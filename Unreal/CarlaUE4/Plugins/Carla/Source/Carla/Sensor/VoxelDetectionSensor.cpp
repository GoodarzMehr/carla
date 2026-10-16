#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"

#include "DrawDebugHelpers.h"
#include "CollisionQueryParams.h"
#include "CustomPhysXPayload.h"
#include "Physics/PhysicsFiltering.h"
#include "PhysicsInterfaceUtilsCore.h"
#include "PhysicsEngine/BodyInstance.h"
#include "PhysXInterfaceWrapper.h"

#include "Engine/World.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"

#include <PxScene.h>
#include <PxSceneLock.h>
#include <geometry/PxGeometryQuery.h>
#include <extensions/PxShapeExt.h>

namespace
{
    struct FVoxelCandidate
    {
        physx::PxGeometryHolder Geometry;
        physx::PxTransform Pose;
        physx::PxBounds3 Bounds;
        uint8 Label;
        uint8 Priority;

        FVoxelCandidate(const physx::PxShape& Shape, const physx::PxRigidActor& Actor,
            uint8 InLabel, uint8 InPriority)
            : Geometry(Shape.getGeometry())
            , Pose(physx::PxShapeExt::getGlobalPose(Shape, Actor))
            , Bounds(physx::PxGeometryQuery::getWorldBounds(Geometry.any(), Pose))
            , Label(InLabel)
            , Priority(InPriority)
        {
        }
    };

    // Match Engine's private object-query filter using public PhysX data.
    class FVoxelCandidateCollector final : public physx::PxQueryFilterCallback
    {
    public:
        TArray<FVoxelCandidate, TInlineAllocator<16>> Candidates;

        FVoxelCandidateCollector(const FCollisionQueryParams& InParams,
            const FCollisionObjectQueryParams& InObjects,
            const uint8* InPriorities, int32 InNumLabels)
            : Params(InParams)
            , Objects(InObjects)
            , Priorities(InPriorities)
            , NumLabels(InNumLabels)
        {
        }

        physx::PxQueryHitType::Enum preFilter(const physx::PxFilterData&,
            const physx::PxShape* Shape, const physx::PxRigidActor* Actor,
            physx::PxHitFlags&) override
        {
            const physx::PxFilterData Filter = Shape->getQueryFilterData();
            const uint32 Complexity = Params.bTraceComplex ? EPDF_ComplexCollision : EPDF_SimpleCollision;
            FMaskFilter ShapeMask;
            const ECollisionChannel Channel = GetCollisionChannelAndExtraFilter(Filter.word3, ShapeMask);
            if (!(Filter.word3 & Complexity) ||
                !(Objects.GetQueryBitfield() & (1u << Channel)) ||
                (Objects.IgnoreMask & ShapeMask) || Params.bIgnoreTouches ||
                Params.GetIgnoredActors().Contains(Filter.word0))
            {
                return physx::PxQueryHitType::eNONE;
            }

            if (Params.GetIgnoredComponents().Num() &&
                Params.GetIgnoredComponents().Contains(Shape->getSimulationFilterData().word2))
            {
                return physx::PxQueryHitType::eNONE;
            }

            UPrimitiveComponent* Component = nullptr;
            if (const FBodyInstance* Body = FPhysxUserData::Get<FBodyInstance>(Actor->userData))
            {
                Body = FPhysicsInterface::ShapeToOriginalBodyInstance(Body, Shape);
                Component = Body->OwnerComponent.Get();
            }
            else if (const FCustomPhysXPayload* Payload =
                FPhysxUserData::Get<FCustomPhysXPayload>(Shape->userData))
            {
                Component = Payload->GetOwningComponent().Get();
            }

            if (Component)
            {
                const int32 Label = Component->CustomDepthStencilValue;
                if (Label > 0 && Label < NumLabels && Priorities[Label] > 0)
                {
                    Candidates.Emplace(*Shape, *Actor, uint8(Label), Priorities[Label]);
                }
            }

            // Continue gathering candidates without running chunk narrow-phase tests.
            return physx::PxQueryHitType::eNONE;
        }

        physx::PxQueryHitType::Enum postFilter(const physx::PxFilterData&,
            const physx::PxQueryHit&) override
        {
            return physx::PxQueryHitType::eNONE;
        }

    private:
        const FCollisionQueryParams& Params;
        const FCollisionObjectQueryParams& Objects;
        const uint8* Priorities;
        int32 NumLabels;
    };
}

TMap<int32, FLinearColor> AVoxelDetectionSensor::ColorMap = AVoxelDetectionSensor::CreateColorMap();
constexpr uint8 AVoxelDetectionSensor::SemanticPriority[32];

AVoxelDetectionSensor::AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer)
    : Super(ObjectInitializer)
{
    PrimaryActorTick.bCanEverTick = true;
}

FActorDefinition AVoxelDetectionSensor::GetSensorDefinition()
{
    auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(TEXT("other"), TEXT("voxel_detection"));

    FActorVariation Range;
    Range.Id = TEXT("range");
    Range.Type = EActorAttributeType::Float;
    Range.RecommendedValues = { TEXT("50.0") };
    Range.bRestrictToRecommended = false;

    FActorVariation Top;
    Top.Id = TEXT("upper_limit");
    Top.Type = EActorAttributeType::Float;
    Top.RecommendedValues = { TEXT("10.0") };
    Top.bRestrictToRecommended = false;

    FActorVariation Bottom;
    Bottom.Id = TEXT("lower_limit");
    Bottom.Type = EActorAttributeType::Float;
    Bottom.RecommendedValues = { TEXT("-5.0") };
    Bottom.bRestrictToRecommended = false;

    FActorVariation VoxelSize;
    VoxelSize.Id = TEXT("voxel_size");
    VoxelSize.Type = EActorAttributeType::Float;
    VoxelSize.RecommendedValues = { TEXT("0.5") };
    VoxelSize.bRestrictToRecommended = false;

    FActorVariation SelfIgnore;
    SelfIgnore.Id = TEXT("ignore_self");
    SelfIgnore.Type = EActorAttributeType::Bool;
    SelfIgnore.RecommendedValues = { TEXT("False") };
    SelfIgnore.bRestrictToRecommended = false;

    FActorVariation DrawDebug;
    DrawDebug.Id = TEXT("draw_debug");
    DrawDebug.Type = EActorAttributeType::Bool;
    DrawDebug.RecommendedValues = { TEXT("False") };
    DrawDebug.bRestrictToRecommended = false;

    FActorVariation ShowCalculationTime;
    ShowCalculationTime.Id = TEXT("show_calculation_time");
    ShowCalculationTime.Type = EActorAttributeType::Bool;
    ShowCalculationTime.RecommendedValues = { TEXT("False") };
    ShowCalculationTime.bRestrictToRecommended = false;

    FActorVariation UseTraceComplex;
    UseTraceComplex.Id = TEXT("use_complex_collision");
    UseTraceComplex.Type = EActorAttributeType::Bool;
    UseTraceComplex.RecommendedValues = { TEXT("True") };
    UseTraceComplex.bRestrictToRecommended = false;

    FActorVariation ChunkX;
    ChunkX.Id = TEXT("chunk_x");
    ChunkX.Type = EActorAttributeType::Int;
    ChunkX.RecommendedValues = { TEXT("4") };
    ChunkX.bRestrictToRecommended = false;

    FActorVariation ChunkY;
    ChunkY.Id = TEXT("chunk_y");
    ChunkY.Type = EActorAttributeType::Int;
    ChunkY.RecommendedValues = { TEXT("4") };
    ChunkY.bRestrictToRecommended = false;

    FActorVariation ChunkZ;
    ChunkZ.Id = TEXT("chunk_z");
    ChunkZ.Type = EActorAttributeType::Int;
    ChunkZ.RecommendedValues = { TEXT("8") };
    ChunkZ.bRestrictToRecommended = false;

    Definition.Variations.Append(
        {
            Range,
            Top,
            Bottom,
            VoxelSize,
            SelfIgnore,
            DrawDebug,
            ShowCalculationTime,
            UseTraceComplex,
            ChunkX,
            ChunkY,
            ChunkZ,
        }
    );

    return Definition;
}

void AVoxelDetectionSensor::Set(const FActorDescription &Description)
{
    Super::Set(Description);

    constexpr float M_TO_CM = 100.0f;

    BoxRange = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("range", Description.Variations, 50.0f);

    Top = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("upper_limit", Description.Variations, 10.0f);

    Bottom = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("lower_limit", Description.Variations, -5.0f);

    BoxSize = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("voxel_size", Description.Variations, 0.5f);

    SelfIgnore = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("ignore_self", Description.Variations, false);

    DrawDebug = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("draw_debug", Description.Variations, false);

    ShowCalculationTime = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("show_calculation_time", Description.Variations, false);

    UseTraceComplex = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_complex_collision", Description.Variations, true);

    ChunkX = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_x", Description.Variations, 4);

    ChunkY = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_y", Description.Variations, 4);

    ChunkZ = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_z", Description.Variations, 8);

    // Pre-calculate grid dimensions.
    GridSizeX = FMath::CeilToInt(2.0f * BoxRange / BoxSize);
    GridSizeY = FMath::CeilToInt(2.0f * BoxRange / BoxSize);
    GridSizeZ = FMath::CeilToInt((Top - Bottom) / BoxSize);

    UE_LOG(
        LogCarla,
        Log,
        TEXT("VoxelDetectionSensor: Grid size %d x %d x %d = %d voxels"),
        GridSizeX,
        GridSizeY,
        GridSizeZ,
        GridSizeX * GridSizeY * GridSizeZ
    );
}

void AVoxelDetectionSensor::SetOwner(AActor *NewOwner)
{
    Super::SetOwner(NewOwner);
}

FVector AVoxelDetectionSensor::VoxelToWorld(int32 X, int32 Y, int32 Z) const
{
    FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

    return GetTransform().TransformPosition(LocalPos);
}

void AVoxelDetectionSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AVoxelDetectionSensor::PostPhysTick);

    const double StartTime = FPlatformTime::Seconds();

    const int32 TotalVoxels = GridSizeX * GridSizeY * GridSizeZ;

    // Initialize voxel grids with 0 (empty).
    TArray<uint8> SemanticVoxels;
    SemanticVoxels.SetNumZeroed(TotalVoxels);

    // Set up collision query.
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);

    QueryParams.bTraceComplex = UseTraceComplex;
    QueryParams.bReturnPhysicalMaterial = false;

    if (SelfIgnore && GetOwner())
    {
        QueryParams.AddIgnoredActor(GetOwner());
    }

    const float HalfVoxel = BoxSize * 0.5f;

    const FVector BoxExtent(HalfVoxel, HalfVoxel, HalfVoxel);

    FCollisionObjectQueryParams ObjectParams(FCollisionObjectQueryParams::AllObjects);
    ObjectParams.RemoveObjectTypesToQuery(ECC_Vehicle);
    ObjectParams.RemoveObjectTypesToQuery(ECC_Pawn);
    ObjectParams.RemoveObjectTypesToQuery(ECC_Camera);
    ObjectParams.RemoveObjectTypesToQuery(ECC_Visibility);
    ObjectParams.RemoveObjectTypesToQuery(ECC_PhysicsBody);
    ObjectParams.RemoveObjectTypesToQuery(ECC_Destructible);

    physx::PxScene* const Scene = World->GetPhysicsScene()->GetPxScene();

    const FTransform SensorTransform = GetTransform();

    const FQuat VoxelRotation = SensorTransform.GetRotation();

    const FCollisionShape VoxelShape = FCollisionShape::MakeBox(BoxExtent);
    const FPhysicsShapeAdapter VoxelAdapter(VoxelRotation, VoxelShape);
    const physx::PxQuat VoxelQuat = U2PQuat(VoxelAdapter.GetGeomOrientation());
    const physx::PxVec3 VoxelWorldExtent = physx::PxGeometryQuery::getWorldBounds(
        VoxelAdapter.GetGeometry(), physx::PxTransform(physx::PxVec3(0.0f), VoxelQuat)).getExtents();
    const auto ResolvePriorityTie = [&](const FVector& WorldPos)
    {
        TArray<FOverlapResult> Overlaps;
        uint8 Label = 0;
        World->ParallelOverlapMultiByObjectType(
            Overlaps, WorldPos, VoxelRotation, ObjectParams, VoxelShape, QueryParams);
        for (const FOverlapResult& Overlap : Overlaps)
        {
            if (const UPrimitiveComponent* Component = Overlap.Component.Get())
            {
                const int32 CandidateLabel = Component->CustomDepthStencilValue;
                if (CandidateLabel >= 0 && CandidateLabel < UE_ARRAY_COUNT(SemanticPriority) &&
                    SemanticPriority[CandidateLabel] > SemanticPriority[Label])
                {
                    Label = uint8(CandidateLabel);
                }
            }
        }
        return Label;
    };

    const int32 SGridX = (GridSizeX + ChunkX - 1) / ChunkX;
    const int32 SGridY = (GridSizeY + ChunkY - 1) / ChunkY;
    const int32 SGridZ = (GridSizeZ + ChunkZ - 1) / ChunkZ;

    const FVector AbsAxisX = SensorTransform.TransformVector(FVector(1.0f, 0.0f, 0.0f)).GetAbs();
    const FVector AbsAxisY = SensorTransform.TransformVector(FVector(0.0f, 1.0f, 0.0f)).GetAbs();
    const FVector AbsAxisZ = SensorTransform.TransformVector(FVector(0.0f, 0.0f, 1.0f)).GetAbs();

    const int32 TotalChunks = SGridX * SGridY * SGridZ;
    const auto ProcessChunk = [&](int32 Index)
    {
        const int32 CX = Index / (SGridY * SGridZ);

        const int32 Remainder = Index % (SGridY * SGridZ);

        const int32 CY = Remainder / SGridZ;
        const int32 CZ = Remainder % SGridZ;

        const int32 XEnd = FMath::Min((CX + 1) * ChunkX, GridSizeX);
        const int32 YEnd = FMath::Min((CY + 1) * ChunkY, GridSizeY);
        const int32 ZEnd = FMath::Min((CZ + 1) * ChunkZ, GridSizeZ);

        const float LocalMinX = -BoxRange + CX * ChunkX * BoxSize;
        const float LocalMinY = -BoxRange + CY * ChunkY * BoxSize;
        const float LocalMinZ = Bottom + CZ * ChunkZ * BoxSize;

        const float LocalMaxX = -BoxRange + XEnd * BoxSize;
        const float LocalMaxY = -BoxRange + YEnd * BoxSize;
        const float LocalMaxZ = Bottom + ZEnd * BoxSize;

        const FVector LocalMin(LocalMinX, LocalMinY, LocalMinZ);
        const FVector LocalMax(LocalMaxX, LocalMaxY, LocalMaxZ);

        const FVector LocalHalfExtent = (LocalMax - LocalMin) * 0.5f;

        const FVector ChunkCenter = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

        // Enclose voxel centers and query extents, including sensor scale.
        const FVector CenterHalfExtent = LocalHalfExtent - BoxExtent;
        const FVector ChunkWorldExtent = AbsAxisX * CenterHalfExtent.X +
            AbsAxisY * CenterHalfExtent.Y + AbsAxisZ * CenterHalfExtent.Z +
            P2UVector(VoxelWorldExtent) + FVector(0.1f);

        FVoxelCandidateCollector Collector(QueryParams, ObjectParams,
            SemanticPriority, UE_ARRAY_COUNT(SemanticPriority));
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherCandidates);
            // PhysX requires touch storage even though the prefilter returns eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);
            const physx::PxQueryFilterData Filter(
                PhysXInterface::StaticDynamicQueryFlags(QueryParams) | physx::PxQueryFlag::ePREFILTER);
            const physx::PxBoxGeometry ChunkGeometry(U2PVector(ChunkWorldExtent));
            const physx::PxTransform ChunkPose(U2PVector(ChunkCenter));
            Scene->overlap(ChunkGeometry, ChunkPose, UnusedHits, Filter, &Collector);

            // Reject empty mesh regions once before testing individual voxels.
            Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
            {
                return !physx::PxGeometryQuery::overlap(ChunkGeometry, ChunkPose,
                    Candidate.Geometry.any(), Candidate.Pose);
            });
            Collector.Candidates.Sort([](const FVoxelCandidate& A, const FVoxelCandidate& B)
            {
                return A.Priority > B.Priority;
            });
        }

        if (Collector.Candidates.Num() == 0)
        {
            return;
        }

        for (int32 X = CX * ChunkX; X < XEnd; ++X)
        {
            for (int32 Y = CY * ChunkY; Y < YEnd; ++Y)
            {
                for (int32 Z = CZ * ChunkZ; Z < ZEnd; ++Z)
                {
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;

                    const FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

                    const FVector WorldPos = SensorTransform.TransformPosition(LocalPos);

                    const physx::PxTransform VoxelPose(U2PVector(WorldPos), VoxelQuat);
                    const physx::PxBounds3 VoxelBounds = physx::PxBounds3::centerExtents(
                        VoxelPose.p, VoxelWorldExtent);
                    uint8 Label = 0;
                    bool bPriorityTie = false;
                    for (const FVoxelCandidate& Candidate : Collector.Candidates)
                    {
                        if (Candidate.Priority < SemanticPriority[Label])
                        {
                            break;
                        }
                        if (Candidate.Label == Label)
                        {
                            continue;
                        }
                        if (!Candidate.Bounds.intersects(VoxelBounds))
                        {
                            continue;
                        }
                        if (physx::PxGeometryQuery::overlap(VoxelAdapter.GetGeometry(), VoxelPose,
                            Candidate.Geometry.any(), Candidate.Pose))
                        {
                            if (Label && Candidate.Label != Label)
                            {
                                // Preserve PhysX hit order for equal-priority labels.
                                bPriorityTie = true;
                                break;
                            }
                            Label = Candidate.Label;
                        }
                    }
                    if (bPriorityTie)
                    {
                        Label = ResolvePriorityTie(WorldPos);
                    }
                    SemanticVoxels[FlatIndex] = Label;
                }
            }
        }
    };

    // PhysX read locks share a mutex; acquire one per batch on its worker.
    const int32 NumBatches = FMath::Min(TotalChunks,
        16 * (FTaskGraphInterface::Get().GetNumWorkerThreads() + 1));
    ParallelFor(NumBatches, [&](int32 Batch)
    {
        physx::PxSceneReadLock SceneLock(*Scene);
        const int32 Begin = int32(int64(TotalChunks) * Batch / NumBatches);
        const int32 End = int32(int64(TotalChunks) * (Batch + 1) / NumBatches);
        for (int32 Index = Begin; Index < End; ++Index)
        {
            ProcessChunk(Index);
        }
    }, EParallelForFlags::Unbalanced);

    // Debug visualization.
    if (DrawDebug)
    {
        const FQuat ActorQuat = GetActorQuat();

        for (int32 FlatIndex = 0; FlatIndex < GridSizeX * GridSizeY * GridSizeZ; ++FlatIndex)
        {
            const uint8 Label = SemanticVoxels[FlatIndex];

            if (Label > 0)
            {
                // Convert flat index back to X, Y, Z.
                const int32 X = FlatIndex / (GridSizeY * GridSizeZ);
                const int32 Remainder = FlatIndex % (GridSizeY * GridSizeZ);
                const int32 Y = Remainder / GridSizeZ;
                const int32 Z = Remainder % GridSizeZ;

                FVector Pos = VoxelToWorld(X, Y, Z);

                FLinearColor Color = ColorMap.Contains(Label) ? ColorMap[Label] : FLinearColor::Green;

                DrawDebugBox(World, Pos, BoxExtent, ActorQuat, Color.ToFColor(true), false, DeltaTime * 1.1f, 0, 2.0f);
            }
        }
    }

    // Send data
    auto DataStream = GetDataStream(*this);

    DataStream.SerializeAndSend(*this, GetEpisode(), SemanticVoxels);

    const double EndTime = FPlatformTime::Seconds();

    if (ShowCalculationTime)
    {
        carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime),
            " ms (chunks: ", TotalChunks, ", batches: ", NumBatches, ")");
    }
}
