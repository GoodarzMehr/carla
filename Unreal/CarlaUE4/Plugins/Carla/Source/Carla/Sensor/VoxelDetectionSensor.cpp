#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Game/CarlaEngine.h"
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
#include "Templates/IntegralConstant.h"

#include <PxScene.h>
#include <PxSceneLock.h>
#include <geometry/PxGeometryQuery.h>
#include <extensions/PxShapeExt.h>

namespace
{
    struct FVoxelProfile
    {
        uint64 GatherCycles = 0;
        uint64 CullCycles = 0;
        uint64 FineCycles = 0;
        uint64 TieCycles = 0;
        uint64 Regions = 0;
        uint64 Candidates = 0;
        uint64 Chunks = 0;
        uint64 Voxels = 0;
        uint64 Tests = 0;
        uint64 Ties = 0;
    };

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

    FActorVariation SuperChunkFactor;
    SuperChunkFactor.Id = TEXT("super_chunk_factor");
    SuperChunkFactor.Type = EActorAttributeType::Int;
    SuperChunkFactor.RecommendedValues = { TEXT("2"), TEXT("1"), TEXT("4"), TEXT("8") };
    SuperChunkFactor.bRestrictToRecommended = true;

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
            SuperChunkFactor,
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

    SuperChunkFactor = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("super_chunk_factor", Description.Variations, 2);
    if (SuperChunkFactor < 1 || SuperChunkFactor > 8 || (SuperChunkFactor & (SuperChunkFactor - 1)))
    {
        UE_LOG(LogCarla, Warning, TEXT("VoxelDetectionSensor: super_chunk_factor must be 1, 2, 4, or 8; using 2"));
        SuperChunkFactor = 2;
    }

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

    const int32 SGridX = GridSizeX / ChunkX + (GridSizeX % ChunkX != 0);
    const int32 SGridY = GridSizeY / ChunkY + (GridSizeY % ChunkY != 0);
    const int32 SGridZ = GridSizeZ / ChunkZ + (GridSizeZ % ChunkZ != 0);

    const int32 PGridX = SGridX / SuperChunkFactor + (SGridX % SuperChunkFactor != 0);
    const int32 PGridY = SGridY / SuperChunkFactor + (SGridY % SuperChunkFactor != 0);
    const int32 PGridZ = SGridZ / SuperChunkFactor + (SGridZ % SuperChunkFactor != 0);

    const FVector AbsAxisX = SensorTransform.TransformVector(FVector(1.0f, 0.0f, 0.0f)).GetAbs();
    const FVector AbsAxisY = SensorTransform.TransformVector(FVector(0.0f, 1.0f, 0.0f)).GetAbs();
    const FVector AbsAxisZ = SensorTransform.TransformVector(FVector(0.0f, 0.0f, 1.0f)).GetAbs();

    struct FChunkBounds
    {
        physx::PxBoxGeometry Geometry;
        physx::PxTransform Pose;
    };

    const auto MakeChunkBounds = [&](const FIntVector& Begin, const FIntVector& End)
    {
        const FVector LocalMin(-BoxRange + Begin.X * BoxSize,
            -BoxRange + Begin.Y * BoxSize, Bottom + Begin.Z * BoxSize);
        const FVector LocalMax(-BoxRange + End.X * BoxSize,
            -BoxRange + End.Y * BoxSize, Bottom + End.Z * BoxSize);

        const FVector LocalHalfExtent = (LocalMax - LocalMin) * 0.5f;

        const FVector ChunkCenter = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

        // Enclose voxel centers and query extents, including sensor scale.
        const FVector CenterHalfExtent = LocalHalfExtent - BoxExtent;
        const FVector ChunkWorldExtent = AbsAxisX * CenterHalfExtent.X +
            AbsAxisY * CenterHalfExtent.Y + AbsAxisZ * CenterHalfExtent.Z +
            P2UVector(VoxelWorldExtent) + FVector(0.1f);

        return FChunkBounds{physx::PxBoxGeometry(U2PVector(ChunkWorldExtent)),
            physx::PxTransform(U2PVector(ChunkCenter))};
    };

    const auto GetChunkEnd = [&](int32 X, int32 Y, int32 Z)
    {
        return FIntVector(
            int32(FMath::Min<int64>(int64(X) * ChunkX, GridSizeX)),
            int32(FMath::Min<int64>(int64(Y) * ChunkY, GridSizeY)),
            int32(FMath::Min<int64>(int64(Z) * ChunkZ, GridSizeZ)));
    };

    const int32 TotalChunks = SGridX * SGridY * SGridZ;
    const int32 TotalRegions = PGridX * PGridY * PGridZ;

    // Separate instantiations keep profiling clocks and counters out of normal regions.
    const auto ProcessRegion = [&](int32 Index, auto ProfileTag, FVoxelProfile& Profile)
    {
        constexpr bool bProfile = decltype(ProfileTag)::Value;
        const int32 PX = Index / (PGridY * PGridZ);
        const int32 Remainder = Index % (PGridY * PGridZ);
        const int32 PY = Remainder / PGridZ;
        const int32 PZ = Remainder % PGridZ;

        const FIntVector FirstChunk(PX * SuperChunkFactor,
            PY * SuperChunkFactor, PZ * SuperChunkFactor);
        const FIntVector LastChunk(
            FMath::Min(FirstChunk.X + SuperChunkFactor, SGridX),
            FMath::Min(FirstChunk.Y + SuperChunkFactor, SGridY),
            FMath::Min(FirstChunk.Z + SuperChunkFactor, SGridZ));
        const FIntVector Begin(FirstChunk.X * ChunkX,
            FirstChunk.Y * ChunkY, FirstChunk.Z * ChunkZ);
        const FIntVector End = GetChunkEnd(LastChunk.X, LastChunk.Y, LastChunk.Z);
        const FChunkBounds Parent = MakeChunkBounds(Begin, End);

        FVoxelCandidateCollector Collector(QueryParams, ObjectParams,
            SemanticPriority, UE_ARRAY_COUNT(SemanticPriority));
        const uint64 GatherStart = bProfile ? FPlatformTime::Cycles64() : 0;
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherCandidates);
            // PhysX requires touch storage even though the prefilter returns eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);
            const physx::PxQueryFilterData Filter(
                PhysXInterface::StaticDynamicQueryFlags(QueryParams) | physx::PxQueryFlag::ePREFILTER);
            Scene->overlap(Parent.Geometry, Parent.Pose, UnusedHits, Filter, &Collector);
        }

        const uint64 CullStart = bProfile ? FPlatformTime::Cycles64() : 0;
        if (bProfile)
        {
            Profile.GatherCycles += CullStart - GatherStart;
            ++Profile.Regions;
            Profile.Candidates += Collector.Candidates.Num();
        }

        // A rejected parent skips all of its children, including their mesh tests.
        Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
        {
            return !physx::PxGeometryQuery::overlap(Parent.Geometry, Parent.Pose,
                Candidate.Geometry.any(), Candidate.Pose);
        });
        Collector.Candidates.Sort([](const FVoxelCandidate& A, const FVoxelCandidate& B)
        {
            return A.Priority > B.Priority;
        });
        if (bProfile)
        {
            Profile.CullCycles += FPlatformTime::Cycles64() - CullStart;
        }

        if (Collector.Candidates.Num() == 0)
        {
            return;
        }

        const auto FillChunk = [&](const FIntVector& ChunkBegin, const FIntVector& ChunkEnd,
            const TArray<const FVoxelCandidate*, TInlineAllocator<16>>& Candidates)
        {
            if (Candidates.Num() == 0)
            {
                return;
            }

            const uint64 FineStart = bProfile ? FPlatformTime::Cycles64() : 0;
            if (bProfile)
            {
                ++Profile.Chunks;
                Profile.Voxels += uint64(ChunkEnd.X - ChunkBegin.X) *
                    (ChunkEnd.Y - ChunkBegin.Y) * (ChunkEnd.Z - ChunkBegin.Z);
            }

            for (int32 X = ChunkBegin.X; X < ChunkEnd.X; ++X)
            {
                for (int32 Y = ChunkBegin.Y; Y < ChunkEnd.Y; ++Y)
                {
                    for (int32 Z = ChunkBegin.Z; Z < ChunkEnd.Z; ++Z)
                    {
                        const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;

                        const FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

                        const FVector WorldPos = SensorTransform.TransformPosition(LocalPos);

                        const physx::PxTransform VoxelPose(U2PVector(WorldPos), VoxelQuat);
                        const physx::PxBounds3 VoxelBounds = physx::PxBounds3::centerExtents(
                            VoxelPose.p, VoxelWorldExtent);
                        uint8 Label = 0;
                        bool bPriorityTie = false;
                        for (const FVoxelCandidate* Entry : Candidates)
                        {
                            const FVoxelCandidate& Candidate = *Entry;
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
                            if (bProfile)
                            {
                                ++Profile.Tests;
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
                            const uint64 TieStart = bProfile ? FPlatformTime::Cycles64() : 0;
                            Label = ResolvePriorityTie(WorldPos);
                            if (bProfile)
                            {
                                Profile.TieCycles += FPlatformTime::Cycles64() - TieStart;
                                ++Profile.Ties;
                            }
                        }
                        SemanticVoxels[FlatIndex] = Label;
                    }
                }
            }

            if (bProfile)
            {
                Profile.FineCycles += FPlatformTime::Cycles64() - FineStart;
            }
        };

        TArray<const FVoxelCandidate*, TInlineAllocator<16>> Candidates;
        if (SuperChunkFactor == 1)
        {
            // The parent is already an exact-filtered leaf; do not test it twice.
            for (const FVoxelCandidate& Candidate : Collector.Candidates)
            {
                Candidates.Add(&Candidate);
            }
            FillChunk(Begin, End, Candidates);
            return;
        }

        for (int32 CX = FirstChunk.X; CX < LastChunk.X; ++CX)
        {
            for (int32 CY = FirstChunk.Y; CY < LastChunk.Y; ++CY)
            {
                for (int32 CZ = FirstChunk.Z; CZ < LastChunk.Z; ++CZ)
                {
                    const uint64 ChildCullStart = bProfile ? FPlatformTime::Cycles64() : 0;
                    const FIntVector ChildBegin(CX * ChunkX, CY * ChunkY, CZ * ChunkZ);
                    const FIntVector ChildEnd = GetChunkEnd(CX + 1, CY + 1, CZ + 1);
                    const FChunkBounds Child = MakeChunkBounds(ChildBegin, ChildEnd);
                    const physx::PxBounds3 ChildBounds = physx::PxBounds3::centerExtents(
                        Child.Pose.p, Child.Geometry.halfExtents);
                    Candidates.Reset();
                    for (const FVoxelCandidate& Candidate : Collector.Candidates)
                    {
                        if (Candidate.Bounds.intersects(ChildBounds) &&
                            physx::PxGeometryQuery::overlap(Child.Geometry, Child.Pose,
                                Candidate.Geometry.any(), Candidate.Pose))
                        {
                            // Parent order preserves descending semantic priority.
                            Candidates.Add(&Candidate);
                        }
                    }
                    if (bProfile)
                    {
                        Profile.CullCycles += FPlatformTime::Cycles64() - ChildCullStart;
                    }
                    FillChunk(ChildBegin, ChildEnd, Candidates);
                }
            }
        }
    };

    // PhysX read locks share a mutex; acquire one per batch on its worker.
    const int32 NumBatches = FMath::Min(TotalRegions,
        16 * (FTaskGraphInterface::Get().GetNumWorkerThreads() + 1));
    TArray<FVoxelProfile> Profiles;
    if (ShowCalculationTime)
    {
        Profiles.SetNum(NumBatches);
    }
    // A prime stride and rotating offset avoid repeatedly sampling the same grid layers.
    constexpr int32 ProfileStride = 127;
    const int32 ProfileOffset = FCarlaEngine::GetFrameCounter() % ProfileStride;
    const double ParallelStart = ShowCalculationTime ? FPlatformTime::Seconds() : 0.0;
    ParallelFor(NumBatches, [&](int32 Batch)
    {
        TRACE_CPUPROFILER_EVENT_SCOPE(VoxelProcessBatch);
        physx::PxSceneReadLock SceneLock(*Scene);
        FVoxelProfile Profile;
        const int32 Begin = int32(int64(TotalRegions) * Batch / NumBatches);
        const int32 End = int32(int64(TotalRegions) * (Batch + 1) / NumBatches);
        for (int32 Index = Begin; Index < End; ++Index)
        {
            if (ShowCalculationTime &&
                (TotalRegions < ProfileStride || Index % ProfileStride == ProfileOffset))
            {
                ProcessRegion(Index, TIntegralConstant<bool, true>(), Profile);
            }
            else
            {
                ProcessRegion(Index, TIntegralConstant<bool, false>(), Profile);
            }
        }
        if (ShowCalculationTime)
        {
            Profiles[Batch] = Profile;
        }
    }, EParallelForFlags::Unbalanced);
    const double ParallelEnd = ShowCalculationTime ? FPlatformTime::Seconds() : 0.0;

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

    const double SendStart = ShowCalculationTime ? FPlatformTime::Seconds() : 0.0;

    // Send data
    auto DataStream = GetDataStream(*this);

    DataStream.SerializeAndSend(*this, GetEpisode(), SemanticVoxels);

    const double EndTime = FPlatformTime::Seconds();

    if (ShowCalculationTime)
    {
        carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime),
            " ms (chunks: ", TotalChunks, ", parent queries: ", TotalRegions,
            ", factor: ", SuperChunkFactor, ", batches: ", NumBatches, ")");
        carla::log_warning("Voxel wall time (ms): setup=", 1000.0 * (ParallelStart - StartTime),
            ", parallel=", 1000.0 * (ParallelEnd - ParallelStart),
            ", debug=", 1000.0 * (SendStart - ParallelEnd),
            ", serialize/send=", 1000.0 * (EndTime - SendStart));

        FVoxelProfile Profile;
        for (const FVoxelProfile& Batch : Profiles)
        {
            Profile.GatherCycles += Batch.GatherCycles;
            Profile.CullCycles += Batch.CullCycles;
            Profile.FineCycles += Batch.FineCycles;
            Profile.TieCycles += Batch.TieCycles;
            Profile.Regions += Batch.Regions;
            Profile.Candidates += Batch.Candidates;
            Profile.Chunks += Batch.Chunks;
            Profile.Voxels += Batch.Voxels;
            Profile.Tests += Batch.Tests;
            Profile.Ties += Batch.Ties;
        }
        // These are sampled sums across workers, not frame wall time. Fine excludes ties.
        carla::log_warning("Voxel sampled worker time (ms): gather=",
            FPlatformTime::ToMilliseconds64(Profile.GatherCycles),
            ", cull/sort=", FPlatformTime::ToMilliseconds64(Profile.CullCycles),
            ", fine=", FPlatformTime::ToMilliseconds64(Profile.FineCycles - Profile.TieCycles),
            ", ties=", FPlatformTime::ToMilliseconds64(Profile.TieCycles));
        carla::log_warning("Voxel sampled counts: regions=", Profile.Regions, "/", TotalRegions,
            ", candidates=", Profile.Candidates, ", active chunks=", Profile.Chunks,
            ", visited voxels=", Profile.Voxels, ", exact tests=", Profile.Tests,
            ", tie queries=", Profile.Ties);
    }
}
