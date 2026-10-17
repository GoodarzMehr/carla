#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla/Sensor/VoxelTriangleOverlap.h"
#include "Carla/Sensor/VoxelSurfaceRaster.h"
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

#include <PxRigidDynamic.h>
#include <PxScene.h>
#include <PxSceneLock.h>
#include <geometry/PxGeometryQuery.h>
#include <geometry/PxMeshQuery.h>
#include <extensions/PxShapeExt.h>

namespace
{
    // Value that marks voxels nobody wrote when validating the output.
    constexpr uint8 UnwrittenVoxel = 0xFF;

    // Chunk size of the reference implementation used for validation.
    const FIntVector ReferenceChunkSize(4, 4, 8);

    struct FVoxelProfile
    {
        uint64 GatherCycles = 0;
        uint64 CullCycles = 0;
        uint64 TriangleCycles = 0;
        uint64 FineCycles = 0;
        uint64 PhysXCycles = 0;
        uint64 TieCycles = 0;
        uint64 Regions = 0;
        uint64 Candidates = 0;
        uint64 Chunks = 0;
        uint64 Voxels = 0;
        uint64 Tests = 0;
        uint64 PhysXQueries = 0;
        uint64 Ties = 0;
        uint64 CachedMeshes = 0;
        uint64 CachedTriangles = 0;
        uint64 TriangleQueries = 0;
        uint64 BoundaryFallbacks = 0;
    };

    struct FVoxelCandidate
    {
        physx::PxGeometryHolder Geometry;
        physx::PxTransform Pose;
        physx::PxBounds3 Bounds;
        uint8 Label;
        uint8 Priority;
        int32 TriangleStart = INDEX_NONE;
        int32 TriangleCount = 0;

        FVoxelCandidate(const physx::PxShape& Shape, const physx::PxRigidActor& Actor, uint8 InLabel, uint8 InPriority)
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

        // Set when a candidate's scene-query bounds may follow a pending kinematic target instead of its current pose.
        bool bKinematicTarget = false;

        FVoxelCandidateCollector(const FCollisionQueryParams& InParams, const FCollisionObjectQueryParams& InObjects, const uint8* InPriorities, int32 InNumLabels)
            : Params(InParams)
            , Objects(InObjects)
            , Priorities(InPriorities)
            , NumLabels(InNumLabels)
        {
        }

        physx::PxQueryHitType::Enum preFilter(const physx::PxFilterData&, const physx::PxShape* Shape, const physx::PxRigidActor* Actor, physx::PxHitFlags&) override
        {
            const physx::PxFilterData Filter = Shape->getQueryFilterData();

            const uint32 Complexity = Params.bTraceComplex ? EPDF_ComplexCollision : EPDF_SimpleCollision;

            FMaskFilter ShapeMask;

            const ECollisionChannel Channel = GetCollisionChannelAndExtraFilter(Filter.word3, ShapeMask);

            if (!(Filter.word3 & Complexity) || !(Objects.GetQueryBitfield() & (1u << Channel)) || (Objects.IgnoreMask & ShapeMask) || Params.bIgnoreTouches || Params.GetIgnoredActors().Contains(Filter.word0))
            {
                return physx::PxQueryHitType::eNONE;
            }

            if (Params.GetIgnoredComponents().Num() && Params.GetIgnoredComponents().Contains(Shape->getSimulationFilterData().word2))
            {
                return physx::PxQueryHitType::eNONE;
            }

            UPrimitiveComponent* Component = nullptr;

            if (const FBodyInstance* Body = FPhysxUserData::Get<FBodyInstance>(Actor->userData))
            {
                Body = FPhysicsInterface::ShapeToOriginalBodyInstance(Body, Shape);

                Component = Body->OwnerComponent.Get();
            }
            else if (const FCustomPhysXPayload* Payload = FPhysxUserData::Get<FCustomPhysXPayload>(Shape->userData))
            {
                Component = Payload->GetOwningComponent().Get();
            }

            if (Component)
            {
                const int32 Label = Component->CustomDepthStencilValue;

                if (Label > 0 && Label < NumLabels && Priorities[Label] > 0)
                {
                    Candidates.Emplace(*Shape, *Actor, uint8(Label), Priorities[Label]);

                    const physx::PxRigidDynamic* Dynamic = Actor->is<physx::PxRigidDynamic>();

                    physx::PxTransform Target;

                    if (Dynamic && Dynamic->getKinematicTarget(Target) && !(Target == Actor->getGlobalPose()))
                    {
                        bKinematicTarget = true;
                    }
                }
            }

            // Continue gathering candidates without running chunk narrow-phase tests.
            return physx::PxQueryHitType::eNONE;
        }

        physx::PxQueryHitType::Enum postFilter(const physx::PxFilterData&, const physx::PxQueryHit&) override
        {
            return physx::PxQueryHitType::eNONE;
        }

    private:
        const FCollisionQueryParams& Params;
        const FCollisionObjectQueryParams& Objects;
        const uint8* Priorities;
        int32 NumLabels;
    };

    struct FChunkBounds
    {
        physx::PxBoxGeometry Geometry;
        physx::PxTransform Pose;
    };

    // Query setup and grid geometry shared by every chunk of one frame.
    struct FVoxelFrame
    {
        UWorld* World;
        physx::PxScene* Scene;
        FCollisionQueryParams QueryParams;
        FCollisionObjectQueryParams ObjectParams;
        FTransform SensorTransform;
        FQuat VoxelRotation;
        FVector BoxExtent;
        FCollisionShape VoxelShape;
        FPhysicsShapeAdapter VoxelAdapter;
        bool bBoxVoxels = false;
        physx::PxVec3 VoxelHalfExtent;
        physx::PxQuat VoxelQuat;
        physx::PxVec3 VoxelWorldExtent;
        FVector AbsAxisX;
        FVector AbsAxisY;
        FVector AbsAxisZ;
        FVector SensorScale;
        double CenterError = 0.0;
        float BoxSize;
        float BoxRange;
        float Bottom;
        int32 GridSizeX;
        int32 GridSizeY;
        int32 GridSizeZ;
        const uint8* Priorities;
        int32 NumLabels;

        FVoxelFrame(
            UWorld* InWorld,
            const FTransform& InSensorTransform,
            const FCollisionQueryParams& InQueryParams,
            float InBoxSize,
            float InBoxRange,
            float InBottom,
            int32 InGridSizeX,
            int32 InGridSizeY,
            int32 InGridSizeZ,
            const uint8* InPriorities,
            int32 InNumLabels)
            : World(InWorld)
            , Scene(InWorld->GetPhysicsScene()->GetPxScene())
            , QueryParams(InQueryParams)
            , ObjectParams(FCollisionObjectQueryParams::AllObjects)
            , SensorTransform(InSensorTransform)
            , VoxelRotation(InSensorTransform.GetRotation())
            , BoxExtent(InBoxSize * 0.5f, InBoxSize * 0.5f, InBoxSize * 0.5f)
            , VoxelShape(FCollisionShape::MakeBox(BoxExtent))
            , VoxelAdapter(VoxelRotation, VoxelShape)
            , BoxSize(InBoxSize)
            , BoxRange(InBoxRange)
            , Bottom(InBottom)
            , GridSizeX(InGridSizeX)
            , GridSizeY(InGridSizeY)
            , GridSizeZ(InGridSizeZ)
            , Priorities(InPriorities)
            , NumLabels(InNumLabels)
        {
            ObjectParams.RemoveObjectTypesToQuery(ECC_Vehicle);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Pawn);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Camera);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Visibility);
            ObjectParams.RemoveObjectTypesToQuery(ECC_PhysicsBody);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Destructible);

            bBoxVoxels = VoxelAdapter.GetGeometry().getType() == physx::PxGeometryType::eBOX;

            VoxelHalfExtent = bBoxVoxels ? static_cast<const physx::PxBoxGeometry&>(VoxelAdapter.GetGeometry()).halfExtents : physx::PxVec3(0.0f);

            VoxelQuat = U2PQuat(VoxelAdapter.GetGeomOrientation());

            VoxelWorldExtent = physx::PxGeometryQuery::getWorldBounds(VoxelAdapter.GetGeometry(), physx::PxTransform(physx::PxVec3(0.0f), VoxelQuat)).getExtents();

            AbsAxisX = SensorTransform.TransformVector(FVector(1.0f, 0.0f, 0.0f)).GetAbs();
            AbsAxisY = SensorTransform.TransformVector(FVector(0.0f, 1.0f, 0.0f)).GetAbs();
            AbsAxisZ = SensorTransform.TransformVector(FVector(0.0f, 0.0f, 1.0f)).GetAbs();

            SensorScale = SensorTransform.GetScale3D();

            // Bound on the rounding of one voxel center, from the world position through the tile frame.
            const float LocalMagnitude = BoxRange + FMath::Max(FMath::Abs(Bottom), FMath::Abs(Bottom + GridSizeZ * BoxSize)) + BoxSize;

            CenterError = 32.0 * FLT_EPSILON * (SensorTransform.GetLocation().GetAbsMax() + 2.0 * SensorScale.GetAbsMax() * LocalMagnitude + 1.0);
        }

        // The surface raster needs box voxels and a sensor scale it can invert.
        bool SupportsRaster() const
        {
            return bBoxVoxels && VoxelQuat.isFinite() && SensorScale.GetAbsMin() > 1e-3f && FMath::IsFinite(CenterError);
        }

        // Approximate centers of the voxels [Begin, Begin + Size) in the tile frame.
        VoxelSurfaceRaster::FLattice MakeLattice(const FIntVector& Begin, const FIntVector& Size, const physx::PxTransform& TileFrame, int32 FootprintPadding) const
        {
            const int32 LatticeSize[3] = {Size.X, Size.Y, Size.Z};

            const double Step[3] = {double(SensorScale.X) * BoxSize, double(SensorScale.Y) * BoxSize, double(SensorScale.Z) * BoxSize};

            const double Padding = FootprintPadding * double(SensorScale.GetAbsMax()) * BoxSize;

            return VoxelSurfaceRaster::MakeLattice(LatticeSize, Step, CenterError, Padding, [&](int32 X, int32 Y, int32 Z)
            {
                return TileFrame.transformInv(U2PVector(GetWorldPosition(Begin.X + X, Begin.Y + Y, Begin.Z + Z)));
            });
        }

        // Center of voxel (X, Y, Z) in world space.
        FVector GetWorldPosition(int32 X, int32 Y, int32 Z) const
        {
            const FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

            return SensorTransform.TransformPosition(LocalPos);
        }

        // Axis-aligned world box that encloses every voxel query between Begin and End.
        FChunkBounds MakeChunkBounds(const FIntVector& Begin, const FIntVector& End) const
        {
            const FVector LocalMin(-BoxRange + Begin.X * BoxSize, -BoxRange + Begin.Y * BoxSize, Bottom + Begin.Z * BoxSize);
            const FVector LocalMax(-BoxRange + End.X * BoxSize, -BoxRange + End.Y * BoxSize, Bottom + End.Z * BoxSize);

            const FVector LocalHalfExtent = (LocalMax - LocalMin) * 0.5f;

            const FVector ChunkCenter = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

            // Enclose voxel centers and query extents, including sensor scale.
            const FVector CenterHalfExtent = LocalHalfExtent - BoxExtent;

            const FVector ChunkWorldExtent = AbsAxisX * CenterHalfExtent.X + AbsAxisY * CenterHalfExtent.Y + AbsAxisZ * CenterHalfExtent.Z + P2UVector(VoxelWorldExtent) + FVector(0.1f);

            return FChunkBounds{physx::PxBoxGeometry(U2PVector(ChunkWorldExtent)), physx::PxTransform(U2PVector(ChunkCenter))};
        }

        // Exclusive end voxel of chunk (X - 1, Y - 1, Z - 1), clamped to the grid.
        FIntVector GetChunkEnd(const FIntVector& ChunkSize, int32 X, int32 Y, int32 Z) const
        {
            return FIntVector(
                int32(FMath::Min<int64>(int64(X) * ChunkSize.X, GridSizeX)),
                int32(FMath::Min<int64>(int64(Y) * ChunkSize.Y, GridSizeY)),
                int32(FMath::Min<int64>(int64(Z) * ChunkSize.Z, GridSizeZ))
            );
        }

        // Label of one voxel from the original per-voxel query; PhysX hit order breaks equal-priority ties.
        uint8 QueryVoxelLabel(const FVector& WorldPos) const
        {
            TArray<FOverlapResult> Overlaps;

            uint8 Label = 0;

            World->ParallelOverlapMultiByObjectType(
                Overlaps,
                WorldPos,
                VoxelRotation,
                ObjectParams,
                VoxelShape,
                QueryParams
            );

            for (const FOverlapResult& Overlap : Overlaps)
            {
                if (const UPrimitiveComponent* Component = Overlap.Component.Get())
                {
                    const int32 CandidateLabel = Component->CustomDepthStencilValue;

                    if (CandidateLabel >= 0 && CandidateLabel < NumLabels && Priorities[CandidateLabel] > Priorities[Label])
                    {
                        Label = uint8(CandidateLabel);
                    }
                }
            }

            return Label;
        }
    };

    // Label every voxel of one chunk. Separate instantiations keep profiling clocks out of normal chunks.
    template <bool bProfile>
    void ProcessChunk(const FVoxelFrame& Frame, const FIntVector& ChunkSize, const FIntVector& Chunk, uint8* Labels, FVoxelProfile& Profile)
    {
        const FIntVector Begin(Chunk.X * ChunkSize.X, Chunk.Y * ChunkSize.Y, Chunk.Z * ChunkSize.Z);

        const FIntVector End = Frame.GetChunkEnd(ChunkSize, Chunk.X + 1, Chunk.Y + 1, Chunk.Z + 1);

        const FChunkBounds ChunkBounds = Frame.MakeChunkBounds(Begin, End);

        FVoxelCandidateCollector Collector(Frame.QueryParams, Frame.ObjectParams, Frame.Priorities, Frame.NumLabels);

        const uint64 GatherStart = bProfile ? FPlatformTime::Cycles64() : 0;

        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherCandidates);

            // PhysX requires touch storage even though the prefilter returns eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);

            const physx::PxQueryFilterData Filter(PhysXInterface::StaticDynamicQueryFlags(Frame.QueryParams) | physx::PxQueryFlag::ePREFILTER);

            Frame.Scene->overlap(ChunkBounds.Geometry, ChunkBounds.Pose, UnusedHits, Filter, &Collector);
        }

        const uint64 CullStart = bProfile ? FPlatformTime::Cycles64() : 0;

        if (bProfile)
        {
            Profile.GatherCycles += CullStart - GatherStart;

            ++Profile.Regions;

            Profile.Candidates += Collector.Candidates.Num();
        }

        // Reject empty mesh regions once before testing individual voxels.
        Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
        {
            return !physx::PxGeometryQuery::overlap(ChunkBounds.Geometry, ChunkBounds.Pose, Candidate.Geometry.any(), Candidate.Pose);
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
            // Every voxel is written exactly once, so the output buffer is never cleared beforehand.
            for (int32 X = Begin.X; X < End.X; ++X)
            {
                for (int32 Y = Begin.Y; Y < End.Y; ++Y)
                {
                    const int32 FlatIndex = X * Frame.GridSizeY * Frame.GridSizeZ + Y * Frame.GridSizeZ + Begin.Z;

                    FMemory::Memzero(Labels + FlatIndex, End.Z - Begin.Z);
                }
            }

            return;
        }

        TArray<VoxelTriangleOverlap::FTriangle, TInlineAllocator<8>> Triangles;

        const physx::PxTransform TriangleFrame(ChunkBounds.Pose.p, Frame.VoxelQuat);

        if (Frame.bBoxVoxels)
        {
            const uint64 TriangleStart = bProfile ? FPlatformTime::Cycles64() : 0;

            for (FVoxelCandidate& Candidate : Collector.Candidates)
            {
                if (Candidate.Geometry.getType() != physx::PxGeometryType::eTRIANGLEMESH)
                {
                    continue;
                }

                // Dense chunks retain the mesh BVH instead of scanning a long triangle list.
                constexpr int32 MaxTriangles = 64;

                physx::PxU32 Indices[MaxTriangles];

                bool bOverflow = false;

                const auto& Mesh = Candidate.Geometry.triangleMesh();

                const int32 Count = physx::PxMeshQuery::findOverlapTriangleMesh(ChunkBounds.Geometry, ChunkBounds.Pose, Mesh, Candidate.Pose, Indices, MaxTriangles, 0, bOverflow);

                if (bOverflow || Count == 0)
                {
                    continue;
                }

                Candidate.TriangleStart = Triangles.Num();
                Candidate.TriangleCount = Count;

                for (int32 Triangle = 0; Triangle < Count; ++Triangle)
                {
                    Triangles.Add(VoxelTriangleOverlap::MakeTriangle(Mesh, Candidate.Pose, TriangleFrame, Indices[Triangle], Frame.VoxelHalfExtent));
                }

                if (bProfile)
                {
                    ++Profile.CachedMeshes;

                    Profile.CachedTriangles += Count;
                }
            }

            if (bProfile)
            {
                Profile.TriangleCycles += FPlatformTime::Cycles64() - TriangleStart;
            }
        }

        const uint64 FineStart = bProfile ? FPlatformTime::Cycles64() : 0;

        if (bProfile)
        {
            ++Profile.Chunks;

            Profile.Voxels += uint64(End.X - Begin.X) * (End.Y - Begin.Y) * (End.Z - Begin.Z);
        }

        for (int32 X = Begin.X; X < End.X; ++X)
        {
            for (int32 Y = Begin.Y; Y < End.Y; ++Y)
            {
                for (int32 Z = Begin.Z; Z < End.Z; ++Z)
                {
                    const int32 FlatIndex = X * Frame.GridSizeY * Frame.GridSizeZ + Y * Frame.GridSizeZ + Z;

                    const FVector WorldPos = Frame.GetWorldPosition(X, Y, Z);

                    const physx::PxTransform VoxelPose(U2PVector(WorldPos), Frame.VoxelQuat);

                    const physx::PxBounds3 VoxelBounds = physx::PxBounds3::centerExtents(VoxelPose.p, Frame.VoxelWorldExtent);

                    const physx::PxVec3 TriangleCenter = Frame.bBoxVoxels ? TriangleFrame.transformInv(VoxelPose.p) : physx::PxVec3(0.0f);

                    uint8 Label = 0;

                    bool bPriorityTie = false;

                    for (const FVoxelCandidate& Candidate : Collector.Candidates)
                    {
                        if (Candidate.Priority < Frame.Priorities[Label])
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

                        bool bOverlap = false;

                        bool bNeedsMeshQuery = Candidate.TriangleStart == INDEX_NONE;

                        if (!bNeedsMeshQuery)
                        {
                            if (bProfile)
                            {
                                ++Profile.TriangleQueries;
                            }

                            for (int32 Triangle = Candidate.TriangleStart; Triangle < Candidate.TriangleStart + Candidate.TriangleCount; ++Triangle)
                            {
                                const VoxelTriangleOverlap::EResult Result = VoxelTriangleOverlap::Test(Triangles[Triangle], TriangleCenter);

                                if (Result == VoxelTriangleOverlap::EResult::Overlap)
                                {
                                    bOverlap = true;

                                    break;
                                }

                                if (Result == VoxelTriangleOverlap::EResult::Uncertain)
                                {
                                    bNeedsMeshQuery = true;
                                }
                            }

                            if (bProfile && bNeedsMeshQuery && !bOverlap)
                            {
                                ++Profile.BoundaryFallbacks;
                            }
                        }

                        if (bNeedsMeshQuery && !bOverlap)
                        {
                            const uint64 PhysXStart = bProfile ? FPlatformTime::Cycles64() : 0;

                            bOverlap = physx::PxGeometryQuery::overlap(Frame.VoxelAdapter.GetGeometry(), VoxelPose, Candidate.Geometry.any(), Candidate.Pose);

                            if (bProfile)
                            {
                                Profile.PhysXCycles += FPlatformTime::Cycles64() - PhysXStart;

                                ++Profile.PhysXQueries;
                            }
                        }

                        if (bOverlap)
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

                        Label = Frame.QueryVoxelLabel(WorldPos);

                        if (bProfile)
                        {
                            Profile.TieCycles += FPlatformTime::Cycles64() - TieStart;

                            ++Profile.Ties;
                        }
                    }

                    Labels[FlatIndex] = Label;
                }
            }
        }

        if (bProfile)
        {
            Profile.FineCycles += FPlatformTime::Cycles64() - FineStart;
        }
    }

    // Label every voxel chunk by chunk and return the number of chunks.
    int32 RunChunks(const FVoxelFrame& Frame, const FIntVector& ChunkSize, uint8* Labels, bool bShowCalculationTime, TArray<FVoxelProfile>& OutProfiles)
    {
        const int32 SGridX = Frame.GridSizeX / ChunkSize.X + (Frame.GridSizeX % ChunkSize.X != 0);
        const int32 SGridY = Frame.GridSizeY / ChunkSize.Y + (Frame.GridSizeY % ChunkSize.Y != 0);
        const int32 SGridZ = Frame.GridSizeZ / ChunkSize.Z + (Frame.GridSizeZ % ChunkSize.Z != 0);

        const int32 TotalChunks = SGridX * SGridY * SGridZ;

        // PhysX read locks share a mutex; acquire one per batch on its worker.
        const int32 NumBatches = FMath::Min(TotalChunks, 16 * (FTaskGraphInterface::Get().GetNumWorkerThreads() + 1));

        OutProfiles.Reset();

        if (bShowCalculationTime)
        {
            OutProfiles.SetNum(NumBatches);
        }

        // A prime stride and rotating offset avoid repeatedly sampling the same grid layers.
        constexpr int32 ProfileStride = 127;

        const int32 ProfileOffset = FCarlaEngine::GetFrameCounter() % ProfileStride;

        ParallelFor(NumBatches, [&](int32 Batch)
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelProcessBatch);

            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FVoxelProfile Profile;

            const int32 Begin = int32(int64(TotalChunks) * Batch / NumBatches);
            const int32 End = int32(int64(TotalChunks) * (Batch + 1) / NumBatches);

            for (int32 Index = Begin; Index < End; ++Index)
            {
                const int32 CX = Index / (SGridY * SGridZ);
                const int32 Remainder = Index % (SGridY * SGridZ);
                const int32 CY = Remainder / SGridZ;
                const int32 CZ = Remainder % SGridZ;

                const FIntVector Chunk(CX, CY, CZ);

                if (bShowCalculationTime && (TotalChunks < ProfileStride || Index % ProfileStride == ProfileOffset))
                {
                    ProcessChunk<true>(Frame, ChunkSize, Chunk, Labels, Profile);
                }
                else
                {
                    ProcessChunk<false>(Frame, ChunkSize, Chunk, Labels, Profile);
                }
            }

            if (bShowCalculationTime)
            {
                OutProfiles[Batch] = Profile;
            }
        }, EParallelForFlags::Unbalanced);

        return TotalChunks;
    }

    struct FVoxelTileProfile
    {
        uint64 ZeroCycles = 0;
        uint64 GatherCycles = 0;
        uint64 CullCycles = 0;
        uint64 FetchCycles = 0;
        uint64 RasterCycles = 0;
        uint64 DeferredCycles = 0;
        uint64 PrimitiveCycles = 0;
        uint64 TieCycles = 0;
        uint64 FallbackCycles = 0;
        uint64 TotalCycles = 0;
        uint64 SlowestCycles = 0;
        uint64 Tiles = 0;
        uint64 FallbackTiles = 0;
        uint64 Candidates = 0;
        uint64 MeshCandidates = 0;
        uint64 PrimitiveCandidates = 0;
        uint64 Regrowths = 0;
        uint64 Ties = 0;
        VoxelSurfaceRaster::FRasterStats Raster;

        void Add(const FVoxelTileProfile& Other)
        {
            ZeroCycles += Other.ZeroCycles;
            GatherCycles += Other.GatherCycles;
            CullCycles += Other.CullCycles;
            FetchCycles += Other.FetchCycles;
            RasterCycles += Other.RasterCycles;
            DeferredCycles += Other.DeferredCycles;
            PrimitiveCycles += Other.PrimitiveCycles;
            TieCycles += Other.TieCycles;
            FallbackCycles += Other.FallbackCycles;
            TotalCycles += Other.TotalCycles;
            SlowestCycles = FMath::Max(SlowestCycles, Other.SlowestCycles);
            Tiles += Other.Tiles;
            FallbackTiles += Other.FallbackTiles;
            Candidates += Other.Candidates;
            MeshCandidates += Other.MeshCandidates;
            PrimitiveCandidates += Other.PrimitiveCandidates;
            Regrowths += Other.Regrowths;
            Ties += Other.Ties;

            Raster.Add(Other.Raster);
        }
    };

    // Tiles are whole multiples of the reference chunk, so legacy chunks and primitive bricks align with the tile grid.
    FIntVector GetTileSize(const FIntVector& ChunkSize)
    {
        return FIntVector(
            FMath::DivideAndRoundUp(ChunkSize.X, ReferenceChunkSize.X) * ReferenceChunkSize.X,
            FMath::DivideAndRoundUp(ChunkSize.Y, ReferenceChunkSize.Y) * ReferenceChunkSize.Y,
            FMath::DivideAndRoundUp(ChunkSize.Z, ReferenceChunkSize.Z) * ReferenceChunkSize.Z
        );
    }

    // Connects the surface raster to the output labels and to the PhysX queries of the current candidate.
    struct FTileEnv
    {
        const FVoxelFrame& Frame;
        const VoxelSurfaceRaster::FLabelRules& Rules;
        const VoxelSurfaceRaster::FLattice& Lattice;
        const physx::PxTransform& TileFrame;
        const FIntVector Begin;
        uint8* Labels;
        VoxelSurfaceRaster::FVoxelSet& Deferred;
        TArray<int32>& Ties;
        const FVoxelCandidate* Candidate;

        int32 GetFlatIndex(int32 X, int32 Y, int32 Z) const
        {
            return (Begin.X + X) * Frame.GridSizeY * Frame.GridSizeZ + (Begin.Y + Y) * Frame.GridSizeZ + Begin.Z + Z;
        }

        // Same world position as the chunked path.
        physx::PxVec3 GetWorldPosition(int32 X, int32 Y, int32 Z) const
        {
            return U2PVector(Frame.GetWorldPosition(Begin.X + X, Begin.Y + Y, Begin.Z + Z));
        }

        bool IsOpen(int32 X, int32 Y, int32 Z) const
        {
            return !Rules.IsClosed(Labels[GetFlatIndex(X, Y, Z)], Candidate->Label, Candidate->Priority);
        }

        bool GetCenter(int32 X, int32 Y, int32 Z, physx::PxVec3& Center) const
        {
            const physx::PxVec3 Position = GetWorldPosition(X, Y, Z);

            if (!Candidate->Bounds.intersects(physx::PxBounds3::centerExtents(Position, Frame.VoxelWorldExtent)))
            {
                return false;
            }

            Center = TileFrame.transformInv(Position);

            return true;
        }

        void Hit(int32 X, int32 Y, int32 Z)
        {
            const int32 FlatIndex = GetFlatIndex(X, Y, Z);

            if (Rules.Apply(Labels[FlatIndex], Candidate->Label))
            {
                Ties.Add(FlatIndex);
            }
        }

        void Defer(int32 X, int32 Y, int32 Z)
        {
            Deferred.Add(Lattice.GetIndex(X, Y, Z));
        }

        bool Overlaps(int32 X, int32 Y, int32 Z) const
        {
            const physx::PxTransform VoxelPose(GetWorldPosition(X, Y, Z), Frame.VoxelQuat);

            return physx::PxGeometryQuery::overlap(Frame.VoxelAdapter.GetGeometry(), VoxelPose, Candidate->Geometry.any(), Candidate->Pose);
        }

        bool OverlapsBrick(const int32* BrickBegin, const int32* BrickEnd) const
        {
            const FChunkBounds Bounds = Frame.MakeChunkBounds(
                FIntVector(Begin.X + BrickBegin[0], Begin.Y + BrickBegin[1], Begin.Z + BrickBegin[2]),
                FIntVector(Begin.X + BrickEnd[0], Begin.Y + BrickEnd[1], Begin.Z + BrickEnd[2])
            );

            return physx::PxGeometryQuery::overlap(Bounds.Geometry, Bounds.Pose, Candidate->Geometry.any(), Candidate->Pose);
        }
    };

    // Label every voxel of one tile with the surface raster. Gives the same labels as the chunked path, which still runs tiles the raster cannot handle exactly.
    template <bool bProfile>
    void ProcessTile(const FVoxelFrame& Frame, const VoxelSurfaceRaster::FLabelRules& Rules, const FIntVector& TileSize, const FIntVector& Tile, int32 FootprintPadding, uint8* Labels, FVoxelTileProfile& Profile)
    {
        const uint64 TileStart = bProfile ? FPlatformTime::Cycles64() : 0;

        uint64 Clock = TileStart;

        // Add the cycles since the previous lap to one phase.
        const auto Lap = [&](uint64& Cycles)
        {
            if (bProfile)
            {
                const uint64 Now = FPlatformTime::Cycles64();

                Cycles += Now - Clock;

                Clock = Now;
            }
        };

        const auto Finish = [&]()
        {
            if (bProfile)
            {
                const uint64 Cycles = FPlatformTime::Cycles64() - TileStart;

                Profile.TotalCycles += Cycles;

                Profile.SlowestCycles = FMath::Max(Profile.SlowestCycles, Cycles);

                ++Profile.Tiles;
            }
        };

        const FIntVector Begin(Tile.X * TileSize.X, Tile.Y * TileSize.Y, Tile.Z * TileSize.Z);

        const FIntVector End = Frame.GetChunkEnd(TileSize, Tile.X + 1, Tile.Y + 1, Tile.Z + 1);

        const FIntVector Size = End - Begin;

        // Labels accumulate in the output, so the tile starts empty.
        for (int32 X = Begin.X; X < End.X; ++X)
        {
            for (int32 Y = Begin.Y; Y < End.Y; ++Y)
            {
                FMemory::Memzero(Labels + X * Frame.GridSizeY * Frame.GridSizeZ + Y * Frame.GridSizeZ + Begin.Z, Size.Z);
            }
        }

        Lap(Profile.ZeroCycles);

        const FChunkBounds TileBounds = Frame.MakeChunkBounds(Begin, End);

        FVoxelCandidateCollector Collector(Frame.QueryParams, Frame.ObjectParams, Frame.Priorities, Frame.NumLabels);

        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherTileCandidates);

            // PhysX requires touch storage even though the prefilter returns eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);

            const physx::PxQueryFilterData Filter(PhysXInterface::StaticDynamicQueryFlags(Frame.QueryParams) | physx::PxQueryFlag::ePREFILTER);

            Frame.Scene->overlap(TileBounds.Geometry, TileBounds.Pose, UnusedHits, Filter, &Collector);
        }

        Lap(Profile.GatherCycles);

        // The chunked path labels this tile when the raster cannot reproduce it.
        const auto RunReferenceChunks = [&]()
        {
            FVoxelProfile UnusedProfile;

            for (int32 CX = Begin.X / ReferenceChunkSize.X; CX * ReferenceChunkSize.X < End.X; ++CX)
            {
                for (int32 CY = Begin.Y / ReferenceChunkSize.Y; CY * ReferenceChunkSize.Y < End.Y; ++CY)
                {
                    for (int32 CZ = Begin.Z / ReferenceChunkSize.Z; CZ * ReferenceChunkSize.Z < End.Z; ++CZ)
                    {
                        ProcessChunk<false>(Frame, ReferenceChunkSize, FIntVector(CX, CY, CZ), Labels, UnusedProfile);
                    }
                }
            }

            if (bProfile)
            {
                ++Profile.FallbackTiles;
            }

            Lap(Profile.FallbackCycles);

            Finish();
        };

        // Chunk gathers see a kinematic body at its target while its shapes are tested at the current pose, so their output depends on the chunk boundaries.
        if (Collector.bKinematicTarget)
        {
            RunReferenceChunks();

            return;
        }

        // Meshes are culled by their triangle fetch below.
        Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
        {
            return Candidate.Geometry.getType() != physx::PxGeometryType::eTRIANGLEMESH && !physx::PxGeometryQuery::overlap(TileBounds.Geometry, TileBounds.Pose, Candidate.Geometry.any(), Candidate.Pose);
        });

        Collector.Candidates.StableSort([](const FVoxelCandidate& A, const FVoxelCandidate& B)
        {
            return A.Priority > B.Priority;
        });

        if (bProfile)
        {
            Profile.Candidates += Collector.Candidates.Num();
        }

        Lap(Profile.CullCycles);

        if (Collector.Candidates.Num() == 0)
        {
            Finish();

            return;
        }

        const physx::PxTransform TileFrame(TileBounds.Pose.p, Frame.VoxelQuat);

        const VoxelSurfaceRaster::FLattice Lattice = Frame.MakeLattice(Begin, Size, TileFrame, FootprintPadding);

        if (!Lattice.IsValid())
        {
            RunReferenceChunks();

            return;
        }

        // Sensor-aligned box around every voxel query box of the tile, with the same 0.1 cm margin as the chunk bounds.
        physx::PxVec3 FetchExtent;

        for (int32 Axis = 0; Axis < 3; ++Axis)
        {
            const double Extent = FMath::Max(FMath::Abs(Lattice.Center(Axis, 0)), FMath::Abs(Lattice.Center(Axis, Lattice.Size[Axis] - 1)));

            FetchExtent[Axis] = float(Extent + Lattice.Margin) + Frame.VoxelHalfExtent[Axis] + 0.1f;
        }

        const physx::PxBoxGeometry FetchBox(FetchExtent);

        const int32 BrickSize[3] = {ReferenceChunkSize.X, ReferenceChunkSize.Y, ReferenceChunkSize.Z};

        // Separating axes are only built for triangles that can still change a label.
        struct FFetchedTriangle
        {
            physx::PxTriangle Triangle;
            float PositionError;
        };

        TArray<physx::PxU32> Indices;

        Indices.SetNumUninitialized(4096);

        TArray<FFetchedTriangle> Triangles;

        VoxelSurfaceRaster::FVoxelSet Deferred;

        Deferred.Reset(Lattice.GetNumVoxels());

        TArray<int32> Ties;

        FTileEnv Env{Frame, Rules, Lattice, TileFrame, Begin, Labels, Deferred, Ties, nullptr};

        // Candidates go in descending priority, so a voxel closes as soon as no later candidate can change its label.
        for (const FVoxelCandidate& Candidate : Collector.Candidates)
        {
            Env.Candidate = &Candidate;

            if (Candidate.Geometry.getType() != physx::PxGeometryType::eTRIANGLEMESH)
            {
                if (bProfile)
                {
                    ++Profile.PrimitiveCandidates;
                }

                const VoxelSurfaceRaster::FIndexBox Box = VoxelSurfaceRaster::GetBoundsBox(Candidate.Bounds, Frame.VoxelWorldExtent, TileFrame, Lattice);

                VoxelSurfaceRaster::RasterizeBounds<bProfile>(Box, BrickSize, Lattice, Env, Profile.Raster);

                Lap(Profile.PrimitiveCycles);

                continue;
            }

            if (bProfile)
            {
                ++Profile.MeshCandidates;
            }

            const physx::PxTriangleMeshGeometry& Mesh = Candidate.Geometry.triangleMesh();

            bool bOverflow = false;

            physx::PxU32 Count = physx::PxMeshQuery::findOverlapTriangleMesh(FetchBox, TileFrame, Mesh, Candidate.Pose, Indices.GetData(), physx::PxU32(Indices.Num()), 0, bOverflow);

            if (bOverflow)
            {
                // Paging with a start index reruns the midphase, so query once more with room for every triangle.
                Indices.SetNumUninitialized(FMath::Max<int32>(2 * Indices.Num(), Mesh.triangleMesh->getNbTriangles()));

                bOverflow = false;

                Count = physx::PxMeshQuery::findOverlapTriangleMesh(FetchBox, TileFrame, Mesh, Candidate.Pose, Indices.GetData(), physx::PxU32(Indices.Num()), 0, bOverflow);

                if (bProfile)
                {
                    ++Profile.Regrowths;
                }

                if (bOverflow)
                {
                    RunReferenceChunks();

                    return;
                }
            }

            Triangles.Reset();

            const physx::PxTransform MeshToTile = TileFrame.transformInv(Candidate.Pose);

            Triangles.SetNumUninitialized(Count);

            for (physx::PxU32 Index = 0; Index < Count; ++Index)
            {
                Triangles[Index].PositionError = VoxelTriangleOverlap::FetchTriangle(Mesh, Candidate.Pose, TileFrame, MeshToTile, Indices[Index], Frame.VoxelHalfExtent, Triangles[Index].Triangle);
            }

            Lap(Profile.FetchCycles);

            for (const FFetchedTriangle& Fetched : Triangles)
            {
                VoxelSurfaceRaster::RasterizeMeshTriangle<bProfile>(Fetched.Triangle, Fetched.PositionError, Frame.VoxelHalfExtent, Lattice, Env, Profile.Raster);
            }

            Lap(Profile.RasterCycles);

            VoxelSurfaceRaster::ResolveDeferred<bProfile>(Deferred, Lattice, Env, Profile.Raster);

            Lap(Profile.DeferredCycles);
        }

        // Two different labels share the highest priority; the per-voxel query keeps PhysX hit order, as in the chunked path.
        for (const int32 FlatIndex : Ties)
        {
            const int32 X = FlatIndex / (Frame.GridSizeY * Frame.GridSizeZ);
            const int32 Remainder = FlatIndex % (Frame.GridSizeY * Frame.GridSizeZ);
            const int32 Y = Remainder / Frame.GridSizeZ;
            const int32 Z = Remainder % Frame.GridSizeZ;

            Labels[FlatIndex] = Frame.QueryVoxelLabel(Frame.GetWorldPosition(X, Y, Z));
        }

        if (bProfile)
        {
            Profile.Ties += Ties.Num();
        }

        Lap(Profile.TieCycles);

        Finish();
    }

    // Label every voxel tile by tile and return the number of tiles.
    int32 RunTiles(const FVoxelFrame& Frame, const FIntVector& TileSize, int32 FootprintPadding, uint8* Labels, bool bShowCalculationTime, TArray<FVoxelTileProfile>& OutProfiles)
    {
        const int32 STilesX = FMath::DivideAndRoundUp(Frame.GridSizeX, TileSize.X);
        const int32 STilesY = FMath::DivideAndRoundUp(Frame.GridSizeY, TileSize.Y);
        const int32 STilesZ = FMath::DivideAndRoundUp(Frame.GridSizeZ, TileSize.Z);

        const int32 TotalTiles = STilesX * STilesY * STilesZ;

        const VoxelSurfaceRaster::FLabelRules Rules(Frame.Priorities, Frame.NumLabels);

        // Tile costs vary a lot, so batches hold a single tile unless tiles are tiny.
        const int32 NumBatches = FMath::Min(TotalTiles, 64 * (FTaskGraphInterface::Get().GetNumWorkerThreads() + 1));

        OutProfiles.Reset();

        if (bShowCalculationTime)
        {
            OutProfiles.SetNum(NumBatches);
        }

        ParallelFor(NumBatches, [&](int32 Batch)
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelProcessTiles);

            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FVoxelTileProfile Profile;

            const int32 Begin = int32(int64(TotalTiles) * Batch / NumBatches);
            const int32 End = int32(int64(TotalTiles) * (Batch + 1) / NumBatches);

            for (int32 Index = Begin; Index < End; ++Index)
            {
                const int32 TX = Index / (STilesY * STilesZ);
                const int32 Remainder = Index % (STilesY * STilesZ);
                const int32 TY = Remainder / STilesZ;
                const int32 TZ = Remainder % STilesZ;

                const FIntVector Tile(TX, TY, TZ);

                if (bShowCalculationTime)
                {
                    ProcessTile<true>(Frame, Rules, TileSize, Tile, FootprintPadding, Labels, Profile);
                }
                else
                {
                    ProcessTile<false>(Frame, Rules, TileSize, Tile, FootprintPadding, Labels, Profile);
                }
            }

            if (bShowCalculationTime)
            {
                OutProfiles[Batch] = Profile;
            }
        }, EParallelForFlags::Unbalanced);

        return TotalTiles;
    }

    // Fill the whole voxel buffer with one value, in parallel slices.
    void FillVoxels(uint8* Labels, int32 TotalVoxels, uint8 Value)
    {
        constexpr int32 SliceSize = 1 << 20;

        const int32 NumSlices = FMath::DivideAndRoundUp(TotalVoxels, SliceSize);

        ParallelFor(NumSlices, [&](int32 Slice)
        {
            const int32 Begin = Slice * SliceSize;

            FMemory::Memset(Labels + Begin, Value, FMath::Min(SliceSize, TotalVoxels - Begin));
        });
    }

    // Compare the output with the chunked reference on the same frame and log every difference class.
    void ValidateVoxels(const FVoxelFrame& Frame, const uint8* Labels)
    {
        const int32 TotalVoxels = Frame.GridSizeX * Frame.GridSizeY * Frame.GridSizeZ;

        TArray<uint8> Reference;

        Reference.SetNumUninitialized(TotalVoxels);

        FillVoxels(Reference.GetData(), TotalVoxels, UnwrittenVoxel);

        TArray<FVoxelProfile> UnusedProfiles;

        RunChunks(Frame, ReferenceChunkSize, Reference.GetData(), false, UnusedProfiles);

        // Labels 0 to 31 get their own bucket; every other value shares the last one.
        constexpr int32 NumBuckets = 33;
        constexpr int32 MaxExamples = 8;
        constexpr int32 SliceSize = 1 << 20;

        struct FSliceResult
        {
            uint64 Mismatches = 0;
            uint64 Unwritten = 0;
            uint64 Invalid = 0;
            uint64 ReferenceUnwritten = 0;
            uint64 Confusion[NumBuckets][NumBuckets] = {};
            int32 Examples[MaxExamples] = {};
            int32 NumExamples = 0;
        };

        const int32 NumSlices = FMath::DivideAndRoundUp(TotalVoxels, SliceSize);

        TArray<FSliceResult> Slices;

        Slices.SetNum(NumSlices);

        ParallelFor(NumSlices, [&](int32 SliceIndex)
        {
            FSliceResult& Slice = Slices[SliceIndex];

            const int32 Begin = SliceIndex * SliceSize;
            const int32 End = FMath::Min(Begin + SliceSize, TotalVoxels);

            for (int32 FlatIndex = Begin; FlatIndex < End; ++FlatIndex)
            {
                const uint8 Output = Labels[FlatIndex];
                const uint8 Expected = Reference[FlatIndex];

                Slice.Unwritten += Output == UnwrittenVoxel;
                Slice.Invalid += Output >= Frame.NumLabels && Output != UnwrittenVoxel;
                Slice.ReferenceUnwritten += Expected == UnwrittenVoxel;

                if (Output != Expected)
                {
                    ++Slice.Mismatches;

                    ++Slice.Confusion[FMath::Min<int32>(Expected, NumBuckets - 1)][FMath::Min<int32>(Output, NumBuckets - 1)];

                    if (Slice.NumExamples < MaxExamples)
                    {
                        Slice.Examples[Slice.NumExamples++] = FlatIndex;
                    }
                }
            }
        });

        FSliceResult Total;

        TArray<int32> Examples;

        for (const FSliceResult& Slice : Slices)
        {
            Total.Mismatches += Slice.Mismatches;
            Total.Unwritten += Slice.Unwritten;
            Total.Invalid += Slice.Invalid;
            Total.ReferenceUnwritten += Slice.ReferenceUnwritten;

            for (int32 Expected = 0; Expected < NumBuckets; ++Expected)
            {
                for (int32 Output = 0; Output < NumBuckets; ++Output)
                {
                    Total.Confusion[Expected][Output] += Slice.Confusion[Expected][Output];
                }
            }

            for (int32 Example = 0; Example < Slice.NumExamples && Examples.Num() < MaxExamples; ++Example)
            {
                Examples.Add(Slice.Examples[Example]);
            }
        }

        carla::log_warning("Voxel validation: ", Total.Mismatches, " mismatched voxels of ", TotalVoxels,
            " against chunks ", ReferenceChunkSize.X, "x", ReferenceChunkSize.Y, "x", ReferenceChunkSize.Z,
            "; unwritten=", Total.Unwritten, ", invalid=", Total.Invalid, ", reference unwritten=", Total.ReferenceUnwritten);

        if (Total.Mismatches == 0)
        {
            return;
        }

        // Report the most frequent label confusions first.
        TArray<FIntVector> Pairs;

        for (int32 Expected = 0; Expected < NumBuckets; ++Expected)
        {
            for (int32 Output = 0; Output < NumBuckets; ++Output)
            {
                if (Total.Confusion[Expected][Output] > 0)
                {
                    Pairs.Add(FIntVector(Expected, Output, 0));
                }
            }
        }

        Pairs.Sort([&](const FIntVector& A, const FIntVector& B)
        {
            return Total.Confusion[A.X][A.Y] > Total.Confusion[B.X][B.Y];
        });

        for (int32 Index = 0; Index < FMath::Min(Pairs.Num(), 10); ++Index)
        {
            carla::log_warning("  reference label ", Pairs[Index].X, " -> output label ", Pairs[Index].Y, ": ", Total.Confusion[Pairs[Index].X][Pairs[Index].Y]);
        }

        // The original per-voxel query tells which side matches the PhysX ground truth.
        physx::PxSceneReadLock SceneLock(*Frame.Scene);

        for (const int32 FlatIndex : Examples)
        {
            const int32 X = FlatIndex / (Frame.GridSizeY * Frame.GridSizeZ);
            const int32 Remainder = FlatIndex % (Frame.GridSizeY * Frame.GridSizeZ);
            const int32 Y = Remainder / Frame.GridSizeZ;
            const int32 Z = Remainder % Frame.GridSizeZ;

            const int32 Oracle = Frame.QueryVoxelLabel(Frame.GetWorldPosition(X, Y, Z));

            carla::log_warning("  voxel (", X, ", ", Y, ", ", Z, "): output=", int32(Labels[FlatIndex]), ", reference=", int32(Reference[FlatIndex]), ", per-voxel query=", Oracle);
        }
    }

    // Compare every Stride-th voxel with the original per-voxel query. The offset rotates so Stride frames cover the grid.
    void CheckAgainstOracle(const FVoxelFrame& Frame, const uint8* Labels, int32 Stride)
    {
        const int32 TotalVoxels = Frame.GridSizeX * Frame.GridSizeY * Frame.GridSizeZ;

        const int32 Offset = FCarlaEngine::GetFrameCounter() % Stride;

        const int32 NumSamples = TotalVoxels > Offset ? FMath::DivideAndRoundUp(TotalVoxels - Offset, Stride) : 0;

        constexpr int32 BatchSize = 4096;
        constexpr int32 MaxExamples = 8;

        struct FBatchResult
        {
            uint64 Mismatches = 0;
            int32 Examples[MaxExamples] = {};
            uint8 OracleLabels[MaxExamples] = {};
            int32 NumExamples = 0;
        };

        const int32 NumBatches = FMath::DivideAndRoundUp(NumSamples, BatchSize);

        TArray<FBatchResult> Batches;

        Batches.SetNum(NumBatches);

        ParallelFor(NumBatches, [&](int32 BatchIndex)
        {
            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FBatchResult& Batch = Batches[BatchIndex];

            const int32 End = FMath::Min((BatchIndex + 1) * BatchSize, NumSamples);

            for (int32 Sample = BatchIndex * BatchSize; Sample < End; ++Sample)
            {
                const int32 FlatIndex = Offset + Sample * Stride;

                const int32 X = FlatIndex / (Frame.GridSizeY * Frame.GridSizeZ);
                const int32 Remainder = FlatIndex % (Frame.GridSizeY * Frame.GridSizeZ);
                const int32 Y = Remainder / Frame.GridSizeZ;
                const int32 Z = Remainder % Frame.GridSizeZ;

                const uint8 Oracle = Frame.QueryVoxelLabel(Frame.GetWorldPosition(X, Y, Z));

                if (Oracle != Labels[FlatIndex])
                {
                    ++Batch.Mismatches;

                    if (Batch.NumExamples < MaxExamples)
                    {
                        Batch.Examples[Batch.NumExamples] = FlatIndex;
                        Batch.OracleLabels[Batch.NumExamples] = Oracle;

                        ++Batch.NumExamples;
                    }
                }
            }
        }, EParallelForFlags::Unbalanced);

        uint64 Mismatches = 0;

        for (const FBatchResult& Batch : Batches)
        {
            Mismatches += Batch.Mismatches;
        }

        carla::log_warning("Voxel oracle check: ", Mismatches, " mismatched voxels of ", NumSamples, " sampled (stride ", Stride, ", offset ", Offset, ")");

        int32 Logged = 0;

        for (const FBatchResult& Batch : Batches)
        {
            for (int32 Example = 0; Example < Batch.NumExamples && Logged < MaxExamples; ++Example, ++Logged)
            {
                const int32 FlatIndex = Batch.Examples[Example];

                const int32 X = FlatIndex / (Frame.GridSizeY * Frame.GridSizeZ);
                const int32 Remainder = FlatIndex % (Frame.GridSizeY * Frame.GridSizeZ);
                const int32 Y = Remainder / Frame.GridSizeZ;
                const int32 Z = Remainder % Frame.GridSizeZ;

                carla::log_warning("  voxel (", X, ", ", Y, ", ", Z, "): output=", int32(Labels[FlatIndex]), ", per-voxel query=", int32(Batch.OracleLabels[Example]));
            }
        }
    }

    // Log the sampled chunk profile.
    void LogChunkProfile(const TArray<FVoxelProfile>& Profiles, int32 TotalChunks)
    {
        FVoxelProfile Profile;

        for (const FVoxelProfile& Batch : Profiles)
        {
            Profile.GatherCycles += Batch.GatherCycles;
            Profile.CullCycles += Batch.CullCycles;
            Profile.TriangleCycles += Batch.TriangleCycles;
            Profile.FineCycles += Batch.FineCycles;
            Profile.PhysXCycles += Batch.PhysXCycles;
            Profile.TieCycles += Batch.TieCycles;
            Profile.Regions += Batch.Regions;
            Profile.Candidates += Batch.Candidates;
            Profile.Chunks += Batch.Chunks;
            Profile.Voxels += Batch.Voxels;
            Profile.Tests += Batch.Tests;
            Profile.PhysXQueries += Batch.PhysXQueries;
            Profile.Ties += Batch.Ties;
            Profile.CachedMeshes += Batch.CachedMeshes;
            Profile.CachedTriangles += Batch.CachedTriangles;
            Profile.TriangleQueries += Batch.TriangleQueries;
            Profile.BoundaryFallbacks += Batch.BoundaryFallbacks;
        }

        // These are sampled sums across workers, not frame wall time. Fine time is split into triangle tests and per-voxel PhysX queries; both exclude ties.
        carla::log_warning("Voxel sampled worker time (ms): gather=", FPlatformTime::ToMilliseconds64(Profile.GatherCycles),
            ", cull/sort=", FPlatformTime::ToMilliseconds64(Profile.CullCycles),
            ", triangle cache=", FPlatformTime::ToMilliseconds64(Profile.TriangleCycles),
            ", fine triangles=", FPlatformTime::ToMilliseconds64(Profile.FineCycles - Profile.TieCycles - Profile.PhysXCycles),
            ", fine PhysX=", FPlatformTime::ToMilliseconds64(Profile.PhysXCycles),
            ", ties=", FPlatformTime::ToMilliseconds64(Profile.TieCycles));

        carla::log_warning("Voxel sampled counts: regions=", Profile.Regions, "/", TotalChunks,
            ", candidates=", Profile.Candidates,
            ", active chunks=", Profile.Chunks,
            ", visited voxels=", Profile.Voxels,
            ", candidate tests=", Profile.Tests,
            ", PhysX voxel queries=", Profile.PhysXQueries,
            ", tie queries=", Profile.Ties);

        carla::log_warning("Voxel sampled triangle counts: cached meshes=", Profile.CachedMeshes,
            ", cached triangles=", Profile.CachedTriangles,
            ", reused queries=", Profile.TriangleQueries,
            ", boundary fallbacks=", Profile.BoundaryFallbacks);
    }

    // Log the tile profile. Every tile is profiled.
    void LogTileProfile(const TArray<FVoxelTileProfile>& Profiles, int32 TotalTiles)
    {
        FVoxelTileProfile Profile;

        for (const FVoxelTileProfile& Batch : Profiles)
        {
            Profile.Add(Batch);
        }

        const VoxelSurfaceRaster::FRasterStats& Raster = Profile.Raster;

        // Phase times are sums across workers, not frame wall time.
        carla::log_warning("Voxel tile worker time (ms): zero=", FPlatformTime::ToMilliseconds64(Profile.ZeroCycles),
            ", gather=", FPlatformTime::ToMilliseconds64(Profile.GatherCycles),
            ", cull/sort=", FPlatformTime::ToMilliseconds64(Profile.CullCycles),
            ", fetch=", FPlatformTime::ToMilliseconds64(Profile.FetchCycles),
            ", raster=", FPlatformTime::ToMilliseconds64(Profile.RasterCycles),
            ", deferred PhysX=", FPlatformTime::ToMilliseconds64(Profile.DeferredCycles),
            ", primitive PhysX=", FPlatformTime::ToMilliseconds64(Profile.PrimitiveCycles),
            ", ties=", FPlatformTime::ToMilliseconds64(Profile.TieCycles),
            ", legacy fallback=", FPlatformTime::ToMilliseconds64(Profile.FallbackCycles),
            ", total=", FPlatformTime::ToMilliseconds64(Profile.TotalCycles),
            ", slowest tile=", FPlatformTime::ToMilliseconds64(Profile.SlowestCycles));

        carla::log_warning("Voxel tile counts: tiles=", Profile.Tiles, "/", TotalTiles,
            ", legacy fallback tiles=", Profile.FallbackTiles,
            ", candidates=", Profile.Candidates,
            ", meshes=", Profile.MeshCandidates,
            ", primitives=", Profile.PrimitiveCandidates,
            ", rasterized triangles=", Raster.Triangles,
            ", skipped closed triangles=", Raster.Skipped,
            ", degenerate=", Raster.Degenerate,
            ", index regrowths=", Profile.Regrowths);

        carla::log_warning("Voxel tile voxel counts: visited=", Raster.Visited,
            ", closed=", Raster.Closed,
            ", certain=", Raster.Certain,
            ", gated=", Raster.Gated,
            ", overlap=", Raster.Overlaps,
            ", uncertain=", Raster.Uncertain,
            ", outside=", Raster.Outside,
            ", deferred queries=", Raster.DeferredQueries,
            ", deferred hits=", Raster.DeferredHits,
            ", bricks=", Raster.Bricks,
            ", culled bricks=", Raster.CulledBricks,
            ", primitive queries=", Raster.PrimitiveQueries,
            ", primitive hits=", Raster.PrimitiveHits,
            ", ties=", Profile.Ties);
    }
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

    FActorVariation ValidateOutput;
    ValidateOutput.Id = TEXT("validate_output");
    ValidateOutput.Type = EActorAttributeType::Bool;
    ValidateOutput.RecommendedValues = { TEXT("False") };
    ValidateOutput.bRestrictToRecommended = false;

    FActorVariation OracleStride;
    OracleStride.Id = TEXT("oracle_stride");
    OracleStride.Type = EActorAttributeType::Int;
    OracleStride.RecommendedValues = { TEXT("0") };
    OracleStride.bRestrictToRecommended = false;

    FActorVariation UseSurfaceRaster;
    UseSurfaceRaster.Id = TEXT("use_surface_raster");
    UseSurfaceRaster.Type = EActorAttributeType::Bool;
    UseSurfaceRaster.RecommendedValues = { TEXT("False") };
    UseSurfaceRaster.bRestrictToRecommended = false;

    FActorVariation FootprintPadding;
    FootprintPadding.Id = TEXT("footprint_padding");
    FootprintPadding.Type = EActorAttributeType::Int;
    FootprintPadding.RecommendedValues = { TEXT("0") };
    FootprintPadding.bRestrictToRecommended = false;

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
            ValidateOutput,
            OracleStride,
            UseSurfaceRaster,
            FootprintPadding,
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

    // Chunks need at least one voxel per axis.
    ChunkX = FMath::Max(1, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_x", Description.Variations, 4));

    ChunkY = FMath::Max(1, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_y", Description.Variations, 4));

    ChunkZ = FMath::Max(1, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_z", Description.Variations, 8));

    ValidateOutput = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("validate_output", Description.Variations, false);

    // Zero disables the comparison with the original per-voxel query.
    OracleStride = FMath::Max(0, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("oracle_stride", Description.Variations, 0));

    // With the surface raster, chunk_x, chunk_y and chunk_z give the tile size.
    UseSurfaceRaster = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_surface_raster", Description.Variations, false);

    // Padding only grows footprints and must not change the output.
    FootprintPadding = FMath::Max(0, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("footprint_padding", Description.Variations, 0));

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

    // Labels go straight into a pooled stream buffer. The pool keeps one or two allocations of this size per sensor.
    auto DataStream = GetDataStream(*this);

    carla::Buffer Voxels = DataStream.PopBufferFromPool();

    Voxels.reset(static_cast<carla::Buffer::size_type>(TotalVoxels));

    uint8* const Labels = Voxels.data();

    // Set up collision query.
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);

    QueryParams.bTraceComplex = UseTraceComplex;
    QueryParams.bReturnPhysicalMaterial = false;

    if (SelfIgnore && GetOwner())
    {
        QueryParams.AddIgnoredActor(GetOwner());
    }

    const FVoxelFrame Frame(
        World,
        GetTransform(),
        QueryParams,
        BoxSize,
        BoxRange,
        Bottom,
        GridSizeX,
        GridSizeY,
        GridSizeZ,
        SemanticPriority,
        UE_ARRAY_COUNT(SemanticPriority)
    );

    if (ValidateOutput)
    {
        // Poison the buffer so voxels that are never written show up in validation.
        FillVoxels(Labels, TotalVoxels, UnwrittenVoxel);
    }

    const double ParallelStart = FPlatformTime::Seconds();

    const FIntVector ChunkSize(ChunkX, ChunkY, ChunkZ);

    const FIntVector TileSize = GetTileSize(ChunkSize);

    // Configurations the raster cannot handle exactly keep the chunked path.
    const bool bSurfaceRaster = UseSurfaceRaster && Frame.SupportsRaster();

    TArray<FVoxelProfile> Profiles;

    TArray<FVoxelTileProfile> TileProfiles;

    const int32 TotalRegions = bSurfaceRaster ? RunTiles(Frame, TileSize, FootprintPadding, Labels, ShowCalculationTime, TileProfiles) : RunChunks(Frame, ChunkSize, Labels, ShowCalculationTime, Profiles);

    const double ParallelEnd = FPlatformTime::Seconds();

    if (ValidateOutput)
    {
        ValidateVoxels(Frame, Labels);
    }

    if (OracleStride > 0)
    {
        CheckAgainstOracle(Frame, Labels, OracleStride);
    }

    const double ValidateEnd = FPlatformTime::Seconds();

    // Debug visualization.
    if (DrawDebug)
    {
        const FQuat ActorQuat = GetActorQuat();

        for (int32 FlatIndex = 0; FlatIndex < TotalVoxels; ++FlatIndex)
        {
            const uint8 Label = Labels[FlatIndex];

            if (Label > 0)
            {
                // Convert flat index back to X, Y, Z.
                const int32 X = FlatIndex / (GridSizeY * GridSizeZ);
                const int32 Remainder = FlatIndex % (GridSizeY * GridSizeZ);
                const int32 Y = Remainder / GridSizeZ;
                const int32 Z = Remainder % GridSizeZ;

                FVector Pos = VoxelToWorld(X, Y, Z);

                FLinearColor Color = ColorMap.Contains(Label) ? ColorMap[Label] : FLinearColor::Green;

                DrawDebugBox(World, Pos, Frame.BoxExtent, ActorQuat, Color.ToFColor(true), false, DeltaTime * 1.1f, 0, 2.0f);
            }
        }
    }

    const double SendStart = FPlatformTime::Seconds();

    // Send data. This is what SerializeAndSend does after serializing, without copying the voxels.
    DataStream.Send(*this, carla::BufferView::CreateFrom(std::move(Voxels)));

    const double EndTime = FPlatformTime::Seconds();

    if (ShowCalculationTime)
    {
        if (bSurfaceRaster)
        {
            carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms (tiles: ", TotalRegions,
                " of ", TileSize.X, "x", TileSize.Y, "x", TileSize.Z, ", batches: ", TileProfiles.Num(), ")");
        }
        else
        {
            carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms (chunks: ", TotalRegions, ", batches: ", Profiles.Num(), ")");
        }

        carla::log_warning("Voxel wall time (ms): setup=", 1000.0 * (ParallelStart - StartTime),
            ", parallel=", 1000.0 * (ParallelEnd - ParallelStart),
            ", validate=", 1000.0 * (ValidateEnd - ParallelEnd),
            ", debug=", 1000.0 * (SendStart - ValidateEnd),
            ", send=", 1000.0 * (EndTime - SendStart));

        if (bSurfaceRaster)
        {
            LogTileProfile(TileProfiles, TotalRegions);
        }
        else
        {
            LogChunkProfile(Profiles, TotalRegions);
        }
    }
}
