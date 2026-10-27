#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla/Sensor/VoxelSurfaceRaster.h"
#include "Carla/Sensor/VoxelTriangleOverlap.h"
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

#include <PxRigidDynamic.h>
#include <PxScene.h>
#include <PxSceneLock.h>
#include <geometry/PxGeometryQuery.h>
#include <geometry/PxHeightField.h>
#include <geometry/PxMeshQuery.h>
#include <extensions/PxShapeExt.h>

// Voxels are labeled tile by tile. Each tile gathers its candidate shapes with one scene query, then every
// triangle of a candidate mesh visits only the voxels near its surface (see VoxelSurfaceRaster.h). Labels
// accumulate in the output buffer, so candidates are applied from the highest priority down.
namespace
{
    // Shapes without triangles are queried brick by brick, and tiles are whole multiples of a brick.
    const FIntVector BrickSize(4, 4, 8);

    // Pose of a shape as PhysX scene queries see it. Unreal moves kinematic bodies by setting a target, and queries treat
    // that target as the current pose, while the simulated pose only reaches it at the next physics step.
    physx::PxTransform GetQueryPose(const physx::PxShape& Shape, const physx::PxRigidActor& Actor)
    {
        if (const physx::PxRigidDynamic* Dynamic = Actor.is<physx::PxRigidDynamic>())
        {
            const physx::PxRigidBodyFlags Flags = Dynamic->getRigidBodyFlags();

            physx::PxTransform Target;

            if ((Flags & physx::PxRigidBodyFlag::eKINEMATIC) && (Flags & physx::PxRigidBodyFlag::eUSE_KINEMATIC_TARGET_FOR_SCENE_QUERIES) && Dynamic->getKinematicTarget(Target))
            {
                return Target * Shape.getLocalPose();
            }
        }

        return physx::PxShapeExt::getGlobalPose(Shape, Actor);
    }

    struct FVoxelCandidate
    {
        physx::PxGeometryHolder Geometry;
        physx::PxTransform Pose;
        physx::PxBounds3 Bounds;
        uint8 Label;
        uint8 Priority;

        FVoxelCandidate(const physx::PxShape& Shape, const physx::PxRigidActor& Actor, uint8 InLabel, uint8 InPriority)
            : Geometry(Shape.getGeometry())
            , Pose(GetQueryPose(Shape, Actor))
            , Bounds(physx::PxGeometryQuery::getWorldBounds(Geometry.any(), Pose))
            , Label(InLabel)
            , Priority(InPriority)
        {
        }
    };

    // Triangle meshes and heightfields go through the surface raster; other shapes are queried voxel by voxel.
    bool HasTriangles(const FVoxelCandidate& Candidate)
    {
        const physx::PxGeometryType::Enum Type = Candidate.Geometry.getType();

        return Type == physx::PxGeometryType::eTRIANGLEMESH || Type == physx::PxGeometryType::eHEIGHTFIELD;
    }

    int32 GetNumTriangles(const FVoxelCandidate& Candidate)
    {
        if (Candidate.Geometry.getType() == physx::PxGeometryType::eHEIGHTFIELD)
        {
            const physx::PxHeightField& HeightField = *Candidate.Geometry.heightField().heightField;

            const int64 Cells = int64(FMath::Max(1u, HeightField.getNbRows()) - 1) * (FMath::Max(1u, HeightField.getNbColumns()) - 1);

            return int32(FMath::Min<int64>(2 * Cells, MAX_int32));
        }

        return int32(Candidate.Geometry.triangleMesh().triangleMesh->getNbTriangles());
    }

    // Match Engine's private object-query filter using public PhysX data.
    class FVoxelCandidateCollector final : public physx::PxQueryFilterCallback
    {
    public:
        TArray<FVoxelCandidate, TInlineAllocator<16>> Candidates;

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
                }
            }

            // Continue gathering candidates without running narrow-phase tests.
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

    // World-space box for PhysX queries.
    struct FQueryBox
    {
        physx::PxBoxGeometry Geometry;
        physx::PxTransform Pose;
    };

    // Query setup and grid geometry shared by every tile of one frame.
    struct FVoxelFrame
    {
        UWorld* World;
        physx::PxScene* Scene;
        FCollisionQueryParams QueryParams;
        FCollisionObjectQueryParams ObjectParams;
        FTransform SensorTransform;
        FQuat VoxelRotation;
        // Voxel query boxes keep this extent whatever the sensor's scale, while centers follow the full sensor transform. A
        // scaled sensor therefore spaces boxes of BoxSize by its scale, as the per-voxel engine query does. The API spawns
        // sensors with unit scale, which gives a contiguous grid; only a scaled parent actor changes it.
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

            // Engine turns nearly empty boxes into spheres.
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

        // The tiles need box voxels, a finite sensor transform and a sensor scale they can invert.
        bool IsSupported() const
        {
            return bBoxVoxels && !SensorTransform.ContainsNaN() && VoxelQuat.isFinite() && SensorScale.GetAbsMin() > 1e-3f && FMath::IsFinite(CenterError);
        }

        // Center of voxel (X, Y, Z) in world space.
        FVector GetWorldPosition(int32 X, int32 Y, int32 Z) const
        {
            const FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

            return SensorTransform.TransformPosition(LocalPos);
        }

        // Exclusive end voxel of block Index in a grid of blocks of the given size, clamped to the voxel grid.
        FIntVector GetBlockEnd(const FIntVector& Size, const FIntVector& Index) const
        {
            return FIntVector(
                int32(FMath::Min<int64>(int64(Index.X + 1) * Size.X, GridSizeX)),
                int32(FMath::Min<int64>(int64(Index.Y + 1) * Size.Y, GridSizeY)),
                int32(FMath::Min<int64>(int64(Index.Z + 1) * Size.Z, GridSizeZ))
            );
        }

        // Axis-aligned world box that encloses every voxel query between Begin and End.
        FQueryBox MakeQueryBox(const FIntVector& Begin, const FIntVector& End) const
        {
            const FVector LocalMin(-BoxRange + Begin.X * BoxSize, -BoxRange + Begin.Y * BoxSize, Bottom + Begin.Z * BoxSize);
            const FVector LocalMax(-BoxRange + End.X * BoxSize, -BoxRange + End.Y * BoxSize, Bottom + End.Z * BoxSize);

            const FVector LocalHalfExtent = (LocalMax - LocalMin) * 0.5f;

            const FVector Center = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

            // Enclose voxel centers and query extents, including sensor scale.
            const FVector CenterHalfExtent = LocalHalfExtent - BoxExtent;

            const FVector WorldExtent = AbsAxisX * CenterHalfExtent.X + AbsAxisY * CenterHalfExtent.Y + AbsAxisZ * CenterHalfExtent.Z + P2UVector(VoxelWorldExtent) + FVector(0.1f);

            return FQueryBox{physx::PxBoxGeometry(U2PVector(WorldExtent)), physx::PxTransform(U2PVector(Center))};
        }

        // Approximate centers of the voxels [Begin, Begin + Size) in the tile frame.
        VoxelSurfaceRaster::FLattice MakeLattice(const FIntVector& Begin, const FIntVector& Size, const physx::PxTransform& TileFrame) const
        {
            const int32 LatticeSize[3] = {Size.X, Size.Y, Size.Z};

            const double Step[3] = {double(SensorScale.X) * BoxSize, double(SensorScale.Y) * BoxSize, double(SensorScale.Z) * BoxSize};

            return VoxelSurfaceRaster::MakeLattice(LatticeSize, Step, CenterError, 0.0, [&](int32 X, int32 Y, int32 Z)
            {
                return TileFrame.transformInv(U2PVector(GetWorldPosition(Begin.X + X, Begin.Y + Y, Begin.Z + Z)));
            });
        }

        // Label of one voxel from a per-voxel engine query; PhysX hit order breaks equal-priority ties.
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

    // Phase times and counts summed over tiles, for show_calculation_time.
    struct FVoxelProfile
    {
        uint64 ZeroCycles = 0;
        uint64 GatherCycles = 0;
        uint64 CullCycles = 0;
        uint64 FetchCycles = 0;
        uint64 RasterCycles = 0;
        uint64 DeferredCycles = 0;
        uint64 PrimitiveCycles = 0;
        uint64 TieCycles = 0;
        uint64 TotalCycles = 0;
        uint64 SlowestCycles = 0;
        uint64 Tiles = 0;
        uint64 Candidates = 0;
        uint64 MeshCandidates = 0;
        uint64 HeightFieldCandidates = 0;
        uint64 PrimitiveCandidates = 0;
        uint64 Regrowths = 0;
        uint64 Ties = 0;
        VoxelSurfaceRaster::FRasterStats Raster;

        void Add(const FVoxelProfile& Other)
        {
            ZeroCycles += Other.ZeroCycles;
            GatherCycles += Other.GatherCycles;
            CullCycles += Other.CullCycles;
            FetchCycles += Other.FetchCycles;
            RasterCycles += Other.RasterCycles;
            DeferredCycles += Other.DeferredCycles;
            PrimitiveCycles += Other.PrimitiveCycles;
            TieCycles += Other.TieCycles;
            TotalCycles += Other.TotalCycles;
            SlowestCycles = FMath::Max(SlowestCycles, Other.SlowestCycles);
            Tiles += Other.Tiles;
            Candidates += Other.Candidates;
            MeshCandidates += Other.MeshCandidates;
            HeightFieldCandidates += Other.HeightFieldCandidates;
            PrimitiveCandidates += Other.PrimitiveCandidates;
            Regrowths += Other.Regrowths;
            Ties += Other.Ties;

            Raster.Add(Other.Raster);
        }
    };

    // Tiles are whole multiples of a brick, so bricks align with the tile grid. A tile larger than the grid covers it just
    // the same, so requested sizes are clamped to the grid first, and the rounding runs in 64 bits so it cannot overflow.
    FIntVector GetTileSize(const FIntVector& Requested, const FIntVector& GridSize)
    {
        const auto Round = [](int32 Tile, int32 Grid, int32 Brick)
        {
            const int64 Clamped = FMath::Clamp<int64>(Tile, 1, FMath::Max(1, Grid));

            return int32(FMath::Min<int64>(FMath::DivideAndRoundUp<int64>(Clamped, Brick) * Brick, MAX_int32 / Brick * Brick));
        };

        return FIntVector(
            Round(Requested.X, GridSize.X, BrickSize.X),
            Round(Requested.Y, GridSize.Y, BrickSize.Y),
            Round(Requested.Z, GridSize.Z, BrickSize.Z)
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
            const FQueryBox Box = Frame.MakeQueryBox(
                FIntVector(Begin.X + BrickBegin[0], Begin.Y + BrickBegin[1], Begin.Z + BrickBegin[2]),
                FIntVector(Begin.X + BrickEnd[0], Begin.Y + BrickEnd[1], Begin.Z + BrickEnd[2])
            );

            return physx::PxGeometryQuery::overlap(Box.Geometry, Box.Pose, Candidate->Geometry.any(), Candidate->Pose);
        }
    };

    // Label every voxel of one tile. Separate instantiations keep profiling clocks out of normal frames.
    template <bool bProfile>
    void ProcessTile(const FVoxelFrame& Frame, const VoxelSurfaceRaster::FLabelRules& Rules, const FIntVector& TileSize, const FIntVector& Tile, uint8* Labels, FVoxelProfile& Profile)
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

        const FIntVector End = Frame.GetBlockEnd(TileSize, Tile);

        const FIntVector Size = End - Begin;

        const FQueryBox TileBox = Frame.MakeQueryBox(Begin, End);

        FVoxelCandidateCollector Collector(Frame.QueryParams, Frame.ObjectParams, Frame.Priorities, Frame.NumLabels);

        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherCandidates);

            // PhysX requires touch storage even though the prefilter returns eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);

            const physx::PxQueryFilterData Filter(PhysXInterface::StaticDynamicQueryFlags(Frame.QueryParams) | physx::PxQueryFlag::ePREFILTER);

            Frame.Scene->overlap(TileBox.Geometry, TileBox.Pose, UnusedHits, Filter, &Collector);
        }

        Lap(Profile.GatherCycles);

        // Labels accumulate in the output, so the tile starts empty.
        for (int32 X = Begin.X; X < End.X; ++X)
        {
            for (int32 Y = Begin.Y; Y < End.Y; ++Y)
            {
                FMemory::Memzero(Labels + X * Frame.GridSizeY * Frame.GridSizeZ + Y * Frame.GridSizeZ + Begin.Z, Size.Z);
            }
        }

        Lap(Profile.ZeroCycles);

        // Meshes and heightfields are culled by their triangle fetch below.
        Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
        {
            return !HasTriangles(Candidate) && !physx::PxGeometryQuery::overlap(TileBox.Geometry, TileBox.Pose, Candidate.Geometry.any(), Candidate.Pose);
        });

        // Equal priorities need no order: the same label is closed to itself, and different labels tie and go to the engine query.
        Collector.Candidates.Sort([](const FVoxelCandidate& A, const FVoxelCandidate& B)
        {
            return A.Priority > B.Priority;
        });

        if (bProfile)
        {
            Profile.Candidates += Collector.Candidates.Num();
        }

        Lap(Profile.CullCycles);

        const physx::PxTransform TileFrame(TileBox.Pose.p, Frame.VoxelQuat);

        const VoxelSurfaceRaster::FLattice Lattice = Frame.MakeLattice(Begin, Size, TileFrame);

        // An invalid lattice only comes from a non-finite transform, which leaves nothing to label.
        if (Collector.Candidates.Num() == 0 || !Lattice.IsValid())
        {
            Finish();

            return;
        }

        // Sensor-aligned box around every voxel query box of the tile, with the same 0.1 cm margin as the query boxes.
        physx::PxVec3 FetchExtent;

        for (int32 Axis = 0; Axis < 3; ++Axis)
        {
            const double Extent = FMath::Max(FMath::Abs(Lattice.Center(Axis, 0)), FMath::Abs(Lattice.Center(Axis, Lattice.Size[Axis] - 1)));

            FetchExtent[Axis] = float(Extent + Lattice.Margin) + Frame.VoxelHalfExtent[Axis] + 0.1f;
        }

        const physx::PxBoxGeometry FetchBox(FetchExtent);

        const int32 BrickDimensions[3] = {BrickSize.X, BrickSize.Y, BrickSize.Z};

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

            if (!HasTriangles(Candidate))
            {
                if (bProfile)
                {
                    ++Profile.PrimitiveCandidates;
                }

                const VoxelSurfaceRaster::FIndexBox Box = VoxelSurfaceRaster::GetBoundsBox(Candidate.Bounds, Frame.VoxelWorldExtent, TileFrame, Lattice);

                VoxelSurfaceRaster::RasterizeBounds<bProfile>(Box, BrickDimensions, Lattice, Env, Profile.Raster);

                Lap(Profile.PrimitiveCycles);

                continue;
            }

            const bool bHeightField = Candidate.Geometry.getType() == physx::PxGeometryType::eHEIGHTFIELD;

            if (bProfile && bHeightField)
            {
                ++Profile.HeightFieldCandidates;
            }
            else if (bProfile)
            {
                ++Profile.MeshCandidates;
            }

            const physx::PxTransform MeshToTile = TileFrame.transformInv(Candidate.Pose);

            // PhysX treats a thin layer next to a heightfield as solid, so its fetch box grows by that skin.
            VoxelSurfaceRaster::FSurface Surface;

            if (bHeightField)
            {
                Surface = VoxelSurfaceRaster::MakeSurface(MeshToTile.q, Candidate.Geometry.heightField().heightField->getThickness());
            }

            const physx::PxBoxGeometry CandidateFetchBox(FetchBox.halfExtents + physx::PxVec3(Surface.Skin));

            // Every query reruns the midphase, so a full buffer is regrown to hold every triangle and queried once more.
            const auto FetchIndices = [&](bool& bOverflow)
            {
                if (bHeightField)
                {
                    return physx::PxMeshQuery::findOverlapHeightField(CandidateFetchBox, TileFrame, Candidate.Geometry.heightField(), Candidate.Pose, Indices.GetData(), physx::PxU32(Indices.Num()), 0, bOverflow);
                }

                return physx::PxMeshQuery::findOverlapTriangleMesh(CandidateFetchBox, TileFrame, Candidate.Geometry.triangleMesh(), Candidate.Pose, Indices.GetData(), physx::PxU32(Indices.Num()), 0, bOverflow);
            };

            bool bOverflow = false;

            physx::PxU32 Count = FetchIndices(bOverflow);

            if (bOverflow)
            {
                Indices.SetNumUninitialized(FMath::Max<int32>(2 * Indices.Num(), GetNumTriangles(Candidate)));

                Count = FetchIndices(bOverflow);

                if (bProfile)
                {
                    ++Profile.Regrowths;
                }
            }

            Triangles.SetNumUninitialized(Count);

            for (physx::PxU32 Index = 0; Index < Count; ++Index)
            {
                FFetchedTriangle& Fetched = Triangles[Index];

                if (bHeightField)
                {
                    Fetched.PositionError = VoxelTriangleOverlap::FetchTriangle(Candidate.Geometry.heightField(), Candidate.Pose, TileFrame, MeshToTile, Indices[Index], Frame.VoxelHalfExtent, Fetched.Triangle);
                }
                else
                {
                    Fetched.PositionError = VoxelTriangleOverlap::FetchTriangle(Candidate.Geometry.triangleMesh(), Candidate.Pose, TileFrame, MeshToTile, Indices[Index], Frame.VoxelHalfExtent, Fetched.Triangle);
                }
            }

            Lap(Profile.FetchCycles);

            for (const FFetchedTriangle& Fetched : Triangles)
            {
                VoxelSurfaceRaster::RasterizeMeshTriangle<bProfile>(Fetched.Triangle, Fetched.PositionError, Frame.VoxelHalfExtent, Lattice, Env, Profile.Raster, bHeightField ? &Surface : nullptr);
            }

            Lap(Profile.RasterCycles);

            VoxelSurfaceRaster::ResolveDeferred<bProfile>(Deferred, Lattice, Env, Profile.Raster);

            Lap(Profile.DeferredCycles);
        }

        // Two different labels share the highest priority; the per-voxel query keeps PhysX hit order.
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
    int32 RunTiles(const FVoxelFrame& Frame, const FIntVector& TileSize, uint8* Labels, bool bShowCalculationTime, TArray<FVoxelProfile>& OutProfiles)
    {
        // In 64 bits, since a grid size plus a tile size can exceed an int32. Each count is at most its grid size.
        const int32 TilesX = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeX, TileSize.X));
        const int32 TilesY = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeY, TileSize.Y));
        const int32 TilesZ = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeZ, TileSize.Z));

        const int32 TotalTiles = TilesX * TilesY * TilesZ;

        const VoxelSurfaceRaster::FLabelRules Rules(Frame.Priorities, Frame.NumLabels);

        // Tile costs vary a lot, so batches hold a single tile unless tiles are tiny. PhysX read locks share a mutex,
        // so each batch takes one.
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

            FVoxelProfile Profile;

            const int32 Begin = int32(int64(TotalTiles) * Batch / NumBatches);
            const int32 End = int32(int64(TotalTiles) * (Batch + 1) / NumBatches);

            for (int32 Index = Begin; Index < End; ++Index)
            {
                const int32 TX = Index / (TilesY * TilesZ);
                const int32 Remainder = Index % (TilesY * TilesZ);
                const int32 TY = Remainder / TilesZ;
                const int32 TZ = Remainder % TilesZ;

                const FIntVector Tile(TX, TY, TZ);

                if (bShowCalculationTime)
                {
                    ProcessTile<true>(Frame, Rules, TileSize, Tile, Labels, Profile);
                }
                else
                {
                    ProcessTile<false>(Frame, Rules, TileSize, Tile, Labels, Profile);
                }
            }

            if (bShowCalculationTime)
            {
                OutProfiles[Batch] = Profile;
            }
        }, EParallelForFlags::Unbalanced);

        return TotalTiles;
    }

    // Compare every Stride-th voxel with the per-voxel engine query. The offset rotates so Stride frames cover the grid.
    void CheckAgainstOracle(const FVoxelFrame& Frame, const uint8* Labels, int32 Stride)
    {
        const int32 TotalVoxels = Frame.GridSizeX * Frame.GridSizeY * Frame.GridSizeZ;

        const int32 Offset = FCarlaEngine::GetFrameCounter() % Stride;

        // Counts and batch ends are computed in 64 bits, since a count plus the stride or batch size can exceed an int32.
        const int32 NumSamples = TotalVoxels > Offset ? int32(FMath::DivideAndRoundUp<int64>(TotalVoxels - Offset, Stride)) : 0;

        constexpr int32 BatchSize = 4096;
        constexpr int32 MaxExamples = 8;

        struct FBatchResult
        {
            uint64 Mismatches = 0;
            int32 Examples[MaxExamples] = {};
            uint8 OracleLabels[MaxExamples] = {};
            int32 NumExamples = 0;
        };

        const int32 NumBatches = int32(FMath::DivideAndRoundUp<int64>(NumSamples, BatchSize));

        TArray<FBatchResult> Batches;

        Batches.SetNum(NumBatches);

        ParallelFor(NumBatches, [&](int32 BatchIndex)
        {
            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FBatchResult& Batch = Batches[BatchIndex];

            const int32 End = int32(FMath::Min<int64>(int64(BatchIndex + 1) * BatchSize, NumSamples));

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

    void LogProfile(const TArray<FVoxelProfile>& Profiles, int32 TotalTiles)
    {
        FVoxelProfile Profile;

        for (const FVoxelProfile& Batch : Profiles)
        {
            Profile.Add(Batch);
        }

        const VoxelSurfaceRaster::FRasterStats& Raster = Profile.Raster;

        // Phase times are sums across workers, not frame wall time.
        carla::log_warning("Voxel worker time (ms): zero=", FPlatformTime::ToMilliseconds64(Profile.ZeroCycles),
            ", gather=", FPlatformTime::ToMilliseconds64(Profile.GatherCycles),
            ", cull/sort=", FPlatformTime::ToMilliseconds64(Profile.CullCycles),
            ", fetch=", FPlatformTime::ToMilliseconds64(Profile.FetchCycles),
            ", raster=", FPlatformTime::ToMilliseconds64(Profile.RasterCycles),
            ", deferred PhysX=", FPlatformTime::ToMilliseconds64(Profile.DeferredCycles),
            ", primitive PhysX=", FPlatformTime::ToMilliseconds64(Profile.PrimitiveCycles),
            ", ties=", FPlatformTime::ToMilliseconds64(Profile.TieCycles),
            ", total=", FPlatformTime::ToMilliseconds64(Profile.TotalCycles),
            ", slowest tile=", FPlatformTime::ToMilliseconds64(Profile.SlowestCycles));

        carla::log_warning("Voxel tile counts: tiles=", TotalTiles,
            ", processed=", Profile.Tiles,
            ", candidates=", Profile.Candidates,
            ", meshes=", Profile.MeshCandidates,
            ", heightfields=", Profile.HeightFieldCandidates,
            ", primitives=", Profile.PrimitiveCandidates,
            ", rasterized triangles=", Raster.Triangles,
            ", skipped closed triangles=", Raster.Skipped,
            ", degenerate=", Raster.Degenerate,
            ", index regrowths=", Profile.Regrowths);

        carla::log_warning("Voxel counts: visited=", Raster.Visited,
            ", closed=", Raster.Closed,
            ", certain=", Raster.Certain,
            ", gated=", Raster.Gated,
            ", overlap=", Raster.Overlaps,
            ", heightfield voxels left to PhysX=", Raster.Unconfirmed,
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

    FActorVariation TileX;
    TileX.Id = TEXT("tile_x");
    TileX.Type = EActorAttributeType::Int;
    TileX.RecommendedValues = { TEXT("32") };
    TileX.bRestrictToRecommended = false;

    FActorVariation TileY;
    TileY.Id = TEXT("tile_y");
    TileY.Type = EActorAttributeType::Int;
    TileY.RecommendedValues = { TEXT("32") };
    TileY.bRestrictToRecommended = false;

    FActorVariation TileZ;
    TileZ.Id = TEXT("tile_z");
    TileZ.Type = EActorAttributeType::Int;
    TileZ.RecommendedValues = { TEXT("128") };
    TileZ.bRestrictToRecommended = false;

    FActorVariation OracleStride;
    OracleStride.Id = TEXT("oracle_stride");
    OracleStride.Type = EActorAttributeType::Int;
    OracleStride.RecommendedValues = { TEXT("0") };
    OracleStride.bRestrictToRecommended = false;

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
            TileX,
            TileY,
            TileZ,
            OracleStride,
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

    // Voxels are processed in tiles of this size, rounded up to whole 4 x 4 x 8 bricks.
    TileX = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("tile_x", Description.Variations, 32);

    TileY = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("tile_y", Description.Variations, 32);

    TileZ = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("tile_z", Description.Variations, 128);

    // Zero disables the comparison with the per-voxel engine query.
    OracleStride = FMath::Max(0, UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("oracle_stride", Description.Variations, 0));

    // Voxels along one axis, or zero unless the count is positive and at most 2^31. The count is checked before its integer
    // conversion, so infinite, NaN and huge counts cannot overflow.
    const auto GetGridSize = [](float Length, float Size)
    {
        const float Count = Length / Size;

        return (Count > 0.0f && Count <= float(MAX_int32)) ? int64(FMath::CeilToDouble(Count)) : int64(0);
    };

    const int64 SizeX = GetGridSize(2.0f * BoxRange, BoxSize);
    const int64 SizeY = GetGridSize(2.0f * BoxRange, BoxSize);
    const int64 SizeZ = GetGridSize(Top - Bottom, BoxSize);

    // Voxels are indexed with 32-bit integers and sent as one buffer, so larger grids would overflow. Each size is at most
    // 2^31, so SizeX * SizeY fits an int64, and dividing the limit avoids the last multiplication. Send empty frames instead.
    const bool bValidGrid = SizeX > 0 && SizeY > 0 && SizeZ > 0 && SizeX * SizeY <= MAX_int32 / SizeZ;

    GridSizeX = bValidGrid ? int32(SizeX) : 0;
    GridSizeY = bValidGrid ? int32(SizeY) : 0;
    GridSizeZ = bValidGrid ? int32(SizeZ) : 0;

    if (!bValidGrid)
    {
        UE_LOG(
            LogCarla,
            Error,
            TEXT("VoxelDetectionSensor: range %g m, limits %g to %g m and voxel size %g m do not give a grid of 1 to %d voxels. Increase voxel_size or reduce the range or height."),
            BoxRange / M_TO_CM,
            Bottom / M_TO_CM,
            Top / M_TO_CM,
            BoxSize / M_TO_CM,
            MAX_int32
        );
    }

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

    const FIntVector TileSize = GetTileSize(FIntVector(TileX, TileY, TileZ), FIntVector(GridSizeX, GridSizeY, GridSizeZ));

    const double ParallelStart = FPlatformTime::Seconds();

    TArray<FVoxelProfile> Profiles;

    int32 TotalTiles = 0;

    if (Frame.IsSupported())
    {
        TotalTiles = RunTiles(Frame, TileSize, Labels, ShowCalculationTime, Profiles);
    }
    else
    {
        // A collapsed or non-finite sensor transform, or a voxel size of almost zero, has no meaningful grid.
        FMemory::Memzero(Labels, TotalVoxels);

        if (!bReportedUnsupported)
        {
            UE_LOG(LogCarla, Warning, TEXT("VoxelDetectionSensor: unsupported sensor scale, transform or voxel size; sending empty grids."));

            bReportedUnsupported = true;
        }
    }

    const double ParallelEnd = FPlatformTime::Seconds();

    if (OracleStride > 0)
    {
        CheckAgainstOracle(Frame, Labels, OracleStride);
    }

    const double OracleEnd = FPlatformTime::Seconds();

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

                FVector Pos = Frame.GetWorldPosition(X, Y, Z);

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
        carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms (tiles: ", TotalTiles,
            " of ", TileSize.X, "x", TileSize.Y, "x", TileSize.Z, ", batches: ", Profiles.Num(), ")");

        carla::log_warning("Voxel wall time (ms): setup=", 1000.0 * (ParallelStart - StartTime),
            ", parallel=", 1000.0 * (ParallelEnd - ParallelStart),
            ", oracle=", 1000.0 * (OracleEnd - ParallelEnd),
            ", debug=", 1000.0 * (SendStart - OracleEnd),
            ", send=", 1000.0 * (EndTime - SendStart));

        LogProfile(Profiles, TotalTiles);
    }
}
