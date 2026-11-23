#include "Carla.h"

#include "Carla/Game/Tagger.h"
#include "Carla/Game/CarlaEngine.h"
#include "Carla/Game/CarlaEpisode.h"

#include "Carla/Sensor/VoxelGpuApi.h"
#include "Carla/Sensor/VoxelSurfaceRaster.h"
#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla/Sensor/VoxelTriangleOverlap.h"

#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"

#include "DrawDebugHelpers.h"
#include "CustomPhysXPayload.h"
#include "CollisionQueryParams.h"
#include "PhysXInterfaceWrapper.h"
#include "PhysicsInterfaceUtilsCore.h"

#include "Physics/PhysicsFiltering.h"

#include "PhysicsEngine/BodyInstance.h"

#include "Engine/World.h"
#include "Misc/Paths.h"
#include "HAL/PlatformProcess.h"
#include "HAL/PlatformMemory.h"

#include "Runtime/Core/Public/Async/ParallelFor.h"

#include <PxScene.h>
#include <PxSceneLock.h>
#include <PxRigidDynamic.h>

#include <geometry/PxMeshQuery.h>
#include <geometry/PxConvexMesh.h>
#include <geometry/PxTriangleMesh.h>
#include <geometry/PxHeightField.h>
#include <geometry/PxGeometryQuery.h>

#include <extensions/PxShapeExt.h>

#include <array>
#include <cmath>
#include <type_traits>

// Voxels are labeled tile by tile. Each tile gathers its candidate shapes
// with one scene query, then uses exact voxel queries or visits the voxels
// near each triangle's surface (see VoxelSurfaceRaster.h). Labels accumulate
// in the output buffer, so candidates are applied from the highest priority
// down. Labels that tell instances apart also record which actor a voxel
// belongs to.
namespace
{
    // A voxel of such a label and the identifier of its actor, as they are
    // sent after the labels: two little-endian 64-bit integers, so the index
    // covers grids of more than 2^32 voxels.
    struct FVoxelInstance
    {
        uint64 Index;
        uint64 Instance;
    };

    struct FVoxelResult
    {
        uint8 Label = 0;
        uint16 Instance = 0;
    };

    // Low 16 bits of the CARLA actor ID, or the Unreal unique ID for an
    // unregistered actor, matching the tagger's normal runtime assignment.
    uint16 GetInstance(const UPrimitiveComponent& Component, const UCarlaEpisode& Episode)
    {
        AActor* Owner = Component.GetOwner();

        if (!Owner)
            return 0;

        const FCarlaActor* CarlaActor = Episode.FindCarlaActor(Owner);

        return uint16(CarlaActor ? CarlaActor->GetActorId() : Owner->GetUniqueID());
    }

    // Shapes without triangles are queried brick by brick, and tiles are
    // whole multiples of a brick.
    const FIntVector BrickSize(4, 4, 4);

    // Pose of a shape as PhysX scene queries see it. Unreal moves kinematic
    // bodies by setting a target, and queries treat that target as the
    // current pose, while the simulated pose only reaches it at the next
    // physics step.
    physx::PxTransform GetQueryPose(const physx::PxShape& Shape, const physx::PxRigidActor& Actor)
    {
        if (const physx::PxRigidDynamic* Dynamic = Actor.is<physx::PxRigidDynamic>())
        {
            const physx::PxRigidBodyFlags Flags = Dynamic->getRigidBodyFlags();

            physx::PxTransform Target;

            if (
                (Flags & physx::PxRigidBodyFlag::eKINEMATIC) &&
                (Flags & physx::PxRigidBodyFlag::eUSE_KINEMATIC_TARGET_FOR_SCENE_QUERIES) &&
                Dynamic->getKinematicTarget(Target)
            )
                return Target * Shape.getLocalPose();
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
        uint16 Instance;
        bool bInstanced;

        // The actor can move or deform, as vehicles and walkers do.
        bool bDynamic;

        FVoxelCandidate(
            const physx::PxShape& Shape,
            const physx::PxRigidActor& Actor,
            uint8 InLabel,
            uint8 InPriority,
            uint16 InInstance,
            bool bInInstanced
        )
            : Geometry(Shape.getGeometry())
            , Pose(GetQueryPose(Shape, Actor))
            , Bounds(physx::PxGeometryQuery::getWorldBounds(Geometry.any(), Pose))
            , Label(InLabel)
            , Priority(InPriority)
            , Instance(InInstance)
            , bInstanced(bInInstanced)
            , bDynamic(Actor.is<physx::PxRigidDynamic>() != nullptr)
        {
        }
    };

    // Triangle meshes and heightfields can use the surface raster; other
    // shapes are queried voxel by voxel.
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

            const int64 Rows = int64(FMath::Max(1u, HeightField.getNbRows()) - 1);
            const int64 Columns = int64(FMath::Max(1u, HeightField.getNbColumns()) - 1);

            return int32(FMath::Min<int64>(2 * Rows * Columns, MAX_int32));
        }

        return int32(Candidate.Geometry.triangleMesh().triangleMesh->getNbTriangles());
    }

    // Match Engine's private object-query filter using public PhysX data.
    class FVoxelCandidateCollector final : public physx::PxQueryFilterCallback
    {
    public:
        TArray<FVoxelCandidate, TInlineAllocator<16>> Candidates;

        FVoxelCandidateCollector(
            const FCollisionQueryParams& InParams,
            const FCollisionObjectQueryParams& InObjects,
            const uint8* InPriorities,
            int32 InNumLabels,
            uint32 InInstancedLabels,
            const UCarlaEpisode& InEpisode
        )
            : Params(InParams)
            , Objects(InObjects)
            , Priorities(InPriorities)
            , NumLabels(InNumLabels)
            , InstancedLabels(InInstancedLabels)
            , Episode(InEpisode)
        {
        }

        physx::PxQueryHitType::Enum preFilter(
            const physx::PxFilterData&,
            const physx::PxShape* Shape,
            const physx::PxRigidActor* Actor,
            physx::PxHitFlags&
        ) override
        {
            const physx::PxFilterData Filter = Shape->getQueryFilterData();

            const uint32 Complexity = Params.bTraceComplex ? EPDF_ComplexCollision : EPDF_SimpleCollision;

            FMaskFilter ShapeMask;

            const ECollisionChannel Channel = GetCollisionChannelAndExtraFilter(Filter.word3, ShapeMask);

            if (
                !(Filter.word3 & Complexity) ||
                !(Objects.GetQueryBitfield() & (1u << Channel)) ||
                (Objects.IgnoreMask & ShapeMask) ||
                Params.bIgnoreTouches ||
                Params.GetIgnoredActors().Contains(Filter.word0)
            )
                return physx::PxQueryHitType::eNONE;

            if (
                Params.GetIgnoredComponents().Num() &&
                Params.GetIgnoredComponents().Contains(Shape->getSimulationFilterData().word2)
            )
                return physx::PxQueryHitType::eNONE;

            UPrimitiveComponent* Component = nullptr;

            if (const FBodyInstance* Body = FPhysxUserData::Get<FBodyInstance>(Actor->userData))
            {
                Body = FPhysicsInterface::ShapeToOriginalBodyInstance(Body, Shape);

                Component = Body->OwnerComponent.Get();
            }
            else if (const FCustomPhysXPayload* Payload = FPhysxUserData::Get<FCustomPhysXPayload>(Shape->userData))
                Component = Payload->GetOwningComponent().Get();

            if (Component)
            {
                const int32 Label = Component->CustomDepthStencilValue;

                if (Label > 0 && Label < NumLabels && Priorities[Label] > 0)
                {
                    const bool bInstanced = (InstancedLabels >> Label) & 1;

                    const uint16 Instance = bInstanced ? GetInstance(*Component, Episode) : 0;

                    Candidates.Emplace(*Shape, *Actor, uint8(Label), Priorities[Label], Instance, bInstanced);
                }
            }

            // Continue gathering candidates without running narrow-phase
            // tests.
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
        uint32 InstancedLabels;
        const UCarlaEpisode& Episode;
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
        // Voxel query boxes keep this extent whatever the sensor's scale,
        // while centers follow the full sensor transform. A scaled sensor
        // therefore spaces boxes of BoxSize by its scale, as the per-voxel
        // engine query does. The API spawns sensors with unit scale, which
        // gives a contiguous grid; only a scaled parent actor changes it.
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
        // Outer corner of voxel (0, 0, 0) in the sensor frame: the back,
        // left and lower limits.
        FVector Origin;
        int32 GridSizeX;
        int32 GridSizeY;
        int32 GridSizeZ;
        const uint8* Priorities;
        int32 NumLabels;
        // One bit for each label that tells instances apart; none when the
        // sensor does not report instances.
        uint32 InstancedLabels;
        const UCarlaEpisode& Episode;

        FVoxelFrame(
            UWorld* InWorld,
            const FTransform& InSensorTransform,
            const FCollisionQueryParams& InQueryParams,
            float InBoxSize,
            const FVector& InOrigin,
            int32 InGridSizeX,
            int32 InGridSizeY,
            int32 InGridSizeZ,
            const uint8* InPriorities,
            int32 InNumLabels,
            uint32 InInstancedLabels,
            const UCarlaEpisode& InEpisode
        )
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
            , Origin(InOrigin)
            , GridSizeX(InGridSizeX)
            , GridSizeY(InGridSizeY)
            , GridSizeZ(InGridSizeZ)
            , Priorities(InPriorities)
            , NumLabels(InNumLabels)
            , InstancedLabels(InInstancedLabels)
            , Episode(InEpisode)
        {
            ObjectParams.RemoveObjectTypesToQuery(ECC_Vehicle);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Pawn);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Camera);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Visibility);
            ObjectParams.RemoveObjectTypesToQuery(ECC_PhysicsBody);
            ObjectParams.RemoveObjectTypesToQuery(ECC_Destructible);

            const physx::PxGeometry& VoxelGeometry = VoxelAdapter.GetGeometry();

            // Engine turns nearly empty boxes into spheres.
            bBoxVoxels = VoxelGeometry.getType() == physx::PxGeometryType::eBOX;

            if (bBoxVoxels)
                VoxelHalfExtent = static_cast<const physx::PxBoxGeometry&>(VoxelGeometry).halfExtents;
            else
                VoxelHalfExtent = physx::PxVec3(0.0f);

            VoxelQuat = U2PQuat(VoxelAdapter.GetGeomOrientation());

            const physx::PxTransform VoxelPose(physx::PxVec3(0.0f), VoxelQuat);

            VoxelWorldExtent = physx::PxGeometryQuery::getWorldBounds(VoxelGeometry, VoxelPose).getExtents();

            AbsAxisX = SensorTransform.TransformVector(FVector(1.0f, 0.0f, 0.0f)).GetAbs();
            AbsAxisY = SensorTransform.TransformVector(FVector(0.0f, 1.0f, 0.0f)).GetAbs();
            AbsAxisZ = SensorTransform.TransformVector(FVector(0.0f, 0.0f, 1.0f)).GetAbs();

            SensorScale = SensorTransform.GetScale3D();

            // Bound on the rounding of one voxel center, from the world
            // position through the tile frame. Every center lies between the
            // grid's corners in the sensor frame.
            const FVector GridEnd = Origin + FVector(FIntVector(GridSizeX, GridSizeY, GridSizeZ)) * BoxSize;

            const FVector LocalExtent = Origin.GetAbs().ComponentMax(GridEnd.GetAbs());

            const float LocalMagnitude = LocalExtent.X + LocalExtent.Y + LocalExtent.Z;

            CenterError = 32.0 * FLT_EPSILON * (
                SensorTransform.GetLocation().GetAbsMax() +
                2.0 * SensorScale.GetAbsMax() * LocalMagnitude +
                1.0
            );
        }

        // The tiles need box voxels, a finite sensor transform and a sensor
        // scale they can invert.
        bool IsSupported() const
        {
            return bBoxVoxels &&
                   !SensorTransform.ContainsNaN() &&
                   VoxelQuat.isFinite() &&
                   SensorScale.GetAbsMin() > 1e-3f &&
                   FMath::IsFinite(CenterError);
        }

        int64 GetNumVoxels() const
        {
            return int64(GridSizeX) * GridSizeY * GridSizeZ;
        }

        // Grid-wide index of voxel (X, Y, Z). Grids can hold more than 2^31
        // voxels, so it is 64-bit.
        int64 GetFlatIndex(int32 X, int32 Y, int32 Z) const
        {
            return (int64(X) * GridSizeY + Y) * GridSizeZ + Z;
        }

        FIntVector GetVoxel(int64 FlatIndex) const
        {
            const int64 Column = FlatIndex / GridSizeZ;

            return FIntVector(int32(Column / GridSizeY), int32(Column % GridSizeY), int32(FlatIndex % GridSizeZ));
        }

        // Center of voxel (X, Y, Z) in world space.
        FVector GetWorldPosition(int32 X, int32 Y, int32 Z) const
        {
            const FVector LocalPos = Origin + FVector(X + 0.5f, Y + 0.5f, Z + 0.5f) * BoxSize;

            return SensorTransform.TransformPosition(LocalPos);
        }

        // Exclusive end voxel of block Index in a grid of blocks of the given
        // size, clamped to the voxel grid.
        FIntVector GetBlockEnd(const FIntVector& Size, const FIntVector& Index) const
        {
            return FIntVector(
                int32(FMath::Min<int64>(int64(Index.X + 1) * Size.X, GridSizeX)),
                int32(FMath::Min<int64>(int64(Index.Y + 1) * Size.Y, GridSizeY)),
                int32(FMath::Min<int64>(int64(Index.Z + 1) * Size.Z, GridSizeZ))
            );
        }

        // Axis-aligned world box that encloses every voxel query between
        // Begin and End.
        FQueryBox MakeQueryBox(const FIntVector& Begin, const FIntVector& End) const
        {
            const FVector LocalMin = Origin + FVector(Begin) * BoxSize;
            const FVector LocalMax = Origin + FVector(End) * BoxSize;

            const FVector LocalHalfExtent = (LocalMax - LocalMin) * 0.5f;

            const FVector Center = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

            // Enclose voxel centers and query extents, including sensor
            // scale.
            const FVector CenterHalfExtent = LocalHalfExtent - BoxExtent;

            const FVector WorldExtent = AbsAxisX * CenterHalfExtent.X +
                                        AbsAxisY * CenterHalfExtent.Y +
                                        AbsAxisZ * CenterHalfExtent.Z +
                                        P2UVector(VoxelWorldExtent) +
                                        FVector(0.1f);

            return FQueryBox{physx::PxBoxGeometry(U2PVector(WorldExtent)), physx::PxTransform(U2PVector(Center))};
        }

        // Approximate centers of the voxels [Begin, Begin + Size) in the tile
        // frame.
        VoxelSurfaceRaster::FLattice MakeLattice(
            const FIntVector& Begin,
            const FIntVector& Size,
            const physx::PxTransform& TileFrame
        ) const
        {
            const int32 LatticeSize[3] = {Size.X, Size.Y, Size.Z};

            const double Step[3] = {
                double(SensorScale.X) * BoxSize,
                double(SensorScale.Y) * BoxSize,
                double(SensorScale.Z) * BoxSize
            };

            const auto GetCenter = [&](int32 X, int32 Y, int32 Z)
            {
                return TileFrame.transformInv(U2PVector(GetWorldPosition(Begin.X + X, Begin.Y + Y, Begin.Z + Z)));
            };

            return VoxelSurfaceRaster::MakeLattice(LatticeSize, Step, CenterError, 0.0, GetCenter);
        }

        bool IsInstanced(uint8 Label) const
        {
            return (InstancedLabels >> Label) & 1;
        }

        // Label of one voxel from a per-voxel engine query, with its instance
        // if the label has them; PhysX hit order breaks equal-priority ties.
        FVoxelResult QueryVoxel(const FVector& WorldPos) const
        {
            TArray<FOverlapResult> Overlaps;

            FVoxelResult Result;

            const UPrimitiveComponent* Winner = nullptr;

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

                    const bool bKnown = CandidateLabel >= 0 && CandidateLabel < NumLabels;

                    if (bKnown && Priorities[CandidateLabel] > Priorities[Result.Label])
                    {
                        Result.Label = uint8(CandidateLabel);

                        Winner = Component;
                    }
                }
            }

            if (Winner && IsInstanced(Result.Label))
                Result.Instance = GetInstance(*Winner, Episode);

            return Result;
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
        uint64 VoxelQueryCycles = 0;
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
        uint64 Instances = 0;
        VoxelSurfaceRaster::FRasterStats Raster;

        void Add(const FVoxelProfile& Other)
        {
            ZeroCycles += Other.ZeroCycles;
            GatherCycles += Other.GatherCycles;
            CullCycles += Other.CullCycles;
            FetchCycles += Other.FetchCycles;
            RasterCycles += Other.RasterCycles;
            DeferredCycles += Other.DeferredCycles;
            VoxelQueryCycles += Other.VoxelQueryCycles;
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
            Instances += Other.Instances;

            Raster.Add(Other.Raster);
        }
    };

    // Tiles are whole multiples of a brick, so bricks align with the tile
    // grid. A tile larger than the grid covers it just the same, so requested
    // sizes are clamped to the grid first, and the rounding runs in 64 bits
    // so it cannot overflow.
    constexpr int64 MaxTileVoxels = int64(1) << 30;

    FIntVector GetTileSize(const FIntVector& Requested, const FIntVector& GridSize)
    {
        const auto Round = [](int32 Tile, int32 Grid, int32 Brick)
        {
            const int64 Clamped = FMath::Clamp<int64>(Tile, 1, FMath::Max(1, Grid));

            const int64 Rounded = FMath::DivideAndRoundUp<int64>(Clamped, Brick) * Brick;

            return int32(FMath::Min<int64>(Rounded, MAX_int32 / Brick * Brick));
        };

        int32 Size[3] = {
            Round(Requested.X, GridSize.X, BrickSize.X),
            Round(Requested.Y, GridSize.Y, BrickSize.Y),
            Round(Requested.Z, GridSize.Z, BrickSize.Z)
        };

        const int32 Brick[3] = {BrickSize.X, BrickSize.Y, BrickSize.Z};

        // Voxels are indexed within a tile with 32-bit integers, so halve the
        // longest side of a larger tile, in whole bricks, until it fits.
        while (int64(Size[0]) * Size[1] * Size[2] > MaxTileVoxels)
        {
            const int Axis = Size[0] >= Size[1] && Size[0] >= Size[2] ? 0 : (Size[1] >= Size[2] ? 1 : 2);

            Size[Axis] = FMath::Max(Brick[Axis], FMath::DivideAndRoundUp(Size[Axis] / 2, Brick[Axis]) * Brick[Axis]);
        }

        return FIntVector(Size[0], Size[1], Size[2]);
    }

    // Connects the surface raster to the output labels and to the PhysX
    // queries of the current candidate.
    struct FTileEnv
    {
        const FVoxelFrame& Frame;
        const VoxelSurfaceRaster::FLabelRules& Rules;
        const VoxelSurfaceRaster::FLattice& Lattice;
        const physx::PxTransform& TileFrame;
        const FIntVector Begin;
        uint8* Labels;
        VoxelSurfaceRaster::FVoxelSet& Deferred;
        TArray<int64>& Ties;
        // Instance of every tile voxel that holds a label with instances;
        // the other voxels are not initialized.
        uint16* Instances;
        TArray<FVoxelInstance>& Marked;
        const FVoxelCandidate* Candidate;

        int64 GetFlatIndex(int32 X, int32 Y, int32 Z) const
        {
            return Frame.GetFlatIndex(Begin.X + X, Begin.Y + Y, Begin.Z + Z);
        }

        physx::PxVec3 GetWorldPosition(int32 X, int32 Y, int32 Z) const
        {
            return U2PVector(Frame.GetWorldPosition(Begin.X + X, Begin.Y + Y, Begin.Z + Z));
        }

        bool IsOpen(int32 X, int32 Y, int32 Z) const
        {
            const uint8 State = Labels[GetFlatIndex(X, Y, Z)];

            if (!Candidate->bInstanced)
                return !Rules.IsClosed(State, Candidate->Label, Candidate->Priority);

            const auto GetVoxelInstance = [&]()
            {
                return Instances[Lattice.GetIndex(X, Y, Z)];
            };

            return !Rules.IsClosed(
                State,
                Candidate->Label,
                Candidate->Priority,
                Candidate->Instance,
                GetVoxelInstance
            );
        }

        bool GetCenter(int32 X, int32 Y, int32 Z, physx::PxVec3& Center) const
        {
            const physx::PxVec3 Position = GetWorldPosition(X, Y, Z);

            if (!Candidate->Bounds.intersects(physx::PxBounds3::centerExtents(Position, Frame.VoxelWorldExtent)))
                return false;

            Center = TileFrame.transformInv(Position);

            return true;
        }

        void Hit(int32 X, int32 Y, int32 Z)
        {
            const int64 FlatIndex = GetFlatIndex(X, Y, Z);

            if (Rules.Apply(Labels[FlatIndex], Candidate->Label))
                Ties.Add(FlatIndex);
            else if (Candidate->bInstanced)
            {
                Instances[Lattice.GetIndex(X, Y, Z)] = Candidate->Instance;

                Marked.Add(FVoxelInstance{uint64(FlatIndex), Candidate->Instance});
            }
        }

        void Defer(int32 X, int32 Y, int32 Z)
        {
            Deferred.Add(Lattice.GetIndex(X, Y, Z));
        }

        bool Overlaps(int32 X, int32 Y, int32 Z) const
        {
            const physx::PxTransform VoxelPose(GetWorldPosition(X, Y, Z), Frame.VoxelQuat);

            return physx::PxGeometryQuery::overlap(
                Frame.VoxelAdapter.GetGeometry(),
                VoxelPose,
                Candidate->Geometry.any(),
                Candidate->Pose
            );
        }

        bool OverlapsBrick(const int32* BrickBegin, const int32* BrickEnd) const
        {
            const FQueryBox Box = Frame.MakeQueryBox(
                FIntVector(Begin.X + BrickBegin[0], Begin.Y + BrickBegin[1], Begin.Z + BrickBegin[2]),
                FIntVector(Begin.X + BrickEnd[0], Begin.Y + BrickEnd[1], Begin.Z + BrickEnd[2])
            );

            return physx::PxGeometryQuery::overlap(
                Box.Geometry,
                Box.Pose,
                Candidate->Geometry.any(),
                Candidate->Pose
            );
        }
    };

    // Label every voxel of one tile and append its voxels with an instance
    // to OutInstances. Separate instantiations keep profiling clocks out of
    // normal frames.
    template <bool bProfile>
    void ProcessTile(
        const FVoxelFrame& Frame,
        const VoxelSurfaceRaster::FLabelRules& Rules,
        const FIntVector& TileSize,
        const FIntVector& Tile,
        uint8* Labels,
        TArray<FVoxelInstance>& OutInstances,
        FVoxelProfile& Profile
    )
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

        FVoxelCandidateCollector Collector(
            Frame.QueryParams,
            Frame.ObjectParams,
            Frame.Priorities,
            Frame.NumLabels,
            Frame.InstancedLabels,
            Frame.Episode
        );

        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelGatherCandidates);

            // PhysX requires touch storage even though the prefilter returns
            // eNONE.
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);

            const physx::PxQueryFlags Flags = PhysXInterface::StaticDynamicQueryFlags(Frame.QueryParams);

            const physx::PxQueryFilterData Filter(Flags | physx::PxQueryFlag::ePREFILTER);

            Frame.Scene->overlap(TileBox.Geometry, TileBox.Pose, UnusedHits, Filter, &Collector);
        }

        Lap(Profile.GatherCycles);

        // Labels accumulate in the output, so the tile starts empty.
        for (int32 X = Begin.X; X < End.X; ++X)
        {
            for (int32 Y = Begin.Y; Y < End.Y; ++Y)
            {
                const int64 RowStart = Frame.GetFlatIndex(X, Y, Begin.Z);

                FMemory::Memzero(Labels + RowStart, Size.Z);
            }
        }

        Lap(Profile.ZeroCycles);

        // Meshes and heightfields are culled by their triangle fetch below.
        Collector.Candidates.RemoveAll([&](const FVoxelCandidate& Candidate)
        {
            if (HasTriangles(Candidate))
                return false;

            return !physx::PxGeometryQuery::overlap(
                TileBox.Geometry,
                TileBox.Pose,
                Candidate.Geometry.any(),
                Candidate.Pose
            );
        });

        // Equal priorities need no order: the same label is closed to itself,
        // and different labels tie and go to the engine query.
        Collector.Candidates.Sort([](const FVoxelCandidate& A, const FVoxelCandidate& B)
        {
            return A.Priority > B.Priority;
        });

        if (bProfile)
            Profile.Candidates += Collector.Candidates.Num();

        Lap(Profile.CullCycles);

        const physx::PxTransform TileFrame(TileBox.Pose.p, Frame.VoxelQuat);

        const VoxelSurfaceRaster::FLattice Lattice = Frame.MakeLattice(Begin, Size, TileFrame);

        // An invalid lattice only comes from a non-finite transform, which
        // leaves nothing to label.
        if (Collector.Candidates.Num() == 0 || !Lattice.IsValid())
        {
            Finish();

            return;
        }

        // Sensor-aligned box around every voxel query box of the tile, with
        // the same 0.1 cm margin as the query boxes.
        physx::PxVec3 FetchExtent;

        for (int32 Axis = 0; Axis < 3; ++Axis)
        {
            const double Extent = FMath::Max(
                FMath::Abs(Lattice.Center(Axis, 0)),
                FMath::Abs(Lattice.Center(Axis, Lattice.Size[Axis] - 1))
            );

            FetchExtent[Axis] = float(Extent + Lattice.Margin) + Frame.VoxelHalfExtent[Axis] + 0.1f;
        }

        const physx::PxBoxGeometry FetchBox(FetchExtent);

        const int32 BrickDimensions[3] = {BrickSize.X, BrickSize.Y, BrickSize.Z};

        // Separating axes are only built for triangles that can still change
        // a label.
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

        TArray<int64> Ties;

        // A voxel only reads its instance after a candidate wrote it.
        TArray<uint16> Instances;

        const bool bInstances = Collector.Candidates.ContainsByPredicate([](const FVoxelCandidate& Candidate)
        {
            return Candidate.bInstanced;
        });

        if (bInstances)
            Instances.SetNumUninitialized(Lattice.GetNumVoxels());

        const int32 FirstInstance = OutInstances.Num();

        FTileEnv Env{
            Frame,
            Rules,
            Lattice,
            TileFrame,
            Begin,
            Labels,
            Deferred,
            Ties,
            Instances.GetData(),
            OutInstances,
            nullptr
        };

        // Candidates go in descending priority, so a voxel closes as soon as
        // no later candidate can change its label.
        for (const FVoxelCandidate& Candidate : Collector.Candidates)
        {
            Env.Candidate = &Candidate;

            // Walker meshes favor exact voxel queries above the measured
            // crossover near 2.0 cm; finer grids keep the surface raster.
            const bool bQueryPedestrian =
                Candidate.Label == uint8(carla::rpc::CityObjectLabel::Pedestrians) &&
                Candidate.Geometry.getType() == physx::PxGeometryType::eTRIANGLEMESH &&
                Frame.BoxSize >= 2.0f;

            if (!HasTriangles(Candidate) || bQueryPedestrian)
            {
                if (bProfile)
                {
                    if (bQueryPedestrian)
                        ++Profile.MeshCandidates;
                    else
                        ++Profile.PrimitiveCandidates;
                }

                const VoxelSurfaceRaster::FIndexBox Box = VoxelSurfaceRaster::GetBoundsBox(
                    Candidate.Bounds,
                    Frame.VoxelWorldExtent,
                    TileFrame,
                    Lattice
                );

                VoxelSurfaceRaster::RasterizeBounds<bProfile>(Box, BrickDimensions, Lattice, Env, Profile.Raster);

                Lap(Profile.VoxelQueryCycles);

                continue;
            }

            const bool bHeightField = Candidate.Geometry.getType() == physx::PxGeometryType::eHEIGHTFIELD;

            if (bProfile && bHeightField)
                ++Profile.HeightFieldCandidates;
            else if (bProfile)
                ++Profile.MeshCandidates;

            const physx::PxTransform MeshToTile = TileFrame.transformInv(Candidate.Pose);

            // PhysX treats a thin layer next to a heightfield as solid, so
            // its fetch box grows by that skin.
            VoxelSurfaceRaster::FSurface Surface;

            if (bHeightField)
            {
                const float Thickness = Candidate.Geometry.heightField().heightField->getThickness();

                Surface = VoxelSurfaceRaster::MakeSurface(MeshToTile.q, Thickness);
            }

            const physx::PxBoxGeometry CandidateFetchBox(FetchBox.halfExtents + physx::PxVec3(Surface.Skin));

            // Every query reruns the midphase, so a full buffer is regrown to
            // hold every triangle and queried once more.
            const auto FetchIndices = [&](bool& bOverflow)
            {
                if (bHeightField)
                {
                    return physx::PxMeshQuery::findOverlapHeightField(
                        CandidateFetchBox,
                        TileFrame,
                        Candidate.Geometry.heightField(),
                        Candidate.Pose,
                        Indices.GetData(),
                        physx::PxU32(Indices.Num()),
                        0,
                        bOverflow
                    );
                }

                return physx::PxMeshQuery::findOverlapTriangleMesh(
                    CandidateFetchBox,
                    TileFrame,
                    Candidate.Geometry.triangleMesh(),
                    Candidate.Pose,
                    Indices.GetData(),
                    physx::PxU32(Indices.Num()),
                    0,
                    bOverflow
                );
            };

            bool bOverflow = false;

            physx::PxU32 Count = FetchIndices(bOverflow);

            if (bOverflow)
            {
                Indices.SetNumUninitialized(FMath::Max<int32>(2 * Indices.Num(), GetNumTriangles(Candidate)));

                Count = FetchIndices(bOverflow);

                if (bProfile)
                    ++Profile.Regrowths;
            }

            Triangles.SetNumUninitialized(Count);

            for (physx::PxU32 Index = 0; Index < Count; ++Index)
            {
                FFetchedTriangle& Fetched = Triangles[Index];

                if (bHeightField)
                {
                    Fetched.PositionError = VoxelTriangleOverlap::FetchTriangle(
                        Candidate.Geometry.heightField(),
                        Candidate.Pose,
                        TileFrame,
                        MeshToTile,
                        Indices[Index],
                        Frame.VoxelHalfExtent,
                        Fetched.Triangle
                    );
                }
                else
                {
                    Fetched.PositionError = VoxelTriangleOverlap::FetchTriangle(
                        Candidate.Geometry.triangleMesh(),
                        Candidate.Pose,
                        TileFrame,
                        MeshToTile,
                        Indices[Index],
                        Frame.VoxelHalfExtent,
                        Fetched.Triangle
                    );
                }
            }

            Lap(Profile.FetchCycles);

            for (const FFetchedTriangle& Fetched : Triangles)
            {
                VoxelSurfaceRaster::RasterizeMeshTriangle<bProfile>(
                    Fetched.Triangle,
                    Fetched.PositionError,
                    Frame.VoxelHalfExtent,
                    Lattice,
                    Env,
                    Profile.Raster,
                    bHeightField ? &Surface : nullptr
                );
            }

            Lap(Profile.RasterCycles);

            VoxelSurfaceRaster::ResolveDeferred<bProfile>(Deferred, Lattice, Env, Profile.Raster);

            Lap(Profile.DeferredCycles);
        }

        // A tie replaces what the first candidate of its voxel recorded.
        for (int32 Index = OutInstances.Num() - 1; Index >= FirstInstance; --Index)
        {
            if (Labels[OutInstances[Index].Index] & VoxelSurfaceRaster::TieFlag)
                OutInstances.RemoveAtSwap(Index, 1, false);
        }

        // Two different labels share the highest priority, or two instances
        // of one label; the per-voxel query keeps PhysX hit order.
        for (const int64 FlatIndex : Ties)
        {
            const FIntVector Voxel = Frame.GetVoxel(FlatIndex);

            const FVoxelResult Result = Frame.QueryVoxel(Frame.GetWorldPosition(Voxel.X, Voxel.Y, Voxel.Z));

            Labels[FlatIndex] = Result.Label;

            if (Frame.IsInstanced(Result.Label))
                OutInstances.Add(FVoxelInstance{uint64(FlatIndex), Result.Instance});
        }

        if (bProfile)
        {
            Profile.Ties += Ties.Num();

            Profile.Instances += OutInstances.Num() - FirstInstance;
        }

        Lap(Profile.TieCycles);

        Finish();
    }

    // Label every voxel tile by tile and return the number of tiles. The
    // voxels with an instance come back in one list per batch.
    int64 RunTiles(
        const FVoxelFrame& Frame,
        const FIntVector& TileSize,
        uint8* Labels,
        bool bShowCalculationTime,
        TArray<TArray<FVoxelInstance>>& OutInstances,
        TArray<FVoxelProfile>& OutProfiles
    )
    {
        // In 64 bits, since a grid size plus a tile size can exceed an int32.
        // Each count is at most its grid size.
        const int32 TilesX = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeX, TileSize.X));
        const int32 TilesY = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeY, TileSize.Y));
        const int32 TilesZ = int32(FMath::DivideAndRoundUp<int64>(Frame.GridSizeZ, TileSize.Z));

        const int64 TotalTiles = int64(TilesX) * TilesY * TilesZ;

        const VoxelSurfaceRaster::FLabelRules Rules(Frame.Priorities, Frame.NumLabels);

        // Tile costs vary a lot, so batches hold a single tile unless tiles
        // are tiny. PhysX read locks share a mutex, so each batch takes one.
        const int32 NumBatches = int32(FMath::Min<int64>(
            TotalTiles,
            64 * (FTaskGraphInterface::Get().GetNumWorkerThreads() + 1)
        ));

        OutProfiles.Reset();

        if (bShowCalculationTime)
            OutProfiles.SetNum(NumBatches);

        OutInstances.Reset();

        OutInstances.SetNum(NumBatches);

        ParallelFor(NumBatches, [&](int32 Batch)
        {
            TRACE_CPUPROFILER_EVENT_SCOPE(VoxelProcessTiles);

            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FVoxelProfile Profile;

            const int64 Begin = TotalTiles * Batch / NumBatches;
            const int64 End = TotalTiles * (Batch + 1) / NumBatches;

            for (int64 Index = Begin; Index < End; ++Index)
            {
                const int64 Column = Index / TilesZ;

                const int32 TX = int32(Column / TilesY);
                const int32 TY = int32(Column % TilesY);
                const int32 TZ = int32(Index % TilesZ);

                const FIntVector Tile(TX, TY, TZ);

                if (bShowCalculationTime)
                    ProcessTile<true>(Frame, Rules, TileSize, Tile, Labels, OutInstances[Batch], Profile);
                else
                    ProcessTile<false>(Frame, Rules, TileSize, Tile, Labels, OutInstances[Batch], Profile);
            }

            if (bShowCalculationTime)
                OutProfiles[Batch] = Profile;
        }, EParallelForFlags::Unbalanced);

        return TotalTiles;
    }

    // Compare every Stride-th voxel with the per-voxel engine query. The
    // offset rotates so Stride frames cover the grid. A voxel has to match in
    // its label, in whether it has an instance, and in that instance.
    void CheckAgainstOracle(
        const FVoxelFrame& Frame,
        const uint8* Labels,
        const TArray<TArray<FVoxelInstance>>& Instances,
        int32 Stride
    )
    {
        const int64 TotalVoxels = Frame.GetNumVoxels();

        TMap<uint64, uint64> InstanceOf;

        int64 NumInstances = 0;

        for (const TArray<FVoxelInstance>& Batch : Instances)
        {
            NumInstances += Batch.Num();

            for (const FVoxelInstance& Voxel : Batch)
            {
                InstanceOf.Add(Voxel.Index, Voxel.Instance);
            }
        }

        const int32 Offset = FCarlaEngine::GetFrameCounter() % Stride;

        // Counts and batch ends are computed in 64 bits, since a count plus
        // the stride or batch size can exceed an int32.
        const int64 Remaining = FMath::Max<int64>(TotalVoxels - Offset, 0);

        const int64 NumSamples = FMath::DivideAndRoundUp<int64>(Remaining, Stride);

        constexpr int32 BatchSize = 4096;
        constexpr int32 MaxExamples = 8;

        struct FBatchResult
        {
            uint64 Mismatches = 0;
            int64 Examples[MaxExamples] = {};
            FVoxelResult Oracles[MaxExamples] = {};
            int32 NumExamples = 0;
        };

        const int32 NumBatches = int32(FMath::DivideAndRoundUp<int64>(NumSamples, BatchSize));

        TArray<FBatchResult> Batches;

        Batches.SetNum(NumBatches);

        ParallelFor(NumBatches, [&](int32 BatchIndex)
        {
            physx::PxSceneReadLock SceneLock(*Frame.Scene);

            FBatchResult& Batch = Batches[BatchIndex];

            const int64 End = FMath::Min<int64>(int64(BatchIndex + 1) * BatchSize, NumSamples);

            for (int64 Sample = int64(BatchIndex) * BatchSize; Sample < End; ++Sample)
            {
                const int64 FlatIndex = Offset + Sample * Stride;

                const FIntVector Voxel = Frame.GetVoxel(FlatIndex);

                const FVoxelResult Oracle = Frame.QueryVoxel(Frame.GetWorldPosition(Voxel.X, Voxel.Y, Voxel.Z));

                const uint64* Instance = InstanceOf.Find(uint64(FlatIndex));

                const bool bSameInstance = Frame.IsInstanced(Oracle.Label) ?
                    Instance && *Instance == Oracle.Instance :
                    !Instance;

                if (Oracle.Label != Labels[FlatIndex] || !bSameInstance)
                {
                    ++Batch.Mismatches;

                    if (Batch.NumExamples < MaxExamples)
                    {
                        Batch.Examples[Batch.NumExamples] = FlatIndex;
                        Batch.Oracles[Batch.NumExamples] = Oracle;

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

        carla::log_warning("Voxel oracle check: ", Mismatches,
            " mismatched voxels of ", NumSamples,
            " sampled (stride ", Stride,
            ", offset ", Offset,
            "), ", NumInstances,
            " voxels with an instance, ", NumInstances - InstanceOf.Num(),
            " of them listed twice");

        int32 Logged = 0;

        for (const FBatchResult& Batch : Batches)
        {
            for (int32 Example = 0; Example < Batch.NumExamples && Logged < MaxExamples; ++Example, ++Logged)
            {
                const int64 FlatIndex = Batch.Examples[Example];

                const FIntVector Voxel = Frame.GetVoxel(FlatIndex);

                const uint64* Instance = InstanceOf.Find(uint64(FlatIndex));

                carla::log_warning("  voxel (", Voxel.X, ", ", Voxel.Y, ", ", Voxel.Z, ")",
                    ": output=", int32(Labels[FlatIndex]),
                    ", instance=", Instance ? int64(*Instance) : int64(-1),
                    ", per-voxel query=", int32(Batch.Oracles[Example].Label),
                    ", instance=", int32(Batch.Oracles[Example].Instance));
            }
        }
    }

    // One bit for each label the tagger counts as a thing: the labels whose
    // actors the instance segmentation camera tells apart.
    uint32 GetInstancedLabels(int32 NumLabels)
    {
        uint32 Labels = 0;

        for (int32 Label = 0; Label < NumLabels; ++Label)
        {
            if (ATagger::IsThing(static_cast<carla::rpc::CityObjectLabel>(Label)))
                Labels |= 1u << Label;
        }

        return Labels;
    }

    // The GPU path of use_gpu, from libcarla_voxel_gpu, which is loaded at
    // run time so CARLA builds and runs without CUDA.
    struct FVoxelGpuLibrary
    {
        decltype(&VoxelGpuApiVersion) ApiVersion = nullptr;
        decltype(&VoxelGpuCreate) Create = nullptr;
        decltype(&VoxelGpuDestroy) Destroy = nullptr;
        decltype(&VoxelGpuError) Error = nullptr;
        decltype(&VoxelGpuFindMesh) FindMesh = nullptr;
        decltype(&VoxelGpuUploadMesh) UploadMesh = nullptr;
        decltype(&VoxelGpuRun) Run = nullptr;
        decltype(&VoxelGpuGetPairs) GetPairs = nullptr;

        FString LoadError;

        bool IsLoaded() const
        {
            return Run != nullptr;
        }

        static const FVoxelGpuLibrary& Get()
        {
            static const FVoxelGpuLibrary Library = Load();

            return Library;
        }

    private:

        static FVoxelGpuLibrary Load()
        {
            FVoxelGpuLibrary Library;

            const FString Path = FPaths::ConvertRelativePathToFull(FPaths::Combine(
                FPaths::ProjectPluginsDir(),
                TEXT("Carla/CarlaDependencies/lib/libcarla_voxel_gpu.so")
            ));

            void* Handle = FPlatformProcess::GetDllHandle(*Path);

            if (!Handle)
            {
                Library.LoadError = FString::Printf(
                    TEXT("cannot load %s; it is built when nvcc is available"),
                    *Path
                );

                return Library;
            }

            const auto Find = [&](auto& Function, const TCHAR* Name)
            {
                using FFunction = typename std::remove_reference<decltype(Function)>::type;

                Function = reinterpret_cast<FFunction>(FPlatformProcess::GetDllExport(Handle, Name));

                return Function != nullptr;
            };

            const bool bFound = Find(Library.ApiVersion, TEXT("VoxelGpuApiVersion")) &&
                                Find(Library.Create, TEXT("VoxelGpuCreate")) &&
                                Find(Library.Destroy, TEXT("VoxelGpuDestroy")) &&
                                Find(Library.Error, TEXT("VoxelGpuError")) &&
                                Find(Library.FindMesh, TEXT("VoxelGpuFindMesh")) &&
                                Find(Library.UploadMesh, TEXT("VoxelGpuUploadMesh")) &&
                                Find(Library.Run, TEXT("VoxelGpuRun")) &&
                                Find(Library.GetPairs, TEXT("VoxelGpuGetPairs"));

            if (!bFound || Library.ApiVersion() != VOXEL_GPU_API_VERSION)
            {
                Library = FVoxelGpuLibrary();

                Library.LoadError = FString::Printf(TEXT("%s does not match this sensor; rebuild it"), *Path);
            }

            return Library;
        }
    };

    using FPrecise = std::array<double, 3>;

    FPrecise Cross(const FPrecise& A, const FPrecise& B)
    {
        return FPrecise{A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]};
    }

    double Dot(const FPrecise& A, const FPrecise& B)
    {
        return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    }

    FPrecise Subtract(const FPrecise& A, const FPrecise& B)
    {
        return FPrecise{A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    }

    // A 3 x 4 affine map in double precision, row-major, as the GPU path
    // takes shapes into grid space.
    struct FAffine
    {
        double M[12] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0};

        static FAffine FromRotation(double X, double Y, double Z, double W)
        {
            const double Norm = std::sqrt(X * X + Y * Y + Z * Z + W * W);

            X /= Norm;
            Y /= Norm;
            Z /= Norm;
            W /= Norm;

            FAffine A;

            A.M[0] = 1.0 - 2.0 * (Y * Y + Z * Z);
            A.M[1] = 2.0 * (X * Y - Z * W);
            A.M[2] = 2.0 * (X * Z + Y * W);
            A.M[4] = 2.0 * (X * Y + Z * W);
            A.M[5] = 1.0 - 2.0 * (X * X + Z * Z);
            A.M[6] = 2.0 * (Y * Z - X * W);
            A.M[8] = 2.0 * (X * Z - Y * W);
            A.M[9] = 2.0 * (Y * Z + X * W);
            A.M[10] = 1.0 - 2.0 * (X * X + Y * Y);

            return A;
        }

        static FAffine FromPose(const physx::PxTransform& Pose)
        {
            FAffine A = FromRotation(Pose.q.x, Pose.q.y, Pose.q.z, Pose.q.w);

            A.M[3] = Pose.p.x;
            A.M[7] = Pose.p.y;
            A.M[11] = Pose.p.z;

            return A;
        }

        static FAffine FromMatrix(const physx::PxMat33& Matrix)
        {
            FAffine A;

            for (int Row = 0; Row < 3; ++Row)
            {
                A.M[4 * Row] = Matrix.column0[Row];
                A.M[4 * Row + 1] = Matrix.column1[Row];
                A.M[4 * Row + 2] = Matrix.column2[Row];
            }

            return A;
        }

        // This map after Other.
        FAffine operator*(const FAffine& Other) const
        {
            FAffine A;

            for (int Row = 0; Row < 3; ++Row)
            {
                for (int Column = 0; Column < 4; ++Column)
                {
                    double Sum = Column == 3 ? M[4 * Row + 3] : 0.0;

                    for (int K = 0; K < 3; ++K)
                        Sum += M[4 * Row + K] * Other.M[4 * K + Column];

                    A.M[4 * Row + Column] = Sum;
                }
            }

            return A;
        }

        FPrecise Apply(const FPrecise& Point) const
        {
            return FPrecise{
                M[0] * Point[0] + M[1] * Point[1] + M[2] * Point[2] + M[3],
                M[4] * Point[0] + M[5] * Point[1] + M[6] * Point[2] + M[7],
                M[8] * Point[0] + M[9] * Point[1] + M[10] * Point[2] + M[11]
            };
        }

        FPrecise Rotate(const FPrecise& Direction) const
        {
            return FPrecise{
                M[0] * Direction[0] + M[1] * Direction[1] + M[2] * Direction[2],
                M[4] * Direction[0] + M[5] * Direction[1] + M[6] * Direction[2],
                M[8] * Direction[0] + M[9] * Direction[1] + M[10] * Direction[2]
            };
        }
    };

    // The voxels whose unit boxes may touch grid-space bounds [Low, High].
    void SetVoxelBox(FVoxelGpuShape& Shape, const FPrecise& Low, const FPrecise& High, const FIntVector& GridSize)
    {
        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const double Begin = std::ceil(Low[Axis]) - 1.0;
            const double End = std::floor(High[Axis]) + 1.0;

            Shape.Begin[Axis] = Begin > 0.0 ? int32(FMath::Min<double>(Begin, GridSize[Axis])) : 0;
            Shape.End[Axis] = End > 0.0 ? int32(FMath::Min<double>(End, GridSize[Axis])) : 0;
        }
    }

    // A convex shape for the GPU: every axis that can separate it from a
    // voxel (the grid axes, its face normals, and its edges crossed with the
    // grid axes), with its extent along each.
    void AddHull(
        const TArray<FPrecise>& Points,
        const TArray<FPrecise>& Normals,
        const TArray<FPrecise>& Edges,
        const FIntVector& GridSize,
        FVoxelGpuShape& Shape,
        TArray<FVoxelGpuAxis>& Axes
    )
    {
        Shape.Type = VOXEL_GPU_HULL;
        Shape.FirstAxis = uint32(Axes.Num());

        TArray<FPrecise> Directions = {FPrecise{1.0, 0.0, 0.0}, FPrecise{0.0, 1.0, 0.0}, FPrecise{0.0, 0.0, 1.0}};

        Directions.Append(Normals);

        for (const FPrecise& Edge : Edges)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
                Directions.Add(Cross(Edge, Directions[Axis]));
        }

        for (const FPrecise& Direction : Directions)
        {
            if (Direction[0] == 0.0 && Direction[1] == 0.0 && Direction[2] == 0.0)
                continue;

            FVoxelGpuAxis Axis{{Direction[0], Direction[1], Direction[2]}, INFINITY, -INFINITY};

            for (const FPrecise& Point : Points)
            {
                const double Projection = Dot(Direction, Point);

                Axis.Low = FMath::Min(Axis.Low, Projection);
                Axis.High = FMath::Max(Axis.High, Projection);
            }

            Axes.Add(Axis);
        }

        Shape.NumAxes = uint32(Axes.Num()) - Shape.FirstAxis;

        FPrecise Low{INFINITY, INFINITY, INFINITY};
        FPrecise High{-INFINITY, -INFINITY, -INFINITY};

        for (const FPrecise& Point : Points)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
            {
                Low[Axis] = FMath::Min(Low[Axis], Point[Axis]);
                High[Axis] = FMath::Max(High[Axis], Point[Axis]);
            }
        }

        SetVoxelBox(Shape, Low, High, GridSize);
    }

    // Timings and counts of one frame on the GPU path.
    struct FVoxelGpuReport
    {
        double GatherMs = 0.0;
        double GeometryMs = 0.0;
        double RunMs = 0.0;
        int32 Candidates = 0;
        int32 Meshes = 0;
        int32 Uploads = 0;
        int64 UploadBytes = 0;
        int32 HeightFields = 0;
        int32 Shapes = 0;
        int32 Unsupported = 0;
        FVoxelGpuStats Stats = {};
    };

    // Label the frame on the GPU without PhysX queries: one scene query
    // gathers the labeled shapes, their geometry goes to the GPU in grid
    // space, and every overlap is decided there in double precision. Ties
    // between equal priorities go to the lower label, then the lower
    // instance, instead of the engine query's hit order.
    bool LabelOnGpu(
        const FVoxelFrame& Frame,
        FVoxelGpuContext* Context,
        uint8* Labels,
        TArray<FVoxelInstance>& OutInstances,
        FVoxelGpuReport& Report,
        FString& OutError
    )
    {
        const FVoxelGpuLibrary& Gpu = FVoxelGpuLibrary::Get();

        // Unit voxels in grid space need an unscaled sensor.
        if (!Frame.SensorScale.Equals(FVector::OneVector, 1e-4f))
        {
            OutError = TEXT("use_gpu needs a sensor scale of 1");

            return false;
        }

        const double Start = FPlatformTime::Seconds();

        const FIntVector GridSize(Frame.GridSizeX, Frame.GridSizeY, Frame.GridSizeZ);

        // Grid space: (sensor frame - Origin) / BoxSize, so voxel (X, Y, Z)
        // spans [X, X + 1] x [Y, Y + 1] x [Z, Z + 1].
        const FQuat Rotation = Frame.SensorTransform.GetRotation();
        const FVector Location = Frame.SensorTransform.GetLocation();

        const FAffine SensorRotation = FAffine::FromRotation(Rotation.X, Rotation.Y, Rotation.Z, Rotation.W);

        FAffine WorldToGrid;

        for (int Row = 0; Row < 3; ++Row)
        {
            double Translation = 0.0;

            for (int Column = 0; Column < 3; ++Column)
            {
                WorldToGrid.M[4 * Row + Column] = SensorRotation.M[4 * Column + Row] / Frame.BoxSize;

                Translation -= SensorRotation.M[4 * Column + Row] * Location[Column];
            }

            WorldToGrid.M[4 * Row + 3] = (Translation - Frame.Origin[Row]) / Frame.BoxSize;
        }

        physx::PxSceneReadLock SceneLock(*Frame.Scene);

        FVoxelCandidateCollector Collector(
            Frame.QueryParams,
            Frame.ObjectParams,
            Frame.Priorities,
            Frame.NumLabels,
            Frame.InstancedLabels,
            Frame.Episode
        );

        {
            physx::PxOverlapHit UnusedHit;
            physx::PxOverlapBuffer UnusedHits(&UnusedHit, 1);

            const physx::PxQueryFlags Flags = PhysXInterface::StaticDynamicQueryFlags(Frame.QueryParams);

            const physx::PxQueryFilterData Filter(Flags | physx::PxQueryFlag::ePREFILTER);

            const FQueryBox GridBox = Frame.MakeQueryBox(FIntVector(0), GridSize);

            Frame.Scene->overlap(GridBox.Geometry, GridBox.Pose, UnusedHits, Filter, &Collector);
        }

        Report.Candidates = Collector.Candidates.Num();

        const double Gathered = FPlatformTime::Seconds();

        TArray<FVoxelGpuInstance> Instances;
        TArray<FVoxelGpuShape> Shapes;
        TArray<FVoxelGpuAxis> Axes;

        TArray<uint32> Indices;
        TArray<float> Vertices;
        TSet<uint64> Sent;

        // Cache a mesh on the GPU, or refresh the vertices of one that can
        // deform, once per frame. A failed upload fails the frame, which then
        // goes to the CPU path instead of leaving the shape out.
        const auto Upload = [&](
            uint64 MeshKey,
            const float* MeshVertices,
            uint32 NumVertices,
            const uint32* MeshIndices,
            uint32 NumTriangles
        )
        {
            const FVoxelGpuMesh Mesh{MeshKey, MeshVertices, NumVertices, MeshIndices, NumTriangles};

            if (Gpu.UploadMesh(Context, &Mesh) != 0)
            {
                OutError = UTF8_TO_TCHAR(Gpu.Error(Context));

                return false;
            }

            Sent.Add(MeshKey);

            ++Report.Uploads;

            Report.UploadBytes += 12 * int64(NumVertices) + (MeshIndices ? 12 * int64(NumTriangles) : 0);

            return true;
        };

        for (const FVoxelCandidate& Candidate : Collector.Candidates)
        {
            const uint32 InstanceId = Candidate.bInstanced ? Candidate.Instance : 0;

            const uint32 Key = VOXEL_GPU_KEY(Candidate.Priority, Candidate.Label, InstanceId);

            const FAffine ShapeToGrid = WorldToGrid * FAffine::FromPose(Candidate.Pose);

            switch (Candidate.Geometry.getType())
            {
            case physx::PxGeometryType::eTRIANGLEMESH:
            {
                const physx::PxTriangleMeshGeometry& Geometry = Candidate.Geometry.triangleMesh();
                const physx::PxTriangleMesh& Mesh = *Geometry.triangleMesh;

                const uint64 MeshKey = uint64(UPTRINT(&Mesh));
                const uint32 NumVertices = Mesh.getNbVertices();
                const uint32 NumTriangles = Mesh.getNbTriangles();
                const float* MeshVertices = reinterpret_cast<const float*>(Mesh.getVertices());

                uint32 CachedVertices = 0;
                uint32 CachedTriangles = 0;

                const bool bCached = Gpu.FindMesh(Context, MeshKey, &CachedVertices, &CachedTriangles) &&
                                     CachedVertices == NumVertices &&
                                     CachedTriangles == NumTriangles;

                if (!bCached)
                {
                    Indices.SetNumUninitialized(3 * NumTriangles);

                    if (Mesh.getTriangleMeshFlags() & physx::PxTriangleMeshFlag::e16_BIT_INDICES)
                    {
                        const physx::PxU16* Source = static_cast<const physx::PxU16*>(Mesh.getTriangles());

                        for (uint32 Index = 0; Index < 3 * NumTriangles; ++Index)
                            Indices[Index] = Source[Index];
                    }
                    else
                        FMemory::Memcpy(Indices.GetData(), Mesh.getTriangles(), 12 * size_t(NumTriangles));

                    if (!Upload(MeshKey, MeshVertices, NumVertices, Indices.GetData(), NumTriangles))
                        return false;
                }
                else if (Candidate.bDynamic && !Sent.Contains(MeshKey))
                {
                    if (!Upload(MeshKey, MeshVertices, NumVertices, nullptr, NumTriangles))
                        return false;
                }

                FVoxelGpuInstance Instance{};

                const FAffine MeshToGrid = ShapeToGrid * FAffine::FromMatrix(Geometry.scale.toMat33());

                Instance.MeshKey = MeshKey;
                Instance.Key = Key;

                FMemory::Memcpy(Instance.ToGrid, MeshToGrid.M, sizeof(Instance.ToGrid));

                Instances.Add(Instance);

                ++Report.Meshes;

                break;
            }
            case physx::PxGeometryType::eHEIGHTFIELD:
            {
                // Landscapes do not move, so their triangles are taken once,
                // in world space, without holes.
                const physx::PxHeightFieldGeometry& Geometry = Candidate.Geometry.heightField();
                const physx::PxHeightField& HeightField = *Geometry.heightField;

                const uint64 MeshKey = uint64(UPTRINT(&HeightField)) | 1;

                uint32 CachedVertices = 0;
                uint32 CachedTriangles = 0;

                if (!Gpu.FindMesh(Context, MeshKey, &CachedVertices, &CachedTriangles))
                {
                    // PhysX numbers the two triangles of the cell at a row and
                    // column 2 * (Row * Columns + Column), so the last row and
                    // column of samples start no cell.
                    const uint32 NumRows = HeightField.getNbRows();
                    const uint32 NumColumns = HeightField.getNbColumns();

                    Vertices.Reset();
                    Indices.Reset();

                    for (uint32 Row = 0; Row + 1 < NumRows; ++Row)
                    {
                        for (uint32 Column = 0; Column + 1 < NumColumns; ++Column)
                        {
                            for (uint32 Half = 0; Half < 2; ++Half)
                            {
                                const uint32 Index = 2 * (Row * NumColumns + Column) + Half;

                                const bool bHole = HeightField.getTriangleMaterialIndex(Index) ==
                                                   physx::PxHeightFieldMaterial::eHOLE;

                                if (bHole)
                                    continue;

                                physx::PxTriangle Triangle;

                                physx::PxMeshQuery::getTriangle(Geometry, Candidate.Pose, Index, Triangle);

                                for (int Vertex = 0; Vertex < 3; ++Vertex)
                                {
                                    Indices.Add(uint32(Vertices.Num() / 3));

                                    Vertices.Add(Triangle.verts[Vertex].x);
                                    Vertices.Add(Triangle.verts[Vertex].y);
                                    Vertices.Add(Triangle.verts[Vertex].z);
                                }
                            }
                        }
                    }

                    if (Indices.Num() == 0)
                        break;

                    const bool bUploaded = Upload(
                        MeshKey,
                        Vertices.GetData(),
                        uint32(Vertices.Num() / 3),
                        Indices.GetData(),
                        uint32(Indices.Num() / 3)
                    );

                    if (!bUploaded)
                        return false;
                }

                // PhysX counts a layer of |Thickness| on the side away from
                // Up, and Unreal leaves the thickness negative.
                const float Thickness = HeightField.getThickness();

                const float Side = Thickness > 0.0f ? -1.0f : 1.0f;

                const physx::PxVec3 Up = Candidate.Pose.q.rotate(physx::PxVec3(0.0f, Side, 0.0f));

                const FPrecise GridUp = WorldToGrid.Rotate(FPrecise{Up.x, Up.y, Up.z});

                const double UpLength = std::sqrt(Dot(GridUp, GridUp));

                FVoxelGpuInstance Instance{};

                Instance.MeshKey = MeshKey;
                Instance.Key = Key;
                Instance.Flags = VOXEL_GPU_HEIGHTFIELD;
                Instance.Skin = std::abs(Thickness) / Frame.BoxSize;

                for (int Axis = 0; Axis < 3; ++Axis)
                    Instance.Up[Axis] = GridUp[Axis] / UpLength;

                FMemory::Memcpy(Instance.ToGrid, WorldToGrid.M, sizeof(Instance.ToGrid));

                Instances.Add(Instance);

                ++Report.HeightFields;

                break;
            }
            case physx::PxGeometryType::eSPHERE:
            {
                FVoxelGpuShape Shape{};

                const FPrecise Center = ShapeToGrid.Apply(FPrecise{0.0, 0.0, 0.0});

                Shape.Type = VOXEL_GPU_SPHERE;
                Shape.Key = Key;
                Shape.Radius = Candidate.Geometry.sphere().radius / Frame.BoxSize;

                for (int Axis = 0; Axis < 3; ++Axis)
                    Shape.Points[0][Axis] = Center[Axis];

                SetVoxelBox(
                    Shape,
                    FPrecise{Center[0] - Shape.Radius, Center[1] - Shape.Radius, Center[2] - Shape.Radius},
                    FPrecise{Center[0] + Shape.Radius, Center[1] + Shape.Radius, Center[2] + Shape.Radius},
                    GridSize
                );

                Shapes.Add(Shape);

                ++Report.Shapes;

                break;
            }
            case physx::PxGeometryType::eCAPSULE:
            {
                // A capsule is a segment along its local x axis grown by its
                // radius.
                const physx::PxCapsuleGeometry& Geometry = Candidate.Geometry.capsule();

                FVoxelGpuShape Shape{};

                const FPrecise Ends[2] = {
                    ShapeToGrid.Apply(FPrecise{-Geometry.halfHeight, 0.0, 0.0}),
                    ShapeToGrid.Apply(FPrecise{Geometry.halfHeight, 0.0, 0.0})
                };

                Shape.Type = VOXEL_GPU_CAPSULE;
                Shape.Key = Key;
                Shape.Radius = Geometry.radius / Frame.BoxSize;

                FPrecise Low;
                FPrecise High;

                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    Shape.Points[0][Axis] = Ends[0][Axis];
                    Shape.Points[1][Axis] = Ends[1][Axis];

                    Low[Axis] = FMath::Min(Ends[0][Axis], Ends[1][Axis]) - Shape.Radius;
                    High[Axis] = FMath::Max(Ends[0][Axis], Ends[1][Axis]) + Shape.Radius;
                }

                SetVoxelBox(Shape, Low, High, GridSize);

                Shapes.Add(Shape);

                ++Report.Shapes;

                break;
            }
            case physx::PxGeometryType::eBOX:
            {
                const physx::PxVec3 Half = Candidate.Geometry.box().halfExtents;

                TArray<FPrecise> Points;

                for (int Corner = 0; Corner < 8; ++Corner)
                {
                    Points.Add(ShapeToGrid.Apply(FPrecise{
                        (Corner & 1) ? Half.x : -Half.x,
                        (Corner & 2) ? Half.y : -Half.y,
                        (Corner & 4) ? Half.z : -Half.z
                    }));
                }

                const TArray<FPrecise> Sides = {
                    ShapeToGrid.Rotate(FPrecise{1.0, 0.0, 0.0}),
                    ShapeToGrid.Rotate(FPrecise{0.0, 1.0, 0.0}),
                    ShapeToGrid.Rotate(FPrecise{0.0, 0.0, 1.0})
                };

                FVoxelGpuShape Shape{};

                Shape.Key = Key;

                AddHull(Points, Sides, Sides, GridSize, Shape, Axes);

                Shapes.Add(Shape);

                ++Report.Shapes;

                break;
            }
            case physx::PxGeometryType::eCONVEXMESH:
            {
                const physx::PxConvexMeshGeometry& Geometry = Candidate.Geometry.convexMesh();
                const physx::PxConvexMesh& Mesh = *Geometry.convexMesh;

                const FAffine HullToGrid = ShapeToGrid * FAffine::FromMatrix(Geometry.scale.toMat33());

                TArray<FPrecise> Points;

                for (physx::PxU32 Vertex = 0; Vertex < Mesh.getNbVertices(); ++Vertex)
                {
                    const physx::PxVec3& Point = Mesh.getVertices()[Vertex];

                    Points.Add(HullToGrid.Apply(FPrecise{Point.x, Point.y, Point.z}));
                }

                TArray<FPrecise> Normals;
                TArray<FPrecise> Edges;

                const physx::PxU8* HullIndices = Mesh.getIndexBuffer();

                for (physx::PxU32 Face = 0; Face < Mesh.getNbPolygons(); ++Face)
                {
                    physx::PxHullPolygon Polygon;

                    Mesh.getPolygonData(Face, Polygon);

                    // Newell's method gives the face normal from its vertices.
                    FPrecise Normal{0.0, 0.0, 0.0};

                    for (physx::PxU16 Corner = 0; Corner < Polygon.mNbVerts; ++Corner)
                    {
                        const FPrecise& A = Points[HullIndices[Polygon.mIndexBase + Corner]];
                        const FPrecise& B = Points[HullIndices[Polygon.mIndexBase + (Corner + 1) % Polygon.mNbVerts]];

                        Normal[0] += (A[1] - B[1]) * (A[2] + B[2]);
                        Normal[1] += (A[2] - B[2]) * (A[0] + B[0]);
                        Normal[2] += (A[0] - B[0]) * (A[1] + B[1]);

                        Edges.Add(Subtract(B, A));
                    }

                    Normals.Add(Normal);
                }

                FVoxelGpuShape Shape{};

                Shape.Key = Key;

                AddHull(Points, Normals, Edges, GridSize, Shape, Axes);

                Shapes.Add(Shape);

                ++Report.Shapes;

                break;
            }
            default:
                ++Report.Unsupported;

                break;
            }
        }

        const double Built = FPlatformTime::Seconds();

        const FVoxelGpuGrid Grid{{Frame.GridSizeX, Frame.GridSizeY, Frame.GridSizeZ}, Frame.InstancedLabels};

        const int32 Status = Gpu.Run(
            Context,
            &Grid,
            Instances.GetData(),
            uint32(Instances.Num()),
            Shapes.GetData(),
            uint32(Shapes.Num()),
            Axes.GetData(),
            uint32(Axes.Num()),
            Labels,
            &Report.Stats
        );

        if (Status != 0)
        {
            OutError = UTF8_TO_TCHAR(Gpu.Error(Context));

            return false;
        }

        const FVoxelGpuPair* Pairs = nullptr;
        uint64_t NumPairs = 0;

        Gpu.GetPairs(Context, &Pairs, &NumPairs);

        static_assert(sizeof(FVoxelGpuPair) == sizeof(FVoxelInstance), "the GPU pairs are sent as they are");

        OutInstances.SetNumUninitialized(int32(NumPairs));

        FMemory::Memcpy(OutInstances.GetData(), Pairs, NumPairs * sizeof(FVoxelInstance));

        const double End = FPlatformTime::Seconds();

        Report.GatherMs = 1000.0 * (Gathered - Start);
        Report.GeometryMs = 1000.0 * (Built - Gathered);
        Report.RunMs = 1000.0 * (End - Built);

        return true;
    }

    void LogGpuReport(const FVoxelGpuReport& Report)
    {
        const FVoxelGpuStats& Stats = Report.Stats;

        carla::log_warning("Voxel GPU time (ms): gather=", Report.GatherMs,
            ", geometry=", Report.GeometryMs,
            ", GPU run=", Report.RunMs,
            " (prepare=", Stats.PrepareMs,
            ", raster=", Stats.RasterMs,
            ", shapes=", Stats.ShapesMs,
            ", resolve=", Stats.ResolveMs,
            ", readback=", Stats.ReadbackMs, ")");

        carla::log_warning("Voxel GPU counts: candidates=", Report.Candidates,
            ", meshes=", Report.Meshes,
            ", heightfields=", Report.HeightFields,
            ", shapes=", Report.Shapes,
            ", unsupported=", Report.Unsupported,
            ", uploads=", Report.Uploads,
            ", uploaded MB=", double(Report.UploadBytes) / 1e6,
            ", triangles=", Stats.Triangles,
            ", columns=", Stats.Columns,
            ", shape voxels=", Stats.ShapeVoxels,
            ", double-precision tests=", Stats.ExactTests,
            ", instances=", Stats.Pairs,
            ", slabs=", Stats.Slabs);
    }

    void LogProfile(const TArray<FVoxelProfile>& Profiles, int64 TotalTiles)
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
            ", voxel PhysX=", FPlatformTime::ToMilliseconds64(Profile.VoxelQueryCycles),
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
            ", box triangles=", Raster.BoxTriangles,
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
            ", voxel queries=", Raster.PrimitiveQueries,
            ", voxel hits=", Raster.PrimitiveHits,
            ", ties=", Profile.Ties,
            ", instances=", Profile.Instances);
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
    auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(
        TEXT("other"),
        TEXT("voxel_detection")
    );

    // Grid limits in meters in the sensor frame, where X points forward, Y
    // right and Z up.
    FActorVariation Front;
    Front.Id = TEXT("front_limit");
    Front.Type = EActorAttributeType::Float;
    Front.RecommendedValues = { TEXT("50.0") };
    Front.bRestrictToRecommended = false;

    FActorVariation Back;
    Back.Id = TEXT("back_limit");
    Back.Type = EActorAttributeType::Float;
    Back.RecommendedValues = { TEXT("-50.0") };
    Back.bRestrictToRecommended = false;

    FActorVariation Left;
    Left.Id = TEXT("left_limit");
    Left.Type = EActorAttributeType::Float;
    Left.RecommendedValues = { TEXT("-50.0") };
    Left.bRestrictToRecommended = false;

    FActorVariation Right;
    Right.Id = TEXT("right_limit");
    Right.Type = EActorAttributeType::Float;
    Right.RecommendedValues = { TEXT("50.0") };
    Right.bRestrictToRecommended = false;

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

    FActorVariation InstanceIds;
    InstanceIds.Id = TEXT("instance_ids");
    InstanceIds.Type = EActorAttributeType::Bool;
    InstanceIds.RecommendedValues = { TEXT("False") };
    InstanceIds.bRestrictToRecommended = false;

    FActorVariation UseGpuVariation;
    UseGpuVariation.Id = TEXT("use_gpu");
    UseGpuVariation.Type = EActorAttributeType::Bool;
    UseGpuVariation.RecommendedValues = { TEXT("False") };
    UseGpuVariation.bRestrictToRecommended = false;

    FActorVariation GpuDeviceVariation;
    GpuDeviceVariation.Id = TEXT("gpu_device");
    GpuDeviceVariation.Type = EActorAttributeType::Int;
    GpuDeviceVariation.RecommendedValues = { TEXT("0") };
    GpuDeviceVariation.bRestrictToRecommended = false;

    FActorVariation OracleStride;
    OracleStride.Id = TEXT("oracle_stride");
    OracleStride.Type = EActorAttributeType::Int;
    OracleStride.RecommendedValues = { TEXT("0") };
    OracleStride.bRestrictToRecommended = false;

    Definition.Variations.Append(
        {
            Front,
            Back,
            Left,
            Right,
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
            InstanceIds,
            OracleStride,
            UseGpuVariation,
            GpuDeviceVariation,
        }
    );

    return Definition;
}

void AVoxelDetectionSensor::Set(const FActorDescription &Description)
{
    Super::Set(Description);

    constexpr float M_TO_CM = 100.0f;

    const auto GetFloat = [&](const FString& Id, float Default)
    {
        return UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(Id, Description.Variations, Default);
    };

    const auto GetInt = [&](const FString& Id, int32 Default)
    {
        return UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(Id, Description.Variations, Default);
    };

    const auto GetBool = [&](const FString& Id, bool Default)
    {
        return UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool(Id, Description.Variations, Default);
    };

    Front = M_TO_CM * GetFloat("front_limit", 50.0f);

    Back = M_TO_CM * GetFloat("back_limit", -50.0f);

    Left = M_TO_CM * GetFloat("left_limit", -50.0f);

    Right = M_TO_CM * GetFloat("right_limit", 50.0f);

    Top = M_TO_CM * GetFloat("upper_limit", 10.0f);

    Bottom = M_TO_CM * GetFloat("lower_limit", -5.0f);

    BoxSize = M_TO_CM * GetFloat("voxel_size", 0.5f);

    SelfIgnore = GetBool("ignore_self", false);

    DrawDebug = GetBool("draw_debug", false);

    ShowCalculationTime = GetBool("show_calculation_time", false);

    UseTraceComplex = GetBool("use_complex_collision", true);

    // Voxels are processed in tiles of this size, rounded up to whole
    // 4 x 4 x 8 bricks.
    TileX = GetInt("tile_x", 32);

    TileY = GetInt("tile_y", 32);

    TileZ = GetInt("tile_z", 128);

    InstanceIds = GetBool("instance_ids", false);

    // Zero disables the comparison with the per-voxel engine query.
    OracleStride = FMath::Max(0, GetInt("oracle_stride", 0));

    // Label on a CUDA GPU without PhysX queries, if the GPU library is there.
    UseGpu = GetBool("use_gpu", false);

    GpuDevice = FMath::Max(0, GetInt("gpu_device", 0));

    // Voxels along one axis, or zero unless the count is positive and below
    // 2^31, so it fits an int32. The count is checked before its integer
    // conversion, so infinite, NaN and huge counts cannot overflow.
    const auto GetGridSize = [](float Length, float Size)
    {
        const float Count = Length / Size;

        return (Count > 0.0f && Count < float(MAX_int32)) ? int64(FMath::CeilToDouble(Count)) : int64(0);
    };

    const int64 SizeX = GetGridSize(Front - Back, BoxSize);
    const int64 SizeY = GetGridSize(Right - Left, BoxSize);
    const int64 SizeZ = GetGridSize(Top - Bottom, BoxSize);

    // Voxels are indexed with 64-bit integers, so the limit is memory: the
    // labels of one frame may take at most half of it, since the stream's
    // pool keeps up to two buffers. Each size is at most 2^31, so
    // SizeX * SizeY fits an int64, and dividing the limit avoids the last
    // multiplication. Send empty frames instead.
    const int64 MaxVoxels = int64(FPlatformMemory::GetConstants().TotalPhysical / 2);

    const bool bValidGrid = SizeX > 0 && SizeY > 0 && SizeZ > 0 && SizeX * SizeY <= MaxVoxels / SizeZ;

    GridSizeX = bValidGrid ? int32(SizeX) : 0;
    GridSizeY = bValidGrid ? int32(SizeY) : 0;
    GridSizeZ = bValidGrid ? int32(SizeZ) : 0;

    if (!bValidGrid)
    {
        UE_LOG(
            LogCarla,
            Error,
            TEXT("VoxelDetectionSensor: limits x %g to %g m, y %g to %g m, z %g to %g m and voxel size %g m ")
            TEXT("do not give a grid of 1 to %lld voxels, half of this machine's memory in bytes. Each front, ")
            TEXT("right and upper limit must exceed its opposite limit; otherwise increase voxel_size or bring ")
            TEXT("the limits closer."),
            Back / M_TO_CM,
            Front / M_TO_CM,
            Left / M_TO_CM,
            Right / M_TO_CM,
            Bottom / M_TO_CM,
            Top / M_TO_CM,
            BoxSize / M_TO_CM,
            MaxVoxels
        );
    }

    UE_LOG(
        LogCarla,
        Log,
        TEXT("VoxelDetectionSensor: Grid size %d x %d x %d = %lld voxels"),
        GridSizeX,
        GridSizeY,
        GridSizeZ,
        int64(GridSizeX) * GridSizeY * GridSizeZ
    );
}

void AVoxelDetectionSensor::EndPlay(EEndPlayReason::Type EndPlayReason)
{
    if (GpuContext)
    {
        FVoxelGpuLibrary::Get().Destroy(GpuContext);

        GpuContext = nullptr;
    }

    Super::EndPlay(EndPlayReason);
}

void AVoxelDetectionSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AVoxelDetectionSensor::PostPhysTick);

    const double StartTime = FPlatformTime::Seconds();

    const int64 TotalVoxels = int64(GridSizeX) * GridSizeY * GridSizeZ;

    // Labels go straight into a pooled stream buffer, with room after them
    // for the voxels that have an instance. The pool keeps one or two
    // allocations of this size per sensor.
    auto DataStream = GetDataStream(*this);

    carla::Buffer Voxels = DataStream.PopBufferFromPool();

    const auto GetBufferSize = [&](int64 NumInstances)
    {
        return carla::Buffer::size_type(TotalVoxels + NumInstances * sizeof(FVoxelInstance));
    };

    Voxels.reset(GetBufferSize(InstanceIds ? InstanceReserve : 0));

    uint8* Labels = Voxels.data();

    // Set up collision query.
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);

    QueryParams.bTraceComplex = UseTraceComplex;
    QueryParams.bReturnPhysicalMaterial = false;

    if (SelfIgnore && GetOwner())
        QueryParams.AddIgnoredActor(GetOwner());

    const FVoxelFrame Frame(
        World,
        GetTransform(),
        QueryParams,
        BoxSize,
        FVector(Back, Left, Bottom),
        GridSizeX,
        GridSizeY,
        GridSizeZ,
        SemanticPriority,
        UE_ARRAY_COUNT(SemanticPriority),
        InstanceIds ? GetInstancedLabels(UE_ARRAY_COUNT(SemanticPriority)) : 0,
        GetEpisode()
    );

    const FIntVector GridSize(GridSizeX, GridSizeY, GridSizeZ);

    const FIntVector TileSize = GetTileSize(FIntVector(TileX, TileY, TileZ), GridSize);

    const double ParallelStart = FPlatformTime::Seconds();

    TArray<FVoxelProfile> Profiles;

    TArray<TArray<FVoxelInstance>> Instances;

    int64 TotalTiles = 0;

    // The GPU path, when asked for and available; the CPU path otherwise.
    bool bOnGpu = false;

    FVoxelGpuReport GpuReport;

    if (UseGpu && Frame.IsSupported())
    {
        const FVoxelGpuLibrary& Gpu = FVoxelGpuLibrary::Get();

        FString GpuError = Gpu.LoadError;

        if (Gpu.IsLoaded() && !GpuContext)
        {
            char Error[256] = {};

            if (Gpu.Create(GpuDevice, &GpuContext, Error, sizeof(Error)) != 0)
                GpuError = UTF8_TO_TCHAR(Error);
        }

        if (GpuContext)
        {
            TArray<FVoxelInstance> GpuInstances;

            bOnGpu = LabelOnGpu(Frame, GpuContext, Labels, GpuInstances, GpuReport, GpuError);

            if (bOnGpu)
                Instances.Add(MoveTemp(GpuInstances));
        }

        // Shipping builds drop UE_LOG, so the reason also goes to the
        // simulator's output.
        if (!bOnGpu && !bReportedGpuFailure)
        {
            UE_LOG(
                LogCarla,
                Error,
                TEXT("VoxelDetectionSensor: use_gpu failed (%s); using the CPU path."),
                *GpuError
            );

            carla::log_warning(
                "VoxelDetectionSensor: use_gpu failed (",
                TCHAR_TO_UTF8(*GpuError),
                "); using the CPU path."
            );

            bReportedGpuFailure = true;
        }
    }

    if (!bOnGpu && Frame.IsSupported())
        TotalTiles = RunTiles(Frame, TileSize, Labels, ShowCalculationTime, Instances, Profiles);
    else if (!bOnGpu)
    {
        // A collapsed or non-finite sensor transform, or a voxel size of
        // almost zero, has no meaningful grid.
        FMemory::Memzero(Labels, TotalVoxels);

        if (!bReportedUnsupported)
        {
            UE_LOG(
                LogCarla,
                Warning,
                TEXT("VoxelDetectionSensor: unsupported sensor scale, transform or voxel size; sending empty grids.")
            );

            bReportedUnsupported = true;
        }
    }

    const double ParallelEnd = FPlatformTime::Seconds();

    int64 NumInstances = 0;

    for (const TArray<FVoxelInstance>& Batch : Instances)
    {
        NumInstances += Batch.Num();
    }

    // A frame that outgrows the reserve moves to a larger buffer, which the
    // pool then keeps.
    if (NumInstances > InstanceReserve)
    {
        InstanceReserve = 2 * NumInstances;

        carla::Buffer Larger = DataStream.PopBufferFromPool();

        Larger.reset(GetBufferSize(InstanceReserve));

        FMemory::Memcpy(Larger.data(), Labels, TotalVoxels);

        Voxels = std::move(Larger);

        Labels = Voxels.data();
    }

    Voxels.reset(GetBufferSize(NumInstances));

    uint8* Tail = Labels + TotalVoxels;

    for (const TArray<FVoxelInstance>& Batch : Instances)
    {
        const int64 Size = int64(Batch.Num()) * sizeof(FVoxelInstance);

        FMemory::Memcpy(Tail, Batch.GetData(), Size);

        Tail += Size;
    }

    const double InstanceEnd = FPlatformTime::Seconds();

    if (OracleStride > 0)
        CheckAgainstOracle(Frame, Labels, Instances, OracleStride);

    const double OracleEnd = FPlatformTime::Seconds();

    // Debug visualization.
    if (DrawDebug)
    {
        const FQuat ActorQuat = GetActorQuat();

        for (int64 FlatIndex = 0; FlatIndex < TotalVoxels; ++FlatIndex)
        {
            const uint8 Label = Labels[FlatIndex];

            if (Label > 0)
            {
                const FIntVector Voxel = Frame.GetVoxel(FlatIndex);

                FVector Pos = Frame.GetWorldPosition(Voxel.X, Voxel.Y, Voxel.Z);

                FLinearColor Color = ColorMap.Contains(Label) ? ColorMap[Label] : FLinearColor::Green;

                DrawDebugBox(
                    World,
                    Pos,
                    Frame.BoxExtent,
                    ActorQuat,
                    Color.ToFColor(true),
                    false,
                    DeltaTime * 1.1f,
                    0,
                    2.0f
                );
            }
        }
    }

    const double SendStart = FPlatformTime::Seconds();

    // Send data. This is what SerializeAndSend does after serializing,
    // without copying the voxels.
    DataStream.Send(*this, carla::BufferView::CreateFrom(std::move(Voxels)));

    const double EndTime = FPlatformTime::Seconds();

    if (ShowCalculationTime && bOnGpu)
    {
        carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms (GPU ", GpuDevice,
            ", slabs: ", GpuReport.Stats.Slabs, ")");

        carla::log_warning("Voxel wall time (ms): setup=", 1000.0 * (ParallelStart - StartTime),
            ", labeling=", 1000.0 * (ParallelEnd - ParallelStart),
            ", instances=", 1000.0 * (InstanceEnd - ParallelEnd),
            ", oracle=", 1000.0 * (OracleEnd - InstanceEnd),
            ", debug=", 1000.0 * (SendStart - OracleEnd),
            ", send=", 1000.0 * (EndTime - SendStart));

        LogGpuReport(GpuReport);
    }
    else if (ShowCalculationTime)
    {
        carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms (tiles: ", TotalTiles,
            " of ", TileSize.X, "x", TileSize.Y, "x", TileSize.Z, ", batches: ", Profiles.Num(), ")");

        carla::log_warning("Voxel wall time (ms): setup=", 1000.0 * (ParallelStart - StartTime),
            ", parallel=", 1000.0 * (ParallelEnd - ParallelStart),
            ", instances=", 1000.0 * (InstanceEnd - ParallelEnd),
            ", oracle=", 1000.0 * (OracleEnd - InstanceEnd),
            ", debug=", 1000.0 * (SendStart - OracleEnd),
            ", send=", 1000.0 * (EndTime - SendStart));

        LogProfile(Profiles, TotalTiles);
    }
}
