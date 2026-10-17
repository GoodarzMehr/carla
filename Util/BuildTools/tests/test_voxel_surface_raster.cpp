// Standalone check of the voxel sensor's surface raster against the actual PhysX geometry queries.
// From the repository root, with UE4_ROOT pointing to UnrealEngine_4.26:
//   px="$UE4_ROOT/Engine/Source/ThirdParty/PhysX3"
//   libs="$px/Lib/Linux/x86_64-unknown-linux-gnu"
//   g++ -std=c++14 -O2 -DNDEBUG -DPX_PHYSX_STATIC_LIB \
//     -I"$px/PhysX_3.4/Include" -I"$px/PxShared/include" \
//     Util/BuildTools/tests/test_voxel_surface_raster.cpp -Wl,--start-group \
//     "$libs/libPhysX3.a" "$libs/libPhysX3Common.a" "$libs/libPxFoundation.a" \
//     "$libs/libPxPvdSDK.a" "$libs/libPhysX3Cooking.a" -Wl,--end-group \
//     -pthread -ldl -o /tmp/test_voxel_surface_raster
//   /tmp/test_voxel_surface_raster

#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelSurfaceRaster.h"
#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelTriangleOverlap.h"
#include "PxPhysicsAPI.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

using namespace physx;

struct Allocator final : PxAllocatorCallback
{
    void* allocate(size_t size, const char*, const char*, int) override
    {
        void* result = nullptr;
        return posix_memalign(&result, 16, size) ? nullptr : result;
    }
    void deallocate(void* data) override { free(data); }
};

struct Errors final : PxErrorCallback
{
    void reportError(PxErrorCode::Enum code, const char* message, const char*, int) override
    {
        // Unreal drops logging info, such as the heightfield query's note that it flushed a full buffer.
        if (code != PxErrorCode::eLOGGING_INFO) fprintf(stderr, "PhysX: %s\n", message);
    }
};

struct Buffer final : PxOutputStream, PxInputData
{
    std::vector<PxU8> bytes;
    PxU32 position = 0;
    PxU32 write(const void* source, PxU32 count) override
    {
        const PxU8* data = static_cast<const PxU8*>(source);
        bytes.insert(bytes.end(), data, data + count);
        return count;
    }
    PxU32 read(void* destination, PxU32 count) override
    {
        count = std::min(count, getLength() - position);
        memcpy(destination, bytes.data() + position, count);
        position += count;
        return count;
    }
    PxU32 getLength() const override { return PxU32(bytes.size()); }
    PxU32 tell() const override { return position; }
    void seek(PxU32 offset) override { position = std::min(offset, getLength()); }
};

static PxTriangleMesh* makeMesh(PxPhysics& physics, PxCooking& cooking,
    const std::vector<PxVec3>& vertices, const std::vector<PxU32>& indices)
{
    PxTriangleMeshDesc desc;
    desc.points.count = PxU32(vertices.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = vertices.data();
    desc.triangles.count = PxU32(indices.size() / 3);
    desc.triangles.stride = 3 * sizeof(PxU32);
    desc.triangles.data = indices.data();
    Buffer buffer;
    if (!cooking.cookTriangleMesh(desc, buffer)) throw std::runtime_error("Cooking failed");
    PxTriangleMesh* result = physics.createTriangleMesh(buffer);
    if (!result) throw std::runtime_error("Mesh creation failed");
    return result;
}

static PxConvexMesh* makeConvex(PxPhysics& physics, PxCooking& cooking, const std::vector<PxVec3>& points)
{
    PxConvexMeshDesc desc;
    desc.points.count = PxU32(points.size());
    desc.points.stride = sizeof(PxVec3);
    desc.points.data = points.data();
    desc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    Buffer buffer;
    if (!cooking.cookConvexMesh(desc, buffer)) throw std::runtime_error("Convex cooking failed");
    PxConvexMesh* result = physics.createConvexMesh(buffer);
    if (!result) throw std::runtime_error("Convex creation failed");
    return result;
}

// One tile of a sensor grid: voxel (x, y, z) has local center origin + (index + .5) * voxel, scaled
// and placed by the sensor pose, as in AVoxelDetectionSensor.
struct Tile
{
    int size[3];
    float voxel;
    PxVec3 scale;
    PxTransform sensor;
    PxVec3 origin;
};

// Raster environment with a single candidate; the reference is the bounds gate plus PhysX per voxel.
struct TileEnv
{
    VoxelSurfaceRaster::FLattice lattice;
    std::vector<PxVec3> world;
    PxTransform tileFrame;
    PxQuat rotation;
    PxBoxGeometry box;
    PxVec3 worldExtent;
    const PxGeometry* geometry = nullptr;
    PxTransform pose;
    PxBounds3 bounds;
    std::vector<unsigned char> state;
    VoxelSurfaceRaster::FVoxelSet deferred;

    int index(int x, int y, int z) const { return lattice.GetIndex(x, y, z); }
    bool IsOpen(int x, int y, int z) const { return !state[index(x, y, z)]; }
    bool GetCenter(int x, int y, int z, PxVec3& center) const
    {
        const PxVec3& position = world[index(x, y, z)];
        if (!bounds.intersects(PxBounds3::centerExtents(position, worldExtent))) return false;
        center = tileFrame.transformInv(position);
        return true;
    }
    void Hit(int x, int y, int z) { state[index(x, y, z)] = 1; }
    void Defer(int x, int y, int z) { deferred.Add(index(x, y, z)); }
    bool Overlaps(int x, int y, int z) const
    {
        return PxGeometryQuery::overlap(box, PxTransform(world[index(x, y, z)], rotation), *geometry, pose);
    }
    bool OverlapsBrick(const int32_t* begin, const int32_t* end) const
    {
        PxBounds3 brick = PxBounds3::empty();
        for (int corner = 0; corner < 8; ++corner)
            brick.include(world[index(corner & 1 ? end[0] - 1 : begin[0],
                corner & 2 ? end[1] - 1 : begin[1], corner & 4 ? end[2] - 1 : begin[2])]);
        return PxGeometryQuery::overlap(PxBoxGeometry(brick.getExtents() + worldExtent + PxVec3(.1f)),
            PxTransform(brick.getCenter()), *geometry, pose);
    }
    bool expected(int i) const
    {
        return bounds.intersects(PxBounds3::centerExtents(world[i], worldExtent)) &&
            PxGeometryQuery::overlap(box, PxTransform(world[i], rotation), *geometry, pose);
    }
};

static void buildTile(const Tile& tile, double padding, TileEnv& env)
{
    const auto toWorld = [&](const PxVec3& local) { return tile.sensor.transform(tile.scale.multiply(local)); };
    const PxVec3 extent(tile.size[0] * tile.voxel, tile.size[1] * tile.voxel, tile.size[2] * tile.voxel);
    env.tileFrame = PxTransform(toWorld(tile.origin + extent * .5f), tile.sensor.q);
    env.rotation = tile.sensor.q;
    env.box = PxBoxGeometry(PxVec3(tile.voxel * .5f));
    env.worldExtent = PxGeometryQuery::getWorldBounds(env.box, PxTransform(PxVec3(0), env.rotation)).getExtents();
    env.world.clear();
    for (int x = 0; x < tile.size[0]; ++x)
        for (int y = 0; y < tile.size[1]; ++y)
            for (int z = 0; z < tile.size[2]; ++z)
                env.world.push_back(toWorld(PxVec3(tile.origin.x + (x + .5f) * tile.voxel,
                    tile.origin.y + (y + .5f) * tile.voxel, tile.origin.z + (z + .5f) * tile.voxel)));
    const double localMagnitude = std::max(VoxelTriangleOverlap::MaxAbs(tile.origin),
        VoxelTriangleOverlap::MaxAbs(tile.origin + extent)) + tile.voxel;
    const double centerError = 32.0 * FLT_EPSILON * (VoxelTriangleOverlap::MaxAbs(tile.sensor.p) +
        2.0 * VoxelTriangleOverlap::MaxAbs(tile.scale) * localMagnitude + 1.0);
    const double step[3] = {double(tile.scale.x) * tile.voxel, double(tile.scale.y) * tile.voxel,
        double(tile.scale.z) * tile.voxel};
    env.lattice = VoxelSurfaceRaster::MakeLattice(tile.size, step, centerError, padding * tile.voxel *
        VoxelTriangleOverlap::MaxAbs(tile.scale), [&](int x, int y, int z)
        { return env.tileFrame.transformInv(env.world[(x * tile.size[1] + y) * tile.size[2] + z]); });
    env.state.assign(env.world.size(), 0);
    env.deferred.Reset(int32_t(env.world.size()));
}

struct Totals
{
    unsigned tiles = 0, voxels = 0, hits = 0, certain = 0;
    VoxelSurfaceRaster::FRasterStats stats;
};

static bool compareTile(const TileEnv& env, const char* what, Totals& totals)
{
    for (size_t i = 0; i < env.world.size(); ++i)
    {
        if (bool(env.state[i]) != env.expected(int(i)))
        {
            fprintf(stderr, "Raster mismatch (%s) voxel=%zu raster=%d physx=%d\n", what, i, env.state[i], !env.state[i]);
            return false;
        }
        totals.hits += env.state[i];
    }
    ++totals.tiles;
    totals.voxels += unsigned(env.world.size());
    return true;
}

// Rasterize every triangle the sensor would fetch for this tile and compare with PhysX per voxel. Also checks
// that each footprint and early-out box holds every voxel where VoxelTriangleOverlap::Test is not Outside, and
// that the test agrees with every voxel the footprint reports as certain.
static bool checkMeshTile(const Tile& tile, const PxTriangleMeshGeometry& geometry, const PxTransform& meshPose,
    double padding, Totals& totals)
{
    TileEnv env;
    buildTile(tile, padding, env);
    if (!env.lattice.IsValid()) { fprintf(stderr, "Invalid lattice\n"); return false; }
    env.geometry = &geometry;
    env.pose = meshPose;
    env.bounds = PxGeometryQuery::getWorldBounds(geometry, meshPose);
    const PxVec3 halfExtent(tile.voxel * .5f);
    PxVec3 fetchExtent;
    for (int axis = 0; axis < 3; ++axis)
        fetchExtent[axis] = float(std::max(std::abs(env.lattice.Center(axis, 0)),
            std::abs(env.lattice.Center(axis, tile.size[axis] - 1))) + env.lattice.Margin) + halfExtent[axis] + .1f;
    std::vector<PxU32> indices(geometry.triangleMesh->getNbTriangles());
    bool overflow = false;
    const PxU32 count = PxMeshQuery::findOverlapTriangleMesh(PxBoxGeometry(fetchExtent), env.tileFrame,
        geometry, meshPose, indices.data(), PxU32(indices.size()), 0, overflow);
    if (overflow) { fprintf(stderr, "Unexpected overflow\n"); return false; }
    const PxTransform meshToTile = env.tileFrame.transformInv(meshPose);
    std::vector<bool> visited(env.world.size());
    for (PxU32 i = 0; i < count; ++i)
    {
        PxTriangle triangle;
        const float error = VoxelTriangleOverlap::FetchTriangle(geometry, meshPose, env.tileFrame, meshToTile,
            indices[i], halfExtent, triangle);
        const VoxelTriangleOverlap::FTriangle prepared = VoxelTriangleOverlap::MakeTriangle(triangle, halfExtent, error);
        if (!prepared.Degenerate)
        {
            std::fill(visited.begin(), visited.end(), false);
            bool sound = true;
            VoxelSurfaceRaster::ForEachTriangleVoxel(prepared, env.lattice, halfExtent, [&](int x, int y, int z, bool certain)
            {
                const int voxel = env.index(x, y, z);
                visited[voxel] = true;
                PxVec3 center;
                if (certain && (!env.GetCenter(x, y, z, center) || VoxelTriangleOverlap::Test(prepared, center) !=
                    VoxelTriangleOverlap::EResult::Overlap))
                {
                    fprintf(stderr, "Unsound certain voxel: triangle=%u voxel=%d\n", indices[i], voxel);
                    sound = false;
                }
                totals.certain += certain;
            });
            if (!sound) return false;
            const VoxelSurfaceRaster::FIndexBox box = VoxelSurfaceRaster::GetTriangleBox(triangle, error, halfExtent, env.lattice);
            for (int x = 0; x < tile.size[0]; ++x)
                for (int y = 0; y < tile.size[1]; ++y)
                    for (int z = 0; z < tile.size[2]; ++z)
                    {
                        const int voxel = env.index(x, y, z);
                        const bool inBox = x >= box.Begin[0] && x < box.End[0] && y >= box.Begin[1] &&
                            y < box.End[1] && z >= box.Begin[2] && z < box.End[2];
                        if ((!visited[voxel] || !inBox) && VoxelTriangleOverlap::Test(prepared,
                            env.tileFrame.transformInv(env.world[voxel])) != VoxelTriangleOverlap::EResult::Outside)
                        {
                            fprintf(stderr, "Incomplete %s: triangle=%u voxel=%d\n", inBox ? "footprint" : "box", indices[i], voxel);
                            return false;
                        }
                    }
        }
        VoxelSurfaceRaster::RasterizeMeshTriangle<true>(triangle, error, halfExtent, env.lattice, env, totals.stats);
    }
    VoxelSurfaceRaster::ResolveDeferred<true>(env.deferred, env.lattice, env, totals.stats);
    return compareTile(env, "mesh", totals);
}

// Heightfield of rows x columns samples with random heights, optional spikes and holes, cooked like a landscape.
static PxHeightField* makeHeightField(PxPhysics& physics, PxCooking& cooking, int rows, int columns,
    int roughness, int spikes, int holes, std::mt19937& random)
{
    std::vector<PxHeightFieldSample> samples(size_t(rows * columns));
    std::uniform_int_distribution<int> height(-roughness, roughness);
    for (PxHeightFieldSample& sample : samples)
    {
        sample.height = PxI16(height(random));
        sample.materialIndex0 = 0;
        sample.materialIndex1 = 0;
    }
    for (int spike = 0; spike < spikes; ++spike)
        samples[random() % samples.size()].height = PxI16((random() % 2 ? 1 : -1) * (4000 + int(random() % 4000)));
    for (int hole = 0; hole < holes; ++hole)
    {
        PxHeightFieldSample& sample = samples[random() % samples.size()];
        sample.materialIndex0 = PxHeightFieldMaterial::eHOLE;
        sample.materialIndex1 = PxHeightFieldMaterial::eHOLE;
    }
    PxHeightFieldDesc desc;
    desc.format = PxHeightFieldFormat::eS16_TM;
    desc.nbRows = PxU32(rows);
    desc.nbColumns = PxU32(columns);
    desc.samples.data = samples.data();
    desc.samples.stride = sizeof(PxHeightFieldSample);
    desc.flags = PxHeightFieldFlag::eNO_BOUNDARY_EDGES;
    PxHeightField* result = cooking.createHeightField(desc, physics.getPhysicsInsertionCallback());
    if (!result) throw std::runtime_error("Heightfield creation failed");
    return result;
}

// Rasterize every heightfield triangle the sensor would fetch for this tile, with the skin and edge-crossing rules,
// and compare with PhysX per voxel.
static bool checkHeightFieldTile(const Tile& tile, const PxHeightFieldGeometry& geometry, const PxTransform& pose,
    Totals& totals, VoxelSurfaceRaster::FRasterStats& surfaceStats)
{
    TileEnv env;
    buildTile(tile, 0.0, env);
    if (!env.lattice.IsValid()) { fprintf(stderr, "Invalid lattice\n"); return false; }
    env.geometry = &geometry;
    env.pose = pose;
    env.bounds = PxGeometryQuery::getWorldBounds(geometry, pose);
    const PxVec3 halfExtent(tile.voxel * .5f);
    const PxTransform hfToTile = env.tileFrame.transformInv(pose);
    VoxelSurfaceRaster::FSurface surface;
    surface.Up = hfToTile.q.rotate(PxVec3(0, 1, 0));
    surface.Skin = std::abs(geometry.heightField->getThickness());
    PxVec3 fetchExtent;
    for (int axis = 0; axis < 3; ++axis)
        fetchExtent[axis] = float(std::max(std::abs(env.lattice.Center(axis, 0)),
            std::abs(env.lattice.Center(axis, tile.size[axis] - 1))) + env.lattice.Margin) + halfExtent[axis] + .1f + surface.Skin;
    std::vector<PxU32> indices(2 * geometry.heightField->getNbRows() * geometry.heightField->getNbColumns());
    bool overflow = false;
    const PxU32 count = PxMeshQuery::findOverlapHeightField(PxBoxGeometry(fetchExtent), env.tileFrame,
        geometry, pose, indices.data(), PxU32(indices.size()), 0, overflow);
    if (overflow) { fprintf(stderr, "Unexpected overflow\n"); return false; }
    VoxelSurfaceRaster::FRasterStats stats;
    for (PxU32 i = 0; i < count; ++i)
    {
        PxTriangle triangle;
        const float error = VoxelTriangleOverlap::FetchTriangle(geometry, pose, env.tileFrame, hfToTile,
            indices[i], halfExtent, triangle);
        VoxelSurfaceRaster::RasterizeMeshTriangle<true>(triangle, error, halfExtent, env.lattice, env, stats, &surface);
    }
    VoxelSurfaceRaster::ResolveDeferred<true>(env.deferred, env.lattice, env, stats);
    surfaceStats.Add(stats);
    totals.stats.Add(stats);
    return compareTile(env, "heightfield", totals);
}

static bool checkPrimitiveTile(const Tile& tile, const PxGeometry& geometry, const PxTransform& pose, Totals& totals)
{
    TileEnv env;
    buildTile(tile, 0.0, env);
    env.geometry = &geometry;
    env.pose = pose;
    env.bounds = PxGeometryQuery::getWorldBounds(geometry, pose);
    const int32_t brick[3] = {4, 4, 8};
    VoxelSurfaceRaster::RasterizeBounds<true>(VoxelSurfaceRaster::GetBoundsBox(env.bounds, env.worldExtent,
        env.tileFrame, env.lattice), brick, env.lattice, env, totals.stats);
    return compareTile(env, "primitive", totals);
}

// The label rules against their definition on random candidate sets: the unique highest-priority label among the
// overlapping candidates, or a tie between two different labels of that priority.
static bool checkLabelRules()
{
    const unsigned char priorities[32] = {0,7,7,1,1,1,1,2,2,1,1,1,6,6,4,3,3,1,5,5,1,1,1,1,8,1,1,1,1,1,2,2};
    const VoxelSurfaceRaster::FLabelRules rules(priorities, 32);
    std::mt19937 random(311);
    struct Candidate { unsigned char label; bool overlap; };
    for (int trial = 0; trial < 200000; ++trial)
    {
        std::vector<Candidate> candidates(1 + random() % 6);
        for (Candidate& candidate : candidates)
            candidate = Candidate{(unsigned char)(1 + random() % 31), random() % 3 == 0};
        std::stable_sort(candidates.begin(), candidates.end(), [&](const Candidate& a, const Candidate& b)
            { return priorities[a.label] > priorities[b.label]; });
        int best = 0, expected = 0;
        for (const Candidate& candidate : candidates)
            if (candidate.overlap) best = std::max(best, int(priorities[candidate.label]));
        for (const Candidate& candidate : candidates)
            if (candidate.overlap && priorities[candidate.label] == best)
                expected = expected == 0 || expected == candidate.label ? candidate.label : -1;
        unsigned char state = 0;
        for (const Candidate& candidate : candidates)
            if (!rules.IsClosed(state, candidate.label, priorities[candidate.label]) && candidate.overlap)
                rules.Apply(state, candidate.label);
        const int actual = state & VoxelSurfaceRaster::TieFlag ? -1 : state;
        if (actual != expected)
        {
            fprintf(stderr, "Label rules: trial=%d expected=%d actual=%d\n", trial, expected, actual);
            return false;
        }
    }
    return true;
}

static bool checkClip()
{
    std::mt19937 random(17);
    std::uniform_real_distribution<double> value(-40, 40);
    VoxelSurfaceRaster::FLattice lattice;
    for (int trial = 0; trial < 100000; ++trial)
    {
        lattice.Size[0] = 1 + random() % 20;
        lattice.Pivot[0] = value(random);
        lattice.Step[0] = (random() % 2 ? 1 : -1) * (.1 + std::abs(value(random)) / 10);
        lattice.InverseStep[0] = 1.0 / lattice.Step[0];
        double low = value(random), high = value(random);
        if (trial % 7 == 0) std::swap(low, high);
        if (trial % 11 == 0) low = -INFINITY;
        if (trial % 13 == 0) high = NAN;
        int32_t begin = int32_t(random() % 3), end = lattice.Size[0] - int32_t(random() % 3);
        const int32_t first = begin, last = end;
        lattice.Clip(0, low, high, begin, end);
        for (int32_t index = first; index < last; ++index)
        {
            const double center = lattice.Center(0, index);
            const bool inside = center >= low && center <= high;
            if (inside != (index >= begin && index < end))
            {
                fprintf(stderr, "Clip: trial=%d index=%d\n", trial, index);
                return false;
            }
        }
    }
    return true;
}

// Degenerate triangles are never tested; the raster defers every voxel they may touch, or the whole tile when
// their vertices are not finite.
static bool checkDegenerate()
{
    for (const PxTriangle triangle : {PxTriangle(PxVec3(0), PxVec3(0), PxVec3(0)),
        PxTriangle(PxVec3(-1, 0, 0), PxVec3(0), PxVec3(1, 0, 0))})
        if (VoxelTriangleOverlap::Test(VoxelTriangleOverlap::MakeTriangle(triangle, PxVec3(5), .001f), PxVec3(0)) !=
            VoxelTriangleOverlap::EResult::Uncertain)
        {
            fprintf(stderr, "Degenerate triangle was tested\n");
            return false;
        }
    const Tile tile{{8, 8, 8}, 10.f, PxVec3(1), PxTransform(PxVec3(3, -2, 1)), PxVec3(-40)};
    PxPlaneGeometry plane;
    for (const PxTriangle triangle : {PxTriangle(PxVec3(-20, 3, -2), PxVec3(0, 3, -2), PxVec3(20, 3, -2)),
        PxTriangle(PxVec3(NAN), PxVec3(0), PxVec3(1))})
    {
        TileEnv env;
        buildTile(tile, 0, env);
        env.geometry = &plane;
        env.bounds = PxBounds3(PxVec3(-1e6f), PxVec3(1e6f));
        VoxelSurfaceRaster::FRasterStats stats;
        VoxelSurfaceRaster::RasterizeDegenerate<false>(triangle, .001f, PxVec3(5), env.lattice, env, stats);
        std::vector<bool> deferred(env.world.size());
        for (const int32_t index : env.deferred.GetItems()) deferred[index] = true;
        const bool finite = triangle.verts[0].isFinite();
        for (size_t voxel = 0; voxel < env.world.size(); ++voxel)
        {
            const PxVec3 center = env.tileFrame.transformInv(env.world[voxel]);
            const bool near = std::abs(center.y - 3) <= 5 && std::abs(center.z + 2) <= 5 && std::abs(center.x) <= 25;
            if ((!finite || near) && !deferred[voxel])
            {
                fprintf(stderr, "Degenerate triangle missed voxel %zu\n", voxel);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    Allocator allocator;
    Errors errors;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, allocator, errors);
    if (!foundation) throw std::runtime_error("PhysX foundation initialization failed");
    PxPhysics* physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), false);
    if (!physics) throw std::runtime_error("PhysX initialization failed");
    PxCooking* cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation,
        PxCookingParams(physics->getTolerancesScale()));
    if (!cooking) throw std::runtime_error("PhysX cooking initialization failed");

    if (!checkLabelRules() || !checkClip() || !checkDegenerate()) return 1;

    // Undulating terrain, a flat plane, a plane far from its mesh origin, and both again with the BVH34 midphase.
    std::vector<PxVec3> vertices;
    std::vector<PxU32> indices;
    for (int y = 0; y <= 32; ++y)
        for (int x = 0; x <= 32; ++x)
            vertices.emplace_back((x - 16) * 10.f, (y - 16) * 10.f,
                2.f * std::sin(float(x)) * std::cos(float(y)));
    for (int y = 0; y < 32; ++y)
        for (int x = 0; x < 32; ++x)
        {
            const PxU32 a = PxU32(y * 33 + x);
            indices.insert(indices.end(), {a, a + 1, a + 34, a, a + 34, a + 33});
        }
    const std::vector<PxVec3> square = {{-100,-100,0}, {100,-100,0}, {100,100,0}, {-100,100,0}};
    const PxVec3 localOffset(100000,-80000,120000);
    std::vector<PxVec3> shiftedSquare;
    for (const PxVec3& corner : square) shiftedSquare.push_back(localOffset + corner);
    PxTriangleMesh* terrain = makeMesh(*physics, *cooking, vertices, indices);
    PxTriangleMesh* plane = makeMesh(*physics, *cooking, square, {0,1,2,0,2,3});
    PxTriangleMesh* shifted = makeMesh(*physics, *cooking, shiftedSquare, {0,1,2,0,2,3});
    PxCookingParams alternate = cooking->getParams();
    alternate.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH34);
    cooking->setParams(alternate);
    PxTriangleMesh* terrain34 = makeMesh(*physics, *cooking, vertices, indices);
    PxTriangleMesh* plane34 = makeMesh(*physics, *cooking, square, {0,1,2,0,2,3});

    std::mt19937 random(4242);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    std::vector<PxVec3> hull;
    for (int index = 0; index < 24; ++index)
        hull.push_back(PxVec3(value(random) * 30, value(random) * 20, value(random) * 25));
    PxConvexMesh* convex = makeConvex(*physics, *cooking, hull);

    Totals totals;
    const auto start = std::chrono::steady_clock::now();
    for (PxTriangleMesh* mesh : {terrain, plane, shifted, terrain34, plane34})
    for (const PxVec3 scale : {PxVec3(1), PxVec3(.6f,1.7f,.8f), PxVec3(-1,1.2f,.7f)})
    for (const PxVec3 offset : {PxVec3(0), PxVec3(100000,-80000,120000)})
    for (int rotation = 0; rotation < 3; ++rotation)
    for (int variant = 0; variant < 4; ++variant)
    {
        // Rotation 2 turns the mesh upside down, so triangles face away from the dominant axis.
        const PxQuat meshRotation = rotation == 2 ? PxQuat(2.9f, PxVec3(1,0,0)) :
            rotation ? PxQuat(.35f, PxVec3(1,0,0)) : PxQuat(PxIdentity);
        const PxQuat sensorRotation = rotation == 1 ? PxQuat(.6457718f, PxVec3(0,0,1)) *
            PxQuat(.1745329f, PxVec3(0,1,0)) : PxQuat(PxIdentity);
        const PxTriangleMeshGeometry geometry(mesh, PxMeshScale(scale, PxQuat(.2f, PxVec3(0,0,1))));
        const PxTransform meshPose(offset, meshRotation);
        const PxVec3 meshCenter = meshPose.transform(mesh == shifted ? geometry.scale.transform(localOffset) : PxVec3(0));
        // Variant 0 puts voxel faces on the mesh plane; the others jitter the tile, scale the sensor or use 50 cm voxels.
        Tile tile{{16, 16, 24}, 10.f, PxVec3(1), PxTransform(meshCenter, sensorRotation), PxVec3(-80, -80, -120)};
        if (variant > 0) tile.sensor.p += PxVec3(value(random), value(random), value(random)) * 7.f;
        if (variant == 2) tile.scale = PxVec3(1.f, -1.f, 1.25f);
        if (variant == 3) { tile.voxel = 50.f; tile.size[0] = tile.size[1] = 8; tile.size[2] = 8; tile.origin = PxVec3(-200); }
        // Padding only grows footprints, so it must not change the result.
        for (double padding : {0.0, 1.0})
            if (!checkMeshTile(tile, geometry, meshPose, padding, totals))
            {
                fprintf(stderr, "  offset=%g scale=%g,%g,%g rotation=%d variant=%d padding=%g\n",
                    offset.x, scale.x, scale.y, scale.z, rotation, variant, padding);
                return 2;
            }
    }

    const PxConvexMeshGeometry convexGeometry(convex, PxMeshScale(PxVec3(1.3f, .7f, 1.f), PxQuat(.4f, PxVec3(0,1,0))));
    const PxBoxGeometry boxGeometry(PxVec3(37, 12, 21));
    const PxSphereGeometry sphereGeometry(33);
    const PxCapsuleGeometry capsuleGeometry(14, 46);
    const PxGeometry* primitives[] = {&boxGeometry, &sphereGeometry, &capsuleGeometry, &convexGeometry};
    for (const PxGeometry* geometry : primitives)
    for (const PxVec3 offset : {PxVec3(0), PxVec3(100000,-80000,120000)})
    for (int variant = 0; variant < 3; ++variant)
    {
        const PxTransform pose(offset + PxVec3(value(random), value(random), value(random)) * 20.f,
            PxQuat(value(random) * 3.f, PxVec3(value(random), value(random), value(random) + 2.f).getNormalized()));
        Tile tile{{16, 16, 24}, 10.f, PxVec3(1), PxTransform(offset, PxQuat(.3f * variant, PxVec3(0,0,1))),
            PxVec3(-80, -80, -120)};
        if (variant == 2) tile.scale = PxVec3(1.f, -1.f, 1.25f);
        if (!checkPrimitiveTile(tile, *geometry, pose, totals))
        {
            fprintf(stderr, "  geometry=%d offset=%g variant=%d\n", int(geometry->getType()), offset.x, variant);
            return 3;
        }
    }

    // Landscapes: heightfields turned so their height axis is world z, with smooth, rough and spiky terrain and holes,
    // scaled like Unreal landscapes (50 or 100 cm quads, fine height steps).
    unsigned heightFieldTiles = 0, heightFieldColumns = 0;
    VoxelSurfaceRaster::FRasterStats surfaceStats[3];
    const PxQuat heightUp(PxHalfPi, PxVec3(1, 0, 0));
    for (int kind = 0; kind < 4; ++kind)
    {
        const int roughness = kind == 0 ? 200 : 1500;
        PxHeightField* heightField = makeHeightField(*physics, *cooking, 13, 13, roughness,
            kind >= 2 ? 12 : 0, kind == 3 ? 20 : 0, random);
        for (const float quad : {50.f, 100.f})
        for (int place = 0; place < 3; ++place)
        for (int variant = 0; variant < 4; ++variant)
        {
            // At the origin, at Town-like coordinates and a kilometre out.
            const PxVec3 offset = place == 0 ? PxVec3(0) : place == 1 ? PxVec3(15000, -9000, 300) : PxVec3(100000, -80000, 120000);
            const PxHeightFieldGeometry geometry(heightField, PxMeshGeometryFlags(), 1.f / 64.f, quad, quad);
            const PxTransform pose(offset, heightUp);
            const PxVec3 middle = pose.transform(PxVec3(6 * quad, 0, 6 * quad));
            const PxQuat sensorRotation = variant == 1 ? PxQuat(.6457718f, PxVec3(0,0,1)) * PxQuat(.07f, PxVec3(0,1,0)) :
                variant == 3 ? PxQuat(.3f, PxVec3(1,0,0)) : PxQuat(PxIdentity);
            Tile tile{{16, 16, 24}, 10.f, PxVec3(1), PxTransform(middle + PxVec3(value(random), value(random), 0) * 200.f,
                sensorRotation), PxVec3(-80, -80, -120)};
            tile.sensor.p.z += value(random) * 20.f;
            if (variant == 2) { tile.voxel = 50.f; tile.size[0] = tile.size[1] = 8; tile.size[2] = 8; tile.origin = PxVec3(-200); }
            if (!checkHeightFieldTile(tile, geometry, pose, totals, surfaceStats[place]))
            {
                fprintf(stderr, "  terrain kind=%d quad=%g offset=%g variant=%d\n", kind, quad, offset.x, variant);
                return 4;
            }
            ++heightFieldTiles;
            heightFieldColumns += unsigned(tile.size[0] * tile.size[1]);
        }
        heightField->release();
    }

    printf("PASS: raster matched PhysX on %u tiles (%u voxels, %u hits) in %.0f ms\n", totals.tiles, totals.voxels,
        totals.hits, std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count());
    printf("  visited=%llu tests=%llu uncertain=%llu deferred=%llu certain=%llu (%u checked) skipped triangles=%llu\n",
        (unsigned long long)totals.stats.Visited,
        (unsigned long long)(totals.stats.Overlaps + totals.stats.Uncertain + totals.stats.Outside),
        (unsigned long long)totals.stats.Uncertain, (unsigned long long)totals.stats.DeferredQueries,
        (unsigned long long)totals.stats.Certain, totals.certain, (unsigned long long)totals.stats.Skipped);
    printf("  primitive queries=%llu culled bricks=%llu\n", (unsigned long long)totals.stats.PrimitiveQueries,
        (unsigned long long)totals.stats.CulledBricks);
    printf("  heightfield tiles=%u (%u columns)\n", heightFieldTiles, heightFieldColumns);
    const char* places[3] = {"origin", "150 m out", "1.7 km out"};
    for (int place = 0; place < 3; ++place)
    {
        const VoxelSurfaceRaster::FRasterStats& stats = surfaceStats[place];
        printf("    %-10s per column: tests=%.2f overlaps=%.2f unconfirmed=%.3f uncertain=%.3f PhysX queries=%.3f\n", places[place],
            3.0 * stats.Visited / heightFieldColumns, 3.0 * stats.Overlaps / heightFieldColumns,
            3.0 * stats.Unconfirmed / heightFieldColumns, 3.0 * stats.Uncertain / heightFieldColumns,
            3.0 * stats.DeferredQueries / heightFieldColumns);
    }

    convex->release();
    terrain->release(); plane->release(); shifted->release(); terrain34->release(); plane34->release();
    cooking->release(); physics->release(); foundation->release();
}
