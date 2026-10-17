// Standalone regression check against the actual PhysX geometry-query oracle.
// From the repository root, with UE4_ROOT pointing to UnrealEngine_4.26:
//   px="$UE4_ROOT/Engine/Source/ThirdParty/PhysX3"
//   libs="$px/Lib/Linux/x86_64-unknown-linux-gnu"
//   g++ -std=c++14 -O2 -DNDEBUG -DPX_PHYSX_STATIC_LIB \
//     -I"$px/PhysX_3.4/Include" -I"$px/PxShared/include" \
//     Util/BuildTools/tests/test_voxel_triangle_overlap.cpp -Wl,--start-group \
//     "$libs/libPhysX3.a" "$libs/libPhysX3Common.a" "$libs/libPxFoundation.a" \
//     "$libs/libPxPvdSDK.a" "$libs/libPhysX3Cooking.a" -Wl,--end-group \
//     -pthread -ldl -o /tmp/test_voxel_triangle_overlap
//   /tmp/test_voxel_triangle_overlap

#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelTriangleOverlap.h"
#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelSurfaceRaster.h"
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
    void reportError(PxErrorCode::Enum, const char* message, const char*, int) override
    {
        fprintf(stderr, "PhysX: %s\n", message);
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

struct Cache
{
    std::vector<VoxelTriangleOverlap::FTriangle> triangles;
    bool fallback = false;
};

static Cache makeCache(const PxTriangleMeshGeometry& geometry, const PxTransform& meshPose,
    const PxTransform& cachePose, const PxVec3& halfExtent)
{
    const PxBounds3 bounds = PxGeometryQuery::getWorldBounds(
        PxBoxGeometry(halfExtent.multiply(PxVec3(4, 4, 8))), cachePose);
    PxU32 indices[64];
    bool overflow = false;
    const PxU32 count = PxMeshQuery::findOverlapTriangleMesh(
        PxBoxGeometry(bounds.getExtents() + PxVec3(.1f)), PxTransform(bounds.getCenter()),
        geometry, meshPose, indices, 64, 0, overflow);
    Cache cache;
    cache.fallback = overflow || count == 0;
    if (!cache.fallback)
        for (PxU32 index = 0; index < count; ++index)
            cache.triangles.push_back(VoxelTriangleOverlap::MakeTriangle(
                geometry, meshPose, cachePose, indices[index], halfExtent));
    return cache;
}

static bool cachedOverlap(const Cache& cache, const PxTransform& cachePose,
    const PxBoxGeometry& box, const PxTransform& boxPose, const PxTriangleMeshGeometry& mesh,
    const PxTransform& meshPose, unsigned& fallbacks)
{
    bool uncertain = cache.fallback;
    const PxVec3 center = cachePose.transformInv(boxPose.p);
    for (const auto& triangle : cache.triangles)
    {
        const auto result = VoxelTriangleOverlap::Test(triangle, center);
        if (result == VoxelTriangleOverlap::EResult::Overlap) return true;
        uncertain |= result == VoxelTriangleOverlap::EResult::Uncertain;
    }
    if (!uncertain) return false;
    ++fallbacks;
    return PxGeometryQuery::overlap(box, boxPose, mesh, meshPose);
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

struct RasterTotals
{
    unsigned tiles = 0, voxels = 0, hits = 0;
    VoxelSurfaceRaster::FRasterStats stats;
};

static bool compareTile(const TileEnv& env, const char* what, RasterTotals& totals)
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

// Rasterize every triangle the sensor would fetch for this tile and compare with PhysX per voxel.
// Also checks that each footprint holds every voxel where VoxelTriangleOverlap::Test is not Outside.
static bool rasterMesh(const Tile& tile, const PxTriangleMeshGeometry& geometry, const PxTransform& meshPose,
    double padding, RasterTotals& totals)
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
        if (prepared.Degenerate)
        {
            VoxelSurfaceRaster::RasterizeDegenerate<true>(triangle, error, halfExtent, env.lattice, env, totals.stats);
            continue;
        }
        std::fill(visited.begin(), visited.end(), false);
        VoxelSurfaceRaster::ForEachTriangleVoxel(prepared, env.lattice, [&](int x, int y, int z)
        {
            visited[env.index(x, y, z)] = true;
        });
        for (size_t voxel = 0; voxel < env.world.size(); ++voxel)
            if (!visited[voxel] && VoxelTriangleOverlap::Test(prepared, env.tileFrame.transformInv(env.world[voxel])) !=
                VoxelTriangleOverlap::EResult::Outside)
            {
                fprintf(stderr, "Incomplete footprint: triangle=%u voxel=%zu\n", indices[i], voxel);
                return false;
            }
        VoxelSurfaceRaster::RasterizeTriangle<true>(prepared, env.lattice, env, totals.stats);
    }
    VoxelSurfaceRaster::ResolveDeferred<true>(env.deferred, env.lattice, env, totals.stats);
    return compareTile(env, "mesh", totals);
}

static bool rasterPrimitive(const Tile& tile, const PxGeometry& geometry, const PxTransform& pose,
    double padding, RasterTotals& totals)
{
    TileEnv env;
    buildTile(tile, padding, env);
    env.geometry = &geometry;
    env.pose = pose;
    env.bounds = PxGeometryQuery::getWorldBounds(geometry, pose);
    const int32_t brick[3] = {4, 4, 8};
    VoxelSurfaceRaster::RasterizeBounds<true>(VoxelSurfaceRaster::GetBoundsBox(env.bounds, env.worldExtent,
        env.tileFrame, env.lattice), brick, env.lattice, env, totals.stats);
    return compareTile(env, "primitive", totals);
}

// The tile label rules against the definition and against the chunked per-voxel loop, on random candidate sets.
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
        // Definition: the unique highest-priority label, or a tie between two of them.
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
        const int raster = state & VoxelSurfaceRaster::TieFlag ? -1 : state;
        int label = 0, chunked = 0;
        for (const Candidate& candidate : candidates)
        {
            if (priorities[candidate.label] < priorities[label]) break;
            if (candidate.label == label || !candidate.overlap) continue;
            if (label) { chunked = -1; break; }
            label = candidate.label;
        }
        if (chunked == 0) chunked = label;
        if (raster != expected || chunked != expected)
        {
            fprintf(stderr, "Label rules: trial=%d expected=%d raster=%d chunked=%d\n", trial, expected, raster, chunked);
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

// Degenerate triangles defer every voxel they may touch; non-finite ones defer the whole tile.
static bool checkDegenerate()
{
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

static bool checkRaster(PxPhysics& physics, const std::vector<PxTriangleMesh*>& meshes, const PxVec3& localOffset,
    PxTriangleMesh* shifted, PxConvexMesh* convex)
{
    if (!checkLabelRules() || !checkClip() || !checkDegenerate()) return false;
    RasterTotals totals;
    std::mt19937 random(4242);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    const auto start = std::chrono::steady_clock::now();
    for (PxTriangleMesh* mesh : meshes)
    for (const PxVec3 scale : {PxVec3(1), PxVec3(.6f,1.7f,.8f), PxVec3(-1,1.2f,.7f)})
    for (const PxVec3 offset : {PxVec3(0), PxVec3(100000,-80000,120000)})
    for (int rotation = 0; rotation < 2; ++rotation)
    for (int variant = 0; variant < 4; ++variant)
    {
        const PxQuat meshRotation = rotation ? PxQuat(.35f, PxVec3(1,0,0)) : PxQuat(PxIdentity);
        const PxQuat sensorRotation = rotation ? PxQuat(.6457718f, PxVec3(0,0,1)) *
            PxQuat(.1745329f, PxVec3(0,1,0)) : PxQuat(PxIdentity);
        const PxTriangleMeshGeometry geometry(mesh, PxMeshScale(scale, PxQuat(.2f, PxVec3(0,0,1))));
        const PxTransform meshPose(offset, meshRotation);
        const PxVec3 meshCenter = meshPose.transform(mesh == shifted ? geometry.scale.transform(localOffset) : PxVec3(0));
        // Variant 0 puts voxel faces on the mesh plane; the others jitter the tile, scale the sensor or use 50 cm voxels.
        Tile tile{{16, 16, 24}, 10.f, PxVec3(1), PxTransform(meshCenter, sensorRotation), PxVec3(-80, -80, -120)};
        if (variant > 0) tile.sensor.p += PxVec3(value(random), value(random), value(random)) * 7.f;
        if (variant == 2) tile.scale = PxVec3(1.f, -1.f, 1.25f);
        if (variant == 3) { tile.voxel = 50.f; tile.size[0] = tile.size[1] = 8; tile.size[2] = 8; tile.origin = PxVec3(-200); }
        for (double padding : {0.0, 1.0})
            if (!rasterMesh(tile, geometry, meshPose, padding, totals))
            {
                fprintf(stderr, "  offset=%g scale=%g,%g,%g rotation=%d variant=%d padding=%g\n",
                    offset.x, scale.x, scale.y, scale.z, rotation, variant, padding);
                return false;
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
        if (!rasterPrimitive(tile, *geometry, pose, 0.0, totals))
        {
            fprintf(stderr, "  geometry=%d offset=%g variant=%d\n", int(geometry->getType()), offset.x, variant);
            return false;
        }
    }
    printf("PASS: raster matched PhysX on %u tiles (%u voxels, %u hits); visited=%llu tests=%llu uncertain=%llu "
        "deferred=%llu primitive queries=%llu culled bricks=%llu degenerate=%llu; %.0f ms\n",
        totals.tiles, totals.voxels, totals.hits, (unsigned long long)totals.stats.Visited,
        (unsigned long long)(totals.stats.Overlaps + totals.stats.Uncertain + totals.stats.Outside),
        (unsigned long long)totals.stats.Uncertain, (unsigned long long)totals.stats.DeferredQueries,
        (unsigned long long)totals.stats.PrimitiveQueries, (unsigned long long)totals.stats.CulledBricks,
        (unsigned long long)totals.stats.Degenerate,
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count());
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
    PxTriangleMesh* terrain = makeMesh(*physics, *cooking, vertices, indices);
    PxTriangleMesh* plane = makeMesh(*physics, *cooking,
        {{-100,-100,0}, {100,-100,0}, {100,100,0}, {-100,100,0}}, {0,1,2,0,2,3});
    const PxVec3 localOffset(100000,-80000,120000);
    PxTriangleMesh* shifted = makeMesh(*physics, *cooking,
        {localOffset + PxVec3(-100,-100,0), localOffset + PxVec3(100,-100,0),
         localOffset + PxVec3(100,100,0), localOffset + PxVec3(-100,100,0)}, {0,1,2,0,2,3});
    PxCookingParams alternate = cooking->getParams();
    alternate.midphaseDesc.setToDefault(PxMeshMidPhase::eBVH34);
    cooking->setParams(alternate);
    PxTriangleMesh* terrain34 = makeMesh(*physics, *cooking, vertices, indices);
    PxTriangleMesh* plane34 = makeMesh(*physics, *cooking,
        {{-100,-100,0}, {100,-100,0}, {100,100,0}, {-100,100,0}}, {0,1,2,0,2,3});
    std::mt19937 random(9071);
    std::uniform_real_distribution<float> value(-1.f, 1.f);
    unsigned queries = 0, fallbacks = 0;
    double preparationMs = 0, referenceMs = 0, cachedMs = 0;
    double gridReferenceMs = 0, gridCachedMs = 0;
    for (PxTriangleMesh* mesh : {terrain, plane, shifted, terrain34, plane34})
    for (const PxVec3 scale : {PxVec3(1), PxVec3(.6f,1.7f,.8f), PxVec3(-1,1.2f,.7f)})
    for (const PxVec3 offset : {PxVec3(0), PxVec3(100000,-80000,120000)})
    for (int rotation = 0; rotation < 2; ++rotation)
    {
        const PxQuat meshRotation = rotation ? PxQuat(.35f, PxVec3(1,0,0)) : PxQuat(PxIdentity);
        const PxQuat boxRotation = rotation ? PxQuat(.6457718f, PxVec3(0,0,1)) *
            PxQuat(.1745329f, PxVec3(0,1,0)) : PxQuat(PxIdentity);
        const PxTriangleMeshGeometry geometry(mesh, PxMeshScale(scale, PxQuat(.2f, PxVec3(0,0,1))));
        const PxTransform meshPose(offset, meshRotation);
        const PxVec3 meshCenter = mesh == shifted ? geometry.scale.transform(localOffset) : PxVec3(0);
        const PxTransform cachePose(meshPose.transform(meshCenter), boxRotation);
        const PxVec3 halfExtent(5);
        const PxBoxGeometry box(halfExtent);
        auto start = std::chrono::steady_clock::now();
        const Cache cache = makeCache(geometry, meshPose, cachePose, halfExtent);
        preparationMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
        std::vector<PxTransform> poses;
        for (int x = 0; x < 4; ++x)
            for (int y = 0; y < 4; ++y)
                for (int z = 0; z < 8; ++z)
                    poses.emplace_back(cachePose.transform(PxVec3((x-1.5f)*10, (y-1.5f)*10, (z-3.5f)*10)), boxRotation);
        for (int index = 0; index < 2000; ++index)
            poses.emplace_back(cachePose.transform(PxVec3(value(random)*15, value(random)*15,
                value(random)*35)), boxRotation);
        std::vector<bool> expected(poses.size());
        std::vector<bool> actual(poses.size());
        start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < poses.size(); ++index)
        {
            if (index == 128)
                gridReferenceMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
            expected[index] = PxGeometryQuery::overlap(box, poses[index], geometry, meshPose);
        }
        referenceMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
        start = std::chrono::steady_clock::now();
        for (size_t index = 0; index < poses.size(); ++index)
        {
            if (index == 128)
                gridCachedMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
            actual[index] = cachedOverlap(cache, cachePose, box, poses[index], geometry, meshPose, fallbacks);
        }
        cachedMs += std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
        for (size_t index = 0; index < poses.size(); ++index)
        {
            if (actual[index] != expected[index])
            {
                fprintf(stderr, "Mismatch query=%u index=%zu offset=%g,%g,%g scale=%g,%g,%g\n",
                    queries, index, offset.x, offset.y, offset.z, scale.x, scale.y, scale.z);
                return 1;
            }
            ++queries;
        }
    }
    // Degenerate inputs must defer to the geometry oracle, rather than claim a hit.
    for (const PxTriangle triangle : {PxTriangle(PxVec3(0),PxVec3(0),PxVec3(0)),
        PxTriangle(PxVec3(-1,0,0),PxVec3(0),PxVec3(1,0,0))})
        if (VoxelTriangleOverlap::Test(VoxelTriangleOverlap::MakeTriangle(triangle, PxVec3(5), .001f),
            PxVec3(0)) != VoxelTriangleOverlap::EResult::Uncertain) return 2;
    std::vector<PxVec3> hull;
    for (int index = 0; index < 24; ++index)
        hull.push_back(PxVec3(value(random) * 30, value(random) * 20, value(random) * 25));
    PxConvexMeshDesc hullDesc;
    hullDesc.points.count = PxU32(hull.size());
    hullDesc.points.stride = sizeof(PxVec3);
    hullDesc.points.data = hull.data();
    hullDesc.flags = PxConvexFlag::eCOMPUTE_CONVEX;
    Buffer hullBuffer;
    if (!cooking->cookConvexMesh(hullDesc, hullBuffer)) throw std::runtime_error("Convex cooking failed");
    PxConvexMesh* convex = physics->createConvexMesh(hullBuffer);
    if (!convex) throw std::runtime_error("Convex creation failed");
    if (!checkRaster(*physics, {terrain, plane, shifted, terrain34, plane34}, localOffset, shifted, convex)) return 3;
    convex->release();
    printf("PASS: %u oracle comparisons; %u whole-mesh fallbacks\n", queries, fallbacks);
    printf("Synthetic timings only: prepare=%.3f ms, direct=%.3f ms, cached=%.3f ms\n",
        preparationMs, referenceMs, cachedMs);
    printf("Actual 128-voxel chunks, including preparation: direct=%.3f ms, cached=%.3f ms\n",
        gridReferenceMs, gridCachedMs + preparationMs);
    terrain->release(); plane->release(); shifted->release(); terrain34->release(); plane34->release();
    cooking->release(); physics->release(); foundation->release();
}
