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
    printf("PASS: %u oracle comparisons; %u whole-mesh fallbacks\n", queries, fallbacks);
    printf("Synthetic timings only: prepare=%.3f ms, direct=%.3f ms, cached=%.3f ms\n",
        preparationMs, referenceMs, cachedMs);
    printf("Actual 128-voxel chunks, including preparation: direct=%.3f ms, cached=%.3f ms\n",
        gridReferenceMs, gridCachedMs + preparationMs);
    terrain->release(); plane->release(); shifted->release(); terrain34->release(); plane34->release();
    cooking->release(); physics->release(); foundation->release();
}
