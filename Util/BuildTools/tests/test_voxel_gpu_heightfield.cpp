// Standalone check of libcarla_voxel_gpu's heightfields against PhysX's own
// box-heightfield overlap, with the triangles placed the way the voxel
// detection sensor places them: taken once in sample space, then scaled and
// posed per shape. From the repository root, with UE4_ROOT pointing to
// UnrealEngine_4.26, after Util/BuildTools/BuildVoxelGpu.sh:
//   px="$UE4_ROOT/Engine/Source/ThirdParty/PhysX3"
//   libs="$px/Lib/Linux/x86_64-unknown-linux-gnu"
//   lib=Unreal/CarlaUE4/Plugins/Carla/CarlaDependencies/lib
//   g++-11 -std=c++17 -O2 -DNDEBUG -DPX_PHYSX_STATIC_LIB \
//     -I"$px/PhysX_3.4/Include" -I"$px/PxShared/include" \
//     Util/BuildTools/tests/test_voxel_gpu_heightfield.cpp -Wl,--start-group \
//     "$libs/libPhysX3.a" "$libs/libPhysX3Common.a" "$libs/libPxFoundation.a" \
//     "$libs/libPxPvdSDK.a" "$libs/libPhysX3Cooking.a" -Wl,--end-group \
//     -L$lib -lcarla_voxel_gpu -Wl,-rpath,$PWD/$lib -pthread -ldl -o /tmp/test_voxel_gpu_heightfield
//   /tmp/test_voxel_gpu_heightfield

#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelGpuApi.h"
#include "PxPhysicsAPI.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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
        if (code != PxErrorCode::eLOGGING_INFO)
            fprintf(stderr, "PhysX: %s\n", message);
    }
};

static PxPhysics* physics = nullptr;
static PxCooking* cooking = nullptr;
static FVoxelGpuContext* context = nullptr;

// Heights in -roughness..roughness, with the given share of hole
// triangles, and every cell a hole from row firstHoleRow on, if set.
static PxHeightField* makeHeightField(int rows, int columns, int roughness, double holes, float thickness,
                                      std::mt19937& random, int firstHoleRow = -1)
{
    std::vector<PxHeightFieldSample> samples(size_t(rows * columns));
    std::uniform_int_distribution<int> height(-roughness, roughness);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (PxHeightFieldSample& sample : samples)
    {
        sample.height = PxI16(height(random));
        sample.materialIndex0 = unit(random) < holes ? PxU8(PxHeightFieldMaterial::eHOLE) : PxU8(0);
        sample.materialIndex1 = unit(random) < holes ? PxU8(PxHeightFieldMaterial::eHOLE) : PxU8(0);
        if (unit(random) < 0.5)
            sample.setTessFlag();
    }
    if (firstHoleRow >= 0)
        for (int index = firstHoleRow * columns; index < rows * columns; ++index)
        {
            samples[size_t(index)].materialIndex0 = PxU8(PxHeightFieldMaterial::eHOLE);
            samples[size_t(index)].materialIndex1 = PxU8(PxHeightFieldMaterial::eHOLE);
        }
    PxHeightFieldDesc desc;
    desc.format = PxHeightFieldFormat::eS16_TM;
    desc.nbRows = PxU32(rows);
    desc.nbColumns = PxU32(columns);
    desc.samples.data = samples.data();
    desc.samples.stride = sizeof(PxHeightFieldSample);
    desc.flags = PxHeightFieldFlag::eNO_BOUNDARY_EDGES;
    desc.thickness = thickness;
    PxHeightField* result = cooking->createHeightField(desc, physics->getPhysicsInsertionCallback());
    if (!result)
        throw std::runtime_error("heightfield creation failed");
    return result;
}

// The sensor's extraction: every cell's non-hole triangles in sample space.
static void uploadSamples(const PxHeightFieldGeometry& geometry, uint64_t key)
{
    PxHeightFieldGeometry samples = geometry;
    samples.rowScale = samples.heightScale = samples.columnScale = 1.0f;
    const PxHeightField& field = *geometry.heightField;
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    for (PxU32 row = 0; row + 1 < field.getNbRows(); ++row)
        for (PxU32 column = 0; column + 1 < field.getNbColumns(); ++column)
            for (PxU32 half = 0; half < 2; ++half)
            {
                const PxU32 index = 2 * (row * field.getNbColumns() + column) + half;
                if (field.getTriangleMaterialIndex(index) == PxHeightFieldMaterial::eHOLE)
                    continue;
                PxTriangle triangle;
                PxMeshQuery::getTriangle(samples, PxTransform(PxIdentity), index, triangle);
                for (int vertex = 0; vertex < 3; ++vertex)
                {
                    indices.push_back(uint32_t(vertices.size() / 3));
                    for (int axis = 0; axis < 3; ++axis)
                        vertices.push_back(triangle.verts[vertex][axis]);
                }
            }
    const FVoxelGpuMesh mesh{key, vertices.data(), uint32_t(vertices.size() / 3), indices.data(),
                             uint32_t(indices.size() / 3)};
    if (VoxelGpuUploadMesh(context, &mesh) != 0)
        throw std::runtime_error(VoxelGpuError(context));
}

// Grid space is (world - origin) / boxSize, rotated by yaw about y.
struct Grid
{
    int size[3];
    double boxSize;
    PxVec3 origin;
    double yaw;

    PxQuat rotation() const { return PxQuat(PxReal(yaw), PxVec3(0, 1, 0)); }

    // Row-major 3 x 4 from world to grid.
    void worldToGrid(double* m) const
    {
        const PxMat33 r(rotation());
        for (int row = 0; row < 3; ++row)
        {
            double t = 0;
            for (int column = 0; column < 3; ++column)
            {
                // The transpose of the rotation.
                const double value = r[row][column];
                m[4 * row + column] = value / boxSize;
                t -= value * origin[column];
            }
            m[4 * row + 3] = t / boxSize;
        }
    }

    PxTransform voxelPose(int x, int y, int z) const
    {
        const PxVec3 local(PxReal((x + 0.5) * boxSize), PxReal((y + 0.5) * boxSize), PxReal((z + 0.5) * boxSize));
        return PxTransform(origin + rotation().rotate(local), rotation());
    }
};

struct Shape
{
    PxHeightFieldGeometry geometry;
    PxTransform pose;
    uint32_t label;
};

static void multiply(const double* a, const double* b, double* out)
{
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 4; ++column)
        {
            double sum = column == 3 ? a[4 * row + 3] : 0.0;
            for (int k = 0; k < 3; ++k)
                sum += a[4 * row + k] * b[4 * k + column];
            out[4 * row + column] = sum;
        }
}

// The sensor's instance for a shape: grid from world, world from shape,
// shape from samples.
static FVoxelGpuInstance makeInstance(const Grid& grid, const Shape& shape, uint64_t key)
{
    double worldToGrid[12];
    grid.worldToGrid(worldToGrid);
    double shapeToWorld[12] = {};
    const PxMat33 r(shape.pose.q);
    for (int row = 0; row < 3; ++row)
    {
        for (int column = 0; column < 3; ++column)
            shapeToWorld[4 * row + column] = r[column][row];
        shapeToWorld[4 * row + 3] = shape.pose.p[row];
    }
    const double scales[12] = {shape.geometry.rowScale, 0, 0, 0, 0, shape.geometry.heightScale, 0, 0,
                               0, 0, shape.geometry.columnScale, 0};
    double shapeToGrid[12];
    multiply(worldToGrid, shapeToWorld, shapeToGrid);
    FVoxelGpuInstance instance{};
    multiply(shapeToGrid, scales, instance.ToGrid);
    instance.MeshKey = key;
    instance.Key = VOXEL_GPU_KEY(1, shape.label, 0);
    instance.Flags = VOXEL_GPU_HEIGHTFIELD;
    const PxHeightField& field = *shape.geometry.heightField;
    instance.Thickness = double(field.getThickness()) / shape.geometry.heightScale;
    instance.Rows = field.getNbRows();
    instance.Columns = field.getNbColumns();
    return instance;
}

// Labels the grid on the GPU and with PhysX, where the lower label wins a
// voxel both shapes touch, and returns the number of voxels that differ.
static long compare(const char* name, const Grid& grid, const std::vector<Shape>& shapes,
                    const std::vector<uint64_t>& keys, long* occupied = nullptr)
{
    std::vector<FVoxelGpuInstance> instances;
    for (size_t index = 0; index < shapes.size(); ++index)
        instances.push_back(makeInstance(grid, shapes[index], keys[index]));
    const FVoxelGpuGrid gpuGrid{{grid.size[0], grid.size[1], grid.size[2]}, 0};
    std::vector<uint8_t> labels(size_t(grid.size[0]) * grid.size[1] * grid.size[2], 0xEE);
    FVoxelGpuStats stats{};
    if (VoxelGpuRun(context, &gpuGrid, instances.data(), uint32_t(instances.size()), nullptr, 0, nullptr, 0,
                    labels.data(), &stats) != 0)
        throw std::runtime_error(VoxelGpuError(context));
    const PxBoxGeometry box(PxReal(grid.boxSize / 2), PxReal(grid.boxSize / 2), PxReal(grid.boxSize / 2));
    long mismatches = 0, hits = 0;
    for (int x = 0; x < grid.size[0]; ++x)
        for (int y = 0; y < grid.size[1]; ++y)
            for (int z = 0; z < grid.size[2]; ++z)
            {
                uint8_t expected = 0;
                for (const Shape& shape : shapes)
                    if ((expected == 0 || shape.label < expected) &&
                        PxGeometryQuery::overlap(box, grid.voxelPose(x, y, z), shape.geometry, shape.pose))
                        expected = uint8_t(shape.label);
                const uint8_t actual = labels[(size_t(x) * grid.size[1] + y) * grid.size[2] + z];
                hits += expected != 0;
                if (actual != expected)
                {
                    if (mismatches < 5)
                        printf("  %s voxel (%d, %d, %d): gpu %d, PhysX %d\n", name, x, y, z, actual, expected);
                    ++mismatches;
                }
            }
    printf("%s: %ld of %d voxels differ, %ld occupied in PhysX\n", name, mismatches,
           grid.size[0] * grid.size[1] * grid.size[2], hits);
    if (occupied)
        *occupied = hits;
    return mismatches;
}

int main()
{
    Allocator allocator;
    Errors errors;
    PxFoundation* foundation = PxCreateFoundation(PX_FOUNDATION_VERSION, allocator, errors);
    physics = PxCreatePhysics(PX_PHYSICS_VERSION, *foundation, PxTolerancesScale(), false);
    cooking = PxCreateCooking(PX_PHYSICS_VERSION, *foundation, PxCookingParams(physics->getTolerancesScale()));
    if (!foundation || !physics || !cooking)
        throw std::runtime_error("PhysX initialization failed");
    char error[256] = {};
    if (VoxelGpuCreate(0, &context, error, sizeof(error)) != 0)
    {
        printf("create failed: %s\n", error);
        return 1;
    }

    std::mt19937 random(7);
    bool pass = true;

    // Flat terrain through the middle of voxels aligned with its cells:
    // every box edge meets the surface on a triangle edge or vertex.
    PxHeightField* flat = makeHeightField(7, 7, 0, 0.0, -1.0f, random);
    {
        const Shape shape{PxHeightFieldGeometry(flat, PxMeshGeometryFlags(), 1, 1, 1), PxTransform(PxVec3(0, 0.5f, 0)),
                          10};
        uploadSamples(shape.geometry, 3);
        const Grid grid{{4, 1, 4}, 1.0, PxVec3(1, 0, 1), 0.0};
        long occupied = 0;
        pass &= compare("aligned flat, cell-sized voxels", grid, {shape}, {3}, &occupied) == 0 && occupied == 16;
        const Grid half{{8, 2, 8}, 0.5, PxVec3(1, 0, 1), 0.0};
        pass &= compare("aligned flat, half-cell voxels", half, {shape}, {3}) == 0;
    }

    // Rough terrain on an aligned grid. Power-of-two scales keep PhysX's
    // single-precision arithmetic exact, so ties stay ties: box corners on
    // the surface and box edges through triangle edges.
    PxHeightField* rough = makeHeightField(24, 20, 6, 0.0, -2.0f, random);
    {
        const Shape shape{PxHeightFieldGeometry(rough, PxMeshGeometryFlags(), 0.25f, 2.0f, 0.5f),
                          PxTransform(PxVec3(-3, 1, 2)), 10};
        uploadSamples(shape.geometry, 5);
        const Grid grid{{40, 12, 17}, 0.5, PxVec3(-2, -2, 2.5f), 0.0};
        pass &= compare("aligned rough", grid, {shape}, {5}) == 0;

        // Past the last column, which PhysX's corner test leaves out.
        const Grid past{{40, 12, 36}, 0.5, PxVec3(-2, -2, 2.5f), 0.0};
        pass &= compare("aligned rough, past the last column", past, {shape}, {5}) == 0;

        // The same cached triangles under a second shape with another pose
        // and scales, as when two shapes share a heightfield or the world
        // origin moves.
        const Shape moved{PxHeightFieldGeometry(rough, PxMeshGeometryFlags(), 0.5f, 1.0f, 1.0f),
                          PxTransform(PxVec3(4, -1, 6)), 11};
        pass &= compare("one heightfield, two poses and scales", grid, {shape, moved}, {5, 5}) == 0;
    }

    // Ties at the edges of holes, which PhysX gives to the cell its
    // floor() picks, and a hole there to no one: a flat field whose rows
    // from 3 on are holes, under cell-sized voxels on both sides of the
    // surface, with Unreal's thickness and a deep one.
    for (const float thickness : {-0.01f, -2.0f})
    {
        PxHeightField* field = makeHeightField(6, 6, 0, 0.0, thickness, random, 3);
        const Shape shape{PxHeightFieldGeometry(field, PxMeshGeometryFlags(), 1, 0.25f, 0.25f),
                          PxTransform(PxVec3(0, 0, 0)), 10};
        uploadSamples(shape.geometry, 7);
        const Grid grid{{7, 2, 7}, 0.25, PxVec3(-0.25f, -0.25f, -0.25f), 0.0};
        char name[64];
        snprintf(name, sizeof(name), "hole edge, thickness %g", thickness);
        pass &= compare(name, grid, {shape}, {7}) == 0;
        VoxelGpuReleaseMesh(context, 7);
        field->release();
    }

    // Rough terrain with scattered holes on an aligned grid that reaches
    // past the field. Voxels of half a cell keep heightfield vertices on
    // voxel faces, where PhysX's vertex test, which the GPU follows only
    // in part, never applies.
    {
        PxHeightField* field = makeHeightField(20, 20, 6, 0.1, -1.0f, random);
        const Shape shape{PxHeightFieldGeometry(field, PxMeshGeometryFlags(), 0.25f, 1.0f, 1.0f),
                          PxTransform(PxVec3(0, 0, 0)), 10};
        uploadSamples(shape.geometry, 8);
        const Grid grid{{42, 10, 42}, 0.5, PxVec3(-0.5f, -2, -0.5f), 0.0};
        pass &= compare("aligned rough with holes", grid, {shape}, {8}) == 0;
        VoxelGpuReleaseMesh(context, 8);
        field->release();
    }

    // A positive thickness, which PhysX bounds without its ends.
    {
        PxHeightField* field = makeHeightField(6, 6, 0, 0.0, 2.0f, random);
        const Shape shape{PxHeightFieldGeometry(field, PxMeshGeometryFlags(), 1, 1, 1), PxTransform(PxVec3(0, 0, 0)),
                          10};
        uploadSamples(shape.geometry, 9);
        const Grid grid{{4, 5, 4}, 1.0, PxVec3(0.5f, -1, 0.5f), 0.0};
        pass &= compare("positive thickness", grid, {shape}, {9}) == 0;
        VoxelGpuReleaseMesh(context, 9);
        field->release();
    }

    // Random poses, grids and holes. Single-precision PhysX and the
    // double-precision GPU may disagree on near-ties, and next to holes
    // PhysX tests only the heightfield vertices that are convex or on the
    // field's edge, which the GPU does not follow, so a few differences are
    // allowed.
    long total = 0, voxels = 0;
    for (int trial = 0; trial < 6; ++trial)
    {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        PxHeightField* field = makeHeightField(30, 30, 40, trial % 2 ? 0.05 : 0.0, -0.5f, random);
        const Shape shape{
            PxHeightFieldGeometry(field, PxMeshGeometryFlags(), PxReal(0.01 + unit(random) * 0.02),
                                  PxReal(0.5 + unit(random)), PxReal(0.5 + unit(random))),
            PxTransform(PxVec3(PxReal(unit(random) * 4), PxReal(unit(random)), PxReal(unit(random) * 4)),
                        PxQuat(PxReal(unit(random) * 0.3), PxVec3(0, 1, 0))),
            10};
        uploadSamples(shape.geometry, 100 + trial);
        const Grid grid{{40, 10, 40}, 0.3,
                        PxVec3(PxReal(4 + unit(random)), PxReal(-1 + unit(random)), PxReal(4 + unit(random))),
                        unit(random) * 0.5};
        char name[64];
        snprintf(name, sizeof(name), "random trial %d", trial);
        total += compare(name, grid, {shape}, {uint64_t(100 + trial)});
        voxels += 40 * 10 * 40;
        field->release();
    }
    pass &= total * 10000 <= voxels;

    // A released mesh is gone, so a new mesh may take its key.
    VoxelGpuReleaseMesh(context, 3);
    uint32_t numVertices = 0, numTriangles = 0;
    const bool released = VoxelGpuFindMesh(context, 3, &numVertices, &numTriangles) == 0;
    printf("released mesh %s\n", released ? "dropped" : "still cached");
    pass &= released;

    printf("%s\n", pass ? "PASS" : "FAIL");
    VoxelGpuDestroy(context);
    flat->release();
    rough->release();
    cooking->release();
    physics->release();
    foundation->release();
    return pass ? 0 : 1;
}
