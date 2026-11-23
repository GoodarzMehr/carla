#pragma once

// C interface of libcarla_voxel_gpu, which labels a voxel grid on a CUDA GPU
// without PhysX. The voxel detection sensor loads the library at run time
// when use_gpu is set, so CARLA builds and runs without CUDA.
//
// Everything is in grid space, where voxel (X, Y, Z) is the unit cube
// [X, X + 1] x [Y, Y + 1] x [Z, Z + 1]. A voxel takes the label of the
// largest key among the shapes that touch it, decided in double precision.

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define VOXEL_GPU_API_VERSION 3

// A key orders the shapes that touch a voxel: a higher priority wins, then
// the lower label, then the lower instance.
#define VOXEL_GPU_KEY(Priority, Label, Instance) \
    ((uint32_t)(Priority) << 21 | (uint32_t)(31 - (Label)) << 16 | (uint32_t)(0xFFFF - (Instance)))

// Instance flags.
#define VOXEL_GPU_HEIGHTFIELD 1u

// Shape types.
#define VOXEL_GPU_SPHERE 0u
#define VOXEL_GPU_CAPSULE 1u
#define VOXEL_GPU_HULL 2u

typedef struct FVoxelGpuContext FVoxelGpuContext;

// A triangle mesh in its own space, cached on the GPU under Key. Indices
// may be null to replace only the vertices of a cached mesh, as for a mesh
// that deforms.
typedef struct
{
    uint64_t Key;
    const float* Vertices;
    uint32_t NumVertices;
    const uint32_t* Indices;
    uint32_t NumTriangles;
} FVoxelGpuMesh;

// A cached mesh placed in the grid. ToGrid maps mesh space to grid space as
// a row-major 3 x 4 matrix.
//
// A heightfield (Flags has VOXEL_GPU_HEIGHTFIELD) touches a voxel by
// PhysX's rules instead of the triangle test. Its mesh is the triangles of
// a PhysX heightfield of Rows x Columns samples in sample space: rows along
// x, heights along y and columns along z, with the triangles of holes left
// out. ToGrid must scale those axes by positive factors. Thickness is
// PhysX's, in sample heights.
typedef struct
{
    uint64_t MeshKey;
    double ToGrid[12];
    double Thickness;
    uint32_t Rows;
    uint32_t Columns;
    uint32_t Key;
    uint32_t Flags;
} FVoxelGpuInstance;

// One separating axis of a convex shape and the shape's extent along it.
typedef struct
{
    double Direction[3];
    double Low;
    double High;
} FVoxelGpuAxis;

// A sphere (center in Points[0]), capsule (segment from Points[0] to
// Points[1]) or convex hull (axes FirstAxis to FirstAxis + NumAxes), with
// the voxels it may touch from Begin to End.
typedef struct
{
    double Points[2][3];
    double Radius;
    uint32_t Type;
    uint32_t Key;
    uint32_t FirstAxis;
    uint32_t NumAxes;
    int32_t Begin[3];
    int32_t End[3];
} FVoxelGpuShape;

// The grid and the labels whose voxels are listed with their instance.
typedef struct
{
    int32_t Size[3];
    uint32_t InstancedLabels;
} FVoxelGpuGrid;

// A voxel of a label with instances, as the sensor sends it.
typedef struct
{
    uint64_t Index;
    uint64_t Instance;
} FVoxelGpuPair;

typedef struct
{
    double PrepareMs;
    double RasterMs;
    double ShapesMs;
    double ResolveMs;
    double ReadbackMs;
    uint64_t Triangles;
    uint64_t Columns;
    uint64_t ShapeVoxels;
    uint64_t ExactTests;
    uint64_t Pairs;
    uint32_t Slabs;
} FVoxelGpuStats;

int32_t VoxelGpuApiVersion(void);

// Returns 0 and a context on success; otherwise writes the reason to Error.
int32_t VoxelGpuCreate(int32_t Device, FVoxelGpuContext** OutContext, char* Error, int32_t ErrorSize);

void VoxelGpuDestroy(FVoxelGpuContext* Context);

const char* VoxelGpuError(const FVoxelGpuContext* Context);

// Returns 1 and the sizes of a cached mesh, or 0.
int32_t VoxelGpuFindMesh(FVoxelGpuContext* Context, uint64_t Key, uint32_t* NumVertices, uint32_t* NumTriangles);

int32_t VoxelGpuUploadMesh(FVoxelGpuContext* Context, const FVoxelGpuMesh* Mesh);

// Drops a cached mesh, if there is one under Key, as when the mesh it came
// from is released and its key may come back for a different mesh.
void VoxelGpuReleaseMesh(FVoxelGpuContext* Context, uint64_t Key);

// Labels the grid into Labels, one byte per voxel, in slabs along X when it
// does not fit the GPU at once. Returns 0 on success.
int32_t VoxelGpuRun(
    FVoxelGpuContext* Context,
    const FVoxelGpuGrid* Grid,
    const FVoxelGpuInstance* Instances,
    uint32_t NumInstances,
    const FVoxelGpuShape* Shapes,
    uint32_t NumShapes,
    const FVoxelGpuAxis* Axes,
    uint32_t NumAxes,
    uint8_t* Labels,
    FVoxelGpuStats* Stats
);

// The voxels with an instance from the last run, valid until the next one.
int32_t VoxelGpuGetPairs(FVoxelGpuContext* Context, const FVoxelGpuPair** Pairs, uint64_t* NumPairs);

#ifdef __cplusplus
}
#endif
