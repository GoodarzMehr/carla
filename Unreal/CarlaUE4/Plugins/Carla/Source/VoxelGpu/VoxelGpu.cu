// libcarla_voxel_gpu: labels the voxel detection sensor's grid on a CUDA GPU
// without PhysX. See VoxelGpuApi.h for the interface. Built by
// Util/BuildTools/BuildVoxelGpu.sh when nvcc is available.
//
// Each frame the sensor uploads the meshes it has not cached yet and lists
// every shape it gathered with its transform into grid space. The grid is
// labeled in slabs along X that fit the GPU. Per slab, every triangle finds
// the voxel columns along its dominant normal axis that its plane may cross,
// each column tests its one to three voxels in double precision, and an
// atomic maximum on a 32-bit key keeps the winning label per voxel. Most
// voxels are empty, so a slab comes back as a bitmap of its occupied voxels
// and their labels, which the host expands into the sensor's buffer.

#include "../Carla/Sensor/VoxelGpuApi.h"

#include <cuda_runtime.h>
#include <cub/device/device_scan.cuh>

#include <emmintrin.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define VOXEL_GPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace
{
    // Meshes unused for this many runs are dropped from the GPU.
    constexpr uint64_t EvictAfterRuns = 600;

    constexpr int ThreadsPerBlock = 256;

    // Grid-stride loops keep launches within the block limit for any count.
    constexpr uint64_t MaxBlocks = 1u << 20;

    unsigned GetBlocks(uint64_t Count)
    {
        return unsigned(std::max<uint64_t>(1, std::min<uint64_t>((Count + ThreadsPerBlock - 1) / ThreadsPerBlock, MaxBlocks)));
    }

    template <typename T>
    struct FDeviceArray
    {
        T* Data = nullptr;
        size_t Capacity = 0;

        cudaError_t Reserve(size_t Count)
        {
            if (Count <= Capacity)
                return cudaSuccess;

            const size_t NewCapacity = std::max(Count, Capacity + Capacity / 2);

            Free();

            const cudaError_t Status = cudaMalloc(&Data, NewCapacity * sizeof(T));

            if (Status == cudaSuccess)
                Capacity = NewCapacity;
            else
                Data = nullptr;

            return Status;
        }

        void Free()
        {
            if (Data)
                cudaFree(Data);

            Data = nullptr;
            Capacity = 0;
        }
    };

    // Page-locked host memory, which the GPU copies to at full speed.
    template <typename T>
    struct FHostArray
    {
        T* Data = nullptr;
        size_t Capacity = 0;

        cudaError_t Reserve(size_t Count)
        {
            if (Count <= Capacity)
                return cudaSuccess;

            const size_t NewCapacity = std::max(Count, Capacity + Capacity / 2);

            Free();

            const cudaError_t Status = cudaMallocHost(&Data, NewCapacity * sizeof(T));

            if (Status == cudaSuccess)
                Capacity = NewCapacity;
            else
                Data = nullptr;

            return Status;
        }

        void Free()
        {
            if (Data)
                cudaFreeHost(Data);

            Data = nullptr;
            Capacity = 0;
        }
    };

    struct FMesh
    {
        FDeviceArray<float> Vertices;
        FDeviceArray<uint32_t> Indices;
        uint32_t NumVertices = 0;
        uint32_t NumTriangles = 0;
        uint64_t LastUsed = 0;
    };

    // An instance as the kernels read it. A heightfield also has the map
    // back to its sample space, and Skin, its thickness in voxels, which
    // bounds the voxels it may reach.
    struct FInstanceGpu
    {
        const float* Vertices;
        const uint32_t* Indices;
        double ToGrid[12];
        double ToSamples[12];
        double Skin;
        double Thickness;
        uint64_t FirstTriangle;
        uint32_t Rows;
        uint32_t Columns;
        uint32_t Key;
        uint32_t Flags;
    };

    constexpr int NumAxes = 13;

    // Values per axis: direction x, y and z, low and high.
    constexpr int AxisValues = NumAxes * 5;

    // A triangle with the voxel box it may touch in the slab. Dominant is its
    // largest normal axis, or -1 for a triangle of zero area. Unless bExact,
    // its 13 separating axes are kept in single precision, scaled to unit L1
    // length, with the range of voxel centers that may touch it along each,
    // measured from voxel Begin; Error bounds the rounding of a test against
    // them. The axes are stored column-wise, value V of triangle T at
    // V * NumTriangles + T, so neighboring threads write neighboring words.
    // Heightfields and zero-area triangles are tested in double precision
    // throughout.
    struct FPrepared
    {
        uint64_t Triangle;
        int32_t Begin[3];
        int32_t End[3];
        uint32_t Instance;
        int32_t Dominant;
        uint32_t bExact;
        float Error;
    };

    struct FShapeGpu
    {
        FVoxelGpuShape Shape;
        uint64_t FirstVoxel;
    };

    // ---------------------------------------------------------------------------------------------------------------
    // Device geometry, all in double precision.

    __device__ void CrossProduct(const double* A, const double* B, double* Out)
    {
        Out[0] = A[1] * B[2] - A[2] * B[1];
        Out[1] = A[2] * B[0] - A[0] * B[2];
        Out[2] = A[0] * B[1] - A[1] * B[0];
    }

    __device__ double DotProduct(const double* A, const double* B)
    {
        return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    }

    // Whether the unit voxel around Center and the triangle share a point:
    // the 13-axis test. A zero-length axis separates nothing, so a triangle
    // of zero area is tested as the segment or point it is.
    __device__ bool BoxTouchesTriangle(const double* Center, const double (*V)[3])
    {
        double P[3][3];

        for (int Vertex = 0; Vertex < 3; ++Vertex)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
                P[Vertex][Axis] = V[Vertex][Axis] - Center[Axis];
        }

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const double Low = fmin(P[0][Axis], fmin(P[1][Axis], P[2][Axis]));
            const double High = fmax(P[0][Axis], fmax(P[1][Axis], P[2][Axis]));

            if (Low > 0.5 || High < -0.5)
                return false;
        }

        double Edges[3][3];

        for (int Edge = 0; Edge < 3; ++Edge)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
                Edges[Edge][Axis] = P[(Edge + 1) % 3][Axis] - P[Edge][Axis];
        }

        double Normal[3];

        CrossProduct(Edges[0], Edges[1], Normal);

        const double Plane = DotProduct(Normal, P[0]);
        const double PlaneRadius = 0.5 * (fabs(Normal[0]) + fabs(Normal[1]) + fabs(Normal[2]));

        if (Plane > PlaneRadius || Plane < -PlaneRadius)
            return false;

        for (int Edge = 0; Edge < 3; ++Edge)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
            {
                double Unit[3] = {0.0, 0.0, 0.0};

                Unit[Axis] = 1.0;

                double Direction[3];

                CrossProduct(Edges[Edge], Unit, Direction);

                const double A = DotProduct(Direction, P[0]);
                const double B = DotProduct(Direction, P[1]);
                const double C = DotProduct(Direction, P[2]);

                const double Radius = 0.5 * (fabs(Direction[0]) + fabs(Direction[1]) + fabs(Direction[2]));

                if (fmin(A, fmin(B, C)) > Radius || fmax(A, fmax(B, C)) < -Radius)
                    return false;
            }
        }

        return true;
    }

    // A heightfield is tested in its sample space, where its cells are unit
    // squares between integer rows and columns. A coordinate within
    // SampleTolerance of an integer is taken as that integer, and a height
    // within it of the surface as on it: the map back from grid space
    // rounds, while PhysX decides its ties exactly there.
    constexpr double SampleTolerance = 1e-9;

    // How far outside a triangle PhysX's box edge test still finds a hit,
    // in barycentric coordinates.
    constexpr double EdgeSlack = 1e-4;

    __device__ double Snap(double Value)
    {
        const double Nearest = rint(Value);

        return fabs(Value - Nearest) <= SampleTolerance ? Nearest : Value;
    }

    __device__ void Transform(const double* M, const double* Point, double* Out)
    {
        for (int Row = 0; Row < 3; ++Row)
            Out[Row] = M[4 * Row] * Point[0] + M[4 * Row + 1] * Point[1] + M[4 * Row + 2] * Point[2] + M[4 * Row + 3];
    }

    // A heightfield triangle in sample space: its cell, whether the cell's
    // diagonal runs from its first sample to its last, whether it is the
    // second triangle of the cell, as PhysX numbers them, and its plane's
    // height at row and column 0 and slopes along them.
    struct FSurfaceTriangle
    {
        double Row;
        double Column;
        bool bZerothShared;
        bool bSecond;
        double Height;
        double RowSlope;
        double ColumnSlope;
    };

    __device__ FSurfaceTriangle GetSurfaceTriangle(const double (*S)[3])
    {
        FSurfaceTriangle T;

        T.Row = fmin(S[0][0], fmin(S[1][0], S[2][0]));
        T.Column = fmin(S[0][2], fmin(S[1][2], S[2][2]));

        bool bHas[2][2] = {{false, false}, {false, false}};

        for (int Vertex = 0; Vertex < 3; ++Vertex)
            bHas[S[Vertex][0] > T.Row][S[Vertex][2] > T.Column] = true;

        // Gu::HeightField::getTriangleVertices: with the diagonal from the
        // first sample, the first triangle has the sample one row on;
        // otherwise it has the first sample.
        T.bZerothShared = bHas[0][0] && bHas[1][1];
        T.bSecond = T.bZerothShared ? !bHas[1][0] : !bHas[0][0];

        // A cell triangle's normal has a y of plus or minus one.
        const double A[3] = {S[1][0] - S[0][0], S[1][1] - S[0][1], S[1][2] - S[0][2]};
        const double B[3] = {S[2][0] - S[0][0], S[2][1] - S[0][1], S[2][2] - S[0][2]};

        double Normal[3];

        CrossProduct(A, B, Normal);

        T.RowSlope = -Normal[0] / Normal[1];
        T.ColumnSlope = -Normal[2] / Normal[1];
        T.Height = S[0][1] - T.RowSlope * S[0][0] - T.ColumnSlope * S[0][2];

        return T;
    }

    // The surface's height at a row and column, on the triangle's plane.
    __device__ double GetSurfaceHeight(const FSurfaceTriangle& T, double Row, double Column)
    {
        return T.Height + T.RowSlope * Row + T.ColumnSlope * Column;
    }

    // Whether the triangle is the one PhysX takes at a row and column, as
    // HeightFieldUtil::getFaceIndexAtShapePoint does: the point must lie
    // on the heightfield short of its last row and column, and its cell and
    // half decide. A hole there is no triangle, so nothing takes it.
    __device__ bool TakesPoint(const FSurfaceTriangle& T, double Row, double Column, uint32_t Rows, uint32_t Columns)
    {
        if (!(Row >= 0.0 && Column >= 0.0 && Row < Rows - 1.0 && Column < Columns - 1.0))
            return false;

        if (floor(Row) != T.Row || floor(Column) != T.Column)
            return false;

        const double FracX = Row - T.Row;
        const double FracZ = Column - T.Column;

        const bool bSecond = T.bZerothShared ?
            FracZ - FracX > SampleTolerance :
            FracX + FracZ - 1.0 > SampleTolerance;

        return bSecond == T.bSecond;
    }

    // Whether PhysX's trace of the box edge from A to B, all in sample
    // space, visits the triangle's cell, which it must for the edge to hit
    // the triangle. The trace visits the cells the edge passes through. An
    // edge along a cell line visits the cells on the side of higher rows or
    // columns, as the trace nudges it there, except on the last line, which
    // the trace clamps into the last cells.
    __device__ bool VisitsCell(
        const FSurfaceTriangle& T,
        const double* A,
        const double* B,
        uint32_t Rows,
        uint32_t Columns
    )
    {
        double Low = 0.0;
        double High = 1.0;

        const int Axes[2] = {0, 2};
        const double Cells[2] = {T.Row, T.Column};
        const double Lines[2] = {Rows - 1.0, Columns - 1.0};

        for (int Index = 0; Index < 2; ++Index)
        {
            const double From = Snap(A[Axes[Index]]);
            const double To = Snap(B[Axes[Index]]);

            if (From == To)
            {
                if (!(From >= 0.0 && From <= Lines[Index]) || fmin(floor(From), Lines[Index] - 1.0) != Cells[Index])
                    return false;

                continue;
            }

            const double First = (Cells[Index] - From) / (To - From);
            const double Last = (Cells[Index] + 1.0 - From) / (To - From);

            Low = fmax(Low, fmin(First, Last));
            High = fmin(High, fmax(First, Last));
        }

        return Low < High;
    }

    // PhysX's heightfield rules (GuOverlapTestsHF.cpp, intersectHeightField
    // Box), with the triangle V in grid space and S in sample space: a box
    // corner within the thickness of the surface the heightfield has under
    // it, a box edge hitting the triangle, or a triangle vertex inside the
    // box. PhysX has no face test, and neither has this. PhysX also takes a
    // vertex only where its heightfield is convex or ends, which this does
    // not follow.
    __device__ bool BoxTouchesSurface(
        const double* Center,
        const double (*V)[3],
        const double (*S)[3],
        const FInstanceGpu& Instance
    )
    {
        for (int Vertex = 0; Vertex < 3; ++Vertex)
        {
            if (
                fabs(V[Vertex][0] - Center[0]) < 0.5 &&
                fabs(V[Vertex][1] - Center[1]) < 0.5 &&
                fabs(V[Vertex][2] - Center[2]) < 0.5
            )
                return true;
        }

        const FSurfaceTriangle T = GetSurfaceTriangle(S);

        // Corners in sample space, from the center and half the grid axes,
        // and their heights above the triangle's plane in sample heights.
        const double* M = Instance.ToSamples;

        double Middle[3];

        Transform(M, Center, Middle);

        double Corners[8][3];
        double Heights[8];

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            for (int Row = 0; Row < 3; ++Row)
            {
                double Value = Middle[Row];

                for (int Axis = 0; Axis < 3; ++Axis)
                    Value += ((Corner >> Axis) & 1 ? 0.5 : -0.5) * M[4 * Row + Axis];

                Corners[Corner][Row] = Value;
            }

            Heights[Corner] = Corners[Corner][1] - GetSurfaceHeight(T, Corners[Corner][0], Corners[Corner][2]);
        }

        // A corner counts within the thickness, which PhysX bounds with its
        // ends for a negative thickness, as Unreal's, and without them
        // otherwise.
        const double Thickness = Instance.Thickness;

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            const double Height = Heights[Corner];

            const bool bWithin = Thickness <= 0.0 ?
                Height <= SampleTolerance && Height >= Thickness - SampleTolerance :
                Height > SampleTolerance && Height < Thickness - SampleTolerance;

            if (!bWithin)
                continue;

            const double Row = Snap(Corners[Corner][0]);
            const double Column = Snap(Corners[Corner][2]);

            if (TakesPoint(T, Row, Column, Instance.Rows, Instance.Columns))
                return true;
        }

        // An edge hits where it meets the plane, ends included, unless it
        // lies in it, within EdgeSlack of the triangle.
        const double Area = (S[1][0] - S[0][0]) * (S[2][2] - S[0][2]) - (S[2][0] - S[0][0]) * (S[1][2] - S[0][2]);

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
            {
                const int Other = Corner | (1 << Axis);

                if (Other == Corner)
                    continue;

                const double HeightA = fabs(Heights[Corner]) <= SampleTolerance ? 0.0 : Heights[Corner];
                const double HeightB = fabs(Heights[Other]) <= SampleTolerance ? 0.0 : Heights[Other];

                if ((HeightA > 0.0 && HeightB > 0.0) || (HeightA < 0.0 && HeightB < 0.0))
                    continue;

                if (HeightA == 0.0 && HeightB == 0.0)
                    continue;

                const double Fraction = HeightA / (HeightA - HeightB);

                const double* A = Corners[Corner];
                const double* B = Corners[Other];

                const double Row = A[0] + (B[0] - A[0]) * Fraction;
                const double Column = A[2] + (B[2] - A[2]) * Fraction;

                bool bInside = true;

                for (int Vertex = 0; Vertex < 3; ++Vertex)
                {
                    const double* P = S[(Vertex + 1) % 3];
                    const double* Q = S[(Vertex + 2) % 3];

                    const double Weight = ((P[0] - Row) * (Q[2] - Column) - (Q[0] - Row) * (P[2] - Column)) / Area;

                    bInside = bInside && Weight >= -EdgeSlack;
                }

                if (bInside && VisitsCell(T, A, B, Instance.Rows, Instance.Columns))
                    return true;
            }
        }

        return false;
    }

    // Squared distance from a point to the unit voxel around Center.
    __device__ double BoxDistanceSquared(const double* Center, const double* Point)
    {
        double Sum = 0.0;

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const double Outside = fmax(0.0, fabs(Point[Axis] - Center[Axis]) - 0.5);

            Sum += Outside * Outside;
        }

        return Sum;
    }

    __device__ bool BoxTouchesShape(const double* Center, const FVoxelGpuShape& Shape, const FVoxelGpuAxis* Axes)
    {
        if (Shape.Type == VOXEL_GPU_SPHERE)
            return BoxDistanceSquared(Center, Shape.Points[0]) <= Shape.Radius * Shape.Radius;

        if (Shape.Type == VOXEL_GPU_CAPSULE)
        {
            // The squared distance to the box is convex along the segment,
            // so a ternary search finds its minimum.
            const auto Distance = [&](double T)
            {
                const double Point[3] = {
                    Shape.Points[0][0] + T * (Shape.Points[1][0] - Shape.Points[0][0]),
                    Shape.Points[0][1] + T * (Shape.Points[1][1] - Shape.Points[0][1]),
                    Shape.Points[0][2] + T * (Shape.Points[1][2] - Shape.Points[0][2])
                };

                return BoxDistanceSquared(Center, Point);
            };

            double Low = 0.0;
            double High = 1.0;

            for (int Step = 0; Step < 100; ++Step)
            {
                const double First = Low + (High - Low) / 3.0;
                const double Second = High - (High - Low) / 3.0;

                if (Distance(First) <= Distance(Second))
                    High = Second;
                else
                    Low = First;
            }

            const double Nearest = fmin(Distance(0.5 * (Low + High)), fmin(Distance(0.0), Distance(1.0)));

            return Nearest <= Shape.Radius * Shape.Radius;
        }

        for (uint32_t Index = 0; Index < Shape.NumAxes; ++Index)
        {
            const FVoxelGpuAxis& Axis = Axes[Shape.FirstAxis + Index];

            const double Projection = DotProduct(Axis.Direction, Center);
            const double Radius = 0.5 * (fabs(Axis.Direction[0]) + fabs(Axis.Direction[1]) + fabs(Axis.Direction[2]));

            if (Projection + Radius < Axis.Low || Projection - Radius > Axis.High)
                return false;
        }

        return true;
    }

    __device__ int64_t GetSlabIndex(const int32_t* Cell, int32_t SlabBegin, int32_t SizeY, int32_t SizeZ)
    {
        return (int64_t(Cell[0] - SlabBegin) * SizeY + Cell[1]) * SizeZ + Cell[2];
    }

    // Voxel indices from Low to High, clamped to [Min, Max] before the
    // conversion so huge or non-finite values cannot overflow.
    __device__ int32_t ClampIndex(double Value, int32_t Min, int32_t Max)
    {
        if (!(Value > double(Min)))
            return Min;

        if (!(Value < double(Max)))
            return Max;

        return int32_t(Value);
    }

    // ---------------------------------------------------------------------------------------------------------------
    // Kernels.

    // Triangle Local of an instance in its mesh's space.
    __device__ void LoadSamples(const FInstanceGpu& Instance, uint64_t Local, double (*S)[3])
    {
        for (int Vertex = 0; Vertex < 3; ++Vertex)
        {
            const float* Source = Instance.Vertices + 3 * uint64_t(Instance.Indices[3 * Local + Vertex]);

            for (int Axis = 0; Axis < 3; ++Axis)
                S[Vertex][Axis] = Source[Axis];
        }
    }

    // Triangle Local of an instance in grid space, in double precision.
    __device__ void LoadTriangle(const FInstanceGpu& Instance, uint64_t Local, double (*V)[3])
    {
        for (int Vertex = 0; Vertex < 3; ++Vertex)
        {
            const float* Source = Instance.Vertices + 3 * uint64_t(Instance.Indices[3 * Local + Vertex]);

            for (int Row = 0; Row < 3; ++Row)
            {
                const double* M = Instance.ToGrid + 4 * Row;

                V[Vertex][Row] = M[0] * Source[0] + M[1] * Source[1] + M[2] * Source[2] + M[3];
            }
        }
    }

    // Writes triangle T's axes at unit L1 length, with its range along each
    // widened by the voxel's radius, 0.5, and returns the band that bounds
    // the rounding of a test against them. An axis of zero length separates
    // nothing and passes every voxel. Q holds the vertices relative to voxel
    // Begin.
    __device__ float WriteAxes(
        const double (*Directions)[3],
        const double (*Q)[3],
        const FPrepared& P,
        uint64_t T,
        uint64_t NumTriangles,
        float* Axes
    )
    {
        double Magnitude = 0.0;

        for (int Index = 0; Index < NumAxes; ++Index)
        {
            const double* Direction = Directions[Index];

            const double Length = fabs(Direction[0]) + fabs(Direction[1]) + fabs(Direction[2]);

            float* Axis = Axes + uint64_t(5 * Index) * NumTriangles + T;

            if (!(Length > 0.0))
            {
                Axis[0] = 0.0f;
                Axis[NumTriangles] = 0.0f;
                Axis[2 * NumTriangles] = 0.0f;
                Axis[3 * NumTriangles] = -FLT_MAX;
                Axis[4 * NumTriangles] = FLT_MAX;

                continue;
            }

            const double Inverse = 1.0 / Length;

            const double Unit[3] = {Direction[0] * Inverse, Direction[1] * Inverse, Direction[2] * Inverse};

            const double A = DotProduct(Unit, Q[0]);
            const double B = DotProduct(Unit, Q[1]);
            const double C = DotProduct(Unit, Q[2]);

            const double Low = fmin(A, fmin(B, C)) - 0.5;
            const double High = fmax(A, fmax(B, C)) + 0.5;

            Axis[0] = float(Unit[0]);
            Axis[NumTriangles] = float(Unit[1]);
            Axis[2 * NumTriangles] = float(Unit[2]);
            Axis[3 * NumTriangles] = float(Low);
            Axis[4 * NumTriangles] = float(High);

            Magnitude = fmax(Magnitude, fmax(fabs(Low), fabs(High)));
        }

        // Centers are at most the box size from Begin. Rounding the axes and
        // ranges to single precision and projecting a center costs a few
        // units in the last place of these magnitudes; twice that is kept as
        // the band.
        double Extent = 0.0;

        for (int Axis = 0; Axis < 3; ++Axis)
            Extent = fmax(Extent, double(P.End[Axis] - P.Begin[Axis]));

        return float(8.0 * FLT_EPSILON * (Magnitude + Extent + 2.0));
    }

    __global__ void PrepareTriangles(
        const FInstanceGpu* Instances,
        uint32_t NumInstances,
        uint64_t NumTriangles,
        int32_t SlabBegin,
        int32_t SlabEnd,
        int32_t SizeY,
        int32_t SizeZ,
        FPrepared* Prepared,
        float* Axes,
        uint64_t* Columns
    )
    {
        for (uint64_t T = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; T < NumTriangles; T += uint64_t(gridDim.x) * blockDim.x)
        {
            // The last instance whose first triangle is at or before T.
            uint32_t Low = 0;
            uint32_t High = NumInstances;

            while (High - Low > 1)
            {
                const uint32_t Middle = (Low + High) / 2;

                if (Instances[Middle].FirstTriangle <= T)
                    Low = Middle;
                else
                    High = Middle;
            }

            const FInstanceGpu& Instance = Instances[Low];

            FPrepared P;

            P.Triangle = T - Instance.FirstTriangle;
            P.Instance = Low;

            double V[3][3];

            LoadTriangle(Instance, P.Triangle, V);

            double Min[3] = {INFINITY, INFINITY, INFINITY};
            double Max[3] = {-INFINITY, -INFINITY, -INFINITY};

            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    Min[Axis] = fmin(Min[Axis], V[Vertex][Axis]);
                    Max[Axis] = fmax(Max[Axis], V[Vertex][Axis]);
                }
            }

            // Voxel I spans [I, I + 1], so it may touch [Min, Max] when
            // ceil(Min) - 1 <= I <= floor(Max). A heightfield also reaches
            // its skin, and its edge test EdgeSlack outside the triangle,
            // which in barycentric coordinates is within twice EdgeSlack of
            // its extent past its bounds.
            const bool bSurface = (Instance.Flags & VOXEL_GPU_HEIGHTFIELD) != 0;

            const int32_t Lower[3] = {SlabBegin, 0, 0};
            const int32_t Upper[3] = {SlabEnd, SizeY, SizeZ};

            bool bEmpty = false;

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                const double Reach = bSurface ? Instance.Skin + 2.0 * EdgeSlack * (Max[Axis] - Min[Axis]) : 0.0;

                P.Begin[Axis] = ClampIndex(ceil(Min[Axis] - Reach) - 1.0, Lower[Axis], Upper[Axis]);
                P.End[Axis] = ClampIndex(floor(Max[Axis] + Reach) + 1.0, Lower[Axis], Upper[Axis]);

                bEmpty = bEmpty || P.Begin[Axis] >= P.End[Axis] || !(Min[Axis] == Min[Axis]) || !(Max[Axis] == Max[Axis]);
            }

            // Most triangles of a large mesh miss the slab; a triangle without
            // columns is never looked up, so it needs nothing else.
            if (bEmpty)
            {
                Columns[T] = 0;

                continue;
            }

            // Vertices relative to voxel Begin keep the numbers small.
            double Q[3][3];

            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                for (int Axis = 0; Axis < 3; ++Axis)
                    Q[Vertex][Axis] = V[Vertex][Axis] - P.Begin[Axis];
            }

            const double Edges[3][3] = {
                {Q[1][0] - Q[0][0], Q[1][1] - Q[0][1], Q[1][2] - Q[0][2]},
                {Q[2][0] - Q[1][0], Q[2][1] - Q[1][1], Q[2][2] - Q[1][2]},
                {Q[0][0] - Q[2][0], Q[0][1] - Q[2][1], Q[0][2] - Q[2][2]}
            };

            double Directions[NumAxes][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};

            CrossProduct(Edges[0], Edges[1], Directions[3]);

            for (int Edge = 0; Edge < 3; ++Edge)
            {
                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    const double Unit[3] = {Axis == 0 ? 1.0 : 0.0, Axis == 1 ? 1.0 : 0.0, Axis == 2 ? 1.0 : 0.0};

                    CrossProduct(Edges[Edge], Unit, Directions[4 + 3 * Edge + Axis]);
                }
            }

            P.Dominant = -1;

            double Largest = 0.0;

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                if (fabs(Directions[3][Axis]) > Largest)
                {
                    Largest = fabs(Directions[3][Axis]);

                    P.Dominant = Axis;
                }
            }

            P.bExact = bSurface || P.Dominant < 0;

            // Heightfield and zero-area triangles are tested from their
            // vertices alone and need no axes.
            P.Error = P.bExact ? 0.0f : WriteAxes(Directions, Q, P, T, NumTriangles, Axes);

            const int D = P.Dominant < 0 ? 2 : P.Dominant;
            const int U = (D + 1) % 3;
            const int W = (D + 2) % 3;

            Prepared[T] = P;

            Columns[T] = uint64_t(P.End[U] - P.Begin[U]) * uint64_t(P.End[W] - P.Begin[W]);
        }
    }

    // The voxels of one column whose box the triangle's plane may cross,
    // from double-precision vertices, with one voxel to spare on each side.
    __device__ void GetExactRange(
        const double (*V)[3],
        const int32_t* Cell,
        int D,
        int U,
        int W,
        double Skin,
        const FPrepared& P,
        int32_t& First,
        int32_t& Last
    )
    {
        double Edges[2][3];

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            Edges[0][Axis] = V[1][Axis] - V[0][Axis];
            Edges[1][Axis] = V[2][Axis] - V[0][Axis];
        }

        double Normal[3];

        CrossProduct(Edges[0], Edges[1], Normal);

        const double Rest = DotProduct(Normal, V[0]) - Normal[U] * (Cell[U] + 0.5) - Normal[W] * (Cell[W] + 0.5);

        const double Reach = 0.5 * (fabs(Normal[0]) + fabs(Normal[1]) + fabs(Normal[2])) +
                             Skin * sqrt(DotProduct(Normal, Normal));

        double A = (Rest - Reach) / Normal[D];
        double B = (Rest + Reach) / Normal[D];

        if (A > B)
        {
            const double Swap = A;

            A = B;
            B = Swap;
        }

        First = max(First, ClampIndex(floor(A - 0.5) - 1.0, P.Begin[D], P.End[D]));
        Last = min(Last, ClampIndex(floor(B - 0.5) + 2.0, P.Begin[D], P.End[D]));
    }

    __global__ void RasterizeColumns(
        const FPrepared* Prepared,
        const float* Axes,
        const FInstanceGpu* Instances,
        const uint64_t* Offsets,
        uint64_t NumTriangles,
        uint64_t NumColumns,
        int32_t SlabBegin,
        int32_t SizeY,
        int32_t SizeZ,
        uint32_t* Keys,
        unsigned long long* Exact
    )
    {
        for (uint64_t Work = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; Work < NumColumns; Work += uint64_t(gridDim.x) * blockDim.x)
        {
            // The last triangle whose first column is at or before Work.
            uint64_t Low = 0;
            uint64_t High = NumTriangles;

            while (High - Low > 1)
            {
                const uint64_t Middle = (Low + High) / 2;

                if (Offsets[Middle] <= Work)
                    Low = Middle;
                else
                    High = Middle;
            }

            const FPrepared& P = Prepared[Low];
            const FInstanceGpu& Instance = Instances[P.Instance];

            const int D = P.Dominant < 0 ? 2 : P.Dominant;
            const int U = (D + 1) % 3;
            const int W = (D + 2) % 3;

            const uint64_t Local = Work - Offsets[Low];
            const uint64_t NumW = uint64_t(P.End[W] - P.Begin[W]);

            int32_t Cell[3];

            Cell[U] = P.Begin[U] + int32_t(Local / NumW);
            Cell[W] = P.Begin[W] + int32_t(Local % NumW);

            int32_t First = P.Begin[D];
            int32_t Last = P.End[D];

            if (P.bExact)
            {
                const bool bSurface = (Instance.Flags & VOXEL_GPU_HEIGHTFIELD) != 0;

                double V[3][3];
                double S[3][3];

                LoadTriangle(Instance, P.Triangle, V);

                if (bSurface)
                    LoadSamples(Instance, P.Triangle, S);

                if (P.Dominant >= 0)
                    GetExactRange(V, Cell, D, U, W, bSurface ? Instance.Skin : 0.0, P, First, Last);

                for (int32_t K = First; K < Last; ++K)
                {
                    Cell[D] = K;

                    const double Center[3] = {Cell[0] + 0.5, Cell[1] + 0.5, Cell[2] + 0.5};

                    const bool bTouches = bSurface ?
                        BoxTouchesSurface(Center, V, S, Instance) :
                        BoxTouchesTriangle(Center, V);

                    if (bTouches)
                        atomicMax(Keys + GetSlabIndex(Cell, SlabBegin, SizeY, SizeZ), Instance.Key);
                }

                continue;
            }

            // The triangle's axes, and the projections of the column's center
            // on them without D.
            float Axis[NumAxes][5];

            for (int Value = 0; Value < AxisValues; ++Value)
                Axis[Value / 5][Value % 5] = Axes[uint64_t(Value) * NumTriangles + Low];

            const float CU = float(Cell[U] - P.Begin[U]) + 0.5f;
            const float CW = float(Cell[W] - P.Begin[W]) + 0.5f;

            float Base[NumAxes];

            for (int Index = 0; Index < NumAxes; ++Index)
                Base[Index] = Axis[Index][U] * CU + Axis[Index][W] * CW;

            // Along D, the plane axis keeps one to three voxels.
            {
                const float* Plane = Axis[3];

                double A = (double(Plane[3]) - P.Error - Base[3]) / Plane[D];
                double B = (double(Plane[4]) + P.Error - Base[3]) / Plane[D];

                if (A > B)
                {
                    const double Swap = A;

                    A = B;
                    B = Swap;
                }

                First = max(First, ClampIndex(floor(A - 0.5) - 1.0 + P.Begin[D], P.Begin[D], P.End[D]));
                Last = min(Last, ClampIndex(floor(B - 0.5) + 2.0 + P.Begin[D], P.Begin[D], P.End[D]));
            }

            for (int32_t K = First; K < Last; ++K)
            {
                const float CD = float(K - P.Begin[D]) + 0.5f;

                bool bOutside = false;
                bool bInside = true;

                for (int Index = 0; Index < NumAxes && !bOutside; ++Index)
                {
                    const float Projection = Base[Index] + Axis[Index][D] * CD;

                    const float Distance = fmaxf(Axis[Index][3] - Projection, Projection - Axis[Index][4]);

                    bOutside = Distance > P.Error;
                    bInside = bInside && Distance < -P.Error;
                }

                if (bOutside)
                    continue;

                Cell[D] = K;

                // Within the band of some axis only the double-precision test
                // can tell.
                if (!bInside)
                {
                    atomicAdd(Exact, 1ull);

                    double V[3][3];

                    LoadTriangle(Instance, P.Triangle, V);

                    const double Center[3] = {Cell[0] + 0.5, Cell[1] + 0.5, Cell[2] + 0.5};

                    if (!BoxTouchesTriangle(Center, V))
                        continue;
                }

                atomicMax(Keys + GetSlabIndex(Cell, SlabBegin, SizeY, SizeZ), Instance.Key);
            }
        }
    }

    __global__ void RasterizeShapes(
        const FShapeGpu* Shapes,
        uint32_t NumShapes,
        const FVoxelGpuAxis* Axes,
        uint64_t NumVoxels,
        int32_t SlabBegin,
        int32_t SizeY,
        int32_t SizeZ,
        uint32_t* Keys
    )
    {
        for (uint64_t W = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x; W < NumVoxels; W += uint64_t(gridDim.x) * blockDim.x)
        {
            uint32_t Low = 0;
            uint32_t High = NumShapes;

            while (High - Low > 1)
            {
                const uint32_t Middle = (Low + High) / 2;

                if (Shapes[Middle].FirstVoxel <= W)
                    Low = Middle;
                else
                    High = Middle;
            }

            const FVoxelGpuShape& Shape = Shapes[Low].Shape;

            const uint64_t Local = W - Shapes[Low].FirstVoxel;
            const uint64_t NumY = uint64_t(Shape.End[1] - Shape.Begin[1]);
            const uint64_t NumZ = uint64_t(Shape.End[2] - Shape.Begin[2]);

            const int32_t Cell[3] = {
                Shape.Begin[0] + int32_t(Local / (NumY * NumZ)),
                Shape.Begin[1] + int32_t(Local / NumZ % NumY),
                Shape.Begin[2] + int32_t(Local % NumZ)
            };

            const double Center[3] = {Cell[0] + 0.5, Cell[1] + 0.5, Cell[2] + 0.5};

            if (BoxTouchesShape(Center, Shape, Axes))
                atomicMax(Keys + GetSlabIndex(Cell, SlabBegin, SizeY, SizeZ), Shape.Key);
        }
    }

    __device__ uint8_t GetLabel(uint32_t Key)
    {
        return Key ? uint8_t(31 - ((Key >> 16) & 31)) : uint8_t(0);
    }

    // The slab goes back to the host packed: a bitmap of the occupied voxels
    // and their labels in order. Bit B of the bitmap is voxel B - Shift of
    // the slab, where Shift is the host address of the slab's first label
    // modulo 64, so every 64-bit word fills one host cache line. A warp
    // handles one word, its lanes one voxel in each half.
    __device__ uint32_t GetSlabKey(const uint32_t* Keys, uint64_t NumVoxels, uint64_t Shift, uint64_t Bit)
    {
        return Bit >= Shift && Bit - Shift < NumVoxels ? Keys[Bit - Shift] : 0u;
    }

    __device__ bool IsInstanced(uint32_t Key, uint32_t InstancedLabels)
    {
        const uint8_t Label = GetLabel(Key);

        return Label && ((InstancedLabels >> Label) & 1);
    }

    __global__ void PackBitmap(
        const uint32_t* Keys,
        uint64_t NumVoxels,
        uint64_t Shift,
        uint64_t NumWords,
        uint32_t InstancedLabels,
        uint64_t* Bitmap,
        uint64_t* Counts,
        unsigned long long* NumPairs
    )
    {
        const uint32_t Lane = threadIdx.x % 32;

        const uint64_t First = (blockIdx.x * uint64_t(blockDim.x) + threadIdx.x) / 32;
        const uint64_t Stride = uint64_t(gridDim.x) * blockDim.x / 32;

        for (uint64_t Word = First; Word < NumWords; Word += Stride)
        {
            const uint32_t Low = GetSlabKey(Keys, NumVoxels, Shift, 64 * Word + Lane);
            const uint32_t High = GetSlabKey(Keys, NumVoxels, Shift, 64 * Word + 32 + Lane);

            const uint32_t LowBits = __ballot_sync(0xFFFFFFFFu, GetLabel(Low) != 0);
            const uint32_t HighBits = __ballot_sync(0xFFFFFFFFu, GetLabel(High) != 0);

            const uint32_t Instanced = __popc(__ballot_sync(0xFFFFFFFFu, IsInstanced(Low, InstancedLabels))) +
                                       __popc(__ballot_sync(0xFFFFFFFFu, IsInstanced(High, InstancedLabels)));

            if (Lane == 0)
            {
                Bitmap[Word] = uint64_t(HighBits) << 32 | LowBits;
                Counts[Word] = __popc(LowBits) + __popc(HighBits);

                if (Instanced > 0)
                    atomicAdd(NumPairs, (unsigned long long)Instanced);
            }
        }
    }

    __global__ void PackLabels(
        const uint32_t* Keys,
        uint64_t NumVoxels,
        uint64_t Shift,
        uint64_t NumWords,
        uint64_t FirstIndex,
        uint32_t InstancedLabels,
        const uint64_t* Offsets,
        uint8_t* Packed,
        FVoxelGpuPair* Pairs,
        unsigned long long* NextPair
    )
    {
        const uint32_t Lane = threadIdx.x % 32;
        const uint32_t Before = (1u << Lane) - 1;

        const uint64_t First = (blockIdx.x * uint64_t(blockDim.x) + threadIdx.x) / 32;
        const uint64_t Stride = uint64_t(gridDim.x) * blockDim.x / 32;

        for (uint64_t Word = First; Word < NumWords; Word += Stride)
        {
            const uint32_t Low = GetSlabKey(Keys, NumVoxels, Shift, 64 * Word + Lane);
            const uint32_t High = GetSlabKey(Keys, NumVoxels, Shift, 64 * Word + 32 + Lane);

            const uint32_t LowBits = __ballot_sync(0xFFFFFFFFu, GetLabel(Low) != 0);
            const uint32_t HighBits = __ballot_sync(0xFFFFFFFFu, GetLabel(High) != 0);

            const uint64_t Offset = Offsets[Word];

            const uint32_t HalfKeys[2] = {Low, High};
            const uint64_t Slots[2] = {
                Offset + __popc(LowBits & Before),
                Offset + __popc(LowBits) + __popc(HighBits & Before)
            };

            for (int Half = 0; Half < 2; ++Half)
            {
                const uint32_t Key = HalfKeys[Half];

                if (!GetLabel(Key))
                    continue;

                Packed[Slots[Half]] = GetLabel(Key);

                if (!IsInstanced(Key, InstancedLabels))
                    continue;

                const unsigned long long Slot = atomicAdd(NextPair, 1ull);

                Pairs[Slot].Index = FirstIndex + 64 * Word + 32 * Half + Lane - Shift;
                Pairs[Slot].Instance = 0xFFFF - (Key & 0xFFFF);
            }
        }
    }

    // Writes the labels of words FirstWord to LastWord of a packed slab,
    // starting from packed label Next. A line inside the slab is built in a
    // small buffer and streamed to memory, so the host writes the slab at
    // its memory bandwidth; the first and last lines may reach past the
    // slab and are written byte by byte.
    void ExpandWords(
        uint8_t* Labels,
        uint64_t NumVoxels,
        uint64_t Shift,
        const uint64_t* Bitmap,
        const uint8_t* Packed,
        uint64_t FirstWord,
        uint64_t LastWord,
        uint64_t Next
    )
    {
        alignas(64) uint8_t Buffer[64];

        for (uint64_t Word = FirstWord; Word < LastWord; ++Word)
        {
            std::memset(Buffer, 0, sizeof(Buffer));

            for (uint64_t Bits = Bitmap[Word]; Bits != 0; Bits &= Bits - 1)
                Buffer[__builtin_ctzll(Bits)] = Packed[Next++];

            if (64 * Word < Shift || 64 * Word + 64 > Shift + NumVoxels)
            {
                for (uint64_t Byte = 0; Byte < 64; ++Byte)
                {
                    const uint64_t Voxel = 64 * Word + Byte - Shift;

                    if (64 * Word + Byte >= Shift && Voxel < NumVoxels)
                        Labels[Voxel] = Buffer[Byte];
                }

                continue;
            }

            uint8_t* Line = Labels + (64 * Word - Shift);

            for (int Part = 0; Part < 4; ++Part)
            {
                const __m128i Value = _mm_load_si128(reinterpret_cast<const __m128i*>(Buffer + 16 * Part));

                _mm_stream_si128(reinterpret_cast<__m128i*>(Line + 16 * Part), Value);
            }
        }

        _mm_sfence();
    }
}

struct FVoxelGpuContext
{
    int32_t Device = 0;
    std::string Error;
    uint64_t RunCount = 0;
    std::unordered_map<uint64_t, FMesh> Meshes;

    FDeviceArray<FInstanceGpu> Instances;
    FDeviceArray<FPrepared> Prepared;
    FDeviceArray<float> TriangleAxes;
    FDeviceArray<uint64_t> Columns;
    FDeviceArray<uint64_t> Offsets;
    FDeviceArray<uint8_t> ScanTemp;
    FDeviceArray<FShapeGpu> Shapes;
    FDeviceArray<FVoxelGpuAxis> Axes;
    FDeviceArray<uint32_t> Keys;
    FDeviceArray<uint64_t> Bitmap;
    FDeviceArray<uint64_t> Counts;
    FDeviceArray<uint64_t> WordOffsets;
    FDeviceArray<uint8_t> Packed;
    FDeviceArray<FVoxelGpuPair> Pairs;
    FDeviceArray<unsigned long long> Counters;

    // The packed slab on the host: the caller's label buffer is never
    // registered with CUDA, since it may be freed and reallocated at the
    // same address while still registered, which sends copies to its old
    // pages.
    FHostArray<uint64_t> HostBitmap;
    FHostArray<uint8_t> HostPacked;
    FHostArray<uint64_t> HostStarts;

    std::vector<FVoxelGpuPair> HostPairs;

    cudaEvent_t Events[2] = {nullptr, nullptr};

    bool Fail(cudaError_t Status, const char* What)
    {
        if (Status == cudaSuccess)
            return false;

        Error = std::string(What) + ": " + cudaGetErrorString(Status);

        return true;
    }

    // Milliseconds since the last call, once the GPU has caught up.
    double Lap()
    {
        cudaEventRecord(Events[1]);
        cudaEventSynchronize(Events[1]);

        float Milliseconds = 0.0f;

        cudaEventElapsedTime(&Milliseconds, Events[0], Events[1]);

        std::swap(Events[0], Events[1]);

        // Timing is best effort; its errors must not reach the next check.
        cudaGetLastError();

        return Milliseconds;
    }

    // The buffers rebuilt every frame.
    void ReleaseScratch()
    {
        Instances.Free();
        Prepared.Free();
        TriangleAxes.Free();
        Columns.Free();
        Offsets.Free();
        ScanTemp.Free();
        Shapes.Free();
        Axes.Free();
        Keys.Free();
        Bitmap.Free();
        Counts.Free();
        WordOffsets.Free();
        Packed.Free();
        Pairs.Free();
        HostBitmap.Free();
        HostPacked.Free();
        HostStarts.Free();
    }

    void Release()
    {
        cudaGetLastError();

        for (auto& Entry : Meshes)
        {
            Entry.second.Vertices.Free();
            Entry.second.Indices.Free();
        }

        Meshes.clear();

        ReleaseScratch();

        Counters.Free();

        for (cudaEvent_t& Event : Events)
        {
            if (Event)
                cudaEventDestroy(Event);

            Event = nullptr;
        }
    }
};

#define VOXEL_GPU_CHECK(Call)                     \
    do                                            \
    {                                             \
        if (Context->Fail((Call), #Call))         \
            return -1;                            \
    } while (false)

VOXEL_GPU_EXPORT int32_t VoxelGpuApiVersion(void)
{
    return VOXEL_GPU_API_VERSION;
}

VOXEL_GPU_EXPORT int32_t VoxelGpuCreate(int32_t Device, FVoxelGpuContext** OutContext, char* Error, int32_t ErrorSize)
{
    *OutContext = nullptr;

    int Count = 0;

    cudaError_t Status = cudaGetDeviceCount(&Count);

    if (Status == cudaSuccess && (Device < 0 || Device >= Count))
    {
        std::snprintf(Error, size_t(ErrorSize), "CUDA device %d does not exist; %d found", Device, Count);

        return -1;
    }

    if (Status == cudaSuccess)
        Status = cudaSetDevice(Device);

    FVoxelGpuContext* Context = new FVoxelGpuContext;

    Context->Device = Device;

    if (Status == cudaSuccess)
        Status = cudaEventCreate(&Context->Events[0]);

    if (Status == cudaSuccess)
        Status = cudaEventCreate(&Context->Events[1]);

    if (Status == cudaSuccess)
        Status = cudaEventRecord(Context->Events[0]);

    if (Status == cudaSuccess)
        Status = Context->Counters.Reserve(3);

    if (Status != cudaSuccess)
    {
        std::snprintf(Error, size_t(ErrorSize), "CUDA: %s", cudaGetErrorString(Status));

        Context->Release();

        delete Context;

        return -1;
    }

    *OutContext = Context;

    return 0;
}

VOXEL_GPU_EXPORT void VoxelGpuDestroy(FVoxelGpuContext* Context)
{
    if (!Context)
        return;

    cudaSetDevice(Context->Device);

    Context->Release();

    delete Context;
}

VOXEL_GPU_EXPORT const char* VoxelGpuError(const FVoxelGpuContext* Context)
{
    return Context ? Context->Error.c_str() : "no context";
}

VOXEL_GPU_EXPORT int32_t VoxelGpuFindMesh(
    FVoxelGpuContext* Context,
    uint64_t Key,
    uint32_t* NumVertices,
    uint32_t* NumTriangles
)
{
    const auto Found = Context->Meshes.find(Key);

    if (Found == Context->Meshes.end())
        return 0;

    *NumVertices = Found->second.NumVertices;
    *NumTriangles = Found->second.NumTriangles;

    return 1;
}

static int32_t UploadMesh(FVoxelGpuContext* Context, const FVoxelGpuMesh* Mesh, FMesh& Entry)
{

    if (Mesh->Indices)
    {
        VOXEL_GPU_CHECK(Entry.Indices.Reserve(3 * size_t(Mesh->NumTriangles)));

        VOXEL_GPU_CHECK(cudaMemcpy(
            Entry.Indices.Data,
            Mesh->Indices,
            3 * size_t(Mesh->NumTriangles) * sizeof(uint32_t),
            cudaMemcpyHostToDevice
        ));

        Entry.NumTriangles = Mesh->NumTriangles;
    }
    else if (Entry.NumTriangles != Mesh->NumTriangles || Entry.NumVertices != Mesh->NumVertices)
    {
        Context->Error = "vertices replaced for a mesh whose size changed";

        return -1;
    }

    VOXEL_GPU_CHECK(Entry.Vertices.Reserve(3 * size_t(Mesh->NumVertices)));

    VOXEL_GPU_CHECK(cudaMemcpy(
        Entry.Vertices.Data,
        Mesh->Vertices,
        3 * size_t(Mesh->NumVertices) * sizeof(float),
        cudaMemcpyHostToDevice
    ));

    Entry.NumVertices = Mesh->NumVertices;
    Entry.LastUsed = Context->RunCount + 1;

    return 0;
}

VOXEL_GPU_EXPORT int32_t VoxelGpuUploadMesh(FVoxelGpuContext* Context, const FVoxelGpuMesh* Mesh)
{
    VOXEL_GPU_CHECK(cudaSetDevice(Context->Device));

    FMesh& Entry = Context->Meshes[Mesh->Key];

    // A mesh left half written would pair new indices with old vertices, so
    // it is dropped and uploaded whole next time.
    if (UploadMesh(Context, Mesh, Entry) != 0)
    {
        Entry.Vertices.Free();
        Entry.Indices.Free();

        Context->Meshes.erase(Mesh->Key);

        return -1;
    }

    return 0;
}

VOXEL_GPU_EXPORT void VoxelGpuReleaseMesh(FVoxelGpuContext* Context, uint64_t Key)
{
    const auto Found = Context->Meshes.find(Key);

    if (Found == Context->Meshes.end())
        return;

    cudaSetDevice(Context->Device);

    Found->second.Vertices.Free();
    Found->second.Indices.Free();

    Context->Meshes.erase(Found);

    // An error here must not fail the next call that checks.
    cudaGetLastError();
}

// Bitmap words per chunk of the host expansion: 4 MB of labels.
constexpr uint64_t ChunkWords = uint64_t(1) << 16;

// A few threads reach the host's memory bandwidth.
constexpr uint64_t MaxExpandThreads = 8;

// Writes a packed slab into Labels. Starts holds the first packed label of
// each chunk of ChunkWords words.
static void ExpandSlab(
    uint8_t* Labels,
    uint64_t NumVoxels,
    uint64_t Shift,
    uint64_t NumWords,
    const uint64_t* Bitmap,
    const uint8_t* Packed,
    const uint64_t* Starts
)
{
    const uint64_t NumChunks = (NumWords + ChunkWords - 1) / ChunkWords;

    std::atomic<uint64_t> NextChunk(0);

    const auto Work = [&]()
    {
        for (uint64_t Chunk = NextChunk++; Chunk < NumChunks; Chunk = NextChunk++)
        {
            const uint64_t FirstWord = Chunk * ChunkWords;
            const uint64_t LastWord = std::min(NumWords, FirstWord + ChunkWords);

            ExpandWords(Labels, NumVoxels, Shift, Bitmap, Packed, FirstWord, LastWord, Starts[Chunk]);
        }
    };

    const uint64_t Cores = std::max(1u, std::thread::hardware_concurrency());

    const uint64_t NumThreads = std::min({NumChunks, MaxExpandThreads, std::max<uint64_t>(1, Cores / 4)});

    std::vector<std::thread> Helpers;

    // Without helpers this thread does all the work.
    try
    {
        for (uint64_t Index = 1; Index < NumThreads; ++Index)
            Helpers.emplace_back(Work);
    }
    catch (const std::system_error&)
    {
    }

    Work();

    for (std::thread& Helper : Helpers)
        Helper.join();
}

// The inverse of a row-major 3 x 4 affine map, if it has one.
static bool Invert(const double* M, double* Out)
{
    const double Determinant =
        M[0] * (M[5] * M[10] - M[6] * M[9]) -
        M[1] * (M[4] * M[10] - M[6] * M[8]) +
        M[2] * (M[4] * M[9] - M[5] * M[8]);

    if (!(std::fabs(Determinant) > 0.0) || !std::isfinite(Determinant))
        return false;

    const double Inverse = 1.0 / Determinant;

    double R[9];

    R[0] = (M[5] * M[10] - M[6] * M[9]) * Inverse;
    R[1] = (M[2] * M[9] - M[1] * M[10]) * Inverse;
    R[2] = (M[1] * M[6] - M[2] * M[5]) * Inverse;
    R[3] = (M[6] * M[8] - M[4] * M[10]) * Inverse;
    R[4] = (M[0] * M[10] - M[2] * M[8]) * Inverse;
    R[5] = (M[2] * M[4] - M[0] * M[6]) * Inverse;
    R[6] = (M[4] * M[9] - M[5] * M[8]) * Inverse;
    R[7] = (M[1] * M[8] - M[0] * M[9]) * Inverse;
    R[8] = (M[0] * M[5] - M[1] * M[4]) * Inverse;

    for (int Row = 0; Row < 3; ++Row)
    {
        for (int Column = 0; Column < 3; ++Column)
            Out[4 * Row + Column] = R[3 * Row + Column];

        Out[4 * Row + 3] = -(R[3 * Row] * M[3] + R[3 * Row + 1] * M[7] + R[3 * Row + 2] * M[11]);
    }

    return true;
}

// One attempt at labeling the grid, with slabs sized to MemoryShare of the
// GPU's memory unless buffers from earlier frames already set their size.
static int32_t RunOnce(
    FVoxelGpuContext* Context,
    const FVoxelGpuGrid* Grid,
    const FVoxelGpuInstance* Instances,
    uint32_t NumInstances,
    const FVoxelGpuShape* Shapes,
    uint32_t NumShapes,
    const FVoxelGpuAxis* Axes,
    uint32_t NumAxes,
    uint8_t* Labels,
    FVoxelGpuStats* Stats,
    double MemoryShare
)
{
    VOXEL_GPU_CHECK(cudaSetDevice(Context->Device));

    *Stats = FVoxelGpuStats{};

    Context->HostPairs.clear();

    const int32_t SizeX = Grid->Size[0];
    const int32_t SizeY = Grid->Size[1];
    const int32_t SizeZ = Grid->Size[2];

    if (SizeX <= 0 || SizeY <= 0 || SizeZ <= 0)
        return 0;

    // Instances, each with its mesh and its first triangle in the frame.
    std::vector<FInstanceGpu> Gpu;

    Gpu.reserve(NumInstances);

    uint64_t NumTriangles = 0;

    for (uint32_t Index = 0; Index < NumInstances; ++Index)
    {
        const FVoxelGpuInstance& Instance = Instances[Index];

        const auto Found = Context->Meshes.find(Instance.MeshKey);

        if (Found == Context->Meshes.end() || Found->second.NumTriangles == 0)
            continue;

        FMesh& Mesh = Found->second;

        Mesh.LastUsed = Context->RunCount;

        FInstanceGpu Entry;

        Entry.Vertices = Mesh.Vertices.Data;
        Entry.Indices = Mesh.Indices.Data;

        std::memcpy(Entry.ToGrid, Instance.ToGrid, sizeof(Entry.ToGrid));

        Entry.Skin = 0.0;
        Entry.Thickness = Instance.Thickness;
        Entry.Rows = Instance.Rows;
        Entry.Columns = Instance.Columns;

        if (Instance.Flags & VOXEL_GPU_HEIGHTFIELD)
        {
            if (!Invert(Instance.ToGrid, Entry.ToSamples))
            {
                Context->Error = "a heightfield's map to the grid cannot be inverted";

                return -1;
            }

            // Heights run along the map's second column.
            const double* M = Instance.ToGrid;

            Entry.Skin = std::fabs(Instance.Thickness) * std::sqrt(M[1] * M[1] + M[5] * M[5] + M[9] * M[9]);
        }

        Entry.FirstTriangle = NumTriangles;
        Entry.Key = Instance.Key;
        Entry.Flags = Instance.Flags;

        Gpu.push_back(Entry);

        NumTriangles += Mesh.NumTriangles;
    }

    Stats->Triangles = NumTriangles;

    if (!Gpu.empty())
    {
        VOXEL_GPU_CHECK(Context->Instances.Reserve(Gpu.size()));

        VOXEL_GPU_CHECK(cudaMemcpy(
            Context->Instances.Data,
            Gpu.data(),
            Gpu.size() * sizeof(FInstanceGpu),
            cudaMemcpyHostToDevice
        ));

        VOXEL_GPU_CHECK(Context->Prepared.Reserve(NumTriangles));
        VOXEL_GPU_CHECK(Context->TriangleAxes.Reserve(size_t(AxisValues) * NumTriangles));
        VOXEL_GPU_CHECK(Context->Columns.Reserve(NumTriangles));
        VOXEL_GPU_CHECK(Context->Offsets.Reserve(NumTriangles));

        size_t ScanBytes = 0;

        VOXEL_GPU_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            ScanBytes,
            Context->Columns.Data,
            Context->Offsets.Data,
            NumTriangles
        ));

        VOXEL_GPU_CHECK(Context->ScanTemp.Reserve(std::max<size_t>(ScanBytes, 1)));
    }

    if (NumAxes > 0)
    {
        VOXEL_GPU_CHECK(Context->Axes.Reserve(NumAxes));

        VOXEL_GPU_CHECK(cudaMemcpy(
            Context->Axes.Data,
            Axes,
            NumAxes * sizeof(FVoxelGpuAxis),
            cudaMemcpyHostToDevice
        ));
    }

    VOXEL_GPU_CHECK(Context->Shapes.Reserve(std::max<uint32_t>(NumShapes, 1)));

    // Slabs along X, at up to five bytes per voxel: four for the keys, the
    // rest for the bitmap, its counts and offsets, and the packed labels.
    // Buffers kept from earlier frames set the slab size, so a GPU whose free
    // memory changes from frame to frame, as one that also renders, is not
    // asked for new buffers every frame. Otherwise MemoryShare of the free
    // memory sets it.
    const int64_t PlaneVoxels = int64_t(SizeY) * SizeZ;
    const int64_t TotalVoxels = PlaneVoxels * SizeX;

    const int64_t Held = int64_t(Context->Keys.Capacity);

    int64_t Planes = 0;

    if (Held >= PlaneVoxels)
        Planes = std::min<int64_t>(SizeX, Held / PlaneVoxels);
    else
    {
        size_t FreeBytes = 0;
        size_t TotalBytes = 0;

        VOXEL_GPU_CHECK(cudaMemGetInfo(&FreeBytes, &TotalBytes));

        Planes = std::max<int64_t>(1, std::min<int64_t>(SizeX, int64_t(FreeBytes * MemoryShare / 5) / PlaneVoxels));
    }

    // VOXEL_GPU_SLAB_PLANES caps the slab size, to test the slabs on small
    // grids.
    if (const char* Cap = std::getenv("VOXEL_GPU_SLAB_PLANES"))
        Planes = std::max<int64_t>(1, std::min<int64_t>(Planes, std::atoll(Cap)));

    const int32_t SlabPlanes = int32_t(Planes);

    const int64_t SlabVoxels = PlaneVoxels * SlabPlanes;

    VOXEL_GPU_CHECK(Context->Keys.Reserve(size_t(std::min(SlabVoxels, TotalVoxels))));

    // A slab's bitmap may start up to 63 bits early and end in a part word.
    const size_t MaxWords = size_t(std::min(SlabVoxels, TotalVoxels) / 64 + 2);

    VOXEL_GPU_CHECK(Context->Bitmap.Reserve(MaxWords));
    VOXEL_GPU_CHECK(Context->Counts.Reserve(MaxWords));
    VOXEL_GPU_CHECK(Context->WordOffsets.Reserve(MaxWords));
    VOXEL_GPU_CHECK(Context->HostBitmap.Reserve(MaxWords));
    VOXEL_GPU_CHECK(Context->HostStarts.Reserve(MaxWords / ChunkWords + 1));

    Context->Lap();

    // Slab bounds are summed in 64 bits: two plane counts near the axis
    // limit overflow 32.
    for (int64_t First = 0; First < SizeX; First += SlabPlanes)
    {
        const int32_t SlabBegin = int32_t(First);
        const int32_t SlabEnd = int32_t(std::min<int64_t>(SizeX, First + SlabPlanes));
        const int64_t Voxels = PlaneVoxels * (SlabEnd - SlabBegin);

        ++Stats->Slabs;

        VOXEL_GPU_CHECK(cudaMemset(Context->Keys.Data, 0, size_t(Voxels) * sizeof(uint32_t)));

        VOXEL_GPU_CHECK(cudaMemset(Context->Counters.Data + 2, 0, sizeof(unsigned long long)));

        Stats->PrepareMs += Context->Lap();

        if (NumTriangles > 0)
        {
            PrepareTriangles<<<GetBlocks(NumTriangles), ThreadsPerBlock>>>(
                Context->Instances.Data,
                uint32_t(Gpu.size()),
                NumTriangles,
                SlabBegin,
                SlabEnd,
                SizeY,
                SizeZ,
                Context->Prepared.Data,
                Context->TriangleAxes.Data,
                Context->Columns.Data
            );

            VOXEL_GPU_CHECK(cudaGetLastError());

            size_t ScanBytes = Context->ScanTemp.Capacity;

            VOXEL_GPU_CHECK(cub::DeviceScan::ExclusiveSum(
                Context->ScanTemp.Data,
                ScanBytes,
                Context->Columns.Data,
                Context->Offsets.Data,
                NumTriangles
            ));

            uint64_t Last[2] = {0, 0};

            VOXEL_GPU_CHECK(cudaMemcpy(
                &Last[0],
                Context->Offsets.Data + NumTriangles - 1,
                sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            ));

            VOXEL_GPU_CHECK(cudaMemcpy(
                &Last[1],
                Context->Columns.Data + NumTriangles - 1,
                sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            ));

            const uint64_t NumColumns = Last[0] + Last[1];

            Stats->Columns += NumColumns;

            Stats->PrepareMs += Context->Lap();

            if (NumColumns > 0)
            {
                RasterizeColumns<<<GetBlocks(NumColumns), ThreadsPerBlock>>>(
                    Context->Prepared.Data,
                    Context->TriangleAxes.Data,
                    Context->Instances.Data,
                    Context->Offsets.Data,
                    NumTriangles,
                    NumColumns,
                    SlabBegin,
                    SizeY,
                    SizeZ,
                    Context->Keys.Data,
                    Context->Counters.Data + 2
                );

                VOXEL_GPU_CHECK(cudaGetLastError());
            }

            Stats->RasterMs += Context->Lap();

            unsigned long long ExactTests = 0;

            VOXEL_GPU_CHECK(cudaMemcpy(&ExactTests, Context->Counters.Data + 2, sizeof(ExactTests), cudaMemcpyDeviceToHost));

            Stats->ExactTests += ExactTests;
        }

        // Shapes, clipped to the slab.
        std::vector<FShapeGpu> SlabShapes;

        uint64_t ShapeVoxels = 0;

        for (uint32_t Index = 0; Index < NumShapes; ++Index)
        {
            FShapeGpu Entry;

            Entry.Shape = Shapes[Index];
            Entry.Shape.Begin[0] = std::max(Entry.Shape.Begin[0], SlabBegin);
            Entry.Shape.End[0] = std::min(Entry.Shape.End[0], SlabEnd);

            bool bEmpty = false;

            uint64_t Count = 1;

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                bEmpty = bEmpty || Entry.Shape.End[Axis] <= Entry.Shape.Begin[Axis];

                Count *= bEmpty ? 0 : uint64_t(Entry.Shape.End[Axis] - Entry.Shape.Begin[Axis]);
            }

            if (bEmpty)
                continue;

            Entry.FirstVoxel = ShapeVoxels;

            ShapeVoxels += Count;

            SlabShapes.push_back(Entry);
        }

        Stats->ShapeVoxels += ShapeVoxels;

        if (!SlabShapes.empty())
        {
            VOXEL_GPU_CHECK(Context->Shapes.Reserve(SlabShapes.size()));

            VOXEL_GPU_CHECK(cudaMemcpy(
                Context->Shapes.Data,
                SlabShapes.data(),
                SlabShapes.size() * sizeof(FShapeGpu),
                cudaMemcpyHostToDevice
            ));

            RasterizeShapes<<<GetBlocks(ShapeVoxels), ThreadsPerBlock>>>(
                Context->Shapes.Data,
                uint32_t(SlabShapes.size()),
                Context->Axes.Data,
                ShapeVoxels,
                SlabBegin,
                SizeY,
                SizeZ,
                Context->Keys.Data
            );

            VOXEL_GPU_CHECK(cudaGetLastError());
        }

        Stats->ShapesMs += Context->Lap();

        // The occupied voxels' bitmap and labels, then the voxels with an
        // instance.
        uint8_t* SlabLabels = Labels + int64_t(SlabBegin) * PlaneVoxels;

        const uint64_t Shift = uint64_t(reinterpret_cast<uintptr_t>(SlabLabels) % 64);
        const uint64_t NumWords = (Shift + uint64_t(Voxels) + 63) / 64;
        const uint64_t NumChunks = (NumWords + ChunkWords - 1) / ChunkWords;

        VOXEL_GPU_CHECK(cudaMemset(Context->Counters.Data, 0, 2 * sizeof(unsigned long long)));

        PackBitmap<<<GetBlocks(32 * NumWords), ThreadsPerBlock>>>(
            Context->Keys.Data,
            uint64_t(Voxels),
            Shift,
            NumWords,
            Grid->InstancedLabels,
            Context->Bitmap.Data,
            Context->Counts.Data,
            Context->Counters.Data
        );

        VOXEL_GPU_CHECK(cudaGetLastError());

        size_t ScanBytes = 0;

        VOXEL_GPU_CHECK(cub::DeviceScan::ExclusiveSum(
            nullptr,
            ScanBytes,
            Context->Counts.Data,
            Context->WordOffsets.Data,
            NumWords
        ));

        VOXEL_GPU_CHECK(Context->ScanTemp.Reserve(std::max<size_t>(ScanBytes, 1)));

        ScanBytes = Context->ScanTemp.Capacity;

        VOXEL_GPU_CHECK(cub::DeviceScan::ExclusiveSum(
            Context->ScanTemp.Data,
            ScanBytes,
            Context->Counts.Data,
            Context->WordOffsets.Data,
            NumWords
        ));

        uint64_t Last[2] = {0, 0};

        VOXEL_GPU_CHECK(cudaMemcpy(
            &Last[0],
            Context->WordOffsets.Data + NumWords - 1,
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost
        ));

        VOXEL_GPU_CHECK(cudaMemcpy(
            &Last[1],
            Context->Counts.Data + NumWords - 1,
            sizeof(uint64_t),
            cudaMemcpyDeviceToHost
        ));

        const uint64_t NumOccupied = Last[0] + Last[1];

        unsigned long long NumPairs = 0;

        VOXEL_GPU_CHECK(cudaMemcpy(&NumPairs, Context->Counters.Data, sizeof(NumPairs), cudaMemcpyDeviceToHost));

        VOXEL_GPU_CHECK(Context->Packed.Reserve(std::max<uint64_t>(NumOccupied, 1)));
        VOXEL_GPU_CHECK(Context->Pairs.Reserve(std::max<unsigned long long>(NumPairs, 1)));

        PackLabels<<<GetBlocks(32 * NumWords), ThreadsPerBlock>>>(
            Context->Keys.Data,
            uint64_t(Voxels),
            Shift,
            NumWords,
            uint64_t(SlabBegin) * uint64_t(PlaneVoxels),
            Grid->InstancedLabels,
            Context->WordOffsets.Data,
            Context->Packed.Data,
            Context->Pairs.Data,
            Context->Counters.Data + 1
        );

        VOXEL_GPU_CHECK(cudaGetLastError());

        Stats->ResolveMs += Context->Lap();

        const auto ReadStart = std::chrono::steady_clock::now();

        VOXEL_GPU_CHECK(Context->HostPacked.Reserve(std::max<uint64_t>(NumOccupied, 1)));

        VOXEL_GPU_CHECK(cudaMemcpy(
            Context->HostBitmap.Data,
            Context->Bitmap.Data,
            NumWords * sizeof(uint64_t),
            cudaMemcpyDeviceToHost
        ));

        VOXEL_GPU_CHECK(cudaMemcpy(
            Context->HostPacked.Data,
            Context->Packed.Data,
            NumOccupied,
            cudaMemcpyDeviceToHost
        ));

        // The offset of every chunk's first word.
        VOXEL_GPU_CHECK(cudaMemcpy2D(
            Context->HostStarts.Data,
            sizeof(uint64_t),
            Context->WordOffsets.Data,
            ChunkWords * sizeof(uint64_t),
            sizeof(uint64_t),
            NumChunks,
            cudaMemcpyDeviceToHost
        ));

        ExpandSlab(
            SlabLabels,
            uint64_t(Voxels),
            Shift,
            NumWords,
            Context->HostBitmap.Data,
            Context->HostPacked.Data,
            Context->HostStarts.Data
        );

        if (NumPairs > 0)
        {
            const size_t First = Context->HostPairs.size();

            Context->HostPairs.resize(First + NumPairs);

            VOXEL_GPU_CHECK(cudaMemcpy(
                Context->HostPairs.data() + First,
                Context->Pairs.Data,
                NumPairs * sizeof(FVoxelGpuPair),
                cudaMemcpyDeviceToHost
            ));
        }

        Stats->ReadbackMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ReadStart).count();

        Context->Lap();
    }

    Stats->Pairs = Context->HostPairs.size();

    return 0;
}

VOXEL_GPU_EXPORT int32_t VoxelGpuRun(
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
)
{
    ++Context->RunCount;

    // Errors left by earlier calls must not fail this run.
    cudaGetLastError();

    int32_t Status = RunOnce(Context, Grid, Instances, NumInstances, Shapes, NumShapes, Axes, NumAxes, Labels, Stats, 0.5);

    // A failure, most likely memory on a GPU that also renders, gets one
    // more try with the frame's buffers released and smaller slabs.
    if (Status != 0)
    {
        const std::string FirstError = Context->Error;

        Context->ReleaseScratch();

        cudaGetLastError();

        Status = RunOnce(Context, Grid, Instances, NumInstances, Shapes, NumShapes, Axes, NumAxes, Labels, Stats, 0.2);

        if (Status != 0)
            Context->Error = FirstError + "; then " + Context->Error;
    }

    if (Status != 0)
        return Status;

    // Meshes of shapes that left the scene.
    for (auto It = Context->Meshes.begin(); It != Context->Meshes.end();)
    {
        if (It->second.LastUsed + EvictAfterRuns < Context->RunCount)
        {
            It->second.Vertices.Free();
            It->second.Indices.Free();

            It = Context->Meshes.erase(It);
        }
        else
            ++It;
    }

    return 0;
}

VOXEL_GPU_EXPORT int32_t VoxelGpuGetPairs(FVoxelGpuContext* Context, const FVoxelGpuPair** Pairs, uint64_t* NumPairs)
{
    *Pairs = Context->HostPairs.data();
    *NumPairs = Context->HostPairs.size();

    return 0;
}
