// Standalone check of libcarla_voxel_gpu against a brute-force reference that
// tests every voxel against every shape. Heightfields follow PhysX's rules,
// which test_voxel_gpu_heightfield.cpp checks against PhysX itself. From
// the repository root, after Util/BuildTools/BuildVoxelGpu.sh:
//   lib=Unreal/CarlaUE4/Plugins/Carla/CarlaDependencies/lib
//   g++-11 -std=c++17 -O2 Util/BuildTools/tests/test_voxel_gpu.cpp -L$lib -lcarla_voxel_gpu \
//     -Wl,-rpath,$PWD/$lib -o /tmp/test_voxel_gpu
//   VOXEL_GPU_SLAB_PLANES=7 /tmp/test_voxel_gpu

#include "../../../Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/VoxelGpuApi.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

using Vec = std::array<double, 3>;

static Vec Sub(const Vec& A, const Vec& B) { return {A[0] - B[0], A[1] - B[1], A[2] - B[2]}; }
static double Dot(const Vec& A, const Vec& B) { return A[0] * B[0] + A[1] * B[1] + A[2] * B[2]; }
static Vec Cross(const Vec& A, const Vec& B)
{
    return {A[1] * B[2] - A[2] * B[1], A[2] * B[0] - A[0] * B[2], A[0] * B[1] - A[1] * B[0]};
}

// Reference predicates, written separately from the kernels.
static bool RefTriangle(const Vec& C, const Vec* V)
{
    Vec P[3] = {Sub(V[0], C), Sub(V[1], C), Sub(V[2], C)};
    std::vector<Vec> Axes = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    Vec E[3] = {Sub(P[1], P[0]), Sub(P[2], P[1]), Sub(P[0], P[2])};
    Axes.push_back(Cross(E[0], E[1]));
    for (const Vec& Edge : E)
        for (int A = 0; A < 3; ++A)
        {
            Vec U{0, 0, 0};
            U[A] = 1;
            Axes.push_back(Cross(Edge, U));
        }
    for (const Vec& D : Axes)
    {
        const double R = 0.5 * (std::fabs(D[0]) + std::fabs(D[1]) + std::fabs(D[2]));
        const double A = Dot(D, P[0]), B = Dot(D, P[1]), Cc = Dot(D, P[2]);
        if (std::min({A, B, Cc}) > R || std::max({A, B, Cc}) < -R)
            return false;
    }
    return true;
}

static double BoxDistance2(const Vec& C, const Vec& P)
{
    double S = 0;
    for (int A = 0; A < 3; ++A)
    {
        const double O = std::max(0.0, std::fabs(P[A] - C[A]) - 0.5);
        S += O * O;
    }
    return S;
}

int main()
{
    std::mt19937 Random(1234);
    std::uniform_real_distribution<double> Unit(0.0, 1.0);

    const int32_t Size[3] = {40, 36, 30};
    const int64_t NumVoxels = int64_t(Size[0]) * Size[1] * Size[2];
    const uint32_t Instanced = (1u << 12) | (1u << 14);

    FVoxelGpuContext* Context = nullptr;
    char Error[256];
    if (VoxelGpuCreate(0, &Context, Error, sizeof(Error)) != 0)
    {
        std::printf("create failed: %s\n", Error);
        return 1;
    }

    struct FRefTriangle { Vec V[3]; uint32_t Key; };
    std::vector<FRefTriangle> Triangles;
    std::vector<FVoxelGpuInstance> Instances;

    const auto RandomKey = [&]()
    {
        const uint32_t Labels[] = {1, 2, 3, 9, 12, 14};
        const uint32_t Label = Labels[Random() % 6];
        const uint32_t Priority = 1 + Random() % 3;
        const uint32_t Instance = (Instanced >> Label) & 1 ? 1 + Random() % 5 : 0;
        return VOXEL_GPU_KEY(Priority, Label, Instance);
    };

    // Meshes in their own space, placed with random rotations, scales and
    // offsets: small, large, zero-area and grid-aligned triangles.
    for (int MeshIndex = 0; MeshIndex < 40; ++MeshIndex)
    {
        const int NumTriangles = 1 + Random() % 30;
        std::vector<float> Vertices;
        std::vector<uint32_t> Indices;
        const int Kind = MeshIndex % 5;
        for (int T = 0; T < NumTriangles; ++T)
        {
            const double Scale = Kind == 0 ? 0.6 : (Kind == 1 ? 12.0 : 3.0);
            Vec Base{Unit(Random) * 8 - 4, Unit(Random) * 8 - 4, Unit(Random) * 8 - 4};
            for (int K = 0; K < 3; ++K)
            {
                Vec P{Base[0] + (Unit(Random) - 0.5) * Scale, Base[1] + (Unit(Random) - 0.5) * Scale,
                      Base[2] + (Unit(Random) - 0.5) * Scale};
                if (Kind == 2 && K == 2)
                    P = {Base[0], Base[1], Base[2]};  // zero area: a segment
                if (Kind == 3)
                    P[2] = std::round(Base[2]);       // in a plane between voxel layers
                for (double X : P)
                    Vertices.push_back(float(X));
                Indices.push_back(uint32_t(3 * T + K));
            }
        }
        FVoxelGpuMesh Mesh{uint64_t(100 + MeshIndex), Vertices.data(), uint32_t(Vertices.size() / 3), Indices.data(),
                           uint32_t(NumTriangles)};
        VoxelGpuUploadMesh(Context, &Mesh);

        // A rotation about z (exact for the grid-aligned kind) or a random
        // one, and an offset inside the grid.
        FVoxelGpuInstance Instance{};
        Instance.MeshKey = Mesh.Key;
        const double Angle = Kind == 3 ? 0.0 : Unit(Random) * 6.283;
        const double Tilt = Kind == 3 ? 0.0 : Unit(Random) * 1.0;
        const double Cz = std::cos(Angle), Sz = std::sin(Angle), Cx = std::cos(Tilt), Sx = std::sin(Tilt);
        const double R[3][3] = {{Cz, -Sz * Cx, Sz * Sx}, {Sz, Cz * Cx, -Cz * Sx}, {0, Sx, Cx}};
        const double Offset[3] = {Unit(Random) * 44 - 2, Unit(Random) * 40 - 2, Kind == 3 ? 10.0 : Unit(Random) * 34 - 2};
        for (int Row = 0; Row < 3; ++Row)
        {
            for (int Col = 0; Col < 3; ++Col)
                Instance.ToGrid[4 * Row + Col] = R[Row][Col];
            Instance.ToGrid[4 * Row + 3] = Offset[Row];
        }
        Instance.Key = RandomKey();
        Instances.push_back(Instance);
        for (int T = 0; T < NumTriangles; ++T)
        {
            FRefTriangle Ref;
            for (int K = 0; K < 3; ++K)
            {
                const float* S = &Vertices[3 * (3 * T + K)];
                for (int Row = 0; Row < 3; ++Row)
                    Ref.V[K][Row] = Instance.ToGrid[4 * Row] * S[0] + Instance.ToGrid[4 * Row + 1] * S[1] +
                                    Instance.ToGrid[4 * Row + 2] * S[2] + Instance.ToGrid[4 * Row + 3];
            }
            Ref.Key = Instance.Key;
            Triangles.push_back(Ref);
        }
    }

    // Shapes: spheres, capsules, and boxes as hulls.
    std::vector<FVoxelGpuShape> Shapes;
    std::vector<FVoxelGpuAxis> Axes;
    struct FBox { Vec Center; Vec Axis[3]; Vec Half; };
    std::vector<FBox> Boxes;
    for (int S = 0; S < 18; ++S)
    {
        FVoxelGpuShape Shape{};
        Shape.Key = RandomKey();
        Vec C{Unit(Random) * 40, Unit(Random) * 36, Unit(Random) * 30};
        double Lo[3], Hi[3];
        if (S % 3 == 0)
        {
            Shape.Type = VOXEL_GPU_SPHERE;
            Shape.Radius = 0.3 + Unit(Random) * 4;
            for (int A = 0; A < 3; ++A)
            {
                Shape.Points[0][A] = C[A];
                Lo[A] = C[A] - Shape.Radius;
                Hi[A] = C[A] + Shape.Radius;
            }
        }
        else if (S % 3 == 1)
        {
            Shape.Type = VOXEL_GPU_CAPSULE;
            Shape.Radius = 0.3 + Unit(Random) * 2;
            for (int A = 0; A < 3; ++A)
            {
                Shape.Points[0][A] = C[A];
                Shape.Points[1][A] = C[A] + (Unit(Random) - 0.5) * 10;
                Lo[A] = std::min(Shape.Points[0][A], Shape.Points[1][A]) - Shape.Radius;
                Hi[A] = std::max(Shape.Points[0][A], Shape.Points[1][A]) + Shape.Radius;
            }
        }
        else
        {
            Shape.Type = VOXEL_GPU_HULL;
            FBox Box;
            Box.Center = C;
            const double Angle = Unit(Random) * 6.283;
            Box.Axis[0] = {std::cos(Angle), std::sin(Angle), 0};
            Box.Axis[1] = {-std::sin(Angle), std::cos(Angle), 0};
            Box.Axis[2] = {0, 0, 1};
            Box.Half = {0.2 + Unit(Random) * 4, 0.2 + Unit(Random) * 4, 0.2 + Unit(Random) * 4};
            std::vector<Vec> Corners;
            for (int K = 0; K < 8; ++K)
            {
                Vec P = C;
                for (int A = 0; A < 3; ++A)
                    for (int I = 0; I < 3; ++I)
                        P[I] += ((K >> A) & 1 ? 1 : -1) * Box.Half[A] * Box.Axis[A][I];
                Corners.push_back(P);
            }
            std::vector<Vec> Directions = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
            for (int A = 0; A < 3; ++A)
            {
                Directions.push_back(Box.Axis[A]);
                for (int U = 0; U < 3; ++U)
                {
                    Vec Unit3{0, 0, 0};
                    Unit3[U] = 1;
                    Directions.push_back(Cross(Box.Axis[A], Unit3));
                }
            }
            Shape.FirstAxis = uint32_t(Axes.size());
            for (const Vec& D : Directions)
            {
                if (D[0] == 0 && D[1] == 0 && D[2] == 0)
                    continue;
                FVoxelGpuAxis Axis{{D[0], D[1], D[2]}, 1e300, -1e300};
                for (const Vec& P : Corners)
                {
                    Axis.Low = std::min(Axis.Low, Dot(D, P));
                    Axis.High = std::max(Axis.High, Dot(D, P));
                }
                Axes.push_back(Axis);
            }
            Shape.NumAxes = uint32_t(Axes.size()) - Shape.FirstAxis;
            for (int A = 0; A < 3; ++A)
            {
                Lo[A] = 1e300;
                Hi[A] = -1e300;
                for (const Vec& P : Corners)
                {
                    Lo[A] = std::min(Lo[A], P[A]);
                    Hi[A] = std::max(Hi[A], P[A]);
                }
            }
        }
        for (int A = 0; A < 3; ++A)
        {
            Shape.Begin[A] = std::max(0, int(std::ceil(Lo[A])) - 1);
            Shape.End[A] = std::min(Size[A], int(std::floor(Hi[A])) + 1);
        }
        Shapes.push_back(Shape);
    }

    // GPU.
    std::vector<uint8_t> Labels(NumVoxels, 0xEE);
    FVoxelGpuGrid Grid{{Size[0], Size[1], Size[2]}, Instanced};
    FVoxelGpuStats Stats;
    if (VoxelGpuRun(Context, &Grid, Instances.data(), uint32_t(Instances.size()), Shapes.data(), uint32_t(Shapes.size()),
                    Axes.data(), uint32_t(Axes.size()), Labels.data(), &Stats) != 0)
    {
        std::printf("run failed: %s\n", VoxelGpuError(Context));
        return 1;
    }
    const FVoxelGpuPair* Pairs = nullptr;
    uint64_t NumPairs = 0;
    VoxelGpuGetPairs(Context, &Pairs, &NumPairs);
    std::map<uint64_t, uint64_t> GpuPairs;
    for (uint64_t I = 0; I < NumPairs; ++I)
        GpuPairs[Pairs[I].Index] = Pairs[I].Instance;

    // Reference.
    int64_t Mismatches = 0, Occupied = 0, PairMismatches = 0, RefPairs = 0;
    for (int X = 0; X < Size[0]; ++X)
        for (int Y = 0; Y < Size[1]; ++Y)
            for (int Z = 0; Z < Size[2]; ++Z)
            {
                const Vec C{X + 0.5, Y + 0.5, Z + 0.5};
                uint32_t Key = 0;
                for (const FRefTriangle& T : Triangles)
                    if (T.Key > Key && RefTriangle(C, T.V))
                        Key = T.Key;
                for (const FVoxelGpuShape& S : Shapes)
                {
                    if (S.Key <= Key)
                        continue;
                    bool bTouch = false;
                    if (S.Type == VOXEL_GPU_SPHERE)
                        bTouch = BoxDistance2(C, {S.Points[0][0], S.Points[0][1], S.Points[0][2]}) <= S.Radius * S.Radius;
                    else if (S.Type == VOXEL_GPU_CAPSULE)
                    {
                        double Best = 1e300;
                        for (int I = 0; I <= 20000; ++I)
                        {
                            const double T = I / 20000.0;
                            Vec P;
                            for (int A = 0; A < 3; ++A)
                                P[A] = S.Points[0][A] + T * (S.Points[1][A] - S.Points[0][A]);
                            Best = std::min(Best, BoxDistance2(C, P));
                        }
                        bTouch = Best <= S.Radius * S.Radius;
                    }
                    else
                    {
                        bTouch = true;
                        for (uint32_t I = 0; I < S.NumAxes && bTouch; ++I)
                        {
                            const FVoxelGpuAxis& A = Axes[S.FirstAxis + I];
                            const Vec D{A.Direction[0], A.Direction[1], A.Direction[2]};
                            const double R = 0.5 * (std::fabs(D[0]) + std::fabs(D[1]) + std::fabs(D[2]));
                            bTouch = !(Dot(D, C) + R < A.Low || Dot(D, C) - R > A.High);
                        }
                    }
                    if (bTouch)
                        Key = S.Key;
                }
                const uint8_t Label = Key ? uint8_t(31 - ((Key >> 16) & 31)) : 0;
                const int64_t Index = (int64_t(X) * Size[1] + Y) * Size[2] + Z;
                Occupied += Label != 0;
                if (Labels[Index] != Label)
                {
                    if (Mismatches < 10)
                        std::printf("  voxel (%d, %d, %d): gpu %d, reference %d\n", X, Y, Z, Labels[Index], Label);
                    ++Mismatches;
                }
                if (Label && ((Instanced >> Label) & 1))
                {
                    ++RefPairs;
                    const auto Found = GpuPairs.find(uint64_t(Index));
                    if (Found == GpuPairs.end() || Found->second != 0xFFFF - (Key & 0xFFFF))
                        ++PairMismatches;
                }
            }

    const bool bPass = Mismatches == 0 && PairMismatches == 0 && uint64_t(RefPairs) == NumPairs;

    std::printf("%s: %lld of %lld voxels differ (%lld occupied); pairs %llu gpu, %lld reference, %lld differ; "
                "triangles=%llu columns=%llu shape voxels=%llu slabs=%u\n",
                bPass ? "PASS" : "FAIL",
                (long long)Mismatches, (long long)NumVoxels, (long long)Occupied, (unsigned long long)NumPairs,
                (long long)RefPairs, (long long)PairMismatches, (unsigned long long)Stats.Triangles,
                (unsigned long long)Stats.Columns, (unsigned long long)Stats.ShapeVoxels, Stats.Slabs);
    VoxelGpuDestroy(Context);
    return bPass ? 0 : 1;
}
