// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB). This work is licensed under the terms of the MIT license.

#pragma once

// Triangle-major labeling of one tile of voxels. Each triangle visits only the voxels where
// VoxelTriangleOverlap::Test can return anything but Outside. Depends on PhysX math only, so the
// standalone oracle test can use it. The sensor supplies voxel centers, labels and PhysX queries
// through an environment object with these members:
//   bool IsOpen(X, Y, Z)                  The current candidate can still change the voxel.
//   bool GetCenter(X, Y, Z, Center)       Exact center in the tile frame; false if the bounds gate fails.
//   void Hit(X, Y, Z)                     The current candidate overlaps the voxel.
//   void Defer(X, Y, Z)                   Decide the voxel with Overlaps once all triangles are done.
//   bool Overlaps(X, Y, Z)                PhysX overlap of the voxel and the current candidate.
//   bool OverlapsBrick(Begin, End)        PhysX overlap of a brick's query bounds and the current candidate.

#include "VoxelTriangleOverlap.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <vector>

#include <foundation/PxBounds3.h>

namespace VoxelSurfaceRaster
{
    // Half-open range of tile voxel indices along each axis.
    struct FIndexBox
    {
        int32_t Begin[3];
        int32_t End[3];

        bool IsEmpty() const
        {
            return Begin[0] >= End[0] || Begin[1] >= End[1] || Begin[2] >= End[2];
        }
    };

    // Approximate voxel centers of one tile in the tile frame. The exact center of voxel I lies within Margin of Pivot + I * Step on every axis.
    struct FLattice
    {
        double Pivot[3] = {0.0, 0.0, 0.0};
        double Step[3] = {0.0, 0.0, 0.0};
        int32_t Size[3] = {0, 0, 0};
        double Margin = 0.0;

        // Largest absolute center coordinate, which bounds rounding in VoxelTriangleOverlap::Test.
        double Magnitude = 0.0;

        // Extra distance added to every footprint. Only used to check that footprints are complete.
        double Padding = 0.0;

        double Center(int Axis, int32_t Index) const
        {
            return Pivot[Axis] + Index * Step[Axis];
        }

        FIndexBox Full() const
        {
            return FIndexBox{{0, 0, 0}, {Size[0], Size[1], Size[2]}};
        }

        int32_t GetIndex(int32_t X, int32_t Y, int32_t Z) const
        {
            return (X * Size[1] + Y) * Size[2] + Z;
        }

        int32_t GetNumVoxels() const
        {
            return Size[0] * Size[1] * Size[2];
        }

        bool IsValid() const
        {
            bool bValid = std::isfinite(Margin) && std::isfinite(Magnitude) && std::isfinite(Padding);

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                bValid = bValid && std::isfinite(Pivot[Axis]) && std::isfinite(Step[Axis]) && Step[Axis] != 0.0 && Size[Axis] > 0;
            }

            return bValid;
        }

        // Narrow [Begin, End) to the voxels whose approximate center along Axis lies in [Low, High].
        void Clip(int Axis, double Low, double High, int32_t& Begin, int32_t& End) const
        {
            double First = (Low - Pivot[Axis]) / Step[Axis];
            double Last = (High - Pivot[Axis]) / Step[Axis];

            if (Step[Axis] < 0.0)
            {
                std::swap(First, Last);
            }

            // NaN bounds fail this comparison and leave the range empty.
            if (!(First <= Last))
            {
                End = Begin;

                return;
            }

            First = std::ceil(First);
            Last = std::floor(Last) + 1.0;

            if (First > Begin)
            {
                Begin = First < End ? int32_t(First) : End;
            }

            if (Last < End)
            {
                End = Last > Begin ? int32_t(Last) : Begin;
            }
        }
    };

    // Build the lattice of a tile. GetCenter(X, Y, Z) returns the exact center of a voxel in the tile frame and CenterError bounds its rounding.
    // The deviation measured at the eight tile corners covers the linear part of the error, since it is largest at a corner.
    template <typename FGetCenter>
    FLattice MakeLattice(const int32_t Size[3], const double Step[3], double CenterError, double Padding, FGetCenter&& GetCenter)
    {
        FLattice Lattice;

        const physx::PxVec3 Pivot = GetCenter(0, 0, 0);

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            Lattice.Pivot[Axis] = Pivot[Axis];
            Lattice.Step[Axis] = Step[Axis];
            Lattice.Size[Axis] = Size[Axis];
        }

        double CornerError = 0.0;

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            const int32_t Index[3] = {(Corner & 1) ? Size[0] - 1 : 0, (Corner & 2) ? Size[1] - 1 : 0, (Corner & 4) ? Size[2] - 1 : 0};

            const physx::PxVec3 Exact = GetCenter(Index[0], Index[1], Index[2]);

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                const double Deviation = std::abs(double(Exact[Axis]) - Lattice.Center(Axis, Index[Axis]));

                // Written so that NaN propagates and invalidates the lattice.
                if (!(Deviation <= CornerError))
                {
                    CornerError = Deviation;
                }
            }
        }

        // The pivot and every other center each carry up to CenterError, on top of the corner measurement.
        Lattice.Margin = CornerError + 4.0 * CenterError;

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const double Extent = std::max(std::abs(Lattice.Center(Axis, 0)), std::abs(Lattice.Center(Axis, Size[Axis] - 1)));

            Lattice.Magnitude = std::max(Lattice.Magnitude, Extent + Lattice.Margin);
        }

        Lattice.Padding = Padding;

        return Lattice;
    }

    // Label state of a voxel: the label in bits 0 to 4, and TieFlag once two different labels of the highest priority overlap it.
    constexpr uint8_t TieFlag = 0x80;

    // Candidates are applied in descending priority, so the label rule does not depend on their order within a priority.
    struct FLabelRules
    {
        uint8_t Rank[256];

        FLabelRules(const uint8_t* Priorities, int NumLabels)
        {
            for (int State = 0; State < 256; ++State)
            {
                const int Label = State & 31;

                const int Priority = Label < NumLabels ? Priorities[Label] : 0;

                Rank[State] = uint8_t(2 * Priority + (State >> 7));
            }
        }

        // The candidate cannot change the final label of a voxel that holds a higher priority, its own label or a tie.
        bool IsClosed(uint8_t State, uint8_t Label, uint8_t Priority) const
        {
            return Rank[State] > 2 * Priority || State == Label;
        }

        // Record an overlap of an open voxel. Returns true when this creates a tie.
        bool Apply(uint8_t& State, uint8_t Label) const
        {
            if (State == 0)
            {
                State = Label;

                return false;
            }

            State |= TieFlag;

            return true;
        }
    };

    // Set of tile voxels that remembers insertion order.
    class FVoxelSet
    {
    public:
        void Reset(int32_t NumVoxels)
        {
            Bits.assign((NumVoxels + 63) / 64, 0);

            List.clear();
        }

        void Add(int32_t Index)
        {
            uint64_t& Word = Bits[Index >> 6];

            const uint64_t Bit = uint64_t(1) << (Index & 63);

            if (!(Word & Bit))
            {
                Word |= Bit;

                List.push_back(Index);
            }
        }

        void Clear()
        {
            for (const int32_t Index : List)
            {
                Bits[Index >> 6] = 0;
            }

            List.clear();
        }

        const std::vector<int32_t>& GetItems() const
        {
            return List;
        }

    private:
        std::vector<uint64_t> Bits;
        std::vector<int32_t> List;
    };

    struct FRasterStats
    {
        uint64_t Triangles = 0;
        uint64_t Degenerate = 0;
        uint64_t Visited = 0;
        uint64_t Closed = 0;
        uint64_t Gated = 0;
        uint64_t Overlaps = 0;
        uint64_t Uncertain = 0;
        uint64_t Outside = 0;
        uint64_t DeferredQueries = 0;
        uint64_t DeferredHits = 0;
        uint64_t Bricks = 0;
        uint64_t CulledBricks = 0;
        uint64_t PrimitiveQueries = 0;
        uint64_t PrimitiveHits = 0;

        void Add(const FRasterStats& Other)
        {
            Triangles += Other.Triangles;
            Degenerate += Other.Degenerate;
            Visited += Other.Visited;
            Closed += Other.Closed;
            Gated += Other.Gated;
            Overlaps += Other.Overlaps;
            Uncertain += Other.Uncertain;
            Outside += Other.Outside;
            DeferredQueries += Other.DeferredQueries;
            DeferredHits += Other.DeferredHits;
            Bricks += Other.Bricks;
            CulledBricks += Other.CulledBricks;
            PrimitiveQueries += Other.PrimitiveQueries;
            PrimitiveHits += Other.PrimitiveHits;
        }
    };

    template <typename FVisit>
    void ForEachBoxVoxel(const FIndexBox& Box, FVisit&& Visit)
    {
        for (int32_t X = Box.Begin[0]; X < Box.End[0]; ++X)
        {
            for (int32_t Y = Box.Begin[1]; Y < Box.End[1]; ++Y)
            {
                for (int32_t Z = Box.Begin[2]; Z < Box.End[2]; ++Z)
                {
                    Visit(X, Y, Z);
                }
            }
        }
    }

    // Distance by which a triangle footprint grows to cover the lattice margin, rounding in VoxelTriangleOverlap::Test and padding.
    inline double GetTrianglePad(const VoxelTriangleOverlap::FTriangle& Triangle, const FLattice& Lattice)
    {
        double Magnitude = 0.0;

        for (physx::PxU32 Index = 0; Index < Triangle.NumAxes; ++Index)
        {
            const VoxelTriangleOverlap::FAxis& Axis = Triangle.Axes[Index];

            Magnitude = std::max(Magnitude, std::max(std::abs(double(Axis.Minimum)), std::abs(double(Axis.Maximum))) + Axis.Error);
        }

        return Lattice.Margin + Lattice.Padding + 8.0 * FLT_EPSILON * (Lattice.Magnitude + Magnitude);
    }

    // Visit every voxel where VoxelTriangleOverlap::Test of a non-degenerate triangle may return anything but Outside.
    // Test returns Outside as soon as one axis separates by more than its error, so each subset of axes bounds the footprint:
    // the three box axes give an index box, the 2D edge axes clip rows across the dominant normal axis, and the plane axis
    // gives the range along it.
    template <typename FVisit>
    void ForEachTriangleVoxel(const VoxelTriangleOverlap::FTriangle& Triangle, const FLattice& Lattice, FVisit&& Visit)
    {
        const double Pad = GetTrianglePad(Triangle, Lattice);

        FIndexBox Box = Lattice.Full();

        // The first three axes are the voxel box axes.
        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const VoxelTriangleOverlap::FAxis& BoxAxis = Triangle.Axes[Axis];

            Lattice.Clip(Axis, double(BoxAxis.Minimum) - BoxAxis.Error - Pad, double(BoxAxis.Maximum) + BoxAxis.Error + Pad, Box.Begin[Axis], Box.End[Axis]);

            if (Box.Begin[Axis] >= Box.End[Axis])
            {
                return;
            }
        }

        // The fourth axis is the triangle normal, normalized to unit L1 length.
        const VoxelTriangleOverlap::FAxis& Plane = Triangle.Axes[3];

        const physx::PxVec3 Normal = Plane.Direction;

        const int D = (std::abs(Normal.x) >= std::abs(Normal.y) && std::abs(Normal.x) >= std::abs(Normal.z)) ? 0 : (std::abs(Normal.y) >= std::abs(Normal.z) ? 1 : 2);
        const int U = (D + 1) % 3;
        const int V = (D + 2) % 3;

        const double PlaneLow = double(Plane.Minimum) - Plane.Error - Pad;
        const double PlaneHigh = double(Plane.Maximum) + Plane.Error + Pad;

        // Edge axes perpendicular to D bound the triangle's projection onto the U-V plane.
        struct FEdge
        {
            double U;
            double V;
            double Low;
            double High;
        };

        FEdge Edges[9];

        int NumEdges = 0;

        for (physx::PxU32 Index = 4; Index < Triangle.NumAxes; ++Index)
        {
            const VoxelTriangleOverlap::FAxis& Axis = Triangle.Axes[Index];

            if (Axis.Direction[D] == 0.0f)
            {
                Edges[NumEdges++] = FEdge{Axis.Direction[U], Axis.Direction[V], double(Axis.Minimum) - Axis.Error - Pad, double(Axis.Maximum) + Axis.Error + Pad};
            }
        }

        int32_t Index[3];

        for (Index[U] = Box.Begin[U]; Index[U] < Box.End[U]; ++Index[U])
        {
            const double CenterU = Lattice.Center(U, Index[U]);

            int32_t BeginV = Box.Begin[V];
            int32_t EndV = Box.End[V];

            for (int Edge = 0; Edge < NumEdges && BeginV < EndV; ++Edge)
            {
                const double Offset = Edges[Edge].U * CenterU;

                if (Edges[Edge].V == 0.0)
                {
                    if (!(Offset >= Edges[Edge].Low && Offset <= Edges[Edge].High))
                    {
                        EndV = BeginV;
                    }
                }
                else
                {
                    double Low = (Edges[Edge].Low - Offset) / Edges[Edge].V;
                    double High = (Edges[Edge].High - Offset) / Edges[Edge].V;

                    if (Edges[Edge].V < 0.0)
                    {
                        std::swap(Low, High);
                    }

                    Lattice.Clip(V, Low, High, BeginV, EndV);
                }
            }

            for (Index[V] = BeginV; Index[V] < EndV; ++Index[V])
            {
                const double Offset = Normal[U] * CenterU + Normal[V] * Lattice.Center(V, Index[V]);

                double Low = (PlaneLow - Offset) / Normal[D];
                double High = (PlaneHigh - Offset) / Normal[D];

                if (Normal[D] < 0.0f)
                {
                    std::swap(Low, High);
                }

                int32_t BeginD = Box.Begin[D];
                int32_t EndD = Box.End[D];

                Lattice.Clip(D, Low, High, BeginD, EndD);

                for (Index[D] = BeginD; Index[D] < EndD; ++Index[D])
                {
                    Visit(Index[0], Index[1], Index[2]);
                }
            }
        }
    }

    // Test one triangle of the current candidate against its footprint.
    template <bool bStats, typename FEnv>
    void RasterizeTriangle(const VoxelTriangleOverlap::FTriangle& Triangle, const FLattice& Lattice, FEnv& Env, FRasterStats& Stats)
    {
        if (bStats)
        {
            ++Stats.Triangles;
        }

        ForEachTriangleVoxel(Triangle, Lattice, [&](int32_t X, int32_t Y, int32_t Z)
        {
            if (bStats)
            {
                ++Stats.Visited;
            }

            if (!Env.IsOpen(X, Y, Z))
            {
                if (bStats)
                {
                    ++Stats.Closed;
                }

                return;
            }

            physx::PxVec3 Center;

            if (!Env.GetCenter(X, Y, Z, Center))
            {
                if (bStats)
                {
                    ++Stats.Gated;
                }

                return;
            }

            const VoxelTriangleOverlap::EResult Result = VoxelTriangleOverlap::Test(Triangle, Center);

            if (Result == VoxelTriangleOverlap::EResult::Overlap)
            {
                if (bStats)
                {
                    ++Stats.Overlaps;
                }

                Env.Hit(X, Y, Z);
            }
            else if (Result == VoxelTriangleOverlap::EResult::Uncertain)
            {
                if (bStats)
                {
                    ++Stats.Uncertain;
                }

                Env.Defer(X, Y, Z);
            }
            else if (bStats)
            {
                ++Stats.Outside;
            }
        });
    }

    // A degenerate triangle cannot be tested, so every voxel it may touch is decided by PhysX. Its vertices lie within
    // PositionError of the PhysX mesh-space vertices; non-finite input defers the whole tile.
    template <bool bStats, typename FEnv>
    void RasterizeDegenerate(const physx::PxTriangle& Triangle, float PositionError, const physx::PxVec3& HalfExtent, const FLattice& Lattice, FEnv& Env, FRasterStats& Stats)
    {
        FIndexBox Box = Lattice.Full();

        const bool bFinite = Triangle.verts[0].isFinite() && Triangle.verts[1].isFinite() && Triangle.verts[2].isFinite() && HalfExtent.isFinite() && physx::PxIsFinite(PositionError);

        if (bFinite)
        {
            const double VertexMagnitude = std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[0]), std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[1]), VoxelTriangleOverlap::MaxAbs(Triangle.verts[2])));

            const double Band = 2.0 * std::abs(PositionError) + Lattice.Margin + Lattice.Padding + 8.0 * FLT_EPSILON * (Lattice.Magnitude + VertexMagnitude + VoxelTriangleOverlap::MaxAbs(HalfExtent));

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                const double Low = std::min(Triangle.verts[0][Axis], std::min(Triangle.verts[1][Axis], Triangle.verts[2][Axis]));
                const double High = std::max(Triangle.verts[0][Axis], std::max(Triangle.verts[1][Axis], Triangle.verts[2][Axis]));

                Lattice.Clip(Axis, Low - HalfExtent[Axis] - Band, High + HalfExtent[Axis] + Band, Box.Begin[Axis], Box.End[Axis]);
            }
        }

        if (bStats)
        {
            ++Stats.Degenerate;
        }

        ForEachBoxVoxel(Box, [&](int32_t X, int32_t Y, int32_t Z)
        {
            if (bStats)
            {
                ++Stats.Visited;
            }

            physx::PxVec3 Center;

            if (Env.IsOpen(X, Y, Z) && Env.GetCenter(X, Y, Z, Center))
            {
                Env.Defer(X, Y, Z);
            }
        });
    }

    // Decide the deferred voxels of the current candidate that are still open with PhysX, then empty the set.
    template <bool bStats, typename FEnv>
    void ResolveDeferred(FVoxelSet& Deferred, const FLattice& Lattice, FEnv& Env, FRasterStats& Stats)
    {
        for (const int32_t Index : Deferred.GetItems())
        {
            const int32_t X = Index / (Lattice.Size[1] * Lattice.Size[2]);
            const int32_t Y = Index / Lattice.Size[2] % Lattice.Size[1];
            const int32_t Z = Index % Lattice.Size[2];

            if (!Env.IsOpen(X, Y, Z))
            {
                continue;
            }

            if (bStats)
            {
                ++Stats.DeferredQueries;
            }

            if (Env.Overlaps(X, Y, Z))
            {
                if (bStats)
                {
                    ++Stats.DeferredHits;
                }

                Env.Hit(X, Y, Z);
            }
        }

        Deferred.Clear();
    }

    // Tile voxels whose bounds may pass the gate of a candidate with world bounds Bounds, for voxels with world extent VoxelExtent.
    inline FIndexBox GetBoundsBox(const physx::PxBounds3& Bounds, const physx::PxVec3& VoxelExtent, const physx::PxTransform& TileFrame, const FLattice& Lattice)
    {
        FIndexBox Box = Lattice.Full();

        // PxBounds3::intersects accepts NaN bounds, so non-finite bounds keep the whole tile.
        if (!Bounds.minimum.isFinite() || !Bounds.maximum.isFinite())
        {
            return Box;
        }

        const physx::PxVec3 Minimum = Bounds.minimum - VoxelExtent;
        const physx::PxVec3 Maximum = Bounds.maximum + VoxelExtent;

        double Low[3] = {DBL_MAX, DBL_MAX, DBL_MAX};
        double High[3] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            const physx::PxVec3 World((Corner & 1) ? Maximum.x : Minimum.x, (Corner & 2) ? Maximum.y : Minimum.y, (Corner & 4) ? Maximum.z : Minimum.z);

            const physx::PxVec3 Local = TileFrame.transformInv(World);

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                Low[Axis] = std::min(Low[Axis], double(Local[Axis]));
                High[Axis] = std::max(High[Axis], double(Local[Axis]));
            }
        }

        const double Magnitude = std::max(VoxelTriangleOverlap::MaxAbs(Minimum), VoxelTriangleOverlap::MaxAbs(Maximum)) + VoxelTriangleOverlap::MaxAbs(TileFrame.p);

        const double Pad = Lattice.Margin + Lattice.Padding + 8.0 * FLT_EPSILON * (Lattice.Magnitude + Magnitude);

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            // Inverted bounds never pass the gate and give an empty range here.
            Lattice.Clip(Axis, Low[Axis] - Pad, High[Axis] + Pad, Box.Begin[Axis], Box.End[Axis]);
        }

        return Box;
    }

    // Decide the voxels of Box for a candidate without triangles. Bricks are aligned to multiples of BrickSize in tile indices;
    // each brick with open voxels is culled as a whole before its voxels are queried one by one.
    template <bool bStats, typename FEnv>
    void RasterizeBounds(const FIndexBox& Box, const int32_t BrickSize[3], const FLattice& Lattice, FEnv& Env, FRasterStats& Stats)
    {
        if (Box.IsEmpty())
        {
            return;
        }

        int32_t First[3];

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            First[Axis] = Box.Begin[Axis] / BrickSize[Axis] * BrickSize[Axis];
        }

        int32_t Brick[3];

        for (Brick[0] = First[0]; Brick[0] < Box.End[0]; Brick[0] += BrickSize[0])
        {
            for (Brick[1] = First[1]; Brick[1] < Box.End[1]; Brick[1] += BrickSize[1])
            {
                for (Brick[2] = First[2]; Brick[2] < Box.End[2]; Brick[2] += BrickSize[2])
                {
                    FIndexBox Voxels;

                    int32_t BrickEnd[3];

                    for (int Axis = 0; Axis < 3; ++Axis)
                    {
                        BrickEnd[Axis] = std::min(Brick[Axis] + BrickSize[Axis], Lattice.Size[Axis]);

                        Voxels.Begin[Axis] = std::max(Brick[Axis], Box.Begin[Axis]);
                        Voxels.End[Axis] = std::min(BrickEnd[Axis], Box.End[Axis]);
                    }

                    bool bOpen = false;

                    ForEachBoxVoxel(Voxels, [&](int32_t X, int32_t Y, int32_t Z)
                    {
                        bOpen = bOpen || Env.IsOpen(X, Y, Z);
                    });

                    if (!bOpen)
                    {
                        continue;
                    }

                    if (bStats)
                    {
                        ++Stats.Bricks;
                    }

                    if (!Env.OverlapsBrick(Brick, BrickEnd))
                    {
                        if (bStats)
                        {
                            ++Stats.CulledBricks;
                        }

                        continue;
                    }

                    ForEachBoxVoxel(Voxels, [&](int32_t X, int32_t Y, int32_t Z)
                    {
                        physx::PxVec3 Center;

                        if (!Env.IsOpen(X, Y, Z) || !Env.GetCenter(X, Y, Z, Center))
                        {
                            return;
                        }

                        if (bStats)
                        {
                            ++Stats.PrimitiveQueries;
                        }

                        if (Env.Overlaps(X, Y, Z))
                        {
                            if (bStats)
                            {
                                ++Stats.PrimitiveHits;
                            }

                            Env.Hit(X, Y, Z);
                        }
                    });
                }
            }
        }
    }
}
