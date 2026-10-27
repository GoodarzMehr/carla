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
        double InverseStep[3] = {0.0, 0.0, 0.0};
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
                bValid = bValid && std::isfinite(Pivot[Axis]) && std::isfinite(Step[Axis]) && std::isfinite(InverseStep[Axis]) && Step[Axis] != 0.0 && Size[Axis] > 0;
            }

            return bValid;
        }

        // Narrow [Begin, End) to the voxels whose approximate center along Axis lies in [Low, High].
        void Clip(int Axis, double Low, double High, int32_t& Begin, int32_t& End) const
        {
            double First = (Low - Pivot[Axis]) * InverseStep[Axis];
            double Last = (High - Pivot[Axis]) * InverseStep[Axis];

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

            // Rounding by truncation, only on values already inside the range: std::ceil and std::floor are library calls without SSE4.1.
            if (First > Begin)
            {
                if (First < End)
                {
                    const int32_t Truncated = int32_t(First);

                    Begin = Truncated < First ? Truncated + 1 : Truncated;
                }
                else
                {
                    Begin = End;
                }
            }

            if (Last + 1.0 < End)
            {
                if (Last + 1.0 > Begin)
                {
                    const int32_t Truncated = int32_t(Last);

                    End = (Truncated > Last ? Truncated - 1 : Truncated) + 1;
                }
                else
                {
                    End = Begin;
                }
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
            Lattice.InverseStep[Axis] = 1.0 / Step[Axis];
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
        uint64_t Skipped = 0;
        uint64_t Degenerate = 0;
        uint64_t Visited = 0;
        uint64_t Certain = 0;
        uint64_t Closed = 0;
        uint64_t Gated = 0;
        uint64_t Overlaps = 0;
        uint64_t Unconfirmed = 0;
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
            Skipped += Other.Skipped;
            Degenerate += Other.Degenerate;
            Visited += Other.Visited;
            Certain += Other.Certain;
            Closed += Other.Closed;
            Gated += Other.Gated;
            Overlaps += Other.Overlaps;
            Unconfirmed += Other.Unconfirmed;
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

    // Visit every voxel where VoxelTriangleOverlap::Test of a non-degenerate triangle may return anything but Outside.
    // Test returns Outside as soon as one axis separates by more than its error, so each subset of axes bounds the footprint:
    // the three box axes give an index box, the 2D edge axes clip rows across the dominant normal axis D, and the plane axis
    // gives the range along D. Visit(X, Y, Z, bCertain) also learns which voxels certainly give Overlap: those whose exact
    // center projects into the triangle along D and lies within the half extent, less more than the error band, of the
    // triangle along D. The triangle point straight along D then lies in the voxel box shrunk by that margin on every axis,
    // so every axis passes by more than its error. That point also lies in the candidate's inflated world bounds.
    template <typename FVisit>
    void ForEachTriangleVoxel(const VoxelTriangleOverlap::FTriangle& Triangle, const FLattice& Lattice, const physx::PxVec3& HalfExtent, FVisit&& Visit)
    {
        double Magnitude = 0.0;

        double MaxError = 0.0;

        for (physx::PxU32 Index = 0; Index < Triangle.NumAxes; ++Index)
        {
            const VoxelTriangleOverlap::FAxis& Axis = Triangle.Axes[Index];

            Magnitude = std::max(Magnitude, std::max(std::abs(double(Axis.Minimum)), std::abs(double(Axis.Maximum))) + Axis.Error);

            MaxError = std::max(MaxError, double(Axis.Error));
        }

        // Bounds rounding in VoxelTriangleOverlap::Test and in the projections recovered below.
        const double Slack = 8.0 * FLT_EPSILON * (Lattice.Magnitude + Magnitude);

        const double Pad = Lattice.Margin + Lattice.Padding + Slack;

        // Inner margin for certain voxels: approximate centers are within Lattice.Margin of the exact ones.
        const double Inset = Lattice.Margin + Slack;

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

        // The dominant component holds at least a third of the unit L1 normal, so this is finite.
        const double InverseNormal = 1.0 / Normal[D];

        // Range of the triangle's own projection on the normal, and how far along D a certain center may lie from the triangle.
        const double PlaneRadius = HalfExtent.x * std::abs(double(Normal.x)) + HalfExtent.y * std::abs(double(Normal.y)) + HalfExtent.z * std::abs(double(Normal.z));

        const double Depth = std::abs(double(Normal[D])) * (HalfExtent[D] - MaxError - 2.0 * Slack) - Inset;

        const double CertainPlaneLow = double(Plane.Maximum) - PlaneRadius - Depth;
        const double CertainPlaneHigh = double(Plane.Minimum) + PlaneRadius + Depth;

        // Edge axes perpendicular to D bound the triangle's projection onto the U-V plane.
        struct FEdge
        {
            double U;
            double InverseV;
            double Low;
            double High;
            double CertainLow;
            double CertainHigh;
        };

        FEdge Edges[9];

        int NumEdges = 0;

        for (physx::PxU32 Index = 4; Index < Triangle.NumAxes; ++Index)
        {
            const VoxelTriangleOverlap::FAxis& Axis = Triangle.Axes[Index];

            if (Axis.Direction[D] == 0.0f)
            {
                const double InverseV = Axis.Direction[V] == 0.0f ? 0.0 : 1.0 / Axis.Direction[V];

                const double Radius = HalfExtent[U] * std::abs(double(Axis.Direction[U])) + HalfExtent[V] * std::abs(double(Axis.Direction[V]));

                Edges[NumEdges++] = FEdge{
                    Axis.Direction[U],
                    InverseV,
                    double(Axis.Minimum) - Axis.Error - Pad,
                    double(Axis.Maximum) + Axis.Error + Pad,
                    double(Axis.Minimum) + Radius + Inset,
                    double(Axis.Maximum) - Radius - Inset
                };
            }
        }

        // The three edges of the projected triangle are always among these axes; without them nothing is certain.
        const bool bCertainPossible = Depth > 0.0 && NumEdges >= 3;

        // Narrow [Begin, End) along V to the row voxels whose edge projections lie in [Low, High] for every edge.
        const auto ClipRow = [&](double CenterU, bool bCertain, int32_t& Begin, int32_t& End)
        {
            for (int Edge = 0; Edge < NumEdges && Begin < End; ++Edge)
            {
                const double Offset = Edges[Edge].U * CenterU;

                const double EdgeLow = bCertain ? Edges[Edge].CertainLow : Edges[Edge].Low;
                const double EdgeHigh = bCertain ? Edges[Edge].CertainHigh : Edges[Edge].High;

                if (Edges[Edge].InverseV == 0.0)
                {
                    if (!(Offset >= EdgeLow && Offset <= EdgeHigh))
                    {
                        End = Begin;
                    }
                }
                else
                {
                    double Low = (EdgeLow - Offset) * Edges[Edge].InverseV;
                    double High = (EdgeHigh - Offset) * Edges[Edge].InverseV;

                    if (Edges[Edge].InverseV < 0.0)
                    {
                        std::swap(Low, High);
                    }

                    Lattice.Clip(V, Low, High, Begin, End);
                }
            }
        };

        // Narrow [Begin, End) along D to the column voxels whose normal projection lies in [Low, High].
        const auto ClipColumn = [&](double Offset, double Low, double High, int32_t& Begin, int32_t& End)
        {
            double First = (Low - Offset) * InverseNormal;
            double Last = (High - Offset) * InverseNormal;

            if (InverseNormal < 0.0)
            {
                std::swap(First, Last);
            }

            Lattice.Clip(D, First, Last, Begin, End);
        };

        int32_t Index[3];

        for (Index[U] = Box.Begin[U]; Index[U] < Box.End[U]; ++Index[U])
        {
            const double CenterU = Lattice.Center(U, Index[U]);

            int32_t BeginV = Box.Begin[V];
            int32_t EndV = Box.End[V];

            ClipRow(CenterU, false, BeginV, EndV);

            int32_t CertainBeginV = BeginV;
            int32_t CertainEndV = BeginV;

            if (bCertainPossible && BeginV < EndV)
            {
                CertainEndV = EndV;

                ClipRow(CenterU, true, CertainBeginV, CertainEndV);
            }

            for (Index[V] = BeginV; Index[V] < EndV; ++Index[V])
            {
                const double Offset = Normal[U] * CenterU + Normal[V] * Lattice.Center(V, Index[V]);

                int32_t BeginD = Box.Begin[D];
                int32_t EndD = Box.End[D];

                ClipColumn(Offset, PlaneLow, PlaneHigh, BeginD, EndD);

                int32_t CertainBeginD = BeginD;
                int32_t CertainEndD = BeginD;

                if (Index[V] >= CertainBeginV && Index[V] < CertainEndV)
                {
                    CertainEndD = EndD;

                    ClipColumn(Offset, CertainPlaneLow, CertainPlaneHigh, CertainBeginD, CertainEndD);
                }

                for (Index[D] = BeginD; Index[D] < EndD; ++Index[D])
                {
                    Visit(Index[0], Index[1], Index[2], Index[D] >= CertainBeginD && Index[D] < CertainEndD);
                }
            }
        }
    }

    // A heightfield as seen from the tile frame. PhysX reports a box as touching a heightfield when a box corner lies within Skin
    // under the surface along Up, when a box edge crosses the surface, or when a surface vertex lies inside the box. It has no
    // face test, so a surface that cuts a box between its edges and corners is not reported.
    struct FSurface
    {
        physx::PxVec3 Up = physx::PxVec3(0.0f, 0.0f, 1.0f);
        float Skin = 0.0f;
    };

    // The skin of a heightfield whose height axis is rotated into the tile frame by HeightFieldToTile. PhysX counts a layer of
    // |Thickness| below the surface for a negative thickness, as Unreal leaves it, and above it for a positive one, so Up points
    // away from that layer.
    inline FSurface MakeSurface(const physx::PxQuat& HeightFieldToTile, float Thickness)
    {
        FSurface Surface;

        Surface.Up = HeightFieldToTile.rotate(physx::PxVec3(0.0f, Thickness > 0.0f ? -1.0f : 1.0f, 0.0f));

        Surface.Skin = physx::PxAbs(Thickness);

        return Surface;
    }

    // One heightfield triangle in the tile frame, prepared for ConfirmsSurfaceOverlap. Margin bounds how far its vertices and the
    // voxel corners may be from what PhysX computes.
    struct FSurfaceTriangle
    {
        double Origin[3] = {0.0, 0.0, 0.0};
        double Vertices[3][3] = {};
        double Normal[3] = {0.0, 0.0, 0.0};
        double EdgeNormals[3][3] = {};
        double Up[3] = {0.0, 0.0, 1.0};
        double NormalUp = 0.0;
        double Skin = 0.0;
        double Margin = 0.0;
        bool bValid = false;

        FSurfaceTriangle(const physx::PxTriangle& Triangle, const FSurface& Surface, double InMargin)
            : Skin(Surface.Skin)
            , Margin(InMargin)
        {
            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    Vertices[Vertex][Axis] = Triangle.verts[Vertex][Axis];
                }
            }

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                Origin[Axis] = Vertices[0][Axis];
                Up[Axis] = Surface.Up[Axis];
            }

            double Edges[3][3];

            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    Edges[Vertex][Axis] = Vertices[(Vertex + 1) % 3][Axis] - Vertices[Vertex][Axis];
                }
            }

            Cross(Edges[0], Edges[1], Normal);

            const double NormalLength = std::sqrt(Dot(Normal, Normal));

            if (!(NormalLength > 0.0) || !std::isfinite(NormalLength))
            {
                return;
            }

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                Normal[Axis] /= NormalLength;
            }

            // In-plane normals of the edges, pointing inside, so a point's distance inside an edge is one dot product.
            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                Cross(Normal, Edges[Vertex], EdgeNormals[Vertex]);

                const double Length = std::sqrt(Dot(EdgeNormals[Vertex], EdgeNormals[Vertex]));

                if (!(Length > 0.0))
                {
                    return;
                }

                for (int Axis = 0; Axis < 3; ++Axis)
                {
                    EdgeNormals[Vertex][Axis] /= Length;
                }
            }

            NormalUp = Dot(Normal, Up);

            bValid = true;
        }

        static double Dot(const double* A, const double* B)
        {
            return A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
        }

        static void Cross(const double* A, const double* B, double* Out)
        {
            Out[0] = A[1] * B[2] - A[2] * B[1];
            Out[1] = A[2] * B[0] - A[0] * B[2];
            Out[2] = A[0] * B[1] - A[1] * B[0];
        }

        // A point of the triangle plane lies inside the triangle by more than Margin.
        bool IsInside(const double* Point) const
        {
            for (int Vertex = 0; Vertex < 3; ++Vertex)
            {
                const double Offset[3] = {Point[0] - Vertices[Vertex][0], Point[1] - Vertices[Vertex][1], Point[2] - Vertices[Vertex][2]};

                if (!(Dot(EdgeNormals[Vertex], Offset) > Margin))
                {
                    return false;
                }
            }

            return true;
        }
    };

    // True when one of PhysX's own rules reports the voxel box around Center as touching the heightfield through this triangle,
    // with more than Margin to spare: a box edge crossing the triangle, or a box corner within the skin under it.
    inline bool ConfirmsSurfaceOverlap(const FSurfaceTriangle& Triangle, const physx::PxVec3& Center, const physx::PxVec3& HalfExtent)
    {
        if (!Triangle.bValid)
        {
            return false;
        }

        const double Margin = Triangle.Margin;

        // Signed distance of every box corner to the triangle plane.
        double Corners[8][3];
        double Sides[8];

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            double Offset[3];

            for (int Axis = 0; Axis < 3; ++Axis)
            {
                Corners[Corner][Axis] = double(Center[Axis]) + ((Corner >> Axis) & 1 ? 1.0 : -1.0) * HalfExtent[Axis];

                Offset[Axis] = Corners[Corner][Axis] - Triangle.Origin[Axis];
            }

            Sides[Corner] = FSurfaceTriangle::Dot(Triangle.Normal, Offset);
        }

        // A box edge crosses the triangle. Each of the 12 edges joins two corners that differ along one axis.
        for (int Corner = 0; Corner < 8; ++Corner)
        {
            for (int Axis = 0; Axis < 3; ++Axis)
            {
                const int Other = Corner | (1 << Axis);

                const bool bCrosses = Other != Corner && ((Sides[Corner] > Margin && Sides[Other] < -Margin) || (Sides[Corner] < -Margin && Sides[Other] > Margin));

                if (!bCrosses)
                {
                    continue;
                }

                const double Fraction = Sides[Corner] / (Sides[Corner] - Sides[Other]);

                double Point[3];

                for (int Index = 0; Index < 3; ++Index)
                {
                    Point[Index] = Corners[Corner][Index] + (Corners[Other][Index] - Corners[Corner][Index]) * Fraction;
                }

                if (Triangle.IsInside(Point))
                {
                    return true;
                }
            }
        }

        // A box corner lies within the skin under the triangle, measured along Up as PhysX measures heights.
        if (std::abs(Triangle.NormalUp) < 1e-3)
        {
            return false;
        }

        for (int Corner = 0; Corner < 8; ++Corner)
        {
            const double Height = Sides[Corner] / Triangle.NormalUp;

            if (Height > -Margin || Height < -Triangle.Skin + Margin)
            {
                continue;
            }

            double Foot[3];

            for (int Index = 0; Index < 3; ++Index)
            {
                Foot[Index] = Corners[Corner][Index] - Height * Triangle.Up[Index];
            }

            if (Triangle.IsInside(Foot))
            {
                return true;
            }
        }

        return false;
    }

    // Test one triangle of the current candidate against its footprint. For a heightfield triangle, set in Surface, a voxel
    // the test does not rule out counts only when ConfirmsSurfaceOverlap holds; PhysX decides the others.
    template <bool bStats, typename FEnv>
    void RasterizeTriangle(const VoxelTriangleOverlap::FTriangle& Triangle, const physx::PxVec3& HalfExtent, const FLattice& Lattice, FEnv& Env, FRasterStats& Stats, const FSurfaceTriangle* Surface = nullptr)
    {
        if (bStats)
        {
            ++Stats.Triangles;
        }

        ForEachTriangleVoxel(Triangle, Lattice, HalfExtent, [&](int32_t X, int32_t Y, int32_t Z, bool bCertain)
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

            if (bCertain && !Surface)
            {
                if (bStats)
                {
                    ++Stats.Certain;
                }

                Env.Hit(X, Y, Z);

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

            if (Result == VoxelTriangleOverlap::EResult::Outside)
            {
                if (bStats)
                {
                    ++Stats.Outside;
                }
            }
            else if (Surface && !ConfirmsSurfaceOverlap(*Surface, Center, HalfExtent))
            {
                if (bStats)
                {
                    ++Stats.Unconfirmed;
                }

                Env.Defer(X, Y, Z);
            }
            else if (Surface || Result == VoxelTriangleOverlap::EResult::Overlap)
            {
                if (bStats)
                {
                    ++Stats.Overlaps;
                }

                Env.Hit(X, Y, Z);
            }
            else
            {
                if (bStats)
                {
                    ++Stats.Uncertain;
                }

                Env.Defer(X, Y, Z);
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

    // Index box around every voxel where a mesh triangle can matter: where VoxelTriangleOverlap::Test may return anything but Outside, or,
    // for a degenerate triangle, where RasterizeDegenerate defers. Cheaper than MakeTriangle, so closed triangles can be skipped first.
    inline FIndexBox GetTriangleBox(const physx::PxTriangle& Triangle, float PositionError, const physx::PxVec3& HalfExtent, const FLattice& Lattice)
    {
        FIndexBox Box = Lattice.Full();

        if (!Triangle.verts[0].isFinite() || !Triangle.verts[1].isFinite() || !Triangle.verts[2].isFinite() || !HalfExtent.isFinite() || !physx::PxIsFinite(PositionError))
        {
            return Box;
        }

        const double VertexMagnitude = std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[0]), std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[1]), VoxelTriangleOverlap::MaxAbs(Triangle.verts[2])));

        const double HalfMagnitude = VoxelTriangleOverlap::MaxAbs(HalfExtent);

        // At least twice the box-axis error of MakeTriangle and the degenerate band.
        const double Error = 2.0 * std::abs(PositionError) + 128.0 * FLT_EPSILON * (VertexMagnitude + HalfMagnitude + std::abs(PositionError));

        const double Pad = Lattice.Margin + Lattice.Padding + 8.0 * FLT_EPSILON * (Lattice.Magnitude + VertexMagnitude + HalfMagnitude + Error);

        for (int Axis = 0; Axis < 3; ++Axis)
        {
            const double Low = std::min(Triangle.verts[0][Axis], std::min(Triangle.verts[1][Axis], Triangle.verts[2][Axis]));
            const double High = std::max(Triangle.verts[0][Axis], std::max(Triangle.verts[1][Axis], Triangle.verts[2][Axis]));

            Lattice.Clip(Axis, Low - HalfExtent[Axis] - Error - Pad, High + HalfExtent[Axis] + Error + Pad, Box.Begin[Axis], Box.End[Axis]);
        }

        return Box;
    }

    // Rasterize one fetched triangle of the current candidate, unless every voxel it can matter for is already closed. Surface is
    // set for heightfield triangles: their error band grows by the skin along its up direction, so boxes PhysX counts through the
    // skin are never declared outside, and their overlaps need a box edge crossing.
    template <bool bStats, typename FEnv>
    void RasterizeMeshTriangle(const physx::PxTriangle& Triangle, float PositionError, const physx::PxVec3& HalfExtent, const FLattice& Lattice, FEnv& Env, FRasterStats& Stats, const FSurface* Surface = nullptr)
    {
        const float Band = Surface ? PositionError + Surface->Skin : PositionError;

        const FIndexBox Box = GetTriangleBox(Triangle, Band, HalfExtent, Lattice);

        bool bOpen = false;

        for (int32_t X = Box.Begin[0]; X < Box.End[0] && !bOpen; ++X)
        {
            for (int32_t Y = Box.Begin[1]; Y < Box.End[1] && !bOpen; ++Y)
            {
                for (int32_t Z = Box.Begin[2]; Z < Box.End[2] && !bOpen; ++Z)
                {
                    bOpen = Env.IsOpen(X, Y, Z);
                }
            }
        }

        if (!bOpen)
        {
            if (bStats)
            {
                ++Stats.Skipped;
            }

            return;
        }

        VoxelTriangleOverlap::FTriangle Prepared = VoxelTriangleOverlap::MakeTriangle(Triangle, HalfExtent, PositionError);

        if (Prepared.Degenerate)
        {
            RasterizeDegenerate<bStats>(Triangle, Band, HalfExtent, Lattice, Env, Stats);

            return;
        }

        if (!Surface)
        {
            RasterizeTriangle<bStats>(Prepared, HalfExtent, Lattice, Env, Stats);

            return;
        }

        // A corner up to Skin under the surface lies in the triangle swept Skin down along Up, whose projection on each axis reaches
        // Skin * (Direction . Up) further on one side only. Extending that side keeps such boxes from being declared outside. The
        // extra error covers rounding in Up and in the shifted bounds.
        for (physx::PxU32 Index = 0; Index < Prepared.NumAxes; ++Index)
        {
            VoxelTriangleOverlap::FAxis& Axis = Prepared.Axes[Index];

            const float Shift = Surface->Skin * Axis.Direction.dot(Surface->Up);

            if (Shift > 0.0f)
            {
                Axis.Minimum -= Shift;
            }
            else
            {
                Axis.Maximum -= Shift;
            }

            Axis.Error += 1e-4f * Surface->Skin + 4.0f * FLT_EPSILON * physx::PxMax(physx::PxAbs(Axis.Minimum), physx::PxAbs(Axis.Maximum));
        }

        const double VertexMagnitude = std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[0]), std::max(VoxelTriangleOverlap::MaxAbs(Triangle.verts[1]), VoxelTriangleOverlap::MaxAbs(Triangle.verts[2])));

        // The checks use exact voxel centers. As in the overlap test itself, the position error bounds how far our vertices and box
        // corners are from those PhysX computes; it grows with the world positions, so it also covers their rounding.
        const FSurfaceTriangle SurfaceTriangle(Triangle, *Surface, PositionError + 16.0 * FLT_EPSILON * (Lattice.Magnitude + VertexMagnitude + VoxelTriangleOverlap::MaxAbs(HalfExtent)));

        RasterizeTriangle<bStats>(Prepared, HalfExtent, Lattice, Env, Stats, &SurfaceTriangle);
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
