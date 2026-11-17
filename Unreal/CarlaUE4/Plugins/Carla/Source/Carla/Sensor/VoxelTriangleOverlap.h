#pragma once

#include <cfloat>

#include <emmintrin.h>

#include <foundation/PxMath.h>
#include <foundation/PxTransform.h>

#include <geometry/PxTriangle.h>
#include <geometry/PxMeshQuery.h>
#include <geometry/PxHeightFieldGeometry.h>
#include <geometry/PxTriangleMeshGeometry.h>

namespace VoxelTriangleOverlap
{
    enum class EResult
    {
        Outside,
        Overlap,
        Uncertain
    };

    // Separating axes of one triangle and a voxel box, stored axis by axis in
    // lanes of four so SSE2 tests them together. Each axis has a direction of
    // unit L1 length, the range of box centers that may touch the triangle
    // along it, and the error of that range. Lanes from NumAxes on are
    // padding that passes every test.
    struct FTriangle
    {
        enum
        {
            NumLanes = 16
        };

        alignas(16) float DirectionX[NumLanes];
        alignas(16) float DirectionY[NumLanes];
        alignas(16) float DirectionZ[NumLanes];
        alignas(16) float Minimum[NumLanes];
        alignas(16) float Maximum[NumLanes];
        alignas(16) float Error[NumLanes];

        physx::PxU32 NumAxes = 0;

        bool Degenerate = false;

        physx::PxVec3 GetDirection(physx::PxU32 Axis) const
        {
            return physx::PxVec3(DirectionX[Axis], DirectionY[Axis], DirectionZ[Axis]);
        }
    };

    inline float MaxAbs(const physx::PxVec3& Value)
    {
        return physx::PxMax(physx::PxAbs(Value.x), physx::PxMax(physx::PxAbs(Value.y), physx::PxAbs(Value.z)));
    }

    inline __m128 Abs(__m128 Value)
    {
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), Value);
    }

    // Bit mask of the lanes that hold a finite value.
    inline int GetFiniteMask(__m128 Value)
    {
        return _mm_movemask_ps(_mm_cmple_ps(Abs(Value), _mm_set1_ps(FLT_MAX)));
    }

    // Build the axes of a triangle: the three box axes, the triangle normal
    // and the nine cross products of an edge and a box axis, less those that
    // vanish. A triangle that cannot be tested comes back Degenerate.
    inline FTriangle MakeTriangle(
        const physx::PxTriangle& Triangle,
        const physx::PxVec3& HalfExtent,
        float PositionError
    )
    {
        FTriangle Result;

        const physx::PxVec3* Vertices = Triangle.verts;
        const physx::PxVec3 Edges[] = {
            Vertices[1] - Vertices[0],
            Vertices[2] - Vertices[1],
            Vertices[0] - Vertices[2]
        };
        const physx::PxVec3 Normal = Edges[0].cross(Edges[1]);

        const float EdgeScale = physx::PxMax(MaxAbs(Edges[0]), physx::PxMax(MaxAbs(Edges[1]), MaxAbs(Edges[2])));

        if (
            !physx::PxIsFinite(PositionError) ||
            PositionError < 0.0f ||
            !HalfExtent.isFinite() ||
            !Vertices[0].isFinite() ||
            !Vertices[1].isFinite() ||
            !Vertices[2].isFinite() ||
            !Normal.isFinite() ||
            MaxAbs(Normal) <= 16.0f * FLT_EPSILON * EdgeScale * EdgeScale
        )
        {
            Result.Degenerate = true;

            return Result;
        }

        constexpr int NumAxes = 13;

        // Directions before normalization. The last three lanes are padding
        // and only need a length.
        alignas(16) float RawX[FTriangle::NumLanes];
        alignas(16) float RawY[FTriangle::NumLanes];
        alignas(16) float RawZ[FTriangle::NumLanes];

        for (int Lane = 0; Lane < FTriangle::NumLanes; ++Lane)
        {
            RawX[Lane] = Lane % 3 == 0 ? 1.0f : 0.0f;
            RawY[Lane] = Lane % 3 == 1 ? 1.0f : 0.0f;
            RawZ[Lane] = Lane % 3 == 2 ? 1.0f : 0.0f;
        }

        RawX[3] = Normal.x;
        RawY[3] = Normal.y;
        RawZ[3] = Normal.z;

        for (int Edge = 0; Edge < 3; ++Edge)
        {
            const physx::PxVec3& Vector = Edges[Edge];

            const int Lane = 4 + 3 * Edge;

            RawX[Lane] = 0.0f;
            RawY[Lane] = Vector.z;
            RawZ[Lane] = -Vector.y;

            RawX[Lane + 1] = -Vector.z;
            RawY[Lane + 1] = 0.0f;
            RawZ[Lane + 1] = Vector.x;

            RawX[Lane + 2] = Vector.y;
            RawY[Lane + 2] = -Vector.x;
            RawZ[Lane + 2] = 0.0f;
        }

        const __m128 One = _mm_set1_ps(1.0f);
        const __m128 HalfX = _mm_set1_ps(HalfExtent.x);
        const __m128 HalfY = _mm_set1_ps(HalfExtent.y);
        const __m128 HalfZ = _mm_set1_ps(HalfExtent.z);
        const __m128 Position = _mm_set1_ps(PositionError);
        const __m128 Roundoff = _mm_set1_ps(64.0f * FLT_EPSILON);

        // The projection of each vertex on the four axes of a group.
        const auto Project = [&](const physx::PxVec3& Vertex, __m128 X, __m128 Y, __m128 Z)
        {
            return _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(X, _mm_set1_ps(Vertex.x)), _mm_mul_ps(Y, _mm_set1_ps(Vertex.y))),
                _mm_mul_ps(Z, _mm_set1_ps(Vertex.z))
            );
        };

        int ZeroMask = 0;
        int FiniteMask = 0;

        for (int Lane = 0; Lane < FTriangle::NumLanes; Lane += 4)
        {
            const __m128 RawDirectionX = _mm_load_ps(RawX + Lane);
            const __m128 RawDirectionY = _mm_load_ps(RawY + Lane);
            const __m128 RawDirectionZ = _mm_load_ps(RawZ + Lane);

            const __m128 Length = _mm_add_ps(_mm_add_ps(Abs(RawDirectionX), Abs(RawDirectionY)), Abs(RawDirectionZ));

            const __m128 Inverse = _mm_div_ps(One, Length);

            const __m128 X = _mm_mul_ps(RawDirectionX, Inverse);
            const __m128 Y = _mm_mul_ps(RawDirectionY, Inverse);
            const __m128 Z = _mm_mul_ps(RawDirectionZ, Inverse);

            const __m128 P0 = Project(Vertices[0], X, Y, Z);
            const __m128 P1 = Project(Vertices[1], X, Y, Z);
            const __m128 P2 = Project(Vertices[2], X, Y, Z);

            const __m128 Radius = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(HalfX, Abs(X)), _mm_mul_ps(HalfY, Abs(Y))),
                _mm_mul_ps(HalfZ, Abs(Z))
            );

            const __m128 Minimum = _mm_sub_ps(_mm_min_ps(P0, _mm_min_ps(P1, P2)), Radius);
            const __m128 Maximum = _mm_add_ps(_mm_max_ps(P0, _mm_max_ps(P1, P2)), Radius);

            const __m128 Magnitude = _mm_max_ps(Abs(Minimum), Abs(Maximum));

            // Bound projection roundoff at both interval boundaries once per
            // triangle.
            const __m128 Error = _mm_add_ps(Position, _mm_mul_ps(Roundoff, _mm_add_ps(Magnitude, Position)));

            _mm_store_ps(Result.DirectionX + Lane, X);
            _mm_store_ps(Result.DirectionY + Lane, Y);
            _mm_store_ps(Result.DirectionZ + Lane, Z);
            _mm_store_ps(Result.Minimum + Lane, Minimum);
            _mm_store_ps(Result.Maximum + Lane, Maximum);
            _mm_store_ps(Result.Error + Lane, Error);

            ZeroMask |= _mm_movemask_ps(_mm_cmpeq_ps(Length, _mm_setzero_ps())) << Lane;

            FiniteMask |= (
                GetFiniteMask(Length) &
                GetFiniteMask(X) &
                GetFiniteMask(Y) &
                GetFiniteMask(Z) &
                GetFiniteMask(Minimum) &
                GetFiniteMask(Maximum) &
                GetFiniteMask(Error)
            ) << Lane;
        }

        constexpr int AxisMask = (1 << NumAxes) - 1;

        ZeroMask &= AxisMask;

        // An axis of zero length does not separate anything and is dropped;
        // any other axis must be finite.
        if ((~FiniteMask & ~ZeroMask & AxisMask) != 0)
        {
            Result.Degenerate = true;

            return Result;
        }

        Result.NumAxes = NumAxes;

        if (ZeroMask != 0)
        {
            Result.NumAxes = 0;

            for (int Lane = 0; Lane < NumAxes; ++Lane)
            {
                if (ZeroMask & (1 << Lane))
                    continue;

                const physx::PxU32 Axis = Result.NumAxes++;

                Result.DirectionX[Axis] = Result.DirectionX[Lane];
                Result.DirectionY[Axis] = Result.DirectionY[Lane];
                Result.DirectionZ[Axis] = Result.DirectionZ[Lane];
                Result.Minimum[Axis] = Result.Minimum[Lane];
                Result.Maximum[Axis] = Result.Maximum[Lane];
                Result.Error[Axis] = Result.Error[Lane];
            }
        }

        for (int Lane = int(Result.NumAxes); Lane < FTriangle::NumLanes; ++Lane)
        {
            Result.DirectionX[Lane] = 0.0f;
            Result.DirectionY[Lane] = 0.0f;
            Result.DirectionZ[Lane] = 0.0f;
            Result.Minimum[Lane] = -FLT_MAX;
            Result.Maximum[Lane] = FLT_MAX;
            Result.Error[Lane] = 0.0f;
        }

        return Result;
    }

    // Fetch a triangle of a mesh or heightfield in the cache frame.
    // MeshToCache must equal CachePose.transformInv(MeshPose). Returns the
    // position error of the vertices.
    template <typename FGeometry>
    inline float FetchTriangle(
        const FGeometry& Geometry,
        const physx::PxTransform& MeshPose,
        const physx::PxTransform& CachePose,
        const physx::PxTransform& MeshToCache,
        physx::PxU32 Index,
        const physx::PxVec3& HalfExtent,
        physx::PxTriangle& Triangle
    )
    {
        physx::PxMeshQuery::getTriangle(Geometry, MeshToCache, Index, Triangle);

        const float Magnitude = 1.0f + MaxAbs(MeshPose.p) + MaxAbs(CachePose.p) +
            physx::PxMax(MaxAbs(Triangle.verts[0]), physx::PxMax(MaxAbs(Triangle.verts[1]), MaxAbs(Triangle.verts[2]))) +
            MaxAbs(HalfExtent);

        // Cached and PhysX mesh-space transforms round differently. Resolve
        // contacts within this error band with the original whole-mesh query.
        return 64.0f * FLT_EPSILON * Magnitude;
    }

    // Distance of a box center outside the range of four axes, negative
    // inside. PartXY is the projection of the center's x and y.
    inline __m128 GetDistance(const FTriangle& Triangle, int Lane, __m128 PartXY, __m128 Z)
    {
        const __m128 Projection = _mm_add_ps(PartXY, _mm_mul_ps(_mm_load_ps(Triangle.DirectionZ + Lane), Z));

        return _mm_max_ps(
            _mm_sub_ps(_mm_load_ps(Triangle.Minimum + Lane), Projection),
            _mm_sub_ps(Projection, _mm_load_ps(Triangle.Maximum + Lane))
        );
    }

    // Projection of a box center's x and y on four axes.
    inline __m128 GetPartXY(const FTriangle& Triangle, int Lane, __m128 X, __m128 Y)
    {
        return _mm_add_ps(
            _mm_mul_ps(_mm_load_ps(Triangle.DirectionX + Lane), X),
            _mm_mul_ps(_mm_load_ps(Triangle.DirectionY + Lane), Y)
        );
    }

    inline EResult Test(const FTriangle& Triangle, const physx::PxVec3& Center)
    {
        if (Triangle.Degenerate || !Center.isFinite())
            return EResult::Uncertain;

        const __m128 X = _mm_set1_ps(Center.x);
        const __m128 Y = _mm_set1_ps(Center.y);
        const __m128 Z = _mm_set1_ps(Center.z);

        int UncertainMask = 0;

        for (int Lane = 0; Lane < int(Triangle.NumAxes); Lane += 4)
        {
            const __m128 Distance = GetDistance(Triangle, Lane, GetPartXY(Triangle, Lane, X, Y), Z);

            const __m128 Error = _mm_load_ps(Triangle.Error + Lane);

            // If the distance is greater than the error, the box is outside
            // the triangle.
            if (_mm_movemask_ps(_mm_cmpgt_ps(Distance, Error)) != 0)
                return EResult::Outside;

            UncertainMask |= _mm_movemask_ps(_mm_cmpge_ps(Distance, _mm_sub_ps(_mm_setzero_ps(), Error)));
        }

        return UncertainMask != 0 ? EResult::Uncertain : EResult::Overlap;
    }
}
