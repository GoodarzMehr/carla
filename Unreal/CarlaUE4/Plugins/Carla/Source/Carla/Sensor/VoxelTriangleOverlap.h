#pragma once

#include <cfloat>

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

    struct FAxis
    {
        physx::PxVec3 Direction;
        
        float Minimum;
        float Maximum;
        float Error;
    };

    struct FTriangle
    {
        FAxis Axes[13];
        
        physx::PxU32 NumAxes = 0;
        
        bool Degenerate = false;
    };

    inline float MaxAbs(const physx::PxVec3& Value)
    {
        return physx::PxMax(physx::PxAbs(Value.x), physx::PxMax(physx::PxAbs(Value.y), physx::PxAbs(Value.z)));
    }

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

        const auto AddAxis = [&](const physx::PxVec3& Direction)
        {
            const float Length = physx::PxAbs(Direction.x) + physx::PxAbs(Direction.y) + physx::PxAbs(Direction.z);
            
            if (Length == 0.0f)
                return;
            
            if (!physx::PxIsFinite(Length))
            {
                Result.Degenerate = true;
                
                return;
            }
            
            FAxis& Axis = Result.Axes[Result.NumAxes++];
            
            Axis.Direction = Direction / Length;
            
            const float P0 = Axis.Direction.dot(Vertices[0]);
            const float P1 = Axis.Direction.dot(Vertices[1]);
            const float P2 = Axis.Direction.dot(Vertices[2]);
            
            const float Radius = HalfExtent.dot(Axis.Direction.abs());
            
            Axis.Minimum = physx::PxMin(P0, physx::PxMin(P1, P2)) - Radius;
            Axis.Maximum = physx::PxMax(P0, physx::PxMax(P1, P2)) + Radius;
            
            const float Magnitude = physx::PxMax(physx::PxAbs(Axis.Minimum), physx::PxAbs(Axis.Maximum));
            
            // Bound projection roundoff at both interval boundaries once per
            // triangle.
            Axis.Error = PositionError + 64.0f * FLT_EPSILON * (Magnitude + PositionError);
            
            if (!physx::PxIsFinite(Axis.Minimum) || !physx::PxIsFinite(Axis.Maximum) || !physx::PxIsFinite(Axis.Error))
                Result.Degenerate = true;
        };

        AddAxis(physx::PxVec3(1.0f, 0.0f, 0.0f));
        AddAxis(physx::PxVec3(0.0f, 1.0f, 0.0f));
        AddAxis(physx::PxVec3(0.0f, 0.0f, 1.0f));
        
        AddAxis(Normal);
        
        for (const physx::PxVec3& Edge : Edges)
        {
            AddAxis(physx::PxVec3(0.0f, Edge.z, -Edge.y));
            AddAxis(physx::PxVec3(-Edge.z, 0.0f, Edge.x));
            AddAxis(physx::PxVec3(Edge.y, -Edge.x, 0.0f));
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

    inline EResult Test(const FTriangle& Triangle, const physx::PxVec3& Center)
    {
        if (Triangle.Degenerate || !Center.isFinite())
            return EResult::Uncertain;
        
        bool Uncertain = false;
        
        for (physx::PxU32 Index = 0; Index < Triangle.NumAxes; ++Index)
        {
            const FAxis& Axis = Triangle.Axes[Index];
            
            const float Projection = Axis.Direction.dot(Center);
            const float Distance = physx::PxMax(Axis.Minimum - Projection, Projection - Axis.Maximum);
            
            // If the distance is greater than the error, the point is outside the triangle.
            if (Distance > Axis.Error)
                return EResult::Outside;

            if (Distance >= -Axis.Error)
                Uncertain = true;
        }
        
        return Uncertain ? EResult::Uncertain : EResult::Overlap;
    }
}
