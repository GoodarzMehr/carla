// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB). This work is licensed under the terms of the MIT license.

#pragma once

#include "CoreMinimal.h"

// Immutable, chunk-local SAT snapshots. No PhysX objects cross the GPU wait.
namespace VoxelDetectionGPU
{
    constexpr uint64 MaxInputBytes = 128ull * 1024 * 1024;
    constexpr uint32 MaxOutputWords = 4u * 1024 * 1024;
    constexpr uint8 FallbackLabel = 255;

    struct FUint4
    {
        uint32 X = 0;
        uint32 Y = 0;
        uint32 Z = 0;
        uint32 W = 0;
    };

    struct FChunk
    {
        FVector4 OriginAndError; // First voxel center in the triangle frame; w bounds center error.
        FVector4 StepX;
        FVector4 StepY;
        FVector4 StepZ;
        FUint4 SizeAndOutput; // XYZ voxel dimensions; W first packed output word.
        FUint4 Candidates; // X first candidate, Y count; candidates sorted by descending priority.
    };

    struct FCandidate
    {
        uint32 TriangleOffset = 0;
        uint32 TriangleCount = 0;
        uint32 Label = 0;
        uint32 Priority = 0;
    };

    struct FTriangle
    {
        uint32 AxisOffset = 0;
        uint32 NumAxes = 0;
        uint32 Degenerate = 0;
        uint32 Padding = 0;
    };

    struct FAxis
    {
        FVector4 DirectionAndMin; // xyz normalized SAT direction; w expanded minimum.
        FVector4 MaxAndError; // x expanded maximum, y CPU SAT error; zw unused.
    };

    struct FBatch
    {
        TArray<FChunk> Chunks;
        TArray<FCandidate> Candidates;
        TArray<FTriangle> Triangles;
        TArray<FAxis> Axes;
        uint32 NumOutputWords = 0;

        uint64 InputBytes() const
        {
            return uint64(Chunks.Num()) * sizeof(FChunk) +
                uint64(Candidates.Num()) * sizeof(FCandidate) +
                uint64(Triangles.Num()) * sizeof(FTriangle) +
                uint64(Axes.Num()) * sizeof(FAxis);
        }
    };

    static_assert(sizeof(FUint4) == 16, "GPU uint4 packing changed");
    static_assert(sizeof(FChunk) == 96, "GPU chunk packing changed");
    static_assert(sizeof(FCandidate) == 16, "GPU candidate packing changed");
    static_assert(sizeof(FTriangle) == 16, "GPU triangle packing changed");
    static_assert(sizeof(FAxis) == 32, "GPU SAT axis packing changed");

    // Initial backend targets Vulkan SM5, including offscreen rendering; NullRHI is unsupported.
    bool IsSupported();

    // Game-thread call. Waits for rendering and readback before returning.
    // Each word holds four little-endian byte labels; 255 requests whole-voxel CPU fallback.
    // Failure leaves output empty. Input/upload work is bounded; RHI buffers persist between calls.
    bool Run(const FBatch& Batch, TArray<uint32>& OutputWords, FString& Error);
}
