// Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB). This work is licensed under the terms of the MIT license.

#include "Carla.h"
#include "Carla/Sensor/VoxelDetectionGPU.h"

#include "GlobalShader.h"
#include "HAL/PlatformProcess.h"
#include "HAL/ThreadSafeBool.h"
#include "Misc/App.h"
#include "RenderGraphUtils.h"
#include "RenderingThread.h"
#include "RenderResource.h"
#include "RHIGPUReadback.h"
#include "RHIUtilities.h"
#include "ShaderParameterStruct.h"

namespace
{
    class FVoxelDetectionCS final : public FGlobalShader
    {
        DECLARE_GLOBAL_SHADER(FVoxelDetectionCS);
        SHADER_USE_PARAMETER_STRUCT(FVoxelDetectionCS, FGlobalShader);

        BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
            SHADER_PARAMETER(uint32, ChunkCount)
            SHADER_PARAMETER(uint32, DispatchWidth)
            SHADER_PARAMETER(uint32, OutputWidth)
            SHADER_PARAMETER_SRV(StructuredBuffer<FVoxelChunk>, Chunks)
            SHADER_PARAMETER_SRV(StructuredBuffer<FVoxelCandidate>, Candidates)
            SHADER_PARAMETER_SRV(StructuredBuffer<FVoxelTriangle>, Triangles)
            SHADER_PARAMETER_SRV(StructuredBuffer<FVoxelAxis>, Axes)
            SHADER_PARAMETER_UAV(RWTexture2D<uint>, OutputWords)
        END_SHADER_PARAMETER_STRUCT()

        static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
        {
            return IsVulkanPlatform(Parameters.Platform) &&
                IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::SM5);
        }
    };

    IMPLEMENT_GLOBAL_SHADER(FVoxelDetectionCS, "/Plugin/Carla/Private/VoxelDetection.usf", "MainCS", SF_Compute);

    struct FInputBuffer
    {
        FStructuredBufferRHIRef Buffer;
        FShaderResourceViewRHIRef SRV;
        uint32 CapacityBytes = 0;

        void Release()
        {
            SRV.SafeRelease();
            Buffer.SafeRelease();
            CapacityBytes = 0;
        }

        void Upload(const void* Data, uint32 Bytes, uint32 Stride, uint32 Capacity, const TCHAR* Name)
        {
            if (CapacityBytes != Capacity)
            {
                Release();
                FRHIResourceCreateInfo CreateInfo(Name);
                Buffer = RHICreateStructuredBuffer(Stride, Capacity,
                    BUF_ShaderResource | BUF_Dynamic, ERHIAccess::SRVCompute, CreateInfo);
                SRV = RHICreateShaderResourceView(Buffer);
                CapacityBytes = Capacity;
            }
            void* Destination = RHILockStructuredBuffer(Buffer, 0, Bytes, RLM_WriteOnly);
            FMemory::Memcpy(Destination, Data, Bytes);
            RHIUnlockStructuredBuffer(Buffer);
        }
    };

    class FVoxelGPUResources : public FRenderResource
    {
    public:
        FInputBuffer Inputs[4];
        FTexture2DRHIRef Output;
        FUnorderedAccessViewRHIRef OutputUAV;
        uint32 OutputCapacityWords = 0;
        TUniquePtr<FRHIGPUTextureReadback> Readback;

        void ReleaseRHI() override
        {
            for (FInputBuffer& Input : Inputs)
            {
                Input.Release();
            }
            OutputUAV.SafeRelease();
            Output.SafeRelease();
            OutputCapacityWords = 0;
            Readback.Reset();
        }
    };

    TGlobalResource<FVoxelGPUResources> Resources;
    FThreadSafeBool BackendFailed(false);

    bool Validate(const VoxelDetectionGPU::FBatch& Batch, FString& Error)
    {
        if (Batch.InputBytes() > VoxelDetectionGPU::MaxInputBytes ||
            Batch.NumOutputWords > VoxelDetectionGPU::MaxOutputWords ||
            Batch.Candidates.Num() == 0 || Batch.Triangles.Num() == 0 || Batch.Axes.Num() == 0)
        {
            Error = TEXT("GPU packet exceeds its budget or has empty SAT inputs");
            return false;
        }
        uint64 ExpectedOutput = 0;
        for (const VoxelDetectionGPU::FChunk& Chunk : Batch.Chunks)
        {
            const auto& Size = Chunk.SizeAndOutput;
            const auto& Candidates = Chunk.Candidates;
            const uint64 MaxVoxels = uint64(Batch.NumOutputWords) * 4;
            if (!Size.X || !Size.Y || !Size.Z || Size.X > MaxVoxels ||
                Size.Y > MaxVoxels / Size.X || Size.Z > MaxVoxels / (uint64(Size.X) * Size.Y))
            {
                Error = TEXT("GPU chunk voxel count is invalid");
                return false;
            }
            const uint64 Voxels = uint64(Size.X) * Size.Y * Size.Z;
            const uint64 Words = (Voxels + 3) / 4;
            if (Size.W != ExpectedOutput || ExpectedOutput > Batch.NumOutputWords ||
                Words > Batch.NumOutputWords - ExpectedOutput ||
                uint64(Candidates.X) + Candidates.Y > uint64(Batch.Candidates.Num()))
            {
                Error = TEXT("GPU chunk dimensions, candidate span, or output packing are invalid");
                return false;
            }
            ExpectedOutput += Words;
            uint32 LastPriority = MAX_uint32;
            for (uint32 Index = Candidates.X; Index < Candidates.X + Candidates.Y; ++Index)
            {
                const auto& Candidate = Batch.Candidates[Index];
                if (!Candidate.Label || Candidate.Label > 31 || !Candidate.Priority ||
                    Candidate.Priority > LastPriority)
                {
                    Error = TEXT("GPU candidate labels or priority ordering are invalid");
                    return false;
                }
                LastPriority = Candidate.Priority;
            }
        }
        if (ExpectedOutput != Batch.NumOutputWords)
        {
            Error = TEXT("GPU packet output contains unwritten words");
            return false;
        }
        for (const auto& Candidate : Batch.Candidates)
        {
            if (!Candidate.TriangleCount || uint64(Candidate.TriangleOffset) + Candidate.TriangleCount >
                uint64(Batch.Triangles.Num()))
            {
                Error = TEXT("GPU candidate triangle span is invalid");
                return false;
            }
        }
        for (const auto& Triangle : Batch.Triangles)
        {
            if (Triangle.NumAxes > 13 || uint64(Triangle.AxisOffset) + Triangle.NumAxes >
                uint64(Batch.Axes.Num()))
            {
                Error = TEXT("GPU triangle axis span is invalid");
                return false;
            }
        }
        return true;
    }
}

bool VoxelDetectionGPU::IsSupported()
{
    return !BackendFailed && GDynamicRHI && !GUsingNullRHI && FApp::CanEverRender() &&
        GMaxRHIFeatureLevel >= ERHIFeatureLevel::SM5 && IsVulkanPlatform(GMaxRHIShaderPlatform);
}

bool VoxelDetectionGPU::Run(const FBatch& Batch, TArray<uint32>& OutputWords, FString& Error)
{
    OutputWords.Reset();
    Error.Reset();
    if (!IsInGameThread() || !IsSupported())
    {
        Error = TEXT("Voxel GPU backend requires the game thread and Vulkan SM5 with rendering enabled");
        return false;
    }
    if (Batch.Chunks.Num() == 0 && Batch.NumOutputWords == 0)
    {
        return true;
    }
    if (!Validate(Batch, Error))
    {
        return false;
    }

    bool Success = false;
    // Run is synchronous: packet arrays and destination remain alive until this command completes.
    ENQUEUE_RENDER_COMMAND(CarlaVoxelDetection)(
        [&Batch, &OutputWords, &Error, &Success](FRHICommandListImmediate& RHICmdList)
    {
        FGlobalShaderMap* ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);
        if (!ShaderMap || !ShaderMap->HasShader(&FVoxelDetectionCS::StaticType, 0))
        {
            Error = TEXT("Voxel compute shader is unavailable; rebuild/cook plugin shaders");
            BackendFailed = true;
            return;
        }

        const uint32 Counts[] = {uint32(Batch.Chunks.Num()), uint32(Batch.Candidates.Num()),
            uint32(Batch.Triangles.Num()), uint32(Batch.Axes.Num())};
        const uint32 Strides[] = {sizeof(FChunk), sizeof(FCandidate), sizeof(FTriangle), sizeof(FAxis)};
        const void* Data[] = {Batch.Chunks.GetData(), Batch.Candidates.GetData(),
            Batch.Triangles.GetData(), Batch.Axes.GetData()};
        const TCHAR* Names[] = {TEXT("VoxelChunks"), TEXT("VoxelCandidates"),
            TEXT("VoxelTriangles"), TEXT("VoxelAxes")};
        uint32 Capacities[4];
        uint64 CapacitySum = 0;
        for (int32 Index = 0; Index < 4; ++Index)
        {
            Capacities[Index] = FMath::Max(Resources.Inputs[Index].CapacityBytes,
                FMath::RoundUpToPowerOfTwo(Counts[Index]) * Strides[Index]);
            CapacitySum += Capacities[Index];
        }
        // Different packet shapes must not accumulate unbounded historical buffer peaks.
        if (CapacitySum > MaxInputBytes)
        {
            for (int32 Index = 0; Index < 4; ++Index)
            {
                Capacities[Index] = Counts[Index] * Strides[Index];
            }
        }
        for (int32 Index = 0; Index < 4; ++Index)
        {
            Resources.Inputs[Index].Upload(Data[Index], Counts[Index] * Strides[Index],
                Strides[Index], Capacities[Index], Names[Index]);
        }
        const uint32 OutputCapacity = FMath::RoundUpToPowerOfTwo(Batch.NumOutputWords);
        const uint32 OutputWidth = FMath::Min(OutputCapacity, 2048u);
        if (Resources.OutputCapacityWords != OutputCapacity)
        {
            Resources.Readback.Reset();
            Resources.OutputUAV.SafeRelease();
            Resources.Output.SafeRelease();
            FRHIResourceCreateInfo CreateInfo(TEXT("VoxelPackedLabels"));
            Resources.Output = RHICreateTexture2D(OutputWidth, OutputCapacity / OutputWidth,
                PF_R32_UINT, 1, 1, TexCreate_ShaderResource | TexCreate_UAV,
                ERHIAccess::UAVCompute, CreateInfo);
            Resources.OutputUAV = RHICreateUnorderedAccessView(Resources.Output);
            Resources.OutputCapacityWords = OutputCapacity;
        }
        if (!Resources.Readback)
        {
            Resources.Readback = MakeUnique<FRHIGPUTextureReadback>(TEXT("VoxelPackedReadback"));
        }

        TShaderMapRef<FVoxelDetectionCS> ComputeShader(ShaderMap);
        FVoxelDetectionCS::FParameters Parameters;
        Parameters.ChunkCount = Batch.Chunks.Num();
        Parameters.DispatchWidth = FMath::Min<uint32>(Parameters.ChunkCount, 65535);
        Parameters.OutputWidth = OutputWidth;
        Parameters.Chunks = Resources.Inputs[0].SRV;
        Parameters.Candidates = Resources.Inputs[1].SRV;
        Parameters.Triangles = Resources.Inputs[2].SRV;
        Parameters.Axes = Resources.Inputs[3].SRV;
        Parameters.OutputWords = Resources.OutputUAV;
        RHICmdList.Transition(FRHITransitionInfo(Resources.Output, ERHIAccess::Unknown,
            ERHIAccess::UAVCompute));
        FComputeShaderUtils::Dispatch(RHICmdList, ComputeShader, Parameters,
            FIntVector(Parameters.DispatchWidth,
                FMath::DivideAndRoundUp(Parameters.ChunkCount, Parameters.DispatchWidth), 1));
        // The Vulkan texture mapping path invalidates noncoherent host memory; its buffer
        // readback counterpart in UE4.26 does not. The texture still holds packed byte labels.
        Resources.Readback->EnqueueCopy(RHICmdList, Resources.Output);
        RHICmdList.SubmitCommandsHint();
        RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);

        const double Deadline = FPlatformTime::Seconds() + 30.0;
        while (!Resources.Readback->IsReady())
        {
            if (FPlatformTime::Seconds() >= Deadline)
            {
                // Keep in-flight resources alive; never reuse them after a failed wait.
                BackendFailed = true;
                Error = TEXT("Voxel GPU readback timed out after 30 seconds");
                return;
            }
            FPlatformProcess::SleepNoStats(0.001f);
            // Vulkan fence Poll reads cached status; submitting a hint refreshes that status.
            RHICmdList.SubmitCommandsHint();
            RHICmdList.ImmediateFlush(EImmediateFlushType::FlushRHIThread);
        }
        void* ReadbackData = nullptr;
        int32 RowPitch = 0;
        Resources.Readback->LockTexture(RHICmdList, ReadbackData, RowPitch);
        if (!ReadbackData || RowPitch < int32(OutputWidth))
        {
            if (ReadbackData)
            {
                Resources.Readback->Unlock();
            }
            BackendFailed = true;
            Error = TEXT("Voxel GPU staging texture could not be mapped");
            return;
        }
        OutputWords.SetNumUninitialized(Batch.NumOutputWords);
        const uint32* Source = static_cast<const uint32*>(ReadbackData);
        for (uint32 Offset = 0, Row = 0; Offset < Batch.NumOutputWords; Offset += OutputWidth, ++Row)
        {
            const uint32 Words = FMath::Min(OutputWidth, Batch.NumOutputWords - Offset);
            FMemory::Memcpy(OutputWords.GetData() + Offset, Source + uint64(Row) * RowPitch,
                Words * sizeof(uint32));
        }
        Resources.Readback->Unlock();
        Success = true;
    });
    FlushRenderingCommands();
    return Success;
}
