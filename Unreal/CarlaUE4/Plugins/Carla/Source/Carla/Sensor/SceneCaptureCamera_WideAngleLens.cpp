// Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "Carla.h"
#include "Carla/Sensor/SceneCaptureCamera_WideAngleLens.h"
#include "Carla/Game/CarlaEngine.h"
#include <chrono>

#include "Actor/ActorBlueprintFunctionLibrary.h"

#include "Runtime/RenderCore/Public/RenderingThread.h"

FActorDefinition ASceneCaptureCamera_WideAngleLens::GetSensorDefinition()
{
    constexpr bool bEnableModifyingPostProcessEffects = true;
    return UActorBlueprintFunctionLibrary::MakeWideAngleLensCameraDefinition(
        TEXT("rgb"),
        bEnableModifyingPostProcessEffects);
}

ASceneCaptureCamera_WideAngleLens::ASceneCaptureCamera_WideAngleLens(const FObjectInitializer& ObjectInitializer)
    : Super(ObjectInitializer)
{
    Super::SetCubemapSampler(CameraModelUtil::GetSampler(ESamplerFilter::SF_AnisotropicLinear));

    EnablePostProcessingEffects(true);
}

void ASceneCaptureCamera_WideAngleLens::BeginPlay()
{
    Super::BeginPlay();
}

void ASceneCaptureCamera_WideAngleLens::OnFirstClientConnected()
{
}

void ASceneCaptureCamera_WideAngleLens::OnLastClientDisconnected()
{
}

void ASceneCaptureCamera_WideAngleLens::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
    Super::EndPlay(EndPlayReason);
}

void ASceneCaptureCamera_WideAngleLens::PostPhysTick(UWorld* World, ELevelTick TickType, float DeltaSeconds)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(ASceneCaptureCamera_WideAngleLens::PostPhysTick);
    /*
    ENQUEUE_RENDER_COMMAND(MeasureTime)([](auto& InRHICmdList)
    {
        std::chrono::time_point<std::chrono::high_resolution_clock> Time =
            std::chrono::high_resolution_clock::now();
        auto Duration = std::chrono::duration_cast<std::chrono::milliseconds>(Time.time_since_epoch());
        uint64_t Milliseconds = Duration.count();
        FString ProfilerText =
            FString("(Render)Frame: ") +
            FString::FromInt(FCarlaEngine::GetFrameCounter()) +
            FString(" Time: ") +
            FString::FromInt(Milliseconds);
        TRACE_CPUPROFILER_EVENT_SCOPE_TEXT(*ProfilerText);
    });
    */
    FPixelReader::SendPixelsInRenderThread<ASceneCaptureCamera_WideAngleLens, FColor>(*this);
}

// void ASceneCaptureCamera_WideAngleLens::SendGBufferTextures(FGBufferRequest& GBuffer)
// {
//     SendGBufferTexturesInternal(*this, GBuffer);
// }
