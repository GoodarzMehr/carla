// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"
#include "Carla/Sensor/ShaderBasedSensor.h"
#include "Sensor/Sensor.h"
#include "Carla/Actor/ActorDefinition.h"
#include "Components/LineBatchComponent.h"
#include "Carla/Game/Tagger.h"
#include "Carla/Sensor/UE4_Overridden/LineBatchComponent_CARLA.h"
#include "Dom/JsonObject.h"
#include "CosmosControlSensor.generated.h"

/**
 * Sensor that produces an input control visualization for Cosmos
 */
UCLASS()
class CARLA_API ACosmosControlSensor : public AShaderBasedSensor
{
	GENERATED_BODY()

public:

  struct CosmosRenderConfig
  {
    float RoadLineThickness = 8.0f;
    float VehicleBoxThickness = 5.0f;
    float PoleThickness = 8.0f;
    float StopLineThickness = 8.0f;
    
    // Colors with defaults matching cosmos_writer
    FColor LaneLinesColor = FColor(98, 183, 249, 255);
    FColor RoadBoundariesColor = FColor(200, 36, 35, 255);
    FColor WaitLinesColor = FColor(185, 63, 34, 255);
    FColor CrosswalksColor = FColor(206, 131, 63, 255);
    FColor RoadMarkingsColor = FColor(126, 204, 205, 255);
    FColor TrafficSignsColor = FColor(131, 175, 155, 255);
    FColor TrafficLightsColor = FColor(252, 157, 155, 255);
    FColor CarsColor = FColor(255, 0, 0, 255);
    FColor TrucksColor = FColor(0, 0, 255, 255);
    FColor PedestriansColor = FColor(0, 255, 0, 255);
    FColor CyclistsColor = FColor(255, 255, 0, 255);
    FColor PolesColor = FColor(66, 40, 144, 255);
  };

	static FActorDefinition GetSensorDefinition();

  ACosmosControlSensor(const FObjectInitializer& ObjectInitializer);
  
  virtual void Set(const FActorDescription &Description) override;

protected:

  void SetUpSceneCaptureComponent(USceneCaptureComponent2D &SceneCapture) override;
  void PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaSeconds) override;

  //Duplicate functions from DrawDebugHelpers to guarantee they work outside editor
  void DrawDebugLine(const UWorld* InWorld, FVector const& LineStart, FVector const& LineEnd, FColor const& Color, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0, float Thickness = 0);
  void DrawDebugSolidBox(const UWorld* InWorld, FVector const& Center, FVector const& Extent, FQuat const& Rotation, FColor const& Color, bool bPersistent = false, float LifeTime = -1.f, uint8 DepthPriority = 0);
  void DrawDebugBox(const UWorld* InWorld, FVector const& Center, FVector const& Box, const FQuat& Rotation, FColor const& Color, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0, float Thickness = 0);
  void DrawDebugCapsule(const UWorld* InWorld, FVector const& Center, float HalfHeight, float Radius, const FQuat& Rotation, FColor const& Color, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0, float Thickness = 0);
  void DrawDebugMesh(const UWorld* InWorld, const TArray<FVector>& Vertices, const TArray<int32>& Indices, const FColor& Color, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0);
  void DrawCircle(const UWorld* InWorld, const FVector& Base, const FVector& X, const FVector& Y, const FColor& Color, float Radius, int32 NumSides, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0, float Thickness = 0);
  void DrawHalfCircle(const UWorld* InWorld, const FVector& Base, const FVector& X, const FVector& Y, const FColor& Color, float Radius, int32 NumSides, bool bPersistentLines = false, float LifeTime = -1.f, uint8 DepthPriority = 0, float Thickness = 0);

  UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
  ULineBatchComponent_CARLA* DynamicLines;
  UPROPERTY(VisibleAnywhere, BlueprintReadOnly)
  ULineBatchComponent_CARLA* PersistentLines;

private:
  ULineBatchComponent_CARLA* GetDebugLineBatcher(bool bPersistentLines);
  FColor GetColorByTag(carla::rpc::CityObjectLabel Tag);
  void LoadConfigFromFile();

private:
  bool added_persisted_stop_lines;
  bool added_persisted_route_lines;
  bool added_persisted_crosswalks;
  bool added_persisted_stencils;
  
  // Configuration for rendering parameters
  CosmosRenderConfig RenderConfig;
};
