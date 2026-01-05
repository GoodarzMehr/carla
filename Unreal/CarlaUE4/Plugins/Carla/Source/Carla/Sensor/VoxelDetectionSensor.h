#pragma once

#include "Carla/Sensor/Sensor.h"
#include "Carla/Actor/ActorDefinition.h"
#include "Carla/Actor/ActorDescription.h"

#include "VoxelDetectionSensor.generated.h"

UCLASS()
class CARLA_API AVoxelDetectionSensor : public ASensor {
    
	GENERATED_BODY()

public:

    AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer);

    static FActorDefinition GetSensorDefinition();

    void Set(const FActorDescription &ActorDescription) override;

    void SetOwner(AActor *NewOwner) override;

    void PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime) override;

    FVector VoxelToWorld(int32 X, int32 Y, int32 Z) const;
    
    static TMap<int32, FLinearColor> ColorMap;
    
    static TMap<int32, FLinearColor> CreateColorMap()
    {
        TMap<int32, FLinearColor> NewColorMap;
        
		NewColorMap.Add(0, FLinearColor::FromSRGBColor(FColor(0, 0, 0, 0)));
        NewColorMap.Add(1, FLinearColor::FromSRGBColor(FColor(128, 64, 128, 255)));
        NewColorMap.Add(2, FLinearColor::FromSRGBColor(FColor(244, 35, 232, 255)));
        NewColorMap.Add(3, FLinearColor::FromSRGBColor(FColor(70, 70, 70, 255)));
        NewColorMap.Add(4, FLinearColor::FromSRGBColor(FColor(102, 102, 156, 255)));
        NewColorMap.Add(5, FLinearColor::FromSRGBColor(FColor(190, 153, 153, 255)));
        NewColorMap.Add(6, FLinearColor::FromSRGBColor(FColor(153, 153, 153, 255)));
        NewColorMap.Add(7, FLinearColor::FromSRGBColor(FColor(250, 170, 30, 255)));
        NewColorMap.Add(8, FLinearColor::FromSRGBColor(FColor(220, 220, 0, 255)));
        NewColorMap.Add(9, FLinearColor::FromSRGBColor(FColor(107, 142, 35, 255)));
        NewColorMap.Add(10, FLinearColor::FromSRGBColor(FColor(152, 251, 152, 255)));
        NewColorMap.Add(11, FLinearColor::FromSRGBColor(FColor(70, 130, 180, 255)));
        NewColorMap.Add(12, FLinearColor::FromSRGBColor(FColor(220, 20, 60, 255)));
        NewColorMap.Add(13, FLinearColor::FromSRGBColor(FColor(255, 0, 0, 255)));
        NewColorMap.Add(14, FLinearColor::FromSRGBColor(FColor(0, 0, 142, 255)));
        NewColorMap.Add(15, FLinearColor::FromSRGBColor(FColor(0, 0, 70, 255)));
        NewColorMap.Add(16, FLinearColor::FromSRGBColor(FColor(0, 60, 100, 255)));
        NewColorMap.Add(17, FLinearColor::FromSRGBColor(FColor(0, 80, 100, 255)));
        NewColorMap.Add(18, FLinearColor::FromSRGBColor(FColor(0, 0, 230, 255)));
        NewColorMap.Add(19, FLinearColor::FromSRGBColor(FColor(119, 11, 32, 255)));
        NewColorMap.Add(20, FLinearColor::FromSRGBColor(FColor(110, 190, 160, 255)));
        NewColorMap.Add(21, FLinearColor::FromSRGBColor(FColor(170, 120, 50, 255)));
        NewColorMap.Add(22, FLinearColor::FromSRGBColor(FColor(55, 90, 80, 255)));
        NewColorMap.Add(23, FLinearColor::FromSRGBColor(FColor(45, 60, 150, 255)));
        NewColorMap.Add(24, FLinearColor::FromSRGBColor(FColor(157, 234, 50, 255)));
        NewColorMap.Add(25, FLinearColor::FromSRGBColor(FColor(81, 0, 81, 255)));
        NewColorMap.Add(26, FLinearColor::FromSRGBColor(FColor(150, 100, 100, 255)));
        NewColorMap.Add(27, FLinearColor::FromSRGBColor(FColor(230, 150, 140, 255)));
        NewColorMap.Add(28, FLinearColor::FromSRGBColor(FColor(180, 165, 180, 255)));
        NewColorMap.Add(29, FLinearColor::FromSRGBColor(FColor(110, 110, 110, 255)));
        NewColorMap.Add(30, FLinearColor::FromSRGBColor(FColor(255, 165, 0, 255)));
        NewColorMap.Add(31, FLinearColor::FromSRGBColor(FColor(200, 128, 128, 255)));
        
		return NewColorMap;
    };

private:
    static constexpr uint8 SemanticPriority[32] = {
        0, // 0:  Unlabeled
        7, // 1:  Road
        7, // 2:  Sidewalk
        1, // 3:  Building
        1, // 4:  Wall
        1, // 5:  Fence
        1, // 6:  Pole
        2, // 7:  Traffic light
        2, // 8:  Traffic sign
        1, // 9:  Vegetation
        1, // 10: Terrain
        1, // 11: Sky
        6, // 12: Pedestrian
        6, // 13: Rider
        4, // 14: Car
        3, // 15: Truck
        3, // 16: Bus
        1, // 17: Train
        5, // 18: Motorcycle
        5, // 19: Bicycle
        1, // 20: Static
        1, // 21: Dynamic
        1, // 22: Other
        1, // 23: Water
        8, // 24: Road line
        1, // 25: Ground
        1, // 26: Bridge
        1, // 27: Rail track
        1, // 28: Guard rail
        1, // 29: Rock
        2, // 30: Traffic cone
        2  // 31: Barrier
    };

	float BoxSize = 50.0f;
    float Top = 1000.0f; 
    float Bottom = -100.0f;
    float BoxRange = 5000.0f;
    
    int32 GridSizeX = 0;
    int32 GridSizeY = 0;
    int32 GridSizeZ = 0;
    
    bool SelfIgnore = false;
    bool DrawDebug = false;
	bool UseTraceComplex = true;

	bool UseZTop = true;
	bool UseZBottom = true;
	bool UseXFront = true;
	bool UseXBack = true;
	bool UseYRight = true;
	bool UseYLeft = true;
};