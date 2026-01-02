#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"
#include "DrawDebugHelpers.h"
#include "Engine/World.h"
#include "CollisionQueryParams.h"
#include <PxScene.h>

TMap<int32, FLinearColor> AVoxelDetectionSensor::ColorMap = AVoxelDetectionSensor::CreateColorMap();

AVoxelDetectionSensor::AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
    PrimaryActorTick.bCanEverTick = true;
}

FActorDefinition AVoxelDetectionSensor::GetSensorDefinition()
{
    auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(
        TEXT("other"),
        TEXT("voxel_detection"));

    FActorVariation Range;
    Range.Id = TEXT("range");
    Range.Type = EActorAttributeType::Float;
    Range.RecommendedValues = { TEXT("50.0") };
    Range.bRestrictToRecommended = false;

    FActorVariation Top;
    Top.Id = TEXT("upper_limit");
    Top.Type = EActorAttributeType::Float;
    Top.RecommendedValues = { TEXT("10.0") };
    Top.bRestrictToRecommended = false;
    
    FActorVariation Bottom;
    Bottom.Id = TEXT("lower_limit");
    Bottom.Type = EActorAttributeType::Float;
    Bottom.RecommendedValues = { TEXT("-5.0") };
    Bottom.bRestrictToRecommended = false;

    FActorVariation VoxelSize;
    VoxelSize.Id = TEXT("voxel_size");
    VoxelSize.Type = EActorAttributeType::Float;
    VoxelSize.RecommendedValues = { TEXT("0.5") };
    VoxelSize.bRestrictToRecommended = false;

    FActorVariation SelfIgnore;
    SelfIgnore.Id = TEXT("ignore_self");
    SelfIgnore.Type = EActorAttributeType::Int;
    SelfIgnore.RecommendedValues = { TEXT("0") };
    SelfIgnore.bRestrictToRecommended = false;

    FActorVariation DrawDebug;
    DrawDebug.Id = TEXT("draw_debug");
    DrawDebug.Type = EActorAttributeType::Int;
    DrawDebug.RecommendedValues = { TEXT("0") };
    DrawDebug.bRestrictToRecommended = false;

    Definition.Variations.Append({ Range, Top, Bottom, VoxelSize, SelfIgnore, DrawDebug });

    return Definition;
}

void AVoxelDetectionSensor::Set(const FActorDescription &Description)
{
    Super::Set(Description);

    constexpr float M_TO_CM = 100.0f;
    
    DetectedLen = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
        "range", Description.Variations, 50.0f);
    
    Top = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
        "upper_limit", Description.Variations, 10.0f);
    
    Bottom = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
        "lower_limit", Description.Variations, -5.0f);

    BoxSize = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat(
        "voxel_size", Description.Variations, 0.5f);

    SelfIgnore = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(
        "ignore_self", Description.Variations, 1);

    DrawDebug = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt(
        "draw_debug", Description.Variations, 0);

    // Pre-calculate grid dimensions
    GridSizeX = FMath::CeilToInt(2.0f * DetectedLen / BoxSize);
    GridSizeY = FMath::CeilToInt(2.0f * DetectedLen / BoxSize);
    GridSizeZ = FMath::CeilToInt((Top - Bottom) / BoxSize);
    
    UE_LOG(LogCarla, Log, TEXT("VoxelDetectionSensor: Grid size %d x %d x %d = %d voxels"), 
        GridSizeX, GridSizeY, GridSizeZ, GridSizeX * GridSizeY * GridSizeZ);
}

void AVoxelDetectionSensor::SetOwner(AActor *NewOwner)
{
    Super::SetOwner(NewOwner);
}

FVector AVoxelDetectionSensor::VoxelToWorld(int32 X, int32 Y, int32 Z) const
{
    FVector LocalPos(
        -DetectedLen + (X + 0.5f) * BoxSize,
        -DetectedLen + (Y + 0.5f) * BoxSize,
        Bottom + (Z + 0.5f) * BoxSize
    );
    return GetTransform().TransformPosition(LocalPos);
}

void AVoxelDetectionSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AVoxelDetectionSensor::PostPhysTick);

    const double StartTime = FPlatformTime::Seconds();
    
    const int32 TotalVoxels = GridSizeX * GridSizeY * GridSizeZ;
    
    // Initialize voxel grid with 0 (empty)
    TArray<uint8> SemanticVoxels;
    SemanticVoxels.SetNumZeroed(TotalVoxels);
    
    // Setup collision query
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);
    QueryParams.bTraceComplex = true;
    QueryParams.bReturnPhysicalMaterial = false;
    
    if (SelfIgnore > 0 && GetOwner())
    {
        QueryParams.AddIgnoredActor(GetOwner());
    }

    const FTransform SensorTransform = GetTransform();
    const FTransform InvSensorTransform = SensorTransform.Inverse();
    const float HalfVoxel = BoxSize * 0.5f;
    const float InvBoxSize = 1.0f / BoxSize;

    const double RaycastStart = FPlatformTime::Seconds();
    
    // Phase 1: Ray casting from 6 directions (±X, ±Y, ±Z) using ParallelLineTraceMultiByChannel
    {
        TRACE_CPUPROFILER_EVENT_SCOPE(VoxelRayCasting);
        
        // Direction 1 & 2: Rays along Z axis (top-down and bottom-up)
        ParallelFor(GridSizeX * GridSizeY, [&](int32 Index)
        {
            const int32 X = Index / GridSizeY;
            const int32 Y = Index % GridSizeY;
            
            const FVector LocalStart(
                -DetectedLen + (X + 0.5f) * BoxSize,
                -DetectedLen + (Y + 0.5f) * BoxSize,
                Top + HalfVoxel
            );
            const FVector LocalEnd(LocalStart.X, LocalStart.Y, Bottom - HalfVoxel);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            // Trace down (top to bottom)
            TArray<FHitResult> Hits;
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldStart, WorldEnd,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 Z = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.Z - Bottom) * InvBoxSize),
                        0, GridSizeZ - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
            
            // Trace up (bottom to top)
            Hits.Reset();
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldEnd, WorldStart,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 Z = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.Z - Bottom) * InvBoxSize),
                        0, GridSizeZ - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    if (SemanticVoxels[FlatIndex] == 0)
                        SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
        });
        
        // Direction 3 & 4: Rays along X axis (left-to-right and right-to-left)
        ParallelFor(GridSizeY * GridSizeZ, [&](int32 Index)
        {
            const int32 Y = Index / GridSizeZ;
            const int32 Z = Index % GridSizeZ;
            
            const FVector LocalStart(
                -DetectedLen - HalfVoxel,
                -DetectedLen + (Y + 0.5f) * BoxSize,
                Bottom + (Z + 0.5f) * BoxSize
            );
            const FVector LocalEnd(DetectedLen + HalfVoxel, LocalStart.Y, LocalStart.Z);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            // Trace +X (left to right)
            TArray<FHitResult> Hits;
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldStart, WorldEnd,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 X = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.X + DetectedLen) * InvBoxSize),
                        0, GridSizeX - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    if (SemanticVoxels[FlatIndex] == 0)
                        SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
            
            // Trace -X (right to left)
            Hits.Reset();
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldEnd, WorldStart,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 X = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.X + DetectedLen) * InvBoxSize),
                        0, GridSizeX - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    if (SemanticVoxels[FlatIndex] == 0)
                        SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
        });
        
        // Direction 5 & 6: Rays along Y axis (front-to-back and back-to-front)
        ParallelFor(GridSizeX * GridSizeZ, [&](int32 Index)
        {
            const int32 X = Index / GridSizeZ;
            const int32 Z = Index % GridSizeZ;
            
            const FVector LocalStart(
                -DetectedLen + (X + 0.5f) * BoxSize,
                -DetectedLen - HalfVoxel,
                Bottom + (Z + 0.5f) * BoxSize
            );
            const FVector LocalEnd(LocalStart.X, DetectedLen + HalfVoxel, LocalStart.Z);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            // Trace +Y (front to back)
            TArray<FHitResult> Hits;
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldStart, WorldEnd,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 Y = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.Y + DetectedLen) * InvBoxSize),
                        0, GridSizeY - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    if (SemanticVoxels[FlatIndex] == 0)
                        SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
            
            // Trace -Y (back to front)
            Hits.Reset();
            GetWorld()->ParallelLineTraceMultiByChannel(
                Hits, WorldEnd, WorldStart,
                ECC_GameTraceChannel2, QueryParams
            );
            
            for (const FHitResult& Hit : Hits)
            {
                if (Hit.bBlockingHit && Hit.Component.IsValid())
                {
                    const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
                    const int32 Y = FMath::Clamp(
                        FMath::FloorToInt((LocalHit.Y + DetectedLen) * InvBoxSize),
                        0, GridSizeY - 1
                    );
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    if (SemanticVoxels[FlatIndex] == 0)
                        SemanticVoxels[FlatIndex] = FMath::Clamp(Hit.Component->CustomDepthStencilValue, 0, 255);
                }
            }
        });
    }
    
    const double RaycastEnd = FPlatformTime::Seconds();
    
    // Phase 2: Fill interior voxels between surface hits
    const double FillStart = FPlatformTime::Seconds();
    {
        TRACE_CPUPROFILER_EVENT_SCOPE(FillInterior);
        
        ParallelFor(GridSizeX * GridSizeY, [&](int32 Index)
        {
            const int32 X = Index / GridSizeY;
            const int32 Y = Index % GridSizeY;
            
            int32 LastLabel = 0;
            int32 LastZ = -1;
            
            for (int32 Z = 0; Z < GridSizeZ; ++Z)
            {
                const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                const uint8 CurrentLabel = SemanticVoxels[FlatIndex];
                
                if (CurrentLabel > 0)
                {
                    // Fill gap if same label
                    if (LastLabel == CurrentLabel && LastZ >= 0 && (Z - LastZ) <= 30)
                    {
                        for (int32 FillZ = LastZ + 1; FillZ < Z; ++FillZ)
                        {
                            const int32 FillIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + FillZ;
                            if (SemanticVoxels[FillIndex] == 0)
                                SemanticVoxels[FillIndex] = CurrentLabel;
                        }
                    }
                    LastLabel = CurrentLabel;
                    LastZ = Z;
                }
            }
        });
    }
    const double FillEnd = FPlatformTime::Seconds();
    
    // Debug visualization
    if (DrawDebug > 0)
    {
        for (int32 X = 0; X < GridSizeX; ++X)
        {
            for (int32 Y = 0; Y < GridSizeY; ++Y)
            {
                for (int32 Z = 0; Z < GridSizeZ; ++Z)
                {
                    const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
                    const uint8 Label = SemanticVoxels[FlatIndex];
                    
                    if (Label > 0)
                    {
                        FVector Pos = VoxelToWorld(X, Y, Z);
                        FLinearColor Color = ColorMap.Contains(Label) ? ColorMap[Label] : FLinearColor::Green;
                        DrawDebugBox(GetWorld(), Pos, FVector(BoxSize * 0.4f), GetActorQuat(), 
                            Color.ToFColor(true), false, DeltaTime * 1.1f, 0, 1.0f);
                    }
                }
            }
        }
    }
    
    // Send data
    auto DataStream = GetDataStream(*this);
    DataStream.SerializeAndSend(*this, GetEpisode(), SemanticVoxels);
    
    const double EndTime = FPlatformTime::Seconds();
    
    // Performance logging
    carla::log_warning("VoxelDetectionSensor: Raycasting %.1f ms, Fill %.1f ms, Total %.1f ms",
        1000.0 * (RaycastEnd - RaycastStart),
        1000.0 * (FillEnd - FillStart),
        1000.0 * (EndTime - StartTime));
}