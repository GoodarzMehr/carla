#include "Carla.h"
#include "Carla/Game/CarlaEpisode.h"
#include "Carla/Sensor/VoxelDetectionSensor.h"
#include "Carla/Actor/ActorBlueprintFunctionLibrary.h"

#include "DrawDebugHelpers.h"
#include "CollisionQueryParams.h"

#include "Engine/World.h"
#include "Runtime/Core/Public/Async/ParallelFor.h"

#include <PxScene.h>

TMap<int32, FLinearColor> AVoxelDetectionSensor::ColorMap = AVoxelDetectionSensor::CreateColorMap();

AVoxelDetectionSensor::AVoxelDetectionSensor(const FObjectInitializer &ObjectInitializer)
  : Super(ObjectInitializer)
{
    PrimaryActorTick.bCanEverTick = true;
}

FActorDefinition AVoxelDetectionSensor::GetSensorDefinition()
{
    auto Definition = UActorBlueprintFunctionLibrary::MakeGenericSensorDefinition(TEXT("other"), TEXT("voxel_detection"));

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
    SelfIgnore.Type = EActorAttributeType::Bool;
    SelfIgnore.RecommendedValues = { TEXT("False") };
    SelfIgnore.bRestrictToRecommended = false;

    FActorVariation DrawDebug;
    DrawDebug.Id = TEXT("draw_debug");
    DrawDebug.Type = EActorAttributeType::Bool;
    DrawDebug.RecommendedValues = { TEXT("False") };
    DrawDebug.bRestrictToRecommended = false;

	FActorVariation ShowCalculationTime;
    ShowCalculationTime.Id = TEXT("show_calculation_time");
    ShowCalculationTime.Type = EActorAttributeType::Bool;
    ShowCalculationTime.RecommendedValues = { TEXT("False") };
    ShowCalculationTime.bRestrictToRecommended = false;

	FActorVariation UseTraceComplex;
    UseTraceComplex.Id = TEXT("use_complex_collision");
    UseTraceComplex.Type = EActorAttributeType::Bool;
    UseTraceComplex.RecommendedValues = { TEXT("True") };
    UseTraceComplex.bRestrictToRecommended = false;

	FActorVariation ChunkX;
    ChunkX.Id = TEXT("chunk_x");
    ChunkX.Type = EActorAttributeType::Int;
    ChunkX.RecommendedValues = { TEXT("4") };
    ChunkX.bRestrictToRecommended = false;

	FActorVariation ChunkY;
    ChunkY.Id = TEXT("chunk_y");
    ChunkY.Type = EActorAttributeType::Int;
    ChunkY.RecommendedValues = { TEXT("4") };
    ChunkY.bRestrictToRecommended = false;

	FActorVariation ChunkZ;
	ChunkZ.Id = TEXT("chunk_z");
	ChunkZ.Type = EActorAttributeType::Int;
	ChunkZ.RecommendedValues = { TEXT("4") };
	ChunkZ.bRestrictToRecommended = false;

    Definition.Variations.Append(
		{
			Range,
			Top,
			Bottom,
			VoxelSize,
			SelfIgnore,
			DrawDebug,
			ShowCalculationTime,
			UseTraceComplex,
			ChunkX,
			ChunkY,
			ChunkZ,
		}
	);

    return Definition;
}

void AVoxelDetectionSensor::Set(const FActorDescription &Description)
{
    Super::Set(Description);

    constexpr float M_TO_CM = 100.0f;
    
    BoxRange = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("range", Description.Variations, 50.0f);
    
    Top = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("upper_limit", Description.Variations, 10.0f);
    
    Bottom = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("lower_limit", Description.Variations, -5.0f);

    BoxSize = M_TO_CM * UActorBlueprintFunctionLibrary::RetrieveActorAttributeToFloat("voxel_size", Description.Variations, 0.5f);

    SelfIgnore = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("ignore_self", Description.Variations, false);

    DrawDebug = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("draw_debug", Description.Variations, false);

	ShowCalculationTime = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("show_calculation_time", Description.Variations, false);
	
	UseTraceComplex = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_complex_collision", Description.Variations, true);

	ChunkX = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_x", Description.Variations, 4);

	ChunkY = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_y", Description.Variations, 4);

	ChunkZ = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToInt("chunk_z", Description.Variations, 4);

    // Pre-calculate grid dimensions.
    GridSizeX = FMath::CeilToInt(2.0f * BoxRange / BoxSize);
    GridSizeY = FMath::CeilToInt(2.0f * BoxRange / BoxSize);
    GridSizeZ = FMath::CeilToInt((Top - Bottom) / BoxSize);
    
    UE_LOG(
		LogCarla,
		Log,
		TEXT("VoxelDetectionSensor: Grid size %d x %d x %d = %d voxels"), 
        GridSizeX,
		GridSizeY,
		GridSizeZ,
		GridSizeX * GridSizeY * GridSizeZ
	);
}

void AVoxelDetectionSensor::SetOwner(AActor *NewOwner)
{
    Super::SetOwner(NewOwner);
}

FVector AVoxelDetectionSensor::VoxelToWorld(int32 X, int32 Y, int32 Z) const
{
    FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

    return GetTransform().TransformPosition(LocalPos);
}

void AVoxelDetectionSensor::PostPhysTick(UWorld *World, ELevelTick TickType, float DeltaTime)
{
    TRACE_CPUPROFILER_EVENT_SCOPE(AVoxelDetectionSensor::PostPhysTick);

	const double StartTime = FPlatformTime::Seconds();
    
    const int32 TotalVoxels = GridSizeX * GridSizeY * GridSizeZ;
    
    // Initialize voxel grids with 0 (empty).
    TArray<uint8> SemanticVoxels;
    SemanticVoxels.SetNumZeroed(TotalVoxels);
    
    // Set up collision query.
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);
    
	QueryParams.bTraceComplex = UseTraceComplex;
    QueryParams.bReturnPhysicalMaterial = false;
    
    if (SelfIgnore && GetOwner())
    {
        QueryParams.AddIgnoredActor(GetOwner());
    }
    
	const float HalfVoxel = BoxSize * 0.5f;

	const FVector BoxExtent(HalfVoxel, HalfVoxel, HalfVoxel);

	FCollisionObjectQueryParams ObjectParams(FCollisionObjectQueryParams::AllObjects);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Vehicle);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Pawn);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Camera);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Visibility);
	ObjectParams.RemoveObjectTypesToQuery(ECC_PhysicsBody);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Destructible);
    
	GetWorld()->GetPhysicsScene()->GetPxScene()->lockRead();

	const FTransform SensorTransform = GetTransform();

	const FQuat VoxelRotation = SensorTransform.GetRotation();
	
	const int32 SGridX = (GridSizeX + ChunkX - 1) / ChunkX;
	const int32 SGridY = (GridSizeY + ChunkY - 1) / ChunkY;
	const int32 SGridZ = (GridSizeZ + ChunkZ - 1) / ChunkZ;
	
	ParallelFor(SGridX * SGridY * SGridZ, [&](int32 Index)
	{	
		const int32 CX = Index / (SGridY * SGridZ);
		
		const int32 Remainder = Index % (SGridY * SGridZ);
		
		const int32 CY = Remainder / SGridZ;
		const int32 CZ = Remainder % SGridZ;

		const int32 XEnd = FMath::Min((CX + 1) * ChunkX, GridSizeX);
		const int32 YEnd = FMath::Min((CY + 1) * ChunkY, GridSizeY);
		const int32 ZEnd = FMath::Min((CZ + 1) * ChunkZ, GridSizeZ);

		const float LocalMinX = -BoxRange + CX * ChunkX * BoxSize;
		const float LocalMinY = -BoxRange + CY * ChunkY * BoxSize;
		const float LocalMinZ = Bottom + CZ * ChunkZ * BoxSize;

		const float LocalMaxX = -BoxRange + XEnd * BoxSize;
		const float LocalMaxY = -BoxRange + YEnd * BoxSize;
		const float LocalMaxZ = Bottom + ZEnd * BoxSize;

		const FVector LocalMin(LocalMinX, LocalMinY, LocalMinZ);
		const FVector LocalMax(LocalMaxX, LocalMaxY, LocalMaxZ);

		const FVector ChunkCenter = SensorTransform.TransformPosition((LocalMin + LocalMax) * 0.5f);

		const FVector ChunkExtent = (LocalMax - LocalMin) * 0.5f + FVector(0.1f);

		const FCollisionShape ChunkShape = FCollisionShape::MakeBox(ChunkExtent);

		bool bChunkEmpty = !GetWorld()->ParallelOverlapAnyTestByObjectType(
			ChunkCenter,
			VoxelRotation,
			ObjectParams,
			ChunkShape,
			QueryParams
		);

		if (bChunkEmpty)
			return;

		for (int32 X = CX * ChunkX; X < XEnd; ++X)
		{
			for (int32 Y = CY * ChunkY; Y < YEnd; ++Y)
			{
				for (int32 Z = CZ * ChunkZ; Z < ZEnd; ++Z)
				{
					const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
		
					const FVector LocalPos(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);

					const FVector WorldPos = SensorTransform.TransformPosition(LocalPos);

					TArray<FOverlapResult> Overlaps;
									
					bool bOverlap = GetWorld()->ParallelOverlapMultiByObjectType(
						Overlaps,
						WorldPos,
						VoxelRotation,
						ObjectParams,
						FCollisionShape::MakeBox(BoxExtent),
						QueryParams
					);
		
					if (bOverlap)
					{
						for (const FOverlapResult& Overlap : Overlaps)
						{
							if (Overlap.Component.IsValid())
							{
								if (SemanticPriority[Overlap.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
									SemanticVoxels[FlatIndex] = Overlap.Component->CustomDepthStencilValue;
							}
						}
					}
				}
			}
		}
	});

	GetWorld()->GetPhysicsScene()->GetPxScene()->unlockRead();
    
    // Debug visualization.
    if (DrawDebug)
    {
        UWorld* World = GetWorld();
    	const FQuat ActorQuat = GetActorQuat();
		
		for (int32 FlatIndex = 0; FlatIndex < GridSizeX * GridSizeY * GridSizeZ; ++FlatIndex)
		{
			const uint8 Label = SemanticVoxels[FlatIndex];
			
			if (Label > 0)
			{
				// Convert flat index back to X, Y, Z.
				const int32 X = FlatIndex / (GridSizeY * GridSizeZ);
				const int32 Remainder = FlatIndex % (GridSizeY * GridSizeZ);
				const int32 Y = Remainder / GridSizeZ;
				const int32 Z = Remainder % GridSizeZ;
				
				FVector Pos = VoxelToWorld(X, Y, Z);
				
				FLinearColor Color = ColorMap.Contains(Label) ? ColorMap[Label] : FLinearColor::Green;
				
				DrawDebugBox(World, Pos, BoxExtent, ActorQuat, Color.ToFColor(true), false, DeltaTime * 1.1f, 0, 2.0f);
			}
		}
    }
    
    // Send data
    auto DataStream = GetDataStream(*this);

    DataStream.SerializeAndSend(*this, GetEpisode(), SemanticVoxels);

	const double EndTime = FPlatformTime::Seconds();
	
	if (ShowCalculationTime)
		carla::log_warning("Calculation time: ", 1000.0 * (EndTime - StartTime), " ms");
}