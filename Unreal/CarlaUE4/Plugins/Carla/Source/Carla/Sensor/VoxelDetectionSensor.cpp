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

	FActorVariation UseTraceComplex;
    UseTraceComplex.Id = TEXT("use_complex_collision");
    UseTraceComplex.Type = EActorAttributeType::Bool;
    UseTraceComplex.RecommendedValues = { TEXT("True") };
    UseTraceComplex.bRestrictToRecommended = false;

	FActorVariation UseZTop;
    UseZTop.Id = TEXT("use_z_top");
    UseZTop.Type = EActorAttributeType::Bool;
    UseZTop.RecommendedValues = { TEXT("True") };
    UseZTop.bRestrictToRecommended = false;

	FActorVariation UseZBottom;
    UseZBottom.Id = TEXT("use_z_bottom");
    UseZBottom.Type = EActorAttributeType::Bool;
    UseZBottom.RecommendedValues = { TEXT("True") };
    UseZBottom.bRestrictToRecommended = false;

	FActorVariation UseXFront;
    UseXFront.Id = TEXT("use_x_front");
    UseXFront.Type = EActorAttributeType::Bool;
    UseXFront.RecommendedValues = { TEXT("True") };
    UseXFront.bRestrictToRecommended = false;

	FActorVariation UseXBack;
    UseXBack.Id = TEXT("use_x_back");
    UseXBack.Type = EActorAttributeType::Bool;
    UseXBack.RecommendedValues = { TEXT("True") };
    UseXBack.bRestrictToRecommended = false;

	FActorVariation UseYRight;
	UseYRight.Id = TEXT("use_y_right");
	UseYRight.Type = EActorAttributeType::Bool;
	UseYRight.RecommendedValues = { TEXT("True") };
	UseYRight.bRestrictToRecommended = false;

	FActorVariation UseYLeft;
	UseYLeft.Id = TEXT("use_y_left");
	UseYLeft.Type = EActorAttributeType::Bool;
	UseYLeft.RecommendedValues = { TEXT("True") };
	UseYLeft.bRestrictToRecommended = false;

    Definition.Variations.Append(
		{
			Range,
			Top,
			Bottom,
			VoxelSize,
			SelfIgnore,
			DrawDebug,
			UseTraceComplex,
			UseZTop,
			UseZBottom,
			UseXFront,
			UseXBack,
			UseYRight,
			UseYLeft
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
	
	UseTraceComplex = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_complex_collision", Description.Variations, true);
	
	UseZTop = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_z_top", Description.Variations, true);
	
	UseZBottom = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_z_bottom", Description.Variations, true);
	
	UseXFront = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_x_front", Description.Variations, true);
	
	UseXBack = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_x_back", Description.Variations, true);

	UseYRight = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_y_right", Description.Variations, true);

	UseYLeft = UActorBlueprintFunctionLibrary::RetrieveActorAttributeToBool("use_y_left", Description.Variations, true);

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
    
    const int32 TotalVoxels = GridSizeX * GridSizeY * GridSizeZ;
    
    // Initialize voxel grids with 0 (empty).
    TArray<uint8> SemanticVoxels;
    SemanticVoxels.SetNumZeroed(TotalVoxels);

	TArray<uint8> SemanticVoxelsZ;
    SemanticVoxelsZ.SetNumZeroed(TotalVoxels);
    
    // Set up collision query.
    FCollisionQueryParams QueryParams(FName(TEXT("VoxelTrace")), true, this);
    
	QueryParams.bTraceComplex = UseTraceComplex;
    QueryParams.bReturnPhysicalMaterial = false;
    
    if (SelfIgnore && GetOwner())
    {
        QueryParams.AddIgnoredActor(GetOwner());
    }
    
	const float HalfVoxel = BoxSize * 0.5f;
    const float InvBoxSize = 1.0f / BoxSize;

	const FVector BoxExtent(HalfVoxel, HalfVoxel, HalfVoxel);
	const FQuat BoxRotation = FQuat::Identity;

	FCollisionObjectQueryParams ObjectParams(FCollisionObjectQueryParams::AllObjects);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Vehicle);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Pawn);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Camera);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Visibility);
	ObjectParams.RemoveObjectTypesToQuery(ECC_PhysicsBody);
	ObjectParams.RemoveObjectTypesToQuery(ECC_Destructible);
    
	GetWorld()->GetPhysicsScene()->GetPxScene()->lockRead();

	const FTransform SensorTransform = GetTransform();
    const FTransform InvSensorTransform = SensorTransform.Inverse();
    
	// First, sweep boxes along the Z axis (top-to-bottom and bottom-to-top)
	// and record the hits in each voxel column. Then, fill the voxels between
	// the hit pairs.
	{
        TRACE_CPUPROFILER_EVENT_SCOPE(VoxelRayCasting);
        
        ParallelFor(GridSizeX * GridSizeY, [&](int32 Index)
        {
			const int32 X = Index / GridSizeY;
            const int32 Y = Index % GridSizeY;
            
            const FVector LocalColumnBase(-BoxRange + (X + 0.5f) * BoxSize, -BoxRange + (Y + 0.5f) * BoxSize, 0.0f);
            
            const FVector LocalStart(LocalColumnBase.X, LocalColumnBase.Y, Top + HalfVoxel);
            const FVector LocalEnd(LocalColumnBase.X, LocalColumnBase.Y, Bottom - HalfVoxel);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            TArray<FHitResult> Hits;

			bool FirstHit = false;
			bool FirstHitGround = false;
			bool OnlyHitGround = true;

			int32 FirstZ = -1;
            
            if (UseZTop)
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldStart,
					WorldEnd,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							if (!FirstHit)
							{
								FirstHit = true;
								
								if (Hit.Component->CustomDepthStencilValue == 1 || 
									Hit.Component->CustomDepthStencilValue == 2 || 
									Hit.Component->CustomDepthStencilValue == 10 || 
									Hit.Component->CustomDepthStencilValue == 24 || 
									Hit.Component->CustomDepthStencilValue == 25)
									FirstHitGround = true;
							}
							
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
							
							const int32 Z = FMath::Clamp(FMath::FloorToInt((LocalHit.Z - Bottom) * InvBoxSize), 0, GridSizeZ - 1);
							
							if (FirstZ == -1)
								FirstZ = Z;
							
							if (Z < FirstZ)
								OnlyHitGround = false;
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
								SemanticVoxels[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}

			Hits.Reset();

			if (UseZBottom && !(FirstHitGround && OnlyHitGround))
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldEnd,
					WorldStart,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
							
							const int32 Z = FMath::Clamp(FMath::FloorToInt((LocalHit.Z - Bottom) * InvBoxSize), 0, GridSizeZ - 1);
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxelsZ[FlatIndex]])
								SemanticVoxelsZ[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}
			
			if (!(FirstHitGround && OnlyHitGround))
			{
				// Find pairs of voxels from the two sweeps and fill between
				// them if they have the same semantic class.
				int32 StartZ = -1;
				uint8 StartLabel = 0;
				
				for (int32 Z = 0; Z < GridSizeZ; ++Z)
				{
					const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
					const uint8 TopDownLabel = SemanticVoxels[FlatIndex];
					const uint8 BottomUpLabel = SemanticVoxelsZ[FlatIndex];
					
					// Check if we have a voxel from the bottom-up sweep
					// (potential floor). If so, mark it as the start of a
					// potential pair.
					if (BottomUpLabel > 0)
					{
						StartZ = Z;
						StartLabel = BottomUpLabel;
					}
					
					// Check if we have a voxel from the top-down sweep
					// (potential ceiling). If so, see if we have a matching
					// start voxel to form a pair.
					if (TopDownLabel > 0)
					{
						if (StartZ >= 0 && StartLabel == TopDownLabel)
						{
							for (int32 FillZ = StartZ; FillZ < Z; ++FillZ)
							{
								const int32 FillIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + FillZ;
								
								if (SemanticPriority[StartLabel] > SemanticPriority[SemanticVoxels[FillIndex]])
									SemanticVoxels[FillIndex] = StartLabel;
							}
						}
						
						// Reset for the next potential pair.
						StartZ = -1;
						StartLabel = 0;
					}
				}
			}
			
			if (!UseZTop)
			{
				for (int32 Z = 0; Z < GridSizeZ; ++Z)
				{
					const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;

					if (SemanticPriority[SemanticVoxelsZ[FlatIndex]] > SemanticPriority[SemanticVoxels[FlatIndex]])
						SemanticVoxels[FlatIndex] = SemanticVoxelsZ[FlatIndex];
				}
			}
        });
    }

	// Next, sweep boxes along the X axis (front-to-back and back-to-front)
	// and record the hits in each voxel row.
	{
        ParallelFor(GridSizeY * GridSizeZ, [&](int32 Index)
        {
            const int32 Y = Index / GridSizeZ;
            const int32 Z = Index % GridSizeZ;
            
            const FVector LocalRowBase(0.0f, -BoxRange + (Y + 0.5f) * BoxSize, Bottom + (Z + 0.5f) * BoxSize);
            
            const FVector LocalStart(BoxRange + HalfVoxel, LocalRowBase.Y, LocalRowBase.Z);
            const FVector LocalEnd(-BoxRange - HalfVoxel, LocalRowBase.Y, LocalRowBase.Z);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            TArray<FHitResult> Hits;

			if (UseXFront)
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldStart,
					WorldEnd,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);

							const int32 X = FMath::Clamp(FMath::FloorToInt((LocalHit.X + BoxRange) * InvBoxSize), 0, GridSizeX - 1);
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
								SemanticVoxels[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}

			Hits.Reset();
            
            if (UseXBack)
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldEnd,
					WorldStart,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
							
							const int32 X = FMath::Clamp(FMath::FloorToInt((LocalHit.X + BoxRange) * InvBoxSize), 0, GridSizeX - 1);
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
								SemanticVoxels[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}
        });
	}
        
    // Finally, sweep boxes along the Y axis (right-to-left and left-to-right)
	// and record the hits in each voxel row.
	{
        ParallelFor(GridSizeX * GridSizeZ, [&](int32 Index)
        {
            const int32 X = Index / GridSizeZ;
            const int32 Z = Index % GridSizeZ;
            
            const FVector LocalRowBase(-BoxRange + (X + 0.5f) * BoxSize, 0.0f, Bottom + (Z + 0.5f) * BoxSize);
            
            const FVector LocalStart(LocalRowBase.X, BoxRange + HalfVoxel, LocalRowBase.Z);
            const FVector LocalEnd(LocalRowBase.X, -BoxRange - HalfVoxel, LocalRowBase.Z);
            
            const FVector WorldStart = SensorTransform.TransformPosition(LocalStart);
            const FVector WorldEnd = SensorTransform.TransformPosition(LocalEnd);
            
            TArray<FHitResult> Hits;

			if (UseYRight)
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldStart,
					WorldEnd,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
							
							const int32 Y = FMath::Clamp(FMath::FloorToInt((LocalHit.Y + BoxRange) * InvBoxSize), 0, GridSizeY - 1);
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
								SemanticVoxels[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}

			Hits.Reset();
            
            if (UseYLeft)
			{
				bool bHit = GetWorld()->ParallelSweepMultiByObjectType(
					Hits,
					WorldEnd,
					WorldStart,
					BoxRotation,
					ObjectParams,
					FCollisionShape::MakeBox(BoxExtent),
					QueryParams
				);
				
				if (bHit)
				{
					for (const FHitResult& Hit : Hits)
					{
						if (Hit.bBlockingHit && Hit.Component.IsValid())
						{
							const FVector LocalHit = InvSensorTransform.TransformPosition(Hit.ImpactPoint);
							
							const int32 Y = FMath::Clamp(FMath::FloorToInt((LocalHit.Y + BoxRange) * InvBoxSize), 0, GridSizeY - 1);
							
							const int32 FlatIndex = X * GridSizeY * GridSizeZ + Y * GridSizeZ + Z;
							
							if (SemanticPriority[Hit.Component->CustomDepthStencilValue] > SemanticPriority[SemanticVoxels[FlatIndex]])
								SemanticVoxels[FlatIndex] = Hit.Component->CustomDepthStencilValue;
						}
					}
				}
			}
        });
    }
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
}