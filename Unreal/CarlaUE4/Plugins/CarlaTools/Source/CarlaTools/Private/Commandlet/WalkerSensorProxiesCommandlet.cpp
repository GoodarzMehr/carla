// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "WalkerSensorProxiesCommandlet.h"

#include "WalkerSensorProxyBlueprints.h"
#include "WalkerSensorProxyMeshes.h"

#include "AnimGraphNode_CopyPoseFromMesh.h"
#include "AnimGraphNode_Root.h"
#include "Animation/AnimBlueprint.h"
#include "Animation/AnimInstance.h"
#include "Animation/Skeleton.h"
#include "AnimationGraphSchema.h"
#include "AssetToolsModule.h"
#include "Components/SkeletalMeshComponent.h"
#include "EdGraph/EdGraph.h"
#include "EdGraph/EdGraphPin.h"
#include "Engine/SkeletalMesh.h"
#include "Factories/AnimBlueprintFactory.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "Misc/PackageName.h"
#include "Misc/Parse.h"
#include "UObject/Package.h"

DEFINE_LOG_CATEGORY_STATIC(LogWalkerSensorProxies, Log, All);

namespace
{
UEdGraphPin* FindPosePin(UEdGraphNode& Node, EEdGraphPinDirection Direction)
{
  for (UEdGraphPin* Pin : Node.Pins)
  {
    if (Pin->Direction == Direction && UAnimationGraphSchema::IsLocalSpacePosePin(Pin->PinType))
    {
      return Pin;
    }
  }
  return nullptr;
}

bool IsPoseCopyBlueprint(UAnimBlueprint& Blueprint, const USkeleton& Skeleton)
{
  if (Blueprint.TargetSkeleton != &Skeleton || !Blueprint.GeneratedClass ||
      Blueprint.Status == BS_Error)
  {
    return false;
  }

  for (UEdGraph* Graph : Blueprint.FunctionGraphs)
  {
    if (Graph->GetFName() != TEXT("AnimGraph") || Graph->Nodes.Num() != 2)
    {
      continue;
    }
    for (UEdGraphNode* Node : Graph->Nodes)
    {
      if (UAnimGraphNode_Root* Root = Cast<UAnimGraphNode_Root>(Node))
      {
        UEdGraphPin* Input = FindPosePin(*Root, EGPD_Input);
        if (Input && Input->LinkedTo.Num() == 1)
        {
          auto* CopyPose = Cast<UAnimGraphNode_CopyPoseFromMesh>(Input->LinkedTo[0]->GetOwningNode());
          if (CopyPose && CopyPose->Node.bUseAttachedParent && !CopyPose->Node.bUseMeshPose &&
              CopyPose->Node.RootBoneToCopy.IsNone())
          {
            UEdGraphPin* Source = CopyPose->FindPin(TEXT("SourceMeshComponent"));
            return !Source || (Source->LinkedTo.Num() == 0 && !Source->DefaultObject);
          }
        }
      }
    }
  }
  return false;
}

bool FindOrCreateAnimation(USkeleton& Skeleton, bool bDryRun, UClass*& OutClass,
    TSet<UPackage*>& Packages)
{
  const FString Directory = TEXT("/Game/Carla/Static/Pedestrian/Animations/SensorProxies");
  const FString Name = Skeleton.GetFName() == TEXT("SK_Pedestrian_Generan")
      ? TEXT("ABP_Pedestrian_SensorProxy")
      : Skeleton.GetFName() == TEXT("Skel__GEN3")
          ? TEXT("ABP_GEN3_SensorProxy")
          : FString::Printf(TEXT("ABP_%s_SensorProxy"), *Skeleton.GetName());
  const FString PackageName = Directory / Name;

  if (FindPackage(nullptr, *PackageName) || FPackageName::DoesPackageExist(PackageName))
  {
    UAnimBlueprint* Blueprint = LoadObject<UAnimBlueprint>(nullptr, *(PackageName + TEXT(".") + Name));
    if (!Blueprint || !IsPoseCopyBlueprint(*Blueprint, Skeleton))
    {
      UE_LOG(LogWalkerSensorProxies, Error, TEXT("%s exists but is not the expected pose-copy graph for %s"),
          *PackageName, *Skeleton.GetPathName());
      return false;
    }
    OutClass = Blueprint->GeneratedClass;
    UE_LOG(LogWalkerSensorProxies, Display, TEXT("Reuse animation %s"), *PackageName);
    return true;
  }

  UE_LOG(LogWalkerSensorProxies, Display, TEXT("%s animation %s for %s"),
      bDryRun ? TEXT("Would create") : TEXT("Create"), *PackageName, *Skeleton.GetPathName());
  if (bDryRun)
  {
    return true;
  }

  UAnimBlueprintFactory* Factory = NewObject<UAnimBlueprintFactory>();
  Factory->TargetSkeleton = &Skeleton;
  Factory->ParentClass = UAnimInstance::StaticClass();
  UAnimBlueprint* Blueprint = Cast<UAnimBlueprint>(FAssetToolsModule::GetModule().Get().CreateAsset(
      Name, Directory, UAnimBlueprint::StaticClass(), Factory));
  if (!Blueprint)
  {
    UE_LOG(LogWalkerSensorProxies, Error, TEXT("Could not create %s"), *PackageName);
    return false;
  }
  for (UEdGraph* Graph : Blueprint->FunctionGraphs)
  {
    if (Graph->GetFName() != TEXT("AnimGraph"))
    {
      continue;
    }
    UAnimGraphNode_Root* Root = nullptr;
    for (UEdGraphNode* Node : Graph->Nodes)
    {
      Root = Cast<UAnimGraphNode_Root>(Node);
      if (Root)
      {
        break;
      }
    }
    if (!Root)
    {
      break;
    }

    FGraphNodeCreator<UAnimGraphNode_CopyPoseFromMesh> NodeCreator(*Graph);
    auto* CopyPose = NodeCreator.CreateNode(false);
    CopyPose->Node.bUseAttachedParent = true;
    CopyPose->NodePosX = Root->NodePosX - 300;
    CopyPose->NodePosY = Root->NodePosY;
    NodeCreator.Finalize();

    UEdGraphPin* OutputPose = FindPosePin(*CopyPose, EGPD_Output);
    UEdGraphPin* InputPose = FindPosePin(*Root, EGPD_Input);
    if (OutputPose && InputPose && Graph->GetSchema()->TryCreateConnection(OutputPose, InputPose))
    {
      FKismetEditorUtilities::CompileBlueprint(Blueprint);
      if (IsPoseCopyBlueprint(*Blueprint, Skeleton))
      {
        OutClass = Blueprint->GeneratedClass;
        Packages.Add(Blueprint->GetOutermost());
        return true;
      }
    }
    break;
  }
  UE_LOG(LogWalkerSensorProxies, Error, TEXT("Could not compile the pose-copy graph for %s"), *PackageName);
  return false;
}

bool SavePackages(const TSet<UPackage*>& Packages)
{
  TArray<UPackage*> Ordered = Packages.Array();
  Ordered.Sort([](const UPackage& A, const UPackage& B) { return A.GetName() < B.GetName(); });
  for (UPackage* Package : Ordered)
  {
    const FString Filename = FPackageName::LongPackageNameToFilename(
        Package->GetName(), FPackageName::GetAssetPackageExtension());
    if (!UPackage::SavePackage(Package, nullptr, RF_Public | RF_Standalone, *Filename))
    {
      UE_LOG(LogWalkerSensorProxies, Error, TEXT("Failed to save %s"), *Filename);
      return false;
    }
    UE_LOG(LogWalkerSensorProxies, Display, TEXT("Saved %s"), *Package->GetName());
  }
  return true;
}
}

UWalkerSensorProxiesCommandlet::UWalkerSensorProxiesCommandlet()
{
  IsClient = false;
  IsServer = false;
  IsEditor = true;
  LogToConsole = true;
}

int32 UWalkerSensorProxiesCommandlet::Main(const FString& Params)
{
  using namespace WalkerSensorProxies;

  const bool bDryRun = FParse::Param(*Params, TEXT("DryRun"));
  const bool bNoSave = FParse::Param(*Params, TEXT("NoSave"));
  TArray<FWalker> Walkers;
  if (!FindWalkers(Walkers) || Walkers.Num() == 0)
  {
    return 1;
  }

  // Compiling a Blueprint can replace its generated class and component templates.
  // Snapshot mesh assets for every walker before compiling any Blueprint.
  TArray<USkeletalMesh*> SourceMeshes;
  for (const FWalker& Walker : Walkers)
  {
    SourceMeshes.Add(Walker.Mesh->SkeletalMesh);
  }

  TMap<USkeletalMesh*, USkeletalMesh*> Meshes;
  TMap<USkeleton*, UClass*> Animations;
  TSet<UPackage*> AssetPackages;
  TSet<UPackage*> WalkerPackages;
  for (USkeletalMesh* Source : SourceMeshes)
  {
    USkeleton* Skeleton = Source->Skeleton;
    if (!Skeleton)
    {
      UE_LOG(LogWalkerSensorProxies, Error, TEXT("%s has no Skeleton"), *Source->GetPathName());
      return 1;
    }
    if (!Meshes.Contains(Source) &&
        !FindOrCreateProxyMesh(Source, bDryRun, Meshes.Add(Source, nullptr), AssetPackages))
    {
      return 1;
    }
    if (!Animations.Contains(Skeleton) &&
        !FindOrCreateAnimation(*Skeleton, bDryRun, Animations.Add(Skeleton, nullptr), AssetPackages))
    {
      return 1;
    }
  }

  for (int32 Index = 0; Index < Walkers.Num(); ++Index)
  {
    const FWalker& Walker = Walkers[Index];
    USkeletalMesh* Source = SourceMeshes[Index];
    if (bDryRun)
    {
      UE_LOG(LogWalkerSensorProxies, Display, TEXT("Would configure %s (%s) from %s"),
          *Walker.Id, *Walker.Blueprint->GetName(), *Source->GetPathName());
      continue;
    }
    bool bChanged = false;
    if (!ConfigureWalker(Walker, Meshes.FindChecked(Source),
          Animations.FindChecked(Source->Skeleton), bChanged))
    {
      return 1;
    }
    if (bChanged)
    {
      WalkerPackages.Add(Walker.Blueprint->GetOutermost());
    }
  }

  UE_LOG(LogWalkerSensorProxies, Display,
      TEXT("Processed %d walker IDs, %d source meshes and %d Skeletons. %d asset packages and %d walker packages changed."),
      Walkers.Num(), Meshes.Num(), Animations.Num(), AssetPackages.Num(), WalkerPackages.Num());
  if (bDryRun || bNoSave)
  {
    UE_LOG(LogWalkerSensorProxies, Display, TEXT("No packages were saved."));
    return 0;
  }
  return SavePackages(AssetPackages) && SavePackages(WalkerPackages) ? 0 : 1;
}
