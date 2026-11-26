// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "WalkerSensorProxyBlueprints.h"

#include "Carla/Actor/CarlaActorFactoryBlueprint.h"
#include "Components/SkeletalMeshComponent.h"
#include "Engine/Blueprint.h"
#include "Engine/BlueprintGeneratedClass.h"
#include "Engine/CollisionProfile.h"
#include "Engine/InheritableComponentHandler.h"
#include "Engine/SCS_Node.h"
#include "Engine/SimpleConstructionScript.h"
#include "GameFramework/Character.h"
#include "Kismet2/BlueprintEditorUtils.h"
#include "Kismet2/KismetEditorUtilities.h"
#include "UObject/UnrealType.h"

namespace
{
  DEFINE_LOG_CATEGORY_STATIC(LogWalkerSensorProxyBlueprints, Log, All);

  const FName ProxyName(TEXT("SensorCollisionProxy"));
  const FName ProxyProfile(TEXT("CustomSensorCollision"));

  USkeletalMeshComponent* GetWalkerMesh(UClass* Class)
  {
    ACharacter* DefaultWalker = Class ? Cast<ACharacter>(Class->GetDefaultObject()) : nullptr;
    return DefaultWalker ? DefaultWalker->GetMesh() : nullptr;
  }

  int32 GetClassDepth(const UClass* Class)
  {
    int32 Depth = 0;
    for (; Class; Class = Class->GetSuperClass())
    {
      ++Depth;
    }
    return Depth;
  }

  bool HasPoseTickSettings(const USkeletalMeshComponent& Component)
  {
    return Component.VisibilityBasedAnimTickOption == EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones
        && !Component.bEnableUpdateRateOptimizations
        && Component.PrimaryComponentTick.bStartWithTickEnabled
        && Component.PrimaryComponentTick.TickInterval == 0.0f;
  }

  void SetPoseTickSettings(USkeletalMeshComponent& Component)
  {
    Component.VisibilityBasedAnimTickOption = EVisibilityBasedAnimTickOption::AlwaysTickPoseAndRefreshBones;
    Component.bEnableUpdateRateOptimizations = false;
    Component.PrimaryComponentTick.bStartWithTickEnabled = true;
    Component.PrimaryComponentTick.TickInterval = 0.0f;
  }

  bool HasProxySettings(USkeletalMeshComponent& Component, const USkeletalMesh* Proxy, const UClass* AnimClass)
  {
    return Component.SkeletalMesh == Proxy
        && Component.GetAnimationMode() == EAnimationMode::AnimationBlueprint
        && Component.GetAnimClass() == AnimClass
        && Component.GetRelativeTransform().Equals(FTransform::Identity)
        && !Component.IsUsingAbsoluteLocation()
        && !Component.IsUsingAbsoluteRotation()
        && !Component.IsUsingAbsoluteScale()
        && Component.GetVisibleFlag()
        && Component.bHiddenInGame
        && !Component.CastShadow
        && Component.bCastHiddenShadow
        && Component.bEnablePerPolyCollision
        && (Component.GetCollisionProfileName() == ProxyProfile)
        && (Component.BodyInstance.GetCollisionEnabled() == ECollisionEnabled::QueryOnly)
        && (Component.GetCollisionObjectType() == ECC_GameTraceChannel1)
        && (Component.GetCollisionResponseToChannel(ECC_GameTraceChannel2) == ECR_Block)
        && !Component.BodyInstance.bSimulatePhysics
        && !Component.GetGenerateOverlapEvents()
        && !Component.CanEverAffectNavigation()
        && Component.PrimaryComponentTick.TickGroup == TG_PrePhysics
        && HasPoseTickSettings(Component);
  }

  void SetProxySettings(USkeletalMeshComponent& Component, USkeletalMesh* Proxy, UClass* AnimClass)
  {
    Component.Modify();
    Component.SetSkeletalMesh(Proxy);
    Component.SetAnimInstanceClass(AnimClass);
    Component.SetAbsolute(false, false, false);
    Component.SetRelativeTransform(FTransform::Identity);
    Component.SetVisibility(true);
    Component.SetHiddenInGame(true);
    Component.SetCastShadow(false);
    // Keep the hidden mesh eligible for the mesh object used to refit collision triangles.
    Component.bCastHiddenShadow = true;
    Component.bEnablePerPolyCollision = true;
    Component.BodyInstance.SetCollisionProfileName(ProxyProfile);
    Component.BodyInstance.bSimulatePhysics = false;
    Component.SetGenerateOverlapEvents(false);
    Component.SetCanEverAffectNavigation(false);
    Component.PrimaryComponentTick.TickGroup = TG_PrePhysics;
    SetPoseTickSettings(Component);
  }

  bool HasNativeMeshSettings(const USkeletalMeshComponent& Mesh)
  {
    return Mesh.GetCollisionProfileName() == UCollisionProfile::NoCollision_ProfileName
        && Mesh.BodyInstance.GetCollisionEnabled() == ECollisionEnabled::NoCollision
        && !Mesh.bEnablePerPolyCollision
        && HasPoseTickSettings(Mesh);
  }

  bool IsAttachedToNativeMesh(USCS_Node& Node, const USkeletalMeshComponent& Mesh)
  {
    return Node.bIsParentComponentNative
        && Node.ParentComponentOrVariableName == Mesh.GetFName()
        && Node.ParentComponentOwnerClassName.IsNone()
        && Node.AttachToName.IsNone()
        && Node.GetSCS()->GetRootNodes().Contains(&Node);
  }

  USCS_Node* FindProxyNode(UBlueprint& Blueprint, bool& bConflict)
  {
    TArray<UBlueprint*> Hierarchy;
    UBlueprint::GetBlueprintHierarchyFromClass(Blueprint.GeneratedClass, Hierarchy);
    USCS_Node* Result = nullptr;
    for (UBlueprint* Entry : Hierarchy)
    {
      USimpleConstructionScript* Script = Entry->SimpleConstructionScript;
      USCS_Node* Node = Script ? Script->FindSCSNode(ProxyName) : nullptr;
      if (Node)
      {
        bConflict |= Result != nullptr || !Cast<USkeletalMeshComponent>(Node->ComponentTemplate);
        Result = Node;
      }
    }
    return Result;
  }
}

bool WalkerSensorProxies::FindWalkers(TArray<FWalker>& OutWalkers)
{
  OutWalkers.Reset();
  UBlueprint* Factory = LoadObject<UBlueprint>(
      nullptr, TEXT("/Game/Carla/Blueprints/Walkers/WalkerFactory.WalkerFactory"));
  UClass* FactoryClass = Factory ? Factory->GeneratedClass : nullptr;
  auto* FactoryDefaults = FactoryClass ? Cast<ACarlaActorFactoryBlueprint>(FactoryClass->GetDefaultObject()) : nullptr;
  if (!FactoryDefaults)
  {
    UE_LOG(LogWalkerSensorProxyBlueprints, Error, TEXT("Could not load the WalkerFactory Blueprint defaults."));
    return false;
  }

  // WalkerFactory builds the definitions from the GenerateDefinitions graph.
  // This graph reads class defaults and calls the world-independent definition builder.
  const TArray<FActorDefinition> Definitions = FactoryDefaults->GetDefinitions();
  TSet<FString> SeenIds;
  bool bValid = Definitions.Num() > 0;
  if (!bValid)
  {
    UE_LOG(LogWalkerSensorProxyBlueprints, Error, TEXT("WalkerFactory has no registered walkers."));
  }
  for (const FActorDefinition& Entry : Definitions)
  {
    UClass* WalkerClass = Entry.Class.Get();
    UBlueprint* Blueprint = WalkerClass ? Cast<UBlueprint>(WalkerClass->ClassGeneratedBy) : nullptr;
    USkeletalMeshComponent* Mesh = GetWalkerMesh(WalkerClass);
    if (Entry.Id.IsEmpty() || SeenIds.Contains(Entry.Id) || !Blueprint || !Blueprint->SimpleConstructionScript
        || !Mesh || !Mesh->SkeletalMesh || Mesh->CreationMethod != EComponentCreationMethod::Native)
    {
      UE_LOG(LogWalkerSensorProxyBlueprints, Error,
          TEXT("WalkerFactory entry %s must have a unique ID and a Blueprint with a native skeletal Mesh."),
          *Entry.Id);
      bValid = false;
      continue;
    }
    SeenIds.Add(Entry.Id);
    OutWalkers.Add({Entry.Id, Blueprint, Mesh});
  }
  OutWalkers.Sort([](const FWalker& A, const FWalker& B)
  {
    const int32 DepthA = GetClassDepth(A.Blueprint->GeneratedClass);
    const int32 DepthB = GetClassDepth(B.Blueprint->GeneratedClass);
    return DepthA != DepthB ? DepthA < DepthB : A.Id < B.Id;
  });
  return bValid;
}

bool WalkerSensorProxies::ConfigureWalker(const FWalker& Walker, USkeletalMesh* Proxy, UClass* AnimClass, bool& bChanged)
{
  bChanged = false;
  UBlueprint& Blueprint = *Walker.Blueprint;
  UBlueprintGeneratedClass* GeneratedClass = CastChecked<UBlueprintGeneratedClass>(Blueprint.GeneratedClass);
  USkeletalMeshComponent* Mesh = GetWalkerMesh(GeneratedClass);
  USimpleConstructionScript* Script = Blueprint.SimpleConstructionScript;
  bool bConflict = false;
  USCS_Node* Node = FindProxyNode(Blueprint, bConflict);
  if (!Mesh || !Proxy || !AnimClass || bConflict
      || (!Node && FindFProperty<FProperty>(GeneratedClass, ProxyName)))
  {
    UE_LOG(LogWalkerSensorProxyBlueprints, Error, TEXT("%s has invalid defaults or a conflicting %s component."),
        *Blueprint.GetPathName(), *ProxyName.ToString());
    return false;
  }

  const bool bNewNode = Node == nullptr;
  if (bNewNode)
  {
    Node = Script->CreateNode(USkeletalMeshComponent::StaticClass(), ProxyName);
  }
  auto* Component = CastChecked<USkeletalMeshComponent>(Node->GetActualComponentTemplate(GeneratedClass));
  const bool bInherited = Node->GetSCS() != Script;
  const bool bFixAttachment = !IsAttachedToNativeMesh(*Node, *Mesh);
  if (bInherited && bFixAttachment)
  {
    UE_LOG(LogWalkerSensorProxyBlueprints, Error, TEXT("Inherited %s on %s is not attached to the native Mesh."),
        *ProxyName.ToString(), *Blueprint.GetPathName());
    return false;
  }

  const bool bFixProxy = !HasProxySettings(*Component, Proxy, AnimClass);
  if (!bNewNode && !bFixAttachment && !bFixProxy && HasNativeMeshSettings(*Mesh))
  {
    return true;
  }

  Blueprint.Modify();
  if (bNewNode || bFixAttachment)
  {
    Node->Modify();
    if (USCS_Node* PreviousParent = Script->FindParentNode(Node))
    {
      PreviousParent->RemoveChildNode(Node);
    }
    Node->SetParent(Mesh);
    Node->AttachToName = NAME_None;
    Script->AddNode(Node);
  }
  if (bFixProxy)
  {
    if (bInherited)
    {
      UInheritableComponentHandler* Handler = Blueprint.GetInheritableComponentHandler(true);
      const FComponentKey Key(Node);
      UActorComponent* Override = Handler->GetOverridenComponentTemplate(Key);
      Component = CastChecked<USkeletalMeshComponent>(Override ? Override : Handler->CreateOverridenComponentTemplate(Key));
    }
    SetProxySettings(*Component, Proxy, AnimClass);
  }
  if (!HasNativeMeshSettings(*Mesh))
  {
    Mesh->Modify();
    Mesh->BodyInstance.SetCollisionProfileName(UCollisionProfile::NoCollision_ProfileName);
    Mesh->bEnablePerPolyCollision = false;
    SetPoseTickSettings(*Mesh);
  }

  if (bNewNode || bFixAttachment)
  {
    FBlueprintEditorUtils::MarkBlueprintAsStructurallyModified(&Blueprint);
  }
  else
  {
    FBlueprintEditorUtils::MarkBlueprintAsModified(&Blueprint);
  }
  FKismetEditorUtilities::CompileBlueprint(&Blueprint);

  GeneratedClass = CastChecked<UBlueprintGeneratedClass>(Blueprint.GeneratedClass);
  Mesh = GetWalkerMesh(GeneratedClass);
  Node = FindProxyNode(Blueprint, bConflict);
  Component = Node ? Cast<USkeletalMeshComponent>(Node->GetActualComponentTemplate(GeneratedClass)) : nullptr;
  if (Blueprint.Status == BS_Error || bConflict || !Mesh || !Node || !Component
      || !IsAttachedToNativeMesh(*Node, *Mesh) || !HasNativeMeshSettings(*Mesh)
      || !HasProxySettings(*Component, Proxy, AnimClass))
  {
    UE_LOG(LogWalkerSensorProxyBlueprints, Error, TEXT("Compiled walker has invalid sensor proxy settings: %s."),
        *Blueprint.GetPathName());
    return false;
  }

  bChanged = true;
  return true;
}
