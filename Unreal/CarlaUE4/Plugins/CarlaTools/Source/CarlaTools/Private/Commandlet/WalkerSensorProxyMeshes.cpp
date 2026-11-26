// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#include "WalkerSensorProxyMeshes.h"

#include "AssetRegistry/AssetRegistryModule.h"
#include "AssetToolsModule.h"
#include "Engine/SkeletalMesh.h"
#include "IAssetTools.h"
#include "LODUtilities.h"
#include "Misc/PackageName.h"
#include "Modules/ModuleManager.h"
#include "PhysicsEngine/BodySetup.h"
#include "Rendering/SkeletalMeshRenderData.h"
#include "UObject/Package.h"

DEFINE_LOG_CATEGORY_STATIC(LogWalkerSensorProxyMeshes, Log, All);

namespace WalkerSensorProxies
{
  namespace
  {
    const FString PedestrianAssetPath = TEXT("/Game/Carla/Static/Pedestrian");
    constexpr int32 TargetTriangles = 5000;

    int32 GetLOD0Triangles(const USkeletalMesh& Mesh)
    {
      const FSkeletalMeshRenderData* RenderData = Mesh.GetResourceForRendering();
      return RenderData && RenderData->LODRenderData.Num() > 0 ?
          RenderData->LODRenderData[0].GetTotalFaces() : 0;
    }

    bool MatchesBindPose(const FReferenceSkeleton& Source, const FReferenceSkeleton& Proxy)
    {
      const int32 BoneCount = Source.GetRawBoneNum();
      if (BoneCount != Proxy.GetRawBoneNum())
      {
        return false;
      }
      for (int32 Bone = 0; Bone < BoneCount; ++Bone)
      {
        if (Source.GetRawRefBoneInfo()[Bone].Name != Proxy.GetRawRefBoneInfo()[Bone].Name ||
            Source.GetRawParentIndex(Bone) != Proxy.GetRawParentIndex(Bone) ||
            !Source.GetRawRefBonePose()[Bone].Equals(Proxy.GetRawRefBonePose()[Bone]))
        {
          return false;
        }
      }
      return true;
    }

    bool FindProxyAsset(const USkeletalMesh& Source, const FString& SourceFolder, FAssetData& OutAsset)
    {
      IAssetRegistry& Registry = FModuleManager::LoadModuleChecked<FAssetRegistryModule>(TEXT("AssetRegistry")).Get();
      Registry.ScanPathsSynchronous({PedestrianAssetPath, SourceFolder});

      const FString Stem = Source.GetName();
      const FName DirectName(*(Stem + TEXT("SensorProxy")));
      const FName UnderscoreName(*(Stem + TEXT("_SensorProxy")));
      const auto KeepProxyNames = [&](TArray<FAssetData>& Assets)
      {
        Assets.RemoveAll([&](const FAssetData& Asset)
        {
          return Asset.AssetName != DirectName && Asset.AssetName != UnderscoreName;
        });
      };

      TArray<FAssetData> Matches;
      Registry.GetAssetsByPath(FName(*SourceFolder), Matches);
      KeepProxyNames(Matches);
      if (Matches.Num() == 0)
      {
        Registry.GetAssetsByPath(FName(*PedestrianAssetPath), Matches, true);
        KeepProxyNames(Matches);
        Matches.RemoveAll([](const FAssetData& Asset)
        {
          return Asset.AssetClass != USkeletalMesh::StaticClass()->GetFName();
        });
      }

      if (Matches.Num() > 1)
      {
        UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("Multiple sensor proxies match %s:"), *Source.GetPathName());
        for (const FAssetData& Match : Matches)
        {
          UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("  %s"), *Match.ObjectPath.ToString());
        }
        return false;
      }
      if (Matches.Num() == 1)
      {
        OutAsset = Matches[0];
      }
      return true;
    }

    bool SimplifyProxy(USkeletalMesh& Proxy)
    {
      Proxy.bEnablePerPolyCollision = false;
      Proxy.BodySetup = nullptr;
      Proxy.SetLODSettings(nullptr);
      // RegenerateLOD only removes LODs in a call that lowers the LOD count.
      if (Proxy.GetLODNum() > 1 && !FLODUtilities::RegenerateLOD(&Proxy, 1))
      {
        return false;
      }
      FSkeletalMeshLODInfo* LOD0 = Proxy.GetLODInfo(0);
      if (!LOD0)
      {
        return false;
      }
      if (GetLOD0Triangles(Proxy) > TargetTriangles)
      {
        LOD0->BonesToRemove.Reset();
        LOD0->ReductionSettings.BaseLOD = 0;
        LOD0->ReductionSettings.TerminationCriterion = SMTC_AbsNumOfTriangles;
        LOD0->ReductionSettings.MaxNumOfTriangles = TargetTriangles;
        if (!FLODUtilities::RegenerateLOD(&Proxy, 1, true, true))
        {
          return false;
        }
      }
      const int32 Triangles = GetLOD0Triangles(Proxy);
      return Proxy.GetLODNum() == 1 && Triangles > 0 && Triangles <= TargetTriangles;
    }

    bool PrepareCollision(USkeletalMesh& Proxy, int32 Triangles, bool bDryRun, TSet<UPackage*>& OutPackages)
    {
      const bool bEnabledBefore = Proxy.bEnablePerPolyCollision;
      UBodySetup* const PreviousBodySetup = Proxy.BodySetup;
      const bool bOwnBodySetup = PreviousBodySetup && PreviousBodySetup->GetOuter() == &Proxy;
      const bool bNeedsBuild = !bEnabledBefore || !bOwnBodySetup || PreviousBodySetup->bFailedToCreatePhysicsMeshes;
      if (bDryRun && bNeedsBuild)
      {
        UE_LOG(LogWalkerSensorProxyMeshes, Display, TEXT("Would build per-poly collision for %s"), *Proxy.GetPathName());
        return true;
      }
      if (bNeedsBuild)
      {
        Proxy.Modify();
        Proxy.bEnablePerPolyCollision = true;
        if (!bOwnBodySetup)
        {
          Proxy.BodySetup = nullptr;
        }
        Proxy.BuildPhysicsData();
      }

      FTriMeshCollisionData CollisionData;
      if (!Proxy.BodySetup || Proxy.BodySetup->bFailedToCreatePhysicsMeshes ||
          !Proxy.GetPhysicsTriMeshData(&CollisionData, true) || CollisionData.Indices.Num() != Triangles)
      {
        Proxy.bEnablePerPolyCollision = bEnabledBefore;
        Proxy.BodySetup = PreviousBodySetup;
        UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("Invalid LOD 0 triangle collision for %s"), *Proxy.GetPathName());
        return false;
      }
      if (bNeedsBuild)
      {
        Proxy.MarkPackageDirty();
        OutPackages.Add(Proxy.GetOutermost());
      }
      return true;
    }
  }

  bool FindOrCreateProxyMesh(
      USkeletalMesh* Source, bool bDryRun, USkeletalMesh*& OutProxy, TSet<UPackage*>& OutPackages)
  {
    OutProxy = nullptr;
    if (!Source || !Source->Skeleton || !Source->PhysicsAsset || GetLOD0Triangles(*Source) == 0)
    {
      UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("%s needs a Skeleton, Physics Asset, and LOD 0 geometry"),
          *GetPathNameSafe(Source));
      return false;
    }

    const FString SourceFolder = FPackageName::GetLongPackagePath(Source->GetOutermost()->GetName());
    FAssetData ExistingAsset;
    if (!FindProxyAsset(*Source, SourceFolder, ExistingAsset))
    {
      return false;
    }

    USkeletalMesh* Proxy = nullptr;
    const bool bFoundExisting = ExistingAsset.IsValid();
    if (bFoundExisting)
    {
      Proxy = Cast<USkeletalMesh>(ExistingAsset.GetAsset());
      if (!Proxy)
      {
        UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("%s is not a loadable Skeletal Mesh"),
            *ExistingAsset.ObjectPath.ToString());
        return false;
      }
    }
    else
    {
      const FString ProxyName = Source->GetName() +
          (Source->GetName().EndsWith(TEXT("_")) ? TEXT("SensorProxy") : TEXT("_SensorProxy"));
      if (bDryRun)
      {
        UE_LOG(LogWalkerSensorProxyMeshes, Display, TEXT("Would generate %s/%s"), *SourceFolder, *ProxyName);
        return true;
      }

      IAssetTools& AssetTools = FModuleManager::LoadModuleChecked<FAssetToolsModule>(TEXT("AssetTools")).Get();
      Proxy = Cast<USkeletalMesh>(AssetTools.DuplicateAsset(ProxyName, SourceFolder, Source));
      if (!Proxy || !SimplifyProxy(*Proxy))
      {
        UE_LOG(LogWalkerSensorProxyMeshes, Error, TEXT("Could not reduce %s/%s to %d LOD 0 triangles"),
            *SourceFolder, *ProxyName, TargetTriangles);
        return false;
      }
    }

    const int32 Triangles = GetLOD0Triangles(*Proxy);
    if (Triangles == 0 || Source->Skeleton != Proxy->Skeleton || Source->PhysicsAsset != Proxy->PhysicsAsset ||
        !MatchesBindPose(Source->RefSkeleton, Proxy->RefSkeleton))
    {
      UE_LOG(LogWalkerSensorProxyMeshes, Error,
          TEXT("%s needs LOD 0 geometry and the same Skeleton, bind pose, and Physics Asset as %s"),
          *Proxy->GetPathName(), *Source->GetPathName());
      return false;
    }
    if (bFoundExisting && Triangles > TargetTriangles)
    {
      UE_LOG(LogWalkerSensorProxyMeshes, Warning, TEXT("Preserving existing proxy %s with %d LOD 0 triangles"),
          *Proxy->GetPathName(), Triangles);
    }
    if (!PrepareCollision(*Proxy, Triangles, bDryRun, OutPackages))
    {
      return false;
    }

    UE_LOG(LogWalkerSensorProxyMeshes, Display, TEXT("%s %s (%d LOD 0 triangles) for %s"),
        bFoundExisting ? TEXT("Reusing") : TEXT("Generated"), *Proxy->GetPathName(), Triangles, *Source->GetPathName());
    OutProxy = Proxy;
    return true;
  }
}
