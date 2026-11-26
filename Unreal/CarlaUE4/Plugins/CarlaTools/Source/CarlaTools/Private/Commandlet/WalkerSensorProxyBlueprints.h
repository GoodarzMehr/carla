// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"

class UBlueprint;
class USkeletalMesh;
class USkeletalMeshComponent;

namespace WalkerSensorProxies
{
  struct FWalker
  {
    FString Id;
    UBlueprint* Blueprint = nullptr;
    USkeletalMeshComponent* Mesh = nullptr;
  };

  bool FindWalkers(TArray<FWalker>& OutWalkers);

  bool ConfigureWalker(const FWalker& Walker, USkeletalMesh* Proxy, UClass* AnimClass, bool& bChanged);
}
