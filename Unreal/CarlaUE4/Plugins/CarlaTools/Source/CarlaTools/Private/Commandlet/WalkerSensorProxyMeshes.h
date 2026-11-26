// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "CoreMinimal.h"

class UPackage;
class USkeletalMesh;

namespace WalkerSensorProxies
{
  // In a dry run, a missing proxy is reported successfully with OutProxy set to null.
  bool FindOrCreateProxyMesh(
      USkeletalMesh* Source, bool bDryRun, USkeletalMesh*& OutProxy, TSet<UPackage*>& OutPackages);
}
