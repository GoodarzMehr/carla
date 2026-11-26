// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "Commandlets/Commandlet.h"
#include "WalkerSensorProxiesCommandlet.generated.h"

UCLASS()
class UWalkerSensorProxiesCommandlet : public UCommandlet
{
  GENERATED_BODY()

public:
  UWalkerSensorProxiesCommandlet();

  // -DryRun reports the plan. -NoSave also reduces meshes and compiles Blueprints in memory.
  int32 Main(const FString& Params) override;
};
