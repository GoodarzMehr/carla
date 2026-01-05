// Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/sensor/data/Array.h"

#include <cstdint>

namespace carla {
namespace sensor {
namespace data {

  class VoxelDetectionEvent : public Array<uint8_t> {
  
  public:
  
    explicit VoxelDetectionEvent(RawData &&data)
      : Array<uint8_t>(0u, std::move(data)) {}
  };

} // namespace data
} // namespace sensor
} // namespace carla