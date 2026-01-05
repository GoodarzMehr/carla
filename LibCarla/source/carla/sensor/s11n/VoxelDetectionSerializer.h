// Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/Memory.h"
#include "carla/sensor/RawData.h"

#include <cstdint>
#include <cstring>

namespace carla {
namespace sensor {

  class SensorData;

namespace s11n {

  class VoxelDetectionSerializer {
  
  public:

    template <typename SensorT, typename EpisodeT, typename VoxelArrayT>
    
    static Buffer Serialize(const SensorT &, const EpisodeT &, const VoxelArrayT &voxels) {
      const uint32_t size_in_bytes = voxels.Num() * sizeof(uint8_t);
      
      Buffer buffer{size_in_bytes};
      
      if (size_in_bytes > 0) {
        std::memcpy(buffer.data(), voxels.GetData(), size_in_bytes);
      }
      
      return buffer;
    }

    static SharedPtr<SensorData> Deserialize(RawData &&data);
  };

} // namespace s11n
} // namespace sensor
} // namespace carla