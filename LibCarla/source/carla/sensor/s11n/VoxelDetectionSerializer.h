// Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/Memory.h"
#include "carla/rpc/ActorId.h"
#include "carla/sensor/RawData.h"

#include <cstdint>
#include <cstring>

namespace carla {
namespace sensor {

  class SensorData;

namespace s11n {

  class VoxelDetectionSerializer {
  public:

    template <typename SensorT, typename EpisodeT, typename ActorListT>
    static Buffer Serialize(
        const SensorT &,
        const EpisodeT &episode,
        const ActorListT &detected_voxels) {
      const uint32_t size_in_bytes = static_cast<uint32_t>(sizeof(ActorId) * detected_voxels.Num());
      Buffer buffer{size_in_bytes};
      unsigned char *it = buffer.data();
      for (int32_t semantic_id : detected_voxels) {
        ActorId id = static_cast<ActorId>(semantic_id);
        std::memcpy(it, &id, sizeof(ActorId));
        it += sizeof(ActorId);
      }
      return buffer;
    }

    static SharedPtr<SensorData> Deserialize(RawData &&data);
  };

} // namespace s11n
} // namespace sensor
} // namespace carla