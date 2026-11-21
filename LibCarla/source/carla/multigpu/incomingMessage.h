// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once
#include "carla/streaming/detail/IncomingMessage.h"

namespace carla {
namespace multigpu {

  /// The secondary servers frame their messages as the sensor streams do.
  using IncomingMessage = carla::streaming::detail::IncomingMessage;

} // namespace multigpu
} // namespace carla
