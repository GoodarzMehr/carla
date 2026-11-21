// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/Buffer.h"

#include <cstdint>
#include <type_traits>

namespace carla {
namespace streaming {
namespace detail {

  using stream_id_type = uint32_t;

  /// Size of a message in bytes.
  using message_size_type = uint64_t;

  static_assert(
      std::is_same<message_size_type, Buffer::size_type>::value,
      "uint type mismatch!");

  /// On the wire, a message starts with its size as a 32-bit integer. A
  /// message of message_size_escape bytes or more starts with
  /// message_size_escape instead, followed by its size as a 64-bit integer.
  /// Smaller messages are therefore sent exactly as with 32-bit sizes.
  using message_short_size_type = uint32_t;

  using message_long_size_type = uint64_t;

  constexpr message_short_size_type message_size_escape = 0xFFFFFFFFu;

} // namespace detail
} // namespace streaming
} // namespace carla
