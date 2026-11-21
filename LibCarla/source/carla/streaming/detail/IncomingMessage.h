// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/Buffer.h"
#include "carla/Debug.h"
#include "carla/streaming/detail/Types.h"

#include <boost/asio/buffer.hpp>

namespace carla {
namespace streaming {
namespace detail {

  /// Helper for reading incoming TCP messages. Allocates the whole message in
  /// a single buffer.
  ///
  /// The size comes first, in one or two reads: read size_as_buffer(), and
  /// if has_long_size() then read long_size_as_buffer() as well. See
  /// message_size_escape.
  class IncomingMessage {
  public:

    explicit IncomingMessage(Buffer &&buffer) : _message(std::move(buffer)) {}

    boost::asio::mutable_buffer size_as_buffer() {
      return boost::asio::buffer(&_short_size, sizeof(_short_size));
    }

    /// Whether the size continues in long_size_as_buffer().
    bool has_long_size() const {
      return _short_size == message_size_escape;
    }

    boost::asio::mutable_buffer long_size_as_buffer() {
      return boost::asio::buffer(&_long_size, sizeof(_long_size));
    }

    boost::asio::mutable_buffer buffer() {
      DEBUG_ASSERT(size() > 0u);
      _message.reset(size());
      return _message.buffer();
    }

    message_size_type size() const {
      return has_long_size() ? _long_size : _short_size;
    }

    auto pop() {
      return std::move(_message);
    }

  private:

    message_short_size_type _short_size = 0u;

    message_long_size_type _long_size = 0u;

    Buffer _message;
  };

} // namespace detail
} // namespace streaming
} // namespace carla
