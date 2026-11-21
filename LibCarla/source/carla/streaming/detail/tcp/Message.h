// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "carla/ListView.h"
#include "carla/Buffer.h"
#include "carla/BufferView.h"
#include "carla/Debug.h"
#include "carla/NonCopyable.h"
#include "carla/streaming/detail/Types.h"

#include <boost/asio/buffer.hpp>

#include <array>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <string>

namespace carla {
namespace streaming {
namespace detail {
namespace tcp {

  /// Serialization of a set of buffers to be sent over a TCP socket as a single
  /// message. Template paramenter @a MaxNumberOfBuffers imposes a compile-time
  /// limit on the maximum number of buffers that can be included in a single
  /// message.
  template <size_t MaxNumberOfBuffers>
  class MessageTmpl
    : public std::enable_shared_from_this<MessageTmpl<MaxNumberOfBuffers>>,
      private NonCopyable {
  public:

    static constexpr size_t max_size() {
      return MaxNumberOfBuffers;
    }

  private:

    MessageTmpl(size_t) {}

    template <typename... Buffers>
    MessageTmpl(size_t size, SharedBufferView buffer, Buffers... buffers)
      : MessageTmpl(size, buffers...) {
      ++_number_of_buffers;
      _total_size += buffer->size();
      _buffer_views[1u + size - _number_of_buffers] = buffer->cbuffer();
      _buffers[size - _number_of_buffers] = buffer;
    }

  public:

    template <typename... Buffers>
    MessageTmpl(SharedBufferView buf, Buffers... buffers)
      : MessageTmpl(sizeof...(Buffers) + 1u, buf, buffers...) {
      static_assert(sizeof...(Buffers) < max_size(), "Too many buffers!");
      _buffer_views[0u] = EncodeSize();
    }

    /// Size in bytes of the message excluding the header.
    auto size() const noexcept {
      return _total_size;
    }

    /// Size in bytes of the header that carries the message size.
    size_t header_size() const noexcept {
      return boost::asio::buffer_size(_buffer_views[0u]);
    }

    bool empty() const noexcept {
      return size() == 0u;
    }

    auto GetBufferSequence() const {
      auto begin = _buffer_views.begin();
      return MakeListView(begin, begin + _number_of_buffers + 1u);
    }

  private:

    /// The size in its short form, or the escape value followed by the long
    /// form; see message_size_escape.
    boost::asio::const_buffer EncodeSize() {
      if (_total_size < message_size_escape) {
        const auto size = static_cast<message_short_size_type>(_total_size);
        std::memcpy(_header.data(), &size, sizeof(size));
        return boost::asio::buffer(_header.data(), sizeof(size));
      }
      const message_short_size_type escape = message_size_escape;
      const message_long_size_type size = _total_size;
      std::memcpy(_header.data(), &escape, sizeof(escape));
      std::memcpy(_header.data() + sizeof(escape), &size, sizeof(size));
      return boost::asio::buffer(_header.data(), sizeof(escape) + sizeof(size));
    }

    size_t _number_of_buffers = 0u;

    message_size_type _total_size = 0u;

    std::array<unsigned char, sizeof(message_short_size_type) + sizeof(message_long_size_type)> _header;

    std::array<SharedBufferView, MaxNumberOfBuffers> _buffers;

    std::array<boost::asio::const_buffer, MaxNumberOfBuffers + 1u> _buffer_views;
  };

  /// A TCP message containing a maximum of 2 buffers. This is optimized for a
  /// header and body sort of messages.
  using Message = MessageTmpl<2u>;

} // namespace tcp
} // namespace detail
} // namespace streaming
} // namespace carla
