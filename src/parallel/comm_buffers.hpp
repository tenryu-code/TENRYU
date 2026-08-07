#pragma once

#include <cstddef>

namespace tenryu::parallel {

struct DeviceArray {
  void* ptr = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;
  const char* pool_tag = nullptr;

  DeviceArray() = default;
  explicit DeviceArray(const char* pool_tag_in);
  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept;
  DeviceArray& operator=(DeviceArray&& other) noexcept;

  void resize(std::size_t new_capacity);

  template <typename T>
  T* as() {
    return static_cast<T*>(ptr);
  }

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(ptr);
  }

  ~DeviceArray();
};

struct PinnedArray {
  void* ptr = nullptr;
  std::size_t size = 0;
  std::size_t capacity = 0;

  PinnedArray() = default;
  PinnedArray(const PinnedArray&) = delete;
  PinnedArray& operator=(const PinnedArray&) = delete;
  PinnedArray(PinnedArray&& other) noexcept;
  PinnedArray& operator=(PinnedArray&& other) noexcept;

  void resize(std::size_t new_capacity);

  template <typename T>
  T* as() {
    return static_cast<T*>(ptr);
  }

  template <typename T>
  const T* as() const {
    return static_cast<const T*>(ptr);
  }

  ~PinnedArray();
};

enum Direction : int {
  LEFT = 0,
  RIGHT = 1,
  BOTTOM = 2,
  TOP = 3,
  NE = 4,
  NW = 5,
  SE = 6,
  SW = 7
};

static constexpr int MAX_NEIGHBORS = 8;

struct CommBuffers {
  DeviceArray send_halo[MAX_NEIGHBORS];
  DeviceArray recv_halo[MAX_NEIGHBORS];
  PinnedArray host_send[MAX_NEIGHBORS];
  PinnedArray host_recv[MAX_NEIGHBORS];
  DeviceArray emigrant_send;
  DeviceArray emigrant_recv;

  void resize_if_needed(std::size_t required);
  bool gpu_aware_mpi = false;
};

bool detect_gpu_aware_mpi();

}  // namespace tenryu::parallel
