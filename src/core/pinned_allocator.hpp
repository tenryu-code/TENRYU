#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

#include "core/error.hpp"

namespace tenryu::core {

template <class T>
struct PinnedAllocator {
  using value_type = T;

  PinnedAllocator() = default;

  template <class U>
  PinnedAllocator(const PinnedAllocator<U>&) {}

  T* allocate(const std::size_t n) {
    T* ptr = nullptr;
    const cudaError_t err = cudaHostAlloc(
        reinterpret_cast<void**>(&ptr), n * sizeof(T), cudaHostAllocDefault);
    TENRYU_ASSERT(err == cudaSuccess && ptr != nullptr,
                  "PinnedAllocator::allocate cudaHostAlloc failed");
    return ptr;
  }

  void deallocate(T* p, std::size_t) {
    if (p != nullptr) {
      const cudaError_t err = cudaFreeHost(p);
      TENRYU_ASSERT(err == cudaSuccess,
                    "PinnedAllocator::deallocate cudaFreeHost failed");
    }
  }

  template <class U>
  bool operator==(const PinnedAllocator<U>&) const noexcept {
    return true;
  }

  template <class U>
  bool operator!=(const PinnedAllocator<U>&) const noexcept {
    return false;
  }
};

template <class T>
using PinnedVector = std::vector<T, PinnedAllocator<T>>;

}  // namespace tenryu::core
