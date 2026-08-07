#include "core/device_scratch.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>

#include "core/error.hpp"

namespace tenryu::core {
namespace {

struct ScratchEntry {
  void* ptr = nullptr;
  std::size_t bytes = 0;
};

std::map<std::string, ScratchEntry>& scratch_map() {
  static std::map<std::string, ScratchEntry> m;
  return m;
}

std::map<std::string, ScratchEntry>& pinned_map() {
  static std::map<std::string, ScratchEntry> m;
  return m;
}

}  // namespace

void* device_scratch_acquire(const char* tag, const std::size_t bytes) {
  TENRYU_ASSERT(tag != nullptr && bytes > 0,
                "device_scratch_acquire requires a tag and bytes > 0");
  auto& entry = scratch_map()[tag];
  if (entry.bytes < bytes) {
    if (entry.ptr != nullptr) {
      const cudaError_t err_free = cudaFree(entry.ptr);
      TENRYU_ASSERT(err_free == cudaSuccess,
                    "device_scratch_acquire: cudaFree during grow failed");
      entry.ptr = nullptr;
      entry.bytes = 0;
    }
    const cudaError_t err = cudaMalloc(&entry.ptr, bytes);
    TENRYU_ASSERT(err == cudaSuccess && entry.ptr != nullptr,
                  "device_scratch_acquire: cudaMalloc failed");
    entry.bytes = bytes;
  }
  static const bool poison = [] {
    const char* v = std::getenv("TENRYU_SCRATCH_POISON");
    return v != nullptr && v[0] == '1';
  }();
  if (poison) {
    // BUG-22 hunt: fill EVERY acquire with a recognizable pattern so any
    // read-before-write surfaces deterministically (0xFF bytes = NaN as
    // double). Debug-only knob; default path untouched.
    const cudaError_t err_fill = cudaMemset(entry.ptr, 0xFF, bytes);
    TENRYU_ASSERT(err_fill == cudaSuccess,
                  "device_scratch_acquire: poison memset failed");
  }
  return entry.ptr;
}

void* host_pinned_scratch_acquire(const char* tag, const std::size_t bytes) {
  TENRYU_ASSERT(tag != nullptr && bytes > 0,
                "host_pinned_scratch_acquire requires a tag and bytes > 0");
  auto& entry = pinned_map()[tag];
  if (entry.bytes < bytes) {
    if (entry.ptr != nullptr) {
      const cudaError_t err_free = cudaFreeHost(entry.ptr);
      TENRYU_ASSERT(
          err_free == cudaSuccess,
          "host_pinned_scratch_acquire: cudaFreeHost during grow failed");
      entry.ptr = nullptr;
      entry.bytes = 0;
    }
    const cudaError_t err =
        cudaHostAlloc(&entry.ptr, bytes, cudaHostAllocPortable);
    TENRYU_ASSERT(err == cudaSuccess && entry.ptr != nullptr,
                  "host_pinned_scratch_acquire: cudaHostAlloc failed");
    entry.bytes = bytes;
  }
  return entry.ptr;
}

void device_scratch_shutdown() {
  for (auto& kv : scratch_map()) {
    if (kv.second.ptr != nullptr) {
      static_cast<void>(cudaFree(kv.second.ptr));
      kv.second.ptr = nullptr;
      kv.second.bytes = 0;
    }
  }
  scratch_map().clear();
  for (auto& kv : pinned_map()) {
    if (kv.second.ptr != nullptr) {
      static_cast<void>(cudaFreeHost(kv.second.ptr));
      kv.second.ptr = nullptr;
      kv.second.bytes = 0;
    }
  }
  pinned_map().clear();
}

}  // namespace tenryu::core
