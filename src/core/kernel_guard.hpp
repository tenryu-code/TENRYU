#pragma once

#include <cuda_runtime.h>

#include <cstdlib>

namespace tenryu::core {

// TENRYU_SYNC_EVERY_KERNEL=1 restores a full device synchronization at every
// post-launch check point (precise error attribution for debugging). Default
// (unset/0) elides the sync: on the single in-order default stream the next
// blocking CUDA call provides the same completion guarantee.
inline bool sync_every_kernel_enabled() {
  static const bool enabled = [] {
    const char* v = std::getenv("TENRYU_SYNC_EVERY_KERNEL");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return enabled;
}

// Drop-in replacement for a post-launch cudaDeviceSynchronize(): polls the
// sticky/launch error state in production; full sync under the env toggle.
inline cudaError_t debug_kernel_sync() {
  if (sync_every_kernel_enabled()) {
    return cudaDeviceSynchronize();
  }
  return cudaGetLastError();
}

}  // namespace tenryu::core
