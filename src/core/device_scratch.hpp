#pragma once

#include <cstddef>

namespace tenryu::core {

// Persistent device scratch buffers keyed by a stable tag (string literal,
// unique per call site). Grow-only; contents are NOT zeroed (same contract
// as cudaMalloc -- call sites keep their own memsets). Single-host-thread
// use only. Buffers live until device_scratch_shutdown() (or process end);
// shutdown also frees pinned-host scratch buffers.
void* device_scratch_acquire(const char* tag, std::size_t bytes);
// Persistent PINNED HOST scratch buffers, same contract as device_scratch_acquire
// (grow-only, NOT zeroed, single-host-thread, freed at device_scratch_shutdown()).
// Backed by cudaHostAlloc(cudaHostAllocPortable) for fast D2H/H2D staging.
void* host_pinned_scratch_acquire(const char* tag, std::size_t bytes);
void device_scratch_shutdown();

}  // namespace tenryu::core
