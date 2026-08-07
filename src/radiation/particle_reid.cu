#include "radiation/particle_reid.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kBlockSize = 256;

enum ReIDCounter : int {
  kChecked = 0,
  kKept = 1,
  kUpdated = 2,
  kSkippedDead = 3,
  kSkippedDDMC = 4,
  kSkippedNaN = 5,
  kBinarySearch = 6,
  kClamped = 7,
  kNumCounters = 8,
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline bool contains_position_1d(const double* __restrict__ node_r,
                                            const int cell,
                                            const int n_cells,
                                            const double r) {
  if (cell < 0 || cell >= n_cells) {
    return false;
  }
  return r >= node_r[cell] && r <= node_r[cell + 1];
}

__device__ int binary_search_cell_1d(const double* __restrict__ node_r,
                                     const int n_cells,
                                     const double r,
                                     int* __restrict__ counters) {
  int lo = 0;
  int hi = n_cells - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (r < node_r[mid]) {
      hi = mid - 1;
    } else if (r > node_r[mid + 1]) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }

  atomicAdd(&counters[kClamped], 1);
  if (r <= node_r[0]) {
    return 0;
  }
  return n_cells - 1;
}

__global__ void reidentify_particles_1d_kernel(std::int32_t* __restrict__ cell_id,
                                               const double* __restrict__ pos_r,
                                               const std::uint8_t* __restrict__ mode,
                                               const std::uint8_t* __restrict__ alive,
                                               const double* __restrict__ node_r,
                                               const int n_particles,
                                               const int n_cells,
                                               int* __restrict__ counters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  if (alive[tid] != static_cast<std::uint8_t>(kAlive)) {
    atomicAdd(&counters[kSkippedDead], 1);
    return;
  }
  if (mode[tid] == static_cast<std::uint8_t>(kModeDDMC)) {
    atomicAdd(&counters[kSkippedDDMC], 1);
    return;
  }

  const double r = pos_r[tid];
  if (!isfinite(r)) {
    atomicAdd(&counters[kSkippedNaN], 1);
    return;
  }

  atomicAdd(&counters[kChecked], 1);
  const int old_cell = cell_id[tid];
  if (contains_position_1d(node_r, old_cell, n_cells, r)) {
    atomicAdd(&counters[kKept], 1);
    return;
  }

  for (int offset = 1; offset <= 2; ++offset) {
    const int left = old_cell - offset;
    if (contains_position_1d(node_r, left, n_cells, r)) {
      cell_id[tid] = left;
      atomicAdd(&counters[kUpdated], 1);
      return;
    }
    const int right = old_cell + offset;
    if (contains_position_1d(node_r, right, n_cells, r)) {
      cell_id[tid] = right;
      atomicAdd(&counters[kUpdated], 1);
      return;
    }
  }

  atomicAdd(&counters[kBinarySearch], 1);
  const int new_cell = binary_search_cell_1d(node_r, n_cells, r, counters);
  if (new_cell != old_cell) {
    cell_id[tid] = new_cell;
    atomicAdd(&counters[kUpdated], 1);
  } else {
    atomicAdd(&counters[kKept], 1);
  }
}

}  // namespace

ParticleReIDStats reidentify_finite_position_particles_1d_cuda(
    PhotonPool& pool,
    const double* node_r,
    const int n_cells,
    const int n_particles) {
  const int count = (n_particles >= 0) ? n_particles : pool.n_alive;
  if (count <= 0) {
    return {};
  }
  TENRYU_ASSERT(count <= pool.capacity,
                "particle re-ID requires n_particles <= pool capacity");
  TENRYU_ASSERT(n_cells > 0, "particle re-ID requires positive n_cells");
  TENRYU_ASSERT(pool.cell_id != nullptr, "particle re-ID requires cell_id");
  TENRYU_ASSERT(pool.pos_r != nullptr, "particle re-ID requires pos_r");
  TENRYU_ASSERT(pool.mode != nullptr, "particle re-ID requires mode");
  TENRYU_ASSERT(pool.alive != nullptr, "particle re-ID requires alive");
  TENRYU_ASSERT(node_r != nullptr, "particle re-ID requires node_r");

  int* d_counters = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_counters),
                        sizeof(int) * static_cast<std::size_t>(kNumCounters)),
             "particle re-ID cudaMalloc counters failed");
  cuda_check(cudaMemset(d_counters, 0, sizeof(int) * static_cast<std::size_t>(kNumCounters)),
             "particle re-ID cudaMemset counters failed");

  const int blocks = (count + kBlockSize - 1) / kBlockSize;
  reidentify_particles_1d_kernel<<<blocks, kBlockSize>>>(
      pool.cell_id, pool.pos_r, pool.mode, pool.alive, node_r, count, n_cells, d_counters);
  cuda_check(cudaGetLastError(), "particle re-ID kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "particle re-ID kernel execution failed");

  std::array<int, kNumCounters> counters{};
  cuda_check(cudaMemcpy(counters.data(),
                        d_counters,
                        sizeof(int) * counters.size(),
                        cudaMemcpyDeviceToHost),
             "particle re-ID copy counters failed");
  cuda_check(cudaFree(d_counters), "particle re-ID cudaFree counters failed");

  ParticleReIDStats stats;
  stats.checked = counters[kChecked];
  stats.kept = counters[kKept];
  stats.updated = counters[kUpdated];
  stats.skipped_dead = counters[kSkippedDead];
  stats.skipped_ddmc = counters[kSkippedDDMC];
  stats.skipped_nan = counters[kSkippedNaN];
  stats.binary_search = counters[kBinarySearch];
  stats.clamped = counters[kClamped];
  return stats;
}

}  // namespace tenryu::radiation
