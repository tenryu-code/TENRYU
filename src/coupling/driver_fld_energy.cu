#include "coupling/driver_fld_energy.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::coupling {
namespace {

constexpr int kBlockSize = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

struct FldEnergyScratch {
  ~FldEnergyScratch() {
    release();
  }

  FldEnergyScratch() = default;
  FldEnergyScratch(const FldEnergyScratch&) = delete;
  FldEnergyScratch& operator=(const FldEnergyScratch&) = delete;

  void ensure_capacity(const std::size_t n_contrib,
                       const std::size_t n_block_sums) {
    int current_device = 0;
    cuda_check(cudaGetDevice(&current_device),
               "FLD radiation energy cudaGetDevice failed");
    if (current_device != device_id) {
      release();
      device_id = current_device;
    }
    if (contrib_capacity < n_contrib) {
      if (d_contrib != nullptr) {
        cuda_check(cudaFree(d_contrib),
                   "FLD radiation energy cudaFree contrib resize failed");
      }
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_contrib),
                            n_contrib * sizeof(double)),
                 "FLD radiation energy cudaMalloc contrib failed");
      contrib_capacity = n_contrib;
    }
    if (block_sums_capacity < n_block_sums) {
      if (d_block_sums != nullptr) {
        cuda_check(cudaFree(d_block_sums),
                   "FLD radiation energy cudaFree block sums resize failed");
      }
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_block_sums),
                            n_block_sums * sizeof(double)),
                 "FLD radiation energy cudaMalloc block sums failed");
      block_sums_capacity = n_block_sums;
    }
    if (d_counts == nullptr) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_counts),
                            2U * sizeof(int)),
                 "FLD radiation energy cudaMalloc counts failed");
    }
  }

  void release() {
    if (d_block_sums != nullptr) {
      (void)cudaFree(d_block_sums);
      d_block_sums = nullptr;
    }
    if (d_contrib != nullptr) {
      (void)cudaFree(d_contrib);
      d_contrib = nullptr;
    }
    if (d_counts != nullptr) {
      (void)cudaFree(d_counts);
      d_counts = nullptr;
    }
    contrib_capacity = 0;
    block_sums_capacity = 0;
    device_id = -1;
  }

  double* d_contrib = nullptr;
  double* d_block_sums = nullptr;
  int* d_counts = nullptr;  // [0]=negative E entries, [1]=nonfinite/invalid
  std::size_t contrib_capacity = 0;
  std::size_t block_sums_capacity = 0;
  int device_id = -1;
};

FldEnergyScratch& fld_energy_scratch() {
  static FldEnergyScratch scratch;
  return scratch;
}

// AI review k15 C-3 (2026-07-26): an energy AUDIT must not sanitize the
// field it audits. Negative E_g now contributes with its sign and is
// counted; non-finite E or an invalid volume is zero-contribution but
// counted so the failure is surfaced instead of silently folded into a
// clean-looking total. Bit-identical while the solvers' nonnegative
// publish invariant holds (every 1D FLD/SN publish clamps E >= 0).
__device__ inline void fld_rad_energy_contrib_kernel_body(
    const std::size_t k,
    double* __restrict__ contrib,
    const double* __restrict__ rad_E,
    const double* __restrict__ vol,
    const std::size_t n_groups,
    int* __restrict__ negative_count,
    int* __restrict__ invalid_count) {
  const std::size_t c = k / n_groups;
  const double V = vol[c];
  const double E = rad_E[k];
  if (!isfinite(V) || !(V > 0.0) || !isfinite(E)) {
    contrib[k] = 0.0;
    if (invalid_count != nullptr) {
      atomicAdd(invalid_count, 1);
    }
    return;
  }
  if (E < 0.0 && negative_count != nullptr) {
    atomicAdd(negative_count, 1);
  }
  contrib[k] = E * V;
}

__global__ void fld_rad_energy_contrib_kernel(
    double* __restrict__ contrib,
    const double* __restrict__ rad_E,
    const double* __restrict__ vol,
    const std::size_t n_cell_groups,
    const std::size_t n_groups,
    const std::size_t c_begin,
    const std::size_t c_end,
    int* __restrict__ negative_count,
    int* __restrict__ invalid_count) {
  for (std::size_t k = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       k < n_cell_groups;
       k += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    // Mode-aware ledger window (Option C, M18e): non-owned cells
    // contribute zero, so the per-rank sum is a disjoint partial that
    // the budget Allreduce completes (2D distributed); the 1D replicated
    // and serial paths pass the full range.
    const std::size_t c = k / n_groups;
    if (c < c_begin || c >= c_end) {
      contrib[k] = 0.0;
      continue;
    }
    fld_rad_energy_contrib_kernel_body(k, contrib, rad_E, vol, n_groups,
                                       negative_count, invalid_count);
  }
}

__device__ inline void reduce_sum_block_kernel_body(
    const std::size_t idx,
    const int tid,
    double* s,
    const double* __restrict__ in,
    double* __restrict__ block_sums,
    const std::size_t n) {
  const std::size_t base = 2U * idx *
                           static_cast<std::size_t>(blockDim.x);
  const std::size_t i0 = base + static_cast<std::size_t>(tid);
  const std::size_t i1 = i0 + static_cast<std::size_t>(blockDim.x);

  double value = 0.0;
  if (i0 < n) {
    value += in[i0];
  }
  if (i1 < n) {
    value += in[i1];
  }
  s[tid] = value;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    block_sums[idx] = s[0];
  }
}

__global__ void reduce_sum_block_kernel(const double* __restrict__ in,
                                        double* __restrict__ block_sums,
                                        const std::size_t n) {
  __shared__ double s[kBlockSize];
  const int tid = threadIdx.x;
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x);
  reduce_sum_block_kernel_body(idx, tid, s, in, block_sums, n);
}

double reduce_device_sum_persistent(FldEnergyScratch& scratch,
                                    const std::size_t n) {
  if (n == 0U) {
    return 0.0;
  }

  const std::size_t reduce_blocks =
      (n + 2U * static_cast<std::size_t>(kBlockSize) - 1U) /
      (2U * static_cast<std::size_t>(kBlockSize));
  TENRYU_ASSERT(reduce_blocks <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "FLD radiation energy reduce block count exceeds CUDA grid limit");
  const int blocks = static_cast<int>(reduce_blocks);

  reduce_sum_block_kernel<<<blocks, kBlockSize>>>(
      scratch.d_contrib, scratch.d_block_sums, n);
  cuda_check(cudaGetLastError(), "FLD radiation energy reduce launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "FLD radiation energy reduce execution failed");

  std::vector<double> host_block_sums(reduce_blocks, 0.0);
  cuda_check(cudaMemcpy(host_block_sums.data(),
                        scratch.d_block_sums,
                        reduce_blocks * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "FLD radiation energy block sums copy failed");

  long double total = 0.0L;
  for (const double value : host_block_sums) {
    total += static_cast<long double>(value);
  }
  return static_cast<double>(total);
}

}  // namespace

double compute_fld_rad_energy_total_device(const core::State& state,
                                           const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups =
      static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  // Empty rad_E = radiation inactive (legitimate zero). A NONEMPTY field
  // with the wrong size is state corruption and must not silently audit as
  // "total energy 0" (AI review k15 C-3/9, 2026-07-26).
  if (state.rad_E.empty() || n_cell_groups == 0U) {
    return 0.0;
  }
  TENRYU_ASSERT(state.rad_E.size() == n_cell_groups &&
                    state.vol.size() == n_cells,
                "FLD radiation energy audit: rad_E/vol size mismatch");

  const std::size_t reduce_blocks =
      (n_cell_groups + 2U * static_cast<std::size_t>(kBlockSize) - 1U) /
      (2U * static_cast<std::size_t>(kBlockSize));
  TENRYU_ASSERT(reduce_blocks <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "FLD radiation energy reduce block count exceeds CUDA grid limit");

  FldEnergyScratch& scratch = fld_energy_scratch();
  scratch.ensure_capacity(n_cell_groups, reduce_blocks);

  const std::size_t raw_contrib_blocks =
      (n_cell_groups + static_cast<std::size_t>(kBlockSize) - 1U) /
      static_cast<std::size_t>(kBlockSize);
  const int contrib_blocks = static_cast<int>(
      std::max<std::size_t>(
          std::size_t{1},
          std::min<std::size_t>(raw_contrib_blocks, std::size_t{4096})));

  cuda_check(cudaMemset(scratch.d_counts, 0, 2U * sizeof(int)),
             "FLD radiation energy counts reset failed");
  std::size_t led_c_begin = 0;
  std::size_t led_c_end = n_cells;
  if (state.mesh.dim == 2) {
    const auto led_cw = state.owned_cell_window(static_cast<int>(n_cells));
    led_c_begin = static_cast<std::size_t>(led_cw.begin);
    led_c_end = static_cast<std::size_t>(led_cw.end);
  }
  fld_rad_energy_contrib_kernel<<<contrib_blocks, kBlockSize>>>(
      scratch.d_contrib,
      state.rad_E.data(),
      state.vol.data(),
      n_cell_groups,
      n_groups,
      led_c_begin,
      led_c_end,
      scratch.d_counts,
      scratch.d_counts + 1);
  cuda_check(cudaGetLastError(), "FLD radiation energy contrib launch failed");

  const double total = reduce_device_sum_persistent(scratch, n_cell_groups);
  int h_counts[2] = {0, 0};
  cuda_check(cudaMemcpy(h_counts, scratch.d_counts, sizeof(h_counts),
                        cudaMemcpyDeviceToHost),
             "FLD radiation energy counts copy failed");
  if (h_counts[0] != 0 || h_counts[1] != 0) {
    static int audit_warn_count = 0;
    ++audit_warn_count;
    if (audit_warn_count <= 5 || audit_warn_count % 500 == 0) {
      core::log_warning(
          "FLD radiation energy audit: negative E entries=" +
          std::to_string(h_counts[0]) +
          ", nonfinite/invalid entries=" + std::to_string(h_counts[1]) +
          " (signed total " + std::to_string(total) + " erg; warning #" +
          std::to_string(audit_warn_count) + ")");
    }
  }
  return total;
}

int fld_rad_energy_reduce_blocks(const core::State& state,
                                 const core::Config& cfg) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups =
      static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  if (state.rad_E.size() != n_cell_groups || state.vol.size() != n_cells) {
    return 0;
  }
  if (n_cell_groups == 0U) {
    return 0;
  }
  const std::size_t reduce_blocks =
      (n_cell_groups + 2U * static_cast<std::size_t>(kBlockSize) - 1U) /
      (2U * static_cast<std::size_t>(kBlockSize));
  TENRYU_ASSERT(reduce_blocks <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "FLD radiation energy reduce block count exceeds CUDA grid limit");
  return static_cast<int>(reduce_blocks);
}

int compute_fld_rad_energy_total_device_to_slot(const core::State& state,
                                                const core::Config& cfg,
                                                double* d_slot) {
  const std::size_t n_cells = state.rho.size();
  const std::size_t n_groups =
      static_cast<std::size_t>(std::max(cfg.radiation.groups, 1));
  const std::size_t n_cell_groups = n_cells * n_groups;
  // Same audit contract as the synchronous variant: empty = inactive,
  // nonempty mismatch = corruption (assert). This path stays asynchronous,
  // so the negative/invalid counters are not read back here (nullptr).
  if (state.rad_E.empty() || n_cell_groups == 0U) {
    return 0;
  }
  TENRYU_ASSERT(state.rad_E.size() == n_cell_groups &&
                    state.vol.size() == n_cells,
                "FLD radiation energy audit: rad_E/vol size mismatch");

  const std::size_t reduce_blocks =
      (n_cell_groups + 2U * static_cast<std::size_t>(kBlockSize) - 1U) /
      (2U * static_cast<std::size_t>(kBlockSize));
  TENRYU_ASSERT(reduce_blocks <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "FLD radiation energy reduce block count exceeds CUDA grid limit");

  FldEnergyScratch& scratch = fld_energy_scratch();
  scratch.ensure_capacity(n_cell_groups, reduce_blocks);

  const std::size_t raw_contrib_blocks =
      (n_cell_groups + static_cast<std::size_t>(kBlockSize) - 1U) /
      static_cast<std::size_t>(kBlockSize);
  const int contrib_blocks = static_cast<int>(
      std::max<std::size_t>(
          std::size_t{1},
          std::min<std::size_t>(raw_contrib_blocks, std::size_t{4096})));

  std::size_t led_c_begin = 0;
  std::size_t led_c_end = n_cells;
  if (state.mesh.dim == 2) {
    const auto led_cw = state.owned_cell_window(static_cast<int>(n_cells));
    led_c_begin = static_cast<std::size_t>(led_cw.begin);
    led_c_end = static_cast<std::size_t>(led_cw.end);
  }
  fld_rad_energy_contrib_kernel<<<contrib_blocks, kBlockSize>>>(
      scratch.d_contrib,
      state.rad_E.data(),
      state.vol.data(),
      n_cell_groups,
      n_groups,
      led_c_begin,
      led_c_end,
      nullptr,
      nullptr);
  cuda_check(cudaGetLastError(), "FLD radiation energy contrib launch failed");

  reduce_sum_block_kernel<<<static_cast<int>(reduce_blocks), kBlockSize>>>(
      scratch.d_contrib, d_slot, n_cell_groups);
  cuda_check(cudaGetLastError(),
             "FLD radiation energy slot reduce launch failed");
  return static_cast<int>(reduce_blocks);
}

double materialize_fld_rad_energy(const double* h_slot, const int blocks) {
  long double total = 0.0L;
  for (int b = 0; b < blocks; ++b) {
    total += static_cast<long double>(h_slot[b]);
  }
  return static_cast<double>(total);
}

}  // namespace tenryu::coupling
