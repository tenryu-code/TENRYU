#include "coupling/thermal_subcycle_scan.hpp"

#include <cuda_runtime.h>

#include <string>

#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::coupling {
namespace {

constexpr int kBlock = 256;
constexpr int kMaxBlocks = 256;
constexpr double kNoMargin = 1.0e30;

inline void cuda_check(const cudaError_t err, const char* what) {
  TENRYU_ASSERT(err == cudaSuccess, std::string(what) + ": " + cudaGetErrorString(err));
}

inline int block_count(const int n) {
  const int blocks = (n + kBlock - 1) / kBlock;
  return blocks < 1 ? 1 : (blocks > kMaxBlocks ? kMaxBlocks : blocks);
}

__global__ void floor_hit_kernel(const double* __restrict__ Te,
                                 const double* __restrict__ rho,
                                 const int n,
                                 const double rho_min,
                                 const double te_threshold,
                                 int* __restrict__ hit) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    if (rho[i] > rho_min && Te[i] <= te_threshold) {
      *hit = 1;
      return;
    }
  }
}

// Per-block minimum of the margin ratio; the host loop's predicate exactly
// (a NaN rho is not skipped by `rho < rho_min`, a NaN Te fails `Te > guard`).
__global__ void margin_ratio_block_min_kernel(const double* __restrict__ Te,
                                              const double* __restrict__ rho,
                                              const int begin,
                                              const int end,
                                              const double rho_min,
                                              const double te_guard,
                                              const double margin_max,
                                              double* __restrict__ block_min) {
  __shared__ double s_min[kBlock];
  double local = kNoMargin;
  for (int i = begin + blockIdx.x * blockDim.x + threadIdx.x; i < end;
       i += gridDim.x * blockDim.x) {
    if (rho[i] < rho_min) {
      continue;
    }
    const double te = Te[i];
    const double margin = te - te_guard;
    if (margin < margin_max && te > te_guard) {
      const double ratio = margin / fmax(te, 1.0);
      local = fmin(local, ratio);
    }
  }
  s_min[threadIdx.x] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      s_min[threadIdx.x] = fmin(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    block_min[blockIdx.x] = s_min[0];
  }
}

__global__ void final_min_kernel(const double* __restrict__ block_min,
                                 const int n_blocks,
                                 double* __restrict__ out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    double m = kNoMargin;
    for (int b = 0; b < n_blocks; ++b) {
      m = fmin(m, block_min[b]);
    }
    *out = m;
  }
}

}  // namespace

bool thermal_subcycle_floor_hit(const double* d_Te,
                                const double* d_rho,
                                const int n,
                                const double rho_min,
                                const double te_threshold) {
  if (n <= 0) {
    return false;
  }
  auto* d_hit = static_cast<int*>(
      core::device_scratch_acquire("thermal_subcycle:floor_hit", sizeof(int)));
  cuda_check(cudaMemset(d_hit, 0, sizeof(int)), "thermal subcycle floor-hit reset");
  floor_hit_kernel<<<block_count(n), kBlock>>>(d_Te, d_rho, n, rho_min, te_threshold, d_hit);
  cuda_check(cudaGetLastError(), "thermal subcycle floor-hit launch");
  int hit = 0;
  cuda_check(cudaMemcpy(&hit, d_hit, sizeof(int), cudaMemcpyDeviceToHost),
             "thermal subcycle floor-hit readback");
  return hit != 0;
}

double thermal_subcycle_min_margin_ratio(const double* d_Te,
                                         const double* d_rho,
                                         const int begin,
                                         const int end,
                                         const double rho_min,
                                         const double te_guard,
                                         const double margin_max) {
  if (end <= begin) {
    return kNoMargin;
  }
  const int blocks = block_count(end - begin);
  auto* d_partial = static_cast<double*>(core::device_scratch_acquire(
      "thermal_subcycle:margin_partial", sizeof(double) * (kMaxBlocks + 1)));
  margin_ratio_block_min_kernel<<<blocks, kBlock>>>(d_Te, d_rho, begin, end, rho_min, te_guard,
                                                    margin_max, d_partial);
  cuda_check(cudaGetLastError(), "thermal subcycle margin launch");
  final_min_kernel<<<1, 1>>>(d_partial, blocks, d_partial + kMaxBlocks);
  cuda_check(cudaGetLastError(), "thermal subcycle margin final launch");
  double result = kNoMargin;
  cuda_check(cudaMemcpy(&result, d_partial + kMaxBlocks, sizeof(double), cudaMemcpyDeviceToHost),
             "thermal subcycle margin readback");
  return result;
}

}  // namespace tenryu::coupling
