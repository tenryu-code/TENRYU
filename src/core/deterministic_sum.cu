#include "core/deterministic_sum.hpp"

#include "core/device_block_primitives.cuh"
#include "core/error.hpp"

namespace tenryu::core {
namespace {

constexpr int kDeterministicSumBlock = 256;

__global__ void deterministic_sum_kernel(const double* __restrict__ values,
                                         const int n,
                                         double* __restrict__ out,
                                         const int accumulate) {
  __shared__ double smem[kDeterministicSumBlock];
  double partial = 0.0;
  for (int i = static_cast<int>(threadIdx.x); i < n; i += kDeterministicSumBlock) {
    partial += values[i];
  }
  const double total =
      block_reduce_sum_fixed_order<kDeterministicSumBlock>(partial, smem);
  if (threadIdx.x == 0) {
    *out = (accumulate != 0) ? (*out + total) : total;
  }
}

}  // namespace

void deterministic_sum(const double* d_values, const int n, double* d_out,
                       const bool accumulate, cudaStream_t stream) {
  TENRYU_ASSERT(d_out != nullptr, "deterministic_sum requires an output");
  if (n <= 0 || d_values == nullptr) {
    if (!accumulate) {
      TENRYU_ASSERT(cudaMemsetAsync(d_out, 0, sizeof(double), stream) == cudaSuccess,
                    "deterministic_sum zero failed");
    }
    return;
  }
  deterministic_sum_kernel<<<1, kDeterministicSumBlock, 0, stream>>>(
      d_values, n, d_out, accumulate ? 1 : 0);
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "deterministic_sum launch failed");
}

}  // namespace tenryu::core
