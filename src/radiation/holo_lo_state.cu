#include "radiation/holo_lo_state.cuh"

#include <limits>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void initialize_holo_lo_state_kernel(double* __restrict__ E_lo,
                                                const int n_total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }
  const double E = E_lo[idx];
  E_lo[idx] = (isfinite(E) && E > 0.0) ? E : 0.0;
}

__global__ void initialize_holo_lo_from_lte_kernel(double* __restrict__ E_lo,
                                                   const double* __restrict__ Te,
                                                   const int n_total,
                                                   const int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_total) {
    return;
  }

  const double E = E_lo[idx];
  if (isfinite(E) && E > 0.0) {
    return;
  }

  const int cell = idx / n_groups;
  const double T = Te[cell];
  const double T_safe = (isfinite(T) && T > 0.0) ? T : 1.0e-12;
  const double T2 = T_safe * T_safe;
  const double T4 = T2 * T2;
  const double b_g = (n_groups == 1) ? 1.0 : (1.0 / static_cast<double>(n_groups));
  const double E_lte = tenryu::core::constants::a_eV * T4 * b_g;
  E_lo[idx] = (isfinite(E_lte) && E_lte > 0.0) ? E_lte : 0.0;
}

}  // namespace

void initialize_holo_lo_state_cuda(double* E_lo,
                                   const int n_cells,
                                   const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  TENRYU_ASSERT(E_lo != nullptr,
                "initialize_holo_lo_state_cuda requires E_lo");
  TENRYU_ASSERT(n_groups <= std::numeric_limits<int>::max() / n_cells,
                "initialize_holo_lo_state_cuda n_cells*n_groups overflow");

  const int n_total = n_cells * n_groups;
  constexpr int kBlock = 256;
  const int grid = (n_total + kBlock - 1) / kBlock;
  initialize_holo_lo_state_kernel<<<grid, kBlock>>>(E_lo, n_total);
  cuda_check(cudaGetLastError(), "initialize_holo_lo_state_cuda launch failed");
  cuda_check(cudaDeviceSynchronize(), "initialize_holo_lo_state_cuda execution failed");
}

void initialize_holo_lo_from_lte_cuda(double* E_lo,
                                      const double* Te,
                                      const int n_cells,
                                      const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  TENRYU_ASSERT(E_lo != nullptr,
                "initialize_holo_lo_from_lte_cuda requires E_lo");
  TENRYU_ASSERT(Te != nullptr,
                "initialize_holo_lo_from_lte_cuda requires Te");
  TENRYU_ASSERT(n_groups <= std::numeric_limits<int>::max() / n_cells,
                "initialize_holo_lo_from_lte_cuda n_cells*n_groups overflow");

  const int n_total = n_cells * n_groups;
  constexpr int kBlock = 256;
  const int grid = (n_total + kBlock - 1) / kBlock;
  initialize_holo_lo_from_lte_kernel<<<grid, kBlock>>>(E_lo, Te, n_total, n_groups);
  cuda_check(cudaGetLastError(), "initialize_holo_lo_from_lte_cuda launch failed");
  cuda_check(cudaDeviceSynchronize(), "initialize_holo_lo_from_lte_cuda execution failed");
}

}  // namespace tenryu::radiation
