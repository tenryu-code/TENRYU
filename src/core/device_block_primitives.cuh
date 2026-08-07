#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math_constants.h>

#include <climits>
#include <cmath>

namespace tenryu::core {

namespace detail {

template <int kBlockSize>
struct BlockSizeIsValid {
  static constexpr bool value =
      kBlockSize >= 32 && (kBlockSize & (kBlockSize - 1)) == 0;
};

__device__ inline bool is_nan(const double v) {
  return v != v;
}

__device__ inline int padded_power_of_two(const int n) {
  int padded = 1;
  while (padded < n) {
    padded <<= 1;
  }
  return padded;
}

}  // namespace detail

template <bool kGrid>
__device__ inline void pcr_sync() {
  if constexpr (kGrid) {
    cooperative_groups::this_grid().sync();
  } else {
    __syncthreads();
  }
}

// Fixed-order block sum reduction. smem must point to kBlockSize doubles.
// Callers with fewer live items than kBlockSize pass the identity, 0.0, for
// inactive lanes. The final barrier makes the returned smem[0] value valid for
// every thread in the block.
template <int kBlockSize>
__device__ double block_reduce_sum_fixed_order(double v, double* smem) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();

  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  return smem[0];
}

// Fixed-order block max reduction. smem must point to kBlockSize doubles.
// Callers with fewer live items than kBlockSize pass the identity, -infinity,
// for inactive lanes. CUDA fmax returns the non-NaN operand when exactly one
// operand is NaN; this primitive performs no additional NaN handling. The final
// barrier makes the returned smem[0] value valid for every thread in the block.
template <int kBlockSize>
__device__ double block_reduce_max_fixed_order(double v, double* smem) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  const int tid = threadIdx.x;
  smem[tid] = v;
  __syncthreads();

  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] = ::fmax(smem[tid], smem[tid + s]);
    }
    __syncthreads();
  }

  __syncthreads();
  return smem[0];
}

// Fixed-order block argmin reduction. smem_v and smem_i must point to
// kBlockSize entries. Callers with fewer live items than kBlockSize pass the
// identity, {+infinity, INT_MAX}, for inactive lanes. Ties on exactly equal
// values are broken by the smaller index. NaN input values are converted to the
// identity before the comparison tree, so NaN entries never win; ordered
// comparisons with NaN are otherwise false. The final barrier leaves the result
// in smem slot 0 for every thread in the block to read.
template <int kBlockSize>
__device__ void block_reduce_argmin_fixed_order(double v,
                                                int idx,
                                                double* smem_v,
                                                int* smem_i,
                                                double* out_v,
                                                int* out_i) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  const int tid = threadIdx.x;
  if (detail::is_nan(v)) {
    smem_v[tid] = CUDART_INF;
    smem_i[tid] = INT_MAX;
  } else {
    smem_v[tid] = v;
    smem_i[tid] = idx;
  }
  __syncthreads();

  for (int s = kBlockSize / 2; s > 0; s >>= 1) {
    if (tid < s) {
      const double av = smem_v[tid];
      const int ai = smem_i[tid];
      const double bv = smem_v[tid + s];
      const int bi = smem_i[tid + s];
      if (bv < av || (bv == av && bi < ai)) {
        smem_v[tid] = bv;
        smem_i[tid] = bi;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out_v = smem_v[0];
    *out_i = smem_i[0];
  }
  __syncthreads();
}

// Fixed-order cooperative-grid sum reduction. Stage A performs the same P9
// block reduction in each resident block and writes one partial per block.
// Stage B uses block 0 to reduce the partial slab over an identity-padded
// power-of-two count with the same halving tree. The result is deterministic
// for a fixed cooperative grid size.
template <int kBlockSize>
__device__ double grid_reduce_sum_fixed_order(double v,
                                              double* smem,
                                              double* partials,
                                              double* scalar,
                                              const int grid_count) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const double block_value =
      block_reduce_sum_fixed_order<kBlockSize>(v, smem);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_value;
  }
  grid.sync();

  if (blockIdx.x == 0) {
    const int n = detail::padded_power_of_two(grid_count);
    for (int i = grid_count + threadIdx.x; i < n; i += blockDim.x) {
      partials[i] = 0.0;
    }
    __syncthreads();
    for (int s = n / 2; s > 0; s >>= 1) {
      for (int i = threadIdx.x; i < s; i += blockDim.x) {
        partials[i] += partials[i + s];
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      *scalar = partials[0];
    }
  }
  grid.sync();
  return *scalar;
}

// Fixed-order cooperative-grid max reduction. See
// grid_reduce_sum_fixed_order for the two-stage contract.
template <int kBlockSize>
__device__ double grid_reduce_max_fixed_order(double v,
                                              double* smem,
                                              double* partials,
                                              double* scalar,
                                              const int grid_count) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  const double block_value =
      block_reduce_max_fixed_order<kBlockSize>(v, smem);
  if (threadIdx.x == 0) {
    partials[blockIdx.x] = block_value;
  }
  grid.sync();

  if (blockIdx.x == 0) {
    const int n = detail::padded_power_of_two(grid_count);
    for (int i = grid_count + threadIdx.x; i < n; i += blockDim.x) {
      partials[i] = -CUDART_INF;
    }
    __syncthreads();
    for (int s = n / 2; s > 0; s >>= 1) {
      for (int i = threadIdx.x; i < s; i += blockDim.x) {
        partials[i] = ::fmax(partials[i], partials[i + s]);
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      *scalar = partials[0];
    }
  }
  grid.sync();
  return *scalar;
}

// Fixed-order cooperative-grid argmin reduction. Ties are broken by the
// smaller index, matching block_reduce_argmin_fixed_order.
template <int kBlockSize>
__device__ void grid_reduce_argmin_fixed_order(double v,
                                               int idx,
                                               double* smem_v,
                                               int* smem_i,
                                               double* partial_v,
                                               int* partial_i,
                                               double* scalar_v,
                                               int* scalar_i,
                                               const int grid_count,
                                               double* out_v,
                                               int* out_i) {
  static_assert(detail::BlockSizeIsValid<kBlockSize>::value,
                "kBlockSize must be a power of two and >= 32");

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  double block_v = CUDART_INF;
  int block_i = INT_MAX;
  block_reduce_argmin_fixed_order<kBlockSize>(
      v, idx, smem_v, smem_i, &block_v, &block_i);
  if (threadIdx.x == 0) {
    partial_v[blockIdx.x] = block_v;
    partial_i[blockIdx.x] = block_i;
  }
  grid.sync();

  if (blockIdx.x == 0) {
    const int n = detail::padded_power_of_two(grid_count);
    for (int i = grid_count + threadIdx.x; i < n; i += blockDim.x) {
      partial_v[i] = CUDART_INF;
      partial_i[i] = INT_MAX;
    }
    __syncthreads();
    for (int s = n / 2; s > 0; s >>= 1) {
      for (int i = threadIdx.x; i < s; i += blockDim.x) {
        const double av = partial_v[i];
        const int ai = partial_i[i];
        const double bv = partial_v[i + s];
        const int bi = partial_i[i + s];
        if (bv < av || (bv == av && bi < ai)) {
          partial_v[i] = bv;
          partial_i[i] = bi;
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      *scalar_v = partial_v[0];
      *scalar_i = partial_i[0];
    }
  }
  grid.sync();
  *out_v = *scalar_v;
  *out_i = *scalar_i;
}

// Serial Thomas solve for one tridiagonal system in the
// cusparseDgtsv2StridedBatch layout. System s occupies
// [s * stride, s * stride + m) in dl, d, du, rhs, and cp_work. The unused
// fringes dl[0] and du[m - 1] for each system are treated as 0.0 and are not
// trusted as input values. cp_work is caller-provided storage in the same
// strided layout; no allocation is performed here. The solution overwrites rhs,
// matching gtsv2's output convention. This serial Thomas algorithm is NOT
// bit-identical to cuSPARSE gtsv2 because the algorithm differs; it is
// run-to-run deterministic.
__device__ inline void thomas_solve_strided(const double* dl,
                                            const double* d,
                                            const double* du,
                                            double* rhs,
                                            double* cp_work,
                                            int m,
                                            int stride,
                                            int system) {
  if (m < 1) {
    return;
  }

  const int base = system * stride;
  if (m == 1) {
    rhs[base] /= d[base];
    return;
  }

  const double b0 = d[base];
  const double c0 = du[base];
  cp_work[base] = c0 / b0;
  rhs[base] /= b0;

  for (int i = 1; i < m; ++i) {
    const int k = base + i;
    const double a_i = dl[k];
    const double c_i = (i == m - 1) ? 0.0 : du[k];
    const double denom = d[k] - a_i * cp_work[k - 1];
    cp_work[k] = c_i / denom;
    rhs[k] = (rhs[k] - a_i * rhs[k - 1]) / denom;
  }

  for (int i = m - 2; i >= 0; --i) {
    const int k = base + i;
    rhs[k] -= cp_work[k] * rhs[k + 1];
  }
}

// Parallel cyclic reduction for the same strided tridiagonal layout as
// thomas_solve_strided. System s occupies [s * stride, s * stride + m) in dl,
// d, du, rhs, and each work slab. The unused fringes dl[0] and du[m - 1] for
// each system are treated as 0.0. The coefficient arrays are used as one round
// buffer and may be overwritten; rhs receives the final solution. The caller
// must provide four work slabs in the same strided layout and must call this
// from all threads participating in the selected block/grid synchronization.
template <bool kGrid>
__device__ inline void pcr_solve_strided(double* dl,
                                         double* d,
                                         double* du,
                                         double* rhs,
                                         double* dl_work,
                                         double* d_work,
                                         double* du_work,
                                         double* rhs_work,
                                         int m,
                                         int stride,
                                         int n_systems,
                                         int thread_id,
                                         int thread_stride) {
  if (m < 1 || n_systems < 1) {
    return;
  }

  const int total = m * n_systems;
  for (int idx = thread_id; idx < total; idx += thread_stride) {
    const int system = idx / m;
    const int row = idx - system * m;
    const int k = system * stride + row;
    dl_work[k] = (row == 0) ? 0.0 : dl[k];
    d_work[k] = d[k];
    du_work[k] = (row == m - 1) ? 0.0 : du[k];
    rhs_work[k] = rhs[k];
  }
  pcr_sync<kGrid>();

  bool read_work = true;
  for (int offset = 1; offset < m; offset <<= 1) {
    double* const src_dl = read_work ? dl_work : dl;
    double* const src_d = read_work ? d_work : d;
    double* const src_du = read_work ? du_work : du;
    double* const src_rhs = read_work ? rhs_work : rhs;
    double* const dst_dl = read_work ? dl : dl_work;
    double* const dst_d = read_work ? d : d_work;
    double* const dst_du = read_work ? du : du_work;
    double* const dst_rhs = read_work ? rhs : rhs_work;

    for (int idx = thread_id; idx < total; idx += thread_stride) {
      const int system = idx / m;
      const int row = idx - system * m;
      const int k = system * stride + row;
      double a_new = 0.0;
      double b_new = src_d[k];
      double c_new = 0.0;
      double r_new = src_rhs[k];

      if (row >= offset) {
        const int kl = k - offset;
        const double alpha = src_dl[k] / src_d[kl];
        b_new -= alpha * src_du[kl];
        r_new -= alpha * src_rhs[kl];
        a_new = -alpha * src_dl[kl];
      }
      if (row + offset < m) {
        const int kr = k + offset;
        const double beta = src_du[k] / src_d[kr];
        b_new -= beta * src_dl[kr];
        r_new -= beta * src_rhs[kr];
        c_new = -beta * src_du[kr];
      }

      dst_dl[k] = a_new;
      dst_d[k] = b_new;
      dst_du[k] = c_new;
      dst_rhs[k] = r_new;
    }
    pcr_sync<kGrid>();
    read_work = !read_work;
  }

  double* const src_d = read_work ? d_work : d;
  double* const src_rhs = read_work ? rhs_work : rhs;
  for (int idx = thread_id; idx < total; idx += thread_stride) {
    const int system = idx / m;
    const int row = idx - system * m;
    const int k = system * stride + row;
    rhs[k] = src_rhs[k] / src_d[k];
  }
  pcr_sync<kGrid>();
}

}  // namespace tenryu::core
