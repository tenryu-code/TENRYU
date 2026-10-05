#pragma once

// Block-level building blocks for device code that must reproduce a host loop's order: the
// exclusive prefix of one count per thread (the order-preserving compaction of runs), and the sum
// of an array in index order by one thread after the nonzero entries are compacted into shared
// memory. Skipping the zero entries leaves the sum unchanged as long as the running sum is never
// a negative zero, which holds when it starts at +0 or a nonzero value (an exact cancellation
// gives +0): a nonzero entry is added as the loop would add it, and adding +0 or -0 to a value
// that is not -0 returns that value.

namespace tenryu::core::device_ordered {

// The exclusive prefix of `value` over the threads of a block of kBlock threads, in thread order;
// *total is the sum. Every thread of the block must call it (it synchronizes the block).
template <int kBlock>
__device__ inline int block_exclusive_prefix(const int value, int* sh_scan, int* total) {
  const int t = static_cast<int>(threadIdx.x);
  sh_scan[t] = value;
  __syncthreads();
  for (int offset = 1; offset < kBlock; offset <<= 1) {
    const int add = (t >= offset) ? sh_scan[t - offset] : 0;
    __syncthreads();
    sh_scan[t] += add;
    __syncthreads();
  }
  const int inclusive = sh_scan[t];
  *total = sh_scan[kBlock - 1];
  __syncthreads();
  return inclusive - value;
}

// init + x[0] + x[1] + ... + x[n-1], added in index order (the zero entries skipped), in thread 0
// (the other threads' return value is meaningless). sh_values holds kBlock * kPerThread doubles.
// Every thread of the block must call it.
template <int kBlock, int kPerThread>
__device__ inline double block_ordered_sum_nonzero(const double* x, const int n, const double init,
                                                   double* sh_values, int* sh_scan) {
  constexpr int kChunk = kBlock * kPerThread;
  const int t = static_cast<int>(threadIdx.x);
  double acc = init;
  for (int chunk = 0; chunk < n; chunk += kChunk) {
    const int begin = chunk + t * kPerThread;
    int local = 0;
    for (int j = 0; j < kPerThread; ++j) {
      const int i = begin + j;
      if (i < n && x[i] != 0.0) {
        ++local;
      }
    }
    int total = 0;
    int at = block_exclusive_prefix<kBlock>(local, sh_scan, &total);
    for (int j = 0; j < kPerThread; ++j) {
      const int i = begin + j;
      if (i < n) {
        const double v = x[i];
        if (v != 0.0) {
          sh_values[at] = v;
          ++at;
        }
      }
    }
    __syncthreads();
    if (t == 0) {
      for (int m = 0; m < total; ++m) {
        acc += sh_values[m];
      }
    }
    __syncthreads();
  }
  return acc;
}

// The same sum for the terms term(0), term(1), ..., term(n-1) of a callable (each term evaluated
// once): init + term(0) + ... + term(n-1) in index order, the zero terms skipped, in thread 0.
template <int kBlock, int kPerThread, typename Term>
__device__ inline double block_ordered_sum_nonzero_terms(const int n, const double init,
                                                         const Term& term, double* sh_values,
                                                         int* sh_scan) {
  constexpr int kChunk = kBlock * kPerThread;
  const int t = static_cast<int>(threadIdx.x);
  double acc = init;
  for (int chunk = 0; chunk < n; chunk += kChunk) {
    const int begin = chunk + t * kPerThread;
    double values[kPerThread];
    int local = 0;
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      const int i = begin + j;
      values[j] = (i < n) ? term(i) : 0.0;
      if (values[j] != 0.0) {
        ++local;
      }
    }
    int total = 0;
    int at = block_exclusive_prefix<kBlock>(local, sh_scan, &total);
#pragma unroll
    for (int j = 0; j < kPerThread; ++j) {
      if (values[j] != 0.0) {
        sh_values[at] = values[j];
        ++at;
      }
    }
    __syncthreads();
    if (t == 0) {
      for (int m = 0; m < total; ++m) {
        acc += sh_values[m];
      }
    }
    __syncthreads();
  }
  return acc;
}

// fmax / fmin as the x86-64 glibc functions return them: a NaN operand is passed over (the other
// one is returned), and of two equal operands the second one is returned (maxsd / minsd), which
// decides the sign of a zero result. A fold of a sequence with them returns, among the values
// equal to the extremum, the last one; folding contiguous chunks in order and then the chunks'
// results in order gives the same value.
__host__ __device__ inline double fmax_like_glibc(const double x, const double y) {
  if (y != y) {
    return x;
  }
  if (x != x) {
    return y;
  }
  return (x > y) ? x : y;
}

__host__ __device__ inline double fmin_like_glibc(const double x, const double y) {
  if (y != y) {
    return x;
  }
  if (x != x) {
    return y;
  }
  return (x < y) ? x : y;
}

}  // namespace tenryu::core::device_ordered
