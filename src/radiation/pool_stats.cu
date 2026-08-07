#include "radiation/pool_stats.cuh"

#include <algorithm>
#include <cfloat>
#include <cstddef>
#include <cstdint>

#include <cub/device/device_reduce.cuh>
#include <cuda_runtime.h>

#include "core/error.hpp"
#include "core/fancy_iterators.cuh"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void alloc_device(T** ptr, const int n, const char* msg) {
  if (n > 0) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(ptr),
                          sizeof(T) * static_cast<std::size_t>(n)),
               msg);
  } else {
    *ptr = nullptr;
  }
}

template <typename T>
void free_device(T** ptr, const char* msg) {
  if (*ptr != nullptr) {
    cuda_check(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

void alloc_bytes(void** ptr, const std::size_t bytes, const char* msg) {
  if (bytes > 0U) {
    cuda_check(cudaMalloc(ptr, bytes), msg);
  } else {
    *ptr = nullptr;
  }
}

void free_bytes(void** ptr, const char* msg) {
  if (*ptr != nullptr) {
    cuda_check(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

__host__ __device__ inline double nonnegative_weight(const double value) {
  return value < 0.0 ? 0.0 : value;
}

struct MinWeightOp {
  const double* weight = nullptr;
  const std::uint8_t* alive = nullptr;

  __host__ __device__ double operator()(const int idx) const {
    if (alive[idx] != static_cast<std::uint8_t>(kAlive)) {
      return DBL_MAX;
    }
    return nonnegative_weight(weight[idx]);
  }
};

struct MaxWeightOp {
  const double* weight = nullptr;
  const std::uint8_t* alive = nullptr;

  __host__ __device__ double operator()(const int idx) const {
    if (alive[idx] != static_cast<std::uint8_t>(kAlive)) {
      return 0.0;
    }
    return nonnegative_weight(weight[idx]);
  }
};

struct SumWeightOp {
  const double* weight = nullptr;
  const std::uint8_t* alive = nullptr;

  __host__ __device__ double operator()(const int idx) const {
    if (alive[idx] != static_cast<std::uint8_t>(kAlive)) {
      return 0.0;
    }
    return nonnegative_weight(weight[idx]);
  }
};

struct AliveCountOp {
  const std::uint8_t* alive = nullptr;

  __host__ __device__ int operator()(const int idx) const {
    return alive[idx] == static_cast<std::uint8_t>(kAlive) ? 1 : 0;
  }
};

struct PoolStatsScratch {
  void* d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0U;
  double* d_values = nullptr;
  int* d_count = nullptr;
  int query_capacity = 0;

  void ensure_outputs() {
    if (d_values == nullptr) {
      alloc_device(&d_values, 3, "pool stats cudaMalloc persistent values failed");
    }
    if (d_count == nullptr) {
      alloc_device(&d_count, 1, "pool stats cudaMalloc persistent count failed");
    }
  }

  std::size_t query_temp_bytes(const int n,
                               const double* weight,
                               const std::uint8_t* alive) {
    using CountingIt = core::CountingInputIterator<int>;
    CountingIt indices(0);
    auto min_it = core::TransformInputIterator<double, MinWeightOp, CountingIt>(
        indices, MinWeightOp{weight, alive});
    auto max_it = core::TransformInputIterator<double, MaxWeightOp, CountingIt>(
        indices, MaxWeightOp{weight, alive});
    auto sum_it = core::TransformInputIterator<double, SumWeightOp, CountingIt>(
        indices, SumWeightOp{weight, alive});
    auto count_it = core::TransformInputIterator<int, AliveCountOp, CountingIt>(
        indices, AliveCountOp{alive});

    std::size_t min_bytes = 0U;
    std::size_t max_bytes = 0U;
    std::size_t sum_bytes = 0U;
    std::size_t count_bytes = 0U;
    cuda_check(cub::DeviceReduce::Min(nullptr, min_bytes, min_it, d_values, n),
               "pool stats CUB min temp-size query failed");
    cuda_check(cub::DeviceReduce::Max(nullptr, max_bytes, max_it, d_values + 1, n),
               "pool stats CUB max temp-size query failed");
    cuda_check(cub::DeviceReduce::Sum(nullptr, sum_bytes, sum_it, d_values + 2, n),
               "pool stats CUB sum temp-size query failed");
    cuda_check(cub::DeviceReduce::Sum(nullptr, count_bytes, count_it, d_count, n),
               "pool stats CUB count temp-size query failed");
    return std::max(std::max(min_bytes, max_bytes), std::max(sum_bytes, count_bytes));
  }

  void ensure(const int required_capacity,
              const double* weight,
              const std::uint8_t* alive) {
    TENRYU_ASSERT(required_capacity >= 0,
                  "PoolStatsScratch::ensure requires capacity >= 0");
    if (required_capacity <= 0) {
      return;
    }
    ensure_outputs();
    if (required_capacity <= query_capacity && d_temp_storage != nullptr) {
      return;
    }
    const std::size_t required_bytes =
        query_temp_bytes(required_capacity, weight, alive);
    if (required_bytes > temp_storage_bytes) {
      free_bytes(&d_temp_storage, "pool stats cudaFree persistent temp storage failed");
      alloc_bytes(&d_temp_storage,
                  required_bytes,
                  "pool stats cudaMalloc persistent temp storage failed");
      temp_storage_bytes = required_bytes;
    }
    query_capacity = required_capacity;
  }

  void release() {
    free_bytes(&d_temp_storage, "pool stats cudaFree persistent temp storage failed");
    free_device(&d_count, "pool stats cudaFree persistent count failed");
    free_device(&d_values, "pool stats cudaFree persistent values failed");
    temp_storage_bytes = 0U;
    query_capacity = 0;
  }
};

PoolStatsScratch& pool_stats_scratch() {
  static PoolStatsScratch scratch;
  return scratch;
}

}  // namespace

void compute_weight_stats_device(const PhotonPool& pool,
                                 double* weight_min,
                                 double* weight_mean,
                                 double* weight_max) {
  TENRYU_ASSERT(weight_min != nullptr, "compute_weight_stats_device requires weight_min");
  TENRYU_ASSERT(weight_mean != nullptr, "compute_weight_stats_device requires weight_mean");
  TENRYU_ASSERT(weight_max != nullptr, "compute_weight_stats_device requires weight_max");
  if (pool.n_alive <= 0 || pool.weight == nullptr || pool.alive == nullptr) {
    *weight_min = 0.0;
    *weight_mean = 0.0;
    *weight_max = 0.0;
    return;
  }
  TENRYU_ASSERT(pool.n_alive <= pool.capacity,
                "compute_weight_stats_device pool.n_alive exceeds pool.capacity");

  PoolStatsScratch& scratch = pool_stats_scratch();
  scratch.ensure(pool.capacity, pool.weight, pool.alive);

  using CountingIt = core::CountingInputIterator<int>;
  CountingIt indices(0);
  auto min_it = core::TransformInputIterator<double, MinWeightOp, CountingIt>(
      indices, MinWeightOp{pool.weight, pool.alive});
  auto max_it = core::TransformInputIterator<double, MaxWeightOp, CountingIt>(
      indices, MaxWeightOp{pool.weight, pool.alive});
  auto sum_it = core::TransformInputIterator<double, SumWeightOp, CountingIt>(
      indices, SumWeightOp{pool.weight, pool.alive});
  auto count_it = core::TransformInputIterator<int, AliveCountOp, CountingIt>(
      indices, AliveCountOp{pool.alive});

  cuda_check(cub::DeviceReduce::Min(scratch.d_temp_storage,
                                    scratch.temp_storage_bytes,
                                    min_it,
                                    scratch.d_values,
                                    pool.n_alive),
             "pool stats CUB min failed");
  cuda_check(cub::DeviceReduce::Max(scratch.d_temp_storage,
                                    scratch.temp_storage_bytes,
                                    max_it,
                                    scratch.d_values + 1,
                                    pool.n_alive),
             "pool stats CUB max failed");
  cuda_check(cub::DeviceReduce::Sum(scratch.d_temp_storage,
                                    scratch.temp_storage_bytes,
                                    sum_it,
                                    scratch.d_values + 2,
                                    pool.n_alive),
             "pool stats CUB sum failed");
  cuda_check(cub::DeviceReduce::Sum(scratch.d_temp_storage,
                                    scratch.temp_storage_bytes,
                                    count_it,
                                    scratch.d_count,
                                    pool.n_alive),
             "pool stats CUB count failed");
  cuda_check(cudaGetLastError(), "pool stats CUB reduction launch failed");

  double values[3] = {0.0, 0.0, 0.0};
  int count = 0;
  cuda_check(cudaMemcpy(values,
                        scratch.d_values,
                        sizeof(values),
                        cudaMemcpyDeviceToHost),
             "pool stats copy values failed");
  cuda_check(cudaMemcpy(&count,
                        scratch.d_count,
                        sizeof(count),
                        cudaMemcpyDeviceToHost),
             "pool stats copy count failed");

  if (count <= 0) {
    *weight_min = 0.0;
    *weight_mean = 0.0;
    *weight_max = 0.0;
    return;
  }
  *weight_min = values[0];
  *weight_mean =
      static_cast<double>(static_cast<long double>(values[2]) /
                          static_cast<long double>(count));
  *weight_max = values[1];
}

void reserve_pool_stats_scratch(const int capacity) {
  pool_stats_scratch().ensure(capacity, nullptr, nullptr);
}

void release_pool_stats_scratch() {
  pool_stats_scratch().release();
}

}  // namespace tenryu::radiation
