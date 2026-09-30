#include "radiation/composite_sort.cuh"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::uint64_t kInvalidSortKey = std::numeric_limits<std::uint64_t>::max();

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_ull;
  unsigned long long int assumed = 0ULL;
  do {
    assumed = old;
    old = atomicCAS(address_ull,
                    assumed,
                    __double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
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
void alloc_managed(T** ptr, const int n, const char* msg) {
  if (n > 0) {
    cuda_check(cudaMallocManaged(reinterpret_cast<void**>(ptr),
                                 sizeof(T) * static_cast<std::size_t>(n)),
               msg);
  } else {
    *ptr = nullptr;
  }
}

template <typename T>
void d2d_copy(T* dst, const T* src, const int n, const char* msg) {
  if (n > 0) {
    cuda_check(cudaMemcpy(dst,
                          src,
                          sizeof(T) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToDevice),
               msg);
  }
}

template <typename T>
void free_device(T** ptr, const char* msg) {
  if (*ptr != nullptr) {
    cuda_check(cudaFree(*ptr), msg);
    *ptr = nullptr;
  }
}

struct PhotonPoolReadView {
  const double* pos_r = nullptr;
  const double* pos_z = nullptr;
  const double* dir_r = nullptr;
  const double* dir_z = nullptr;
  const double* dir_phi = nullptr;
  const double* energy = nullptr;
  const double* weight = nullptr;
  const double* time_remain = nullptr;
  const double* birth_energy = nullptr;
  const std::int8_t* sign = nullptr;
  const std::uint64_t* global_id = nullptr;
  const std::uint32_t* rng_counter = nullptr;
  const std::int32_t* cell_id = nullptr;
  const std::uint16_t* group_id = nullptr;
  const std::uint8_t* mode = nullptr;
  const std::uint8_t* alive = nullptr;
};

struct PhotonPoolWriteView {
  double* pos_r = nullptr;
  double* pos_z = nullptr;
  double* dir_r = nullptr;
  double* dir_z = nullptr;
  double* dir_phi = nullptr;
  double* energy = nullptr;
  double* weight = nullptr;
  double* time_remain = nullptr;
  double* birth_energy = nullptr;
  std::int8_t* sign = nullptr;
  std::uint64_t* global_id = nullptr;
  std::uint32_t* rng_counter = nullptr;
  std::int32_t* cell_id = nullptr;
  std::uint16_t* group_id = nullptr;
  std::uint8_t* mode = nullptr;
  std::uint8_t* alive = nullptr;
};

PhotonPoolReadView make_read_view(const PhotonPool& pool) {
  PhotonPoolReadView view;
  view.pos_r = pool.pos_r;
  view.pos_z = pool.pos_z;
  view.dir_r = pool.dir_r;
  view.dir_z = pool.dir_z;
  view.dir_phi = pool.dir_phi;
  view.energy = pool.energy;
  view.weight = pool.weight;
  view.time_remain = pool.time_remain;
  view.birth_energy = pool.birth_energy;
  view.sign = pool.sign;
  view.global_id = pool.global_id;
  view.rng_counter = pool.rng_counter;
  view.cell_id = pool.cell_id;
  view.group_id = pool.group_id;
  view.mode = pool.mode;
  view.alive = pool.alive;
  return view;
}

PhotonPoolWriteView make_write_view(PhotonPool& pool) {
  PhotonPoolWriteView view;
  view.pos_r = pool.pos_r;
  view.pos_z = pool.pos_z;
  view.dir_r = pool.dir_r;
  view.dir_z = pool.dir_z;
  view.dir_phi = pool.dir_phi;
  view.energy = pool.energy;
  view.weight = pool.weight;
  view.time_remain = pool.time_remain;
  view.birth_energy = pool.birth_energy;
  view.sign = pool.sign;
  view.global_id = pool.global_id;
  view.rng_counter = pool.rng_counter;
  view.cell_id = pool.cell_id;
  view.group_id = pool.group_id;
  view.mode = pool.mode;
  view.alive = pool.alive;
  return view;
}

struct CompactAliveScratch {
  PhotonPool scratch_pool;
  int* d_compact_indices = nullptr;
  int* d_counts = nullptr;
  double* d_dropped_energy = nullptr;
  int index_capacity = 0;

  void ensure(const int required_capacity) {
    TENRYU_ASSERT(required_capacity >= 0,
                  "CompactAliveScratch::ensure requires capacity >= 0");
    if (d_counts == nullptr) {
      alloc_device(&d_counts,
                   4,
                   "compact_alive_only cudaMalloc persistent d_counts failed");
    }
    if (d_dropped_energy == nullptr) {
      alloc_device(&d_dropped_energy,
                   1,
                   "compact_alive_only cudaMalloc persistent d_dropped_energy failed");
    }
    if (required_capacity > scratch_pool.capacity) {
      scratch_pool.allocate(required_capacity);
    }
    if (required_capacity > index_capacity) {
      free_device(&d_compact_indices,
                  "compact_alive_only cudaFree persistent d_compact_indices failed");
      alloc_device(&d_compact_indices,
                   required_capacity,
                   "compact_alive_only cudaMalloc persistent d_compact_indices failed");
      index_capacity = required_capacity;
    }
  }

  void release() {
    free_device(&d_compact_indices,
                "compact_alive_only cudaFree persistent d_compact_indices failed");
    free_device(&d_dropped_energy,
                "compact_alive_only cudaFree persistent d_dropped_energy failed");
    free_device(&d_counts,
                "compact_alive_only cudaFree persistent d_counts failed");
    scratch_pool.release();
    index_capacity = 0;
  }
};

CompactAliveScratch& compact_alive_scratch() {
  static CompactAliveScratch scratch;
  return scratch;
}

__global__ void build_sort_keys_kernel(const std::int32_t* __restrict__ cell_id,
                                       const std::uint16_t* __restrict__ group_id,
                                       const std::uint8_t* __restrict__ mode,
                                       const std::uint8_t* __restrict__ alive,
                                       const double* __restrict__ energy,
                                       const std::uint64_t* __restrict__ global_id,
                                       std::uint64_t* __restrict__ keys,
                                       int* __restrict__ indices,
                                       const int n_particles,
                                       const int n_cells,
                                       const int n_groups,
                                       const int bucket_shift,
                                       const std::uint64_t id_mask,
                                       int* __restrict__ counts,
                                       double* __restrict__ dropped_energy) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  indices[tid] = tid;

  const bool alive_ok =
      (alive[tid] == static_cast<std::uint8_t>(kAlive)) && (cell_id[tid] >= 0) &&
      (cell_id[tid] < n_cells) && (static_cast<int>(group_id[tid]) < n_groups);

  if (!alive_ok) {
    keys[tid] = kInvalidSortKey;
    const double E = energy[tid];
    if (isfinite(E) && E != 0.0) {
      atomic_add_double(dropped_energy, E);
    }
    return;
  }

  const std::uint8_t mode_i = mode[tid];
  const std::uint64_t mode_stride = static_cast<std::uint64_t>(n_cells) *
                                    static_cast<std::uint64_t>(n_groups);
  const std::uint64_t cell_stride = static_cast<std::uint64_t>(n_groups);
  const std::uint64_t bucket_key =
      static_cast<std::uint64_t>(mode_i) * mode_stride +
      static_cast<std::uint64_t>(cell_id[tid]) * cell_stride +
      static_cast<std::uint64_t>(group_id[tid]);
  keys[tid] = (bucket_key << bucket_shift) | (global_id[tid] & id_mask);

  atomicAdd(&counts[0], 1);
  if (mode_i == static_cast<std::uint8_t>(kModeIMC)) {
    atomicAdd(&counts[1], 1);
  } else if (mode_i == static_cast<std::uint8_t>(kModeDDMC)) {
    atomicAdd(&counts[2], 1);
  } else if (mode_i == static_cast<std::uint8_t>(kModeRW)) {
    atomicAdd(&counts[3], 1);
  }
}

__global__ void gather_sorted_kernel(const PhotonPoolReadView src,
                                     const PhotonPoolWriteView dst,
                                     const int* __restrict__ sorted_indices,
                                     const int* __restrict__ counts,
                                     const int n_particles) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  const int src_idx = sorted_indices[tid];
  const int n_alive = counts[0];

  dst.pos_r[tid] = src.pos_r[src_idx];
  dst.pos_z[tid] = src.pos_z[src_idx];
  dst.dir_r[tid] = src.dir_r[src_idx];
  dst.dir_z[tid] = src.dir_z[src_idx];
  dst.dir_phi[tid] = src.dir_phi[src_idx];
  dst.energy[tid] = src.energy[src_idx];
  dst.weight[tid] = src.weight[src_idx];
  dst.time_remain[tid] = src.time_remain[src_idx];
  dst.birth_energy[tid] = src.birth_energy[src_idx];
  dst.sign[tid] = src.sign[src_idx];
  dst.global_id[tid] = src.global_id[src_idx];
  dst.rng_counter[tid] = src.rng_counter[src_idx];
  dst.cell_id[tid] = src.cell_id[src_idx];
  dst.group_id[tid] = src.group_id[src_idx];
  dst.mode[tid] = src.mode[src_idx];
  dst.alive[tid] = (tid < n_alive) ? static_cast<std::uint8_t>(kAlive)
                                   : static_cast<std::uint8_t>(kDead);
}

__global__ void compact_alive_kernel(const PhotonPoolReadView src,
                                     int* __restrict__ compact_indices,
                                     const int n_particles,
                                     const int n_cells,
                                     const int n_groups,
                                     int* __restrict__ counts,
                                     double* __restrict__ dropped_energy) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  const bool alive_ok =
      (src.alive[tid] == static_cast<std::uint8_t>(kAlive)) && (src.cell_id[tid] >= 0) &&
      (src.cell_id[tid] < n_cells) && (static_cast<int>(src.group_id[tid]) < n_groups);
  if (!alive_ok) {
    const double E = src.energy[tid];
    if (isfinite(E) && E != 0.0) {
      atomic_add_double(dropped_energy, E);
    }
    return;
  }

  const int pos = atomicAdd(&counts[0], 1);
  compact_indices[pos] = tid;

  const std::uint8_t mode_i = src.mode[tid];
  if (mode_i == static_cast<std::uint8_t>(kModeIMC)) {
    atomicAdd(&counts[1], 1);
  } else if (mode_i == static_cast<std::uint8_t>(kModeDDMC)) {
    atomicAdd(&counts[2], 1);
  } else if (mode_i == static_cast<std::uint8_t>(kModeRW)) {
    atomicAdd(&counts[3], 1);
  }
}

__global__ void gather_compacted_soa_kernel(const PhotonPoolReadView src,
                                            const PhotonPoolWriteView dst,
                                            const int* __restrict__ compact_indices,
                                            const int* __restrict__ counts,
                                            double* __restrict__ E_numerical_loss,
                                            const double* __restrict__ dropped_energy,
                                            const int n_particles) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  if (tid == 0 && E_numerical_loss != nullptr) {
    const double dropped = dropped_energy[0];
    if (dropped != 0.0) {
      atomic_add_double(E_numerical_loss, fabs(dropped));
    }
  }

  const int n_alive = counts[0];
  if (tid >= n_alive) {
    return;
  }

  const int src_idx = compact_indices[tid];
  dst.pos_r[tid] = src.pos_r[src_idx];
  dst.pos_z[tid] = src.pos_z[src_idx];
  dst.dir_r[tid] = src.dir_r[src_idx];
  dst.dir_z[tid] = src.dir_z[src_idx];
  dst.dir_phi[tid] = src.dir_phi[src_idx];
  dst.energy[tid] = src.energy[src_idx];
  dst.weight[tid] = src.weight[src_idx];
  dst.time_remain[tid] = src.time_remain[src_idx];
  dst.birth_energy[tid] = src.birth_energy[src_idx];
  dst.sign[tid] = src.sign[src_idx];
  dst.global_id[tid] = src.global_id[src_idx];
  dst.rng_counter[tid] = src.rng_counter[src_idx];
  dst.cell_id[tid] = src.cell_id[src_idx];
  dst.group_id[tid] = src.group_id[src_idx];
  dst.mode[tid] = src.mode[src_idx];
  dst.alive[tid] = static_cast<std::uint8_t>(kAlive);
}

__global__ void accumulate_dropped_energy_kernel(double* __restrict__ E_numerical_loss,
                                                 const double* __restrict__ dropped_energy) {
  if (blockIdx.x != 0 || threadIdx.x != 0 || E_numerical_loss == nullptr) {
    return;
  }

  const double dropped = dropped_energy[0];
  if (dropped != 0.0) {
    atomic_add_double(E_numerical_loss, fabs(dropped));
  }
}

}  // namespace

CompositeSortResult composite_sort_and_partition(PhotonPool& pool,
                                                 const int n_particles,
                                                 const int n_cells,
                                                 const int n_groups,
                                                 double* const E_numerical_loss) {
  TENRYU_ASSERT(n_particles >= 0, "composite_sort requires n_particles >= 0");
  TENRYU_ASSERT(n_cells >= 0, "composite_sort requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 1, "composite_sort requires n_groups >= 1");
  if (n_particles == 0) {
    pool.n_alive = 0;
    return {};
  }

  std::uint64_t* d_keys_in = nullptr;
  std::uint64_t* d_keys_out = nullptr;
  int* d_indices_in = nullptr;
  int* d_indices_out = nullptr;
  void* d_sort_temp_storage = nullptr;
  std::size_t sort_temp_storage_bytes = 0;
  int* d_counts_managed = nullptr;        // [0]=n_alive, [1]=n_imc, [2]=n_ddmc, [3]=n_rw
  double* d_dropped_energy = nullptr;     // signed sum over dropped particles

  alloc_device(&d_keys_in,
               n_particles,
               "composite_sort cudaMalloc d_keys_in failed");
  alloc_device(&d_keys_out,
               n_particles,
               "composite_sort cudaMalloc d_keys_out failed");
  alloc_device(&d_indices_in,
               n_particles,
               "composite_sort cudaMalloc d_indices_in failed");
  alloc_device(&d_indices_out,
               n_particles,
               "composite_sort cudaMalloc d_indices_out failed");
  alloc_managed(&d_counts_managed,
                4,
                "composite_sort cudaMallocManaged d_counts failed");
  alloc_managed(&d_dropped_energy,
                1,
                "composite_sort cudaMallocManaged d_dropped_energy failed");
  cuda_check(cudaMemset(d_counts_managed, 0, sizeof(int) * 4),
             "composite_sort cudaMemset d_counts failed");
  cuda_check(cudaMemset(d_dropped_energy, 0, sizeof(double)),
             "composite_sort cudaMemset d_dropped_energy failed");

  // Compute bucket_shift dynamically to avoid overflow.
  // bucket_key ∈ [0, 3*n_cells*n_groups) needs ceil(log2(3*n_cells*n_groups)) bits.
  // Remaining bits store global_id for deterministic tie-breaking.
  const std::uint64_t max_bucket =
      static_cast<std::uint64_t>(3) * static_cast<std::uint64_t>(n_cells) *
      static_cast<std::uint64_t>(n_groups);
  int bucket_bits = 1;
  {
    std::uint64_t v = max_bucket;
    while (v > (1ULL << bucket_bits)) {
      ++bucket_bits;
    }
  }
  // Reserve 1 extra bit for the dead-particle sentinel (kInvalidSortKey)
  bucket_bits = std::min(bucket_bits + 1, 63);
  const int bucket_shift = 64 - bucket_bits;
  const std::uint64_t id_mask = (bucket_shift >= 64) ? 0ULL : ((1ULL << bucket_shift) - 1ULL);

  const int blocks = (n_particles + kThreadsPerBlock - 1) / kThreadsPerBlock;
  build_sort_keys_kernel<<<blocks, kThreadsPerBlock>>>(pool.cell_id,
                                                        pool.group_id,
                                                        pool.mode,
                                                        pool.alive,
                                                        pool.energy,
                                                        pool.global_id,
                                                        d_keys_in,
                                                        d_indices_in,
                                                        n_particles,
                                                        n_cells,
                                                        n_groups,
                                                        bucket_shift,
                                                        id_mask,
                                                        d_counts_managed,
                                                        d_dropped_energy);
  cuda_check(cudaGetLastError(),
             "composite_sort launch build_sort_keys_kernel failed");

  cub::DoubleBuffer<std::uint64_t> d_keys(d_keys_in, d_keys_out);
  cub::DoubleBuffer<int> d_indices(d_indices_in, d_indices_out);
  cuda_check(cub::DeviceRadixSort::SortPairs(nullptr,
                                             sort_temp_storage_bytes,
                                             d_keys,
                                             d_indices,
                                             n_particles,
                                             0,
                                             64),
             "composite_sort CUB sort temp-storage query failed");
  if (sort_temp_storage_bytes > 0) {
    cuda_check(cudaMalloc(&d_sort_temp_storage, sort_temp_storage_bytes),
               "composite_sort cudaMalloc d_sort_temp_storage failed");
  }
  cuda_check(cub::DeviceRadixSort::SortPairs(d_sort_temp_storage,
                                             sort_temp_storage_bytes,
                                             d_keys,
                                             d_indices,
                                             n_particles,
                                             0,
                                             64),
             "composite_sort CUB sort failed");

  PhotonPool sorted_pool;
  sorted_pool.allocate(n_particles);
  const PhotonPoolReadView src_view = make_read_view(pool);
  const PhotonPoolWriteView dst_view = make_write_view(sorted_pool);
  gather_sorted_kernel<<<blocks, kThreadsPerBlock>>>(
      src_view, dst_view, d_indices.Current(), d_counts_managed, n_particles);
  cuda_check(cudaGetLastError(),
             "composite_sort launch gather_sorted_kernel failed");

  d2d_copy(pool.pos_r,
           sorted_pool.pos_r,
           n_particles,
           "composite_sort copy back pos_r failed");
  d2d_copy(pool.pos_z,
           sorted_pool.pos_z,
           n_particles,
           "composite_sort copy back pos_z failed");
  d2d_copy(pool.dir_r,
           sorted_pool.dir_r,
           n_particles,
           "composite_sort copy back dir_r failed");
  d2d_copy(pool.dir_z,
           sorted_pool.dir_z,
           n_particles,
           "composite_sort copy back dir_z failed");
  d2d_copy(pool.dir_phi,
           sorted_pool.dir_phi,
           n_particles,
           "composite_sort copy back dir_phi failed");
  d2d_copy(pool.energy,
           sorted_pool.energy,
           n_particles,
           "composite_sort copy back energy failed");
  d2d_copy(pool.weight,
           sorted_pool.weight,
           n_particles,
           "composite_sort copy back weight failed");
  d2d_copy(pool.time_remain,
           sorted_pool.time_remain,
           n_particles,
           "composite_sort copy back time_remain failed");
  d2d_copy(pool.birth_energy,
           sorted_pool.birth_energy,
           n_particles,
           "composite_sort copy back birth_energy failed");
  d2d_copy(pool.sign,
           sorted_pool.sign,
           n_particles,
           "composite_sort copy back sign failed");
  d2d_copy(pool.global_id,
           sorted_pool.global_id,
           n_particles,
           "composite_sort copy back global_id failed");
  d2d_copy(pool.rng_counter,
           sorted_pool.rng_counter,
           n_particles,
           "composite_sort copy back rng_counter failed");
  d2d_copy(pool.cell_id,
           sorted_pool.cell_id,
           n_particles,
           "composite_sort copy back cell_id failed");
  d2d_copy(pool.group_id,
           sorted_pool.group_id,
           n_particles,
           "composite_sort copy back group_id failed");
  d2d_copy(pool.mode,
           sorted_pool.mode,
           n_particles,
           "composite_sort copy back mode failed");
  d2d_copy(pool.alive,
           sorted_pool.alive,
           n_particles,
           "composite_sort copy back alive failed");

  if (E_numerical_loss != nullptr) {
    accumulate_dropped_energy_kernel<<<1, 1>>>(E_numerical_loss, d_dropped_energy);
    cuda_check(cudaGetLastError(),
               "composite_sort launch accumulate_dropped_energy_kernel failed");
  }

  cuda_check(cudaDeviceSynchronize(), "composite_sort cudaDeviceSynchronize failed");

  const int n_alive = d_counts_managed[0];
  const int n_imc = d_counts_managed[1];
  const int n_ddmc = d_counts_managed[2];
  const int n_rw = d_counts_managed[3];

  if (d_sort_temp_storage != nullptr) {
    cuda_check(cudaFree(d_sort_temp_storage),
               "composite_sort cudaFree d_sort_temp_storage failed");
  }
  cuda_check(cudaFree(d_indices_out),
             "composite_sort cudaFree d_indices_out failed");
  cuda_check(cudaFree(d_indices_in),
             "composite_sort cudaFree d_indices_in failed");
  cuda_check(cudaFree(d_keys_out), "composite_sort cudaFree d_keys_out failed");
  cuda_check(cudaFree(d_keys_in), "composite_sort cudaFree d_keys_in failed");
  cuda_check(cudaFree(d_dropped_energy),
             "composite_sort cudaFree d_dropped_energy failed");
  cuda_check(cudaFree(d_counts_managed),
             "composite_sort cudaFree d_counts_managed failed");

  pool.n_alive = n_alive;
  CompositeSortResult out;
  out.n_alive = n_alive;
  out.n_imc = n_imc;
  out.n_ddmc = n_ddmc;
  out.n_rw = n_rw;
  return out;
}

CompositeSortResult compact_alive_only(PhotonPool& pool,
                                       const int n_particles,
                                       const int n_cells,
                                       const int n_groups,
                                       double* const E_numerical_loss) {
  TENRYU_ASSERT(n_particles >= 0, "compact_alive_only requires n_particles >= 0");
  TENRYU_ASSERT(n_cells >= 0, "compact_alive_only requires n_cells >= 0");
  TENRYU_ASSERT(n_groups >= 1, "compact_alive_only requires n_groups >= 1");
  if (n_particles == 0) {
    pool.n_alive = 0;
    return {};
  }
  TENRYU_ASSERT(n_particles <= pool.capacity,
                "compact_alive_only requires n_particles <= pool.capacity");

  CompactAliveScratch& scratch = compact_alive_scratch();
  scratch.ensure(pool.capacity);

  cuda_check(cudaMemset(scratch.d_counts, 0, sizeof(int) * 4),
             "compact_alive_only cudaMemset d_counts failed");
  cuda_check(cudaMemset(scratch.d_dropped_energy, 0, sizeof(double)),
             "compact_alive_only cudaMemset d_dropped_energy failed");

  const int blocks = (n_particles + kThreadsPerBlock - 1) / kThreadsPerBlock;
  const PhotonPoolReadView src_view = make_read_view(pool);
  const PhotonPoolWriteView dst_view = make_write_view(scratch.scratch_pool);
  compact_alive_kernel<<<blocks, kThreadsPerBlock>>>(src_view,
                                                     scratch.d_compact_indices,
                                                     n_particles,
                                                     n_cells,
                                                     n_groups,
                                                     scratch.d_counts,
                                                     scratch.d_dropped_energy);
  cuda_check(cudaGetLastError(),
             "compact_alive_only launch compact_alive_kernel failed");
  gather_compacted_soa_kernel<<<blocks, kThreadsPerBlock>>>(src_view,
                                                            dst_view,
                                                            scratch.d_compact_indices,
                                                            scratch.d_counts,
                                                            E_numerical_loss,
                                                            scratch.d_dropped_energy,
                                                            n_particles);
  cuda_check(cudaGetLastError(),
             "compact_alive_only launch gather_compacted_soa_kernel failed");

  cuda_check(cudaDeviceSynchronize(),
             "compact_alive_only cudaDeviceSynchronize(gather) failed");

  int h_counts[4] = {0, 0, 0, 0};
  cuda_check(cudaMemcpy(h_counts,
                        scratch.d_counts,
                        sizeof(h_counts),
                        cudaMemcpyDeviceToHost),
             "compact_alive_only copy counts failed");

  const int n_alive = h_counts[0];
  const int n_imc = h_counts[1];
  const int n_ddmc = h_counts[2];
  const int n_rw = h_counts[3];

  const int old_n_census = pool.n_census;
  scratch.scratch_pool.n_alive = n_alive;
  scratch.scratch_pool.n_census = old_n_census;
  pool.swap(scratch.scratch_pool);
  pool.n_alive = n_alive;
  pool.n_census = old_n_census;
  CompositeSortResult out;
  out.n_alive = n_alive;
  out.n_imc = n_imc;
  out.n_ddmc = n_ddmc;
  out.n_rw = n_rw;
  return out;
}

void reserve_compact_alive_scratch(const int capacity) {
  compact_alive_scratch().ensure(capacity);
}

void release_compact_alive_scratch() {
  compact_alive_scratch().release();
}

}  // namespace tenryu::radiation
