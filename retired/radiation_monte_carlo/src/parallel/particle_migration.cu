#include "parallel/particle_migration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include "core/error.hpp"

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::parallel {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr double kTwoPi = 6.28318530717958647692;
#if TENRYU_ENABLE_MPI
constexpr int kMigrationPhaseId = 50;
constexpr int kMigrationPayloadSlot = 0;
constexpr int kMigrationCountSlot = 1;
#endif

std::string cuda_error_message(const char* operation, const cudaError_t err) {
  return std::string(operation) + " failed: " + cudaGetErrorString(err);
}

template <typename T>
void release_device_ptr(T*& ptr) {
  if (ptr == nullptr) {
    return;
  }
  const cudaError_t err = cudaFree(ptr);
  TENRYU_ASSERT(err == cudaSuccess, cuda_error_message("cudaFree", err));
  ptr = nullptr;
}

void clear_per_dest(EmigrantBuffer& buffer) {
  for (int dir = 0; dir < MAX_NEIGHBORS; ++dir) {
    buffer.per_dest_count[dir] = 0;
    buffer.per_dest_offset[dir] = 0;
  }
}

bool is_direction_active(const PartitionInfo& part, const int direction) {
  if (direction == LEFT || direction == RIGHT) {
    return true;
  }
  if (part.cart_dims[1] == 1) {
    return false;
  }
  return true;
}

bool census_verify_enabled() {
  const char* env = std::getenv("TENRYU_VERIFY_CENSUS");
  if (env == nullptr || env[0] == '\0') {
    return false;
  }
  const char c = env[0];
  return (c == '1' || c == 't' || c == 'T' || c == 'y' || c == 'Y');
}

bool census_verify_enabled_cached() {
  static const bool enabled = census_verify_enabled();
  return enabled;
}

#if TENRYU_ENABLE_MPI
int opposite_direction(const int direction) {
  switch (direction) {
    case LEFT:
      return RIGHT;
    case RIGHT:
      return LEFT;
    case BOTTOM:
      return TOP;
    case TOP:
      return BOTTOM;
    case NE:
      return SW;
    case NW:
      return SE;
    case SE:
      return NW;
    case SW:
      return NE;
    default:
      return direction;
  }
}

int migration_tag(const int phase_id, const int direction, const int slot) {
  return phase_id * 1000 + direction * 100 + slot;
}

std::string mpi_error_message(const char* operation) {
  return std::string(operation) + " failed in particle migration";
}

MPI_Comm migration_comm(const PartitionInfo& part) {
  if (part.cart_comm != MPI_COMM_NULL) {
    return part.cart_comm;
  }
  return MPI_COMM_WORLD;
}
#endif

__device__ inline std::uint64_t splitmix64_next(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

__device__ inline double u01_from_u64(const std::uint64_t x) {
  constexpr double kInv53 = 1.0 / 9007199254740992.0;
  return static_cast<double>(x >> 11) * kInv53;
}

__device__ inline void sample_isotropic_direction_from_seed(
    const std::uint64_t seed, double* const dir_r, double* const dir_z,
    double* const dir_phi) {
  const std::uint64_t s0 = splitmix64_next(seed);
  const std::uint64_t s1 = splitmix64_next(s0);
  const double xi_mu = u01_from_u64(s0);
  const double xi_phi = u01_from_u64(s1);
  const double mu = 2.0 * xi_mu - 1.0;
  const double phi = kTwoPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  *dir_r = mu;
  *dir_z = sin_theta * cos(phi);
  *dir_phi = sin_theta * sin(phi);
}

__global__ void detect_emigrants_kernel(
    ParticleEmigrant* __restrict__ packed,
    std::int32_t* __restrict__ packed_dest_rank, int* __restrict__ packed_count,
    const double* __restrict__ pos_r, const double* __restrict__ pos_z,
    const double* __restrict__ dir_r, const double* __restrict__ dir_z,
    const double* __restrict__ dir_phi, const double* __restrict__ energy,
    const double* __restrict__ weight, const double* __restrict__ time_remain,
    const double* __restrict__ birth_energy,
    const std::int8_t* __restrict__ sign,
    const std::uint64_t* __restrict__ global_id,
    const std::uint32_t* __restrict__ rng_counter,
    const std::int32_t* __restrict__ cell_id,
    const std::uint16_t* __restrict__ group_id,
    const std::uint8_t* __restrict__ mode, std::uint8_t* __restrict__ alive,
    const int n_particles, const int capacity, const int left_rank,
    const int right_rank, const int bottom_rank, const int top_rank) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles) {
    return;
  }

  if (alive[idx] != static_cast<std::uint8_t>(radiation::kAlive)) {
    return;
  }

  const int marker = static_cast<int>(cell_id[idx]);
  if (marker >= 0) {
    return;
  }

  const int face = -marker - 100;
  if (face < 0 || face > 3) {
    return;
  }

  int dest = -1;
  switch (face) {
    case 0:
      dest = left_rank;
      break;
    case 1:
      dest = right_rank;
      break;
    case 2:
      dest = bottom_rank;
      break;
    case 3:
      dest = top_rank;
      break;
    default:
      break;
  }
  if (dest < 0) {
    return;
  }

  const int slot = atomicAdd(packed_count, 1);
  if (slot >= capacity) {
    return;
  }

  ParticleEmigrant out{};
  out.position[0] = pos_r[idx];
  out.position[1] = pos_z[idx];
  out.position[2] = 0.0;
  out.direction[0] = dir_r[idx];
  out.direction[1] = dir_z[idx];
  out.direction[2] = dir_phi[idx];
  out.energy = energy[idx];
  out.weight = weight[idx];
  out.time_remain = time_remain[idx];
  out.birth_energy = birth_energy[idx];
  out.global_id = global_id[idx];
  out.rng_counter = rng_counter[idx];
  out.cell_id_src = marker;
  out.group = group_id[idx];
  out.sign = sign[idx];
  out.leak_face = static_cast<std::int8_t>(face);
  out.mode = mode[idx];
  out.padding[0] = 0;
  out.padding[1] = 0;
  out.padding[2] = 0;

  packed[slot] = out;
  packed_dest_rank[slot] = dest;
  alive[idx] = static_cast<std::uint8_t>(radiation::kDead);
}

__global__ void gather_emigrants_kernel(
    ParticleEmigrant* __restrict__ dst,
    const ParticleEmigrant* __restrict__ src,
    const int* __restrict__ indices, const int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  dst[i] = src[indices[i]];
}

__global__ void merge_immigrants_kernel(
    const ParticleEmigrant* __restrict__ recv, const int count, const int base,
    double* __restrict__ pos_r, double* __restrict__ pos_z,
    double* __restrict__ dir_r, double* __restrict__ dir_z,
    double* __restrict__ dir_phi, double* __restrict__ energy,
    double* __restrict__ weight, double* __restrict__ time_remain,
    double* __restrict__ birth_energy, std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter, std::int32_t* __restrict__ cell_id,
    std::uint16_t* __restrict__ group_id, std::int8_t* __restrict__ sign,
    std::uint8_t* __restrict__ mode, std::uint8_t* __restrict__ alive,
    const std::int8_t* __restrict__ ddmc_mode_map,
    const int ddmc_mode_map_size, const int ddmc_mode_stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }

  const int dst = base + idx;
  const ParticleEmigrant in = recv[idx];
  double pos_r_out = in.position[0];
  double pos_z_out = in.position[1];
  double dir_r_out = in.direction[0];
  double dir_z_out = in.direction[1];
  double dir_phi_out = in.direction[2];
  double t_remain_out = in.time_remain;
  std::uint8_t mode_out = in.mode;

  if (ddmc_mode_map != nullptr && ddmc_mode_map_size > 0 && in.cell_id_src >= 0) {
    const int stride = (ddmc_mode_stride > 0) ? ddmc_mode_stride : 1;
    const int mode_idx =
        in.cell_id_src * stride + (static_cast<int>(in.group) % stride);
    if (mode_idx >= 0 && mode_idx < ddmc_mode_map_size) {
      const std::int8_t mode_map_value = ddmc_mode_map[mode_idx];
      std::uint8_t dst_mode = static_cast<std::uint8_t>(radiation::kModeIMC);
      if (mode_map_value == static_cast<std::int8_t>(radiation::kModeDDMC)) {
        dst_mode = static_cast<std::uint8_t>(radiation::kModeDDMC);
      } else if (mode_map_value == static_cast<std::int8_t>(radiation::kModeRW)) {
        dst_mode = static_cast<std::uint8_t>(radiation::kModeRW);
      }
      if (mode_out != dst_mode) {
        if (mode_out == static_cast<std::uint8_t>(radiation::kModeDDMC) &&
            dst_mode == static_cast<std::uint8_t>(radiation::kModeIMC)) {
          // Cell-center geometry is not available in this layer. Keep finite position.
          if (!isfinite(pos_r_out)) {
            pos_r_out = 0.0;
          }
          if (!isfinite(pos_z_out)) {
            pos_z_out = 0.0;
          }
          const std::uint64_t seed =
              in.global_id ^
              (static_cast<std::uint64_t>(in.rng_counter) << 32) ^
              0xD6E8FEB86659FD93ULL;
          sample_isotropic_direction_from_seed(seed, &dir_r_out, &dir_z_out, &dir_phi_out);
          t_remain_out = 0.0;
          mode_out = dst_mode;
        } else if (mode_out == static_cast<std::uint8_t>(radiation::kModeIMC) &&
                   dst_mode == static_cast<std::uint8_t>(radiation::kModeDDMC)) {
          const double nan = NAN;
          pos_r_out = nan;
          pos_z_out = nan;
          dir_r_out = nan;
          dir_z_out = nan;
          dir_phi_out = nan;
          t_remain_out = 0.0;
          mode_out = dst_mode;
        }
      }
    }
  }

  pos_r[dst] = pos_r_out;
  pos_z[dst] = pos_z_out;
  dir_r[dst] = dir_r_out;
  dir_z[dst] = dir_z_out;
  dir_phi[dst] = dir_phi_out;
  energy[dst] = in.energy;
  weight[dst] = in.weight;
  time_remain[dst] = t_remain_out;
  birth_energy[dst] = in.birth_energy;
  global_id[dst] = in.global_id;
  rng_counter[dst] = in.rng_counter;
  cell_id[dst] = -1;
  group_id[dst] = in.group;
  sign[dst] = in.sign;
  mode[dst] = mode_out;
  alive[dst] = static_cast<std::uint8_t>(radiation::kAlive);
}

__global__ void count_alive_negative_cell_kernel(
    const std::uint8_t* __restrict__ alive, const std::int32_t* __restrict__ cell_id,
    const int n_particles, int* __restrict__ count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_particles) {
    return;
  }
  if (alive[idx] == static_cast<std::uint8_t>(radiation::kAlive) &&
      cell_id[idx] < 0) {
    atomicAdd(count, 1);
  }
}

void verify_census(const PartitionInfo& part, const radiation::PhotonPool* pool,
                   const int local_sent, const int local_recv,
                   cudaStream_t stream) {
  const bool verify_enabled = census_verify_enabled_cached();

  int local_unpacked = 0;
  if (verify_enabled) {
    if (pool != nullptr && pool->n_alive > 0 && pool->alive != nullptr &&
        pool->cell_id != nullptr) {
      int* d_unpacked = nullptr;
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_unpacked), sizeof(int)));
      CUDA_CHECK(cudaMemsetAsync(d_unpacked, 0, sizeof(int), stream));
      const int blocks = (pool->n_alive + kThreadsPerBlock - 1) / kThreadsPerBlock;
      count_alive_negative_cell_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
          pool->alive, pool->cell_id, pool->n_alive, d_unpacked);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpyAsync(&local_unpacked, d_unpacked, sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaFree(d_unpacked));
    }
  }

  long long global_sent = -1;
  long long global_recv = -1;
#if TENRYU_ENABLE_MPI
  if (part.n_ranks > 1 && local_sent >= 0 && local_recv >= 0) {
    long long local_counts[2] = {
        static_cast<long long>(local_sent),
        static_cast<long long>(local_recv)};
    long long global_counts[2] = {0LL, 0LL};
    const int allreduce_err =
        MPI_Allreduce(local_counts, global_counts, 2, MPI_LONG_LONG, MPI_SUM,
                      migration_comm(part));
    TENRYU_ASSERT(allreduce_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Allreduce"));
    global_sent = global_counts[0];
    global_recv = global_counts[1];
  }
#else
  (void)part;
#endif

  if (!verify_enabled) {
    return;
  }

  if (local_unpacked > 0) {
    core::log_warning("particle migration census verify: alive particles with cell_id<0 remain "
                      "(rank=" +
                      std::to_string(part.rank) + ", count=" +
                      std::to_string(local_unpacked) + ")");
  }

  if (global_sent >= 0 && global_recv >= 0 && global_sent != global_recv) {
    core::log_warning("particle migration census verify: global send/recv mismatch "
                      "(rank=" + std::to_string(part.rank) +
                      ", local_sent=" + std::to_string(local_sent) +
                      ", local_recv=" + std::to_string(local_recv) +
                      ", global_sent=" + std::to_string(global_sent) +
                      ", global_recv=" + std::to_string(global_recv) + ")");
  }
}

}  // namespace

EmigrantBuffer::EmigrantBuffer(EmigrantBuffer&& other) noexcept {
  *this = std::move(other);
}

EmigrantBuffer& EmigrantBuffer::operator=(EmigrantBuffer&& other) noexcept {
  if (this != &other) {
    release_device_ptr(data);
    release_device_ptr(dest_rank);

    data = other.data;
    dest_rank = other.dest_rank;
    count = other.count;
    capacity = other.capacity;
    for (int dir = 0; dir < MAX_NEIGHBORS; ++dir) {
      per_dest_count[dir] = other.per_dest_count[dir];
      per_dest_offset[dir] = other.per_dest_offset[dir];
      other.per_dest_count[dir] = 0;
      other.per_dest_offset[dir] = 0;
    }

    other.data = nullptr;
    other.dest_rank = nullptr;
    other.count = 0;
    other.capacity = 0;
  }
  return *this;
}

void EmigrantBuffer::resize_if_needed(const int required) {
  TENRYU_ASSERT(required >= 0,
                "EmigrantBuffer::resize_if_needed requires required >= 0");
  if (required <= capacity) {
    return;
  }

  ParticleEmigrant* new_data = nullptr;
  std::int32_t* new_dest_rank = nullptr;

  if (required > 0) {
    const cudaError_t alloc_data_err = cudaMalloc(
        reinterpret_cast<void**>(&new_data),
        sizeof(ParticleEmigrant) * static_cast<std::size_t>(required));
    TENRYU_ASSERT(alloc_data_err == cudaSuccess,
                  cuda_error_message("cudaMalloc", alloc_data_err));

    const cudaError_t alloc_rank_err =
        cudaMalloc(reinterpret_cast<void**>(&new_dest_rank),
                   sizeof(std::int32_t) * static_cast<std::size_t>(required));
    TENRYU_ASSERT(alloc_rank_err == cudaSuccess,
                  cuda_error_message("cudaMalloc", alloc_rank_err));
  }

  if (count > 0 && data != nullptr && dest_rank != nullptr) {
    const int copy_count = std::min(count, required);
    const cudaError_t copy_data_err = cudaMemcpy(
        new_data, data, sizeof(ParticleEmigrant) * static_cast<std::size_t>(copy_count),
        cudaMemcpyDeviceToDevice);
    TENRYU_ASSERT(copy_data_err == cudaSuccess,
                  cuda_error_message("cudaMemcpy", copy_data_err));

    const cudaError_t copy_rank_err = cudaMemcpy(
        new_dest_rank, dest_rank,
        sizeof(std::int32_t) * static_cast<std::size_t>(copy_count),
        cudaMemcpyDeviceToDevice);
    TENRYU_ASSERT(copy_rank_err == cudaSuccess,
                  cuda_error_message("cudaMemcpy", copy_rank_err));
  }

  release_device_ptr(data);
  release_device_ptr(dest_rank);
  data = new_data;
  dest_rank = new_dest_rank;
  capacity = required;
  if (count > capacity) {
    count = capacity;
  }
}

EmigrantBuffer::~EmigrantBuffer() {
  release_device_ptr(data);
  release_device_ptr(dest_rank);
  count = 0;
  capacity = 0;
  clear_per_dest(*this);
}

int detect_emigrants(const radiation::PhotonPool& pool, const PartitionInfo& part,
                     EmigrantBuffer& emigrants, cudaStream_t stream) {
  clear_per_dest(emigrants);
  emigrants.count = 0;

  if (part.n_ranks <= 1 || pool.n_alive <= 0) {
    return 0;
  }

  emigrants.resize_if_needed(pool.n_alive);
  TENRYU_ASSERT(emigrants.data != nullptr && emigrants.dest_rank != nullptr,
                "detect_emigrants requires allocated EmigrantBuffer");

  int* device_count = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_count), sizeof(int)));
  CUDA_CHECK(cudaMemsetAsync(device_count, 0, sizeof(int), stream));

  const int blocks = (pool.n_alive + kThreadsPerBlock - 1) / kThreadsPerBlock;
  detect_emigrants_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      emigrants.data, emigrants.dest_rank, device_count, pool.pos_r, pool.pos_z,
      pool.dir_r, pool.dir_z, pool.dir_phi, pool.energy, pool.weight,
      pool.time_remain, pool.birth_energy, pool.sign, pool.global_id, pool.rng_counter,
      pool.cell_id, pool.group_id, pool.mode, pool.alive, pool.n_alive,
      emigrants.capacity, part.neighbor_ranks[LEFT], part.neighbor_ranks[RIGHT],
      part.neighbor_ranks[BOTTOM], part.neighbor_ranks[TOP]);
  CUDA_CHECK(cudaGetLastError());

  int detected = 0;
  CUDA_CHECK(cudaMemcpyAsync(&detected, device_count, sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaFree(device_count));

  verify_census(part, &pool, detected, -1, stream);

  if (detected <= 0) {
    return 0;
  }

  TENRYU_ASSERT(detected <= emigrants.capacity,
                "detect_emigrants exceeded EmigrantBuffer capacity");
  emigrants.count = detected;

  int* d_indices = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices),
                        sizeof(int) * static_cast<std::size_t>(detected)));
  ParticleEmigrant* d_temp = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),
                        sizeof(ParticleEmigrant) *
                            static_cast<std::size_t>(detected)));

  auto keys = thrust::device_pointer_cast(emigrants.dest_rank);
  auto indices = thrust::device_pointer_cast(d_indices);
  thrust::sequence(thrust::cuda::par.on(stream), indices, indices + detected);
  CUDA_CHECK(cudaGetLastError());

  thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + detected, indices);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(
      d_temp, emigrants.data,
      sizeof(ParticleEmigrant) * static_cast<std::size_t>(detected),
      cudaMemcpyDeviceToDevice, stream));

  const int gather_blocks = (detected + kThreadsPerBlock - 1) / kThreadsPerBlock;
  gather_emigrants_kernel<<<gather_blocks, kThreadsPerBlock, 0, stream>>>(
      emigrants.data, d_temp, d_indices, detected);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaFree(d_temp));
  CUDA_CHECK(cudaFree(d_indices));

  std::vector<std::int32_t> host_dest(static_cast<std::size_t>(detected));
  CUDA_CHECK(cudaMemcpyAsync(host_dest.data(), emigrants.dest_rank,
                             sizeof(std::int32_t) * host_dest.size(),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  for (int dir = 0; dir < MAX_NEIGHBORS; ++dir) {
    if (!is_direction_active(part, dir)) {
      continue;
    }
    const int dest = part.neighbor_ranks[dir];
    if (dest < 0) {
      continue;
    }
    const auto range = std::equal_range(host_dest.begin(), host_dest.end(), dest);
    emigrants.per_dest_offset[dir] =
        static_cast<int>(range.first - host_dest.begin());
    emigrants.per_dest_count[dir] = static_cast<int>(range.second - range.first);
  }

  return detected;
}

void exchange_emigrants(const PartitionInfo& part, CommBuffers& buffers,
                        const EmigrantBuffer& send, EmigrantBuffer& recv,
                        cudaStream_t stream) {
  clear_per_dest(recv);
  recv.count = 0;

  if (part.n_ranks <= 1) {
    return;
  }

#if TENRYU_ENABLE_MPI
  CUDA_CHECK(cudaStreamSynchronize(stream));

  bool use_gpu_aware_mpi = buffers.gpu_aware_mpi;
#if TENRYU_GPU_AWARE_MPI_COMPILE == 0
  use_gpu_aware_mpi = false;
#endif
  if (!use_gpu_aware_mpi && buffers.gpu_aware_mpi) {
    buffers.gpu_aware_mpi = false;
  }

  std::array<int, MAX_NEIGHBORS> active_dirs{};
  int active_count = 0;
  for (int dir = 0; dir < MAX_NEIGHBORS; ++dir) {
    if (!is_direction_active(part, dir)) {
      continue;
    }
    if (part.neighbor_ranks[dir] < 0) {
      continue;
    }
    active_dirs[active_count++] = dir;
  }
  if (active_count == 0) {
    verify_census(part, nullptr, send.count, recv.count, stream);
    return;
  }

  std::array<int, MAX_NEIGHBORS> send_counts{};
  std::array<int, MAX_NEIGHBORS> recv_counts{};
  std::array<MPI_Request, MAX_NEIGHBORS * 2> count_requests{};
  int request_count = 0;
  const MPI_Comm comm = migration_comm(part);

  for (int idx = 0; idx < active_count; ++idx) {
    const int dir = active_dirs[idx];
    send_counts[dir] = std::max(0, send.per_dest_count[dir]);
    const int recv_dir = opposite_direction(dir);
    const int recv_err =
        MPI_Irecv(&recv_counts[dir], 1, MPI_INT, part.neighbor_ranks[dir],
                  migration_tag(kMigrationPhaseId, recv_dir, kMigrationCountSlot),
                  comm, &count_requests[request_count++]);
    TENRYU_ASSERT(recv_err == MPI_SUCCESS, mpi_error_message("MPI_Irecv"));
  }

  for (int idx = 0; idx < active_count; ++idx) {
    const int dir = active_dirs[idx];
    const int send_err =
        MPI_Isend(&send_counts[dir], 1, MPI_INT, part.neighbor_ranks[dir],
                  migration_tag(kMigrationPhaseId, dir, kMigrationCountSlot),
                  comm, &count_requests[request_count++]);
    TENRYU_ASSERT(send_err == MPI_SUCCESS, mpi_error_message("MPI_Isend"));
  }

  const int wait_count_err =
      MPI_Waitall(request_count, count_requests.data(), MPI_STATUSES_IGNORE);
  TENRYU_ASSERT(wait_count_err == MPI_SUCCESS,
                mpi_error_message("MPI_Waitall"));

  int total_recv = 0;
  for (int idx = 0; idx < active_count; ++idx) {
    const int dir = active_dirs[idx];
    recv.per_dest_offset[dir] = total_recv;
    recv.per_dest_count[dir] = std::max(0, recv_counts[dir]);
    total_recv += recv.per_dest_count[dir];
  }

  recv.resize_if_needed(total_recv);
  recv.count = total_recv;

  std::vector<ParticleEmigrant> host_send;
  std::vector<ParticleEmigrant> host_recv;
  if (!use_gpu_aware_mpi) {
    if (send.count > 0) {
      TENRYU_ASSERT(send.data != nullptr,
                    "exchange_emigrants host-staging requires send.data");
      host_send.resize(static_cast<std::size_t>(send.count));
      CUDA_CHECK(cudaMemcpy(host_send.data(), send.data,
                            sizeof(ParticleEmigrant) * host_send.size(),
                            cudaMemcpyDeviceToHost));
    }
    if (total_recv > 0) {
      host_recv.resize(static_cast<std::size_t>(total_recv));
    }
  }

  std::array<MPI_Request, MAX_NEIGHBORS * 2> data_requests{};
  request_count = 0;

  for (int idx = 0; idx < active_count; ++idx) {
    const int dir = active_dirs[idx];
    const int recv_count = recv.per_dest_count[dir];
    if (recv_count <= 0) {
      continue;
    }
    const std::size_t recv_bytes =
        static_cast<std::size_t>(recv_count) * EmigrantBuffer::BYTES_PER_PARTICLE;
    TENRYU_ASSERT(
        recv_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "particle migration receive message exceeds MPI int byte-count limit");
    void* recv_ptr = use_gpu_aware_mpi
                         ? static_cast<void*>(recv.data + recv.per_dest_offset[dir])
                         : static_cast<void*>(host_recv.data() + recv.per_dest_offset[dir]);
    const int recv_dir = opposite_direction(dir);
    const int recv_err =
        MPI_Irecv(recv_ptr, static_cast<int>(recv_bytes), MPI_BYTE,
                  part.neighbor_ranks[dir],
                  migration_tag(kMigrationPhaseId, recv_dir, kMigrationPayloadSlot),
                  comm, &data_requests[request_count++]);
    TENRYU_ASSERT(recv_err == MPI_SUCCESS, mpi_error_message("MPI_Irecv"));
  }

  for (int idx = 0; idx < active_count; ++idx) {
    const int dir = active_dirs[idx];
    const int send_count = send_counts[dir];
    if (send_count <= 0) {
      continue;
    }
    TENRYU_ASSERT(send.per_dest_offset[dir] >= 0,
                  "exchange_emigrants requires non-negative send offset");
    TENRYU_ASSERT(send.per_dest_offset[dir] + send_count <= send.count,
                  "exchange_emigrants send slice exceeds send.count");

    const std::size_t send_bytes =
        static_cast<std::size_t>(send_count) * EmigrantBuffer::BYTES_PER_PARTICLE;
    TENRYU_ASSERT(
        send_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "particle migration send message exceeds MPI int byte-count limit");
    const void* send_ptr = use_gpu_aware_mpi
                               ? static_cast<const void*>(
                                     send.data + send.per_dest_offset[dir])
                               : static_cast<const void*>(
                                     host_send.data() + send.per_dest_offset[dir]);
    const int send_err =
        MPI_Isend(const_cast<void*>(send_ptr), static_cast<int>(send_bytes),
                  MPI_BYTE, part.neighbor_ranks[dir],
                  migration_tag(kMigrationPhaseId, dir, kMigrationPayloadSlot),
                  comm, &data_requests[request_count++]);
    TENRYU_ASSERT(send_err == MPI_SUCCESS, mpi_error_message("MPI_Isend"));
  }

  if (request_count > 0) {
    const int wait_data_err =
        MPI_Waitall(request_count, data_requests.data(), MPI_STATUSES_IGNORE);
    TENRYU_ASSERT(wait_data_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Waitall"));
  }

  if (!use_gpu_aware_mpi && total_recv > 0) {
    TENRYU_ASSERT(recv.data != nullptr,
                  "exchange_emigrants host-staging requires recv.data");
    CUDA_CHECK(cudaMemcpy(recv.data, host_recv.data(),
                          sizeof(ParticleEmigrant) * host_recv.size(),
                          cudaMemcpyHostToDevice));
  }
  verify_census(part, nullptr, send.count, recv.count, stream);
#else
  (void)buffers;
  (void)send;
  (void)stream;
#endif
}

void merge_immigrants(radiation::PhotonPool& pool, const EmigrantBuffer& recv,
                      const bool sort_by_global_id, cudaStream_t stream,
                      const std::int8_t* ddmc_mode_map,
                      const int ddmc_mode_map_size, const int ddmc_mode_stride) {
  (void)sort_by_global_id;
  if (recv.count <= 0) {
    return;
  }

  TENRYU_ASSERT(recv.data != nullptr, "merge_immigrants requires recv.data");
  TENRYU_ASSERT(pool.n_alive + recv.count <= pool.capacity,
                "merge_immigrants requires sufficient PhotonPool capacity");

  const int base = pool.n_alive;
  const int blocks = (recv.count + kThreadsPerBlock - 1) / kThreadsPerBlock;
  merge_immigrants_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      recv.data, recv.count, base, pool.pos_r, pool.pos_z, pool.dir_r, pool.dir_z,
      pool.dir_phi, pool.energy, pool.weight, pool.time_remain, pool.birth_energy,
      pool.global_id, pool.rng_counter, pool.cell_id, pool.group_id, pool.sign,
      pool.mode, pool.alive, ddmc_mode_map, ddmc_mode_map_size, ddmc_mode_stride);
  CUDA_CHECK(cudaGetLastError());

  pool.n_alive += recv.count;
}

void merge_immigrants(radiation::PhotonPool& pool, const EmigrantBuffer& recv,
                      const bool sort_by_global_id, cudaStream_t stream) {
  merge_immigrants(pool,
                   recv,
                   sort_by_global_id,
                   stream,
                   nullptr,
                   0,
                   1);
}

void ParticleMigration::migrate() {
}

}  // namespace tenryu::parallel
