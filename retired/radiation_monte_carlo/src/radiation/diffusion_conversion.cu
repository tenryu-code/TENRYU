#include "radiation/diffusion_conversion.cuh"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr int kBlock = 256;
constexpr double kPi = 3.141592653589793238462643383279502884;

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

__device__ inline bool is_entering_diffusion(const std::uint8_t* __restrict__ diff_cell,
                                             const std::uint8_t* __restrict__ diff_cell_prev,
                                             const int c) {
  return diff_cell[c] != 0U && diff_cell_prev[c] == 0U;
}

__device__ inline bool is_exiting_diffusion(const std::uint8_t* __restrict__ diff_cell,
                                            const std::uint8_t* __restrict__ diff_cell_prev,
                                            const int c) {
  return diff_cell[c] == 0U && diff_cell_prev[c] != 0U;
}

__device__ inline double sample_uniform_shell_radius_1d(const double xi_r,
                                                        const double r_lo,
                                                        const double r_hi) {
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  const double xi = fmin(fmax(xi_r, 0.0), 1.0);
  return cbrt(r_lo3 + xi * (r_hi3 - r_lo3));
}

__global__ void zero_entering_diffusion_energy_kernel(
    double* __restrict__ diff_E,
    const std::uint8_t* __restrict__ diff_cell,
    const std::uint8_t* __restrict__ diff_cell_prev,
    const int n_cells,
    const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }
  const int c = tid / n_groups;
  if (is_entering_diffusion(diff_cell, diff_cell_prev, c)) {
    diff_E[tid] = 0.0;
  }
}

__global__ void fold_entering_particles_kernel(
    double* __restrict__ entry_E,
    unsigned long long* __restrict__ folded_count,
    double* __restrict__ folded_energy,
    double* __restrict__ energy,
    double* __restrict__ weight,
    double* __restrict__ time_remain,
    double* __restrict__ birth_energy,
    std::int32_t* __restrict__ cell_id,
    std::uint16_t* __restrict__ group_id,
    std::uint8_t* __restrict__ alive,
    const std::uint8_t* __restrict__ diff_cell,
    const std::uint8_t* __restrict__ diff_cell_prev,
    const int n_particles,
    const int n_cells,
    const int n_groups) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_particles || alive[p] != static_cast<std::uint8_t>(kAlive)) {
    return;
  }

  const int c = cell_id[p];
  const int g = static_cast<int>(group_id[p]);
  if (c < 0 || c >= n_cells || g < 0 || g >= n_groups ||
      !is_entering_diffusion(diff_cell, diff_cell_prev, c)) {
    return;
  }

  const double E = energy[p];
  atomic_add_double(&entry_E[c * n_groups + g], E);
  atomic_add_double(folded_energy, E);
  atomicAdd(folded_count, 1ULL);

  alive[p] = static_cast<std::uint8_t>(kDead);
  energy[p] = 0.0;
  weight[p] = 0.0;
  time_remain[p] = 0.0;
  birth_energy[p] = 0.0;
}

__global__ void finalize_entering_diffusion_energy_kernel(
    double* __restrict__ diff_E,
    const double* __restrict__ entry_E,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ diff_cell,
    const std::uint8_t* __restrict__ diff_cell_prev,
    const int n_cells,
    const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }
  const int c = tid / n_groups;
  if (!is_entering_diffusion(diff_cell, diff_cell_prev, c)) {
    return;
  }
  const double V = vol[c];
  diff_E[tid] = (V > 0.0) ? (entry_E[tid] / V) : 0.0;
}

__global__ void fill_exit_phase_space_1d_kernel(
    const int start,
    const int n_new,
    const double* __restrict__ node_r,
    const std::int32_t* __restrict__ cell_id,
    double* __restrict__ time_remain,
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    std::int8_t* __restrict__ sign,
    const std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt,
    const int n_cells) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = start + t;
  const int c = cell_id[idx];
  if (c < 0 || c >= n_cells) {
    return;
  }

  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  const double xi_r = curand_uniform_double(&rng);
  const double xi_mu = curand_uniform_double(&rng);
  const double xi_phi = curand_uniform_double(&rng);

  pos_r[idx] = sample_uniform_shell_radius_1d(xi_r, node_r[c], node_r[c + 1]);
  pos_z[idx] = 0.0;

  const double mu = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  dir_r[idx] = mu;
  dir_z[idx] = sin_theta * cos(phi);
  dir_phi[idx] = sin_theta * sin(phi);
  time_remain[idx] = dt;
  sign[idx] = 1;

  rng_counter[idx] += 3U;
}

__global__ void zero_exiting_diffusion_energy_kernel(
    double* __restrict__ diff_E,
    const std::uint8_t* __restrict__ diff_cell,
    const std::uint8_t* __restrict__ diff_cell_prev,
    const int n_cells,
    const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_total = n_cells * n_groups;
  if (tid >= n_total) {
    return;
  }
  const int c = tid / n_groups;
  if (is_exiting_diffusion(diff_cell, diff_cell_prev, c)) {
    diff_E[tid] = 0.0;
  }
}

}  // namespace

DiffusionConversionStats fold_entering_diffusion_particles_cuda(
    PhotonPool& pool,
    const int n_particles,
    const std::uint8_t* diff_cell,
    const std::uint8_t* diff_cell_prev,
    const double* vol,
    double* diff_E,
    double* entry_E,
    const int n_cells,
    const int n_groups) {
  TENRYU_ASSERT(n_particles >= 0,
                "fold_entering_diffusion_particles_cuda requires n_particles >= 0");
  TENRYU_ASSERT(n_cells >= 0,
                "fold_entering_diffusion_particles_cuda requires n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0,
                "fold_entering_diffusion_particles_cuda requires n_groups > 0");
  TENRYU_ASSERT(diff_cell != nullptr,
                "fold_entering_diffusion_particles_cuda requires diff_cell");
  TENRYU_ASSERT(diff_cell_prev != nullptr,
                "fold_entering_diffusion_particles_cuda requires diff_cell_prev");
  TENRYU_ASSERT(vol != nullptr,
                "fold_entering_diffusion_particles_cuda requires vol");
  TENRYU_ASSERT(diff_E != nullptr,
                "fold_entering_diffusion_particles_cuda requires diff_E");
  TENRYU_ASSERT(entry_E != nullptr,
                "fold_entering_diffusion_particles_cuda requires entry_E");
  if (n_cells == 0) {
    return {};
  }

  const int n_total = n_cells * n_groups;
  cuda_check(cudaMemset(entry_E, 0, sizeof(double) * static_cast<std::size_t>(n_total)),
             "fold_entering_diffusion_particles_cuda zero entry_E failed");

  unsigned long long* d_count = nullptr;
  double* d_energy = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_count), sizeof(unsigned long long)),
             "fold_entering_diffusion_particles_cuda cudaMalloc count failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_energy), sizeof(double)),
             "fold_entering_diffusion_particles_cuda cudaMalloc energy failed");
  cuda_check(cudaMemset(d_count, 0, sizeof(unsigned long long)),
             "fold_entering_diffusion_particles_cuda zero count failed");
  cuda_check(cudaMemset(d_energy, 0, sizeof(double)),
             "fold_entering_diffusion_particles_cuda zero energy failed");

  const int grid_total = (n_total + kBlock - 1) / kBlock;
  zero_entering_diffusion_energy_kernel<<<grid_total, kBlock>>>(
      diff_E, diff_cell, diff_cell_prev, n_cells, n_groups);
  cuda_check(cudaGetLastError(),
             "fold_entering_diffusion_particles_cuda zero entering launch failed");

  if (n_particles > 0) {
    const int grid_particles = (n_particles + kBlock - 1) / kBlock;
    fold_entering_particles_kernel<<<grid_particles, kBlock>>>(
        entry_E,
        d_count,
        d_energy,
        pool.energy,
        pool.weight,
        pool.time_remain,
        pool.birth_energy,
        pool.cell_id,
        pool.group_id,
        pool.alive,
        diff_cell,
        diff_cell_prev,
        n_particles,
        n_cells,
        n_groups);
    cuda_check(cudaGetLastError(),
               "fold_entering_diffusion_particles_cuda fold launch failed");
  }

  finalize_entering_diffusion_energy_kernel<<<grid_total, kBlock>>>(
      diff_E, entry_E, vol, diff_cell, diff_cell_prev, n_cells, n_groups);
  cuda_check(cudaGetLastError(),
             "fold_entering_diffusion_particles_cuda finalize launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "fold_entering_diffusion_particles_cuda execution failed");

  unsigned long long host_count = 0ULL;
  double host_energy = 0.0;
  cuda_check(cudaMemcpy(&host_count,
                        d_count,
                        sizeof(host_count),
                        cudaMemcpyDeviceToHost),
             "fold_entering_diffusion_particles_cuda copy count failed");
  cuda_check(cudaMemcpy(&host_energy,
                        d_energy,
                        sizeof(host_energy),
                        cudaMemcpyDeviceToHost),
             "fold_entering_diffusion_particles_cuda copy energy failed");
  cuda_check(cudaFree(d_energy),
             "fold_entering_diffusion_particles_cuda free energy failed");
  cuda_check(cudaFree(d_count),
             "fold_entering_diffusion_particles_cuda free count failed");

  DiffusionConversionStats stats;
  stats.n_particles = static_cast<std::uint64_t>(host_count);
  stats.energy = host_energy;
  return stats;
}

void fill_diffusion_exit_phase_space_1d_cuda(PhotonPool& pool,
                                             const int start,
                                             const int n_new,
                                             const double* node_r,
                                             const int n_cells,
                                             const std::uint64_t user_seed,
                                             const std::uint64_t step_number,
                                             const double dt) {
  TENRYU_ASSERT(start >= 0, "fill_diffusion_exit_phase_space_1d_cuda requires start >= 0");
  TENRYU_ASSERT(n_new >= 0, "fill_diffusion_exit_phase_space_1d_cuda requires n_new >= 0");
  TENRYU_ASSERT(n_cells >= 0, "fill_diffusion_exit_phase_space_1d_cuda requires n_cells >= 0");
  TENRYU_ASSERT(node_r != nullptr,
                "fill_diffusion_exit_phase_space_1d_cuda requires node_r");
  if (n_new == 0) {
    return;
  }
  const int grid = (n_new + kBlock - 1) / kBlock;
  fill_exit_phase_space_1d_kernel<<<grid, kBlock>>>(start,
                                                    n_new,
                                                    node_r,
                                                    pool.cell_id,
                                                    pool.time_remain,
                                                    pool.pos_r,
                                                    pool.pos_z,
                                                    pool.dir_r,
                                                    pool.dir_z,
                                                    pool.dir_phi,
                                                    pool.sign,
                                                    pool.global_id,
                                                    pool.rng_counter,
                                                    user_seed,
                                                    step_number,
                                                    dt,
                                                    n_cells);
  cuda_check(cudaGetLastError(), "fill_diffusion_exit_phase_space_1d_cuda launch failed");
  cuda_check(cudaDeviceSynchronize(), "fill_diffusion_exit_phase_space_1d_cuda execution failed");
}

void zero_exiting_diffusion_energy_cuda(double* diff_E,
                                        const std::uint8_t* diff_cell,
                                        const std::uint8_t* diff_cell_prev,
                                        const int n_cells,
                                        const int n_groups) {
  TENRYU_ASSERT(diff_E != nullptr, "zero_exiting_diffusion_energy_cuda requires diff_E");
  TENRYU_ASSERT(diff_cell != nullptr,
                "zero_exiting_diffusion_energy_cuda requires diff_cell");
  TENRYU_ASSERT(diff_cell_prev != nullptr,
                "zero_exiting_diffusion_energy_cuda requires diff_cell_prev");
  TENRYU_ASSERT(n_cells >= 0, "zero_exiting_diffusion_energy_cuda requires n_cells >= 0");
  TENRYU_ASSERT(n_groups > 0, "zero_exiting_diffusion_energy_cuda requires n_groups > 0");
  const int n_total = n_cells * n_groups;
  if (n_total == 0) {
    return;
  }
  const int grid = (n_total + kBlock - 1) / kBlock;
  zero_exiting_diffusion_energy_kernel<<<grid, kBlock>>>(
      diff_E, diff_cell, diff_cell_prev, n_cells, n_groups);
  cuda_check(cudaGetLastError(), "zero_exiting_diffusion_energy_cuda launch failed");
  cuda_check(cudaDeviceSynchronize(), "zero_exiting_diffusion_energy_cuda execution failed");
}

}  // namespace tenryu::radiation
