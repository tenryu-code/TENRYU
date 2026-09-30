#include "radiation/source.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/device/device_scan.cuh>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "parallel/reduction.hpp"
#include "radiation/boundary.cuh"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaxRejectSamples = 64;
constexpr int kMaxRejectSamples2DMax = 128;
constexpr double kMaxTemperatureForT4 = 1.0e6;
constexpr std::uint64_t kStepLocalIdBits = 40ULL;
constexpr std::uint64_t kStepLocalIdMask = (1ULL << kStepLocalIdBits) - 1ULL;

__global__ void rebin_census_particles_1d_kernel(std::int32_t* __restrict__ cell_id,
                                                 const double* __restrict__ pos_r,
                                                 const double* __restrict__ node_r,
                                                 const int n_particles,
                                                 const int n_cells) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) {
    return;
  }

  const double r = pos_r[tid];
  if (!isfinite(r)) {
    return;
  }
  const int old_cell = cell_id[tid];
  if (old_cell >= 0 && old_cell < n_cells &&
      r >= node_r[old_cell] && r <= node_r[old_cell + 1]) {
    return;
  }

  int lo = 0;
  int hi = n_cells - 1;
  while (lo <= hi) {
    const int mid = lo + (hi - lo) / 2;
    if (r < node_r[mid]) {
      hi = mid - 1;
    } else if (r > node_r[mid + 1]) {
      lo = mid + 1;
    } else {
      cell_id[tid] = mid;
      return;
    }
  }

  if (r <= node_r[0]) {
    cell_id[tid] = 0;
  } else if (r >= node_r[n_cells]) {
    cell_id[tid] = n_cells - 1;
  }
}

__device__ inline double draw_uniform(curandStatePhilox4_32_10_t* rng,
                                      std::uint32_t* draws) {
  ++(*draws);
  return curand_uniform_double(rng);
}

int mpi_world_size() {
#if TENRYU_ENABLE_MPI
  int mpi_initialized = 0;
  MPI_Initialized(&mpi_initialized);
  if (!mpi_initialized) {
    return 1;
  }
  int n_ranks = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
  return std::max(n_ranks, 1);
#else
  return 1;
#endif
}

std::uint64_t compute_emission_gid_base(const std::uint64_t step_base_gid,
                                        const std::int64_t n_new_local,
                                        const int local_start,
                                        const parallel::Reduction* reduction) {
  TENRYU_ASSERT(n_new_local >= 0, "source gid base requires non-negative n_new_local");
  TENRYU_ASSERT(local_start >= 0, "source gid base requires non-negative local_start");

  if (reduction == nullptr) {
    const std::uint64_t local_start_u64 = static_cast<std::uint64_t>(local_start);
    TENRYU_ASSERT(
        step_base_gid <= std::numeric_limits<std::uint64_t>::max() - local_start_u64,
        "source gid base overflow (single-rank path)");
    return step_base_gid + local_start_u64;
  }

  const std::uint64_t step_prefix = step_base_gid & ~kStepLocalIdMask;
  const std::uint64_t local_base_offset = step_base_gid & kStepLocalIdMask;
  const double global_base_offset_d =
      reduction->allreduce_sum(static_cast<double>(local_base_offset));
  TENRYU_ASSERT(std::isfinite(global_base_offset_d),
                "source gid base allreduce produced non-finite offset");
  TENRYU_ASSERT(global_base_offset_d >= 0.0,
                "source gid base allreduce produced negative offset");
  const std::int64_t global_base_offset_i64 =
      static_cast<std::int64_t>(std::llround(global_base_offset_d));
  TENRYU_ASSERT(std::abs(global_base_offset_d -
                         static_cast<double>(global_base_offset_i64)) <= 0.25,
                "source gid base allreduce produced non-integer offset");
  TENRYU_ASSERT(global_base_offset_i64 >= 0,
                "source gid base allreduce offset underflow");
  const std::uint64_t global_base_offset =
      static_cast<std::uint64_t>(global_base_offset_i64);
  TENRYU_ASSERT(global_base_offset <= kStepLocalIdMask,
                "source gid base global offset exceeds 40-bit step space");
  TENRYU_ASSERT(step_prefix <=
                    std::numeric_limits<std::uint64_t>::max() - global_base_offset,
                "source gid base step prefix overflow");
  const std::uint64_t step_gid_base = step_prefix + global_base_offset;

  const std::int64_t rank_offset_i64 = reduction->exscan_sum(n_new_local);
  TENRYU_ASSERT(rank_offset_i64 >= 0, "source gid base exscan produced negative rank offset");
  const std::uint64_t rank_offset = static_cast<std::uint64_t>(rank_offset_i64);
  TENRYU_ASSERT(step_gid_base <= std::numeric_limits<std::uint64_t>::max() - rank_offset,
                "source gid base rank offset overflow");
  return step_gid_base + rank_offset;
}

std::size_t expected_node_count_2d(const int nr, const int nz) {
  TENRYU_ASSERT(nr >= 0 && nz >= 0, "expected_node_count_2d requires non-negative nr/nz");
  return static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
}

void upload_source_tilt_2d(const core::State& state,
                           const std::vector<double>& source_tilt,
                           double** d_source_tilt_r,
                           double** d_source_tilt_z,
                           double** d_cell_centroid_r,
                           double** d_cell_centroid_z) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_nodes = expected_node_count_2d(nr, nz);
  TENRYU_ASSERT(source_tilt.size() == 2U * static_cast<std::size_t>(n_cells),
                "upload_source_tilt_2d source_tilt size mismatch");
  TENRYU_ASSERT(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) ==
                    static_cast<std::size_t>(n_cells),
                "upload_source_tilt_2d requires nr*nz cells");
  TENRYU_ASSERT(state.x_r.size() == n_nodes,
                "upload_source_tilt_2d requires x_r node count to match topology");
  TENRYU_ASSERT(state.x_z.size() == n_nodes,
                "upload_source_tilt_2d requires x_z node count to match topology");

  std::vector<double> host_tilt_r(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_tilt_z(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_centroid_r(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_centroid_z(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> host_node_r(n_nodes, 0.0);
  std::vector<double> host_node_z(n_nodes, 0.0);
  state.x_r.copy_to_host(host_node_r.data());
  state.x_z.copy_to_host(host_node_z.data());

  const int stride = nz + 1;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const std::size_t c_us = static_cast<std::size_t>(c);
      host_tilt_r[c_us] = source_tilt[2U * c_us];
      host_tilt_z[c_us] = source_tilt[2U * c_us + 1U];

      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n11 = (i + 1) * stride + (j + 1);
      const int n01 = i * stride + (j + 1);
      host_centroid_r[c_us] =
          0.25 * (host_node_r[static_cast<std::size_t>(n00)] +
                  host_node_r[static_cast<std::size_t>(n10)] +
                  host_node_r[static_cast<std::size_t>(n11)] +
                  host_node_r[static_cast<std::size_t>(n01)]);
      host_centroid_z[c_us] =
          0.25 * (host_node_z[static_cast<std::size_t>(n00)] +
                  host_node_z[static_cast<std::size_t>(n10)] +
                  host_node_z[static_cast<std::size_t>(n11)] +
                  host_node_z[static_cast<std::size_t>(n01)]);
    }
  }

  const std::size_t bytes_cells = sizeof(double) * static_cast<std::size_t>(n_cells);
  *d_source_tilt_r = static_cast<double*>(
      core::device_scratch_acquire("source:source_tilt_r_2d", bytes_cells));
  *d_source_tilt_z = static_cast<double*>(
      core::device_scratch_acquire("source:source_tilt_z_2d", bytes_cells));
  *d_cell_centroid_r = static_cast<double*>(
      core::device_scratch_acquire("source:cell_centroid_r_2d", bytes_cells));
  *d_cell_centroid_z = static_cast<double*>(
      core::device_scratch_acquire("source:cell_centroid_z_2d", bytes_cells));
  cuda_check(cudaMemcpy(*d_source_tilt_r,
                        host_tilt_r.data(),
                        bytes_cells,
                        cudaMemcpyHostToDevice),
             "upload_source_tilt_2d copy source_tilt_r failed");
  cuda_check(cudaMemcpy(*d_source_tilt_z,
                        host_tilt_z.data(),
                        bytes_cells,
                        cudaMemcpyHostToDevice),
             "upload_source_tilt_2d copy source_tilt_z failed");
  cuda_check(cudaMemcpy(*d_cell_centroid_r,
                        host_centroid_r.data(),
                        bytes_cells,
                        cudaMemcpyHostToDevice),
             "upload_source_tilt_2d copy cell_centroid_r failed");
  cuda_check(cudaMemcpy(*d_cell_centroid_z,
                        host_centroid_z.data(),
                        bytes_cells,
                        cudaMemcpyHostToDevice),
             "upload_source_tilt_2d copy cell_centroid_z failed");
}

void warn_if_rng_counter_near_wrap_range(const std::uint32_t* d_rng_counter,
                                         const int offset,
                                         const int n_count,
                                         const std::uint32_t draws_required,
                                         const char* stage_name) {
  if (d_rng_counter == nullptr || n_count <= 0) {
    return;
  }

  std::vector<std::uint32_t> host_counter(static_cast<std::size_t>(n_count), 0U);
  cuda_check(cudaMemcpy(host_counter.data(),
                        d_rng_counter + offset,
                        sizeof(std::uint32_t) * host_counter.size(),
                        cudaMemcpyDeviceToHost),
             "source rng_counter overflow check copy failed");

  std::uint32_t max_counter = 0U;
  for (const std::uint32_t c : host_counter) {
    max_counter = std::max(max_counter, c);
  }

  constexpr std::uint32_t kMaxCounter = std::numeric_limits<std::uint32_t>::max();
  const std::uint32_t threshold =
      (draws_required < kMaxCounter) ? (kMaxCounter - draws_required) : 0U;
  if (max_counter >= threshold) {
    static bool warned_rng_wrap = false;
    if (!warned_rng_wrap) {
      core::log_warning(std::string(stage_name) +
                        ": rng_counter is near uint32 wrap; RNG stream may restart");
      warned_rng_wrap = true;
    }
  }
}

__host__ __device__ inline double clamped_temperature_for_t4(const double temperature_eV) {
  if (!isfinite(temperature_eV) || temperature_eV <= 0.0) {
    return 0.0;
  }
  return fmin(temperature_eV, kMaxTemperatureForT4);
}

__host__ __device__ inline double safe_temperature_pow4(const double temperature_eV) {
  const double t = clamped_temperature_for_t4(temperature_eV);
  return t * t * t * t;
}

__device__ inline void bilinear_map(const double eta,
                                    const double zeta,
                                    const double r00,
                                    const double z00,
                                    const double r10,
                                    const double z10,
                                    const double r11,
                                    const double z11,
                                    const double r01,
                                    const double z01,
                                    double* r_out,
                                    double* z_out) {
  const double n00 = (1.0 - eta) * (1.0 - zeta);
  const double n10 = eta * (1.0 - zeta);
  const double n11 = eta * zeta;
  const double n01 = (1.0 - eta) * zeta;
  *r_out = n00 * r00 + n10 * r10 + n11 * r11 + n01 * r01;
  *z_out = n00 * z00 + n10 * z10 + n11 * z11 + n01 * z01;
}

__device__ inline int radial_cell_from_nodes(const double r_sample,
                                             const double* __restrict__ node_r,
                                             const int nr,
                                             const int nz) {
  // Limitation: this bins by the j=0 node row only, so strongly warped/non-rectilinear
  // meshes can produce imperfect radial binning for Marshak face sampling.
  const int stride = nz + 1;
  int lo = 0;
  int hi = nr;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) / 2;
    const double r_mid = node_r[mid * stride];
    if (r_sample < r_mid) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  if (lo < 0) {
    return 0;
  }
  if (lo >= nr) {
    return nr - 1;
  }
  return lo;
}

__device__ inline double eval_tilted_shell_cdf_1d(const double u,
                                                   const double r_lo,
                                                   const double dr,
                                                   const double tilt) {
  const double a = 1.0 - tilt;
  const double b = 2.0 * tilt;
  const double r_lo2 = r_lo * r_lo;
  const double dr2 = dr * dr;
  const double c0 = r_lo2 * a;
  const double c1 = r_lo2 * b + 2.0 * r_lo * dr * a;
  const double c2 = 2.0 * r_lo * dr * b + dr2 * a;
  const double c3 = dr2 * b;
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  return c0 * u + 0.5 * c1 * u2 + (c2 * u3) / 3.0 + 0.25 * c3 * u4;
}

__device__ inline double sample_uniform_shell_radius_1d(const double xi_r,
                                                        const double r_lo,
                                                        const double r_hi) {
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  const double xi = fmin(fmax(xi_r, 0.0), 1.0);
  return cbrt(r_lo3 + xi * (r_hi3 - r_lo3));
}

__device__ inline double sample_thermal_radius_1d(const double xi_r,
                                                  const double r_lo,
                                                  const double r_hi,
                                                  const double tilt) {
  if (!(r_hi > r_lo)) {
    return r_lo;
  }

  const double xi = fmin(fmax(xi_r, 0.0), 1.0);
  const double tilt_clamped = fmin(fmax(tilt, -1.0), 1.0);
  if (fabs(tilt_clamped) <= 1.0e-12) {
    return sample_uniform_shell_radius_1d(xi, r_lo, r_hi);
  }

  const double dr = r_hi - r_lo;
  const double total = eval_tilted_shell_cdf_1d(1.0, r_lo, dr, tilt_clamped);
  if (!(total > 0.0) || !isfinite(total)) {
    return sample_uniform_shell_radius_1d(xi, r_lo, r_hi);
  }

  const double target = xi * total;
  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < 24; ++iter) {
    const double mid = 0.5 * (lo + hi);
    if (eval_tilted_shell_cdf_1d(mid, r_lo, dr, tilt_clamped) < target) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return r_lo + 0.5 * (lo + hi) * dr;
}

__device__ inline double sample_localized_radius_1d(const double xi_r,
                                                    const double xi_mix,
                                                    const double xi_n1,
                                                    const double xi_n2,
                                                    const double r_lo,
                                                    const double r_hi,
                                                    const double mu,
                                                    const double sigma,
                                                    const double alpha,
                                                    const double tilt,
                                                    const bool use_tilt_fallback) {
  const double alpha_clamped = fmin(fmax(alpha, 0.0), 1.0);
  if (!(xi_mix < alpha_clamped)) {
    return use_tilt_fallback ? sample_thermal_radius_1d(xi_r, r_lo, r_hi, tilt)
                             : sample_uniform_shell_radius_1d(xi_r, r_lo, r_hi);
  }

  const double sigma_default = 0.25 * fmax(r_hi - r_lo, 0.0);
  const double sigma_eff = (sigma > 0.0) ? sigma : sigma_default;
  const double z =
      sqrt(-2.0 * log(fmax(xi_n1, 1.0e-30))) * cos(2.0 * kPi * xi_n2);
  return fmin(fmax(mu + sigma_eff * z, r_lo), r_hi);
}

__global__ void fill_thermal_phase_space_kernel(const int offset,
                                                const int n_new,
                                                const double* __restrict__ node_r,
                                                const double* __restrict__ source_tilt,
                                                const double* __restrict__ sloc_mean_r,
                                                const double* __restrict__ sloc_sigma,
                                                const double* __restrict__ sloc_alpha,
                                                const double* __restrict__ sloc_prev_E,
                                                const std::int32_t* __restrict__ cell_id,
                                                double* __restrict__ time_remain,
                                                double* __restrict__ pos_r,
                                                double* __restrict__ pos_z,
                                                double* __restrict__ dir_r,
                                                double* __restrict__ dir_z,
                                                double* __restrict__ dir_phi,
                                                std::uint64_t* __restrict__ global_id,
                                                std::uint32_t* __restrict__ rng_counter,
                                                const std::uint64_t user_seed,
                                                const std::uint64_t step_number,
                                                const double dt,
                                                const int n_cells) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = offset + t;
  const int c = cell_id[idx];
  if (c < 0 || c >= n_cells) {
    return;
  }

  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  std::uint32_t draws = 0U;
  const double xi_r = draw_uniform(&rng, &draws);
  const double xi_mu = draw_uniform(&rng, &draws);
  const double xi_phi = draw_uniform(&rng, &draws);
  const double xi_t = draw_uniform(&rng, &draws);

  const double r_lo = node_r[c];
  const double r_hi = node_r[c + 1];
  const bool use_tilt_fallback = (source_tilt != nullptr);
  if (sloc_mean_r != nullptr && sloc_prev_E != nullptr && sloc_prev_E[c] > 0.0) {
    const double xi_mix = draw_uniform(&rng, &draws);
    const double xi_n1 = draw_uniform(&rng, &draws);
    const double xi_n2 = draw_uniform(&rng, &draws);
    const double sigma = (sloc_sigma != nullptr) ? sloc_sigma[c] : 0.25 * (r_hi - r_lo);
    const double alpha = (sloc_alpha != nullptr) ? sloc_alpha[c] : 1.0;
    const double tilt = use_tilt_fallback ? source_tilt[c] : 0.0;
    pos_r[idx] = sample_localized_radius_1d(xi_r,
                                            xi_mix,
                                            xi_n1,
                                            xi_n2,
                                            r_lo,
                                            r_hi,
                                            sloc_mean_r[c],
                                            sigma,
                                            alpha,
                                            tilt,
                                            use_tilt_fallback);
  } else if (use_tilt_fallback) {
    pos_r[idx] = sample_thermal_radius_1d(xi_r, r_lo, r_hi, source_tilt[c]);
  } else {
    pos_r[idx] = sample_uniform_shell_radius_1d(xi_r, r_lo, r_hi);
  }
  pos_z[idx] = 0.0;

  const double mu = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  dir_r[idx] = mu;
  dir_z[idx] = sin_theta * cos(phi);
  dir_phi[idx] = sin_theta * sin(phi);
  time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);

  rng_counter[idx] += draws;
}

__global__ void fill_thermal_phase_space_tilted_kernel(
    const int offset,
    const int n_new,
    const double* __restrict__ node_r,
    const double* __restrict__ source_tilt,
    const double* __restrict__ sloc_mean_r,
    const double* __restrict__ sloc_prev_E,
    const std::int32_t* __restrict__ cell_id,
    double* __restrict__ time_remain,
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt,
    const int n_cells) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = offset + t;
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
  const double xi_t = curand_uniform_double(&rng);

  const double r_lo = node_r[c];
  const double r_hi = node_r[c + 1];
  if (sloc_mean_r != nullptr && sloc_prev_E != nullptr && sloc_prev_E[c] > 0.0) {
    pos_r[idx] = fmin(fmax(sloc_mean_r[c], r_lo), r_hi);
  } else {
    const double tilt = source_tilt[c];
    pos_r[idx] = sample_thermal_radius_1d(xi_r, r_lo, r_hi, tilt);
  }
  pos_z[idx] = 0.0;

  const double mu = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  dir_r[idx] = mu;
  dir_z[idx] = sin_theta * cos(phi);
  dir_phi[idx] = sin_theta * sin(phi);
  time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);

  rng_counter[idx] += 4U;
}

__global__ void fill_thermal_phase_space_2d_kernel(
    const int offset,
    const int n_new,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const std::int32_t* __restrict__ cell_id,
    double* __restrict__ time_remain,
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double* __restrict__ source_tilt_r,
    const double* __restrict__ source_tilt_z,
    const double* __restrict__ cell_centroid_r,
    const double* __restrict__ cell_centroid_z,
    const double dt,
    const int n_cells,
    const int nr,
    const int nz) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = offset + t;
  const int c = cell_id[idx];
  if (c < 0 || c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const double r00 = node_r[n00];
  const double r10 = node_r[n10];
  const double r11 = node_r[n11];
  const double r01 = node_r[n01];
  const double z00 = node_z[n00];
  const double z10 = node_z[n10];
  const double z11 = node_z[n11];
  const double z01 = node_z[n01];
  const double r_max_cell = fmax(fmax(r00, r10), fmax(r11, r01));

  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  std::uint32_t draws = 0U;
  double r_p = 0.25 * (r00 + r10 + r11 + r01);
  double z_p = 0.25 * (z00 + z10 + z11 + z01);
  bool accepted = false;
  const bool use_tilt = source_tilt_r != nullptr && source_tilt_z != nullptr;
  const int max_reject = use_tilt ? kMaxRejectSamples2DMax : kMaxRejectSamples;
  for (int n_try = 0; n_try < kMaxRejectSamples2DMax; ++n_try) {
    if (n_try >= max_reject) {
      break;
    }
    const double xi_eta = curand_uniform_double(&rng);
    const double xi_zeta = curand_uniform_double(&rng);
    const double xi_reject = curand_uniform_double(&rng);
    draws += 3U;

    bilinear_map(xi_eta,
                 xi_zeta,
                 r00,
                 z00,
                 r10,
                 z10,
                 r11,
                 z11,
                 r01,
                 z01,
                 &r_p,
                 &z_p);
    double w =
        (r_max_cell > 0.0) ? fmin(1.0, fmax(0.0, r_p / r_max_cell)) : 1.0;
    if (use_tilt) {
      const double r_c = cell_centroid_r[c];
      const double z_c = cell_centroid_z[c];
      const double ell_r = fmax(fmax(fabs(r00 - r_c), fabs(r10 - r_c)),
                                fmax(fabs(r11 - r_c), fabs(r01 - r_c)));
      const double ell_z = fmax(fmax(fabs(z00 - z_c), fabs(z10 - z_c)),
                                fmax(fabs(z11 - z_c), fabs(z01 - z_c)));
      const double q_r =
          (ell_r > 0.0) ? fmin(1.0, fmax(-1.0, (r_p - r_c) / ell_r)) : 0.0;
      const double q_z =
          (ell_z > 0.0) ? fmin(1.0, fmax(-1.0, (z_p - z_c) / ell_z)) : 0.0;
      const double t_r = source_tilt_r[c];
      const double t_z = source_tilt_z[c];
      const double b = 1.0 + t_r * q_r + t_z * q_z;
      const double b_max = 1.0 + fabs(t_r) + fabs(t_z);
      w *= fmax(b, 0.0) / fmax(b_max, 1.0);
    }
    if (xi_reject <= w) {
      accepted = true;
      break;
    }
  }
  if (!accepted) {
    r_p = fmax(r_p, 0.0);
  }
  pos_r[idx] = fmax(r_p, 0.0);
  pos_z[idx] = z_p;

  const double xi_mu = curand_uniform_double(&rng);
  const double xi_phi = curand_uniform_double(&rng);
  const double xi_t = curand_uniform_double(&rng);
  draws += 3U;

  const double mu_z = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_z * mu_z));
  dir_r[idx] = sin_theta * cos(phi);
  dir_z[idx] = mu_z;
  dir_phi[idx] = sin_theta * sin(phi);
  time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);

  rng_counter[idx] += draws;
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

__device__ inline double cdf_row_probability(const double* __restrict__ row_cdf,
                                             const int group) {
  if (row_cdf == nullptr || group < 0) {
    return 0.0;
  }
  const double cdf_hi = fmin(fmax(row_cdf[group], 0.0), 1.0);
  const double cdf_lo = (group > 0) ? fmin(fmax(row_cdf[group - 1], 0.0), 1.0) : 0.0;
  return fmax(cdf_hi - cdf_lo, 0.0);
}

__global__ void compute_source_energy_kernel(
    const double* __restrict__ Te,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ reference_E,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ nlte_eta,
    const double* __restrict__ nlte_f,
    double* __restrict__ source_E,
    double* __restrict__ emit_E,
    double* __restrict__ source_cell_total,
    double* __restrict__ source_total,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double Te_floor,
    const bool linearized,
    const double cv_override,
    const bool nlte_mode,
    const PlanckTableDeviceView planck) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_bins = n_cells * n_groups;
  if (idx >= n_bins) {
    return;
  }

  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    source_E[idx] = 0.0;
    emit_E[idx] = 0.0;
    return;
  }

  const double T =
      clamped_temperature_for_t4(fmax(Te[c], Te_floor));
  const double V = fmax(vol[c], 0.0);
  double S = 0.0;
  if (nlte_mode) {
    const double f = fmax(nlte_f[c], 0.0);
    const double eta = fmax(nlte_eta[idx], 0.0);
    S = f * eta;
  } else {
    const double b_g = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
    if (linearized && cv_override > 0.0) {
      S = tenryu::core::constants::c_light * sigma_a_eff[idx] * cv_override * T * b_g;
    } else {
      const double T4 = safe_temperature_pow4(T);
      S = tenryu::core::constants::c_light * sigma_a_eff[idx] *
          tenryu::core::constants::a_eV * T4 * b_g;
    }
  }

  const double E_emit = fmax(S * V * dt, 0.0);
  double E_source = E_emit;
  if (reference_E != nullptr) {
    const double E_ref = reference_E[idx];
    const double E_ref_abs = tenryu::core::constants::c_light *
                             fmax(sigma_a_eff[idx], 0.0) * E_ref * V * dt;
    E_source = E_emit - E_ref_abs;
  }
  if (!isfinite(E_source)) {
    E_source = 0.0;
  }
  emit_E[idx] = E_emit;
  source_E[idx] = E_source;
  const double E_source_abs = fabs(E_source);
  if (E_source_abs > 0.0) {
    if (source_cell_total != nullptr) {
      atomic_add_double(source_cell_total + c, E_source_abs);
    }
    atomic_add_double(source_total, E_source_abs);
  }
}

__global__ void source_particle_count_kernel(const double* __restrict__ source_E,
                                             const double* __restrict__ source_cell_total,
                                             const double* __restrict__ emission_bias_cdf,
                                             std::int32_t* __restrict__ count,
                                             double* __restrict__ source_lost,
                                             const int n_bins,
                                             const int n_groups,
                                             const double source_sum,
                                             const int N_tot) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_bins) {
    return;
  }

  const double E_bin = fabs(source_E[idx]);
  int n_p = 0;
  if (E_bin > 0.0 && source_sum > 0.0) {
    double ratio = E_bin / source_sum;
    if (source_cell_total != nullptr && emission_bias_cdf != nullptr && n_groups > 0) {
      const int c = idx / n_groups;
      const int g = idx - c * n_groups;
      const int cell_base = c * n_groups;
      const double cell_total = fmax(source_cell_total[c], 0.0);
      if (cell_total > 0.0) {
        const double* const row_cdf = emission_bias_cdf + cell_base;
        double q_sum = 0.0;
        for (int gg = 0; gg < n_groups; ++gg) {
          if (fabs(source_E[cell_base + gg]) > 0.0) {
            q_sum += cdf_row_probability(row_cdf, gg);
          }
        }
        const double q_g = cdf_row_probability(row_cdf, g);
        if (q_g > 0.0 && q_sum > 0.0) {
          ratio = (cell_total / source_sum) * (q_g / q_sum);
        }
      }
    }
    n_p = static_cast<int>(floor(static_cast<double>(N_tot) * ratio + 0.5));
    if (ratio > 0.0 && n_p <= 0) {
      n_p = 1;
    }
  }

  count[idx] = static_cast<std::int32_t>(n_p);
  if (E_bin > 0.0 && n_p <= 0) {
    atomic_add_double(source_lost, E_bin);
  }
}

__global__ void finalize_scan_tail_kernel(const int n_bins,
                                          const std::int32_t* __restrict__ count,
                                          std::int32_t* __restrict__ offset) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (n_bins <= 0) {
    offset[0] = 0;
    return;
  }
  offset[n_bins] = offset[n_bins - 1] + count[n_bins - 1];
}

__global__ void fill_emission_metadata_kernel(const int start,
                                              const int n_bins,
                                              const int n_groups,
                                              const std::int32_t* __restrict__ count,
                                              const std::int32_t* __restrict__ offset,
                                              const double* __restrict__ source_E,
                                              const std::uint64_t gid_base,
                                              const double dt,
                                              double* __restrict__ energy,
                                              double* __restrict__ weight,
                                              double* __restrict__ time_remain,
                                              double* __restrict__ birth_energy,
                                              std::int8_t* __restrict__ sign,
                                              std::uint64_t* __restrict__ global_id,
                                              std::uint32_t* __restrict__ rng_counter,
                                              std::int32_t* __restrict__ cell_id,
                                              std::uint16_t* __restrict__ group_id,
                                              std::uint8_t* __restrict__ mode,
                                              std::uint8_t* __restrict__ alive) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_bins) {
    return;
  }

  const int n_p = static_cast<int>(count[idx]);
  if (n_p <= 0) {
    return;
  }

  const int begin = static_cast<int>(offset[idx]);
  const double E_bin = source_E[idx];
  const double E_p = fabs(E_bin) / static_cast<double>(n_p);
  const int c = idx / n_groups;
  const std::uint16_t g = static_cast<std::uint16_t>(idx - c * n_groups);

  for (int k = 0; k < n_p; ++k) {
    const int local_index = begin + k;
    const int p = start + local_index;
    energy[p] = E_p;
    weight[p] = 1.0;
    time_remain[p] = dt;
    birth_energy[p] = E_p;
    sign[p] = (E_bin < 0.0) ? -1 : 1;
    global_id[p] = gid_base + static_cast<std::uint64_t>(local_index);
    rng_counter[p] = 0U;
    cell_id[p] = c;
    group_id[p] = g;
    mode[p] = kModeIMC;
    alive[p] = kAlive;
  }
}

__global__ void fill_marshak_phase_space_kernel(const int offset,
                                                const int n_new,
                                                const double r_boundary,
                                                double* __restrict__ time_remain,
                                                double* __restrict__ pos_r,
                                                double* __restrict__ pos_z,
                                                double* __restrict__ dir_r,
                                                double* __restrict__ dir_z,
                                                double* __restrict__ dir_phi,
                                                std::uint16_t* __restrict__ group_id,
                                                std::uint64_t* __restrict__ global_id,
                                                std::uint32_t* __restrict__ rng_counter,
                                                const std::uint64_t user_seed,
                                                const std::uint64_t step_number,
                                                const double dt,
                                                const int n_groups,
                                                const double T_r_eV,
                                                const PlanckTableDeviceView planck) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = offset + t;
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  const double xi_mu = curand_uniform_double(&rng);
  const double xi_phi = curand_uniform_double(&rng);
  const double xi_g = curand_uniform_double(&rng);
  const double xi_t = curand_uniform_double(&rng);

  pos_r[idx] = r_boundary;
  pos_z[idx] = 0.0;

  const double mu_inward = -sqrt(xi_mu);  // cosine-weighted inward hemisphere
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_inward * mu_inward));
  dir_r[idx] = mu_inward;
  dir_z[idx] = sin_theta * cos(phi);
  dir_phi[idx] = sin_theta * sin(phi);

  if (group_id != nullptr) {
    const int g = (n_groups > 1) ? planck.sample_group(xi_g, T_r_eV) : 0;
    group_id[idx] = static_cast<std::uint16_t>(g);
  }
  time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);

  rng_counter[idx] += 4U;
}

__global__ void fill_marshak_phase_space_2d_kernel(
    const int offset,
    const int n_new,
    const int cell_j,
    const double mu_sign,
    const double* __restrict__ face_seg_cdf,
    const double* __restrict__ face_seg_r0,
    const double* __restrict__ face_seg_z0,
    const double* __restrict__ face_seg_r1,
    const double* __restrict__ face_seg_z1,
    const int n_face_segments,
    const double* __restrict__ node_r,
    std::int32_t* __restrict__ cell_id,
    double* __restrict__ time_remain,
    double* __restrict__ pos_r,
    double* __restrict__ pos_z,
    double* __restrict__ dir_r,
    double* __restrict__ dir_z,
    double* __restrict__ dir_phi,
    std::uint16_t* __restrict__ group_id,
    std::uint64_t* __restrict__ global_id,
    std::uint32_t* __restrict__ rng_counter,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt,
    const int n_groups,
    const double T_r_eV,
    const PlanckTableDeviceView planck,
    const int nr,
    const int nz) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_new) {
    return;
  }

  const int idx = offset + t;
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter[idx]),
              &rng);

  if (n_face_segments <= 0 || face_seg_cdf == nullptr || face_seg_r0 == nullptr ||
      face_seg_z0 == nullptr || face_seg_r1 == nullptr || face_seg_z1 == nullptr) {
    return;
  }

  const double xi_face = curand_uniform_double(&rng);
  const double xi_u = curand_uniform_double(&rng);
  const double xi_mu = curand_uniform_double(&rng);
  const double xi_phi = curand_uniform_double(&rng);
  const double xi_g = curand_uniform_double(&rng);
  const double xi_t = curand_uniform_double(&rng);

  int seg = 0;
  if (n_face_segments > 1) {
    int lo = 0;
    int hi = n_face_segments - 1;
    while (lo < hi) {
      const int mid = lo + (hi - lo) / 2;
      const double cdf_mid = face_seg_cdf[mid];
      if (xi_face <= cdf_mid) {
        hi = mid;
      } else {
        lo = mid + 1;
      }
    }
    seg = lo;
  }

  const double r0 = face_seg_r0[seg];
  const double z0 = face_seg_z0[seg];
  const double r1 = face_seg_r1[seg];
  const double z1 = face_seg_z1[seg];
  const double r_emit = fmax(r0 + xi_u * (r1 - r0), 0.0);
  const double z_emit = z0 + xi_u * (z1 - z0);
  pos_r[idx] = r_emit;
  pos_z[idx] = z_emit;

  const double mu_abs = sqrt(fmax(xi_mu, 0.0));
  const double mu_z = mu_sign * mu_abs;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_z * mu_z));
  dir_r[idx] = sin_theta * cos(phi);
  dir_z[idx] = mu_z;
  dir_phi[idx] = sin_theta * sin(phi);

  if (group_id != nullptr) {
    const int g = (n_groups > 1) ? planck.sample_group(xi_g, T_r_eV) : 0;
    group_id[idx] = static_cast<std::uint16_t>(g);
  }

  const int i = radial_cell_from_nodes(r_emit, node_r, nr, nz);
  const int j = (cell_j < 0) ? 0 : ((cell_j >= nz) ? (nz - 1) : cell_j);
  cell_id[idx] = i * nz + j;
  time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);

  rng_counter[idx] += 6U;
}

void append_particles(PhotonPool& pool,
                      const int start,
                      const std::vector<std::int32_t>& cell_id,
                      const std::vector<std::uint16_t>& group_id,
                      const std::vector<double>& energy,
                      const double dt,
                      const std::uint64_t gid_base,
                      const std::vector<std::int8_t>* sign_override = nullptr) {
  const int n_new = static_cast<int>(energy.size());
  TENRYU_ASSERT(static_cast<int>(cell_id.size()) == n_new, "append_particles cell size mismatch");
  TENRYU_ASSERT(static_cast<int>(group_id.size()) == n_new,
                "append_particles group size mismatch");
  TENRYU_ASSERT(sign_override == nullptr ||
                    static_cast<int>(sign_override->size()) == n_new,
                "append_particles sign size mismatch");

  // weight=1.0: energy accounting uses pool.energy directly.
  // Physical emission weight is embedded in pool.energy = E_group / n_particles.
  std::vector<double> weight(static_cast<std::size_t>(n_new), 1.0);
  std::vector<double> time_remain(static_cast<std::size_t>(n_new), dt);
  std::vector<double> birth_energy = energy;
  std::vector<std::int8_t> sign =
      (sign_override != nullptr)
          ? *sign_override
          : std::vector<std::int8_t>(static_cast<std::size_t>(n_new), 1);
  std::vector<std::uint64_t> global_id(static_cast<std::size_t>(n_new), 0);
  std::vector<std::uint32_t> rng_counter(static_cast<std::size_t>(n_new), 0U);
  std::vector<std::uint8_t> mode(static_cast<std::size_t>(n_new), kModeIMC);
  std::vector<std::uint8_t> alive(static_cast<std::size_t>(n_new), kAlive);

  for (int i = 0; i < n_new; ++i) {
    global_id[static_cast<std::size_t>(i)] = gid_base + static_cast<std::uint64_t>(i);
  }

  const std::size_t bytes_d = sizeof(double) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i32 = sizeof(std::int32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u16 = sizeof(std::uint16_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u64 = sizeof(std::uint64_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u32 = sizeof(std::uint32_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_i8 = sizeof(std::int8_t) * static_cast<std::size_t>(n_new);
  const std::size_t bytes_u8 = sizeof(std::uint8_t) * static_cast<std::size_t>(n_new);

  cuda_check(cudaMemcpy(pool.energy + start,
                        energy.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_particles copy energy failed");
  cuda_check(cudaMemcpy(pool.weight + start,
                        weight.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_particles copy weight failed");
  cuda_check(cudaMemcpy(pool.time_remain + start,
                        time_remain.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_particles copy time_remain failed");
  cuda_check(cudaMemcpy(pool.birth_energy + start,
                        birth_energy.data(),
                        bytes_d,
                        cudaMemcpyHostToDevice),
             "append_particles copy birth_energy failed");
  cuda_check(cudaMemcpy(pool.sign + start,
                        sign.data(),
                        bytes_i8,
                        cudaMemcpyHostToDevice),
             "append_particles copy sign failed");
  cuda_check(cudaMemcpy(pool.cell_id + start,
                        cell_id.data(),
                        bytes_i32,
                        cudaMemcpyHostToDevice),
             "append_particles copy cell_id failed");
  cuda_check(cudaMemcpy(pool.group_id + start,
                        group_id.data(),
                        bytes_u16,
                        cudaMemcpyHostToDevice),
             "append_particles copy group_id failed");
  cuda_check(cudaMemcpy(pool.global_id + start,
                        global_id.data(),
                        bytes_u64,
                        cudaMemcpyHostToDevice),
             "append_particles copy global_id failed");
  cuda_check(cudaMemcpy(pool.rng_counter + start,
                        rng_counter.data(),
                        bytes_u32,
                        cudaMemcpyHostToDevice),
             "append_particles copy rng_counter failed");
  cuda_check(cudaMemcpy(pool.mode + start,
                        mode.data(),
                        bytes_u8,
                        cudaMemcpyHostToDevice),
             "append_particles copy mode failed");
  cuda_check(cudaMemcpy(pool.alive + start,
                        alive.data(),
                        bytes_u8,
                        cudaMemcpyHostToDevice),
             "append_particles copy alive failed");
}

}  // namespace

void rebin_census_particles_1d_cuda(std::int32_t* cell_id,
                                    const double* pos_r,
                                    const double* node_r,
                                    const int n_particles,
                                    const int n_cells) {
  if (n_particles <= 0) {
    return;
  }
  TENRYU_ASSERT(cell_id != nullptr, "census rebin requires cell_id");
  TENRYU_ASSERT(pos_r != nullptr, "census rebin requires pos_r");
  TENRYU_ASSERT(node_r != nullptr, "census rebin requires node_r");
  TENRYU_ASSERT(n_cells > 0, "census rebin requires positive n_cells");

  constexpr int kBlockSize = 256;
  const int n_blocks = (n_particles + kBlockSize - 1) / kBlockSize;
  rebin_census_particles_1d_kernel<<<n_blocks, kBlockSize>>>(
      cell_id, pos_r, node_r, n_particles, n_cells);
  cuda_check(cudaGetLastError(), "census rebin kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "census rebin kernel execution failed");
}

__global__ void flip_negative_signs_kernel(std::int8_t* __restrict__ sign,
                                           const int n_particles) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_particles) return;
  if (sign[tid] < 0) sign[tid] = 1;
}

void flip_negative_census_signs_cuda(std::int8_t* sign, const int n_particles) {
  if (n_particles <= 0 || sign == nullptr) return;
  constexpr int kBlock = 256;
  const int grid = (n_particles + kBlock - 1) / kBlock;
  flip_negative_signs_kernel<<<grid, kBlock>>>(sign, n_particles);
  cuda_check(cudaGetLastError(), "flip negative signs launch failed");
  cuda_check(cudaDeviceSynchronize(), "flip negative signs execution failed");
}

SourceStats IMCSource::emit_thermal(core::State& state,
                                    const core::Config& cfg,
                                    const PlanckTable& planck,
                                    const double* sigma_a_eff,
                                    PhotonPool& pool,
                                    const int max_pool_size,
                                    const double dt,
                                    const std::uint64_t step_number,
                                    const std::uint64_t user_seed,
                                    const std::uint64_t step_base_gid,
                                    const std::vector<double>* nlte_eta,
                                    const std::vector<double>* nlte_f,
                                    const std::vector<double>* source_tilt,
                                    const double* sloc_mean_r,
                                    const double* sloc_sigma,
                                    const double* sloc_alpha,
                                    const double* sloc_prev_E,
                                    const int ppcg_override,
                                    const double* emission_bias_cdf,
                                    const std::vector<std::uint8_t>* source_skip_cell,
                                    const double* reference_E,
                                    const double* d_nlte_eta_input,
                                    const double* d_nlte_f_input) {
  SourceStats stats{};
  if (!cfg.radiation.enabled || dt <= 0.0 || state.rho.empty()) {
    return stats;
  }
  if (cfg.materials.materials.empty()) {
    return stats;
  }
  const int n_ranks = mpi_world_size();
  const parallel::Reduction reduction(n_ranks);
  const parallel::Reduction* reduction_ptr = (n_ranks > 1) ? &reduction : nullptr;

  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "emit_thermal requires cell_is_void size to match n_cells");
  const int n_groups = std::max(cfg.radiation.groups, 1);
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "emit_thermal planck/group size mismatch");
  if (state.mesh.dim == 1) {
    TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n_cells + 1),
                  "emit_thermal 1D requires x_r node count = n_cells + 1");
  } else if (state.mesh.dim == 2) {
    const std::size_t n_nodes_expected =
        expected_node_count_2d(state.mesh.topo.nr, state.mesh.topo.nz);
    TENRYU_ASSERT(state.x_r.size() == n_nodes_expected,
                  "emit_thermal 2D requires x_r node count to match topology");
    TENRYU_ASSERT(state.x_z.size() == n_nodes_expected,
                  "emit_thermal 2D requires x_z node count to match topology");
  }

  const auto& mat = cfg.materials.materials.front();
  const bool linearized = cfg.radiation.imc.linearized_planck;
  const double cv_override = mat.cv_e_override;
  const std::size_t n_bins = static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(n_bins <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "emit_thermal n_cells*n_groups exceeds int range");
  const int n_bins_i = static_cast<int>(n_bins);

  TENRYU_ASSERT((nlte_eta == nullptr) == (nlte_f == nullptr),
                "emit_thermal host NLTE eta/f pointers must both be null or both set");
  TENRYU_ASSERT((d_nlte_eta_input == nullptr) == (d_nlte_f_input == nullptr),
                "emit_thermal device NLTE eta/f pointers must both be null or both set");
  const bool nlte_host_mode = (nlte_eta != nullptr);
  const bool nlte_device_mode = (d_nlte_eta_input != nullptr);
  TENRYU_ASSERT(!nlte_host_mode || !nlte_device_mode,
                "emit_thermal NLTE coefficients must be host or device, not both");
  const bool nlte_mode = nlte_host_mode || nlte_device_mode;
  if (nlte_host_mode) {
    TENRYU_ASSERT(nlte_eta->size() == n_bins,
                  "emit_thermal NLTE eta size mismatch");
    TENRYU_ASSERT(nlte_f->size() == static_cast<std::size_t>(n_cells),
                  "emit_thermal NLTE f size mismatch");
  }
  if (nlte_device_mode) {
    TENRYU_ASSERT(d_nlte_eta_input != nullptr,
                  "emit_thermal device NLTE eta pointer is null");
    TENRYU_ASSERT(d_nlte_f_input != nullptr,
                  "emit_thermal device NLTE f pointer is null");
  }

  std::uint8_t* d_cell_is_void = nullptr;
  double* d_nlte_eta_owned = nullptr;
  double* d_nlte_f_owned = nullptr;
  const double* d_nlte_eta = d_nlte_eta_input;
  const double* d_nlte_f = d_nlte_f_input;
  double* d_source_tilt = nullptr;
  double* d_source_tilt_r_2d = nullptr;
  double* d_source_tilt_z_2d = nullptr;
  double* d_cell_centroid_r_2d = nullptr;
  double* d_cell_centroid_z_2d = nullptr;
  double* d_source_E = nullptr;
  double* d_emit_E = nullptr;
  double* d_source_cell_total = nullptr;
  double* d_source_total = nullptr;
  double* d_source_lost = nullptr;
  std::int32_t* d_count = nullptr;
  std::int32_t* d_offset = nullptr;
  void* d_scan_temp = nullptr;

  std::vector<std::uint8_t> source_cell_mask = state.cell_is_void;
  if (source_skip_cell != nullptr) {
    TENRYU_ASSERT(source_skip_cell->size() == static_cast<std::size_t>(n_cells),
                  "emit_thermal source_skip_cell size mismatch");
    for (int c = 0; c < n_cells; ++c) {
      if ((*source_skip_cell)[static_cast<std::size_t>(c)] != 0U) {
        source_cell_mask[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }

  const std::size_t bytes_void = sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "source:emit_thermal:d_cell_is_void", bytes_void));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        source_cell_mask.data(),
                        bytes_void,
                        cudaMemcpyHostToDevice),
             "emit_thermal copy cell_is_void failed");

  if (nlte_host_mode) {
    const std::size_t bytes_bins = sizeof(double) * n_bins;
    const std::size_t bytes_cells = sizeof(double) * static_cast<std::size_t>(n_cells);
    d_nlte_eta_owned = static_cast<double*>(core::device_scratch_acquire(
        "source:emit_thermal:d_nlte_eta_owned", bytes_bins));
    cuda_check(cudaMemcpy(d_nlte_eta_owned,
                          nlte_eta->data(),
                          bytes_bins,
                          cudaMemcpyHostToDevice),
               "emit_thermal copy nlte_eta failed");
    d_nlte_eta = d_nlte_eta_owned;
    d_nlte_f_owned = static_cast<double*>(core::device_scratch_acquire(
        "source:emit_thermal:d_nlte_f_owned", bytes_cells));
    cuda_check(cudaMemcpy(d_nlte_f_owned,
                          nlte_f->data(),
                          bytes_cells,
                          cudaMemcpyHostToDevice),
               "emit_thermal copy nlte_f failed");
    d_nlte_f = d_nlte_f_owned;
  }

  const bool use_source_tilt = (source_tilt != nullptr && state.mesh.dim == 1);
  const bool use_source_tilt_2d = (source_tilt != nullptr && state.mesh.dim == 2);
  if (use_source_tilt) {
    TENRYU_ASSERT(source_tilt->size() == static_cast<std::size_t>(n_cells),
                  "emit_thermal source_tilt size mismatch");
    const std::size_t bytes_cells = sizeof(double) * static_cast<std::size_t>(n_cells);
    d_source_tilt = static_cast<double*>(core::device_scratch_acquire(
        "source:emit_thermal:d_source_tilt", bytes_cells));
    cuda_check(cudaMemcpy(d_source_tilt,
                          source_tilt->data(),
                          bytes_cells,
                          cudaMemcpyHostToDevice),
               "emit_thermal copy source_tilt failed");
  }
  if (use_source_tilt_2d) {
    TENRYU_ASSERT(source_tilt->size() == 2U * static_cast<std::size_t>(n_cells),
                  "emit_thermal 2D source_tilt size mismatch");
  }

  const std::size_t bytes_bins = sizeof(double) * n_bins;
  const std::size_t bytes_cells = sizeof(double) * static_cast<std::size_t>(n_cells);
  d_source_E = static_cast<double*>(core::device_scratch_acquire(
      "source:emit_thermal:d_source_E", bytes_bins));
  d_emit_E = static_cast<double*>(core::device_scratch_acquire(
      "source:emit_thermal:d_emit_E", bytes_bins));
  if (emission_bias_cdf != nullptr) {
    d_source_cell_total = static_cast<double*>(core::device_scratch_acquire(
        "source:emit_thermal:d_source_cell_total", bytes_cells));
    cuda_check(cudaMemset(d_source_cell_total, 0, bytes_cells),
               "emit_thermal cudaMemset source_cell_total failed");
  }
  d_source_total = static_cast<double*>(core::device_scratch_acquire(
      "source:emit_thermal:d_source_total", sizeof(double)));
  cuda_check(cudaMemset(d_source_total, 0, sizeof(double)),
             "emit_thermal cudaMemset source_total failed");

  constexpr int kBlock = 256;
  const int grid_bins = (n_bins_i + kBlock - 1) / kBlock;
  compute_source_energy_kernel<<<grid_bins, kBlock>>>(state.Te.data(),
                                                      state.vol.data(),
                                                      sigma_a_eff,
                                                      reference_E,
                                                      d_cell_is_void,
                                                      d_nlte_eta,
                                                      d_nlte_f,
                                                      d_source_E,
                                                      d_emit_E,
                                                      d_source_cell_total,
                                                      d_source_total,
                                                      n_cells,
                                                      n_groups,
                                                      dt,
                                                      cfg.numerics.floors.Te,
                                                      linearized,
                                                      cv_override,
                                                      nlte_mode,
                                                      planck.device_view());
  cuda_check(cudaGetLastError(), "emit_thermal source-energy kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "emit_thermal source-energy kernel execution failed");

  state.rad_emit.reset(n_bins);
  cuda_check(cudaMemcpy(state.rad_emit.data(),
                        d_emit_E,
                        bytes_bins,
                        cudaMemcpyDeviceToDevice),
             "emit_thermal copy emit_E to rad_emit failed");

  double source_sum = 0.0;
  cuda_check(cudaMemcpy(&source_sum,
                        d_source_total,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "emit_thermal copy source_sum failed");

  if (source_sum <= 0.0 && reduction_ptr == nullptr) {
    return stats;
  }
  const bool difference_source = (reference_E != nullptr);
  const auto clear_rad_emit_if_legacy = [&]() {
    if (!difference_source) {
      state.rad_emit.reset(n_bins);
      state.rad_emit.fill(0.0);
    }
  };

  const int ppcg_raw = (ppcg_override > 0) ? ppcg_override
                                           : cfg.radiation.imc.particles_per_cell_group;
  const std::int64_t ppcg_ll = static_cast<std::int64_t>(ppcg_raw);
  const std::int64_t n_cells_ll = static_cast<std::int64_t>(n_cells);
  const std::int64_t n_groups_ll = static_cast<std::int64_t>(n_groups);
  TENRYU_ASSERT(ppcg_ll > 0 && n_cells_ll > 0 && n_groups_ll > 0,
                "emit_thermal requires positive particle count factors");
  TENRYU_ASSERT(ppcg_ll <= (std::numeric_limits<std::int64_t>::max() / n_cells_ll),
                "emit_thermal particle-count overflow (ppcg*n_cells)");
  const std::int64_t particles_per_step_ll = ppcg_ll * n_cells_ll;
  TENRYU_ASSERT(particles_per_step_ll <=
                    (std::numeric_limits<std::int64_t>::max() / n_groups_ll),
                "emit_thermal particle-count overflow (ppcg*n_cells*n_groups)");
  const std::int64_t requested_particles_ll = particles_per_step_ll * n_groups_ll;
  const std::int64_t N_tot_ll = std::max<std::int64_t>(1, requested_particles_ll);
  TENRYU_ASSERT(N_tot_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
                "emit_thermal particle count exceeds int range");
  int N_tot = static_cast<int>(N_tot_ll);
  const double E_per_target = source_sum / static_cast<double>(N_tot);
  constexpr double E_floor_photon = 1.0e-2;
  if (E_per_target < E_floor_photon && source_sum > 0.0) {
    N_tot = std::max(1, static_cast<int>(std::ceil(source_sum / E_floor_photon)));
  }

  d_source_lost = static_cast<double*>(core::device_scratch_acquire(
      "source:emit_thermal:d_source_lost", sizeof(double)));
  cuda_check(cudaMemset(d_source_lost, 0, sizeof(double)),
             "emit_thermal cudaMemset source_lost failed");
  d_count = static_cast<std::int32_t*>(core::device_scratch_acquire(
      "source:emit_thermal:d_count", sizeof(std::int32_t) * n_bins));
  d_offset = static_cast<std::int32_t*>(core::device_scratch_acquire(
      "source:emit_thermal:d_offset", sizeof(std::int32_t) * (n_bins + 1)));

  source_particle_count_kernel<<<grid_bins, kBlock>>>(
      d_source_E,
      d_source_cell_total,
      emission_bias_cdf,
      d_count,
      d_source_lost,
      n_bins_i,
      n_groups,
      source_sum,
      N_tot);
  cuda_check(cudaGetLastError(), "emit_thermal source-count kernel launch failed");

  std::size_t scan_temp_bytes = 0;
  cuda_check(cub::DeviceScan::ExclusiveSum(
                 nullptr, scan_temp_bytes, d_count, d_offset, n_bins_i),
             "emit_thermal CUB ExclusiveSum size query failed");
  if (scan_temp_bytes > 0) {
    d_scan_temp = static_cast<void*>(core::device_scratch_acquire(
        "source:emit_thermal:d_scan_temp", scan_temp_bytes));
  }
  cuda_check(cub::DeviceScan::ExclusiveSum(
                 d_scan_temp, scan_temp_bytes, d_count, d_offset, n_bins_i),
             "emit_thermal CUB ExclusiveSum failed");
  finalize_scan_tail_kernel<<<1, 1>>>(n_bins_i, d_count, d_offset);
  cuda_check(cudaGetLastError(), "emit_thermal finalize scan tail launch failed");

  std::int32_t n_new_i32 = 0;
  cuda_check(cudaMemcpy(&n_new_i32,
                        d_offset + n_bins_i,
                        sizeof(std::int32_t),
                        cudaMemcpyDeviceToHost),
             "emit_thermal copy n_new failed");
  TENRYU_ASSERT(n_new_i32 >= 0, "emit_thermal n_new must be non-negative");

  // Energy accounting: all overflow/skip early-return paths below set
  // stats.E_thermal_lost = source_sum to ensure no energy disappears
  // from the coupled system (see audit A5/A12 fix).
  const auto n_new_i64 = static_cast<std::int64_t>(n_new_i32);
  const std::uint64_t gid_base =
      compute_emission_gid_base(step_base_gid, n_new_i64, pool.n_alive, reduction_ptr);
  if (n_new_i64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    // CPU-side assignment; no atomicExch needed (single-threaded source generation).
    stats.pool_overflow = 1;
    stats.E_thermal_lost = source_sum;  // ALL source energy lost on overflow
    state.radiation_device_flags.pool_overflow = 1;
    clear_rad_emit_if_legacy();
    core::log_warning("source: emit_thermal n_new exceeds int32 max; skipping injection");
    return stats;
  }
  const int n_new = static_cast<int>(n_new_i64);
  if (n_new == 0) {
    stats.E_thermal_lost = source_sum;
    clear_rad_emit_if_legacy();
    return stats;
  }

  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + n_new_i64;
  const bool overflow = required_i64 > static_cast<std::int64_t>(max_pool_size);
  if (overflow) {
    // CPU-side assignment; no atomicExch needed (single-threaded source generation).
    stats.pool_overflow = 1;
    stats.E_thermal_lost = source_sum;
    state.radiation_device_flags.pool_overflow = 1;
    clear_rad_emit_if_legacy();
    core::log_error("[imc:pool] reserve overflow stage=thermal, step=" +
                    std::to_string(step_number) + ", n_alive=" +
                    std::to_string(pool.n_alive) + ", n_new=" +
                    std::to_string(n_new) + ", required=" +
                    std::to_string(required_i64) + ", max_pool_size=" +
                    std::to_string(max_pool_size));
    return stats;
  }
  const int required_capacity = static_cast<int>(required_i64);
  pool.reserve(required_capacity, max_pool_size);
  const int start = pool.n_alive;

  fill_emission_metadata_kernel<<<grid_bins, kBlock>>>(start,
                                                       n_bins_i,
                                                       n_groups,
                                                       d_count,
                                                       d_offset,
                                                       d_source_E,
                                                       gid_base,
                                                       dt,
                                                       pool.energy,
                                                       pool.weight,
                                                       pool.time_remain,
                                                       pool.birth_energy,
                                                       pool.sign,
                                                       pool.global_id,
                                                       pool.rng_counter,
                                                       pool.cell_id,
                                                       pool.group_id,
                                                       pool.mode,
                                                       pool.alive);
  cuda_check(cudaGetLastError(), "emit_thermal metadata kernel launch failed");

  const int grid = (n_new + kBlock - 1) / kBlock;
  if (state.mesh.dim == 2) {
    if (use_source_tilt_2d) {
      upload_source_tilt_2d(state,
                            *source_tilt,
                            &d_source_tilt_r_2d,
                            &d_source_tilt_z_2d,
                            &d_cell_centroid_r_2d,
                            &d_cell_centroid_z_2d);
    }
    const std::uint32_t max_draws_2d =
        use_source_tilt_2d
            ? (3U * static_cast<std::uint32_t>(kMaxRejectSamples2DMax) + 3U)
            : (3U * static_cast<std::uint32_t>(kMaxRejectSamples) + 3U);
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, max_draws_2d, "emit_thermal(2d)");
    TENRYU_ASSERT(state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0,
                  "emit_thermal 2D requires valid mesh topology");
    TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                  "emit_thermal 2D requires x_z/x_r node size match");
    fill_thermal_phase_space_2d_kernel<<<grid, kBlock>>>(start,
                                                         n_new,
                                                         state.x_r.data(),
                                                         state.x_z.data(),
                                                         pool.cell_id,
                                                         pool.time_remain,
                                                         pool.pos_r,
                                                         pool.pos_z,
                                                         pool.dir_r,
                                                         pool.dir_z,
                                                         pool.dir_phi,
                                                         pool.global_id,
                                                         pool.rng_counter,
                                                         user_seed,
                                                         step_number,
                                                         d_source_tilt_r_2d,
                                                         d_source_tilt_z_2d,
                                                         d_cell_centroid_r_2d,
                                                         d_cell_centroid_z_2d,
                                                         dt,
                                                         n_cells,
                                                         state.mesh.topo.nr,
                                                         state.mesh.topo.nz);
  } else {
    const bool use_source_localization =
        (sloc_mean_r != nullptr && sloc_prev_E != nullptr);
    warn_if_rng_counter_near_wrap_range(pool.rng_counter,
                                        start,
                                        n_new,
                                        use_source_localization ? 7U : 4U,
                                        "emit_thermal(1d)");
    if (use_source_localization) {
      fill_thermal_phase_space_kernel<<<grid, kBlock>>>(start,
                                                        n_new,
                                                        state.x_r.data(),
                                                        d_source_tilt,
                                                        sloc_mean_r,
                                                        sloc_sigma,
                                                        sloc_alpha,
                                                        sloc_prev_E,
                                                        pool.cell_id,
                                                        pool.time_remain,
                                                        pool.pos_r,
                                                        pool.pos_z,
                                                        pool.dir_r,
                                                        pool.dir_z,
                                                        pool.dir_phi,
                                                        pool.global_id,
                                                        pool.rng_counter,
                                                        user_seed,
                                                        step_number,
                                                        dt,
                                                        n_cells);
    } else if (d_source_tilt != nullptr) {
      fill_thermal_phase_space_tilted_kernel<<<grid, kBlock>>>(start,
                                                               n_new,
                                                               state.x_r.data(),
                                                               d_source_tilt,
                                                               sloc_mean_r,
                                                               sloc_prev_E,
                                                               pool.cell_id,
                                                               pool.time_remain,
                                                               pool.pos_r,
                                                               pool.pos_z,
                                                               pool.dir_r,
                                                               pool.dir_z,
                                                               pool.dir_phi,
                                                               pool.global_id,
                                                               pool.rng_counter,
                                                               user_seed,
                                                               step_number,
                                                               dt,
                                                               n_cells);
    } else {
      fill_thermal_phase_space_kernel<<<grid, kBlock>>>(start,
                                                        n_new,
                                                        state.x_r.data(),
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        nullptr,
                                                        pool.cell_id,
                                                        pool.time_remain,
                                                        pool.pos_r,
                                                        pool.pos_z,
                                                        pool.dir_r,
                                                        pool.dir_z,
                                                        pool.dir_phi,
                                                        pool.global_id,
                                                        pool.rng_counter,
                                                        user_seed,
                                                        step_number,
                                                        dt,
                                                        n_cells);
    }
  }
  cuda_check(cudaGetLastError(), "emit_thermal phase-space kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "emit_thermal phase-space kernel execution failed");

  double source_lost = 0.0;
  cuda_check(cudaMemcpy(&source_lost,
                        d_source_lost,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "emit_thermal copy source_lost failed");

  pool.n_alive += n_new;
  stats.n_thermal = n_new;
  stats.E_thermal_lost = std::max(source_lost, 0.0);
  stats.E_thermal = std::max(source_sum - stats.E_thermal_lost, 0.0);
  if (source_sum > 0.0 && stats.E_thermal_lost / source_sum > 1.0e-4) {
    static std::uint64_t last_warned_step = 0;
    if (step_number <= 1 || step_number >= last_warned_step + 100) {
      core::log_warning("[imc:emit_thermal] E_thermal_lost/source_sum = " +
                        std::to_string(stats.E_thermal_lost / source_sum) +
                        " exceeds threshold 1e-4; step=" + std::to_string(step_number));
      last_warned_step = step_number;
    }
  }
  return stats;
}

SourceStats IMCSource::append_census_residual_1d(
    core::State& state,
    PhotonPool& pool,
    const int max_pool_size,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    const std::uint64_t gid_base,
    const std::vector<std::int32_t>& cell_id,
    const std::vector<std::uint16_t>& group_id,
    const std::vector<double>& energy,
    const std::vector<std::int8_t>& sign) {
  SourceStats stats{};
  const int n_new = static_cast<int>(energy.size());
  TENRYU_ASSERT(state.mesh.dim == 1 || state.mesh.dim == 2,
                "append_census_residual_1d requires 1D_SPH or 2D_RZ");
  TENRYU_ASSERT(static_cast<int>(cell_id.size()) == n_new,
                "append_census_residual_1d cell size mismatch");
  TENRYU_ASSERT(static_cast<int>(group_id.size()) == n_new,
                "append_census_residual_1d group size mismatch");
  TENRYU_ASSERT(static_cast<int>(sign.size()) == n_new,
                "append_census_residual_1d sign size mismatch");
  if (n_new <= 0) {
    return stats;
  }
  if (state.mesh.dim == 1) {
    TENRYU_ASSERT(state.x_r.size() >= state.rho.size() + 1U,
                  "append_census_residual_1d requires node-centered x_r");
  } else {
    TENRYU_ASSERT(state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0,
                  "append_census_residual_1d 2D requires valid mesh topology");
    TENRYU_ASSERT(static_cast<std::size_t>(state.mesh.topo.nr) *
                      static_cast<std::size_t>(state.mesh.topo.nz) ==
                      state.rho.size(),
                  "append_census_residual_1d 2D requires nr*nz cells");
    const std::size_t n_nodes_expected =
        expected_node_count_2d(state.mesh.topo.nr, state.mesh.topo.nz);
    TENRYU_ASSERT(state.x_r.size() == n_nodes_expected,
                  "append_census_residual_1d 2D requires x_r node count to match topology");
    TENRYU_ASSERT(state.x_z.size() == n_nodes_expected,
                  "append_census_residual_1d 2D requires x_z node count to match topology");
  }

  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + static_cast<std::int64_t>(n_new);
  if (required_i64 > static_cast<std::int64_t>(max_pool_size)) {
    stats.pool_overflow = 1;
    state.radiation_device_flags.pool_overflow = 1;
    core::log_error("[imc:pool] reserve overflow stage=difference_census_residual, step=" +
                    std::to_string(step_number) + ", n_alive=" +
                    std::to_string(pool.n_alive) + ", n_new=" +
                    std::to_string(n_new) + ", required=" +
                    std::to_string(required_i64) + ", max_pool_size=" +
                    std::to_string(max_pool_size));
    return stats;
  }

  const int required_capacity = static_cast<int>(required_i64);
  pool.reserve(required_capacity, max_pool_size);
  const int start = pool.n_alive;
  append_particles(pool,
                   start,
                   cell_id,
                   group_id,
                   energy,
                   dt,
                   gid_base,
                   &sign);

  constexpr int kBlock = 256;
  const int grid = (n_new + kBlock - 1) / kBlock;
  if (state.mesh.dim == 2) {
    constexpr std::uint32_t max_draws_2d =
        3U * static_cast<std::uint32_t>(kMaxRejectSamples) + 3U;
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, max_draws_2d, "append_census_residual_2d");
    fill_thermal_phase_space_2d_kernel<<<grid, kBlock>>>(start,
                                                         n_new,
                                                         state.x_r.data(),
                                                         state.x_z.data(),
                                                         pool.cell_id,
                                                         pool.time_remain,
                                                         pool.pos_r,
                                                         pool.pos_z,
                                                         pool.dir_r,
                                                         pool.dir_z,
                                                         pool.dir_phi,
                                                         pool.global_id,
                                                         pool.rng_counter,
                                                         user_seed,
                                                         step_number,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         dt,
                                                         static_cast<int>(state.rho.size()),
                                                         state.mesh.topo.nr,
                                                         state.mesh.topo.nz);
  } else {
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, 4U, "append_census_residual_1d");
    fill_thermal_phase_space_kernel<<<grid, kBlock>>>(start,
                                                      n_new,
                                                      state.x_r.data(),
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      pool.cell_id,
                                                      pool.time_remain,
                                                      pool.pos_r,
                                                      pool.pos_z,
                                                      pool.dir_r,
                                                      pool.dir_z,
                                                      pool.dir_phi,
                                                      pool.global_id,
                                                      pool.rng_counter,
                                                      user_seed,
                                                      step_number,
                                                      dt,
                                                      static_cast<int>(state.rho.size()));
  }
  cuda_check(cudaGetLastError(),
             "append_census_residual_1d phase-space kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "append_census_residual_1d phase-space kernel execution failed");

  pool.n_alive += n_new;
  stats.n_thermal = n_new;
  for (const double E : energy) {
    stats.E_thermal += std::max(E, 0.0);
  }
  return stats;
}

SourceStats IMCSource::emit_volume_source(core::State& state,
                                          const core::Config& cfg,
                                          PhotonPool& pool,
                                          const int max_pool_size,
                                          const double dt,
                                          const std::uint64_t step_number,
                                          const std::uint64_t user_seed,
                                          const std::uint64_t step_base_gid) {
  SourceStats stats{};
  if (!cfg.radiation.enabled || dt <= 0.0 || state.vol.empty() || state.x_r.empty()) {
    return stats;
  }
  if (cfg.radiation.volume_source_rate <= 0.0 || cfg.radiation.volume_source_x_max <= 0.0) {
    return stats;
  }
  const int n_ranks = mpi_world_size();
  const parallel::Reduction reduction(n_ranks);
  const parallel::Reduction* reduction_ptr = (n_ranks > 1) ? &reduction : nullptr;

  const int n_cells = static_cast<int>(state.vol.size());
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "emit_volume_source requires cell_is_void size to match n_cells");
  const int n_groups = std::max(cfg.radiation.groups, 1);
  if (state.mesh.dim == 1) {
    TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n_cells + 1),
                  "emit_volume_source 1D requires node-centered x_r");
  } else {
    TENRYU_ASSERT(static_cast<int>(state.mesh.cell_centroid_z.size()) == n_cells,
                  "emit_volume_source 2D requires cell centroid z");
    const std::size_t n_nodes_expected =
        expected_node_count_2d(state.mesh.topo.nr, state.mesh.topo.nz);
    TENRYU_ASSERT(state.x_r.size() == n_nodes_expected,
                  "emit_volume_source 2D requires x_r node count to match topology");
    TENRYU_ASSERT(state.x_z.size() == n_nodes_expected,
                  "emit_volume_source 2D requires x_z node count to match topology");
  }

  std::vector<double> vol(state.vol.size(), 0.0);
  std::vector<double> x_r(state.x_r.size(), 0.0);
  state.vol.copy_to_host(vol.data());
  state.x_r.copy_to_host(x_r.data());

  std::vector<double> source_E(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  int n_source_cells = 0;
  double source_sum = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    double x_center = 0.0;
    if (state.mesh.dim == 2) {
      x_center = state.mesh.cell_centroid_z[static_cast<std::size_t>(c)];
    } else {
      x_center = 0.5 * (x_r[static_cast<std::size_t>(c)] +
                        x_r[static_cast<std::size_t>(c + 1)]);
    }
    if (x_center > cfg.radiation.volume_source_x_max) {
      continue;
    }
    ++n_source_cells;
    const double V = std::max(vol[static_cast<std::size_t>(c)], 0.0);
    const double E_cell = std::max(cfg.radiation.volume_source_rate * V * dt, 0.0);
    for (int g = 0; g < n_groups; ++g) {
      const int idx = c * n_groups + g;
      const double b_g = (n_groups == 1) ? 1.0 : (1.0 / static_cast<double>(n_groups));
      const double E = E_cell * b_g;
      source_E[static_cast<std::size_t>(idx)] = E;
      source_sum += E;
    }
  }

  if ((n_source_cells == 0 || source_sum <= 0.0) && reduction_ptr == nullptr) {
    return stats;
  }

  const std::int64_t ppcg_ll =
      static_cast<std::int64_t>(cfg.radiation.imc.particles_per_cell_group);
  const std::int64_t n_source_cells_ll = static_cast<std::int64_t>(n_source_cells);
  const std::int64_t n_groups_ll = static_cast<std::int64_t>(n_groups);
  TENRYU_ASSERT(ppcg_ll > 0 && n_groups_ll > 0,
                "emit_volume_source requires positive particle count factors");
  std::int64_t requested_particles_ll = 0;
  if (n_source_cells_ll > 0) {
    TENRYU_ASSERT(ppcg_ll <=
                      (std::numeric_limits<std::int64_t>::max() / n_source_cells_ll),
                  "emit_volume_source particle-count overflow (ppcg*n_source_cells)");
    const std::int64_t particles_per_step_ll = ppcg_ll * n_source_cells_ll;
    TENRYU_ASSERT(particles_per_step_ll <=
                      (std::numeric_limits<std::int64_t>::max() / n_groups_ll),
                  "emit_volume_source particle-count overflow (ppcg*n_source_cells*n_groups)");
    requested_particles_ll = particles_per_step_ll * n_groups_ll;
  }
  const std::int64_t N_tot_ll = std::max<std::int64_t>(1, requested_particles_ll);
  TENRYU_ASSERT(N_tot_ll <= static_cast<std::int64_t>(std::numeric_limits<int>::max()),
                "emit_volume_source particle count exceeds int range");
  const int N_tot = static_cast<int>(N_tot_ll);

  std::vector<std::int32_t> out_cell;
  std::vector<std::uint16_t> out_group;
  std::vector<double> out_energy;
  out_cell.reserve(static_cast<std::size_t>(N_tot));
  out_group.reserve(static_cast<std::size_t>(N_tot));
  out_energy.reserve(static_cast<std::size_t>(N_tot));

  for (int c = 0; c < n_cells; ++c) {
    if (state.cell_is_void[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    for (int g = 0; g < n_groups; ++g) {
      const int idx = c * n_groups + g;
      const double E_bin = source_E[static_cast<std::size_t>(idx)];
      if (E_bin <= 0.0) {
        continue;
      }
      const double ratio = E_bin / source_sum;
      const int n_p = std::max(1, static_cast<int>(std::floor(N_tot * ratio + 0.5)));
      const double E_p = E_bin / static_cast<double>(n_p);
      for (int k = 0; k < n_p; ++k) {
        out_cell.push_back(c);
        out_group.push_back(static_cast<std::uint16_t>(g));
        out_energy.push_back(E_p);
      }
    }
  }
  const auto n_new_i64 = static_cast<std::int64_t>(out_energy.size());
  const std::uint64_t gid_base =
      compute_emission_gid_base(step_base_gid, n_new_i64, pool.n_alive, reduction_ptr);
  if (n_new_i64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    // CPU-side assignment; no atomicExch needed (single-threaded source generation).
    stats.pool_overflow = 1;
    state.radiation_device_flags.pool_overflow = 1;
    core::log_warning("source: emit_volume_source n_new exceeds int32 max; skipping injection");
    return stats;
  }
  const int n_new = static_cast<int>(n_new_i64);
  if (n_new == 0) {
    return stats;
  }

  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + n_new_i64;
  const bool overflow = required_i64 > static_cast<std::int64_t>(max_pool_size);
  if (overflow) {
    // CPU-side assignment; no atomicExch needed (single-threaded source generation).
    stats.pool_overflow = 1;
    state.radiation_device_flags.pool_overflow = 1;
    core::log_error("[imc:pool] reserve overflow stage=volume, step=" +
                    std::to_string(step_number) + ", n_alive=" +
                    std::to_string(pool.n_alive) + ", n_new=" +
                    std::to_string(n_new) + ", required=" +
                    std::to_string(required_i64) + ", max_pool_size=" +
                    std::to_string(max_pool_size));
    return stats;
  }
  const int required_capacity = static_cast<int>(required_i64);
  pool.reserve(required_capacity, max_pool_size);
  const int start = pool.n_alive;
  append_particles(pool,
                   start,
                   out_cell,
                   out_group,
                   out_energy,
                   dt,
                   gid_base);

  constexpr int kBlock = 256;
  const int grid = (n_new + kBlock - 1) / kBlock;
  if (state.mesh.dim == 2) {
    constexpr bool use_2d_tilt = false;
    const std::uint32_t max_draws_2d =
        use_2d_tilt ? (3U * static_cast<std::uint32_t>(kMaxRejectSamples2DMax) + 3U)
                    : (3U * static_cast<std::uint32_t>(kMaxRejectSamples) + 3U);
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, max_draws_2d, "emit_volume_source(2d)");
    TENRYU_ASSERT(state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0,
                  "emit_volume_source 2D requires valid mesh topology");
    fill_thermal_phase_space_2d_kernel<<<grid, kBlock>>>(start,
                                                         n_new,
                                                         state.x_r.data(),
                                                         state.x_z.data(),
                                                         pool.cell_id,
                                                         pool.time_remain,
                                                         pool.pos_r,
                                                         pool.pos_z,
                                                         pool.dir_r,
                                                         pool.dir_z,
                                                         pool.dir_phi,
                                                         pool.global_id,
                                                         pool.rng_counter,
                                                         user_seed,
                                                         step_number,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         nullptr,
                                                         dt,
                                                         n_cells,
                                                         state.mesh.topo.nr,
                                                         state.mesh.topo.nz);
  } else {
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, 4U, "emit_volume_source(1d)");
    fill_thermal_phase_space_kernel<<<grid, kBlock>>>(start,
                                                      n_new,
                                                      state.x_r.data(),
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      nullptr,
                                                      pool.cell_id,
                                                      pool.time_remain,
                                                      pool.pos_r,
                                                      pool.pos_z,
                                                      pool.dir_r,
                                                      pool.dir_z,
                                                      pool.dir_phi,
                                                      pool.global_id,
                                                      pool.rng_counter,
                                                      user_seed,
                                                      step_number,
                                                      dt,
                                                      n_cells);
  }
  cuda_check(cudaGetLastError(), "emit_volume_source phase-space kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "emit_volume_source phase-space kernel execution failed");

  pool.n_alive += n_new;
  stats.n_thermal = n_new;
  stats.E_thermal = std::accumulate(out_energy.begin(), out_energy.end(), 0.0);
  return stats;
}

SourceStats IMCSource::emit_marshak(core::State& state,
                                    const core::Config& cfg,
                                    const PlanckTableDeviceView& planck,
                                    PhotonPool& pool,
                                    const int max_pool_size,
                                    const double dt,
                                    const std::uint64_t step_number,
                                    const std::uint64_t user_seed,
                                    const std::uint64_t step_base_gid) {
  SourceStats stats{};
  if (!cfg.radiation.enabled || dt <= 0.0) {
    return stats;
  }
  if (state.Te.empty() || state.x_r.empty()) {
    return stats;
  }

  const bool is_2d = (state.mesh.dim == 2);
  if (!is_2d && cfg.radiation.boundary.outer_r != "marshak") {
    return stats;
  }
  if (is_2d && cfg.radiation.boundary.bottom_z != "marshak" &&
      cfg.radiation.boundary.top_z != "marshak") {
    return stats;
  }

  std::vector<double> x_r(state.x_r.size(), 0.0);
  state.x_r.copy_to_host(x_r.data());
  std::vector<double> x_z;
  if (is_2d) {
    x_z.assign(state.x_z.size(), 0.0);
    state.x_z.copy_to_host(x_z.data());
  }
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const int n_ranks = mpi_world_size();
  const parallel::Reduction reduction(n_ranks);
  const parallel::Reduction* reduction_ptr = (n_ranks > 1) ? &reduction : nullptr;

  double T_r = cfg.radiation.boundary.marshak_Tr_eV;
  if (!(T_r > 0.0) && state.marshak_Tr_1d.has_value()) {
    T_r = state.marshak_Tr_1d->eval(state.t);
  }

  if (is_2d) {
    TENRYU_ASSERT(state.mesh.topo.nr > 0 && state.mesh.topo.nz > 0,
                  "emit_marshak 2D requires valid mesh topology");
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    const int stride = nz + 1;
    const auto node_index = [stride](const int i, const int j) {
      return i * stride + j;
    };

    struct FaceSpec {
      std::string key;
      std::string alias;
      double area = 0.0;
      double mu_sign = 1.0;
      int cell_j = 0;
      int n_particles = 0;
      double energy = 0.0;
      double T_r_face = 0.0;
      std::vector<double> seg_cdf;
      std::vector<double> seg_r0;
      std::vector<double> seg_z0;
      std::vector<double> seg_r1;
      std::vector<double> seg_z1;
    };
    auto eval_face_temperature = [&](const std::string& key,
                                     const std::string& alias) {
      double t_face = T_r;
      auto eval_table = [&](const std::string& table_key) {
        const auto it = state.marshak_Tr_face_tables.find(table_key);
        if (it == state.marshak_Tr_face_tables.end()) {
          return false;
        }
        t_face = it->second.eval(state.t);
        return true;
      };
      if (!eval_table(key) && !eval_table(alias)) {
        t_face = T_r;
      }
      return t_face;
    };

    auto build_boundary_segments = [&](FaceSpec& face, const int j_face) {
      face.seg_cdf.clear();
      face.seg_r0.clear();
      face.seg_z0.clear();
      face.seg_r1.clear();
      face.seg_z1.clear();
      face.area = 0.0;

      for (int i = 0; i < nr; ++i) {
        const int n0 = node_index(i, j_face);
        const int n1 = node_index(i + 1, j_face);
        const double r0 = std::max(x_r[static_cast<std::size_t>(n0)], 0.0);
        const double z0 = x_z[static_cast<std::size_t>(n0)];
        const double r1 = std::max(x_r[static_cast<std::size_t>(n1)], 0.0);
        const double z1 = x_z[static_cast<std::size_t>(n1)];
        const double edge_len = std::hypot(r1 - r0, z1 - z0);
        if (!(edge_len > 0.0)) {
          continue;
        }
        const double area_seg = 2.0 * kPi * 0.5 * (r0 + r1) * edge_len;
        if (!(area_seg > 0.0) || !std::isfinite(area_seg)) {
          continue;
        }
        face.area += area_seg;
        face.seg_cdf.push_back(face.area);
        face.seg_r0.push_back(r0);
        face.seg_z0.push_back(z0);
        face.seg_r1.push_back(r1);
        face.seg_z1.push_back(z1);
      }

      if (!(face.area > 0.0)) {
        return;
      }
      for (double& v : face.seg_cdf) {
        v /= face.area;
      }
      face.seg_cdf.back() = 1.0;
    };

    std::vector<FaceSpec> faces;
    if (cfg.radiation.boundary.bottom_z == "marshak") {
      FaceSpec f{};
      f.key = "bottom_z";
      f.alias = "z_bottom";
      f.mu_sign = +1.0;
      f.cell_j = 0;
      f.T_r_face = eval_face_temperature(f.key, f.alias);
      build_boundary_segments(f, 0);
      if (f.area > 0.0 && f.T_r_face > 0.0) {
        faces.push_back(std::move(f));
      }
    }
    if (cfg.radiation.boundary.top_z == "marshak") {
      FaceSpec f{};
      f.key = "top_z";
      f.alias = "z_top";
      f.mu_sign = -1.0;
      f.cell_j = nz - 1;
      f.T_r_face = eval_face_temperature(f.key, f.alias);
      build_boundary_segments(f, nz);
      if (f.area > 0.0 && f.T_r_face > 0.0) {
        faces.push_back(std::move(f));
      }
    }
    if (faces.empty()) {
      if (reduction_ptr != nullptr) {
        static_cast<void>(compute_emission_gid_base(
            step_base_gid, 0LL, pool.n_alive, reduction_ptr));
      }
      return stats;
    }

    const int n_faces = static_cast<int>(faces.size());
    const int n_total = std::max(n_faces, cfg.radiation.boundary.marshak_particles);
    double area_total = 0.0;
    for (const auto& face : faces) {
      area_total += std::max(face.area, 0.0);
    }
    // TODO(mpi): n_face allocation must use global Marshak face area via MPI_Allreduce(SUM)
    // once distributed 2D Marshak source emission is enabled.
    TENRYU_ASSERT(area_total > 0.0,
                  "emit_marshak 2D requires positive total marshak face area");

    int sum_particles = 0;
    int idx_max_area = 0;
    double max_area = -1.0;
    for (int k = 0; k < n_faces; ++k) {
      const double ratio = faces[static_cast<std::size_t>(k)].area / area_total;
      int n_face = static_cast<int>(std::floor(n_total * ratio + 0.5));
      n_face = std::max(1, n_face);
      faces[static_cast<std::size_t>(k)].n_particles = n_face;
      sum_particles += n_face;
      if (faces[static_cast<std::size_t>(k)].area > max_area) {
        max_area = faces[static_cast<std::size_t>(k)].area;
        idx_max_area = k;
      }
    }

    int delta_particles = n_total - sum_particles;
    while (delta_particles > 0) {
      faces[static_cast<std::size_t>(idx_max_area)].n_particles += 1;
      --delta_particles;
    }
    while (delta_particles < 0) {
      bool adjusted = false;
      for (int pass = 0; pass < n_faces && delta_particles < 0; ++pass) {
        const int idx = (idx_max_area + pass) % n_faces;
        auto& n_face = faces[static_cast<std::size_t>(idx)].n_particles;
        if (n_face > 1) {
          --n_face;
          ++delta_particles;
          adjusted = true;
        }
      }
      if (!adjusted) {
        break;
      }
    }

    std::vector<std::int32_t> out_cell;
    std::vector<std::uint16_t> out_group;
    std::vector<double> out_energy;
    out_cell.reserve(static_cast<std::size_t>(n_total));
    out_group.reserve(static_cast<std::size_t>(n_total));
    out_energy.reserve(static_cast<std::size_t>(n_total));

    struct FaceLaunch {
      int rel_offset = 0;
      int n_new = 0;
      int cell_j = 0;
      double mu_sign = 1.0;
      double T_r_face = 0.0;
      int face_idx = -1;
    };
    std::vector<FaceLaunch> launches;
    launches.reserve(faces.size());

    int running_offset = 0;
    for (int face_idx = 0; face_idx < n_faces; ++face_idx) {
      auto& face = faces[static_cast<std::size_t>(face_idx)];
      const int n_face = face.n_particles;
      if (n_face <= 0) {
        continue;
      }

      face.energy = 0.25 * tenryu::core::constants::a_eV *
                    tenryu::core::constants::c_light *
                    safe_temperature_pow4(face.T_r_face) * face.area * dt;
      if (face.energy <= 0.0) {
        continue;
      }

      const double E_p = face.energy / static_cast<double>(n_face);
      for (int p = 0; p < n_face; ++p) {
        out_cell.push_back(0);
        out_group.push_back(0);
        out_energy.push_back(E_p);
      }

      launches.push_back(FaceLaunch{running_offset,
                                    n_face,
                                    face.cell_j,
                                    face.mu_sign,
                                    face.T_r_face,
                                    face_idx});
      running_offset += n_face;
      stats.n_marshak += n_face;
      stats.E_marshak += face.energy;
    }

    const auto n_new_i64 = static_cast<std::int64_t>(out_energy.size());
    const std::uint64_t gid_base =
        compute_emission_gid_base(step_base_gid, n_new_i64, pool.n_alive, reduction_ptr);
    if (n_new_i64 > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      // CPU-side assignment; no atomicExch needed (single-threaded source generation).
      stats.pool_overflow = 1;
      state.radiation_device_flags.pool_overflow = 1;
      core::log_warning("source: emit_marshak(2d) n_new exceeds int32 max; skipping injection");
      return stats;
    }
    const int n_new = static_cast<int>(n_new_i64);
    if (n_new == 0) {
      return stats;
    }

    const std::int64_t required_i64 =
        static_cast<std::int64_t>(pool.n_alive) + n_new_i64;
    const bool overflow = required_i64 > static_cast<std::int64_t>(max_pool_size);
    if (overflow) {
      // CPU-side assignment; no atomicExch needed (single-threaded source generation).
      stats.pool_overflow = 1;
      state.radiation_device_flags.pool_overflow = 1;
      core::log_error("[imc:pool] reserve overflow stage=marshak_2d, step=" +
                      std::to_string(step_number) + ", n_alive=" +
                      std::to_string(pool.n_alive) + ", n_new=" +
                      std::to_string(n_new) + ", required=" +
                      std::to_string(required_i64) + ", max_pool_size=" +
                      std::to_string(max_pool_size));
      return stats;
    }
    const int required_capacity = static_cast<int>(required_i64);
    pool.reserve(required_capacity, max_pool_size);
    const int start = pool.n_alive;
    append_particles(pool,
                     start,
                     out_cell,
                     out_group,
                     out_energy,
                     dt,
                     gid_base);

    constexpr int kBlock = 256;
    warn_if_rng_counter_near_wrap_range(
        pool.rng_counter, start, n_new, 6U, "emit_marshak(2d)");
    for (const auto& launch : launches) {
      const auto& face = faces[static_cast<std::size_t>(launch.face_idx)];
      const int n_segments = static_cast<int>(face.seg_cdf.size());
      if (n_segments <= 0) {
        continue;
      }

      double* d_seg_cdf = nullptr;
      double* d_seg_r0 = nullptr;
      double* d_seg_z0 = nullptr;
      double* d_seg_r1 = nullptr;
      double* d_seg_z1 = nullptr;
      const std::size_t bytes = static_cast<std::size_t>(n_segments) * sizeof(double);
      d_seg_cdf = static_cast<double*>(core::device_scratch_acquire(
          "source:emit_marshak:d_seg_cdf", bytes));
      d_seg_r0 = static_cast<double*>(core::device_scratch_acquire(
          "source:emit_marshak:d_seg_r0", bytes));
      d_seg_z0 = static_cast<double*>(core::device_scratch_acquire(
          "source:emit_marshak:d_seg_z0", bytes));
      d_seg_r1 = static_cast<double*>(core::device_scratch_acquire(
          "source:emit_marshak:d_seg_r1", bytes));
      d_seg_z1 = static_cast<double*>(core::device_scratch_acquire(
          "source:emit_marshak:d_seg_z1", bytes));
      cuda_check(cudaMemcpy(d_seg_cdf, face.seg_cdf.data(), bytes, cudaMemcpyHostToDevice),
                 "emit_marshak 2D copy d_seg_cdf failed");
      cuda_check(cudaMemcpy(d_seg_r0, face.seg_r0.data(), bytes, cudaMemcpyHostToDevice),
                 "emit_marshak 2D copy d_seg_r0 failed");
      cuda_check(cudaMemcpy(d_seg_z0, face.seg_z0.data(), bytes, cudaMemcpyHostToDevice),
                 "emit_marshak 2D copy d_seg_z0 failed");
      cuda_check(cudaMemcpy(d_seg_r1, face.seg_r1.data(), bytes, cudaMemcpyHostToDevice),
                 "emit_marshak 2D copy d_seg_r1 failed");
      cuda_check(cudaMemcpy(d_seg_z1, face.seg_z1.data(), bytes, cudaMemcpyHostToDevice),
                 "emit_marshak 2D copy d_seg_z1 failed");

      const int grid = (launch.n_new + kBlock - 1) / kBlock;
      fill_marshak_phase_space_2d_kernel<<<grid, kBlock>>>(
          start + launch.rel_offset,
          launch.n_new,
          launch.cell_j,
          launch.mu_sign,
          d_seg_cdf,
          d_seg_r0,
          d_seg_z0,
          d_seg_r1,
          d_seg_z1,
          n_segments,
          state.x_r.data(),
          pool.cell_id,
          pool.time_remain,
          pool.pos_r,
          pool.pos_z,
          pool.dir_r,
          pool.dir_z,
          pool.dir_phi,
          pool.group_id,
          pool.global_id,
          pool.rng_counter,
          user_seed,
          step_number,
          dt,
          n_groups,
          launch.T_r_face,
          planck,
          nr,
          nz);
    }
    cuda_check(cudaGetLastError(), "emit_marshak 2D phase-space kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "emit_marshak 2D phase-space kernel execution failed");

    pool.n_alive += n_new;
    return stats;
  }

  if (!(T_r > 0.0)) {
    return stats;
  }

  const double r_boundary = x_r.back();
  const double area = 4.0 * kPi * r_boundary * r_boundary;
  const double E_marshak = 0.25 * tenryu::core::constants::a_eV *
                           tenryu::core::constants::c_light *
                           safe_temperature_pow4(T_r) * area * dt;
  if (E_marshak <= 0.0) {
    return stats;
  }

  const int n_new = std::max(1, cfg.radiation.boundary.marshak_particles);
  const double E_p = E_marshak / static_cast<double>(n_new);
  const std::int64_t n_new_i64 = static_cast<std::int64_t>(n_new);
  const std::uint64_t gid_base =
      compute_emission_gid_base(step_base_gid, n_new_i64, pool.n_alive, reduction_ptr);

  std::vector<std::int32_t> out_cell(static_cast<std::size_t>(n_new),
                                     static_cast<std::int32_t>(state.Te.size() - 1));
  std::vector<std::uint16_t> out_group(static_cast<std::size_t>(n_new), 0);
  std::vector<double> out_energy(static_cast<std::size_t>(n_new), E_p);

  const std::int64_t required_i64 =
      static_cast<std::int64_t>(pool.n_alive) + static_cast<std::int64_t>(n_new);
  const bool overflow = required_i64 > static_cast<std::int64_t>(max_pool_size);
  if (overflow) {
    // CPU-side assignment; no atomicExch needed (single-threaded source generation).
    stats.pool_overflow = 1;
    state.radiation_device_flags.pool_overflow = 1;
    core::log_error("[imc:pool] reserve overflow stage=marshak, step=" +
                    std::to_string(step_number) + ", n_alive=" +
                    std::to_string(pool.n_alive) + ", n_new=" +
                    std::to_string(n_new) + ", required=" +
                    std::to_string(required_i64) + ", max_pool_size=" +
                    std::to_string(max_pool_size));
    return stats;
  }
  const int required_capacity = static_cast<int>(required_i64);
  pool.reserve(required_capacity, max_pool_size);
  const int start = pool.n_alive;
  append_particles(pool,
                   start,
                   out_cell,
                   out_group,
                   out_energy,
                   dt,
                   gid_base);

  constexpr int kBlock = 256;
  const int grid = (n_new + kBlock - 1) / kBlock;
  warn_if_rng_counter_near_wrap_range(pool.rng_counter, start, n_new, 4U, "emit_marshak(1d)");
  fill_marshak_phase_space_kernel<<<grid, kBlock>>>(start,
                                                    n_new,
                                                    r_boundary,
                                                    pool.time_remain,
                                                    pool.pos_r,
                                                    pool.pos_z,
                                                    pool.dir_r,
                                                    pool.dir_z,
                                                    pool.dir_phi,
                                                    pool.group_id,
                                                    pool.global_id,
                                                    pool.rng_counter,
                                                    user_seed,
                                                    step_number,
                                                    dt,
                                                    n_groups,
                                                    T_r,
                                                    planck);
  cuda_check(cudaGetLastError(), "emit_marshak phase-space kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "emit_marshak phase-space kernel execution failed");

  pool.n_alive += n_new;
  stats.n_marshak = n_new;
  stats.E_marshak = E_marshak;
  return stats;
}

}  // namespace tenryu::radiation
