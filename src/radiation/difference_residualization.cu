#include "radiation/difference_residualization.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/error.hpp"
#include "core/state.hpp"
#include "parallel/comm_buffers.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kConditionFloor = 1.0e-300;
constexpr double kConditionRel = 1.0e-12;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kMaxRejectSamples2D = 64;

enum DifferenceBinAction : int {
  kDifferenceBinNone = 0,
  kDifferenceBinScale = 1,
  kDifferenceBinRebuild = 2
};

struct DifferenceResidualDeviceTotals {
  int scaled_bins;
  int rebuilt_bins;
  int killed_bins;
  int empty_created;
  int valid_particles;
  int scaled_particles;
  int rebuild_particles;
  int n_out;
};

struct DifferenceResidualWorkspaceView {
  double* signed_U = nullptr;
  double* abs_U = nullptr;
  double* U_phys = nullptr;
  double* target_U = nullptr;
  double* ratio_abs = nullptr;
  int* count = nullptr;
  int* head = nullptr;
  int* action = nullptr;
  int* sign_flip = nullptr;
  int* particle_count = nullptr;
  int* particle_offset = nullptr;
  int* rebuild_flag = nullptr;
  int* rebuild_offset = nullptr;
  int* empty_flag = nullptr;
  int* empty_offset = nullptr;
  DifferenceResidualDeviceTotals* totals = nullptr;
};

struct PhotonPoolDeviceView {
  double* pos_r;
  double* pos_z;
  double* dir_r;
  double* dir_z;
  double* dir_phi;
  double* energy;
  double* weight;
  double* time_remain;
  double* birth_energy;
  std::int8_t* sign;
  std::uint64_t* global_id;
  std::uint32_t* rng_counter;
  std::int32_t* cell_id;
  std::uint16_t* group_id;
  std::uint8_t* mode;
  std::uint8_t* alive;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

PhotonPoolDeviceView make_photon_pool_view(const PhotonPool& pool) {
  return {pool.pos_r,
          pool.pos_z,
          pool.dir_r,
          pool.dir_z,
          pool.dir_phi,
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
          pool.alive};
}

std::size_t align_up(const std::size_t value, const std::size_t alignment) {
  return ((value + alignment - 1U) / alignment) * alignment;
}

template <typename T>
void bump_layout(std::size_t& offset, const std::size_t count) {
  offset = align_up(offset, alignof(T));
  offset += sizeof(T) * count;
}

template <typename T>
T* place_array(std::uint8_t* base, std::size_t& offset, const std::size_t count) {
  offset = align_up(offset, alignof(T));
  T* ptr = reinterpret_cast<T*>(base + offset);
  offset += sizeof(T) * count;
  return ptr;
}

std::size_t difference_workspace_bytes(const std::size_t n_bins,
                                       const std::size_t n_particles) {
  std::size_t offset = 0;
  bump_layout<double>(offset, n_bins);
  bump_layout<double>(offset, n_bins);
  bump_layout<double>(offset, n_bins);
  bump_layout<double>(offset, n_bins);
  bump_layout<double>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_particles);
  bump_layout<int>(offset, n_particles);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<int>(offset, n_bins);
  bump_layout<DifferenceResidualDeviceTotals>(offset, 1U);
  return offset;
}

DifferenceResidualWorkspaceView make_difference_workspace_view(
    parallel::DeviceArray& workspace,
    const std::size_t n_bins,
    const std::size_t n_particles) {
  workspace.resize(difference_workspace_bytes(n_bins, n_particles));
  std::uint8_t* base = static_cast<std::uint8_t*>(workspace.ptr);
  std::size_t offset = 0;
  DifferenceResidualWorkspaceView view{};
  view.signed_U = place_array<double>(base, offset, n_bins);
  view.abs_U = place_array<double>(base, offset, n_bins);
  view.U_phys = place_array<double>(base, offset, n_bins);
  view.target_U = place_array<double>(base, offset, n_bins);
  view.ratio_abs = place_array<double>(base, offset, n_bins);
  view.count = place_array<int>(base, offset, n_bins);
  view.head = place_array<int>(base, offset, n_bins);
  view.action = place_array<int>(base, offset, n_bins);
  view.sign_flip = place_array<int>(base, offset, n_bins);
  view.particle_count = place_array<int>(base, offset, n_particles);
  view.particle_offset = place_array<int>(base, offset, n_particles);
  view.rebuild_flag = place_array<int>(base, offset, n_bins);
  view.rebuild_offset = place_array<int>(base, offset, n_bins);
  view.empty_flag = place_array<int>(base, offset, n_bins);
  view.empty_offset = place_array<int>(base, offset, n_bins);
  view.totals = place_array<DifferenceResidualDeviceTotals>(base, offset, 1U);
  return view;
}

__device__ inline double finite_or_zero_device(const double value) {
  return isfinite(value) ? value : 0.0;
}

__device__ inline double finite_nonnegative_energy_device(const double value) {
  return (isfinite(value) && value > 0.0) ? value : 0.0;
}

__device__ inline double difference_draw_uniform(curandStatePhilox4_32_10_t* rng,
                                                 std::uint32_t* draws) {
  ++(*draws);
  return curand_uniform_double(rng);
}

__device__ inline double difference_sample_uniform_shell_radius_1d(
    const double xi_r,
    const double r_lo,
    const double r_hi) {
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  return cbrt(r_lo3 + xi_r * (r_hi3 - r_lo3));
}

__device__ inline void difference_bilinear_map(const double eta,
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

__device__ inline std::int8_t normalized_particle_sign_device(const std::int8_t sign) {
  return (sign < 0) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(1);
}

__device__ inline bool valid_difference_particle(const PhotonPoolDeviceView pool,
                                                 const int p,
                                                 const int n_cells,
                                                 const int n_groups,
                                                 int* key_out) {
  if (pool.alive[p] != kAlive) {
    return false;
  }
  const int cell = pool.cell_id[p];
  const int group = static_cast<int>(pool.group_id[p]);
  if (cell < 0 || cell >= n_cells || group < 0 || group >= n_groups) {
    return false;
  }
  *key_out = cell * n_groups + group;
  return true;
}

__global__ void difference_accumulate_bins_kernel(
    PhotonPoolDeviceView pool,
    const int n_particles,
    const int n_cells,
    const int n_groups,
    double* __restrict__ signed_U,
    double* __restrict__ abs_U,
    int* __restrict__ count,
    int* __restrict__ head,
    DifferenceResidualDeviceTotals* __restrict__ totals) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_particles) {
    return;
  }
  int key = -1;
  if (!valid_difference_particle(pool, p, n_cells, n_groups, &key)) {
    return;
  }
  const double E = finite_nonnegative_energy_device(pool.energy[p]);
  const std::int8_t sign = normalized_particle_sign_device(pool.sign[p]);
  atomicAdd(&signed_U[key], static_cast<double>(sign) * E);
  atomicAdd(&abs_U[key], E);
  atomicAdd(&count[key], 1);
  atomicMax(&head[key], p);
  atomicAdd(&totals->valid_particles, 1);
}

__global__ void kill_difference_census_in_holo_core_kernel(
    PhotonPoolDeviceView pool,
    const std::uint8_t* __restrict__ holo_core,
    const int n_particles,
    const int n_cells,
    int* __restrict__ killed_count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_particles || pool.alive[p] != kAlive) {
    return;
  }
  const int c = pool.cell_id[p];
  if (c >= 0 && c < n_cells && holo_core[c] != 0U) {
    pool.alive[p] = kDead;
    atomicAdd(killed_count, 1);
  }
}

__global__ void difference_prepare_physical_density_kernel(
    const double* __restrict__ signed_U,
    const double* __restrict__ previous_reference_U,
    const double* __restrict__ rad_E,
    const double* __restrict__ vol,
    const DifferenceResidualDeviceTotals* __restrict__ totals,
    const int have_previous_reference,
    const int n_bins,
    const int n_groups,
    double* __restrict__ U_phys,
    double* __restrict__ physical_E_density) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_bins) {
    return;
  }
  const int cell = idx / n_groups;
  const double V = fmax(vol[cell], 0.0);
  const bool use_bins = (have_previous_reference != 0) || totals->valid_particles > 0;
  double U = 0.0;
  if (use_bins) {
    U = finite_or_zero_device(signed_U[idx]);
    if (have_previous_reference != 0) {
      U += finite_or_zero_device(previous_reference_U[idx]);
    }
  } else if (rad_E != nullptr) {
    U = finite_or_zero_device(rad_E[idx]) * V;
  }
  U_phys[idx] = U;
  physical_E_density[idx] = (V > 0.0) ? (finite_or_zero_device(U) / V) : 0.0;
}

__global__ void difference_plan_bins_kernel(
    const double* __restrict__ signed_U,
    const double* __restrict__ abs_U,
    const int* __restrict__ count,
    const double* __restrict__ U_phys,
    const double* __restrict__ E_ref_start,
    const double* __restrict__ vol,
    const int n_bins,
    const int n_groups,
    double* __restrict__ target_U,
    double* __restrict__ ratio_abs,
    int* __restrict__ action,
    int* __restrict__ sign_flip,
    int* __restrict__ rebuild_flag,
    int* __restrict__ empty_flag,
    double* __restrict__ previous_reference_U_start,
    DifferenceResidualDeviceTotals* __restrict__ totals) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_bins) {
    return;
  }
  const int cell = b / n_groups;
  const double V = fmax(vol[cell], 0.0);
  const double U_ref = finite_or_zero_device(E_ref_start[b]) * V;
  previous_reference_U_start[b] = U_ref;
  const double target = finite_or_zero_device(U_phys[b]) - U_ref;
  target_U[b] = target;
  ratio_abs[b] = 0.0;
  action[b] = kDifferenceBinNone;
  sign_flip[b] = 0;
  rebuild_flag[b] = 0;
  empty_flag[b] = 0;

  const int n_bin = count[b];
  if (n_bin <= 0) {
    if (target != 0.0) {
      empty_flag[b] = 1;
    }
    return;
  }

  if (target == 0.0) {
    atomicAdd(&totals->killed_bins, 1);
    return;
  }

  const double R_old = finite_or_zero_device(signed_U[b]);
  const double scale = fmax(fmax(fabs(target), fabs(R_old)),
                           fmax(abs_U[b], kConditionFloor));
  const bool well_conditioned = fabs(R_old) > kConditionRel * scale;
  if (well_conditioned) {
    const double ratio = target / R_old;
    if (isfinite(ratio)) {
      ratio_abs[b] = fabs(ratio);
      action[b] = kDifferenceBinScale;
      sign_flip[b] = (ratio < 0.0) ? 1 : 0;
      atomicAdd(&totals->scaled_bins, 1);
      return;
    }
  }

  action[b] = kDifferenceBinRebuild;
  rebuild_flag[b] = 1;
  atomicAdd(&totals->rebuilt_bins, 1);
}

__global__ void difference_count_scaled_particles_kernel(
    PhotonPoolDeviceView pool,
    const int n_particles,
    const int n_cells,
    const int n_groups,
    const int* __restrict__ action,
    const double* __restrict__ ratio_abs,
    int* __restrict__ particle_count) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_particles) {
    return;
  }
  particle_count[p] = 0;
  int key = -1;
  if (!valid_difference_particle(pool, p, n_cells, n_groups, &key)) {
    return;
  }
  if (action[key] != kDifferenceBinScale) {
    return;
  }
  const double E_new = finite_nonnegative_energy_device(pool.energy[p]) * ratio_abs[key];
  particle_count[p] = (E_new > 0.0) ? 1 : 0;
}

__global__ void difference_finalize_totals_kernel(
    const int* __restrict__ particle_count,
    const int* __restrict__ particle_offset,
    const int n_particles,
    const int* __restrict__ rebuild_flag,
    const int* __restrict__ rebuild_offset,
    const int* __restrict__ empty_flag,
    const int* __restrict__ empty_offset,
    const int n_bins,
    DifferenceResidualDeviceTotals* __restrict__ totals) {
  int scaled = 0;
  if (n_particles > 0) {
    scaled = particle_offset[n_particles - 1] + particle_count[n_particles - 1];
  }
  int rebuilt = 0;
  int empty = 0;
  if (n_bins > 0) {
    rebuilt = rebuild_offset[n_bins - 1] + rebuild_flag[n_bins - 1];
    empty = empty_offset[n_bins - 1] + empty_flag[n_bins - 1];
  }
  totals->scaled_particles = scaled;
  totals->rebuild_particles = rebuilt;
  totals->empty_created = empty;
  totals->n_out = scaled + rebuilt;
}

__device__ inline void copy_particle_geometry(const PhotonPoolDeviceView in,
                                              PhotonPoolDeviceView out,
                                              const int src,
                                              const int dst) {
  out.pos_r[dst] = in.pos_r[src];
  out.pos_z[dst] = in.pos_z[src];
  out.dir_r[dst] = in.dir_r[src];
  out.dir_z[dst] = in.dir_z[src];
  out.dir_phi[dst] = in.dir_phi[src];
  out.time_remain[dst] =
      fmax(finite_or_zero_device(in.time_remain[src]), 0.0);
  out.global_id[dst] = in.global_id[src];
  out.rng_counter[dst] = in.rng_counter[src];
  out.cell_id[dst] = in.cell_id[src];
  out.group_id[dst] = in.group_id[src];
  out.mode[dst] = in.mode[src];
  out.alive[dst] = kAlive;
}

__global__ void difference_write_scaled_particles_kernel(
    PhotonPoolDeviceView in,
    PhotonPoolDeviceView out,
    const int n_particles,
    const int n_cells,
    const int n_groups,
    const int* __restrict__ action,
    const int* __restrict__ sign_flip,
    const double* __restrict__ ratio_abs,
    const int* __restrict__ particle_count,
    const int* __restrict__ particle_offset) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= n_particles || particle_count[p] == 0) {
    return;
  }
  int key = -1;
  if (!valid_difference_particle(in, p, n_cells, n_groups, &key) ||
      action[key] != kDifferenceBinScale) {
    return;
  }
  const int out_idx = particle_offset[p];
  const double ratio = ratio_abs[key];
  const double E_new = finite_nonnegative_energy_device(in.energy[p]) * ratio;
  if (!(E_new > 0.0)) {
    return;
  }
  const double birth_new = finite_nonnegative_energy_device(in.birth_energy[p]) * ratio;
  const double birth_out = (birth_new > 0.0) ? birth_new : E_new;
  const std::int8_t old_sign = normalized_particle_sign_device(in.sign[p]);
  const std::int8_t new_sign =
      (sign_flip[key] != 0) ? static_cast<std::int8_t>(-old_sign) : old_sign;
  copy_particle_geometry(in, out, p, out_idx);
  out.energy[out_idx] = E_new;
  out.birth_energy[out_idx] = birth_out;
  out.weight[out_idx] = (birth_out > 0.0) ? (E_new / birth_out) : 0.0;
  out.sign[out_idx] = new_sign;
}

__global__ void difference_write_rebuilt_bins_kernel(
    PhotonPoolDeviceView in,
    PhotonPoolDeviceView out,
    const int n_bins,
    const int* __restrict__ head,
    const int* __restrict__ rebuild_flag,
    const int* __restrict__ rebuild_offset,
    const double* __restrict__ target_U,
    const int scaled_total) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_bins || rebuild_flag[b] == 0) {
    return;
  }
  const int src = head[b];
  if (src < 0) {
    return;
  }
  const double E = fabs(target_U[b]);
  if (!(E > 0.0)) {
    return;
  }
  const int out_idx = scaled_total + rebuild_offset[b];
  copy_particle_geometry(in, out, src, out_idx);
  out.energy[out_idx] = E;
  out.birth_energy[out_idx] = E;
  out.weight[out_idx] = 1.0;
  out.sign[out_idx] =
      (target_U[b] < 0.0) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(1);
}

__global__ void difference_append_empty_particles_kernel(
    PhotonPoolDeviceView out,
    const double* __restrict__ target_U,
    const int* __restrict__ empty_flag,
    const int* __restrict__ empty_offset,
    const double* __restrict__ node_r,
    const int n_bins,
    const int n_groups,
    const int start,
    const std::uint64_t gid_base,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt,
    const int n_cells) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_bins || empty_flag[b] == 0) {
    return;
  }
  const int c = b / n_groups;
  if (c < 0 || c >= n_cells) {
    return;
  }
  const double E = fabs(target_U[b]);
  if (!(E > 0.0)) {
    return;
  }
  const int local = empty_offset[b];
  const int idx = start + local;
  out.energy[idx] = E;
  out.weight[idx] = 1.0;
  out.birth_energy[idx] = E;
  out.sign[idx] =
      (target_U[b] < 0.0) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(1);
  out.global_id[idx] = gid_base + static_cast<std::uint64_t>(local);
  out.rng_counter[idx] = 0U;
  out.cell_id[idx] = static_cast<std::int32_t>(c);
  out.group_id[idx] = static_cast<std::uint16_t>(b % n_groups);
  out.mode[idx] = kModeIMC;
  out.alive[idx] = kAlive;

  curandStatePhilox4_32_10_t rng;
  curand_init(out.global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(out.rng_counter[idx]),
              &rng);
  std::uint32_t draws = 0U;
  const double xi_r = difference_draw_uniform(&rng, &draws);
  const double xi_mu = difference_draw_uniform(&rng, &draws);
  const double xi_phi = difference_draw_uniform(&rng, &draws);
  const double xi_t = difference_draw_uniform(&rng, &draws);
  out.pos_r[idx] =
      difference_sample_uniform_shell_radius_1d(xi_r, node_r[c], node_r[c + 1]);
  out.pos_z[idx] = 0.0;
  const double mu = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  out.dir_r[idx] = mu;
  out.dir_z[idx] = sin_theta * cos(phi);
  out.dir_phi[idx] = sin_theta * sin(phi);
  out.time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);
  out.rng_counter[idx] = draws;
}

__global__ void difference_append_empty_particles_2d_kernel(
    PhotonPoolDeviceView out,
    const double* __restrict__ target_U,
    const int* __restrict__ empty_flag,
    const int* __restrict__ empty_offset,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const int n_bins,
    const int n_groups,
    const int start,
    const std::uint64_t gid_base,
    const std::uint64_t user_seed,
    const std::uint64_t step_number,
    const double dt,
    const int n_cells,
    const int nr,
    const int nz) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= n_bins || empty_flag[b] == 0) {
    return;
  }
  const int c = b / n_groups;
  if (c < 0 || c >= n_cells) {
    return;
  }
  const double E = fabs(target_U[b]);
  if (!(E > 0.0)) {
    return;
  }
  const int local = empty_offset[b];
  const int idx = start + local;
  out.energy[idx] = E;
  out.weight[idx] = 1.0;
  out.birth_energy[idx] = E;
  out.sign[idx] =
      (target_U[b] < 0.0) ? static_cast<std::int8_t>(-1) : static_cast<std::int8_t>(1);
  out.global_id[idx] = gid_base + static_cast<std::uint64_t>(local);
  out.rng_counter[idx] = 0U;
  out.cell_id[idx] = static_cast<std::int32_t>(c);
  out.group_id[idx] = static_cast<std::uint16_t>(b % n_groups);
  out.mode[idx] = kModeIMC;
  out.alive[idx] = kAlive;

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
  curand_init(out.global_id[idx] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(out.rng_counter[idx]),
              &rng);
  std::uint32_t draws = 0U;
  double r_p = 0.25 * (r00 + r10 + r11 + r01);
  double z_p = 0.25 * (z00 + z10 + z11 + z01);
  bool accepted = false;
  for (int n_try = 0; n_try < kMaxRejectSamples2D; ++n_try) {
    const double xi_eta = difference_draw_uniform(&rng, &draws);
    const double xi_zeta = difference_draw_uniform(&rng, &draws);
    const double xi_reject = difference_draw_uniform(&rng, &draws);
    if (accepted) {
      continue;
    }
    difference_bilinear_map(xi_eta,
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
    const double w =
        (r_max_cell > 0.0) ? fmin(1.0, fmax(0.0, r_p / r_max_cell)) : 1.0;
    accepted = (xi_reject <= w);
  }
  out.pos_r[idx] = fmax(r_p, 0.0);
  out.pos_z[idx] = z_p;

  const double xi_mu = difference_draw_uniform(&rng, &draws);
  const double xi_phi = difference_draw_uniform(&rng, &draws);
  const double xi_t = difference_draw_uniform(&rng, &draws);
  const double mu_z = 2.0 * xi_mu - 1.0;
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu_z * mu_z));
  out.dir_r[idx] = sin_theta * cos(phi);
  out.dir_z[idx] = mu_z;
  out.dir_phi[idx] = sin_theta * sin(phi);
  out.time_remain[idx] = fmax((1.0 - xi_t) * dt, 0.0);
  out.rng_counter[idx] = draws;
}

void exclusive_scan_int(const int* input,
                        int* output,
                        const int count,
                        parallel::DeviceArray& scan_workspace,
                        const char* message) {
  if (count <= 0) {
    return;
  }
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceScan::ExclusiveSum(nullptr,
                                           temp_bytes,
                                           input,
                                           output,
                                           count),
             message);
  scan_workspace.resize(temp_bytes);
  cuda_check(cub::DeviceScan::ExclusiveSum(scan_workspace.ptr,
                                           temp_bytes,
                                           input,
                                           output,
                                           count),
             message);
}

void reset_difference_workspace(const DifferenceResidualWorkspaceView& view,
                                const std::size_t n_bins,
                                const std::size_t n_particles) {
  if (n_bins > 0U) {
    cuda_check(cudaMemset(view.signed_U, 0, sizeof(double) * n_bins),
               "difference residualization memset signed_U failed");
    cuda_check(cudaMemset(view.abs_U, 0, sizeof(double) * n_bins),
               "difference residualization memset abs_U failed");
    cuda_check(cudaMemset(view.count, 0, sizeof(int) * n_bins),
               "difference residualization memset count failed");
    cuda_check(cudaMemset(view.head, 0xff, sizeof(int) * n_bins),
               "difference residualization memset head failed");
  }
  if (n_particles > 0U) {
    cuda_check(cudaMemset(view.particle_count, 0, sizeof(int) * n_particles),
               "difference residualization memset particle_count failed");
  }
  cuda_check(cudaMemset(view.totals, 0, sizeof(DifferenceResidualDeviceTotals)),
             "difference residualization memset totals failed");
}

}  // namespace

void prepare_difference_census_reference_cuda(const PhotonPool& pool,
                                              const double* rad_E,
                                              const double* vol,
                                              const double* previous_reference_U,
                                              const bool have_previous_reference,
                                              const int n_cells,
                                              const int n_groups,
                                              double* physical_E_density,
                                              parallel::DeviceArray& workspace) {
  TENRYU_ASSERT(vol != nullptr, "prepare_difference_census_reference_cuda requires vol");
  TENRYU_ASSERT(physical_E_density != nullptr,
                "prepare_difference_census_reference_cuda requires output density");
  const int n_particles = std::max(pool.n_alive, 0);
  const int n_bins = std::max(n_cells, 0) * std::max(n_groups, 0);
  if (n_bins <= 0) {
    return;
  }
  TENRYU_ASSERT(!have_previous_reference || previous_reference_U != nullptr,
                "prepare_difference_census_reference_cuda missing previous reference");

  DifferenceResidualWorkspaceView view =
      make_difference_workspace_view(workspace,
                                     static_cast<std::size_t>(n_bins),
                                     static_cast<std::size_t>(n_particles));
  reset_difference_workspace(view,
                             static_cast<std::size_t>(n_bins),
                             static_cast<std::size_t>(n_particles));

  constexpr int kBlock = 256;
  const PhotonPoolDeviceView pool_view = make_photon_pool_view(pool);
  if (n_particles > 0) {
    const int grid_particles = (n_particles + kBlock - 1) / kBlock;
    difference_accumulate_bins_kernel<<<grid_particles, kBlock>>>(pool_view,
                                                                  n_particles,
                                                                  n_cells,
                                                                  n_groups,
                                                                  view.signed_U,
                                                                  view.abs_U,
                                                                  view.count,
                                                                  view.head,
                                                                  view.totals);
    cuda_check(cudaGetLastError(),
               "difference residualization bin accumulate kernel launch failed");
  }

  const int grid_bins = (n_bins + kBlock - 1) / kBlock;
  difference_prepare_physical_density_kernel<<<grid_bins, kBlock>>>(
      view.signed_U,
      previous_reference_U,
      rad_E,
      vol,
      view.totals,
      have_previous_reference ? 1 : 0,
      n_bins,
      n_groups,
      view.U_phys,
      physical_E_density);
  cuda_check(cudaGetLastError(),
             "difference residualization physical density kernel launch failed");
}

int kill_difference_census_in_holo_core_cuda(PhotonPool& pool,
                                             const std::uint8_t* holo_core,
                                             const int n_particles,
                                             const int n_cells) {
  TENRYU_ASSERT(holo_core != nullptr,
                "kill_difference_census_in_holo_core_cuda requires holo_core");
  TENRYU_ASSERT(n_particles >= 0,
                "kill_difference_census_in_holo_core_cuda requires n_particles >= 0");
  TENRYU_ASSERT(n_cells >= 0,
                "kill_difference_census_in_holo_core_cuda requires n_cells >= 0");
  if (n_particles <= 0 || n_cells <= 0) {
    return 0;
  }

  int* d_killed_count = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_killed_count), sizeof(int)),
             "kill_difference_census_in_holo_core cudaMalloc killed_count failed");
  cuda_check(cudaMemset(d_killed_count, 0, sizeof(int)),
             "kill_difference_census_in_holo_core cudaMemset killed_count failed");

  constexpr int kBlock = 256;
  const int grid = (n_particles + kBlock - 1) / kBlock;
  const PhotonPoolDeviceView pool_view = make_photon_pool_view(pool);
  kill_difference_census_in_holo_core_kernel<<<grid, kBlock>>>(pool_view,
                                                               holo_core,
                                                               n_particles,
                                                               n_cells,
                                                               d_killed_count);
  cuda_check(cudaGetLastError(),
             "kill_difference_census_in_holo_core kernel launch failed");
  int killed_count = 0;
  cuda_check(cudaMemcpy(&killed_count,
                        d_killed_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "kill_difference_census_in_holo_core copy killed_count failed");
  cuda_check(cudaFree(d_killed_count),
             "kill_difference_census_in_holo_core cudaFree killed_count failed");
  return killed_count;
}

DifferenceResidualizationDeviceStats residualize_census_against_reference_cuda(
    core::State& state,
    PhotonPool& pool,
    const int max_pool_size,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    const std::uint64_t empty_gid_base,
    const double* E_ref_start,
    const double* vol,
    double* previous_reference_U_start,
    const int n_cells,
    const int n_groups,
    PhotonPool& output_workspace,
    parallel::DeviceArray& workspace,
    parallel::DeviceArray& scan_workspace) {
  DifferenceResidualizationDeviceStats stats{};
  stats.n_before = std::max(pool.n_alive, 0);
  const int n_particles = stats.n_before;
  const int n_bins = std::max(n_cells, 0) * std::max(n_groups, 0);
  TENRYU_ASSERT(E_ref_start != nullptr,
                "residualize_census_against_reference_cuda requires E_ref_start");
  TENRYU_ASSERT(vol != nullptr,
                "residualize_census_against_reference_cuda requires vol");
  TENRYU_ASSERT(previous_reference_U_start != nullptr,
                "residualize_census_against_reference_cuda requires previous reference output");
  if (n_bins <= 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return stats;
  }

  DifferenceResidualWorkspaceView view =
      make_difference_workspace_view(workspace,
                                     static_cast<std::size_t>(n_bins),
                                     static_cast<std::size_t>(n_particles));

  constexpr int kBlock = 256;
  const int grid_bins = (n_bins + kBlock - 1) / kBlock;
  difference_plan_bins_kernel<<<grid_bins, kBlock>>>(view.signed_U,
                                                     view.abs_U,
                                                     view.count,
                                                     view.U_phys,
                                                     E_ref_start,
                                                     vol,
                                                     n_bins,
                                                     n_groups,
                                                     view.target_U,
                                                     view.ratio_abs,
                                                     view.action,
                                                     view.sign_flip,
                                                     view.rebuild_flag,
                                                     view.empty_flag,
                                                     previous_reference_U_start,
                                                     view.totals);
  cuda_check(cudaGetLastError(),
             "difference residualization plan kernel launch failed");

  if (n_particles > 0) {
    const int grid_particles = (n_particles + kBlock - 1) / kBlock;
    const PhotonPoolDeviceView pool_view = make_photon_pool_view(pool);
    difference_count_scaled_particles_kernel<<<grid_particles, kBlock>>>(pool_view,
                                                                         n_particles,
                                                                         n_cells,
                                                                         n_groups,
                                                                         view.action,
                                                                         view.ratio_abs,
                                                                         view.particle_count);
    cuda_check(cudaGetLastError(),
               "difference residualization particle count kernel launch failed");
    exclusive_scan_int(view.particle_count,
                       view.particle_offset,
                       n_particles,
                       scan_workspace,
                       "difference residualization particle scan failed");
  }

  exclusive_scan_int(view.rebuild_flag,
                     view.rebuild_offset,
                     n_bins,
                     scan_workspace,
                     "difference residualization rebuild scan failed");
  exclusive_scan_int(view.empty_flag,
                     view.empty_offset,
                     n_bins,
                     scan_workspace,
                     "difference residualization empty scan failed");

  difference_finalize_totals_kernel<<<1, 1>>>(view.particle_count,
                                              view.particle_offset,
                                              n_particles,
                                              view.rebuild_flag,
                                              view.rebuild_offset,
                                              view.empty_flag,
                                              view.empty_offset,
                                              n_bins,
                                              view.totals);
  cuda_check(cudaGetLastError(),
             "difference residualization totals kernel launch failed");

  DifferenceResidualDeviceTotals host_totals{};
  cuda_check(cudaMemcpy(&host_totals,
                        view.totals,
                        sizeof(host_totals),
                        cudaMemcpyDeviceToHost),
             "difference residualization copy totals failed");
  stats.scaled_bins = host_totals.scaled_bins;
  stats.rebuilt_bins = host_totals.rebuilt_bins;
  stats.killed_bins = host_totals.killed_bins;
  stats.empty_created = host_totals.empty_created;

  const int n_out = host_totals.n_out;
  const int required_after_empty = n_out + host_totals.empty_created;
  TENRYU_ASSERT(required_after_empty <= max_pool_size,
                "difference census residualization exceeded max_pool_size");

  if (required_after_empty > 0) {
    output_workspace.n_alive = 0;
    output_workspace.n_census = 0;
    output_workspace.reserve(required_after_empty, max_pool_size);
    output_workspace.n_alive = n_out;
    output_workspace.n_census = n_out;
    const PhotonPoolDeviceView input_view = make_photon_pool_view(pool);
    const PhotonPoolDeviceView output_view = make_photon_pool_view(output_workspace);
    if (host_totals.scaled_particles > 0) {
      const int grid_particles = (n_particles + kBlock - 1) / kBlock;
      difference_write_scaled_particles_kernel<<<grid_particles, kBlock>>>(
          input_view,
          output_view,
          n_particles,
          n_cells,
          n_groups,
          view.action,
          view.sign_flip,
          view.ratio_abs,
          view.particle_count,
          view.particle_offset);
      cuda_check(cudaGetLastError(),
                 "difference residualization write scaled kernel launch failed");
    }
    if (host_totals.rebuild_particles > 0) {
      difference_write_rebuilt_bins_kernel<<<grid_bins, kBlock>>>(input_view,
                                                                  output_view,
                                                                  n_bins,
                                                                  view.head,
                                                                  view.rebuild_flag,
                                                                  view.rebuild_offset,
                                                                  view.target_U,
                                                                  host_totals.scaled_particles);
      cuda_check(cudaGetLastError(),
                 "difference residualization write rebuilt kernel launch failed");
    }
    pool.swap(output_workspace);
  } else {
    pool.n_alive = 0;
    pool.n_census = 0;
  }

  if (host_totals.empty_created > 0) {
    const PhotonPoolDeviceView pool_view = make_photon_pool_view(pool);
    if (state.mesh.dim == 1) {
      TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n_cells) + 1U,
                    "difference residual empty append requires node-centered x_r");
      difference_append_empty_particles_kernel<<<grid_bins, kBlock>>>(pool_view,
                                                                      view.target_U,
                                                                      view.empty_flag,
                                                                      view.empty_offset,
                                                                      state.x_r.data(),
                                                                      n_bins,
                                                                      n_groups,
                                                                      n_out,
                                                                      empty_gid_base,
                                                                      user_seed,
                                                                      step_number,
                                                                      dt,
                                                                      n_cells);
    } else if (state.mesh.dim == 2) {
      const int nr = state.mesh.topo.nr;
      const int nz = state.mesh.topo.nz;
      TENRYU_ASSERT(nr > 0 && nz > 0,
                    "difference residual empty append requires valid 2D topology");
      TENRYU_ASSERT(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) ==
                        static_cast<std::size_t>(n_cells),
                    "difference residual empty append requires nr*nz cells");
      const std::size_t n_nodes =
          static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
      TENRYU_ASSERT(state.x_r.size() == n_nodes,
                    "difference residual empty append requires 2D node_r");
      TENRYU_ASSERT(state.x_z.size() == n_nodes,
                    "difference residual empty append requires 2D node_z");
      difference_append_empty_particles_2d_kernel<<<grid_bins, kBlock>>>(pool_view,
                                                                         view.target_U,
                                                                         view.empty_flag,
                                                                         view.empty_offset,
                                                                         state.x_r.data(),
                                                                         state.x_z.data(),
                                                                         n_bins,
                                                                         n_groups,
                                                                         n_out,
                                                                         empty_gid_base,
                                                                         user_seed,
                                                                         step_number,
                                                                         dt,
                                                                         n_cells,
                                                                         nr,
                                                                         nz);
    } else {
      TENRYU_ASSERT(false, "difference residual empty append requires 1D_SPH or 2D_RZ");
    }
    cuda_check(cudaGetLastError(),
               "difference residualization append empty kernel launch failed");
    pool.n_alive = required_after_empty;
    pool.n_census = required_after_empty;
  }

  stats.n_after = pool.n_alive;
  pool.n_census = pool.n_alive;
  return stats;
}

}  // namespace tenryu::radiation
