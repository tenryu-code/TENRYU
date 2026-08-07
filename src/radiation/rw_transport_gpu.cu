#include "radiation/rw_transport_gpu.cuh"

#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaFloor = 1.0e-30;
constexpr double kRngEps = 1.0e-16;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kMaxEventsRw = 100000;

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

__device__ inline void atomic_inc_counter(unsigned long long* counter) {
  if (counter != nullptr) {
    atomicAdd(counter, 1ULL);
  }
}

__device__ inline double rng_uniform(curandStatePhilox4_32_10_t* rng,
                                     std::uint32_t* counter) {
  *counter += 1U;
  const double xi = curand_uniform_double(rng);
  return fmin(fmax(xi, kRngEps), 1.0 - kRngEps);
}

__device__ inline void sample_isotropic_direction_1d_device(
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    double* dir_r,
    double* dir_z,
    double* dir_phi) {
  const double mu = 2.0 * rng_uniform(rng, counter) - 1.0;
  const double phi = kTwoPi * rng_uniform(rng, counter);
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  *dir_r = mu;
  *dir_z = sin_theta * cos(phi);
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline void sample_half_space_direction_1d_device(
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    const double outward_sign,
    double* dir_r,
    double* dir_z,
    double* dir_phi) {
  const double mu_mag = rng_uniform(rng, counter);
  const double mu = outward_sign * fmax(fmin(mu_mag, 1.0), 0.0);
  const double phi = kTwoPi * rng_uniform(rng, counter);
  const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
  *dir_r = mu;
  *dir_z = sin_theta * cos(phi);
  *dir_phi = sin_theta * sin(phi);
}

__device__ inline void set_cell_center_position_1d(const double r_lo,
                                                   const double r_hi,
                                                   double* pos_r,
                                                   double* pos_z) {
  *pos_r = fmax(0.5 * (r_lo + r_hi), 0.0);
  *pos_z = 0.0;
}

__device__ inline double integrate_rw_residence(const double energy, const double dt_seg) {
  if (!(dt_seg > 0.0) || !(energy > 0.0)) {
    return 0.0;
  }
  return core::constants::c_light * energy * dt_seg;
}

__global__ __launch_bounds__(128, 16) void rw_event_loop(
    double* __restrict__ pos_r_arr,
    double* __restrict__ pos_z_arr,
    double* __restrict__ dir_r_arr,
    double* __restrict__ dir_z_arr,
    double* __restrict__ dir_phi_arr,
    double* __restrict__ energy_arr,
    double* __restrict__ weight_arr,
    const double* __restrict__ birth_energy_arr,
    double* __restrict__ time_remain_arr,
    const std::int8_t* __restrict__ sign_arr,
    const std::uint64_t* __restrict__ global_id_arr,
    std::uint32_t* __restrict__ rng_counter_arr,
    std::int32_t* __restrict__ cell_id_arr,
    std::uint16_t* __restrict__ group_id_arr,
    std::uint8_t* __restrict__ mode_arr,
    std::uint8_t* __restrict__ alive_arr,
    const double* __restrict__ sigma_R,
    const TransportMode* __restrict__ mode_map,
    const double* __restrict__ node_r,
    double* __restrict__ rad_E_tally,
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,
    unsigned long long* __restrict__ rw_census,
    unsigned long long* __restrict__ rw_leak_left,
    unsigned long long* __restrict__ rw_leak_right,
    unsigned long long* __restrict__ rw_escaped,
    unsigned long long* __restrict__ rw_converted_to_imc,
    unsigned long long* __restrict__ rw_converted_to_ddmc,
    const int n_cells,
    const int n_groups,
    const int n_rw,
    const int rw_start,
    const int pool_capacity,
    const int bc_inner,
    const int bc_outer,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    core::DeviceErrorFlags* __restrict__ error_flags) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_rw) {
    return;
  }

  const int p = rw_start + tid;
  if (p < 0 || p >= pool_capacity) {
    return;
  }
  std::uint8_t alive_l = alive_arr[p];
  std::uint8_t mode_l = mode_arr[p];
  if (alive_l != kAlive || mode_l != kModeRW) {
    return;
  }

  std::int32_t cell_l = cell_id_arr[p];
  int group_l = static_cast<int>(group_id_arr[p]);
  double E_l = energy_arr[p];
  double t_remain_l = time_remain_arr[p];
  const std::int8_t sign_l = sign_arr[p];
  const double sign_d = static_cast<double>(sign_l);
  if (!isfinite(E_l) || E_l < 0.0) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (isfinite(E_l) && E_l != 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * fabs(E_l));
    }
    E_l = 0.0;
    alive_l = kDead;
    t_remain_l = 0.0;
  }
  if (t_remain_l <= 0.0) {
    t_remain_l = dt;
  }

  std::uint32_t rng_counter_l = rng_counter_arr[p];
  curandStatePhilox4_32_10_t rng;
  curand_init(global_id_arr[p] ^ user_seed,
              static_cast<unsigned long long>(step_number),
              static_cast<unsigned long long>(rng_counter_l),
              &rng);

  int events = 0;
  while (alive_l == kAlive && mode_l == kModeRW && t_remain_l > 0.0 && events < kMaxEventsRw) {
    ++events;

    if (cell_l < 0 || cell_l >= n_cells || group_l < 0 || group_l >= n_groups) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (E_l > 0.0 && E_numerical_loss != nullptr) {
        atomic_add_double(E_numerical_loss, sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      t_remain_l = 0.0;
      break;
    }
    if (!(E_l > 0.0)) {
      E_l = 0.0;
      alive_l = kDead;
      t_remain_l = 0.0;
      break;
    }

    const int key = cell_l * n_groups + group_l;
    const double sigma_t = fmax(sigma_R[key], 0.0);
    const double dr = fmax(node_r[cell_l + 1] - node_r[cell_l], 0.0);
    const double R0 = 0.5 * dr;
    if (!(sigma_t > kSigmaFloor) || !(R0 > 0.0) || !isfinite(R0)) {
      mode_l = kModeIMC;
      atomic_inc_counter(rw_converted_to_imc);
      set_cell_center_position_1d(node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
      sample_isotropic_direction_1d_device(
          &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
      break;
    }

    const double mean_t_leak = (R0 * R0 * sigma_t) / (2.0 * core::constants::c_light);
    const double t_leak = -mean_t_leak * log(rng_uniform(&rng, &rng_counter_l));
    const double t_remain_before = t_remain_l;
    const bool hits_census = (t_leak >= t_remain_before);
    const double dt_seg = hits_census ? t_remain_before : t_leak;

    const double residence = integrate_rw_residence(E_l, dt_seg);
    if (residence > 0.0) {
      atomic_add_double(&rad_E_tally[key], sign_d * residence);
    }
    t_remain_l -= dt_seg;

    if (hits_census) {
      atomic_inc_counter(rw_census);
      t_remain_l = 0.0;
      set_cell_center_position_1d(node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
      sample_isotropic_direction_1d_device(
          &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
      break;
    }

    const bool leak_left = (rng_uniform(&rng, &rng_counter_l) < 0.5);
    if (leak_left) {
      atomic_inc_counter(rw_leak_left);
      if (cell_l == 0) {
        if (is_escape_boundary(bc_inner)) {
          atomic_add_double(&E_escape[group_l], sign_d * E_l);
          E_l = 0.0;
          alive_l = kDead;
          t_remain_l = 0.0;
          atomic_inc_counter(rw_escaped);
          break;
        }
        set_cell_center_position_1d(
            node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
        sample_isotropic_direction_1d_device(
            &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
        continue;
      }

      const int neighbor = cell_l - 1;
      const TransportMode target_mode =
          (mode_map != nullptr) ? mode_map[neighbor * n_groups + group_l] : TransportMode::IMC;
      if (target_mode == TransportMode::RW) {
        cell_l = neighbor;
        set_cell_center_position_1d(
            node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
        sample_isotropic_direction_1d_device(
            &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
        continue;
      }
      if (target_mode == TransportMode::DDMC) {
        cell_l = neighbor;
        mode_l = kModeDDMC;
        t_remain_l = 0.0;
        const double nan = NAN;
        pos_r_arr[p] = nan;
        pos_z_arr[p] = nan;
        dir_r_arr[p] = nan;
        dir_z_arr[p] = nan;
        dir_phi_arr[p] = nan;
        atomic_inc_counter(rw_converted_to_ddmc);
        break;
      }

      cell_l = neighbor;
      mode_l = kModeIMC;
      pos_r_arr[p] = node_r[cell_l + 1];
      pos_z_arr[p] = 0.0;
      sample_half_space_direction_1d_device(
          &rng, &rng_counter_l, -1.0, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
      atomic_inc_counter(rw_converted_to_imc);
      break;
    }

    atomic_inc_counter(rw_leak_right);
    if (cell_l == n_cells - 1) {
      if (is_escape_boundary(bc_outer)) {
        atomic_add_double(&E_escape[group_l], sign_d * E_l);
        E_l = 0.0;
        alive_l = kDead;
        t_remain_l = 0.0;
        atomic_inc_counter(rw_escaped);
        break;
      }
      set_cell_center_position_1d(node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
      sample_isotropic_direction_1d_device(
          &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
      continue;
    }

    const int neighbor = cell_l + 1;
    const TransportMode target_mode =
        (mode_map != nullptr) ? mode_map[neighbor * n_groups + group_l] : TransportMode::IMC;
    if (target_mode == TransportMode::RW) {
      cell_l = neighbor;
      set_cell_center_position_1d(
          node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
      sample_isotropic_direction_1d_device(
          &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
      continue;
    }
    if (target_mode == TransportMode::DDMC) {
      cell_l = neighbor;
      mode_l = kModeDDMC;
      t_remain_l = 0.0;
      const double nan = NAN;
      pos_r_arr[p] = nan;
      pos_z_arr[p] = nan;
      dir_r_arr[p] = nan;
      dir_z_arr[p] = nan;
      dir_phi_arr[p] = nan;
      atomic_inc_counter(rw_converted_to_ddmc);
      break;
    }

    cell_l = neighbor;
    mode_l = kModeIMC;
    pos_r_arr[p] = node_r[cell_l];
    pos_z_arr[p] = 0.0;
    sample_half_space_direction_1d_device(
        &rng, &rng_counter_l, +1.0, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
    atomic_inc_counter(rw_converted_to_imc);
    break;
  }

  if (events >= kMaxEventsRw && alive_l == kAlive && mode_l == kModeRW) {
    if (E_l > 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * E_l);
    }
    if (error_flags != nullptr && (cell_l < 0 || cell_l >= n_cells || group_l < 0 || group_l >= n_groups)) {
      atomicExch(&error_flags->invalid_cell, 1);
    }
    E_l = 0.0;
    alive_l = kDead;
    t_remain_l = 0.0;
    if (error_flags != nullptr) {
      atomicAdd(&error_flags->infinite_loop, 1);
    }
  }

  energy_arr[p] = E_l;
  time_remain_arr[p] = fmax(t_remain_l, 0.0);
  cell_id_arr[p] = cell_l;
  group_id_arr[p] = static_cast<std::uint16_t>(group_l);
  mode_arr[p] = mode_l;
  alive_arr[p] = alive_l;
  rng_counter_arr[p] = rng_counter_l;

  const double birth = birth_energy_arr[p];
  weight_arr[p] = (birth > 0.0) ? (fmax(E_l, 0.0) / birth) : 0.0;
}

}  // namespace

void rw_transport_1d_gpu_cuda(const RWTransportGPUInputs& in) {
  TENRYU_ASSERT(in.pool != nullptr, "rw_transport_1d requires pool");
  TENRYU_ASSERT(in.sigma_R != nullptr, "rw_transport_1d requires sigma_R");
  TENRYU_ASSERT(in.mode_map != nullptr, "rw_transport_1d requires mode_map");
  TENRYU_ASSERT(in.node_r != nullptr, "rw_transport_1d requires node_r");
  TENRYU_ASSERT(in.rad_E_tally != nullptr, "rw_transport_1d requires rad_E_tally");
  TENRYU_ASSERT(in.E_escape != nullptr, "rw_transport_1d requires E_escape");
  TENRYU_ASSERT(in.E_numerical_loss != nullptr,
                "rw_transport_1d requires E_numerical_loss");
  TENRYU_ASSERT(in.n_cells >= 0, "rw_transport_1d requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 1, "rw_transport_1d requires n_groups >= 1");
  TENRYU_ASSERT(in.n_rw >= 0, "rw_transport_1d requires n_rw >= 0");
  TENRYU_ASSERT(in.rw_start >= 0, "rw_transport_1d requires rw_start >= 0");
  const auto rw_end =
      static_cast<long long>(in.rw_start) + static_cast<long long>(in.n_rw);
  TENRYU_ASSERT(rw_end <= static_cast<long long>(in.pool->capacity),
                "RW slice out of bounds: start=" + std::to_string(in.rw_start) +
                    " n=" + std::to_string(in.n_rw) +
                    " capacity=" + std::to_string(in.pool->capacity));

  if (in.n_rw == 0) {
    return;
  }

  TENRYU_ASSERT(in.pool->pos_r != nullptr, "rw_transport_1d requires pool->pos_r");
  TENRYU_ASSERT(in.pool->pos_z != nullptr, "rw_transport_1d requires pool->pos_z");
  TENRYU_ASSERT(in.pool->dir_r != nullptr, "rw_transport_1d requires pool->dir_r");
  TENRYU_ASSERT(in.pool->dir_z != nullptr, "rw_transport_1d requires pool->dir_z");
  TENRYU_ASSERT(in.pool->dir_phi != nullptr, "rw_transport_1d requires pool->dir_phi");
  TENRYU_ASSERT(in.pool->energy != nullptr, "rw_transport_1d requires pool->energy");
  TENRYU_ASSERT(in.pool->weight != nullptr, "rw_transport_1d requires pool->weight");
  TENRYU_ASSERT(in.pool->birth_energy != nullptr,
                "rw_transport_1d requires pool->birth_energy");
  TENRYU_ASSERT(in.pool->time_remain != nullptr,
                "rw_transport_1d requires pool->time_remain");
  TENRYU_ASSERT(in.pool->sign != nullptr, "rw_transport_1d requires pool->sign");
  TENRYU_ASSERT(in.pool->global_id != nullptr, "rw_transport_1d requires pool->global_id");
  TENRYU_ASSERT(in.pool->rng_counter != nullptr,
                "rw_transport_1d requires pool->rng_counter");
  TENRYU_ASSERT(in.pool->cell_id != nullptr, "rw_transport_1d requires pool->cell_id");
  TENRYU_ASSERT(in.pool->group_id != nullptr, "rw_transport_1d requires pool->group_id");
  TENRYU_ASSERT(in.pool->mode != nullptr, "rw_transport_1d requires pool->mode");
  TENRYU_ASSERT(in.pool->alive != nullptr, "rw_transport_1d requires pool->alive");

  constexpr int kBlock = 128;
  const int grid = (in.n_rw + kBlock - 1) / kBlock;
  rw_event_loop<<<grid, kBlock>>>(in.pool->pos_r,
                                  in.pool->pos_z,
                                  in.pool->dir_r,
                                  in.pool->dir_z,
                                  in.pool->dir_phi,
                                  in.pool->energy,
                                  in.pool->weight,
                                  in.pool->birth_energy,
                                  in.pool->time_remain,
                                  in.pool->sign,
                                  in.pool->global_id,
                                  in.pool->rng_counter,
                                  in.pool->cell_id,
                                  in.pool->group_id,
                                  in.pool->mode,
                                  in.pool->alive,
                                  in.sigma_R,
                                  in.mode_map,
                                  in.node_r,
                                  in.rad_E_tally,
                                  in.E_escape,
                                  in.E_numerical_loss,
                                  in.rw_census,
                                  in.rw_leak_left,
                                  in.rw_leak_right,
                                  in.rw_escaped,
                                  in.rw_converted_to_imc,
                                  in.rw_converted_to_ddmc,
                                  in.n_cells,
                                  in.n_groups,
                                  in.n_rw,
                                  in.rw_start,
                                  in.pool->capacity,
                                  in.bc_inner,
                                  in.bc_outer,
                                  in.dt,
                                  in.step_number,
                                  in.user_seed,
                                  in.error_flags);

  cuda_check(cudaGetLastError(), "rw_transport_1d kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "rw_transport_1d kernel execution failed");
}

}  // namespace tenryu::radiation
