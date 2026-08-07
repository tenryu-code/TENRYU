#include "radiation/ddmc_transport_gpu.cuh"

#include <cmath>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_event.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaTotFloor = 1.0e-30;
constexpr double kLeakTol = 1.0e-14;
constexpr double kRngEps = 1.0e-16;  // Clamp for uniform RNG to avoid log(0) and log(1)
constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kMaxEventsDdmc = 100000;

constexpr std::uint8_t kBcInternal =
    static_cast<std::uint8_t>(DDMCBoundaryType::Internal);
constexpr std::uint8_t kBcVacuum =
    static_cast<std::uint8_t>(DDMCBoundaryType::Vacuum);
constexpr std::uint8_t kBcReflective =
    static_cast<std::uint8_t>(DDMCBoundaryType::Reflective);
constexpr std::uint8_t kBcInterface =
    static_cast<std::uint8_t>(DDMCBoundaryType::Interface);
constexpr std::uint8_t kInterfaceExitCosine = 0U;
constexpr std::uint8_t kInterfaceExitHalfIsotropic = 1U;

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

__device__ inline double sample_ddmc_event_time_device(const double sigma_tot,
                                                       const double xi) {
  if (sigma_tot <= 0.0) {
    return 1.0e300;  // Infinity surrogate when sigma_tot <= 0 (matches CPU ddmc_event.hpp)
  }
  return -log(fmin(fmax(xi, kRngEps), 1.0 - kRngEps)) /
         (core::constants::c_light * sigma_tot);
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

__device__ inline void sample_volume_uniform_position_1d_device(
    const double r_lo,
    const double r_hi,
    curandStatePhilox4_32_10_t* rng,
    std::uint32_t* counter,
    double* pos_r,
    double* pos_z) {
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  const double xi_r = rng_uniform(rng, counter);
  if (r_hi > r_lo) {
    *pos_r = fmax(cbrt(r_lo3 + xi_r * (r_hi3 - r_lo3)), 0.0);
  } else {
    *pos_r = fmax(0.5 * (r_lo + r_hi), 0.0);
  }
  *pos_z = 0.0;
}

__device__ inline void set_cell_center_position_1d_device(const double r_lo,
                                                          const double r_hi,
                                                          double* pos_r,
                                                          double* pos_z) {
  *pos_r = fmax(0.5 * (r_lo + r_hi), 0.0);
  *pos_z = 0.0;
}

__global__ __launch_bounds__(128, 16) void ddmc_event_loop(
    double* __restrict__ pos_r_arr,
    double* __restrict__ pos_z_arr,
    double* __restrict__ dir_r_arr,
    double* __restrict__ dir_z_arr,
    double* __restrict__ dir_phi_arr,
    double* __restrict__ energy_arr,
    double* __restrict__ time_remain_arr,
    const std::int8_t* __restrict__ sign_arr,
    const std::uint64_t* __restrict__ global_id_arr,
    std::uint32_t* __restrict__ rng_counter_arr,
    std::int32_t* __restrict__ cell_id_arr,
    std::uint16_t* __restrict__ group_id_arr,
    std::uint8_t* __restrict__ mode_arr,
    std::uint8_t* __restrict__ alive_arr,
    const double* __restrict__ sigma_a_eff,
    const double* __restrict__ sigma_s_eff,
    const double* __restrict__ sigma_leak_left,
    const double* __restrict__ sigma_leak_right,
    const std::uint8_t* __restrict__ bc_left,
    const std::uint8_t* __restrict__ bc_right,
    const double* __restrict__ eta_cdf,
    const TransportMode* __restrict__ ddmc_mode,
    const double* __restrict__ node_r,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_E_tally,
    double* __restrict__ E_escape,
    double* __restrict__ E_numerical_loss,
    unsigned long long* __restrict__ ddmc_absorbed,
    unsigned long long* __restrict__ ddmc_census,
    unsigned long long* __restrict__ ddmc_leak_left,
    unsigned long long* __restrict__ ddmc_leak_right,
    unsigned long long* __restrict__ ddmc_leak_boundary,
    unsigned long long* __restrict__ ddmc_vacuum_leak_left,
    unsigned long long* __restrict__ ddmc_vacuum_leak_right,
    unsigned long long* __restrict__ ddmc_converted_to_imc,
    unsigned long long* __restrict__ ddmc_converted_to_rw,
    unsigned long long* __restrict__ ddmc_sigma_tot_zero,
    unsigned long long* __restrict__ ddmc_max_events_reached,
    const int n_cells,
    const int n_groups,
    const int ghost_layers,
    const int nr_local,
    const bool has_left_boundary,
    const bool has_right_boundary,
    const int n_ddmc,
    const int ddmc_start,
    const int pool_capacity,
    const std::uint8_t interface_exit_distribution,
    const double dt,
    const std::uint64_t step_number,
    const std::uint64_t user_seed,
    core::DeviceErrorFlags* __restrict__ error_flags) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_ddmc) {
    return;
  }

  const int p = ddmc_start + tid;
  if (p < 0 || p >= pool_capacity) {
    return;
  }
  std::uint8_t alive_l = alive_arr[p];
  std::uint8_t mode_l = mode_arr[p];
  if (alive_l != kAlive || mode_l != kModeDDMC) {
    return;
  }

  std::int32_t cell_l = cell_id_arr[p];
  int group_l = static_cast<int>(group_id_arr[p]);
  double E_l = energy_arr[p];
  double t_remain_l = time_remain_arr[p];
  const std::int8_t sign_l = sign_arr[p];
  const double sign_d = static_cast<double>(sign_l);
  if (!isfinite(E_l) || !isfinite(t_remain_l)) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (isfinite(E_l) && E_l > 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * E_l);
    }
    energy_arr[p] = 0.0;
    time_remain_arr[p] = 0.0;
    alive_arr[p] = kDead;
    return;
  }
  if (E_l <= 0.0) {
    if (E_l < 0.0 && E_numerical_loss != nullptr && isfinite(E_l)) {
      atomic_add_double(E_numerical_loss, sign_d * (-E_l));
    }
    energy_arr[p] = 0.0;
    time_remain_arr[p] = 0.0;
    alive_arr[p] = kDead;
    return;
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
  bool preserved_census = false;
  while (alive_l == kAlive && mode_l == kModeDDMC &&
         t_remain_l > 0.0 && events < kMaxEventsDdmc) {
    ++events;

    if (!isfinite(E_l) || !isfinite(t_remain_l)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      if (isfinite(E_l) && E_l > 0.0 && E_numerical_loss != nullptr) {
        atomic_add_double(E_numerical_loss, sign_d * E_l);
      }
      E_l = 0.0;
      t_remain_l = 0.0;
      alive_l = kDead;
      break;
    }
    if (E_l <= 0.0) {
      if (E_l < 0.0 && E_numerical_loss != nullptr && isfinite(E_l)) {
        atomic_add_double(E_numerical_loss, sign_d * (-E_l));
      }
      E_l = 0.0;
      alive_l = kDead;
      break;
    }

    if (cell_l < 0 || cell_l >= n_cells ||
        group_l < 0 || group_l >= n_groups) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (E_l > 0.0 && E_numerical_loss != nullptr) {
        atomic_add_double(E_numerical_loss, sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      break;
    }

    const int key = cell_l * n_groups + group_l;
    const double sigma_a = fmax(sigma_a_eff[key], 0.0);
    const double sigma_s =
        (sigma_s_eff != nullptr && eta_cdf != nullptr) ? fmax(sigma_s_eff[key], 0.0) : 0.0;
    const double sigma_leak_left_raw = sigma_leak_left[key];
    const double sigma_leak_right_raw = sigma_leak_right[key];
    if (sigma_leak_left_raw < -kLeakTol || sigma_leak_right_raw < -kLeakTol) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }
    const std::uint8_t left_bc = bc_left[key];
    const std::uint8_t right_bc = bc_right[key];

    const bool left_is_vacuum = (left_bc == kBcVacuum);
    const bool right_is_vacuum = (right_bc == kBcVacuum);
    const double sigma_leak_left_vacuum =
        left_is_vacuum ? fmax(sigma_leak_left_raw, 0.0) : 0.0;
    const double sigma_leak_right_vacuum =
        right_is_vacuum ? fmax(sigma_leak_right_raw, 0.0) : 0.0;
    const double sigma_leak_left_out =
        left_is_vacuum ? 0.0 : fmax(sigma_leak_left_raw, 0.0);
    const double sigma_leak_right_out =
        right_is_vacuum ? 0.0 : fmax(sigma_leak_right_raw, 0.0);
    const double sigma_leak_bnd = sigma_leak_left_vacuum + sigma_leak_right_vacuum;
    // GPU transport keeps a single LeakBoundary channel for vacuum faces for
    // physics/event sampling; side-resolved vacuum leak diagnostics are tracked
    // separately without changing energy accounting.
    const double sigma_tot =
        sigma_a + sigma_s + sigma_leak_left_out + sigma_leak_right_out + sigma_leak_bnd;

    if (sigma_tot <= kSigmaTotFloor) {
      atomic_add_double(&rad_E_tally[key],
                        sign_d * core::constants::c_light * E_l * t_remain_l);
      t_remain_l = 0.0;
      atomic_inc_counter(ddmc_sigma_tot_zero);
      if (error_flags != nullptr) {
        atomicExch(&error_flags->ddmc_sigma_tot_zero, 1);
      }
      break;
    }

    const double dt_evt =
        sample_ddmc_event_time_device(sigma_tot, rng_uniform(&rng, &rng_counter_l));
    const double dt_res = fmin(dt_evt, t_remain_l);
    atomic_add_double(&rad_E_tally[key],
                      sign_d * core::constants::c_light * E_l * dt_res);

    if (dt_evt >= t_remain_l) {
      t_remain_l = 0.0;
      atomic_inc_counter(ddmc_census);
      preserved_census = true;
      break;
    }

    t_remain_l -= dt_evt;

    const double xi_event = rng_uniform(&rng, &rng_counter_l);
    const DDMCEvent event = select_ddmc_event(
        sigma_a,
        sigma_s,
        sigma_leak_left_out,
        sigma_leak_right_out,
        sigma_leak_bnd,
        sigma_tot,
        xi_event);

    if (event == DDMCEvent::Absorb) {
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    if (event == DDMCEvent::Scatter) {
      const int g_new = sample_group_from_cdf(
          eta_cdf + cell_l * n_groups,
          n_groups,
          rng_uniform(&rng, &rng_counter_l));
      if (ddmc_mode != nullptr) {
        const TransportMode target_mode = ddmc_mode[cell_l * n_groups + g_new];
        if (target_mode == TransportMode::RW) {
          mode_l = kModeRW;
          group_l = g_new;
          atomic_inc_counter(ddmc_converted_to_rw);
          set_cell_center_position_1d_device(
              node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
          sample_isotropic_direction_1d_device(
              &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
          break;
        }
        if (target_mode != TransportMode::DDMC) {
          mode_l = kModeIMC;
          group_l = g_new;
          atomic_inc_counter(ddmc_converted_to_imc);
          sample_volume_uniform_position_1d_device(node_r[cell_l],
                                                   node_r[cell_l + 1],
                                                   &rng,
                                                   &rng_counter_l,
                                                   &pos_r_arr[p],
                                                   &pos_z_arr[p]);
          sample_isotropic_direction_1d_device(
              &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
          break;
        }
      }
      group_l = g_new;
      continue;
    }

    if (event == DDMCEvent::LeakLeft) {
      if (left_bc == kBcInternal) {
        atomic_inc_counter(ddmc_leak_left);
        cell_l -= 1;
        if (ghost_layers > 0 && nr_local > 0 && cell_l < ghost_layers) {
          if (has_left_boundary) {
            // Safety: physical boundary reached via internal BC - should be
            // unreachable (sigma_leak=0 at reflective). Absorb as fallback.
            atomic_add_double(&rad_dep[key], sign_d * E_l);
            E_l = 0.0;
            alive_l = kDead;
            atomic_inc_counter(ddmc_absorbed);
            break;
          }
          cell_l = -(100 + 0);
          alive_l = kAlive;
          break;
        }
        continue;
      }
      if (left_bc == kBcInterface) {
        atomic_inc_counter(ddmc_leak_left);
        cell_l -= 1;
        if (cell_l >= 0 && cell_l < n_cells) {
          const TransportMode target_mode =
              (ddmc_mode != nullptr)
                  ? ddmc_mode[cell_l * n_groups + group_l]
                  : TransportMode::IMC;
          if (target_mode == TransportMode::RW) {
            mode_l = kModeRW;
            atomic_inc_counter(ddmc_converted_to_rw);
            set_cell_center_position_1d_device(
                node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
            sample_isotropic_direction_1d_device(
                &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
          } else {
            mode_l = kModeIMC;
            atomic_inc_counter(ddmc_converted_to_imc);
            const double xi_mu = rng_uniform(&rng, &rng_counter_l);
            const double mu =
                (interface_exit_distribution == kInterfaceExitHalfIsotropic)
                    ? fmax(fmin(xi_mu, 1.0), 0.0)
                    : sqrt(fmax(xi_mu, 0.0));
            const double xi_phi = rng_uniform(&rng, &rng_counter_l);
            const double phi = kTwoPi * xi_phi;
            const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
            pos_r_arr[p] = node_r[cell_l + 1];
            pos_z_arr[p] = 0.0;
            dir_r_arr[p] = -mu;
            dir_z_arr[p] = sin_theta * cos(phi);
            dir_phi_arr[p] = sin_theta * sin(phi);
          }
        } else {
          if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
            atomic_add_double(&E_escape[group_l], sign_d * E_l);
          }
          E_l = 0.0;
          alive_l = kDead;
          atomic_inc_counter(ddmc_leak_boundary);
        }
        break;
      }
      if (left_bc == kBcVacuum) {
        if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
          atomic_add_double(&E_escape[group_l], sign_d * E_l);
        }
        E_l = 0.0;
        alive_l = kDead;
        atomic_inc_counter(ddmc_vacuum_leak_left);
        atomic_inc_counter(ddmc_leak_boundary);
        break;
      }
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    if (event == DDMCEvent::LeakRight) {
      if (right_bc == kBcInternal) {
        atomic_inc_counter(ddmc_leak_right);
        cell_l += 1;
        if (ghost_layers > 0 && nr_local > 0 && cell_l >= (ghost_layers + nr_local)) {
          if (has_right_boundary) {
            // Safety: physical boundary reached via internal BC - should be
            // unreachable (sigma_leak=0 at reflective). Absorb as fallback.
            atomic_add_double(&rad_dep[key], sign_d * E_l);
            E_l = 0.0;
            alive_l = kDead;
            atomic_inc_counter(ddmc_absorbed);
            break;
          }
          cell_l = -(100 + 1);
          alive_l = kAlive;
          break;
        }
        continue;
      }
      if (right_bc == kBcInterface) {
        atomic_inc_counter(ddmc_leak_right);
        cell_l += 1;
        if (cell_l >= 0 && cell_l < n_cells) {
          const TransportMode target_mode =
              (ddmc_mode != nullptr)
                  ? ddmc_mode[cell_l * n_groups + group_l]
                  : TransportMode::IMC;
          if (target_mode == TransportMode::RW) {
            mode_l = kModeRW;
            atomic_inc_counter(ddmc_converted_to_rw);
            set_cell_center_position_1d_device(
                node_r[cell_l], node_r[cell_l + 1], &pos_r_arr[p], &pos_z_arr[p]);
            sample_isotropic_direction_1d_device(
                &rng, &rng_counter_l, &dir_r_arr[p], &dir_z_arr[p], &dir_phi_arr[p]);
          } else {
            mode_l = kModeIMC;
            atomic_inc_counter(ddmc_converted_to_imc);
            const double xi_mu = rng_uniform(&rng, &rng_counter_l);
            const double mu =
                (interface_exit_distribution == kInterfaceExitHalfIsotropic)
                    ? fmax(fmin(xi_mu, 1.0), 0.0)
                    : sqrt(fmax(xi_mu, 0.0));
            const double xi_phi = rng_uniform(&rng, &rng_counter_l);
            const double phi = kTwoPi * xi_phi;
            const double sin_theta = sqrt(fmax(0.0, 1.0 - mu * mu));
            pos_r_arr[p] = node_r[cell_l];
            pos_z_arr[p] = 0.0;
            dir_r_arr[p] = mu;
            dir_z_arr[p] = sin_theta * cos(phi);
            dir_phi_arr[p] = sin_theta * sin(phi);
          }
        } else {
          if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
            atomic_add_double(&E_escape[group_l], sign_d * E_l);
          }
          E_l = 0.0;
          alive_l = kDead;
          atomic_inc_counter(ddmc_leak_boundary);
        }
        break;
      }
      if (right_bc == kBcVacuum) {
        if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
          atomic_add_double(&E_escape[group_l], sign_d * E_l);
        }
        E_l = 0.0;
        alive_l = kDead;
        atomic_inc_counter(ddmc_vacuum_leak_right);
        atomic_inc_counter(ddmc_leak_boundary);
        break;
      }
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_boundary, 1);
      }
      atomic_add_double(&rad_dep[key], sign_d * E_l);
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_absorbed);
      break;
    }

    if (event == DDMCEvent::LeakBoundary) {
      if (left_is_vacuum || right_is_vacuum) {
        if (left_is_vacuum && !right_is_vacuum) {
          atomic_inc_counter(ddmc_vacuum_leak_left);
        } else if (!left_is_vacuum && right_is_vacuum) {
          atomic_inc_counter(ddmc_vacuum_leak_right);
        } else {
          const double threshold =
              fmin(fmax(xi_event, 0.0), 1.0 - 1.0e-12) * sigma_tot;
          const double cdf_boundary_start =
              fmax(sigma_a, 0.0) + fmax(sigma_leak_left_out, 0.0) +
              fmax(sigma_leak_right_out, 0.0);
          const double boundary_offset = fmax(threshold - cdf_boundary_start, 0.0);
          if (boundary_offset < sigma_leak_left_vacuum) {
            atomic_inc_counter(ddmc_vacuum_leak_left);
          } else {
            atomic_inc_counter(ddmc_vacuum_leak_right);
          }
        }
      }
      if (group_l >= 0 && group_l < n_groups && E_l > 0.0) {
        atomic_add_double(&E_escape[group_l], sign_d * E_l);
      }
      E_l = 0.0;
      alive_l = kDead;
      atomic_inc_counter(ddmc_leak_boundary);
      break;
    }

    if (event == DDMCEvent::Census) {
      t_remain_l = 0.0;
      atomic_inc_counter(ddmc_census);
      preserved_census = true;
      break;
    }
  }

  if (events >= kMaxEventsDdmc && alive_l == kAlive &&
      mode_l == kModeDDMC && !preserved_census) {
    if (cell_l >= 0 && cell_l < n_cells &&
        group_l >= 0 && group_l < n_groups) {
      const int key = cell_l * n_groups + group_l;
      atomic_add_double(&rad_dep[key], sign_d * E_l);
    } else if (E_l > 0.0 && E_numerical_loss != nullptr) {
      atomic_add_double(E_numerical_loss, sign_d * E_l);
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
    }
    E_l = 0.0;
    alive_l = kDead;
    atomic_inc_counter(ddmc_max_events_reached);
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
}

}  // namespace

void ddmc_transport_gpu_cuda(const DDMCTransportGPUInputs& in) {
  TENRYU_ASSERT(in.pool != nullptr, "ddmc_transport_gpu requires pool");
  TENRYU_ASSERT(in.sigma_a_eff != nullptr, "ddmc_transport_gpu requires sigma_a_eff");
  TENRYU_ASSERT(in.sigma_leak_left != nullptr,
                "ddmc_transport_gpu requires sigma_leak_left");
  TENRYU_ASSERT(in.sigma_leak_right != nullptr,
                "ddmc_transport_gpu requires sigma_leak_right");
  TENRYU_ASSERT(in.bc_left != nullptr, "ddmc_transport_gpu requires bc_left");
  TENRYU_ASSERT(in.bc_right != nullptr, "ddmc_transport_gpu requires bc_right");
  TENRYU_ASSERT(in.node_r != nullptr, "ddmc_transport_gpu requires node_r");
  TENRYU_ASSERT(in.rad_dep != nullptr, "ddmc_transport_gpu requires rad_dep");
  TENRYU_ASSERT(in.rad_E_tally != nullptr, "ddmc_transport_gpu requires rad_E_tally");
  TENRYU_ASSERT(in.E_escape != nullptr, "ddmc_transport_gpu requires E_escape");
  TENRYU_ASSERT(in.E_numerical_loss != nullptr,
                "ddmc_transport_gpu requires E_numerical_loss");
  TENRYU_ASSERT(in.n_cells >= 0, "ddmc_transport_gpu requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 1, "ddmc_transport_gpu requires n_groups >= 1");
  TENRYU_ASSERT(in.n_ddmc >= 0, "ddmc_transport_gpu requires n_ddmc >= 0");
  TENRYU_ASSERT(in.ddmc_start >= 0, "ddmc_transport_gpu requires ddmc_start >= 0");
  TENRYU_ASSERT(in.interface_exit_distribution == kInterfaceExitCosine ||
                    in.interface_exit_distribution == kInterfaceExitHalfIsotropic,
                "ddmc_transport_gpu requires interface_exit_distribution in {0,1}");
  const auto ddmc_end =
      static_cast<long long>(in.ddmc_start) + static_cast<long long>(in.n_ddmc);
  TENRYU_ASSERT(ddmc_end <= static_cast<long long>(in.pool->capacity),
                "DDMC slice out of bounds: start=" + std::to_string(in.ddmc_start) +
                    " n=" + std::to_string(in.n_ddmc) +
                    " capacity=" + std::to_string(in.pool->capacity));

  if (in.n_ddmc == 0) {
    return;
  }

  TENRYU_ASSERT(in.pool->pos_r != nullptr, "ddmc_transport_gpu requires pool->pos_r");
  TENRYU_ASSERT(in.pool->pos_z != nullptr, "ddmc_transport_gpu requires pool->pos_z");
  TENRYU_ASSERT(in.pool->dir_r != nullptr, "ddmc_transport_gpu requires pool->dir_r");
  TENRYU_ASSERT(in.pool->dir_z != nullptr, "ddmc_transport_gpu requires pool->dir_z");
  TENRYU_ASSERT(in.pool->dir_phi != nullptr, "ddmc_transport_gpu requires pool->dir_phi");
  TENRYU_ASSERT(in.pool->energy != nullptr, "ddmc_transport_gpu requires pool->energy");
  TENRYU_ASSERT(in.pool->time_remain != nullptr,
                "ddmc_transport_gpu requires pool->time_remain");
  TENRYU_ASSERT(in.pool->sign != nullptr, "ddmc_transport_gpu requires pool->sign");
  TENRYU_ASSERT(in.pool->global_id != nullptr, "ddmc_transport_gpu requires pool->global_id");
  TENRYU_ASSERT(in.pool->rng_counter != nullptr,
                "ddmc_transport_gpu requires pool->rng_counter");
  TENRYU_ASSERT(in.pool->cell_id != nullptr, "ddmc_transport_gpu requires pool->cell_id");
  TENRYU_ASSERT(in.pool->group_id != nullptr, "ddmc_transport_gpu requires pool->group_id");
  TENRYU_ASSERT(in.pool->mode != nullptr, "ddmc_transport_gpu requires pool->mode");
  TENRYU_ASSERT(in.pool->alive != nullptr, "ddmc_transport_gpu requires pool->alive");

  constexpr int kBlock = 128;
  const int grid = (in.n_ddmc + kBlock - 1) / kBlock;

  ddmc_event_loop<<<grid, kBlock>>>(in.pool->pos_r,
                                    in.pool->pos_z,
                                    in.pool->dir_r,
                                    in.pool->dir_z,
                                    in.pool->dir_phi,
                                    in.pool->energy,
                                    in.pool->time_remain,
                                    in.pool->sign,
                                    in.pool->global_id,
                                    in.pool->rng_counter,
                                    in.pool->cell_id,
                                    in.pool->group_id,
                                    in.pool->mode,
                                    in.pool->alive,
                                    in.sigma_a_eff,
                                    in.sigma_s_eff,
                                    in.sigma_leak_left,
                                    in.sigma_leak_right,
      in.bc_left,
      in.bc_right,
      in.eta_cdf,
      in.ddmc_mode,
      in.node_r,
                                    in.rad_dep,
                                    in.rad_E_tally,
                                    in.E_escape,
                                    in.E_numerical_loss,
                                    in.ddmc_absorbed,
                                    in.ddmc_census,
                                    in.ddmc_leak_left,
                                    in.ddmc_leak_right,
                                    in.ddmc_leak_boundary,
                                    in.ddmc_vacuum_leak_left,
                                    in.ddmc_vacuum_leak_right,
                                    in.ddmc_converted_to_imc,
                                    in.ddmc_converted_to_rw,
                                    in.ddmc_sigma_tot_zero,
                                    in.ddmc_max_events_reached,
                                    in.n_cells,
                                    in.n_groups,
                                    in.ghost_layers,
                                    in.nr_local,
                                    in.has_left_boundary,
                                    in.has_right_boundary,
                                    in.n_ddmc,
                                    in.ddmc_start,
                                    in.pool->capacity,
                                    in.interface_exit_distribution,
                                    in.dt,
                                    in.step_number,
                                    in.user_seed,
                                    in.error_flags);

  cuda_check(cudaGetLastError(), "ddmc_transport_gpu kernel launch failed");
  cuda_check(cudaDeviceSynchronize(), "ddmc_transport_gpu kernel execution failed");
}

}  // namespace tenryu::radiation
