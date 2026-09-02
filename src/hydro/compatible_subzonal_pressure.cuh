#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace tenryu::hydro {

struct SubzonalPressureDtArgmin {
  int cell_id = -1;
  double dt = 0.0;
};

struct HydroEOSContext;

struct SubzonalPressureProjectionDebugBuffers {
  // Pair fields use [2 * cell + {inner, outer}].
  double* rho_axis_raw = nullptr;
  double* rho_partner_raw = nullptr;
  double* rho_projected = nullptr;
  double* merit = nullptr;
  double* x_merit = nullptr;
};

struct SubzonalPressureShadowReplay {
  std::vector<double> axis_source_force_r;
  std::vector<double> axis_source_force_z;
  std::vector<double> offaxis_source_force_r;
  std::vector<double> offaxis_source_force_z;
};

enum class SubzonalMeritMode : int {
  CaramanaAuto = 0,
  Constant = 1,
  Off = 2,
};

__host__ __device__ inline double compatible_subzonal_pressure_pow_int(
    const double base,
    const int power) {
  if (power <= 0) {
    return 1.0;
  }
  double out = 1.0;
  for (int i = 0; i < power; ++i) {
    out *= base;
  }
  return out;
}

__host__ __device__ inline double compatible_subzonal_pressure_merit_factor(
    const double x,
    const SubzonalMeritMode mode,
    const double alpha1,
    const double alpha2,
    const int power,
    const double constant_value) {
  if (mode == SubzonalMeritMode::Off) {
    return 1.0;
  }
  if (mode == SubzonalMeritMode::Constant) {
    return constant_value;
  }
  const double x_clamped = (x > 0.0 && isfinite(x)) ? x : 0.0;
  if (!(alpha2 > 0.0) || x_clamped > 2.0 * alpha2) {
    return compatible_subzonal_pressure_pow_int(alpha1, power);
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double base =
      0.5 * alpha1 * (1.0 - cos(pi * x_clamped / (2.0 * alpha2)));
  return compatible_subzonal_pressure_pow_int(base, power);
}

__host__ __device__ inline void compatible_subzonal_pressure_work_split_fractions(
    const double pe,
    const double pi,
    double* electron_fraction,
    double* ion_fraction) {
  const double p = pe + pi;
  if (isfinite(p) && p >= 1.0e-20) {
    *electron_fraction = pe / p;
    *ion_fraction = pi / p;
    return;
  }
  *electron_fraction = 0.5;
  *ion_fraction = 0.5;
}

double compute_compatible_subzonal_pressure_dt_2d(
    const core::State& state,
    const core::Config& cfg,
    const std::int8_t* d_hydro_active,
    SubzonalPressureDtArgmin* argmin = nullptr);

void update_compatible_subzonal_pressure_observability_2d(
    core::State& state,
    const core::Config& cfg);

void compute_compatible_subzonal_pressure_force_2d(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const std::int8_t* d_hydro_active,
    bool aw_axis_slave_theta0_active = false,
    bool aw_axis_slave_theta_pi_active = false,
    const SubzonalPressureProjectionDebugBuffers* projection_debug = nullptr);

SubzonalPressureShadowReplay
compute_compatible_subzonal_pressure_shadow_replay_2d(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const std::int8_t* d_hydro_active,
    bool aw_axis_slave_theta0_active,
    bool aw_axis_slave_theta_pi_active);

}  // namespace tenryu::hydro
