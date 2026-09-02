#pragma once

#include <cstdint>
#include <limits>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::compatible {

struct TensorAvCflArgmin {
  double dt = std::numeric_limits<double>::infinity();
  int cell_id = -1;
  double rho = 0.0;
  double length = 0.0;
  double mu = 0.0;
  double coefficient = 0.25;
};

void launch_compute_tensor_av_2d(
    core::State& state,
    const core::Config& cfg,
    const core::CellField1D& cell_cs,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active,
    bool aw_axis_slave_theta0_active,
    bool aw_axis_slave_theta_pi_active);

void launch_sum_tensor_av_corner_forces_to_nodes_2d(
    double* force_r,
    double* force_z,
    const std::int8_t* hydro_active,
    const core::State& state);

double compute_tensor_av_cfl_dt(
    const core::State& state,
    const core::Config& cfg,
    bool aw_axis_slave_theta0_active,
    bool aw_axis_slave_theta_pi_active,
    TensorAvCflArgmin* argmin = nullptr);

void launch_compute_tensor_av_work_2d(
    core::State& state,
    const core::Config& cfg,
    const double* v_r,
    const double* v_z,
    const std::int8_t* hydro_active);

}  // namespace tenryu::hydro::compatible
