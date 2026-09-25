#pragma once

#include <limits>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct Fld1DDiagnostics {
  int outer_iterations = 0;
  bool converged = false;
  double outer_residual = 0.0;
  double escaped_energy = 0.0;
};

// drive_time_s: time at which time-dependent boundary drives (Marshak T_r(t),
// pulsed flux) are evaluated — the midpoint of the interval this solve
// advances. NaN (default) keeps the historic state.t for callers that step
// from state.t themselves.
void advance_radiation_step_fld_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt,
    double drive_time_s = std::numeric_limits<double>::quiet_NaN());

double fld_compute_max_reduced_flux_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat);

double fld_compute_fv_uniform_residual_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt);

}  // namespace tenryu::radiation
