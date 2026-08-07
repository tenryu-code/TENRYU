#pragma once

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

void advance_radiation_step_fld_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt);

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
