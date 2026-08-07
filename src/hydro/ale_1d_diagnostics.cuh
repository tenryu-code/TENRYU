#pragma once

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_remap.cuh"
#include "hydro/ale_1d_types.cuh"
#include "hydro/ale_1d_velocity_project.cuh"

namespace tenryu::hydro::ale1d {

struct Ale1dDiagnosticsResult {
  double mass_conservation_rel_err = 0.0;
  double material_internal_energy_rel_err = 0.0;
  double material_internal_energy_component_rel_err = 0.0;
  double global_total_energy_rel_err = 0.0;
  double radiation_conservation_rel_err = 0.0;
  double kinetic_energy_drift_rel = 0.0;
  bool hard_tolerance_passed = true;
};

Ale1dDiagnosticsResult compute_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const Ale1dRemapScratch& remap_scratch,
    const Ale1dRemapResult& remap_result,
    const Ale1dVelocityProjectResult& velocity_result);

}  // namespace tenryu::hydro::ale1d
