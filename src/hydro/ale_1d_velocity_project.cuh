#pragma once

#include <vector>

#include "core/state.hpp"
#include "hydro/ale_1d_remap.cuh"

namespace tenryu::hydro::ale1d {

struct Ale1dVelocityProjectScratch {
  DeviceArray<double> p_old_cell;
  DeviceArray<double> p_new_cell;
  DeviceArray<double> u_new_cell;
  DeviceArray<double> v_new_node;

  void resize(int n_cells);
  [[nodiscard]] bool size_matches(int n_cells) const;
};

struct Ale1dVelocityProjectResult {
  bool success = false;
  double kinetic_energy_drift_rel = 0.0;
  double kinetic_energy_old = 0.0;
  double kinetic_energy_new = 0.0;
};

Ale1dVelocityProjectResult project_velocity(
    const core::State& state,
    const std::vector<double>& mass_new,
    const std::vector<double>& delta_Y,
    const std::vector<int>& donor,
    const std::vector<double>& phi_face,
    Ale1dVelocityProjectScratch& scratch);

}  // namespace tenryu::hydro::ale1d
