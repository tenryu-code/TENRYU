#pragma once

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

struct LeadingShockState {
  bool valid = false;
  int i0 = -1;
  int i1 = -1;
  double q_sum = 0.0;
  double r_s = 0.0;
  double u_s = 0.0;
  double dr_s = 0.0;
  double shell_mean_div_u = 0.0;
  bool bounce_seen = false;
};

LeadingShockState update_leading_shock_tracker_1d(
    core::State& state,
    const core::Config& cfg,
    const core::CellField1D& q_probe,
    const core::CellField1D& div_u_probe,
    double t,
    double dt);

}  // namespace tenryu::hydro
