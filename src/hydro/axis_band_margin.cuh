#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {
struct State;
}

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::hydro::ale {

struct AxisBandMarginResult {
  bool valid = false;
  int K = 0;
  double row_K_margin = 0.0;
  int argmin_j = -1;
  double v_actual_min = 0.0;
  double v_target_min = 0.0;
  double R_K_min = 0.0;
};

AxisBandMarginResult compute_row_K_margin(
    const core::State& state, int K,
    const parallel::Reduction* reduction);

struct AxisBandKSelection {
  int K_chosen = 0;
  bool any_K_healthy = false;
  std::vector<double> per_K_margins;
};

AxisBandKSelection select_axis_band_K(
    const core::State& state, int K_initial, int K_max,
    double margin_threshold,
    const parallel::Reduction* reduction);

}  // namespace tenryu::hydro::ale
