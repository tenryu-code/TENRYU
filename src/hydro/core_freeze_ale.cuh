#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::ale::core_freeze {

struct CoreFreezeDiagnostics {
  bool enabled = false;
  int frozen_cell_count = 0;
  int frozen_node_count = 0;
  int velocity_projection_skipped_node_count = 0;
  double max_frozen_node_displacement = 0.0;
  double frozen_cell_swept_volume_abs_sum = 0.0;
};

CoreFreezeDiagnostics restore_target_if_enabled(
    core::State& state,
    const core::Config& cfg,
    double* d_target_r,
    double* d_target_z,
    const double* d_current_r,
    const double* d_current_z,
    bool for_axis_rezone,
    const char* path_tag,
    core::DeviceArray<std::uint8_t>* d_frozen_nodes_out = nullptr);

}  // namespace tenryu::hydro::ale::core_freeze
