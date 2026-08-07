#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::pole_angular_coarsen {

struct Macro {
  int block_id = -1;
  int representative_cell = -1;
  int local_i_begin = -1;
  int local_i_end = -1;
  int local_j_begin = -1;
  int local_j_end = -1;
  int level = 0;
  int span = 0;
  int skipped_nodes = 0;
  bool single_apex_boundary = false;
  int canonical_apex_node = -1;
  std::vector<int> boundary_nodes_ordered;
};

struct Overlay {
  bool active = false;
  bool skip_fine_child_paths = true;
  bool supports_deref_macro_repair = false;
  int block_id = -1;
  int representative_cell = -1;
  int q_min = -1;
  int q_max = -1;
  int level_max = 0;
  int n_i_cells = 0;
  int n_j_cells = 0;
  int cell_begin = -1;
  int owned_node_begin = -1;
  std::vector<std::uint8_t> pole_coarsen_inactive_fine_mask;
  std::vector<Macro> macros;
};

struct MotionTaperRow {
  int q = -1;
  double alpha = 0.0;
};

struct MotionPilotResult {
  bool active = false;
  bool applied = false;
  int macro_count = 0;
  int nodes_overwritten = 0;
  int transition_nodes = 0;
  int transition_i = -1;
  int transition_i_min = -1;
  int transition_i_max = -1;
  int transition_rows = 0;
  int q_free = -1;
  int q_full = -1;
  const char* profile = "";
  std::vector<MotionTaperRow> taper_rows;
};

bool pilot_enabled();
bool motion_pilot_enabled();
Overlay build_overlay(const core::State& state, const core::Config& cfg);
Overlay build_motion_overlay(const core::State& state, const core::Config& cfg);
int span_for_level(int level);
Macro make_macro(const Overlay& overlay, int j_begin, int j_end);
void mark_macro_cells(const Overlay& overlay,
                      const Macro& macro,
                      std::vector<std::uint8_t>& mask);
MotionPilotResult apply_coherent_motion_host(
    const Overlay& accepted,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    std::vector<double>& pos_v_r,
    std::vector<double>& pos_v_z,
    double dt);

}  // namespace tenryu::hydro::pole_angular_coarsen
