#include "hydro/pole_angular_coarsen.cuh"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <string>

#include "core/error.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::pole_angular_coarsen {
namespace {

int polar_shell_block_id(const mesh::MultiBlockTopology& mb) {
  int block_id = -1;
  for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
    if (mb.blocks[static_cast<std::size_t>(b)].role ==
        mesh::BlockRole::POLAR_SHELL) {
      TENRYU_ASSERT(block_id < 0,
                    "pole coarsen pilot found multiple POLAR_SHELL blocks");
      block_id = b;
    }
  }
  return block_id;
}

int shell_cell_id(const mesh::BlockInfo& shell, const int i, const int j) {
  return shell.cell_begin + i * shell.n_j_cells + j;
}

int shell_node_id(const mesh::BlockInfo& shell, const int i, const int j) {
  const int stride = shell.n_j_cells + 1;
  return shell.owned_node_begin + i * stride + j;
}

int shell_node_id(const Overlay& overlay, const int i, const int j) {
  const int stride = overlay.n_j_cells + 1;
  return overlay.owned_node_begin + i * stride + j;
}

int shell_cell_id(const Overlay& overlay, const int i, const int j) {
  return overlay.cell_begin + i * overlay.n_j_cells + j;
}

int env_int(const char* name, const int default_value) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  TENRYU_ASSERT(errno == 0 && end != raw && *end == '\0' &&
                    value >= static_cast<long>(INT_MIN) &&
                    value <= static_cast<long>(INT_MAX),
                std::string("invalid integer environment value for ") + name);
  return static_cast<int>(value);
}

bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

int motion_transition_rows() {
  static const int rows = [] {
    const int value =
        env_int("TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS", 4);
    TENRYU_ASSERT(value >= 0,
                  "pole motion pilot TRANSITION_ROWS must be nonnegative");
    return value;
  }();
  return rows;
}

const std::string& motion_profile() {
  static const std::string profile = [] {
    const char* raw = std::getenv("TENRYU_I1B_POLE_MOTION_PROFILE");
    std::string value =
        raw == nullptr || raw[0] == '\0' ? "smoothstep" : std::string(raw);
    TENRYU_ASSERT(value == "smoothstep",
                  "pole motion pilot PROFILE must be smoothstep");
    return value;
  }();
  return profile;
}

double smoothstep(const double s_in) {
  const double s = std::max(0.0, std::min(1.0, s_in));
  return s * s * (3.0 - 2.0 * s);
}

int motion_q_free(const int q_full, const int transition_rows) {
  const long value = static_cast<long>(q_full) -
                     static_cast<long>(transition_rows) - 1L;
  TENRYU_ASSERT(value >= static_cast<long>(INT_MIN) &&
                    value <= static_cast<long>(INT_MAX),
                "pole motion pilot transition row range overflow");
  return static_cast<int>(value);
}

double motion_alpha(const int q, const int q_full, const int q_free) {
  if (q >= q_full) {
    return 1.0;
  }
  if (q <= q_free) {
    return 0.0;
  }
  const double denom = static_cast<double>(q_full - q_free);
  return smoothstep(static_cast<double>(q - q_free) / denom);
}

Overlay build_overlay_impl(const core::State& state,
                           const core::Config& cfg,
                           const bool enabled,
                           const bool skip_fine_child_paths) {
  Overlay overlay;
  if (enabled && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "pole angular coarsen/motion pilot is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  if (!enabled || !mesh::mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.n_cells <= 0) {
    return overlay;
  }

  const auto& mb = *state.mesh.topo.multiblock;
  const int block_id = polar_shell_block_id(mb);
  if (block_id < 0) {
    return overlay;
  }
  const mesh::BlockInfo& shell = mb.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(shell.role == mesh::BlockRole::POLAR_SHELL,
                "pole coarsen pilot requires POLAR_SHELL block");
  TENRYU_ASSERT(shell.n_i_cells > 0 && shell.n_j_cells > 0,
                "pole coarsen pilot requires non-empty shell block");
  TENRYU_ASSERT(shell.owned_node_count ==
                    (shell.n_i_cells + 1) * (shell.n_j_cells + 1),
                "pole coarsen pilot requires contiguous shell grid nodes");

  const int q_min = env_int("TENRYU_I1B_POLE_COARSEN_Q_MIN", 7);
  const int q_max = env_int("TENRYU_I1B_POLE_COARSEN_Q_MAX", 7);
  const int level_max =
      env_int("TENRYU_I1B_POLE_COARSEN_LEVEL_MAX", 3);
  TENRYU_ASSERT(q_min >= 0 && q_min < shell.n_i_cells &&
                    q_max >= q_min && q_max < shell.n_i_cells,
                "pole coarsen pilot Q_MIN/Q_MAX out of POLAR_SHELL range");
  TENRYU_ASSERT(level_max >= 1 && level_max < 30,
                "pole coarsen pilot LEVEL_MAX must be in [1, 29]");
  TENRYU_ASSERT(span_for_level(level_max) <= shell.n_j_cells,
                "pole coarsen pilot LEVEL_MAX span exceeds POLAR_SHELL angular cells");

  overlay.active = true;
  overlay.skip_fine_child_paths = skip_fine_child_paths;
  overlay.block_id = block_id;
  overlay.q_min = q_min;
  overlay.q_max = q_max;
  overlay.level_max = level_max;
  overlay.n_i_cells = shell.n_i_cells;
  overlay.n_j_cells = shell.n_j_cells;
  overlay.cell_begin = shell.cell_begin;
  overlay.owned_node_begin = shell.owned_node_begin;
  overlay.representative_cell = shell_cell_id(shell, q_min, 0);
  return overlay;
}

}  // namespace

bool pilot_enabled() {
  static const bool enabled = [] {
    return env_flag_enabled("TENRYU_I1B_POLE_COARSEN_PILOT");
  }();
  return enabled;
}

bool motion_pilot_enabled() {
  static const bool enabled = [] {
    return env_flag_enabled("TENRYU_I1B_POLE_MOTION_PILOT");
  }();
  return enabled;
}

int span_for_level(const int level) {
  TENRYU_ASSERT(level >= 1 && level < 30,
                "pole coarsen pilot level must be in [1, 29]");
  return 1 << level;
}

Macro make_macro(const Overlay& overlay, const int j_begin, const int j_end) {
  TENRYU_ASSERT(overlay.active, "pole coarsen macro requires active overlay");
  TENRYU_ASSERT(overlay.q_min >= 0 && overlay.q_max >= overlay.q_min &&
                    overlay.q_max < overlay.n_i_cells,
                "pole coarsen macro requires valid radial band");
  TENRYU_ASSERT(j_begin >= 0 && j_begin < j_end &&
                    j_end <= overlay.n_j_cells,
                "pole coarsen macro angular interval out of range");

  Macro macro;
  macro.block_id = overlay.block_id;
  macro.local_i_begin = overlay.q_min;
  macro.local_i_end = overlay.q_max + 1;
  macro.local_j_begin = j_begin;
  macro.local_j_end = j_end;
  macro.span = j_end - j_begin;
  macro.level = 0;
  for (int level = 1; level <= overlay.level_max; ++level) {
    if (span_for_level(level) == macro.span) {
      macro.level = level;
      break;
    }
  }
  if (macro.level == 0) {
    macro.level = overlay.level_max;
  }
  macro.skipped_nodes = 2 * std::max(0, macro.span - 1);
  macro.representative_cell = shell_cell_id(overlay, overlay.q_min, j_begin);

  const int i0 = overlay.q_min;
  const int i1 = overlay.q_max + 1;
  macro.boundary_nodes_ordered.reserve(
      static_cast<std::size_t>(2 * (i1 - i0 + 1)));
  macro.boundary_nodes_ordered.push_back(shell_node_id(overlay, i0, j_begin));
  macro.boundary_nodes_ordered.push_back(shell_node_id(overlay, i0, j_end));
  for (int i = i0 + 1; i <= i1; ++i) {
    macro.boundary_nodes_ordered.push_back(shell_node_id(overlay, i, j_end));
  }
  macro.boundary_nodes_ordered.push_back(shell_node_id(overlay, i1, j_begin));
  for (int i = i1 - 1; i > i0; --i) {
    macro.boundary_nodes_ordered.push_back(shell_node_id(overlay, i, j_begin));
  }
  return macro;
}

void mark_macro_cells(const Overlay& overlay,
                      const Macro& macro,
                      std::vector<std::uint8_t>& mask) {
  if (mask.empty()) {
    return;
  }
  TENRYU_ASSERT(mask.size() > static_cast<std::size_t>(overlay.cell_begin),
                "pole coarsen mask does not cover shell cells");
  for (int i = macro.local_i_begin; i < macro.local_i_end; ++i) {
    for (int j = macro.local_j_begin; j < macro.local_j_end; ++j) {
      const int cell = shell_cell_id(overlay, i, j);
      TENRYU_ASSERT(cell >= 0 && cell < static_cast<int>(mask.size()),
                    "pole coarsen mask child cell out of range");
      mask[static_cast<std::size_t>(cell)] = 1U;
    }
  }
}

Overlay build_overlay(const core::State& state, const core::Config& cfg) {
  return build_overlay_impl(state, cfg, pilot_enabled(), true);
}

Overlay build_motion_overlay(const core::State& state, const core::Config& cfg) {
  return build_overlay_impl(state, cfg, motion_pilot_enabled(), false);
}

MotionPilotResult apply_coherent_motion_host(
    const Overlay& accepted,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    std::vector<double>& pos_v_r,
    std::vector<double>& pos_v_z,
    const double dt) {
  MotionPilotResult result;
  result.active = motion_pilot_enabled();
  if (!result.active || !accepted.active || accepted.macros.empty()) {
    return result;
  }
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "pole motion pilot requires finite dt > 0");
  TENRYU_ASSERT(x_r_old.size() == x_z_old.size() &&
                    x_r_old.size() == pos_v_r.size() &&
                    x_r_old.size() == pos_v_z.size(),
                "pole motion pilot vector size mismatch");

  std::vector<std::uint8_t> touched(x_r_old.size(), 0U);
  std::vector<std::uint8_t> transition_touched(x_r_old.size(), 0U);
  const int transition_rows = motion_transition_rows();
  const int q_full = accepted.q_min;
  const int q_free = motion_q_free(q_full, transition_rows);
  const int transition_i_min = std::max(0, q_free + 1);
  const int transition_i_max = q_full - 1;
  result.profile = motion_profile().c_str();
  result.transition_rows = transition_rows;
  result.q_free = q_free;
  result.q_full = q_full;
  result.transition_i_min =
      transition_i_min <= transition_i_max ? transition_i_min : -1;
  result.transition_i_max =
      transition_i_min <= transition_i_max ? transition_i_max : -1;
  result.transition_i =
      transition_i_max >= 0 && transition_i_min <= transition_i_max
          ? transition_i_max
          : -1;
  for (int i = transition_i_min; i <= transition_i_max; ++i) {
    const double alpha = motion_alpha(i, q_full, q_free);
    if (alpha > 0.0 && alpha < 1.0) {
      MotionTaperRow row;
      row.q = i;
      row.alpha = alpha;
      result.taper_rows.push_back(row);
    }
  }

  const auto apply_row = [&](const Macro& macro, const int i,
                             const double blend) {
    if (i < 0 || i > accepted.n_i_cells ||
        macro.local_j_begin < 0 ||
        macro.local_j_end <= macro.local_j_begin ||
        macro.local_j_end > accepted.n_j_cells) {
      return;
    }
    const int j0 = macro.local_j_begin;
    const int j1 = macro.local_j_end;
    const int n0 = shell_node_id(accepted, i, j0);
    const int n1 = shell_node_id(accepted, i, j1);
    TENRYU_ASSERT(n0 >= 0 && n1 >= 0 &&
                      n0 < static_cast<int>(x_r_old.size()) &&
                      n1 < static_cast<int>(x_r_old.size()),
                  "pole motion pilot endpoint node out of range");
    const double r0_new =
        x_r_old[static_cast<std::size_t>(n0)] +
        dt * pos_v_r[static_cast<std::size_t>(n0)];
    const double z0_new =
        x_z_old[static_cast<std::size_t>(n0)] +
        dt * pos_v_z[static_cast<std::size_t>(n0)];
    const double r1_new =
        x_r_old[static_cast<std::size_t>(n1)] +
        dt * pos_v_r[static_cast<std::size_t>(n1)];
    const double z1_new =
        x_z_old[static_cast<std::size_t>(n1)] +
        dt * pos_v_z[static_cast<std::size_t>(n1)];
    const double inv_span = 1.0 / static_cast<double>(j1 - j0);
    for (int j = j0; j <= j1; ++j) {
      const int node = shell_node_id(accepted, i, j);
      TENRYU_ASSERT(node >= 0 &&
                        node < static_cast<int>(x_r_old.size()),
                    "pole motion pilot child node out of range");
      const std::size_t idx = static_cast<std::size_t>(node);
      const double frac = static_cast<double>(j - j0) * inv_span;
      const double r_hydro_new = x_r_old[idx] + dt * pos_v_r[idx];
      const double z_hydro_new = x_z_old[idx] + dt * pos_v_z[idx];
      const double r_coherent = (1.0 - frac) * r0_new + frac * r1_new;
      const double z_coherent = (1.0 - frac) * z0_new + frac * z1_new;
      const double r_target =
          r_hydro_new + blend * (r_coherent - r_hydro_new);
      const double z_target =
          z_hydro_new + blend * (z_coherent - z_hydro_new);
      pos_v_r[idx] = (r_target - x_r_old[idx]) / dt;
      pos_v_z[idx] = (z_target - x_z_old[idx]) / dt;
      if (touched[idx] == 0U) {
        touched[idx] = 1U;
        ++result.nodes_overwritten;
      }
      if (blend < 1.0 && transition_touched[idx] == 0U) {
        transition_touched[idx] = 1U;
        ++result.transition_nodes;
      }
    }
  };

  for (const auto& macro : accepted.macros) {
    for (int i = macro.local_i_begin; i <= macro.local_i_end; ++i) {
      apply_row(macro, i, 1.0);
    }
    for (int i = transition_i_min; i <= transition_i_max; ++i) {
      const double alpha = motion_alpha(i, q_full, q_free);
      if (alpha > 0.0 && alpha < 1.0) {
        apply_row(macro, i, alpha);
      }
    }
  }

  result.applied = result.nodes_overwritten > 0;
  result.macro_count = static_cast<int>(accepted.macros.size());
  return result;
}

}  // namespace tenryu::hydro::pole_angular_coarsen
