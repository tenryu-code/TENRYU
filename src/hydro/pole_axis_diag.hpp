#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::pole_axis_diag {

struct Snapshot {
  int step = -1;
  bool valid = false;
  std::vector<double> node_mass;
  std::vector<double> force_r;
  std::vector<double> force_z;
  std::vector<double> accel_r;
  std::vector<double> accel_z;
  std::vector<double> v_z_post_accel;
  std::vector<double> x_r_post_accel;
  std::vector<double> x_z_post_accel;
  std::vector<double> x_z_initial;
  std::vector<std::uint8_t> node_active;
  std::vector<double> prev_gap_north;
  std::vector<double> prev_gap_south;
  bool emitted_post_accel = false;
  bool emitted_post_remap = false;
};

inline Snapshot g_snapshot;

inline bool enabled() {
  const char* env = std::getenv("TENRYU_I1B_POLE_AXIS_DIAG");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

inline int every() {
  const char* env = std::getenv("TENRYU_I1B_POLE_AXIS_DIAG_EVERY");
  if (env == nullptr || env[0] == '\0') {
    return 20;
  }
  return std::max(1, std::atoi(env));
}

inline bool due(const core::State& state) {
  return enabled() && every() > 0 && state.step % every() == 0 &&
         state.mesh.topo.multiblock.has_value();
}

inline void capture_accel_snapshot(const core::State& state,
                                   const core::NodeField1D& node_mass,
                                   const core::NodeField1D& force_r,
                                   const core::NodeField1D& force_z,
                                   const core::NodeField1D& accel_r,
                                   const core::NodeField1D& accel_z,
                                   const std::uint8_t* d_node_active) {
  if (!due(state)) {
    return;
  }
  g_snapshot.step = state.step;
  g_snapshot.valid = true;
  node_mass.copy_to_host(g_snapshot.node_mass);
  force_r.copy_to_host(g_snapshot.force_r);
  force_z.copy_to_host(g_snapshot.force_z);
  accel_r.copy_to_host(g_snapshot.accel_r);
  accel_z.copy_to_host(g_snapshot.accel_z);
  g_snapshot.v_z_post_accel.clear();
  state.x_r.copy_to_host(g_snapshot.x_r_post_accel);
  state.x_z.copy_to_host(g_snapshot.x_z_post_accel);
  g_snapshot.x_z_initial.clear();
  if (state.x_z_initial.size() == state.x_z.size()) {
    state.x_z_initial.copy_to_host(g_snapshot.x_z_initial);
  }
  g_snapshot.node_active.assign(state.x_r.size(), 0U);
#ifdef __CUDACC__
  if (d_node_active != nullptr && !g_snapshot.node_active.empty()) {
    CUDA_CHECK(cudaMemcpy(g_snapshot.node_active.data(),
                          d_node_active,
                          g_snapshot.node_active.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost));
  }
#else
  (void)d_node_active;
#endif
  g_snapshot.emitted_post_accel = false;
  g_snapshot.emitted_post_remap = false;
}

inline void capture_post_accel_velocity(const core::State& state) {
  if (!g_snapshot.valid || state.step != g_snapshot.step) {
    return;
  }
  state.v_z.copy_to_host(g_snapshot.v_z_post_accel);
}

inline bool mask_has(const std::vector<std::uint8_t>& mask, const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < mask.size() &&
         mask[static_cast<std::size_t>(n)] != 0U;
}

inline bool pole_deref_macro_contains_node(const core::State& state,
                                           const int q,
                                           const int j) {
  const auto& pc = state.pole_angular_derefine;
  if (!pc.configured || !pc.built || !pc.valid) {
    return false;
  }
  for (const auto& macro : pc.macros) {
    if (q >= macro.local_i_begin && q <= macro.local_i_end &&
        j >= macro.local_j_begin && j <= macro.local_j_end) {
      return true;
    }
  }
  return false;
}

inline double at_or_nan(const std::vector<double>& values, const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < values.size()
             ? values[static_cast<std::size_t>(n)]
             : std::numeric_limits<double>::quiet_NaN();
}

inline std::uint8_t u8_at_or_zero(const std::vector<std::uint8_t>& values,
                                  const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < values.size()
             ? values[static_cast<std::size_t>(n)]
             : 0U;
}

inline bool snapshot_matches_remap_context(const core::State& state) {
  return g_snapshot.valid &&
         (state.step == g_snapshot.step || state.step == g_snapshot.step + 1);
}

inline void emit_snapshot(const core::State& state,
                          const char* phase,
                          const std::vector<double>& r,
                          const std::vector<double>& z,
                          const std::vector<double>& z0,
                          const std::vector<double>* v_z_post_remap) {
  if (!snapshot_matches_remap_context(state) ||
      !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  int shell_block_id = -1;
  const auto& blocks = state.mesh.topo.multiblock->blocks;
  for (int b = 0; b < static_cast<int>(blocks.size()); ++b) {
    if (&blocks[static_cast<std::size_t>(b)] == &shell) {
      shell_block_id = b;
      break;
    }
  }
  const int n_i = shell.n_i_cells;
  const int n_j = shell.n_j_cells;
  if (n_i <= 0 || n_j <= 0) {
    return;
  }

  const int stride = n_j + 1;
  const auto shell_node = [&](const int q, const int j) {
    return shell.owned_node_begin + q * stride + j;
  };
  const int center_ref_q = n_i;
  const int north_ref = shell_node(center_ref_q, 0);
  const int south_ref = shell_node(center_ref_q, n_j);
  const std::vector<double>& z_center_source = z0.empty() ? z : z0;
  const double z_c = 0.5 * (at_or_nan(z_center_source, north_ref) +
                            at_or_nan(z_center_source, south_ref));

  if (g_snapshot.prev_gap_north.size() != static_cast<std::size_t>(n_i + 1)) {
    g_snapshot.prev_gap_north.assign(static_cast<std::size_t>(n_i + 1),
                                     std::numeric_limits<double>::quiet_NaN());
    g_snapshot.prev_gap_south.assign(static_cast<std::size_t>(n_i + 1),
                                     std::numeric_limits<double>::quiet_NaN());
  }

  const int columns[2] = {0, n_j};
  for (const int j : columns) {
    std::vector<double>& prev_gap =
        (j == 0) ? g_snapshot.prev_gap_north : g_snapshot.prev_gap_south;
    for (int q = 0; q <= n_i; ++q) {
      const int n = shell_node(q, j);
      const int outer = (q < n_i) ? shell_node(q + 1, j) : -1;
      const double s = std::abs(at_or_nan(z, n) - z_c);
      const double s_outer =
          outer >= 0 ? std::abs(at_or_nan(z, outer) - z_c)
                     : std::numeric_limits<double>::quiet_NaN();
      const double gap = outer >= 0 ? s_outer - s
                                    : std::numeric_limits<double>::quiet_NaN();
      const double old_gap = prev_gap[static_cast<std::size_t>(q)];
      const double dgap = std::isfinite(old_gap) ? gap - old_gap
                                                 : std::numeric_limits<double>::quiet_NaN();
      prev_gap[static_cast<std::size_t>(q)] = gap;
      const bool pole_member = pole_deref_macro_contains_node(state, q, j);
      const bool pole_boundary =
          mask_has(state.pole_angular_derefine.boundary_node_mask, n);
      const bool central_boundary =
          mask_has(state.central_pseudo_core.boundary_node_mask, n);
      const std::uint8_t flags =
          n >= 0 && static_cast<std::size_t>(n) < state.mesh.topo.node_flags.size()
              ? state.mesh.topo.node_flags[static_cast<std::size_t>(n)]
              : 0U;

      std::ostringstream os;
      os << std::setprecision(17)
         << "[pole_axis_diag]"
         << " step=" << g_snapshot.step
         << " phase=" << (phase != nullptr ? phase : "post_remap")
         << " node_id=" << n
         << " block=" << shell_block_id
         << " local_i=" << q
         << " local_j=" << j
         << " r=" << at_or_nan(r, n)
         << " z=" << at_or_nan(z, n)
         << " z_c=" << z_c
         << " s=" << s
         << " node_mass=" << at_or_nan(g_snapshot.node_mass, n)
         << " mass_le0=" << (at_or_nan(g_snapshot.node_mass, n) <= 0.0 ? 1 : 0)
         << " node_active=" << static_cast<int>(u8_at_or_zero(g_snapshot.node_active, n))
         << " force_r=" << at_or_nan(g_snapshot.force_r, n)
         << " force_z=" << at_or_nan(g_snapshot.force_z, n)
         << " accel_r=" << at_or_nan(g_snapshot.accel_r, n)
         << " accel_z=" << at_or_nan(g_snapshot.accel_z, n)
         << " v_z_post_accel=" << at_or_nan(g_snapshot.v_z_post_accel, n)
         << " v_z_post_remap="
         << (v_z_post_remap != nullptr
                 ? at_or_nan(*v_z_post_remap, n)
                 : std::numeric_limits<double>::quiet_NaN())
         << " gap=" << gap
         << " dgap=" << dgap
         << " pole_deref_member=" << (pole_member ? 1 : 0)
         << " pole_deref_boundary=" << (pole_boundary ? 1 : 0)
         << " pole_deref_boundary_force_node=" << (pole_boundary ? 1 : 0)
         << " central_macro_boundary=" << (central_boundary ? 1 : 0)
         << " node_flags=" << static_cast<int>(flags);
      core::log_warning(os.str());
    }
  }
}

inline void emit_post_remap(const core::State& state, const char* phase) {
  if (!snapshot_matches_remap_context(state) || g_snapshot.emitted_post_remap ||
      g_snapshot.v_z_post_accel.empty()) {
    return;
  }
  if (!g_snapshot.emitted_post_accel) {
    emit_snapshot(state,
                  "post_accel",
                  g_snapshot.x_r_post_accel,
                  g_snapshot.x_z_post_accel,
                  g_snapshot.x_z_initial,
                  nullptr);
    g_snapshot.emitted_post_accel = true;
  }

  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> z0;
  std::vector<double> v_z_post_remap;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  if (state.x_z_initial.size() == state.x_z.size()) {
    state.x_z_initial.copy_to_host(z0);
  }
  state.v_z.copy_to_host(v_z_post_remap);
  emit_snapshot(state,
                phase != nullptr ? phase : "post_remap",
                r,
                z,
                z0,
                &v_z_post_remap);
  g_snapshot.emitted_post_remap = true;
  g_snapshot.valid = false;
}

}  // namespace tenryu::hydro::pole_axis_diag
