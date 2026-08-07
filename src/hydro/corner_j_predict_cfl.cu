#include "hydro/corner_j_predict_cfl.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/button_morph_indexing.hpp"

namespace tenryu::hydro {

double compute_corner_j_predict_cfl_dt(const core::State& state,
                                       const core::Config& cfg) {
  if (cfg.mesh.topology_scheme !=
      core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
    static bool warned_non_button_topology = false;
    if (!warned_non_button_topology) {
      core::log_warning(
          "[corner_j_predict] enabled but topology is not "
          "multiblock_cart_core_polar_shell - inert");
      warned_non_button_topology = true;
    }
    return std::numeric_limits<double>::infinity();
  }

  const auto indexing = button_morph::make_button_indexing(cfg.mesh);
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  TENRYU_ASSERT(state.mesh.dim == 2,
                "corner-J prediction requires a 2D mesh");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "corner-J prediction requires multiblock topology metadata");
  TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                "corner-J prediction requires matching node position arrays");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "corner-J prediction requires matching node velocity arrays");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "corner-J prediction requires topo/state node-size consistency");
  TENRYU_ASSERT(cfg.numerics.hydro.corner_j_predict_cfl_safety > 0.0 &&
                    cfg.numerics.hydro.corner_j_predict_cfl_safety <= 1.0,
                "corner-J prediction safety must be in (0, 1]");
  TENRYU_ASSERT(cfg.numerics.hydro.corner_j_predict_shell_rings >= 0,
                "corner-J prediction shell rings must be non-negative");

  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "corner-J prediction requires multiblock cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) * 4U,
                "corner-J prediction requires multiblock cell-node CSR indices");

  std::vector<double> node_r(state.x_r.size(), 0.0);
  std::vector<double> node_z(state.x_z.size(), 0.0);
  std::vector<double> v_r(state.v_r.size(), 0.0);
  std::vector<double> v_z(state.v_z.size(), 0.0);
  state.x_r.copy_to_host(node_r.data());
  state.x_z.copy_to_host(node_z.data());
  state.v_r.copy_to_host(v_r.data());
  state.v_z.copy_to_host(v_z.data());

  const long long scoped_end =
      static_cast<long long>(indexing.shell_cell_offset) +
      static_cast<long long>(cfg.numerics.hydro.corner_j_predict_shell_rings) *
          static_cast<long long>(indexing.ntheta);
  const int scoped_cells = static_cast<int>(
      std::min(static_cast<long long>(n_cells), scoped_end));
  double dt_max = std::numeric_limits<double>::infinity();
  for (int c = 0; c < scoped_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    const int off = mb.cell_node_csr_offsets[idx];
    double xr[4] = {};
    double xz[4] = {};
    double vr[4] = {};
    double vz[4] = {};
    for (int k = 0; k < 4; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(node >= 0 && node < state.mesh.topo.n_nodes,
                    "corner-J prediction cell-node index out of range");
      const std::size_t n = static_cast<std::size_t>(node);
      xr[k] = node_r[n];
      xz[k] = node_z[n];
      vr[k] = v_r[n];
      vz[k] = v_z[n];
    }
    dt_max = std::min(
        dt_max,
        corner_j_predict_dt_max(
            xr, xz, vr, vz,
            cfg.numerics.hydro.corner_j_predict_max_shrink));
  }

  return cfg.numerics.hydro.corner_j_predict_cfl_safety * dt_max;
}

}  // namespace tenryu::hydro
