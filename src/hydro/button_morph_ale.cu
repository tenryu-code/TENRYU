#include "hydro/button_morph_ale.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/button_morph_indexing.hpp"
#include "hydro/reference_barrier_ale.hpp"
#include "mesh/button_align_target.hpp"
#include "mesh/mesh.hpp"
#include "mesh/multiblock_theta_ladder.hpp"

namespace tenryu::hydro::button_morph {
namespace {

int core_boundary_node_id(const ButtonIndexing& idx, const int k) {
  TENRYU_ASSERT(k >= 0 && k <= idx.ntheta,
                "button morph core boundary index out of range");
  if (k <= idx.n_c) {
    return core_node_id(idx, k, 2 * idx.n_c);
  }
  if (k <= 3 * idx.n_c) {
    return core_node_id(idx, idx.n_c, 3 * idx.n_c - k);
  }
  return core_node_id(idx, 4 * idx.n_c - k, 0);
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss.setf(std::ios::scientific);
  oss.precision(16);
  oss << value;
  return oss.str();
}

std::vector<double> compute_local_node_spacing(
    const tenryu::core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z) {
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "button morph requires multiblock topology CSR");
  TENRYU_ASSERT(x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    x_z.size() == static_cast<std::size_t>(n_nodes),
                "button morph current node coordinate size mismatch");
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "button morph cell-node CSR offset size mismatch");

  std::vector<double> h_local(static_cast<std::size_t>(n_nodes),
                              std::numeric_limits<double>::infinity());
  for (int c = 0; c < n_cells; ++c) {
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    TENRYU_ASSERT(begin >= 0 && end > begin &&
                      static_cast<std::size_t>(end) <=
                          mb.cell_node_csr_indices.size(),
                  "button morph cell-node CSR range invalid");
    const int n_cell_nodes = end - begin;
    for (int k = 0; k < n_cell_nodes; ++k) {
      const int n0 = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + k)];
      const int n1 = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + (k + 1) % n_cell_nodes)];
      TENRYU_ASSERT(n0 >= 0 && n0 < n_nodes && n1 >= 0 && n1 < n_nodes,
                    "button morph cell-node CSR node id out of range");
      const double edge_length =
          std::hypot(x_r[static_cast<std::size_t>(n1)] -
                         x_r[static_cast<std::size_t>(n0)],
                     x_z[static_cast<std::size_t>(n1)] -
                         x_z[static_cast<std::size_t>(n0)]);
      if (std::isfinite(edge_length) && edge_length > 0.0) {
        h_local[static_cast<std::size_t>(n0)] =
            std::min(h_local[static_cast<std::size_t>(n0)], edge_length);
        h_local[static_cast<std::size_t>(n1)] =
            std::min(h_local[static_cast<std::size_t>(n1)], edge_length);
      }
    }
  }
  return h_local;
}

double max_displacement_over_spacing(const std::vector<double>& target_r,
                                     const std::vector<double>& target_z,
                                     const std::vector<double>& x_r,
                                     const std::vector<double>& x_z,
                                     const std::vector<double>& h_local) {
  TENRYU_ASSERT(target_r.size() == target_z.size() &&
                    target_r.size() == x_r.size() &&
                    target_r.size() == x_z.size() &&
                    target_r.size() == h_local.size(),
                "button morph displacement vector size mismatch");
  double max_ratio = 0.0;
  for (std::size_t n = 0; n < target_r.size(); ++n) {
    if (!(std::isfinite(h_local[n]) && h_local[n] > 0.0)) {
      continue;
    }
    const double displacement =
        std::hypot(target_r[n] - x_r[n], target_z[n] - x_z[n]);
    max_ratio = std::max(max_ratio, displacement / h_local[n]);
  }
  return max_ratio;
}

}  // namespace

double button_morph_alpha(const double t,
                          const double t_start,
                          const double t_end) {
  TENRYU_ASSERT(t_end > t_start,
                "button morph schedule end must exceed start");
  if (t <= t_start) {
    return 0.0;
  }
  if (t >= t_end) {
    return 1.0;
  }
  const double u =
      std::clamp((t - t_start) / (t_end - t_start), 0.0, 1.0);
  const double u2 = u * u;
  const double u3 = u2 * u;
  const double u4 = u3 * u;
  const double u5 = u4 * u;
  return 6.0 * u5 - 15.0 * u4 + 10.0 * u3;
}

void build_button_morph_targets(const tenryu::core::State& state,
                                const tenryu::core::Config& cfg,
                                const double alpha,
                                const std::vector<double>& current_r,
                                const std::vector<double>& current_z,
                                std::vector<double>& tr,
                                std::vector<double>& tz) {
  const auto idx = button_morph::make_button_indexing(cfg.mesh);
  TENRYU_ASSERT(idx.n_nodes_total == state.mesh.topo.n_nodes,
                "button morph node count mismatch");
  const std::size_t n_nodes = static_cast<std::size_t>(idx.n_nodes_total);
  TENRYU_ASSERT(current_r.size() == n_nodes && current_z.size() == n_nodes,
                "button morph current coordinate size mismatch");

  std::vector<double> init_r(n_nodes, 0.0);
  std::vector<double> init_z(n_nodes, 0.0);
  state.x_r_reference.copy_to_host(init_r.data());
  state.x_z_reference.copy_to_host(init_z.data());
  TENRYU_ASSERT(init_r.size() == n_nodes && init_z.size() == n_nodes,
                "button morph reference coordinate size mismatch");

  tr = current_r;
  tz = current_z;
  if (alpha <= 0.0) {
    return;
  }

  for (int i = 0; i <= idx.n_c; ++i) {
    for (int j = 0; j <= 2 * idx.n_c; ++j) {
      const int id = core_node_id(idx, i, j);
      const std::size_t n = static_cast<std::size_t>(id);
      const double r0 = init_r[n];
      const double z0 = init_z[n];
      const double a = std::max(r0, std::abs(z0));
      double aligned_r = 0.0;
      double aligned_z = 0.0;
      if (a > 0.0) {
        tenryu::mesh::assembly::button_align_core_target(
            r0, z0, &aligned_r, &aligned_z);
      }
      if (i == 0) {
        TENRYU_ASSERT(r0 == 0.0,
                      "button morph axis source must have zero radius");
        TENRYU_ASSERT(aligned_r == 0.0,
                      "button morph aligned axis radius must be exactly zero");
      }
      tr[n] = r0 + alpha * (aligned_r - r0);
      tz[n] = z0 + alpha * (aligned_z - z0);
    }
  }

  TENRYU_ASSERT(cfg.mesh.explicit_nodes.size() >= 2U,
                "button morph requires an explicit shell radial ladder");
  const double h_seam =
      cfg.mesh.explicit_nodes[1] - cfg.mesh.explicit_nodes[0];
  TENRYU_ASSERT(std::isfinite(h_seam) && h_seam > 0.0,
                "button morph shell first-row spacing must be positive and "
                "finite");

  for (int layer = 1; layer < idx.n_b; ++layer) {
    for (int k = 0; k <= idx.ntheta; ++k) {
      const int core_id = core_boundary_node_id(idx, k);
      const std::size_t core_n = static_cast<std::size_t>(core_id);
      const double t_core =
          tenryu::mesh::assembly::button_align_perimeter_fraction(
              init_r[core_n], init_z[core_n]);
      const double theta_seam = tenryu::mesh::multiblock_theta_node(
          k, idx.ntheta, cfg.mesh.multiblock_theta_cap_widen_factor);
      double aligned_r = 0.0;
      double aligned_z = 0.0;
      tenryu::mesh::assembly::button_align_bridge_target(
          idx.r_c, idx.r_match, layer, idx.n_b, h_seam, t_core, theta_seam,
          &aligned_r, &aligned_z);

      const int id = bridge_interior_node_id(idx, layer, k);
      const std::size_t n = static_cast<std::size_t>(id);
      tr[n] = init_r[n] + alpha * (aligned_r - init_r[n]);
      tz[n] = init_z[n] + alpha * (aligned_z - init_z[n]);
    }
  }
}

ButtonMorphStepResult apply_button_morph_if_due(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double t_end_of_step,
    const int step) {
  ButtonMorphStepResult result;
  const auto& morph_cfg = cfg.numerics.ale.button_morph;
  if (!morph_cfg.enabled) {
    return result;
  }
  if (cfg.mesh.topology_scheme !=
      tenryu::core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
    static bool warned_non_button_topology = false;
    if (!warned_non_button_topology) {
      tenryu::core::log_warning(
          "[button_morph] enabled but topology is not cart_core_polar_shell — inert");
      warned_non_button_topology = true;
    }
    return result;
  }
  if (t_end_of_step < morph_cfg.t_start_s ||
      t_end_of_step > morph_cfg.t_end_s) {
    return result;
  }
  if (step % morph_cfg.every_n_steps != 0) {
    return result;
  }

  result.attempted = true;
  result.alpha = button_morph_alpha(
      t_end_of_step, morph_cfg.t_start_s, morph_cfg.t_end_s);

  const std::size_t n_nodes =
      static_cast<std::size_t>(state.mesh.topo.n_nodes);
  std::vector<double> current_r(n_nodes, 0.0);
  std::vector<double> current_z(n_nodes, 0.0);
  state.x_r.copy_to_host(current_r.data());
  state.x_z.copy_to_host(current_z.data());

  std::vector<double> target_r;
  std::vector<double> target_z;
  build_button_morph_targets(state, cfg, result.alpha, current_r, current_z,
                             target_r, target_z);

  const std::vector<double> h_local =
      compute_local_node_spacing(state, current_r, current_z);
  result.max_step_over_h = max_displacement_over_spacing(
      target_r, target_z, current_r, current_z, h_local);
  if (result.max_step_over_h > morph_cfg.max_step_fraction) {
    result.lambda_cap =
        morph_cfg.max_step_fraction / result.max_step_over_h;
  }

  std::vector<double> capped_r(target_r.size(), 0.0);
  std::vector<double> capped_z(target_z.size(), 0.0);
  for (std::size_t n = 0; n < target_r.size(); ++n) {
    capped_r[n] =
        current_r[n] + result.lambda_cap * (target_r[n] - current_r[n]);
    capped_z[n] =
        current_z[n] + result.lambda_cap * (target_z[n] - current_z[n]);
  }

  const std::size_t node_bytes = target_r.size() * sizeof(double);
  double* const d_target_r = static_cast<double*>(
      tenryu::core::device_scratch_acquire("button_morph:d_target_r",
                                           node_bytes));
  double* const d_target_z = static_cast<double*>(
      tenryu::core::device_scratch_acquire("button_morph:d_target_z",
                                           node_bytes));
  CUDA_CHECK(cudaMemcpy(d_target_r, capped_r.data(), node_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_target_z, capped_z.data(), node_bytes,
                        cudaMemcpyHostToDevice));

  const auto idx = make_button_indexing(cfg.mesh);
  std::vector<std::uint8_t> cell_mask(
      static_cast<std::size_t>(idx.n_cells_total), 0);
  for (int cell = 0; cell < idx.shell_cell_offset; ++cell) {
    cell_mask[static_cast<std::size_t>(cell)] = 1;
  }
  std::vector<std::uint8_t> node_freeze_mask(
      static_cast<std::size_t>(idx.n_nodes_total), 0);
  for (int node = idx.shell_node_offset; node < idx.n_nodes_total; ++node) {
    node_freeze_mask[static_cast<std::size_t>(node)] = 1;
  }
  auto* const d_cell_mask = static_cast<std::uint8_t*>(
      tenryu::core::device_scratch_acquire("button_morph:d_cell_mask",
                                           cell_mask.size()));
  auto* const d_node_freeze = static_cast<std::uint8_t*>(
      tenryu::core::device_scratch_acquire("button_morph:d_node_freeze",
                                           node_freeze_mask.size()));
  CUDA_CHECK(cudaMemcpy(d_cell_mask, cell_mask.data(), cell_mask.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_node_freeze,
                        node_freeze_mask.data(),
                        node_freeze_mask.size(),
                        cudaMemcpyHostToDevice));
  // The transaction must be bitwise inert outside the mandate; the shock
  // lives in the shell during the morph window and repeated global velocity
  // projections were measured to pump the post-shock axis columns
  // (+2.2e4 cm/s per pass).
  tenryu::hydro::ReferenceBarrierScope scope{d_cell_mask, d_node_freeze};
  const auto barrier = tenryu::hydro::apply_reference_barrier_rezone(
      state, cfg, d_target_r, d_target_z, &scope);
  result.lambda_accepted = barrier.lambda_accepted;
  result.applied = barrier.succeeded && barrier.lambda_accepted > 0.0;

  std::vector<double> post_r(target_r.size(), 0.0);
  std::vector<double> post_z(target_z.size(), 0.0);
  state.x_r.copy_to_host(post_r.data());
  state.x_z.copy_to_host(post_z.data());
  result.residual_max_over_h = max_displacement_over_spacing(
      target_r, target_z, post_r, post_z, h_local);

  tenryu::core::log_info(
      "[button_morph] step=" + std::to_string(step) +
      " t=" + format_scientific(t_end_of_step) +
      " alpha=" + format_scientific(result.alpha) +
      " lambda_cap=" + format_scientific(result.lambda_cap) +
      " lambda_accepted=" + format_scientific(result.lambda_accepted) +
      " max_step_over_h=" + format_scientific(result.max_step_over_h) +
      " residual_over_h=" + format_scientific(result.residual_max_over_h) +
      " applied=" + (result.applied ? "true" : "false"));
  return result;
}

}  // namespace tenryu::hydro::button_morph
