#include "hydro/axis_edge_collapse.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/evacuated_cell_shadow.hpp"
#include "hydro/flank_tangential_strip.hpp"
#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro {
namespace {

double median(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const std::size_t mid = values.size() / 2U;
  if ((values.size() & 1U) != 0U) {
    return values[mid];
  }
  return 0.5 * (values[mid - 1U] + values[mid]);
}

int structured_cell_corner_for_node(const int cell,
                                    const int node,
                                    const int nz) {
  const int i = cell / nz;
  const int j = cell - i * nz;
  const std::array<int, 4> nodes = {
      i * (nz + 1) + j,
      (i + 1) * (nz + 1) + j,
      (i + 1) * (nz + 1) + j + 1,
      i * (nz + 1) + j + 1,
  };
  for (int corner = 0; corner < 4; ++corner) {
    if (nodes[static_cast<std::size_t>(corner)] == node) {
      return corner;
    }
  }
  return -1;
}

bool audit_equal(const double lhs, const double rhs) {
  const double scale = std::max(std::abs(lhs), std::abs(rhs));
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  return std::isfinite(lhs) && std::isfinite(rhs) &&
         std::abs(lhs - rhs) <= tolerance;
}

struct AxisCollapseChainSlotMatch {
  core::EvacContactSlot* slot = nullptr;
  int pair = -1;
  int side = 0;
};

AxisCollapseChainSlotMatch axis_collapse_chain_slot(
    core::State& state,
    const int failing_cell,
    const int node_lo,
    const int node_hi,
    const int nz,
    const char** fail_reason) {
  if (fail_reason != nullptr) {
    *fail_reason = "";
  }
  const auto find_matching_slot =
      [&](const int cell, const int anchor_node) {
        for (int pair = 0; pair < 2; ++pair) {
          const auto matching = std::find_if(
              state.contact_graph.records.begin(),
              state.contact_graph.records.end(),
              [&](const core::EvacContactSlot& slot) {
                return slot.state == core::EvacContactState::kActive &&
                       slot.cell == cell && slot.pair_engaged[pair] != 0U &&
                       slot.node_b[pair] == anchor_node;
              });
          if (matching != state.contact_graph.records.end()) {
            return AxisCollapseChainSlotMatch{&*matching, pair, 0};
          }
        }
        return AxisCollapseChainSlotMatch{};
      };

  int cell = failing_cell - 1;
  const auto& collapsed =
      state.evacuated_cells.cell_axis_edge_collapsed;
  while (cell >= 0 && cell / nz == 0 && !collapsed.empty() &&
         collapsed[static_cast<std::size_t>(cell)] != 0U) {
    --cell;
  }
  const bool low_side_reached_terminal = cell >= 0 && cell / nz == 0;
  const char* low_side_fail_reason = "walk_left_row0";
  if (cell < 0 || cell / nz != 0) {
    low_side_fail_reason = "walk_left_row0";
  } else {
    const auto active = std::find_if(
        state.contact_graph.records.begin(),
        state.contact_graph.records.end(),
        [&](const core::EvacContactSlot& slot) {
          return slot.state == core::EvacContactState::kActive &&
                 slot.cell == cell;
        });
    if (active == state.contact_graph.records.end()) {
      low_side_fail_reason = "no_active_slot_at_chain_end";
    } else {
      const auto engaged = std::find_if(
          state.contact_graph.records.begin(),
          state.contact_graph.records.end(),
          [&](const core::EvacContactSlot& slot) {
            return slot.state == core::EvacContactState::kActive &&
                   slot.cell == cell &&
                   (slot.pair_engaged[0] != 0U ||
                    slot.pair_engaged[1] != 0U);
          });
      if (engaged == state.contact_graph.records.end()) {
        low_side_fail_reason = "pair_not_engaged";
      } else {
        AxisCollapseChainSlotMatch matching =
            find_matching_slot(cell, node_lo);
        if (matching.slot != nullptr) {
          matching.side = -1;
          return matching;
        }
        low_side_fail_reason = "node_b_mismatch";
      }
    }
  }

  cell = failing_cell + 1;
  while (cell < nz && !collapsed.empty() &&
         collapsed[static_cast<std::size_t>(cell)] != 0U) {
    ++cell;
  }
  if (cell < nz) {
    AxisCollapseChainSlotMatch matching =
        find_matching_slot(cell, node_hi);
    if (matching.slot != nullptr) {
      matching.side = 1;
      return matching;
    }
  }

  if (fail_reason != nullptr) {
    *fail_reason = low_side_reached_terminal
                       ? low_side_fail_reason
                       : "no_slot_either_side";
  }
  return {};
}

}  // namespace

void rebuild_geometry_policy_exempt(core::State& state) {
  auto& evacuated = state.evacuated_cells;
  const auto& cell_axis_edge_collapsed =
      evacuated.cell_axis_edge_collapsed;
  auto& geometry_policy_exempt_cells =
      evacuated.geometry_policy_exempt_cells;
  if (cell_axis_edge_collapsed.empty() &&
      state.contact_graph.records.empty()) {
    geometry_policy_exempt_cells.clear();
  } else {
    geometry_policy_exempt_cells.assign(
        static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
    for (std::size_t cell = 0; cell < cell_axis_edge_collapsed.size(); ++cell) {
      if (cell_axis_edge_collapsed[cell] != 0U) {
        geometry_policy_exempt_cells[cell] = 1U;
      }
    }
    for (const core::EvacContactSlot& slot : state.contact_graph.records) {
      if (slot.devolumized != 0U) {
        geometry_policy_exempt_cells[static_cast<std::size_t>(slot.cell)] = 1U;
      }
    }
  }
  evacuated.d_geometry_policy_exempt_cells.reset(
      geometry_policy_exempt_cells.size());
  evacuated.d_geometry_policy_exempt_cells.copy_from_host(
      geometry_policy_exempt_cells);
}

void rebuild_axis_edge_collapse_device_state(core::State& state) {
  auto& evacuated = state.evacuated_cells;
  const auto& cell_axis_edge_collapsed =
      evacuated.cell_axis_edge_collapsed;
  if (!cell_axis_edge_collapsed.empty()) {
    evacuated.d_cell_axis_edge_collapsed.reset(
        cell_axis_edge_collapsed.size());
    evacuated.d_cell_axis_edge_collapsed.copy_from_host(
        cell_axis_edge_collapsed);
  }

  const auto& node_axis_alias = evacuated.node_axis_alias;
  if (!node_axis_alias.empty()) {
    evacuated.d_node_axis_alias.reset(node_axis_alias.size());
    evacuated.d_node_axis_alias.copy_from_host(node_axis_alias);
  } else {
    evacuated.d_node_axis_alias.reset(0);
  }
  std::vector<int> alias_pairs;
  for (std::size_t node = 0; node < node_axis_alias.size(); ++node) {
    const int survivor = node_axis_alias[node];
    if (survivor >= 0) {
      alias_pairs.push_back(static_cast<int>(node));
      alias_pairs.push_back(survivor);
    }
  }
  evacuated.d_node_axis_alias_pairs.reset(alias_pairs.size());
  evacuated.d_node_axis_alias_pairs.copy_from_host(alias_pairs);
  evacuated.n_node_axis_aliases =
      static_cast<int>(alias_pairs.size() / 2U);
  rebuild_geometry_policy_exempt(state);
}

AxisEdgeCollapsePredicateResult axis_edge_collapse_update_and_evaluate(
    core::State& state,
    const core::Config& cfg,
    const int failing_cell,
    const double retry_dt_next,
    const double dt_min_s) {
  AxisEdgeCollapsePredicateResult result;
  const auto& collapse_cfg =
      cfg.numerics.ale.evacuated_cell.closure_contact.axis_edge_collapse;
  const int nz = state.mesh.topo.nz;
  if (!collapse_cfg.enabled || state.mesh.dim != 2 || nz <= 0 ||
      failing_cell < 0 || failing_cell >= state.mesh.topo.n_cells ||
      failing_cell / nz != 0) {
    return result;
  }

  const int j = failing_cell % nz;
  const int node_lo = state.mesh.topo.node_index(0, j);
  const int node_hi = state.mesh.topo.node_index(0, j + 1);
  std::vector<double> z;
  std::vector<double> vz;
  state.x_z.copy_to_host(z);
  state.v_z.copy_to_host(vz);

  const double h0 = z[static_cast<std::size_t>(node_hi)] -
                    z[static_cast<std::size_t>(node_lo)];
  const double hdot = vz[static_cast<std::size_t>(node_hi)] -
                      vz[static_cast<std::size_t>(node_lo)];
  auto& monitor = state.evacuated_cells.axis_edge_collapse_monitor;
  if (monitor.cell != failing_cell) {
    const int repair_step = monitor.repair_step;
    monitor = core::EvacuatedCellsState::AxisEdgeCollapseMonitor{};
    monitor.repair_step = repair_step;
    monitor.cell = failing_cell;
    monitor.node_lo = node_lo;
    monitor.node_hi = node_hi;
    std::vector<double> local_axis_spacing;
    const int first_edge = std::max(0, j - 4);
    const int last_edge = std::min(nz - 1, j + 4);
    local_axis_spacing.reserve(
        static_cast<std::size_t>(last_edge - first_edge));
    for (int k = first_edge; k <= last_edge; ++k) {
      if (k == j) {
        continue;
      }
      const int k_lo = state.mesh.topo.node_index(0, k);
      const int k_hi = state.mesh.topo.node_index(0, k + 1);
      local_axis_spacing.push_back(std::abs(
          z[static_cast<std::size_t>(k_hi)] -
          z[static_cast<std::size_t>(k_lo)]));
    }
    monitor.h_ref = std::max(median(std::move(local_axis_spacing)),
                             std::abs(h0));
  }

  const int persistence_window = collapse_cfg.persistence_window;
  if (state.step != monitor.last_sample_step) {
    const int head = monitor.ring_head;
    monitor.h0[head] = h0;
    monitor.hdot[head] = hdot;
    monitor.dt_sample[head] = retry_dt_next;
    monitor.ring_head = (head + 1) % persistence_window;
    monitor.n_samples = std::min(monitor.n_samples + 1,
                                 persistence_window);
    monitor.last_sample_step = state.step;
  }

  const double inf = std::numeric_limits<double>::infinity();
  const double z_lo_abs = std::abs(z[static_cast<std::size_t>(node_lo)]);
  const double z_hi_abs = std::abs(z[static_cast<std::size_t>(node_hi)]);
  const double ulp_lo = std::nextafter(z_lo_abs, inf) - z_lo_abs;
  const double ulp_hi = std::nextafter(z_hi_abs, inf) - z_hi_abs;
  const double ulp_max = std::max(ulp_lo, ulp_hi);
  const double h_retire =
      std::max(collapse_cfg.ulp_count * ulp_max,
               collapse_cfg.h_ref_fraction * monitor.h_ref);

  const int window = monitor.n_samples;
  const int oldest =
      window == persistence_window ? monitor.ring_head : 0;
  const auto ring_index = [&](const int sample) {
    return (oldest + sample) % persistence_window;
  };
  std::vector<double> acceleration_samples;
  if (window > 1) {
    acceleration_samples.reserve(static_cast<std::size_t>(window - 1));
  }
  int closing_count = 0;
  for (int sample = 0; sample < window; ++sample) {
    const int index = ring_index(sample);
    if (monitor.hdot[index] < 0.0) {
      ++closing_count;
    }
    if (sample > 0) {
      const int older = ring_index(sample - 1);
      acceleration_samples.push_back(
          (monitor.hdot[index] - monitor.hdot[older]) /
          monitor.dt_sample[index]);
    }
  }
  const double a_h = median(std::move(acceleration_samples));
  const double h_min_pred =
      a_h > 0.0 ? h0 - hdot * hdot / (2.0 * a_h) : -inf;

  const AxisCollapseChainSlotMatch chain_slot = axis_collapse_chain_slot(
      state, failing_cell, node_lo, node_hi, nz,
      &result.ineligible_reason);
  result.eligible = chain_slot.slot != nullptr;
  result.h0 = h0;
  result.hdot = hdot;
  result.h_retire = h_retire;
  result.h_min_pred = h_min_pred;
  result.closing_count = closing_count;
  result.window = window;
  result.would_fire =
      result.eligible && h0 <= h_retire && hdot < 0.0 &&
      h_min_pred <= h_retire && window >= persistence_window &&
      closing_count >= collapse_cfg.persistence_min_closing;
  // The retry-dt policy now clamps geometric-failure retries to the dt floor, so a terminally dt-limited step sits AT the floor rather than below it; at-floor qualifies as terminal.
  result.terminal_waiver =
      result.eligible && retry_dt_next <= dt_min_s &&
      h0 < 4.0 * h_retire && hdot < 0.0;
  return result;
}

AxisEdgeCollapseTransactionResult axis_edge_collapse_execute(
    core::State& state,
    const core::Config& cfg,
    const int failing_cell) {
  AxisEdgeCollapseTransactionResult result;
  result.attempted = true;
  const auto reject = [&](const char* const reason) {
    result.reject_reason = reason;
    return result;
  };

  const auto& collapse_cfg =
      cfg.numerics.ale.evacuated_cell.closure_contact.axis_edge_collapse;
  if (!collapse_cfg.enabled) {
    return reject("config_disabled");
  }
  if (state.mesh.dim != 2) {
    return reject("not_2d");
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (nz <= 0 || failing_cell < 0 || failing_cell >= n_cells ||
      failing_cell / nz != 0) {
    return reject("not_axis_row");
  }
  if (!state.hydro_active.empty() &&
      state.hydro_active[static_cast<std::size_t>(failing_cell)] == 0) {
    return reject("inactive_cell");
  }
  const auto& collapsed = state.evacuated_cells.cell_axis_edge_collapsed;
  if (!collapsed.empty() &&
      collapsed[static_cast<std::size_t>(failing_cell)] != 0U) {
    return reject("already_collapsed");
  }

  const int j = failing_cell % nz;
  const int node_lo = state.mesh.topo.node_index(0, j);
  const int node_hi = state.mesh.topo.node_index(0, j + 1);
  const int node_lo_out = state.mesh.topo.node_index(1, j);
  const int node_hi_out = state.mesh.topo.node_index(1, j + 1);
  const AxisCollapseChainSlotMatch chain_slot = axis_collapse_chain_slot(
      state, failing_cell, node_lo, node_hi, nz, nullptr);
  core::EvacContactSlot* const contact_slot = chain_slot.slot;
  if (contact_slot == nullptr) {
    return reject("axis_partner_not_engaged");
  }
  TENRYU_ASSERT((chain_slot.side == -1 || chain_slot.side == 1) &&
                    chain_slot.pair >= 0 && chain_slot.pair < 2,
                "axis-edge collapse chain match mismatch");
  const int pair = chain_slot.pair;
  const int survivor_node = node_hi;
  const int retired_node = node_lo;
  const int pair_anchored_node =
      chain_slot.side == -1 ? node_lo : node_hi;
  const bool reanchor_pair = chain_slot.side == -1;
  const int survivor_corner =
      structured_cell_corner_for_node(failing_cell, survivor_node, nz);
  const int retired_corner =
      structured_cell_corner_for_node(failing_cell, retired_node, nz);
  TENRYU_ASSERT(survivor_corner >= 0 && retired_corner >= 0,
                "axis-edge collapse corner convention mismatch");
  auto& monitor = state.evacuated_cells.axis_edge_collapse_monitor;
  if (monitor.repair_step == state.step) {
    return reject("already_this_step");
  }

  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<double> corner_mass;
  std::vector<double> mass;
  std::vector<double> ei;
  std::vector<double> volume;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);
  state.corner_mass.copy_to_host(corner_mass);
  state.mass.copy_to_host(mass);
  state.ei.copy_to_host(ei);
  state.vol.copy_to_host(volume);
  TENRYU_ASSERT(x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    x_z.size() == x_r.size() && v_r.size() == x_r.size() &&
                    v_z.size() == x_r.size() &&
                    corner_mass.size() == 4U * static_cast<std::size_t>(n_cells) &&
                    mass.size() == static_cast<std::size_t>(n_cells) &&
                    ei.size() == mass.size() && volume.size() == mass.size() &&
                    state.evacuated_cells.inactive_member_mask.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    (state.hydro_active.empty() ||
                     state.hydro_active.size() == static_cast<std::size_t>(n_cells)),
                "axis-edge collapse field size mismatch");

  const std::size_t corner_base =
      4U * static_cast<std::size_t>(failing_cell);
  const double m_a =
      corner_mass[corner_base + static_cast<std::size_t>(retired_corner)];
  const std::vector<std::int8_t>* const hydro_active =
      state.hydro_active.empty() ? nullptr : &state.hydro_active;
  const double m_b = evacuated_cell_active_node_mass(
      survivor_node,
      nr,
      nz,
      corner_mass,
      state.evacuated_cells.inactive_member_mask,
      hydro_active);
  if (!(m_a > 0.0) || !(m_b > 0.0)) {
    return reject("nonpositive_nodal_mass");
  }

  const std::size_t retired = static_cast<std::size_t>(retired_node);
  const std::size_t survivor = static_cast<std::size_t>(survivor_node);
  const std::size_t pair_anchor =
      static_cast<std::size_t>(pair_anchored_node);
  const double v_lo_r_old = v_r[retired];
  const double v_lo_z_old = v_z[retired];
  const double v_hi_r_old = v_r[survivor];
  const double v_hi_z_old = v_z[survivor];
  const double merged_mass = m_a + m_b;
  const double v_plus_r =
      (m_a * v_lo_r_old + m_b * v_hi_r_old) / merged_mass;
  const double v_plus_z =
      (m_a * v_lo_z_old + m_b * v_hi_z_old) / merged_mass;
  const double dv_r = v_lo_r_old - v_hi_r_old;
  const double dv_z = v_lo_z_old - v_hi_z_old;
  result.q_node =
      0.5 * m_a * m_b / merged_mass * (dv_r * dv_r + dv_z * dv_z);

  const double corner_mass_sum_before =
      corner_mass[corner_base] + corner_mass[corner_base + 1U] +
      corner_mass[corner_base + 2U] + corner_mass[corner_base + 3U];
  const double pre_collapse_volume =
      volume[static_cast<std::size_t>(failing_cell)];
  const core::EvacContactSlot slot_before = *contact_slot;

  x_z[survivor] = x_z[pair_anchor];
  x_r[survivor] = 0.0;
  x_z[retired] = x_z[survivor];
  x_r[retired] = 0.0;
  v_r[retired] = v_plus_r;
  v_z[retired] = v_plus_z;
  v_r[survivor] = v_plus_r;
  v_z[survivor] = v_plus_z;
  corner_mass[corner_base + static_cast<std::size_t>(survivor_corner)] +=
      corner_mass[corner_base + static_cast<std::size_t>(retired_corner)];
  corner_mass[corner_base + static_cast<std::size_t>(retired_corner)] = 0.0;
  TENRYU_ASSERT(mass[static_cast<std::size_t>(failing_cell)] > 0.0,
                "axis-edge collapse heat target has nonpositive mass");
  ei[static_cast<std::size_t>(failing_cell)] +=
      result.q_node / mass[static_cast<std::size_t>(failing_cell)];

  if (reanchor_pair) {
    contact_slot->node_b[pair] = survivor_node;
  }
  std::fill(std::begin(contact_slot->mortar_g_hold_valid),
            std::end(contact_slot->mortar_g_hold_valid), 0U);
  contact_slot->lambda_last = 0.0;
  contact_slot->tensile_streak_pair[pair] = 0;
  const int node_a = contact_slot->node_a[pair];
  contact_slot->gap_pair[pair] =
      (x_r[survivor] - x_r[static_cast<std::size_t>(node_a)]) *
          contact_slot->normal_pair_r[pair] +
      (x_z[survivor] - x_z[static_cast<std::size_t>(node_a)]) *
          contact_slot->normal_pair_z[pair];
  contact_slot->gap_prev_pair[pair] = contact_slot->gap_pair[pair];
  contact_slot->gap = std::numeric_limits<double>::infinity();
  for (int engaged_pair = 0; engaged_pair < 2; ++engaged_pair) {
    if (contact_slot->pair_engaged[engaged_pair] != 0U) {
      contact_slot->gap =
          std::min(contact_slot->gap,
                   contact_slot->gap_pair[engaged_pair]);
    }
  }

  const double momentum_r_before =
      m_a * v_lo_r_old + m_b * v_hi_r_old;
  const double momentum_r_after = merged_mass * v_plus_r;
  const double momentum_z_before =
      m_a * v_lo_z_old + m_b * v_hi_z_old;
  const double momentum_z_after = merged_mass * v_plus_z;
  result.mom_r_residual = momentum_r_before - momentum_r_after;
  result.mom_z_residual = momentum_z_before - momentum_z_after;
  const double kinetic_before =
      0.5 * m_a * (v_lo_r_old * v_lo_r_old + v_lo_z_old * v_lo_z_old) +
      0.5 * m_b * (v_hi_r_old * v_hi_r_old + v_hi_z_old * v_hi_z_old);
  const double kinetic_and_heat_after =
      0.5 * merged_mass * (v_plus_r * v_plus_r + v_plus_z * v_plus_z) +
      result.q_node;
  result.energy_residual = kinetic_before - kinetic_and_heat_after;
  const double corner_mass_sum_after =
      corner_mass[corner_base] + corner_mass[corner_base + 1U] +
      corner_mass[corner_base + 2U] + corner_mass[corner_base + 3U];
  result.mass_residual = corner_mass_sum_before - corner_mass_sum_after;

  const int quad_nodes[4] = {node_lo, node_lo_out, node_hi_out, node_hi};
  double quad_r[4];
  double quad_z[4];
  for (int corner = 0; corner < 4; ++corner) {
    const std::size_t node =
        static_cast<std::size_t>(quad_nodes[corner]);
    quad_r[corner] = x_r[node];
    quad_z[corner] = x_z[node];
  }
  result.tri_volume = rz::rz_polygon_volume_exact(quad_r, quad_z, 4);
  const double corner_j_1 = corner_jacobian_from_quad(quad_r, quad_z, 1);
  const double corner_j_2 = corner_jacobian_from_quad(quad_r, quad_z, 2);
  result.min_offaxis_corner_j = std::min(corner_j_1, corner_j_2);

  const auto reject_after_mutation = [&](const char* const reason) {
    *contact_slot = slot_before;
    result.reject_reason = reason;
    return result;
  };
  if (!audit_equal(momentum_r_before, momentum_r_after)) {
    return reject_after_mutation("momentum_r_audit");
  }
  if (!audit_equal(momentum_z_before, momentum_z_after)) {
    return reject_after_mutation("momentum_z_audit");
  }
  if (!audit_equal(kinetic_before, kinetic_and_heat_after)) {
    return reject_after_mutation("energy_audit");
  }
  if (!audit_equal(corner_mass_sum_before, corner_mass_sum_after)) {
    return reject_after_mutation("mass_audit");
  }
  if (!std::isfinite(result.tri_volume) || result.tri_volume == 0.0 ||
      !std::isfinite(pre_collapse_volume) || pre_collapse_volume == 0.0 ||
      result.tri_volume * pre_collapse_volume <= 0.0) {
    return reject_after_mutation("geometry_audit");
  }
  if (!(corner_j_1 > 0.0) || !(corner_j_2 > 0.0) ||
      !std::isfinite(corner_j_1) || !std::isfinite(corner_j_2)) {
    return reject_after_mutation("offaxis_corner_j_audit");
  }

  state.x_r.copy_from_host(x_r);
  state.x_z.copy_from_host(x_z);
  state.v_r.copy_from_host(v_r);
  state.v_z.copy_from_host(v_z);
  state.corner_mass.copy_from_host(corner_mass);
  state.ei.copy_from_host(ei);

  auto& node_axis_alias = state.evacuated_cells.node_axis_alias;
  if (node_axis_alias.empty()) {
    node_axis_alias.assign(static_cast<std::size_t>(n_nodes), -1);
  }
  TENRYU_ASSERT(node_axis_alias.size() == static_cast<std::size_t>(n_nodes),
                "axis-edge collapse node alias size mismatch");
  auto& cell_axis_edge_collapsed =
      state.evacuated_cells.cell_axis_edge_collapsed;
  if (cell_axis_edge_collapsed.empty()) {
    cell_axis_edge_collapsed.assign(static_cast<std::size_t>(n_cells), 0U);
  }
  TENRYU_ASSERT(
      cell_axis_edge_collapsed.size() == static_cast<std::size_t>(n_cells),
      "axis-edge collapse cell mask size mismatch");
  node_axis_alias[retired] = survivor_node;
  for (int& alias_target : node_axis_alias) {
    if (alias_target == retired_node) {
      alias_target = survivor_node;
    }
  }
  cell_axis_edge_collapsed[static_cast<std::size_t>(failing_cell)] = 1U;

  rebuild_axis_edge_collapse_device_state(state);
  // The transaction updates pair state without changing the engaged set, so
  // refresh the device contact buffers explicitly.
  evacuated_cell_upload_contact_device_state(state);
  monitor.repair_step = state.step;
  monitor.cell = -1;
  result.committed = true;
  flank_strip_capture_reference(state, cfg);
  return result;
}

}  // namespace tenryu::hydro
