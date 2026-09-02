#include "hydro/cavity_boundary_shadow_detect.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/cavity_boundary_graph.hpp"
#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
namespace {

double orientation(const double ar,
                   const double az,
                   const double br,
                   const double bz,
                   const double cr,
                   const double cz) {
  return (br - ar) * (cz - az) - (bz - az) * (cr - ar);
}

bool strict_opposite_signs(const double lhs, const double rhs) {
  return (lhs < 0.0 && rhs > 0.0) || (lhs > 0.0 && rhs < 0.0);
}

bool proper_segment_intersection(
    const std::size_t segment_a,
    const std::size_t segment_b,
    const core::CavityBoundaryGraph& graph,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const std::size_t a =
      static_cast<std::size_t>(graph.seg_node_a[segment_a]);
  const std::size_t b =
      static_cast<std::size_t>(graph.seg_node_b[segment_a]);
  const std::size_t c =
      static_cast<std::size_t>(graph.seg_node_a[segment_b]);
  const std::size_t d =
      static_cast<std::size_t>(graph.seg_node_b[segment_b]);
  const double abc = orientation(node_r[a], node_z[a], node_r[b], node_z[b],
                                 node_r[c], node_z[c]);
  const double abd = orientation(node_r[a], node_z[a], node_r[b], node_z[b],
                                 node_r[d], node_z[d]);
  const double cda = orientation(node_r[c], node_z[c], node_r[d], node_z[d],
                                 node_r[a], node_z[a]);
  const double cdb = orientation(node_r[c], node_z[c], node_r[d], node_z[d],
                                 node_r[b], node_z[b]);
  return strict_opposite_signs(abc, abd) && strict_opposite_signs(cda, cdb);
}

void mark_polyline_exclusions(const std::int32_t source_segment,
                              const core::CavityBoundaryGraph& graph,
                              std::vector<std::uint8_t>& excluded) {
  excluded[static_cast<std::size_t>(source_segment)] = 1U;
  std::int32_t segment = source_segment;
  for (int hop = 0; hop < 2; ++hop) {
    segment = graph.seg_next[static_cast<std::size_t>(segment)];
    if (segment < 0) {
      break;
    }
    excluded[static_cast<std::size_t>(segment)] = 1U;
  }
  segment = source_segment;
  for (int hop = 0; hop < 2; ++hop) {
    segment = graph.seg_prev[static_cast<std::size_t>(segment)];
    if (segment < 0) {
      break;
    }
    excluded[static_cast<std::size_t>(segment)] = 1U;
  }
}

}  // namespace

void shadow_detect_cavity_contacts(
    core::State& state,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::int32_t>& prev_nodes,
    const std::vector<std::int32_t>& prev_best_ids,
    const bool topology_changed) {
  auto& graph = state.cavity_boundary;
  const std::size_t n_nodes = graph.boundary_nodes.size();
  const std::size_t n_segments = graph.segment_ids.size();
  graph.shadow_best_seg.assign(n_nodes, -1);
  graph.shadow_best_xi.assign(n_nodes, 0.0);
  graph.shadow_best_gap.assign(
      n_nodes, std::numeric_limits<double>::infinity());
  graph.shadow_prev_best_seg.assign(n_nodes, -1);

  std::vector<std::uint8_t> excluded(n_segments, 0U);
  for (std::size_t node_index = 0; node_index < n_nodes; ++node_index) {
    std::fill(excluded.begin(), excluded.end(), 0U);
    const std::int32_t slave_node = graph.boundary_nodes[node_index];
    double slave_normal_r = 0.0;
    double slave_normal_z = 0.0;
    for (std::size_t segment = 0; segment < n_segments; ++segment) {
      if (graph.seg_node_a[segment] == slave_node ||
          graph.seg_node_b[segment] == slave_node) {
        slave_normal_r += graph.seg_normal_r[segment];
        slave_normal_z += graph.seg_normal_z[segment];
        mark_polyline_exclusions(static_cast<std::int32_t>(segment), graph,
                                 excluded);
      }
    }
    const double slave_normal_magnitude =
        std::hypot(slave_normal_r, slave_normal_z);
    if (slave_normal_magnitude > 0.0) {
      slave_normal_r /= slave_normal_magnitude;
      slave_normal_z /= slave_normal_magnitude;
    } else {
      slave_normal_r = 0.0;
      slave_normal_z = 0.0;
    }

    const std::size_t slave = static_cast<std::size_t>(slave_node);
    for (std::size_t segment = 0; segment < n_segments; ++segment) {
      if (graph.seg_node_a[segment] == slave_node ||
          graph.seg_node_b[segment] == slave_node ||
          excluded[segment] != 0U || graph.seg_length[segment] == 0.0) {
        continue;
      }
      const std::size_t node_a =
          static_cast<std::size_t>(graph.seg_node_a[segment]);
      const std::size_t node_b =
          static_cast<std::size_t>(graph.seg_node_b[segment]);
      const double delta_r = node_r[node_b] - node_r[node_a];
      const double delta_z = node_z[node_b] - node_z[node_a];
      const double length_squared = delta_r * delta_r + delta_z * delta_z;
      const double xi_unclamped =
          ((node_r[slave] - node_r[node_a]) * delta_r +
           (node_z[slave] - node_z[node_a]) * delta_z) /
          length_squared;
      if (xi_unclamped < 0.0 || xi_unclamped > 1.0) {
        continue;
      }
      const double xi_star = xi_unclamped;
      const double master_r =
          (1.0 - xi_star) * node_r[node_a] + xi_star * node_r[node_b];
      const double master_z =
          (1.0 - xi_star) * node_z[node_a] + xi_star * node_z[node_b];
      const double gap =
          graph.seg_normal_r[segment] * (node_r[slave] - master_r) +
          graph.seg_normal_z[segment] * (node_z[slave] - master_z);
      // Deep-negative gaps are shielded by intervening material.
      if (gap <= -2.0 * graph.seg_length[segment]) {
        continue;
      }
      // Contact requires facing surfaces; same-facing segments of the own wall are not candidates.
      if (slave_normal_magnitude > 0.0 &&
          graph.seg_normal_r[segment] * slave_normal_r +
                  graph.seg_normal_z[segment] * slave_normal_z >
              0.0) {
        continue;
      }
      const std::int32_t best_segment = graph.shadow_best_seg[node_index];
      if (best_segment < 0 || gap < graph.shadow_best_gap[node_index] ||
          (gap == graph.shadow_best_gap[node_index] &&
           graph.segment_ids[segment] <
               graph.segment_ids[static_cast<std::size_t>(best_segment)])) {
        graph.shadow_best_seg[node_index] =
            static_cast<std::int32_t>(segment);
        graph.shadow_best_xi[node_index] = xi_star;
        graph.shadow_best_gap[node_index] = gap;
      }
    }

    const std::int32_t best_segment = graph.shadow_best_seg[node_index];
    const std::int32_t current_best_id =
        best_segment < 0
            ? -1
            : graph.segment_ids[static_cast<std::size_t>(best_segment)];
    graph.shadow_prev_best_seg[node_index] = current_best_id;
    const auto previous =
        std::lower_bound(prev_nodes.begin(), prev_nodes.end(), slave_node);
    if (previous != prev_nodes.end() && *previous == slave_node) {
      const std::size_t previous_index =
          static_cast<std::size_t>(previous - prev_nodes.begin());
      if (previous_index < prev_best_ids.size() &&
          prev_best_ids[previous_index] != current_best_id) {
        ++graph.shadow_pair_migrations;
      }
    }
  }

  std::size_t crossing_logs = 0U;
  for (std::size_t first = 0; first < n_segments; ++first) {
    for (std::size_t second = first + 1U; second < n_segments; ++second) {
      if (graph.seg_node_a[first] == graph.seg_node_a[second] ||
          graph.seg_node_a[first] == graph.seg_node_b[second] ||
          graph.seg_node_b[first] == graph.seg_node_a[second] ||
          graph.seg_node_b[first] == graph.seg_node_b[second]) {
        continue;
      }
      if (!proper_segment_intersection(first, second, graph, node_r, node_z)) {
        continue;
      }
      ++graph.shadow_crossing_events;
      if (crossing_logs < 4U) {
        core::log_warning(
            "[cavity-shadow] crossing step=" + std::to_string(state.step) +
            " segA=" + std::to_string(graph.segment_ids[first]) +
            " segB=" + std::to_string(graph.segment_ids[second]));
        ++crossing_logs;
      }
    }
  }

  std::size_t divergence_logs = 0U;
  const auto& contact_graph = state.contact_graph;
  for (std::size_t record_index = 0;
       record_index < contact_graph.records.size(); ++record_index) {
    if (record_index >= contact_graph.record_boundary_valid.size() ||
        contact_graph.record_boundary_valid[record_index] != 1U) {
      continue;
    }
    const core::EvacContactSlot& record =
        contact_graph.records[record_index];
    for (int pair = 0; pair < 2; ++pair) {
      const auto node = std::lower_bound(graph.boundary_nodes.begin(),
                                         graph.boundary_nodes.end(),
                                         record.node_b[pair]);
      if (node == graph.boundary_nodes.end() || *node != record.node_b[pair]) {
        continue;
      }
      const std::size_t node_index =
          static_cast<std::size_t>(node - graph.boundary_nodes.begin());
      const std::size_t node_a =
          static_cast<std::size_t>(record.node_a[pair]);
      const std::size_t node_b =
          static_cast<std::size_t>(record.node_b[pair]);
      const double legacy_gap =
          (node_r[node_b] - node_r[node_a]) * record.normal_pair_r[pair] +
          (node_z[node_b] - node_z[node_a]) * record.normal_pair_z[pair];
      const double shadow_gap = graph.shadow_best_gap[node_index];
      if (std::abs(legacy_gap - shadow_gap) <=
              0.5 * std::max(std::abs(legacy_gap), std::abs(shadow_gap)) ||
          std::min(std::abs(legacy_gap), std::abs(shadow_gap)) >=
              record.g_arm * 4.0) {
        continue;
      }
      if (divergence_logs < 4U) {
        const std::int32_t best_segment = graph.shadow_best_seg[node_index];
        const std::int32_t best_id =
            best_segment < 0
                ? -1
                : graph.segment_ids[static_cast<std::size_t>(best_segment)];
        std::ostringstream line;
        line << std::scientific << std::setprecision(17)
             << "[cavity-shadow] divergence step=" << state.step
             << " cell=" << record.cell << " pair=" << pair
             << " legacy=" << legacy_gap << " shadow=" << shadow_gap
             << " seg=" << best_id << " xi=" << graph.shadow_best_xi[node_index];
        core::log_info(line.str());
        ++divergence_logs;
      }
    }
  }

  if (n_nodes > 0U &&
      (state.step % 5000 == 0 || topology_changed)) {
    std::size_t minimum_node_index = 0U;
    for (std::size_t node_index = 1U; node_index < n_nodes; ++node_index) {
      if (graph.shadow_best_gap[node_index] <
          graph.shadow_best_gap[minimum_node_index]) {
        minimum_node_index = node_index;
      }
    }
    std::ostringstream line;
    line << std::scientific << std::setprecision(17)
         << "[cavity-shadow] step=" << state.step << " nodes=" << n_nodes
         << " min_gap=" << graph.shadow_best_gap[minimum_node_index]
         << " min_gap_node=" << graph.boundary_nodes[minimum_node_index]
         << " migrations=" << graph.shadow_pair_migrations
         << " crossings=" << graph.shadow_crossing_events;
    core::log_info(line.str());
  }
}

}  // namespace tenryu::hydro
