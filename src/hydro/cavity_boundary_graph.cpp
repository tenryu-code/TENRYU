#include "hydro/cavity_boundary_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "core/cavity_boundary_graph.hpp"
#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/cavity_boundary_shadow_detect.hpp"

namespace tenryu::hydro {
namespace {

bool cavity_void_cell(const core::EvacuatedCellsState& evacuated,
                      const int cell) {
  const std::size_t index = static_cast<std::size_t>(cell);
  return (index < evacuated.inactive_member_mask.size() &&
          evacuated.inactive_member_mask[index] != 0U) ||
         (index < evacuated.cell_axis_edge_collapsed.size() &&
          evacuated.cell_axis_edge_collapsed[index] != 0U);
}

class VoidRegionUnionFind {
 public:
  explicit VoidRegionUnionFind(const std::size_t size)
      : parent_(size, -1) {}

  void make_set(const int cell) {
    parent_[static_cast<std::size_t>(cell)] = cell;
  }

  int find(const int cell) {
    int root = cell;
    while (parent_[static_cast<std::size_t>(root)] != root) {
      root = parent_[static_cast<std::size_t>(root)];
    }
    int current = cell;
    while (current != root) {
      const int next = parent_[static_cast<std::size_t>(current)];
      parent_[static_cast<std::size_t>(current)] = root;
      current = next;
    }
    return root;
  }

  void unite(const int lhs, const int rhs) {
    const int lhs_root = find(lhs);
    const int rhs_root = find(rhs);
    if (lhs_root == rhs_root) {
      return;
    }
    const int root = std::min(lhs_root, rhs_root);
    const int child = std::max(lhs_root, rhs_root);
    parent_[static_cast<std::size_t>(child)] = root;
  }

 private:
  std::vector<int> parent_;
};

void reserve_from_previous(core::CavityBoundaryGraph& graph,
                           const core::CavityBoundaryGraph& previous) {
  graph.seg_cell.reserve(previous.seg_cell.size());
  graph.seg_face.reserve(previous.seg_face.size());
  graph.seg_void_cell.reserve(previous.seg_void_cell.size());
  graph.seg_node_a.reserve(previous.seg_node_a.size());
  graph.seg_node_b.reserve(previous.seg_node_b.size());
  graph.seg_next.reserve(previous.seg_next.size());
  graph.seg_prev.reserve(previous.seg_prev.size());
  graph.seg_component.reserve(previous.seg_component.size());
  graph.seg_tangent_r.reserve(previous.seg_tangent_r.size());
  graph.seg_tangent_z.reserve(previous.seg_tangent_z.size());
  graph.seg_normal_r.reserve(previous.seg_normal_r.size());
  graph.seg_normal_z.reserve(previous.seg_normal_z.size());
  graph.seg_length.reserve(previous.seg_length.size());
  graph.boundary_nodes.reserve(previous.boundary_nodes.size());
  graph.nonmanifold_nodes.reserve(previous.nonmanifold_nodes.size());
  graph.segment_ids.reserve(previous.segment_ids.size());
  graph.shadow_best_seg.reserve(previous.shadow_best_seg.size());
  graph.shadow_best_xi.reserve(previous.shadow_best_xi.size());
  graph.shadow_best_gap.reserve(previous.shadow_best_gap.size());
  graph.shadow_prev_best_seg.reserve(previous.shadow_prev_best_seg.size());
}

std::int32_t select_same_region_segment(
    const std::vector<std::int32_t>& candidates,
    const std::int32_t source_segment,
    const std::vector<int>& segment_region,
    const core::CavityBoundaryGraph& graph) {
  std::int32_t selected = -1;
  bool selected_same_cell = false;
  const std::size_t source = static_cast<std::size_t>(source_segment);
  for (const std::int32_t candidate_segment : candidates) {
    const std::size_t candidate =
        static_cast<std::size_t>(candidate_segment);
    if (segment_region[candidate] != segment_region[source]) {
      continue;
    }
    const bool same_cell = graph.seg_cell[candidate] == graph.seg_cell[source];
    if (selected < 0 || (same_cell && !selected_same_cell) ||
        (same_cell == selected_same_cell &&
         graph.segment_ids[candidate] <
             graph.segment_ids[static_cast<std::size_t>(selected)])) {
      selected = candidate_segment;
      selected_same_cell = same_cell;
    }
  }
  return selected;
}

}  // namespace

void rebuild_cavity_boundary_graph(core::State& state,
                                   const core::Config& cfg) {
  const auto& topo = state.mesh.topo;
  if (cfg.main.dim != 2 ||
      cfg.mesh.topology_scheme != core::TopologyScheme::SINGLE_BLOCK ||
      topo.multiblock.has_value() || topo.nr <= 0 || topo.nz <= 0 ||
      topo.n_cells != topo.nr * topo.nz) {
    return;
  }

  const int nr = topo.nr;
  const int nz = topo.nz;
  const int n_cells = topo.n_cells;
  const auto& evacuated = state.evacuated_cells;
  std::vector<std::uint8_t> void_cells(static_cast<std::size_t>(n_cells), 0U);
  VoidRegionUnionFind void_regions(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    if (!cavity_void_cell(evacuated, cell)) {
      continue;
    }
    void_cells[static_cast<std::size_t>(cell)] = 1U;
    void_regions.make_set(cell);
    const int i = cell / nz;
    const int j = cell - i * nz;
    if (i > 0 && void_cells[static_cast<std::size_t>(cell - nz)] != 0U) {
      void_regions.unite(cell, cell - nz);
    }
    if (j > 0 && void_cells[static_cast<std::size_t>(cell - 1)] != 0U) {
      void_regions.unite(cell, cell - 1);
    }
  }

  auto& previous = state.cavity_boundary;
  core::CavityBoundaryGraph next;
  next.topology_epoch = previous.topology_epoch;
  next.shadow_pair_migrations = previous.shadow_pair_migrations;
  next.shadow_crossing_events = previous.shadow_crossing_events;
  reserve_from_previous(next, previous);

  for (int cell = 0; cell < n_cells; ++cell) {
    if (void_cells[static_cast<std::size_t>(cell)] != 0U) {
      continue;
    }
    const int i = cell / nz;
    const int j = cell - i * nz;
    for (int face = 0; face < 4; ++face) {
      int neighbor = -1;
      int node_a = -1;
      int node_b = -1;
      if (face == 0 && j > 0) {
        neighbor = cell - 1;
        node_a = i * (nz + 1) + j;
        node_b = (i + 1) * (nz + 1) + j;
      } else if (face == 1 && i + 1 < nr) {
        neighbor = cell + nz;
        node_a = (i + 1) * (nz + 1) + j;
        node_b = (i + 1) * (nz + 1) + j + 1;
      } else if (face == 2 && j + 1 < nz) {
        neighbor = cell + 1;
        node_a = (i + 1) * (nz + 1) + j + 1;
        node_b = i * (nz + 1) + j + 1;
      } else if (face == 3 && i > 0) {
        neighbor = cell - nz;
        node_a = i * (nz + 1) + j + 1;
        node_b = i * (nz + 1) + j;
      }
      if (neighbor < 0 ||
          void_cells[static_cast<std::size_t>(neighbor)] == 0U) {
        continue;
      }
      next.seg_cell.push_back(static_cast<std::int32_t>(cell));
      next.seg_face.push_back(static_cast<std::uint8_t>(face));
      next.seg_void_cell.push_back(static_cast<std::int32_t>(neighbor));
      next.seg_node_a.push_back(static_cast<std::int32_t>(node_a));
      next.seg_node_b.push_back(static_cast<std::int32_t>(node_b));
      next.segment_ids.push_back(
          static_cast<std::int32_t>(cell * 4 + face));
    }
  }

  const std::size_t n_segments = next.seg_cell.size();
  std::vector<double> node_r;
  std::vector<double> node_z;
  if (n_segments > 0U) {
    state.x_r.copy_to_host(node_r);
    state.x_z.copy_to_host(node_z);
    next.seg_tangent_r.reserve(n_segments);
    next.seg_tangent_z.reserve(n_segments);
    next.seg_normal_r.reserve(n_segments);
    next.seg_normal_z.reserve(n_segments);
    next.seg_length.reserve(n_segments);
    for (std::size_t segment = 0; segment < n_segments; ++segment) {
      const std::size_t node_a =
          static_cast<std::size_t>(next.seg_node_a[segment]);
      const std::size_t node_b =
          static_cast<std::size_t>(next.seg_node_b[segment]);
      const double delta_r = node_r[node_b] - node_r[node_a];
      const double delta_z = node_z[node_b] - node_z[node_a];
      const double length = std::hypot(delta_r, delta_z);
      const double tangent_r = length == 0.0 ? 0.0 : delta_r / length;
      const double tangent_z = length == 0.0 ? 0.0 : delta_z / length;
      next.seg_tangent_r.push_back(tangent_r);
      next.seg_tangent_z.push_back(tangent_z);
      next.seg_normal_r.push_back(length == 0.0 ? 0.0 : tangent_z);
      next.seg_normal_z.push_back(length == 0.0 ? 0.0 : -tangent_r);
      next.seg_length.push_back(length);
    }
  }

  next.seg_next.assign(n_segments, -1);
  next.seg_prev.assign(n_segments, -1);
  next.seg_component.assign(n_segments, -1);
  std::vector<std::vector<std::int32_t>> outgoing(
      static_cast<std::size_t>(topo.n_nodes));
  std::vector<std::vector<std::int32_t>> incoming(
      static_cast<std::size_t>(topo.n_nodes));
  std::vector<int> segment_region(n_segments, -1);
  for (std::size_t segment = 0; segment < n_segments; ++segment) {
    const std::int32_t segment_index =
        static_cast<std::int32_t>(segment);
    const std::size_t node_a =
        static_cast<std::size_t>(next.seg_node_a[segment]);
    const std::size_t node_b =
        static_cast<std::size_t>(next.seg_node_b[segment]);
    outgoing[node_a].push_back(segment_index);
    incoming[node_b].push_back(segment_index);
    next.boundary_nodes.push_back(next.seg_node_a[segment]);
    next.boundary_nodes.push_back(next.seg_node_b[segment]);
    segment_region[segment] =
        void_regions.find(next.seg_void_cell[segment]);
  }
  std::sort(next.boundary_nodes.begin(), next.boundary_nodes.end());
  next.boundary_nodes.erase(
      std::unique(next.boundary_nodes.begin(), next.boundary_nodes.end()),
      next.boundary_nodes.end());

  for (std::size_t node = 0; node < outgoing.size(); ++node) {
    if (outgoing[node].size() > 1U || incoming[node].size() > 1U) {
      next.nonmanifold_nodes.push_back(static_cast<std::int32_t>(node));
    }
  }
  for (std::size_t segment = 0; segment < n_segments; ++segment) {
    const std::int32_t segment_index =
        static_cast<std::int32_t>(segment);
    const std::size_t node_a =
        static_cast<std::size_t>(next.seg_node_a[segment]);
    const std::size_t node_b =
        static_cast<std::size_t>(next.seg_node_b[segment]);
    next.seg_next[segment] = select_same_region_segment(
        outgoing[node_b], segment_index, segment_region, next);
    next.seg_prev[segment] = select_same_region_segment(
        incoming[node_a], segment_index, segment_region, next);
    if (next.seg_next[segment] < 0) {
      ++next.open_end_count;
    }
  }

  std::map<int, std::int32_t> minimum_segment_id_by_region;
  for (std::size_t segment = 0; segment < n_segments; ++segment) {
    const int region = segment_region[segment];
    const std::int32_t segment_id = next.segment_ids[segment];
    const auto [it, inserted] =
        minimum_segment_id_by_region.emplace(region, segment_id);
    if (!inserted && segment_id < it->second) {
      it->second = segment_id;
    }
  }
  next.component_count = static_cast<std::int32_t>(
      minimum_segment_id_by_region.size());
  for (std::size_t segment = 0; segment < n_segments; ++segment) {
    next.seg_component[segment] =
        minimum_segment_id_by_region.at(segment_region[segment]);
  }

  const bool topology_changed = next.segment_ids != previous.segment_ids;
  if (topology_changed) {
    ++next.topology_epoch;
  }
  const std::vector<std::int32_t> prev_nodes = previous.boundary_nodes;
  const std::vector<std::int32_t> prev_best_ids =
      previous.shadow_prev_best_seg;
  state.cavity_boundary = std::move(next);
  auto& contact_graph = state.contact_graph;
  contact_graph.record_boundary_valid.assign(contact_graph.records.size(), 0U);
  std::size_t slots_on_boundary = 0U;
  for (std::size_t record = 0; record < contact_graph.records.size();
       ++record) {
    const core::EvacContactSlot& slot = contact_graph.records[record];
    const auto& boundary_nodes = state.cavity_boundary.boundary_nodes;
    const bool boundary_valid =
        std::binary_search(boundary_nodes.begin(), boundary_nodes.end(),
                           slot.node_a[0]) &&
        std::binary_search(boundary_nodes.begin(), boundary_nodes.end(),
                           slot.node_a[1]) &&
        std::binary_search(boundary_nodes.begin(), boundary_nodes.end(),
                           slot.node_b[0]) &&
        std::binary_search(boundary_nodes.begin(), boundary_nodes.end(),
                           slot.node_b[1]);
    if (boundary_valid) {
      contact_graph.record_boundary_valid[record] = 1U;
      ++slots_on_boundary;
    }
  }
  if (topology_changed) {
    const auto& graph = state.cavity_boundary;
    core::log_info(
        "[cavity-boundary] step=" + std::to_string(state.step) +
        " epoch=" + std::to_string(graph.topology_epoch) +
        " segments=" + std::to_string(graph.segment_ids.size()) +
        " nodes=" + std::to_string(graph.boundary_nodes.size()) +
        " components=" + std::to_string(graph.component_count) +
        " open_ends=" + std::to_string(graph.open_end_count) +
        " nonmanifold=" + std::to_string(graph.nonmanifold_nodes.size()) +
        " slots_on_boundary=" + std::to_string(slots_on_boundary) + "/" +
        std::to_string(contact_graph.records.size()));
  }
  if (n_segments > 0U) {
    shadow_detect_cavity_contacts(state, node_r, node_z, prev_nodes,
                                  prev_best_ids, topology_changed);
  } else {
    auto& graph = state.cavity_boundary;
    graph.shadow_best_seg.clear();
    graph.shadow_best_xi.clear();
    graph.shadow_best_gap.clear();
    graph.shadow_prev_best_seg.clear();
  }
}

}  // namespace tenryu::hydro
