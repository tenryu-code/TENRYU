#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {

// Oriented material-void boundary graph (shadow, observational).
// Segment s is a directed half-edge with MATERIAL on the LEFT and void on the
// right; the unit normal (normal_r, normal_z) = (t_z, -t_r) points from
// material into void. Persistent SegmentID = cell * 4 + face with the face
// numbering below; IDs survive topology epochs while the (cell, face) pair
// remains on the boundary.
struct CavityBoundaryGraph {
  // face numbering: 0 = z-low, 1 = r-high, 2 = z-high, 3 = r-low
  std::vector<std::int32_t> seg_cell;  // material cell index
  std::vector<std::uint8_t> seg_face;  // 0..3
  std::vector<std::int32_t> seg_void_cell;  // void neighbor across (cell, face)
  std::vector<std::int32_t> seg_node_a;  // directed a -> b (mesh node indices)
  std::vector<std::int32_t> seg_node_b;
  // Segment index whose node_a == this node_b, -1 open.
  std::vector<std::int32_t> seg_next;
  // Segment index whose node_b == this node_a, -1 open.
  std::vector<std::int32_t> seg_prev;
  // Component label = min SegmentID in component.
  std::vector<std::int32_t> seg_component;
  std::vector<double> seg_tangent_r, seg_tangent_z;  // unit tangent a->b
  // (t_z, -t_r), material -> void.
  std::vector<double> seg_normal_r, seg_normal_z;
  std::vector<double> seg_length;
  std::vector<std::int32_t> boundary_nodes;  // sorted unique mesh node indices
  // Sorted; nodes with >1 outgoing or >1 incoming.
  std::vector<std::int32_t> nonmanifold_nodes;
  std::vector<std::int32_t> segment_ids;  // cell*4+face, ascending (canonical order)
  // Shadow point-to-segment detection (observational). Parallel to
  // boundary_nodes. best_seg = -1 when no admissible candidate exists.
  std::vector<std::int32_t> shadow_best_seg;
  std::vector<double> shadow_best_xi;
  std::vector<double> shadow_best_gap;
  std::vector<std::int32_t> shadow_prev_best_seg;  // by node id, from previous step
  std::uint64_t shadow_pair_migrations = 0;   // cumulative best-seg changes
  std::uint64_t shadow_crossing_events = 0;   // cumulative crossing detections
  std::uint64_t topology_epoch = 0;
  std::int32_t component_count = 0;
  std::int32_t open_end_count = 0;  // segments with seg_next == -1

  void clear() {
    seg_cell.clear();
    seg_face.clear();
    seg_void_cell.clear();
    seg_node_a.clear();
    seg_node_b.clear();
    seg_next.clear();
    seg_prev.clear();
    seg_component.clear();
    seg_tangent_r.clear();
    seg_tangent_z.clear();
    seg_normal_r.clear();
    seg_normal_z.clear();
    seg_length.clear();
    boundary_nodes.clear();
    nonmanifold_nodes.clear();
    segment_ids.clear();
    shadow_best_seg.clear();
    shadow_best_xi.clear();
    shadow_best_gap.clear();
    shadow_prev_best_seg.clear();
    component_count = 0;
    open_end_count = 0;
  }
};

}  // namespace tenryu::core
