#include "hydro/central_pseudo_core.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "core/error.hpp"
#include "diagnostics/diagnostics.hpp"
#include "hydro/central_core_1d_submodel.hpp"
#include "hydro/conservation_audit.hpp"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"
#include "mesh/path_admissibility.cuh"

namespace tenryu::hydro::central_pseudo_core {
namespace {

constexpr double kFourPiOverThree =
    4.1887902047863909846168578443726705122628925325000;

constexpr std::uint8_t kActiveOrdinary =
    static_cast<std::uint8_t>(
        core::CentralPseudoCoreActiveClass::ACTIVE_ORDINARY);
constexpr std::uint8_t kInactiveCoreMember =
    static_cast<std::uint8_t>(
        core::CentralPseudoCoreActiveClass::INACTIVE_CORE_MEMBER);

inline void cuda_check(const cudaError_t err, const char* const message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool is_supported_topology(const core::Config& cfg) {
  return cfg.main.dimension == "2D_RZ" &&
         (cfg.mesh.topology_scheme ==
              core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_5BLOCK ||
          cfg.mesh.topology_scheme ==
              core::TopologyScheme::MULTIBLOCK_HALF_BUTTERFLY_TRIFAN_CAP_5BLOCK ||
          (cfg.mesh.topology_scheme ==
               core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER &&
           cfg.mesh.polar_tier_center_kind == "trifan_cap"));
}

struct CapTopoView {
  bool has_cap = false;  // Pseudo-core-admissible cap exists.
  bool hybrid = false;   // Cap-center hybrid (BRIDGE-row growth).
  int n_cap = 0;         // Cap ring count.
  int ntheta = 0;        // Angular cells (== 4*n_cap, asserted).
};

int active_nverts(const mesh::Mesh& mesh, const int c) {
  if (mesh.cell_nverts.size() == static_cast<std::size_t>(mesh.topo.n_cells)) {
    return mesh::mesh_topo_cell_active_nverts(mesh.cell_nverts, c);
  }
  return mesh::kMeshTopoCellStorageSlots;
}

int central_core_block_id(const mesh::MultiBlockTopology& mb) {
  int block_id = -1;
  for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
    if (mb.blocks[static_cast<std::size_t>(b)].role ==
        mesh::BlockRole::CENTRAL_CORE) {
      TENRYU_ASSERT(block_id < 0,
                    "central pseudo-core found multiple central-core blocks");
      block_id = b;
    }
  }
  TENRYU_ASSERT(block_id >= 0,
                "central pseudo-core requires a central-core block");
  return block_id;
}

CapTopoView make_cap_topo_view(const core::Config& cfg,
                               const mesh::MultiBlockTopology& mb) {
  if (mb.has_trifan_cap) {
    return {true, false, mb.n_cap, 4 * mb.n_cap};
  }
  if (cfg.mesh.topology_scheme ==
          core::TopologyScheme::MULTIBLOCK_POLAR_TIER_CART_CENTER &&
      cfg.mesh.polar_tier_center_kind == "trifan_cap") {
    const int block_id = central_core_block_id(mb);
    const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
    TENRYU_ASSERT(block.n_j_cells == 4 * block.n_i_cells,
                  "central pseudo-core hybrid cap angular count mismatch");
    return {true, true, block.n_i_cells, block.n_j_cells};
  }
  return {};
}

int block_cell_id(const mesh::BlockInfo& block, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i < block.n_i_cells,
                "central pseudo-core block i index out of range");
  TENRYU_ASSERT(j >= 0 && j < block.n_j_cells,
                "central pseudo-core block j index out of range");
  return block.cell_begin + i * block.n_j_cells + j;
}

const mesh::BlockInfo& hybrid_bridge_block(
    const mesh::MultiBlockTopology& mb,
    const CapTopoView& view) {
  TENRYU_ASSERT(view.hybrid,
                "central pseudo-core hybrid bridge requires hybrid cap");
  const mesh::BlockInfo* bridge = nullptr;
  for (const auto& block : mb.blocks) {
    if (block.role == mesh::BlockRole::BRIDGE) {
      TENRYU_ASSERT(bridge == nullptr,
                    "central pseudo-core found multiple hybrid bridge blocks");
      bridge = &block;
    }
  }
  TENRYU_ASSERT(bridge != nullptr && bridge->n_i_cells > 0 &&
                    bridge->n_j_cells == view.ntheta &&
                    bridge->cell_count ==
                        bridge->n_i_cells * bridge->n_j_cells &&
                    bridge->owned_node_count ==
                        bridge->n_i_cells * (view.ntheta + 1),
                "central pseudo-core hybrid bridge layout mismatch");
  return *bridge;
}

double bridge_row_mean_node_radius(
    const mesh::MultiBlockTopology& mb,
    const mesh::BlockInfo& bridge,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const int row) {
  TENRYU_ASSERT(row >= 0 && row < bridge.n_i_cells,
                "central pseudo-core hybrid bridge row out of range");
  long double radius_sum = 0.0L;
  int count = 0;
  for (int j = 0; j < bridge.n_j_cells; ++j) {
    const int c = block_cell_id(bridge, row, j);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
      const int n =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && static_cast<std::size_t>(n) < node_r.size() &&
                        static_cast<std::size_t>(n) < node_z.size(),
                    "central pseudo-core hybrid bridge node out of range");
      const double radius =
          std::hypot(node_r[static_cast<std::size_t>(n)],
                     node_z[static_cast<std::size_t>(n)]);
      TENRYU_ASSERT(std::isfinite(radius),
                    "central pseudo-core hybrid bridge radius must be finite");
      radius_sum += static_cast<long double>(radius);
      ++count;
    }
  }
  TENRYU_ASSERT(count > 0,
                "central pseudo-core hybrid bridge row must contain nodes");
  return static_cast<double>(radius_sum / static_cast<long double>(count));
}

bool hybrid_bridge_row_zero_is_inner(
    const mesh::MultiBlockTopology& mb,
    const mesh::BlockInfo& bridge,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const double row_zero =
      bridge_row_mean_node_radius(mb, bridge, node_r, node_z, 0);
  const double row_last = bridge_row_mean_node_radius(
      mb, bridge, node_r, node_z, bridge.n_i_cells - 1);
  // Fixed tiebreak: row 0 is the inner row when the means are equal.
  return row_zero <= row_last;
}

int hybrid_bridge_storage_row(const mesh::BlockInfo& bridge,
                              const int radial_row,
                              const bool row_zero_is_inner) {
  TENRYU_ASSERT(radial_row >= 0 && radial_row < bridge.n_i_cells,
                "central pseudo-core hybrid bridge radial row out of range");
  return row_zero_is_inner ? radial_row
                           : bridge.n_i_cells - 1 - radial_row;
}

int hybrid_bridge_radial_row(const mesh::BlockInfo& bridge,
                             const int storage_row,
                             const bool row_zero_is_inner) {
  TENRYU_ASSERT(storage_row >= 0 && storage_row < bridge.n_i_cells,
                "central pseudo-core hybrid bridge storage row out of range");
  return row_zero_is_inner ? storage_row
                           : bridge.n_i_cells - 1 - storage_row;
}

int hybrid_bridge_boundary_node_row_index(
    const mesh::MultiBlockTopology& mb,
    const mesh::BlockInfo& bridge,
    const int absorbed_rows,
    const bool row_zero_is_inner) {
  TENRYU_ASSERT(absorbed_rows > 0 && absorbed_rows <= bridge.n_i_cells,
                "central pseudo-core hybrid bridge boundary depth out of range");
  const int cell_row = hybrid_bridge_storage_row(
      bridge, absorbed_rows - 1, row_zero_is_inner);
  const int first_cell = block_cell_id(bridge, cell_row, 0);
  const int first_off =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(first_cell)];
  const int first_slot = row_zero_is_inner ? 3 : 0;
  const int last_slot = row_zero_is_inner ? 2 : 1;
  const int first_node = mb.cell_node_csr_indices[
      static_cast<std::size_t>(first_off + first_slot)];
  const int stride = bridge.n_j_cells + 1;
  const int row_node_index =
      row_zero_is_inner ? bridge.n_i_cells - 1 - absorbed_rows
                        : absorbed_rows - 1;
  const int expected_begin =
      bridge.owned_node_begin + row_node_index * stride;
  TENRYU_ASSERT(first_node == expected_begin,
                "central pseudo-core hybrid bridge node-row orientation mismatch");
  for (int g = 0; g < bridge.n_j_cells; ++g) {
    const int c = block_cell_id(bridge, cell_row, g);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    TENRYU_ASSERT(
        mb.cell_node_csr_indices[static_cast<std::size_t>(off + first_slot)] ==
            expected_begin + g,
        "central pseudo-core hybrid bridge boundary row stride mismatch");
  }
  const int last_cell =
      block_cell_id(bridge, cell_row, bridge.n_j_cells - 1);
  const int last_off =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(last_cell)];
  TENRYU_ASSERT(
      mb.cell_node_csr_indices[static_cast<std::size_t>(last_off + last_slot)] ==
          expected_begin + bridge.n_j_cells,
      "central pseudo-core hybrid bridge boundary row end mismatch");
  return row_node_index;
}

double cell_max_node_s(const mesh::Mesh& mesh,
                       const mesh::MultiBlockTopology& mb,
                       const std::vector<double>& node_r,
                       const std::vector<double>& node_z,
                       const int c) {
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  const int nverts = active_nverts(mesh, c);
  double max_s = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    TENRYU_ASSERT(n >= 0 &&
                      static_cast<std::size_t>(n) < node_r.size() &&
                      static_cast<std::size_t>(n) < node_z.size(),
                  "central pseudo-core cell node out of range");
    max_s = std::max(max_s,
                     std::hypot(node_r[static_cast<std::size_t>(n)],
                                node_z[static_cast<std::size_t>(n)]));
  }
  TENRYU_ASSERT(std::isfinite(max_s),
                "central pseudo-core ring node radius must be finite");
  return max_s;
}

void mark_block_cell(const mesh::MultiBlockTopology& mb,
                     const int block_id,
                     const int c,
                     std::vector<std::uint8_t>& member_mask) {
  TENRYU_ASSERT(c >= 0 &&
                    static_cast<std::size_t>(c) < member_mask.size(),
                "central pseudo-core member cell out of range");
  TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(c)] == block_id,
                "central pseudo-core block-major cell mapping mismatch");
  member_mask[static_cast<std::size_t>(c)] = 1U;
}

int select_trifan_cap_core_rings(const mesh::Mesh& mesh,
                                 const mesh::MultiBlockTopology& mb,
                                 const CapTopoView& view,
                                 const int block_id,
                                 const std::vector<double>& node_r,
                                 const std::vector<double>& node_z,
                                 const double s_c,
                                 const int min_ring_count,
                                 std::vector<std::uint8_t>& member_mask) {
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(block.role == mesh::BlockRole::CENTRAL_CORE,
                "central pseudo-core ring selection requires central core");
  TENRYU_ASSERT(block.n_i_cells == view.n_cap &&
                    block.n_j_cells == view.ntheta &&
                    view.ntheta == 4 * view.n_cap,
                "central pseudo-core trifan cap block dimensions mismatch");
  const double s_limit =
      std::nextafter(s_c, std::numeric_limits<double>::infinity());
  int selected_rings = 0;
  for (int i = 0; i < block.n_i_cells; ++i) {
    double ring_max_s = 0.0;
    for (int j = 0; j < block.n_j_cells; ++j) {
      ring_max_s =
          std::max(ring_max_s,
                   cell_max_node_s(mesh,
                                   mb,
                                   node_r,
                                   node_z,
                                   block_cell_id(block, i, j)));
    }
    if (ring_max_s <= s_limit) {
      ++selected_rings;
    } else {
      break;
    }
  }
  // Dynamic ring absorption can force additional complete rings beyond the
  // static s_c selection.
  selected_rings =
      std::max(selected_rings, std::min(min_ring_count, block.n_i_cells));

  for (int i = 0; i < selected_rings; ++i) {
    for (int j = 0; j < block.n_j_cells; ++j) {
      mark_block_cell(mb,
                      block_id,
                      block_cell_id(block, i, j),
                      member_mask);
    }
  }
  int depth = selected_rings;
  // Once every cap ring is absorbed, hybrid requests grow through complete
  // BRIDGE rows; non-hybrid requests grow through complete FAN LAYERS and
  // then PURE-GAS polar-shell rows. The material guard lives at the request/
  // trigger decision sites (absorbable_shell_row_count), so the requested
  // depth is trusted here.
  if (selected_rings == block.n_i_cells &&
      min_ring_count > block.n_i_cells) {
    if (view.hybrid) {
      const auto& bridge = hybrid_bridge_block(mb, view);
      const bool row_zero_is_inner = hybrid_bridge_row_zero_is_inner(
          mb, bridge, node_r, node_z);
      const int bridge_rows = std::min(
          min_ring_count - block.n_i_cells, bridge.n_i_cells);
      const int bridge_block_id =
          static_cast<int>(&bridge - mb.blocks.data());
      for (int q = 0; q < bridge_rows; ++q) {
        const int row =
            hybrid_bridge_storage_row(bridge, q, row_zero_is_inner);
        for (int j = 0; j < bridge.n_j_cells; ++j) {
          mark_block_cell(mb,
                          bridge_block_id,
                          block_cell_id(bridge, row, j),
                          member_mask);
        }
      }
      depth += bridge_rows;
      return depth;
    }
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    const int fan_layers = std::min(min_ring_count - block.n_i_cells,
                                    north.n_i_cells);
    for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                      mesh::BlockRole::EAST_FAN,
                                      mesh::BlockRole::SOUTH_FAN}) {
      const auto& fan_block = mesh::mesh_topo_trifan_fan_block(mb, fan);
      TENRYU_ASSERT(fan_block.n_i_cells == north.n_i_cells,
                    "central pseudo-core fan blocks must share layer count");
      const int fan_block_id =
          static_cast<int>(&fan_block - mb.blocks.data());
      for (int l = 0; l < fan_layers; ++l) {
        for (int j = 0; j < fan_block.n_j_cells; ++j) {
          mark_block_cell(mb,
                          fan_block_id,
                          block_cell_id(fan_block, l, j),
                          member_mask);
        }
      }
    }
    depth += fan_layers;
    if (fan_layers == north.n_i_cells &&
        min_ring_count > block.n_i_cells + north.n_i_cells) {
      const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
      const int shell_rows =
          std::min(min_ring_count - block.n_i_cells - north.n_i_cells,
                   shell.n_i_cells - 1);
      const int shell_block_id =
          static_cast<int>(&shell - mb.blocks.data());
      for (int q = 0; q < shell_rows; ++q) {
        for (int j = 0; j < shell.n_j_cells; ++j) {
          mark_block_cell(mb,
                          shell_block_id,
                          block_cell_id(shell, q, j),
                          member_mask);
        }
      }
      depth += shell_rows;
    }
  }
  return depth;
}

int select_centered_core_rings(const mesh::Mesh& mesh,
                               const mesh::MultiBlockTopology& mb,
                               const int block_id,
                               const std::vector<double>& node_r,
                               const std::vector<double>& node_z,
                               const double s_c,
                               const int min_ring_count,
                               std::vector<std::uint8_t>& member_mask) {
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(block.role == mesh::BlockRole::CENTRAL_CORE,
                "central pseudo-core ring selection requires central core");
  TENRYU_ASSERT(block.n_j_cells == 2 * block.n_i_cells,
                "central pseudo-core legacy core must be 2:1 rectangular");
  const double s_limit =
      std::nextafter(s_c, std::numeric_limits<double>::infinity());
  const int j_mid = block.n_j_cells / 2;
  int selected_rings = 0;
  for (int ring_count = 1; ring_count <= block.n_i_cells; ++ring_count) {
    const int j_begin = j_mid - ring_count;
    const int j_end = j_mid + ring_count;
    TENRYU_ASSERT(j_begin >= 0 && j_end <= block.n_j_cells,
                  "central pseudo-core centered ring j range out of range");
    double patch_max_s = 0.0;
    for (int i = 0; i < ring_count; ++i) {
      for (int j = j_begin; j < j_end; ++j) {
        patch_max_s =
            std::max(patch_max_s,
                     cell_max_node_s(mesh,
                                     mb,
                                     node_r,
                                     node_z,
                                     block_cell_id(block, i, j)));
      }
    }
    if (patch_max_s <= s_limit) {
      selected_rings = ring_count;
    } else {
      break;
    }
  }
  selected_rings =
      std::max(selected_rings, std::min(min_ring_count, block.n_i_cells));

  const int j_begin = j_mid - selected_rings;
  const int j_end = j_mid + selected_rings;
  for (int i = 0; i < selected_rings; ++i) {
    for (int j = j_begin; j < j_end; ++j) {
      mark_block_cell(mb,
                      block_id,
                      block_cell_id(block, i, j),
                      member_mask);
    }
  }
  return selected_rings;
}

int select_ring_conforming_members(const mesh::Mesh& mesh,
                                   const mesh::MultiBlockTopology& mb,
                                   const CapTopoView& view,
                                   const std::vector<double>& node_r,
                                   const std::vector<double>& node_z,
                                   const double s_c,
                                   const int min_ring_count,
                                   std::vector<std::uint8_t>& member_mask) {
  const int block_id = central_core_block_id(mb);
  if (view.has_cap) {
    return select_trifan_cap_core_rings(
        mesh, mb, view, block_id, node_r, node_z, s_c, min_ring_count,
        member_mask);
  }
  return select_centered_core_rings(
      mesh, mb, block_id, node_r, node_z, s_c, min_ring_count, member_mask);
}

std::pair<int, int> face_nodes(const int nverts,
                               const std::array<int, 4>& nodes,
                               const int face) {
  if (nverts == 3) {
    switch (face) {
      case 1:
        return {nodes[1], nodes[2]};
      case 2:
        return {nodes[0], nodes[1]};
      case 3:
        return {nodes[2], nodes[0]};
      default:
        return {-1, -1};
    }
  }
  switch (face) {
    case 0:
      return {nodes[0], nodes[3]};
    case 1:
      return {nodes[1], nodes[2]};
    case 2:
      return {nodes[0], nodes[1]};
    case 3:
      return {nodes[3], nodes[2]};
    default:
      return {-1, -1};
  }
}

std::vector<int> order_boundary_nodes(
    const std::vector<std::pair<int, int>>& boundary_edges,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  TENRYU_ASSERT(!boundary_edges.empty(),
                "central pseudo-core boundary has no member/non-member faces");
  std::map<int, std::vector<int>> graph;
  for (const auto& edge : boundary_edges) {
    TENRYU_ASSERT(edge.first >= 0 && edge.second >= 0 && edge.first != edge.second,
                  "central pseudo-core boundary edge is degenerate");
    graph[edge.first].push_back(edge.second);
    graph[edge.second].push_back(edge.first);
  }
  for (const auto& [node, neighbors] : graph) {
    (void)node;
    TENRYU_ASSERT(neighbors.size() == 2U,
                  "central pseudo-core boundary is not one closed chain");
  }

  const int start = graph.begin()->first;
  std::vector<int> ordered;
  ordered.reserve(graph.size());
  int prev = -1;
  int cur = start;
  for (;;) {
    ordered.push_back(cur);
    const auto& nbr = graph[cur];
    const int next = (nbr[0] == prev) ? nbr[1] : nbr[0];
    prev = cur;
    cur = next;
    if (cur == start) {
      break;
    }
    TENRYU_ASSERT(static_cast<int>(ordered.size()) <=
                      static_cast<int>(graph.size()),
                  "central pseudo-core boundary has multiple loops or a branch");
  }
  TENRYU_ASSERT(ordered.size() == graph.size(),
                "central pseudo-core boundary did not consume exactly one loop");

  std::vector<double> r(ordered.size());
  std::vector<double> z(ordered.size());
  for (std::size_t k = 0; k < ordered.size(); ++k) {
    const int n = ordered[k];
    r[k] = node_r[static_cast<std::size_t>(n)];
    z[k] = node_z[static_cast<std::size_t>(n)];
  }
  const double signed_volume =
      rz::rz_polygon_volume_exact(r.data(), z.data(), static_cast<int>(ordered.size()));
  TENRYU_ASSERT(std::isfinite(signed_volume) && signed_volume != 0.0,
                "central pseudo-core boundary has zero signed RZ volume");
  if (signed_volume < 0.0) {
    std::reverse(ordered.begin(), ordered.end());
  }
  return ordered;
}

double boundary_loop_volume(const core::CentralPseudoCoreState& pc,
                            const std::vector<double>& node_r,
                            const std::vector<double>& node_z) {
  const int n_boundary =
      static_cast<int>(pc.boundary_nodes_ordered.size());
  TENRYU_ASSERT(n_boundary >= 3,
                "central pseudo-core macro boundary requires at least 3 nodes");
  std::vector<double> r(static_cast<std::size_t>(n_boundary));
  std::vector<double> z(static_cast<std::size_t>(n_boundary));
  for (int k = 0; k < n_boundary; ++k) {
    const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(k)];
    TENRYU_ASSERT(n >= 0 &&
                      static_cast<std::size_t>(n) < node_r.size() &&
                      static_cast<std::size_t>(n) < node_z.size(),
                  "central pseudo-core boundary node out of range");
    r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
    z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
  }
  const double volume =
      rz::rz_polygon_volume_exact(r.data(), z.data(), n_boundary);
  TENRYU_ASSERT(std::isfinite(volume) && volume > 0.0,
                "central pseudo-core macro boundary volume must be positive");
  return volume;
}

int cap_apex_node_id(const mesh::MultiBlockTopology& mb,
                     const CapTopoView& view) {
  if (!view.hybrid) {
    return mesh::mesh_topo_cap_apex_node_id(mb);
  }
  const int block_id = central_core_block_id(mb);
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  const int cell = block_cell_id(block, 0, 0);
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(cell)] == block_id,
                "central pseudo-core hybrid cap apex cell mapping mismatch");
  return mb.cell_node_csr_indices[static_cast<std::size_t>(off)];
}

int cap_ring_node_id(const mesh::MultiBlockTopology& mb,
                     const CapTopoView& view,
                     const int l,
                     const int k) {
  if (!view.hybrid) {
    return mesh::mesh_topo_cap_ring_node_id(mb, l, k);
  }
  TENRYU_ASSERT(l >= 1 && l <= view.n_cap,
                "central pseudo-core hybrid cap ring layer out of range");
  TENRYU_ASSERT(k >= 0 && k <= view.ntheta,
                "central pseudo-core hybrid cap angle out of range");
  const int block_id = central_core_block_id(mb);
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  const int cell_k = k < view.ntheta ? k : view.ntheta - 1;
  const int cell = block_cell_id(block, l - 1, cell_k);
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  TENRYU_ASSERT(mb.cell_block_id[static_cast<std::size_t>(cell)] == block_id,
                "central pseudo-core hybrid cap ring cell mapping mismatch");
  const int outer_slot = l == 1 ? (k < view.ntheta ? 2 : 1)
                                : (k < view.ntheta ? 3 : 2);
  return mb.cell_node_csr_indices[
      static_cast<std::size_t>(off + outer_slot)];
}

int trifan_member_ring_count(const core::CentralPseudoCoreState& pc,
                             const mesh::MultiBlockTopology& mb,
                             const CapTopoView& view,
                             const std::vector<double>& node_r,
                             const std::vector<double>& node_z,
                             const mesh::BlockInfo** hybrid_bridge_out,
                             bool* bridge_zero_is_inner_out) {
  TENRYU_ASSERT(view.has_cap,
                "central pseudo-core virtual star-map requires trifan cap topology");
  *hybrid_bridge_out = nullptr;
  *bridge_zero_is_inner_out = true;
  const int block_id = central_core_block_id(mb);
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(block.role == mesh::BlockRole::CENTRAL_CORE &&
                    block.n_i_cells == view.n_cap &&
                    block.n_j_cells == view.ntheta &&
                    view.ntheta == 4 * view.n_cap,
                "central pseudo-core trifan cap block dimensions mismatch");

  int selected_rings = 0;
  bool seen_unselected = false;
  for (int i = 0; i < block.n_i_cells; ++i) {
    bool full_ring = true;
    bool any_ring = false;
    for (int j = 0; j < block.n_j_cells; ++j) {
      const int c = block_cell_id(block, i, j);
      TENRYU_ASSERT(c >= 0 &&
                        static_cast<std::size_t>(c) < pc.member_mask.size(),
                    "central pseudo-core trifan member cell out of range");
      const bool is_member =
          pc.member_mask[static_cast<std::size_t>(c)] != 0U;
      full_ring = full_ring && is_member;
      any_ring = any_ring || is_member;
    }
    if (full_ring && !seen_unselected) {
      ++selected_rings;
      continue;
    }
    seen_unselected = true;
    TENRYU_ASSERT(!any_ring,
                  "central pseudo-core trifan members must be full prefix rings");
  }
  TENRYU_ASSERT(selected_rings > 0 && selected_rings <= view.n_cap,
                "central pseudo-core trifan member ring count out of range");
  std::size_t expected_members =
      static_cast<std::size_t>(selected_rings * block.n_j_cells);
  if (view.hybrid) {
    int bridge_rows = 0;
    if (selected_rings == block.n_i_cells &&
        pc.member_cells.size() > expected_members) {
      const auto& bridge = hybrid_bridge_block(mb, view);
      const bool row_zero_is_inner = hybrid_bridge_row_zero_is_inner(
          mb, bridge, node_r, node_z);
      *hybrid_bridge_out = &bridge;
      *bridge_zero_is_inner_out = row_zero_is_inner;
      bool seen_open_row = false;
      for (int q = 0; q < bridge.n_i_cells; ++q) {
        const int row =
            hybrid_bridge_storage_row(bridge, q, row_zero_is_inner);
        bool full_row = true;
        bool any_row = false;
        for (int j = 0; j < bridge.n_j_cells; ++j) {
          const int c = block_cell_id(bridge, row, j);
          TENRYU_ASSERT(c >= 0 && static_cast<std::size_t>(c) <
                                      pc.member_mask.size(),
                        "central pseudo-core bridge member cell out of range");
          const bool is_member =
              pc.member_mask[static_cast<std::size_t>(c)] != 0U;
          full_row = full_row && is_member;
          any_row = any_row || is_member;
        }
        if (full_row && !seen_open_row) {
          ++bridge_rows;
          expected_members += static_cast<std::size_t>(bridge.n_j_cells);
          continue;
        }
        seen_open_row = true;
        TENRYU_ASSERT(
            !any_row,
            "central pseudo-core bridge members must be full radial-prefix rows");
      }
    }
    TENRYU_ASSERT(pc.member_cells.size() == expected_members,
                  "central pseudo-core hybrid member count does not match depth");
    return selected_rings + bridge_rows;
  }
  // Beyond the cap, membership grows by complete fan LAYERS (the union of
  // layer l across the three fan blocks); count the contiguous full-layer
  // prefix the same way.
  int fan_layers = 0;
  if (selected_rings == block.n_i_cells) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    bool seen_open_layer = false;
    for (int l = 0; l < north.n_i_cells; ++l) {
      bool full_layer = true;
      bool any_layer = false;
      for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                        mesh::BlockRole::EAST_FAN,
                                        mesh::BlockRole::SOUTH_FAN}) {
        const auto& fan_block = mesh::mesh_topo_trifan_fan_block(mb, fan);
        for (int j = 0; j < fan_block.n_j_cells; ++j) {
          const int c = block_cell_id(fan_block, l, j);
          TENRYU_ASSERT(c >= 0 && static_cast<std::size_t>(c) <
                                      pc.member_mask.size(),
                        "central pseudo-core fan member cell out of range");
          const bool is_member =
              pc.member_mask[static_cast<std::size_t>(c)] != 0U;
          full_layer = full_layer && is_member;
          any_layer = any_layer || is_member;
        }
      }
      if (full_layer && !seen_open_layer) {
        ++fan_layers;
        expected_members += static_cast<std::size_t>(4 * north.n_j_cells);
        continue;
      }
      seen_open_layer = true;
      TENRYU_ASSERT(!any_layer,
                    "central pseudo-core fan members must be full prefix layers");
    }
  }
  // Beyond the fans, membership grows by complete polar-shell rows.
  int shell_rows = 0;
  if (fan_layers > 0 &&
      fan_layers ==
          mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN)
              .n_i_cells) {
    const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
    bool seen_open_row = false;
    for (int q = 0; q < shell.n_i_cells; ++q) {
      bool full_row = true;
      bool any_row = false;
      for (int j = 0; j < shell.n_j_cells; ++j) {
        const int c = block_cell_id(shell, q, j);
        TENRYU_ASSERT(c >= 0 && static_cast<std::size_t>(c) <
                                    pc.member_mask.size(),
                      "central pseudo-core shell member cell out of range");
        const bool is_member =
            pc.member_mask[static_cast<std::size_t>(c)] != 0U;
        full_row = full_row && is_member;
        any_row = any_row || is_member;
      }
      if (full_row && !seen_open_row) {
        ++shell_rows;
        expected_members += static_cast<std::size_t>(shell.n_j_cells);
        continue;
      }
      seen_open_row = true;
      TENRYU_ASSERT(!any_row,
                    "central pseudo-core shell members must be full prefix rows");
    }
  }
  TENRYU_ASSERT(pc.member_cells.size() == expected_members,
                "central pseudo-core trifan member count does not match depth");
  return selected_rings + fan_layers + shell_rows;
}

bool rebuild_trifan_virtual_member_geometry_host(
    const core::CentralPseudoCoreState& pc,
    const mesh::MultiBlockTopology& mb,
    const CapTopoView& view,
    std::vector<double>& node_r,
    std::vector<double>& node_z) {
  if (!view.has_cap) {
    return false;
  }
  // Total absorption depth D = full cap rings followed by hybrid bridge rows,
  // or by non-hybrid fan layers and shell rows. The interior radial ladder is
  // m = 1 .. D-1, with the cap outer ring shared by the first outward block.
  const mesh::BlockInfo* bridge = nullptr;
  bool bridge_zero_is_inner = true;
  const int D = trifan_member_ring_count(
      pc, mb, view, node_r, node_z, &bridge, &bridge_zero_is_inner);
  const int n_cap = view.n_cap;
  const int B = view.hybrid ? std::max(0, D - n_cap) : 0;
  if (B > 0) {
    TENRYU_ASSERT(bridge != nullptr && B <= bridge->n_i_cells,
                  "central pseudo-core hybrid bridge depth out of range");
  }
  const int n_b =
      view.hybrid
          ? 0
          : mesh::mesh_topo_trifan_fan_block(
                mb, mesh::BlockRole::NORTH_FAN).n_i_cells;
  // Fan rows absorbed (capped at n_b) and shell rows absorbed beyond them.
  const int L = std::min(std::max(0, D - n_cap), n_b);
  const int S = std::max(0, D - n_cap - n_b);
  const int ntheta = view.ntheta;
  TENRYU_ASSERT(ntheta == 4 * n_cap,
                "central pseudo-core trifan angular count mismatch");
  TENRYU_ASSERT(node_r.size() == node_z.size(),
                "central pseudo-core trifan node arrays are inconsistent");
  const int bridge_boundary_row_index =
      B > 0 ? hybrid_bridge_boundary_node_row_index(
                  mb, *bridge, B, bridge_zero_is_inner)
            : 0;

  const int apex = cap_apex_node_id(mb, view);
  const double r0 = node_r[static_cast<std::size_t>(apex)];
  const double z0 = node_z[static_cast<std::size_t>(apex)];
  TENRYU_ASSERT(std::isfinite(r0) && std::isfinite(z0),
                "central pseudo-core trifan apex position must be finite");

  // Member-boundary node at global angle g in [0, ntheta].
  const auto boundary_node = [&](const int g) -> int {
    if (B > 0) {
      return bridge->owned_node_begin +
             bridge_boundary_row_index * (ntheta + 1) + g;
    }
    if (S > 0) {
      const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
      TENRYU_ASSERT(shell.n_j_cells == ntheta,
                    "polar shell angular count must match trifan ntheta");
      return shell.owned_node_begin + S * (ntheta + 1) + g;
    }
    if (L == 0) {
      return cap_ring_node_id(mb, view, D, g);
    }
    if (g <= n_cap) {
      return mesh::mesh_topo_trifan_fan_node_id(
          mb, mesh::BlockRole::NORTH_FAN, L, g);
    }
    if (g <= 3 * n_cap) {
      return mesh::mesh_topo_trifan_fan_node_id(
          mb, mesh::BlockRole::EAST_FAN, L, g - n_cap);
    }
    return mesh::mesh_topo_trifan_fan_node_id(
        mb, mesh::BlockRole::SOUTH_FAN, L, g - 3 * n_cap);
  };

  // Fold guard: the interior star map is valid only if the boundary
  // anchors are angularly monotone around the apex. Real boundary nodes
  // are live DOF and stay untouched; when their apex-relative angles are
  // non-monotone (violent aspherical interfaces), the DECORATIVE interior
  // uses a minimally repaired anchor fan (deterministic forward
  // projection, radius per anchor preserved). Monotone inputs reproduce
  // the original anchors bit-exactly.
  std::vector<double> anchor_r(static_cast<std::size_t>(ntheta) + 1U);
  std::vector<double> anchor_z(static_cast<std::size_t>(ntheta) + 1U);
  {
    std::vector<double> phi(static_cast<std::size_t>(ntheta) + 1U);
    std::vector<double> rad(static_cast<std::size_t>(ntheta) + 1U);
    for (int g = 0; g <= ntheta; ++g) {
      const int nb = boundary_node(g);
      const double dr = node_r[static_cast<std::size_t>(nb)] - r0;
      const double dz = node_z[static_cast<std::size_t>(nb)] - z0;
      anchor_r[static_cast<std::size_t>(g)] =
          node_r[static_cast<std::size_t>(nb)];
      anchor_z[static_cast<std::size_t>(g)] =
          node_z[static_cast<std::size_t>(nb)];
      phi[static_cast<std::size_t>(g)] = std::atan2(std::max(dr, 0.0), dz);
      rad[static_cast<std::size_t>(g)] = std::hypot(dr, dz);
    }
    bool monotone = true;
    for (int g = 1; g <= ntheta; ++g) {
      if (!(phi[static_cast<std::size_t>(g)] >
            phi[static_cast<std::size_t>(g - 1)])) {
        monotone = false;
        break;
      }
    }
    if (!monotone) {
      const double eps =
          0.02 * (3.14159265358979323846 / static_cast<double>(ntheta));
      int repaired = 0;
      double max_shift = 0.0;
      for (int g = 1; g <= ntheta; ++g) {
        const double lower = phi[static_cast<std::size_t>(g - 1)] + eps;
        if (phi[static_cast<std::size_t>(g)] < lower) {
          max_shift =
              std::max(max_shift, lower - phi[static_cast<std::size_t>(g)]);
          phi[static_cast<std::size_t>(g)] = lower;
          ++repaired;
        }
      }
      for (int g = 1; g <= ntheta; ++g) {
        const std::size_t gi = static_cast<std::size_t>(g);
        anchor_r[gi] = r0 + rad[gi] * std::sin(phi[gi]);
        anchor_z[gi] = z0 + rad[gi] * std::cos(phi[gi]);
      }
      std::fprintf(stderr,
                   "[central_pseudo_core] star-map anchor fold repair "
                   "anchors=%d max_dphi=%.3e\n",
                   repaired, max_shift);
    }
  }

  bool changed = false;
  const auto map_node = [&](const int n_inner, const int g, const double psi) {
    TENRYU_ASSERT(n_inner >= 0 && g >= 0 &&
                      static_cast<std::size_t>(n_inner) < node_r.size() &&
                      static_cast<std::size_t>(g) < anchor_r.size() &&
                      static_cast<std::size_t>(g) < anchor_z.size(),
                  "central pseudo-core trifan star-map node out of range");
    const double rb = anchor_r[static_cast<std::size_t>(g)];
    const double zb = anchor_z[static_cast<std::size_t>(g)];
    TENRYU_ASSERT(std::isfinite(rb) && std::isfinite(zb),
                  "central pseudo-core trifan boundary node must be finite");
    const double r_new = r0 + psi * (rb - r0);
    const double z_new = z0 + psi * (zb - z0);
    if (node_r[static_cast<std::size_t>(n_inner)] != r_new ||
        node_z[static_cast<std::size_t>(n_inner)] != z_new) {
      node_r[static_cast<std::size_t>(n_inner)] = r_new;
      node_z[static_cast<std::size_t>(n_inner)] = z_new;
      changed = true;
    }
  };

  // Interior cap rings m = 1 .. min(D-1, n_cap); the cap k index IS the
  // global angle.
  const int cap_top = std::min(D - 1, n_cap);
  for (int m = 1; m <= cap_top; ++m) {
    const double psi =
        std::cbrt(static_cast<double>(m) / static_cast<double>(D));
    for (int k = 0; k <= ntheta; ++k) {
      map_node(cap_ring_node_id(mb, view, m, k), k, psi);
    }
  }
  // Interior fan rows: l = 1 .. L-1 while the boundary is a fan edge, or all
  // rows 1 .. n_b once shell rows are absorbed (the shared end columns alias
  // the same node ids across fans, so re-mapping is an idempotent overwrite;
  // fan row n_b aliases shell row 0).
  const int fan_top = (S > 0) ? L : (L - 1);
  for (int l = 1; l <= fan_top; ++l) {
    const double psi = std::cbrt(static_cast<double>(n_cap + l) /
                                 static_cast<double>(D));
    for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                      mesh::BlockRole::EAST_FAN,
                                      mesh::BlockRole::SOUTH_FAN}) {
      const auto& fan_block = mesh::mesh_topo_trifan_fan_block(mb, fan);
      const int g_base = (fan == mesh::BlockRole::NORTH_FAN)
                             ? 0
                             : (fan == mesh::BlockRole::EAST_FAN ? n_cap
                                                                 : 3 * n_cap);
      for (int k = 0; k <= fan_block.n_j_cells; ++k) {
        map_node(mesh::mesh_topo_trifan_fan_node_id(mb, fan, l, k),
                 g_base + k,
                 psi);
      }
    }
  }
  // Interior hybrid bridge node-rows q = 1 .. B-1. The same idealized
  // volume-uniform radial ladder used by the shell branch continues outward
  // from the cap.
  if (B > 1) {
    for (int q = 1; q < B; ++q) {
      const double psi = std::cbrt(static_cast<double>(n_cap + q) /
                                   static_cast<double>(D));
      const int row_node_index = hybrid_bridge_boundary_node_row_index(
          mb, *bridge, q, bridge_zero_is_inner);
      for (int g = 0; g <= ntheta; ++g) {
        map_node(bridge->owned_node_begin +
                     row_node_index * (ntheta + 1) + g,
                 g,
                 psi);
      }
    }
  }
  // Interior shell rows q = 1 .. S-1.
  if (!view.hybrid && S > 1) {
    const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
    for (int q = 1; q < S; ++q) {
      const double psi = std::cbrt(static_cast<double>(n_cap + n_b + q) /
                                   static_cast<double>(D));
      for (int g = 0; g <= ntheta; ++g) {
        map_node(shell.owned_node_begin + q * (ntheta + 1) + g, g, psi);
      }
    }
  }
  return changed;
}

void initialize_member_scatter_weights(core::CentralPseudoCoreState& pc,
                                       const std::vector<double>& mass) {
  const std::size_t n_members = pc.member_cells.size();
  TENRYU_ASSERT(n_members > 0U,
                "central pseudo-core scatter requires member cells");
  if (pc.member_scatter_weights.size() == n_members) {
    return;
  }

  bool valid_mass_weights = true;
  long double raw_sum = 0.0L;
  std::vector<long double> raw(n_members, 0.0L);
  for (std::size_t i = 0; i < n_members; ++i) {
    const int c = pc.member_cells[i];
    TENRYU_ASSERT(c >= 0 && static_cast<std::size_t>(c) < mass.size(),
                  "central pseudo-core scatter member out of range");
    const double m = mass[static_cast<std::size_t>(c)];
    if (!(std::isfinite(m) && m > 0.0)) {
      valid_mass_weights = false;
      break;
    }
    raw[i] = static_cast<long double>(m);
    raw_sum += raw[i];
  }
  if (!(valid_mass_weights && raw_sum > 0.0L)) {
    raw_sum = static_cast<long double>(n_members);
    std::fill(raw.begin(), raw.end(), 1.0L);
  }

  pc.member_scatter_weights.assign(n_members, 0.0);
  long double partial = 0.0L;
  for (std::size_t i = 0; i + 1U < n_members; ++i) {
    const long double w = raw[i] / raw_sum;
    pc.member_scatter_weights[i] = static_cast<double>(w);
    partial += static_cast<long double>(pc.member_scatter_weights[i]);
  }
  pc.member_scatter_weights[n_members - 1U] =
      static_cast<double>(1.0L - partial);
  if (!(pc.member_scatter_weights[n_members - 1U] > 0.0 &&
        std::isfinite(pc.member_scatter_weights[n_members - 1U]))) {
    const double uniform = 1.0 / static_cast<double>(n_members);
    std::fill(pc.member_scatter_weights.begin(),
              pc.member_scatter_weights.end(),
              uniform);
  }
}

void upload_overlay(core::State& state) {
  auto& pc = state.central_pseudo_core;
  pc.d_member_mask.reset(pc.member_mask.size());
  pc.d_member_mask.copy_from_host(pc.member_mask);
  pc.d_passive_mask.reset(pc.passive_mask.size());
  pc.d_passive_mask.copy_from_host(pc.passive_mask);
  pc.d_inactive_member_mask.reset(pc.inactive_member_mask.size());
  pc.d_inactive_member_mask.copy_from_host(pc.inactive_member_mask);
  pc.d_active_cell_class.reset(pc.active_cell_class.size());
  pc.d_active_cell_class.copy_from_host(pc.active_cell_class);
  pc.d_boundary_node_mask.reset(pc.boundary_node_mask.size());
  pc.d_boundary_node_mask.copy_from_host(pc.boundary_node_mask);
  pc.d_boundary_nodes_ordered.reset(pc.boundary_nodes_ordered.size());
  pc.d_boundary_nodes_ordered.copy_from_host(pc.boundary_nodes_ordered);
  pc.d_member_scatter_weights.reset(pc.member_scatter_weights.size());
  pc.d_member_scatter_weights.copy_from_host(pc.member_scatter_weights);
}

template <typename Tag>
std::vector<double> copy_field(const core::Field1D<Tag>& field) {
  std::vector<double> out;
  field.copy_to_host(out);
  return out;
}

double relative_delta(const double value, const double reference) {
  const double denom = std::max(std::abs(reference), 1.0e-300);
  return (value - reference) / denom;
}

bool phase_is(const char* phase, const char* name) {
  return phase != nullptr && std::strcmp(phase, name) == 0;
}

bool sync_internal_from_members_phase(const char* phase) {
  return phase_is(phase, "hydro_step_start") ||
         phase_is(phase, "cfl_acoustic") ||
         phase_is(phase, "cfl");
}

bool work_probe_enabled() {
  const char* env = std::getenv("TENRYU_I1B_PSEUDOCELL_WORK_PROBE");
  return env != nullptr && env[0] != '\0' && env[0] != '0';
}

bool phase_ledger_enabled(const core::State& state) {
  static const bool enabled = [] {
    const char* env = std::getenv("TENRYU_I1B_PSEUDOCELL_PHASE_LEDGER");
    return env != nullptr && env[0] != '\0' && env[0] != '0';
  }();
  // Step window, default [0, 3] (the historical activation-forensics
  // window). Overridable for late-time forensics, e.g. the mixed-core
  // terminal-phase energy bracket (TENRYU_I1B_PSEUDOCELL_PHASE_LEDGER_FROM/TO).
  static const int from = [] {
    const char* env =
        std::getenv("TENRYU_I1B_PSEUDOCELL_PHASE_LEDGER_FROM");
    return env != nullptr && env[0] != '\0' ? std::atoi(env) : 0;
  }();
  static const int to = [] {
    const char* env = std::getenv("TENRYU_I1B_PSEUDOCELL_PHASE_LEDGER_TO");
    return env != nullptr && env[0] != '\0' ? std::atoi(env) : 3;
  }();
  return enabled && state.step >= from && state.step <= to;
}

double electron_energy_fraction(const double Ue,
                                const double Ui,
                                const bool has_ion_energy) {
  if (!has_ion_energy) {
    return 1.0;
  }
  const double U_total = Ue + Ui;
  return (std::isfinite(U_total) && U_total > 0.0) ? (Ue / U_total) : 0.5;
}

// A3 asphericity diagnostic (verdict #6): Legendre decomposition of the
// macro boundary radius and radial velocity, R(theta) = sum R_l P_l(cos
// theta) etc. The 1D core sub-model is spherical, so the hybrid handoff is
// only defensible while eps_R = |R_{l>=1}|/|R_0| < 0.03 and eps_u < 0.05.
// Emitted at absorption events (pre/post = the handoff surfaces) and on a
// cadence under TENRYU_I1B_CORE1D_ASPH_EVERY. Host-side, ~50 nodes.
void emit_boundary_asphericity(core::State& state, const char* tag) {
  static const int every = [] {
    const char* raw = std::getenv("TENRYU_I1B_CORE1D_ASPH_EVERY");
    return raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
  }();
  const bool event_tag =
      tag != nullptr && (tag[0] == 'p');  // pre_absorb / post_absorb
  if (!event_tag && (every <= 0 || (state.step % every) != 0)) {
    return;
  }
  auto& pc = state.central_pseudo_core;
  if (!pc.built || pc.boundary_nodes_ordered.size() < 8U) {
    return;
  }
  std::vector<double> nr = copy_field(state.x_r);
  std::vector<double> nz = copy_field(state.x_z);
  std::vector<double> vr = copy_field(state.v_r);
  std::vector<double> vz = copy_field(state.v_z);
  const int n_nodes = static_cast<int>(nr.size());
  const int L = 5;  // l = 0..4
  double ata[5][5] = {};
  double atb_R[5] = {};
  double atb_u[5] = {};
  int used = 0;
  for (const int n : pc.boundary_nodes_ordered) {
    if (n < 0 || n >= n_nodes) {
      continue;
    }
    const std::size_t ni = static_cast<std::size_t>(n);
    const double rho = std::hypot(nr[ni], nz[ni]);
    if (!(rho > 0.0)) {
      continue;
    }
    // The boundary loop closes along the axis (r = 0, rho = |z| sweeping
    // 0..R): those closure nodes are not the physical interface and their
    // inclusion fabricated giant l=2/l=4 amplitudes (measured R_4/R0 = 0.48
    // on a 3%-converged spherical interface). Fit the physical arc only.
    if (nr[ni] < 1.0e-2 * rho) {
      continue;
    }
    const double x = nz[ni] / rho;  // cos(theta)
    const double ur = (vr[ni] * nr[ni] + vz[ni] * nz[ni]) / rho;
    double P[5];
    P[0] = 1.0;
    P[1] = x;
    P[2] = 0.5 * (3.0 * x * x - 1.0);
    P[3] = 0.5 * (5.0 * x * x * x - 3.0 * x);
    P[4] = 0.125 * ((35.0 * x * x - 30.0) * x * x + 3.0);
    for (int a = 0; a < L; ++a) {
      for (int b = 0; b < L; ++b) {
        ata[a][b] += P[a] * P[b];
      }
      atb_R[a] += P[a] * rho;
      atb_u[a] += P[a] * ur;
    }
    ++used;
  }
  if (used < 8) {
    return;
  }
  // solve the shared normal equations for both right-hand sides
  double lu[5][5];
  std::memcpy(lu, ata, sizeof(lu));
  double cR[5] = {}, cu[5] = {};
  std::memcpy(cR, atb_R, sizeof(cR));
  std::memcpy(cu, atb_u, sizeof(cu));
  for (int col = 0; col < L; ++col) {
    int piv = col;
    for (int row = col + 1; row < L; ++row) {
      if (std::abs(lu[row][col]) > std::abs(lu[piv][col])) {
        piv = row;
      }
    }
    if (std::abs(lu[piv][col]) < 1.0e-300) {
      return;
    }
    if (piv != col) {
      for (int k = 0; k < L; ++k) {
        std::swap(lu[col][k], lu[piv][k]);
      }
      std::swap(cR[col], cR[piv]);
      std::swap(cu[col], cu[piv]);
    }
    for (int row = col + 1; row < L; ++row) {
      const double f = lu[row][col] / lu[col][col];
      for (int k = col; k < L; ++k) {
        lu[row][k] -= f * lu[col][k];
      }
      cR[row] -= f * cR[col];
      cu[row] -= f * cu[col];
    }
  }
  for (int row = L - 1; row >= 0; --row) {
    for (int k = row + 1; k < L; ++k) {
      cR[row] -= lu[row][k] * cR[k];
      cu[row] -= lu[row][k] * cu[k];
    }
    cR[row] /= lu[row][row];
    cu[row] /= lu[row][row];
  }
  double hiR = 0.0, hiu = 0.0;
  for (int l = 1; l < L; ++l) {
    hiR += cR[l] * cR[l];
    hiu += cu[l] * cu[l];
  }
  const double u_floor = 1.0e3;  // cm/s scale floor for quiescent boundaries
  const double eps_R = std::sqrt(hiR) / std::max(std::abs(cR[0]), 1.0e-300);
  const double eps_u =
      std::sqrt(hiu) / std::max(std::abs(cu[0]), u_floor);
  std::fprintf(stderr,
               "[core1d_asph] step=%d t=%.6e tag=%s nodes=%d R0=%.6e "
               "u0=%.4e eps_R=%.4e eps_u=%.4e R_l=%.3e,%.3e,%.3e,%.3e\n",
               state.step, state.t, tag != nullptr ? tag : "cadence", used,
               cR[0], cu[0], eps_R, eps_u, cR[1], cR[2], cR[3], cR[4]);
}

double macro_core_pressure(const core::CentralPseudoCoreState& pc,
                           const core::Config& cfg) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "central pseudo-core macro pressure requires a material");
  const auto& mat = cfg.materials.materials.front();
  TENRYU_ASSERT(mat.eos_model == "ideal_gas" && mat.eos_tables == nullptr,
                "central pseudo-core Phase1 requires ideal_gas EOS");
  TENRYU_ASSERT(pc.M_c > 0.0 && pc.V_c > 0.0,
                "central pseudo-core macro pressure requires positive mass/volume");
  // Stratified 1D sub-model override: the boundary force consumes the
  // sub-model's outer-face pressure (P+q of the outermost shell) instead of
  // the pooled 0-D closure.
  if (core1d::enabled() && core1d::built(&pc)) {
    const double p1d =
        core1d::outer_face_pressure(&pc, mat.ideal_gas_gamma);
    if (std::isfinite(p1d) && p1d > 0.0) {
      return p1d;
    }
  }
  const double rho_c = pc.M_c / pc.V_c;
  const double e_total =
      std::max((pc.Ue_c + pc.Ui_c) / pc.M_c, 0.0);
  const double p = (mat.ideal_gas_gamma - 1.0) * rho_c * e_total;
  return (std::isfinite(p) && p > 0.0) ? p : 0.0;
}

struct ScalarTotals {
  double mass = 0.0;
  double tracer_mass = 0.0;
  double Ue = 0.0;
  double Ui = 0.0;
  double E_total = 0.0;
};

struct MemberSums {
  long double M = 0.0L;
  long double MY = 0.0L;
  long double Ue = 0.0L;
  long double Ui = 0.0L;
  long double UY = 0.0L;
  long double V = 0.0L;
};

MemberSums sum_member_state(const core::CentralPseudoCoreState& pc,
                            const std::vector<double>& mass,
                            const std::vector<double>& vol,
                            const std::vector<double>& ee,
                            const std::vector<double>& ei,
                            const std::vector<double>& tracer) {
  MemberSums sums{};
  for (const int c : pc.member_cells) {
    const std::size_t idx = static_cast<std::size_t>(c);
    sums.M += static_cast<long double>(mass[idx]);
    sums.V += static_cast<long double>(vol[idx]);
    sums.Ue += static_cast<long double>(mass[idx]) *
               static_cast<long double>(ee[idx]);
    if (!ei.empty()) {
      sums.Ui += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(ei[idx]);
    }
    if (!tracer.empty()) {
      sums.MY += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(tracer[idx]);
      const long double e_cell =
          static_cast<long double>(ee[idx]) +
          (!ei.empty() ? static_cast<long double>(ei[idx]) : 0.0L);
      sums.UY += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(tracer[idx]) * e_cell;
    }
  }
  return sums;
}

// Update the pooled gas ENERGY fraction (pc.U_gas_frac_c). Exact because
// write_member_mirrors distributes pc.Ue_c/Ui_c and the tracer uniformly
// (telescoping-exact): the pooled OLD members contribute exactly
// y_old*U_old to sums.UY, so the difference isolates the newly absorbed
// cells' real gas internal energy. No-op unless membership changed —
// boundary work and adiabatic compression preserve the fraction under the
// same-gamma pooled closure.
void update_u_gas_energy_fraction(core::CentralPseudoCoreState& pc,
                                  const MemberSums& sums) {
  const int count = static_cast<int>(pc.member_cells.size());
  const double U_new = static_cast<double>(sums.Ue + sums.Ui);
  if (!(U_new > 0.0) || !std::isfinite(U_new)) {
    return;
  }
  if (std::isfinite(pc.U_gas_frac_c) &&
      pc.u_gas_frac_member_count == count) {
    return;
  }
  double f_new;
  if (!std::isfinite(pc.U_gas_frac_c) || !(pc.M_c > 0.0) ||
      !((pc.Ue_c + pc.Ui_c) > 0.0)) {
    f_new = static_cast<double>(sums.UY) / U_new;
  } else {
    const double U_old = pc.Ue_c + pc.Ui_c;
    const double y_old =
        std::min(std::max(pc.M_Y_c / pc.M_c, 0.0), 1.0);
    const double dUY_new = static_cast<double>(sums.UY) - y_old * U_old;
    f_new = (pc.U_gas_frac_c * U_old + std::max(dUY_new, 0.0)) / U_new;
  }
  pc.U_gas_frac_c = std::min(std::max(f_new, 0.0), 1.0);
  pc.u_gas_frac_member_count = count;
}

std::array<int, 4> multiblock_cell_nodes(const mesh::MultiBlockTopology& mb,
                                         const int c) {
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  return {{
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 0)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 1)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 2)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 3)],
  }};
}

std::vector<double> member_corner_mass_host(const core::State& state,
                                            const std::vector<double>& mass,
                                            const std::vector<double>& node_r,
                                            const std::vector<double>& node_z) {
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t expected = static_cast<std::size_t>(n_cells) * 4U;
  if (state.corner_mass.size() == expected) {
    return copy_field(state.corner_mass);
  }

  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "central pseudo-core activation KE requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  std::vector<double> corner_mass(expected, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const std::array<int, 4> nodes = multiblock_cell_nodes(mb, c);
    const int nverts = active_nverts(state.mesh, c);
    double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
    if (nverts == 3) {
      rz::compute_triangle_corner_masses_exact(
          mass[static_cast<std::size_t>(c)],
          node_r[static_cast<std::size_t>(nodes[0])],
          node_z[static_cast<std::size_t>(nodes[0])],
          node_r[static_cast<std::size_t>(nodes[1])],
          node_z[static_cast<std::size_t>(nodes[1])],
          node_r[static_cast<std::size_t>(nodes[2])],
          node_z[static_cast<std::size_t>(nodes[2])],
          m_corner);
    } else {
      rz::compute_quad_corner_masses_exact_subpolygon(
          mass[static_cast<std::size_t>(c)],
          node_r[static_cast<std::size_t>(nodes[0])],
          node_z[static_cast<std::size_t>(nodes[0])],
          node_r[static_cast<std::size_t>(nodes[1])],
          node_z[static_cast<std::size_t>(nodes[1])],
          node_r[static_cast<std::size_t>(nodes[2])],
          node_z[static_cast<std::size_t>(nodes[2])],
          node_r[static_cast<std::size_t>(nodes[3])],
          node_z[static_cast<std::size_t>(nodes[3])],
          m_corner);
    }
    for (int k = 0; k < 4; ++k) {
      corner_mass[static_cast<std::size_t>(4 * c + k)] = m_corner[k];
    }
  }
  return corner_mass;
}

void project_activation_kinetic_energy(core::State& state) {
  auto& pc = state.central_pseudo_core;
  if (pc.activation_ke_valid) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "central pseudo-core activation KE requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = static_cast<int>(state.v_r.size());
  TENRYU_ASSERT(static_cast<int>(state.v_z.size()) == n_nodes &&
                    static_cast<int>(state.x_r.size()) == n_nodes &&
                    static_cast<int>(state.x_z.size()) == n_nodes,
                "central pseudo-core activation KE requires matching node fields");

  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);
  std::vector<double> v_r = copy_field(state.v_r);
  std::vector<double> v_z = copy_field(state.v_z);
  const std::vector<double> corner_mass =
      member_corner_mass_host(state, mass, node_r, node_z);

  std::vector<std::uint8_t> boundary_node(static_cast<std::size_t>(n_nodes), 0U);
  for (const int n : pc.boundary_nodes_ordered) {
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "central pseudo-core boundary node out of range");
    boundary_node[static_cast<std::size_t>(n)] = 1U;
  }

  std::vector<long double> member_node_mass(static_cast<std::size_t>(n_nodes),
                                            0.0L);
  for (const int c : pc.member_cells) {
    TENRYU_ASSERT(c >= 0 && c < n_cells,
                  "central pseudo-core member cell out of range");
    const std::array<int, 4> nodes = multiblock_cell_nodes(mb, c);
    const int nverts = active_nverts(state.mesh, c);
    for (int k = 0; k < nverts; ++k) {
      const int n = nodes[static_cast<std::size_t>(k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "central pseudo-core member node out of range");
      member_node_mass[static_cast<std::size_t>(n)] += static_cast<long double>(
          corner_mass[static_cast<std::size_t>(4 * c + k)]);
    }
  }

  long double K_old = 0.0L;
  long double K_new = 0.0L;
  pc.activation_passive_nodes.clear();
  pc.activation_saved_v_r.clear();
  pc.activation_saved_v_z.clear();
  for (int n = 0; n < n_nodes; ++n) {
    const long double m = member_node_mass[static_cast<std::size_t>(n)];
    if (m == 0.0L) {
      continue;
    }
    const long double vr = static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
    const long double vz = static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
    const long double K = 0.5L * m * (vr * vr + vz * vz);
    K_old += K;
    if (boundary_node[static_cast<std::size_t>(n)] != 0U) {
      K_new += K;
      continue;
    }
    pc.activation_passive_nodes.push_back(n);
    pc.activation_saved_v_r.push_back(v_r[static_cast<std::size_t>(n)]);
    pc.activation_saved_v_z.push_back(v_z[static_cast<std::size_t>(n)]);
    v_r[static_cast<std::size_t>(n)] = 0.0;
    v_z[static_cast<std::size_t>(n)] = 0.0;
  }

  const double E_before = diagnostics::compute_energy_budget_2d(state).E_total;
  state.v_r.copy_from_host(v_r);
  state.v_z.copy_from_host(v_z);
  const double E_after_zero =
      diagnostics::compute_energy_budget_2d(state).E_total;
  double dU = E_before - E_after_zero;
  const double dU_member = static_cast<double>(K_old - K_new);
  if (!std::isfinite(dU)) {
    dU = dU_member;
  }
  const double scale =
      std::max({std::abs(static_cast<double>(K_old)),
                std::abs(static_cast<double>(K_new)),
                std::abs(E_before),
                1.0});
  if (dU < 0.0 && std::abs(dU) <= 1.0e-12 * scale) {
    dU = 0.0;
  }
  TENRYU_ASSERT(std::isfinite(dU) && dU >= 0.0,
                "central pseudo-core activation KE bank must be non-negative");

  pc.activation_ke_K_old = static_cast<double>(K_old);
  pc.activation_ke_K_new = static_cast<double>(K_new);
  pc.activation_ke_dU = dU;
  pc.activation_ke_E_before = E_before;
  pc.activation_ke_E_after_zero = E_after_zero;
  pc.activation_ke_bank_pending = true;
  pc.activation_ke_valid = true;
}

void write_member_mirrors(core::State& state,
                          std::vector<double>& mass,
                          std::vector<double>& vol,
                          std::vector<double>& rho,
                          std::vector<double>& ee,
                          std::vector<double>& ei,
                          std::vector<double>& tracer) {
  const auto& pc = state.central_pseudo_core;
  TENRYU_ASSERT(pc.M_c > 0.0 && pc.V_c > 0.0,
                "central pseudo-core mirror requires positive mass/volume");
  TENRYU_ASSERT(pc.member_scatter_weights.size() == pc.member_cells.size(),
                "central pseudo-core mirror scatter weights are not initialized");
  const double rho_c = pc.M_c / pc.V_c;
  long double mass_prefix = 0.0L;
  long double vol_prefix = 0.0L;
  long double Ue_prefix = 0.0L;
  long double Ui_prefix = 0.0L;
  long double MY_prefix = 0.0L;
  for (std::size_t i = 0; i < pc.member_cells.size(); ++i) {
    const int c = pc.member_cells[i];
    const std::size_t idx = static_cast<std::size_t>(c);
    const bool last = i + 1U == pc.member_cells.size();
    const long double w =
        static_cast<long double>(pc.member_scatter_weights[i]);
    const long double m_i =
        last ? (static_cast<long double>(pc.M_c) - mass_prefix)
             : w * static_cast<long double>(pc.M_c);
    const long double V_i =
        last ? (static_cast<long double>(pc.V_c) - vol_prefix)
             : w * static_cast<long double>(pc.V_c);
    const long double Ue_i =
        last ? (static_cast<long double>(pc.Ue_c) - Ue_prefix)
             : w * static_cast<long double>(pc.Ue_c);
    const long double Ui_i =
        last ? (static_cast<long double>(pc.Ui_c) - Ui_prefix)
             : w * static_cast<long double>(pc.Ui_c);
    const long double MY_i =
        last ? (static_cast<long double>(pc.M_Y_c) - MY_prefix)
             : w * static_cast<long double>(pc.M_Y_c);
    TENRYU_ASSERT(m_i > 0.0L && V_i > 0.0L,
                  "central pseudo-core mirror produced non-positive storage");
    mass[idx] = static_cast<double>(m_i);
    vol[idx] = static_cast<double>(V_i);
    rho[idx] = rho_c;
    ee[idx] = static_cast<double>(Ue_i / m_i);
    if (!ei.empty()) {
      ei[idx] = static_cast<double>(Ui_i / m_i);
    }
    if (!tracer.empty()) {
      tracer[idx] = static_cast<double>(MY_i / m_i);
    }
    mass_prefix += static_cast<long double>(mass[idx]);
    vol_prefix += static_cast<long double>(vol[idx]);
    Ue_prefix += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(ee[idx]);
    if (!ei.empty()) {
      Ui_prefix += static_cast<long double>(mass[idx]) *
                   static_cast<long double>(ei[idx]);
    }
    if (!tracer.empty()) {
      MY_prefix += static_cast<long double>(mass[idx]) *
                   static_cast<long double>(tracer[idx]);
    }
  }
  state.mass.copy_from_host(mass.data());
  state.vol.copy_from_host(vol.data());
  state.rho.copy_from_host(rho.data());
  state.ee.copy_from_host(ee.data());
  if (!ei.empty()) {
    state.ei.copy_from_host(ei.data());
  }
  if (!tracer.empty()) {
    state.gas_tracer_Y.copy_from_host(tracer.data());
  }
}

void deactivate_if_needed(core::State& state) {
  auto& pc = state.central_pseudo_core;
  if (!pc.built) {
    return;
  }

  double dK_restore = 0.0;
  if (pc.activation_ke_valid && !pc.activation_passive_nodes.empty()) {
    std::vector<double> v_r = copy_field(state.v_r);
    std::vector<double> v_z = copy_field(state.v_z);
    TENRYU_ASSERT(pc.activation_passive_nodes.size() ==
                      pc.activation_saved_v_r.size() &&
                      pc.activation_passive_nodes.size() ==
                          pc.activation_saved_v_z.size(),
                  "central pseudo-core deactivation saved velocity mismatch");
    const double E_before = diagnostics::compute_energy_budget_2d(state).E_total;
    for (std::size_t i = 0; i < pc.activation_passive_nodes.size(); ++i) {
      const int n = pc.activation_passive_nodes[i];
      TENRYU_ASSERT(n >= 0 && static_cast<std::size_t>(n) < v_r.size(),
                    "central pseudo-core deactivation passive node out of range");
      v_r[static_cast<std::size_t>(n)] = pc.activation_saved_v_r[i];
      v_z[static_cast<std::size_t>(n)] = pc.activation_saved_v_z[i];
    }
    state.v_r.copy_from_host(v_r);
    state.v_z.copy_from_host(v_z);
    const double E_after = diagnostics::compute_energy_budget_2d(state).E_total;
    dK_restore = E_after - E_before;
    const double scale =
        std::max({std::abs(E_before), std::abs(E_after), 1.0});
    if (dK_restore < 0.0 && std::abs(dK_restore) <= 1.0e-12 * scale) {
      dK_restore = 0.0;
    }
    TENRYU_ASSERT(std::isfinite(dK_restore) && dK_restore >= 0.0,
                  "central pseudo-core deactivation restored KE must be non-negative");
  }

  if (dK_restore > 0.0 && !pc.activation_ke_bank_pending) {
    std::vector<double> mass = copy_field(state.mass);
    std::vector<double> vol = copy_field(state.vol);
    std::vector<double> rho = copy_field(state.rho);
    std::vector<double> ee = copy_field(state.ee);
    std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                              : copy_field(state.ei);
    std::vector<double> tracer =
        state.gas_tracer_Y.empty() ? std::vector<double>{}
                                   : copy_field(state.gas_tracer_Y);
    std::vector<double> node_r = copy_field(state.x_r);
    std::vector<double> node_z = copy_field(state.x_z);
    const MemberSums sums = sum_member_state(pc, mass, vol, ee, ei, tracer);
    TENRYU_ASSERT(sums.M > 0.0L,
                  "central pseudo-core deactivation requires positive aggregate mass");
    const double V_boundary = boundary_loop_volume(pc, node_r, node_z);
    const double Ue = static_cast<double>(sums.Ue);
    const double Ui = !ei.empty() ? static_cast<double>(sums.Ui) : 0.0;
    const double U_total = Ue + Ui;
    const double scale =
        std::max({std::abs(U_total), std::abs(dK_restore), 1.0});
    TENRYU_ASSERT(dK_restore <= U_total + 1.0e-12 * scale,
                  "central pseudo-core deactivation lacks internal energy for KE restore");
    const double chi_e = electron_energy_fraction(Ue, Ui, !ei.empty());
    pc.M_c = static_cast<double>(sums.M);
    pc.M_Y_c = static_cast<double>(sums.MY);
    // Deactivation dissolves the pooled state; the gas energy fraction is
    // meaningless afterwards and re-initializes on the next build.
    pc.U_gas_frac_c = std::numeric_limits<double>::quiet_NaN();
    pc.u_gas_frac_member_count = -1;
    pc.core1d_V_gas_c = std::numeric_limits<double>::quiet_NaN();
    core1d::reset(&pc);
    pc.V_c = V_boundary;
    pc.Ue_c = Ue - chi_e * dK_restore;
    pc.Ui_c = !ei.empty() ? (Ui - (1.0 - chi_e) * dK_restore) : 0.0;
    TENRYU_ASSERT(std::isfinite(pc.Ue_c) && std::isfinite(pc.Ui_c) &&
                      pc.Ue_c >= -1.0e-12 * scale &&
                      pc.Ui_c >= -1.0e-12 * scale,
                  "central pseudo-core deactivation produced negative internal energy");
    pc.Ue_c = std::max(pc.Ue_c, 0.0);
    pc.Ui_c = std::max(pc.Ui_c, 0.0);
    write_member_mirrors(state, mass, vol, rho, ee, ei, tracer);
  }

  pc.activation_ke_bank_pending = false;
  pc.activation_ke_valid = false;
  pc.activation_ke_K_old = 0.0;
  pc.activation_ke_K_new = 0.0;
  pc.activation_ke_dU = 0.0;
  pc.activation_ke_chi_e = 1.0;
  pc.activation_ke_E_before = 0.0;
  pc.activation_ke_E_after_zero = 0.0;
  pc.V_c_initial = std::numeric_limits<double>::quiet_NaN();
  pc.rho_c_initial = std::numeric_limits<double>::quiet_NaN();
  pc.activation_passive_nodes.clear();
  pc.activation_saved_v_r.clear();
  pc.activation_saved_v_z.clear();
  pc.boundary_work_pending = false;
  pc.boundary_work_valid = false;
  pc.configured = false;
  pc.built = false;
  pc.valid = false;
  pc.member_mask.clear();
  pc.passive_mask.clear();
  pc.inactive_member_mask.clear();
  pc.active_cell_class.clear();
  pc.boundary_node_mask.clear();
  pc.member_cells.clear();
  pc.passive_cells.clear();
  pc.boundary_nodes_ordered.clear();
  pc.member_scatter_weights.clear();
  pc.d_member_mask.reset(0);
  pc.d_passive_mask.reset(0);
  pc.d_inactive_member_mask.reset(0);
  pc.d_active_cell_class.reset(0);
  pc.d_boundary_node_mask.reset(0);
  pc.d_boundary_nodes_ordered.reset(0);
  pc.d_member_scatter_weights.reset(0);
  pc.d_boundary_impulse_r.reset(0);
  pc.d_boundary_impulse_z.reset(0);
}

ScalarTotals compute_global_totals(const core::State& state) {
  const std::vector<double> mass = copy_field(state.mass);
  const std::vector<double> ee = copy_field(state.ee);
  const std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                                  : copy_field(state.ei);
  const std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  ScalarTotals totals{};
  for (std::size_t c = 0; c < mass.size(); ++c) {
    const double m = mass[c];
    totals.mass += m;
    totals.Ue += m * ee[c];
    if (!ei.empty()) {
      totals.Ui += m * ei[c];
    }
    if (!tracer.empty()) {
      totals.tracer_mass += m * tracer[c];
    }
  }
  totals.E_total = diagnostics::compute_energy_budget_2d(state).E_total;
  return totals;
}

__global__ void zero_member_compatible_cell_buffers_kernel(
    double* __restrict__ corner_force_p_r,
    double* __restrict__ corner_force_p_z,
    double* __restrict__ corner_force_sub_r,
    double* __restrict__ corner_force_sub_z,
    double* __restrict__ corner_force_q_r,
    double* __restrict__ corner_force_q_z,
    double* __restrict__ work_p,
    double* __restrict__ work_sub,
    double* __restrict__ work_av,
    const std::uint8_t* __restrict__ member_mask,
    const int n_cells,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  const int base = 4 * c;
  for (int k = 0; k < 4; ++k) {
    corner_force_p_r[base + k] = 0.0;
    corner_force_p_z[base + k] = 0.0;
    corner_force_sub_r[base + k] = 0.0;
    corner_force_sub_z[base + k] = 0.0;
  }
  if (corner_force_q_r != nullptr) {
    const int q_base = c * corner_stride;
    for (int k = 0; k < corner_stride; ++k) {
      corner_force_q_r[q_base + k] = 0.0;
      corner_force_q_z[q_base + k] = 0.0;
    }
  }
  work_p[c] = 0.0;
  work_sub[c] = 0.0;
  work_av[c] = 0.0;
}

__global__ void zero_member_work_kernel(double* __restrict__ work_p,
                                        double* __restrict__ work_sub,
                                        double* __restrict__ work_av,
                                        const std::uint8_t* __restrict__ member_mask,
                                        const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  work_p[c] = 0.0;
  work_sub[c] = 0.0;
  work_av[c] = 0.0;
}

__global__ void zero_member_aw_pressure_work_force_kernel(
    double* __restrict__ corner_force_p_rz_r,
    double* __restrict__ corner_force_p_rz_z,
    const std::uint8_t* __restrict__ member_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  const int base = 4 * c;
  for (int k = 0; k < 4; ++k) {
    corner_force_p_rz_r[base + k] = 0.0;
    corner_force_p_rz_z[base + k] = 0.0;
  }
}

__global__ void zero_internal_member_edge_forces_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    const int* __restrict__ cell_a,
    const int* __restrict__ cell_b,
    const std::uint8_t* __restrict__ member_mask,
    const int n_edges,
    const int offset) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_edges || member_mask == nullptr) {
    return;
  }
  const int a = cell_a[e];
  const int b = cell_b[e];
  if ((a >= 0 && member_mask[a] != 0U) || (b >= 0 && member_mask[b] != 0U)) {
    edge_force_r[offset + e] = 0.0;
    edge_force_z[offset + e] = 0.0;
  }
}

__global__ void zero_boundary_member_edge_forces_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    const int* __restrict__ cell,
    const std::uint8_t* __restrict__ member_mask,
    const int n_edges,
    const int offset) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_edges || member_mask == nullptr) {
    return;
  }
  const int c = cell[e];
  if (c >= 0 && member_mask[c] != 0U) {
    edge_force_r[offset + e] = 0.0;
    edge_force_z[offset + e] = 0.0;
  }
}

__global__ void boundary_pressure_force_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    double* __restrict__ boundary_impulse_r,
    double* __restrict__ boundary_impulse_z,
    const int* __restrict__ boundary_nodes,
    const int n_boundary_nodes,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double pressure,
    const double impulse_dt,
    const int rz_scheme_id) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n_boundary_nodes) {
    return;
  }
  if (rz_scheme_id == 1) {
    // Closed-loop contract per order_boundary_nodes; on-axis segment area vectors
    // are physically inert for r-forces via the axis projector.
    const int km1 = (k + n_boundary_nodes - 1) % n_boundary_nodes;
    const int kp1 = (k + 1) % n_boundary_nodes;
    const int nm1 = boundary_nodes[km1];
    const int n = boundary_nodes[k];
    const int np1 = boundary_nodes[kp1];
    double sr = 0.0;
    double sz = 0.0;
    const auto area_m = detail::planar_boundary_pressure_segment_area_vector(
        x_r[nm1], x_z[nm1], x_r[n], x_z[n]);
    sr += area_m.r;
    sz += area_m.z;
    const auto area_p = detail::planar_boundary_pressure_segment_area_vector(
        x_r[n], x_z[n], x_r[np1], x_z[np1]);
    sr += area_p.r;
    sz += area_p.z;
    const double fr = pressure * sr;
    const double fz = pressure * sz;
    atomicAdd(force_r + n, fr);
    atomicAdd(force_z + n, fz);
    boundary_impulse_r[k] = impulse_dt * fr;
    boundary_impulse_z[k] = impulse_dt * fz;
    return;
  }
  const int km1 = (k + n_boundary_nodes - 1) % n_boundary_nodes;
  const int kp1 = (k + 1) % n_boundary_nodes;
  const int nm1 = boundary_nodes[km1];
  const int n = boundary_nodes[k];
  const int np1 = boundary_nodes[kp1];
  const double rkm1 = x_r[nm1];
  const double rk = x_r[n];
  const double rkp1 = x_r[np1];
  const double zkm1 = x_z[nm1];
  const double zk = x_z[n];
  const double zkp1 = x_z[np1];
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double sr =
      pi_over_three *
      (2.0 * rk * (zkp1 - zkm1) + rkp1 * (zkp1 - zk) +
       rkm1 * (zk - zkm1));
  const double sz =
      pi_over_three *
      (rkm1 * (rkm1 + rk) - rkp1 * (rk + rkp1));
  const double fr = pressure * sr;
  const double fz = pressure * sz;
  atomicAdd(force_r + n, fr);
  atomicAdd(force_z + n, fz);
  boundary_impulse_r[k] = impulse_dt * fr;
  boundary_impulse_z[k] = impulse_dt * fz;
}

}  // namespace

bool configured(const core::Config& cfg) {
  return cfg.numerics.ale.central_pseudo_core_enabled;
}

bool active(const core::State& state) {
  return state.central_pseudo_core.configured &&
         state.central_pseudo_core.built &&
         state.central_pseudo_core.valid;
}

void rebuild_virtual_member_geometry(core::State& state,
                                     const core::Config& cfg,
                                     const char* phase) {
  (void)phase;
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "central pseudo-core virtual star-map requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  const CapTopoView view = make_cap_topo_view(cfg, mb);
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);
  if (!rebuild_trifan_virtual_member_geometry_host(
          state.central_pseudo_core, mb, view, node_r, node_z)) {
    return;
  }
  state.x_r.copy_from_host(node_r);
  state.x_z.copy_from_host(node_z);
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;
}

void ensure_built(core::State& state, const core::Config& cfg) {
  auto& pc = state.central_pseudo_core;
  if (pc.built) {
    return;
  }
  if (cfg.numerics.ale.central_pseudo_core_activation_time_s > 0.0 &&
      state.t < cfg.numerics.ale.central_pseudo_core_activation_time_s &&
      !state.central_pseudo_core.built) {
    return;
  }
  if (!configured(cfg)) {
    return;
  }
  core1d::Core1DParams core1d_params;
  core1d_params.enabled =
      cfg.numerics.ale.central_pseudo_core_core1d_enabled;
  core1d_params.build_shells =
      cfg.numerics.ale.central_pseudo_core_core1d_build_shells;
  core1d_params.split_append =
      cfg.numerics.ale.central_pseudo_core_core1d_split_append;
  core1d_params.av_c1 = cfg.numerics.ale.central_pseudo_core_core1d_av_c1;
  core1d_params.av_c2 = cfg.numerics.ale.central_pseudo_core_core1d_av_c2;
  core1d_params.cfl = cfg.numerics.ale.central_pseudo_core_core1d_cfl;
  core1d_params.piston_cap =
      cfg.numerics.ale.central_pseudo_core_core1d_piston_cap;
  core1d_params.max_substeps =
      cfg.numerics.ale.central_pseudo_core_core1d_max_substeps;
  core1d_params.dist_append =
      cfg.numerics.ale.central_pseudo_core_core1d_dist_append;
  core1d::set_params(core1d_params);
  TENRYU_ASSERT(is_supported_topology(cfg),
                "central pseudo-core Exp1 requires a 2D_RZ 5-block multiblock "
                "or polar-tier cart-center trifan_cap topology");
  TENRYU_ASSERT(!cfg.numerics.materials.per_material_conservation_enabled,
                "central pseudo-core Exp1 does not support per-material "
                "conservation");
  TENRYU_ASSERT(cfg.numerics.ale.central_pseudo_core_s_c > 0.0,
                "central pseudo-core requires central_pseudo_core_s_c > 0");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "central pseudo-core requires initialized multiblock topology");

  const auto& mb = *state.mesh.topo.multiblock;
  const CapTopoView view = make_cap_topo_view(cfg, mb);
  if (view.hybrid &&
      cfg.numerics.ale.central_pseudo_core_s_c >
          cfg.mesh.multiblock_cart_core_r_c) {
    std::ostringstream msg;
    msg << "central pseudo-core hybrid requires central_pseudo_core_s_c="
        << cfg.numerics.ale.central_pseudo_core_s_c
        << " <= multiblock_cart_core_r_c="
        << cfg.mesh.multiblock_cart_core_r_c;
    TENRYU_ASSERT(false, msg.str());
  }
  const int n_cells = state.mesh.topo.n_cells;
  pc.configured = true;
  pc.s_c = cfg.numerics.ale.central_pseudo_core_s_c;
  pc.member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.passive_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.inactive_member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.active_cell_class.assign(static_cast<std::size_t>(n_cells),
                              kActiveOrdinary);
  // Rebuild support (dynamic ring absorption): clear the derived containers
  // so a second ensure_built does not duplicate or reuse stale entries.
  pc.member_cells.clear();
  pc.passive_cells.clear();
  pc.boundary_nodes_ordered.clear();
  pc.boundary_node_mask.clear();
  pc.member_scatter_weights.clear();

  std::vector<double> node_r;
  std::vector<double> node_z;
  if (state.x_r_initial.size() == state.x_r.size()) {
    state.x_r_initial.copy_to_host(node_r);
    state.x_z_initial.copy_to_host(node_z);
  } else {
    state.x_r.copy_to_host(node_r);
    state.x_z.copy_to_host(node_z);
  }

  const int member_ring_count =
      select_ring_conforming_members(state.mesh,
                                     mb,
                                     view,
                                     node_r,
                                     node_z,
                                     pc.s_c,
                                     pc.min_member_ring_count,
                                     pc.member_mask);
  pc.member_ring_count = member_ring_count;

  for (int c = 0; c < n_cells; ++c) {
    if (pc.member_mask[static_cast<std::size_t>(c)] != 0U) {
      pc.member_cells.push_back(c);
    }
  }
  TENRYU_ASSERT(!pc.member_cells.empty(),
                "central pseudo-core selected no member cells");
  if (view.hybrid) {
    const int block_id = central_core_block_id(mb);
    const auto& bridge = hybrid_bridge_block(mb, view);
    const int bridge_block_id =
        static_cast<int>(&bridge - mb.blocks.data());
    for (const int c : pc.member_cells) {
      const int member_block_id =
          mb.cell_block_id[static_cast<std::size_t>(c)];
      TENRYU_ASSERT(member_block_id == block_id ||
                        member_block_id == bridge_block_id,
                    "central pseudo-core hybrid members must stay in the "
                    "central-core or bridge block");
    }
  }
  pc.representative_cell = pc.member_cells.front();
  for (const int c : pc.member_cells) {
    const std::size_t idx = static_cast<std::size_t>(c);
    pc.passive_mask[idx] = 1U;
    pc.inactive_member_mask[idx] = 1U;
    pc.active_cell_class[idx] = kInactiveCoreMember;
    pc.passive_cells.push_back(c);
  }
  tenryu::hydro::pole_angular_derefine::refresh_geometry_exempt_cells(state);

  std::vector<std::pair<int, int>> boundary_edges;
  for (const int c : pc.member_cells) {
    const int off_nodes = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int off_faces = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(state.mesh, c);
    std::array<int, 4> nodes = {-1, -1, -1, -1};
    for (int k = 0; k < nverts; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off_nodes + k)];
    }
    for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
      const int nb = mb.face_adj_csr_indices[static_cast<std::size_t>(off_faces + f)];
      if (nb >= 0 && nb < n_cells &&
          pc.member_mask[static_cast<std::size_t>(nb)] != 0U) {
        continue;
      }
      const auto edge = face_nodes(nverts, nodes, f);
      if (edge.first >= 0 && edge.second >= 0) {
        boundary_edges.push_back(edge);
      }
    }
  }
  pc.boundary_nodes_ordered =
      order_boundary_nodes(boundary_edges, node_r, node_z);
  const int n_nodes = static_cast<int>(node_r.size());
  pc.boundary_node_mask.assign(static_cast<std::size_t>(n_nodes), 0U);
  for (const int n : pc.boundary_nodes_ordered) {
    TENRYU_ASSERT(n >= 0 && n < n_nodes,
                  "central pseudo-core boundary node out of range");
    pc.boundary_node_mask[static_cast<std::size_t>(n)] = 1U;
  }
  std::vector<double> mass = copy_field(state.mass);
  initialize_member_scatter_weights(pc, mass);
  pc.V_c = boundary_loop_volume(pc, node_r, node_z);
  project_activation_kinetic_energy(state);
  upload_overlay(state);
  pc.built = true;
  pc.valid = true;

  // Stratified 1D sub-model: first build only (rebuilds after absorption
  // keep the running sub-model; absorb deltas are fed by
  // execute_ring_absorption).
  if (core1d::enabled() && !core1d::built(&pc) &&
      !cfg.materials.materials.empty() &&
      cfg.materials.materials.front().eos_model == "ideal_gas") {
    const auto& mb1d = *state.mesh.topo.multiblock;
    std::vector<double> vol_h = copy_field(state.vol);
    std::vector<double> ee_h = copy_field(state.ee);
    std::vector<double> ei_h = state.ei.empty() ? std::vector<double>{}
                                                : copy_field(state.ei);
    std::vector<double> tr_h = state.gas_tracer_Y.empty()
                                   ? std::vector<double>{}
                                   : copy_field(state.gas_tracer_Y);
    std::vector<core1d::BuildCell> cells1d;
    cells1d.reserve(pc.member_cells.size());
    for (const int cidx : pc.member_cells) {
      const std::size_t ci = static_cast<std::size_t>(cidx);
      const auto nodes = multiblock_cell_nodes(mb1d, cidx);
      double rr = 0.0;
      for (const int nn : nodes) {
        rr += std::hypot(node_r[static_cast<std::size_t>(nn)],
                         node_z[static_cast<std::size_t>(nn)]);
      }
      core1d::BuildCell bc;
      bc.r = 0.25 * rr;
      bc.m = mass[ci];
      bc.v = vol_h[ci];
      bc.e = ee_h[ci] + (ei_h.empty() ? 0.0 : ei_h[ci]);
      bc.Y = tr_h.empty() ? 0.0 : tr_h[ci];
      cells1d.push_back(bc);
    }
    core1d::build(&pc, std::move(cells1d), pc.V_c,
                  cfg.materials.materials.front().ideal_gas_gamma);
  }

  std::fprintf(stderr,
               "[central_pseudo_core] built step=%d s_c=%.17e rings=%d rep=%d "
               "members=%zu passive=%zu boundary_nodes=%zu\n",
               state.step,
               pc.s_c,
               member_ring_count,
               pc.representative_cell,
               pc.member_cells.size(),
               pc.passive_cells.size(),
               pc.boundary_nodes_ordered.size());
}

namespace {

// Namelist-first with environment override: the historical
// TENRYU_I1B_RING_ABSORB* variables win when SET (experimental decks),
// otherwise the Numerics.ale.central_pseudo_core_ring_absorption_* keys
// apply.
bool ring_absorption_enabled(const core::Config& cfg) {
  static const int env_state = [] {
    const char* raw = std::getenv("TENRYU_I1B_RING_ABSORB");
    if (raw == nullptr || raw[0] == '\0') {
      return -1;
    }
    return raw[0] != '0' ? 1 : 0;
  }();
  if (env_state >= 0) {
    return env_state == 1;
  }
  return cfg.numerics.ale.central_pseudo_core_ring_absorption_enabled;
}

int ring_absorption_max_rings(const core::Config& cfg) {
  // Default: no fixed cap — the material/topology guard is the binding
  // limit, so the cap scales with resolution. Env or namelist provide an
  // optional tighter override (0 = unlimited).
  static const int env_value = [] {
    const char* raw = std::getenv("TENRYU_I1B_RING_ABSORB_MAX_RINGS");
    return raw != nullptr ? std::atoi(raw) : -1;
  }();
  const int v =
      env_value >= 0
          ? env_value
          : cfg.numerics.ale.central_pseudo_core_ring_absorption_max_rings;
  return v > 0 ? v : std::numeric_limits<int>::max();
}

// Count absorbable PURE-GAS rows. For a hybrid cap this is the contiguous
// radial BRIDGE prefix beyond the current membership, and an absent tracer
// accepts every remaining row. Non-hybrid shell behavior is unchanged: it
// returns the full shell prefix and requires a tracer field.
int absorbable_shell_row_count(core::State& state,
                               const core::Config& cfg,
                               const mesh::MultiBlockTopology& mb,
                               const mesh::BlockInfo* bridge_hint = nullptr,
                               const bool* bridge_zero_is_inner_hint = nullptr) {
  const CapTopoView view = make_cap_topo_view(cfg, mb);
  if (!view.has_cap) {
    return 0;
  }
  const mesh::BlockInfo* hybrid_bridge = nullptr;
  bool bridge_zero_is_inner = true;
  int first_bridge_row = 0;
  if (view.hybrid) {
    if (bridge_hint != nullptr) {
      TENRYU_ASSERT(bridge_zero_is_inner_hint != nullptr &&
                        bridge_hint == &hybrid_bridge_block(mb, view),
                    "central pseudo-core hybrid bridge orientation hint mismatch");
      hybrid_bridge = bridge_hint;
      bridge_zero_is_inner = *bridge_zero_is_inner_hint;
    } else {
      TENRYU_ASSERT(bridge_zero_is_inner_hint == nullptr,
                    "central pseudo-core hybrid bridge hint is incomplete");
      hybrid_bridge = &hybrid_bridge_block(mb, view);
      std::vector<double> node_r = copy_field(state.x_r);
      std::vector<double> node_z = copy_field(state.x_z);
      bridge_zero_is_inner = hybrid_bridge_row_zero_is_inner(
          mb, *hybrid_bridge, node_r, node_z);
    }
    first_bridge_row =
        std::max(0, state.central_pseudo_core.member_ring_count - view.n_cap);
    TENRYU_ASSERT(first_bridge_row <= hybrid_bridge->n_i_cells,
                  "central pseudo-core hybrid bridge membership depth out of range");
    if (state.gas_tracer_Y.empty()) {
      return hybrid_bridge->n_i_cells - first_bridge_row;
    }
  } else if (state.gas_tracer_Y.empty()) {
    return 0;
  }
  // The binding statistic is the MASS-WEIGHTED row tracer fraction: a
  // per-cell row minimum is hostage to single pole-column cells whose
  // tracer dips from remap diffusion while the row is still physically
  // gas. A genuinely mixed interface row fails the weighted gate by a
  // wide margin. A loose per-cell hard bound additionally refuses any
  // row whose worst cell is majority shell material.
  static const double env_tracer_floor = [] {
    const char* raw = std::getenv("TENRYU_I1B_RING_ABSORB_GAS_TRACER_MIN");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v < 1.0) ? v : -1.0;
  }();
  static const double env_cell_floor = [] {
    const char* raw =
        std::getenv("TENRYU_I1B_RING_ABSORB_GAS_TRACER_CELL_MIN");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v < 1.0) ? v : -1.0;
  }();
  const double tracer_floor =
      env_tracer_floor > 0.0
          ? env_tracer_floor
          : cfg.numerics.ale
                .central_pseudo_core_ring_absorption_gas_tracer_min;
  const double cell_floor =
      env_cell_floor > 0.0
          ? env_cell_floor
          : cfg.numerics.ale
                .central_pseudo_core_ring_absorption_gas_tracer_cell_min;
  if (view.hybrid) {
    const std::vector<double> tracer = copy_field(state.gas_tracer_Y);
    const std::vector<double> mass = copy_field(state.mass);
    int rows = 0;
    for (int q = first_bridge_row; q < hybrid_bridge->n_i_cells; ++q) {
      const int row = hybrid_bridge_storage_row(
          *hybrid_bridge, q, bridge_zero_is_inner);
      double row_min = std::numeric_limits<double>::infinity();
      double mass_sum = 0.0;
      double mass_tracer_sum = 0.0;
      for (int j = 0; j < hybrid_bridge->n_j_cells; ++j) {
        const int c = block_cell_id(*hybrid_bridge, row, j);
        TENRYU_ASSERT(c >= 0 &&
                          static_cast<std::size_t>(c) < tracer.size() &&
                          static_cast<std::size_t>(c) < mass.size(),
                      "central pseudo-core bridge tracer cell out of range");
        const double y = tracer[static_cast<std::size_t>(c)];
        const double m = mass[static_cast<std::size_t>(c)];
        row_min = std::min(row_min, y);
        mass_sum += m;
        mass_tracer_sum += m * y;
      }
      const double weighted =
          mass_sum > 0.0 ? mass_tracer_sum / mass_sum : 0.0;
      if (!(weighted >= tracer_floor) || !(row_min >= cell_floor)) {
        break;
      }
      ++rows;
    }
    return rows;
  }
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  const std::vector<double> tracer = copy_field(state.gas_tracer_Y);
  const std::vector<double> mass = copy_field(state.mass);
  int rows = 0;
  for (int q = 0; q < shell.n_i_cells; ++q) {
    double row_min = std::numeric_limits<double>::infinity();
    double mass_sum = 0.0;
    double mass_tracer_sum = 0.0;
    for (int j = 0; j < shell.n_j_cells; ++j) {
      const int c = block_cell_id(shell, q, j);
      TENRYU_ASSERT(c >= 0 &&
                        static_cast<std::size_t>(c) < tracer.size() &&
                        static_cast<std::size_t>(c) < mass.size(),
                    "central pseudo-core shell tracer cell out of range");
      const double y = tracer[static_cast<std::size_t>(c)];
      const double m = mass[static_cast<std::size_t>(c)];
      row_min = std::min(row_min, y);
      mass_sum += m;
      mass_tracer_sum += m * y;
    }
    const double weighted =
        mass_sum > 0.0 ? mass_tracer_sum / mass_sum : 0.0;
    if (!(weighted >= tracer_floor) || !(row_min >= cell_floor)) {
      break;
    }
    ++rows;
  }
  return rows;
}

// Emergency mixed-material absorption (macro-boundary terminal-phase verdict,
// rank-1 direction (a), ideal-gas MVP): when the pure-gas ladder is
// exhausted and the macro boundary itself is the failure surface, allow
// absorbing the immediately-next DENSE shell row — bypassing the gas-tracer
// guard but NOT the structural row limit — gated on the static
// admissibility (positive RZ volume + simple loop at CURRENT node
// positions) of the WOULD-BE new outer boundary. The verdict's key point:
// the check belongs on the proposed NEW boundary, because absorbing the
// failed boundary IS the rescue. Thermodynamics: with the present
// single-gamma ideal gas, the pooled macro closure p_C=(gamma-1)U_C/V_C is
// EXACTLY the two-material pressure-equilibrium answer
// (p_C = sum_m (gamma_m-1)U_m / V_C degenerates when gamma_g = gamma_s), so
// no closure change is needed; a true multi-material macro state is the
// contract for any future per-material-EOS physics. Env-gated, default off.
bool mixed_absorption_enabled(const core::Config& cfg) {
  static const char* raw = std::getenv("TENRYU_I1B_MIXED_ABSORB");
  if (raw != nullptr && raw[0] != '\0') {
    return raw[0] != '0';
  }
  return cfg.numerics.ale.central_pseudo_core_mixed_absorb_enabled;
}

int absorb_watch_rows(const core::Config& cfg) {
  static const char* raw = std::getenv("TENRYU_I1B_ABSORB_WATCH_ROWS");
  if (raw != nullptr && raw[0] != '\0') {
    const int v = std::atoi(raw);
    return v > 0 ? v : 1;
  }
  return cfg.numerics.ale.central_pseudo_core_absorb_watch_rows;
}

double spherical_absorb_alpha(const core::Config& cfg) {
  static const char* raw = std::getenv("TENRYU_I1B_SPHERICAL_ABSORB_ALPHA");
  if (raw != nullptr && raw[0] != '\0') {
    const double v = std::atof(raw);
    return (std::isfinite(v) && v > 0.0 && v < 1.0) ? v : 0.0;
  }
  return cfg.numerics.ale.central_pseudo_core_spherical_absorb_alpha;
}

double spherical_absorb_pjump(const core::Config& cfg) {
  static const char* raw = std::getenv("TENRYU_I1B_SPHERICAL_ABSORB_PJUMP");
  if (raw != nullptr && raw[0] != '\0') {
    const double v = std::atof(raw);
    return (std::isfinite(v) && v > 1.0) ? v : 0.0;
  }
  return cfg.numerics.ale.central_pseudo_core_spherical_absorb_pjump;
}

bool spherical_absorb_gasfront(const core::Config& cfg) {
  static const char* raw =
      std::getenv("TENRYU_I1B_SPHERICAL_ABSORB_GASFRONT");
  if (raw != nullptr && raw[0] != '\0') {
    return raw[0] != '0';
  }
  return cfg.numerics.ale.central_pseudo_core_spherical_absorb_gasfront;
}

// Static admissibility of the hypothetical boundary loop after absorbing to
// `target_rings`: membership is selected on the INITIAL geometry (the same
// topological rule ensure_built uses) and the loop ORDER is derived there,
// but volume and simplicity are evaluated at the CURRENT committed node
// positions — the question is whether today's mesh can host the new loop.
bool hypothetical_boundary_loop_admissible(core::State& state,
                                           const CapTopoView& view,
                                           const int target_rings,
                                           double* v_out,
                                           int* simple_out) {
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  auto& pc = state.central_pseudo_core;

  std::vector<double> node_r_sel;
  std::vector<double> node_z_sel;
  if (state.x_r_initial.size() == state.x_r.size()) {
    state.x_r_initial.copy_to_host(node_r_sel);
    state.x_z_initial.copy_to_host(node_z_sel);
  } else {
    state.x_r.copy_to_host(node_r_sel);
    state.x_z.copy_to_host(node_z_sel);
  }
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(n_cells), 0U);
  const int got_rings = select_ring_conforming_members(
      state.mesh, mb, view, node_r_sel, node_z_sel, pc.s_c, target_rings,
      mask);
  if (got_rings < target_rings) {
    return false;
  }
  std::vector<std::pair<int, int>> boundary_edges;
  for (int c = 0; c < n_cells; ++c) {
    if (mask[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off_nodes =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int off_faces =
        mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(state.mesh, c);
    std::array<int, 4> nodes = {-1, -1, -1, -1};
    for (int k = 0; k < nverts; ++k) {
      nodes[k] =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off_nodes + k)];
    }
    for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
      const int nb =
          mb.face_adj_csr_indices[static_cast<std::size_t>(off_faces + f)];
      if (nb >= 0 && nb < n_cells && mask[static_cast<std::size_t>(nb)] != 0U) {
        continue;
      }
      const auto edge = face_nodes(nverts, nodes, f);
      if (edge.first >= 0 && edge.second >= 0) {
        boundary_edges.push_back(edge);
      }
    }
  }
  if (boundary_edges.size() < 3U) {
    return false;
  }
  const std::vector<int> loop =
      order_boundary_nodes(boundary_edges, node_r_sel, node_z_sel);
  if (loop.size() < 3U) {
    return false;
  }
  std::vector<double> node_r_cur;
  std::vector<double> node_z_cur;
  state.x_r.copy_to_host(node_r_cur);
  state.x_z.copy_to_host(node_z_cur);
  const int n_nodes = static_cast<int>(node_r_cur.size());
  for (const int n : loop) {
    if (n < 0 || n >= n_nodes) {
      return false;
    }
  }
  const double v = mesh::path_admissibility_detail::rz_polygon_volume(
      node_r_cur, node_z_cur, loop);
  const bool is_simple = mesh::path_admissibility_detail::simple_loop(
      node_r_cur, node_z_cur, loop);
  if (v_out != nullptr) {
    *v_out = v;
  }
  if (simple_out != nullptr) {
    *simple_out = is_simple ? 1 : 0;
  }
  return std::isfinite(v) && v > 0.0 && is_simple;
}

// Mass-weighted material stats of one polar-shell row (for the emergency
// absorption log: how mixed is the row the macro is about to swallow).
void shell_row_material_stats(const core::State& state,
                              const mesh::MultiBlockTopology& mb,
                              const int q_row,
                              double* mass_out,
                              double* weighted_y_out) {
  *mass_out = 0.0;
  *weighted_y_out = 0.0;
  if (state.gas_tracer_Y.empty()) {
    return;
  }
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  if (q_row < 0 || q_row >= shell.n_i_cells) {
    return;
  }
  const std::vector<double> tracer = copy_field(state.gas_tracer_Y);
  const std::vector<double> mass = copy_field(state.mass);
  double mass_sum = 0.0;
  double mass_tracer_sum = 0.0;
  for (int j = 0; j < shell.n_j_cells; ++j) {
    const int c = block_cell_id(shell, q_row, j);
    if (c < 0 || static_cast<std::size_t>(c) >= mass.size()) {
      continue;
    }
    mass_sum += mass[static_cast<std::size_t>(c)];
    mass_tracer_sum += mass[static_cast<std::size_t>(c)] *
                       tracer[static_cast<std::size_t>(c)];
  }
  *mass_out = mass_sum;
  *weighted_y_out = mass_sum > 0.0 ? mass_tracer_sum / mass_sum : 0.0;
}

// Rebuild the macro cell with `target_rings` member rings. Must run at the
// start of a Lagrangian step (before the "hydro_step_start" sync aggregate):
// project_activation_kinetic_energy banks the newly passive nodes' KE and the
// immediately following sync aggregate deposits it while re-summing Ue/Ui
// from the absorbed ring's still-real cell state.
// Endgame reference consistency (verdict #5 Q4; measured 2026-07-02): once
// absorption grows the core, the persistent reference still describes the
// ORIGINAL core extent — reference cells overlap the new members, the
// near-boundary swept volumes of the per-step CSR reference remap cannot
// close, and mass is CREATED inside `ale_csr_swept_remap` (+3.3e-7/step
// measured at stage level; also seen as negative reference target volumes at
// core-adjacent cells). After every absorption commit, sync the reference
// coordinates to the CURRENT mesh for all nodes of member/passive cells plus
// a one-cell halo: the remap then demands zero motion there and the swept
// partition closes by construction. Farther out the reference is untouched.
void sync_reference_to_current_around_core(core::State& state) {
  auto& pc = state.central_pseudo_core;
  if (!state.mesh.topo.multiblock.has_value() ||
      state.x_r_reference.size() != state.x_r.size() ||
      state.x_z_reference.size() != state.x_z.size() ||
      state.x_r_reference.empty()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<std::uint8_t> cell_mark(static_cast<std::size_t>(n_cells), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    if ((idx < pc.member_mask.size() && pc.member_mask[idx] != 0U) ||
        (idx < pc.passive_mask.size() && pc.passive_mask[idx] != 0U) ||
        (idx < pc.inactive_member_mask.size() &&
         pc.inactive_member_mask[idx] != 0U)) {
      cell_mark[idx] = 1U;
    }
  }
  std::vector<std::uint8_t> halo = cell_mark;
  for (int c = 0; c < n_cells; ++c) {
    if (cell_mark[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
      const int adj =
          mb.face_adj_csr_indices[static_cast<std::size_t>(off + f)];
      if (adj >= 0 && adj < n_cells) {
        halo[static_cast<std::size_t>(adj)] = 1U;
      }
    }
  }
  std::vector<double> ref_r = copy_field(state.x_r_reference);
  std::vector<double> ref_z = copy_field(state.x_z_reference);
  std::vector<double> cur_r = copy_field(state.x_r);
  std::vector<double> cur_z = copy_field(state.x_z);
  int synced = 0;
  for (int c = 0; c < n_cells; ++c) {
    if (halo[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    for (int k = off; k < end; ++k) {
      const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(k)];
      if (node < 0 || node >= n_nodes) {
        continue;
      }
      const std::size_t nn = static_cast<std::size_t>(node);
      if (ref_r[nn] != cur_r[nn] || ref_z[nn] != cur_z[nn]) {
        ++synced;
      }
      ref_r[nn] = cur_r[nn];
      ref_z[nn] = cur_z[nn];
    }
  }
  state.x_r_reference.copy_from_host(ref_r);
  state.x_z_reference.copy_from_host(ref_z);
  if (synced > 0 && (synced > 50 || (state.step % 200) == 0)) {
    std::fprintf(stderr,
                 "[central_pseudo_core] reference_synced_around_core "
                 "nodes_updated=%d step=%d\n",
                 synced,
                 state.step);
  }
}

// Detect-gated reference repair (verdict #5 Q2 "refuse broken targets";
// replaces the REJECTED unconditional per-step tracking, which interleaved
// with the axis-rezone persistent-reference override dance and corrupted it).
// At hydro_step_start (outside any ALE-phase reference override): compute the
// exact signed RZ polygon volume of every core-halo cell on the PERSISTENT
// reference coordinates; if any is non-positive (or vanished vs the current
// volume), the stale reference would make the next CSR swept remap create
// mass — repair by syncing the core-halo reference to the current mesh.
double host_rz_polygon_volume(const std::vector<double>& r,
                              const std::vector<double>& z,
                              const int* nodes,
                              const int count) {
  long double acc = 0.0L;
  for (int k = 0; k < count; ++k) {
    const int a = nodes[k];
    const int b = nodes[(k + 1) % count];
    const long double ra = r[static_cast<std::size_t>(a)];
    const long double rb = r[static_cast<std::size_t>(b)];
    const long double za = z[static_cast<std::size_t>(a)];
    const long double zb = z[static_cast<std::size_t>(b)];
    acc += (ra * ra + ra * rb + rb * rb) * (zb - za);
  }
  constexpr long double kPiOver3 = 1.04719755119659774615L;
  return static_cast<double>(kPiOver3 * acc);
}

void detect_and_repair_stale_core_reference_impl(core::State& state) {
  auto& pc = state.central_pseudo_core;
  if (!state.mesh.topo.multiblock.has_value() ||
      state.x_r_reference.size() != state.x_r.size() ||
      state.x_r_reference.empty() || !pc.built || !pc.valid) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  std::vector<std::uint8_t> cell_mark(static_cast<std::size_t>(n_cells), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    if ((idx < pc.member_mask.size() && pc.member_mask[idx] != 0U) ||
        (idx < pc.passive_mask.size() && pc.passive_mask[idx] != 0U) ||
        (idx < pc.inactive_member_mask.size() &&
         pc.inactive_member_mask[idx] != 0U)) {
      cell_mark[idx] = 1U;
    }
  }
  std::vector<std::uint8_t> halo = cell_mark;
  for (int c = 0; c < n_cells; ++c) {
    if (cell_mark[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
      const int adj =
          mb.face_adj_csr_indices[static_cast<std::size_t>(off + f)];
      if (adj >= 0 && adj < n_cells) {
        halo[static_cast<std::size_t>(adj)] = 1U;
      }
    }
  }
  const std::vector<double> ref_r = copy_field(state.x_r_reference);
  const std::vector<double> ref_z = copy_field(state.x_z_reference);
  const std::vector<double> cur_r = copy_field(state.x_r);
  const std::vector<double> cur_z = copy_field(state.x_z);
  int nodes[8];
  int broken_cell = -1;
  double broken_ref_vol = 0.0;
  double broken_cur_vol = 0.0;
  for (int c = 0; c < n_cells && broken_cell < 0; ++c) {
    if (halo[static_cast<std::size_t>(c)] == 0U ||
        cell_mark[static_cast<std::size_t>(c)] != 0U) {
      continue;  // repair targets the RESOLVED halo; members are exempt
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    int count = 0;
    for (int k = off; k < end && count < 8; ++k) {
      const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(k)];
      if (node >= 0 &&
          node < static_cast<int>(ref_r.size())) {
        nodes[count++] = node;
      }
    }
    if (count < 3) {
      continue;
    }
    const double v_ref = host_rz_polygon_volume(ref_r, ref_z, nodes, count);
    const double v_cur = host_rz_polygon_volume(cur_r, cur_z, nodes, count);
    if (!(std::isfinite(v_ref)) || v_ref * v_cur <= 0.0 ||
        std::abs(v_ref) < 1.0e-3 * std::abs(v_cur)) {
      broken_cell = c;
      broken_ref_vol = v_ref;
      broken_cur_vol = v_cur;
    }
  }
  if (broken_cell >= 0) {
    std::fprintf(stderr,
                 "[central_pseudo_core] stale_reference_repair step=%d "
                 "cell=%d v_ref=%.6e v_cur=%.6e -> syncing core-halo "
                 "reference to current\n",
                 state.step,
                 broken_cell,
                 broken_ref_vol,
                 broken_cur_vol);
    sync_reference_to_current_around_core(state);
  }
}

void execute_ring_absorption(core::State& state,
                             const core::Config& cfg,
                             const int target_rings,
                             const char* trigger) {
  auto& pc = state.central_pseudo_core;
  const int previous_rings = pc.member_ring_count;
  emit_boundary_asphericity(state, "pre_absorb");
  std::vector<int> core1d_old_members;
  if (core1d::enabled() && core1d::built(&pc)) {
    core1d_old_members = pc.member_cells;
  }
  std::fprintf(stderr,
               "[central_pseudo_core] ring absorption (%s): rebuild with %d "
               "rings (was %d) step=%d\n",
               trigger,
               target_rings,
               previous_rings,
               state.step);
  pc.min_member_ring_count = target_rings;
  pc.built = false;
  pc.valid = false;
  pc.activation_ke_valid = false;
  ensure_built(state, cfg);
  // Feed the newly absorbed cells to the stratified 1D sub-model as one
  // outer shell (mass-weighted specific energy, radial velocity and gas
  // fraction), preserving momentum instead of thermalizing it. The delta is
  // taken against the pre-rebuild membership; cell state is still the real
  // pre-pool state here (mirrors of the new members are rewritten later by
  // the aggregate that follows).
  if (core1d::enabled() && core1d::built(&pc) &&
      !core1d_old_members.empty() &&
      state.mesh.topo.multiblock.has_value()) {
    const int n_cells = state.mesh.topo.n_cells;
    std::vector<std::uint8_t> was(static_cast<std::size_t>(n_cells), 0U);
    for (const int c : core1d_old_members) {
      if (c >= 0 && c < n_cells) {
        was[static_cast<std::size_t>(c)] = 1U;
      }
    }
    std::vector<double> mass_h = copy_field(state.mass);
    std::vector<double> vol1d_h = copy_field(state.vol);
    std::vector<double> ee_h = copy_field(state.ee);
    std::vector<double> ei_h = state.ei.empty() ? std::vector<double>{}
                                                : copy_field(state.ei);
    std::vector<double> tr_h = state.gas_tracer_Y.empty()
                                   ? std::vector<double>{}
                                   : copy_field(state.gas_tracer_Y);
    std::vector<double> vr_h = copy_field(state.v_r);
    std::vector<double> vz_h = copy_field(state.v_z);
    std::vector<double> nr_h = copy_field(state.x_r);
    std::vector<double> nz_h = copy_field(state.x_z);
    const auto& mb1d = *state.mesh.topo.multiblock;
    const CapTopoView view1d = make_cap_topo_view(cfg, mb1d);
    // Verdict #6 Q3: ONE 1D shell per absorption UNIT (radial order), not
    // one pooled shell per event — a cascade otherwise destroys the radial
    // stratification. Unit depth uses the same mapping arithmetic as
    // request_ring_absorption.
    const int cb_id = central_core_block_id(mb1d);
    const auto& cb = mb1d.blocks[static_cast<std::size_t>(cb_id)];
    const mesh::BlockInfo* bridge1d = nullptr;
    bool bridge_zero_is_inner = true;
    if (view1d.hybrid) {
      bridge1d = &hybrid_bridge_block(mb1d, view1d);
      bridge_zero_is_inner = hybrid_bridge_row_zero_is_inner(
          mb1d, *bridge1d, nr_h, nz_h);
    }
    const auto depth_of = [&](const int c) -> int {
      if (c >= cb.cell_begin && c < cb.cell_begin + cb.cell_count) {
        return (c - cb.cell_begin) / cb.n_j_cells;
      }
      if (view1d.has_cap) {
        if (view1d.hybrid) {
          if (c >= bridge1d->cell_begin &&
              c < bridge1d->cell_begin + bridge1d->cell_count) {
            const int storage_row =
                (c - bridge1d->cell_begin) / bridge1d->n_j_cells;
            return cb.n_i_cells + hybrid_bridge_radial_row(
                                      *bridge1d,
                                      storage_row,
                                      bridge_zero_is_inner);
          }
          return -1;
        }
        for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                          mesh::BlockRole::EAST_FAN,
                                          mesh::BlockRole::SOUTH_FAN}) {
          const auto& fb = mesh::mesh_topo_trifan_fan_block(mb1d, fan);
          if (c >= fb.cell_begin && c < fb.cell_begin + fb.cell_count) {
            return cb.n_i_cells + (c - fb.cell_begin) / fb.n_j_cells;
          }
        }
        const auto& sh = mesh::mesh_topo_multiblock_polar_shell_block(mb1d);
        if (c >= sh.cell_begin && c < sh.cell_begin + sh.cell_count) {
          const auto& north = mesh::mesh_topo_trifan_fan_block(
              mb1d, mesh::BlockRole::NORTH_FAN);
          return cb.n_i_cells + north.n_i_cells +
                 (c - sh.cell_begin) / sh.n_j_cells;
        }
      }
      return -1;
    };
    struct Core1dGroup {
      double M = 0.0;
      double U = 0.0;
      double Pr = 0.0;
      double Kfull = 0.0;
      double MY = 0.0;
      double V = 0.0;
    };
    std::map<int, Core1dGroup> groups;
    std::map<int, std::vector<core1d::ShellSample>> group_samples;
    const bool dist_append = core1d::dist_append_enabled();
    for (const int c : pc.member_cells) {
      if (c < 0 || c >= n_cells || was[static_cast<std::size_t>(c)] != 0U) {
        continue;
      }
      const std::size_t ci = static_cast<std::size_t>(c);
      const double m = mass_h[ci];
      const double e = ee_h[ci] + (ei_h.empty() ? 0.0 : ei_h[ci]);
      const double Y = tr_h.empty() ? 0.0 : tr_h[ci];
      const auto nodes = multiblock_cell_nodes(mb1d, c);
      double vr_bar = 0.0, vz_bar = 0.0, rc = 0.0, zc = 0.0;
      for (const int nn : nodes) {
        const std::size_t ni = static_cast<std::size_t>(nn);
        vr_bar += vr_h[ni];
        vz_bar += vz_h[ni];
        rc += nr_h[ni];
        zc += nz_h[ni];
      }
      vr_bar *= 0.25;
      vz_bar *= 0.25;
      rc *= 0.25;
      zc *= 0.25;
      const double s = std::hypot(rc, zc);
      const double ur =
          s > 0.0 ? (vr_bar * rc + vz_bar * zc) / s : 0.0;
      Core1dGroup& g = groups[depth_of(c)];
      g.M += m;
      g.U += m * e;
      g.Pr += m * ur;
      // Full cell-mean kinetic energy; the radially unresolved part is
      // thermalized on append (verdict #6 Q3-2: K_r = p_r^2/2M is the
      // only kinetic energy a 1D shell can carry — the remainder must
      // not vanish).
      g.Kfull += 0.5 * m * (vr_bar * vr_bar + vz_bar * vz_bar);
      g.MY += m * Y;
      g.V += vol1d_h[ci];
      if (dist_append && m > 0.0) {
        core1d::ShellSample sample;
        sample.m = m;
        // The cell's own non-radial KE thermalizes into ITS sample; the
        // ring's radial-velocity DISTRIBUTION survives into the sub-model.
        sample.e = e + std::max(0.5 * (vr_bar * vr_bar + vz_bar * vz_bar) -
                                    0.5 * ur * ur,
                                0.0);
        sample.u_r = ur;
        sample.Y = Y;
        sample.V = vol1d_h[ci];
        group_samples[depth_of(c)].push_back(sample);
      }
    }
    for (const auto& kv : groups) {
      const Core1dGroup& g = kv.second;
      if (!(g.M > 0.0) || !(g.V > 0.0)) {
        continue;
      }
      if (dist_append) {
        auto it_s = group_samples.find(kv.first);
        if (it_s != group_samples.end() && !it_s->second.empty()) {
          core1d::absorb_event_distribution(&pc, std::move(it_s->second));
          continue;
        }
      }
      const double Kr = g.Pr * g.Pr / (2.0 * g.M);
      const double e_spec =
          (g.U + std::max(g.Kfull - Kr, 0.0)) / g.M;
      core1d::absorb_event(&pc, g.M, e_spec, g.Pr / g.M, g.MY / g.M, g.V);
    }
  }
  pc.absorbed_ring_count += std::max(target_rings - previous_rings, 0);
  pc.absorb_watch_birth_by_depth.clear();
  sync_reference_to_current_around_core(state);
  emit_boundary_asphericity(state, "post_absorb");
}

}  // namespace

bool request_ring_absorption(core::State& state,
                             const core::Config& cfg,
                             const int failing_cell) {
  if (!ring_absorption_enabled(cfg) || !configured(cfg)) {
    return false;
  }
  auto& pc = state.central_pseudo_core;
  if (!pc.built || !pc.valid || !state.mesh.topo.multiblock.has_value()) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const CapTopoView view = make_cap_topo_view(cfg, mb);
  const int block_id = central_core_block_id(mb);
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  const mesh::BlockInfo* hybrid_bridge = nullptr;
  bool bridge_zero_is_inner = true;
  if (view.hybrid) {
    hybrid_bridge = &hybrid_bridge_block(mb, view);
    std::vector<double> node_r = copy_field(state.x_r);
    std::vector<double> node_z = copy_field(state.x_z);
    bridge_zero_is_inner = hybrid_bridge_row_zero_is_inner(
        mb, *hybrid_bridge, node_r, node_z);
  }
  // Map the failing cell to an absorption depth: core ring i -> depth i;
  // a hybrid bridge row in increasing-radius order -> depth n_cap + q;
  // fan layer l (non-hybrid trifan only) -> depth n_cap + l.
  int depth = -1;
  int max_depth = block.n_i_cells - 1;
  if (failing_cell == mesh::kCentralMacroCoreSentinelCell) {
    // The macro boundary ITSELF is path-inadmissible (sentinel failure):
    // absorb the first active unit so the boundary moves outward onto
    // healthier mesh. All downstream guards apply unchanged.
    depth = pc.member_ring_count;
  } else if (failing_cell >= block.cell_begin &&
             failing_cell < block.cell_begin + block.cell_count) {
    depth = (failing_cell - block.cell_begin) / block.n_j_cells;
  } else if (view.has_cap) {
    if (view.hybrid) {
      if (failing_cell >= hybrid_bridge->cell_begin &&
          failing_cell <
              hybrid_bridge->cell_begin + hybrid_bridge->cell_count) {
        const int storage_row =
            (failing_cell - hybrid_bridge->cell_begin) /
            hybrid_bridge->n_j_cells;
        depth = block.n_i_cells + hybrid_bridge_radial_row(
                                      *hybrid_bridge,
                                      storage_row,
                                      bridge_zero_is_inner);
      }
    } else {
      for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                        mesh::BlockRole::EAST_FAN,
                                        mesh::BlockRole::SOUTH_FAN}) {
        const auto& fan_block = mesh::mesh_topo_trifan_fan_block(mb, fan);
        if (failing_cell >= fan_block.cell_begin &&
            failing_cell < fan_block.cell_begin + fan_block.cell_count) {
          depth = block.n_i_cells +
                  (failing_cell - fan_block.cell_begin) /
                      fan_block.n_j_cells;
          break;
        }
      }
      if (depth < 0) {
        const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
        if (failing_cell >= shell.cell_begin &&
            failing_cell < shell.cell_begin + shell.cell_count) {
          const auto& north = mesh::mesh_topo_trifan_fan_block(
              mb, mesh::BlockRole::NORTH_FAN);
          depth = block.n_i_cells + north.n_i_cells +
                  (failing_cell - shell.cell_begin) / shell.n_j_cells;
        }
      }
    }
  }
  if (view.has_cap) {
    if (view.hybrid) {
      const int absorbed_bridge_rows =
          std::max(0, pc.member_ring_count - view.n_cap);
      max_depth = view.n_cap + absorbed_bridge_rows +
                  absorbable_shell_row_count(
                      state, cfg, mb, hybrid_bridge, &bridge_zero_is_inner);
    } else {
      const auto& north =
          mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
      // Shell-row absorption is bounded by the pure-gas prefix: the material
      // interface must never be absorbed into the macro CV.
      max_depth = block.n_i_cells + north.n_i_cells +
                  absorbable_shell_row_count(state, cfg, mb);
    }
  }
  if (depth < 0) {
    std::fprintf(stderr,
                 "[central_pseudo_core] ring absorption REFUSED: "
                 "failing_cell=%d is not absorbable (no unit mapping) "
                 "step=%d\n",
                 failing_cell,
                 state.step);
    return false;
  }
  // Member cells are geometry-exempt; a member failure is a different bug
  // and must not be masked by absorption.
  if (depth < pc.member_ring_count) {
    std::fprintf(stderr,
                 "[central_pseudo_core] ring absorption REFUSED: "
                 "failing_cell=%d depth=%d is already a member "
                 "(members=%d) step=%d\n",
                 failing_cell,
                 depth,
                 pc.member_ring_count,
                 state.step);
    return false;
  }
  const int target = depth + 1;
  if (target > max_depth || target > ring_absorption_max_rings(cfg)) {
    // Emergency mixed-material absorption (terminal-phase verdict direction (a)):
    // the gas prefix is exhausted but the failure sits AT or BEYOND the
    // macro boundary. Absorb the immediately-NEXT unit as a MIXED row —
    // one row per request, so repeated failures WALK the boundary outward
    // toward the failing cell (Exp2-v2 empirical: the dying cell sat 3
    // rows beyond the members, so a failing-depth-equality condition never
    // fired through 299 refusals). Gated on the static admissibility of
    // the would-be new outer boundary on today's mesh; the structural
    // backstop and the max_rings cap still bind.
    const int emergency_target = pc.member_ring_count + 1;
    if (mixed_absorption_enabled(cfg) && view.has_cap &&
        depth >= pc.member_ring_count &&
        emergency_target <= ring_absorption_max_rings(cfg) &&
        pc.min_member_ring_count < emergency_target) {
      const mesh::BlockInfo* north_fan = nullptr;
      int structural_max = view.n_cap;
      if (view.hybrid) {
        structural_max = view.n_cap + hybrid_bridge->n_i_cells;
      } else {
        north_fan = &mesh::mesh_topo_trifan_fan_block(
            mb, mesh::BlockRole::NORTH_FAN);
        const auto& shell_blk =
            mesh::mesh_topo_multiblock_polar_shell_block(mb);
        // Matches the membership selector's topology backstop: the LAST shell
        // row is never absorbable (at least one active shell row remains).
        structural_max = block.n_i_cells + north_fan->n_i_cells +
                         (shell_blk.n_i_cells - 1);
      }
      if (emergency_target <= structural_max) {
        double v_new_loop = 0.0;
        int new_loop_simple = 0;
        const bool new_loop_ok = hypothetical_boundary_loop_admissible(
            state, view, emergency_target, &v_new_loop, &new_loop_simple);
        const int q_row =
            view.hybrid
                ? -1
                : emergency_target - 1 - block.n_i_cells -
                      north_fan->n_i_cells;
        double row_mass = 0.0;
        double row_weighted_y = 0.0;
        if (!view.hybrid) {
          shell_row_material_stats(state, mb, q_row, &row_mass,
                                   &row_weighted_y);
        }
        if (new_loop_ok) {
          pc.min_member_ring_count = emergency_target;
          // Only rows past the gas prefix are MIXED; an emergency walk
          // step that is still within the prefix is a normal-legal row
          // absorbed under the emergency banner.
          const bool is_mixed_row = emergency_target > max_depth;
          if (is_mixed_row) {
            pc.mixed_absorbed_row_count += 1;
          }
          std::fprintf(stderr,
                       "[central_pseudo_core] EMERGENCY %s absorption "
                       "granted%s: failing_cell=%d "
                       "failing_depth=%d emergency_target=%d "
                       "(gas max_depth=%d) shell_row=%d "
                       "row_mass=%.6e row_weighted_Y=%.6f "
                       "new_loop_V=%.6e new_loop_simple=%d "
                       "mixed_rows=%d step=%d\n",
                       is_mixed_row ? "mixed" : "walk",
                       is_mixed_row ? " (mixed-core terminal phase)" : "",
                       failing_cell,
                       depth,
                       emergency_target,
                       max_depth,
                       q_row,
                       row_mass,
                       row_weighted_y,
                       v_new_loop,
                       new_loop_simple,
                       pc.mixed_absorbed_row_count,
                       state.step);
          return true;
        }
        std::fprintf(stderr,
                     "[central_pseudo_core] EMERGENCY mixed absorption "
                     "REFUSED by new-loop admissibility: failing_cell=%d "
                     "failing_depth=%d emergency_target=%d "
                     "new_loop_V=%.6e new_loop_simple=%d "
                     "step=%d\n",
                     failing_cell,
                     depth,
                     emergency_target,
                     v_new_loop,
                     new_loop_simple,
                     state.step);
        // The macro-band rezone taper keeps the band Lagrangian, which
        // lets the band's angular sawtooth grow untreated — and the
        // would-be new boundary loop is then exactly the tangled row that
        // blocks the walk (observed: taper-clean roll died at 2.62 ns on
        // repeated new-loop refusals). Lift the taper for a window so the
        // rezone can repair the band, then the retried walk can proceed.
        {
          static const int lift_steps = [] {
            const char* raw =
                std::getenv("TENRYU_I1B_MACRO_BAND_TAPER_LIFT_STEPS");
            const int v = raw != nullptr ? std::atoi(raw) : 0;
            return v > 0 ? v : 200;
          }();
          const int until = state.step + lift_steps;
          if (until > pc.taper_lift_until_step) {
            pc.taper_lift_until_step = until;
            std::fprintf(stderr,
                         "[central_pseudo_core] macro-band taper LIFT armed "
                         "until step=%d (new-loop refusal at step=%d)\n",
                         until,
                         state.step);
          }
        }
      }
    }
    std::fprintf(stderr,
                 "[central_pseudo_core] ring absorption REFUSED: "
                 "failing_cell=%d target=%d exceeds max_depth=%d "
                 "(max_rings=%d) step=%d\n",
                 failing_cell,
                 target,
                 max_depth,
                 ring_absorption_max_rings(cfg),
                 state.step);
    return false;
  }
  // A request at this depth is already pending (the retry has not executed
  // it yet): a repeat failure means absorption did not cure it — refuse so
  // the caller falls through to the hard assert instead of looping.
  if (pc.min_member_ring_count >= target) {
    std::fprintf(stderr,
                 "[central_pseudo_core] ring absorption REFUSED: "
                 "failing_cell=%d target=%d already pending "
                 "(min_depth=%d, members=%d) step=%d\n",
                 failing_cell,
                 target,
                 pc.min_member_ring_count,
                 pc.member_ring_count,
                 state.step);
    return false;
  }
  pc.min_member_ring_count = target;
  std::fprintf(stderr,
               "[central_pseudo_core] ring absorption requested: "
               "failing_cell=%d depth=%d -> min_depth=%d (members now %d) "
               "step=%d\n",
               failing_cell,
               depth,
               target,
               pc.member_ring_count,
               state.step);
  return true;
}

bool request_macro_boundary_absorption(core::State& state,
                                       const core::Config& cfg) {
  return request_ring_absorption(state,
                                 cfg,
                                 mesh::kCentralMacroCoreSentinelCell);
}

void detect_and_repair_stale_core_reference(core::State& state) {
  detect_and_repair_stale_core_reference_impl(state);
}

void maybe_absorb_first_active_ring(core::State& state,
                                    const core::Config& cfg) {
  if (!ring_absorption_enabled(cfg) || !configured(cfg)) {
    return;
  }
  auto& pc = state.central_pseudo_core;
  if (!pc.built || !pc.valid || !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  // Failure-triggered pending request (armed by request_ring_absorption at
  // the ALE center-patch admissibility site, executed here on the restored
  // pre-step state after the driver full-step retry).
  if (pc.min_member_ring_count > pc.member_ring_count) {
    execute_ring_absorption(
        state, cfg, pc.min_member_ring_count, "failure_retry");
    return;
  }
  static const double env_tau = [] {
    const char* raw = std::getenv("TENRYU_I1B_RING_ABSORB_TAU");
    const double v = raw != nullptr ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0 && v < 1.0) ? v : -1.0;
  }();
  const double tau =
      env_tau > 0.0
          ? env_tau
          : cfg.numerics.ale.central_pseudo_core_ring_absorption_tau;
  const auto& mb = *state.mesh.topo.multiblock;
  const CapTopoView view = make_cap_topo_view(cfg, mb);
  const int block_id = central_core_block_id(mb);
  const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
  // Watch the first W ACTIVE absorption units starting at depth D = member
  // depth: a core ring for D < n_cap, then a hybrid bridge row or a non-hybrid
  // fan layer, and finally a non-hybrid polar-shell row. W defaults to 1 (the
  // historical single-ring watch); a wider window is the endgame verdict's
  // PRE-trigger — a row beyond the first active unit can collapse to a
  // committed-degenerate volume within one accepted step (observed:
  // cell=1023 vol=-0.0 hard assert at t=2.78 ns while the single-ring
  // watch looked only at the first unit).
  const int watch_rows = absorb_watch_rows(cfg);
  int max_depth = block.n_i_cells - 1;
  int n_fan_layers = 0;
  const mesh::BlockInfo* hybrid_bridge = nullptr;
  bool bridge_zero_is_inner = true;
  if (view.hybrid) {
    hybrid_bridge = &hybrid_bridge_block(mb, view);
    std::vector<double> node_r = copy_field(state.x_r);
    std::vector<double> node_z = copy_field(state.x_z);
    bridge_zero_is_inner = hybrid_bridge_row_zero_is_inner(
        mb, *hybrid_bridge, node_r, node_z);
    const int absorbed_bridge_rows =
        std::max(0, pc.member_ring_count - view.n_cap);
    const int absorbable_bridge_rows =
        absorbed_bridge_rows +
        absorbable_shell_row_count(
            state, cfg, mb, hybrid_bridge, &bridge_zero_is_inner);
    max_depth = view.n_cap + absorbable_bridge_rows - 1;
  } else if (view.has_cap) {
    const auto& north =
        mesh::mesh_topo_trifan_fan_block(mb, mesh::BlockRole::NORTH_FAN);
    n_fan_layers = north.n_i_cells;
    // Absorbing a watched unit must keep within the pure-gas shell prefix.
    max_depth = block.n_i_cells + n_fan_layers +
                absorbable_shell_row_count(state, cfg, mb) - 1;
  }
  std::vector<double> vol = copy_field(state.vol);
  // Per-cell birth volumes for the sliver-ness discriminator. The row's
  // raw min/mean volume is structurally tiny in RZ (pole cells carry
  // vol ~ 2*pi*rbar -> 0 within the SAME row), so the lift must compare
  // each cell's vol/birth RATIO: uniform compression shrinks all ratios
  // together, the angular sawtooth crashes individual ones. TU-static is
  // acceptable for this experimental env-gated control (no restart
  // contract).
  static std::map<int, double> watch_birth_vol_by_cell;
  const auto unit_vol_stats = [&](const int depth, double* min_ratio_out,
                                  double* mean_ratio_out) {
    double min_vol = std::numeric_limits<double>::infinity();
    double min_ratio = std::numeric_limits<double>::infinity();
    long double sum = 0.0L;
    int count = 0;
    const auto accumulate_unit = [&](const mesh::BlockInfo& unit_block,
                                     const int layer) {
      for (int j = 0; j < unit_block.n_j_cells; ++j) {
        const int c = block_cell_id(unit_block, layer, j);
        if (c >= 0 && c < static_cast<int>(vol.size())) {
          const double v = vol[static_cast<std::size_t>(c)];
          min_vol = std::min(min_vol, v);
          if (!std::isfinite(v) || v <= 0.0) {
            min_ratio = std::min(min_ratio, 0.0);
            continue;
          }
          auto it = watch_birth_vol_by_cell.find(c);
          if (it == watch_birth_vol_by_cell.end()) {
            it = watch_birth_vol_by_cell.emplace(c, v).first;
          }
          const double ratio = it->second > 0.0 ? v / it->second : 1.0;
          min_ratio = std::min(min_ratio, ratio);
          sum += ratio;
          ++count;
        }
      }
    };
    if (depth < block.n_i_cells) {
      accumulate_unit(block, depth);
    } else if (view.hybrid) {
      const int radial_row = depth - block.n_i_cells;
      accumulate_unit(
          *hybrid_bridge,
          hybrid_bridge_storage_row(
              *hybrid_bridge, radial_row, bridge_zero_is_inner));
    } else if (depth < block.n_i_cells + n_fan_layers) {
      const int fan_layer = depth - block.n_i_cells;
      for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                        mesh::BlockRole::EAST_FAN,
                                        mesh::BlockRole::SOUTH_FAN}) {
        accumulate_unit(mesh::mesh_topo_trifan_fan_block(mb, fan),
                        fan_layer);
      }
    } else {
      accumulate_unit(
          mesh::mesh_topo_multiblock_polar_shell_block(mb),
          depth - block.n_i_cells - n_fan_layers);
    }
    if (min_ratio_out != nullptr) {
      *min_ratio_out = min_ratio;
    }
    if (mean_ratio_out != nullptr) {
      *mean_ratio_out = count > 0 ? static_cast<double>(sum / count) : 0.0;
    }
    return min_vol;
  };
  // Proactive SPHERICAL absorption schedule (verdict #6 A3 + verdict #5
  // Q4): failure-triggered absorption hands the spherical 1D core a folded
  // boundary (measured eps_R = 0.32-0.72 at every dyncore12 handoff vs the
  // 0.03 acceptance gate). Instead absorb the first active unit while it is
  // still spherical: when its mean node radius has converged below ALPHA
  // times its first-active (birth) value. Env-gated, default off
  // (TENRYU_I1B_SPHERICAL_ABSORB_ALPHA in (0,1)).
  const double spherical_alpha = spherical_absorb_alpha(cfg);
  // Shock-arrival criterion (dyncore13: the radius criterion NEVER fired —
  // the inner gas is quiet until the shock arrives and folds it within a
  // few steps, so a geometric convergence fraction loses the race to the
  // failure ladder). Absorb the first active unit when its max cell
  // pressure jumps above PJUMP times its birth value: that IS the
  // convergence interface passing the unit, and it is still spherical
  // there.
  const double spherical_pjump = spherical_absorb_pjump(cfg);
  static const int spherical_probe_every = [] {
    const char* raw = std::getenv("TENRYU_I1B_SPHERICAL_ABSORB_PROBE");
    return raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
  }();
  // GASFRONT mode (dyncore14 lesson): the fold forms at the SHOCK FRONT in
  // the mid-gas (depth ~15) while the first active unit is still pre-shock
  // quiet (p/birth = 1.023 at step 800, failure cascade 7->16 at step 960)
  // — watching the innermost unit can never win. Instead watch the
  // OUTERMOST gas unit (adjacent to the shell): its pressure jump IS the
  // shock entering the cushion, and at that moment the whole gas prefix is
  // still spherical. Fire absorbs the ENTIRE prefix (per-ring shells keep
  // the stratification in the 1D core) = the clean 2D-shell / 1D-cushion
  // split with a material-interface, pre-shock handoff.
  const bool spherical_gasfront = spherical_absorb_gasfront(cfg);
  if (spherical_alpha > 0.0 || spherical_pjump > 0.0) {
    const int depth =
        spherical_gasfront ? max_depth : pc.member_ring_count;
    if (depth > 0 && depth <= max_depth &&
        depth >= pc.member_ring_count &&
        depth < ring_absorption_max_rings(cfg)) {
      std::vector<double> nr_h = copy_field(state.x_r);
      std::vector<double> nz_h = copy_field(state.x_z);
      std::vector<double> rho_h = copy_field(state.rho);
      std::vector<double> ee_h = copy_field(state.ee);
      std::vector<double> ei_h = state.ei.empty() ? std::vector<double>{}
                                                  : copy_field(state.ei);
      const int nn_total = static_cast<int>(nr_h.size());
      const int nc_total = static_cast<int>(rho_h.size());
      double acc = 0.0;
      int cnt = 0;
      double p_max = 0.0;
      const double gamma_sph =
          !cfg.materials.materials.empty()
              ? cfg.materials.materials.front().ideal_gas_gamma
              : 5.0 / 3.0;
      const auto accumulate_unit_sph = [&](const mesh::BlockInfo& ub,
                                           const int layer) {
        for (int j = 0; j < ub.n_j_cells; ++j) {
          const int c = block_cell_id(ub, layer, j);
          if (c < 0 || c >= nc_total) {
            continue;
          }
          const std::size_t ci = static_cast<std::size_t>(c);
          const double e_c =
              ee_h[ci] + (ei_h.empty() ? 0.0 : ei_h[ci]);
          p_max = std::max(p_max, (gamma_sph - 1.0) * rho_h[ci] * e_c);
          const auto nodes = multiblock_cell_nodes(mb, c);
          for (const int nn : nodes) {
            if (nn >= 0 && nn < nn_total) {
              acc += std::hypot(nr_h[static_cast<std::size_t>(nn)],
                                nz_h[static_cast<std::size_t>(nn)]);
              ++cnt;
            }
          }
        }
      };
      if (depth < block.n_i_cells) {
        accumulate_unit_sph(block, depth);
      } else if (view.hybrid) {
        const int radial_row = depth - block.n_i_cells;
        accumulate_unit_sph(
            *hybrid_bridge,
            hybrid_bridge_storage_row(
                *hybrid_bridge, radial_row, bridge_zero_is_inner));
      } else if (depth < block.n_i_cells + n_fan_layers) {
        for (const mesh::BlockRole fan : {mesh::BlockRole::NORTH_FAN,
                                          mesh::BlockRole::EAST_FAN,
                                          mesh::BlockRole::SOUTH_FAN}) {
          accumulate_unit_sph(mesh::mesh_topo_trifan_fan_block(mb, fan),
                              depth - block.n_i_cells);
        }
      } else {
        accumulate_unit_sph(
            mesh::mesh_topo_multiblock_polar_shell_block(mb),
            depth - block.n_i_cells - n_fan_layers);
      }
      if (cnt > 0) {
        const double mean_R = acc / cnt;
        // Birth = the unit's state when it first became the first active
        // unit (TU-static, single-state experimental scope).
        static std::map<int, std::pair<double, double>>
            spherical_birth_by_depth;  // depth -> (mean_R, p_max)
        const auto itb = spherical_birth_by_depth.find(depth);
        if (itb == spherical_birth_by_depth.end()) {
          spherical_birth_by_depth.emplace(depth,
                                           std::make_pair(mean_R, p_max));
        } else {
          const double birth_R = itb->second.first;
          const double birth_p = std::max(itb->second.second, 1.0e-300);
          const bool radius_fire =
              spherical_alpha > 0.0 && mean_R < spherical_alpha * birth_R;
          const bool pjump_fire =
              spherical_pjump > 0.0 && p_max > spherical_pjump * birth_p;
          if (spherical_probe_every > 0 &&
              (state.step % spherical_probe_every) == 0) {
            std::fprintf(stderr,
                         "[sph_absorb_probe] step=%d depth=%d "
                         "R/birth=%.4f p/birth=%.4e\n",
                         state.step, depth, mean_R / birth_R,
                         p_max / birth_p);
          }
          if (radius_fire || pjump_fire) {
            std::fprintf(stderr,
                         "[central_pseudo_core] spherical absorption "
                         "schedule (%s%s): depth=%d R/birth=%.4f "
                         "p/birth=%.3e step=%d\n",
                         pjump_fire ? "pjump" : "radius",
                         spherical_gasfront ? ",gasfront" : "", depth,
                         mean_R / birth_R, p_max / birth_p, state.step);
            execute_ring_absorption(state, cfg, depth + 1,
                                    "spherical_schedule");
            return;
          }
        }
      }
    }
  }
  for (int depth = pc.member_ring_count;
       depth < pc.member_ring_count + watch_rows; ++depth) {
    if (depth <= 0 || depth > max_depth ||
        depth >= ring_absorption_max_rings(cfg)) {
      break;
    }
    double min_ratio = 0.0;
    double mean_ratio = 0.0;
    const double min_vol = unit_vol_stats(depth, &min_ratio, &mean_ratio);
    if (!std::isfinite(min_vol) || min_vol <= 0.0) {
      continue;
    }
    if (pc.absorb_watch_birth_by_depth.size() <=
        static_cast<std::size_t>(depth)) {
      pc.absorb_watch_birth_by_depth.resize(
          static_cast<std::size_t>(depth) + 1U, 0.0);
    }
    double& birth =
        pc.absorb_watch_birth_by_depth[static_cast<std::size_t>(depth)];
    if (!(birth > 0.0)) {
      // Birth volume = the unit's state when it first entered the watch
      // window; the trigger compares against this, not t=0.
      birth = min_vol;
      continue;
    }
    // Proactive macro-band taper lift: with the rezone taper holding the
    // band Lagrangian, a degrading watched row can throttle dt into an
    // asymptotic crawl WITHOUT ever reaching the dt floor or an
    // admissibility refusal (observed: 21k steps advancing 0.003 ns at
    // dt~1e-16 with zero rescue events). Discriminator = the row's
    // min/mean of PER-CELL vol/birth ratios: uniform compression shrinks
    // every ratio together, the angular sawtooth crashes individual ones.
    // (A raw min/mean volume test is structurally tiny in RZ — pole cells
    // carry vol ~ 2 pi rbar -> 0 within the same row — and armed the lift
    // permanently from step 0; a min/birth test conflates compression.)
    {
      static const double lift_frac = [] {
        const char* raw =
            std::getenv("TENRYU_I1B_MACRO_BAND_TAPER_LIFT_FRAC");
        const double v = raw != nullptr ? std::atof(raw) : 0.0;
        return (std::isfinite(v) && v > 0.0 && v < 1.0) ? v : 0.35;
      }();
      static const int lift_steps = [] {
        const char* raw =
            std::getenv("TENRYU_I1B_MACRO_BAND_TAPER_LIFT_STEPS");
        const int v = raw != nullptr ? std::atoi(raw) : 0;
        return v > 0 ? v : 200;
      }();
      if (mean_ratio > 0.0 && min_ratio < lift_frac * mean_ratio) {
        const int until = state.step + lift_steps;
        if (until > pc.taper_lift_until_step) {
          pc.taper_lift_until_step = until;
          std::fprintf(stderr,
                       "[central_pseudo_core] macro-band taper LIFT armed "
                       "until step=%d (watch row=%d min_ratio=%.3e "
                       "mean_ratio=%.3e frac=%.2f)\n",
                       until,
                       depth,
                       min_ratio,
                       mean_ratio,
                       lift_frac);
        }
      }
    }
    if (min_vol >= tau * birth) {
      continue;
    }
    std::fprintf(stderr,
                 "[central_pseudo_core] ring absorption trigger: ring=%d "
                 "min_vol=%.6e birth=%.6e tau=%.3f watch_rows=%d step=%d\n",
                 depth,
                 min_vol,
                 birth,
                 tau,
                 watch_rows,
                 state.step);
    execute_ring_absorption(state, cfg, depth + 1, "volume");
    return;
  }
}

void aggregate_state(core::State& state,
                     const core::Config& cfg,
                     const char* phase,
                     const bool emit_audit) {
  if (!configured(cfg)) {
    deactivate_if_needed(state);
    return;
  }
  ensure_built(state, cfg);
  auto& pc = state.central_pseudo_core;
  if (!active(state)) {
    return;
  }
  if (phase_is(phase, "post_ale_remap")) {
    rebuild_virtual_member_geometry(state, cfg, phase);
  }
  // Per-step reference tracking (env TENRYU_I1B_CORE_REF_TRACK): between
  // absorptions the core-adjacent resolved cells keep converging while their
  // reference stays at the last absorb-time sync — the divergence eventually
  // re-breaks the CSR swept closure (measured: one +3.6e-7 creation event at
  // a stale-reference halo cell). Track the reference to the current mesh
  // around the core EVERY step: the halo becomes effectively Lagrangian
  // (zero remap motion demanded) and quality there is owned by the
  // interior-patch remap.
  {
    static const bool ref_track = [] {
      const char* raw = std::getenv("TENRYU_I1B_CORE_REF_TRACK");
      return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    if (ref_track && phase_is(phase, "hydro_step_start")) {
      sync_reference_to_current_around_core(state);
    }
    static const bool ref_repair = [] {
      const char* raw = std::getenv("TENRYU_I1B_CORE_REF_REPAIR");
      return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
    }();
    if (ref_repair && phase_is(phase, "hydro_step_start")) {
      detect_and_repair_stale_core_reference_impl(state);
    }
  }
  TENRYU_ASSERT(!cfg.numerics.materials.per_material_conservation_enabled,
                "central pseudo-core Exp1 does not aggregate per-material "
                "arrays");

  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> vol = copy_field(state.vol);
  std::vector<double> rho = copy_field(state.rho);
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);

  const MemberSums sums = sum_member_state(pc, mass, vol, ee, ei, tracer);
  TENRYU_ASSERT(sums.M > 0.0L,
                "central pseudo-core aggregate mass must be positive");
  const double V_boundary = boundary_loop_volume(pc, node_r, node_z);
  // Stratified 1D sub-model: once per 2D step (at hydro_step_start, with
  // rollback on driver full-step retries) track the macro boundary loop
  // volume with the kinematic piston and subcycle the interior hydro.
  if (core1d::enabled() && core1d::built(&pc) &&
      phase_is(phase, "hydro_step_start") &&
      !cfg.materials.materials.empty() &&
      cfg.materials.materials.front().eos_model == "ideal_gas") {
    const double gamma1d = cfg.materials.materials.front().ideal_gas_gamma;
    if (core1d::begin_step(&pc, state.step)) {
      const double dt2d = state.dt > 0.0 ? state.dt : 1.0e-13;
      // A6 pairing audit: compare the sub-model's internal piston integral,
      // Pi times the shared volume change, and the 2D conjugate-impulse
      // booking for the same step.
      static const int pair_every = [] {
        const char* raw = std::getenv("TENRYU_I1B_CORE_1D_LEDGER_EVERY");
        return raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
      }();
      const double pw0 = core1d::piston_work_total(&pc);
      const double v_sub0 = core1d::current_volume(&pc);
      const double pi_face = core1d::outer_face_pressure(&pc, gamma1d);
      // Dynamic (force-driven) outer boundary by default: external
      // pressure = mass-weighted mean of the ACTIVE cells touching the
      // boundary loop; the kinematic piston (measured slosh-pump after
      // massive appends) remains via TENRYU_I1B_CORE_1D_PISTON=kinematic.
      // Massless-face kinematic piston is the default (the pure
      // force-driven face un-tethers and freely expands — measured).
      static const bool piston_dynamic = [] {
        const char* raw = std::getenv("TENRYU_I1B_CORE_1D_PISTON");
        return raw != nullptr && std::strcmp(raw, "dynamic") == 0;
      }();
      // A2 overlap instrument (verdict #6): the 2D boundary-adjacent
      // mass-weighted pressure vs the sub-model outer-face pressure —
      // logged each pairing sample below; also feeds the dynamic piston.
      double pm = 0.0, mm = 0.0;
      {
        if (state.mesh.topo.multiblock.has_value()) {
          const auto& mbp = *state.mesh.topo.multiblock;
          const int n_cells_p = state.mesh.topo.n_cells;
          for (int cix = 0; cix < n_cells_p; ++cix) {
            const std::size_t cxi = static_cast<std::size_t>(cix);
            if (cxi < pc.member_mask.size() && pc.member_mask[cxi] != 0U) {
              continue;
            }
            const auto nds = multiblock_cell_nodes(mbp, cix);
            bool touches = false;
            for (const int nn : nds) {
              if (nn >= 0 &&
                  static_cast<std::size_t>(nn) <
                      pc.boundary_node_mask.size() &&
                  pc.boundary_node_mask[static_cast<std::size_t>(nn)] !=
                      0U) {
                touches = true;
                break;
              }
            }
            if (!touches) {
              continue;
            }
            const double m_c2 = mass[cxi];
            const double rho_c2 =
                vol[cxi] > 0.0 ? m_c2 / vol[cxi] : 0.0;
            const double e_c2 =
                ee[cxi] + (ei.empty() ? 0.0 : ei[cxi]);
            pm += m_c2 * (gamma1d - 1.0) * rho_c2 * e_c2;
            mm += m_c2;
          }
        }
      }
      if (!piston_dynamic) {
        core1d::advance_to_volume(&pc, V_boundary, dt2d, gamma1d);
      } else {
        core1d::advance_dynamic(&pc, mm > 0.0 ? pm / mm : 0.0, dt2d,
                                gamma1d);
      }
      if (pair_every > 0 && (state.step % pair_every) == 0) {
        const double w_1d = core1d::piston_work_total(&pc) - pw0;
        const double w_pair = pi_face * (v_sub0 - V_boundary);
        const double p2d = mm > 0.0 ? pm / mm : 0.0;
        std::fprintf(stderr,
                     "[core1d_pair] step=%d t=%.6e W_1d=%.6e "
                     "PidV=%.6e W_2d_booked=%.6e dV=%.6e\n",
                     state.step, state.t, w_1d, w_pair,
                     pc.boundary_work_dU, V_boundary - v_sub0);
        // A2 overlap (verdict #6): interface pressure agreement between
        // the sub-model face and the 2D boundary-adjacent average. P_stat
        // separates the static face pressure from the AV component so the
        // comparison is like-with-like (the 2D average is static-only).
        const double p_stat =
            core1d::outer_face_pressure_static(&pc, gamma1d);
        std::fprintf(stderr,
                     "[core1d_a2] step=%d t=%.6e P_face=%.6e P_stat=%.6e "
                     "P_2d=%.6e rel=%.4e rel_stat=%.4e\n",
                     state.step, state.t, pi_face, p_stat, p2d,
                     (pi_face - p2d) / std::max(p2d, 1.0e-300),
                     (p_stat - p2d) / std::max(p2d, 1.0e-300));
      }
      core1d::emit_ledger(&pc, state.step, state.t, gamma1d);
      const core1d::GasView g1d = core1d::gas_view(&pc, gamma1d);
      pc.core1d_V_gas_c = (g1d.valid && g1d.V_gas > 0.0)
                              ? g1d.V_gas
                              : std::numeric_limits<double>::quiet_NaN();
      emit_boundary_asphericity(state, nullptr);
    }
  }
  const bool sync_internal =
      sync_internal_from_members_phase(phase) || !(pc.M_c > 0.0 && pc.V_c > 0.0);
  update_u_gas_energy_fraction(pc, sums);
  pc.M_c = static_cast<double>(sums.M);
  pc.M_Y_c = static_cast<double>(sums.MY);
  pc.V_c = V_boundary;
  if (!std::isfinite(pc.V_c_initial) &&
      pc.M_c > 0.0 && pc.V_c > 0.0) {
    pc.V_c_initial = pc.V_c;
    pc.rho_c_initial = pc.M_c / pc.V_c;
  }
  if (sync_internal) {
    long double Ue_c = sums.Ue;
    long double Ui_c = !ei.empty() ? sums.Ui : 0.0L;
    if (pc.activation_ke_bank_pending) {
      const double chi_e = electron_energy_fraction(
          static_cast<double>(Ue_c), static_cast<double>(Ui_c), !ei.empty());
      pc.activation_ke_chi_e = chi_e;
      Ue_c += static_cast<long double>(chi_e) *
              static_cast<long double>(pc.activation_ke_dU);
      if (!ei.empty()) {
        Ui_c += static_cast<long double>(1.0 - chi_e) *
                static_cast<long double>(pc.activation_ke_dU);
      }
      pc.activation_ke_bank_pending = false;
    }
    pc.Ue_c = static_cast<double>(Ue_c);
    pc.Ui_c = !ei.empty() ? static_cast<double>(Ui_c) : 0.0;
    pc.boundary_work_pending = false;
  }
  write_member_mirrors(state, mass, vol, rho, ee, ei, tracer);
  if (conservation_audit::enabled()) {
    std::string audit_stage = std::string("central_aggregate:") +
                              (phase != nullptr ? phase : "unspecified");
    conservation_audit::emit_stage(state, audit_stage.c_str());
  }
  if (phase_is(phase, "hydro_step_start") || phase_is(phase, "post_predictor")) {
    emit_phase_ledger(state, cfg, "E_after_aggregate", phase);
  } else if (phase_is(phase, "post_energy")) {
    emit_phase_ledger(state, cfg, "E_post_energy", phase);
  } else if (phase_is(phase, "post_ale_remap")) {
    emit_phase_ledger(state, cfg, "E_after_remap", phase);
  }

  const ScalarTotals totals = compute_global_totals(state);
  if (!std::isfinite(pc.E_total_ref)) {
    pc.E_total_ref = totals.E_total;
    pc.mass_ref = totals.mass;
    pc.tracer_mass_ref = totals.tracer_mass;
    pc.Ue_ref = totals.Ue;
    pc.Ui_ref = totals.Ui;
  }
  if (emit_audit) {
    std::fprintf(stderr,
                 "[central_pseudo_core_audit] step=%d t=%.17e phase=%s "
                 "members=%zu rep=%d Vc=%.17e Mc=%.17e "
                 "mass=%.17e dmass_rel=%.17e tracer_mass=%.17e "
                 "dtracer_rel=%.17e Ue=%.17e dUe_rel=%.17e "
                 "Ui=%.17e dUi_rel=%.17e E_total=%.17e dE_rel=%.17e\n",
                 state.step,
                 state.t,
                 phase != nullptr ? phase : "unspecified",
                 pc.member_cells.size(),
                 pc.representative_cell,
                 pc.V_c,
                 pc.M_c,
                 totals.mass,
                 relative_delta(totals.mass, pc.mass_ref),
                 totals.tracer_mass,
                 relative_delta(totals.tracer_mass, pc.tracer_mass_ref),
                 totals.Ue,
                 relative_delta(totals.Ue, pc.Ue_ref),
                 totals.Ui,
                 relative_delta(totals.Ui, pc.Ui_ref),
                 totals.E_total,
                 relative_delta(totals.E_total, pc.E_total_ref));
    if (work_probe_enabled() && pc.activation_ke_valid) {
      std::fprintf(stderr,
                   "[central_pseudo_core_activation_probe] step=%d t=%.17e "
                   "phase=%s K_old=%.17e K_new=%.17e dU_activation=%.17e "
                   "E_before=%.17e E_after_zero=%.17e passive_nodes=%zu "
                   "chi_e=%.17e\n",
                   state.step,
                   state.t,
                   phase != nullptr ? phase : "unspecified",
                   pc.activation_ke_K_old,
                   pc.activation_ke_K_new,
                   pc.activation_ke_dU,
                   pc.activation_ke_E_before,
                   pc.activation_ke_E_after_zero,
                   pc.activation_passive_nodes.size(),
                   pc.activation_ke_chi_e);
    }
    if (work_probe_enabled() && pc.boundary_work_valid) {
      std::fprintf(stderr,
                   "[central_pseudo_core_work_probe] step=%d t=%.17e phase=%s "
                   "W_old=%.17e W_new=%.17e W_mid=%.17e dU_C=%.17e "
                   "R_C=%.17e chi_e=%.17e\n",
                   state.step,
                   state.t,
                   phase != nullptr ? phase : "unspecified",
                   pc.boundary_work_W_old,
                   pc.boundary_work_W_new,
                   pc.boundary_work_W_mid,
                   pc.boundary_work_dU,
                   pc.boundary_work_residual,
                   pc.boundary_work_chi_e);
    }
  }
}

void emit_phase_ledger(core::State& state,
                       const core::Config& cfg,
                       const char* phase,
                       const char* source_phase) {
  if (!phase_ledger_enabled(state) || !configured(cfg) || !active(state)) {
    return;
  }

  struct LedgerState {
    int step = -1;
    bool has_previous = false;
    double previous_E = 0.0;
    std::string previous_phase;
    std::string previous_source_phase;
  };
  static LedgerState ledger;

  if (ledger.step != state.step) {
    ledger.step = state.step;
    ledger.has_previous = false;
    ledger.previous_E = 0.0;
    ledger.previous_phase.clear();
    ledger.previous_source_phase.clear();
  }

  const double E_total = diagnostics::compute_energy_budget_2d(state).E_total;
  const double dE =
      ledger.has_previous ? (E_total - ledger.previous_E) : 0.0;
  const double dE_rel =
      ledger.has_previous
          ? dE / std::max(std::abs(ledger.previous_E), 1.0e-300)
          : 0.0;
  const char* phase_name = phase != nullptr ? phase : "unspecified";
  const char* source_name =
      source_phase != nullptr ? source_phase : "unspecified";
  const char* previous_phase =
      ledger.has_previous ? ledger.previous_phase.c_str() : "none";
  const char* previous_source =
      ledger.has_previous ? ledger.previous_source_phase.c_str() : "none";

  std::fprintf(stderr,
               "[i1b_pseudocell_phase_ledger] step=%d t=%.17e phase=%s "
               "source_phase=%s E_total=%.17e dE_prev=%.17e "
               "dE_prev_rel=%.17e prev_phase=%s prev_source_phase=%s\n",
               state.step,
               state.t,
               phase_name,
               source_name,
               E_total,
               dE,
               dE_rel,
               previous_phase,
               previous_source);

  ledger.has_previous = true;
  ledger.previous_E = E_total;
  ledger.previous_phase = phase_name;
  ledger.previous_source_phase = source_name;
}

const std::uint8_t* member_mask_device(core::State& state,
                                       const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.central_pseudo_core.d_member_mask.data() : nullptr;
}

const std::uint8_t* inactive_member_mask_device(core::State& state,
                                                const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.central_pseudo_core.d_inactive_member_mask.data()
                       : nullptr;
}

const std::uint8_t* active_cell_class_device(core::State& state,
                                             const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.central_pseudo_core.d_active_cell_class.data()
                       : nullptr;
}

const std::uint8_t* boundary_node_mask_device(core::State& state,
                                              const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.central_pseudo_core.d_boundary_node_mask.data()
                       : nullptr;
}

double acoustic_dt(core::State& state, const core::Config& cfg) {
  if (!configured(cfg)) {
    return std::numeric_limits<double>::infinity();
  }
  ensure_built(state, cfg);
  const auto& pc = state.central_pseudo_core;
  if (!active(state) || !(pc.V_c > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> cs;
  if (!state.cs.empty()) {
    state.cs.copy_to_host(cs);
  }
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  const int c0 = pc.representative_cell;
  const double gamma = cfg.materials.materials.empty()
                           ? 5.0 / 3.0
                           : cfg.materials.materials.front().ideal_gas_gamma;
  const double e_e = ee[static_cast<std::size_t>(c0)] > 0.0
                         ? ee[static_cast<std::size_t>(c0)]
                         : 0.0;
  const double e_i = (!ei.empty() && ei[static_cast<std::size_t>(c0)] > 0.0)
                         ? ei[static_cast<std::size_t>(c0)]
                         : 0.0;
  const double cs_val =
      !cs.empty() ? cs[static_cast<std::size_t>(c0)]
                  : std::sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
  const double c_eff = std::max(cs_val, 0.0);
  if (!(c_eff > 0.0) || !std::isfinite(c_eff)) {
    return std::numeric_limits<double>::infinity();
  }
  const double h = std::cbrt(std::max(pc.V_c / kFourPiOverThree, 0.0));
  if (!(h > 0.0) || !std::isfinite(h)) {
    return std::numeric_limits<double>::infinity();
  }
  const double denom = c_eff * (1.0 + cfg.numerics.hydro.av_linear);
  return cfg.numerics.dt.cfl_hydro * h / denom;
}

void zero_member_compatible_cell_buffers(core::State& state) {
  if (!active(state)) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 || state.corner_force_p_r.empty() ||
      state.corner_force_sub_r.empty() || state.work_p_per_cell.empty()) {
    return;
  }
  const int blocks = (n_cells + 255) / 256;
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  const bool has_corner_q = state.corner_force_q_r.size() == n_corners &&
                            state.corner_force_q_z.size() == n_corners;
  zero_member_compatible_cell_buffers_kernel<<<blocks, 256>>>(
      state.corner_force_p_r.data(),
      state.corner_force_p_z.data(),
      state.corner_force_sub_r.data(),
      state.corner_force_sub_z.data(),
      has_corner_q ? state.corner_force_q_r.data() : nullptr,
      has_corner_q ? state.corner_force_q_z.data() : nullptr,
      state.work_p_per_cell.data(),
      state.work_sub_per_cell.data(),
      state.work_av_per_cell.data(),
      state.central_pseudo_core.d_member_mask.data(),
      n_cells,
      state.corner_stride);
  cuda_check(cudaGetLastError(),
             "central pseudo-core zero member compatible cell buffers launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "central pseudo-core zero member compatible cell buffers failed");
  if (!state.corner_force_p_rz_r.empty()) {
    zero_member_aw_pressure_work_force_kernel<<<blocks, 256>>>(
        state.corner_force_p_rz_r.data(),
        state.corner_force_p_rz_z.data(),
        state.central_pseudo_core.d_member_mask.data(),
        n_cells);
    cuda_check(
        cudaGetLastError(),
        "central pseudo-core zero member AW pressure work force launch failed");
    cuda_check(
        cudaDeviceSynchronize(),
        "central pseudo-core zero member AW pressure work force failed");
  }
}

void zero_member_edge_av_forces(core::State& state) {
  if (!active(state) || state.edge_force_av_r.empty()) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "central pseudo-core edge zero requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
  const int n_boundary = static_cast<int>(mb.boundary_faces.size());
  if (n_internal > 0) {
    const int blocks = (n_internal + 255) / 256;
    zero_internal_member_edge_forces_kernel<<<blocks, 256>>>(
        state.edge_force_av_r.data(),
        state.edge_force_av_z.data(),
        thrust::raw_pointer_cast(mb.d_unique_face_cell_a.data()),
        thrust::raw_pointer_cast(mb.d_unique_face_cell_b.data()),
        state.central_pseudo_core.d_member_mask.data(),
        n_internal,
        0);
  }
  if (n_boundary > 0) {
    const int blocks = (n_boundary + 255) / 256;
    zero_boundary_member_edge_forces_kernel<<<blocks, 256>>>(
        state.edge_force_av_r.data(),
        state.edge_force_av_z.data(),
        thrust::raw_pointer_cast(mb.d_boundary_face_cell.data()),
        state.central_pseudo_core.d_member_mask.data(),
        n_boundary,
        n_internal);
  }
  cuda_check(cudaGetLastError(),
             "central pseudo-core zero member edge AV launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "central pseudo-core zero member edge AV failed");
}

void add_boundary_pressure_force(core::State& state,
                                 const core::Config& cfg,
                                 const core::CellField1D& cell_pressure,
                                 double* force_r,
                                 double* force_z,
                                 const double impulse_dt,
                                 const char* diag_tag) {
  static const bool node_diag =
      std::getenv("TENRYU_PSC_BFORCE_NODE_DIAG") != nullptr;
  static const bool node_diag_every =
      std::getenv("TENRYU_PSC_BFORCE_NODE_DIAG_EVERY") != nullptr;
  static long long g_bforce_calls = 0;
  ++g_bforce_calls;
  const bool sample =
      node_diag_every || g_bforce_calls <= 64 || g_bforce_calls % 200 == 1;
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return;
  }
  auto& pc = state.central_pseudo_core;
  const int n_boundary = static_cast<int>(pc.boundary_nodes_ordered.size());
  if (n_boundary <= 0) {
    return;
  }
  (void)cell_pressure;
  const double p_c = macro_core_pressure(pc, cfg);
  pc.d_boundary_nodes_ordered.reset(pc.boundary_nodes_ordered.size());
  pc.d_boundary_nodes_ordered.copy_from_host(pc.boundary_nodes_ordered);
  if (pc.d_boundary_impulse_r.size() != pc.boundary_nodes_ordered.size()) {
    pc.d_boundary_impulse_r.reset(pc.boundary_nodes_ordered.size());
    pc.d_boundary_impulse_z.reset(pc.boundary_nodes_ordered.size());
  }
  double Fpre_out = 0.0;
  double fpre_nodemax = 0.0;
  int k_max = 0;
  std::vector<double> pre_force_r;
  std::vector<double> pre_force_z;
  if (sample) {
    cuda_check(cudaDeviceSynchronize(),
               "central pseudo-core boundary pressure force pre-sample sync failed");
    const std::size_t n_nodes = state.x_r.size();
    const std::vector<double> node_r = copy_field(state.x_r);
    const std::vector<double> node_z = copy_field(state.x_z);
    std::vector<double> force_r_h(n_nodes);
    std::vector<double> force_z_h(n_nodes);
    cuda_check(cudaMemcpy(force_r_h.data(), force_r,
                          n_nodes * sizeof(double), cudaMemcpyDeviceToHost),
               "central pseudo-core boundary pressure force pre-sample copy r failed");
    cuda_check(cudaMemcpy(force_z_h.data(), force_z,
                          n_nodes * sizeof(double), cudaMemcpyDeviceToHost),
               "central pseudo-core boundary pressure force pre-sample copy z failed");
    if (node_diag) {
      pre_force_r = force_r_h;
      pre_force_z = force_z_h;
    }
    for (int k = 0; k < n_boundary; ++k) {
      const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(k)];
      const double xr_n = node_r[static_cast<std::size_t>(n)];
      const double xz_n = node_z[static_cast<std::size_t>(n)];
      const double s_n = std::hypot(xr_n, xz_n);
      const double fpre_out =
          (force_r_h[static_cast<std::size_t>(n)] * xr_n +
           force_z_h[static_cast<std::size_t>(n)] * xz_n) /
          std::max(s_n, 1.0e-300);
      Fpre_out += fpre_out;
      if (std::abs(fpre_out) > fpre_nodemax) {
        fpre_nodemax = std::abs(fpre_out);
        k_max = k;
      }
    }
  }
  const int blocks = (n_boundary + 255) / 256;
  boundary_pressure_force_kernel<<<blocks, 256>>>(
      force_r,
      force_z,
      pc.d_boundary_impulse_r.data(),
      pc.d_boundary_impulse_z.data(),
      pc.d_boundary_nodes_ordered.data(),
      n_boundary,
      state.x_r.data(),
      state.x_z.data(),
      p_c,
      impulse_dt,
      cfg.numerics.hydro.rz_momentum_scheme_id);
  cuda_check(cudaGetLastError(),
             "central pseudo-core boundary pressure force launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "central pseudo-core boundary pressure force failed");

  if (sample) {
    const std::size_t n_nodes = state.x_r.size();
    const std::vector<double> node_r = copy_field(state.x_r);
    const std::vector<double> node_z = copy_field(state.x_z);
    std::vector<double> force_r_h(n_nodes);
    std::vector<double> force_z_h(n_nodes);
    cuda_check(cudaMemcpy(force_r_h.data(), force_r,
                          n_nodes * sizeof(double), cudaMemcpyDeviceToHost),
               "central pseudo-core boundary pressure force post-sample copy r failed");
    cuda_check(cudaMemcpy(force_z_h.data(), force_z,
                          n_nodes * sizeof(double), cudaMemcpyDeviceToHost),
               "central pseudo-core boundary pressure force post-sample copy z failed");
    if (node_diag) {
      if (!pre_force_r.empty() && !pre_force_z.empty()) {
        for (int k = 0; k < n_boundary; ++k) {
          const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(k)];
          const double f_r = force_r_h[static_cast<std::size_t>(n)] -
                             pre_force_r[static_cast<std::size_t>(n)];
          const double f_z = force_z_h[static_cast<std::size_t>(n)] -
                             pre_force_z[static_cast<std::size_t>(n)];
          std::fprintf(
              stderr,
              "[psc_bforce_node] call=%lld k=%d n=%d R=%.17e Z=%.17e "
              "fR=%.17e fZ=%.17e\n",
              g_bforce_calls, k, n, node_r[static_cast<std::size_t>(n)],
              node_z[static_cast<std::size_t>(n)], f_r, f_z);
        }
      }
    }
    const std::vector<double> velocity_r = copy_field(state.v_r);
    const std::vector<double> velocity_z = copy_field(state.v_z);
    double r_mean = 0.0;
    double Fpost_out = 0.0;
    double ur_ring_mean = 0.0;
    for (int k = 0; k < n_boundary; ++k) {
      const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(k)];
      const double xr_n = node_r[static_cast<std::size_t>(n)];
      const double xz_n = node_z[static_cast<std::size_t>(n)];
      const double s_n = std::hypot(xr_n, xz_n);
      r_mean += s_n;
      Fpost_out +=
          (force_r_h[static_cast<std::size_t>(n)] * xr_n +
           force_z_h[static_cast<std::size_t>(n)] * xz_n) /
          std::max(s_n, 1.0e-300);
      ur_ring_mean +=
          (velocity_r[static_cast<std::size_t>(n)] * xr_n +
           velocity_z[static_cast<std::size_t>(n)] * xz_n) /
          std::max(s_n, 1.0e-300);
    }
    r_mean /= static_cast<double>(n_boundary);
    ur_ring_mean /= static_cast<double>(n_boundary);
    const double p1d = core1d::outer_face_pressure(
        &pc, cfg.materials.materials.front().ideal_gas_gamma);
    const double U_pool = pc.Ue_c + pc.Ui_c;
    std::fprintf(
        stderr,
        "[psc_bforce] call=%lld tag=%s p_used=%.6e p1d_outer=%.6e U_pool=%.6e "
        "V_c=%.6e n_b=%d r_ring_mean=%.6e Fpre_out=%.6e "
        "Fpre_nodemax=%.3e@%d Fpost_out=%.6e ur_ring=%.6e "
        "impulse_dt=%.3e\n",
        g_bforce_calls, diag_tag, p_c, p1d, U_pool, pc.V_c, n_boundary,
        r_mean, Fpre_out, fpre_nodemax, k_max, Fpost_out, ur_ring_mean,
        impulse_dt);
  }
}

void set_boundary_pressure_work(core::State& state,
                                const core::Config& cfg,
                                const double dt,
                                const core::NodeField1D& node_mass,
                                const double* old_velocity_r,
                                const double* old_velocity_z,
                                const double* new_velocity_r,
                                const double* new_velocity_z) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state) || state.work_p_per_cell.empty()) {
    return;
  }
  auto& pc = state.central_pseudo_core;
  const int n_boundary = static_cast<int>(pc.boundary_nodes_ordered.size());
  const int n_cells = static_cast<int>(state.rho.size());
  pc.boundary_work_pending = false;
  pc.boundary_work_valid = false;
  if (pc.d_boundary_impulse_r.empty() || pc.d_boundary_impulse_z.empty() ||
      !(dt > 0.0)) {
    zero_member_compatible_cell_buffers(state);
    return;
  }
  const int cell_blocks = (n_cells + 255) / 256;
  zero_member_work_kernel<<<cell_blocks, 256>>>(
      state.work_p_per_cell.data(),
      state.work_sub_per_cell.data(),
      state.work_av_per_cell.data(),
      pc.d_member_mask.data(),
      n_cells);
  cuda_check(cudaGetLastError(),
             "central pseudo-core zero member compatible work launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "central pseudo-core zero member compatible work failed");

  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(old_velocity_r != nullptr && old_velocity_z != nullptr &&
                    new_velocity_r != nullptr && new_velocity_z != nullptr,
                "central pseudo-core boundary pressure work requires old/new velocities");
  std::vector<double> impulse_r;
  std::vector<double> impulse_z;
  std::vector<double> old_r(static_cast<std::size_t>(n_nodes));
  std::vector<double> old_z(static_cast<std::size_t>(n_nodes));
  std::vector<double> new_r(static_cast<std::size_t>(n_nodes));
  std::vector<double> new_z(static_cast<std::size_t>(n_nodes));
  std::vector<double> rz_mass;
  std::vector<double> planar_mass;
  pc.d_boundary_impulse_r.copy_to_host(impulse_r);
  pc.d_boundary_impulse_z.copy_to_host(impulse_z);
  TENRYU_ASSERT(impulse_r.size() == static_cast<std::size_t>(n_boundary) &&
                    impulse_z.size() == static_cast<std::size_t>(n_boundary),
                "central pseudo-core boundary impulse size mismatch");
  cuda_check(cudaMemcpy(old_r.data(), old_velocity_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "central pseudo-core boundary work copy old vr failed");
  cuda_check(cudaMemcpy(old_z.data(), old_velocity_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "central pseudo-core boundary work copy old vz failed");
  cuda_check(cudaMemcpy(new_r.data(), new_velocity_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "central pseudo-core boundary work copy new vr failed");
  cuda_check(cudaMemcpy(new_z.data(), new_velocity_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "central pseudo-core boundary work copy new vz failed");
  switch (cfg.numerics.hydro.rz_momentum_scheme_id) {
    case 0:
      break;
    case 1:
      TENRYU_ASSERT(node_mass.size() == static_cast<std::size_t>(n_nodes),
                    "central pseudo-core AWS boundary work requires RZ node mass");
      TENRYU_ASSERT(state.node_planar_mass.size() ==
                        static_cast<std::size_t>(n_nodes),
                    "central pseudo-core AWS boundary work requires planar node mass");
      rz_mass = copy_field(node_mass);
      planar_mass = copy_field(state.node_planar_mass);
      break;
    default:
      TENRYU_ASSERT(false,
                    "central pseudo-core boundary work has invalid RZ momentum scheme");
  }

  long double dot_old = 0.0L;
  long double dot_new = 0.0L;
  long double dot_mid = 0.0L;
  for (int k = 0; k < n_boundary; ++k) {
    const int n = pc.boundary_nodes_ordered[static_cast<std::size_t>(k)];
    const long double Ir =
        static_cast<long double>(impulse_r[static_cast<std::size_t>(k)]);
    const long double Iz =
        static_cast<long double>(impulse_z[static_cast<std::size_t>(k)]);
    long double conjugate_scale = 1.0L;
    if (cfg.numerics.hydro.rz_momentum_scheme_id == 1) {
      const double mass_a = planar_mass[static_cast<std::size_t>(n)];
      conjugate_scale =
          mass_a > 0.0
              ? static_cast<long double>(rz_mass[static_cast<std::size_t>(n)]) /
                    static_cast<long double>(mass_a)
              : 0.0L;
    }
    const long double old_dot =
        conjugate_scale *
        (Ir * static_cast<long double>(old_r[static_cast<std::size_t>(n)]) +
         Iz * static_cast<long double>(old_z[static_cast<std::size_t>(n)]));
    const long double new_dot =
        conjugate_scale *
        (Ir * static_cast<long double>(new_r[static_cast<std::size_t>(n)]) +
         Iz * static_cast<long double>(new_z[static_cast<std::size_t>(n)]));
    dot_old += old_dot;
    dot_new += new_dot;
    dot_mid += 0.5L * (old_dot + new_dot);
  }

  // Book W_conj = -sum_k alpha_k I_k . u_mid,k, with alpha=1 for the
  // volume-weighted scheme and alpha=M_RZ/M_A for AWS.  AWS advances with
  // delta-u=I_A/M_A while global nodal KE uses M_RZ, so this is exactly the
  // negative macro-impulse contribution to the global KE change.
  pc.boundary_work_W_old = static_cast<double>(-dot_old);
  pc.boundary_work_W_new = static_cast<double>(-dot_new);
  pc.boundary_work_W_mid = static_cast<double>(-dot_mid);
  pc.boundary_work_dU = pc.boundary_work_W_mid;
  pc.boundary_work_residual =
      static_cast<double>(static_cast<long double>(pc.boundary_work_dU) + dot_mid);
  pc.boundary_work_chi_e =
      electron_energy_fraction(pc.Ue_c, pc.Ui_c, !state.ei.empty());
  pc.boundary_work_valid = true;
  pc.boundary_work_pending = true;
}

void apply_boundary_pressure_work(core::State& state,
                                  const core::Config& cfg,
                                  const bool use_two_temp) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return;
  }
  auto& pc = state.central_pseudo_core;
  if (!pc.boundary_work_pending) {
    return;
  }

  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> vol = copy_field(state.vol);
  std::vector<double> rho = copy_field(state.rho);
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);
  const MemberSums sums = sum_member_state(pc, mass, vol, ee, ei, tracer);
  TENRYU_ASSERT(sums.M > 0.0L,
                "central pseudo-core pressure work requires positive mass");
  const double V_boundary = boundary_loop_volume(pc, node_r, node_z);
  update_u_gas_energy_fraction(pc, sums);
  pc.M_c = static_cast<double>(sums.M);
  pc.M_Y_c = static_cast<double>(sums.MY);
  pc.V_c = V_boundary;
  if (use_two_temp && !ei.empty()) {
    const double chi_e = pc.boundary_work_chi_e;
    pc.Ue_c = static_cast<double>(sums.Ue) + chi_e * pc.boundary_work_dU;
    pc.Ui_c = static_cast<double>(sums.Ui) +
              (1.0 - chi_e) * pc.boundary_work_dU;
  } else {
    pc.Ue_c = static_cast<double>(sums.Ue) + pc.boundary_work_dU;
    pc.Ui_c = !ei.empty() ? static_cast<double>(sums.Ui) : 0.0;
  }
  TENRYU_ASSERT(std::isfinite(pc.Ue_c) && std::isfinite(pc.Ui_c),
                "central pseudo-core pressure work produced non-finite energy");
  write_member_mirrors(state, mass, vol, rho, ee, ei, tracer);
  conservation_audit::emit_stage(
      state, "central_boundary_pressure_work", pc.boundary_work_dU);
  pc.boundary_work_pending = false;
}

}  // namespace tenryu::hydro::central_pseudo_core
