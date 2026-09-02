#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>

#include "core/error.hpp"
#include "mesh/poly_geom.cuh"

namespace tenryu::mesh::tess {
namespace {

std::size_t canonical_rotation(
    const mesh::voronoi::VoronoiCell& cell) {
  const std::size_t vertex_count = cell.r.size();
  std::size_t best = 0;
  for (std::size_t candidate = 1; candidate < vertex_count; ++candidate) {
    for (std::size_t offset = 0; offset < vertex_count; ++offset) {
      const std::size_t candidate_index =
          (candidate + offset) % vertex_count;
      const std::size_t best_index = (best + offset) % vertex_count;
      const auto candidate_key = std::make_tuple(
          cell.neighbor_ids[candidate_index],
          std::bit_cast<std::uint64_t>(cell.r[candidate_index]),
          std::bit_cast<std::uint64_t>(cell.z[candidate_index]));
      const auto best_key = std::make_tuple(
          cell.neighbor_ids[best_index],
          std::bit_cast<std::uint64_t>(cell.r[best_index]),
          std::bit_cast<std::uint64_t>(cell.z[best_index]));
      if (candidate_key < best_key) {
        best = candidate;
        break;
      }
      if (best_key < candidate_key) {
        break;
      }
    }
  }
  return best;
}

}  // namespace

mesh::voronoi::Tessellation to_legacy_tessellation(
    const DelaunayTriangulation& dt,
    const RestrictedDcel& dcel) {
  TENRYU_ASSERT(
      dcel.nodes.size() <=
          static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "legacy adapter node count exceeds int range");

  mesh::voronoi::Tessellation result;
  result.node_r.reserve(dcel.nodes.size());
  result.node_z.reserve(dcel.nodes.size());
  result.node_boundary.reserve(dcel.nodes.size());
  for (const RestrictedNode& node : dcel.nodes) {
    result.node_r.push_back(node.constructed[0]);
    result.node_z.push_back(node.constructed[1]);
    mesh::voronoi::TessNodeBoundary boundary;
    if (!node.domain_vertices.empty()) {
      TENRYU_ASSERT(!node.boundary_param.valid,
                    "carrier_provenance_overlap");
      boundary.cls = mesh::voronoi::TessNodeClass::kCarrierMaster;
      boundary.domain_vertex = *std::min_element(
          node.domain_vertices.begin(), node.domain_vertices.end());
    } else if (node.boundary_param.valid) {
      boundary.cls = mesh::voronoi::TessNodeClass::kCarrierSlave;
      boundary.carrier_edge = node.boundary_param.edge;
      boundary.lambda_cache = node.lambda_cache;
    }
    result.node_boundary.push_back(boundary);
  }

  std::map<std::pair<SiteId, RestrictedNodeId>, RestrictedHalfEdgeId>
      edge_by_face_origin;
  for (RestrictedHalfEdgeId edge = 0;
       edge < static_cast<RestrictedHalfEdgeId>(dcel.half_edges.size());
       ++edge) {
    edge_by_face_origin.emplace(
        std::make_pair(dcel.half_edges[edge].left_face,
                       dcel.half_edges[edge].origin),
        edge);
  }

  result.cells.reserve(dcel.cells.size());
  for (const RestrictedCell& restricted_cell : dcel.cells) {
    mesh::voronoi::VoronoiCell cell;
    cell.generator_id = dt.sites[restricted_cell.site].stable_id;
    const std::size_t vertex_count = restricted_cell.cycle.size();
    cell.node_indices.reserve(vertex_count);
    cell.r.reserve(vertex_count);
    cell.z.reserve(vertex_count);
    cell.neighbor_ids.reserve(vertex_count);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
      const RestrictedNodeId node = restricted_cell.cycle[vertex];
      cell.node_indices.push_back(static_cast<int>(node));
      cell.r.push_back(dcel.nodes[node].constructed[0]);
      cell.z.push_back(dcel.nodes[node].constructed[1]);

      const RestrictedHalfEdgeId edge = edge_by_face_origin.at(
          std::make_pair(restricted_cell.site, node));
      const RestrictedHalfEdge& half_edge = dcel.half_edges[edge];
      const RestrictedHalfEdge& twin = dcel.half_edges[half_edge.twin];
      TENRYU_ASSERT(
          twin.origin == restricted_cell.cycle[(vertex + 1) % vertex_count],
          "legacy adapter cycle-halfedge mismatch");
      cell.neighbor_ids.push_back(
          twin.left_face == dcel.exterior_face
              ? mesh::voronoi::kBoundaryNeighbor
              : dt.sites[twin.left_face].stable_id);
    }
    cell.volume_rz = mesh::reale::polygon_rz_moments(
                         cell.r.data(), cell.z.data(),
                         static_cast<int>(cell.r.size()))
                         .volume;
    result.cells.push_back(std::move(cell));
  }

  result.collapsed_edges = 0;
  result.valid = true;
  result.reject_reason.clear();
  return result;
}

TessellationComparison compare_tessellations(
    const mesh::voronoi::Tessellation& legacy,
    const mesh::voronoi::Tessellation& shadow) {
  TessellationComparison comparison;
  comparison.nodes_old = legacy.node_r.size();
  comparison.nodes_new = shadow.node_r.size();
  for (const mesh::voronoi::VoronoiCell& cell : shadow.cells) {
    const int valence = static_cast<int>(cell.r.size());
    comparison.max_valence = std::max(comparison.max_valence, valence);
    if (cell.r.size() > 8U) {
      ++comparison.nv_over_8;
    }
  }

  if (legacy.cells.size() != shadow.cells.size()) {
    comparison.first_divergence = "cell_count";
    return comparison;
  }
  for (std::size_t cell_index = 0; cell_index < legacy.cells.size();
       ++cell_index) {
    if (legacy.cells[cell_index].generator_id !=
        shadow.cells[cell_index].generator_id) {
      comparison.first_divergence = "generator_ids";
      return comparison;
    }
  }

  for (std::size_t cell_index = 0; cell_index < legacy.cells.size();
       ++cell_index) {
    const mesh::voronoi::VoronoiCell& old_cell = legacy.cells[cell_index];
    const mesh::voronoi::VoronoiCell& new_cell = shadow.cells[cell_index];
    const std::size_t old_vertex_count = old_cell.r.size();
    const std::size_t new_vertex_count = new_cell.r.size();
    if (old_vertex_count != new_vertex_count) {
      comparison.first_divergence =
          "cell " + std::to_string(old_cell.generator_id) + " nv " +
          std::to_string(old_vertex_count) + " vs " +
          std::to_string(new_vertex_count);
      return comparison;
    }

    const std::size_t old_rotation = canonical_rotation(old_cell);
    const std::size_t new_rotation = canonical_rotation(new_cell);
    for (std::size_t edge = 0; edge < old_vertex_count; ++edge) {
      const std::size_t old_index =
          (old_rotation + edge) % old_vertex_count;
      const std::size_t new_index =
          (new_rotation + edge) % new_vertex_count;
      const std::uint64_t old_neighbor = old_cell.neighbor_ids[old_index];
      const std::uint64_t new_neighbor = new_cell.neighbor_ids[new_index];
      if (old_neighbor != new_neighbor) {
        comparison.first_divergence =
            "cell " + std::to_string(old_cell.generator_id) + " edge " +
            std::to_string(edge) + " neighbor " +
            std::to_string(old_neighbor) + " vs " +
            std::to_string(new_neighbor);
        return comparison;
      }
      comparison.max_coord_delta = std::max(
          comparison.max_coord_delta,
          std::abs(old_cell.r[old_index] - new_cell.r[new_index]));
      comparison.max_coord_delta = std::max(
          comparison.max_coord_delta,
          std::abs(old_cell.z[old_index] - new_cell.z[new_index]));
    }
    comparison.max_volume_delta = std::max(
        comparison.max_volume_delta,
        std::abs(old_cell.volume_rz - new_cell.volume_rz));
  }

  comparison.topology_equal = true;
  return comparison;
}

}  // namespace tenryu::mesh::tess
