#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::tess {
namespace {

void append_u32(std::vector<std::uint8_t>& bytes,
                const std::uint32_t value) {
  for (unsigned shift = 0; shift < 32; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_i32(std::vector<std::uint8_t>& bytes,
                const std::int32_t value) {
  append_u32(bytes, static_cast<std::uint32_t>(value));
}

void append_u64(std::vector<std::uint8_t>& bytes,
                const std::uint64_t value) {
  for (unsigned shift = 0; shift < 64; shift += 8) {
    bytes.push_back(static_cast<std::uint8_t>(value >> shift));
  }
}

void append_double(std::vector<std::uint8_t>& bytes, double value) {
  if (value == 0.0) {
    value = 0.0;
  }
  std::uint64_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  append_u64(bytes, bits);
}

void append_key(std::vector<std::uint8_t>& bytes,
                const CanonicalNodeKey& key) {
  bytes.push_back(static_cast<std::uint8_t>(key.kind));
  if (key.kind == CanonicalNodeKeyKind::kDomainVertex) {
    append_i32(bytes, key.domain_vertex);
    return;
  }
  if (key.kind == CanonicalNodeKeyKind::kVoronoiTriple) {
    for (const std::uint64_t stable_id : key.voronoi_triple.stable_ids) {
      append_u64(bytes, stable_id);
    }
    return;
  }
  append_i32(bytes, key.domain_segment);
  append_u64(bytes, key.crossing_sites[0]);
  append_u64(bytes, key.crossing_sites[1]);
}

}  // namespace

std::vector<std::uint8_t> canonical_serialize(
    const DelaunayTriangulation& dt,
    const RestrictedDcel& dcel) {
  TENRYU_ASSERT(dcel.cells.size() == dt.sites.size(),
                "canonical serializer cell count mismatch");
  std::vector<std::uint8_t> bytes;
  bytes.insert(bytes.end(), {'T', 'R', 'V', '1'});
  std::vector<std::size_t> cells(dcel.cells.size());
  std::iota(cells.begin(), cells.end(), std::size_t{0});
  std::sort(cells.begin(), cells.end(), [&](const std::size_t lhs,
                                            const std::size_t rhs) {
    return dt.sites[dcel.cells[lhs].site].stable_id <
           dt.sites[dcel.cells[rhs].site].stable_id;
  });

  std::vector<RestrictedNodeId> node_remap(dcel.nodes.size(), kInvalidId);
  std::vector<RestrictedNodeId> canonical_nodes;
  canonical_nodes.reserve(dcel.nodes.size());
  for (const std::size_t cell_index : cells) {
    std::vector<RestrictedNodeId> cycle = dcel.cells[cell_index].cycle;
    TENRYU_ASSERT(!cycle.empty(),
                  "canonical serializer has an empty cell");
    const auto first = std::min_element(
        cycle.begin(), cycle.end(),
        [&](const RestrictedNodeId lhs, const RestrictedNodeId rhs) {
          return dcel.nodes[lhs].key < dcel.nodes[rhs].key;
        });
    std::rotate(cycle.begin(), first, cycle.end());
    for (const RestrictedNodeId old_node : cycle) {
      TENRYU_ASSERT(old_node >= 0 &&
                        old_node < static_cast<RestrictedNodeId>(
                                       dcel.nodes.size()),
                    "canonical serializer node is out of range");
      if (node_remap[old_node] == kInvalidId) {
        node_remap[old_node] =
            static_cast<RestrictedNodeId>(canonical_nodes.size());
        canonical_nodes.push_back(old_node);
      }
    }
  }
  TENRYU_ASSERT(canonical_nodes.size() == dcel.nodes.size(),
                "canonical serializer found an unused node");

  append_u64(bytes, canonical_nodes.size());
  for (const RestrictedNodeId old_node : canonical_nodes) {
    const RestrictedNode& node = dcel.nodes[old_node];
    append_key(bytes, node.key);
    append_double(bytes, node.constructed[0]);
    append_double(bytes, node.constructed[1]);
    append_u64(bytes, node.aliases.size());
    for (const CanonicalNodeKey& alias : node.aliases) {
      append_key(bytes, alias);
    }
    append_u64(bytes, node.boundary_segments.size());
    for (const DomainSegmentId segment : node.boundary_segments) {
      append_i32(bytes, segment);
    }
    append_u64(bytes, node.domain_vertices.size());
    for (const DomainVertexId vertex : node.domain_vertices) {
      append_i32(bytes, vertex);
    }
  }

  append_u64(bytes, cells.size());
  for (const std::size_t cell_index : cells) {
    const RestrictedCell& cell = dcel.cells[cell_index];
    TENRYU_ASSERT(cell.site >= 0 &&
                      cell.site < static_cast<SiteId>(dt.sites.size()) &&
                      !cell.cycle.empty(),
                  "canonical serializer has an invalid cell");
    append_u64(bytes, dt.sites[cell.site].stable_id);
    std::vector<RestrictedNodeId> cycle = cell.cycle;
    for (RestrictedNodeId& node : cycle) {
      node = node_remap[node];
    }
    const auto minimum = std::min_element(cycle.begin(), cycle.end());
    std::rotate(cycle.begin(), minimum, cycle.end());
    append_u64(bytes, cycle.size());
    for (const RestrictedNodeId node : cycle) {
      append_i32(bytes, node);
    }
  }
  return bytes;
}

}  // namespace tenryu::mesh::tess
