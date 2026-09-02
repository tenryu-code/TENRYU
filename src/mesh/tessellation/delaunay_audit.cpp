#include "mesh/tessellation/delaunay_2d.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {
namespace {

struct EdgeKey {
  SiteId lo;
  SiteId hi;

  bool operator<(const EdgeKey& other) const {
    return lo < other.lo || (lo == other.lo && hi < other.hi);
  }
};

EdgeKey edge_key(const SiteId a, const SiteId b) {
  return EdgeKey{std::min(a, b), std::max(a, b)};
}

DelaunayAuditResult failure(const char* code) {
  return DelaunayAuditResult{false, code};
}

bool valid_half_edge(const DelaunayTriangulation& dt,
                     const HalfEdgeId edge) {
  return edge >= 0 &&
         edge < static_cast<HalfEdgeId>(dt.he_origin.size());
}

}  // namespace

DelaunayAuditResult audit_delaunay(const DelaunayTriangulation& dt) {
  const std::size_t half_edge_count = dt.he_origin.size();
  if (dt.he_twin.size() != half_edge_count ||
      dt.he_next.size() != half_edge_count ||
      dt.he_face.size() != half_edge_count ||
      half_edge_count != 3 * dt.tri_edge.size()) {
    return failure("nonmanifold_edge");
  }

  std::vector<unsigned char> visited(half_edge_count, 0);
  std::vector<unsigned char> represented(dt.sites.size(), 0);
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const HalfEdgeId edge0 = dt.tri_edge[triangle];
    if (!valid_half_edge(dt, edge0)) {
      return failure("nonmanifold_edge");
    }
    const HalfEdgeId edge1 = dt.he_next[edge0];
    if (!valid_half_edge(dt, edge1)) {
      return failure("nonmanifold_edge");
    }
    const HalfEdgeId edge2 = dt.he_next[edge1];
    if (!valid_half_edge(dt, edge2) || dt.he_next[edge2] != edge0 ||
        edge0 == edge1 || edge1 == edge2 || edge2 == edge0) {
      return failure("nonmanifold_edge");
    }

    const std::array<HalfEdgeId, 3> edges{{edge0, edge1, edge2}};
    std::array<SiteId, 3> vertices{};
    for (int local = 0; local < 3; ++local) {
      const HalfEdgeId edge = edges[local];
      if (visited[edge] || dt.he_face[edge] != triangle) {
        return failure("nonmanifold_edge");
      }
      visited[edge] = 1;
      const SiteId origin = dt.he_origin[edge];
      if (origin < 0 || origin >= static_cast<SiteId>(dt.sites.size())) {
        return failure("nonmanifold_edge");
      }
      vertices[local] = origin;
      represented[origin] = 1;
    }
    if (vertices[0] == vertices[1] || vertices[1] == vertices[2] ||
        vertices[2] == vertices[0]) {
      return failure("nonmanifold_edge");
    }
    if (orient2d_sign_tess(dt.sites[vertices[0]], dt.sites[vertices[1]],
                           dt.sites[vertices[2]]) <= 0) {
      return failure("orientation_violation");
    }
  }
  if (std::find(visited.begin(), visited.end(), 0) != visited.end()) {
    return failure("nonmanifold_edge");
  }

  std::vector<std::pair<EdgeKey, HalfEdgeId>> incidence_entries;
  incidence_entries.reserve(half_edge_count);
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(half_edge_count); ++edge) {
    const HalfEdgeId next = dt.he_next[edge];
    if (!valid_half_edge(dt, next)) {
      return failure("nonmanifold_edge");
    }
    const SiteId a = dt.he_origin[edge];
    const SiteId b = dt.he_origin[next];
    if (a == b) {
      return failure("nonmanifold_edge");
    }
    incidence_entries.emplace_back(edge_key(a, b), edge);
  }

  std::sort(incidence_entries.begin(), incidence_entries.end(),
            [](const auto& lhs, const auto& rhs) {
              if (lhs.first < rhs.first) {
                return true;
              }
              if (rhs.first < lhs.first) {
                return false;
              }
              return lhs.second < rhs.second;
            });

  std::size_t incidence_group_count = 0;
  std::size_t group_begin = 0;
  while (group_begin < incidence_entries.size()) {
    ++incidence_group_count;
    std::size_t group_end = group_begin + 1;
    while (group_end < incidence_entries.size() &&
           !(incidence_entries[group_begin].first <
             incidence_entries[group_end].first) &&
           !(incidence_entries[group_end].first <
             incidence_entries[group_begin].first)) {
      ++group_end;
    }
    const std::size_t group_size = group_end - group_begin;
    if (group_size == 1) {
      if (dt.he_twin[incidence_entries[group_begin].second] != kInvalidId) {
        return failure("nonmanifold_edge");
      }
      group_begin = group_end;
      continue;
    }
    if (group_size != 2) {
      return failure("nonmanifold_edge");
    }
    const HalfEdgeId first = incidence_entries[group_begin].second;
    const HalfEdgeId second = incidence_entries[group_begin + 1].second;
    if (dt.he_twin[first] != second || dt.he_twin[second] != first ||
        dt.he_origin[first] != dt.he_origin[dt.he_next[second]] ||
        dt.he_origin[second] != dt.he_origin[dt.he_next[first]]) {
      return failure("nonmanifold_edge");
    }

    const SiteId a = dt.he_origin[first];
    const SiteId b = dt.he_origin[dt.he_next[first]];
    const SiteId c = dt.he_origin[dt.he_next[dt.he_next[first]]];
    const SiteId d = dt.he_origin[dt.he_next[dt.he_next[second]]];
    if (incircle_sos(dt.sites[a], dt.sites[b], dt.sites[c],
                     dt.sites[d]) > 0) {
      return failure("delaunay_illegal_edge");
    }
    group_begin = group_end;
  }

  if (std::find(represented.begin(), represented.end(), 0) !=
      represented.end()) {
    return failure("euler_invariant_violation");
  }
  const std::int64_t vertices = static_cast<std::int64_t>(dt.sites.size());
  const std::int64_t edges =
      static_cast<std::int64_t>(incidence_group_count);
  const std::int64_t faces =
      static_cast<std::int64_t>(dt.tri_edge.size()) + 1;
  if (vertices - edges + faces != 2) {
    return failure("euler_invariant_violation");
  }

  return DelaunayAuditResult{true, {}};
}

}  // namespace tenryu::mesh::tess
