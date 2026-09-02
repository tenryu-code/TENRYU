#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::tess {
namespace {

bool same_exact_point(const HomogeneousPoint& first,
                      const HomogeneousPoint& second) {
  return expansion_sign(expansion_difference(
             expansion_product(first.x, second.w),
             expansion_product(second.x, first.w))) == 0 &&
         expansion_sign(expansion_difference(
             expansion_product(first.y, second.w),
             expansion_product(second.y, first.w))) == 0;
}

bool bitwise_identical(const std::array<double, 2>& first,
                       const std::array<double, 2>& second) {
  std::array<std::uint64_t, 2> first_bits{};
  std::array<std::uint64_t, 2> second_bits{};
  std::memcpy(first_bits.data(), first.data(), sizeof(first_bits));
  std::memcpy(second_bits.data(), second.data(), sizeof(second_bits));
  return first_bits == second_bits;
}

Expansion rational_difference(const Expansion& first_coordinate,
                              const Expansion& first_weight,
                              const Expansion& second_coordinate,
                              const Expansion& second_weight) {
  return expansion_difference(
      expansion_product(first_coordinate, second_weight),
      expansion_product(second_coordinate, first_weight));
}

int projection_compare(const HomogeneousPoint& first,
                       const HomogeneousPoint& second,
                       const HomogeneousPoint& segment_start,
                       const HomogeneousPoint& segment_end) {
  const Expansion difference_r = rational_difference(
      first.x, first.w, second.x, second.w);
  const Expansion difference_z = rational_difference(
      first.y, first.w, second.y, second.w);
  const Expansion direction_r = rational_difference(
      segment_end.x, segment_end.w, segment_start.x, segment_start.w);
  const Expansion direction_z = rational_difference(
      segment_end.y, segment_end.w, segment_start.y, segment_start.w);
  return expansion_sign(expansion_sum(
      expansion_product(difference_r, direction_r),
      expansion_product(difference_z, direction_z)));
}

template <typename T>
void sort_unique(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

// One ascending-node pass replaces the per-segment all-node
// binary_search scans (~193 x ~208k probes per call). Ascending node
// order reproduces the original filtered-scan order exactly, so every
// per-segment event list is identical to the scan it replaces.
std::vector<std::vector<RestrictedNodeId>> collect_segment_events(
    const RestrictedGeometry& geometry) {
  std::vector<std::vector<RestrictedNodeId>> events_by_segment(
      geometry.domain_segments.size());
  for (RestrictedNodeId node = 0;
       node < static_cast<RestrictedNodeId>(geometry.nodes.size()); ++node) {
    for (const DomainSegmentId segment_id :
         geometry.nodes[node].boundary_segments) {
      TENRYU_ASSERT(segment_id >= 0 &&
                        segment_id < static_cast<DomainSegmentId>(
                                         events_by_segment.size()),
                    "boundary segment id out of range");
      events_by_segment[static_cast<std::size_t>(segment_id)].push_back(node);
    }
  }
  return events_by_segment;
}

class NodeUnionFind {
 public:
  explicit NodeUnionFind(const std::vector<RestrictedNode>& nodes)
      : parent_(nodes.size()), nodes_(nodes) {
    std::iota(parent_.begin(), parent_.end(), RestrictedNodeId{0});
  }

  RestrictedNodeId find(const RestrictedNodeId node) {
    TENRYU_ASSERT(node >= 0 &&
                      node < static_cast<RestrictedNodeId>(parent_.size()),
                  "restricted node is out of range");
    RestrictedNodeId root = node;
    while (parent_[root] != root) {
      root = parent_[root];
    }
    RestrictedNodeId current = node;
    while (parent_[current] != current) {
      const RestrictedNodeId next = parent_[current];
      parent_[current] = root;
      current = next;
    }
    return root;
  }

  void unite(const RestrictedNodeId first,
             const RestrictedNodeId second) {
    RestrictedNodeId first_root = find(first);
    RestrictedNodeId second_root = find(second);
    if (first_root == second_root) {
      return;
    }
    if (nodes_[second_root].key < nodes_[first_root].key) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
  }

 private:
  std::vector<RestrictedNodeId> parent_;
  const std::vector<RestrictedNode>& nodes_;
};

void compact_unioned_nodes(RestrictedGeometry& geometry,
                           NodeUnionFind& union_find) {
  std::vector<RestrictedNodeId> root_of(geometry.nodes.size());
  std::vector<std::size_t> member_counts(geometry.nodes.size(), 0);
  for (RestrictedNodeId node = 0;
       node < static_cast<RestrictedNodeId>(geometry.nodes.size()); ++node) {
    root_of[node] = union_find.find(node);
    ++member_counts[root_of[node]];
  }
  std::vector<RestrictedNodeId> roots;
  for (RestrictedNodeId node = 0;
       node < static_cast<RestrictedNodeId>(member_counts.size()); ++node) {
    if (member_counts[node] != 0 && root_of[node] == node) {
      roots.push_back(node);
    }
  }
  std::sort(roots.begin(), roots.end(), [&](const RestrictedNodeId lhs,
                                            const RestrictedNodeId rhs) {
    return geometry.nodes[lhs].key < geometry.nodes[rhs].key;
  });

  std::vector<std::size_t> member_offsets(geometry.nodes.size() + 1, 0);
  std::partial_sum(member_counts.begin(), member_counts.end(),
                   member_offsets.begin() + 1);
  std::vector<std::size_t> member_cursors(member_offsets.begin(),
                                          member_offsets.end() - 1);
  std::vector<RestrictedNodeId> members(geometry.nodes.size());
  for (RestrictedNodeId node = 0;
       node < static_cast<RestrictedNodeId>(geometry.nodes.size()); ++node) {
    members[member_cursors[root_of[node]]++] = node;
  }

  std::vector<RestrictedNodeId> remap(geometry.nodes.size(), kInvalidId);
  std::vector<RestrictedNode> compact;
  compact.reserve(roots.size());
  for (const RestrictedNodeId root : roots) {
    TENRYU_ASSERT(compact.size() <= static_cast<std::size_t>(
                                        std::numeric_limits<RestrictedNodeId>::max()),
                  "restricted node count exceeds RestrictedNodeId range");
    // Move is safe: each root appears exactly once in the sorted roots list,
    // the key-ordered roots sort completed before this loop, and the member
    // loop below only reads geometry.nodes[member] for member != root.
    RestrictedNode merged = std::move(geometry.nodes[root]);
    for (std::size_t member_index = member_offsets[root];
         member_index < member_offsets[root + 1]; ++member_index) {
      const RestrictedNodeId member = members[member_index];
      remap[member] = static_cast<RestrictedNodeId>(compact.size());
      if (member == root) {
        continue;
      }
      TENRYU_ASSERT(
          same_exact_point(merged.exact_point,
                           geometry.nodes[member].exact_point) ||
              bitwise_identical(merged.constructed,
                                geometry.nodes[member].constructed),
          "domain_vertex_alias_conflict");
      merged.aliases.insert(merged.aliases.end(),
                            geometry.nodes[member].aliases.begin(),
                            geometry.nodes[member].aliases.end());
      merged.exact_witnesses.insert(
          merged.exact_witnesses.end(),
          geometry.nodes[member].exact_witnesses.begin(),
          geometry.nodes[member].exact_witnesses.end());
      merged.boundary_segments.insert(
          merged.boundary_segments.end(),
          geometry.nodes[member].boundary_segments.begin(),
          geometry.nodes[member].boundary_segments.end());
      merged.domain_vertices.insert(
          merged.domain_vertices.end(),
          geometry.nodes[member].domain_vertices.begin(),
          geometry.nodes[member].domain_vertices.end());
    }
    sort_unique(merged.aliases);
    std::sort(merged.exact_witnesses.begin(),
              merged.exact_witnesses.end(),
              [](const RestrictedExactWitness& lhs,
                 const RestrictedExactWitness& rhs) {
                return lhs.key < rhs.key;
              });
    merged.exact_witnesses.erase(
        std::unique(merged.exact_witnesses.begin(),
                    merged.exact_witnesses.end(),
                    [](const RestrictedExactWitness& lhs,
                       const RestrictedExactWitness& rhs) {
                      return lhs.key == rhs.key;
                    }),
        merged.exact_witnesses.end());
    sort_unique(merged.boundary_segments);
    sort_unique(merged.domain_vertices);
    merged.key = merged.aliases.front();
    compact.push_back(std::move(merged));
  }

  for (RestrictedDualEdge& edge : geometry.interior_edges) {
    edge.endpoint0 = remap[edge.endpoint0];
    edge.endpoint1 = remap[edge.endpoint1];
  }
  geometry.interior_edges.erase(
      std::remove_if(geometry.interior_edges.begin(),
                     geometry.interior_edges.end(),
                     [](const RestrictedDualEdge& edge) {
                       return edge.endpoint0 == edge.endpoint1;
                     }),
      geometry.interior_edges.end());
  for (DomainSegmentRecord& segment : geometry.domain_segments) {
    segment.start_node = remap[segment.start_node];
    segment.end_node = remap[segment.end_node];
  }
  geometry.nodes = std::move(compact);
  geometry.identity_contractions.clear();
}

HomogeneousPoint exact_midpoint(const HomogeneousPoint& first,
                                const HomogeneousPoint& second) {
  const Expansion common_weight =
      expansion_product(first.w, second.w);
  return HomogeneousPoint{
      expansion_sum(expansion_product(first.x, second.w),
                    expansion_product(second.x, first.w)),
      expansion_sum(expansion_product(first.y, second.w),
                    expansion_product(second.y, first.w)),
      expansion_scale(common_weight, 2.0)};
}

Expansion squared_distance_numerator(const HomogeneousPoint& point,
                                     const Site& site) {
  const Expansion dr = expansion_difference(
      point.x, expansion_scale(point.w, site.r));
  const Expansion dz = expansion_difference(
      point.y, expansion_scale(point.w, site.z));
  return expansion_sum(expansion_product(dr, dr),
                       expansion_product(dz, dz));
}

int compare_distance(const HomogeneousPoint& point,
                     const Site& first,
                     const Site& second) {
  return expansion_sign(expansion_difference(
      squared_distance_numerator(point, first),
      squared_distance_numerator(point, second)));
}

struct ApproxHomogeneous {
  ApproxBound x;
  ApproxBound y;
  ApproxBound w;
};

ApproxHomogeneous approx_of_point(const HomogeneousPoint& point) {
  return ApproxHomogeneous{approx_of_expansion(point.x),
                           approx_of_expansion(point.y),
                           approx_of_expansion(point.w)};
}

// Filtered variant of projection_compare: the double screen answers
// certain-sign cases and falls back to the exact expansions otherwise —
// same guaranteed sign, so this is exact-equivalent (fix36 semantics).
int projection_compare_screened(const HomogeneousPoint& first,
                                const HomogeneousPoint& second,
                                const HomogeneousPoint& segment_start,
                                const HomogeneousPoint& segment_end) {
  const ApproxHomogeneous a_first = approx_of_point(first);
  const ApproxHomogeneous a_second = approx_of_point(second);
  const ApproxHomogeneous a_start = approx_of_point(segment_start);
  const ApproxHomogeneous a_end = approx_of_point(segment_end);
  const ApproxBound difference_r = approx_sub(
      approx_mul(a_first.x, a_second.w), approx_mul(a_second.x, a_first.w));
  const ApproxBound difference_z = approx_sub(
      approx_mul(a_first.y, a_second.w), approx_mul(a_second.y, a_first.w));
  const ApproxBound direction_r = approx_sub(
      approx_mul(a_end.x, a_start.w), approx_mul(a_start.x, a_end.w));
  const ApproxBound direction_z = approx_sub(
      approx_mul(a_end.y, a_start.w), approx_mul(a_start.y, a_end.w));
  const ApproxSign screened = approx_sign(
      approx_add(approx_mul(difference_r, direction_r),
                 approx_mul(difference_z, direction_z)));
  if (screened.certain) {
    return screened.sign;
  }
  return projection_compare(first, second, segment_start, segment_end);
}

// Certain-nonzero screen for same_exact_point: if either homogeneous
// coordinate difference is certainly nonzero, the points differ; only
// the uncertain case runs the exact test. Exact-equivalent.
bool same_exact_point_screened(const HomogeneousPoint& first,
                               const HomogeneousPoint& second) {
  const ApproxHomogeneous a_first = approx_of_point(first);
  const ApproxHomogeneous a_second = approx_of_point(second);
  const ApproxSign x_sign = approx_sign(approx_sub(
      approx_mul(a_first.x, a_second.w), approx_mul(a_second.x, a_first.w)));
  if (x_sign.certain && x_sign.sign != 0) {
    return false;
  }
  const ApproxSign y_sign = approx_sign(approx_sub(
      approx_mul(a_first.y, a_second.w), approx_mul(a_second.y, a_first.w)));
  if (y_sign.certain && y_sign.sign != 0) {
    return false;
  }
  return same_exact_point(first, second);
}

ApproxBound approx_squared_distance(const ApproxHomogeneous& point,
                                    const Site& site) {
  const ApproxBound dr = approx_sub(
      point.x, approx_mul(point.w, ApproxBound{site.r, 0.0}));
  const ApproxBound dz = approx_sub(
      point.y, approx_mul(point.w, ApproxBound{site.z, 0.0}));
  return approx_add(approx_mul(dr, dr), approx_mul(dz, dz));
}

// Filtered variant of compare_distance: the double screen answers
// certain-sign cases and falls back to the exact expansions otherwise —
// same guaranteed sign, so this is exact-equivalent (fix36 semantics).
int compare_distance_screened(const ApproxHomogeneous& approx_point,
                              const HomogeneousPoint& point,
                              const Site& first,
                              const Site& second) {
  const ApproxSign screened = approx_sign(
      approx_sub(approx_squared_distance(approx_point, first),
                 approx_squared_distance(approx_point, second)));
  if (screened.certain) {
    return screened.sign;
  }
  return compare_distance(point, first, second);
}

std::vector<std::vector<SiteId>> delaunay_neighbors(
    const DelaunayTriangulation& dt) {
  std::vector<std::size_t> degrees(dt.sites.size(), 0);
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    const SiteId first = dt.he_origin[edge];
    const SiteId second = dt.he_origin[dt.he_next[edge]];
    ++degrees[first];
    ++degrees[second];
  }
  std::vector<std::size_t> offsets(dt.sites.size() + 1, 0);
  std::partial_sum(degrees.begin(), degrees.end(), offsets.begin() + 1);
  std::vector<std::size_t> cursors(offsets.begin(), offsets.end() - 1);
  std::vector<SiteId> flat_neighbors(offsets.back());
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    const SiteId first = dt.he_origin[edge];
    const SiteId second = dt.he_origin[dt.he_next[edge]];
    flat_neighbors[cursors[first]++] = second;
    flat_neighbors[cursors[second]++] = first;
  }

  std::vector<std::vector<SiteId>> neighbors(dt.sites.size());
  for (SiteId site = 0;
       site < static_cast<SiteId>(neighbors.size()); ++site) {
    std::vector<SiteId>& list = neighbors[site];
    list.assign(flat_neighbors.begin() + offsets[site],
                flat_neighbors.begin() + offsets[site + 1]);
    std::sort(list.begin(), list.end(), [&](const SiteId lhs,
                                            const SiteId rhs) {
      return dt.sites[lhs].stable_id < dt.sites[rhs].stable_id;
    });
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  return neighbors;
}

SiteId nearest_site(const DelaunayTriangulation& dt,
                    const std::vector<std::vector<SiteId>>& neighbors,
                    const HomogeneousPoint& point,
                    SiteId start) {
  TENRYU_ASSERT(!dt.sites.empty(), "boundary_interval_unowned");
  const ApproxHomogeneous approx_point = approx_of_point(point);
  if (start == kInvalidId) {
    start = 0;
  }
  SiteId current = start;
  while (true) {
    SiteId best = current;
    for (const SiteId candidate : neighbors[current]) {
      const int comparison =
          compare_distance_screened(approx_point, point, dt.sites[candidate],
                                    dt.sites[best]);
      if (comparison < 0 ||
          (comparison == 0 &&
           dt.sites[candidate].stable_id < dt.sites[best].stable_id)) {
        best = candidate;
      }
    }
    if (best == current) {
      for (const SiteId candidate : neighbors[current]) {
        TENRYU_ASSERT(
            compare_distance_screened(approx_point, point,
                                      dt.sites[candidate],
                                      dt.sites[current]) != 0,
            "boundary_interval_unowned");
      }
      return current;
    }
    TENRYU_ASSERT(compare_distance_screened(approx_point, point,
                                            dt.sites[best],
                                            dt.sites[current]) < 0,
                  "boundary_interval_unowned");
    current = best;
  }
}

}  // namespace

void build_boundary_arrangement(const DelaunayTriangulation& dt,
                                RestrictedGeometry& geometry) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  double uf_ms = 0.0;
  double events_ms = 0.0;
  double sort_ms = 0.0;
  double adj_ms = 0.0;
  double compact_ms = 0.0;
  std::size_t n_events = 0;
  std::chrono::steady_clock::time_point uf_start;
  if (timing_enabled) {
    uf_start = std::chrono::steady_clock::now();
  }
  NodeUnionFind union_find(geometry.nodes);
  for (const auto& contraction : geometry.identity_contractions) {
    union_find.unite(contraction[0], contraction[1]);
  }
  if (timing_enabled) {
    uf_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - uf_start)
                .count();
  }

  std::chrono::steady_clock::time_point events_start;
  if (timing_enabled) {
    events_start = std::chrono::steady_clock::now();
  }
  const std::vector<std::vector<RestrictedNodeId>> events_by_segment =
      collect_segment_events(geometry);
  if (timing_enabled) {
    events_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - events_start)
                    .count();
  }
  for (const DomainSegmentRecord& segment : geometry.domain_segments) {
    std::vector<RestrictedNodeId> events =
        events_by_segment[static_cast<std::size_t>(segment.id)];
    if (timing_enabled) {
      n_events += events.size();
    }
    const HomogeneousPoint& start =
        geometry.nodes[segment.start_node].exact_point;
    const HomogeneousPoint& end =
        geometry.nodes[segment.end_node].exact_point;
    std::chrono::steady_clock::time_point sort_start;
    if (timing_enabled) {
      sort_start = std::chrono::steady_clock::now();
    }
    std::sort(events.begin(), events.end(),
              [&](const RestrictedNodeId lhs, const RestrictedNodeId rhs) {
      const int order = projection_compare_screened(
          geometry.nodes[lhs].exact_point,
          geometry.nodes[rhs].exact_point, start, end);
      if (order != 0) {
        return order < 0;
      }
      return geometry.nodes[lhs].key < geometry.nodes[rhs].key;
    });
    if (timing_enabled) {
      sort_ms += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - sort_start)
                     .count();
    }
    std::chrono::steady_clock::time_point adj_start;
    if (timing_enabled) {
      adj_start = std::chrono::steady_clock::now();
    }
    for (std::size_t index = 1; index < events.size(); ++index) {
      const RestrictedNode& previous = geometry.nodes[events[index - 1]];
      const RestrictedNode& current = geometry.nodes[events[index]];
      if (bitwise_identical(previous.constructed, current.constructed) ||
          same_exact_point_screened(previous.exact_point,
                                    current.exact_point)) {
        union_find.unite(events[index - 1], events[index]);
      }
    }
    if (timing_enabled) {
      adj_ms += std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - adj_start)
                    .count();
    }
  }
  std::chrono::steady_clock::time_point compact_start;
  if (timing_enabled) {
    compact_start = std::chrono::steady_clock::now();
  }
  compact_unioned_nodes(geometry, union_find);
  if (timing_enabled) {
    compact_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - compact_start)
                     .count();
  }

  if (timing_enabled) {
    std::fprintf(stderr,
                 "[tess_timing_arr] uf_ms=%.1f events_ms=%.1f "
                 "sort_ms=%.1f adj_ms=%.1f compact_ms=%.1f n_segments=%zu "
                 "n_events=%zu\n",
                 uf_ms, events_ms, sort_ms, adj_ms, compact_ms,
                 geometry.domain_segments.size(), n_events);
  }
  rebuild_boundary_intervals(dt, geometry);
}

void rebuild_boundary_intervals(const DelaunayTriangulation& dt,
                                RestrictedGeometry& geometry) {
  static const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  double nbrs_ms = 0.0;
  double events2_ms = 0.0;
  double rsort_ms = 0.0;
  double mid_ms = 0.0;
  double near_ms = 0.0;
  std::chrono::steady_clock::time_point nbrs_start;
  if (timing_enabled) {
    nbrs_start = std::chrono::steady_clock::now();
  }
  const std::vector<std::vector<SiteId>> neighbors =
      delaunay_neighbors(dt);
  if (timing_enabled) {
    nbrs_ms = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - nbrs_start)
                  .count();
  }
  geometry.boundary_intervals.clear();
  SiteId previous_owner = kInvalidId;
  std::chrono::steady_clock::time_point events2_start;
  if (timing_enabled) {
    events2_start = std::chrono::steady_clock::now();
  }
  const std::vector<std::vector<RestrictedNodeId>> events_by_segment =
      collect_segment_events(geometry);
  if (timing_enabled) {
    events2_ms = std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - events2_start)
                     .count();
  }
  for (const DomainSegmentRecord& segment : geometry.domain_segments) {
    std::vector<RestrictedNodeId> events =
        events_by_segment[static_cast<std::size_t>(segment.id)];
    const HomogeneousPoint& start =
        geometry.nodes[segment.start_node].exact_point;
    const HomogeneousPoint& end =
        geometry.nodes[segment.end_node].exact_point;
    std::chrono::steady_clock::time_point rsort_start;
    if (timing_enabled) {
      rsort_start = std::chrono::steady_clock::now();
    }
    std::sort(events.begin(), events.end(),
              [&](const RestrictedNodeId lhs, const RestrictedNodeId rhs) {
      const int order = projection_compare_screened(
          geometry.nodes[lhs].exact_point,
          geometry.nodes[rhs].exact_point, start, end);
      if (order != 0) {
        return order < 0;
      }
      return geometry.nodes[lhs].key < geometry.nodes[rhs].key;
    });
    if (timing_enabled) {
      rsort_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - rsort_start)
                      .count();
    }
    events.erase(std::unique(events.begin(), events.end()), events.end());
    TENRYU_ASSERT(events.size() >= 2 &&
                      events.front() == segment.start_node &&
                      events.back() == segment.end_node,
                  "boundary_support_missing");
    for (std::size_t index = 1; index < events.size(); ++index) {
      const RestrictedNodeId first = events[index - 1];
      const RestrictedNodeId second = events[index];
      TENRYU_ASSERT(first != second &&
                        !bitwise_identical(geometry.nodes[first].constructed,
                                           geometry.nodes[second].constructed) &&
                        !same_exact_point_screened(
                            geometry.nodes[first].exact_point,
                            geometry.nodes[second].exact_point),
                    "boundary_interval_unowned");
      std::chrono::steady_clock::time_point mid_start;
      if (timing_enabled) {
        mid_start = std::chrono::steady_clock::now();
      }
      const HomogeneousPoint midpoint = exact_midpoint(
          geometry.nodes[first].exact_point,
          geometry.nodes[second].exact_point);
      if (timing_enabled) {
        mid_ms += std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - mid_start)
                      .count();
      }
      std::chrono::steady_clock::time_point near_start;
      if (timing_enabled) {
        near_start = std::chrono::steady_clock::now();
      }
      previous_owner = nearest_site(dt, neighbors, midpoint, previous_owner);
      if (timing_enabled) {
        near_ms += std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - near_start)
                       .count();
      }
      TENRYU_ASSERT(previous_owner != kInvalidId,
                    "boundary_interval_unowned");
      geometry.boundary_intervals.push_back(BoundaryInterval{
          segment.id, first, second, previous_owner});
    }
  }
  if (timing_enabled) {
    std::fprintf(stderr,
                 "[tess_timing_rbi] nbrs_ms=%.1f events2_ms=%.1f "
                 "rsort_ms=%.1f mid_ms=%.1f near_ms=%.1f "
                 "n_intervals=%zu\n",
                 nbrs_ms, events2_ms, rsort_ms, mid_ms, near_ms,
                 geometry.boundary_intervals.size());
  }
}

}  // namespace tenryu::mesh::tess
