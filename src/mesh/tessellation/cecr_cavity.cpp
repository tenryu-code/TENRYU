#include "mesh/tessellation/delaunay_2d.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {

struct CecrTimingData {
  double hull_ms = 0.0;
  double seed_ms = 0.0;
  double components_ms = 0.0;
  double d0_ms = 0.0;
  double d1_ms = 0.0;
  double d2_ms = 0.0;
  double d3_ms = 0.0;
  double d4_ms = 0.0;
  int growth_iterations = 0;
  double retriangulate_ms = 0.0;
  double splice_ms = 0.0;
  double audit_ms = 0.0;
};

class CecrStageTimer {
 public:
  explicit CecrStageTimer(double* elapsed_ms) : elapsed_ms_(elapsed_ms) {
    if (elapsed_ms_ != nullptr) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~CecrStageTimer() { stop(); }

  void stop() {
    if (elapsed_ms_ != nullptr) {
      *elapsed_ms_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - start_)
              .count();
      elapsed_ms_ = nullptr;
    }
  }

  CecrStageTimer(const CecrStageTimer&) = delete;
  CecrStageTimer& operator=(const CecrStageTimer&) = delete;

 private:
  double* elapsed_ms_;
  std::chrono::steady_clock::time_point start_{};
};

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change,
    std::vector<std::uint8_t> in_patch,
    CecrTimingData* timing);

namespace {

// Growth-round ceiling for the CECR cavity loop. Production p99 is ~23
// rounds; runaway growth (384 rounds / 10.2 s components observed in the
// 2026-08-16 kick smoke) is cheaper to serve as a cold rebuild via the
// existing static-fallback path, whose output is byte-identical by the
// warm==static contract.
inline constexpr int kCecrGrowthRoundCap = 64;

using FaceKey = std::array<std::uint64_t, 3>;

bool tess_axis_census_enabled() {
  static const bool enabled =
      std::getenv("TENRYU_TESS_AXIS") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_AXIS")) == std::string("1");
  return enabled;
}

class DisjointSet {
 public:
  explicit DisjointSet(const std::size_t size) : parent_(size), rank_(size, 0) {
    for (std::size_t index = 0; index < size; ++index) {
      parent_[index] = static_cast<int>(index);
    }
  }

  int find(const int value) {
    TENRYU_ASSERT(value >= 0 &&
                      value < static_cast<int>(parent_.size()),
                  "CECR disjoint-set index is invalid");
    if (parent_[value] != value) {
      parent_[value] = find(parent_[value]);
    }
    return parent_[value];
  }

  void unite(const int first, const int second) {
    int first_root = find(first);
    int second_root = find(second);
    if (first_root == second_root) {
      return;
    }
    if (rank_[first_root] < rank_[second_root]) {
      std::swap(first_root, second_root);
    }
    parent_[second_root] = first_root;
    if (rank_[first_root] == rank_[second_root]) {
      ++rank_[first_root];
    }
  }

 private:
  std::vector<int> parent_;
  std::vector<int> rank_;
};

std::array<HalfEdgeId, 3> triangle_edges(
    const DelaunayTriangulation& dt,
    const TriangleId triangle) {
  TENRYU_ASSERT(triangle >= 0 &&
                    triangle < static_cast<TriangleId>(dt.tri_edge.size()),
                "CECR triangle id is invalid");
  const HalfEdgeId first = dt.tri_edge[triangle];
  TENRYU_ASSERT(first >= 0 &&
                    first < static_cast<HalfEdgeId>(dt.he_origin.size()),
                "CECR triangle edge is invalid");
  const HalfEdgeId second = dt.he_next[first];
  TENRYU_ASSERT(second >= 0 &&
                    second < static_cast<HalfEdgeId>(dt.he_origin.size()),
                "CECR triangle cycle is invalid");
  const HalfEdgeId third = dt.he_next[second];
  TENRYU_ASSERT(third >= 0 &&
                    third < static_cast<HalfEdgeId>(dt.he_origin.size()) &&
                    dt.he_next[third] == first,
                "CECR triangle cycle is invalid");
  return {{first, second, third}};
}

std::array<SiteId, 3> triangle_vertices(
    const DelaunayTriangulation& dt,
    const TriangleId triangle) {
  const std::array<HalfEdgeId, 3> edges = triangle_edges(dt, triangle);
  return {{dt.he_origin[edges[0]], dt.he_origin[edges[1]],
           dt.he_origin[edges[2]]}};
}

SiteId edge_destination(const DelaunayTriangulation& dt,
                        const HalfEdgeId edge) {
  return dt.he_origin[dt.he_next[edge]];
}

FaceKey face_key(const DelaunayTriangulation& dt,
                 const TriangleId triangle) {
  const std::array<SiteId, 3> vertices = triangle_vertices(dt, triangle);
  FaceKey key{{dt.sites[vertices[0]].stable_id,
               dt.sites[vertices[1]].stable_id,
               dt.sites[vertices[2]].stable_id}};
  std::sort(key.begin(), key.end());
  return key;
}

bool face_key_less(const DelaunayTriangulation& dt,
                   const TriangleId first,
                   const TriangleId second) {
  const FaceKey first_key = face_key(dt, first);
  const FaceKey second_key = face_key(dt, second);
  return first_key < second_key ||
         (first_key == second_key && first < second);
}

void add_closed_vertex_star(const DelaunayTriangulation& dt,
                            const std::vector<std::uint8_t>& vertices,
                            std::vector<std::uint8_t>& in_patch) {
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const std::array<SiteId, 3> sites = triangle_vertices(dt, triangle);
    if (vertices[sites[0]] != 0 || vertices[sites[1]] != 0 ||
        vertices[sites[2]] != 0) {
      in_patch[triangle] = 1;
    }
  }
}

bool old_edge_is_exact_cocircular(const DelaunayTriangulation& previous,
                                  const HalfEdgeId edge) {
  const HalfEdgeId twin = previous.he_twin[edge];
  if (twin == kInvalidId) {
    return false;
  }
  const HalfEdgeId next = previous.he_next[edge];
  const HalfEdgeId opposite = previous.he_next[next];
  const HalfEdgeId twin_opposite =
      previous.he_next[previous.he_next[twin]];
  const SiteId a = previous.he_origin[edge];
  const SiteId b = previous.he_origin[next];
  const SiteId c = previous.he_origin[opposite];
  const SiteId d = previous.he_origin[twin_opposite];
  return incircle_sign_forced_exact(previous.sites[a], previous.sites[b],
                                    previous.sites[c], previous.sites[d]) == 0;
}

struct FrontierEdge {
  HalfEdgeId half_edge = kInvalidId;
  SiteId origin = kInvalidId;
  SiteId destination = kInvalidId;
};

std::vector<FrontierEdge> component_frontier(
    const DelaunayTriangulation& dt,
    const std::vector<int>& component_of_triangle,
    const int component) {
  std::vector<FrontierEdge> frontier;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (component_of_triangle[triangle] != component) {
      continue;
    }
    for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
      const HalfEdgeId twin = dt.he_twin[edge];
      if (twin != kInvalidId &&
          component_of_triangle[dt.he_face[twin]] == component) {
        continue;
      }
      frontier.push_back(
          FrontierEdge{edge, dt.he_origin[edge], edge_destination(dt, edge)});
    }
  }
  return frontier;
}

std::vector<FrontierEdge> component_frontier_from_list(
    const DelaunayTriangulation& dt,
    const std::vector<int>& component_of_triangle,
    const int component,
    const TriangleId* triangles,
    const std::size_t count) {
  std::vector<FrontierEdge> frontier;
  for (std::size_t index = 0; index < count; ++index) {
    const TriangleId triangle = triangles[index];
    for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
      const HalfEdgeId twin = dt.he_twin[edge];
      if (twin != kInvalidId &&
          component_of_triangle[dt.he_face[twin]] == component) {
        continue;
      }
      frontier.push_back(
          FrontierEdge{edge, dt.he_origin[edge], edge_destination(dt, edge)});
    }
  }
  return frontier;
}

bool point_on_segment(const Site& point,
                      const Site& first,
                      const Site& second) {
  if (point.r < std::min(first.r, second.r) ||
      point.r > std::max(first.r, second.r) ||
      point.z < std::min(first.z, second.z) ||
      point.z > std::max(first.z, second.z)) {
    return false;
  }
  return orient2d_sign_tess(first, second, point) == 0;
}

bool proper_segment_intersection(const Site& a,
                                 const Site& b,
                                 const Site& c,
                                 const Site& d) {
  const int abc = orient2d_sign_tess(a, b, c);
  const int abd = orient2d_sign_tess(a, b, d);
  const int cda = orient2d_sign_tess(c, d, a);
  const int cdb = orient2d_sign_tess(c, d, b);
  return abc != 0 && abd != 0 && cda != 0 && cdb != 0 &&
         abc != abd && cda != cdb;
}

struct ComponentKey {
  std::uint64_t minimum_vertex = std::numeric_limits<std::uint64_t>::max();
  FaceKey minimum_face{{std::numeric_limits<std::uint64_t>::max(),
                        std::numeric_limits<std::uint64_t>::max(),
                        std::numeric_limits<std::uint64_t>::max()}};

  bool operator<(const ComponentKey& other) const {
    return std::tie(minimum_vertex, minimum_face) <
           std::tie(other.minimum_vertex, other.minimum_face);
  }
};

ComponentKey component_key(const DelaunayTriangulation& dt,
                           const std::vector<int>& labels,
                           const int label) {
  ComponentKey key;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (labels[triangle] != label) {
      continue;
    }
    const FaceKey triangle_key = face_key(dt, triangle);
    key.minimum_face = std::min(key.minimum_face, triangle_key);
    for (const SiteId vertex : triangle_vertices(dt, triangle)) {
      key.minimum_vertex =
          std::min(key.minimum_vertex, dt.sites[vertex].stable_id);
    }
  }
  return key;
}

int normalize_component_labels(const DelaunayTriangulation& dt,
                               std::vector<int>& labels) {
  int maximum_label = -1;
  for (const int label : labels) {
    maximum_label = std::max(maximum_label, label);
  }
  if (maximum_label < 0) {
    return 0;
  }

  DisjointSet merged(static_cast<std::size_t>(maximum_label + 1));
  std::vector<int> first_component_at_vertex(dt.sites.size(), -1);
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const int label = labels[triangle];
    if (label < 0) {
      continue;
    }
    for (const SiteId vertex : triangle_vertices(dt, triangle)) {
      int& first = first_component_at_vertex[vertex];
      if (first < 0) {
        first = label;
      } else {
        merged.unite(first, label);
      }
    }
  }

  for (int& label : labels) {
    if (label >= 0) {
      label = merged.find(label);
    }
  }

  std::map<int, ComponentKey> keys;
  for (const int label : labels) {
    if (label >= 0 && keys.find(label) == keys.end()) {
      keys.emplace(label, component_key(dt, labels, label));
    }
  }
  std::vector<std::pair<ComponentKey, int>> ordered;
  ordered.reserve(keys.size());
  for (const auto& entry : keys) {
    ordered.emplace_back(entry.second, entry.first);
  }
  std::sort(ordered.begin(), ordered.end());

  std::map<int, int> canonical_label;
  for (std::size_t index = 0; index < ordered.size(); ++index) {
    canonical_label.emplace(ordered[index].second,
                            static_cast<int>(index));
  }
  for (int& label : labels) {
    if (label >= 0) {
      label = canonical_label.at(label);
    }
  }
  return static_cast<int>(ordered.size());
}

bool patch_touches_hull(const DelaunayTriangulation& dt,
                        const std::vector<std::uint8_t>& in_patch) {
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (in_patch[triangle] == 0) {
      continue;
    }
    for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
      if (dt.he_twin[edge] == kInvalidId) {
        return true;
      }
    }
  }
  return false;
}

bool fill_first_enclosed_complement(
    const DelaunayTriangulation& dt,
    const int component,
    std::vector<int>& labels,
    std::vector<std::uint8_t>& in_patch,
    std::vector<std::uint32_t>& visited,
    std::uint32_t& epoch) {
  ++epoch;
  for (TriangleId seed = 0;
       seed < static_cast<TriangleId>(dt.tri_edge.size()); ++seed) {
    if (labels[seed] == component || visited[seed] == epoch) {
      continue;
    }
    std::vector<TriangleId> region;
    std::queue<TriangleId> pending;
    bool touches_hull = false;
    visited[seed] = epoch;
    pending.push(seed);
    while (!pending.empty()) {
      const TriangleId triangle = pending.front();
      pending.pop();
      region.push_back(triangle);
      for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
        const HalfEdgeId twin = dt.he_twin[edge];
        if (twin == kInvalidId) {
          touches_hull = true;
          continue;
        }
        const TriangleId neighbor = dt.he_face[twin];
        if (labels[neighbor] == component || visited[neighbor] == epoch) {
          continue;
        }
        visited[neighbor] = epoch;
        pending.push(neighbor);
      }
    }
    if (touches_hull) {
      continue;
    }

    std::set<int> absorbed_components;
    for (const TriangleId triangle : region) {
      if (labels[triangle] >= 0 && labels[triangle] != component) {
        absorbed_components.insert(labels[triangle]);
      }
      labels[triangle] = component;
      in_patch[triangle] = 1;
    }
    if (!absorbed_components.empty()) {
      for (int& label : labels) {
        if (absorbed_components.count(label) != 0) {
          label = component;
        }
      }
    }
    return true;
  }
  return false;
}

struct FrontierTopology {
  bool single_cycle = false;
  std::vector<SiteId> pinch_vertices;
};

FrontierTopology analyze_frontier(const DelaunayTriangulation& dt,
                                  const std::vector<FrontierEdge>& frontier) {
  FrontierTopology result;
  if (frontier.empty()) {
    return result;
  }
  std::vector<int> incoming(dt.sites.size(), 0);
  std::vector<int> outgoing(dt.sites.size(), 0);
  std::vector<HalfEdgeId> outgoing_edge(dt.sites.size(), kInvalidId);
  for (const FrontierEdge& edge : frontier) {
    ++outgoing[edge.origin];
    ++incoming[edge.destination];
    outgoing_edge[edge.origin] = edge.half_edge;
  }
  for (SiteId vertex = 0;
       vertex < static_cast<SiteId>(dt.sites.size()); ++vertex) {
    if ((incoming[vertex] != 0 || outgoing[vertex] != 0) &&
        (incoming[vertex] != 1 || outgoing[vertex] != 1)) {
      result.pinch_vertices.push_back(vertex);
    }
  }
  std::sort(result.pinch_vertices.begin(), result.pinch_vertices.end(),
            [&dt](const SiteId first, const SiteId second) {
              return dt.sites[first].stable_id < dt.sites[second].stable_id;
            });
  if (!result.pinch_vertices.empty()) {
    return result;
  }

  const SiteId start = std::min_element(
      frontier.begin(), frontier.end(),
      [&dt](const FrontierEdge& first, const FrontierEdge& second) {
        return dt.sites[first.origin].stable_id <
               dt.sites[second.origin].stable_id;
      })->origin;
  SiteId current = start;
  std::set<HalfEdgeId> visited_edges;
  while (true) {
    const HalfEdgeId edge = outgoing_edge[current];
    if (edge == kInvalidId || !visited_edges.insert(edge).second) {
      break;
    }
    current = edge_destination(dt, edge);
    if (current == start) {
      break;
    }
  }
  result.single_cycle = current == start &&
                        visited_edges.size() == frontier.size();
  return result;
}

void assert_disk_euler_characteristic(const DelaunayTriangulation& dt,
                                      const TriangleId* triangles,
                                      const std::size_t count) {
  std::vector<SiteId> vertices;
  std::vector<std::pair<SiteId, SiteId>> edges;
  const std::int64_t faces = static_cast<std::int64_t>(count);
  for (std::size_t index = 0; index < count; ++index) {
    const TriangleId triangle = triangles[index];
    for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
      const SiteId origin = dt.he_origin[edge];
      const SiteId destination = edge_destination(dt, edge);
      vertices.push_back(origin);
      edges.emplace_back(std::min(origin, destination),
                         std::max(origin, destination));
    }
  }
  std::sort(vertices.begin(), vertices.end());
  vertices.erase(std::unique(vertices.begin(), vertices.end()),
                 vertices.end());
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
  const std::int64_t euler = static_cast<std::int64_t>(vertices.size()) -
                             static_cast<std::int64_t>(edges.size()) + faces;
  TENRYU_ASSERT(euler == 1,
                "CECR disk frontier failed the Euler cross-check");
}

TriangleId smallest_outside_face_at_vertex(
    const DelaunayTriangulation& dt,
    const std::vector<int>& labels,
    const int component,
    const SiteId vertex,
    const std::vector<std::size_t>& site_face_offsets,
    const std::vector<TriangleId>& site_faces) {
  TriangleId best = kInvalidId;
  for (std::size_t index = site_face_offsets[vertex];
       index < site_face_offsets[vertex + 1]; ++index) {
    const TriangleId face = site_faces[index];
    if (labels[face] == component) {
      continue;
    }
    if (best == kInvalidId || face_key_less(dt, face, best)) {
      best = face;
    }
  }
  return best;
}

struct StableEdgeKey {
  std::uint64_t lo;
  std::uint64_t hi;

  bool operator<(const StableEdgeKey& other) const {
    return lo < other.lo || (lo == other.lo && hi < other.hi);
  }

  bool operator==(const StableEdgeKey& other) const {
    return lo == other.lo && hi == other.hi;
  }
};

StableEdgeKey stable_edge_key(const DelaunayTriangulation& dt,
                              const SiteId first,
                              const SiteId second) {
  const std::uint64_t first_id = dt.sites[first].stable_id;
  const std::uint64_t second_id = dt.sites[second].stable_id;
  return StableEdgeKey{std::min(first_id, second_id),
                       std::max(first_id, second_id)};
}

struct OrderedFrontier {
  std::vector<FrontierEdge> edges;
  std::vector<SiteId> cycle;
  bool ok = false;
};

struct SubstitutedPolygon {
  std::vector<SiteId> cycle;
  std::vector<FrontierEdge> edges;
};

OrderedFrontier order_frontier(
    const DelaunayTriangulation& dt,
    const std::vector<FrontierEdge>& frontier) {
  OrderedFrontier result;
  if (frontier.size() < 3) {
    return result;
  }
  std::map<SiteId, FrontierEdge> outgoing;
  std::set<SiteId> incoming;
  for (const FrontierEdge& edge : frontier) {
    if (!outgoing.emplace(edge.origin, edge).second ||
        !incoming.insert(edge.destination).second) {
      return result;
    }
  }
  if (outgoing.size() != incoming.size()) {
    return result;
  }
  const SiteId start = std::min_element(
      frontier.begin(), frontier.end(),
      [&dt](const FrontierEdge& first, const FrontierEdge& second) {
        return dt.sites[first.origin].stable_id <
               dt.sites[second.origin].stable_id;
      })->origin;
  SiteId current = start;
  std::set<HalfEdgeId> visited;
  do {
    const auto next = outgoing.find(current);
    if (next == outgoing.end() ||
        !visited.insert(next->second.half_edge).second) {
      return result;
    }
    result.cycle.push_back(current);
    result.edges.push_back(next->second);
    current = next->second.destination;
  } while (current != start);
  if (visited.size() != frontier.size()) {
    return result;
  }
  result.ok = true;
  return result;
}

enum class PointPolygonLocation {
  kOutside,
  kBoundary,
  kInside,
};

PointPolygonLocation classify_point_in_polygon(
    const DelaunayTriangulation& dt,
    const SiteId point_id,
    const std::vector<SiteId>& cycle) {
  const Site& point = dt.sites[point_id];
  int winding_number = 0;
  for (std::size_t index = 0; index < cycle.size(); ++index) {
    const Site& first = dt.sites[cycle[index]];
    const Site& second = dt.sites[cycle[(index + 1) % cycle.size()]];
    if (point_on_segment(point, first, second)) {
      return PointPolygonLocation::kBoundary;
    }
    if (first.z <= point.z && second.z > point.z) {
      if (orient2d_sign_tess(first, second, point) > 0) {
        ++winding_number;
      }
    } else if (second.z <= point.z && first.z > point.z) {
      if (orient2d_sign_tess(first, second, point) < 0) {
        --winding_number;
      }
    }
  }
  return winding_number == 0 ? PointPolygonLocation::kOutside
                             : PointPolygonLocation::kInside;
}

bool segment_bboxes_overlap(const Site& a,
                            const Site& b,
                            const Site& c,
                            const Site& d) {
  return std::max(std::min(a.r, b.r), std::min(c.r, d.r)) <=
             std::min(std::max(a.r, b.r), std::max(c.r, d.r)) &&
         std::max(std::min(a.z, b.z), std::min(c.z, d.z)) <=
             std::min(std::max(a.z, b.z), std::max(c.z, d.z));
}

bool segments_intersect_exact(const Site& a,
                              const Site& b,
                              const Site& c,
                              const Site& d) {
  if (!segment_bboxes_overlap(a, b, c, d)) {
    return false;
  }
  if (proper_segment_intersection(a, b, c, d)) {
    return true;
  }
  return point_on_segment(a, c, d) || point_on_segment(b, c, d) ||
         point_on_segment(c, a, b) || point_on_segment(d, a, b);
}

Expansion exact_triangle_area(const Site& a,
                              const Site& b,
                              const Site& c) {
  return expansion_cross(exact_difference(b.r, a.r),
                         exact_difference(b.z, a.z),
                         exact_difference(c.r, a.r),
                         exact_difference(c.z, a.z));
}

Expansion exact_polygon_area(const DelaunayTriangulation& dt,
                             const std::vector<SiteId>& cycle) {
  Expansion area;
  for (std::size_t index = 0; index < cycle.size(); ++index) {
    const Site& first = dt.sites[cycle[index]];
    const Site& second = dt.sites[cycle[(index + 1) % cycle.size()]];
    area = expansion_sum(
        area,
        expansion_difference(
            expansion_product(Expansion{first.r}, Expansion{second.z}),
            expansion_product(Expansion{second.r}, Expansion{first.z})));
  }
  return area;
}

CecrComponents edge_connected_components(
    const DelaunayTriangulation& dt,
    const std::vector<std::uint8_t>& in_patch) {
  CecrComponents result;
  result.component_of_triangle.assign(dt.tri_edge.size(), -1);
  std::vector<std::vector<TriangleId>> faces;
  for (TriangleId seed = 0;
       seed < static_cast<TriangleId>(dt.tri_edge.size()); ++seed) {
    if (in_patch[seed] == 0 || result.component_of_triangle[seed] >= 0) {
      continue;
    }
    const int label = static_cast<int>(faces.size());
    faces.emplace_back();
    std::queue<TriangleId> pending;
    pending.push(seed);
    result.component_of_triangle[seed] = label;
    while (!pending.empty()) {
      const TriangleId triangle = pending.front();
      pending.pop();
      faces.back().push_back(triangle);
      for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
        const HalfEdgeId twin = dt.he_twin[edge];
        if (twin == kInvalidId) {
          continue;
        }
        const TriangleId neighbor = dt.he_face[twin];
        if (in_patch[neighbor] == 0 ||
            result.component_of_triangle[neighbor] >= 0) {
          continue;
        }
        result.component_of_triangle[neighbor] = label;
        pending.push(neighbor);
      }
    }
  }

  if (!faces.empty()) {
    DisjointSet touching(faces.size());
    std::vector<int> first_component(dt.sites.size(), -1);
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
      const int label = result.component_of_triangle[triangle];
      if (label < 0) {
        continue;
      }
      for (const SiteId site : triangle_vertices(dt, triangle)) {
        if (first_component[site] < 0) {
          first_component[site] = label;
        } else {
          touching.unite(first_component[site], label);
        }
      }
    }
    for (int& label : result.component_of_triangle) {
      if (label >= 0) {
        label = touching.find(label);
      }
    }
  }

  std::map<int, FaceKey> minimum_keys;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    const int label = result.component_of_triangle[triangle];
    if (label < 0) {
      continue;
    }
    const auto inserted = minimum_keys.emplace(label, face_key(dt, triangle));
    if (!inserted.second) {
      inserted.first->second =
          std::min(inserted.first->second, face_key(dt, triangle));
    }
  }
  std::vector<std::pair<FaceKey, int>> ordered;
  ordered.reserve(minimum_keys.size());
  for (const auto& entry : minimum_keys) {
    ordered.emplace_back(entry.second, entry.first);
  }
  std::sort(ordered.begin(), ordered.end());
  std::vector<int> canonical(faces.size(), -1);
  for (std::size_t index = 0; index < ordered.size(); ++index) {
    canonical[ordered[index].second] = static_cast<int>(index);
  }
  for (int& label : result.component_of_triangle) {
    if (label >= 0) {
      label = canonical[label];
    }
  }
  result.component_count = static_cast<int>(ordered.size());
  return result;
}

std::vector<SiteId> component_sites(
    const DelaunayTriangulation& dt,
    const std::vector<int>& labels,
    const int component) {
  std::vector<SiteId> sites;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (labels[triangle] != component) {
      continue;
    }
    for (const SiteId site : triangle_vertices(dt, triangle)) {
      sites.push_back(site);
    }
  }
  std::sort(sites.begin(), sites.end(),
            [&dt](const SiteId first, const SiteId second) {
              return dt.sites[first].stable_id < dt.sites[second].stable_id;
            });
  sites.erase(std::unique(sites.begin(), sites.end()), sites.end());
  return sites;
}

TriangleId smallest_outside_face_at_edge(
    const DelaunayTriangulation& dt,
    const std::vector<std::uint8_t>& in_patch,
    const SiteId first,
    const SiteId second) {
  TriangleId best = kInvalidId;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (in_patch[triangle] != 0) {
      continue;
    }
    for (const HalfEdgeId edge : triangle_edges(dt, triangle)) {
      const SiteId origin = dt.he_origin[edge];
      const SiteId destination = edge_destination(dt, edge);
      if (!((origin == first && destination == second) ||
            (origin == second && destination == first))) {
        continue;
      }
      if (best == kInvalidId || face_key_less(dt, triangle, best)) {
        best = triangle;
      }
    }
  }
  return best;
}

TriangleId smallest_unpatched_face_at_site(
    const DelaunayTriangulation& dt,
    const std::vector<std::uint8_t>& in_patch,
    const SiteId site) {
  TriangleId best = kInvalidId;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(dt.tri_edge.size()); ++triangle) {
    if (in_patch[triangle] != 0) {
      continue;
    }
    const std::array<SiteId, 3> vertices =
        triangle_vertices(dt, triangle);
    if (vertices[0] != site && vertices[1] != site && vertices[2] != site) {
      continue;
    }
    if (best == kInvalidId || face_key_less(dt, triangle, best)) {
      best = triangle;
    }
  }
  return best;
}

TriangleId growth_face_for_polygon_edge(
    const DelaunayTriangulation& dt,
    const std::vector<std::uint8_t>& in_patch,
    const FrontierEdge& edge) {
  if (edge.half_edge != kInvalidId) {
    TENRYU_ASSERT(
        edge.half_edge >= 0 &&
            edge.half_edge < static_cast<HalfEdgeId>(dt.he_origin.size()) &&
            dt.he_origin[edge.half_edge] == edge.origin &&
            edge_destination(dt, edge.half_edge) == edge.destination,
        "CECR polygon edge half-edge does not match its sites");
    return smallest_outside_face_at_edge(
        dt, in_patch, edge.origin, edge.destination);
  }

  std::array<SiteId, 2> witnesses{{edge.origin, edge.destination}};
  std::sort(witnesses.begin(), witnesses.end(),
            [&dt](const SiteId first, const SiteId second) {
              return dt.sites[first].stable_id < dt.sites[second].stable_id;
            });
  return smallest_unpatched_face_at_site(dt, in_patch, witnesses.front());
}

std::vector<SiteId> triangulation_hull_cycle_for_cecr(
    const DelaunayTriangulation& dt) {
  std::map<SiteId, SiteId> next_site;
  std::set<SiteId> incoming;
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(dt.he_origin.size()); ++edge) {
    if (dt.he_twin[edge] != kInvalidId) {
      continue;
    }
    const SiteId origin = dt.he_origin[edge];
    const SiteId destination = edge_destination(dt, edge);
    TENRYU_ASSERT(next_site.emplace(origin, destination).second,
                  "CECR hull has duplicate outgoing edges");
    TENRYU_ASSERT(incoming.insert(destination).second,
                  "CECR hull has duplicate incoming edges");
  }
  TENRYU_ASSERT(!next_site.empty() && next_site.size() == incoming.size(),
                "CECR hull cycle is invalid");
  const SiteId start = next_site.begin()->first;
  SiteId current = start;
  std::vector<SiteId> cycle;
  do {
    TENRYU_ASSERT(cycle.size() < next_site.size(),
                  "CECR hull cycle does not close");
    cycle.push_back(current);
    const auto next = next_site.find(current);
    TENRYU_ASSERT(next != next_site.end(),
                  "CECR hull cycle is missing an edge");
    current = next->second;
  } while (current != start);
  TENRYU_ASSERT(cycle.size() == next_site.size(),
                "CECR hull contains multiple cycles");
  return cycle;
}

bool stable_cycles_equal(const DelaunayTriangulation& dt,
                         const std::vector<SiteId>& first,
                         const std::vector<SiteId>& second) {
  if (first.size() != second.size() || first.empty()) {
    return false;
  }
  const std::uint64_t first_id = dt.sites[first.front()].stable_id;
  const auto start = std::find_if(
      second.begin(), second.end(),
      [&dt, first_id](const SiteId site) {
        return dt.sites[site].stable_id == first_id;
      });
  if (start == second.end()) {
    return false;
  }
  const std::size_t offset = static_cast<std::size_t>(start - second.begin());
  for (std::size_t index = 0; index < first.size(); ++index) {
    if (dt.sites[first[index]].stable_id !=
        dt.sites[second[(offset + index) % second.size()]].stable_id) {
      return false;
    }
  }
  return true;
}

struct HullArc {
  int component = -1;
  SiteId first_anchor = kInvalidId;
  SiteId second_anchor = kInvalidId;
  std::vector<SiteId> new_span;
};

SubstitutedPolygon substitute_hull_arcs(
    const DelaunayTriangulation& moved,
    const OrderedFrontier& frontier,
    const std::vector<HullArc>& arcs,
    const int component) {
  if (!frontier.ok) {
    return {};
  }
  std::vector<const HullArc*> component_arcs;
  for (const HullArc& arc : arcs) {
    if (arc.component == component) {
      component_arcs.push_back(&arc);
    }
  }
  if (component_arcs.empty()) {
    return SubstitutedPolygon{frontier.cycle, frontier.edges};
  }

  TENRYU_ASSERT(frontier.edges.size() == frontier.cycle.size(),
                "CECR substituted polygon requires an ordered frontier");
  std::map<SiteId, const HullArc*> arc_at_first_anchor;
  for (const HullArc* const arc : component_arcs) {
    TENRYU_ASSERT(arc->new_span.size() >= 2 &&
                      arc->new_span.front() == arc->first_anchor &&
                      arc->new_span.back() == arc->second_anchor &&
                      arc_at_first_anchor.emplace(arc->first_anchor,
                                                  arc).second,
                  "CECR substituted polygon has an invalid hull arc");
  }

  std::size_t start = frontier.edges.size();
  for (std::size_t index = 0; index < frontier.edges.size(); ++index) {
    const FrontierEdge& edge = frontier.edges[index];
    TENRYU_ASSERT(edge.half_edge >= 0 &&
                      edge.half_edge <
                          static_cast<HalfEdgeId>(moved.he_twin.size()),
                  "CECR frontier edge has an invalid half-edge");
    if (moved.he_twin[edge.half_edge] != kInvalidId) {
      start = index;
      break;
    }
  }
  TENRYU_ASSERT(start < frontier.edges.size(),
                "CECR boundary cavity cannot substitute the entire hull");

  SubstitutedPolygon result;
  std::set<const HullArc*> used_arcs;
  std::size_t consumed = 0;
  std::size_t index = start;
  while (consumed < frontier.edges.size()) {
    const FrontierEdge& edge = frontier.edges[index];
    const auto arc_entry = arc_at_first_anchor.find(edge.origin);
    if (moved.he_twin[edge.half_edge] == kInvalidId) {
      TENRYU_ASSERT(arc_entry != arc_at_first_anchor.end(),
                    "CECR hull run is missing its admitted arc");
      const HullArc& arc = *arc_entry->second;
      TENRYU_ASSERT(used_arcs.insert(&arc).second,
                    "CECR admitted hull arc is used more than once");
      for (std::size_t span_index = 0;
           span_index + 1 < arc.new_span.size(); ++span_index) {
        const SiteId origin = arc.new_span[span_index];
        const SiteId destination = arc.new_span[span_index + 1];
        result.cycle.push_back(origin);
        result.edges.push_back(
            FrontierEdge{kInvalidId, origin, destination});
      }
      SiteId destination = edge.origin;
      do {
        TENRYU_ASSERT(consumed < frontier.edges.size(),
                      "CECR admitted hull arc does not terminate");
        const FrontierEdge& old_span_edge = frontier.edges[index];
        TENRYU_ASSERT(
            moved.he_twin[old_span_edge.half_edge] == kInvalidId,
            "CECR admitted hull arc contains an interior frontier edge");
        destination = old_span_edge.destination;
        index = (index + 1) % frontier.edges.size();
        ++consumed;
      } while (destination != arc.second_anchor);
      continue;
    }

    TENRYU_ASSERT(arc_entry == arc_at_first_anchor.end(),
                  "CECR admitted hull arc starts on an interior edge");
    result.cycle.push_back(edge.origin);
    result.edges.push_back(edge);
    index = (index + 1) % frontier.edges.size();
    ++consumed;
  }

  TENRYU_ASSERT(result.cycle.size() == result.edges.size() &&
                    result.cycle.size() >= 3 &&
                    used_arcs.size() == component_arcs.size(),
                "CECR substituted polygon is not a cycle");
  for (std::size_t edge_index = 0; edge_index < result.edges.size();
       ++edge_index) {
    TENRYU_ASSERT(
        result.edges[edge_index].origin == result.cycle[edge_index] &&
            result.edges[edge_index].destination ==
                result.cycle[(edge_index + 1) % result.cycle.size()],
        "CECR substituted polygon edge order is invalid");
  }
  return result;
}

struct D0AdmissionResult {
  std::vector<HullArc> arcs;
  std::size_t boundary_component_count = 0;
  std::uint64_t minimum_boundary_stable_id =
      std::numeric_limits<std::uint64_t>::max();
  std::uint64_t failure_witness_stable_id =
      std::numeric_limits<std::uint64_t>::max();
  SiteId failure_witness = kInvalidId;
  int failure_component = -1;
  bool has_boundary_component = false;
  std::size_t axis_only_components = 0;
  std::size_t outer_only_components = 0;
  std::size_t mixed_components = 0;
  bool ok = true;
};

D0AdmissionResult certify_hull_frontier_admission(
    const DelaunayTriangulation& moved,
    const CecrComponents& components,
    const std::vector<std::uint8_t>& in_patch,
    const CecrHullChange& hull_change,
    const double axis_band_r) {
  D0AdmissionResult result;
  result.has_boundary_component = patch_touches_hull(moved, in_patch);
  if (!result.has_boundary_component) {
    return result;
  }

  std::vector<ComponentKey> component_keys(
      static_cast<std::size_t>(components.component_count));
  for (int component = 0; component < components.component_count;
       ++component) {
    component_keys[component] = component_key(
        moved, components.component_of_triangle, component);
  }
  const auto record_witness =
      [&result, &moved, &component_keys](const SiteId site,
                                         const int component) {
    TENRYU_ASSERT(site >= 0 &&
                      site < static_cast<SiteId>(moved.sites.size()),
                  "CECR D0 witness site is invalid");
    TENRYU_ASSERT(component >= 0 &&
                      component < static_cast<int>(component_keys.size()),
                  "CECR D0 witness component is invalid");
    result.ok = false;
    const std::uint64_t stable_id = moved.sites[site].stable_id;
    if (stable_id < result.failure_witness_stable_id ||
        (stable_id == result.failure_witness_stable_id &&
         (result.failure_component < 0 ||
          component_keys[component] <
              component_keys[result.failure_component]))) {
      result.failure_witness_stable_id = stable_id;
      result.failure_witness = site;
      result.failure_component = component;
    }
  };

  std::map<std::pair<std::uint64_t, std::uint64_t>, std::size_t>
      old_edge_index;
  for (std::size_t index = 0; index < hull_change.h_old.size(); ++index) {
    const SiteId first = hull_change.h_old[index];
    const SiteId second =
        hull_change.h_old[(index + 1) % hull_change.h_old.size()];
    old_edge_index.emplace(
        std::make_pair(moved.sites[first].stable_id,
                       moved.sites[second].stable_id),
        index);
  }

  std::map<std::uint64_t, std::size_t> new_hull_position;
  for (std::size_t index = 0; index < hull_change.h_new.size(); ++index) {
    new_hull_position.emplace(
        moved.sites[hull_change.h_new[index]].stable_id, index);
  }

  for (int component = 0; component < components.component_count;
       ++component) {
    std::vector<std::uint8_t> owned_old_edges(hull_change.h_old.size(), 0);
    bool component_touches_hull = false;
    std::size_t component_axis_edges = 0;
    std::size_t component_outer_edges = 0;
    for (const FrontierEdge& edge : component_frontier(
             moved, components.component_of_triangle, component)) {
      if (moved.he_twin[edge.half_edge] != kInvalidId) {
        continue;
      }
      component_touches_hull = true;
      if (tess_axis_census_enabled()) {
        const double edge_r = std::max(moved.sites[edge.origin].r,
                                       moved.sites[edge.destination].r);
        if (edge_r < axis_band_r) {
          ++component_axis_edges;
        } else {
          ++component_outer_edges;
        }
      }
      result.minimum_boundary_stable_id = std::min(
          result.minimum_boundary_stable_id,
          std::min(moved.sites[edge.origin].stable_id,
                   moved.sites[edge.destination].stable_id));
      const auto old_edge = old_edge_index.find(std::make_pair(
          moved.sites[edge.origin].stable_id,
          moved.sites[edge.destination].stable_id));
      if (old_edge == old_edge_index.end()) {
        record_witness(edge.origin, component);
        record_witness(edge.destination, component);
        continue;
      }
      owned_old_edges[old_edge->second] = 1;
    }
    if (!component_touches_hull) {
      continue;
    }
    ++result.boundary_component_count;
    if (tess_axis_census_enabled()) {
      if (component_axis_edges > 0 && component_outer_edges == 0) {
        ++result.axis_only_components;
      } else if (component_axis_edges == 0 && component_outer_edges > 0) {
        ++result.outer_only_components;
      } else {
        ++result.mixed_components;
      }
    }

    const std::size_t owned_count = static_cast<std::size_t>(std::count(
        owned_old_edges.begin(), owned_old_edges.end(), std::uint8_t{1}));
    if (owned_count == hull_change.h_old.size()) {
      for (const SiteId site : hull_change.h_old) {
        record_witness(site, component);
      }
      continue;
    }

    std::vector<std::pair<std::size_t, std::size_t>> runs;
    for (std::size_t index = 0; index < owned_old_edges.size();) {
      if (owned_old_edges[index] == 0) {
        ++index;
        continue;
      }
      const std::size_t start = index;
      while (index < owned_old_edges.size() && owned_old_edges[index] != 0) {
        ++index;
      }
      runs.emplace_back(start, index - start);
    }
    if (runs.size() > 1 && owned_old_edges.front() != 0 &&
        owned_old_edges.back() != 0) {
      const std::pair<std::size_t, std::size_t> last = runs.back();
      runs.pop_back();
      runs.front().first = last.first;
      runs.front().second += last.second;
    }

    const std::vector<SiteId> sites = component_sites(
        moved, components.component_of_triangle, component);
    std::set<std::uint64_t> component_stable_ids;
    for (const SiteId site : sites) {
      component_stable_ids.insert(moved.sites[site].stable_id);
    }

    for (const auto& run : runs) {
      HullArc arc;
      arc.component = component;
      arc.first_anchor = hull_change.h_old[run.first];
      arc.second_anchor = hull_change.h_old[
          (run.first + run.second) % hull_change.h_old.size()];
      const std::uint64_t first_id =
          moved.sites[arc.first_anchor].stable_id;
      const std::uint64_t second_id =
          moved.sites[arc.second_anchor].stable_id;
      const auto first_position = new_hull_position.find(first_id);
      const auto second_position = new_hull_position.find(second_id);
      if (first_position == new_hull_position.end()) {
        record_witness(arc.first_anchor, component);
      }
      if (second_position == new_hull_position.end()) {
        record_witness(arc.second_anchor, component);
      }
      if (first_position == new_hull_position.end() ||
          second_position == new_hull_position.end()) {
        result.arcs.push_back(std::move(arc));
        continue;
      }

      arc.new_span.push_back(arc.first_anchor);
      std::size_t position = first_position->second;
      bool found_second_anchor = false;
      for (std::size_t step = 1; step < hull_change.h_new.size(); ++step) {
        position = (position + 1) % hull_change.h_new.size();
        const SiteId site = hull_change.h_new[position];
        arc.new_span.push_back(site);
        if (moved.sites[site].stable_id == second_id) {
          found_second_anchor = true;
          break;
        }
      }
      if (!found_second_anchor) {
        record_witness(arc.first_anchor, component);
        record_witness(arc.second_anchor, component);
      } else {
        for (std::size_t index = 1; index + 1 < arc.new_span.size();
             ++index) {
          const SiteId site = arc.new_span[index];
          if (component_stable_ids.count(moved.sites[site].stable_id) == 0) {
            record_witness(site, component);
          }
        }
      }
      result.arcs.push_back(std::move(arc));
    }
  }

  std::map<std::uint64_t, std::size_t> span_owner;
  for (std::size_t arc_index = 0; arc_index < result.arcs.size();
       ++arc_index) {
    for (const SiteId site : result.arcs[arc_index].new_span) {
      const std::uint64_t stable_id = moved.sites[site].stable_id;
      const auto owner = span_owner.emplace(stable_id, arc_index);
      if (!owner.second) {
        const int first_component =
            result.arcs[owner.first->second].component;
        const int second_component = result.arcs[arc_index].component;
        const int growth_component =
            component_keys[second_component] <
                    component_keys[first_component]
                ? second_component
                : first_component;
        record_witness(site, growth_component);
      }
    }
  }
  return result;
}

void log_stage2_admission(const char* const reason,
                          const D0AdmissionResult& admission) {
  const std::uint64_t witness_stable_id =
      admission.ok ? admission.minimum_boundary_stable_id
                   : admission.failure_witness_stable_id;
  if (witness_stable_id ==
      std::numeric_limits<std::uint64_t>::max()) {
    std::fprintf(stderr,
                 "[cecr_stage2] reason=%s boundary_components=%zu arcs=%zu "
                 "axis_components=%zu outer_components=%zu "
                 "mixed_components=%zu\n",
                 reason, admission.boundary_component_count,
                 admission.arcs.size(), admission.axis_only_components,
                 admission.outer_only_components,
                 admission.mixed_components);
    return;
  }
  std::fprintf(stderr,
               "[cecr_stage2] reason=%s boundary_components=%zu arcs=%zu "
               "witness_stable_id=%llu axis_components=%zu "
               "outer_components=%zu mixed_components=%zu\n",
               reason, admission.boundary_component_count,
               admission.arcs.size(),
               static_cast<unsigned long long>(witness_stable_id),
               admission.axis_only_components,
               admission.outer_only_components,
               admission.mixed_components);
}

struct D1Result {
  bool ok = false;
  TriangleId growth_face = kInvalidId;
};

D1Result certify_simple_boundary(
    const DelaunayTriangulation& dt,
    const SubstitutedPolygon& polygon,
    const std::vector<std::uint8_t>& in_patch) {
  D1Result result;
  if (polygon.cycle.size() < 3 ||
      polygon.edges.size() != polygon.cycle.size() ||
      expansion_sign(exact_polygon_area(dt, polygon.cycle)) <= 0) {
    return result;
  }
  for (std::size_t index = 0; index < polygon.edges.size(); ++index) {
    if (polygon.edges[index].origin != polygon.cycle[index] ||
        polygon.edges[index].destination !=
            polygon.cycle[(index + 1) % polygon.cycle.size()]) {
      return result;
    }
  }
  bool found = false;
  std::pair<StableEdgeKey, StableEdgeKey> best_key;
  FrontierEdge best_edge;
  const auto record_witness =
      [&](const FrontierEdge& first_edge,
          const FrontierEdge& second_edge) {
        StableEdgeKey first_key = stable_edge_key(
            dt, first_edge.origin, first_edge.destination);
        StableEdgeKey second_key = stable_edge_key(
            dt, second_edge.origin, second_edge.destination);
        FrontierEdge selected_edge = first_edge;
        if (second_key < first_key) {
          std::swap(first_key, second_key);
          selected_edge = second_edge;
        }
        const std::pair<StableEdgeKey, StableEdgeKey> key{first_key,
                                                          second_key};
        if (!found || key < best_key) {
          found = true;
          best_key = key;
          best_edge = selected_edge;
        }
      };
  for (std::size_t first = 0; first < polygon.edges.size(); ++first) {
    const FrontierEdge& first_edge = polygon.edges[first];
    const FrontierEdge& second_edge =
        polygon.edges[(first + 1) % polygon.edges.size()];
    if (point_on_segment(dt.sites[second_edge.destination],
                         dt.sites[first_edge.origin],
                         dt.sites[first_edge.destination]) ||
        point_on_segment(dt.sites[first_edge.origin],
                         dt.sites[second_edge.origin],
                         dt.sites[second_edge.destination])) {
      record_witness(first_edge, second_edge);
    }
  }
  for (std::size_t first = 0; first < polygon.edges.size(); ++first) {
    const std::size_t first_next = (first + 1) % polygon.edges.size();
    for (std::size_t second = first + 1;
         second < polygon.edges.size(); ++second) {
      const std::size_t second_next = (second + 1) % polygon.edges.size();
      if (first_next == second || second_next == first) {
        continue;
      }
      const FrontierEdge& first_edge = polygon.edges[first];
      const FrontierEdge& second_edge = polygon.edges[second];
      if (!segments_intersect_exact(dt.sites[first_edge.origin],
                                    dt.sites[first_edge.destination],
                                    dt.sites[second_edge.origin],
                                    dt.sites[second_edge.destination])) {
        continue;
      }
      record_witness(first_edge, second_edge);
    }
  }
  if (!found) {
    result.ok = true;
    return result;
  }
  result.growth_face =
      growth_face_for_polygon_edge(dt, in_patch, best_edge);
  return result;
}

struct D2Result {
  bool ok = false;
  TriangleId growth_face = kInvalidId;
  std::vector<SiteId> interior_sites;
};

D2Result certify_patch_containment(
    const DelaunayTriangulation& dt,
    const SubstitutedPolygon& polygon,
    const std::vector<int>& labels,
    const int component,
    const std::vector<std::uint8_t>& in_patch) {
  D2Result result;
  std::set<SiteId> boundary(polygon.cycle.begin(), polygon.cycle.end());
  if (boundary.size() != polygon.cycle.size()) {
    return result;
  }
  const std::vector<SiteId> patch_sites =
      component_sites(dt, labels, component);
  SiteId failing_site = kInvalidId;
  for (const SiteId site : patch_sites) {
    if (boundary.count(site) != 0) {
      continue;
    }
    if (classify_point_in_polygon(dt, site, polygon.cycle) !=
        PointPolygonLocation::kInside) {
      failing_site = site;
      break;
    }
    result.interior_sites.push_back(site);
  }
  if (failing_site == kInvalidId) {
    result.ok = true;
    return result;
  }

  for (const FrontierEdge& edge : polygon.edges) {
    if (orient2d_sign_forced_exact(dt.sites[edge.origin],
                                   dt.sites[edge.destination],
                                   dt.sites[failing_site]) >= 0) {
      continue;
    }
    const TriangleId candidate =
        growth_face_for_polygon_edge(dt, in_patch, edge);
    if (candidate != kInvalidId &&
        (result.growth_face == kInvalidId ||
         face_key_less(dt, candidate, result.growth_face))) {
      result.growth_face = candidate;
    }
  }
  return result;
}

struct D3Result {
  bool ok = false;
  TriangleId growth_face = kInvalidId;
};

D3Result certify_exterior_sites(
    const DelaunayTriangulation& dt,
    const SubstitutedPolygon& polygon,
    const std::vector<int>& labels,
    const int component,
    const std::vector<std::uint8_t>& in_patch) {
  D3Result result;
  const std::vector<SiteId> patch_sites =
      component_sites(dt, labels, component);
  std::set<SiteId> in_component(patch_sites.begin(), patch_sites.end());
  double r_min = dt.sites[polygon.cycle.front()].r;
  double r_max = r_min;
  double z_min = dt.sites[polygon.cycle.front()].z;
  double z_max = z_min;
  for (const SiteId site : polygon.cycle) {
    r_min = std::min(r_min, dt.sites[site].r);
    r_max = std::max(r_max, dt.sites[site].r);
    z_min = std::min(z_min, dt.sites[site].z);
    z_max = std::max(z_max, dt.sites[site].z);
  }
  std::vector<SiteId> candidates(dt.sites.size());
  std::iota(candidates.begin(), candidates.end(), SiteId{0});
  // dt.sites is rank-ordered = ascending stable_id (asserted at
  // build/warm entry), so index order IS stable-id order.
  for (const SiteId site : candidates) {
    if (in_component.count(site) != 0 || dt.sites[site].r < r_min ||
        dt.sites[site].r > r_max || dt.sites[site].z < z_min ||
        dt.sites[site].z > z_max) {
      continue;
    }
    if (classify_point_in_polygon(dt, site, polygon.cycle) ==
        PointPolygonLocation::kOutside) {
      continue;
    }
    result.growth_face =
        smallest_unpatched_face_at_site(dt, in_patch, site);
    return result;
  }
  result.ok = true;
  return result;
}

struct MeshEdge {
  HalfEdgeId half_edge = kInvalidId;
  SiteId first = kInvalidId;
  SiteId second = kInvalidId;
};

struct D4Result {
  bool ok = false;
  TriangleId growth_face = kInvalidId;
};

D4Result certify_embedded_exterior(
    const DelaunayTriangulation& moved,
    const CecrComponents& components,
    const std::vector<std::uint8_t>& in_patch,
    const CecrHullChange& hull_change,
    const D0AdmissionResult& admission,
    const std::vector<SubstitutedPolygon>& polygons) {
  D4Result result;
  const char* const debug_env = std::getenv("TENRYU_CECR_D4_DEBUG");
  const bool debug = debug_env != nullptr && debug_env[0] != '\0';
  TENRYU_ASSERT(polygons.size() ==
                    static_cast<std::size_t>(components.component_count),
                "CECR D4 polygon count must match component count");
  TriangleId failing = kInvalidId;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size());
       ++triangle) {
    if (in_patch[triangle] != 0) {
      continue;
    }
    const std::array<SiteId, 3> vertices =
        triangle_vertices(moved, triangle);
    if (orient2d_sign_tess(moved.sites[vertices[0]],
                           moved.sites[vertices[1]],
                           moved.sites[vertices[2]]) <= 0) {
      if (failing == kInvalidId ||
          face_key_less(moved, triangle, failing)) {
        failing = triangle;
      }
    }
  }
  if (failing != kInvalidId) {
    result.growth_face = failing;
    if (debug) {
      std::fprintf(stderr, "[d4dbg] exterior_nonpositive tri=%d\n",
                   static_cast<int>(failing));
    }
    return result;
  }

  struct MeshEdgeEntry {
    std::pair<SiteId, SiteId> key;
    MeshEdge edge;
    double r_min;
    double r_max;
    double z_min;
    double z_max;
  };
  std::vector<MeshEdgeEntry> mesh_edges;
  mesh_edges.reserve(moved.he_origin.size());
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(moved.he_origin.size()); ++edge) {
    const HalfEdgeId twin = moved.he_twin[edge];
    if (in_patch[moved.he_face[edge]] != 0 &&
        (twin == kInvalidId || in_patch[moved.he_face[twin]] != 0)) {
      continue;
    }
    const SiteId first = moved.he_origin[edge];
    const SiteId second = edge_destination(moved, edge);
    const std::pair<SiteId, SiteId> key{std::min(first, second),
                                        std::max(first, second)};
    mesh_edges.push_back(
        MeshEdgeEntry{key,
                      MeshEdge{edge, first, second},
                      std::min(moved.sites[first].r,
                               moved.sites[second].r),
                      std::max(moved.sites[first].r,
                               moved.sites[second].r),
                      std::min(moved.sites[first].z,
                               moved.sites[second].z),
                      std::max(moved.sites[first].z,
                               moved.sites[second].z)});
  }
  std::sort(mesh_edges.begin(), mesh_edges.end(),
            [](const MeshEdgeEntry& lhs, const MeshEdgeEntry& rhs) {
              if (lhs.key != rhs.key) {
                return lhs.key < rhs.key;
              }
              return lhs.edge.half_edge < rhs.edge.half_edge;
            });
  mesh_edges.erase(
      std::unique(mesh_edges.begin(), mesh_edges.end(),
                  [](const MeshEdgeEntry& lhs, const MeshEdgeEntry& rhs) {
                    return lhs.key == rhs.key;
                  }),
      mesh_edges.end());

  constexpr std::size_t kCrossingBuckets = 256;
  double z_lo = 0.0;
  double z_hi = 0.0;
  std::size_t bucket_count = 0;
  double bucket_width = 0.0;
  std::vector<std::size_t> bucket_offsets;
  std::vector<std::size_t> bucket_entries;
  if (!mesh_edges.empty()) {
    z_lo = mesh_edges.front().z_min;
    z_hi = mesh_edges.front().z_max;
    for (const MeshEdgeEntry& entry : mesh_edges) {
      z_lo = std::min(z_lo, entry.z_min);
      z_hi = std::max(z_hi, entry.z_max);
    }
    bucket_count = z_hi > z_lo ? kCrossingBuckets : 1;
    bucket_width =
        z_hi > z_lo ? (z_hi - z_lo) / static_cast<double>(bucket_count)
                    : 1.0;
  }
  const auto bucket_of = [&](const double z) -> std::size_t {
    if (bucket_count <= 1) {
      return 0;
    }
    const double offset = (z - z_lo) / bucket_width;
    if (offset <= 0.0) {
      return 0;
    }
    const std::size_t index = static_cast<std::size_t>(offset);
    return std::min(index, bucket_count - 1);
  };
  bucket_offsets.assign(bucket_count + 1, 0);
  for (const MeshEdgeEntry& entry : mesh_edges) {
    const std::size_t first_bucket = bucket_of(entry.z_min);
    const std::size_t last_bucket = bucket_of(entry.z_max);
    for (std::size_t bucket = first_bucket; bucket <= last_bucket; ++bucket) {
      ++bucket_offsets[bucket + 1];
    }
  }
  std::partial_sum(bucket_offsets.begin(), bucket_offsets.end(),
                   bucket_offsets.begin());
  bucket_entries.resize(bucket_offsets.back());
  std::vector<std::size_t> bucket_positions = bucket_offsets;
  for (std::size_t index = 0; index < mesh_edges.size(); ++index) {
    const MeshEdgeEntry& entry = mesh_edges[index];
    const std::size_t first_bucket = bucket_of(entry.z_min);
    const std::size_t last_bucket = bucket_of(entry.z_max);
    for (std::size_t bucket = first_bucket; bucket <= last_bucket; ++bucket) {
      bucket_entries[bucket_positions[bucket]++] = index;
    }
  }

  bool found_crossing = false;
  std::pair<StableEdgeKey, StableEdgeKey> best_crossing;
  FrontierEdge best_growth_edge;
  FrontierEdge best_frontier_edge;
  MeshEdge best_mesh_edge;
  std::vector<std::uint32_t> visit_stamp(mesh_edges.size(), 0);
  std::uint32_t visit_epoch = 0;
  for (int component = 0; component < components.component_count;
       ++component) {
    const SubstitutedPolygon& polygon = polygons[component];
    for (const FrontierEdge& frontier_edge : polygon.edges) {
      const StableEdgeKey frontier_key = stable_edge_key(
          moved, frontier_edge.origin, frontier_edge.destination);
      const double frontier_r_min =
          std::min(moved.sites[frontier_edge.origin].r,
                   moved.sites[frontier_edge.destination].r);
      const double frontier_r_max =
          std::max(moved.sites[frontier_edge.origin].r,
                   moved.sites[frontier_edge.destination].r);
      const double frontier_z_min =
          std::min(moved.sites[frontier_edge.origin].z,
                   moved.sites[frontier_edge.destination].z);
      const double frontier_z_max =
          std::max(moved.sites[frontier_edge.origin].z,
                   moved.sites[frontier_edge.destination].z);
      ++visit_epoch;
      if (bucket_count == 0) {
        continue;
      }
      const std::size_t first_bucket = bucket_of(frontier_z_min);
      const std::size_t last_bucket = bucket_of(frontier_z_max);
      for (std::size_t bucket = first_bucket; bucket <= last_bucket;
           ++bucket) {
        for (std::size_t position = bucket_offsets[bucket];
             position < bucket_offsets[bucket + 1]; ++position) {
          const std::size_t index = bucket_entries[position];
          if (visit_stamp[index] == visit_epoch) {
            continue;
          }
          visit_stamp[index] = visit_epoch;
          const MeshEdgeEntry& entry = mesh_edges[index];
          const MeshEdge& candidate = entry.edge;
          const double candidate_r_min = entry.r_min;
          const double candidate_r_max = entry.r_max;
          const double candidate_z_min = entry.z_min;
          const double candidate_z_max = entry.z_max;
          if (frontier_r_max < candidate_r_min ||
              candidate_r_max < frontier_r_min ||
              frontier_z_max < candidate_z_min ||
              candidate_z_max < frontier_z_min) {
            continue;
          }
          const int shared_endpoints =
              static_cast<int>(candidate.first == frontier_edge.origin ||
                               candidate.first == frontier_edge.destination) +
              static_cast<int>(
                  candidate.second == frontier_edge.origin ||
                  candidate.second == frontier_edge.destination);
          if (shared_endpoints == 2) {
            continue;
          }
          bool intersects = false;
          if (shared_endpoints == 1) {
            SiteId shared = candidate.first;
            SiteId candidate_other = candidate.second;
            if (candidate.second == frontier_edge.origin ||
                candidate.second == frontier_edge.destination) {
              shared = candidate.second;
              candidate_other = candidate.first;
            }
            const SiteId frontier_other =
                frontier_edge.origin == shared ? frontier_edge.destination
                                               : frontier_edge.origin;
            intersects = point_on_segment(
                             moved.sites[candidate_other], moved.sites[shared],
                             moved.sites[frontier_other]) ||
                         point_on_segment(
                             moved.sites[frontier_other], moved.sites[shared],
                             moved.sites[candidate_other]);
          } else {
            intersects = segments_intersect_exact(
                moved.sites[frontier_edge.origin],
                moved.sites[frontier_edge.destination],
                moved.sites[candidate.first], moved.sites[candidate.second]);
          }
          if (!intersects) {
            continue;
          }
          StableEdgeKey first_key = frontier_key;
          StableEdgeKey second_key = stable_edge_key(
              moved, candidate.first, candidate.second);
          FrontierEdge selected_edge = frontier_edge;
          if (second_key < first_key) {
            std::swap(first_key, second_key);
            selected_edge = FrontierEdge{candidate.half_edge,
                                         candidate.first,
                                         candidate.second};
          }
          const std::pair<StableEdgeKey, StableEdgeKey> witness{first_key,
                                                                second_key};
          if (!found_crossing || witness < best_crossing) {
            found_crossing = true;
            best_crossing = witness;
            best_growth_edge = selected_edge;
            if (debug) {
              best_frontier_edge = frontier_edge;
              best_mesh_edge = candidate;
            }
          }
        }
      }
    }
  }
  if (found_crossing) {
    result.growth_face = growth_face_for_polygon_edge(
        moved, in_patch, best_growth_edge);
    if (debug) {
      const StableEdgeKey frontier_key = stable_edge_key(
          moved, best_frontier_edge.origin,
          best_frontier_edge.destination);
      const StableEdgeKey mesh_key = stable_edge_key(
          moved, best_mesh_edge.first, best_mesh_edge.second);
      std::fprintf(
          stderr,
          "[d4dbg] crossing frontier_edge=(%llu,%llu) "
          "mesh_edge=(%llu,%llu) span=%d\n",
          static_cast<unsigned long long>(frontier_key.lo),
          static_cast<unsigned long long>(frontier_key.hi),
          static_cast<unsigned long long>(mesh_key.lo),
          static_cast<unsigned long long>(mesh_key.hi),
          best_frontier_edge.half_edge == kInvalidId ? 1 : 0);
    }
    return result;
  }

  if (!admission.has_boundary_component) {
    if (!hull_change.cycles_equal) {
      return result;
    }
  } else {
    using DirectedStableEdge = std::pair<std::uint64_t, std::uint64_t>;
    const auto directed_edge = [&moved](const SiteId first,
                                         const SiteId second) {
      return DirectedStableEdge{moved.sites[first].stable_id,
                                moved.sites[second].stable_id};
    };

    std::vector<std::uint8_t> deleted_old_edges(
        hull_change.h_old.size(), 0);
    std::map<std::size_t, const HullArc*> arc_at_old_position;
    std::map<DirectedStableEdge, std::size_t> span_edge_owners;
    for (const HullArc& arc : admission.arcs) {
      const auto first = std::find(hull_change.h_old.begin(),
                                   hull_change.h_old.end(),
                                   arc.first_anchor);
      const auto second = std::find(hull_change.h_old.begin(),
                                    hull_change.h_old.end(),
                                    arc.second_anchor);
      TENRYU_ASSERT(first != hull_change.h_old.end() &&
                        second != hull_change.h_old.end() &&
                        arc.new_span.size() >= 2,
                    "CECR D4 admitted arc is not on the old hull");
      std::size_t position =
          static_cast<std::size_t>(first - hull_change.h_old.begin());
      TENRYU_ASSERT(arc_at_old_position.emplace(position, &arc).second,
                    "CECR D4 hull arcs have duplicate starts");
      std::size_t steps = 0;
      while (hull_change.h_old[position] != arc.second_anchor) {
        TENRYU_ASSERT(steps < hull_change.h_old.size() &&
                          deleted_old_edges[position] == 0,
                      "CECR D4 hull arcs overlap or do not terminate");
        deleted_old_edges[position] = 1;
        position = (position + 1) % hull_change.h_old.size();
        ++steps;
      }
      TENRYU_ASSERT(steps > 0,
                    "CECR D4 hull arc must delete at least one edge");
      for (std::size_t index = 0; index + 1 < arc.new_span.size();
           ++index) {
        ++span_edge_owners[directed_edge(arc.new_span[index],
                                         arc.new_span[index + 1])];
      }
    }

    std::map<DirectedStableEdge, std::pair<SiteId, SiteId>> new_edges;
    for (std::size_t index = 0; index < hull_change.h_new.size(); ++index) {
      const SiteId first = hull_change.h_new[index];
      const SiteId second =
          hull_change.h_new[(index + 1) % hull_change.h_new.size()];
      new_edges.emplace(directed_edge(first, second),
                        std::make_pair(first, second));
    }

    std::map<DirectedStableEdge, std::pair<SiteId, SiteId>> kept_edges;
    std::vector<SiteId> hull_witnesses;
    for (std::size_t index = 0; index < hull_change.h_old.size(); ++index) {
      if (deleted_old_edges[index] != 0) {
        continue;
      }
      const SiteId first = hull_change.h_old[index];
      const SiteId second =
          hull_change.h_old[(index + 1) % hull_change.h_old.size()];
      const DirectedStableEdge key = directed_edge(first, second);
      kept_edges.emplace(key, std::make_pair(first, second));
      if (new_edges.count(key) == 0) {
        hull_witnesses.push_back(first);
        hull_witnesses.push_back(second);
      }
    }
    for (const auto& entry : new_edges) {
      const bool kept = kept_edges.count(entry.first) != 0;
      const auto span = span_edge_owners.find(entry.first);
      const std::size_t span_count =
          span == span_edge_owners.end() ? 0 : span->second;
      if ((kept && span_count != 0) || (!kept && span_count != 1)) {
        hull_witnesses.push_back(entry.second.first);
        hull_witnesses.push_back(entry.second.second);
      }
    }

    const auto kept_start = std::find(
        deleted_old_edges.begin(), deleted_old_edges.end(), std::uint8_t{0});
    TENRYU_ASSERT(kept_start != deleted_old_edges.end(),
                  "CECR D4 decomposition deleted the entire old hull");
    std::size_t position =
        static_cast<std::size_t>(kept_start - deleted_old_edges.begin());
    std::size_t consumed = 0;
    std::vector<SiteId> expected_new_hull;
    while (consumed < hull_change.h_old.size()) {
      if (deleted_old_edges[position] == 0) {
        expected_new_hull.push_back(hull_change.h_old[position]);
        position = (position + 1) % hull_change.h_old.size();
        ++consumed;
        continue;
      }
      const auto arc_entry = arc_at_old_position.find(position);
      TENRYU_ASSERT(arc_entry != arc_at_old_position.end(),
                    "CECR D4 deleted chain is missing its hull arc");
      const HullArc& arc = *arc_entry->second;
      expected_new_hull.insert(expected_new_hull.end(),
                               arc.new_span.begin(),
                               arc.new_span.end() - 1);
      do {
        TENRYU_ASSERT(consumed < hull_change.h_old.size() &&
                          deleted_old_edges[position] != 0,
                      "CECR D4 deleted hull chain does not match its arc");
        position = (position + 1) % hull_change.h_old.size();
        ++consumed;
      } while (hull_change.h_old[position] != arc.second_anchor);
    }
    if (!stable_cycles_equal(moved, expected_new_hull,
                             hull_change.h_new)) {
      if (!expected_new_hull.empty()) {
        hull_witnesses.push_back(expected_new_hull.front());
      }
      if (!hull_change.h_new.empty()) {
        hull_witnesses.push_back(hull_change.h_new.front());
      }
    }
    if (!hull_witnesses.empty()) {
      std::sort(hull_witnesses.begin(), hull_witnesses.end(),
                [&moved](const SiteId first, const SiteId second) {
                  return moved.sites[first].stable_id <
                         moved.sites[second].stable_id;
                });
      hull_witnesses.erase(
          std::unique(hull_witnesses.begin(), hull_witnesses.end()),
          hull_witnesses.end());
      for (const SiteId witness : hull_witnesses) {
        result.growth_face =
            smallest_unpatched_face_at_site(moved, in_patch, witness);
        if (result.growth_face != kInvalidId) {
          if (debug) {
            std::fprintf(
                stderr,
                "[d4dbg] hull_witnesses n=%zu first_stable=%llu "
                "growth=found\n",
                hull_witnesses.size(),
                static_cast<unsigned long long>(
                    moved.sites[hull_witnesses.front()].stable_id));
          }
          return result;
        }
      }
      if (debug) {
        std::fprintf(
            stderr,
            "[d4dbg] hull_witnesses n=%zu first_stable=%llu "
            "growth=exhausted\n",
            hull_witnesses.size(),
            static_cast<unsigned long long>(
                moved.sites[hull_witnesses.front()].stable_id));
      }
      return result;
    }
  }

  Expansion exterior_area;
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size());
       ++triangle) {
    if (in_patch[triangle] != 0) {
      continue;
    }
    const std::array<SiteId, 3> vertices =
        triangle_vertices(moved, triangle);
    exterior_area = expansion_sum(
        exterior_area,
        exact_triangle_area(moved.sites[vertices[0]],
                            moved.sites[vertices[1]],
                            moved.sites[vertices[2]]));
  }
  Expansion cavity_area;
  for (int component = 0; component < components.component_count;
       ++component) {
    cavity_area = expansion_sum(
        cavity_area, exact_polygon_area(moved, polygons[component].cycle));
  }
  const Expansion expected_exterior = expansion_difference(
      exact_polygon_area(moved, hull_change.h_new), cavity_area);
  if (expansion_sign(
          expansion_difference(exterior_area, expected_exterior)) != 0) {
    if (debug) {
      std::fprintf(stderr, "[d4dbg] area_mismatch\n");
    }
    return result;
  }
  result.ok = true;
  return result;
}

}  // namespace

CecrHullChange cecr_hull_change(const DelaunayTriangulation& previous,
                                const DelaunayTriangulation& moved) {
  TENRYU_ASSERT(previous.sites.size() == moved.sites.size(),
                "CECR hull-change site counts must match");
  for (std::size_t site = 0; site < moved.sites.size(); ++site) {
    TENRYU_ASSERT(previous.sites[site].stable_id ==
                      moved.sites[site].stable_id,
                  "CECR hull-change stable ids must match");
  }
  CecrHullChange result;
  result.h_old = triangulation_hull_cycle_for_cecr(previous);
  result.h_new = exact_convex_hull(moved.sites);
  result.cycles_equal =
      stable_cycles_equal(moved, result.h_old, result.h_new);

  std::set<std::uint64_t> old_ids;
  std::set<std::uint64_t> new_ids;
  for (const SiteId site : result.h_old) {
    old_ids.insert(moved.sites[site].stable_id);
  }
  for (const SiteId site : result.h_new) {
    new_ids.insert(moved.sites[site].stable_id);
  }

  std::vector<SiteId> sites(moved.sites.size());
  std::iota(sites.begin(), sites.end(), SiteId{0});
  std::sort(sites.begin(), sites.end(),
            [&moved](const SiteId first, const SiteId second) {
              return moved.sites[first].stable_id <
                     moved.sites[second].stable_id;
            });
  for (const SiteId site : sites) {
    const std::uint64_t stable_id = moved.sites[site].stable_id;
    if (new_ids.count(stable_id) != 0 && old_ids.count(stable_id) == 0) {
      result.entrants.push_back(site);
    }
    if (old_ids.count(stable_id) != 0 && new_ids.count(stable_id) == 0) {
      result.leavers.push_back(site);
    }
  }
  return result;
}

CecrSeedResult cecr_seed_defects(const DelaunayTriangulation& previous,
                                 const DelaunayTriangulation& moved) {
  return cecr_seed_defects(previous, moved,
                           cecr_hull_change(previous, moved));
}

CecrSeedResult cecr_seed_defects(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change) {
  TENRYU_ASSERT(previous.tri_edge.size() == moved.tri_edge.size(),
                "CECR previous and moved triangle counts must match");
  TENRYU_ASSERT(previous.sites.size() == moved.sites.size(),
                "CECR previous and moved site counts must match");
  TENRYU_ASSERT(previous.he_origin.size() == moved.he_origin.size() &&
                    previous.he_twin.size() == moved.he_twin.size() &&
                    previous.he_next.size() == moved.he_next.size() &&
                    previous.he_face.size() == moved.he_face.size(),
                "CECR previous and moved connectivity sizes must match");

  CecrSeedResult result;
  result.in_patch.assign(moved.tri_edge.size(), 0);
  std::vector<std::uint8_t> seed_vertices(moved.sites.size(), 0);
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size()); ++triangle) {
    const std::array<SiteId, 3> vertices =
        triangle_vertices(moved, triangle);
    if (orient2d_sign_forced_exact(moved.sites[vertices[0]],
                                   moved.sites[vertices[1]],
                                   moved.sites[vertices[2]]) <= 0) {
      result.defect_triangles.push_back(triangle);
      seed_vertices[vertices[0]] = 1;
      seed_vertices[vertices[1]] = 1;
      seed_vertices[vertices[2]] = 1;
    }
  }
  result.any_defect = !result.defect_triangles.empty();
  for (const SiteId entrant : hull_change.entrants) {
    seed_vertices[entrant] = 1;
  }
  for (const SiteId leaver : hull_change.leavers) {
    seed_vertices[leaver] = 1;
  }
  if (!result.any_defect && hull_change.entrants.empty() &&
      hull_change.leavers.empty()) {
    return result;
  }
  add_closed_vertex_star(moved, seed_vertices, result.in_patch);

  std::vector<std::uint8_t> in_defect_class(moved.tri_edge.size(), 0);
  std::queue<TriangleId> pending;
  for (const TriangleId defect : result.defect_triangles) {
    if (in_defect_class[defect] == 0) {
      in_defect_class[defect] = 1;
      pending.push(defect);
    }
  }
  while (!pending.empty()) {
    const TriangleId triangle = pending.front();
    pending.pop();
    for (const HalfEdgeId edge : triangle_edges(previous, triangle)) {
      const HalfEdgeId twin = previous.he_twin[edge];
      if (twin == kInvalidId || !old_edge_is_exact_cocircular(previous, edge)) {
        continue;
      }
      const TriangleId neighbor = previous.he_face[twin];
      if (in_defect_class[neighbor] == 0) {
        in_defect_class[neighbor] = 1;
        pending.push(neighbor);
      }
    }
  }

  // Canon reading: only old cocircular classes containing a defect face are
  // enlarged. Closed-star completion is then applied once, and only to the
  // vertices of faces newly added by that class step; there is no fixpoint.
  std::vector<std::uint8_t> class_added_vertices(moved.sites.size(), 0);
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size()); ++triangle) {
    if (in_defect_class[triangle] == 0 || result.in_patch[triangle] != 0) {
      continue;
    }
    result.in_patch[triangle] = 1;
    for (const SiteId vertex : triangle_vertices(moved, triangle)) {
      class_added_vertices[vertex] = 1;
    }
  }
  add_closed_vertex_star(moved, class_added_vertices, result.in_patch);
  return result;
}

CecrComponents cecr_form_components(
    const DelaunayTriangulation& moved,
    const std::vector<std::uint8_t>& in_patch) {
  TENRYU_ASSERT(in_patch.size() == moved.tri_edge.size(),
                "CECR patch flag count must match triangle count");
  CecrComponents result;
  result.component_of_triangle.assign(moved.tri_edge.size(), -1);

  int initial_component_count = 0;
  for (TriangleId seed = 0;
       seed < static_cast<TriangleId>(moved.tri_edge.size()); ++seed) {
    if (in_patch[seed] == 0 || result.component_of_triangle[seed] >= 0) {
      continue;
    }
    std::queue<TriangleId> pending;
    result.component_of_triangle[seed] = initial_component_count;
    pending.push(seed);
    while (!pending.empty()) {
      const TriangleId triangle = pending.front();
      pending.pop();
      for (const HalfEdgeId edge : triangle_edges(moved, triangle)) {
        const HalfEdgeId twin = moved.he_twin[edge];
        if (twin == kInvalidId) {
          continue;
        }
        const TriangleId neighbor = moved.he_face[twin];
        if (in_patch[neighbor] == 0 ||
            result.component_of_triangle[neighbor] >= 0) {
          continue;
        }
        result.component_of_triangle[neighbor] = initial_component_count;
        pending.push(neighbor);
      }
    }
    ++initial_component_count;
  }
  if (initial_component_count == 0) {
    return result;
  }

  std::vector<std::set<SiteId>> component_vertices(
      static_cast<std::size_t>(initial_component_count));
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size()); ++triangle) {
    const int component = result.component_of_triangle[triangle];
    if (component < 0) {
      continue;
    }
    for (const SiteId vertex : triangle_vertices(moved, triangle)) {
      component_vertices[component].insert(vertex);
    }
  }

  DisjointSet merged(static_cast<std::size_t>(initial_component_count));
  for (int first = 0; first < initial_component_count; ++first) {
    for (int second = first + 1; second < initial_component_count; ++second) {
      bool share_vertex = false;
      const std::set<SiteId>& smaller =
          component_vertices[first].size() < component_vertices[second].size()
              ? component_vertices[first]
              : component_vertices[second];
      const std::set<SiteId>& larger =
          component_vertices[first].size() < component_vertices[second].size()
              ? component_vertices[second]
              : component_vertices[first];
      for (const SiteId vertex : smaller) {
        if (larger.count(vertex) != 0) {
          share_vertex = true;
          break;
        }
      }
      if (share_vertex) {
        merged.unite(first, second);
      }
    }
  }

  for (int& component : result.component_of_triangle) {
    if (component >= 0) {
      component = merged.find(component);
    }
  }
  result.component_count =
      normalize_component_labels(moved, result.component_of_triangle);
  return result;
}

CecrClosureResult cecr_close_disks(
    const DelaunayTriangulation& moved,
    CecrComponents components,
    std::vector<std::uint8_t> in_patch) {
  TENRYU_ASSERT(in_patch.size() == moved.tri_edge.size(),
                "CECR patch flag count must match triangle count");
  TENRYU_ASSERT(components.component_of_triangle.size() ==
                    moved.tri_edge.size(),
                "CECR component label count must match triangle count");
  for (TriangleId triangle = 0;
       triangle < static_cast<TriangleId>(moved.tri_edge.size()); ++triangle) {
    TENRYU_ASSERT((in_patch[triangle] == 0) ==
                      (components.component_of_triangle[triangle] < 0),
                  "CECR patch flags and component labels disagree");
  }
  if (components.component_count == 0) {
    return CecrClosureResult{std::move(in_patch), true, {}};
  }

  std::vector<int> labels = std::move(components.component_of_triangle);
  int component_count = normalize_component_labels(moved, labels);
  // site -> incident faces: each incident face has exactly one
  // half-edge whose origin is the site, so the origin buckets
  // enumerate every incident face exactly once.
  std::vector<std::size_t> site_face_offsets(moved.sites.size() + 1, 0);
  for (HalfEdgeId edge = 0;
       edge < static_cast<HalfEdgeId>(moved.he_origin.size()); ++edge) {
    ++site_face_offsets[static_cast<std::size_t>(moved.he_origin[edge]) + 1];
  }
  std::partial_sum(site_face_offsets.begin(), site_face_offsets.end(),
                   site_face_offsets.begin());
  std::vector<TriangleId> site_faces(site_face_offsets.back());
  {
    std::vector<std::size_t> cursor(site_face_offsets.begin(),
                                    site_face_offsets.end() - 1);
    for (HalfEdgeId edge = 0;
         edge < static_cast<HalfEdgeId>(moved.he_origin.size()); ++edge) {
      site_faces[cursor[static_cast<std::size_t>(moved.he_origin[edge])]++] =
          moved.he_face[edge];
    }
  }
  std::vector<std::uint32_t> fill_visited(moved.tri_edge.size(), 0);
  std::uint32_t fill_epoch = 0;
  std::vector<std::uint8_t> hole_free(
      static_cast<std::size_t>(component_count), 0);
  const auto renormalize_and_remap_flags =
      [&](const std::vector<std::size_t>& offsets,
          const std::vector<TriangleId>& triangles, const int changed) {
        const int old_count = component_count;
        std::vector<TriangleId> representative(
            static_cast<std::size_t>(old_count), kInvalidId);
        for (int component = 0; component < old_count; ++component) {
          representative[static_cast<std::size_t>(component)] =
              triangles[offsets[static_cast<std::size_t>(component)]];
        }
        component_count = normalize_component_labels(moved, labels);
        std::vector<std::uint8_t> next(
            static_cast<std::size_t>(component_count), 0);
        std::vector<std::uint8_t> written(
            static_cast<std::size_t>(component_count), 0);
        for (int component = 0; component < old_count; ++component) {
          const TriangleId rep =
              representative[static_cast<std::size_t>(component)];
          const int target = labels[rep];
          TENRYU_ASSERT(target >= 0 && target < component_count,
                        "CECR closure flag remap lost a component");
          if (written[static_cast<std::size_t>(target)] != 0) {
            // two old components merged here (absorption): unknown set,
            // force re-check
            next[static_cast<std::size_t>(target)] = 0;
            continue;
          }
          written[static_cast<std::size_t>(target)] = 1;
          next[static_cast<std::size_t>(target)] =
              component == changed
                  ? std::uint8_t{0}
                  : hole_free[static_cast<std::size_t>(component)];
        }
        if (changed >= 0) {
          const TriangleId changed_rep =
              representative[static_cast<std::size_t>(changed)];
          next[static_cast<std::size_t>(labels[changed_rep])] = 0;
        }
        hole_free = std::move(next);
      };
  while (true) {
    std::vector<std::size_t> component_offsets(
        static_cast<std::size_t>(component_count) + 1, 0);
    for (TriangleId triangle = 0;
         triangle < static_cast<TriangleId>(moved.tri_edge.size());
         ++triangle) {
      if (labels[triangle] >= 0) {
        ++component_offsets[static_cast<std::size_t>(labels[triangle]) + 1];
      }
    }
    std::partial_sum(component_offsets.begin(), component_offsets.end(),
                     component_offsets.begin());
    std::vector<TriangleId> component_triangles(component_offsets.back());
    {
      std::vector<std::size_t> cursor(component_offsets.begin(),
                                      component_offsets.end() - 1);
      for (TriangleId triangle = 0;
           triangle < static_cast<TriangleId>(moved.tri_edge.size());
           ++triangle) {
        if (labels[triangle] >= 0) {
          component_triangles[cursor[static_cast<std::size_t>(
              labels[triangle])]++] = triangle;
        }
      }
    }
    bool filled_hole = false;
    int changed_component = -1;
    for (int component = 0; component < component_count; ++component) {
      if (hole_free[static_cast<std::size_t>(component)] != 0) {
        continue;
      }
      if (fill_first_enclosed_complement(moved, component, labels,
                                         in_patch, fill_visited,
                                         fill_epoch)) {
        filled_hole = true;
        changed_component = component;
        break;
      }
      hole_free[static_cast<std::size_t>(component)] = 1;
    }
    if (filled_hole) {
      renormalize_and_remap_flags(component_offsets, component_triangles,
                                  changed_component);
      continue;
    }

    bool all_disks = true;
    bool enlarged_pinch = false;
    for (int component = 0; component < component_count; ++component) {
      const std::size_t component_begin =
          component_offsets[static_cast<std::size_t>(component)];
      const std::size_t component_size =
          component_offsets[static_cast<std::size_t>(component) + 1] -
          component_begin;
      const std::vector<FrontierEdge> frontier =
          component_frontier_from_list(
              moved, labels, component,
              component_triangles.data() + component_begin,
              component_size);
      const FrontierTopology topology = analyze_frontier(moved, frontier);
      if (topology.single_cycle) {
        assert_disk_euler_characteristic(
            moved, component_triangles.data() + component_begin,
            component_size);
        continue;
      }
      all_disks = false;
      if (topology.pinch_vertices.empty()) {
        continue;
      }
      const SiteId pinch_vertex = topology.pinch_vertices.front();
      const TriangleId outside = smallest_outside_face_at_vertex(
          moved, labels, component, pinch_vertex, site_face_offsets,
          site_faces);
      if (outside == kInvalidId) {
        continue;
      }
      if (labels[outside] >= 0) {
        const int absorbed = labels[outside];
        for (int& label : labels) {
          if (label == absorbed) {
            label = component;
          }
        }
      } else {
        labels[outside] = component;
        in_patch[outside] = 1;
      }
      enlarged_pinch = true;
      changed_component = component;
      break;
    }
    if (all_disks) {
      return CecrClosureResult{std::move(in_patch), true, {}};
    }
    if (!enlarged_pinch) {
      return CecrClosureResult{std::move(in_patch), false,
                               "cecr_disk_closure_failed"};
    }
    renormalize_and_remap_flags(component_offsets, component_triangles,
                                changed_component);
  }
}

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    std::vector<std::uint8_t> in_patch) {
  return cecr_certify_and_grow(previous, moved,
                               cecr_hull_change(previous, moved),
                               std::move(in_patch));
}

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change,
    std::vector<std::uint8_t> in_patch) {
  return cecr_certify_and_grow(previous, moved, hull_change,
                               std::move(in_patch), nullptr);
}

CecrCavitySet cecr_certify_and_grow(
    const DelaunayTriangulation& previous,
    const DelaunayTriangulation& moved,
    const CecrHullChange& hull_change,
    std::vector<std::uint8_t> in_patch,
    CecrTimingData* timing) {
  TENRYU_ASSERT(previous.sites.size() == moved.sites.size(),
                "CECR certification site counts must match");
  TENRYU_ASSERT(previous.tri_edge.size() == moved.tri_edge.size() &&
                    previous.he_origin.size() == moved.he_origin.size() &&
                    previous.he_twin.size() == moved.he_twin.size() &&
                    previous.he_next.size() == moved.he_next.size() &&
                    previous.he_face.size() == moved.he_face.size(),
                "CECR certification connectivity sizes must match");
  TENRYU_ASSERT(in_patch.size() == moved.tri_edge.size(),
                "CECR certification patch size must match triangle count");
  for (const std::uint8_t flag : in_patch) {
    TENRYU_ASSERT(flag <= 1,
                  "CECR certification patch flags must be binary");
  }
  for (std::size_t site = 0; site < moved.sites.size(); ++site) {
    TENRYU_ASSERT(previous.sites[site].stable_id ==
                      moved.sites[site].stable_id,
                  "CECR certification stable ids must match");
  }

  const auto failure = [](const char* reason) {
    CecrCavitySet result;
    result.reason = reason;
    return result;
  };
  const char* const debug_env = std::getenv("TENRYU_CECR_D4_DEBUG");
  const bool debug = debug_env != nullptr && debug_env[0] != '\0';
  const auto log_growth = [&in_patch, debug](const TriangleId face) {
    if (debug) {
      const std::size_t patch_size = static_cast<std::size_t>(std::count(
          in_patch.begin(), in_patch.end(), std::uint8_t{1}));
      std::fprintf(stderr, "[d4dbg] grow face=%d patch_size=%zu\n",
                   static_cast<int>(face), patch_size);
    }
  };
  if (std::find(in_patch.begin(), in_patch.end(), std::uint8_t{1}) ==
      in_patch.end()) {
    return CecrCavitySet{{}, true, {}};
  }

  // telemetry-only band, matches the S2-w0 census (0.1 * H_old r_max at
  // previous coordinates; hull edges classified at moved coordinates)
  double axis_band_r = 0.0;
  if (tess_axis_census_enabled()) {
    double h_old_r_max = 0.0;
    for (const SiteId site : hull_change.h_old) {
      h_old_r_max = std::max(h_old_r_max, previous.sites[site].r);
    }
    axis_band_r = 0.1 * h_old_r_max;
  }

  int growth_rounds = 0;
  while (true) {
    ++growth_rounds;
    if (growth_rounds > kCecrGrowthRoundCap) {
      return failure("cecr_growth_round_cap");
    }
    if (timing != nullptr) {
      ++timing->growth_iterations;
    }
    CecrStageTimer components_timer(
        timing != nullptr ? &timing->components_ms : nullptr);
    CecrComponents components = edge_connected_components(moved, in_patch);
    CecrClosureResult closure =
        cecr_close_disks(moved, components, in_patch);
    if (!closure.ok) {
      return failure("cecr_d1_selfintersect");
    }
    if (closure.in_patch != in_patch) {
      in_patch = std::move(closure.in_patch);
      continue;
    }
    components = edge_connected_components(moved, in_patch);
    components_timer.stop();

    D0AdmissionResult admission;
    {
      CecrStageTimer d0_timer(
          timing != nullptr ? &timing->d0_ms : nullptr);
      admission = certify_hull_frontier_admission(
          moved, components, in_patch, hull_change, axis_band_r);
    }
    if (!admission.ok) {
      TENRYU_ASSERT(admission.failure_witness != kInvalidId,
                    "CECR D0 failure is missing its site witness");
      const TriangleId growth_face = smallest_unpatched_face_at_site(
          moved, in_patch, admission.failure_witness);
      if (growth_face == kInvalidId) {
        log_stage2_admission("cecr_d0_span", admission);
        return failure("cecr_d0_span");
      }
      in_patch[growth_face] = 1;
      log_growth(growth_face);
      continue;
    }

    std::vector<OrderedFrontier> frontiers;
    std::vector<SubstitutedPolygon> polygons;
    frontiers.reserve(static_cast<std::size_t>(components.component_count));
    polygons.reserve(static_cast<std::size_t>(components.component_count));
    for (int component = 0; component < components.component_count;
         ++component) {
      frontiers.push_back(order_frontier(
          moved, component_frontier(moved,
                                    components.component_of_triangle,
                                    component)));
      polygons.push_back(substitute_hull_arcs(
          moved, frontiers.back(), admission.arcs, component));
    }

    bool grew = false;
    for (int component = 0; component < components.component_count;
         ++component) {
      const SubstitutedPolygon& polygon = polygons[component];

      D1Result d1;
      {
        CecrStageTimer d1_timer(
            timing != nullptr ? &timing->d1_ms : nullptr);
        d1 = certify_simple_boundary(moved, polygon, in_patch);
      }
      if (!d1.ok) {
        if (d1.growth_face == kInvalidId) {
          return failure("cecr_d1_selfintersect");
        }
        in_patch[d1.growth_face] = 1;
        log_growth(d1.growth_face);
        grew = true;
        break;
      }

      D2Result d2;
      {
        CecrStageTimer d2_timer(
            timing != nullptr ? &timing->d2_ms : nullptr);
        d2 = certify_patch_containment(
            moved, polygon, components.component_of_triangle,
            component, in_patch);
      }
      if (!d2.ok) {
        if (d2.growth_face == kInvalidId) {
          return failure("cecr_d2_containment");
        }
        in_patch[d2.growth_face] = 1;
        log_growth(d2.growth_face);
        grew = true;
        break;
      }

      D3Result d3;
      {
        CecrStageTimer d3_timer(
            timing != nullptr ? &timing->d3_ms : nullptr);
        d3 = certify_exterior_sites(
            moved, polygon, components.component_of_triangle,
            component, in_patch);
      }
      if (!d3.ok) {
        if (d3.growth_face == kInvalidId) {
          return failure("cecr_d3_exterior_site");
        }
        in_patch[d3.growth_face] = 1;
        log_growth(d3.growth_face);
        grew = true;
        break;
      }

    }
    if (grew) {
      continue;
    }
    D4Result d4;
    {
      CecrStageTimer d4_timer(
          timing != nullptr ? &timing->d4_ms : nullptr);
      d4 = certify_embedded_exterior(
          moved, components, in_patch, hull_change, admission, polygons);
    }
    if (!d4.ok) {
      if (d4.growth_face == kInvalidId) {
        return failure("cecr_d4_embedding");
      }
      in_patch[d4.growth_face] = 1;
      log_growth(d4.growth_face);
      continue;
    }

    if (admission.has_boundary_component) {
      log_stage2_admission("cecr_stage2_spliced", admission);
    }

    CecrCavitySet result;
    result.ok = true;
    result.cavities.reserve(
        static_cast<std::size_t>(components.component_count));
    for (int component = 0; component < components.component_count;
         ++component) {
      const OrderedFrontier frontier = order_frontier(
          moved, component_frontier(moved,
                                    components.component_of_triangle,
                                    component));
      const SubstitutedPolygon polygon = substitute_hull_arcs(
          moved, frontier, admission.arcs, component);
      D2Result containment;
      {
        CecrStageTimer d2_timer(
            timing != nullptr ? &timing->d2_ms : nullptr);
        containment = certify_patch_containment(
            moved, polygon, components.component_of_triangle,
            component, in_patch);
      }
      TENRYU_ASSERT(frontier.ok && containment.ok,
                    "certified CECR cavity changed during extraction");
      CecrCavity cavity;
      // With no admitted hull arc, polygon is the frontier cycle and both
      // edge lists remain empty, preserving the Stage-1 interior path.
      cavity.boundary_cycle = polygon.cycle;
      for (const HullArc& arc : admission.arcs) {
        if (arc.component != component) {
          continue;
        }
        const auto first = std::find(hull_change.h_old.begin(),
                                     hull_change.h_old.end(),
                                     arc.first_anchor);
        TENRYU_ASSERT(first != hull_change.h_old.end(),
                      "certified CECR old hull arc is missing its anchor");
        std::size_t position =
            static_cast<std::size_t>(first - hull_change.h_old.begin());
        std::size_t steps = 0;
        while (hull_change.h_old[position] != arc.second_anchor) {
          TENRYU_ASSERT(steps < hull_change.h_old.size(),
                        "certified CECR old hull arc does not terminate");
          const std::size_t next =
              (position + 1) % hull_change.h_old.size();
          cavity.old_arc_edges.emplace_back(hull_change.h_old[position],
                                            hull_change.h_old[next]);
          position = next;
          ++steps;
        }
        for (std::size_t index = 0; index + 1 < arc.new_span.size();
             ++index) {
          cavity.span_edges.emplace_back(arc.new_span[index],
                                         arc.new_span[index + 1]);
        }
      }
      cavity.interior_sites = containment.interior_sites;
      cavity.in_patch.assign(in_patch.size(), 0);
      for (TriangleId triangle = 0;
           triangle < static_cast<TriangleId>(moved.tri_edge.size());
           ++triangle) {
        if (components.component_of_triangle[triangle] == component) {
          cavity.in_patch[triangle] = 1;
        }
      }
      cavity.ok = true;
      result.cavities.push_back(std::move(cavity));
    }
    return result;
  }
}

}  // namespace tenryu::mesh::tess
