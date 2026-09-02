#include "mesh/voronoi_rz.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "mesh/poly_geom.cuh"

namespace tenryu::mesh::voronoi {
namespace {

struct Point {
  double r;
  double z;
};

enum class VertexProvenanceKind {
  kIncomplete,
  kDomainCorner,
  kBoundary,
  kInterior,
};

struct VertexProvenance {
  VertexProvenanceKind kind = VertexProvenanceKind::kIncomplete;
  std::array<std::uint64_t, 2> neighbor_ids{};
  std::size_t boundary_index = std::numeric_limits<std::size_t>::max();
};

struct LabeledPolygon {
  std::vector<Point> vertices;
  std::vector<VertexProvenance> vertex_provenance;
  // Entry i labels the edge from vertex i to vertex (i + 1) modulo n.
  std::vector<std::uint64_t> edge_labels;
  std::vector<std::size_t> boundary_segments;
};

struct Bisector {
  Point start;
  Point end;
};

bool use_slow_path_for_tests = false;

constexpr std::size_t kNoBoundarySegment =
    std::numeric_limits<std::size_t>::max();
constexpr std::size_t kMaximumValence = 8;

bool same_point(const Point& lhs, const Point& rhs) {
  return lhs.r == rhs.r && lhs.z == rhs.z;
}

double line_side_value(const Bisector& line, const Point& point) {
  return std::fma(line.end.r - line.start.r, point.z - line.start.z,
                  -((line.end.z - line.start.z) *
                    (point.r - line.start.r)));
}

int line_side_sign(const Bisector& line, const Point& point) {
  // The filtered exact predicate gives an epsilon-free closed-half-plane
  // classification. Points exactly on the bisector are retained; ascending
  // generator-id clip order is the deterministic authority for such ties.
  return reale::orient2d_sign(line.start.r, line.start.z,
                              line.end.r, line.end.z,
                              point.r, point.z);
}

Point segment_line_intersection(const Point& segment_start,
                                const Point& segment_end,
                                const Bisector& line,
                                const Point& retained_endpoint) {
  const double start_side = line_side_value(line, segment_start);
  const double end_side = line_side_value(line, segment_end);
  if (start_side == end_side) {
    // The exact orientation predicate can distinguish a crossing after both
    // floating side values round to zero. In double precision both endpoints
    // then lie on the line, so collapse the crossing to the retained endpoint.
    return retained_endpoint;
  }
  const double fraction = start_side / (start_side - end_side);
  return {
      std::fma(fraction, segment_end.r - segment_start.r, segment_start.r),
      std::fma(fraction, segment_end.z - segment_start.z, segment_start.z),
  };
}

Bisector perpendicular_bisector(const Generator& generator,
                                const Generator& other) {
  const double dr = other.r - generator.r;
  const double dz = other.z - generator.z;
  const Point midpoint{
      generator.r + 0.5 * dr,
      generator.z + 0.5 * dz,
  };
  // This orientation makes the generator side nonnegative.
  return {midpoint, {midpoint.r - dz, midpoint.z + dr}};
}

void append_vertex(std::vector<Point>& vertices,
                   std::vector<VertexProvenance>& vertex_provenance,
                   std::vector<std::uint64_t>& incoming_labels,
                   std::vector<std::size_t>& incoming_boundary_segments,
                   const Point& point,
                   const VertexProvenance& provenance,
                   const std::uint64_t incoming_label,
                   const std::size_t incoming_boundary_segment) {
  if (!vertices.empty() && same_point(vertices.back(), point)) {
    if (provenance.kind == VertexProvenanceKind::kDomainCorner ||
        vertex_provenance.back().kind ==
            VertexProvenanceKind::kIncomplete) {
      vertex_provenance.back() = provenance;
    }
    return;
  }
  vertices.push_back(point);
  vertex_provenance.push_back(provenance);
  incoming_labels.push_back(incoming_label);
  incoming_boundary_segments.push_back(incoming_boundary_segment);
}

VertexProvenance intersection_provenance(
    const std::uint64_t crossed_edge_label,
    const std::size_t crossed_boundary_segment,
    const std::uint64_t clip_label) {
  VertexProvenance provenance;
  if (crossed_edge_label == kBoundaryNeighbor) {
    if (crossed_boundary_segment == kNoBoundarySegment) {
      return provenance;
    }
    provenance.kind = VertexProvenanceKind::kBoundary;
    provenance.neighbor_ids[0] = clip_label;
    provenance.boundary_index = crossed_boundary_segment;
    return provenance;
  }
  if (crossed_edge_label == clip_label) {
    return provenance;
  }
  provenance.kind = VertexProvenanceKind::kInterior;
  provenance.neighbor_ids = {crossed_edge_label, clip_label};
  std::sort(provenance.neighbor_ids.begin(), provenance.neighbor_ids.end());
  return provenance;
}

LabeledPolygon clip_half_plane(const LabeledPolygon& input,
                               const Bisector& line,
                               const std::uint64_t clip_label) {
  LabeledPolygon result;
  if (input.vertices.empty()) {
    return result;
  }

  std::vector<bool> inside(input.vertices.size());
  bool any_outside = false;
  for (std::size_t vertex = 0; vertex < input.vertices.size(); ++vertex) {
    inside[vertex] = line_side_sign(line, input.vertices[vertex]) >= 0;
    any_outside = any_outside || !inside[vertex];
  }
  // A skipped spatial-hash candidate must be bitwise identical to an
  // all-pairs clip that does not remove a vertex.
  if (!any_outside) {
    return input;
  }

  std::vector<std::uint64_t> incoming_labels;
  std::vector<std::size_t> incoming_boundary_segments;
  result.vertices.reserve(input.vertices.size() + 1);
  result.vertex_provenance.reserve(input.vertices.size() + 1);
  incoming_labels.reserve(input.vertices.size() + 1);
  incoming_boundary_segments.reserve(input.vertices.size() + 1);

  for (std::size_t edge = 0; edge < input.vertices.size(); ++edge) {
    const std::size_t next = (edge + 1) % input.vertices.size();
    const Point& start = input.vertices[edge];
    const Point& end = input.vertices[next];
    const bool start_inside = inside[edge];
    const bool end_inside = inside[next];

    if (start_inside != end_inside) {
      const Point intersection =
          segment_line_intersection(start, end, line,
                                    start_inside ? start : end);
      const std::uint64_t incoming_label =
          start_inside ? input.edge_labels[edge] : clip_label;
      const std::size_t incoming_boundary_segment =
          start_inside ? input.boundary_segments[edge]
                       : kNoBoundarySegment;
      append_vertex(result.vertices, result.vertex_provenance,
                    incoming_labels, incoming_boundary_segments, intersection,
                    intersection_provenance(input.edge_labels[edge],
                                            input.boundary_segments[edge],
                                            clip_label),
                    incoming_label,
                    incoming_boundary_segment);
    }
    if (end_inside) {
      append_vertex(result.vertices, result.vertex_provenance,
                    incoming_labels, incoming_boundary_segments, end,
                    input.vertex_provenance[next],
                    input.edge_labels[edge], input.boundary_segments[edge]);
    }
  }

  if (result.vertices.size() > 1 &&
      same_point(result.vertices.front(), result.vertices.back())) {
    incoming_labels.front() = incoming_labels.back();
    incoming_boundary_segments.front() =
        incoming_boundary_segments.back();
    if (result.vertex_provenance.back().kind ==
            VertexProvenanceKind::kDomainCorner ||
        result.vertex_provenance.front().kind ==
            VertexProvenanceKind::kIncomplete) {
      result.vertex_provenance.front() =
          result.vertex_provenance.back();
    }
    result.vertices.pop_back();
    result.vertex_provenance.pop_back();
    incoming_labels.pop_back();
    incoming_boundary_segments.pop_back();
  }

  result.edge_labels.resize(result.vertices.size());
  result.boundary_segments.resize(result.vertices.size());
  for (std::size_t vertex = 0; vertex < result.vertices.size(); ++vertex) {
    const std::size_t next = (vertex + 1) % result.vertices.size();
    result.edge_labels[vertex] = incoming_labels[next];
    result.boundary_segments[vertex] = incoming_boundary_segments[next];
  }
  return result;
}

std::string reject_with_id(const char* reason, const std::uint64_t id) {
  return std::string(reason) + ": " + std::to_string(id);
}

std::string reject_duplicate(const std::uint64_t first_id,
                             const std::uint64_t second_id) {
  return std::string("duplicate_generator: ") +
         std::to_string(first_id) + " " + std::to_string(second_id);
}

Tessellation reject(const std::string& reason) {
  Tessellation result;
  result.reject_reason = reason;
  return result;
}

bool find_duplicate_generators(
    const std::vector<Generator>& sorted_generators,
    std::uint64_t& first_id,
    std::uint64_t& second_id) {
  std::vector<std::size_t> by_position(sorted_generators.size());
  for (std::size_t index = 0; index < by_position.size(); ++index) {
    by_position[index] = index;
  }
  std::sort(by_position.begin(), by_position.end(),
            [&sorted_generators](const std::size_t lhs_index,
                                 const std::size_t rhs_index) {
              const Generator& lhs = sorted_generators[lhs_index];
              const Generator& rhs = sorted_generators[rhs_index];
              const bool lhs_r_nan = std::isnan(lhs.r);
              const bool rhs_r_nan = std::isnan(rhs.r);
              if (lhs_r_nan != rhs_r_nan) {
                return !lhs_r_nan;
              }
              if (!lhs_r_nan && lhs.r != rhs.r) {
                return lhs.r < rhs.r;
              }
              const bool lhs_z_nan = std::isnan(lhs.z);
              const bool rhs_z_nan = std::isnan(rhs.z);
              if (lhs_z_nan != rhs_z_nan) {
                return !lhs_z_nan;
              }
              if (!lhs_z_nan && lhs.z != rhs.z) {
                return lhs.z < rhs.z;
              }
              if (lhs.id != rhs.id) {
                return lhs.id < rhs.id;
              }
              return lhs_index < rhs_index;
            });

  bool found = false;
  for (std::size_t index = 1; index < by_position.size(); ++index) {
    const Generator& first = sorted_generators[by_position[index - 1]];
    const Generator& second = sorted_generators[by_position[index]];
    if (first.r != second.r || first.z != second.z) {
      continue;
    }
    const std::pair<std::uint64_t, std::uint64_t> candidate{
        first.id, second.id};
    if (!found || candidate < std::pair{first_id, second_id}) {
      first_id = candidate.first;
      second_id = candidate.second;
      found = true;
    }
  }
  return found;
}

LabeledPolygon clip_against_candidates(
    const LabeledPolygon& start_polygon,
    const std::vector<Generator>& sorted_generators,
    std::size_t generator_index,
    const std::vector<std::size_t>& candidates) {
  LabeledPolygon cell_polygon = start_polygon;
  for (const std::size_t other_index : candidates) {
    if (other_index == generator_index) {
      continue;
    }
    const Generator& generator = sorted_generators[generator_index];
    const Generator& other = sorted_generators[other_index];
    cell_polygon = clip_half_plane(
        cell_polygon, perpendicular_bisector(generator, other), other.id);
    if (cell_polygon.vertices.size() < 3) {
      break;
    }
  }
  return cell_polygon;
}

void canonicalize_polygon(LabeledPolygon& polygon) {
  if (polygon.vertices.empty()) {
    return;
  }
  const auto first = std::min_element(
      polygon.vertices.begin(), polygon.vertices.end(),
      [](const Point& lhs, const Point& rhs) {
        return lhs.r < rhs.r || (lhs.r == rhs.r && lhs.z < rhs.z);
      });
  const std::size_t offset = static_cast<std::size_t>(
      std::distance(polygon.vertices.begin(), first));
  std::rotate(polygon.vertices.begin(), first, polygon.vertices.end());
  std::rotate(polygon.vertex_provenance.begin(),
              polygon.vertex_provenance.begin() + offset,
              polygon.vertex_provenance.end());
  std::rotate(polygon.edge_labels.begin(),
              polygon.edge_labels.begin() + offset,
              polygon.edge_labels.end());
  std::rotate(polygon.boundary_segments.begin(),
              polygon.boundary_segments.begin() + offset,
              polygon.boundary_segments.end());
}

bool canonical_circumcenter(
    const std::array<std::uint64_t, 3>& generator_ids,
    const std::vector<Generator>& sorted_generators,
    Point& center) {
  std::array<const Generator*, 3> generators{};
  for (std::size_t index = 0; index < generator_ids.size(); ++index) {
    const auto found = std::lower_bound(
        sorted_generators.begin(), sorted_generators.end(),
        generator_ids[index],
        [](const Generator& generator, const std::uint64_t id) {
          return generator.id < id;
        });
    if (found == sorted_generators.end() || found->id != generator_ids[index]) {
      return false;
    }
    generators[index] = &*found;
  }

  const Generator& a = *generators[0];
  const Generator& b = *generators[1];
  const Generator& c = *generators[2];
  const double br = b.r - a.r;
  const double bz = b.z - a.z;
  const double cr = c.r - a.r;
  const double cz = c.z - a.z;
  const double b_squared = std::fma(br, br, bz * bz);
  const double c_squared = std::fma(cr, cr, cz * cz);
  const double denominator = 2.0 * std::fma(br, cz, -(bz * cr));
  if (denominator == 0.0) {
    return false;
  }
  center = {
      std::fma(std::fma(cz, b_squared, -(bz * c_squared)) /
                   denominator,
               1.0, a.r),
      std::fma(std::fma(br, c_squared, -(cr * b_squared)) /
                   denominator,
               1.0, a.z),
  };
  return std::isfinite(center.r) && std::isfinite(center.z);
}

bool canonical_boundary_intersection(
    std::array<std::uint64_t, 2> generator_ids,
    const std::size_t boundary_segment,
    const std::vector<Generator>& sorted_generators,
    const LabeledPolygon& domain_polygon,
    Point& intersection) {
  std::sort(generator_ids.begin(), generator_ids.end());
  std::array<const Generator*, 2> generators{};
  for (std::size_t index = 0; index < generator_ids.size(); ++index) {
    const auto found = std::lower_bound(
        sorted_generators.begin(), sorted_generators.end(),
        generator_ids[index],
        [](const Generator& generator, const std::uint64_t id) {
          return generator.id < id;
        });
    if (found == sorted_generators.end() || found->id != generator_ids[index]) {
      return false;
    }
    generators[index] = &*found;
  }
  if (boundary_segment >= domain_polygon.vertices.size()) {
    return false;
  }

  const Bisector bisector =
      perpendicular_bisector(*generators[0], *generators[1]);
  const Point& start = domain_polygon.vertices[boundary_segment];
  const Point& end = domain_polygon.vertices[
      (boundary_segment + 1) % domain_polygon.vertices.size()];
  const double segment_r = end.r - start.r;
  const double segment_z = end.z - start.z;
  const double bisector_r = bisector.end.r - bisector.start.r;
  const double bisector_z = bisector.end.z - bisector.start.z;
  const double denominator =
      std::fma(segment_r, bisector_z, -(segment_z * bisector_r));
  if (denominator == 0.0) {
    return false;
  }
  const double fraction =
      std::fma(bisector.start.r - start.r, bisector_z,
               -((bisector.start.z - start.z) * bisector_r)) /
      denominator;
  intersection = {
      std::fma(fraction, segment_r, start.r),
      std::fma(fraction, segment_z, start.z),
  };
  return std::isfinite(intersection.r) && std::isfinite(intersection.z);
}

enum class NodeKeyKind : std::uint8_t {
  kDomainCorner,
  kBoundary,
  kInterior,
  kCollapsed,
};

struct NodeKey {
  NodeKeyKind kind = NodeKeyKind::kDomainCorner;
  std::array<std::uint64_t, 3> values{};
};

struct NodeKeyLess {
  bool operator()(const NodeKey& lhs, const NodeKey& rhs) const {
    if (lhs.kind != rhs.kind) {
      return lhs.kind < rhs.kind;
    }
    return lhs.values < rhs.values;
  }
};

class CanonicalNodeTable {
 public:
  CanonicalNodeTable(Tessellation& tessellation,
                     const std::vector<Generator>& sorted_generators,
                     const LabeledPolygon& domain_polygon)
      : tessellation_(tessellation),
        sorted_generators_(sorted_generators),
        domain_polygon_(domain_polygon) {}

  bool resolve(const VertexProvenance& provenance,
               const std::uint64_t generator_id,
               int& node_id) {
    NodeKey key;
    if (provenance.kind == VertexProvenanceKind::kDomainCorner) {
      if (provenance.boundary_index >= domain_polygon_.vertices.size()) {
        return false;
      }
      key.kind = NodeKeyKind::kDomainCorner;
      key.values[0] = provenance.boundary_index;
      return find_or_insert(
          key, domain_polygon_.vertices[provenance.boundary_index], node_id);
    }
    if (provenance.kind == VertexProvenanceKind::kBoundary) {
      const std::uint64_t neighbor_id = provenance.neighbor_ids[0];
      if (neighbor_id == generator_id ||
          neighbor_id == kBoundaryNeighbor ||
          provenance.boundary_index >= domain_polygon_.vertices.size()) {
        return false;
      }
      std::array<std::uint64_t, 2> generator_ids{
          generator_id, neighbor_id};
      std::sort(generator_ids.begin(), generator_ids.end());
      key.kind = NodeKeyKind::kBoundary;
      key.values = {generator_ids[0], generator_ids[1],
                    provenance.boundary_index};
      const auto found = nodes_.find(key);
      if (found != nodes_.end()) {
        node_id = found->second;
        return true;
      }
      Point intersection{};
      if (!canonical_boundary_intersection(
              generator_ids, provenance.boundary_index,
              sorted_generators_, domain_polygon_, intersection)) {
        return false;
      }
      const Point& start =
          domain_polygon_.vertices[provenance.boundary_index];
      const Point& end = domain_polygon_.vertices[
          (provenance.boundary_index + 1) %
          domain_polygon_.vertices.size()];
      if (start.r == 0.0 && end.r == 0.0) {
        intersection.r = 0.0;
      }
      return insert(key, intersection, node_id);
    }
    if (provenance.kind == VertexProvenanceKind::kInterior) {
      std::array<std::uint64_t, 3> generator_ids{
          generator_id, provenance.neighbor_ids[0],
          provenance.neighbor_ids[1]};
      std::sort(generator_ids.begin(), generator_ids.end());
      if (generator_ids[0] == generator_ids[1] ||
          generator_ids[1] == generator_ids[2] ||
          generator_ids[2] == kBoundaryNeighbor) {
        return false;
      }
      key.kind = NodeKeyKind::kInterior;
      key.values = generator_ids;
      const auto found = nodes_.find(key);
      if (found != nodes_.end()) {
        node_id = found->second;
        return true;
      }
      Point center{};
      if (!canonical_circumcenter(
              generator_ids, sorted_generators_, center)) {
        return false;
      }
      return insert(key, center, node_id);
    }
    return false;
  }

  std::pair<int, bool> append_collapsed(const int first,
                                         const int second) {
    const int low = std::min(first, second);
    const int high = std::max(first, second);
    const NodeKey key{
        NodeKeyKind::kCollapsed,
        {static_cast<std::uint64_t>(low),
         static_cast<std::uint64_t>(high), 0}};
    const auto found = nodes_.find(key);
    if (found != nodes_.end()) {
      return {found->second, false};
    }
    const Point first_point = point(first);
    const Point second_point = point(second);
    const Point midpoint{
        0.5 * (first_point.r + second_point.r),
        0.5 * (first_point.z + second_point.z),
    };
    int node_id = -1;
    if (!insert(key, midpoint, node_id)) {
      return {-1, false};
    }
    node_parent_ids_[static_cast<std::size_t>(node_id)] = {low, high};
    return {node_id, true};
  }

  void remove_last_collapsed(const int first,
                             const int second,
                             const int node_id) {
    if (node_id < 0 ||
        static_cast<std::size_t>(node_id + 1) !=
            tessellation_.node_r.size()) {
      return;
    }
    const int low = std::min(first, second);
    const int high = std::max(first, second);
    nodes_.erase({NodeKeyKind::kCollapsed,
                  {static_cast<std::uint64_t>(low),
                   static_cast<std::uint64_t>(high), 0}});
    tessellation_.node_r.pop_back();
    tessellation_.node_z.pop_back();
    node_key_kinds_.pop_back();
    node_keys_.pop_back();
    node_parent_ids_.pop_back();
  }

  Point point(const int node_id) const {
    return {tessellation_.node_r[static_cast<std::size_t>(node_id)],
            tessellation_.node_z[static_cast<std::size_t>(node_id)]};
  }

  NodeKeyKind node_key_kind(const int node_id) const {
    return node_key_kinds_[static_cast<std::size_t>(node_id)];
  }

  bool collapsed_parents(
      int node_id, int& parent0, int& parent1) const {
    if (node_id < 0 ||
        static_cast<std::size_t>(node_id) >= node_key_kinds_.size() ||
        node_key_kinds_[static_cast<std::size_t>(node_id)] !=
            NodeKeyKind::kCollapsed) {
      return false;
    }
    const std::array<int, 2>& parents =
        node_parent_ids_[static_cast<std::size_t>(node_id)];
    parent0 = parents[0];
    parent1 = parents[1];
    return true;
  }

  bool interior_generator_triple(
      const int node_id,
      std::array<std::uint64_t, 3>& generator_ids) const {
    if (node_id < 0 ||
        static_cast<std::size_t>(node_id) >= node_keys_.size()) {
      return false;
    }
    const NodeKey& key = node_keys_[static_cast<std::size_t>(node_id)];
    if (key.kind != NodeKeyKind::kInterior) {
      return false;
    }
    generator_ids = key.values;
    return true;
  }

  bool has_boundary_class_provenance(const int node_id) const {
    if (node_id < 0 ||
        static_cast<std::size_t>(node_id) >= node_key_kinds_.size()) {
      return false;
    }
    const NodeKeyKind kind =
        node_key_kinds_[static_cast<std::size_t>(node_id)];
    if (kind == NodeKeyKind::kDomainCorner ||
        kind == NodeKeyKind::kBoundary) {
      return true;
    }
    if (kind != NodeKeyKind::kCollapsed) {
      return false;
    }
    const std::array<int, 2>& parents =
        node_parent_ids_[static_cast<std::size_t>(node_id)];
    if (parents[0] < 0 || parents[0] >= node_id ||
        parents[1] < 0 || parents[1] >= node_id) {
      return false;
    }
    return has_boundary_class_provenance(parents[0]) &&
           has_boundary_class_provenance(parents[1]);
  }

 private:
  bool find_or_insert(const NodeKey& key,
                      const Point& point,
                      int& node_id) {
    const auto found = nodes_.find(key);
    if (found != nodes_.end()) {
      node_id = found->second;
      return true;
    }
    return insert(key, point, node_id);
  }

  bool insert(const NodeKey& key, const Point& point, int& node_id) {
    if (tessellation_.node_r.size() >=
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return false;
    }
    node_id = static_cast<int>(tessellation_.node_r.size());
    tessellation_.node_r.push_back(point.r);
    tessellation_.node_z.push_back(point.z);
    node_key_kinds_.push_back(key.kind);
    node_keys_.push_back(key);
    node_parent_ids_.push_back({-1, -1});
    nodes_.emplace(key, node_id);
    return true;
  }

  Tessellation& tessellation_;
  const std::vector<Generator>& sorted_generators_;
  const LabeledPolygon& domain_polygon_;
  std::map<NodeKey, int, NodeKeyLess> nodes_;
  std::vector<NodeKeyKind> node_key_kinds_;
  std::vector<NodeKey> node_keys_;
  std::vector<std::array<int, 2>> node_parent_ids_;
};

Point node_point(const Tessellation& tessellation, const int node_id) {
  return {tessellation.node_r[static_cast<std::size_t>(node_id)],
          tessellation.node_z[static_cast<std::size_t>(node_id)]};
}

bool point_on_segment(const Point& point,
                      const Point& start,
                      const Point& end) {
  return reale::orient2d_sign(start.r, start.z, end.r, end.z,
                              point.r, point.z) == 0 &&
         point.r >= std::min(start.r, end.r) &&
         point.r <= std::max(start.r, end.r) &&
         point.z >= std::min(start.z, end.z) &&
         point.z <= std::max(start.z, end.z);
}

bool lies_exactly_on_domain_boundary(
    const Point& point,
    const LabeledPolygon& domain_polygon) {
  for (std::size_t segment = 0;
       segment < domain_polygon.vertices.size(); ++segment) {
    const Point& start = domain_polygon.vertices[segment];
    const Point& end = domain_polygon.vertices[
        (segment + 1) % domain_polygon.vertices.size()];
    if (reale::orient2d_sign(start.r, start.z, end.r, end.z,
                             point.r, point.z) == 0 &&
        point.r >= std::min(start.r, end.r) &&
        point.r <= std::max(start.r, end.r) &&
        point.z >= std::min(start.z, end.z) &&
        point.z <= std::max(start.z, end.z)) {
      return true;
    }
  }
  return false;
}

bool segments_intersect(const Point& a0,
                        const Point& a1,
                        const Point& b0,
                        const Point& b1) {
  const int b0_side = reale::orient2d_sign(
      a0.r, a0.z, a1.r, a1.z, b0.r, b0.z);
  const int b1_side = reale::orient2d_sign(
      a0.r, a0.z, a1.r, a1.z, b1.r, b1.z);
  const int a0_side = reale::orient2d_sign(
      b0.r, b0.z, b1.r, b1.z, a0.r, a0.z);
  const int a1_side = reale::orient2d_sign(
      b0.r, b0.z, b1.r, b1.z, a1.r, a1.z);
  if (b0_side == 0 && point_on_segment(b0, a0, a1)) {
    return true;
  }
  if (b1_side == 0 && point_on_segment(b1, a0, a1)) {
    return true;
  }
  if (a0_side == 0 && point_on_segment(a0, b0, b1)) {
    return true;
  }
  if (a1_side == 0 && point_on_segment(a1, b0, b1)) {
    return true;
  }
  return (b0_side < 0) != (b1_side < 0) &&
         (a0_side < 0) != (a1_side < 0);
}

bool polygon_is_simple(const VoronoiCell& cell,
                       const Tessellation& tessellation) {
  const std::size_t count = cell.node_indices.size();
  if (count < 3 ||
      cell.neighbor_ids.size() != count) {
    return false;
  }
  for (std::size_t first = 0; first < count; ++first) {
    const Point first_point = node_point(
        tessellation, cell.node_indices[first]);
    for (std::size_t second = first + 1; second < count; ++second) {
      if (same_point(first_point,
                     node_point(tessellation, cell.node_indices[second]))) {
        return false;
      }
    }
  }
  for (std::size_t first = 0; first < count; ++first) {
    const std::size_t first_next = (first + 1) % count;
    const Point a0 = node_point(tessellation, cell.node_indices[first]);
    const Point a1 = node_point(tessellation, cell.node_indices[first_next]);
    for (std::size_t second = first + 1; second < count; ++second) {
      const std::size_t second_next = (second + 1) % count;
      if (first_next == second || second_next == first) {
        continue;
      }
      const Point b0 = node_point(tessellation, cell.node_indices[second]);
      const Point b1 = node_point(tessellation, cell.node_indices[second_next]);
      if (segments_intersect(a0, a1, b0, b1)) {
        return false;
      }
    }
  }
  return true;
}

void canonicalize_cell(VoronoiCell& cell,
                       std::vector<std::size_t>& boundary_segments) {
  if (cell.node_indices.empty()) {
    return;
  }
  std::size_t first = 0;
  for (std::size_t vertex = 1; vertex < cell.node_indices.size(); ++vertex) {
    if (cell.node_indices[vertex] < cell.node_indices[first]) {
      first = vertex;
    }
  }
  std::rotate(cell.node_indices.begin(),
              cell.node_indices.begin() + first,
              cell.node_indices.end());
  std::rotate(cell.neighbor_ids.begin(),
              cell.neighbor_ids.begin() + first,
              cell.neighbor_ids.end());
  std::rotate(boundary_segments.begin(),
              boundary_segments.begin() + first,
              boundary_segments.end());
}

void collapse_cell_edge(VoronoiCell& cell,
                        std::vector<std::size_t>& boundary_segments,
                        const int first,
                        const int second,
                        const int midpoint) {
  const std::size_t count = cell.node_indices.size();
  std::vector<int> replaced = cell.node_indices;
  for (std::size_t vertex = 0; vertex < count; ++vertex) {
    if (replaced[vertex] == first || replaced[vertex] == second) {
      replaced[vertex] = midpoint;
    }
  }

  std::vector<int> collapsed_nodes;
  std::vector<std::uint64_t> collapsed_neighbors;
  std::vector<std::size_t> collapsed_boundary_segments;
  collapsed_nodes.reserve(count);
  collapsed_neighbors.reserve(count);
  collapsed_boundary_segments.reserve(count);
  for (std::size_t edge = 0; edge < count; ++edge) {
    const std::size_t next = (edge + 1) % count;
    if (replaced[edge] == replaced[next]) {
      continue;
    }
    collapsed_nodes.push_back(replaced[edge]);
    collapsed_neighbors.push_back(cell.neighbor_ids[edge]);
    collapsed_boundary_segments.push_back(boundary_segments[edge]);
  }
  cell.node_indices = std::move(collapsed_nodes);
  cell.neighbor_ids = std::move(collapsed_neighbors);
  boundary_segments = std::move(collapsed_boundary_segments);
  canonicalize_cell(cell, boundary_segments);
}

struct CollapseEdge {
  int first = -1;
  int second = -1;
  double length = 0.0;
};

CollapseEdge make_collapse_edge(const int first,
                                const int second,
                                const Tessellation& tessellation) {
  const int low = std::min(first, second);
  const int high = std::max(first, second);
  const Point low_point = node_point(tessellation, low);
  const Point high_point = node_point(tessellation, high);
  return {low, high,
          std::hypot(high_point.r - low_point.r,
                     high_point.z - low_point.z)};
}

bool collapse_edge_less(const CollapseEdge& lhs, const CollapseEdge& rhs) {
  if (lhs.length != rhs.length) {
    return lhs.length < rhs.length;
  }
  return std::pair{lhs.first, lhs.second} <
         std::pair{rhs.first, rhs.second};
}

bool same_collapse_edge(const CollapseEdge& lhs, const CollapseEdge& rhs) {
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

using CellIncidence = std::map<int, std::vector<std::size_t>>;

void add_cell_incidence(CellIncidence& incidence,
                        const VoronoiCell& cell,
                        const std::size_t cell_index) {
  for (const int node_id : cell.node_indices) {
    std::vector<std::size_t>& owners = incidence[node_id];
    if (owners.empty() || owners.back() != cell_index) {
      owners.push_back(cell_index);
    }
  }
}

CellIncidence build_cell_incidence(const std::vector<VoronoiCell>& cells) {
  CellIncidence incidence;
  for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    add_cell_incidence(incidence, cells[cell_index], cell_index);
  }
  return incidence;
}

std::vector<std::size_t> boundary_membership(
    const int node_id,
    const std::vector<VoronoiCell>& cells,
    const std::vector<std::vector<std::size_t>>& boundary_segments,
    const CellIncidence& incidence) {
  std::vector<std::size_t> segments;
  const auto owners = incidence.find(node_id);
  if (owners == incidence.end()) {
    return segments;
  }
  for (const std::size_t cell_index : owners->second) {
    const VoronoiCell& cell = cells[cell_index];
    for (std::size_t vertex = 0; vertex < cell.node_indices.size(); ++vertex) {
      if (node_id != cell.node_indices[vertex]) {
        continue;
      }
      const std::size_t incoming =
          boundary_segments[cell_index][
              (vertex + cell.node_indices.size() - 1) %
              cell.node_indices.size()];
      const std::size_t outgoing = boundary_segments[cell_index][vertex];
      if (incoming != kNoBoundarySegment) {
        segments.push_back(incoming);
      }
      if (outgoing != kNoBoundarySegment) {
        segments.push_back(outgoing);
      }
    }
  }
  std::sort(segments.begin(), segments.end());
  segments.erase(std::unique(segments.begin(), segments.end()),
                 segments.end());
  return segments;
}

bool shares_boundary_segment(const std::vector<std::size_t>& first,
                             const std::vector<std::size_t>& second) {
  for (const std::size_t first_segment : first) {
    if (std::binary_search(second.begin(), second.end(), first_segment)) {
      return true;
    }
  }
  return false;
}

bool collapse_edge_is_eligible(const CollapseEdge& edge,
                               const Tessellation& tessellation,
                               const std::vector<VoronoiCell>& cells,
                               const std::vector<std::vector<std::size_t>>&
                                   boundary_segments,
                               const CellIncidence& incidence) {
  if (tessellation.node_r[static_cast<std::size_t>(edge.first)] == 0.0 &&
      tessellation.node_r[static_cast<std::size_t>(edge.second)] == 0.0) {
    return false;
  }
  // Boundary vertices may move together only when they share an original
  // domain segment. This permits boundary-chain shortening without rounding
  // an edge across a domain corner.
  const std::vector<std::size_t> first = boundary_membership(
      edge.first, cells, boundary_segments, incidence);
  const std::vector<std::size_t> second = boundary_membership(
      edge.second, cells, boundary_segments, incidence);
  return first.empty() || second.empty() ||
         shares_boundary_segment(first, second);
}

std::vector<std::size_t> affected_cell_indices(
    const CollapseEdge& edge, const CellIncidence& incidence) {
  std::vector<std::size_t> affected;
  const auto first = incidence.find(edge.first);
  if (first != incidence.end()) {
    affected.insert(affected.end(),
                    first->second.begin(), first->second.end());
  }
  const auto second = incidence.find(edge.second);
  if (second != incidence.end()) {
    affected.insert(affected.end(),
                    second->second.begin(), second->second.end());
  }
  std::sort(affected.begin(), affected.end());
  affected.erase(std::unique(affected.begin(), affected.end()),
                 affected.end());
  return affected;
}

std::vector<CollapseEdge> collect_collapse_edges(
    const Tessellation& tessellation) {
  std::vector<CollapseEdge> edges;
  for (const VoronoiCell& cell : tessellation.cells) {
    bool has_snap_noise_edge = false;
    for (std::size_t edge = 0; edge < cell.node_indices.size(); ++edge) {
      const std::size_t next = (edge + 1) % cell.node_indices.size();
      const std::size_t node_a =
          static_cast<std::size_t>(cell.node_indices[edge]);
      const std::size_t node_b =
          static_cast<std::size_t>(cell.node_indices[next]);
      const double r_a = tessellation.node_r[node_a];
      const double z_a = tessellation.node_z[node_a];
      const double r_b = tessellation.node_r[node_b];
      const double z_b = tessellation.node_z[node_b];
      const double edge_length = std::hypot(r_b - r_a, z_b - z_a);
      const double scale = std::max(
          {std::abs(r_a), std::abs(z_a), std::abs(r_b), std::abs(z_b)});
      // Edges below ~4 ulp of the endpoint coordinate magnitude are canonical-snap noise, not resolved geometry -- the same 1-ulp-per-coordinate premise as the remap's snap_additivity_bound; no fitted constants.
      if (edge_length <= 4.0 * std::numeric_limits<double>::epsilon() *
                             scale) {
        has_snap_noise_edge = true;
        break;
      }
    }
    if (cell.node_indices.size() <= kMaximumValence &&
        polygon_is_simple(cell, tessellation) && !has_snap_noise_edge) {
      continue;
    }
    edges.reserve(edges.size() + cell.node_indices.size());
    for (std::size_t edge = 0; edge < cell.node_indices.size(); ++edge) {
      const std::size_t next =
          (edge + 1) % cell.node_indices.size();
      edges.push_back(make_collapse_edge(
          cell.node_indices[edge], cell.node_indices[next], tessellation));
    }
  }
  std::sort(edges.begin(), edges.end(), collapse_edge_less);
  edges.erase(std::unique(edges.begin(), edges.end(), same_collapse_edge),
              edges.end());
  return edges;
}

std::string collapse_reject_reason(const char* reason,
                                   const std::uint64_t generator_id) {
  return std::string(reason) + ": " + std::to_string(generator_id);
}

static bool all_cells_are_simple(const Tessellation& tessellation,
                                 std::string& reject_reason) {
  for (std::size_t cell_index = 0;
       cell_index < tessellation.cells.size(); ++cell_index) {
    const VoronoiCell& cell = tessellation.cells[cell_index];
    if (!polygon_is_simple(cell, tessellation)) {
      reject_reason = collapse_reject_reason(
          "nonsimple_cell_unresolved", cell.generator_id);
      return false;
    }
  }
  return true;
}

struct TrialCell {
  std::size_t index = 0;
  VoronoiCell cell;
  std::vector<std::size_t> boundary_segments;
};

struct SharedEdgeOwner {
  std::size_t cell_index = 0;
  std::size_t edge_index = 0;
};

using SharedEdge = std::pair<int, int>;
using SharedEdgeOwners = std::map<SharedEdge, std::vector<SharedEdgeOwner>>;

SharedEdge make_shared_edge(const int first, const int second) {
  return std::minmax(first, second);
}

SharedEdgeOwners build_shared_edge_owners(
    const std::vector<VoronoiCell>& cells) {
  SharedEdgeOwners owners;
  for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    const VoronoiCell& cell = cells[cell_index];
    for (std::size_t edge = 0; edge < cell.node_indices.size(); ++edge) {
      const std::size_t next =
          (edge + 1) % cell.node_indices.size();
      owners[make_shared_edge(cell.node_indices[edge],
                              cell.node_indices[next])]
          .push_back({cell_index, edge});
    }
  }
  return owners;
}

bool rebuild_neighbors_and_check_reciprocity(
    Tessellation& tessellation,
    const CanonicalNodeTable& node_table,
    const LabeledPolygon& domain_polygon) {
  std::vector<VoronoiCell>& cells = tessellation.cells;
  const SharedEdgeOwners owners = build_shared_edge_owners(cells);
  for (VoronoiCell& cell : cells) {
    cell.neighbor_ids.assign(cell.node_indices.size(), kBoundaryNeighbor);
  }
  for (const auto& [edge, edge_owners] : owners) {
    (void)edge;
    if (edge_owners.size() != 2 ||
        edge_owners[0].cell_index == edge_owners[1].cell_index) {
      continue;
    }
    const SharedEdgeOwner& first = edge_owners[0];
    const SharedEdgeOwner& second = edge_owners[1];
    cells[first.cell_index].neighbor_ids[first.edge_index] =
        cells[second.cell_index].generator_id;
    cells[second.cell_index].neighbor_ids[second.edge_index] =
        cells[first.cell_index].generator_id;
  }

  for (std::size_t cell_index = 0; cell_index < cells.size(); ++cell_index) {
    const VoronoiCell& cell = cells[cell_index];
    if (cell.neighbor_ids.size() != cell.node_indices.size()) {
      return false;
    }
    for (std::size_t edge = 0; edge < cell.node_indices.size(); ++edge) {
      const std::uint64_t neighbor_id = cell.neighbor_ids[edge];
      if (neighbor_id == kBoundaryNeighbor) {
        const std::size_t next =
            (edge + 1) % cell.node_indices.size();
        const int first_node = cell.node_indices[edge];
        const int second_node = cell.node_indices[next];
        if (!(node_table.has_boundary_class_provenance(first_node) ||
              lies_exactly_on_domain_boundary(
                  node_point(tessellation, first_node), domain_polygon)) ||
            !(node_table.has_boundary_class_provenance(second_node) ||
              lies_exactly_on_domain_boundary(
                  node_point(tessellation, second_node), domain_polygon))) {
          static bool audit_first_fail_reported = false;
          if (!audit_first_fail_reported) {
            audit_first_fail_reported = true;
            const NodeKeyKind first_kind =
                node_table.node_key_kind(first_node);
            const NodeKeyKind second_kind =
                node_table.node_key_kind(second_node);
            const Point first_point =
                node_point(tessellation, first_node);
            const Point second_point =
                node_point(tessellation, second_node);
            std::array<std::uint64_t, 3> first_triple{};
            std::array<std::uint64_t, 3> second_triple{};
            char first_triple_text[128] = {};
            char second_triple_text[128] = {};
            if (node_table.interior_generator_triple(
                    first_node, first_triple)) {
              std::snprintf(
                  first_triple_text, sizeof(first_triple_text),
                  " first_key_generator_ids=(%llu,%llu,%llu)",
                  static_cast<unsigned long long>(first_triple[0]),
                  static_cast<unsigned long long>(first_triple[1]),
                  static_cast<unsigned long long>(first_triple[2]));
            }
            if (node_table.interior_generator_triple(
                    second_node, second_triple)) {
              std::snprintf(
                  second_triple_text, sizeof(second_triple_text),
                  " second_key_generator_ids=(%llu,%llu,%llu)",
                  static_cast<unsigned long long>(second_triple[0]),
                  static_cast<unsigned long long>(second_triple[1]),
                  static_cast<unsigned long long>(second_triple[2]));
            }
            std::fprintf(
                stderr,
                "[audit_first_fail] cell_generator_id=%llu "
                "endpoint_node_ids=(%d,%d) endpoint_kinds=(%d,%d) "
                "endpoint_coordinates=((%.17g,%.17g),(%.17g,%.17g))%s%s\n",
                static_cast<unsigned long long>(cell.generator_id),
                first_node, second_node, static_cast<int>(first_kind),
                static_cast<int>(second_kind), first_point.r, first_point.z,
                second_point.r, second_point.z, first_triple_text,
                second_triple_text);
            const auto print_collapsed_parents =
                [&node_table](const auto& self, const int node_id) -> void {
              int parent0 = -1;
              int parent1 = -1;
              if (!node_table.collapsed_parents(
                      node_id, parent0, parent1)) {
                return;
              }
              std::fprintf(
                  stderr, "[audit_parents] node=%d parents=(%d,%d)\n",
                  node_id, parent0, parent1);
              self(self, parent0);
              self(self, parent1);
            };
            if (first_kind == NodeKeyKind::kCollapsed) {
              print_collapsed_parents(
                  print_collapsed_parents, first_node);
            }
            if (second_kind == NodeKeyKind::kCollapsed) {
              print_collapsed_parents(
                  print_collapsed_parents, second_node);
            }
            for (const VoronoiCell& audit_cell : tessellation.cells) {
              if (std::find(audit_cell.node_indices.begin(),
                            audit_cell.node_indices.end(), first_node) ==
                      audit_cell.node_indices.end() &&
                  std::find(audit_cell.node_indices.begin(),
                            audit_cell.node_indices.end(), second_node) ==
                      audit_cell.node_indices.end()) {
                continue;
              }
              std::fprintf(
                  stderr, "[audit_cell] generator_id=%llu node_indices=",
                  static_cast<unsigned long long>(
                      audit_cell.generator_id));
              for (std::size_t vertex = 0;
                   vertex < audit_cell.node_indices.size(); ++vertex) {
                const int node_id = audit_cell.node_indices[vertex];
                const Point point = node_point(tessellation, node_id);
                std::fprintf(
                    stderr, "%s%d(%d)@(%.17g,%.17g)",
                    vertex == 0 ? "" : " ", node_id,
                    static_cast<int>(node_table.node_key_kind(node_id)),
                    point.r, point.z);
              }
              std::fprintf(stderr, "\n");
            }
          }
          return false;
        }
        continue;
      }
      const std::size_t next =
          (edge + 1) % cell.node_indices.size();
      const auto found = owners.find(
          make_shared_edge(cell.node_indices[edge],
                           cell.node_indices[next]));
      if (found == owners.end() || found->second.size() != 2) {
        return false;
      }
      const SharedEdgeOwner* reciprocal = nullptr;
      for (const SharedEdgeOwner& owner : found->second) {
        if (owner.cell_index == cell_index && owner.edge_index == edge) {
          continue;
        }
        reciprocal = &owner;
      }
      if (reciprocal == nullptr ||
          cells[reciprocal->cell_index].generator_id != neighbor_id ||
          cells[reciprocal->cell_index]
                  .neighbor_ids[reciprocal->edge_index] != cell.generator_id) {
        return false;
      }
    }
  }
  return true;
}

bool bitwise_same_point(const Tessellation& tessellation,
                        const int first,
                        const int second) {
  const std::size_t first_index = static_cast<std::size_t>(first);
  const std::size_t second_index = static_cast<std::size_t>(second);
  return std::bit_cast<std::uint64_t>(tessellation.node_r[first_index]) ==
             std::bit_cast<std::uint64_t>(
                 tessellation.node_r[second_index]) &&
         std::bit_cast<std::uint64_t>(tessellation.node_z[first_index]) ==
             std::bit_cast<std::uint64_t>(
                 tessellation.node_z[second_index]);
}

int representative(std::vector<int>& parent, const int node_id) {
  int root = node_id;
  while (parent[static_cast<std::size_t>(root)] != root) {
    root = parent[static_cast<std::size_t>(root)];
  }
  int current = node_id;
  while (parent[static_cast<std::size_t>(current)] != current) {
    const int next = parent[static_cast<std::size_t>(current)];
    parent[static_cast<std::size_t>(current)] = root;
    current = next;
  }
  return root;
}

void merge_exact_zero_length_edges(
    Tessellation& tessellation,
    std::vector<std::vector<std::size_t>>& boundary_segments) {
  std::vector<int> parent(tessellation.node_r.size());
  for (std::size_t node = 0; node < parent.size(); ++node) {
    parent[node] = static_cast<int>(node);
  }
  for (const VoronoiCell& cell : tessellation.cells) {
    for (std::size_t edge = 0; edge < cell.node_indices.size(); ++edge) {
      const std::size_t next =
          (edge + 1) % cell.node_indices.size();
      const int first = cell.node_indices[edge];
      const int second = cell.node_indices[next];
      if (first == second ||
          !bitwise_same_point(tessellation, first, second)) {
        continue;
      }
      const int first_root = representative(parent, first);
      const int second_root = representative(parent, second);
      parent[static_cast<std::size_t>(std::max(first_root, second_root))] =
          std::min(first_root, second_root);
    }
  }
  for (std::size_t node = 0; node < parent.size(); ++node) {
    parent[node] = representative(parent, static_cast<int>(node));
  }

  for (std::size_t cell_index = 0;
       cell_index < tessellation.cells.size(); ++cell_index) {
    VoronoiCell& cell = tessellation.cells[cell_index];
    std::vector<int> replaced = cell.node_indices;
    for (int& node_id : replaced) {
      node_id = parent[static_cast<std::size_t>(node_id)];
    }
    std::vector<int> merged_nodes;
    std::vector<std::uint64_t> merged_neighbors;
    std::vector<std::size_t> merged_boundary_segments;
    merged_nodes.reserve(replaced.size());
    merged_neighbors.reserve(replaced.size());
    merged_boundary_segments.reserve(replaced.size());
    for (std::size_t edge = 0; edge < replaced.size(); ++edge) {
      const std::size_t next = (edge + 1) % replaced.size();
      if (replaced[edge] == replaced[next]) {
        continue;
      }
      merged_nodes.push_back(replaced[edge]);
      merged_neighbors.push_back(cell.neighbor_ids[edge]);
      merged_boundary_segments.push_back(
          boundary_segments[cell_index][edge]);
    }
    cell.node_indices = std::move(merged_nodes);
    cell.neighbor_ids = std::move(merged_neighbors);
    boundary_segments[cell_index] =
        std::move(merged_boundary_segments);
    canonicalize_cell(cell, boundary_segments[cell_index]);
  }
}

void unify_exact_cocircular_interior_nodes(
    Tessellation& tessellation,
    const CanonicalNodeTable& node_table,
    const std::vector<Generator>& sorted_generators) {
  using GeneratorPair = std::array<std::uint64_t, 2>;
  using GeneratorTriple = std::array<std::uint64_t, 3>;

  std::map<GeneratorPair, std::vector<int>> pair_buckets;
  std::vector<GeneratorTriple> node_triples(tessellation.node_r.size());
  for (std::size_t node = 0; node < node_triples.size(); ++node) {
    GeneratorTriple triple{};
    if (!node_table.interior_generator_triple(
            static_cast<int>(node), triple)) {
      continue;
    }
    node_triples[node] = triple;
    pair_buckets[{triple[0], triple[1]}].push_back(
        static_cast<int>(node));
    pair_buckets[{triple[0], triple[2]}].push_back(
        static_cast<int>(node));
    pair_buckets[{triple[1], triple[2]}].push_back(
        static_cast<int>(node));
  }

  std::vector<int> parent(tessellation.node_r.size());
  for (std::size_t node = 0; node < parent.size(); ++node) {
    parent[node] = static_cast<int>(node);
  }
  const auto canonical_less = [&](const int lhs, const int rhs) {
    const GeneratorTriple& lhs_triple =
        node_triples[static_cast<std::size_t>(lhs)];
    const GeneratorTriple& rhs_triple =
        node_triples[static_cast<std::size_t>(rhs)];
    if (lhs_triple != rhs_triple) {
      return lhs_triple < rhs_triple;
    }
    return lhs < rhs;
  };
  const auto third_generator = [](const GeneratorTriple& triple,
                                  const GeneratorPair& pair) {
    for (const std::uint64_t generator_id : triple) {
      if (generator_id != pair[0] && generator_id != pair[1]) {
        return generator_id;
      }
    }
    return kBoundaryNeighbor;
  };
  const auto generator_for_id = [&](const std::uint64_t generator_id)
      -> const Generator& {
    return *std::lower_bound(
        sorted_generators.begin(), sorted_generators.end(), generator_id,
        [](const Generator& generator, const std::uint64_t id) {
          return generator.id < id;
        });
  };

  for (const auto& [pair, nodes] : pair_buckets) {
    for (std::size_t first_index = 0;
         first_index < nodes.size(); ++first_index) {
      const int first = nodes[first_index];
      const std::uint64_t third_first = third_generator(
          node_triples[static_cast<std::size_t>(first)], pair);
      for (std::size_t second_index = first_index + 1;
           second_index < nodes.size(); ++second_index) {
        const int second = nodes[second_index];
        const std::uint64_t third_second = third_generator(
            node_triples[static_cast<std::size_t>(second)], pair);
        if (third_first == third_second) {
          continue;
        }
        const Generator& a = generator_for_id(pair[0]);
        const Generator& b = generator_for_id(pair[1]);
        const Generator& c = generator_for_id(third_first);
        const Generator& d = generator_for_id(third_second);
        // Argument convention: the ascending shared pair (a,b), followed by
        // node A's third generator c and node B's third generator d.
        if (reale::incircle_sign(
                a.r, a.z, b.r, b.z, c.r, c.z, d.r, d.z) != 0) {
          continue;
        }
        int first_root = representative(parent, first);
        int second_root = representative(parent, second);
        if (first_root == second_root) {
          continue;
        }
        if (canonical_less(second_root, first_root)) {
          std::swap(first_root, second_root);
        }
        parent[static_cast<std::size_t>(second_root)] = first_root;
      }
    }
  }
  for (std::size_t node = 0; node < parent.size(); ++node) {
    parent[node] = representative(parent, static_cast<int>(node));
  }
  for (VoronoiCell& cell : tessellation.cells) {
    for (int& node_id : cell.node_indices) {
      node_id = parent[static_cast<std::size_t>(node_id)];
    }
  }
}

bool unify_bitwise_duplicate_nodes(
    Tessellation& tessellation,
    std::vector<std::vector<std::size_t>>& boundary_segments,
    std::uint64_t& degenerate_generator_id) {
  std::map<std::pair<double, double>, int> canonical_ids;
  std::vector<int> canonical_node_ids(tessellation.node_r.size());
  for (std::size_t node = 0; node < tessellation.node_r.size(); ++node) {
    const std::pair<double, double> coordinates{
        tessellation.node_r[node], tessellation.node_z[node]};
    // Default double ordering treats -0.0 and +0.0 as equal, which is
    // acceptable because they represent the same coordinate.
    const auto [found, inserted] = canonical_ids.emplace(
        coordinates, static_cast<int>(node));
    (void)inserted;
    canonical_node_ids[node] = found->second;
  }

  for (std::size_t cell_index = 0;
       cell_index < tessellation.cells.size(); ++cell_index) {
    VoronoiCell& cell = tessellation.cells[cell_index];
    std::vector<int> replaced = cell.node_indices;
    for (int& node_id : replaced) {
      node_id = canonical_node_ids[static_cast<std::size_t>(node_id)];
    }

    std::vector<int> unified_nodes;
    std::vector<std::uint64_t> unified_neighbors;
    std::vector<std::size_t> unified_boundary_segments;
    unified_nodes.reserve(replaced.size());
    unified_neighbors.reserve(replaced.size());
    unified_boundary_segments.reserve(replaced.size());
    for (std::size_t edge = 0; edge < replaced.size(); ++edge) {
      const std::size_t next = (edge + 1) % replaced.size();
      if (replaced[edge] == replaced[next]) {
        continue;
      }
      unified_nodes.push_back(replaced[edge]);
      unified_neighbors.push_back(cell.neighbor_ids[edge]);
      unified_boundary_segments.push_back(
          boundary_segments[cell_index][edge]);
    }

    std::vector<int> distinct_nodes = unified_nodes;
    std::sort(distinct_nodes.begin(), distinct_nodes.end());
    distinct_nodes.erase(
        std::unique(distinct_nodes.begin(), distinct_nodes.end()),
        distinct_nodes.end());
    if (distinct_nodes.size() < 3) {
      degenerate_generator_id = cell.generator_id;
      return false;
    }
    cell.node_indices = std::move(unified_nodes);
    cell.neighbor_ids = std::move(unified_neighbors);
    boundary_segments[cell_index] =
        std::move(unified_boundary_segments);
    canonicalize_cell(cell, boundary_segments[cell_index]);
  }
  return true;
}

double cell_volume(const VoronoiCell& cell,
                   const Tessellation& tessellation) {
  std::vector<double> r;
  std::vector<double> z;
  r.reserve(cell.node_indices.size());
  z.reserve(cell.node_indices.size());
  for (const int node_id : cell.node_indices) {
    const Point point = node_point(tessellation, node_id);
    r.push_back(point.r);
    z.push_back(point.z);
  }
  return reale::polygon_rz_moments(
      r.data(), z.data(), static_cast<int>(r.size())).volume;
}

void finalize_cell_coordinates(Tessellation& tessellation) {
  for (VoronoiCell& cell : tessellation.cells) {
    cell.r.clear();
    cell.z.clear();
    cell.r.reserve(cell.node_indices.size());
    cell.z.reserve(cell.node_indices.size());
    for (const int node_id : cell.node_indices) {
      const std::size_t index = static_cast<std::size_t>(node_id);
      cell.r.push_back(tessellation.node_r[index]);
      cell.z.push_back(tessellation.node_z[index]);
    }
    cell.volume_rz = reale::polygon_rz_moments(
        cell.r.data(), cell.z.data(), static_cast<int>(cell.r.size())).volume;
  }
}

bool enforce_maximum_valence(
    Tessellation& tessellation,
    CanonicalNodeTable& node_table,
    std::vector<std::vector<std::size_t>>& boundary_segments,
    std::string& reject_reason) {
  static bool collapse_trace_printed = false;
  std::vector<CollapseEdge> candidates =
      collect_collapse_edges(tessellation);
  if (candidates.empty()) {
    if (!all_cells_are_simple(tessellation, reject_reason)) {
      return false;
    }
    return true;
  }
  CellIncidence incidence = build_cell_incidence(tessellation.cells);
  // The loop is finite because every committed collapse strictly reduces the
  // total node-reference count across cells: cells sharing the collapsed edge
  // each lose one reference, while renames elsewhere preserve counts, and
  // candidates are recollected only for cells that remain defective.
  while (true) {
    std::string first_candidate_rejection;
    bool collapsed = false;
    // Length is primary; the sorted node-id pair is the deterministic tie.
    for (const CollapseEdge& candidate : candidates) {
      if (!collapse_edge_is_eligible(candidate, tessellation,
                                     tessellation.cells,
                                     boundary_segments, incidence)) {
        continue;
      }
      const auto [midpoint, inserted] = node_table.append_collapsed(
          candidate.first, candidate.second);
      if (midpoint < 0) {
        reject_reason = "provenance_incomplete";
        return false;
      }
      const bool collapse_trace =
          !collapse_trace_printed &&
          ((candidate.first == 73080 && candidate.second == 73081) ||
           (candidate.first == 73081 && candidate.second == 73080));
      if (collapse_trace) {
        collapse_trace_printed = true;
        std::fprintf(stderr,
                     "[collapse_trace] candidate first=%d second=%d "
                     "midpoint=%d\n",
                     candidate.first, candidate.second, midpoint);
      }
      std::vector<TrialCell> affected;
      const std::vector<std::size_t> affected_indices =
          affected_cell_indices(candidate, incidence);
      for (const std::size_t cell_index : affected_indices) {
        affected.push_back(
            {cell_index, tessellation.cells[cell_index],
             boundary_segments[cell_index]});
      }
      if (collapse_trace) {
        std::fprintf(stderr, "[collapse_trace] affected n=%zu:",
                     affected_indices.size());
        for (const std::size_t cell_index : affected_indices) {
          std::fprintf(stderr, " %zu", cell_index);
        }
        std::fprintf(stderr, "\n");
        for (const TrialCell& trial : affected) {
          std::fprintf(stderr,
                       "[collapse_trace] pre cell=%zu gen=%llu cycle=",
                       trial.index,
                       static_cast<unsigned long long>(
                           trial.cell.generator_id));
          for (std::size_t node = 0;
               node < trial.cell.node_indices.size(); ++node) {
            std::fprintf(stderr, "%s%d", node == 0 ? "" : " ",
                         trial.cell.node_indices[node]);
          }
          std::fprintf(stderr, "\n");
        }
      }

      bool admissible = true;
      for (TrialCell& trial : affected) {
        collapse_cell_edge(trial.cell, trial.boundary_segments,
                           candidate.first, candidate.second, midpoint);
        std::vector<double> trial_r;
        std::vector<double> trial_z;
        trial_r.reserve(trial.cell.node_indices.size());
        trial_z.reserve(trial.cell.node_indices.size());
        for (const int node_id : trial.cell.node_indices) {
          const Point point = node_point(tessellation, node_id);
          trial_r.push_back(point.r);
          trial_z.push_back(point.z);
        }
        trial.cell.volume_rz = reale::polygon_rz_moments(
            trial_r.data(), trial_z.data(),
            static_cast<int>(trial_r.size())).volume;
        if (!(trial.cell.volume_rz > 0.0)) {
          if (first_candidate_rejection.empty()) {
            first_candidate_rejection = collapse_reject_reason(
                "collapse_nonpositive_volume", trial.cell.generator_id);
          }
          admissible = false;
          break;
        }
      }
      if (collapse_trace) {
        for (const TrialCell& trial : affected) {
          std::fprintf(stderr,
                       "[collapse_trace] post cell=%zu gen=%llu "
                       "committed=%d cycle=",
                       trial.index,
                       static_cast<unsigned long long>(
                           trial.cell.generator_id),
                       admissible ? 1 : 0);
          for (std::size_t node = 0;
               node < trial.cell.node_indices.size(); ++node) {
            std::fprintf(stderr, "%s%d", node == 0 ? "" : " ",
                         trial.cell.node_indices[node]);
          }
          std::fprintf(stderr, "\n");
        }
      }
      if (!admissible) {
        if (inserted) {
          node_table.remove_last_collapsed(
              candidate.first, candidate.second, midpoint);
        }
        continue;
      }
      for (TrialCell& trial : affected) {
        tessellation.cells[trial.index] = std::move(trial.cell);
        boundary_segments[trial.index] =
            std::move(trial.boundary_segments);
      }
      // Rebuild incidence from scratch after every commit: the incremental path
      // could go stale and miss renames (hanging collapse products). Candidate
      // ordering and one-at-a-time commit semantics are unchanged, so previously
      // correct collapse sequences are byte-identical.
      incidence = build_cell_incidence(tessellation.cells);
      ++tessellation.collapsed_edges;
      collapsed = true;
      break;
    }
    if (!collapsed) {
      if (!first_candidate_rejection.empty()) {
        reject_reason = first_candidate_rejection;
      } else {
        const auto overflowing = std::find_if(
            tessellation.cells.begin(), tessellation.cells.end(),
            [](const VoronoiCell& cell) {
              return cell.node_indices.size() > kMaximumValence;
            });
        if (overflowing != tessellation.cells.end()) {
          reject_reason = collapse_reject_reason(
              "valence_collapse_stalled", overflowing->generator_id);
        } else {
          const auto nonsimple = std::find_if(
              tessellation.cells.begin(), tessellation.cells.end(),
              [&tessellation](const VoronoiCell& cell) {
                return !polygon_is_simple(cell, tessellation);
              });
          if (nonsimple != tessellation.cells.end()) {
            reject_reason = collapse_reject_reason(
                "nonsimple_cell_unresolved", nonsimple->generator_id);
          } else {
            reject_reason = "valence_collapse_stalled: none";
          }
        }
      }
      return false;
    }
    candidates = collect_collapse_edges(tessellation);
    if (candidates.empty()) {
      if (!all_cells_are_simple(tessellation, reject_reason)) {
        return false;
      }
      return true;
    }
  }
}

LabeledPolygon rebuild_from_active_neighbors(
    const LabeledPolygon& polygon,
    const LabeledPolygon& domain_polygon,
    const std::vector<Generator>& sorted_generators,
    const std::size_t generator_index) {
  // Rebuild from the final active constraints so transient, subsequently
  // removed clips cannot change the output rotation or intersection rounding.
  std::vector<std::size_t> active_neighbors;
  active_neighbors.reserve(polygon.edge_labels.size());
  const Generator& generator = sorted_generators[generator_index];
  for (std::size_t edge = 0; edge < polygon.edge_labels.size(); ++edge) {
    const std::uint64_t neighbor_id = polygon.edge_labels[edge];
    if (neighbor_id == kBoundaryNeighbor) {
      continue;
    }
    const std::size_t next = (edge + 1) % polygon.vertices.size();
    const Point& start = polygon.vertices[edge];
    const Point& end = polygon.vertices[next];
    const double coordinate_scale = std::max(
        {std::abs(generator.r), std::abs(generator.z),
         std::abs(start.r), std::abs(start.z),
         std::abs(end.r), std::abs(end.z)});
    const double edge_length = std::hypot(end.r - start.r,
                                          end.z - start.z);
    if (edge_length <= 64.0 * std::numeric_limits<double>::epsilon() *
                           coordinate_scale) {
      continue;
    }
    const auto neighbor = std::lower_bound(
        sorted_generators.begin(), sorted_generators.end(), neighbor_id,
        [](const Generator& candidate, const std::uint64_t id) {
          return candidate.id < id;
        });
    if (neighbor != sorted_generators.end() && neighbor->id == neighbor_id) {
      active_neighbors.push_back(static_cast<std::size_t>(
          std::distance(sorted_generators.begin(), neighbor)));
    }
  }
  std::sort(active_neighbors.begin(), active_neighbors.end());
  active_neighbors.erase(
      std::unique(active_neighbors.begin(), active_neighbors.end()),
      active_neighbors.end());
  return clip_against_candidates(domain_polygon, sorted_generators,
                                 generator_index, active_neighbors);
}

class UniformSpatialHash {
 public:
  explicit UniformSpatialHash(
      const std::vector<Generator>& sorted_generators)
      : sorted_generators_(sorted_generators) {
    const std::size_t count = sorted_generators.size();
    dimension_ = static_cast<std::size_t>(
        std::ceil(std::sqrt(static_cast<double>(count))));
    dimension_ = std::max<std::size_t>(dimension_, 1);

    if (sorted_generators.empty()) {
      buckets_.resize(1);
      return;
    }

    min_r_ = max_r_ = sorted_generators.front().r;
    min_z_ = max_z_ = sorted_generators.front().z;
    for (const Generator& generator : sorted_generators) {
      min_r_ = std::min(min_r_, generator.r);
      max_r_ = std::max(max_r_, generator.r);
      min_z_ = std::min(min_z_, generator.z);
      max_z_ = std::max(max_z_, generator.z);
    }
    span_r_ = max_r_ - min_r_;
    span_z_ = max_z_ - min_z_;

    buckets_.resize(dimension_ * dimension_);
    const double infinity = std::numeric_limits<double>::infinity();
    column_prefix_max_r_.assign(dimension_, -infinity);
    column_suffix_min_r_.assign(dimension_, infinity);
    row_prefix_max_z_.assign(dimension_, -infinity);
    row_suffix_min_z_.assign(dimension_, infinity);
    for (std::size_t index = 0; index < sorted_generators.size(); ++index) {
      const auto [column, row] = bucket_of(sorted_generators[index]);
      buckets_[row * dimension_ + column].push_back(index);
      const Generator& generator = sorted_generators[index];
      column_prefix_max_r_[column] =
          std::max(column_prefix_max_r_[column], generator.r);
      column_suffix_min_r_[column] =
          std::min(column_suffix_min_r_[column], generator.r);
      row_prefix_max_z_[row] =
          std::max(row_prefix_max_z_[row], generator.z);
      row_suffix_min_z_[row] =
          std::min(row_suffix_min_z_[row], generator.z);
    }
    for (std::size_t index = 1; index < dimension_; ++index) {
      column_prefix_max_r_[index] =
          std::max(column_prefix_max_r_[index],
                   column_prefix_max_r_[index - 1]);
      row_prefix_max_z_[index] =
          std::max(row_prefix_max_z_[index],
                   row_prefix_max_z_[index - 1]);
    }
    for (std::size_t index = dimension_ - 1; index > 0; --index) {
      column_suffix_min_r_[index - 1] =
          std::min(column_suffix_min_r_[index - 1],
                   column_suffix_min_r_[index]);
      row_suffix_min_z_[index - 1] =
          std::min(row_suffix_min_z_[index - 1],
                   row_suffix_min_z_[index]);
    }
  }

  std::vector<std::size_t> candidates(
      std::size_t generator_index,
      const LabeledPolygon& domain_polygon,
      double& radius_upper) const {
    const Generator& generator = sorted_generators_[generator_index];
    const auto [center_column, center_row] = bucket_of(generator);
    std::vector<std::size_t> gathered;
    std::size_t min_column = center_column > 2U ? center_column - 2U : 0U;
    std::size_t max_column =
        std::min(dimension_ - 1U, center_column + 2U);
    std::size_t min_row = center_row > 2U ? center_row - 2U : 0U;
    std::size_t max_row = std::min(dimension_ - 1U, center_row + 2U);
    if (span_r_ == 0.0) {
      min_column = 0;
      max_column = dimension_ - 1;
    }
    if (span_z_ == 0.0) {
      min_row = 0;
      max_row = dimension_ - 1;
    }

    LabeledPolygon seed_polygon = domain_polygon;
    bool first_iteration = true;
    std::size_t old_min_column = min_column;
    std::size_t old_max_column = max_column;
    std::size_t old_min_row = min_row;
    std::size_t old_max_row = max_row;

    // Expand per generator until the physical distance bound to every
    // unvisited bucket's contents exceeds the current cell's 2*radius bound.
    // The seed never reaches the output; the canonical build re-clips the
    // domain polygon in ascending id order. Its internal clip order is thus
    // free, and the returned set only needs to be a certified superset of the
    // active set, which the unchanged security-radius termination guarantees.
    while (true) {
      std::vector<std::size_t> new_candidates;
      if (first_iteration) {
        append_bucket_rectangle(min_column, max_column,
                                min_row, max_row, new_candidates);
      } else {
        if (min_column < old_min_column) {
          append_bucket_rectangle(min_column, old_min_column - 1U,
                                  min_row, max_row, new_candidates);
        }
        if (max_column > old_max_column) {
          append_bucket_rectangle(old_max_column + 1U, max_column,
                                  min_row, max_row, new_candidates);
        }
        if (min_row < old_min_row) {
          append_bucket_rectangle(old_min_column, old_max_column,
                                  min_row, old_min_row - 1U,
                                  new_candidates);
        }
        if (max_row > old_max_row) {
          append_bucket_rectangle(old_min_column, old_max_column,
                                  old_max_row + 1U, max_row,
                                  new_candidates);
        }
      }
      std::sort(
          new_candidates.begin(), new_candidates.end(),
          [&](const std::size_t lhs, const std::size_t rhs) {
            const double lhs_dr = sorted_generators_[lhs].r - generator.r;
            const double lhs_dz = sorted_generators_[lhs].z - generator.z;
            const double rhs_dr = sorted_generators_[rhs].r - generator.r;
            const double rhs_dz = sorted_generators_[rhs].z - generator.z;
            const double lhs_distance = lhs_dr * lhs_dr + lhs_dz * lhs_dz;
            const double rhs_distance = rhs_dr * rhs_dr + rhs_dz * rhs_dz;
            if (lhs_distance != rhs_distance) {
              return lhs_distance < rhs_distance;
            }
            return lhs < rhs;
          });
      gathered.insert(gathered.end(),
                      new_candidates.begin(), new_candidates.end());
      seed_polygon = clip_against_candidates(
          seed_polygon, sorted_generators_, generator_index, new_candidates);
      if (seed_polygon.vertices.size() < 3) {
        radius_upper = std::numeric_limits<double>::infinity();
        sort_and_unique(gathered);
        return gathered;
      }
      if (min_column == 0 && max_column + 1 == dimension_ &&
          min_row == 0 && max_row + 1 == dimension_) {
        double radius = 0.0;
        for (const Point& vertex : seed_polygon.vertices) {
          radius = std::max(radius,
                            std::hypot(vertex.r - generator.r,
                                       vertex.z - generator.z));
        }
        radius_upper = std::nextafter(
            radius, std::numeric_limits<double>::infinity());
        sort_and_unique(gathered);
        return gathered;
      }

      double radius = 0.0;
      for (const Point& vertex : seed_polygon.vertices) {
        radius = std::max(radius,
                          std::hypot(vertex.r - generator.r,
                                     vertex.z - generator.z));
      }
      const double security_radius = std::nextafter(
          2.0 * radius, std::numeric_limits<double>::infinity());
      if (distance_to_unvisited_buckets(
              generator, min_column, max_column, min_row, max_row) >
          security_radius) {
        radius_upper = std::nextafter(
            radius, std::numeric_limits<double>::infinity());
        sort_and_unique(gathered);
        return gathered;
      }

      old_min_column = min_column;
      old_max_column = max_column;
      old_min_row = min_row;
      old_max_row = max_row;
      if (min_column > 0) {
        --min_column;
      }
      if (max_column + 1 < dimension_) {
        ++max_column;
      }
      if (min_row > 0) {
        --min_row;
      }
      if (max_row + 1 < dimension_) {
        ++max_row;
      }
      first_iteration = false;
    }
  }

 private:
  std::size_t coordinate_bucket(const double coordinate,
                                const double minimum,
                                const double span) const {
    if (span == 0.0 || coordinate <= minimum) {
      return 0;
    }
    if (coordinate >= minimum + span) {
      return dimension_ - 1;
    }
    const double scaled =
        (coordinate - minimum) / span * static_cast<double>(dimension_);
    return std::min(static_cast<std::size_t>(scaled), dimension_ - 1);
  }

  std::pair<std::size_t, std::size_t> bucket_of(
      const Generator& generator) const {
    return {coordinate_bucket(generator.r, min_r_, span_r_),
            coordinate_bucket(generator.z, min_z_, span_z_)};
  }

  double distance_to_unvisited_buckets(
      const Generator& generator,
      const std::size_t min_column,
      const std::size_t max_column,
      const std::size_t min_row,
      const std::size_t max_row) const {
    double distance = std::numeric_limits<double>::infinity();
    if (min_column > 0) {
      distance = std::min(
          distance, generator.r -
                        column_prefix_max_r_[min_column - 1]);
    }
    if (max_column + 1 < dimension_) {
      distance = std::min(
          distance, column_suffix_min_r_[max_column + 1] -
                        generator.r);
    }
    if (min_row > 0) {
      distance = std::min(
          distance, generator.z -
                        row_prefix_max_z_[min_row - 1]);
    }
    if (max_row + 1 < dimension_) {
      distance = std::min(
          distance, row_suffix_min_z_[max_row + 1] -
                        generator.z);
    }
    return std::nextafter(distance,
                          -std::numeric_limits<double>::infinity());
  }

  void append_bucket_rectangle(const std::size_t min_column,
                               const std::size_t max_column,
                               const std::size_t min_row,
                               const std::size_t max_row,
                               std::vector<std::size_t>& result) const {
    for (std::size_t row = min_row; row <= max_row; ++row) {
      for (std::size_t column = min_column; column <= max_column; ++column) {
        const std::vector<std::size_t>& bucket =
            buckets_[row * dimension_ + column];
        result.insert(result.end(), bucket.begin(), bucket.end());
      }
    }
  }

  static void sort_and_unique(std::vector<std::size_t>& candidates) {
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
  }

  const std::vector<Generator>& sorted_generators_;
  std::size_t dimension_ = 1;
  double min_r_ = 0.0;
  double max_r_ = 0.0;
  double min_z_ = 0.0;
  double max_z_ = 0.0;
  double span_r_ = 0.0;
  double span_z_ = 0.0;
  std::vector<std::vector<std::size_t>> buckets_;
  std::vector<double> column_prefix_max_r_;
  std::vector<double> column_suffix_min_r_;
  std::vector<double> row_prefix_max_z_;
  std::vector<double> row_suffix_min_z_;
};

}  // namespace

namespace voronoi_detail {

void set_slow_path_for_tests(const bool enabled) {
  use_slow_path_for_tests = enabled;
}

}  // namespace voronoi_detail

Tessellation build_constrained_voronoi(
    const std::vector<Generator>& generators,
    const DomainBoundary& domain) {
  std::vector<Generator> sorted_generators = generators;
  std::stable_sort(sorted_generators.begin(), sorted_generators.end(),
                   [](const Generator& lhs, const Generator& rhs) {
                     return lhs.id < rhs.id;
                   });

  for (const Generator& generator : sorted_generators) {
    if (generator.r < 0.0) {
      return reject(reject_with_id("negative_r", generator.id));
    }
  }
  std::uint64_t duplicate_first_id = 0;
  std::uint64_t duplicate_second_id = 0;
  if (find_duplicate_generators(sorted_generators, duplicate_first_id,
                                duplicate_second_id)) {
    return reject(reject_duplicate(duplicate_first_id, duplicate_second_id));
  }

  LabeledPolygon domain_polygon;
  domain_polygon.vertices.reserve(domain.r.size());
  domain_polygon.vertex_provenance.reserve(domain.r.size());
  domain_polygon.edge_labels.assign(domain.r.size(), kBoundaryNeighbor);
  domain_polygon.boundary_segments.resize(domain.r.size());
  for (std::size_t vertex = 0; vertex < domain.r.size(); ++vertex) {
    domain_polygon.vertices.push_back({domain.r[vertex], domain.z[vertex]});
    VertexProvenance provenance;
    provenance.kind = VertexProvenanceKind::kDomainCorner;
    provenance.boundary_index = vertex;
    domain_polygon.vertex_provenance.push_back(provenance);
    domain_polygon.boundary_segments[vertex] = vertex;
  }

  std::vector<std::size_t> all_candidates(sorted_generators.size());
  for (std::size_t index = 0; index < all_candidates.size(); ++index) {
    all_candidates[index] = index;
  }
  const bool finite_coordinates = std::all_of(
      sorted_generators.begin(), sorted_generators.end(),
      [](const Generator& generator) {
        return std::isfinite(generator.r) && std::isfinite(generator.z);
      });
  const bool canonical_neighbor_ids =
      std::adjacent_find(
          sorted_generators.begin(), sorted_generators.end(),
          [](const Generator& lhs, const Generator& rhs) {
            return lhs.id == rhs.id;
          }) == sorted_generators.end() &&
      std::none_of(sorted_generators.begin(), sorted_generators.end(),
                   [](const Generator& generator) {
                     return generator.id == kBoundaryNeighbor;
                   });
  if (!canonical_neighbor_ids) {
    return reject("provenance_incomplete");
  }
  const bool use_spatial_hash = !use_slow_path_for_tests &&
                                finite_coordinates &&
                                !sorted_generators.empty();
  const UniformSpatialHash spatial_hash(sorted_generators);

  Tessellation result;
  result.cells.reserve(sorted_generators.size());
  CanonicalNodeTable node_table(
      result, sorted_generators, domain_polygon);
  std::vector<std::vector<std::size_t>> cell_boundary_segments;
  cell_boundary_segments.reserve(sorted_generators.size());
  for (std::size_t generator_index = 0;
       generator_index < sorted_generators.size(); ++generator_index) {
    const Generator& generator = sorted_generators[generator_index];
    std::vector<std::size_t> candidates;
    if (use_spatial_hash) {
      double radius_upper = std::numeric_limits<double>::infinity();
      candidates = spatial_hash.candidates(
          generator_index, domain_polygon, radius_upper);
      // If |q-g| > 2*max_v|v-g| over the cell vertices, every vertex lies
      // strictly on g's side of bisector(g,q): the bisector distance is
      // |q-g|/2 > radius. The clip is therefore an exact no-op, and
      // clip_half_plane returns its input bitwise, so pruning preserves the
      // bytes produced by the remaining id-ordered sequence. Opposing
      // nextafter guards conservatively cover <=1-ulp rounding in hypot and
      // the vertex-distance maximum. Marginal candidates are kept as cheap
      // no-op scans, never wrongly dropped.
      const double pruning_radius_upper = std::nextafter(
          2.0 * radius_upper, std::numeric_limits<double>::infinity());
      candidates.erase(
          std::remove_if(
              candidates.begin(), candidates.end(),
              [&](const std::size_t candidate_index) {
                const Generator& candidate =
                    sorted_generators[candidate_index];
                const double candidate_distance_lower = std::nextafter(
                    std::hypot(candidate.r - generator.r,
                               candidate.z - generator.z),
                    -std::numeric_limits<double>::infinity());
                return candidate_distance_lower > pruning_radius_upper;
              }),
          candidates.end());
    } else {
      candidates = all_candidates;
    }
    LabeledPolygon cell_polygon = clip_against_candidates(
        domain_polygon, sorted_generators, generator_index, candidates);

    if (cell_polygon.vertices.size() < 3) {
      return reject(reject_with_id("empty_cell", generator.id));
    }
    cell_polygon = rebuild_from_active_neighbors(
        cell_polygon, domain_polygon, sorted_generators, generator_index);
    canonicalize_polygon(cell_polygon);
    if (cell_polygon.vertices.size() < 3 ||
        cell_polygon.vertex_provenance.size() !=
            cell_polygon.vertices.size()) {
      return reject("provenance_incomplete");
    }

    VoronoiCell cell;
    cell.generator_id = generator.id;
    cell.node_indices.reserve(cell_polygon.vertices.size());
    for (const VertexProvenance& provenance :
         cell_polygon.vertex_provenance) {
      int node_id = -1;
      if (!node_table.resolve(provenance, generator.id, node_id)) {
        return reject("provenance_incomplete");
      }
      cell.node_indices.push_back(node_id);
    }
    cell.neighbor_ids = cell_polygon.edge_labels;
    std::vector<std::size_t> boundary_segments =
        cell_polygon.boundary_segments;
    canonicalize_cell(cell, boundary_segments);
    cell_boundary_segments.push_back(std::move(boundary_segments));
    result.cells.push_back(std::move(cell));
  }

  merge_exact_zero_length_edges(result, cell_boundary_segments);
  // consult-26 Stage-3.5 exact co-circular event resolution: provenance
  // identifies equal interior events even when circumcenters differ by ulps.
  unify_exact_cocircular_interior_nodes(
      result, node_table, sorted_generators);
  // consult-26 Stage-3 exact cleanup: bitwise-coincident nodes under
  // different provenance keys (>=4-cocircular splits) must be one topological
  // node, or shared edges fail to pair by id and cracks masquerade as boundary.
  std::uint64_t degenerate_generator_id = 0;
  if (!unify_bitwise_duplicate_nodes(
          result, cell_boundary_segments, degenerate_generator_id)) {
    return reject(reject_with_id(
        "duplicate_node_unification_degenerate",
        degenerate_generator_id));
  }
  for (VoronoiCell& cell : result.cells) {
    if (cell.node_indices.size() < 3) {
      return reject("provenance_incomplete");
    }
    cell.volume_rz = cell_volume(cell, result);
    if (!(cell.volume_rz > 0.0)) {
      return reject(reject_with_id(
          "nonpositive_volume", cell.generator_id));
    }
  }

  std::string collapse_rejection;
  if (!enforce_maximum_valence(
          result, node_table, cell_boundary_segments,
          collapse_rejection)) {
    Tessellation rejected = reject(collapse_rejection);
    rejected.collapsed_edges = result.collapsed_edges;
    return rejected;
  }
  if (!rebuild_neighbors_and_check_reciprocity(
          result, node_table, domain_polygon)) {
    Tessellation rejected = reject(
        "reciprocity_rebuild_failed_or_interior_crack");
    rejected.collapsed_edges = result.collapsed_edges;
    return rejected;
  }

  finalize_cell_coordinates(result);
  result.valid = true;
  return result;
}

}  // namespace tenryu::mesh::voronoi
