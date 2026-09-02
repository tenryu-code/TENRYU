#include "mesh/tessellation/voronoi_dual.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/tessellation/gpu_classify.hpp"
#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {
namespace {

bool g_tess_gpu_restrict_enabled = false;
std::size_t g_tess_gpu_last_classify_shortlist = 0;

struct DomainVertex {
  DomainPoint point;
  DomainVertexId input_index;
};

struct CandidatePoint {
  HomogeneousPoint point;
  std::vector<CanonicalNodeKey> aliases;
  std::vector<RestrictedExactWitness> exact_witnesses;
  std::vector<DomainSegmentId> supports;
  std::vector<DomainVertexId> domain_vertices;
  BoundaryParam boundary_param;
  VoronoiNodeId dual_node = kInvalidId;
  bool inside = false;
};

struct DomainClassification {
  bool inside = false;
  std::vector<DomainSegmentId> supports;
  std::vector<DomainVertexId> domain_vertices;
};

template <typename T>
void sort_unique(std::vector<T>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

Expansion scalar_expansion(const double value) {
  return Expansion{value};
}

double expansion_value(const Expansion& expansion) {
  return std::accumulate(expansion.begin(), expansion.end(), 0.0);
}

HomogeneousPoint domain_point_exact(const DomainPoint& point) {
  return HomogeneousPoint{scalar_expansion(point.r),
                          scalar_expansion(point.z), Expansion{1.0}};
}

Expansion bisector_phi(const Site& gi,
                       const Site& gj,
                       const DomainPoint& point) {
  const Expansion gi_r = exact_difference(point.r, gi.r);
  const Expansion gi_z = exact_difference(point.z, gi.z);
  const Expansion gj_r = exact_difference(point.r, gj.r);
  const Expansion gj_z = exact_difference(point.z, gj.z);
  return expansion_difference(expansion_lift(gi_r, gi_z),
                              expansion_lift(gj_r, gj_z));
}

ExactRational bisector_edge_parameter(const Site& gi,
                                      const Site& gj,
                                      const DomainPoint& a,
                                      const DomainPoint& b) {
  const Expansion phi_a = bisector_phi(gi, gj, a);
  const Expansion phi_b = bisector_phi(gi, gj, b);
  return make_exact_rational(expansion_negate(phi_a),
                             expansion_difference(phi_b, phi_a));
}

bool same_exact_point(const HomogeneousPoint& first,
                      const HomogeneousPoint& second) {
  return expansion_sign(expansion_difference(
             expansion_product(first.x, second.w),
             expansion_product(second.x, first.w))) == 0 &&
         expansion_sign(expansion_difference(
             expansion_product(first.y, second.w),
             expansion_product(second.y, first.w))) == 0;
}

int orient_domain_edge_point(const DomainPoint& a,
                             const DomainPoint& b,
                             const HomogeneousPoint& point) {
  const double edge_r_value = b.r - a.r;
  const double edge_z_value = b.z - a.z;
  const ApproxBound edge_r_approx{
      edge_r_value,
      kApproxEps * std::abs(edge_r_value) * kApproxSlack};
  const ApproxBound edge_z_approx{
      edge_z_value,
      kApproxEps * std::abs(edge_z_value) * kApproxSlack};
  const ApproxBound point_r_approx = approx_sub(
      approx_of_expansion(point.x),
      approx_mul(approx_of_expansion(point.w), ApproxBound{a.r, 0.0}));
  const ApproxBound point_z_approx = approx_sub(
      approx_of_expansion(point.y),
      approx_mul(approx_of_expansion(point.w), ApproxBound{a.z, 0.0}));
  const ApproxBound determinant_approx = approx_sub(
      approx_mul(edge_r_approx, point_z_approx),
      approx_mul(edge_z_approx, point_r_approx));
  const ApproxSign sign_approx = approx_sign(determinant_approx);
  if (sign_approx.certain) {
    return sign_approx.sign;
  }

  const Expansion edge_r = exact_difference(b.r, a.r);
  const Expansion edge_z = exact_difference(b.z, a.z);
  const Expansion point_r = expansion_difference(
      point.x, expansion_scale(point.w, a.r));
  const Expansion point_z = expansion_difference(
      point.y, expansion_scale(point.w, a.z));
  return expansion_sign(expansion_cross(edge_r, edge_z,
                                        point_r, point_z));
}

bool rational_in_closed_range(const Expansion& numerator,
                              const Expansion& denominator,
                              const double first,
                              const double second) {
  const double low = std::min(first, second);
  const double high = std::max(first, second);
  return expansion_sign(expansion_difference(
             numerator, expansion_scale(denominator, low))) >= 0 &&
         expansion_sign(expansion_difference(
             expansion_scale(denominator, high), numerator)) >= 0;
}

bool point_on_domain_segment(const HomogeneousPoint& point,
                             const DomainPoint& a,
                             const DomainPoint& b) {
  return orient_domain_edge_point(a, b, point) == 0 &&
         rational_in_closed_range(point.x, point.w, a.r, b.r) &&
         rational_in_closed_range(point.y, point.w, a.z, b.z);
}

struct DomainQueryBox {
  double r_lo;
  double r_hi;
  double z_lo;
  double z_hi;
};

struct DomainSegmentBounds {
  double r_lo;
  double r_hi;
  double z_lo;
  double z_hi;
};

struct DomainSegmentIndex {
  explicit DomainSegmentIndex(const std::vector<DomainVertex>& vertices)
      : n_segments(vertices.size()),
        nx(std::max<std::size_t>(
            1, static_cast<std::size_t>(
                   std::ceil(std::sqrt(static_cast<double>(n_segments)))))),
        ny(nx),
        r_lo(vertices.front().point.r),
        r_hi(vertices.front().point.r),
        z_lo(vertices.front().point.z),
        z_hi(vertices.front().point.z) {
    for (const DomainVertex& vertex : vertices) {
      r_lo = std::min(r_lo, vertex.point.r);
      r_hi = std::max(r_hi, vertex.point.r);
      z_lo = std::min(z_lo, vertex.point.z);
      z_hi = std::max(z_hi, vertex.point.z);
    }

    segment_bounds.reserve(n_segments);
    grid_buckets.resize(nx * ny);
    z_buckets.resize(n_segments);
    for (std::size_t segment = 0; segment < n_segments; ++segment) {
      const DomainPoint& a = vertices[segment].point;
      const DomainPoint& b = vertices[(segment + 1) % n_segments].point;
      const DomainSegmentBounds bounds{
          std::min(a.r, b.r), std::max(a.r, b.r),
          std::min(a.z, b.z), std::max(a.z, b.z)};
      segment_bounds.push_back(bounds);

      const std::size_t r_begin =
          cell_index(bounds.r_lo, r_lo, r_hi, nx);
      const std::size_t r_end =
          cell_index(bounds.r_hi, r_lo, r_hi, nx);
      const std::size_t z_begin =
          cell_index(bounds.z_lo, z_lo, z_hi, ny);
      const std::size_t z_end =
          cell_index(bounds.z_hi, z_lo, z_hi, ny);
      for (std::size_t r_cell = r_begin; r_cell <= r_end; ++r_cell) {
        for (std::size_t z_cell = z_begin; z_cell <= z_end; ++z_cell) {
          grid_buckets[r_cell * ny + z_cell].push_back(
              static_cast<DomainSegmentId>(segment));
        }
      }

      const std::size_t bucket_begin =
          cell_index(bounds.z_lo, z_lo, z_hi, n_segments);
      const std::size_t bucket_end =
          cell_index(bounds.z_hi, z_lo, z_hi, n_segments);
      for (std::size_t bucket = bucket_begin; bucket <= bucket_end; ++bucket) {
        z_buckets[bucket].push_back(
            static_cast<DomainSegmentId>(segment));
      }
    }
  }

  const std::vector<DomainSegmentId>& query_box(
      const double query_r_lo,
      const double query_r_hi,
      const double query_z_lo,
      const double query_z_hi) {
    box_scratch.clear();
    if (query_r_lo > query_r_hi || query_z_lo > query_z_hi) {
      return box_scratch;
    }
    const std::size_t r_begin = cell_index(query_r_lo, r_lo, r_hi, nx);
    const std::size_t r_end = cell_index(query_r_hi, r_lo, r_hi, nx);
    const std::size_t z_begin = cell_index(query_z_lo, z_lo, z_hi, ny);
    const std::size_t z_end = cell_index(query_z_hi, z_lo, z_hi, ny);
    for (std::size_t r_cell = r_begin; r_cell <= r_end; ++r_cell) {
      for (std::size_t z_cell = z_begin; z_cell <= z_end; ++z_cell) {
        const std::vector<DomainSegmentId>& bucket =
            grid_buckets[r_cell * ny + z_cell];
        box_scratch.insert(box_scratch.end(), bucket.begin(), bucket.end());
      }
    }
    sort_unique(box_scratch);
    box_scratch.erase(
        std::remove_if(
            box_scratch.begin(), box_scratch.end(),
            [&](const DomainSegmentId segment) {
              const DomainSegmentBounds& bounds =
                  segment_bounds[static_cast<std::size_t>(segment)];
              return bounds.r_hi < query_r_lo || bounds.r_lo > query_r_hi ||
                     bounds.z_hi < query_z_lo || bounds.z_lo > query_z_hi;
            }),
        box_scratch.end());
    return box_scratch;
  }

  const std::vector<DomainSegmentId>& query_z_straddle(
      const double query_z_lo,
      const double query_z_hi) {
    z_scratch.clear();
    if (query_z_lo > query_z_hi) {
      return z_scratch;
    }
    const std::size_t bucket_begin =
        cell_index(query_z_lo, z_lo, z_hi, n_segments);
    const std::size_t bucket_end =
        cell_index(query_z_hi, z_lo, z_hi, n_segments);
    for (std::size_t bucket = bucket_begin; bucket <= bucket_end; ++bucket) {
      z_scratch.insert(z_scratch.end(), z_buckets[bucket].begin(),
                       z_buckets[bucket].end());
    }
    sort_unique(z_scratch);
    z_scratch.erase(
        std::remove_if(
            z_scratch.begin(), z_scratch.end(),
            [&](const DomainSegmentId segment) {
              const DomainSegmentBounds& bounds =
                  segment_bounds[static_cast<std::size_t>(segment)];
              return bounds.z_hi < query_z_lo || bounds.z_lo > query_z_hi;
            }),
        z_scratch.end());
    return z_scratch;
  }

  static std::size_t cell_index(const double coordinate,
                                const double lo,
                                const double hi,
                                const std::size_t bucket_count) {
    if (hi == lo) {
      return 0;
    }
    const double scaled =
        (coordinate - lo) * static_cast<double>(bucket_count) /
        std::max(hi - lo, std::numeric_limits<double>::min());
    if (scaled <= 0.0) {
      return 0;
    }
    if (scaled >= static_cast<double>(bucket_count)) {
      return bucket_count - 1;
    }
    return std::clamp(static_cast<std::size_t>(scaled), std::size_t{0},
                      bucket_count - 1);
  }

  std::size_t n_segments;
  std::size_t nx;
  std::size_t ny;
  double r_lo;
  double r_hi;
  double z_lo;
  double z_hi;
  std::vector<DomainSegmentBounds> segment_bounds;
  std::vector<std::vector<DomainSegmentId>> grid_buckets;
  std::vector<std::vector<DomainSegmentId>> z_buckets;
  std::vector<DomainSegmentId> box_scratch;
  std::vector<DomainSegmentId> z_scratch;
};

DomainQueryBox certified_point_box(const double r,
                                   const double z,
                                   const DomainSegmentIndex& index) {
  const double scale = std::max(
      {std::abs(index.r_lo), std::abs(index.r_hi), std::abs(index.z_lo),
       std::abs(index.z_hi), std::abs(r), std::abs(z)});
  const double delta =
      4.0 * std::numeric_limits<double>::epsilon() * scale;
  return DomainQueryBox{r - delta, r + delta, z - delta, z + delta};
}

DomainClassification classify_domain_point(
    const HomogeneousPoint& point,
    const std::vector<DomainVertex>& vertices) {
  TENRYU_ASSERT(vertices.size() >= 3,
                "domain classification requires a polygon");
  DomainClassification classification;
  for (std::size_t segment = 0; segment < vertices.size(); ++segment) {
    const std::size_t next = (segment + 1) % vertices.size();
    const DomainPoint& a = vertices[segment].point;
    const DomainPoint& b = vertices[next].point;
    if (orient_domain_edge_point(a, b, point) == 0 &&
        point_on_domain_segment(point, a, b)) {
      classification.supports.push_back(
          static_cast<DomainSegmentId>(segment));
      if (same_exact_point(point, domain_point_exact(a))) {
        classification.domain_vertices.push_back(
            vertices[segment].input_index);
        classification.supports.push_back(static_cast<DomainSegmentId>(
            (segment + vertices.size() - 1) % vertices.size()));
      }
      if (same_exact_point(point, domain_point_exact(b))) {
        classification.domain_vertices.push_back(
            vertices[next].input_index);
        classification.supports.push_back(
            static_cast<DomainSegmentId>(next));
      }
    }
  }
  if (!classification.supports.empty()) {
    classification.inside = true;
    sort_unique(classification.supports);
    sort_unique(classification.domain_vertices);
    return classification;
  }

  int winding_number = 0;
  for (std::size_t segment = 0; segment < vertices.size(); ++segment) {
    const DomainPoint& a = vertices[segment].point;
    const DomainPoint& b =
        vertices[(segment + 1) % vertices.size()].point;
    const int sign_a = expansion_sign(expansion_difference(
        expansion_scale(point.w, a.z), point.y));
    const int sign_b = expansion_sign(expansion_difference(
        expansion_scale(point.w, b.z), point.y));
    const int side = orient_domain_edge_point(a, b, point);
    if (sign_a <= 0 && sign_b > 0 && side > 0) {
      ++winding_number;
    } else if (sign_b <= 0 && sign_a > 0 && side < 0) {
      --winding_number;
    }
  }
  classification.inside = winding_number != 0;
  return classification;
}

DomainClassification classify_domain_point(
    const HomogeneousPoint& point,
    const std::vector<DomainVertex>& vertices,
    DomainSegmentIndex& index,
    const DomainQueryBox& box) {
  TENRYU_ASSERT(vertices.size() >= 3,
                "domain classification requires a polygon");
  DomainClassification classification;
  // Accepted support contains the exact point, so its AABB overlaps this box.
  for (const DomainSegmentId segment_id :
       index.query_box(box.r_lo, box.r_hi, box.z_lo, box.z_hi)) {
    const std::size_t segment = static_cast<std::size_t>(segment_id);
    const std::size_t next = (segment + 1) % vertices.size();
    const DomainPoint& a = vertices[segment].point;
    const DomainPoint& b = vertices[next].point;
    if (orient_domain_edge_point(a, b, point) == 0 &&
        point_on_domain_segment(point, a, b)) {
      classification.supports.push_back(
          static_cast<DomainSegmentId>(segment));
      if (same_exact_point(point, domain_point_exact(a))) {
        classification.domain_vertices.push_back(
            vertices[segment].input_index);
        classification.supports.push_back(static_cast<DomainSegmentId>(
            (segment + vertices.size() - 1) % vertices.size()));
      }
      if (same_exact_point(point, domain_point_exact(b))) {
        classification.domain_vertices.push_back(
            vertices[next].input_index);
        classification.supports.push_back(
            static_cast<DomainSegmentId>(next));
      }
    }
  }
  if (!classification.supports.empty()) {
    classification.inside = true;
    sort_unique(classification.supports);
    sort_unique(classification.domain_vertices);
    return classification;
  }

  int winding_number = 0;
  // A winding contribution requires the segment z-range to contain exact z;
  // skipped segments therefore contribute exactly zero to both branches.
  for (const DomainSegmentId segment_id :
       index.query_z_straddle(box.z_lo, box.z_hi)) {
    const std::size_t segment = static_cast<std::size_t>(segment_id);
    const DomainPoint& a = vertices[segment].point;
    const DomainPoint& b =
        vertices[(segment + 1) % vertices.size()].point;
    const int sign_a = expansion_sign(expansion_difference(
        expansion_scale(point.w, a.z), point.y));
    const int sign_b = expansion_sign(expansion_difference(
        expansion_scale(point.w, b.z), point.y));
    const int side = orient_domain_edge_point(a, b, point);
    if (sign_a <= 0 && sign_b > 0 && side > 0) {
      ++winding_number;
    } else if (sign_b <= 0 && sign_a > 0 && side < 0) {
      --winding_number;
    }
  }
  classification.inside = winding_number != 0;
  return classification;
}

bool domain_points_equal(const DomainPoint& first,
                         const DomainPoint& second) {
  return first.r == second.r && first.z == second.z;
}

std::vector<DomainVertex> preprocess_domain(
    std::vector<DomainPoint> domain) {
  TENRYU_ASSERT(domain.size() >= 3, "domain requires at least three vertices");
  std::vector<DomainVertex> vertices;
  vertices.reserve(domain.size());
  for (std::size_t index = 0; index < domain.size(); ++index) {
    DomainPoint point = domain[index];
    TENRYU_ASSERT(std::isfinite(point.r) && std::isfinite(point.z),
                  "nan_coordinate");
    TENRYU_ASSERT(point.r >= 0.0, "negative_r");
    if (point.r == 0.0) {
      point.r = 0.0;
    }
    if (point.z == 0.0) {
      point.z = 0.0;
    }
    if (!vertices.empty()) {
      TENRYU_ASSERT(!domain_points_equal(vertices.back().point, point),
                    "domain has a zero-length edge");
    }
    TENRYU_ASSERT(index <= static_cast<std::size_t>(
                               std::numeric_limits<DomainVertexId>::max()),
                  "domain vertex count exceeds DomainVertexId range");
    vertices.push_back(
        DomainVertex{point, static_cast<DomainVertexId>(index)});
  }
  TENRYU_ASSERT(!domain_points_equal(vertices.front().point,
                                     vertices.back().point),
                "domain repeats its first vertex");

  bool changed = true;
  while (changed) {
    changed = false;
    TENRYU_ASSERT(vertices.size() >= 3,
                  "domain collapsed below three vertices");
    for (std::size_t index = 0; index < vertices.size(); ++index) {
      const std::size_t previous =
          (index + vertices.size() - 1) % vertices.size();
      const std::size_t next = (index + 1) % vertices.size();
      const DomainPoint& a = vertices[previous].point;
      const DomainPoint& b = vertices[index].point;
      const DomainPoint& c = vertices[next].point;
      if (orient2d_exact_sign(a.r, a.z, b.r, b.z, c.r, c.z) == 0 &&
          b.r >= std::min(a.r, c.r) && b.r <= std::max(a.r, c.r) &&
          b.z >= std::min(a.z, c.z) && b.z <= std::max(a.z, c.z)) {
        vertices.erase(vertices.begin() + static_cast<std::ptrdiff_t>(index));
        changed = true;
        break;
      }
    }
  }

  Expansion twice_area;
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    const DomainPoint& a = vertices[index].point;
    const DomainPoint& b = vertices[(index + 1) % vertices.size()].point;
    twice_area = expansion_sum(
        twice_area,
        expansion_difference(
            expansion_product(Expansion{a.r}, Expansion{b.z}),
            expansion_product(Expansion{b.r}, Expansion{a.z})));
  }
  TENRYU_ASSERT(expansion_sign(twice_area) > 0,
                "domain polygon must be counterclockwise");

  auto first = std::min_element(
      vertices.begin(), vertices.end(),
      [](const DomainVertex& lhs, const DomainVertex& rhs) {
        return lhs.input_index < rhs.input_index;
      });
  std::rotate(vertices.begin(), first, vertices.end());
  return vertices;
}

CanonicalNodeKey domain_key(const DomainVertexId vertex) {
  CanonicalNodeKey key;
  key.kind = CanonicalNodeKeyKind::kDomainVertex;
  key.domain_vertex = vertex;
  return key;
}

CanonicalNodeKey voronoi_key(const VoronoiTripleKey& triple) {
  CanonicalNodeKey key;
  key.kind = CanonicalNodeKeyKind::kVoronoiTriple;
  key.voronoi_triple = triple;
  return key;
}

CanonicalNodeKey crossing_key(const DomainSegmentId segment,
                              const std::uint64_t first,
                              const std::uint64_t second) {
  CanonicalNodeKey key;
  key.kind = CanonicalNodeKeyKind::kBoundaryCrossing;
  key.domain_segment = segment;
  key.crossing_sites = {{std::min(first, second), std::max(first, second)}};
  return key;
}

std::array<double, 2> construct_cartesian(const HomogeneousPoint& point) {
  const double denominator = expansion_value(point.w);
  TENRYU_ASSERT(denominator > 0.0,
                "restricted point denominator must be positive");
  return {{expansion_value(point.x) / denominator,
           expansion_value(point.y) / denominator}};
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

int projection_compare_bounded(const HomogeneousPoint& first,
                               const HomogeneousPoint& second,
                               const HomogeneousPoint& direction_start,
                               const HomogeneousPoint& direction_end) {
  const Expansion difference_r = rational_difference(
      first.x, first.w, second.x, second.w);
  const Expansion difference_z = rational_difference(
      first.y, first.w, second.y, second.w);
  const Expansion direction_r = rational_difference(
      direction_end.x, direction_end.w,
      direction_start.x, direction_start.w);
  const Expansion direction_z = rational_difference(
      direction_end.y, direction_end.w,
      direction_start.y, direction_start.w);
  return expansion_sign(expansion_sum(
      expansion_product(difference_r, direction_r),
      expansion_product(difference_z, direction_z)));
}

int projection_compare_ray(const HomogeneousPoint& first,
                           const HomogeneousPoint& second,
                           const ExactDirectionWitness& direction) {
  const Expansion difference_r = rational_difference(
      first.x, first.w, second.x, second.w);
  const Expansion difference_z = rational_difference(
      first.y, first.w, second.y, second.w);
  return expansion_sign(expansion_sum(
      expansion_product(difference_r, direction.r),
      expansion_product(difference_z, direction.z)));
}

bool point_on_bounded_feature(const HomogeneousPoint& point,
                              const HomogeneousPoint& first,
                              const HomogeneousPoint& second) {
  return projection_compare_bounded(point, first, first, second) >= 0 &&
         projection_compare_bounded(point, second, first, second) <= 0;
}

bool point_on_ray_feature(const HomogeneousPoint& point,
                          const HomogeneousPoint& origin,
                          const ExactDirectionWitness& direction) {
  return projection_compare_ray(point, origin, direction) >= 0;
}

void merge_candidate(CandidatePoint& destination,
                     const CandidatePoint& source) {
  TENRYU_ASSERT(same_exact_point(destination.point, source.point),
                "candidate aliases denote different exact points");
  destination.aliases.insert(destination.aliases.end(),
                             source.aliases.begin(), source.aliases.end());
  destination.exact_witnesses.insert(
      destination.exact_witnesses.end(), source.exact_witnesses.begin(),
      source.exact_witnesses.end());
  destination.supports.insert(destination.supports.end(),
                              source.supports.begin(), source.supports.end());
  destination.domain_vertices.insert(destination.domain_vertices.end(),
                                     source.domain_vertices.begin(),
                                     source.domain_vertices.end());
  if (!destination.boundary_param.valid && source.boundary_param.valid) {
    destination.boundary_param = source.boundary_param;
  }
  if (source.dual_node != kInvalidId) {
    TENRYU_ASSERT(destination.dual_node == kInvalidId ||
                      destination.dual_node == source.dual_node,
                  "domain_vertex_alias_conflict");
    destination.dual_node = source.dual_node;
  }
  sort_unique(destination.aliases);
  std::sort(destination.exact_witnesses.begin(),
            destination.exact_witnesses.end(),
            [](const RestrictedExactWitness& lhs,
               const RestrictedExactWitness& rhs) {
              return lhs.key < rhs.key;
            });
  destination.exact_witnesses.erase(
      std::unique(destination.exact_witnesses.begin(),
                  destination.exact_witnesses.end(),
                  [](const RestrictedExactWitness& lhs,
                     const RestrictedExactWitness& rhs) {
                    return lhs.key == rhs.key;
                  }),
      destination.exact_witnesses.end());
  sort_unique(destination.supports);
  sort_unique(destination.domain_vertices);
}

class NodeRegistry {
 public:
  NodeRegistry(const VoronoiDual& dual,
               const std::vector<DomainVertex>& vertices,
               RestrictedGeometry& geometry)
      : dual_(dual), vertices_(vertices), geometry_(geometry) {
    dual_nodes_.assign(dual.nodes.size(), kInvalidId);
    for (std::size_t index = 0; index < vertices.size(); ++index) {
      const DomainVertex& vertex = vertices[index];
      CandidatePoint candidate;
      candidate.point = domain_point_exact(vertex.point);
      candidate.aliases.push_back(domain_key(vertex.input_index));
      candidate.domain_vertices.push_back(vertex.input_index);
      candidate.supports.push_back(static_cast<DomainSegmentId>(index));
      candidate.supports.push_back(static_cast<DomainSegmentId>(
          (index + vertices.size() - 1) % vertices.size()));
      const RestrictedNodeId node = append_node(candidate);
      domain_nodes_.emplace(vertex.input_index, node);
    }
  }

  RestrictedNodeId get(const CandidatePoint& candidate) {
    RestrictedNodeId node = kInvalidId;
    if (!candidate.domain_vertices.empty()) {
      const auto vertex = std::min_element(candidate.domain_vertices.begin(),
                                           candidate.domain_vertices.end());
      const auto found = domain_nodes_.find(*vertex);
      TENRYU_ASSERT(found != domain_nodes_.end(),
                    "domain_vertex_alias_conflict");
      node = found->second;
    } else if (candidate.dual_node != kInvalidId) {
      TENRYU_ASSERT(candidate.dual_node >= 0 &&
                        static_cast<std::size_t>(candidate.dual_node) <
                            dual_nodes_.size(),
                    "dual node id out of interning range");
      node = dual_nodes_[static_cast<std::size_t>(candidate.dual_node)];
      if (node != kInvalidId) {
        // A dual-slot hit re-presents the SAME immutable dual_candidates
        // entry whose first interning already consumed every field
        // (aliases/supports/domain_vertices/point), so the found-path
        // exact identity check and merges below are pure no-ops for it.
        return node;
      }
    } else {
      TENRYU_ASSERT(!candidate.aliases.empty(),
                    "restricted node has no canonical alias");
      const auto alias =
          std::min_element(candidate.aliases.begin(), candidate.aliases.end());
      const auto found = crossing_nodes_.find(*alias);
      if (found != crossing_nodes_.end()) {
        node = found->second;
      }
    }

    if (node == kInvalidId) {
      node = append_node(candidate);
      if (candidate.dual_node != kInvalidId) {
        dual_nodes_[static_cast<std::size_t>(candidate.dual_node)] = node;
      } else {
        crossing_nodes_.emplace(geometry_.nodes[node].key, node);
      }
      return node;
    }

    RestrictedNode& existing = geometry_.nodes[node];
    TENRYU_ASSERT(same_exact_point(existing.exact_point, candidate.point),
                  "domain_vertex_alias_conflict");
    existing.aliases.insert(existing.aliases.end(), candidate.aliases.begin(),
                            candidate.aliases.end());
    if (candidate.exact_witnesses.empty()) {
      const auto alias =
          std::min_element(candidate.aliases.begin(), candidate.aliases.end());
      existing.exact_witnesses.push_back(
          RestrictedExactWitness{*alias, candidate.point});
    } else {
      existing.exact_witnesses.insert(existing.exact_witnesses.end(),
                                      candidate.exact_witnesses.begin(),
                                      candidate.exact_witnesses.end());
    }
    existing.boundary_segments.insert(existing.boundary_segments.end(),
                                      candidate.supports.begin(),
                                      candidate.supports.end());
    existing.domain_vertices.insert(existing.domain_vertices.end(),
                                    candidate.domain_vertices.begin(),
                                    candidate.domain_vertices.end());
    sort_unique(existing.aliases);
    std::sort(existing.exact_witnesses.begin(),
              existing.exact_witnesses.end(),
              [](const RestrictedExactWitness& lhs,
                 const RestrictedExactWitness& rhs) {
                return lhs.key < rhs.key;
              });
    existing.exact_witnesses.erase(
        std::unique(existing.exact_witnesses.begin(),
                    existing.exact_witnesses.end(),
                    [](const RestrictedExactWitness& lhs,
                       const RestrictedExactWitness& rhs) {
                      return lhs.key == rhs.key;
                    }),
        existing.exact_witnesses.end());
    sort_unique(existing.boundary_segments);
    sort_unique(existing.domain_vertices);
    existing.key = existing.aliases.front();
    if (candidate.boundary_param.valid) {
      set_boundary_param(existing, candidate.boundary_param);
    }
    normalize_axis(existing);
    return node;
  }

 private:
  RestrictedNodeId append_node(const CandidatePoint& candidate) {
    TENRYU_ASSERT(
        geometry_.nodes.size() <= static_cast<std::size_t>(
                                      std::numeric_limits<RestrictedNodeId>::max()),
        "restricted node count exceeds RestrictedNodeId range");
    RestrictedNode node;
    node.aliases = candidate.aliases;
    node.exact_witnesses = candidate.exact_witnesses;
    node.boundary_segments = candidate.supports;
    node.domain_vertices = candidate.domain_vertices;
    sort_unique(node.aliases);
    sort_unique(node.boundary_segments);
    sort_unique(node.domain_vertices);
    TENRYU_ASSERT(!node.aliases.empty(),
                  "restricted node has no canonical alias");
    node.key = node.aliases.front();
    node.exact_point = candidate.point;
    if (node.exact_witnesses.empty()) {
      node.exact_witnesses.push_back(
          RestrictedExactWitness{node.key, node.exact_point});
    }
    if (candidate.dual_node != kInvalidId) {
      TENRYU_ASSERT(candidate.dual_node >= 0 &&
                        candidate.dual_node <
                            static_cast<VoronoiNodeId>(dual_.nodes.size()),
                    "restricted dual node is out of range");
      node.constructed = dual_.nodes[candidate.dual_node].constructed;
    } else {
      node.constructed = construct_cartesian(candidate.point);
    }
    if (candidate.boundary_param.valid) {
      set_boundary_param(node, candidate.boundary_param);
    }
    normalize_axis(node);
    const RestrictedNodeId id =
        static_cast<RestrictedNodeId>(geometry_.nodes.size());
    geometry_.nodes.push_back(std::move(node));
    return id;
  }

  void set_boundary_param(RestrictedNode& node,
                          const BoundaryParam& boundary_param) const {
    TENRYU_ASSERT(boundary_param.edge >= 0 &&
                      boundary_param.edge < static_cast<DomainSegmentId>(
                                                vertices_.size()),
                  "boundary parameter edge is out of range");
    node.boundary_param = boundary_param;
    node.lambda_cache = exact_rational_round(boundary_param.lambda);
  }

  void normalize_axis(RestrictedNode& node) const {
    for (const DomainSegmentId segment : node.boundary_segments) {
      if (segment >= 0 &&
          segment < static_cast<DomainSegmentId>(
                        geometry_.domain_segments.size()) &&
          geometry_.domain_segments[segment].axis) {
        node.constructed[0] = 0.0;
        return;
      }
    }
  }

  const VoronoiDual& dual_;
  const std::vector<DomainVertex>& vertices_;
  RestrictedGeometry& geometry_;
  std::map<DomainVertexId, RestrictedNodeId> domain_nodes_;
  // Dense direct-index interning for interior dual nodes: slot ==
  // kInvalidId means "not yet interned" (the former map-miss).
  std::vector<RestrictedNodeId> dual_nodes_;
  std::map<CanonicalNodeKey, RestrictedNodeId> crossing_nodes_;
};

CandidatePoint dual_endpoint_candidate_with(
    const VoronoiDual& dual,
    const VoronoiNodeId node_id,
    const std::vector<DomainVertex>& vertices,
    const DomainClassification& classification) {
  CandidatePoint candidate;
  candidate.dual_node = node_id;
  const VoronoiDualNode& node = dual.nodes[node_id];
  candidate.point =
      dual.classes[node.representative_class].circumcenter;
  for (const VoronoiClassId class_id : node.classes) {
    const CanonicalNodeKey key =
        voronoi_key(dual.classes[class_id].key);
    candidate.aliases.push_back(key);
    candidate.exact_witnesses.push_back(
        RestrictedExactWitness{key,
                               dual.classes[class_id].circumcenter});
  }
  candidate.inside = classification.inside;
  candidate.supports = classification.supports;
  candidate.domain_vertices = classification.domain_vertices;
  for (const DomainVertexId vertex : candidate.domain_vertices) {
    const CanonicalNodeKey key = domain_key(vertex);
    candidate.aliases.push_back(key);
    candidate.exact_witnesses.push_back(
        RestrictedExactWitness{key, candidate.point});
  }
  sort_unique(candidate.aliases);
  sort_unique(candidate.supports);
  sort_unique(candidate.domain_vertices);
  return candidate;
}

CandidatePoint dual_endpoint_candidate(
    const VoronoiDual& dual,
    const VoronoiNodeId node_id,
    const std::vector<DomainVertex>& vertices,
    DomainSegmentIndex& index) {
  TENRYU_ASSERT(node_id >= 0 &&
                    node_id < static_cast<VoronoiNodeId>(dual.nodes.size()),
                "dual endpoint node is out of range");
  const VoronoiDualNode& node = dual.nodes[node_id];
  const DomainQueryBox box = certified_point_box(
      node.constructed[0], node.constructed[1], index);
  const DomainClassification classification =
      classify_domain_point(
          dual.classes[node.representative_class].circumcenter,
          vertices, index, box);
  return dual_endpoint_candidate_with(dual, node_id, vertices,
                                      classification);
}

bool point_in_closed_domain(const HomogeneousPoint& point,
                            const std::vector<DomainVertex>& vertices) {
  return classify_domain_point(point, vertices).inside;
}

bool point_in_closed_domain(const HomogeneousPoint& point,
                            const std::vector<DomainVertex>& vertices,
                            DomainSegmentIndex& index,
                            const DomainQueryBox& box) {
  return classify_domain_point(point, vertices, index, box).inside;
}

Site domain_site(const DomainVertex& vertex) {
  return Site{vertex.point.r, vertex.point.z,
              static_cast<std::uint64_t>(vertex.input_index), 0};
}

CandidatePoint crossing_candidate(
    const DelaunayTriangulation& dt,
    const VoronoiDualFeature& feature,
    const DomainSegmentRecord& segment,
    const std::vector<DomainVertex>& vertices,
    const HomogeneousPoint& point,
    const ExactRational& lambda) {
  CandidatePoint candidate;
  candidate.point = point;
  candidate.supports.push_back(segment.id);
  candidate.aliases.push_back(crossing_key(
      segment.id, dt.sites[feature.primal_sites[0]].stable_id,
      dt.sites[feature.primal_sites[1]].stable_id));
  candidate.exact_witnesses.push_back(
      RestrictedExactWitness{candidate.aliases.back(), point});
  const DomainPoint& a = vertices[segment.id].point;
  const DomainPoint& b =
      vertices[(static_cast<std::size_t>(segment.id) + 1) %
               vertices.size()]
          .point;
  const bool point_at_start =
      same_exact_point(point, domain_point_exact(a));
  const bool point_at_end =
      same_exact_point(point, domain_point_exact(b));
  const bool lambda_at_start = exact_rational_compare_unit(lambda, 0) == 0;
  const bool lambda_at_end = exact_rational_compare_unit(lambda, 1) == 0;
  TENRYU_ASSERT(point_at_start == lambda_at_start,
                "carrier_lambda_endpoint_mismatch");
  TENRYU_ASSERT(point_at_end == lambda_at_end,
                "carrier_lambda_endpoint_mismatch");
  if (point_at_start) {
    candidate.domain_vertices.push_back(segment.start_vertex);
    const CanonicalNodeKey key = domain_key(segment.start_vertex);
    candidate.aliases.push_back(key);
    candidate.exact_witnesses.push_back(
        RestrictedExactWitness{key, point});
  }
  if (point_at_end) {
    candidate.domain_vertices.push_back(segment.end_vertex);
    const CanonicalNodeKey key = domain_key(segment.end_vertex);
    candidate.aliases.push_back(key);
    candidate.exact_witnesses.push_back(
        RestrictedExactWitness{key, point});
  }
  if (!point_at_start && !point_at_end) {
    candidate.boundary_param = BoundaryParam{segment.id, lambda, true};
  }
  sort_unique(candidate.aliases);
  sort_unique(candidate.domain_vertices);
  return candidate;
}

}  // namespace

void set_tess_gpu_restrict_enabled(const bool enabled) {
  g_tess_gpu_restrict_enabled = enabled;
}

bool tess_gpu_restrict_enabled() { return g_tess_gpu_restrict_enabled; }

std::size_t tess_gpu_last_classify_shortlist() {
  return g_tess_gpu_last_classify_shortlist;
}

bool CanonicalNodeKey::operator<(const CanonicalNodeKey& other) const {
  if (kind != other.kind) {
    return kind < other.kind;
  }
  if (kind == CanonicalNodeKeyKind::kDomainVertex) {
    return domain_vertex < other.domain_vertex;
  }
  if (kind == CanonicalNodeKeyKind::kVoronoiTriple) {
    return voronoi_triple < other.voronoi_triple;
  }
  if (domain_segment != other.domain_segment) {
    return domain_segment < other.domain_segment;
  }
  return crossing_sites < other.crossing_sites;
}

bool CanonicalNodeKey::operator==(const CanonicalNodeKey& other) const {
  if (kind != other.kind) {
    return false;
  }
  if (kind == CanonicalNodeKeyKind::kDomainVertex) {
    return domain_vertex == other.domain_vertex;
  }
  if (kind == CanonicalNodeKeyKind::kVoronoiTriple) {
    return voronoi_triple == other.voronoi_triple;
  }
  return domain_segment == other.domain_segment &&
         crossing_sites == other.crossing_sites;
}

RestrictedGeometry restrict_voronoi_dual(
    const DelaunayTriangulation& dt,
    const VoronoiDual& dual,
    std::vector<DomainPoint> domain) {
  const bool timing_enabled =
      std::getenv("TENRYU_TESS_TIMING") != nullptr &&
      std::string(std::getenv("TENRYU_TESS_TIMING")) == std::string("1");
  const auto setup_start = std::chrono::steady_clock::now();
  const std::vector<DomainVertex> vertices =
      preprocess_domain(std::move(domain));
  TENRYU_ASSERT(vertices.size() <= static_cast<std::size_t>(
                                       std::numeric_limits<DomainSegmentId>::max()),
                "domain segment count exceeds DomainSegmentId range");
  DomainSegmentIndex segment_index(vertices);

  RestrictedGeometry geometry;
  geometry.domain_segments.reserve(vertices.size());
  for (std::size_t index = 0; index < vertices.size(); ++index) {
    const std::size_t next = (index + 1) % vertices.size();
    const bool axis = vertices[index].point.r == 0.0 &&
                      vertices[next].point.r == 0.0;
    geometry.domain_segments.push_back(DomainSegmentRecord{
        static_cast<DomainSegmentId>(index), kInvalidId, kInvalidId,
        vertices[index].input_index, vertices[next].input_index, axis});
  }

  NodeRegistry nodes(dual, vertices, geometry);
  for (DomainSegmentRecord& segment : geometry.domain_segments) {
    CandidatePoint start;
    start.point = domain_point_exact(vertices[segment.id].point);
    start.aliases.push_back(domain_key(segment.start_vertex));
    start.domain_vertices.push_back(segment.start_vertex);
    start.supports.push_back(segment.id);
    start.supports.push_back(static_cast<DomainSegmentId>(
        (static_cast<std::size_t>(segment.id) + vertices.size() - 1) %
        vertices.size()));
    segment.start_node = nodes.get(start);

    const std::size_t next =
        (static_cast<std::size_t>(segment.id) + 1) % vertices.size();
    CandidatePoint end;
    end.point = domain_point_exact(vertices[next].point);
    end.aliases.push_back(domain_key(segment.end_vertex));
    end.domain_vertices.push_back(segment.end_vertex);
    end.supports.push_back(segment.id);
    end.supports.push_back(static_cast<DomainSegmentId>(
        (next + vertices.size() - 1) % vertices.size()));
    segment.end_node = nodes.get(end);
  }
  const auto setup_end = std::chrono::steady_clock::now();

  const auto gencheck_start = std::chrono::steady_clock::now();
  for (const Site& site : dt.sites) {
    const HomogeneousPoint point{
        Expansion{site.r}, Expansion{site.z}, Expansion{1.0}};
    const DomainQueryBox box =
        certified_point_box(site.r, site.z, segment_index);
    TENRYU_ASSERT(point_in_closed_domain(point, vertices, segment_index, box),
                  "generator_outside_cell");
  }
  const auto gencheck_end = std::chrono::steady_clock::now();

  const auto classify_start = std::chrono::steady_clock::now();
  std::vector<CandidatePoint> dual_candidates;
  dual_candidates.reserve(dual.nodes.size());
  const bool gpu_classify_active =
      tess_gpu_dual_enabled() && g_tess_gpu_restrict_enabled &&
      !tess_gpu_forced_fallback_for_tests() && !dual.nodes.empty();
  GpuClassifyView classify_view;
  std::vector<std::int32_t> node_reps;
  std::vector<double> segments_rz;
  if (gpu_classify_active) {
    node_reps.reserve(dual.nodes.size());
    for (const VoronoiDualNode& node : dual.nodes) {
      node_reps.push_back(static_cast<std::int32_t>(
          dual.classes[node.representative_class].triangles.front()));
    }
    segments_rz.reserve(4 * vertices.size());
    for (std::size_t segment = 0; segment < vertices.size(); ++segment) {
      const DomainPoint& a = vertices[segment].point;
      const DomainPoint& b =
          vertices[(segment + 1) % vertices.size()].point;
      segments_rz.push_back(a.r);
      segments_rz.push_back(a.z);
      segments_rz.push_back(b.r);
      segments_rz.push_back(b.z);
    }
    classify_view = gpu_classify_batch(
        dual.nodes.size(), node_reps.data(), dual.triangle_class.size(),
        segments_rz.data(), vertices.size());
  }
  for (VoronoiNodeId node = 0;
       node < static_cast<VoronoiNodeId>(dual.nodes.size()); ++node) {
    if (gpu_classify_active &&
        classify_view.status[static_cast<std::size_t>(node)] != 2) {
      DomainClassification classification;
      classification.inside =
          classify_view.status[static_cast<std::size_t>(node)] == 1;
      dual_candidates.push_back(dual_endpoint_candidate_with(
          dual, node, vertices, classification));
    } else {
      dual_candidates.push_back(
          dual_endpoint_candidate(dual, node, vertices, segment_index));
    }
  }
  if (gpu_classify_active) {
    g_tess_gpu_last_classify_shortlist = classify_view.uncertain;
    if (timing_enabled) {
      std::fprintf(stderr,
                   "[tess_timing_restrict_gpu] gpu_classify_ms=%.1f "
                   "cls_in=%zu cls_out=%zu cls_short=%zu\n",
                   classify_view.gpu_ms, classify_view.certified_inside,
                   classify_view.certified_outside,
                   classify_view.uncertain);
    }
  }
  const auto classify_end = std::chrono::steady_clock::now();

  geometry.interior_edges.reserve(dual.features.size());
  const auto features_start = std::chrono::steady_clock::now();
  double fastpath_ns = 0.0;
  double crossing_ns = 0.0;
  for (const VoronoiDualFeature& feature : dual.features) {
    const auto feature_start =
        timing_enabled ? std::chrono::steady_clock::now()
                       : std::chrono::steady_clock::time_point{};
    if (feature.kind == DualFeatureKind::kBoundedSegment) {
      const CandidatePoint& first_ref =
          dual_candidates[feature.first_node];
      const CandidatePoint& second_ref =
          dual_candidates[feature.second_node];
      if (first_ref.inside && second_ref.inside) {
        const RestrictedNodeId endpoint0 = nodes.get(first_ref);
        const RestrictedNodeId endpoint1 = nodes.get(second_ref);
        const bool bits_same = bitwise_identical(
            geometry.nodes[endpoint0].constructed,
            geometry.nodes[endpoint1].constructed);
        if (endpoint0 == endpoint1 || bits_same) {
          // Lazy: the exact identity result only gates the diagnostic
          // below; the contraction decision uses ids/bits alone, so the
          // bulk path (distinct nodes, distinct bits) must not pay for
          // exact expansion arithmetic.
          const bool exact_same = same_exact_point(
              geometry.nodes[endpoint0].exact_point,
              geometry.nodes[endpoint1].exact_point);
          if (!exact_same && bits_same && endpoint0 != endpoint1) {
            std::fprintf(stderr,
                         "[contract_inexact_inside] e0=%d e1=%d e0_bp=%d "
                         "e1_bp=%d c=%.17e,%.17e\n",
                         static_cast<int>(endpoint0),
                         static_cast<int>(endpoint1),
                         geometry.nodes[endpoint0].boundary_param.valid ? 1
                                                                         : 0,
                         geometry.nodes[endpoint1].boundary_param.valid ? 1
                                                                         : 0,
                         geometry.nodes[endpoint0].constructed[0],
                         geometry.nodes[endpoint0].constructed[1]);
          }
          geometry.identity_contractions.push_back({{endpoint0, endpoint1}});
          if (timing_enabled) {
            fastpath_ns += std::chrono::duration<double, std::nano>(
                               std::chrono::steady_clock::now() -
                               feature_start)
                               .count();
          }
          continue;
        }
        geometry.interior_edges.push_back(
            RestrictedDualEdge{feature.primal_sites, endpoint0, endpoint1});
        if (timing_enabled) {
          fastpath_ns += std::chrono::duration<double, std::nano>(
                             std::chrono::steady_clock::now() - feature_start)
                             .count();
        }
        continue;
      }
    }
    CandidatePoint first = dual_candidates[feature.first_node];
    CandidatePoint second;
    if (feature.kind == DualFeatureKind::kBoundedSegment) {
      second = dual_candidates[feature.second_node];
    }

    std::vector<CandidatePoint> candidates;
    if (first.inside) {
      candidates.push_back(first);
    }
    if (feature.kind == DualFeatureKind::kBoundedSegment &&
        second.inside) {
      candidates.push_back(second);
    }

    const bool requires_crossings =
        feature.kind == DualFeatureKind::kUnboundedRay ||
        !first.inside || !second.inside;
    if (requires_crossings) {
      DomainQueryBox feature_box = certified_point_box(
          dual.nodes[feature.first_node].constructed[0],
          dual.nodes[feature.first_node].constructed[1], segment_index);
      if (feature.kind == DualFeatureKind::kBoundedSegment) {
        const DomainQueryBox second_box = certified_point_box(
            dual.nodes[feature.second_node].constructed[0],
            dual.nodes[feature.second_node].constructed[1], segment_index);
        feature_box.r_lo = std::min(feature_box.r_lo, second_box.r_lo);
        feature_box.r_hi = std::max(feature_box.r_hi, second_box.r_hi);
        feature_box.z_lo = std::min(feature_box.z_lo, second_box.z_lo);
        feature_box.z_hi = std::max(feature_box.z_hi, second_box.z_hi);
        // A bounded hit lies in the exact endpoint AABB, which these boxes span.
      } else {
        const int r_sign = expansion_sign(feature.ray_direction.r);
        const int z_sign = expansion_sign(feature.ray_direction.z);
        if (r_sign > 0) {
          feature_box.r_hi = segment_index.r_hi;
        } else if (r_sign < 0) {
          feature_box.r_lo = segment_index.r_lo;
        } else {
          feature_box.r_lo = segment_index.r_lo;
          feature_box.r_hi = segment_index.r_hi;
        }
        if (z_sign > 0) {
          feature_box.z_hi = segment_index.z_hi;
        } else if (z_sign < 0) {
          feature_box.z_lo = segment_index.z_lo;
        } else {
          feature_box.z_lo = segment_index.z_lo;
          feature_box.z_hi = segment_index.z_hi;
        }
        // Exact direction signs bound each ray axis from its origin; accepted
        // domain hits are bounded on the other side by the domain AABB.
      }
      const std::vector<DomainSegmentId>& segment_ids =
          segment_index.query_box(feature_box.r_lo, feature_box.r_hi,
                                  feature_box.z_lo, feature_box.z_hi);
      for (const DomainSegmentId segment_id : segment_ids) {
        const DomainSegmentRecord& segment =
            geometry.domain_segments[static_cast<std::size_t>(segment_id)];
        const std::size_t start = static_cast<std::size_t>(segment.id);
        const std::size_t end = (start + 1) % vertices.size();
        const SegmentHit hit = bisector_segment_intersection_exact(
            dt.sites[feature.primal_sites[0]],
            dt.sites[feature.primal_sites[1]], domain_site(vertices[start]),
            domain_site(vertices[end]));
        if (!hit.exists) {
          continue;
        }
        const ExactRational lambda = bisector_edge_parameter(
            dt.sites[feature.primal_sites[0]],
            dt.sites[feature.primal_sites[1]], vertices[start].point,
            vertices[end].point);
        const bool point_on_segment = point_on_domain_segment(
            hit.point, vertices[start].point, vertices[end].point);
        const bool lambda_on_segment =
            exact_rational_compare_unit(lambda, 0) >= 0 &&
            exact_rational_compare_unit(lambda, 1) <= 0;
        TENRYU_ASSERT(point_on_segment == lambda_on_segment,
                      "carrier_lambda_inconsistent");
        if (!point_on_segment) {
          continue;
        }
        const bool on_feature =
            feature.kind == DualFeatureKind::kBoundedSegment
                ? point_on_bounded_feature(hit.point, first.point,
                                           second.point)
                : point_on_ray_feature(hit.point, first.point,
                                       feature.ray_direction);
        if (!on_feature) {
          continue;
        }
        candidates.push_back(crossing_candidate(
            dt, feature, segment, vertices, hit.point, lambda));
      }
    }

    const auto projection_less = [&](const CandidatePoint& lhs,
                                     const CandidatePoint& rhs) {
      const int order =
          feature.kind == DualFeatureKind::kBoundedSegment
              ? projection_compare_bounded(lhs.point, rhs.point,
                                           first.point, second.point)
              : projection_compare_ray(lhs.point, rhs.point,
                                       feature.ray_direction);
      if (order != 0) {
        return order < 0;
      }
      return lhs.aliases < rhs.aliases;
    };
    std::sort(candidates.begin(), candidates.end(), projection_less);
    std::vector<CandidatePoint> unique;
    for (const CandidatePoint& candidate : candidates) {
      if (!unique.empty() &&
          same_exact_point(unique.back().point, candidate.point)) {
        merge_candidate(unique.back(), candidate);
      } else {
        unique.push_back(candidate);
      }
    }

    if (unique.empty()) {
      if (timing_enabled) {
        crossing_ns += std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - feature_start)
                           .count();
      }
      continue;
    }
    if (unique.size() == 1) {
      if (timing_enabled) {
        crossing_ns += std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - feature_start)
                           .count();
      }
      continue;
    }
    RestrictedNodeId endpoint0 = nodes.get(unique.front());
    RestrictedNodeId endpoint1 = nodes.get(unique.back());
    const bool exact_same = same_exact_point(
        geometry.nodes[endpoint0].exact_point,
        geometry.nodes[endpoint1].exact_point);
    const bool bits_same = bitwise_identical(
        geometry.nodes[endpoint0].constructed,
        geometry.nodes[endpoint1].constructed);
    if (endpoint0 == endpoint1 || exact_same || bits_same) {
      if (!exact_same && bits_same && endpoint0 != endpoint1) {
        std::fprintf(stderr,
                     "[contract_inexact] e0=%d e1=%d e0_bp=%d e1_bp=%d "
                     "c=%.17e,%.17e\n",
                     static_cast<int>(endpoint0),
                     static_cast<int>(endpoint1),
                     geometry.nodes[endpoint0].boundary_param.valid ? 1 : 0,
                     geometry.nodes[endpoint1].boundary_param.valid ? 1 : 0,
                     geometry.nodes[endpoint0].constructed[0],
                     geometry.nodes[endpoint0].constructed[1]);
      }
      geometry.identity_contractions.push_back({{endpoint0, endpoint1}});
      if (timing_enabled) {
        crossing_ns += std::chrono::duration<double, std::nano>(
                           std::chrono::steady_clock::now() - feature_start)
                           .count();
      }
      continue;
    }
    geometry.interior_edges.push_back(
        RestrictedDualEdge{feature.primal_sites, endpoint0, endpoint1});
    if (timing_enabled) {
      crossing_ns += std::chrono::duration<double, std::nano>(
                         std::chrono::steady_clock::now() - feature_start)
                         .count();
    }
  }
  const auto features_end = std::chrono::steady_clock::now();

  std::map<DomainSegmentId, std::vector<const RestrictedNode*>>
      carrier_nodes;
  for (const RestrictedNode& node : geometry.nodes) {
    if (node.boundary_param.valid) {
      carrier_nodes[node.boundary_param.edge].push_back(&node);
    }
  }
  for (auto& [edge, edge_nodes] : carrier_nodes) {
    (void)edge;
    std::sort(edge_nodes.begin(), edge_nodes.end(),
              [](const RestrictedNode* lhs, const RestrictedNode* rhs) {
                return exact_rational_compare(lhs->boundary_param.lambda,
                                              rhs->boundary_param.lambda) < 0;
              });
    for (std::size_t index = 1; index < edge_nodes.size(); ++index) {
      const RestrictedNode& previous = *edge_nodes[index - 1];
      const RestrictedNode& current = *edge_nodes[index];
      if (exact_rational_compare(previous.boundary_param.lambda,
                                 current.boundary_param.lambda) != 0 &&
          bitwise_identical(previous.constructed, current.constructed)) {
        geometry.carrier_representability_ok = false;
      }
    }
  }

  if (timing_enabled) {
    std::fprintf(
        stderr,
        "[tess_timing_restrict] setup_ms=%.1f gencheck_ms=%.1f "
        "classify_ms=%.1f features_ms=%.1f fastpath_ms=%.1f "
        "crossing_ms=%.1f n_domain_vertices=%zu "
        "n_domain_segments=%zu\n",
        std::chrono::duration<double, std::milli>(setup_end - setup_start)
            .count(),
        std::chrono::duration<double, std::milli>(gencheck_end -
                                                  gencheck_start)
            .count(),
        std::chrono::duration<double, std::milli>(classify_end -
                                                  classify_start)
            .count(),
        std::chrono::duration<double, std::milli>(features_end -
                                                  features_start)
            .count(),
        fastpath_ns / 1.0e6, crossing_ns / 1.0e6,
        vertices.size(), geometry.domain_segments.size());
  }
  return geometry;
}

}  // namespace tenryu::mesh::tess
