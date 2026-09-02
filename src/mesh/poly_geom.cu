#include "mesh/poly_geom.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::reale {
namespace {

constexpr double kPiOverThree =
    1.0471975511965977461542144610931676280657231331250352736615;
constexpr double kTwoPi =
    6.283185307179586476925286766559005768394338798750211641949;

// Shewchuk's epsilon is half std::numeric_limits<double>::epsilon(). These are
// the predicates.c ccwerrboundA and iccerrboundA forward-error bounds.
constexpr double kShewchukEpsilon =
    0.5 * std::numeric_limits<double>::epsilon();
constexpr double kCcwErrBoundA =
    (3.0 + 16.0 * kShewchukEpsilon) * kShewchukEpsilon;
constexpr double kIncircleErrBoundA =
    (10.0 + 96.0 * kShewchukEpsilon) * kShewchukEpsilon;

using Expansion = std::vector<double>;

void two_sum(const double a,
             const double b,
             double& sum,
             double& error) {
  sum = a + b;
  const double b_virtual = sum - a;
  const double a_virtual = sum - b_virtual;
  const double b_roundoff = b - b_virtual;
  const double a_roundoff = a - a_virtual;
  error = a_roundoff + b_roundoff;
}

void two_diff(const double a,
              const double b,
              double& difference,
              double& error) {
  difference = a - b;
  const double b_virtual = a - difference;
  const double a_virtual = difference + b_virtual;
  const double b_roundoff = b_virtual - b;
  const double a_roundoff = a - a_virtual;
  error = a_roundoff + b_roundoff;
}

void two_product(const double a,
                 const double b,
                 double& product,
                 double& error) {
  product = a * b;
  error = std::fma(a, b, -product);
}

void append_nonzero(Expansion& expansion, const double value) {
  if (value != 0.0) {
    expansion.push_back(value);
  }
}

Expansion expansion_sum(const Expansion& lhs, const Expansion& rhs) {
  if (lhs.empty()) {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }

  Expansion result;
  result.reserve(lhs.size() + rhs.size());
  std::size_t lhs_index = 0;
  std::size_t rhs_index = 0;
  const auto next_component = [&]() {
    if (rhs_index == rhs.size() ||
        (lhs_index != lhs.size() &&
         std::abs(lhs[lhs_index]) <= std::abs(rhs[rhs_index]))) {
      return lhs[lhs_index++];
    }
    return rhs[rhs_index++];
  };

  double accumulator = next_component();
  while (lhs_index != lhs.size() || rhs_index != rhs.size()) {
    double sum = 0.0;
    double error = 0.0;
    two_sum(accumulator, next_component(), sum, error);
    append_nonzero(result, error);
    accumulator = sum;
  }
  if (accumulator != 0.0 || result.empty()) {
    result.push_back(accumulator);
  }
  return result;
}

Expansion expansion_negate(const Expansion& expansion) {
  Expansion result = expansion;
  for (double& component : result) {
    component = -component;
  }
  return result;
}

Expansion expansion_difference(const Expansion& lhs,
                               const Expansion& rhs) {
  return expansion_sum(lhs, expansion_negate(rhs));
}

Expansion exact_difference(const double a, const double b) {
  double difference = 0.0;
  double error = 0.0;
  two_diff(a, b, difference, error);
  Expansion result;
  result.reserve(2);
  append_nonzero(result, error);
  if (difference != 0.0 || result.empty()) {
    result.push_back(difference);
  }
  return result;
}

Expansion expansion_scale(const Expansion& expansion, const double scale) {
  Expansion result;
  for (const double component : expansion) {
    double product = 0.0;
    double error = 0.0;
    two_product(component, scale, product, error);
    Expansion term;
    term.reserve(2);
    append_nonzero(term, error);
    if (product != 0.0 || term.empty()) {
      term.push_back(product);
    }
    result = expansion_sum(result, term);
  }
  return result;
}

Expansion expansion_product(const Expansion& lhs, const Expansion& rhs) {
  Expansion result;
  for (const double component : rhs) {
    result = expansion_sum(result, expansion_scale(lhs, component));
  }
  return result;
}

Expansion expansion_cross(const Expansion& ax,
                          const Expansion& ay,
                          const Expansion& bx,
                          const Expansion& by) {
  return expansion_difference(expansion_product(ax, by),
                              expansion_product(ay, bx));
}

Expansion expansion_lift(const Expansion& x, const Expansion& y) {
  return expansion_sum(expansion_product(x, x), expansion_product(y, y));
}

int expansion_sign(const Expansion& expansion) {
  for (auto component = expansion.rbegin(); component != expansion.rend();
       ++component) {
    if (*component > 0.0) {
      return 1;
    }
    if (*component < 0.0) {
      return -1;
    }
  }
  return 0;
}

int orient2d_exact_sign(const double ax,
                        const double ay,
                        const double bx,
                        const double by,
                        const double cx,
                        const double cy) {
  const Expansion acx = exact_difference(ax, cx);
  const Expansion acy = exact_difference(ay, cy);
  const Expansion bcx = exact_difference(bx, cx);
  const Expansion bcy = exact_difference(by, cy);
  return expansion_sign(expansion_cross(acx, acy, bcx, bcy));
}

int incircle_exact_sign(const double ax,
                        const double ay,
                        const double bx,
                        const double by,
                        const double cx,
                        const double cy,
                        const double dx,
                        const double dy) {
  const Expansion adx = exact_difference(ax, dx);
  const Expansion ady = exact_difference(ay, dy);
  const Expansion bdx = exact_difference(bx, dx);
  const Expansion bdy = exact_difference(by, dy);
  const Expansion cdx = exact_difference(cx, dx);
  const Expansion cdy = exact_difference(cy, dy);

  const Expansion alift = expansion_lift(adx, ady);
  const Expansion blift = expansion_lift(bdx, bdy);
  const Expansion clift = expansion_lift(cdx, cdy);
  const Expansion bcdet = expansion_cross(bdx, bdy, cdx, cdy);
  const Expansion cadet = expansion_cross(cdx, cdy, adx, ady);
  const Expansion abdet = expansion_cross(adx, ady, bdx, bdy);

  const Expansion determinant = expansion_sum(
      expansion_sum(expansion_product(alift, bcdet),
                    expansion_product(blift, cadet)),
      expansion_product(clift, abdet));
  return expansion_sign(determinant);
}

struct Point {
  double r;
  double z;
};

double edge_side_value(const Point& edge_start,
                       const Point& edge_end,
                       const Point& point) {
  return std::fma(edge_end.r - edge_start.r, point.z - edge_start.z,
                  -((edge_end.z - edge_start.z) *
                    (point.r - edge_start.r)));
}

bool point_inside_edge(const Point& edge_start,
                       const Point& edge_end,
                       const Point& point,
                       const int clip_orientation) {
  const int side = orient2d_sign(edge_start.r, edge_start.z,
                                 edge_end.r, edge_end.z,
                                 point.r, point.z);
  return side == 0 || side == clip_orientation;
}

Point segment_line_intersection(const Point& segment_start,
                                const Point& segment_end,
                                const Point& line_start,
                                const Point& line_end) {
  const double start_side =
      edge_side_value(line_start, line_end, segment_start);
  const double end_side =
      edge_side_value(line_start, line_end, segment_end);
  const double fraction = start_side / (start_side - end_side);
  return {
      std::fma(fraction, segment_end.r - segment_start.r, segment_start.r),
      std::fma(fraction, segment_end.z - segment_start.z, segment_start.z),
  };
}

int convex_orientation(const double* r, const double* z, const int n) {
  int orientation = 0;
  for (int k = 0; k < n; ++k) {
    const int next = (k + 1 == n) ? 0 : k + 1;
    const int next_next = (next + 1 == n) ? 0 : next + 1;
    const int edge_orientation = orient2d_sign(
        r[k], z[k], r[next], z[next], r[next_next], z[next_next]);
    TENRYU_ASSERT(edge_orientation != 0,
                  "ReALE clip polygon must be strictly convex");
    if (orientation == 0) {
      orientation = edge_orientation;
    }
    TENRYU_ASSERT(edge_orientation == orientation,
                  "ReALE clip polygon must be convex");
  }
  return orientation;
}

struct Triangle {
  double r[3];
  double z[3];
  int weight;
};

bool fan_triangle(const double* polygon_r,
                  const double* polygon_z,
                  const int vertex,
                  Triangle& triangle) {
  triangle.r[0] = polygon_r[0];
  triangle.z[0] = polygon_z[0];
  triangle.r[1] = polygon_r[vertex];
  triangle.z[1] = polygon_z[vertex];
  triangle.r[2] = polygon_r[vertex + 1];
  triangle.z[2] = polygon_z[vertex + 1];

  double scale_squared = 0.0;
  for (int edge = 0; edge < 3; ++edge) {
    const int next = (edge + 1 == 3) ? 0 : edge + 1;
    const double dr = triangle.r[next] - triangle.r[edge];
    const double dz = triangle.z[next] - triangle.z[edge];
    scale_squared =
        std::max(scale_squared, std::fma(dr, dr, dz * dz));
  }
  const double area = 0.5 * edge_side_value(
      {triangle.r[0], triangle.z[0]},
      {triangle.r[1], triangle.z[1]},
      {triangle.r[2], triangle.z[2]});
  const double area_guard =
      4.0 * std::numeric_limits<double>::epsilon() * scale_squared;
  if (std::abs(area) <= area_guard) {
    return false;
  }

  const int orientation = orient2d_sign(
      triangle.r[0], triangle.z[0], triangle.r[1], triangle.z[1],
      triangle.r[2], triangle.z[2]);
  if (orientation == 0) {
    return false;
  }
  triangle.weight = orientation;
  if (orientation < 0) {
    std::swap(triangle.r[1], triangle.r[2]);
    std::swap(triangle.z[1], triangle.z[2]);
  }
  return true;
}

}  // namespace

RZMoments polygon_rz_moments(const double* r, const double* z, const int n) {
  RZMoments result{};

  // Local fallback copy of hydro::rz::rz_polygon_volume_exact, the frozen RZ
  // volume authority; including its hydro header here would invert layering.
  double volume_sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const int next = (k + 1 == n) ? 0 : k + 1;
    volume_sum +=
        (r[k] * z[next] - r[next] * z[k]) * (r[k] + r[next]);
  }
  result.volume = kPiOverThree * volume_sum;

  const moments::PolyRZMoments planar =
      moments::poly_rz_moments_fan(r, z, n);
  result.moment_r = kTwoPi * planar.mrr;
  result.moment_z = kTwoPi * planar.mrz;
  return result;
}

int clip_polygon_convex(const double* subj_r,
                        const double* subj_z,
                        const int subj_n,
                        const double* clip_r,
                        const double* clip_z,
                        const int clip_n,
                        double* out_r,
                        double* out_z,
                        const int out_cap) {
  TENRYU_ASSERT(subj_n >= 3, "ReALE subject polygon must have three vertices");
  TENRYU_ASSERT(clip_n >= 3, "ReALE clip polygon must have three vertices");
  TENRYU_ASSERT(out_cap >= 0, "ReALE clip output capacity must be nonnegative");
  const int clip_orientation = convex_orientation(clip_r, clip_z, clip_n);

  std::vector<Point> input;
  input.reserve(static_cast<std::size_t>(subj_n + clip_n));
  for (int k = 0; k < subj_n; ++k) {
    input.push_back({subj_r[k], subj_z[k]});
  }

  std::vector<Point> output;
  output.reserve(static_cast<std::size_t>(subj_n + clip_n));
  for (int edge = 0; edge < clip_n; ++edge) {
    if (input.empty()) {
      return 0;
    }
    output.clear();
    const int next_edge = (edge + 1 == clip_n) ? 0 : edge + 1;
    const Point line_start{clip_r[edge], clip_z[edge]};
    const Point line_end{clip_r[next_edge], clip_z[next_edge]};
    Point previous = input.back();
    bool previous_inside =
        point_inside_edge(line_start, line_end, previous, clip_orientation);
    for (const Point& current : input) {
      const bool current_inside =
          point_inside_edge(line_start, line_end, current, clip_orientation);
      if (current_inside != previous_inside) {
        output.push_back(segment_line_intersection(
            previous, current, line_start, line_end));
      }
      if (current_inside) {
        output.push_back(current);
      }
      previous = current;
      previous_inside = current_inside;
    }
    input.swap(output);
  }

  TENRYU_ASSERT(static_cast<int>(input.size()) <= out_cap,
                "ReALE clip output capacity exceeded");
  for (std::size_t k = 0; k < input.size(); ++k) {
    out_r[k] = input[k].r;
    out_z[k] = input[k].z;
  }
  return static_cast<int>(input.size());
}

RZMoments intersect_general(const double* a_r,
                            const double* a_z,
                            const int a_n,
                            const double* b_r,
                            const double* b_z,
                            const int b_n) {
  RZMoments result{};
  if (a_n < 3 || b_n < 3) {
    return result;
  }

  for (int a_vertex = 1; a_vertex < a_n - 1; ++a_vertex) {
    Triangle a_triangle{};
    if (!fan_triangle(a_r, a_z, a_vertex, a_triangle)) {
      continue;
    }
    for (int b_vertex = 1; b_vertex < b_n - 1; ++b_vertex) {
      Triangle b_triangle{};
      if (!fan_triangle(b_r, b_z, b_vertex, b_triangle)) {
        continue;
      }

      double clipped_r[8]{};
      double clipped_z[8]{};
      const int clipped_n = clip_polygon_convex(
          a_triangle.r, a_triangle.z, 3,
          b_triangle.r, b_triangle.z, 3,
          clipped_r, clipped_z, 8);
      if (clipped_n < 3) {
        continue;
      }
      const RZMoments piece =
          polygon_rz_moments(clipped_r, clipped_z, clipped_n);
      const double weight =
          static_cast<double>(a_triangle.weight * b_triangle.weight);
      result.volume += weight * piece.volume;
      result.moment_r += weight * piece.moment_r;
      result.moment_z += weight * piece.moment_z;
    }
  }
  return result;
}

int orient2d_sign(const double ax,
                  const double ay,
                  const double bx,
                  const double by,
                  const double cx,
                  const double cy) {
  const double acx = ax - cx;
  const double acy = ay - cy;
  const double bcx = bx - cx;
  const double bcy = by - cy;
  const double left = acx * bcy;
  const double right = acy * bcx;
  const double determinant = left - right;
  const double error_bound =
      kCcwErrBoundA * (std::abs(left) + std::abs(right));
  if (std::abs(determinant) > error_bound) {
    return determinant > 0.0 ? 1 : -1;
  }
  return orient2d_exact_sign(ax, ay, bx, by, cx, cy);
}

int incircle_sign(const double ax,
                  const double ay,
                  const double bx,
                  const double by,
                  const double cx,
                  const double cy,
                  const double dx,
                  const double dy) {
  const double adx = ax - dx;
  const double ady = ay - dy;
  const double bdx = bx - dx;
  const double bdy = by - dy;
  const double cdx = cx - dx;
  const double cdy = cy - dy;

  const double bdxcdy = bdx * cdy;
  const double cdxbdy = cdx * bdy;
  const double cdxady = cdx * ady;
  const double adxcdy = adx * cdy;
  const double adxbdy = adx * bdy;
  const double bdxady = bdx * ady;
  const double alift = adx * adx + ady * ady;
  const double blift = bdx * bdx + bdy * bdy;
  const double clift = cdx * cdx + cdy * cdy;
  const double determinant =
      alift * (bdxcdy - cdxbdy) +
      blift * (cdxady - adxcdy) +
      clift * (adxbdy - bdxady);
  const double permanent =
      (std::abs(bdxcdy) + std::abs(cdxbdy)) * alift +
      (std::abs(cdxady) + std::abs(adxcdy)) * blift +
      (std::abs(adxbdy) + std::abs(bdxady)) * clift;
  if (std::abs(determinant) > kIncircleErrBoundA * permanent) {
    return determinant > 0.0 ? 1 : -1;
  }
  return incircle_exact_sign(ax, ay, bx, by, cx, cy, dx, dy);
}

std::uint64_t polygon_topo_hash(const EntityId* node_ids, const int n) {
  TENRYU_ASSERT(n > 0, "ReALE topology hash requires a nonempty polygon");
  int minimum_index = 0;
  for (int k = 1; k < n; ++k) {
    if (node_ids[k] < node_ids[minimum_index]) {
      minimum_index = k;
    }
  }

  constexpr std::uint64_t fnv_offset_basis = 14695981039346656037ULL;
  constexpr std::uint64_t fnv_prime = 1099511628211ULL;
  std::uint64_t hash = fnv_offset_basis;
  for (int k = 0; k < n; ++k) {
    const EntityId id = node_ids[(minimum_index + k) % n];
    for (int byte = 0; byte < 8; ++byte) {
      hash ^= (id >> (8 * byte)) & 0xffULL;
      hash *= fnv_prime;
    }
  }
  return hash;
}

}  // namespace tenryu::mesh::reale
