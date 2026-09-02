#include "mesh/tessellation/exact_types.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

#include "core/error.hpp"
#include "mesh/tessellation/sos_policy.hpp"

namespace tenryu::mesh::tess {
namespace detail {

bool g_forced_exact_for_tests = false;
thread_local PredicateCounters* g_predicate_counters = nullptr;
bool g_predicate_counters_active = false;

}  // namespace detail

namespace {

Expansion scalar_expansion(const double value) {
  return Expansion{value};
}

Expansion expansion_dot(const Expansion& ax,
                        const Expansion& ay,
                        const Expansion& bx,
                        const Expansion& by) {
  return expansion_sum(expansion_product(ax, bx),
                       expansion_product(ay, by));
}

HomogeneousPoint normalize_homogeneous(HomogeneousPoint point) {
  const int w_sign = expansion_sign(point.w);
  TENRYU_ASSERT(w_sign != 0,
                "homogeneous point requires a nonzero denominator");
  if (w_sign < 0) {
    point.x = expansion_negate(point.x);
    point.y = expansion_negate(point.y);
    point.w = expansion_negate(point.w);
  }
  return point;
}

int compare_rational_to_double(const ExactRational& rational,
                               const double value) {
  return expansion_sign(expansion_difference(
      rational.num, expansion_scale(rational.den, value)));
}

int compare_rational_to_midpoint(const ExactRational& rational,
                                 const double first,
                                 const double second) {
  return expansion_sign(expansion_difference(
      expansion_scale(rational.num, 2.0),
      expansion_sum(expansion_scale(rational.den, first),
                    expansion_scale(rational.den, second))));
}

bool double_is_even(const double value) {
  return (std::bit_cast<std::uint64_t>(value) & UINT64_C(1)) == 0;
}

}  // namespace

ExactRational make_exact_rational(Expansion num, Expansion den) {
  const int den_sign = expansion_sign(den);
  TENRYU_ASSERT(den_sign != 0,
                "exact rational requires a nonzero denominator");
  if (den_sign < 0) {
    num = expansion_negate(num);
    den = expansion_negate(den);
  }
  return ExactRational{std::move(num), std::move(den)};
}

int exact_rational_compare(const ExactRational& a,
                           const ExactRational& b) {
  return expansion_sign(expansion_difference(
      expansion_product(a.num, b.den),
      expansion_product(b.num, a.den)));
}

int exact_rational_compare_unit(const ExactRational& a,
                                const int bound01) {
  TENRYU_ASSERT(bound01 == 0 || bound01 == 1,
                "exact rational unit bound must be zero or one");
  return bound01 == 0
             ? expansion_sign(a.num)
             : expansion_sign(expansion_difference(a.num, a.den));
}

double exact_rational_round(const ExactRational& a) {
  TENRYU_ASSERT(expansion_sign(a.den) > 0,
                "exact rational denominator must be positive");
  double candidate = approx_of_expansion(a.num).value /
                     approx_of_expansion(a.den).value;
  TENRYU_ASSERT(std::isfinite(candidate),
                "exact rational approximation must be finite");

  while (true) {
    const int direction = compare_rational_to_double(a, candidate);
    if (direction == 0) {
      return candidate;
    }
    const double neighbor = std::nextafter(
        candidate, direction > 0 ? std::numeric_limits<double>::infinity()
                                 : -std::numeric_limits<double>::infinity());
    TENRYU_ASSERT(std::isfinite(neighbor),
                  "exact rational rounded value must be finite");
    const int midpoint =
        compare_rational_to_midpoint(a, candidate, neighbor);
    const bool neighbor_is_closer =
        direction > 0 ? midpoint > 0 : midpoint < 0;
    const bool tied_toward_even =
        midpoint == 0 && !double_is_even(candidate) &&
        double_is_even(neighbor);
    if (!neighbor_is_closer && !tied_toward_even) {
      return candidate;
    }
    candidate = neighbor;
  }
}

HomogeneousPoint circumcenter_exact(const Site& a,
                                    const Site& b,
                                    const Site& c) {
  const Expansion ar = scalar_expansion(a.r);
  const Expansion az = scalar_expansion(a.z);
  const Expansion br = scalar_expansion(b.r);
  const Expansion bz = scalar_expansion(b.z);
  const Expansion cr = scalar_expansion(c.r);
  const Expansion cz = scalar_expansion(c.z);

  const Expansion a_lift = expansion_lift(ar, az);
  const Expansion b_lift = expansion_lift(br, bz);
  const Expansion c_lift = expansion_lift(cr, cz);

  const Expansion numerator_r = expansion_sum(
      expansion_sum(
          expansion_product(a_lift, exact_difference(b.z, c.z)),
          expansion_product(b_lift, exact_difference(c.z, a.z))),
      expansion_product(c_lift, exact_difference(a.z, b.z)));
  const Expansion numerator_z = expansion_sum(
      expansion_sum(
          expansion_product(a_lift, exact_difference(c.r, b.r)),
          expansion_product(b_lift, exact_difference(a.r, c.r))),
      expansion_product(c_lift, exact_difference(b.r, a.r)));

  const Expansion ba_r = exact_difference(b.r, a.r);
  const Expansion ba_z = exact_difference(b.z, a.z);
  const Expansion ca_r = exact_difference(c.r, a.r);
  const Expansion ca_z = exact_difference(c.z, a.z);
  const Expansion denominator =
      expansion_scale(expansion_cross(ba_r, ba_z, ca_r, ca_z), 2.0);

  return normalize_homogeneous(
      HomogeneousPoint{numerator_r, numerator_z, denominator});
}

SegmentHit bisector_segment_intersection_exact(
    const Site& i,
    const Site& j,
    const Site& segment_a,
    const Site& segment_b) {
  const Expansion site_dr = exact_difference(j.r, i.r);
  const Expansion site_dz = exact_difference(j.z, i.z);
  const Expansion segment_dr = exact_difference(segment_b.r, segment_a.r);
  const Expansion segment_dz = exact_difference(segment_b.z, segment_a.z);

  const Expansion denominator = expansion_scale(
      expansion_dot(site_dr, site_dz, segment_dr, segment_dz), 2.0);
  if (expansion_sign(denominator) == 0) {
    return SegmentHit{false, HomogeneousPoint{}};
  }

  const Expansion i_lift = expansion_lift(
      scalar_expansion(i.r), scalar_expansion(i.z));
  const Expansion j_lift = expansion_lift(
      scalar_expansion(j.r), scalar_expansion(j.z));
  const Expansion twice_site_dot_start = expansion_scale(
      expansion_dot(site_dr, site_dz, scalar_expansion(segment_a.r),
                    scalar_expansion(segment_a.z)),
      2.0);
  const Expansion parameter_numerator = expansion_difference(
      expansion_difference(j_lift, i_lift), twice_site_dot_start);

  const Expansion x = expansion_sum(
      expansion_scale(denominator, segment_a.r),
      expansion_product(segment_dr, parameter_numerator));
  const Expansion y = expansion_sum(
      expansion_scale(denominator, segment_a.z),
      expansion_product(segment_dz, parameter_numerator));

  return SegmentHit{
      true, normalize_homogeneous(HomogeneousPoint{x, y, denominator})};
}

bool on_segment_exact(const Site& a, const Site& b, const Site& p) {
  if (orient2d_sign_tess(a, b, p) != 0) {
    return false;
  }
  return p.r >= std::min(a.r, b.r) && p.r <= std::max(a.r, b.r) &&
         p.z >= std::min(a.z, b.z) && p.z <= std::max(a.z, b.z);
}

int projection_order(const Site& a,
                     const Site& b,
                     const Site& p,
                     const Site& q) {
  const Expansion direction_r = exact_difference(b.r, a.r);
  const Expansion direction_z = exact_difference(b.z, a.z);
  const Expansion pq_r = exact_difference(p.r, q.r);
  const Expansion pq_z = exact_difference(p.z, q.z);
  const int projection_sign = expansion_sign(
      expansion_dot(pq_r, pq_z, direction_r, direction_z));
  if (projection_sign != 0) {
    return projection_sign;
  }
  if (p.stable_id < q.stable_id) {
    return -1;
  }
  if (p.stable_id > q.stable_id) {
    return 1;
  }
  return 0;
}

int compare_event_parameters_exact(const Expansion& numerator_a,
                                   const Expansion& denominator_a,
                                   const Expansion& numerator_b,
                                   const Expansion& denominator_b) {
  TENRYU_ASSERT(expansion_sign(denominator_a) > 0,
                "event parameter denominator must be positive");
  TENRYU_ASSERT(expansion_sign(denominator_b) > 0,
                "event parameter denominator must be positive");
  return expansion_sign(expansion_difference(
      expansion_product(numerator_a, denominator_b),
      expansion_product(numerator_b, denominator_a)));
}

bool event_parameter_less_exact(const Expansion& numerator_a,
                                const Expansion& denominator_a,
                                const Expansion& numerator_b,
                                const Expansion& denominator_b) {
  return compare_event_parameters_exact(numerator_a, denominator_a,
                                        numerator_b, denominator_b) < 0;
}

bool event_parameters_equal_exact(const Expansion& numerator_a,
                                  const Expansion& denominator_a,
                                  const Expansion& numerator_b,
                                  const Expansion& denominator_b) {
  return compare_event_parameters_exact(numerator_a, denominator_a,
                                        numerator_b, denominator_b) == 0;
}

}  // namespace tenryu::mesh::tess
