#include "mesh/reference_flat_path.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "mesh/poly_geom.cuh"

namespace tenryu::mesh {
namespace {

struct ExpansionComponent {
  double significand = 0.0;
  int exponent = 0;
};

// Components are ordered from least to greatest magnitude.  Each component is
// a normalized double significand times an explicit power of two; keeping the
// exponent separate prevents exact products from overflowing or underflowing.
using Expansion = std::vector<ExpansionComponent>;

void two_sum(const double a,
             const double b,
             double& sum,
             double& error) {
  sum = a + b;
  const double b_virtual = sum - a;
  const double a_virtual = sum - b_virtual;
  error = (a - a_virtual) + (b - b_virtual);
}

void two_product(const double a,
                 const double b,
                 double& product,
                 double& error) {
  product = a * b;
  error = std::fma(a, b, -product);
}

ExpansionComponent normalized_component(const double significand,
                                        const int exponent) {
  if (significand == 0.0) {
    return {};
  }
  int adjustment = 0;
  const double normalized = std::frexp(significand, &adjustment);
  return {normalized, exponent + adjustment};
}

void append_nonzero(Expansion& expansion,
                    const ExpansionComponent component) {
  if (component.significand != 0.0) {
    expansion.push_back(component);
  }
}

bool component_less_equal(const ExpansionComponent lhs,
                          const ExpansionComponent rhs) {
  if (lhs.exponent != rhs.exponent) {
    return lhs.exponent < rhs.exponent;
  }
  return std::abs(lhs.significand) <= std::abs(rhs.significand);
}

void two_component_sum(const ExpansionComponent a,
                       const ExpansionComponent b,
                       ExpansionComponent& sum,
                       ExpansionComponent& error) {
  const ExpansionComponent* smaller = &a;
  const ExpansionComponent* larger = &b;
  if (!component_less_equal(a, b)) {
    std::swap(smaller, larger);
  }
  const int exponent_difference = larger->exponent - smaller->exponent;
  if (exponent_difference > 1021) {
    error = *smaller;
    sum = *larger;
    return;
  }

  const int common_exponent = larger->exponent;
  const double scaled_smaller = std::scalbn(
      smaller->significand, smaller->exponent - common_exponent);
  const double scaled_larger = larger->significand;
  double raw_sum = 0.0;
  double raw_error = 0.0;
  two_sum(scaled_smaller, scaled_larger, raw_sum, raw_error);
  error = normalized_component(raw_error, common_exponent);
  sum = normalized_component(raw_sum, common_exponent);
}

void two_component_product(const ExpansionComponent a,
                           const ExpansionComponent b,
                           ExpansionComponent& product,
                           ExpansionComponent& error) {
  double raw_product = 0.0;
  double raw_error = 0.0;
  two_product(a.significand, b.significand, raw_product, raw_error);
  const int exponent = a.exponent + b.exponent;
  error = normalized_component(raw_error, exponent);
  product = normalized_component(raw_product, exponent);
}

Expansion exact_value(const double value) {
  if (value == 0.0) {
    return {};
  }
  int exponent = 0;
  const double significand = std::frexp(value, &exponent);
  return {{significand, exponent}};
}

Expansion expansion_sum(const Expansion& lhs, const Expansion& rhs) {
  if (lhs.empty()) {
    return rhs;
  }
  if (rhs.empty()) {
    return lhs;
  }

  Expansion result = lhs;
  for (const ExpansionComponent addend : rhs) {
    Expansion grown;
    grown.reserve(result.size() + 1U);
    ExpansionComponent accumulator = addend;
    for (const ExpansionComponent component : result) {
      ExpansionComponent sum;
      ExpansionComponent error;
      two_component_sum(accumulator, component, sum, error);
      append_nonzero(grown, error);
      accumulator = sum;
    }
    append_nonzero(grown, accumulator);
    result = std::move(grown);
  }
  return result;
}

Expansion expansion_negate(const Expansion& expansion) {
  Expansion result = expansion;
  for (ExpansionComponent& component : result) {
    component.significand = -component.significand;
  }
  return result;
}

Expansion expansion_difference(const Expansion& lhs,
                               const Expansion& rhs) {
  return expansion_sum(lhs, expansion_negate(rhs));
}

Expansion expansion_scale_component(const Expansion& expansion,
                                    const ExpansionComponent scale) {
  if (scale.significand == 0.0) {
    return {};
  }
  Expansion result;
  for (const ExpansionComponent component : expansion) {
    ExpansionComponent product;
    ExpansionComponent error;
    two_component_product(component, scale, product, error);
    Expansion term;
    term.reserve(2U);
    append_nonzero(term, error);
    append_nonzero(term, product);
    result = expansion_sum(result, term);
  }
  return result;
}

Expansion expansion_scale(const Expansion& expansion, const double scale) {
  const Expansion exact_scale = exact_value(scale);
  return exact_scale.empty()
             ? Expansion{}
             : expansion_scale_component(expansion, exact_scale.front());
}

Expansion expansion_product(const Expansion& lhs, const Expansion& rhs) {
  Expansion result;
  for (const ExpansionComponent component : rhs) {
    result = expansion_sum(result,
                           expansion_scale_component(lhs, component));
  }
  return result;
}

int expansion_sign(const Expansion& expansion) {
  for (auto component = expansion.rbegin(); component != expansion.rend();
       ++component) {
    if (component->significand > 0.0) {
      return 1;
    }
    if (component->significand < 0.0) {
      return -1;
    }
  }
  return 0;
}

bool expansion_equal(const Expansion& lhs, const Expansion& rhs) {
  return expansion_sign(expansion_difference(lhs, rhs)) == 0;
}

bool expansion_less(const Expansion& lhs, const Expansion& rhs) {
  return expansion_sign(expansion_difference(lhs, rhs)) < 0;
}

Expansion midpoint(const Expansion& lhs, const Expansion& rhs) {
  return expansion_scale(expansion_sum(lhs, rhs), 0.5);
}

long double expansion_to_long_double(const Expansion& expansion) {
  long double value = 0.0L;
  for (const ExpansionComponent component : expansion) {
    value += std::scalbn(static_cast<long double>(component.significand),
                         component.exponent);
  }
  return value;
}

struct LinearExpansion {
  Expansion c0;
  Expansion c1;
};

LinearExpansion affine_coordinate(const double start, const double end) {
  return {exact_value(start),
          expansion_difference(exact_value(end), exact_value(start))};
}

LinearExpansion operator-(const LinearExpansion& lhs,
                          const LinearExpansion& rhs) {
  return {expansion_difference(lhs.c0, rhs.c0),
          expansion_difference(lhs.c1, rhs.c1)};
}

struct ExactQuadratic {
  Expansion c0;
  Expansion c1;
  Expansion c2;
};

ExactQuadratic product_difference(const LinearExpansion& a,
                                  const LinearExpansion& b,
                                  const LinearExpansion& c,
                                  const LinearExpansion& d,
                                  const int orientation_sign = 1) {
  const Expansion q0 = expansion_difference(
      expansion_product(a.c0, b.c0), expansion_product(c.c0, d.c0));
  const Expansion q1 = expansion_difference(
      expansion_sum(expansion_product(a.c1, b.c0),
                    expansion_product(a.c0, b.c1)),
      expansion_sum(expansion_product(c.c1, d.c0),
                    expansion_product(c.c0, d.c1)));
  const Expansion q2 = expansion_difference(
      expansion_product(a.c1, b.c1), expansion_product(c.c1, d.c1));
  return {expansion_scale(q0, orientation_sign),
          expansion_scale(q1, orientation_sign),
          expansion_scale(q2, orientation_sign)};
}

ExactQuadratic dot_product(const LinearExpansion& ar,
                           const LinearExpansion& az,
                           const LinearExpansion& br,
                           const LinearExpansion& bz) {
  const Expansion q0 = expansion_sum(expansion_product(ar.c0, br.c0),
                                     expansion_product(az.c0, bz.c0));
  const Expansion q1 = expansion_sum(
      expansion_sum(expansion_product(ar.c1, br.c0),
                    expansion_product(ar.c0, br.c1)),
      expansion_sum(expansion_product(az.c1, bz.c0),
                    expansion_product(az.c0, bz.c1)));
  const Expansion q2 = expansion_sum(expansion_product(ar.c1, br.c1),
                                     expansion_product(az.c1, bz.c1));
  return {q0, q1, q2};
}

ExactQuadratic squared_length(const LinearExpansion& r,
                              const LinearExpansion& z) {
  return dot_product(r, z, r, z);
}

ExactQuadratic orientation_polynomial(const double* start_r,
                                      const double* start_z,
                                      const double* end_r,
                                      const double* end_z,
                                      const int a,
                                      const int b,
                                      const int c,
                                      const int orientation_sign = 1) {
  const LinearExpansion ar = affine_coordinate(start_r[a], end_r[a]);
  const LinearExpansion az = affine_coordinate(start_z[a], end_z[a]);
  const LinearExpansion br = affine_coordinate(start_r[b], end_r[b]);
  const LinearExpansion bz = affine_coordinate(start_z[b], end_z[b]);
  const LinearExpansion cr = affine_coordinate(start_r[c], end_r[c]);
  const LinearExpansion cz = affine_coordinate(start_z[c], end_z[c]);
  return product_difference(br - ar, cz - az, bz - az, cr - ar,
                            orientation_sign);
}

ExactQuadratic segment_order_polynomial(const double* start_r,
                                        const double* start_z,
                                        const double* end_r,
                                        const double* end_z,
                                        const int point,
                                        const int segment_a,
                                        const int segment_b) {
  const LinearExpansion pr = affine_coordinate(start_r[point], end_r[point]);
  const LinearExpansion pz = affine_coordinate(start_z[point], end_z[point]);
  const LinearExpansion ar =
      affine_coordinate(start_r[segment_a], end_r[segment_a]);
  const LinearExpansion az =
      affine_coordinate(start_z[segment_a], end_z[segment_a]);
  const LinearExpansion br =
      affine_coordinate(start_r[segment_b], end_r[segment_b]);
  const LinearExpansion bz =
      affine_coordinate(start_z[segment_b], end_z[segment_b]);
  return dot_product(pr - ar, pz - az, pr - br, pz - bz);
}

ExactQuadratic flat_edge_squared_length(const double* start_r,
                                        const double* start_z,
                                        const double* end_r,
                                        const double* end_z,
                                        const int a,
                                        const int b) {
  const LinearExpansion ar = affine_coordinate(start_r[a], end_r[a]);
  const LinearExpansion az = affine_coordinate(start_z[a], end_z[a]);
  const LinearExpansion br = affine_coordinate(start_r[b], end_r[b]);
  const LinearExpansion bz = affine_coordinate(start_z[b], end_z[b]);
  return squared_length(br - ar, bz - az);
}

ExactQuadratic flat_edge_dot(const double* start_r,
                             const double* start_z,
                             const double* end_r,
                             const double* end_z,
                             const int previous,
                             const int node,
                             const int next) {
  const LinearExpansion ar =
      affine_coordinate(start_r[previous], end_r[previous]);
  const LinearExpansion az =
      affine_coordinate(start_z[previous], end_z[previous]);
  const LinearExpansion br = affine_coordinate(start_r[node], end_r[node]);
  const LinearExpansion bz = affine_coordinate(start_z[node], end_z[node]);
  const LinearExpansion cr = affine_coordinate(start_r[next], end_r[next]);
  const LinearExpansion cz = affine_coordinate(start_z[next], end_z[next]);
  return dot_product(br - ar, bz - az, cr - br, cz - bz);
}

Expansion evaluate(const ExactQuadratic& polynomial,
                   const Expansion& value) {
  return expansion_sum(
      polynomial.c0,
      expansion_product(
          value,
          expansion_sum(polynomial.c1,
                        expansion_product(value, polynomial.c2))));
}

int sign_at(const ExactQuadratic& polynomial, const Expansion& value) {
  return expansion_sign(evaluate(polynomial, value));
}

int degree(const ExactQuadratic& polynomial) {
  return expansion_sign(polynomial.c2) != 0
             ? 2
             : (expansion_sign(polynomial.c1) != 0
                    ? 1
                    : (expansion_sign(polynomial.c0) != 0 ? 0 : -1));
}

Expansion discriminant(const ExactQuadratic& polynomial) {
  return expansion_difference(
      expansion_product(polynomial.c1, polynomial.c1),
      expansion_scale(expansion_product(polynomial.c2, polynomial.c0), 4.0));
}

int sturm_variations(const ExactQuadratic& polynomial,
                     const Expansion& value) {
  const int polynomial_degree = degree(polynomial);
  if (polynomial_degree < 0) {
    return 0;
  }

  std::vector<int> signs;
  if (polynomial_degree == 2) {
    const Expansion determinant = discriminant(polynomial);
    if (expansion_sign(determinant) == 0) {
      signs.push_back(expansion_sign(expansion_sum(
          polynomial.c1,
          expansion_scale(expansion_product(polynomial.c2, value), 2.0))));
      signs.push_back(expansion_sign(polynomial.c2));
    } else {
      signs.push_back(sign_at(polynomial, value));
      signs.push_back(expansion_sign(expansion_sum(
          polynomial.c1,
          expansion_scale(expansion_product(polynomial.c2, value), 2.0))));
      signs.push_back(expansion_sign(determinant) *
                      expansion_sign(polynomial.c2));
    }
  } else if (polynomial_degree == 1) {
    signs.push_back(sign_at(polynomial, value));
    signs.push_back(expansion_sign(polynomial.c1));
  } else {
    signs.push_back(expansion_sign(polynomial.c0));
  }

  int variations = 0;
  int previous_sign = 0;
  for (const int sign : signs) {
    if (sign == 0) {
      continue;
    }
    if (previous_sign != 0 && sign != previous_sign) {
      ++variations;
    }
    previous_sign = sign;
  }
  return variations;
}

int open_root_count(const ExactQuadratic& polynomial,
                    const Expansion& lo,
                    const Expansion& hi) {
  if (!expansion_less(lo, hi) || degree(polynomial) <= 0) {
    return 0;
  }
  return sturm_variations(polynomial, lo) -
         sturm_variations(polynomial, hi);
}

struct AlgebraicRoot {
  ExactQuadratic defining;
  Expansion lo;
  Expansion hi;

  bool exact() const { return expansion_equal(lo, hi); }
};

void refine(AlgebraicRoot& root);

void isolate_interval(const ExactQuadratic& polynomial,
                      const Expansion& lo,
                      const Expansion& hi,
                      std::vector<AlgebraicRoot>& roots) {
  const int count = open_root_count(polynomial, lo, hi);
  if (count == 0) {
    return;
  }
  if (count == 1) {
    roots.push_back({polynomial, lo, hi});
    return;
  }
  const Expansion mid = midpoint(lo, hi);
  isolate_interval(polynomial, lo, mid, roots);
  isolate_interval(polynomial, mid, hi, roots);
}

std::vector<AlgebraicRoot> roots_open_unit(
    const ExactQuadratic& polynomial) {
  ExactQuadratic interior = polynomial;
  const Expansion zero;
  const Expansion one = exact_value(1.0);
  while (degree(interior) > 0 && sign_at(interior, zero) == 0) {
    interior = {interior.c1, interior.c2, {}};
  }
  while (degree(interior) > 0 && sign_at(interior, one) == 0) {
    interior = {expansion_negate(interior.c0), interior.c2, {}};
  }
  if (degree(interior) <= 0) {
    return {};
  }
  std::vector<AlgebraicRoot> roots;
  isolate_interval(interior, zero, one, roots);
  for (AlgebraicRoot& root : roots) {
    while (!root.exact() &&
           (expansion_equal(root.lo, zero) ||
            expansion_equal(root.hi, one))) {
      refine(root);
    }
  }
  return roots;
}

void refine(AlgebraicRoot& root) {
  if (root.exact()) {
    return;
  }
  const Expansion mid = midpoint(root.lo, root.hi);
  if (sign_at(root.defining, mid) == 0) {
    root.lo = mid;
    root.hi = mid;
    return;
  }
  if (open_root_count(root.defining, root.lo, mid) == 1) {
    root.hi = mid;
  } else {
    root.lo = mid;
  }
}

bool intervals_overlap(const AlgebraicRoot& lhs,
                       const AlgebraicRoot& rhs) {
  return !(expansion_less(lhs.hi, rhs.lo) ||
           expansion_less(rhs.hi, lhs.lo));
}

ExactQuadratic linear_common_factor(const ExactQuadratic& quadratic,
                                    const ExactQuadratic& linear) {
  const Expansion value_numerator = expansion_sum(
      expansion_product(
          quadratic.c0, expansion_product(linear.c1, linear.c1)),
      expansion_sum(
          expansion_negate(expansion_product(
              quadratic.c1,
              expansion_product(linear.c0, linear.c1))),
          expansion_product(
              quadratic.c2, expansion_product(linear.c0, linear.c0))));
  return expansion_sign(value_numerator) == 0 ? linear : ExactQuadratic{};
}

ExactQuadratic common_factor(const ExactQuadratic& lhs,
                             const ExactQuadratic& rhs) {
  const int lhs_degree = degree(lhs);
  const int rhs_degree = degree(rhs);
  if (lhs_degree <= 0 || rhs_degree <= 0) {
    return {};
  }
  if (lhs_degree == 1 && rhs_degree == 1) {
    return expansion_sign(expansion_difference(
               expansion_product(lhs.c1, rhs.c0),
               expansion_product(rhs.c1, lhs.c0))) == 0
               ? lhs
               : ExactQuadratic{};
  }
  if (lhs_degree == 2 && rhs_degree == 1) {
    return linear_common_factor(lhs, rhs);
  }
  if (lhs_degree == 1 && rhs_degree == 2) {
    return linear_common_factor(rhs, lhs);
  }

  const Expansion linear_coefficient = expansion_difference(
      expansion_product(rhs.c2, lhs.c1),
      expansion_product(lhs.c2, rhs.c1));
  const Expansion constant_coefficient = expansion_difference(
      expansion_product(rhs.c2, lhs.c0),
      expansion_product(lhs.c2, rhs.c0));
  if (expansion_sign(linear_coefficient) == 0) {
    return expansion_sign(constant_coefficient) == 0 ? lhs
                                                     : ExactQuadratic{};
  }
  return linear_common_factor(
      lhs, {constant_coefficient, linear_coefficient, {}});
}

bool common_root_in_interval(const ExactQuadratic& lhs,
                             const ExactQuadratic& rhs,
                             const Expansion& lo,
                             const Expansion& hi) {
  const ExactQuadratic common = common_factor(lhs, rhs);
  if (degree(common) <= 0) {
    return false;
  }
  if (expansion_equal(lo, hi)) {
    return sign_at(common, lo) == 0;
  }
  return open_root_count(common, lo, hi) > 0;
}

bool same_root(AlgebraicRoot& lhs, AlgebraicRoot& rhs) {
  if (lhs.exact() && rhs.exact()) {
    return expansion_equal(lhs.lo, rhs.lo);
  }
  for (;;) {
    if (!intervals_overlap(lhs, rhs)) {
      return false;
    }
    const Expansion& lo = expansion_less(lhs.lo, rhs.lo) ? rhs.lo : lhs.lo;
    const Expansion& hi = expansion_less(lhs.hi, rhs.hi) ? lhs.hi : rhs.hi;
    if (common_root_in_interval(lhs.defining, rhs.defining, lo, hi)) {
      return true;
    }
    refine(lhs);
    refine(rhs);
  }
}

int sign_at_root(AlgebraicRoot root,
                 const ExactQuadratic& polynomial) {
  if (degree(polynomial) < 0) {
    return 0;
  }
  if (root.exact()) {
    return sign_at(polynomial, root.lo);
  }
  for (;;) {
    if (common_root_in_interval(root.defining, polynomial,
                                root.lo, root.hi)) {
      return 0;
    }
    const int lo_sign = sign_at(polynomial, root.lo);
    const int hi_sign = sign_at(polynomial, root.hi);
    if (lo_sign != 0 && lo_sign == hi_sign &&
        open_root_count(polynomial, root.lo, root.hi) == 0) {
      return lo_sign;
    }
    refine(root);
  }
}

double root_to_double(AlgebraicRoot root) {
  for (int iteration = 0; iteration < 192 && !root.exact(); ++iteration) {
    const long double lo = expansion_to_long_double(root.lo);
    const long double hi = expansion_to_long_double(root.hi);
    if (static_cast<double>(lo) == static_cast<double>(hi)) {
      return std::clamp(static_cast<double>(lo), 0.0, 1.0);
    }
    refine(root);
  }
  const long double value =
      expansion_to_long_double(midpoint(root.lo, root.hi));
  return std::clamp(static_cast<double>(value), 0.0, 1.0);
}

std::vector<AlgebraicRoot> distinct_sorted_roots(
    const std::vector<ExactQuadratic>& polynomials) {
  std::vector<AlgebraicRoot> roots;
  for (const ExactQuadratic& polynomial : polynomials) {
    auto candidates = roots_open_unit(polynomial);
    for (AlgebraicRoot& candidate : candidates) {
      bool duplicate = false;
      for (AlgebraicRoot& root : roots) {
        if (same_root(root, candidate)) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        roots.push_back(std::move(candidate));
      }
    }
  }
  for (std::size_t i = 0; i < roots.size(); ++i) {
    for (std::size_t j = i + 1U; j < roots.size(); ++j) {
      while (intervals_overlap(roots[i], roots[j])) {
        refine(roots[i]);
        refine(roots[j]);
      }
    }
  }
  std::sort(roots.begin(), roots.end(),
            [](const AlgebraicRoot& lhs, const AlgebraicRoot& rhs) {
              return expansion_less(lhs.hi, rhs.lo);
            });
  return roots;
}

bool exact_quadratic_nonnegative(const ExactQuadratic& polynomial) {
  const Expansion endpoint_one = expansion_sum(
      expansion_sum(polynomial.c0, polynomial.c1), polynomial.c2);
  if (expansion_sign(polynomial.c0) < 0 ||
      expansion_sign(endpoint_one) < 0) {
    return false;
  }
  if (expansion_sign(polynomial.c2) > 0 &&
      expansion_less(expansion_scale(polynomial.c2, -2.0), polynomial.c1) &&
      expansion_sign(polynomial.c1) < 0) {
    return expansion_sign(expansion_difference(
               expansion_scale(
                   expansion_product(polynomial.c2, polynomial.c0), 4.0),
               expansion_product(polynomial.c1, polynomial.c1))) >= 0;
  }
  return true;
}

bool exact_quadratic_positive(const ExactQuadratic& polynomial) {
  const Expansion endpoint_one = expansion_sum(
      expansion_sum(polynomial.c0, polynomial.c1), polynomial.c2);
  if (expansion_sign(polynomial.c0) <= 0 ||
      expansion_sign(endpoint_one) <= 0) {
    return false;
  }
  if (expansion_sign(polynomial.c2) > 0 &&
      expansion_less(expansion_scale(polynomial.c2, -2.0), polynomial.c1) &&
      expansion_sign(polynomial.c1) < 0) {
    return expansion_sign(expansion_difference(
               expansion_scale(
                   expansion_product(polynomial.c2, polynomial.c0), 4.0),
               expansion_product(polynomial.c1, polynomial.c1))) > 0;
  }
  return true;
}

bool flat_ordering_at(const ExactQuadratic& previous_edge_length,
                      const ExactQuadratic& next_edge_length,
                      const ExactQuadratic& edge_dot,
                      const Expansion& tau) {
  return sign_at(previous_edge_length, tau) > 0 &&
         sign_at(next_edge_length, tau) > 0 && sign_at(edge_dot, tau) > 0;
}

bool flat_ordering_at_root(const ExactQuadratic& previous_edge_length,
                           const ExactQuadratic& next_edge_length,
                           const ExactQuadratic& edge_dot,
                           AlgebraicRoot root) {
  return sign_at_root(root, previous_edge_length) > 0 &&
         sign_at_root(root, next_edge_length) > 0 &&
         sign_at_root(root, edge_dot) > 0;
}

double quadratic_minimum_approximate(const double* start_r,
                                     const double* start_z,
                                     const double* end_r,
                                     const double* end_z,
                                     const int previous,
                                     const int node,
                                     const int next,
                                     const int orientation_sign) {
  const double a0r = start_r[node] - start_r[previous];
  const double a0z = start_z[node] - start_z[previous];
  const double b0r = start_r[next] - start_r[previous];
  const double b0z = start_z[next] - start_z[previous];
  const double dar = (end_r[node] - end_r[previous]) - a0r;
  const double daz = (end_z[node] - end_z[previous]) - a0z;
  const double dbr = (end_r[next] - end_r[previous]) - b0r;
  const double dbz = (end_z[next] - end_z[previous]) - b0z;
  const double c = orientation_sign * (a0r * b0z - a0z * b0r);
  const double b = orientation_sign *
                   (dar * b0z + a0r * dbz - daz * b0r - a0z * dbr);
  const double a = orientation_sign * (dar * dbz - daz * dbr);
  double minimum = std::min(c, a + b + c);
  if (a > 0.0) {
    const double tau = -b / (2.0 * a);
    if (tau > 0.0 && tau < 1.0) {
      minimum = std::min(minimum, (a * tau + b) * tau + c);
    }
  }
  return minimum;
}

double finite_failure_value(const double approximate, const bool negative) {
  if (std::isfinite(approximate) && (!negative || approximate < 0.0)) {
    return approximate;
  }
  return negative ? -std::numeric_limits<double>::denorm_min() : 0.0;
}

double first_negative_boundary(const ExactQuadratic& polynomial) {
  if (expansion_sign(polynomial.c0) < 0) {
    return 0.0;
  }
  if (expansion_sign(polynomial.c0) == 0 &&
      (expansion_sign(polynomial.c1) < 0 ||
       (expansion_sign(polynomial.c1) == 0 &&
        expansion_sign(polynomial.c2) < 0))) {
    return 0.0;
  }
  auto roots = roots_open_unit(polynomial);
  std::sort(roots.begin(), roots.end(),
            [](const AlgebraicRoot& lhs, const AlgebraicRoot& rhs) {
              return expansion_less(lhs.hi, rhs.lo);
            });
  Expansion left;
  AlgebraicRoot previous_root;
  bool have_previous_root = false;
  for (AlgebraicRoot& root : roots) {
    const Expansion sample = midpoint(left, root.lo);
    if (expansion_less(left, sample) && sign_at(polynomial, sample) < 0) {
      return have_previous_root ? root_to_double(previous_root) : 0.0;
    }
    left = root.hi;
    previous_root = root;
    have_previous_root = true;
  }
  const Expansion one = exact_value(1.0);
  const Expansion tail_sample = midpoint(left, one);
  if (have_previous_root && expansion_less(left, tail_sample) &&
      sign_at(polynomial, tail_sample) < 0) {
    return root_to_double(previous_root);
  }
  return expansion_sign(expansion_sum(
             expansion_sum(polynomial.c0, polynomial.c1), polynomial.c2)) < 0
             ? (have_previous_root ? root_to_double(previous_root) : 0.0)
             : 0.0;
}

double first_strict_failure_boundary(const ExactQuadratic& polynomial) {
  if (expansion_sign(polynomial.c0) <= 0) {
    return 0.0;
  }
  auto roots = roots_open_unit(polynomial);
  if (!roots.empty()) {
    std::sort(roots.begin(), roots.end(),
              [](const AlgebraicRoot& lhs, const AlgebraicRoot& rhs) {
                return expansion_less(lhs.hi, rhs.lo);
              });
    return root_to_double(roots.front());
  }
  return expansion_sign(expansion_sum(
             expansion_sum(polynomial.c0, polynomial.c1), polynomial.c2)) <= 0
             ? 1.0
             : 0.0;
}

bool segments_intersect_at(const std::vector<ExactQuadratic>& predicates,
                           const Expansion& tau) {
  const int o1 = sign_at(predicates[0], tau);
  const int o2 = sign_at(predicates[1], tau);
  const int o3 = sign_at(predicates[2], tau);
  const int o4 = sign_at(predicates[3], tau);
  if ((o1 == 0 && sign_at(predicates[4], tau) <= 0) ||
      (o2 == 0 && sign_at(predicates[5], tau) <= 0) ||
      (o3 == 0 && sign_at(predicates[6], tau) <= 0) ||
      (o4 == 0 && sign_at(predicates[7], tau) <= 0)) {
    return true;
  }
  return o1 * o2 < 0 && o3 * o4 < 0;
}

bool segments_intersect_at_root(const std::vector<ExactQuadratic>& predicates,
                                AlgebraicRoot root) {
  const int o1 = sign_at_root(root, predicates[0]);
  const int o2 = sign_at_root(root, predicates[1]);
  const int o3 = sign_at_root(root, predicates[2]);
  const int o4 = sign_at_root(root, predicates[3]);
  if ((o1 == 0 && sign_at_root(root, predicates[4]) <= 0) ||
      (o2 == 0 && sign_at_root(root, predicates[5]) <= 0) ||
      (o3 == 0 && sign_at_root(root, predicates[6]) <= 0) ||
      (o4 == 0 && sign_at_root(root, predicates[7]) <= 0)) {
    return true;
  }
  return o1 * o2 < 0 && o3 * o4 < 0;
}

std::pair<bool, double> nonincident_edges_intersect_on_path(
    const double* start_r,
    const double* start_z,
    const double* end_r,
    const double* end_z,
    const int a,
    const int b,
    const int c,
    const int d) {
  std::vector<ExactQuadratic> predicates;
  predicates.reserve(8U);
  predicates.push_back(
      orientation_polynomial(start_r, start_z, end_r, end_z, a, b, c));
  predicates.push_back(
      orientation_polynomial(start_r, start_z, end_r, end_z, a, b, d));
  predicates.push_back(
      orientation_polynomial(start_r, start_z, end_r, end_z, c, d, a));
  predicates.push_back(
      orientation_polynomial(start_r, start_z, end_r, end_z, c, d, b));
  predicates.push_back(segment_order_polynomial(
      start_r, start_z, end_r, end_z, c, a, b));
  predicates.push_back(segment_order_polynomial(
      start_r, start_z, end_r, end_z, d, a, b));
  predicates.push_back(segment_order_polynomial(
      start_r, start_z, end_r, end_z, a, c, d));
  predicates.push_back(segment_order_polynomial(
      start_r, start_z, end_r, end_z, b, c, d));

  const Expansion zero;
  const Expansion one = exact_value(1.0);
  if (segments_intersect_at(predicates, zero)) {
    return {true, 0.0};
  }
  auto roots = distinct_sorted_roots(predicates);
  Expansion interval_start;
  AlgebraicRoot previous_root;
  bool have_previous_root = false;
  for (AlgebraicRoot& root : roots) {
    const Expansion sample = midpoint(interval_start, root.lo);
    if (expansion_less(interval_start, sample) &&
        segments_intersect_at(predicates, sample)) {
      return {true,
              have_previous_root ? root_to_double(previous_root) : 0.0};
    }
    if (segments_intersect_at_root(predicates, root)) {
      return {true, root_to_double(root)};
    }
    interval_start = root.hi;
    previous_root = root;
    have_previous_root = true;
  }
  const Expansion tail_sample = midpoint(interval_start, one);
  if (expansion_less(interval_start, tail_sample) &&
      segments_intersect_at(predicates, tail_sample)) {
    return {true,
            have_previous_root ? root_to_double(previous_root) : 0.0};
  }
  if (segments_intersect_at(predicates, one)) {
    return {true, 1.0};
  }
  return {false, 1.0};
}

}  // namespace

ReferenceCornerClass classify_reference_corner(
    const double previous_r,
    const double previous_z,
    const double node_r,
    const double node_z,
    const double next_r,
    const double next_z,
    const int orientation_sign) {
  if (orientation_sign != -1 && orientation_sign != 1) {
    return ReferenceCornerClass::InvalidOrReflexReference;
  }
  for (const double value : {previous_r, previous_z, node_r, node_z, next_r,
                             next_z}) {
    if (!std::isfinite(value)) {
      return ReferenceCornerClass::InvalidOrReflexReference;
    }
  }
  const int turn_sign = orientation_sign * reale::orient2d_sign(
      previous_r, previous_z, node_r, node_z, next_r, next_z);
  if (turn_sign > 0) {
    return ReferenceCornerClass::RegularReference;
  }
  if (turn_sign < 0 ||
      (previous_r == node_r && previous_z == node_z) ||
      (node_r == next_r && node_z == next_z)) {
    return ReferenceCornerClass::InvalidOrReflexReference;
  }
  const LinearExpansion previous_to_node_r =
      {expansion_difference(exact_value(node_r), exact_value(previous_r)), {}};
  const LinearExpansion previous_to_node_z =
      {expansion_difference(exact_value(node_z), exact_value(previous_z)), {}};
  const LinearExpansion node_to_next_r =
      {expansion_difference(exact_value(next_r), exact_value(node_r)), {}};
  const LinearExpansion node_to_next_z =
      {expansion_difference(exact_value(next_z), exact_value(node_z)), {}};
  const ExactQuadratic ordering = dot_product(
      previous_to_node_r, previous_to_node_z, node_to_next_r, node_to_next_z);
  return expansion_sign(ordering.c0) > 0
             ? ReferenceCornerClass::FlatReference
             : ReferenceCornerClass::InvalidOrReflexReference;
}

ReferenceFlatCellPathResult evaluate_reference_flat_cell_path(
    const double* reference_r,
    const double* reference_z,
    const double* start_r,
    const double* start_z,
    const double* end_r,
    const double* end_z,
    const int nverts,
    const int orientation_sign) {
  ReferenceFlatCellPathResult result;
  if (reference_r == nullptr || reference_z == nullptr || start_r == nullptr ||
      start_z == nullptr || end_r == nullptr || end_z == nullptr ||
      nverts < 3 || orientation_sign == 0) {
    return result;
  }

  std::vector<int> flat_corners;
  for (int corner = 0; corner < nverts; ++corner) {
    const int previous = (corner + nverts - 1) % nverts;
    const int next = (corner + 1) % nverts;
    if (classify_reference_corner(
            reference_r[previous], reference_z[previous],
            reference_r[corner], reference_z[corner], reference_r[next],
            reference_z[next], orientation_sign) ==
        ReferenceCornerClass::FlatReference) {
      flat_corners.push_back(corner);
    }
  }
  result.has_flat_corner = !flat_corners.empty();
  if (flat_corners.empty()) {
    return result;
  }
  result.first_flat_corner = flat_corners.front();
  result.min_oriented_turn = std::numeric_limits<double>::infinity();

  for (int corner = 0; corner < nverts; ++corner) {
    if (!std::isfinite(start_r[corner]) ||
        !std::isfinite(start_z[corner]) ||
        !std::isfinite(end_r[corner]) ||
        !std::isfinite(end_z[corner])) {
      result.admissible = false;
      result.first_failure_tau = 0.0;
      result.min_oriented_turn =
          -std::numeric_limits<double>::denorm_min();
      return result;
    }
  }

  for (const int corner : flat_corners) {
    const int previous = (corner + nverts - 1) % nverts;
    const int next = (corner + 1) % nverts;
    const ExactQuadratic turn = orientation_polynomial(
        start_r, start_z, end_r, end_z, previous, corner, next,
        orientation_sign);
    const ExactQuadratic previous_edge_length = flat_edge_squared_length(
        start_r, start_z, end_r, end_z, previous, corner);
    const ExactQuadratic next_edge_length = flat_edge_squared_length(
        start_r, start_z, end_r, end_z, corner, next);
    const ExactQuadratic edge_dot = flat_edge_dot(
        start_r, start_z, end_r, end_z, previous, corner, next);
    const double approximate_minimum = quadratic_minimum_approximate(
        start_r, start_z, end_r, end_z, previous, corner, next,
        orientation_sign);
    result.min_oriented_turn =
        std::min(result.min_oriented_turn, approximate_minimum);

    if (!exact_quadratic_nonnegative(turn)) {
      const double tau = first_negative_boundary(turn);
      if (result.admissible || tau < result.first_failure_tau) {
        result.admissible = false;
        result.first_failure_tau = tau;
        result.first_flat_corner = corner;
        result.min_oriented_turn =
            finite_failure_value(approximate_minimum, true);
      }
      continue;
    }

    if (degree(turn) < 0) {
      if (!exact_quadratic_positive(previous_edge_length) ||
          !exact_quadratic_positive(next_edge_length) ||
          !exact_quadratic_positive(edge_dot)) {
        double tau = 1.0;
        if (!exact_quadratic_positive(previous_edge_length)) {
          tau = std::min(
              tau, first_strict_failure_boundary(previous_edge_length));
        }
        if (!exact_quadratic_positive(next_edge_length)) {
          tau = std::min(
              tau, first_strict_failure_boundary(next_edge_length));
        }
        if (!exact_quadratic_positive(edge_dot)) {
          tau = std::min(tau, first_strict_failure_boundary(edge_dot));
        }
        if (result.admissible || tau < result.first_failure_tau) {
          result.admissible = false;
          result.first_failure_tau = tau;
          result.first_flat_corner = corner;
          result.min_oriented_turn = 0.0;
        }
      }
      continue;
    }

    const Expansion zero;
    const Expansion one = exact_value(1.0);
    if (expansion_sign(turn.c0) == 0 &&
        !flat_ordering_at(previous_edge_length, next_edge_length, edge_dot,
                          zero)) {
      if (result.admissible || 0.0 < result.first_failure_tau) {
        result.admissible = false;
        result.first_failure_tau = 0.0;
        result.first_flat_corner = corner;
        result.min_oriented_turn = 0.0;
      }
    }
    for (AlgebraicRoot root : roots_open_unit(turn)) {
      if (!flat_ordering_at_root(previous_edge_length, next_edge_length,
                                 edge_dot, root)) {
        const double tau = root_to_double(root);
        if (result.admissible || tau < result.first_failure_tau) {
          result.admissible = false;
          result.first_failure_tau = tau;
          result.first_flat_corner = corner;
          result.min_oriented_turn = 0.0;
        }
      }
    }
    if (expansion_sign(expansion_sum(
            expansion_sum(turn.c0, turn.c1), turn.c2)) == 0 &&
        !flat_ordering_at(previous_edge_length, next_edge_length, edge_dot,
                          one)) {
      if (result.admissible || 1.0 < result.first_failure_tau) {
        result.admissible = false;
        result.first_failure_tau = 1.0;
        result.first_flat_corner = corner;
        result.min_oriented_turn = 0.0;
      }
    }
  }

  for (int edge_a = 0; edge_a < nverts; ++edge_a) {
    const int edge_a_next = (edge_a + 1) % nverts;
    for (int edge_b = edge_a + 1; edge_b < nverts; ++edge_b) {
      const int edge_b_next = (edge_b + 1) % nverts;
      if (edge_a == edge_b_next || edge_a_next == edge_b) {
        continue;
      }
      const auto [intersects, tau] = nonincident_edges_intersect_on_path(
          start_r, start_z, end_r, end_z, edge_a, edge_a_next, edge_b,
          edge_b_next);
      if (intersects && (result.admissible || tau < result.first_failure_tau)) {
        result.admissible = false;
        result.embedding_failure = true;
        result.first_failure_tau = tau;
        result.first_flat_corner = flat_corners.front();
        result.min_oriented_turn = 0.0;
      }
    }
  }

  if (!std::isfinite(result.min_oriented_turn)) {
    result.min_oriented_turn = 0.0;
  }
  return result;
}

}  // namespace tenryu::mesh
