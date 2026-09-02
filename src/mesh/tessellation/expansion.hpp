#pragma once

// Clean-room duplicate of the Shewchuk expansion kernels in poly_geom.cu.
// DIVERGENCE IS FORBIDDEN without a tessellator-core contract revision; the
// poly_geom.cu copies remain the frozen originals.

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tenryu::mesh::tess {

using Expansion = std::vector<double>;

inline void two_sum(const double a,
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

inline void two_diff(const double a,
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

inline void two_product(const double a,
                        const double b,
                        double& product,
                        double& error) {
  product = a * b;
  error = std::fma(a, b, -product);
}

inline void append_nonzero(Expansion& expansion, const double value) {
  if (value != 0.0) {
    expansion.push_back(value);
  }
}

inline Expansion expansion_sum(const Expansion& lhs, const Expansion& rhs) {
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

inline Expansion expansion_negate(const Expansion& expansion) {
  Expansion result = expansion;
  for (double& component : result) {
    component = -component;
  }
  return result;
}

inline Expansion expansion_difference(const Expansion& lhs,
                                       const Expansion& rhs) {
  return expansion_sum(lhs, expansion_negate(rhs));
}

inline Expansion exact_difference(const double a, const double b) {
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

inline Expansion expansion_scale(const Expansion& expansion,
                                  const double scale) {
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

inline Expansion expansion_product(const Expansion& lhs,
                                    const Expansion& rhs) {
  Expansion result;
  for (const double component : rhs) {
    result = expansion_sum(result, expansion_scale(lhs, component));
  }
  return result;
}

inline Expansion expansion_cross(const Expansion& ax,
                                 const Expansion& ay,
                                 const Expansion& bx,
                                 const Expansion& by) {
  return expansion_difference(expansion_product(ax, by),
                              expansion_product(ay, bx));
}

inline Expansion expansion_lift(const Expansion& x, const Expansion& y) {
  return expansion_sum(expansion_product(x, x), expansion_product(y, y));
}

inline int expansion_sign(const Expansion& expansion) {
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

// An n-term sequential sum has error <= (n-1) eps sum|c_i| to first order; add/mul use standard first-order bounds with result rounding.
// Per-rule (1+4 eps) slack covers bound rounding, at most three internal roundings, and dropped second-order terms, and approx_of_expansion carries an explicit (1 + n eps) factor for the magnitude accumulation itself.
// This assumes no overflow/underflow; O(1) cm coordinates and their handful-term products stay in range.
struct ApproxBound {
  double value;
  double bound;
};

inline constexpr double kApproxSlack =
    1.0 + 4.0 * std::numeric_limits<double>::epsilon();
inline constexpr double kApproxEps =
    std::numeric_limits<double>::epsilon();

inline ApproxBound approx_of_expansion(const Expansion& expansion) {
  double value = 0.0;
  double magnitude = 0.0;
  for (const double component : expansion) {
    value += component;
    magnitude += std::abs(component);
  }
  const double count = static_cast<double>(
      expansion.size() > 0 ? expansion.size() - 1 : 0);
  const double bound =
      count * kApproxEps * magnitude * (1.0 + count * kApproxEps) * kApproxSlack * kApproxSlack;
  return ApproxBound{value, bound};
}

inline ApproxBound approx_sub(const ApproxBound& a,
                              const ApproxBound& b) {
  const double value = a.value - b.value;
  const double bound =
      (a.bound + b.bound + kApproxEps * std::abs(value)) *
      kApproxSlack;
  return ApproxBound{value, bound};
}

inline ApproxBound approx_add(const ApproxBound& a,
                              const ApproxBound& b) {
  const double value = a.value + b.value;
  const double bound =
      (a.bound + b.bound + kApproxEps * std::abs(value)) *
      kApproxSlack;
  return ApproxBound{value, bound};
}

inline ApproxBound approx_mul(const ApproxBound& a,
                              const ApproxBound& b) {
  const double value = a.value * b.value;
  const double bound =
      (std::abs(a.value) * b.bound + std::abs(b.value) * a.bound +
       a.bound * b.bound + kApproxEps * std::abs(value)) *
      kApproxSlack;
  return ApproxBound{value, bound};
}

struct ApproxSign {
  bool certain;
  int sign;
};

inline ApproxSign approx_sign(const ApproxBound& a) {
  if (std::isfinite(a.value) && std::isfinite(a.bound) &&
      std::abs(a.value) > a.bound) {
    return {true, a.value > 0.0 ? 1 : -1};
  }
  return {false, 0};
}

inline int orient2d_exact_sign(const double ax,
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

inline int incircle_exact_sign(const double ax,
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

}  // namespace tenryu::mesh::tess
