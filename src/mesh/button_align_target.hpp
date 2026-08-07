#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numbers>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::assembly {

// Equal-volume radial rescale (exact ring-wise RZ volume preservation).
inline double button_align_radius(const double a) {
  TENRYU_ASSERT(std::isfinite(a) && a >= 0.0,
                "button alignment ring radius must be non-negative and finite");
  return std::cbrt(1.5) * a;
}

// Half-plane square-ring perimeter fraction measured from the +z axis.
inline double button_align_perimeter_fraction(const double r,
                                              const double z) {
  TENRYU_ASSERT(std::isfinite(r) && r >= 0.0,
                "button alignment radial coordinate must be non-negative "
                "and finite");
  TENRYU_ASSERT(std::isfinite(z),
                "button alignment axial coordinate must be finite");

  const double abs_z = std::abs(z);
  const double a = std::max(r, abs_z);
  if (a == 0.0) {
    return 0.0;
  }

  const double perimeter = 4.0 * a;
  if (abs_z >= r) {
    if (z > 0.0) {
      return r / perimeter;
    }
    return (4.0 * a - r) / perimeter;
  }
  return (2.0 * a - z) / perimeter;
}

// Spherical equal-volume target for a node in the button core.
inline void button_align_core_target(const double r, const double z,
                                     double* const r_t,
                                     double* const z_t) {
  TENRYU_ASSERT(r_t != nullptr && z_t != nullptr,
                "button alignment target outputs must not be null");

  const double a = std::max(r, std::abs(z));
  const double radius = button_align_radius(a);
  const double t = button_align_perimeter_fraction(r, z);
  if (t == 0.0) {
    *r_t = 0.0;
    *z_t = radius;
    return;
  }
  if (t == 1.0) {
    *r_t = 0.0;
    *z_t = -radius;
    return;
  }

  const double theta = std::numbers::pi_v<double> * t;
  *r_t = radius * std::sin(theta);
  *z_t = radius * std::cos(theta);
}

// Circular-annulus bridge target with a geometric radial ladder and a linear
// angle ladder. Graded seam landing requires N_b >= 2 in v1; the graded-button
// planner never emits N_b == 1.
inline void button_align_bridge_target(
    const double r_c, const double s_b, const int k, const int n_b,
    const double h_seam, const double t_core, const double theta_seam,
    double* const r_t, double* const z_t) {
  TENRYU_ASSERT(r_t != nullptr && z_t != nullptr,
                "button alignment target outputs must not be null");
  TENRYU_ASSERT(n_b >= 2,
                "graded seam landing requires at least 2 bridge layers");
  TENRYU_ASSERT(k >= 0 && k <= n_b,
                "button alignment bridge layer index must be in range");
  TENRYU_ASSERT(std::isfinite(t_core) && t_core >= 0.0 && t_core <= 1.0,
                "button alignment core perimeter fraction must be in range");

  constexpr double pi = std::numbers::pi_v<double>;
  TENRYU_ASSERT(std::isfinite(theta_seam) && theta_seam >= 0.0 &&
                    theta_seam <= pi,
                "button alignment seam angle must be in range");
  TENRYU_ASSERT(std::isfinite(s_b),
                "button alignment seam radius must be finite");

  const double s_c = button_align_radius(r_c);
  TENRYU_ASSERT(s_b > s_c,
                "button alignment seam radius must exceed the core radius");
  TENRYU_ASSERT(std::isfinite(h_seam) && h_seam > 0.0,
                "button alignment seam spacing must be positive and finite");

  const double span = s_b - s_c;
  const double scaled_h_seam = h_seam * static_cast<double>(n_b);
  TENRYU_ASSERT(scaled_h_seam > span / 1.0e3 &&
                    scaled_h_seam < span * 1.0e3,
                "button alignment seam spacing is incompatible with the "
                "bridge span");

  const auto graded_span = [h_seam, n_b](const double ratio) {
    double spacing = h_seam;
    double total = 0.0;
    for (int layer = n_b - 1; layer >= 0; --layer) {
      total += spacing;
      spacing /= ratio;
    }
    return total;
  };

  double ratio_lo = 1.0e-3;
  double ratio_hi = 1.0e3;
  for (int iteration = 0; iteration < 64; ++iteration) {
    const double ratio_mid = 0.5 * (ratio_lo + ratio_hi);
    if (graded_span(ratio_mid) > span) {
      ratio_lo = ratio_mid;
    } else {
      ratio_hi = ratio_mid;
    }
  }
  const double ratio = 0.5 * (ratio_lo + ratio_hi);
  TENRYU_ASSERT(ratio >= 1.0 / 1.6 && ratio <= 1.6,
                "button alignment bridge grading is too steep; raise N_b");

  const double first_spacing =
      h_seam / std::pow(ratio, static_cast<double>(n_b - 1));
  std::vector<double> rho(static_cast<std::size_t>(n_b) + 1U, 0.0);
  rho.front() = s_c;
  double spacing = first_spacing;
  for (int layer = 0; layer < n_b - 1; ++layer) {
    rho[static_cast<std::size_t>(layer + 1)] =
        rho[static_cast<std::size_t>(layer)] + spacing;
    spacing *= ratio;
  }
  rho.back() = s_b;
  for (int layer = 1; layer <= n_b; ++layer) {
    TENRYU_ASSERT(rho[static_cast<std::size_t>(layer)] >
                      rho[static_cast<std::size_t>(layer - 1)],
                  "button alignment bridge radial ladder must be strictly "
                  "increasing");
  }
  TENRYU_ASSERT(std::memcmp(&rho.front(), &s_c, sizeof(double)) == 0 &&
                    std::memcmp(&rho.back(), &s_b, sizeof(double)) == 0,
                "button alignment bridge radial ladder endpoints must be "
                "bitwise exact");

  const double mu = static_cast<double>(k) / static_cast<double>(n_b);
  const double radius = rho[static_cast<std::size_t>(k)];
  const double theta_core = pi * t_core;
  const double theta = (1.0 - mu) * theta_core + mu * theta_seam;

  if (theta_core == 0.0 && theta_seam == 0.0) {
    *r_t = 0.0;
    *z_t = radius;
    return;
  }
  if (theta_core == pi && theta_seam == pi) {
    *r_t = 0.0;
    *z_t = -radius;
    return;
  }

  *r_t = radius * std::sin(theta);
  *z_t = radius * std::cos(theta);
}

}  // namespace tenryu::mesh::assembly
