#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/error.hpp"

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

// For quad corner k, extrapolate node positions as x(dt) = x + dt*v and set
//   e1 = x[k+1] - x[k],  e2 = x[k-1] - x[k],
//   d1 = v[k+1] - v[k],  d2 = v[k-1] - v[k].
// With cross(p,q) = p.r*q.z - p.z*q.r (the sign convention used by
// corner_jacobian_from_quad), the corner Jacobian is
//   J(dt) = cross(e1 + dt*d1, e2 + dt*d2)
//         = J0 + b*dt + a*dt^2,
// where J0 = cross(e1,e2),
//       b  = cross(e1,d2) + cross(d1,e2),
//       a  = cross(d1,d2).
// The limiter requires J(dt) >= (1-max_shrink)*J0 for J0 > 0, so it finds
// the smallest positive root of
//   g(dt) = a*dt^2 + b*dt + max_shrink*J0.
// Quadratic roots use q = -(b + sign(b)*sqrt(discriminant))/2 and the pair
// q/a, c/q to avoid cancellation. Corners with non-positive J0 or non-finite
// inputs impose no limit.

namespace detail {

inline double corner_j_predict_cross2(const double ar,
                                      const double az,
                                      const double br,
                                      const double bz) {
  return ar * bz - az * br;
}

inline double corner_j_predict_smallest_positive_root(const double a,
                                                      const double b,
                                                      const double c) {
  const double inf = std::numeric_limits<double>::infinity();
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) {
    return inf;
  }
  if (std::abs(a) <= 1.0e-300) {
    if (b < 0.0) {
      const double root = -c / b;
      return root > 0.0 && std::isfinite(root) ? root : inf;
    }
    return inf;
  }

  const double discriminant = b * b - 4.0 * a * c;
  if (!(discriminant >= 0.0) || !std::isfinite(discriminant)) {
    return inf;
  }
  const double sqrt_discriminant = std::sqrt(discriminant);
  const double q =
      -0.5 * (b + std::copysign(sqrt_discriminant, b));
  double root = inf;
  if (q != 0.0 && std::isfinite(q)) {
    const double root1 = q / a;
    const double root2 = c / q;
    if (root1 > 0.0 && std::isfinite(root1)) {
      root = root1;
    }
    if (root2 > 0.0 && std::isfinite(root2)) {
      root = std::min(root, root2);
    }
  }
  return root;
}

}  // namespace detail

inline double corner_j_predict_dt_max(const double xr[4],
                                      const double xz[4],
                                      const double vr[4],
                                      const double vz[4],
                                      const double max_shrink) {
  TENRYU_ASSERT(max_shrink > 0.0 && max_shrink < 1.0,
                "corner-J prediction max_shrink must be in (0, 1)");

  double dt_max = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const int km = (k + 3) & 3;
    if (!std::isfinite(xr[k]) || !std::isfinite(xz[k]) ||
        !std::isfinite(vr[k]) || !std::isfinite(vz[k]) ||
        !std::isfinite(xr[kp]) || !std::isfinite(xz[kp]) ||
        !std::isfinite(vr[kp]) || !std::isfinite(vz[kp]) ||
        !std::isfinite(xr[km]) || !std::isfinite(xz[km]) ||
        !std::isfinite(vr[km]) || !std::isfinite(vz[km])) {
      continue;
    }

    const double e1r = xr[kp] - xr[k];
    const double e1z = xz[kp] - xz[k];
    const double e2r = xr[km] - xr[k];
    const double e2z = xz[km] - xz[k];
    const double d1r = vr[kp] - vr[k];
    const double d1z = vz[kp] - vz[k];
    const double d2r = vr[km] - vr[k];
    const double d2z = vz[km] - vz[k];
    const double j0 = detail::corner_j_predict_cross2(e1r, e1z, e2r, e2z);
    const double b = detail::corner_j_predict_cross2(e1r, e1z, d2r, d2z) +
                     detail::corner_j_predict_cross2(d1r, d1z, e2r, e2z);
    const double a = detail::corner_j_predict_cross2(d1r, d1z, d2r, d2z);
    if (!(j0 > 0.0) || !std::isfinite(j0) || !std::isfinite(a) ||
        !std::isfinite(b)) {
      continue;
    }

    const double c = max_shrink * j0;
    const double root =
        detail::corner_j_predict_smallest_positive_root(a, b, c);
    dt_max = std::min(dt_max, root);
  }
  return dt_max;
}

double compute_corner_j_predict_cfl_dt(const core::State& state,
                                       const core::Config& cfg);

}  // namespace tenryu::hydro
