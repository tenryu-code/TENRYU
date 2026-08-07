#pragma once

#include <cmath>

#include <cuda_runtime.h>

#include "hydro/ale_remap.cuh"

namespace tenryu::hydro::plic {

using tenryu::hydro::ale::detail::kPi;

constexpr int kPlicMaxPolygonVertices = 8;

struct Polygon {
  double r[kPlicMaxPolygonVertices];
  double z[kPlicMaxPolygonVertices];
  int n;
};

__host__ __device__ inline void append_vertex(Polygon& p,
                                             const double r,
                                             const double z) {
  if (p.n < kPlicMaxPolygonVertices) {
    p.r[p.n] = r;
    p.z[p.n] = z;
    ++p.n;
  }
}

__host__ __device__ inline double polygon_revolved_volume(const Polygon& p) {
  if (p.n < 3) {
    return 0.0;
  }

  double v = 0.0;
  for (int k = 0; k < p.n; ++k) {
    const int kp1 = (k + 1) % p.n;
    v += (p.r[k] + p.r[kp1]) *
         (p.r[k] * p.z[kp1] - p.r[kp1] * p.z[k]);
  }
  return (kPi / 3.0) * v;
}

__host__ __device__ inline void clip_polygon_by_halfplane(
    const Polygon& in,
    const double n_r,
    const double n_z,
    const double alpha,
    Polygon& out) {
  out.n = 0;
  if (in.n < 3) {
    return;
  }

  for (int k = 0; k < in.n; ++k) {
    const int kp1 = (k + 1) % in.n;
    const double s_k = n_r * in.r[k] + n_z * in.z[k] - alpha;
    const double s_kp1 = n_r * in.r[kp1] + n_z * in.z[kp1] - alpha;
    const bool in_k = s_k <= 0.0;
    const bool in_kp1 = s_kp1 <= 0.0;

    if (in_k) {
      append_vertex(out, in.r[k], in.z[k]);
    }
    if (in_k != in_kp1) {
      const double denom = s_k - s_kp1;
      if (fabs(denom) > 0.0) {
        const double t = s_k / denom;
        append_vertex(out,
                      in.r[k] + t * (in.r[kp1] - in.r[k]),
                      in.z[k] + t * (in.z[kp1] - in.z[k]));
      }
    }
  }
}

__host__ __device__ inline double v_below_in_cell(
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    const double r3,
    const double z3,
    const double n_r,
    const double n_z,
    const double alpha) {
  Polygon quad;
  quad.n = 4;
  quad.r[0] = r0;
  quad.z[0] = z0;
  quad.r[1] = r1;
  quad.z[1] = z1;
  quad.r[2] = r2;
  quad.z[2] = z2;
  quad.r[3] = r3;
  quad.z[3] = z3;

  Polygon clipped;
  clip_polygon_by_halfplane(quad, n_r, n_z, alpha, clipped);
  return polygon_revolved_volume(clipped);
}

struct AlphaFinderResult {
  double alpha;
  double volume_fraction_residual;
  int iterations;
  bool converged;
};

__host__ __device__ inline AlphaFinderResult find_alpha_for_volume_fraction(
    const double r0,
    const double z0,
    const double r1,
    const double z1,
    const double r2,
    const double z2,
    const double r3,
    const double z3,
    const double n_r,
    const double n_z,
    const double v_cell,
    const double target_volfrac,
    const int max_iter,
    const double tol_rel) {
  AlphaFinderResult res{0.0, 1.0, 0, false};
  if (v_cell <= 0.0 || target_volfrac <= 0.0 || target_volfrac >= 1.0) {
    res.alpha = 0.0;
    res.volume_fraction_residual = 0.0;
    res.converged = (target_volfrac <= 0.0 || target_volfrac >= 1.0);
    return res;
  }

  double a_lo = fmin(fmin(n_r * r0 + n_z * z0, n_r * r1 + n_z * z1),
                     fmin(n_r * r2 + n_z * z2, n_r * r3 + n_z * z3));
  double a_hi = fmax(fmax(n_r * r0 + n_z * z0, n_r * r1 + n_z * z1),
                     fmax(n_r * r2 + n_z * z2, n_r * r3 + n_z * z3));
  if (!(a_hi > a_lo)) {
    res.alpha = a_lo;
    return res;
  }

  const double target_v = target_volfrac * v_cell;
  for (int it = 0; it < max_iter; ++it) {
    const double a_mid = 0.5 * (a_lo + a_hi);
    const double v_mid =
        v_below_in_cell(r0, z0, r1, z1, r2, z2, r3, z3, n_r, n_z, a_mid);
    if (v_mid < target_v) {
      a_lo = a_mid;
    } else {
      a_hi = a_mid;
    }
    res.iterations = it + 1;
    const double rel = fabs(v_mid - target_v) / fmax(target_v, 1.0e-30);
    if (rel < tol_rel) {
      res.alpha = a_mid;
      res.volume_fraction_residual = rel;
      res.converged = true;
      return res;
    }
  }

  res.alpha = 0.5 * (a_lo + a_hi);
  const double v =
      v_below_in_cell(r0, z0, r1, z1, r2, z2, r3, z3, n_r, n_z, res.alpha);
  res.volume_fraction_residual = fabs(v - target_v) / fmax(target_v, 1.0e-30);
  res.converged = (res.volume_fraction_residual < tol_rel);
  return res;
}

__host__ __device__ inline bool detect_negative_corner_j(
    const double corner_j[4],
    int& which_corner_negative) {
  int worst = -1;
  double worst_val = 0.0;
  for (int k = 0; k < 4; ++k) {
    if (corner_j[k] < worst_val) {
      worst = k;
      worst_val = corner_j[k];
    }
  }
  which_corner_negative = worst;
  return worst >= 0;
}

}  // namespace tenryu::hydro::plic
