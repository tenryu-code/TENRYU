#pragma once

#include <cmath>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::compatible {

inline constexpr int kCsw98MaxSideVecs = 8;

// --- csw_edge_csw98 median-mesh geometry (CSW98 Eq. 16; I1-B Stage-G W1) ---
// For a cell side traversed p_a -> p_b (the caller's canonical edge
// orientation) with cell centroid c (node mean), the median-mesh vector S is
// the 90-degree rotation of (c - m), m = (p_a + p_b)/2, sign-fixed so that
// S . (p_b - p_a) >= 0. With this pairing the discrete identity
//   sum_sides S_k . dv_k = dV/dt,   dv_k = v(p_b) - v(p_a),
// holds exactly for arbitrary polygons and node velocities in the 2D
// Cartesian (planar-area) measure: reversing a side flips (S, dv) jointly,
// leaving S.dv invariant. In RZ no edge-difference form can reproduce the
// full revolution-volume rate (the r-translation mode has
// sum_k dV/dx_k != 0 but is invisible to dv); the RZ variant below uses the
// pi*(r_c + r_m) lateral-revolution weighting of the median segment and
// captures the deformational part only, exact in the Cartesian limit
// r >> dr. Measured precisely in tests/hydro/test_csw98_eq16_identity.cu.

constexpr double kCsw98Pi = 3.1415926535897932384626433832795028841971693993751;

__host__ __device__ inline void csw98_cell_centroid(const double* r,
                                                    const double* z,
                                                    const int nverts,
                                                    double* c_r,
                                                    double* c_z) {
  double sum_r = 0.0;
  double sum_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    sum_r += r[k];
    sum_z += z[k];
  }
  const double inv = 1.0 / static_cast<double>(nverts);
  *c_r = sum_r * inv;
  *c_z = sum_z * inv;
}

__host__ __device__ inline bool csw98_median_svec_cart(const double r_a,
                                                       const double z_a,
                                                       const double r_b,
                                                       const double z_b,
                                                       const double c_r,
                                                       const double c_z,
                                                       double* s_r,
                                                       double* s_z) {
  const double m_r = 0.5 * (r_a + r_b);
  const double m_z = 0.5 * (z_a + z_b);
  const double d_r = c_r - m_r;
  const double d_z = c_z - m_z;
  const double t_r = r_b - r_a;
  const double t_z = z_b - z_a;
  // R(-90deg)(d) = (d_z, -d_r); orientation sign = S0.t = t_r*d_z - t_z*d_r.
  const double orient = t_r * d_z - t_z * d_r;
  const double scale = (fabs(t_r) + fabs(t_z)) * (fabs(d_r) + fabs(d_z));
  if (!(scale > 0.0) || !(fabs(orient) > 1.0e-30 * scale)) {
    *s_r = 0.0;
    *s_z = 0.0;
    return false;
  }
  if (orient > 0.0) {
    *s_r = d_z;
    *s_z = -d_r;
  } else {
    *s_r = -d_z;
    *s_z = d_r;
  }
  return true;
}

__host__ __device__ inline bool csw98_median_svec_rz(const double r_a,
                                                     const double z_a,
                                                     const double r_b,
                                                     const double z_b,
                                                     const double c_r,
                                                     const double c_z,
                                                     double* s_r,
                                                     double* s_z) {
  if (!csw98_median_svec_cart(r_a, z_a, r_b, z_b, c_r, c_z, s_r, s_z)) {
    return false;
  }
  const double m_r = 0.5 * (r_a + r_b);
  const double coeff = kCsw98Pi * (c_r + m_r);
  *s_r *= coeff;
  *s_z *= coeff;
  return true;
}

// --- C2 side vectors (decision 2026-07-04,
// docs/design/i1b_csw98_rz_eq16_decision.md): telescoped from the exact
// revolution-volume corner gradients a_k = dV/dx_k. With
// b_k = a_k - mean(a), the cyclic system S_{k-1} - S_k = b_k with gauge
// sum_k S_k = 0 has the unique solution
//   S_0 = (sum_{j=1..nv-1} (nv - j) b_j) / nv,  S_k = S_{k-1} - b_k,
// and satisfies sum_k S_k . (v_{k+1} - v_k) = sum_k b_k . v_k -- the exact
// deformational volume rate (full dV/dt minus the r-translation part,
// which no edge-difference form can carry) -- to roundoff for any nverts.
// In the planar measure the same construction reproduces the Cartesian
// median-mesh vectors exactly.

// Winding-orientation factor for the corner-gradient formulas below. The
// shoelace-family expressions flip sign with the polygon winding; the pressure
// path (rz_area_weighted.cuh) applies the same factor so that corner vectors
// carry the physical outward orientation for either input winding. Without it,
// clockwise-wound cells (every cell of the spherical-polar logical meshes)
// invert the AV compression switch: the viscosity then fires on expansion and
// is silent at shocks.
__host__ __device__ inline double csw98_winding_orientation(const double* r,
                                                            const double* z,
                                                            const int nverts) {
  double area2 = 0.0;
  double scale = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    area2 += r[k] * z[kp] - r[kp] * z[k];
    scale += r[k] * r[k] + z[k] * z[k];
  }
  const double threshold = 64.0 * 2.220446049250313e-16 * scale;
  if (!(fabs(area2) > threshold)) {
  #ifdef __CUDA_ARCH__
    __trap();  // degenerate/zero-area cell reached AV orientation: fail loud (kernel abort -> CUDA_CHECK)
  #else
    ::tenryu::core::tenryu_abort(
        "fabs(area2) > threshold",
        "csw98_winding_orientation: zero/near-zero signed area (degenerate cell) "
        "must not silently orient +1",
        __FILE__, __LINE__);
  #endif
  }
  return (area2 >= 0.0) ? 1.0 : -1.0;
}

__host__ __device__ inline void csw98_rz_corner_gradients(
    const double* r,
    const double* z,
    const int nverts,
    double* a_r,
    double* a_z) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double orientation = csw98_winding_orientation(r, z, nverts);
  for (int k = 0; k < nverts; ++k) {
    const int km = (k + nverts - 1) % nverts;
    const int kp = (k + 1) % nverts;
    a_r[k] = orientation * (pi_over_three *
        (z[kp] * (r[k] + r[kp]) + (r[k] * z[kp] - r[kp] * z[k]) -
         z[km] * (r[km] + r[k]) + (r[km] * z[k] - r[k] * z[km])));
    a_z[k] = orientation * (pi_over_three *
        (r[km] * (r[km] + r[k]) - r[kp] * (r[k] + r[kp])));
  }
}

__host__ __device__ inline void csw98_c2_side_svecs(
    const double* r,
    const double* z,
    const int nverts,
    double* s_r,
    double* s_z) {
  if (!(nverts >= 3 && nverts <= kCsw98MaxSideVecs)) {
  #ifdef __CUDA_ARCH__
    __trap();  // unsupported vertex count reached C2 geometry: fail loud (kernel abort -> CUDA_CHECK)
  #else
    ::tenryu::core::tenryu_abort(
        "nverts >= 3 && nverts <= kCsw98MaxSideVecs",
        "csw98_c2_side_svecs: vertex count outside supported range "
        "must not overrun local side-vector storage",
        __FILE__, __LINE__);
  #endif
  }
  double a_r[kCsw98MaxSideVecs];
  double a_z[kCsw98MaxSideVecs];
  csw98_rz_corner_gradients(r, z, nverts, a_r, a_z);
  double mean_r = 0.0;
  double mean_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    mean_r += a_r[k];
    mean_z += a_z[k];
  }
  const double inv = 1.0 / static_cast<double>(nverts);
  mean_r *= inv;
  mean_z *= inv;
  double s0_r = 0.0;
  double s0_z = 0.0;
  for (int j = 1; j < nverts; ++j) {
    const double w = static_cast<double>(nverts - j);
    s0_r += w * (a_r[j] - mean_r);
    s0_z += w * (a_z[j] - mean_z);
  }
  s_r[0] = s0_r * inv;
  s_z[0] = s0_z * inv;
  for (int k = 1; k < nverts; ++k) {
    s_r[k] = s_r[k - 1] - (a_r[k] - mean_r);
    s_z[k] = s_z[k - 1] - (a_z[k] - mean_z);
  }
}

// Side vector for the side traversed corner ca -> corner cb (corners in
// cyclic cell order). Reversed traversal returns the negated cyclic
// vector, keeping S.dv invariant. Returns false if (ca, cb) is not a
// cell side.
__host__ __device__ inline bool csw98_c2_svec_for_side(
    const double* r,
    const double* z,
    const int nverts,
    const int ca,
    const int cb,
    double* out_r,
    double* out_z) {
  if (!(nverts >= 3 && nverts <= kCsw98MaxSideVecs)) {
  #ifdef __CUDA_ARCH__
    __trap();  // unsupported vertex count reached C2 geometry: fail loud (kernel abort -> CUDA_CHECK)
  #else
    ::tenryu::core::tenryu_abort(
        "nverts >= 3 && nverts <= kCsw98MaxSideVecs",
        "csw98_c2_svec_for_side: vertex count outside supported range "
        "must not overrun local side-vector storage",
        __FILE__, __LINE__);
  #endif
  }
  double s_r[kCsw98MaxSideVecs];
  double s_z[kCsw98MaxSideVecs];
  csw98_c2_side_svecs(r, z, nverts, s_r, s_z);
  if (cb == ((ca + 1) % nverts)) {
    *out_r = s_r[ca];
    *out_z = s_z[ca];
    return true;
  }
  if (ca == ((cb + 1) % nverts)) {
    *out_r = -s_r[cb];
    *out_z = -s_z[cb];
    return true;
  }
  *out_r = 0.0;
  *out_z = 0.0;
  return false;
}

struct CswEdgeAvDiagnostics {
  int compressive_edge_count = 0;
  int clipped_negative_work_count = 0;
};

void launch_compute_csw_edge_av_2d(core::State& state,
                                   const core::Config& cfg,
                                   const core::CellField1D& cell_cs,
                                   const double* v_r,
                                   const double* v_z,
                                   const std::int8_t* hydro_active);

double compute_csw_edge_av_cfl_dt(const core::State& state,
                                  const core::Config& cfg);

}  // namespace tenryu::hydro::compatible
