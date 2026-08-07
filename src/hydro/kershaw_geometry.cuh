#pragma once

#include <algorithm>
#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::hydro::kershaw {

constexpr double kJFloor = 1.0e-30;
constexpr double kJTol = 1.0e-20;

struct NodeGeometry {
  double Ar = 0.0;
  double Az = 0.0;
  double Br = 0.0;
  double Bz = 0.0;
  double J = kJFloor;
  double A2 = 0.0;
  double B2 = 0.0;
  double AdotB = 0.0;
  int degenerate = 1;
};

TENRYU_HOST_DEVICE inline int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

TENRYU_HOST_DEVICE inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

TENRYU_HOST_DEVICE inline int clampi(const int v, const int lo, const int hi) {
  return (v < lo) ? lo : ((v > hi) ? hi : v);
}

TENRYU_HOST_DEVICE inline double polygon_area4(const double r0,
                                               const double z0,
                                               const double r1,
                                               const double z1,
                                               const double r2,
                                               const double z2,
                                               const double r3,
                                               const double z3) {
  const double cross = r0 * z1 - z0 * r1 + r1 * z2 - z1 * r2 + r2 * z3 - z2 * r3 +
                       r3 * z0 - z3 * r0;
  return 0.5 * fabs(cross);
}

TENRYU_HOST_DEVICE inline NodeGeometry compute_node_geometry(const double* x_r,
                                                             const double* x_z,
                                                             const int i,
                                                             const int j,
                                                             const int nr,
                                                             const int nz) {
  const int ic = clampi(i, 0, nr);
  const int jc = clampi(j, 0, nz);

  const int ip = clampi(ic + 1, 0, nr);
  const int im = clampi(ic - 1, 0, nr);
  const int jp = clampi(jc + 1, 0, nz);
  const int jm = clampi(jc - 1, 0, nz);

  const int n_im_j = node_index(im, jc, nz);
  const int n_ip_j = node_index(ip, jc, nz);
  const int n_i_jm = node_index(ic, jm, nz);
  const int n_i_jp = node_index(ic, jp, nz);

  NodeGeometry g;
  g.Ar = 0.5 * (x_r[n_i_jp] - x_r[n_i_jm]);
  g.Az = 0.5 * (x_z[n_i_jp] - x_z[n_i_jm]);
  g.Br = 0.5 * (x_r[n_ip_j] - x_r[n_im_j]);
  g.Bz = 0.5 * (x_z[n_ip_j] - x_z[n_im_j]);

  // Mirror extrapolation for boundary nodes: the one-sided difference
  // is half the centered difference that a ghost node would produce.
  // Doubling restores correct metric at boundaries (Neumann/reflect BC).
  if (ip == ic || im == ic) {
    g.Br *= 2.0;
    g.Bz *= 2.0;
  }
  if (jp == jc || jm == jc) {
    g.Ar *= 2.0;
    g.Az *= 2.0;
  }

  const double J_raw = g.Ar * g.Bz - g.Az * g.Br;
  // Magnitude is used because mesh orientation can make the signed Jacobian negative
  // for the standard (i=r, j=z) index ordering.
  const double J_abs = fabs(J_raw);
  g.J = fmax(J_abs, kJFloor);
  g.A2 = g.Ar * g.Ar + g.Az * g.Az;
  g.B2 = g.Br * g.Br + g.Bz * g.Bz;
  g.AdotB = g.Ar * g.Br + g.Az * g.Bz;

  if (J_abs >= kJTol) {
    g.degenerate = 0;
  } else {
    // Zero B-operator in near-degenerate cells (NUMERICS App.A §A.3).
    g.Ar = 0.0;
    g.Az = 0.0;
    g.Br = 0.0;
    g.Bz = 0.0;
    g.A2 = 0.0;
    g.B2 = 0.0;
    g.AdotB = 0.0;
    g.degenerate = 1;
  }

  return g;
}

TENRYU_HOST_DEVICE inline void compute_b_operators(const NodeGeometry& g,
                                                   const double phi_nw,
                                                   const double phi_ne,
                                                   const double phi_se,
                                                   const double phi_sw,
                                                   double& b1,
                                                   double& b2,
                                                   double& b3,
                                                   double& b4) {
  if (g.degenerate != 0) {
    b1 = 0.0;
    b2 = 0.0;
    b3 = 0.0;
    b4 = 0.0;
    return;
  }

  const double d_nw_se = phi_nw - phi_se;
  const double d_ne_sw = phi_ne - phi_sw;

  b1 = g.Bz * d_nw_se - g.Az * d_ne_sw;
  b2 = g.Az * d_nw_se - g.Bz * d_ne_sw;
  b3 = g.Ar * d_nw_se - g.Br * d_ne_sw;
  b4 = g.Br * d_nw_se - g.Ar * d_ne_sw;
}

TENRYU_HOST_DEVICE inline void node_gradient_from_cells(const double* phi,
                                                        const double* x_r,
                                                        const double* x_z,
                                                        const int ni,
                                                        const int nj,
                                                        const int nr,
                                                        const int nz,
                                                        double& grad_r,
                                                        double& grad_z) {
  const int i_nw = clampi(ni - 1, 0, nr - 1);
  const int j_nw = clampi(nj, 0, nz - 1);
  const int i_ne = clampi(ni, 0, nr - 1);
  const int j_ne = clampi(nj, 0, nz - 1);
  const int i_se = clampi(ni, 0, nr - 1);
  const int j_se = clampi(nj - 1, 0, nz - 1);
  const int i_sw = clampi(ni - 1, 0, nr - 1);
  const int j_sw = clampi(nj - 1, 0, nz - 1);

  const double phi_nw = phi[cell_index(i_nw, j_nw, nz)];
  const double phi_ne = phi[cell_index(i_ne, j_ne, nz)];
  const double phi_se = phi[cell_index(i_se, j_se, nz)];
  const double phi_sw = phi[cell_index(i_sw, j_sw, nz)];

  const NodeGeometry g = compute_node_geometry(x_r, x_z, ni, nj, nr, nz);
  if (g.degenerate != 0) {
    grad_r = 0.0;
    grad_z = 0.0;
    return;
  }

  double b1 = 0.0;
  double b2 = 0.0;
  double b3 = 0.0;
  double b4 = 0.0;
  compute_b_operators(g, phi_nw, phi_ne, phi_se, phi_sw, b1, b2, b3, b4);

  const double inv_2J = 0.5 / g.J;
  grad_r = b1 * inv_2J;
  grad_z = b4 * inv_2J;
}

TENRYU_HOST_DEVICE inline double r_face_i(const double* x_r,
                                           const int i,
                                           const int j,
                                           const int nz) {
  const int n0 = node_index(i, j, nz);
  const int n1 = node_index(i, j + 1, nz);
  return 0.5 * (x_r[n0] + x_r[n1]);
}

TENRYU_HOST_DEVICE inline double r_face_j(const double* x_r,
                                           const int i,
                                           const int j,
                                           const int nz) {
  const int n0 = node_index(i, j, nz);
  const int n1 = node_index(i + 1, j, nz);
  return 0.5 * (x_r[n0] + x_r[n1]);
}

TENRYU_HOST_DEVICE inline double cell_area_from_nodes(const double* x_r,
                                                       const double* x_z,
                                                       const int i,
                                                       const int j,
                                                       const int nz) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  return polygon_area4(x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11], x_z[n11],
                       x_r[n01], x_z[n01]);
}

}  // namespace tenryu::hydro::kershaw
