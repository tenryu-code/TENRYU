#pragma once

#include <cuda_runtime.h>

#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::forces {

/**
 * Returns V1 = integral(r)dA from the polygon boundary form.
 *
 * This and the other objects in this file are R-WEIGHTED work/energy surface
 * objects. The planar force objects in polygon_forces.cuh are the FIX-2 planar
 * measure and are not interchangeable with these objects.
 */
__host__ __device__ inline double polygon_v1_boundary(
    const double* r, const double* z, int n) {
  double acc = 0.0;
  for (int i = 0; i < n; ++i) {
    const int next = (i + 1 == n) ? 0 : i + 1;
    const double ri = r[i];
    const double rn = r[next];
    const double q_i = fma(ri, ri, fma(ri, rn, rn * rn));
    acc = fma(z[next] - z[i], q_i, acc);
  }
  return acc / 6.0;
}

/**
 * Writes the R-WEIGHTED work/energy surface vector d(V1)/d(r_k,z_k).
 *
 * These FIX-2 surface objects use the r-weighted measure. They are not
 * interchangeable with the planar force objects in polygon_forces.cuh.
 */
__host__ __device__ inline void polygon_rz_work_surface(
    const double* r, const double* z, int n, int k, double& t_r,
    double& t_z) {
  const int prev = (k == 0) ? n - 1 : k - 1;
  const int next = (k + 1 == n) ? 0 : k + 1;
  const double rk = r[k];
  const double rp = r[prev];
  const double rn = r[next];
  const double a = fma(2.0, rk, rn);
  const double b = fma(2.0, rk, rp);
  t_r =
      fma(z[next] - z[k], a, (z[k] - z[prev]) * b) / 6.0;
  t_z =
      fma(rp, rp + rk, -(fma(rn, rk + rn, 0.0))) / 6.0;
}

/**
 * Returns the fixed-order R-WEIGHTED V1 rate sum_k u_k dot d(V1)/dx_k.
 */
__host__ __device__ inline double polygon_v1_rate(
    const double* r, const double* z, const double* ur, const double* uz,
    int n) {
  double acc = 0.0;
  for (int k = 0; k < n; ++k) {
    double t_r = 0.0;
    double t_z = 0.0;
    polygon_rz_work_surface(r, z, n, k, t_r, t_z);
    acc = fma(ur[k], t_r, fma(uz[k], t_z, acc));
  }
  return acc;
}

}  // namespace tenryu::mesh::forces
