#pragma once

#include <cuda_runtime.h>

#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::forces {

/**
 * Writes the planar corner surface vector for vertex i of a CCW polygon.
 *
 * The two contributions are the outward normals of the directed half-edges
 * m_prev -> v_i and v_i -> m_next. Midpoints are formed first so that the
 * contribution of a shared half-edge is the exact FP negation when traversed
 * in the opposite direction. The expression is contraction-free: apart from
 * the exact power-of-two midpoint scales, it contains sums of subtractions
 * only.
 */
__host__ __device__ inline void polygon_corner_surface(
    const double* r, const double* z, int n, int i, double& s_r,
    double& s_z) {
  const int prev = (i == 0) ? n - 1 : i - 1;
  const int next = (i + 1 == n) ? 0 : i + 1;
  const double m_prev_r = 0.5 * (r[prev] + r[i]);
  const double m_prev_z = 0.5 * (z[prev] + z[i]);
  const double m_next_r = 0.5 * (r[i] + r[next]);
  const double m_next_z = 0.5 * (z[i] + z[next]);

  s_r = (z[i] - m_prev_z) + (m_next_z - z[i]);
  s_z = (m_prev_r - r[i]) + (r[i] - m_next_r);
}

/**
 * Writes the planar-measure pressure force F_i = -P*S_i.
 */
__host__ __device__ inline void polygon_corner_force(
    const double* r, const double* z, int n, int i, double P, double& f_r,
    double& f_z) {
  double s_r = 0.0;
  double s_z = 0.0;
  polygon_corner_surface(r, z, n, i, s_r, s_z);
  f_r = fma(-P, s_r, 0.0);
  f_z = fma(-P, s_z, 0.0);
}

/**
 * Writes the fixed-order sum of all polygon corner surface vectors.
 */
__host__ __device__ inline void polygon_surface_sum(
    const double* r, const double* z, int n, double& sum_r,
    double& sum_z) {
  sum_r = 0.0;
  sum_z = 0.0;
  for (int i = 0; i < n; ++i) {
    double s_r = 0.0;
    double s_z = 0.0;
    polygon_corner_surface(r, z, n, i, s_r, s_z);
    sum_r += s_r;
    sum_z += s_z;
  }
}

/**
 * Returns the analytic shoelace area derivative under nodal velocities:
 * 0.5*sum_i [ur_i*(z_next-z_prev) + uz_i*(r_prev-r_next)].
 */
__host__ __device__ inline double polygon_area_rate(
    const double* r, const double* z, const double* ur, const double* uz,
    int n) {
  double twice_rate = 0.0;
  for (int i = 0; i < n; ++i) {
    const int prev = (i == 0) ? n - 1 : i - 1;
    const int next = (i + 1 == n) ? 0 : i + 1;
    const double term =
        fma(ur[i], z[next] - z[prev],
            fma(uz[i], r[prev] - r[next], 0.0));
    twice_rate += term;
  }
  return 0.5 * twice_rate;
}

}  // namespace tenryu::mesh::forces
