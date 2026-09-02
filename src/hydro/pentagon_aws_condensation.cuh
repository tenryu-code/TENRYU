#pragma once

#include "hydro/pentagon_geometry.cuh"
#include "hydro/rz_area_weighted.cuh"

namespace tenryu::hydro {

__host__ __device__ inline void pentagon_aws_condensed_planar_s(
    const PentagonPoint x[5],
    double Sr[5],
    double Sz[5]) {
  const PentagonPoint x_star = pentagon_center(x);
  double S_star_r[5];
  double S_star_z[5];
  double S_L_r[5];
  double S_L_z[5];
  double S_R_r[5];
  double S_R_z[5];
  for (int k = 0; k < 5; ++k) {
    const int kp1 = (k == 4) ? 0 : k + 1;
    const double r[3] = {x_star.r, x[k].r, x[kp1].r};
    const double z[3] = {x_star.z, x[k].z, x[kp1].z};
    double triangle_Sr[4];
    double triangle_Sz[4];
    rz::aw_planar_triangle_corner_vectors(
        r, z, triangle_Sr, triangle_Sz);
    S_star_r[k] = triangle_Sr[0];
    S_star_z[k] = triangle_Sz[0];
    S_L_r[k] = triangle_Sr[1];
    S_L_z[k] = triangle_Sz[1];
    S_R_r[k] = triangle_Sr[2];
    S_R_z[k] = triangle_Sz[2];
  }
  double S_star_total_r = 0.0;
  double S_star_total_z = 0.0;
  for (int j = 0; j < 5; ++j) {
    S_star_total_r = S_star_total_r + S_star_r[j];
    S_star_total_z = S_star_total_z + S_star_z[j];
  }
  for (int k = 0; k < 5; ++k) {
    const int km1 = (k == 0) ? 4 : k - 1;
    const double S_tilde_r = S_R_r[km1] + S_L_r[k];
    const double S_tilde_z = S_R_z[km1] + S_L_z[k];
    Sr[k] = S_tilde_r + 0.2 * S_star_total_r;
    Sz[k] = S_tilde_z + 0.2 * S_star_total_z;
  }
  Sr[2] =
      -((Sr[0] + Sr[1]) +
        (Sr[3] + Sr[4]));
  Sz[2] =
      -((Sz[0] + Sz[1]) +
        (Sz[3] + Sz[4]));
}

}  // namespace tenryu::hydro
