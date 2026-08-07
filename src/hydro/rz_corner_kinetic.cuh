#pragma once

#include <cmath>

#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro::rz {

__device__ inline double corner_nodal_kinetic_for_cell(
    const int c,
    const int nr,
    const int nz,
    const double* __restrict__ corner_mass,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node) {
  (void)nr;
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const int nodes[4] = {n00, n10, n11, n01};

  double k = 0.0;
  for (int q = 0; q < 4; ++q) {
    const int n = nodes[q];
    const double vr = v_r_node[n];
    const double vz = v_z_node[n];
    k += 0.5 * corner_mass[c * 4 + q] * (vr * vr + vz * vz);
  }
  return k;
}

}  // namespace tenryu::hydro::rz
