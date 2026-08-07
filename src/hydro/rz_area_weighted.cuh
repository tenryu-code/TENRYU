#pragma once

#include <cmath>
#include <cstdint>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

namespace tenryu::hydro::rz {

// Planar polygon corner vectors use the physical outward orientation for
// either input winding. This matches the orientation-corrected RZ Svec sign:
// counter-clockwise quads use the raw area gradient and clockwise quads use
// its negative.
__host__ __device__ inline void aw_planar_corner_vectors(
    const double r[4],
    const double z[4],
    double Sr[4],
    double Sz[4]) {
  double area2 = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    area2 += r[k] * z[kp1] - r[kp1] * z[k];
  }
  const double orientation = (area2 >= 0.0) ? 1.0 : -1.0;
  for (int k = 0; k < 4; ++k) {
    const int km1 = (k + 3) & 3;
    const int kp1 = (k + 1) & 3;
    Sr[k] = 0.5 * orientation * (z[kp1] - z[km1]);
    Sz[k] = 0.5 * orientation * (r[km1] - r[kp1]);
  }
}

// Fig. 5 correspondence: edge-center triangles are ordered A_41, A_12,
// A_23, A_34, then distributed to corners 1..4 with the cyclic 12-weight
// quadrature.
__host__ __device__ inline void aw_planar_subzonal_areas(
    const double r[4],
    const double z[4],
    double A[4]) {
  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  double edge_triangle[4];
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    const double cross =
        (r[kp1] - r[k]) * (zc - z[k]) -
        (z[kp1] - z[k]) * (rc - r[k]);
    edge_triangle[k] = 0.5 * fabs(cross);
  }

  const double A41 = edge_triangle[3];
  const double A12 = edge_triangle[0];
  const double A23 = edge_triangle[1];
  const double A34 = edge_triangle[2];
  A[0] = (5.0 * A41 + 5.0 * A12 + A23 + A34) / 12.0;
  A[1] = (A41 + 5.0 * A12 + 5.0 * A23 + A34) / 12.0;
  A[2] = (A41 + A12 + 5.0 * A23 + 5.0 * A34) / 12.0;
  A[3] = (5.0 * A41 + A12 + A23 + 5.0 * A34) / 12.0;
}

__host__ __device__ inline double aw_csr_node_planar_mass(
    const double* rho,
    const double* x_r,
    const double* x_z,
    const int* reverse_csr_node_offsets,
    const int* reverse_csr_node_cells,
    const int* reverse_csr_node_corners,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::int8_t* hydro_active,
    const int node) {
  const int off = reverse_csr_node_offsets[node];
  const int end = reverse_csr_node_offsets[node + 1];
  double sum = 0.0;
  for (int k = off; k < end; ++k) {
    const int c = reverse_csr_node_cells[k];
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      continue;
    }
    const int corner = reverse_csr_node_corners[k];
    const int cell_off = cell_node_csr_offsets[c];
    const int n00 = cell_node_csr_indices[cell_off + 0];
    const int n10 = cell_node_csr_indices[cell_off + 1];
    const int n11 = cell_node_csr_indices[cell_off + 2];
    const int n01 = cell_node_csr_indices[cell_off + 3];
    const double r[4] = {x_r[n00], x_r[n10], x_r[n11], x_r[n01]};
    const double z[4] = {x_z[n00], x_z[n10], x_z[n11], x_z[n01]};
    double area[4];
    aw_planar_subzonal_areas(r, z, area);
    sum += rho[c] * area[corner];
  }
  return sum;
}

}  // namespace tenryu::hydro::rz
