#pragma once

#include <cuda_runtime.h>

#include "mesh/polygon_forces.cuh"
#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::patch {

inline constexpr int kPatchNodes = 10;
inline constexpr int kPatchCells = 6;
inline constexpr int kPatchMaxVerts = 5;

struct PolygonPatch {
  double node_r[kPatchNodes];
  double node_z[kPatchNodes];
  int cell_n[kPatchCells];
  int cell_nodes[kPatchCells][kPatchMaxVerts];
};

__host__ __device__ inline PolygonPatch make_pentagon_ring_patch() {
  PolygonPatch p{};
  const double inner_r[5] = {1.0, 2.0, 2.4, 1.7, 0.8};
  const double inner_z[5] = {0.0, 0.0, 0.8, 1.5, 0.9};
  for (int i = 0; i < 5; ++i) {
    p.node_r[i] = inner_r[i];
    p.node_z[i] = inner_z[i];
    p.cell_nodes[0][i] = i;
  }
  p.cell_n[0] = 5;

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  moments::poly_vertex_mean(
      p.node_r, p.node_z, 5, centroid_r, centroid_z);
  for (int i = 0; i < 5; ++i) {
    const int next = (i + 1 == 5) ? 0 : i + 1;
    p.node_r[5 + i] =
        fma(1.8, p.node_r[i] - centroid_r, centroid_r);
    p.node_z[5 + i] =
        fma(1.8, p.node_z[i] - centroid_z, centroid_z);

    p.cell_n[1 + i] = 4;
    p.cell_nodes[1 + i][0] = next;
    p.cell_nodes[1 + i][1] = i;
    p.cell_nodes[1 + i][2] = 5 + i;
    p.cell_nodes[1 + i][3] = 5 + next;
  }
  return p;
}

namespace detail {

__host__ __device__ inline void patch_cell_vertices(
    const PolygonPatch& p, const int cell, double* r, double* z) {
  for (int i = 0; i < p.cell_n[cell]; ++i) {
    const int node = p.cell_nodes[cell][i];
    r[i] = p.node_r[node];
    z[i] = p.node_z[node];
  }
}

}  // namespace detail

__host__ __device__ inline void patch_node_masses(
    const PolygonPatch& p, const double* rho, double* m_node) {
  for (int node = 0; node < kPatchNodes; ++node) {
    m_node[node] = 0.0;
  }

  for (int cell = 0; cell < kPatchCells; ++cell) {
    double r[kPatchMaxVerts]{};
    double z[kPatchMaxVerts]{};
    double weight[kPatchMaxVerts]{};
    detail::patch_cell_vertices(p, cell, r, z);
    moments::star_p1_vertex_r_moments(
        r, z, p.cell_n[cell], weight);
    for (int i = 0; i < p.cell_n[cell]; ++i) {
      const int node = p.cell_nodes[cell][i];
      m_node[node] = fma(rho[cell], weight[i], m_node[node]);
    }
  }
}

__host__ __device__ inline void patch_node_forces(
    const PolygonPatch& p, const double* pressure, double* f_r,
    double* f_z) {
  for (int node = 0; node < kPatchNodes; ++node) {
    f_r[node] = 0.0;
    f_z[node] = 0.0;
  }

  for (int cell = 0; cell < kPatchCells; ++cell) {
    double r[kPatchMaxVerts]{};
    double z[kPatchMaxVerts]{};
    detail::patch_cell_vertices(p, cell, r, z);
    for (int i = 0; i < p.cell_n[cell]; ++i) {
      double corner_f_r = 0.0;
      double corner_f_z = 0.0;
      forces::polygon_corner_force(
          r, z, p.cell_n[cell], i, pressure[cell],
          corner_f_r, corner_f_z);
      const int node = p.cell_nodes[cell][i];
      f_r[node] += corner_f_r;
      f_z[node] += corner_f_z;
    }
  }
}

__host__ __device__ inline void patch_node_accelerations(
    const PolygonPatch& p, const double* pressure, const double* rho,
    double* a_r, double* a_z) {
  double m_node[kPatchNodes]{};
  patch_node_masses(p, rho, m_node);
  patch_node_forces(p, pressure, a_r, a_z);
  for (int node = 0; node < kPatchNodes; ++node) {
    a_r[node] /= m_node[node];
    a_z[node] /= m_node[node];
  }
}

}  // namespace tenryu::mesh::patch
