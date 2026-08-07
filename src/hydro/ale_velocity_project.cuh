#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "hydro/pole_axis_constraints.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale {

__global__ void compute_cell_velocity_from_nodes_kernel(
    double* __restrict__ v_r_cell,
    double* __restrict__ v_z_cell,
    const double* __restrict__ v_r_node,
    const double* __restrict__ v_z_node,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts == 3) {
    v_r_cell[c] = (v_r_node[n00] + v_r_node[n10] + v_r_node[n11]) / 3.0;
    v_z_cell[c] = (v_z_node[n00] + v_z_node[n10] + v_z_node[n11]) / 3.0;
    return;
  }

  v_r_cell[c] = 0.25 * (v_r_node[n00] + v_r_node[n10] + v_r_node[n11] + v_r_node[n01]);
  v_z_cell[c] = 0.25 * (v_z_node[n00] + v_z_node[n10] + v_z_node[n11] + v_z_node[n01]);
}

__global__ void project_cell_velocity_to_nodes_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ node_flags,
    const int nr,
    const int nz,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  double w_sum = 0.0;
  double vr_sum = 0.0;
  double vz_sum = 0.0;

  // Mass-weighted projection is robust and axisymmetric-friendly, but it is not
  // strictly conservative for total momentum/kinetic energy after node<->cell transfer.
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      const double w = fmax(rho[c] * vol[c], 0.0);
      w_sum += w;
      vr_sum += w * v_r_cell[c];
      vz_sum += w * v_z_cell[c];
    }
  }

  if (w_sum > 0.0) {
    v_r_node[n] = vr_sum / w_sum;
    v_z_node[n] = vz_sum / w_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }

  pole_axis::apply_2d_boundary_vector_constraints(
      v_r_node[n],
      v_z_node[n],
      node_flags,
      n,
      i,
      j,
      nr,
      nz,
      r_outer_bc_mode,
      z_bottom_bc_mode,
      z_top_bc_mode,
      false);
}

__global__ void normalize_volfrac_kernel(double* __restrict__ volfrac,
                                         int* __restrict__ degenerate_count,
                                         const int n_cells,
                                         const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  if (n_mat <= 0) {
    return;
  }
  if (n_mat == 1) {
    volfrac[c * n_mat] = 1.0;
    return;
  }

  double sum = 0.0;
  int imax = 0;
  double vmax = volfrac[c * n_mat];
  for (int m = 0; m < n_mat; ++m) {
    const int idx = c * n_mat + m;
    const double clamped = fmax(volfrac[idx], 0.0);
    volfrac[idx] = clamped;
    sum += clamped;
    if (volfrac[idx] > vmax) {
      vmax = volfrac[idx];
      imax = m;
    }
  }

  if (sum >= 1.0e-30) {
    for (int m = 0; m < n_mat; ++m) {
      const int idx = c * n_mat + m;
      volfrac[idx] /= sum;
    }
    return;
  }

  atomicAdd(degenerate_count, 1);
  for (int m = 0; m < n_mat; ++m) {
    const int idx = c * n_mat + m;
    volfrac[idx] = (m == imax) ? 1.0 : 0.0;
  }
}

__global__ void compute_density_from_mass_kernel(double* __restrict__ rho,
                                                 const double* __restrict__ mass,
                                                 const double* __restrict__ vol,
                                                 const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  rho[c] = (vol[c] > 0.0) ? (mass[c] / vol[c]) : 0.0;
}

}  // namespace tenryu::hydro::ale
