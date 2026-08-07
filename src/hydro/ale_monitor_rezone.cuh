#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "hydro/rz_corner_mass.cuh"
#include "mesh/boundary_kind.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale::monitor_rezone {

/*
Design note: adaptive monitor-weighted rezone
------------------------------------------------
This header is a cfg-free, default-off module.  It does not change the active
ALE path until a caller explicitly wires these kernels.

The monitor is cell-centered:

  w_c_raw = sqrt(1
                 + alpha_cm2 * |grad rho|^2 / rho^2
                 + beta * (V_lag / V0 - 1)^2
                 + gamma * (1 / quality))

Coordinates are in cm, rho is in g/cm^3, and V_lag/V0 uses TENRYU 2D RZ
volumes, dV = 2*pi*r dr dz.  Therefore alpha_cm2 has units cm^2; beta,
gamma, quality, and w are dimensionless.  The raw weight is area-normalized
to mean one over active cells and clamped to [weight_floor, weight_ceiling].
This keeps the monitor bounded before candidate-mesh admissibility gates.

Node monitors are area-weighted averages of incident normalized cell monitors.
A Winslow/Jacobi sweep consumes the node monitor through the graph Laplacian

  sum_m a_nm (x_m - x_n) = 0,  a_nm = 0.5 * (w_n + w_m),

for x = r and z.  In 1D, w*x_xi = const, so larger w gives smaller physical
spacing.  In the implosion use case, high weights near the gas/interface or
converging center make the reference mesh track contraction instead of pulling
the mesh back toward a uniform/open configuration.  Callers still own boundary
constraints, displacement limits, and global CSR admissibility line searches.
*/

struct MonitorWeightParams {
  double alpha_cm2 = 0.0;
  double beta = 0.0;
  double gamma = 0.0;
  double rho_floor = 1.0e-300;
  double quality_floor = 1.0e-12;
  double volume_floor = 1.0e-300;
  double area_floor = 1.0e-300;
  double weight_floor = 0.1;
  double weight_ceiling = 10.0;
};

struct MonitorWeightScratch {
  double* cell_area = nullptr;          // [n_cells]
  double* node_weight_sum = nullptr;    // [n_nodes]
  double* node_area_sum = nullptr;      // [n_nodes]
  double* normalization_sums = nullptr; // [2]: sum(w_raw*A), sum(A)
};

struct WeightedWinslowParams {
  double omega = 1.0;
  double coefficient_floor = 1.0e-12;
  double coefficient_ceiling = 1.0e12;
  bool freeze_boundary_nodes = true;
  bool freeze_center_nodes = true;
  bool freeze_axis_nodes = false;
  bool pin_axis_r = true;
  bool pin_pole_axis_r = true;
  bool enforce_nonnegative_r = true;
  bool smooth_cross_seams = true;
};

namespace detail {

constexpr int kMaxCellVerts = 4;
constexpr int kMaxNeighbors = 64;

__host__ __device__ inline double clamp_double(const double x,
                                               const double lo,
                                               const double hi) {
  return fmin(fmax(x, lo), hi);
}

__host__ __device__ inline MonitorWeightParams
sanitize_weight_params(MonitorWeightParams p) {
  p.alpha_cm2 = fmax(0.0, p.alpha_cm2);
  p.beta = fmax(0.0, p.beta);
  p.gamma = fmax(0.0, p.gamma);
  p.rho_floor = fmax(p.rho_floor, 1.0e-300);
  p.quality_floor = fmax(p.quality_floor, 1.0e-300);
  p.volume_floor = fmax(p.volume_floor, 1.0e-300);
  p.area_floor = fmax(p.area_floor, 1.0e-300);
  p.weight_floor = fmax(p.weight_floor, 1.0e-300);
  p.weight_ceiling = fmax(p.weight_ceiling, p.weight_floor);
  return p;
}

__host__ __device__ inline WeightedWinslowParams
sanitize_winslow_params(WeightedWinslowParams p) {
  p.omega = clamp_double(p.omega, 0.0, 1.0);
  p.coefficient_floor = fmax(p.coefficient_floor, 1.0e-300);
  p.coefficient_ceiling = fmax(p.coefficient_ceiling, p.coefficient_floor);
  return p;
}

__device__ inline bool cell_is_active(const std::uint8_t* cell_active_mask,
                                      const int c) {
  return cell_active_mask == nullptr || cell_active_mask[c] != 0U;
}

__device__ inline bool load_cell_planar_geometry(
    const int c,
    const int n_cells,
    const int n_nodes,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double area_floor,
    double* area_out,
    double* centroid_r_out,
    double* centroid_z_out) {
  if (c < 0 || c >= n_cells ||
      cell_node_csr_offsets == nullptr ||
      cell_node_csr_indices == nullptr ||
      node_r == nullptr || node_z == nullptr) {
    return false;
  }

  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int begin = cell_node_csr_offsets[c];
  const int end = cell_node_csr_offsets[c + 1];
  if (begin < 0 || end < begin + active_nverts) {
    return false;
  }

  double rr[kMaxCellVerts] = {0.0, 0.0, 0.0, 0.0};
  double zz[kMaxCellVerts] = {0.0, 0.0, 0.0, 0.0};
  double avg_r = 0.0;
  double avg_z = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[begin + k];
    if (n < 0 || n >= n_nodes) {
      return false;
    }
    rr[k] = node_r[n];
    zz[k] = node_z[n];
    if (!isfinite(rr[k]) || !isfinite(zz[k])) {
      return false;
    }
    avg_r += rr[k];
    avg_z += zz[k];
  }
  avg_r /= static_cast<double>(active_nverts);
  avg_z /= static_cast<double>(active_nverts);

  const double area2 = tenryu::hydro::rz::rz_polygon_area2_exact(
      rr, zz, active_nverts);
  double area = 0.5 * fabs(area2);
  double centroid_r = avg_r;
  double centroid_z = avg_z;
  if (isfinite(area2) && area > 0.0) {
    tenryu::hydro::rz::rz_polygon_area_centroid_exact(
        rr, zz, active_nverts, &centroid_r, &centroid_z);
    if (!isfinite(centroid_r) || !isfinite(centroid_z)) {
      centroid_r = avg_r;
      centroid_z = avg_z;
    }
  }

  area = isfinite(area) ? fmax(area, area_floor) : area_floor;
  *area_out = area;
  *centroid_r_out = centroid_r;
  *centroid_z_out = centroid_z;
  return isfinite(area) && isfinite(centroid_r) && isfinite(centroid_z);
}

__device__ inline double relative_density_gradient2(
    const int c,
    const int n_cells,
    const int n_nodes,
    const int* __restrict__ cell_adj_csr_offsets,
    const int* __restrict__ cell_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ cell_active_mask,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ rho,
    const double area_floor,
    const double rho_floor,
    const double centroid_r,
    const double centroid_z) {
  if (cell_adj_csr_offsets == nullptr ||
      cell_adj_csr_indices == nullptr ||
      rho == nullptr || !isfinite(rho[c])) {
    return 0.0;
  }

  const double rho_scale = fmax(fabs(rho[c]), rho_floor);
  double a00 = 0.0;
  double a01 = 0.0;
  double a11 = 0.0;
  double b0 = 0.0;
  double b1 = 0.0;
  double fallback_grad2 = 0.0;

  const int begin = cell_adj_csr_offsets[c];
  const int end = cell_adj_csr_offsets[c + 1];
  for (int p = begin; p < end; ++p) {
    const int nb = cell_adj_csr_indices[p];
    if (nb < 0 || nb >= n_cells || nb == c ||
        !cell_is_active(cell_active_mask, nb) ||
        !isfinite(rho[nb])) {
      continue;
    }

    double nb_area = 0.0;
    double nb_r = 0.0;
    double nb_z = 0.0;
    if (!load_cell_planar_geometry(nb,
                                   n_cells,
                                   n_nodes,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   node_r,
                                   node_z,
                                   area_floor,
                                   &nb_area,
                                   &nb_r,
                                   &nb_z)) {
      continue;
    }
    (void)nb_area;

    const double dr = nb_r - centroid_r;
    const double dz = nb_z - centroid_z;
    const double dist2 = dr * dr + dz * dz;
    if (!isfinite(dist2) || !(dist2 > 0.0)) {
      continue;
    }
    const double d_rho = rho[nb] - rho[c];
    const double inv_dist2 = 1.0 / dist2;
    const double ls_weight = inv_dist2;
    a00 += ls_weight * dr * dr;
    a01 += ls_weight * dr * dz;
    a11 += ls_weight * dz * dz;
    b0 += ls_weight * d_rho * dr;
    b1 += ls_weight * d_rho * dz;
    fallback_grad2 = fmax(fallback_grad2, d_rho * d_rho * inv_dist2);
  }

  double grad2 = fallback_grad2;
  const double det = a00 * a11 - a01 * a01;
  if (isfinite(det) && fabs(det) > 1.0e-300) {
    const double inv_det = 1.0 / det;
    const double grad_r = (a11 * b0 - a01 * b1) * inv_det;
    const double grad_z = (-a01 * b0 + a00 * b1) * inv_det;
    const double ls_grad2 = grad_r * grad_r + grad_z * grad_z;
    if (isfinite(ls_grad2)) {
      grad2 = ls_grad2;
    }
  }

  const double rel_grad2 = grad2 / (rho_scale * rho_scale);
  return isfinite(rel_grad2) ? rel_grad2 : 0.0;
}

__device__ inline double cell_monitor_raw(
    const int c,
    const int n_cells,
    const int n_nodes,
    const int* __restrict__ cell_adj_csr_offsets,
    const int* __restrict__ cell_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ cell_active_mask,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const double* __restrict__ rho,
    const double* __restrict__ lagrangian_volume,
    const double* __restrict__ initial_volume,
    const double* __restrict__ cell_quality,
    const MonitorWeightParams params,
    const double centroid_r,
    const double centroid_z) {
  double density_term = 0.0;
  if (params.alpha_cm2 > 0.0) {
    density_term = params.alpha_cm2 *
        relative_density_gradient2(c,
                                   n_cells,
                                   n_nodes,
                                   cell_adj_csr_offsets,
                                   cell_adj_csr_indices,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_nverts,
                                   cell_active_mask,
                                   node_r,
                                   node_z,
                                   rho,
                                   params.area_floor,
                                   params.rho_floor,
                                   centroid_r,
                                   centroid_z);
  }

  double volume_term = 0.0;
  if (params.beta > 0.0 &&
      lagrangian_volume != nullptr &&
      initial_volume != nullptr &&
      isfinite(lagrangian_volume[c]) &&
      isfinite(initial_volume[c])) {
    const double v0_scale = fmax(fabs(initial_volume[c]), params.volume_floor);
    const double ratio = lagrangian_volume[c] / v0_scale;
    const double delta = ratio - 1.0;
    if (isfinite(delta)) {
      volume_term = params.beta * delta * delta;
    }
  }

  double quality_term = 0.0;
  if (params.gamma > 0.0 && cell_quality != nullptr) {
    const double q = isfinite(cell_quality[c])
                         ? fmax(cell_quality[c], params.quality_floor)
                         : params.quality_floor;
    quality_term = params.gamma / q;
  }

  double arg = 1.0 + density_term + volume_term + quality_term;
  if (!isfinite(arg) || arg < 1.0) {
    arg = 1.0;
  }
  const double raw = sqrt(arg);
  return isfinite(raw) ? clamp_double(raw,
                                      params.weight_floor,
                                      params.weight_ceiling)
                       : params.weight_ceiling;
}

static __global__ void cell_monitor_raw_kernel(
    double* __restrict__ cell_weight,
    double* __restrict__ cell_area,
    double* __restrict__ normalization_sums,
    const double* __restrict__ rho,
    const double* __restrict__ lagrangian_volume,
    const double* __restrict__ initial_volume,
    const double* __restrict__ cell_quality,
    const int* __restrict__ cell_adj_csr_offsets,
    const int* __restrict__ cell_adj_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ cell_active_mask,
    const double* __restrict__ node_r,
    const double* __restrict__ node_z,
    const int n_cells,
    const int n_nodes,
    MonitorWeightParams params) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  params = sanitize_weight_params(params);

  double area = 0.0;
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  const bool active = cell_is_active(cell_active_mask, c);
  const bool geometry_ok =
      active &&
      load_cell_planar_geometry(c,
                                n_cells,
                                n_nodes,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell_nverts,
                                node_r,
                                node_z,
                                params.area_floor,
                                &area,
                                &centroid_r,
                                &centroid_z);
  if (!geometry_ok) {
    cell_weight[c] = 1.0;
    cell_area[c] = 0.0;
    return;
  }

  const double raw = cell_monitor_raw(c,
                                      n_cells,
                                      n_nodes,
                                      cell_adj_csr_offsets,
                                      cell_adj_csr_indices,
                                      cell_node_csr_offsets,
                                      cell_node_csr_indices,
                                      cell_nverts,
                                      cell_active_mask,
                                      node_r,
                                      node_z,
                                      rho,
                                      lagrangian_volume,
                                      initial_volume,
                                      cell_quality,
                                      params,
                                      centroid_r,
                                      centroid_z);
  cell_weight[c] = raw;
  cell_area[c] = area;
  atomicAdd(&normalization_sums[0], raw * area);
  atomicAdd(&normalization_sums[1], area);
}

static __global__ void cell_monitor_normalize_kernel(
    double* __restrict__ cell_weight,
    const double* __restrict__ cell_area,
    const double* __restrict__ normalization_sums,
    const int n_cells,
    MonitorWeightParams params) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  params = sanitize_weight_params(params);
  const double denom = normalization_sums[1];
  double mean = 1.0;
  if (isfinite(denom) && denom > 0.0) {
    const double candidate_mean = normalization_sums[0] / denom;
    if (isfinite(candidate_mean) && candidate_mean > 0.0) {
      mean = candidate_mean;
    }
  }
  const double area = cell_area[c];
  if (!isfinite(area) || !(area > 0.0)) {
    cell_weight[c] = 1.0;
    return;
  }
  const double normalized = cell_weight[c] / mean;
  cell_weight[c] = isfinite(normalized)
                       ? clamp_double(normalized,
                                      params.weight_floor,
                                      params.weight_ceiling)
                       : 1.0;
}

static __global__ void node_monitor_accumulate_kernel(
    double* __restrict__ node_weight_sum,
    double* __restrict__ node_area_sum,
    const double* __restrict__ cell_weight,
    const double* __restrict__ cell_area,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const int n_nodes) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double area = cell_area[c];
  const double weight = cell_weight[c];
  if (!isfinite(area) || !(area > 0.0) ||
      !isfinite(weight) || !(weight > 0.0)) {
    return;
  }
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int begin = cell_node_csr_offsets[c];
  const int end = cell_node_csr_offsets[c + 1];
  if (begin < 0 || end < begin + active_nverts) {
    return;
  }
  for (int k = 0; k < active_nverts; ++k) {
    const int n = cell_node_csr_indices[begin + k];
    if (n < 0 || n >= n_nodes) {
      continue;
    }
    atomicAdd(&node_weight_sum[n], weight * area);
    atomicAdd(&node_area_sum[n], area);
  }
}

static __global__ void node_monitor_finalize_kernel(
    double* __restrict__ node_weight,
    const double* __restrict__ node_weight_sum,
    const double* __restrict__ node_area_sum,
    const int n_nodes,
    MonitorWeightParams params) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  params = sanitize_weight_params(params);
  const double area = node_area_sum[n];
  if (!isfinite(area) || !(area > 0.0)) {
    node_weight[n] = 1.0;
    return;
  }
  const double averaged = node_weight_sum[n] / area;
  node_weight[n] = isfinite(averaged)
                       ? clamp_double(averaged,
                                      params.weight_floor,
                                      params.weight_ceiling)
                       : 1.0;
}

__host__ __device__ inline double edge_coefficient(
    const double w_a,
    const double w_b,
    WeightedWinslowParams params) {
  params = sanitize_winslow_params(params);
  const double coeff = 0.5 * (w_a + w_b);
  return isfinite(coeff)
             ? clamp_double(coeff,
                            params.coefficient_floor,
                            params.coefficient_ceiling)
             : params.coefficient_floor;
}

static __global__ void weighted_laplacian_coefficients_graph_kernel(
    double* __restrict__ edge_coefficients,
    const int* __restrict__ node_neighbor_csr_offsets,
    const int* __restrict__ node_neighbor_csr_indices,
    const double* __restrict__ node_weight,
    const int n_nodes,
    WeightedWinslowParams params) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  params = sanitize_winslow_params(params);
  const double w_n = node_weight[n];
  const int begin = node_neighbor_csr_offsets[n];
  const int end = node_neighbor_csr_offsets[n + 1];
  for (int p = begin; p < end; ++p) {
    const int nb = node_neighbor_csr_indices[p];
    edge_coefficients[p] =
        (nb >= 0 && nb < n_nodes)
            ? edge_coefficient(w_n, node_weight[nb], params)
            : params.coefficient_floor;
  }
}

__device__ inline bool is_seam_tag(const std::int8_t tag) {
  return tag ==
             static_cast<std::int8_t>(
                 mesh::BoundaryKind::SEAM_CORE_BRIDGE) ||
         tag ==
             static_cast<std::int8_t>(
                 mesh::BoundaryKind::SEAM_BRIDGE_SHELL);
}

__device__ inline bool fixed_node(const std::uint8_t flags,
                                  const WeightedWinslowParams params) {
  if (params.freeze_boundary_nodes &&
      (flags & mesh::NODE_BOUNDARY) != 0U) {
    return true;
  }
  if (params.freeze_center_nodes &&
      (flags & mesh::NODE_CENTER) != 0U) {
    return true;
  }
  if (params.freeze_axis_nodes &&
      (flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U) {
    return true;
  }
  return false;
}

__device__ inline bool append_neighbor(int* neighbors,
                                       int& n_neighbors,
                                       bool& overflow,
                                       const int self,
                                       const int candidate) {
  if (candidate == self || candidate < 0) {
    return true;
  }
  for (int k = 0; k < n_neighbors; ++k) {
    if (neighbors[k] == candidate) {
      return true;
    }
  }
  if (n_neighbors >= kMaxNeighbors) {
    overflow = true;
    return false;
  }
  neighbors[n_neighbors++] = candidate;
  return true;
}

__device__ inline void apply_node_constraints(double& r,
                                              const double old_r,
                                              const std::uint8_t flags,
                                              const WeightedWinslowParams params) {
  if (params.pin_axis_r && (flags & mesh::NODE_AXIS) != 0U) {
    r = 0.0;
  }
  if (params.pin_pole_axis_r && (flags & mesh::NODE_POLE_AXIS) != 0U) {
    r = 0.0;
  }
  if (params.enforce_nonnegative_r && r < 0.0) {
    r = 0.0;
  }
  if (!isfinite(r)) {
    r = old_r;
  }
}

static __global__ void monitor_weighted_winslow_graph_jacobi_kernel(
    double* __restrict__ x_r_new,
    double* __restrict__ x_z_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ node_weight,
    const int* __restrict__ node_neighbor_csr_offsets,
    const int* __restrict__ node_neighbor_csr_indices,
    const double* __restrict__ edge_coefficients,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_active_mask,
    const int n_nodes,
    WeightedWinslowParams params) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  params = sanitize_winslow_params(params);

  const double old_r = x_r_old[n];
  const double old_z = x_z_old[n];
  x_r_new[n] = old_r;
  x_z_new[n] = old_z;

  const std::uint8_t flags = node_flags == nullptr ? 0U : node_flags[n];
  if ((node_active_mask != nullptr && node_active_mask[n] == 0U) ||
      fixed_node(flags, params)) {
    return;
  }

  const int begin = node_neighbor_csr_offsets[n];
  const int end = node_neighbor_csr_offsets[n + 1];
  double sum_r = 0.0;
  double sum_z = 0.0;
  double denom = 0.0;
  for (int p = begin; p < end; ++p) {
    const int nb = node_neighbor_csr_indices[p];
    if (nb < 0 || nb >= n_nodes) {
      continue;
    }
    const double coeff =
        edge_coefficients != nullptr
            ? edge_coefficients[p]
            : edge_coefficient(node_weight[n], node_weight[nb], params);
    if (!isfinite(coeff) || !(coeff > 0.0)) {
      continue;
    }
    sum_r += coeff * x_r_old[nb];
    sum_z += coeff * x_z_old[nb];
    denom += coeff;
  }
  if (!isfinite(denom) || !(denom > 0.0)) {
    return;
  }

  double candidate_r = (1.0 - params.omega) * old_r +
                       params.omega * (sum_r / denom);
  double candidate_z = (1.0 - params.omega) * old_z +
                       params.omega * (sum_z / denom);
  apply_node_constraints(candidate_r, old_r, flags, params);
  if (!isfinite(candidate_z)) {
    candidate_z = old_z;
  }
  x_r_new[n] = candidate_r;
  x_z_new[n] = candidate_z;
}

static __global__ void monitor_weighted_winslow_csr_jacobi_kernel(
    double* __restrict__ x_r_new,
    double* __restrict__ x_z_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ node_weight,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int* __restrict__ face_adj_csr_offsets,
    const std::int8_t* __restrict__ face_bc_tags,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_active_mask,
    const int n_nodes,
    WeightedWinslowParams params) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  params = sanitize_winslow_params(params);

  const double old_r = x_r_old[n];
  const double old_z = x_z_old[n];
  x_r_new[n] = old_r;
  x_z_new[n] = old_z;

  const std::uint8_t flags = node_flags == nullptr ? 0U : node_flags[n];
  if ((node_active_mask != nullptr && node_active_mask[n] == 0U) ||
      fixed_node(flags, params)) {
    return;
  }

  const int incident_begin = node_cell_csr_offsets[n];
  const int incident_end = node_cell_csr_offsets[n + 1];
  if (incident_end <= incident_begin) {
    return;
  }

  int incident_block = -1;
  bool seam_touching = false;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int c = node_cell_csr_indices[p];
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (cell_block_id != nullptr) {
      const int block = cell_block_id[c];
      if (incident_block < 0) {
        incident_block = block;
      } else if (block != incident_block) {
        seam_touching = true;
      }
    }
    if (face_adj_csr_offsets != nullptr && face_bc_tags != nullptr) {
      const int face_begin = face_adj_csr_offsets[c];
      const int face_end = face_adj_csr_offsets[c + 1];
      for (int f = face_begin; f < face_end; ++f) {
        const int local_face = f - face_begin;
        if (!mesh::mesh_topo_local_face_is_active(active_nverts, local_face)) {
          continue;
        }
        if (is_seam_tag(face_bc_tags[f])) {
          seam_touching = true;
        }
      }
    }
  }
  if (seam_touching && !params.smooth_cross_seams) {
    return;
  }

  int neighbors[kMaxNeighbors];
  int n_neighbors = 0;
  bool overflow = false;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int c = node_cell_csr_indices[p];
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    const int cell_begin = cell_node_csr_offsets[c];
    const int cell_end = cell_node_csr_offsets[c + 1];
    if (cell_begin < 0 || cell_end < cell_begin + active_nverts) {
      overflow = true;
      continue;
    }

    if (seam_touching && params.smooth_cross_seams &&
        node_cell_csr_corners != nullptr) {
      const int corner = node_cell_csr_corners[p];
      if (corner < 0 || corner >= active_nverts) {
        overflow = true;
        continue;
      }
      const int local_a =
          active_nverts == 3 ? (corner + 1) % 3 : (corner + 1) & 3;
      const int local_b =
          active_nverts == 3 ? (corner + 2) % 3 : (corner + 3) & 3;
      append_neighbor(neighbors,
                      n_neighbors,
                      overflow,
                      n,
                      cell_node_csr_indices[cell_begin + local_a]);
      append_neighbor(neighbors,
                      n_neighbors,
                      overflow,
                      n,
                      cell_node_csr_indices[cell_begin + local_b]);
      continue;
    }

    for (int q = cell_begin; q < cell_begin + active_nverts; ++q) {
      append_neighbor(neighbors,
                      n_neighbors,
                      overflow,
                      n,
                      cell_node_csr_indices[q]);
    }
  }
  if (overflow || n_neighbors == 0) {
    return;
  }

  double sum_r = 0.0;
  double sum_z = 0.0;
  double denom = 0.0;
  for (int k = 0; k < n_neighbors; ++k) {
    const int nb = neighbors[k];
    if (nb < 0 || nb >= n_nodes) {
      continue;
    }
    const double coeff = edge_coefficient(node_weight[n], node_weight[nb], params);
    if (!isfinite(coeff) || !(coeff > 0.0)) {
      continue;
    }
    sum_r += coeff * x_r_old[nb];
    sum_z += coeff * x_z_old[nb];
    denom += coeff;
  }
  if (!isfinite(denom) || !(denom > 0.0)) {
    return;
  }

  double candidate_r = (1.0 - params.omega) * old_r +
                       params.omega * (sum_r / denom);
  double candidate_z = (1.0 - params.omega) * old_z +
                       params.omega * (sum_z / denom);
  apply_node_constraints(candidate_r, old_r, flags, params);
  if (!isfinite(candidate_z)) {
    candidate_z = old_z;
  }
  x_r_new[n] = candidate_r;
  x_z_new[n] = candidate_z;
}

inline cudaError_t launch_monitor_weights(
    double* d_cell_weight,
    double* d_node_weight,
    MonitorWeightScratch scratch,
    const double* d_rho,
    const double* d_lagrangian_volume,
    const double* d_initial_volume,
    const double* d_cell_quality,
    const int* d_cell_adj_csr_offsets,
    const int* d_cell_adj_csr_indices,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_cell_active_mask,
    const double* d_node_r,
    const double* d_node_z,
    const int n_cells,
    const int n_nodes,
    MonitorWeightParams params,
    cudaStream_t stream = nullptr) {
  if (n_cells < 0 || n_nodes < 0 ||
      d_cell_weight == nullptr ||
      d_node_weight == nullptr ||
      scratch.cell_area == nullptr ||
      scratch.node_weight_sum == nullptr ||
      scratch.node_area_sum == nullptr ||
      scratch.normalization_sums == nullptr ||
      d_cell_node_csr_offsets == nullptr ||
      d_cell_node_csr_indices == nullptr ||
      d_node_r == nullptr || d_node_z == nullptr ||
      (params.alpha_cm2 > 0.0 &&
       (d_rho == nullptr ||
        d_cell_adj_csr_offsets == nullptr ||
        d_cell_adj_csr_indices == nullptr)) ||
      (params.beta > 0.0 &&
       (d_lagrangian_volume == nullptr || d_initial_volume == nullptr)) ||
      (params.gamma > 0.0 && d_cell_quality == nullptr)) {
    return cudaErrorInvalidValue;
  }
  if (n_cells == 0 || n_nodes == 0) {
    return cudaSuccess;
  }

  cudaError_t err = cudaMemsetAsync(
      scratch.normalization_sums, 0, 2 * sizeof(double), stream);
  if (err != cudaSuccess) {
    return err;
  }

  constexpr int threads = 256;
  const int cell_blocks = (n_cells + threads - 1) / threads;
  cell_monitor_raw_kernel<<<cell_blocks, threads, 0, stream>>>(
      d_cell_weight,
      scratch.cell_area,
      scratch.normalization_sums,
      d_rho,
      d_lagrangian_volume,
      d_initial_volume,
      d_cell_quality,
      d_cell_adj_csr_offsets,
      d_cell_adj_csr_indices,
      d_cell_node_csr_offsets,
      d_cell_node_csr_indices,
      d_cell_nverts,
      d_cell_active_mask,
      d_node_r,
      d_node_z,
      n_cells,
      n_nodes,
      params);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }

  cell_monitor_normalize_kernel<<<cell_blocks, threads, 0, stream>>>(
      d_cell_weight,
      scratch.cell_area,
      scratch.normalization_sums,
      n_cells,
      params);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }

  err = cudaMemsetAsync(
      scratch.node_weight_sum, 0, static_cast<std::size_t>(n_nodes) * sizeof(double), stream);
  if (err != cudaSuccess) {
    return err;
  }
  err = cudaMemsetAsync(
      scratch.node_area_sum, 0, static_cast<std::size_t>(n_nodes) * sizeof(double), stream);
  if (err != cudaSuccess) {
    return err;
  }

  node_monitor_accumulate_kernel<<<cell_blocks, threads, 0, stream>>>(
      scratch.node_weight_sum,
      scratch.node_area_sum,
      d_cell_weight,
      scratch.cell_area,
      d_cell_node_csr_offsets,
      d_cell_node_csr_indices,
      d_cell_nverts,
      n_cells,
      n_nodes);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    return err;
  }

  const int node_blocks = (n_nodes + threads - 1) / threads;
  node_monitor_finalize_kernel<<<node_blocks, threads, 0, stream>>>(
      d_node_weight,
      scratch.node_weight_sum,
      scratch.node_area_sum,
      n_nodes,
      params);
  return cudaGetLastError();
}

inline cudaError_t launch_weighted_laplacian_coefficients_graph(
    double* d_edge_coefficients,
    const int* d_node_neighbor_csr_offsets,
    const int* d_node_neighbor_csr_indices,
    const double* d_node_weight,
    const int n_nodes,
    WeightedWinslowParams params,
    cudaStream_t stream = nullptr) {
  if (n_nodes < 0 ||
      d_edge_coefficients == nullptr ||
      d_node_neighbor_csr_offsets == nullptr ||
      d_node_neighbor_csr_indices == nullptr ||
      d_node_weight == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n_nodes == 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  const int node_blocks = (n_nodes + threads - 1) / threads;
  weighted_laplacian_coefficients_graph_kernel<<<node_blocks, threads, 0, stream>>>(
      d_edge_coefficients,
      d_node_neighbor_csr_offsets,
      d_node_neighbor_csr_indices,
      d_node_weight,
      n_nodes,
      params);
  return cudaGetLastError();
}

inline cudaError_t launch_monitor_weighted_winslow_graph_step(
    double* d_x_r_new,
    double* d_x_z_new,
    const double* d_x_r_old,
    const double* d_x_z_old,
    const double* d_node_weight,
    const int* d_node_neighbor_csr_offsets,
    const int* d_node_neighbor_csr_indices,
    const double* d_edge_coefficients,
    const std::uint8_t* d_node_flags,
    const std::uint8_t* d_node_active_mask,
    const int n_nodes,
    WeightedWinslowParams params,
    cudaStream_t stream = nullptr) {
  if (n_nodes < 0 ||
      d_x_r_new == nullptr || d_x_z_new == nullptr ||
      d_x_r_old == nullptr || d_x_z_old == nullptr ||
      d_node_weight == nullptr ||
      d_node_neighbor_csr_offsets == nullptr ||
      d_node_neighbor_csr_indices == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n_nodes == 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  const int node_blocks = (n_nodes + threads - 1) / threads;
  monitor_weighted_winslow_graph_jacobi_kernel<<<node_blocks, threads, 0, stream>>>(
      d_x_r_new,
      d_x_z_new,
      d_x_r_old,
      d_x_z_old,
      d_node_weight,
      d_node_neighbor_csr_offsets,
      d_node_neighbor_csr_indices,
      d_edge_coefficients,
      d_node_flags,
      d_node_active_mask,
      n_nodes,
      params);
  return cudaGetLastError();
}

inline cudaError_t launch_monitor_weighted_winslow_csr_step(
    double* d_x_r_new,
    double* d_x_z_new,
    const double* d_x_r_old,
    const double* d_x_z_old,
    const double* d_node_weight,
    const int* d_node_cell_csr_offsets,
    const int* d_node_cell_csr_indices,
    const int* d_node_cell_csr_corners,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const std::uint8_t* d_cell_nverts,
    const int* d_cell_block_id,
    const int* d_face_adj_csr_offsets,
    const std::int8_t* d_face_bc_tags,
    const std::uint8_t* d_node_flags,
    const std::uint8_t* d_node_active_mask,
    const int n_nodes,
    WeightedWinslowParams params,
    cudaStream_t stream = nullptr) {
  if (n_nodes < 0 ||
      d_x_r_new == nullptr || d_x_z_new == nullptr ||
      d_x_r_old == nullptr || d_x_z_old == nullptr ||
      d_node_weight == nullptr ||
      d_node_cell_csr_offsets == nullptr ||
      d_node_cell_csr_indices == nullptr ||
      d_cell_node_csr_offsets == nullptr ||
      d_cell_node_csr_indices == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (n_nodes == 0) {
    return cudaSuccess;
  }
  constexpr int threads = 256;
  const int node_blocks = (n_nodes + threads - 1) / threads;
  monitor_weighted_winslow_csr_jacobi_kernel<<<node_blocks, threads, 0, stream>>>(
      d_x_r_new,
      d_x_z_new,
      d_x_r_old,
      d_x_z_old,
      d_node_weight,
      d_node_cell_csr_offsets,
      d_node_cell_csr_indices,
      d_node_cell_csr_corners,
      d_cell_node_csr_offsets,
      d_cell_node_csr_indices,
      d_cell_nverts,
      d_cell_block_id,
      d_face_adj_csr_offsets,
      d_face_bc_tags,
      d_node_flags,
      d_node_active_mask,
      n_nodes,
      params);
  return cudaGetLastError();
}

}  // namespace detail

using detail::edge_coefficient;
using detail::launch_monitor_weighted_winslow_csr_step;
using detail::launch_monitor_weighted_winslow_graph_step;
using detail::launch_monitor_weights;
using detail::launch_weighted_laplacian_coefficients_graph;

}  // namespace tenryu::hydro::ale::monitor_rezone
