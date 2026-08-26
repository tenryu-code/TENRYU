#pragma once

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

#include "core/error.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale::center_patch_barrier {

/*
Design note: CP4 center-patch Phi-barrier optimizer

Scope:
  This header is intentionally cfg-free and uncalled by construction. It owns
  only the local center-patch mesh optimizer requested by the I1-B ALE verdict;
  Driver integration is responsible for trigger hysteresis, mask construction,
  final CSR sigma-linesearch, diagnostics, and default-off config plumbing.

Coordinates and units:
  All coordinates are physical 2D RZ coordinates in cm. Cell volume barriers use
  TENRYU's exact cgs RZ polygon volume
    V = (pi/3) sum_k (r_k z_{k+1} - r_{k+1} z_k)(r_k + r_{k+1})
  and therefore include the 2*pi*r rotation factor. Corner-J barriers use the
  existing planar corner cross product in cm^2. No internal unit conversion is
  introduced; TENRYU remains cgs+eV.

Objective on active patch-interior nodes n:
  Phi_n(x) =
      w_L |x - x_L|^2 / h_n^2
    + w_S sum_edges ((|e(x)| - |e_L|)^2 / h_n^2)
    + w_V sum_{c in C(n)} -log(V_c(x) / V_floor,c)
    + w_J sum_{c in C(n),k} -log(J_{c,k}(x) / J_floor,c,k)

  h_n is a local length scale from incident reference polygon area/edge length.
  V_floor,c=max(volume_floor_abs, volume_floor_rel*|V_ref,c|) and
  J_floor,c,k=max(jacobian_floor_abs, jacobian_floor_rel*|J_ref,c,k|).
  Trial states with non-finite coordinates, V<=floor, or J<=floor are rejected.

Constraints and execution model:
  node_rezone_active[n]==0 and node_patch_boundary[n]!=0 are frozen to the
  supplied Lagrangian coordinate. NODE_AXIS/NODE_POLE_AXIS nodes keep r=0.
  NODE_CENTER and cap_apex_node_id keep (r,z)=(0,0). The optimizer performs
  deterministic in-place projected Gauss-Seidel sweeps: one device thread visits
  active nodes in id order, and each node solves a local finite-difference
  Newton step with backtracking. This avoids race-prone adjacent-node updates;
  the center patch is expected to be small.
*/

struct CenterPatchBarrierParams {
  double weight_lagrangian = 1.0;
  double weight_edge = 0.5;
  double weight_volume = 1.0;
  double weight_jacobian = 1.0;
  double volume_floor_rel = 1.0e-8;
  double jacobian_floor_rel = 1.0e-8;
  double volume_floor_abs = 0.0;
  double jacobian_floor_abs = 0.0;
  double convergence_tol = 1.0e-6;
  double finite_difference_rel = 1.0e-8;
  double min_step_rel = 1.0e-14;
  double max_step_fraction = 0.5;
  double armijo_c1 = 1.0e-4;
  int max_sweeps = 8;
  int max_newton_iters = 10;
  int max_backtrack_iters = 16;
  int cap_apex_node_id = 0;
  bool has_physical_rz_axis = true;
  bool constrain_cap_apex = true;
};

struct CenterPatchBarrierResult {
  bool converged = false;
  int sweeps = 0;
  int active_nodes = 0;
  int accepted_steps = 0;
  int rejected_steps = 0;
  int invalid_nodes = 0;
  // Revert post-pass: active nodes still at an inadmissible position after
  // the sweeps are reverted to their Lagrangian coordinates (identity
  // update) so the emitted rezone target NEVER carries a poisoned node —
  // a retained invalid node was measured to hand the CSR remap phantom
  // swept volumes that drained the cell and injected mass through its
  // boundary face (+1.46e-5 g in one fire). invalid_nodes_final counts
  // nodes inadmissible even at their Lagrangian position (mesh already
  // bad there; identity emitted).
  int reverted_nodes = 0;
  int invalid_nodes_final = 0;
  double max_step = 0.0;
  double max_grad_norm = 0.0;
  double final_phi = 0.0;
};

namespace detail {

constexpr double kTiny = 1.0e-300;
constexpr double kHuge = 1.0e300;

__host__ __device__ inline bool finite(const double x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

__host__ __device__ inline double sqr(const double x) {
  return x * x;
}

__host__ __device__ inline double clamp_positive_floor(const double value) {
  return finite(value) && value > kTiny ? value : kTiny;
}

__host__ __device__ inline int active_nverts(
    const std::uint8_t* __restrict__ cell_nverts,
    const int cell) {
  return tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
}

__host__ __device__ inline double orientation_sign(
    const int* __restrict__ cell_orientation_sign,
    const int cell) {
  if (cell_orientation_sign == nullptr) {
    return 1.0;
  }
  return cell_orientation_sign[cell] < 0 ? -1.0 : 1.0;
}

__host__ __device__ inline double polygon_area2(const double* r,
                                                const double* z,
                                                const int nverts) {
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : k + 1;
    sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return sum;
}

__host__ __device__ inline double rz_volume_oriented(const double* r,
                                                     const double* z,
                                                     const int nverts,
                                                     const double orient) {
  return orient * tenryu::hydro::rz::rz_polygon_volume_exact(r, z, nverts);
}

__host__ __device__ inline double corner_j_oriented(const double* r,
                                                    const double* z,
                                                    const int corner,
                                                    const int nverts,
                                                    const double orient) {
  if (nverts == 3) {
    const int kp = (corner + 1) % 3;
    const int km = (corner + 2) % 3;
    return orient * tenryu::hydro::corner_jacobian_cross2(
                        r[kp] - r[corner], z[kp] - z[corner],
                        r[km] - r[corner], z[km] - z[corner]);
  }
  return orient * tenryu::hydro::corner_jacobian_from_quad(r, z, corner);
}

__host__ __device__ inline bool node_is_axis(
    const std::uint8_t* __restrict__ node_flags,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_r_reference,
    const int n,
    const CenterPatchBarrierParams& params) {
  if (node_flags != nullptr) {
    const std::uint8_t flags = node_flags[n];
    if ((flags & (tenryu::mesh::NODE_AXIS |
                  tenryu::mesh::NODE_POLE_AXIS)) != 0U) {
      return true;
    }
  }
  if (!params.has_physical_rz_axis) {
    return false;
  }
  const double r_lag = x_r_lagrangian != nullptr ? x_r_lagrangian[n] : 0.0;
  const double r_ref = x_r_reference != nullptr ? x_r_reference[n] : r_lag;
  return finite(r_lag) && finite(r_ref) && fabs(r_lag) <= 0.0 &&
         fabs(r_ref) <= 0.0;
}

__host__ __device__ inline bool node_is_apex(
    const std::uint8_t* __restrict__ node_flags,
    const int n,
    const CenterPatchBarrierParams& params) {
  if (node_flags != nullptr &&
      (node_flags[n] & tenryu::mesh::NODE_CENTER) != 0U) {
    return true;
  }
  return params.constrain_cap_apex && n == params.cap_apex_node_id;
}

__host__ __device__ inline void project_constraints(
    double& r,
    double& z,
    const std::uint8_t* __restrict__ node_flags,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_r_reference,
    const int n,
    const CenterPatchBarrierParams& params) {
  if (node_is_apex(node_flags, n, params)) {
    r = 0.0;
    z = 0.0;
    return;
  }
  if (node_is_axis(node_flags, x_r_lagrangian, x_r_reference, n, params)) {
    r = 0.0;
  } else if (params.has_physical_rz_axis && r < 0.0) {
    r = 0.0;
  }
}

__host__ __device__ inline bool node_active(
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const int n) {
  if (node_rezone_active == nullptr || node_rezone_active[n] == 0U) {
    return false;
  }
  return node_patch_boundary == nullptr || node_patch_boundary[n] == 0U;
}

__host__ __device__ inline bool incident_cell_contains_only_patch_nodes(
    const int cell,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const CenterPatchBarrierParams& params) {
  if (node_rezone_active == nullptr) {
    return false;
  }
  const int begin = cell_node_csr_offsets[cell];
  const int end = cell_node_csr_offsets[cell + 1];
  const int nverts = active_nverts(cell_nverts, cell);
  if (end - begin != tenryu::mesh::kMeshTopoCellStorageSlots) {
    return false;
  }
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_node_csr_indices[begin + k];
    const bool active =
        node_rezone_active[node] != 0U &&
        (node_patch_boundary == nullptr || node_patch_boundary[node] == 0U);
    const bool boundary =
        node_patch_boundary != nullptr && node_patch_boundary[node] != 0U;
    const std::uint8_t flags = node_flags != nullptr ? node_flags[node] : 0U;
    const bool constrained_patch_node =
        (flags & (tenryu::mesh::NODE_CENTER |
                  tenryu::mesh::NODE_AXIS |
                  tenryu::mesh::NODE_POLE_AXIS)) != 0U ||
        (params.constrain_cap_apex && node == params.cap_apex_node_id);
    if (!active && !boundary && !constrained_patch_node) {
      return false;
    }
  }
  return true;
}

__host__ __device__ inline int moved_corner_for_incident(
    const int p,
    const int n,
    const int cell,
    const int nverts,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices) {
  if (node_cell_csr_corners != nullptr) {
    return node_cell_csr_corners[p];
  }
  const int begin = cell_node_csr_offsets[cell];
  for (int k = 0; k < nverts; ++k) {
    if (cell_node_csr_indices[begin + k] == n) {
      return k;
    }
  }
  return -1;
}

__host__ __device__ inline void load_cell_coords(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int cell,
    const int nverts,
    const int moved_node,
    const double trial_r,
    const double trial_z,
    double* r,
    double* z) {
  const int begin = cell_node_csr_offsets[cell];
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_node_csr_indices[begin + k];
    r[k] = node == moved_node ? trial_r : x_r[node];
    z[k] = node == moved_node ? trial_z : x_z[node];
  }
}

__host__ __device__ inline double reference_length_scale2(
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n) {
  const bool has_reference =
      x_r_reference != nullptr && x_z_reference != nullptr;
  const double* r_ref = has_reference ? x_r_reference : x_r_lagrangian;
  const double* z_ref = has_reference ? x_z_reference : x_z_lagrangian;
  const int incident_begin = node_cell_csr_offsets[n];
  const int incident_end = node_cell_csr_offsets[n + 1];
  double area_sum = 0.0;
  double edge2_sum = 0.0;
  int area_count = 0;
  int edge_count = 0;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int cell = node_cell_csr_indices[p];
    const int nverts = active_nverts(cell_nverts, cell);
    const int begin = cell_node_csr_offsets[cell];
    if (cell_node_csr_offsets[cell + 1] - begin !=
        tenryu::mesh::kMeshTopoCellStorageSlots) {
      continue;
    }
    double rr[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    double zz[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    bool ok = true;
    for (int k = 0; k < nverts; ++k) {
      const int node = cell_node_csr_indices[begin + k];
      rr[k] = r_ref[node];
      zz[k] = z_ref[node];
      ok = ok && finite(rr[k]) && finite(zz[k]);
    }
    if (!ok) {
      continue;
    }
    const double area = 0.5 * fabs(polygon_area2(rr, zz, nverts));
    if (finite(area) && area > 0.0) {
      area_sum += area;
      ++area_count;
    }
    for (int k = 0; k < nverts; ++k) {
      const int kp = (k + 1 == nverts) ? 0 : k + 1;
      const double edge2 = sqr(rr[kp] - rr[k]) + sqr(zz[kp] - zz[k]);
      if (finite(edge2) && edge2 > 0.0) {
        edge2_sum += edge2;
        ++edge_count;
      }
    }
  }
  const double area_scale =
      area_count > 0 ? area_sum / static_cast<double>(area_count) : 0.0;
  const double edge_scale =
      edge_count > 0 ? edge2_sum / static_cast<double>(edge_count) : 0.0;
  return clamp_positive_floor(fmax(area_scale, edge_scale));
}

__host__ __device__ inline double barrier_floor(const double abs_floor,
                                                const double rel_floor,
                                                const double reference_abs) {
  const double rel = finite(reference_abs) && reference_abs > 0.0
                         ? fmax(rel_floor, 0.0) * reference_abs
                         : 0.0;
  return clamp_positive_floor(fmax(fmax(abs_floor, 0.0), rel));
}

__host__ __device__ inline double edge_preservation_phi(
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int cell,
    const int moved_corner,
    const int nverts,
    const double trial_r,
    const double trial_z,
    const double h2) {
  const int begin = cell_node_csr_offsets[cell];
  double phi = 0.0;
  const int adjacent[2] = {
      (moved_corner + nverts - 1) % nverts,
      (moved_corner + 1) % nverts,
  };
  for (int q = 0; q < 2; ++q) {
    const int other_corner = adjacent[q];
    const int other_node = cell_node_csr_indices[begin + other_corner];
    const double other_r = x_r_lagrangian[other_node];
    const double other_z = x_z_lagrangian[other_node];
    const double trial_len =
        sqrt(sqr(trial_r - other_r) + sqr(trial_z - other_z));
    const int moved_node = cell_node_csr_indices[begin + moved_corner];
    const double lag_len =
        sqrt(sqr(x_r_lagrangian[moved_node] - other_r) +
             sqr(x_z_lagrangian[moved_node] - other_z));
    if (!finite(trial_len) || !finite(lag_len)) {
      return kHuge;
    }
    phi += sqr(trial_len - lag_len) / h2;
  }
  return phi;
}

__host__ __device__ inline double node_phi(
    const double* __restrict__ x_r_current,
    const double* __restrict__ x_z_current,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const int n,
    const double trial_r_in,
    const double trial_z_in,
    const CenterPatchBarrierParams& params,
    bool* admissible_out,
    double* h2_out) {
  double trial_r = trial_r_in;
  double trial_z = trial_z_in;
  project_constraints(trial_r,
                      trial_z,
                      node_flags,
                      x_r_lagrangian,
                      x_r_reference,
                      n,
                      params);
  const double h2 = reference_length_scale2(x_r_reference,
                                            x_z_reference,
                                            x_r_lagrangian,
                                            x_z_lagrangian,
                                            node_cell_csr_offsets,
                                            node_cell_csr_indices,
                                            cell_node_csr_offsets,
                                            cell_node_csr_indices,
                                            cell_nverts,
                                            n);
  if (h2_out != nullptr) {
    *h2_out = h2;
  }
  if (!finite(trial_r) || !finite(trial_z)) {
    if (admissible_out != nullptr) {
      *admissible_out = false;
    }
    return kHuge;
  }

  double phi =
      params.weight_lagrangian *
      (sqr(trial_r - x_r_lagrangian[n]) +
       sqr(trial_z - x_z_lagrangian[n])) /
      h2;
  const int incident_begin = node_cell_csr_offsets[n];
  const int incident_end = node_cell_csr_offsets[n + 1];
  const bool has_reference =
      x_r_reference != nullptr && x_z_reference != nullptr;
  const double* r_ref = has_reference ? x_r_reference : x_r_lagrangian;
  const double* z_ref = has_reference ? x_z_reference : x_z_lagrangian;

  bool admissible = true;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int cell = node_cell_csr_indices[p];
    if (!incident_cell_contains_only_patch_nodes(cell,
                                                cell_node_csr_offsets,
                                                cell_node_csr_indices,
                                                cell_nverts,
                                                node_flags,
                                                node_rezone_active,
                                                node_patch_boundary,
                                                params)) {
      admissible = false;
      continue;
    }
    const int nverts = active_nverts(cell_nverts, cell);
    const int cell_begin = cell_node_csr_offsets[cell];
    const int cell_end = cell_node_csr_offsets[cell + 1];
    if (cell_end - cell_begin != tenryu::mesh::kMeshTopoCellStorageSlots) {
      admissible = false;
      continue;
    }
    const int moved_corner =
        moved_corner_for_incident(p,
                                  n,
                                  cell,
                                  nverts,
                                  node_cell_csr_corners,
                                  cell_node_csr_offsets,
                                  cell_node_csr_indices);
    if (moved_corner < 0 || moved_corner >= nverts) {
      admissible = false;
      continue;
    }

    double r_trial[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    double z_trial[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    double r_ref_cell[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    double z_ref_cell[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    load_cell_coords(x_r_current,
                     x_z_current,
                     cell_node_csr_offsets,
                     cell_node_csr_indices,
                     cell,
                     nverts,
                     n,
                     trial_r,
                     trial_z,
                     r_trial,
                     z_trial);
    bool finite_nodes = true;
    for (int k = 0; k < nverts; ++k) {
      const int node = cell_node_csr_indices[cell_begin + k];
      r_ref_cell[k] = r_ref[node];
      z_ref_cell[k] = z_ref[node];
      finite_nodes = finite_nodes && finite(r_trial[k]) &&
                     finite(z_trial[k]) && finite(r_ref_cell[k]) &&
                     finite(z_ref_cell[k]);
    }
    if (!finite_nodes) {
      admissible = false;
      continue;
    }

    if (params.weight_edge != 0.0) {
      const double edge_phi =
          edge_preservation_phi(x_r_lagrangian,
                                x_z_lagrangian,
                                cell_node_csr_offsets,
                                cell_node_csr_indices,
                                cell,
                                moved_corner,
                                nverts,
                                trial_r,
                                trial_z,
                                h2);
      if (!finite(edge_phi)) {
        admissible = false;
      } else {
        phi += params.weight_edge * edge_phi;
      }
    }

    const double orient = orientation_sign(cell_orientation_sign, cell);
    if (params.weight_volume != 0.0) {
      const double v_ref =
          fabs(rz_volume_oriented(r_ref_cell, z_ref_cell, nverts, orient));
      const double v_floor = barrier_floor(params.volume_floor_abs,
                                           params.volume_floor_rel,
                                           v_ref);
      const double v_trial =
          rz_volume_oriented(r_trial, z_trial, nverts, orient);
      if (!finite(v_trial) || !(v_trial > v_floor)) {
        admissible = false;
      } else {
        phi += params.weight_volume * (-log(v_trial / v_floor));
      }
    }

    if (params.weight_jacobian != 0.0) {
      for (int corner = 0; corner < nverts; ++corner) {
        const double j_ref = fabs(corner_j_oriented(
            r_ref_cell, z_ref_cell, corner, nverts, orient));
        const double j_floor = barrier_floor(params.jacobian_floor_abs,
                                             params.jacobian_floor_rel,
                                             j_ref);
        const double j_trial =
            corner_j_oriented(r_trial, z_trial, corner, nverts, orient);
        if (!finite(j_trial) || !(j_trial > j_floor)) {
          admissible = false;
        } else {
          phi += params.weight_jacobian * (-log(j_trial / j_floor));
        }
      }
    }
  }

  if (admissible_out != nullptr) {
    *admissible_out = admissible && finite(phi);
  }
  return admissible && finite(phi) ? phi : kHuge;
}

struct NodeStepResult {
  bool active = false;
  bool converged = false;
  bool invalid = false;
  int accepted_steps = 0;
  int rejected_steps = 0;
  double final_phi = 0.0;
  double max_step = 0.0;
  double max_grad_norm = 0.0;
  double scale = 1.0;
};

__host__ __device__ inline double finite_difference_step(
    const double x,
    const double scale,
    const CenterPatchBarrierParams& params) {
  const double rel = fmax(params.finite_difference_rel, 1.0e-12);
  const double coord = fmax(fabs(x), scale);
  return clamp_positive_floor(rel * coord);
}

__host__ __device__ inline bool solve_newton_delta(
    const double grad_r,
    const double grad_z,
    const double h_rr,
    const double h_rz,
    const double h_zz,
    const bool axis,
    const double scale,
    double& dr,
    double& dz) {
  dr = 0.0;
  dz = 0.0;
  if (axis) {
    if (finite(h_zz) && h_zz > 0.0) {
      dz = -grad_z / h_zz;
    } else {
      dz = -grad_z * scale;
    }
    return finite(dz);
  }
  const double det = h_rr * h_zz - h_rz * h_rz;
  if (finite(det) && det > 1.0e-30 &&
      finite(h_rr) && finite(h_rz) && finite(h_zz)) {
    dr = (-grad_r * h_zz + grad_z * h_rz) / det;
    dz = (grad_r * h_rz - grad_z * h_rr) / det;
    return finite(dr) && finite(dz);
  }
  const double grad_norm = sqrt(grad_r * grad_r + grad_z * grad_z);
  if (!finite(grad_norm) || !(grad_norm > 0.0)) {
    return false;
  }
  dr = -scale * grad_r / grad_norm;
  dz = -scale * grad_z / grad_norm;
  return finite(dr) && finite(dz);
}

__host__ __device__ inline NodeStepResult optimize_node(
    double* __restrict__ x_r_current,
    double* __restrict__ x_z_current,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const int n,
    const CenterPatchBarrierParams& params) {
  NodeStepResult out;
  if (!node_active(node_rezone_active, node_patch_boundary, n) ||
      node_is_apex(node_flags, n, params)) {
    x_r_current[n] = node_is_apex(node_flags, n, params)
                         ? 0.0
                         : x_r_lagrangian[n];
    x_z_current[n] = node_is_apex(node_flags, n, params)
                         ? 0.0
                         : x_z_lagrangian[n];
    out.converged = true;
    return out;
  }

  out.active = true;
  double r = x_r_current[n];
  double z = x_z_current[n];
  project_constraints(r,
                      z,
                      node_flags,
                      x_r_lagrangian,
                      x_r_reference,
                      n,
                      params);
  x_r_current[n] = r;
  x_z_current[n] = z;

  bool admissible = false;
  double h2 = 1.0;
  double phi0 = node_phi(x_r_current,
                         x_z_current,
                         x_r_lagrangian,
                         x_z_lagrangian,
                         x_r_reference,
                         x_z_reference,
                         node_cell_csr_offsets,
                         node_cell_csr_indices,
                         node_cell_csr_corners,
                         cell_node_csr_offsets,
                         cell_node_csr_indices,
                         cell_orientation_sign,
                         cell_nverts,
                         node_flags,
                         node_rezone_active,
                         node_patch_boundary,
                         n,
                         r,
                         z,
                         params,
                         &admissible,
                         &h2);
  out.scale = sqrt(h2);
  if (!admissible) {
    out.invalid = true;
    out.rejected_steps = 1;
    out.final_phi = phi0;
    return out;
  }

  const bool axis =
      node_is_axis(node_flags, x_r_lagrangian, x_r_reference, n, params);
  const int iter_limit = params.max_newton_iters > 0
                             ? params.max_newton_iters
                             : 0;
  for (int iter = 0; iter < iter_limit; ++iter) {
    const double hr = finite_difference_step(r, out.scale, params);
    const double hz = finite_difference_step(z, out.scale, params);

    bool ok_rp = false;
    bool ok_rm = false;
    bool ok_zp = false;
    bool ok_zm = false;
    bool ok_pp = false;
    bool ok_pm = false;
    bool ok_mp = false;
    bool ok_mm = false;
    const double phi_rp =
        axis ? phi0
             : node_phi(x_r_current,
                        x_z_current,
                        x_r_lagrangian,
                        x_z_lagrangian,
                        x_r_reference,
                        x_z_reference,
                        node_cell_csr_offsets,
                        node_cell_csr_indices,
                        node_cell_csr_corners,
                        cell_node_csr_offsets,
                        cell_node_csr_indices,
                        cell_orientation_sign,
                        cell_nverts,
                        node_flags,
                        node_rezone_active,
                        node_patch_boundary,
                        n,
                        r + hr,
                        z,
                        params,
                        &ok_rp,
                        nullptr);
    const double phi_rm =
        axis ? phi0
             : node_phi(x_r_current,
                        x_z_current,
                        x_r_lagrangian,
                        x_z_lagrangian,
                        x_r_reference,
                        x_z_reference,
                        node_cell_csr_offsets,
                        node_cell_csr_indices,
                        node_cell_csr_corners,
                        cell_node_csr_offsets,
                        cell_node_csr_indices,
                        cell_orientation_sign,
                        cell_nverts,
                        node_flags,
                        node_rezone_active,
                        node_patch_boundary,
                        n,
                        r - hr,
                        z,
                        params,
                        &ok_rm,
                        nullptr);
    const double phi_zp = node_phi(x_r_current,
                                   x_z_current,
                                   x_r_lagrangian,
                                   x_z_lagrangian,
                                   x_r_reference,
                                   x_z_reference,
                                   node_cell_csr_offsets,
                                   node_cell_csr_indices,
                                   node_cell_csr_corners,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_orientation_sign,
                                   cell_nverts,
                                   node_flags,
                                   node_rezone_active,
                                   node_patch_boundary,
                                   n,
                                   r,
                                   z + hz,
                                   params,
                                   &ok_zp,
                                   nullptr);
    const double phi_zm = node_phi(x_r_current,
                                   x_z_current,
                                   x_r_lagrangian,
                                   x_z_lagrangian,
                                   x_r_reference,
                                   x_z_reference,
                                   node_cell_csr_offsets,
                                   node_cell_csr_indices,
                                   node_cell_csr_corners,
                                   cell_node_csr_offsets,
                                   cell_node_csr_indices,
                                   cell_orientation_sign,
                                   cell_nverts,
                                   node_flags,
                                   node_rezone_active,
                                   node_patch_boundary,
                                   n,
                                   r,
                                   z - hz,
                                   params,
                                   &ok_zm,
                                   nullptr);
    if ((!axis && (!ok_rp || !ok_rm)) || !ok_zp || !ok_zm) {
      out.rejected_steps += 1;
      break;
    }

    const double grad_r = axis ? 0.0 : (phi_rp - phi_rm) / (2.0 * hr);
    const double grad_z = (phi_zp - phi_zm) / (2.0 * hz);
    const double grad_norm = sqrt(grad_r * grad_r + grad_z * grad_z);
    out.max_grad_norm = fmax(out.max_grad_norm, grad_norm);
    if (!finite(grad_norm) ||
        grad_norm <= params.convergence_tol * fmax(1.0, fabs(phi0)) /
                         out.scale) {
      out.converged = true;
      break;
    }

    const double h_rr =
        axis ? 1.0 : (phi_rp - 2.0 * phi0 + phi_rm) / (hr * hr);
    const double h_zz = (phi_zp - 2.0 * phi0 + phi_zm) / (hz * hz);
    double h_rz = 0.0;
    if (!axis) {
      const double phi_pp = node_phi(x_r_current,
                                     x_z_current,
                                     x_r_lagrangian,
                                     x_z_lagrangian,
                                     x_r_reference,
                                     x_z_reference,
                                     node_cell_csr_offsets,
                                     node_cell_csr_indices,
                                     node_cell_csr_corners,
                                     cell_node_csr_offsets,
                                     cell_node_csr_indices,
                                     cell_orientation_sign,
                                     cell_nverts,
                                     node_flags,
                                     node_rezone_active,
                                     node_patch_boundary,
                                     n,
                                     r + hr,
                                     z + hz,
                                     params,
                                     &ok_pp,
                                     nullptr);
      const double phi_pm = node_phi(x_r_current,
                                     x_z_current,
                                     x_r_lagrangian,
                                     x_z_lagrangian,
                                     x_r_reference,
                                     x_z_reference,
                                     node_cell_csr_offsets,
                                     node_cell_csr_indices,
                                     node_cell_csr_corners,
                                     cell_node_csr_offsets,
                                     cell_node_csr_indices,
                                     cell_orientation_sign,
                                     cell_nverts,
                                     node_flags,
                                     node_rezone_active,
                                     node_patch_boundary,
                                     n,
                                     r + hr,
                                     z - hz,
                                     params,
                                     &ok_pm,
                                     nullptr);
      const double phi_mp = node_phi(x_r_current,
                                     x_z_current,
                                     x_r_lagrangian,
                                     x_z_lagrangian,
                                     x_r_reference,
                                     x_z_reference,
                                     node_cell_csr_offsets,
                                     node_cell_csr_indices,
                                     node_cell_csr_corners,
                                     cell_node_csr_offsets,
                                     cell_node_csr_indices,
                                     cell_orientation_sign,
                                     cell_nverts,
                                     node_flags,
                                     node_rezone_active,
                                     node_patch_boundary,
                                     n,
                                     r - hr,
                                     z + hz,
                                     params,
                                     &ok_mp,
                                     nullptr);
      const double phi_mm = node_phi(x_r_current,
                                     x_z_current,
                                     x_r_lagrangian,
                                     x_z_lagrangian,
                                     x_r_reference,
                                     x_z_reference,
                                     node_cell_csr_offsets,
                                     node_cell_csr_indices,
                                     node_cell_csr_corners,
                                     cell_node_csr_offsets,
                                     cell_node_csr_indices,
                                     cell_orientation_sign,
                                     cell_nverts,
                                     node_flags,
                                     node_rezone_active,
                                     node_patch_boundary,
                                     n,
                                     r - hr,
                                     z - hz,
                                     params,
                                     &ok_mm,
                                     nullptr);
      if (ok_pp && ok_pm && ok_mp && ok_mm) {
        h_rz = (phi_pp - phi_pm - phi_mp + phi_mm) / (4.0 * hr * hz);
      }
    }

    double dr = 0.0;
    double dz = 0.0;
    if (!solve_newton_delta(grad_r,
                            grad_z,
                            h_rr,
                            h_rz,
                            h_zz,
                            axis,
                            out.scale,
                            dr,
                            dz)) {
      out.rejected_steps += 1;
      break;
    }
    if (axis) {
      dr = 0.0;
    }
    const double step_norm_raw = sqrt(dr * dr + dz * dz);
    const double coord_scale =
        clamp_positive_floor(fmax(out.scale, sqrt(r * r + z * z)));
    if (!finite(step_norm_raw) ||
        step_norm_raw <=
            fmax(params.min_step_rel, 0.0) * coord_scale) {
      out.rejected_steps += 1;
      break;
    }
    const double max_step =
        fmax(params.max_step_fraction, 1.0e-6) * out.scale;
    if (step_norm_raw > max_step) {
      const double scale = max_step / step_norm_raw;
      dr *= scale;
      dz *= scale;
    }

    const double grad_dot_delta = grad_r * dr + grad_z * dz;
    bool accepted = false;
    double accepted_r = r;
    double accepted_z = z;
    double accepted_phi = phi0;
    double lambda = 1.0;
    const int backtrack_limit =
        params.max_backtrack_iters > 0 ? params.max_backtrack_iters : 0;
    for (int bt = 0; bt <= backtrack_limit; ++bt) {
      double trial_r = r + lambda * dr;
      double trial_z = z + lambda * dz;
      project_constraints(trial_r,
                          trial_z,
                          node_flags,
                          x_r_lagrangian,
                          x_r_reference,
                          n,
                          params);
      bool ok_trial = false;
      const double trial_phi = node_phi(x_r_current,
                                        x_z_current,
                                        x_r_lagrangian,
                                        x_z_lagrangian,
                                        x_r_reference,
                                        x_z_reference,
                                        node_cell_csr_offsets,
                                        node_cell_csr_indices,
                                        node_cell_csr_corners,
                                        cell_node_csr_offsets,
                                        cell_node_csr_indices,
                                        cell_orientation_sign,
                                        cell_nverts,
                                        node_flags,
                                        node_rezone_active,
                                        node_patch_boundary,
                                        n,
                                        trial_r,
                                        trial_z,
                                        params,
                                        &ok_trial,
                                        nullptr);
      const double sufficient =
          phi0 + fmax(params.armijo_c1, 0.0) * lambda * grad_dot_delta;
      if (ok_trial && finite(trial_phi) &&
          (trial_phi < phi0 || trial_phi <= sufficient)) {
        accepted = true;
        accepted_r = trial_r;
        accepted_z = trial_z;
        accepted_phi = trial_phi;
        break;
      }
      lambda *= 0.5;
    }

    if (!accepted) {
      out.rejected_steps += 1;
      break;
    }
    const double step_norm =
        sqrt(sqr(accepted_r - r) + sqr(accepted_z - z));
    r = accepted_r;
    z = accepted_z;
    phi0 = accepted_phi;
    x_r_current[n] = r;
    x_z_current[n] = z;
    out.max_step = fmax(out.max_step, step_norm);
    out.accepted_steps += 1;
    if (step_norm <= params.convergence_tol * out.scale) {
      out.converged = true;
      break;
    }
  }

  out.final_phi = phi0;
  return out;
}

#ifdef __CUDACC__
__global__ void initialize_optimizer_coordinates_kernel(
    double* __restrict__ x_r_out,
    double* __restrict__ x_z_out,
    const double* __restrict__ x_r_seed,
    const double* __restrict__ x_z_seed,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const double* __restrict__ x_r_reference,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const int n_nodes,
    const CenterPatchBarrierParams params) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  const bool active =
      node_active(node_rezone_active, node_patch_boundary, n);
  double r = active && x_r_seed != nullptr ? x_r_seed[n] : x_r_lagrangian[n];
  double z = active && x_z_seed != nullptr ? x_z_seed[n] : x_z_lagrangian[n];
  project_constraints(r,
                      z,
                      node_flags,
                      x_r_lagrangian,
                      x_r_reference,
                      n,
                      params);
  x_r_out[n] = r;
  x_z_out[n] = z;
}

__global__ void optimize_serial_kernel(
    double* __restrict__ x_r_current,
    double* __restrict__ x_z_current,
    const double* __restrict__ x_r_lagrangian,
    const double* __restrict__ x_z_lagrangian,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_patch_boundary,
    const int n_nodes,
    const CenterPatchBarrierParams params,
    CenterPatchBarrierResult* __restrict__ result) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  CenterPatchBarrierResult out{};
  int active_count = 0;
  for (int n = 0; n < n_nodes; ++n) {
    if (node_active(node_rezone_active, node_patch_boundary, n)) {
      ++active_count;
    }
  }
  out.active_nodes = active_count;
  if (active_count == 0) {
    out.converged = true;
    *result = out;
    return;
  }

  const int sweep_limit = params.max_sweeps > 0 ? params.max_sweeps : 0;
  for (int sweep = 0; sweep < sweep_limit; ++sweep) {
    bool sweep_converged = true;
    double sweep_max_step = 0.0;
    double sweep_max_grad = 0.0;
    double sweep_max_scale = 1.0;
    double sweep_phi = 0.0;
    bool sweep_invalid = false;
    for (int n = 0; n < n_nodes; ++n) {
      const NodeStepResult step =
          optimize_node(x_r_current,
                        x_z_current,
                        x_r_lagrangian,
                        x_z_lagrangian,
                        x_r_reference,
                        x_z_reference,
                        node_cell_csr_offsets,
                        node_cell_csr_indices,
                        node_cell_csr_corners,
                        cell_node_csr_offsets,
                        cell_node_csr_indices,
                        cell_orientation_sign,
                        cell_nverts,
                        node_flags,
                        node_rezone_active,
                        node_patch_boundary,
                        n,
                        params);
      if (!step.active) {
        continue;
      }
      sweep_converged = sweep_converged && step.converged && !step.invalid;
      sweep_max_step = fmax(sweep_max_step, step.max_step);
      sweep_max_grad = fmax(sweep_max_grad, step.max_grad_norm);
      sweep_max_scale = fmax(sweep_max_scale, step.scale);
      sweep_phi += finite(step.final_phi) ? step.final_phi : kHuge;
      sweep_invalid = sweep_invalid || step.invalid;
      out.accepted_steps += step.accepted_steps;
      out.rejected_steps += step.rejected_steps;
      out.invalid_nodes += step.invalid ? 1 : 0;
    }
    out.sweeps = sweep + 1;
    out.max_step = sweep_max_step;
    out.max_grad_norm = sweep_max_grad;
    out.final_phi = sweep_phi;
    if (!sweep_invalid &&
        sweep_max_step <= params.convergence_tol * sweep_max_scale) {
      sweep_converged = true;
    }
    if (sweep_converged) {
      out.converged = true;
      break;
    }
  }
  // Revert post-pass (fixed point): any active node whose FINAL position is
  // inadmissible reverts to its Lagrangian coordinates. Reverting a node can
  // change a neighbor's admissibility, so iterate; the moved-node set only
  // shrinks (worst case all nodes revert = identity rezone, which is safe).
  for (int pass = 0; pass < 32; ++pass) {
    int reverted = 0;
    for (int n = 0; n < n_nodes; ++n) {
      if (!node_active(node_rezone_active, node_patch_boundary, n) ||
          node_is_apex(node_flags, n, params)) {
        continue;
      }
      const double r = x_r_current[n];
      const double z = x_z_current[n];
      if (r == x_r_lagrangian[n] && z == x_z_lagrangian[n]) {
        continue;
      }
      bool admissible = false;
      double h2 = 1.0;
      (void)node_phi(x_r_current,
                     x_z_current,
                     x_r_lagrangian,
                     x_z_lagrangian,
                     x_r_reference,
                     x_z_reference,
                     node_cell_csr_offsets,
                     node_cell_csr_indices,
                     node_cell_csr_corners,
                     cell_node_csr_offsets,
                     cell_node_csr_indices,
                     cell_orientation_sign,
                     cell_nverts,
                     node_flags,
                     node_rezone_active,
                     node_patch_boundary,
                     n,
                     r,
                     z,
                     params,
                     &admissible,
                     &h2);
      if (!admissible) {
        x_r_current[n] = x_r_lagrangian[n];
        x_z_current[n] = x_z_lagrangian[n];
        ++reverted;
      }
    }
    out.reverted_nodes += reverted;
    if (reverted == 0) {
      break;
    }
  }
  for (int n = 0; n < n_nodes; ++n) {
    if (!node_active(node_rezone_active, node_patch_boundary, n) ||
        node_is_apex(node_flags, n, params)) {
      continue;
    }
    bool admissible = false;
    double h2 = 1.0;
    (void)node_phi(x_r_current,
                   x_z_current,
                   x_r_lagrangian,
                   x_z_lagrangian,
                   x_r_reference,
                   x_z_reference,
                   node_cell_csr_offsets,
                   node_cell_csr_indices,
                   node_cell_csr_corners,
                   cell_node_csr_offsets,
                   cell_node_csr_indices,
                   cell_orientation_sign,
                   cell_nverts,
                   node_flags,
                   node_rezone_active,
                   node_patch_boundary,
                   n,
                   x_r_current[n],
                   x_z_current[n],
                   params,
                   &admissible,
                   &h2);
    if (!admissible) {
      ++out.invalid_nodes_final;
    }
  }
  *result = out;
}

__global__ void count_excluded_active_nodes_kernel(
    int* count,
    const std::uint8_t* __restrict__ node_rezone_active,
    const std::uint8_t* __restrict__ node_rezone_excluded,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes || node_rezone_active == nullptr ||
      node_rezone_excluded == nullptr) {
    return;
  }
  if (node_rezone_excluded[n] != 0U && node_rezone_active[n] != 0U) {
    atomicAdd(count, 1);
  }
}
#endif

}  // namespace detail

#ifdef __CUDACC__
inline CenterPatchBarrierResult optimize_center_patch_phi_barrier(
    double* d_x_r_optimized,
    double* d_x_z_optimized,
    const double* d_x_r_seed,
    const double* d_x_z_seed,
    const double* d_x_r_lagrangian,
    const double* d_x_z_lagrangian,
    const double* d_x_r_reference,
    const double* d_x_z_reference,
    const int* d_node_cell_csr_offsets,
    const int* d_node_cell_csr_indices,
    const int* d_node_cell_csr_corners,
    const int* d_cell_node_csr_offsets,
    const int* d_cell_node_csr_indices,
    const int* d_cell_orientation_sign,
    const std::uint8_t* d_cell_nverts,
    const std::uint8_t* d_node_flags,
    const std::uint8_t* d_node_rezone_active,
    const std::uint8_t* d_node_patch_boundary,
    const int n_nodes,
    const CenterPatchBarrierParams& params,
    const std::uint8_t* d_node_rezone_excluded = nullptr,
    cudaStream_t stream = nullptr) {
  TENRYU_ASSERT(d_x_r_optimized != nullptr && d_x_z_optimized != nullptr,
                "center patch barrier requires output coordinate buffers");
  TENRYU_ASSERT(d_x_r_lagrangian != nullptr && d_x_z_lagrangian != nullptr,
                "center patch barrier requires Lagrangian coordinates");
  TENRYU_ASSERT(d_node_cell_csr_offsets != nullptr &&
                    d_node_cell_csr_indices != nullptr,
                "center patch barrier requires node-cell CSR");
  TENRYU_ASSERT(d_cell_node_csr_offsets != nullptr &&
                    d_cell_node_csr_indices != nullptr,
                "center patch barrier requires cell-node CSR");
  TENRYU_ASSERT(d_node_rezone_active != nullptr,
                "center patch barrier requires node_rezone_active mask");
  TENRYU_ASSERT(n_nodes >= 0,
                "center patch barrier requires non-negative node count");
  if (d_node_rezone_excluded != nullptr && n_nodes > 0) {
    int* d_excluded_active_count = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_excluded_active_count),
                          sizeof(int)));
    CUDA_CHECK(cudaMemsetAsync(d_excluded_active_count, 0, sizeof(int), stream));
    const int threads = 256;
    const int blocks = (n_nodes + threads - 1) / threads;
    detail::count_excluded_active_nodes_kernel<<<blocks, threads, 0, stream>>>(
        d_excluded_active_count,
        d_node_rezone_active,
        d_node_rezone_excluded,
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    int excluded_active_count = 0;
    if (stream == nullptr) {
      CUDA_CHECK(cudaMemcpy(&excluded_active_count,
                            d_excluded_active_count,
                            sizeof(int),
                            cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(&excluded_active_count,
                                 d_excluded_active_count,
                                 sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaFree(d_excluded_active_count));
    TENRYU_ASSERT(excluded_active_count == 0,
                  "center patch active mask includes excluded macro-core nodes");
  }

  CenterPatchBarrierResult* d_result = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_result),
                        sizeof(CenterPatchBarrierResult)));
  CUDA_CHECK(cudaMemsetAsync(d_result,
                             0,
                             sizeof(CenterPatchBarrierResult),
                             stream));
  const int threads = 256;
  const int blocks = (n_nodes + threads - 1) / threads;
  if (n_nodes > 0) {
    detail::initialize_optimizer_coordinates_kernel<<<blocks,
                                                       threads,
                                                       0,
                                                       stream>>>(
        d_x_r_optimized,
        d_x_z_optimized,
        d_x_r_seed,
        d_x_z_seed,
        d_x_r_lagrangian,
        d_x_z_lagrangian,
        d_x_r_reference,
        d_node_flags,
        d_node_rezone_active,
        d_node_patch_boundary,
        n_nodes,
        params);
    CUDA_CHECK(cudaGetLastError());
  }
  detail::optimize_serial_kernel<<<1, 1, 0, stream>>>(
      d_x_r_optimized,
      d_x_z_optimized,
      d_x_r_lagrangian,
      d_x_z_lagrangian,
      d_x_r_reference,
      d_x_z_reference,
      d_node_cell_csr_offsets,
      d_node_cell_csr_indices,
      d_node_cell_csr_corners,
      d_cell_node_csr_offsets,
      d_cell_node_csr_indices,
      d_cell_orientation_sign,
      d_cell_nverts,
      d_node_flags,
      d_node_rezone_active,
      d_node_patch_boundary,
      n_nodes,
      params,
      d_result);
  CUDA_CHECK(cudaGetLastError());

  CenterPatchBarrierResult result{};
  if (stream == nullptr) {
    CUDA_CHECK(cudaMemcpy(&result,
                          d_result,
                          sizeof(CenterPatchBarrierResult),
                          cudaMemcpyDeviceToHost));
  } else {
    CUDA_CHECK(cudaMemcpyAsync(&result,
                               d_result,
                               sizeof(CenterPatchBarrierResult),
                               cudaMemcpyDeviceToHost,
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  CUDA_CHECK(cudaFree(d_result));
  return result;
}
#endif

}  // namespace tenryu::hydro::ale::center_patch_barrier
