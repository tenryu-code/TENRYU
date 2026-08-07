#pragma once

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"

namespace tenryu::laser::laser_mesh_bodies {

constexpr double kProtonMass = tenryu::core::constants::proton_mass;

__device__ inline int locate_cell_1d_device(const double* __restrict__ edges,
                                            const int n_cells,
                                            const double r) {
  if (n_cells <= 0) {
    return 0;
  }
  if (r <= edges[0]) {
    return 0;
  }
  if (r >= edges[n_cells]) {
    return n_cells - 1;
  }

  int lo = 0;
  int hi = n_cells + 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (r < edges[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  const int idx = lo - 1;
  if (idx < 0) {
    return 0;
  }
  if (idx >= n_cells) {
    return n_cells - 1;
  }
  return idx;
}

__device__ inline double clamp_device(const double x, const double lo, const double hi) {
  return fmin(hi, fmax(lo, x));
}

__device__ inline void map_hydro_to_laser_1d_kernel_body(
    const int idx,
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ A_eff_cell,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ r_edges,
    double* __restrict__ n_hat_out,
    double* __restrict__ n_hat_raw_out,
    double* __restrict__ Te_out,
    double* __restrict__ Z_out,
    const int n_nodes_total,
    const int n_nodes_z,
    const int n_cells,
    const double n_crit_safe,
    const int use_ghost_corona,
    const int outer_surface_cell,
    const double ghost_ne_inner,
    const double ghost_scale_length,
    const double ghost_ne_min,
    const double r_surface_outer,
    const double r_ghost_outer,
    const double Te_anchor,
    const double zbar_anchor,
    const double ghost_zbar_min,
    const double ghost_zbar_max,
    const double ghost_Te_min_eV,
    const int critical_clip,
    const double n_hat_margin,
    const int fcrit_cell,
    const double r_crit_interp,
    const double ne_fcrit_center,
    const double r_fcrit_center) {
  const int i = idx / n_nodes_z;
  const int j = idx - i * n_nodes_z;
  const double R = node_R[i];
  const double Z = node_Z[j];
  const double r = sqrt(R * R + Z * Z);

  const int c = locate_cell_1d_device(r_edges, n_cells, r);
  const bool is_void = (cell_is_void[c] != 0U);
  const bool outside_surface = (outer_surface_cell >= 0 && r > r_surface_outer);
  const bool outer_hydro_clamp = outside_surface && !is_void;
  const double rho_c = fmax(0.0, rho[c]);
  double z_c = is_void ? 0.0 : fmax(0.0, zbar[c]);
  double Te_c = Te[c];
  double nh_raw = 0.0;

  if (!is_void && !outer_hydro_clamp) {
    const double A_eff = fmax(A_eff_cell[c], 1.0e-30);
    const double ne_cell_val = rho_c * z_c / (A_eff * kProtonMass);
    // Linear interpolation of n_e in r between adjacent hydro cell
    // centers (2026-07-31 audit): piecewise-constant sampling put a
    // slope*dr/2 staircase on nodal n_hat and a 0/double comb on the
    // segment gradient wherever laser nodes are finer than hydro cells,
    // scattering ray turning points. Linear-in-r interpolation of n_e
    // reproduces linear profiles exactly on any node layout. Te and
    // zbar outputs stay cell-sampled; only n_hat changes.
    double ne_interp = ne_cell_val;
    const double r_c = 0.5 * (r_edges[c] + r_edges[c + 1]);
    int c2 = (r < r_c) ? (c - 1) : (c + 1);
    if (c2 >= 0 && c2 < n_cells && c2 != c &&
        cell_is_void[c2] == 0U &&
        (outer_surface_cell < 0 || c2 <= outer_surface_cell)) {
      const double r_c2 = 0.5 * (r_edges[c2] + r_edges[c2 + 1]);
      const double dr_cc = r_c2 - r_c;
      if (fabs(dr_cc) > 1.0e-300) {
        const double A_eff2 = fmax(A_eff_cell[c2], 1.0e-30);
        const double ne2 =
            fmax(0.0, rho[c2]) * fmax(0.0, zbar[c2]) /
            (A_eff2 * kProtonMass);
        const double t = clamp_device((r - r_c) / dr_cc, 0.0, 1.0);
        ne_interp = ne_cell_val + t * (ne2 - ne_cell_val);
      }
    }
    nh_raw = fmax(0.0, ne_interp / n_crit_safe);

    if (c == fcrit_cell && r_crit_interp > 0.0 && nh_raw < 1.0 &&
        ne_fcrit_center > 0.0 && r < r_fcrit_center) {
      const double denom = fmax(r_fcrit_center - r_crit_interp, 1.0e-30);
      const double t = clamp_device((r - r_crit_interp) / denom, 0.0, 1.0);
      const double log_ne_center = log(fmax(ne_fcrit_center, 1.0e-30));
      nh_raw = exp(t * log_ne_center);
    }
  } else if ((use_ghost_corona != 0 && is_void && r >= r_surface_outer &&
              r <= r_ghost_outer) ||
             (outer_hydro_clamp && use_ghost_corona != 0 && r <= r_ghost_outer)) {
    const double dr = fmax(r - r_surface_outer, 0.0);
    nh_raw = ghost_ne_inner * exp(-dr / fmax(ghost_scale_length, 1.0e-30));
    nh_raw = clamp_device(nh_raw, ghost_ne_min, ghost_ne_inner);
    z_c = clamp_device(zbar_anchor, ghost_zbar_min, fmax(ghost_zbar_max, ghost_zbar_min));
    Te_c = Te_anchor;
  } else if (outer_hydro_clamp) {
    z_c = 0.0;
    Te_c = ghost_Te_min_eV;
  }

  double n_hat = fmin(1.0, fmax(0.0, nh_raw));
  if (critical_clip != 0) {
    n_hat = fmin(n_hat, n_hat_margin);
  }

  n_hat_out[idx] = n_hat;
  n_hat_raw_out[idx] = nh_raw;
  Te_out[idx] = Te_c;
  Z_out[idx] = z_c;
}

__device__ inline void ema_smooth_n_hat_kernel_body(
    const int idx,
    double* __restrict__ n_hat,
    const double* __restrict__ prev_n_hat,
    const int n_nodes_total) {
  constexpr double kNhatEmaAlpha = 0.05;
  const double cur = n_hat[idx];
  const double prev = prev_n_hat[idx];
  if (cur > 0.3 && prev > 0.3 && fabs(cur - prev) < 0.01) {
    n_hat[idx] = kNhatEmaAlpha * cur + (1.0 - kNhatEmaAlpha) * prev;
  }
}

__device__ inline void compute_gradient_kernel_body(
    const int idx,
    double* __restrict__ grad_R,
    double* __restrict__ grad_Z,
    const double* __restrict__ n_hat,
    const double* __restrict__ node_R,
    const double* __restrict__ node_Z,
    const int n_nodes_r,
    const int n_nodes_z) {
  const int i = idx / n_nodes_z;
  const int j = idx - i * n_nodes_z;

  auto node = [&](const int ii, const int jj) -> double {
    return n_hat[ii * n_nodes_z + jj];
  };

  double dndR = 0.0;
  if (i == 0) {
    dndR = 0.0;
  } else if (i == n_nodes_r - 1) {
    const double dR = node_R[i] - node_R[i - 1];
    dndR = (dR > 0.0) ? (node(i, j) - node(i - 1, j)) / dR : 0.0;
  } else {
    const double h_m = node_R[i] - node_R[i - 1];
    const double h_p = node_R[i + 1] - node_R[i];
    const double denom = h_m * h_p * (h_m + h_p);
    if (denom > 0.0) {
      dndR = (-h_p * h_p * node(i - 1, j) + (h_p * h_p - h_m * h_m) * node(i, j) +
              h_m * h_m * node(i + 1, j)) /
             denom;
    }
  }

  double dndZ = 0.0;
  if (j == 0) {
    const double dZ = node_Z[1] - node_Z[0];
    dndZ = (dZ > 0.0) ? (node(i, 1) - node(i, 0)) / dZ : 0.0;
  } else if (j == n_nodes_z - 1) {
    const double dZ = node_Z[j] - node_Z[j - 1];
    dndZ = (dZ > 0.0) ? (node(i, j) - node(i, j - 1)) / dZ : 0.0;
  } else {
    const double h_m = node_Z[j] - node_Z[j - 1];
    const double h_p = node_Z[j + 1] - node_Z[j];
    const double denom = h_m * h_p * (h_m + h_p);
    if (denom > 0.0) {
      dndZ = (-h_p * h_p * node(i, j - 1) + (h_p * h_p - h_m * h_m) * node(i, j) +
              h_m * h_m * node(i, j + 1)) /
             denom;
    }
  }

  if (!isfinite(dndR)) {
    dndR = 0.0;
  }
  if (!isfinite(dndZ)) {
    dndZ = 0.0;
  }

  grad_R[idx] = dndR;
  grad_Z[idx] = dndZ;
}

__device__ inline void compute_smooth_kappa_kernel_body(
    const int idx,
    double* __restrict__ smooth_kappa_factor,
    const double* __restrict__ n_hat,
    const double* __restrict__ T_e,
    const double* __restrict__ Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const int n_nodes_total) {
  smooth_kappa_factor[idx] = compute_kappa_smooth_factor(
      n_hat[idx], T_e[idx], Zbar[idx], lambda_cm, eps_n, coulomb_log_floor);
}

__device__ inline void compute_smooth_kappa_ext_kernel_body(
    const int idx,
    double* __restrict__ smooth_kappa_factor,
    const double* __restrict__ n_hat,
    const double* __restrict__ T_e,
    const double* __restrict__ Zbar,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const int n_nodes_total,
    const LaserPhysExtOptions opt) {
  if (idx >= n_nodes_total) {
    return;
  }
  smooth_kappa_factor[idx] = compute_kappa_smooth_factor_ext(
      opt, n_hat[idx], T_e[idx], Zbar[idx], lambda_cm, eps_n,
      coulomb_log_floor, /*I_wcm2=*/-1.0);
}

__device__ inline void extract_radial_profile_kernel_body(
    const int i,
    double* __restrict__ radial_node_r,
    double* __restrict__ radial_n_hat,
    double* __restrict__ radial_n_hat_raw,
    double* __restrict__ radial_smooth_kappa,
    const double* __restrict__ node_R,
    const double* __restrict__ n_hat,
    const double* __restrict__ n_hat_raw,
    const double* __restrict__ smooth_kappa_factor,
    const int n_nodes_r,
    const int n_nodes_z,
    const int j_center,
    const int copy_smooth) {
  const int n = i * n_nodes_z + j_center;
  radial_node_r[i] = node_R[i];
  radial_n_hat[i] = n_hat[n];
  radial_n_hat_raw[i] = n_hat_raw[n];
  radial_smooth_kappa[i] = (copy_smooth != 0 && smooth_kappa_factor != nullptr)
                               ? smooth_kappa_factor[n]
                               : 0.0;
}

__device__ inline void extract_radial_te_kernel_body(
    const int i,
    double* __restrict__ radial_T_e,
    const double* __restrict__ T_e,
    const int n_nodes_z,
    const int j_center,
    const int n_nodes_r) {
  if (i >= n_nodes_r) {
    return;
  }
  radial_T_e[i] = T_e[i * n_nodes_z + j_center];
}

__device__ inline void compute_radial_gradient_kernel_body(
    const int i,
    double* __restrict__ radial_dn_dr,
    const double* __restrict__ radial_node_r,
    const double* __restrict__ radial_n_hat,
    const int n_nodes_r) {
  double grad = 0.0;
  if (n_nodes_r > 1) {
    // Consistency contract: the force is the exact derivative of the
    // piecewise-linear radial_n_hat interpolant, so linear profiles are exact
    // on any node layout. The last node copies the final segment slope.
    const int segment_i = (i < n_nodes_r - 1) ? i : n_nodes_r - 2;
    const double dr =
        radial_node_r[segment_i + 1] - radial_node_r[segment_i];
    if (dr > 0.0) {
      grad =
          (radial_n_hat[segment_i + 1] - radial_n_hat[segment_i]) / dr;
    }
  }

  radial_dn_dr[i] = isfinite(grad) ? grad : 0.0;
}

}  // namespace tenryu::laser::laser_mesh_bodies
