// W-H: Braginskii plasma shear viscosity kernels (1D planar / cylindrical /
// spherical), ion + electron channels. See braginskii_viscosity.cuh,
// docs/design/wh_braginskii_viscosity_design.md and
// docs/design/electron_viscosity_1d_20260712.md for the derivations.

#include "hydro/braginskii_viscosity.cuh"
#include "hydro/braginskii_viscosity_device.cuh"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_scratch.hpp"
#include "core/state.hpp"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro::braginskii {

namespace {

void brag_cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " +
                             cudaGetErrorString(err));
  }
}

void brag_sync(const char* msg) {
  brag_cuda_check(cudaGetLastError(), msg);
  brag_cuda_check(cudaDeviceSynchronize(), msg);
}

// Trace-free rate-of-strain components per cell:
//   e = du/dr, h = u/r (cell average u-bar/r-bar), div = e + alpha*h,
//   W_rr = 2e - (2/3)div, transverse W_tt = 2h - (2/3)div (spherical twice;
//   cylindrical once plus the zero-strain axial W_zz = -(2/3)div; planar
//   twice with h = 0). pi = -eta_eff * W with eta_eff per SPECIES
//   (0 = ion, 1 = electron, 2 = both, fully separated if-constexpr branches
//   so SPECIES=0 keeps the W-H arithmetic source-identical); heating powers
//   (eta_i/2) W:W vol -> heat_rate and (eta_e/2) W:W vol -> heat_rate_e.
template <int SPECIES, bool kZMom>
__global__ void braginskii_cell_stress_1d_kernel(
    double* __restrict__ pi_rr,
    double* __restrict__ pi_tt,
    double* __restrict__ heat_rate,
    double* __restrict__ heat_rate_e,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int geom,
    const DeviceParams p,
    const double* __restrict__ zmom_r4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  double prr = 0.0;
  double ptt = 0.0;
  double hr = 0.0;
  double hr_e = 0.0;
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (active) {
    const double rl = x_r[i];
    const double rr = x_r[i + 1];
    const double dr = rr - rl;
    if (dr > 0.0) {
      const double e = (v_r[i + 1] - v_r[i]) / dr;
      double h = 0.0;
      if (geom != 2) {
        const double rsum = rl + rr;
        h = (rsum > 0.0) ? (v_r[i] + v_r[i + 1]) / rsum : 0.0;
      }
      const double alpha = (geom == 0) ? 2.0 : ((geom == 1) ? 1.0 : 0.0);
      const double div_u = e + alpha * h;
      const double w_rr = 2.0 * e - (2.0 / 3.0) * div_u;
      const double w_iso = -(2.0 / 3.0) * div_u;
      const double w_tt = (geom == 2) ? w_iso : (2.0 * h + w_iso);
      double wsq = w_rr * w_rr;
      if (geom == 0) {
        wsq += 2.0 * w_tt * w_tt;
      } else if (geom == 1) {
        wsq += w_tt * w_tt + w_iso * w_iso;
      } else {
        wsq += 2.0 * w_iso * w_iso;
      }
      if constexpr (SPECIES == 0) {
        const double ti_ev = (Ti != nullptr) ? Ti[i] : p.ti_floor_ev;
        const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
        const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
        double eta;
        if constexpr (kZMom) {
          eta = braginskii_eta_poise_ext(
              rho[i], ti_ev, a_eff, zb, zmom_r4[i], dr, p);
        } else {
          eta = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p);
        }
        prr = -eta * w_rr;
        ptt = -eta * w_tt;
        if (heat_rate != nullptr) {
          hr = 0.5 * eta * wsq * vol[i];
        }
      } else if constexpr (SPECIES == 1) {
        const double te_ev = (Te != nullptr) ? Te[i] : p.te_floor_ev;
        const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
        const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
        const double eta =
            braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p);
        prr = -eta * w_rr;
        ptt = -eta * w_tt;
        if (heat_rate_e != nullptr) {
          hr_e = 0.5 * eta * wsq * vol[i];
        }
      } else {
        const double ti_ev = (Ti != nullptr) ? Ti[i] : p.ti_floor_ev;
        const double te_ev = (Te != nullptr) ? Te[i] : p.te_floor_ev;
        const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
        const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
        double eta_i;
        double eta_e;
        if (p.model == 1) {
          // "constant" under species="both": eta_eff = eta_const invariantly,
          // split half/half so all three species settings share the same
          // momentum physics and differ only in heat routing.
          eta_i = 0.5 * p.eta_const;
          eta_e = 0.5 * p.eta_const;
        } else {
          if constexpr (kZMom) {
            eta_i = braginskii_eta_poise_ext(
                rho[i], ti_ev, a_eff, zb, zmom_r4[i], dr, p);
          } else {
            eta_i = braginskii_eta_poise(
                rho[i], ti_ev, a_eff, zb, dr, p);
          }
          eta_e = braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p);
        }
        const double eta = eta_i + eta_e;
        prr = -eta * w_rr;
        ptt = -eta * w_tt;
        if (heat_rate != nullptr) {
          hr = 0.5 * eta_i * wsq * vol[i];
        }
        if (heat_rate_e != nullptr) {
          hr_e = 0.5 * eta_e * wsq * vol[i];
        }
      }
    }
  }
  pi_rr[i] = prr;
  pi_tt[i] = ptt;
  if (heat_rate != nullptr) {
    heat_rate[i] = hr;
  }
  if (heat_rate_e != nullptr) {
    heat_rate_e[i] = hr_e;
  }
}

// (div pi)_r = d(pi_rr)/dr + alpha*(pi_rr - pi_tt)/r; the node force uses
// the area-weighted flux difference (same structure as the pressure force
// in compute_acceleration_1d_kernel) plus the hoop term integrated over the
// half-cell node volume. accel is ADDED to (call after the pressure/AV
// acceleration kernel, before the damping filters).
__global__ void braginskii_add_accel_1d_kernel(
    double* __restrict__ accel,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ pi_rr,
    const double* __restrict__ pi_tt,
    const std::uint8_t* __restrict__ node_active,
    const int n_cells,
    const int geom,
    const int skip_outer) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = n_cells + 1;
  if (j >= n_nodes) {
    return;
  }
  if (j == 0) {
    return;
  }
  if (node_active != nullptr && node_active[j] == 0u) {
    return;
  }

  const double alpha = (geom == 0) ? 2.0 : ((geom == 1) ? 1.0 : 0.0);
  const double rj = x_r[j];

  if (j < n_cells) {
    const double node_mass = 0.5 * (mass[j - 1] + mass[j]);
    if (!(node_mass > 0.0)) {
      return;
    }
    const double area = mesh::geometry_1d_face_area(geom, rj);
    double force = area * (pi_rr[j] - pi_rr[j - 1]);
    if (alpha != 0.0 && rj > 0.0) {
      const double v_node = 0.5 * (vol[j - 1] + vol[j]);
      const double pbar_rr = 0.5 * (pi_rr[j - 1] + pi_rr[j]);
      const double pbar_tt = 0.5 * (pi_tt[j - 1] + pi_tt[j]);
      force += v_node * alpha * (pbar_rr - pbar_tt) / rj;
    }
    accel[j] += -force / node_mass;
    return;
  }

  if (skip_outer != 0) {
    return;
  }
  const double node_mass = 0.5 * mass[n_cells - 1];
  if (!(node_mass > 0.0)) {
    return;
  }
  const double area = mesh::geometry_1d_face_area(geom, rj);
  double force = area * (0.0 - pi_rr[n_cells - 1]);
  if (alpha != 0.0 && rj > 0.0) {
    const double v_node = 0.5 * vol[n_cells - 1];
    const double pbar_rr = 0.5 * pi_rr[n_cells - 1];
    const double pbar_tt = 0.5 * pi_tt[n_cells - 1];
    force += v_node * alpha * (pbar_rr - pbar_tt) / rj;
  }
  accel[j] += -force / node_mass;
}

// Longitudinal diffusion stability: nu_L = (4/3) eta_eff / rho, dt <=
// dt_safety * dr^2 / (2 nu_L) = dt_safety * rho dr^2 / ((8/3) eta_eff),
// eta_eff composed per SPECIES (additive for "both": 1/dt = 1/dt_i + 1/dt_e).
template <int SPECIES, bool kZMom>
__global__ void braginskii_dt_1d_kernel(
    double* __restrict__ min_dt,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const DeviceParams p,
    const double dt_safety,
    const double* __restrict__ zmom_r4) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return;
  }
  const double dr = x_r[i + 1] - x_r[i];
  if (!(dr > 0.0)) {
    return;
  }
  double eta;
  if constexpr (SPECIES == 0) {
    const double ti_ev = (Ti != nullptr) ? Ti[i] : p.ti_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
    if constexpr (kZMom) {
      eta = braginskii_eta_poise_ext(
          rho[i], ti_ev, a_eff, zb, zmom_r4[i], dr, p);
    } else {
      eta = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p);
    }
  } else if constexpr (SPECIES == 1) {
    const double te_ev = (Te != nullptr) ? Te[i] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
    eta = braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p);
  } else {
    const double ti_ev = (Ti != nullptr) ? Ti[i] : p.ti_floor_ev;
    const double te_ev = (Te != nullptr) ? Te[i] : p.te_floor_ev;
    const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
    const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
    if (p.model == 1) {
      eta = p.eta_const;
    } else {
      if constexpr (kZMom) {
        eta = braginskii_eta_poise_ext(
                  rho[i], ti_ev, a_eff, zb, zmom_r4[i], dr, p) +
              braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p);
      } else {
        eta = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p) +
              braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p);
      }
    }
  }
  if (!(eta > 0.0)) {
    return;
  }
  const double dt = dt_safety * rho[i] * dr * dr / ((8.0 / 3.0) * eta);
  if (!(dt > 0.0)) {
    return;
  }
  atomic_min_double_brag(min_dt, dt);
}

// Exact Gershgorin row-sum of the assembled node-acceleration operator
// (AI kernel review 2026-07-26, k13 F-05/F-09). One thread per NODE j mirrors
// braginskii_add_accel_1d_kernel: accel_j = -F_j / M_j with
//   F_j = A_j (pi_rr,R - pi_rr,L) + Vn_j alpha (pbar_rr - pbar_tt) / r_j,
// linearized in the node velocities at frozen eta. Per adjacent cell c
// (dr = width, rsum = r_lo + r_hi; h-terms absent for planar or rsum <= 0):
//   d(pi_rr,c)/du_lo  = +eta [ (4/3)/dr + (2/3) alpha / rsum ]
//   d(pi_rr,c)/du_hi  = -eta [ (4/3)/dr - (2/3) alpha / rsum ]
//   d(pi_rr - pi_tt)c/du_lo = +2 eta (1/dr + 1/rsum)
//   d(pi_rr - pi_tt)c/du_hi = -2 eta (1/dr - 1/rsum)
// Uniform planar check: row sum = (16/3) eta / (rho dr^2), i.e. 2/lambda_G
// equals the scalar dt formula's stability limit exactly.
template <int SPECIES, bool kZMom>
__global__ void braginskii_gershgorin_1d_kernel(
    double* __restrict__ max_lambda,
    const double* __restrict__ x_r,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ mass,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ node_active,
    const int n_cells,
    const int geom,
    const int skip_outer,
    const DeviceParams p,
    const double* __restrict__ zmom_r4) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = n_cells + 1;
  if (j >= n_nodes || j == 0) {
    return;
  }
  if (node_active != nullptr && node_active[j] == 0u) {
    return;
  }
  if (j == n_cells && skip_outer != 0) {
    return;
  }

  const double alpha = (geom == 0) ? 2.0 : ((geom == 1) ? 1.0 : 0.0);
  const double rj = x_r[j];
  const double area = mesh::geometry_1d_face_area(geom, rj);

  // Per-cell linearization coefficients; zeroed for inactive/degenerate cells
  // (their stress is identically zero in the stress kernel).
  const auto cell_coeffs = [&](const int c, double& dpi_lo, double& dpi_hi,
                               double& dpd_lo, double& dpd_hi) {
    dpi_lo = dpi_hi = dpd_lo = dpd_hi = 0.0;
    if (c < 0 || c >= n_cells) {
      return;
    }
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      return;
    }
    const double rl = x_r[c];
    const double rr = x_r[c + 1];
    const double dr = rr - rl;
    if (!(dr > 0.0)) {
      return;
    }
    double eta;
    if constexpr (SPECIES == 0) {
      const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
      const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
      const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
      if constexpr (kZMom) {
        eta = braginskii_eta_poise_ext(
            rho[c], ti_ev, a_eff, zb, zmom_r4[c], dr, p);
      } else {
        eta = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, dr, p);
      }
    } else if constexpr (SPECIES == 1) {
      const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
      const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
      const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
      eta = braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, dr, p);
    } else {
      if (p.model == 1) {
        eta = p.eta_const;
      } else {
        const double ti_ev = (Ti != nullptr) ? Ti[c] : p.ti_floor_ev;
        const double te_ev = (Te != nullptr) ? Te[c] : p.te_floor_ev;
        const double a_eff = (A_eff != nullptr) ? A_eff[c] : 1.0;
        const double zb = (zbar != nullptr) ? zbar[c] : 1.0;
        if constexpr (kZMom) {
          eta = braginskii_eta_poise_ext(
                    rho[c], ti_ev, a_eff, zb, zmom_r4[c], dr, p) +
                braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, dr, p);
        } else {
          eta = braginskii_eta_poise(rho[c], ti_ev, a_eff, zb, dr, p) +
                braginskii_eta_e_poise(rho[c], te_ev, a_eff, zb, dr, p);
        }
      }
    }
    if (!(eta > 0.0)) {
      return;
    }
    const double rsum = rl + rr;
    const double inv_dr = 1.0 / dr;
    const double inv_rsum = (geom != 2 && rsum > 0.0) ? (1.0 / rsum) : 0.0;
    dpi_lo = eta * ((4.0 / 3.0) * inv_dr + (2.0 / 3.0) * alpha * inv_rsum);
    dpi_hi = -eta * ((4.0 / 3.0) * inv_dr - (2.0 / 3.0) * alpha * inv_rsum);
    dpd_lo = 2.0 * eta * (inv_dr + inv_rsum);
    dpd_hi = -2.0 * eta * (inv_dr - inv_rsum);
  };

  double dpiL_lo, dpiL_hi, dpdL_lo, dpdL_hi;
  double dpiR_lo, dpiR_hi, dpdR_lo, dpdR_hi;
  cell_coeffs(j - 1, dpiL_lo, dpiL_hi, dpdL_lo, dpdL_hi);
  cell_coeffs((j < n_cells) ? j : -1, dpiR_lo, dpiR_hi, dpdR_lo, dpdR_hi);

  double node_mass;
  double vn;
  if (j < n_cells) {
    node_mass = 0.5 * (mass[j - 1] + mass[j]);
    vn = 0.5 * (vol[j - 1] + vol[j]);
  } else {
    node_mass = 0.5 * mass[n_cells - 1];
    vn = 0.5 * vol[n_cells - 1];
  }
  if (!(node_mass > 0.0)) {
    return;
  }
  const double hoop = (alpha != 0.0 && rj > 0.0) ? (vn * alpha / rj) : 0.0;

  // Assembled row entries of F_j (left cell spans nodes j-1,j; right cell
  // spans nodes j,j+1; the hoop average carries the 1/2 of pbar).
  const double dF_um1 = area * (-dpiL_lo) + hoop * 0.5 * dpdL_lo;
  const double dF_u0 =
      area * (dpiR_lo - dpiL_hi) + hoop * 0.5 * (dpdL_hi + dpdR_lo);
  const double dF_up1 = area * dpiR_hi + hoop * 0.5 * dpdR_hi;

  const double row_sum = fabs(dF_um1) + fabs(dF_u0) + fabs(dF_up1);
  const double lambda_j = row_sum / node_mass;
  if (lambda_j > 0.0 && isfinite(lambda_j)) {
    atomic_max_double_brag(max_lambda, lambda_j);
  }
}

// History-diagnostics kernel: per-cell physical channel etas, configured
// effective eta, and configured instantaneous heat rates. Inactive or
// degenerate cells are marked with eta_i_phys = -1 and skipped on the host.
__global__ void braginskii_history_diag_1d_kernel(
    double* __restrict__ eta_i_phys,
    double* __restrict__ eta_e_phys,
    double* __restrict__ eta_eff,
    double* __restrict__ heat_i,
    double* __restrict__ heat_e,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r,
    const double* __restrict__ rho,
    const double* __restrict__ Ti,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const double* __restrict__ zbar,
    const double* __restrict__ vol,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int geom,
    const int species,
    const DeviceParams p_run,
    const DeviceParams p_phys) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  eta_i_phys[i] = -1.0;
  eta_e_phys[i] = 0.0;
  eta_eff[i] = 0.0;
  heat_i[i] = 0.0;
  heat_e[i] = 0.0;
  const bool active = (hydro_active == nullptr) || (hydro_active[i] != 0);
  if (!active) {
    return;
  }
  const double rl = x_r[i];
  const double rr = x_r[i + 1];
  const double dr = rr - rl;
  if (!(dr > 0.0)) {
    return;
  }
  const double e = (v_r[i + 1] - v_r[i]) / dr;
  double h = 0.0;
  if (geom != 2) {
    const double rsum = rl + rr;
    h = (rsum > 0.0) ? (v_r[i] + v_r[i + 1]) / rsum : 0.0;
  }
  const double alpha = (geom == 0) ? 2.0 : ((geom == 1) ? 1.0 : 0.0);
  const double div_u = e + alpha * h;
  const double w_rr = 2.0 * e - (2.0 / 3.0) * div_u;
  const double w_iso = -(2.0 / 3.0) * div_u;
  const double w_tt = (geom == 2) ? w_iso : (2.0 * h + w_iso);
  double wsq = w_rr * w_rr;
  if (geom == 0) {
    wsq += 2.0 * w_tt * w_tt;
  } else if (geom == 1) {
    wsq += w_tt * w_tt + w_iso * w_iso;
  } else {
    wsq += 2.0 * w_iso * w_iso;
  }
  const double ti_ev = (Ti != nullptr) ? Ti[i] : p_run.ti_floor_ev;
  const double te_ev = (Te != nullptr) ? Te[i] : p_run.te_floor_ev;
  const double a_eff = (A_eff != nullptr) ? A_eff[i] : 1.0;
  const double zb = (zbar != nullptr) ? zbar[i] : 1.0;
  // Physical channels (Braginskii formulas regardless of run model).
  eta_i_phys[i] = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p_phys);
  eta_e_phys[i] = braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p_phys);
  // Configured effective eta and heat split.
  double e_i = 0.0;
  double e_e = 0.0;
  if (species == 1) {
    e_e = braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p_run);
  } else if (species == 2) {
    if (p_run.model == 1) {
      e_i = 0.5 * p_run.eta_const;
      e_e = 0.5 * p_run.eta_const;
    } else {
      e_i = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p_run);
      e_e = braginskii_eta_e_poise(rho[i], te_ev, a_eff, zb, dr, p_run);
    }
  } else {
    e_i = braginskii_eta_poise(rho[i], ti_ev, a_eff, zb, dr, p_run);
  }
  eta_eff[i] = e_i + e_e;
  heat_i[i] = 0.5 * e_i * wsq * vol[i];
  heat_e[i] = 0.5 * e_e * wsq * vol[i];
}

}  // namespace

namespace {

// F-01 (AI kernel review 2026-07-26): strict env parsing — the previous
// atof/atoi path accepted NaN, partial parses ("1.5x"), and negatives, which
// turn the stress anti-diffusive (eta < 0 => pi = +|eta| W, Q < 0) while the
// dt kernel's (eta > 0) guard silently drops the stability limit.
double brag_env_finite_double(const char* name, const char* text) {
  errno = 0;
  char* end = nullptr;
  const double v = std::strtod(text, &end);
  if (end == text || end == nullptr || *end != '\0' || errno == ERANGE ||
      !std::isfinite(v)) {
    throw std::runtime_error(std::string("braginskii: invalid ") + name +
                             " value '" + text +
                             "' (complete finite number required)");
  }
  return v;
}

long brag_env_long(const char* name, const char* text) {
  errno = 0;
  char* end = nullptr;
  const long v = std::strtol(text, &end, 10);
  if (end == text || end == nullptr || *end != '\0' || errno == ERANGE) {
    throw std::runtime_error(std::string("braginskii: invalid ") + name +
                             " value '" + text + "' (integer required)");
  }
  return v;
}

int brag_model_code(const char* name, const std::string& v) {
  if (v == "braginskii") {
    return 0;
  }
  if (v == "constant") {
    return 1;
  }
  throw std::runtime_error(std::string("braginskii: unknown ") + name + " '" +
                           v + "' (expected \"braginskii\" | \"constant\")");
}

int brag_species_code(const char* name, const std::string& v) {
  if (v == "ion") {
    return 0;
  }
  if (v == "electron") {
    return 1;
  }
  if (v == "both") {
    return 2;
  }
  throw std::runtime_error(std::string("braginskii: unknown ") + name + " '" +
                           v + "' (expected \"ion\" | \"electron\" | \"both\")");
}

void apply_env_overrides(Params& p) {
  if (const char* s = std::getenv("TENRYU_BRAG_MODEL")) {
    p.model = brag_model_code("TENRYU_BRAG_MODEL", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_SPECIES")) {
    p.species = brag_species_code("TENRYU_BRAG_SPECIES", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_ETA_CONST")) {
    p.eta_const = brag_env_finite_double("TENRYU_BRAG_ETA_CONST", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_ETA0_SCALE")) {
    p.eta0_scale = brag_env_finite_double("TENRYU_BRAG_ETA0_SCALE", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_MFP_CAP_CELLS")) {
    p.mfp_cap_cells = brag_env_finite_double("TENRYU_BRAG_MFP_CAP_CELLS", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_LNLAMBDA_FIXED")) {
    p.lnlambda_fixed = brag_env_finite_double("TENRYU_BRAG_LNLAMBDA_FIXED", s);
  }
  if (const char* s = std::getenv("TENRYU_BRAG_DT_SAFETY")) {
    p.dt_safety = brag_env_finite_double("TENRYU_BRAG_DT_SAFETY", s);
  }
}

}  // namespace

void validate_params(const Params& p) {
  if (!p.enabled) {
    return;  // disabled params are inert regardless of content
  }
  const auto require = [](const bool ok, const char* what) {
    if (!ok) {
      throw std::runtime_error(
          std::string("braginskii: invalid parameter: ") + what);
    }
  };
  require(p.model == 0 || p.model == 1, "model code (0|1)");
  require(p.species >= 0 && p.species <= 2, "species code (0|1|2)");
  require(std::isfinite(p.eta_const) && p.eta_const >= 0.0,
          "eta_const must be finite and >= 0");
  require(std::isfinite(p.eta0_scale) && p.eta0_scale >= 0.0,
          "eta0_scale must be finite and >= 0 (negative eta is anti-diffusive)");
  require(std::isfinite(p.mfp_cap_cells) && p.mfp_cap_cells >= 0.0,
          "mfp_cap_cells must be finite and >= 0 (0 = off)");
  require(std::isfinite(p.lnlambda_fixed) && p.lnlambda_fixed >= 0.0,
          "lnlambda_fixed must be finite and >= 0 (0 = NRL)");
  require(std::isfinite(p.dt_safety) && p.dt_safety > 0.0,
          "dt_safety must be finite and > 0");
  require(std::isfinite(p.ti_floor_ev) && p.ti_floor_ev >= 0.0,
          "ti_floor_ev must be finite and >= 0");
  require(std::isfinite(p.te_floor_ev) && p.te_floor_ev >= 0.0,
          "te_floor_ev must be finite and >= 0");
}

Params params_from_env() {
  Params p;
  const char* enable = std::getenv("TENRYU_BRAG_ENABLE");
  p.enabled = (enable != nullptr &&
               brag_env_long("TENRYU_BRAG_ENABLE", enable) != 0);
  if (!p.enabled) {
    return p;
  }
  apply_env_overrides(p);
  validate_params(p);
  return p;
}

Params params_from_config(const core::Config& cfg) {
  Params p;
  const auto& pv = cfg.numerics.hydro.plasma_viscosity;
  p.enabled = pv.enabled;
  // Final enabled state first (env can flip it either way), so that a
  // disabled-and-inert config never trips the strict enum mapping below.
  if (const char* s = std::getenv("TENRYU_BRAG_ENABLE")) {
    p.enabled = (brag_env_long("TENRYU_BRAG_ENABLE", s) != 0);
  }
  if (p.enabled) {
    p.model = brag_model_code("Numerics.hydro.plasma_viscosity.model", pv.model);
    p.species = brag_species_code("Numerics.hydro.plasma_viscosity.species",
                                  pv.species);
  } else {
    // Inert legacy mapping (never feeds a kernel).
    p.model = (pv.model == "constant") ? 1 : 0;
    p.species =
        (pv.species == "electron") ? 1 : ((pv.species == "both") ? 2 : 0);
  }
  p.eta_const = pv.eta_const;
  p.eta0_scale = pv.eta0_scale;
  p.mfp_cap_cells = pv.mfp_cap_cells;
  p.lnlambda_fixed = pv.lnlambda_fixed;
  p.dt_safety = pv.dt_safety;
  p.ti_floor_ev = cfg.numerics.floors.Ti;
  p.te_floor_ev = cfg.numerics.floors.Te;
  // Env diagnostic overrides (TENRYU_FLD_* hook precedent): each set
  // variable wins over the namelist value. Strict parse + validation (F-01).
  if (p.enabled) {
    apply_env_overrides(p);
  }
  validate_params(p);
  return p;
}

void compute_cell_stress_1d(double* pi_rr,
                            double* pi_tt,
                            double* heat_rate,
                            double* heat_rate_e,
                            const double* x_r,
                            const double* v_r,
                            const double* rho,
                            const double* Ti,
                            const double* Te,
                            const double* A_eff,
                            const double* zbar,
                            const double* vol,
                            const std::int8_t* hydro_active,
                            const int n_cells,
                            const int geom_code,
                            const Params& p,
                            const double* zmom_r4) {
  if (!p.enabled || n_cells <= 0) {
    // F-02: the header contract is "gated on Params.enabled"; enforce it here
    // instead of relying on every caller (callers all gate today, so this is
    // bit-neutral hardening).
    return;
  }
  const int blocks = (n_cells + 255) / 256;
  if (zmom_r4 != nullptr) {
    switch (p.species) {
      case 1:
        braginskii_cell_stress_1d_kernel<1, true><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            zmom_r4);
        break;
      case 2:
        braginskii_cell_stress_1d_kernel<2, true><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            zmom_r4);
        break;
      default:
        braginskii_cell_stress_1d_kernel<0, true><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            zmom_r4);
        break;
    }
  } else {
    switch (p.species) {
      case 1:
        braginskii_cell_stress_1d_kernel<1, false><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            nullptr);
        break;
      case 2:
        braginskii_cell_stress_1d_kernel<2, false><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            nullptr);
        break;
      default:
        braginskii_cell_stress_1d_kernel<0, false><<<blocks, 256>>>(
            pi_rr, pi_tt, heat_rate, heat_rate_e, x_r, v_r, rho, Ti, Te, A_eff,
            zbar, vol, hydro_active, n_cells, geom_code, to_device_params(p),
            nullptr);
        break;
    }
  }
  brag_sync("braginskii: cell_stress kernel failed");
}

void add_node_accel_1d(double* accel,
                       const double* mass,
                       const double* x_r,
                       const double* vol,
                       const double* pi_rr,
                       const double* pi_tt,
                       const std::uint8_t* node_active,
                       const int n_cells,
                       const int geom_code,
                       const int skip_outer,
                       const Params& p) {
  if (!p.enabled || n_cells <= 0) {
    return;  // F-02: same enabled-contract hardening as compute_cell_stress_1d
  }
  const int n_nodes = n_cells + 1;
  const int blocks = (n_nodes + 255) / 256;
  braginskii_add_accel_1d_kernel<<<blocks, 256>>>(
      accel, mass, x_r, vol, pi_rr, pi_tt, node_active, n_cells, geom_code,
      skip_outer);
  brag_sync("braginskii: add_accel kernel failed");
}

double compute_dt_braginskii_raw(const double* x_r,
                                 const double* rho,
                                 const double* Ti,
                                 const double* Te,
                                 const double* A_eff,
                                 const double* zbar,
                                 const std::int8_t* hydro_active,
                                 const int n_cells,
                                 const Params& p,
                                 const double* zmom_r4) {
  const double inf = std::numeric_limits<double>::infinity();
  if (!p.enabled || n_cells <= 0) {
    return inf;
  }
  // F-21 / W-F conformance: scratch pool instead of per-call cudaMalloc/Free
  // (this runs every driver step when viscosity is enabled).
  double* d_min = static_cast<double*>(
      core::device_scratch_acquire("braginskii:dt:min", sizeof(double)));
  brag_cuda_check(
      cudaMemcpy(d_min, &inf, sizeof(double), cudaMemcpyHostToDevice),
      "braginskii: init dt failed");
  const int blocks = (n_cells + 255) / 256;
  if (zmom_r4 != nullptr) {
    switch (p.species) {
      case 1:
        braginskii_dt_1d_kernel<1, true><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, zmom_r4);
        break;
      case 2:
        braginskii_dt_1d_kernel<2, true><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, zmom_r4);
        break;
      default:
        braginskii_dt_1d_kernel<0, true><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, zmom_r4);
        break;
    }
  } else {
    switch (p.species) {
      case 1:
        braginskii_dt_1d_kernel<1, false><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, nullptr);
        break;
      case 2:
        braginskii_dt_1d_kernel<2, false><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, nullptr);
        break;
      default:
        braginskii_dt_1d_kernel<0, false><<<blocks, 256>>>(
            d_min, x_r, rho, Ti, Te, A_eff, zbar, hydro_active, n_cells,
            to_device_params(p), p.dt_safety, nullptr);
        break;
    }
  }
  brag_sync("braginskii: dt kernel failed");
  double out = inf;
  brag_cuda_check(
      cudaMemcpy(&out, d_min, sizeof(double), cudaMemcpyDeviceToHost),
      "braginskii: copy dt failed");
  return out;
}

double compute_viscous_gershgorin_lambda_1d(const double* x_r,
                                            const double* rho,
                                            const double* Ti,
                                            const double* Te,
                                            const double* A_eff,
                                            const double* zbar,
                                            const double* mass,
                                            const double* vol,
                                            const std::int8_t* hydro_active,
                                            const std::uint8_t* node_active,
                                            const int n_cells,
                                            const int geom_code,
                                            const int skip_outer,
                                            const Params& p,
                                            const double* zmom_r4) {
  if (!p.enabled || n_cells <= 0) {
    return 0.0;
  }
  double* d_max = static_cast<double*>(
      core::device_scratch_acquire("braginskii:gershgorin:max", sizeof(double)));
  const double zero = 0.0;
  brag_cuda_check(
      cudaMemcpy(d_max, &zero, sizeof(double), cudaMemcpyHostToDevice),
      "braginskii: init gershgorin failed");
  const int n_nodes = n_cells + 1;
  const int blocks = (n_nodes + 255) / 256;
  if (zmom_r4 != nullptr) {
    switch (p.species) {
      case 1:
        braginskii_gershgorin_1d_kernel<1, true><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            zmom_r4);
        break;
      case 2:
        braginskii_gershgorin_1d_kernel<2, true><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            zmom_r4);
        break;
      default:
        braginskii_gershgorin_1d_kernel<0, true><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            zmom_r4);
        break;
    }
  } else {
    switch (p.species) {
      case 1:
        braginskii_gershgorin_1d_kernel<1, false><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            nullptr);
        break;
      case 2:
        braginskii_gershgorin_1d_kernel<2, false><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            nullptr);
        break;
      default:
        braginskii_gershgorin_1d_kernel<0, false><<<blocks, 256>>>(
            d_max, x_r, rho, Ti, Te, A_eff, zbar, mass, vol, hydro_active,
            node_active, n_cells, geom_code, skip_outer, to_device_params(p),
            nullptr);
        break;
    }
  }
  brag_sync("braginskii: gershgorin kernel failed");
  double out = 0.0;
  brag_cuda_check(
      cudaMemcpy(&out, d_max, sizeof(double), cudaMemcpyDeviceToHost),
      "braginskii: copy gershgorin failed");
  return out;
}

double compute_dt_braginskii(const core::State& state, const core::Config& cfg) {
  const double inf = std::numeric_limits<double>::infinity();
  const Params p = params_from_config(cfg);
  if (!p.enabled) {
    return inf;
  }
  if (state.mesh.dim == 2) {
    return compute_dt_braginskii_2d(state, cfg);
  }
  if (state.mesh.dim != 1) {
    return inf;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 ||
      state.x_r.size() != static_cast<std::size_t>(n_cells + 1)) {
    return inf;
  }
  std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty() &&
      state.hydro_active.size() == static_cast<std::size_t>(n_cells)) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "braginskii:dt:hydro_active", n_cells * sizeof(std::int8_t)));
    brag_cuda_check(cudaMemcpy(d_active, state.hydro_active.data(),
                               n_cells * sizeof(std::int8_t),
                               cudaMemcpyHostToDevice),
                    "braginskii: copy hydro_active failed");
  }
  const double dt = compute_dt_braginskii_raw(
      state.x_r.data(), state.rho.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.A_eff.empty() ? nullptr : state.A_eff.data(),
      state.zbar.empty() ? nullptr : state.zbar.data(), d_active, n_cells, p,
      state.zmom_active ? state.zmom_r4.data() : nullptr);
  return dt;
}

HistoryDiagnostics compute_history_diagnostics(const core::State& state,
                                               const core::Config& cfg) {
  HistoryDiagnostics out;
  const Params p = params_from_config(cfg);
  if (!p.enabled) {
    return out;
  }
  if (state.mesh.dim == 2) {
    return compute_history_diagnostics_2d(state, cfg);
  }
  if (state.mesh.dim != 1) {
    return out;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 ||
      state.x_r.size() != static_cast<std::size_t>(n_cells + 1)) {
    return out;
  }
  const std::size_t n = static_cast<std::size_t>(n_cells);
  const std::size_t bytes = n * sizeof(double);
  // F-21 / W-F conformance: scratch pool instead of per-call cudaMalloc/Free
  // (the 2D parity design already specifies scratch reuse for this path).
  double* d_buf = static_cast<double*>(
      core::device_scratch_acquire("braginskii:history:diag5", 5 * bytes));
  double* d_eta_i = d_buf;
  double* d_eta_e = d_buf + n;
  double* d_eta_eff = d_buf + 2 * n;
  double* d_heat_i = d_buf + 3 * n;
  double* d_heat_e = d_buf + 4 * n;
  std::int8_t* d_active = nullptr;
  if (!state.hydro_active.empty() &&
      state.hydro_active.size() == static_cast<std::size_t>(n_cells)) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "braginskii:history:hydro_active", n_cells * sizeof(std::int8_t)));
    brag_cuda_check(cudaMemcpy(d_active, state.hydro_active.data(),
                               n_cells * sizeof(std::int8_t),
                               cudaMemcpyHostToDevice),
                    "braginskii: copy diag hydro_active failed");
  }
  DeviceParams p_run = to_device_params(p);
  // F-10 (AI kernel review 2026-07-26): the "physical channel" diagnostics
  // must be the uncapped, unscaled classical Braginskii values with the NRL
  // Coulomb log — otherwise the regime map inherits the mesh through the mfp
  // cap (default 20 cells) and the eta0_scale/lnlambda_fixed knobs, and the
  // cap-active regions read as spurious regime boundaries. The configured
  // channels (eta_eff, heat rates) keep the run's knobs.
  DeviceParams p_phys = p_run;
  p_phys.model = 0;           // physical channels use the Braginskii formulas
  p_phys.eta0_scale = 1.0;    // unscaled
  p_phys.mfp_cap_cells = 0.0; // uncapped
  p_phys.lnlambda_fixed = 0.0;  // NRL log
  const int blocks = (n_cells + 255) / 256;
  braginskii_history_diag_1d_kernel<<<blocks, 256>>>(
      d_eta_i, d_eta_e, d_eta_eff, d_heat_i, d_heat_e, state.x_r.data(),
      state.v_r.data(), state.rho.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.A_eff.empty() ? nullptr : state.A_eff.data(),
      state.zbar.empty() ? nullptr : state.zbar.data(), state.vol.data(),
      d_active, n_cells, state.mesh.geometry_code, p.species, p_run, p_phys);
  brag_sync("braginskii: history diag kernel failed");
  std::vector<double> h(5 * n, 0.0);
  brag_cuda_check(
      cudaMemcpy(h.data(), d_buf, 5 * bytes, cudaMemcpyDeviceToHost),
      "braginskii: copy diag failed");
  std::vector<double> mass_h(n, 0.0);
  brag_cuda_check(cudaMemcpy(mass_h.data(), state.mass.data(), bytes,
                             cudaMemcpyDeviceToHost),
                  "braginskii: copy diag mass failed");
  // Deterministic host reduction (fixed cell order, no atomics).
  double r_min = std::numeric_limits<double>::infinity();
  double r_max = 0.0;
  double sum_m_lnr = 0.0;
  double sum_m = 0.0;
  bool any_ratio = false;
  for (std::size_t i = 0; i < n; ++i) {
    const double ei = h[i];
    if (ei < 0.0) {
      continue;  // inactive cell
    }
    ++out.n_cells_active;
    const double ee = h[n + i];
    out.eta_i_max = std::fmax(out.eta_i_max, ei);
    out.eta_e_max = std::fmax(out.eta_e_max, ee);
    out.eta_eff_max = std::fmax(out.eta_eff_max, h[2 * n + i]);
    out.heat_rate_i_tot += h[3 * n + i];
    out.heat_rate_e_tot += h[4 * n + i];
    if (ei > 0.0 && ee > 0.0) {
      const double r = ee / ei;
      any_ratio = true;
      r_min = std::fmin(r_min, r);
      r_max = std::fmax(r_max, r);
      const double m = mass_h[i];
      sum_m_lnr += m * std::log(r);
      sum_m += m;
      if (r >= 10.0) {
        ++out.n_cells_e_dom;
      } else if (r <= 0.1) {
        ++out.n_cells_i_dom;
      } else {
        ++out.n_cells_mixed;
      }
    }
  }
  if (any_ratio) {
    out.ratio_min = r_min;
    out.ratio_max = r_max;
    out.ratio_geomean_masswt =
        (sum_m > 0.0) ? std::exp(sum_m_lnr / sum_m) : 0.0;
  }
  out.valid = true;
  return out;
}

}  // namespace tenryu::hydro::braginskii
