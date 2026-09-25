#pragma once

// Grey acceleration of the 1D FLD outer iteration (NUMERICS §6.7,
// Radiation.multigroup_diffusion.outer_accel = "grey"; Morel, Larsen &
// Matzen, JQSRT 34 (1985) 243 — the linear multifrequency-grey scheme).
//
// One outer iteration evaluates the Fleck factor f_g and the opacities at
// T_k, solves each group for E_g with the source
//   S_g(T) = f_g(T) c sigma_pe,g B_g(T) + (1 - f_g(T)) c sigma_a,g E^n_g,
// then solves the matter energy for T^{k+1/2} with f_g frozen at f_g(T_k).
// The fixed point has f_g(T*). Linearizing the step from the iterate to the
// fixed point (per cell, rows scaled as in the group assembly, dT_k =
// T^{k+1/2} - T_k, opacities frozen, f_g = 1 / (1 + z) with z ∝ T^3 so that
// f'_g = -3 f_g (1 - f_g) / T):
//   S'_g = f_g c sigma_pe,g B'_g + f'_g (c sigma_pe,g B_g - c sigma_a,g E^n_g),
//   B'_g = 4 a T^3 b_g,   D = rho c_v / dt + sum_g S'_g,   w_g = S'_g / D,
//   P = sum_g f'_g (c sigma_pe,g B_g - c sigma_a,g E^n_g),
//   A_g e_g - dt V w_g sum_h c sigma_a,h e_h = dt V w_g (D - P) dT_k,
//   T* - T^{k+1/2} = (sum_h c sigma_a,h e_h - P dT_k) / D,
// where A_g is group g's tridiagonal matrix of the iteration. With
// e_g = xi_g eps and xi_g the infinite-medium mode w_g / (1 + dt c sigma_a,g)
// normalized to sum_g c sigma_a,g xi_g = 1, the sum over the groups is one
// tridiagonal equation for eps, and the next outer iteration starts from
// T^{k+1/2} + (eps - P dT_k) / D. For one group the grey equation is the
// linearized correction itself (a Newton step). The correction vanishes with
// dT_k, so the converged solution is the plain iteration's.
//
// The spectrum and the grey matrix are built from iteration k's opacities and
// matrix right after its assembly (the group solve may overwrite the matrix);
// the right-hand side and the correction are applied at the start of
// iteration k+1, so an iteration that exits converged keeps its raw output.

#include <cstdint>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "materials/eos_device_table.cuh"
#include "radiation/fld_1d_bodies.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation::fld_1d_grey_accel {

using fld_1d_bodies::eos_log_temperature;
using fld_1d_bodies::finite_or_zero;
using fld_1d_bodies::has_electron_eos_table;
using fld_1d_bodies::nonnegative_finite;
using fld_1d_bodies::safe_pow4;

// Couplings below this (per cell) leave the cell out of the correction.
constexpr double kMinCoupling = 1.0e-300;

// Electron heat capacity per unit mass at T, as the matter update takes it:
// the electron table when there is one, else cv_e, cv_e_override / rho, or
// the ideal gas Z e / (A m_p (gamma - 1)).
template <bool EOS_TAIL>
__device__ inline double matter_cv_mass(const int c,
                                        const double T,
                                        const double rho_c,
                                        const double* __restrict__ zbar,
                                        const double* __restrict__ cv_e,
                                        const int has_cv_e,
                                        const double cv_e_override,
                                        const double* __restrict__ A_eff,
                                        const double* __restrict__ gamma_eff,
                                        const materials::DeviceEOSTableView& electron_eos,
                                        const double temperature_floor_eV) {
  if (has_electron_eos_table(electron_eos)) {
    const materials::RhoBracket br = materials::find_rho_bracket(electron_eos, rho_c);
    double cv;
    if constexpr (EOS_TAIL) {
      cv = materials::device_eos_eval_with_high_t_tail(electron_eos, br, T,
                                                        temperature_floor_eV)
               .cv;
    } else {
      cv = materials::device_eos_cv(electron_eos, br,
                                    eos_log_temperature(T, temperature_floor_eV));
    }
    return fmax(nonnegative_finite(cv), 1.0e-300);
  }
  double cv_mass = 0.0;
  if (has_cv_e != 0 && cv_e != nullptr && cv_e[c] > 0.0 && isfinite(cv_e[c])) {
    cv_mass = cv_e[c];
  } else if (cv_e_override > 0.0) {
    cv_mass = cv_e_override / rho_c;
  } else {
    const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
    const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
    const double z = fmax(finite_or_zero(zbar[c]), 0.0);
    cv_mass = z * core::constants::eV_to_erg /
              (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
  }
  return fmax(finite_or_zero(cv_mass), 1.0e-300);
}

// Per cell at the iteration's linearization temperature Te (= T_k): the
// spectrum xi[c*G+g], D[c], sum_g w_g in sum_w[c] and P[c]; D[c] = 0 marks a
// cell left out (void, or no coupling).
template <bool EOS_TAIL>
__device__ inline void spectrum_body(const int c,
                                     const double* __restrict__ rho,
                                     const double* __restrict__ zbar,
                                     const double* __restrict__ cv_e,
                                     const int has_cv_e,
                                     const double cv_e_override,
                                     const double* __restrict__ A_eff,
                                     const double* __restrict__ gamma_eff,
                                     const materials::DeviceEOSTableView electron_eos,
                                     const double temperature_floor_eV,
                                     const std::uint8_t* __restrict__ cell_is_void,
                                     const double* __restrict__ Te,
                                     const double* __restrict__ sigma_a,
                                     const double* __restrict__ sigma_pe,
                                     const double* __restrict__ fleck,
                                     const double* __restrict__ rad_E_old,
                                     const PlanckTableDeviceView planck,
                                     const int n_groups,
                                     const double dt,
                                     double* __restrict__ xi,
                                     double* __restrict__ D_out,
                                     double* __restrict__ sum_w_out,
                                     double* __restrict__ P_out) {
  const int base = c * n_groups;
  const auto leave_out = [&]() {
    for (int g = 0; g < n_groups; ++g) {
      xi[base + g] = 0.0;
    }
    D_out[c] = 0.0;
    sum_w_out[c] = 0.0;
    P_out[c] = 0.0;
  };
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    leave_out();
    return;
  }
  const double rho_c = fmax(finite_or_zero(rho[c]), 1.0e-300);
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T4 = safe_pow4(T);
  const double aT4 = core::constants::a_eV * T4;
  const double dB_scale = 4.0 * core::constants::a_eV * ((T > 0.0) ? (T4 / T) : 0.0);
  const double cv_mass = matter_cv_mass<EOS_TAIL>(c, T, rho_c, zbar, cv_e, has_cv_e,
                                                  cv_e_override, A_eff, gamma_eff,
                                                  electron_eos, temperature_floor_eV);
  const double dt_safe = fmax(dt, 1.0e-300);
  // Pass 1: S'_g into xi (scratch), D and P.
  double D = rho_c * cv_mass / dt_safe;
  double P = 0.0;
  const PlanckTableDeviceView::Location T_loc = planck.locate_b(T);
  for (int g = 0; g < n_groups; ++g) {
    const double sigma_pa = nonnegative_finite(sigma_a[base + g]);
    const double sigma_pe_g =
        (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[base + g]) : sigma_pa;
    const double f =
        (fleck != nullptr) ? fmin(fmax(finite_or_zero(fleck[base + g]), 0.0), 1.0) : 1.0;
    const double b = fmax(planck.interpolate_b(g, T_loc), 0.0);
    const double df = -3.0 * f * (1.0 - f) / T;
    const double E_n = (rad_E_old != nullptr) ? nonnegative_finite(rad_E_old[base + g]) : 0.0;
    const double imbalance = core::constants::c_light * (sigma_pe_g * aT4 * b - sigma_pa * E_n);
    const double Sp = fmax(f * core::constants::c_light * sigma_pe_g * dB_scale * b +
                               df * imbalance,
                           0.0);
    xi[base + g] = Sp;
    D += Sp;
    P += df * imbalance;
  }
  if (!(isfinite(D) && D > 0.0 && isfinite(P))) {
    leave_out();
    return;
  }
  // Pass 2: w_g, the infinite-medium mode and its normalization.
  double N = 0.0;
  double sum_w = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double sigma_pa = nonnegative_finite(sigma_a[base + g]);
    const double w = xi[base + g] / D;
    const double q = w / (1.0 + dt * core::constants::c_light * sigma_pa);
    xi[base + g] = q;
    N += core::constants::c_light * sigma_pa * q;
    sum_w += w;
  }
  if (!(isfinite(N) && N > kMinCoupling && isfinite(sum_w))) {
    leave_out();
    return;
  }
  const double inv_N = 1.0 / N;
  for (int g = 0; g < n_groups; ++g) {
    xi[base + g] *= inv_N;
  }
  D_out[c] = D;
  sum_w_out[c] = sum_w;
  P_out[c] = P;
}

// spectrum_body with one warp per cell (2026-09-25): the lanes evaluate the
// groups' terms (the Planck fractions dominate the cost) into shared memory
// and lane 0 forms the sums in group order with the same expressions as
// spectrum_body, so the results are bitwise the same. shared: 5 * n_groups
// doubles; all 32 lanes of the warp call it.
template <bool EOS_TAIL>
__device__ inline void spectrum_body_warp(const int c,
                                          const int lane,
                                          double* __restrict__ shared,
                                          const double* __restrict__ rho,
                                          const double* __restrict__ zbar,
                                          const double* __restrict__ cv_e,
                                          const int has_cv_e,
                                          const double cv_e_override,
                                          const double* __restrict__ A_eff,
                                          const double* __restrict__ gamma_eff,
                                          const materials::DeviceEOSTableView electron_eos,
                                          const double temperature_floor_eV,
                                          const std::uint8_t* __restrict__ cell_is_void,
                                          const double* __restrict__ Te,
                                          const double* __restrict__ sigma_a,
                                          const double* __restrict__ sigma_pe,
                                          const double* __restrict__ fleck,
                                          const double* __restrict__ rad_E_old,
                                          const PlanckTableDeviceView planck,
                                          const int n_groups,
                                          const double dt,
                                          double* __restrict__ xi,
                                          double* __restrict__ D_out,
                                          double* __restrict__ sum_w_out,
                                          double* __restrict__ P_out) {
  constexpr unsigned kFull = 0xffffffffu;
  const int base = c * n_groups;
  double* const s_sp = shared;                   // S'_g, then w_g
  double* const s_df = shared + n_groups;        // f'_g
  double* const s_imb = shared + 2 * n_groups;   // c sigma_pe,g B_g - c sigma_a,g E^n_g
  double* const s_csa = shared + 3 * n_groups;   // c sigma_a,g
  double* const s_q = shared + 4 * n_groups;     // q_g
  const auto leave_out = [&]() {
    for (int g = lane; g < n_groups; g += 32) {
      xi[base + g] = 0.0;
    }
    if (lane == 0) {
      D_out[c] = 0.0;
      sum_w_out[c] = 0.0;
      P_out[c] = 0.0;
    }
  };
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    leave_out();
    return;
  }
  const double rho_c = fmax(finite_or_zero(rho[c]), 1.0e-300);
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T4 = safe_pow4(T);
  const double aT4 = core::constants::a_eV * T4;
  const double dB_scale = 4.0 * core::constants::a_eV * ((T > 0.0) ? (T4 / T) : 0.0);
  const double dt_safe = fmax(dt, 1.0e-300);
  // Pass 1 terms.
  const PlanckTableDeviceView::Location T_loc = planck.locate_b(T);
  for (int g = lane; g < n_groups; g += 32) {
    const double sigma_pa = nonnegative_finite(sigma_a[base + g]);
    const double sigma_pe_g =
        (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[base + g]) : sigma_pa;
    const double f =
        (fleck != nullptr) ? fmin(fmax(finite_or_zero(fleck[base + g]), 0.0), 1.0) : 1.0;
    const double b = fmax(planck.interpolate_b(g, T_loc), 0.0);
    const double df = -3.0 * f * (1.0 - f) / T;
    const double E_n = (rad_E_old != nullptr) ? nonnegative_finite(rad_E_old[base + g]) : 0.0;
    const double imbalance = core::constants::c_light * (sigma_pe_g * aT4 * b - sigma_pa * E_n);
    const double Sp = fmax(f * core::constants::c_light * sigma_pe_g * dB_scale * b +
                               df * imbalance,
                           0.0);
    xi[base + g] = Sp;
    s_sp[g] = Sp;
    s_df[g] = df;
    s_imb[g] = imbalance;
    s_csa[g] = core::constants::c_light * sigma_pa;
  }
  __syncwarp(kFull);
  int ok = 0;
  double D = 0.0;
  double P = 0.0;
  if (lane == 0) {
    const double cv_mass = matter_cv_mass<EOS_TAIL>(c, T, rho_c, zbar, cv_e, has_cv_e,
                                                    cv_e_override, A_eff, gamma_eff,
                                                    electron_eos, temperature_floor_eV);
    D = rho_c * cv_mass / dt_safe;
    for (int g = 0; g < n_groups; ++g) {
      D += s_sp[g];
      P += s_df[g] * s_imb[g];
    }
    ok = (isfinite(D) && D > 0.0 && isfinite(P)) ? 1 : 0;
  }
  ok = __shfl_sync(kFull, ok, 0);
  if (ok == 0) {
    leave_out();
    return;
  }
  D = __shfl_sync(kFull, D, 0);
  // Pass 2 terms: w_g and the infinite-medium mode.
  for (int g = lane; g < n_groups; g += 32) {
    const double sigma_pa = nonnegative_finite(sigma_a[base + g]);
    const double w = s_sp[g] / D;
    const double q = w / (1.0 + dt * core::constants::c_light * sigma_pa);
    xi[base + g] = q;
    s_sp[g] = w;
    s_q[g] = q;
  }
  __syncwarp(kFull);
  double inv_N = 0.0;
  if (lane == 0) {
    double N = 0.0;
    double sum_w = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      N += s_csa[g] * s_q[g];
      sum_w += s_sp[g];
    }
    ok = (isfinite(N) && N > kMinCoupling && isfinite(sum_w)) ? 1 : 0;
    if (ok != 0) {
      inv_N = 1.0 / N;
      D_out[c] = D;
      sum_w_out[c] = sum_w;
      P_out[c] = P;
    }
  }
  ok = __shfl_sync(kFull, ok, 0);
  if (ok == 0) {
    leave_out();
    return;
  }
  inv_N = __shfl_sync(kFull, inv_N, 0);
  for (int g = lane; g < n_groups; g += 32) {
    xi[base + g] = s_q[g] * inv_N;
  }
}

// Grey row c from the groups' rows (group-major: row g * n_cells + c) and the
// spectra of c and its neighbours.
__device__ inline void matrix_body(const int c,
                                   const double* __restrict__ lower,
                                   const double* __restrict__ diag,
                                   const double* __restrict__ upper,
                                   const double* __restrict__ xi,
                                   const double* __restrict__ D,
                                   const double* __restrict__ sum_w,
                                   const double* __restrict__ vol,
                                   const int n_cells,
                                   const int n_groups,
                                   const double dt,
                                   double* __restrict__ grey_lower,
                                   double* __restrict__ grey_diag,
                                   double* __restrict__ grey_upper) {
  if (!(D[c] > 0.0)) {
    grey_lower[c] = 0.0;
    grey_diag[c] = 1.0;
    grey_upper[c] = 0.0;
    return;
  }
  double d = 0.0;
  double l = 0.0;
  double u = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int row = g * n_cells + c;
    d += diag[row] * xi[c * n_groups + g];
    if (c > 0) {
      l += lower[row] * xi[(c - 1) * n_groups + g];
    }
    if (c + 1 < n_cells) {
      u += upper[row] * xi[(c + 1) * n_groups + g];
    }
  }
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  d -= dt * V * sum_w[c];
  if (!(isfinite(d) && d > 0.0 && isfinite(l) && isfinite(u))) {
    // Not a usable row: the cell takes no correction.
    grey_lower[c] = 0.0;
    grey_diag[c] = 1.0;
    grey_upper[c] = 0.0;
    return;
  }
  grey_lower[c] = l;
  grey_diag[c] = d;
  grey_upper[c] = u;
}

// Right-hand side at the start of iteration k+1: dT_k = Te - T_lin with
// T_lin = T_k and Te = T^{k+1/2}.
__device__ inline void rhs_body(const int c,
                                const double* __restrict__ D,
                                const double* __restrict__ sum_w,
                                const double* __restrict__ P,
                                const double* __restrict__ T_lin,
                                const double* __restrict__ Te,
                                const double* __restrict__ vol,
                                const double dt,
                                double* __restrict__ grey_rhs) {
  if (!(D[c] > 0.0)) {
    grey_rhs[c] = 0.0;
    return;
  }
  const double dT = finite_or_zero(Te[c]) - finite_or_zero(T_lin[c]);
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  const double rhs = dt * V * sum_w[c] * (D[c] - P[c]) * dT;
  grey_rhs[c] = isfinite(rhs) ? rhs : 0.0;
}

// The correction applied to the matter: T += (eps - P dT_k) / D within
// [-T/2, +T], with the electron energy and pressure closed as the matter
// update closes them (table: from T; otherwise incremental with the constant
// c_v).
template <bool EOS_TAIL>
__device__ inline void apply_body(const int c,
                                  const double* __restrict__ D,
                                  const double* __restrict__ P,
                                  const double* __restrict__ T_lin,
                                  const double* __restrict__ eps,
                                  const double* __restrict__ rho,
                                  const double* __restrict__ zbar,
                                  const double* __restrict__ cv_e,
                                  const int has_cv_e,
                                  const double cv_e_override,
                                  const double* __restrict__ A_eff,
                                  const double* __restrict__ gamma_eff,
                                  const materials::DeviceEOSTableView electron_eos,
                                  const double temperature_floor_eV,
                                  double* __restrict__ Te,
                                  double* __restrict__ ee,
                                  double* __restrict__ Pe) {
  const double Dc = D[c];
  if (!(Dc > 0.0)) {
    return;
  }
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double dT_k = finite_or_zero(Te[c]) - finite_or_zero(T_lin[c]);
  const double corr = (eps[c] - P[c] * dT_k) / Dc;
  if (!isfinite(corr) || corr == 0.0) {
    return;
  }
  const double dT = fmin(fmax(corr, -0.5 * T), T);
  const double T_new = fmax(T + dT, temperature_floor_eV);
  if (!(T_new != T)) {
    return;
  }
  const double rho_c = fmax(finite_or_zero(rho[c]), 1.0e-300);
  if (has_electron_eos_table(electron_eos)) {
    const materials::RhoBracket br = materials::find_rho_bracket(electron_eos, rho_c);
    if constexpr (EOS_TAIL) {
      const auto th =
          materials::device_eos_eval_with_high_t_tail(electron_eos, br, T_new,
                                                      temperature_floor_eV);
      ee[c] = th.e;
      Pe[c] = th.P;
    } else {
      const double logT = eos_log_temperature(T_new, temperature_floor_eV);
      ee[c] = materials::device_eos_energy(electron_eos, br, logT);
      Pe[c] = materials::device_eos_pressure(electron_eos, br, logT);
    }
  } else {
    const double cv_mass = matter_cv_mass<false>(c, T, rho_c, zbar, cv_e, has_cv_e,
                                                 cv_e_override, A_eff, gamma_eff,
                                                 electron_eos, temperature_floor_eV);
    ee[c] = finite_or_zero(ee[c]) + cv_mass * (T_new - T);
    Pe[c] = fmax(fmax(gamma_eff[c], 1.0 + 1.0e-12) - 1.0, 1.0e-12) * rho_c *
            finite_or_zero(ee[c]);
  }
  Te[c] = T_new;
}

}  // namespace tenryu::radiation::fld_1d_grey_accel
