#pragma once

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "materials/eos_device_table.hpp"
#include "materials/ionmix_reader.cuh"
#include "mesh/geometry_1d.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation::fld_1d_bodies {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kFld1dOuterVacuum = 0;
constexpr int kFld1dOuterReflect = 1;
constexpr int kFld1dOuterMarshak = 2;
constexpr int kUpdateMatterWarpSize = 32;

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double x) {
  return isfinite(x) ? fmax(x, 0.0) : 0.0;
}

__host__ __device__ inline double safe_pow4(const double T) {
  const double Tc = fmin(fmax(finite_or_zero(T), 0.0), 1.0e6);
  const double T2 = Tc * Tc;
  return T2 * T2;
}

constexpr double kNlteKappaFloor = 1.0e-100;
constexpr double kNlteEtaFloor = 1.0e-300;

__device__ inline double nlte_safe_log_kappa(const double kappa) {
  return log(fmax(kappa, kNlteKappaFloor));
}

__device__ inline double nlte_edge_log_slope_temperature_upper(
    const materials::IonmixOpacityDeviceView& table,
    const double* __restrict__ kappa_table,
    const int group,
    const double log_ni) {
  if (table.ntemp < 2) {
    return 0.0;
  }
  const double log_T_hi = table.log_temps[table.ntemp - 1];
  const double log_T_lo = table.log_temps[table.ntemp - 2];
  if (!(log_T_hi > log_T_lo)) {
    return 0.0;
  }
  const double k_hi = table.interpolate(kappa_table, group, log_ni, log_T_hi);
  const double k_lo = table.interpolate(kappa_table, group, log_ni, log_T_lo);
  return (nlte_safe_log_kappa(k_hi) - nlte_safe_log_kappa(k_lo)) /
         (log_T_hi - log_T_lo);
}

__device__ inline double nlte_edge_log_slope_density_lower(
    const materials::IonmixOpacityDeviceView& table,
    const double* __restrict__ kappa_table,
    const int group,
    const double log_T) {
  if (table.ndens < 2) {
    return 0.0;
  }
  const double log_ni_lo = table.log_numdens[0];
  const double log_ni_hi = table.log_numdens[1];
  if (!(log_ni_hi > log_ni_lo)) {
    return 0.0;
  }
  const double k_lo = table.interpolate(kappa_table, group, log_ni_lo, log_T);
  const double k_hi = table.interpolate(kappa_table, group, log_ni_hi, log_T);
  return (nlte_safe_log_kappa(k_hi) - nlte_safe_log_kappa(k_lo)) /
         (log_ni_hi - log_ni_lo);
}

__device__ inline double nlte_interpolate_kappa_with_smooth_edges(
    const materials::IonmixOpacityDeviceView& table,
    const double* __restrict__ kappa_table,
    const int group,
    const double ni_cm3,
    const double temperature_eV) {
  if (kappa_table == nullptr || table.ntemp <= 0 || table.ndens <= 0) {
    return 0.0;
  }
  const double ni_min = exp(table.log_ni_min);
  double ni_eval = ni_cm3;
  if (!isfinite(ni_eval) || !(ni_eval > 0.0)) {
    ni_eval = ni_min;
  }
  double T_eval = temperature_eV;
  if (!isfinite(T_eval) || !(T_eval > 0.0)) {
    T_eval = table.T_min;
  }

  const double log_ni_eval = log(ni_eval);
  const double log_T_eval = log(T_eval);
  const bool rarefied_extrapolation = ni_eval < ni_min;
  const bool hot_extrapolation = T_eval > table.T_max;
  const double log_ni_edge =
      fmin(fmax(log_ni_eval, table.log_ni_min), table.log_ni_max);
  const double log_T_edge =
      fmin(fmax(log_T_eval, table.log_T_min), table.log_T_max);

  double log_kappa = nlte_safe_log_kappa(
      table.interpolate(kappa_table, group, log_ni_edge, log_T_edge));
  if (rarefied_extrapolation) {
    const double slope_ni =
        nlte_edge_log_slope_density_lower(table, kappa_table, group,
                                          log_T_edge);
    log_kappa += slope_ni * (log_ni_eval - table.log_ni_min);
  }
  if (hot_extrapolation) {
    const double slope_T =
        nlte_edge_log_slope_temperature_upper(table, kappa_table, group,
                                              log_ni_edge);
    log_kappa += slope_T * (log_T_eval - table.log_T_max);
  }

  const double log_kappa_min = log(kNlteKappaFloor);
  const double log_kappa_max = log(DBL_MAX);
  log_kappa = fmin(fmax(log_kappa, log_kappa_min), log_kappa_max);
  return exp(log_kappa);
}

__device__ inline double nlte_planck_weight(
    const PlanckTableDeviceView& planck,
    const int g,
    const double temperature_eV,
    const int n_groups,
    bool* __restrict__ invalid) {
  if (n_groups <= 1) {
    return 1.0;
  }
  const double b = planck.interpolate_b(g, temperature_eV);
  if (!isfinite(b)) {
    *invalid = true;
    return 0.0;
  }
  return fmax(b, 0.0);
}

__device__ inline void eval_nlte_opacity_emission_kernel_body(
    const int c,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e_table,
    const std::uint8_t* __restrict__ cell_is_void,
    const materials::IonmixOpacityDeviceView table,
    const PlanckTableDeviceView planck,
    double* __restrict__ out_f,
    double* __restrict__ out_sigma_pa,
    double* __restrict__ out_sigma_pe,
    double* __restrict__ out_sigma_R,
    double* __restrict__ out_sigma_a_eff,
    double* __restrict__ out_sigma_s_eff,
    double* __restrict__ out_eta_cdf,
    double* __restrict__ out_eta,
    double* __restrict__ out_sigma_p_em,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double A,
    const double alpha,
    const double sigma_cap,
    const double cv_e_override,
    const double temperature_floor_eV,
    const double gamma_m1) {
  (void)n_cells;
  const int base = c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    if (out_sigma_p_em != nullptr) {
      out_sigma_p_em[c] = 0.0;
    }
    for (int g = 0; g < n_groups; ++g) {
      const int idx = base + g;
      out_f[idx] = 1.0;
      out_sigma_pa[idx] = 0.0;
      if (out_sigma_pe != nullptr) {
        out_sigma_pe[idx] = 0.0;
      }
      out_sigma_R[idx] = 0.0;
      out_sigma_a_eff[idx] = 0.0;
      out_sigma_s_eff[idx] = 0.0;
      out_eta[idx] = 0.0;
      out_eta_cdf[idx] =
          (n_groups == 1) ? 1.0 : (static_cast<double>(g + 1) /
                                   static_cast<double>(n_groups));
    }
    return;
  }

  const double rho_c = fmax(rho[c], 0.0);
  double T_eval = Te[c];
  if (!isfinite(T_eval) || !(T_eval > 0.0)) {
    T_eval = table.T_min;
  }
  double ni_c = exp(table.log_ni_min);
  if (rho_c > 0.0) {
    ni_c = rho_c / (fmax(A, 1.0e-12) * core::constants::proton_mass);
    if (!isfinite(ni_c) || !(ni_c > 0.0)) {
      ni_c = exp(table.log_ni_min);
    }
  }
  const double T4 = safe_pow4(T_eval);

  bool b_invalid = false;
  double b_sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    b_sum += nlte_planck_weight(planck, g, T_eval, n_groups, &b_invalid);
  }
  const bool use_uniform_b =
      (n_groups > 1) &&
      (b_invalid || !(b_sum > 0.0) || !isfinite(b_sum));

  double sigma_p_abs = 0.0;
  double eta_sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = base + g;
    double b_g = 1.0;
    if (n_groups > 1) {
      b_g = use_uniform_b ? (1.0 / static_cast<double>(n_groups))
                          : (nlte_planck_weight(planck, g, T_eval, n_groups,
                                                &b_invalid) /
                             b_sum);
    }

    double sigma_pa_raw = 0.0;
    double sigma_pe_raw = 0.0;
    double sigma_R_raw = 0.0;
    if (rho_c > 0.0) {
      sigma_pa_raw =
          rho_c * nlte_interpolate_kappa_with_smooth_edges(
                      table, table.kappa_PA, g, ni_c, T_eval);
      sigma_pe_raw =
          rho_c * nlte_interpolate_kappa_with_smooth_edges(
                      table, table.kappa_PE, g, ni_c, T_eval);
      sigma_R_raw =
          rho_c * nlte_interpolate_kappa_with_smooth_edges(
                      table, table.kappa_R, g, ni_c, T_eval);
    }

    double sigma_pa = fmax(sigma_pa_raw, 0.0);
    double sigma_pe = fmax(sigma_pe_raw, 0.0);
    double sigma_R = fmax(sigma_R_raw, 0.0);
    if (!isfinite(sigma_pa)) {
      sigma_pa = 0.0;
    }
    if (!isfinite(sigma_pe)) {
      sigma_pe = 0.0;
    }
    if (!isfinite(sigma_R)) {
      sigma_R = 0.0;
    }
    if (sigma_cap > 0.0) {
      sigma_pa = fmin(sigma_pa, sigma_cap);
      sigma_pe = fmin(sigma_pe, sigma_cap);
      sigma_R = fmin(sigma_R, sigma_cap);
    }

    out_sigma_pa[idx] = sigma_pa;
    if (out_sigma_pe != nullptr) {
      out_sigma_pe[idx] = sigma_pe;
    }
    out_sigma_R[idx] = sigma_R;
    sigma_p_abs += b_g * sigma_pa;

    double eta_g = 0.0;
    if (T_eval > temperature_floor_eV) {
      eta_g =
          sigma_pe * core::constants::c_light * core::constants::a_eV * T4 * b_g;
    }
    if (!isfinite(eta_g) || eta_g < 0.0) {
      eta_g = 0.0;
    }
    out_eta[idx] = eta_g;
    eta_sum += eta_g;
  }

  double Cv_e = 0.0;
  if (cv_e_override > 0.0) {
    Cv_e = cv_e_override;
  } else if (cv_e_table != nullptr && cv_e_table[c] > 0.0) {
    Cv_e = fmax(rho_c, 0.0) * cv_e_table[c];
  } else {
    Cv_e = rho_c * fmax(zbar[c], 0.0) * core::constants::eV_to_erg /
           (fmax(A, 1.0e-12) * core::constants::proton_mass *
            fmax(gamma_m1, 1.0e-12));
  }
  Cv_e = fmax(Cv_e, 1.0e-30);

  // No IMC-safety beta <= 1 cap here (NUMERICS section 6.7): beta is the
  // physical stiffness 4 a T^3 / C_v; capping it in the ideal-gas-fallback
  // branch inflated f in hot low-C_v cells and under-damped the emission
  // linearization exactly where the Fleck factor must protect. f = 1/(1+z)
  // stays in (0, 1] for any beta >= 0, so the cap never served its stated
  // "f > 1 prevention" purpose. (2026-07-26 review.)
  double beta =
      4.0 * core::constants::a_eV * T_eval * T_eval * T_eval / Cv_e;
  if (!isfinite(beta) || beta < 0.0) {
    beta = 0.0;
  }

  double sigma_p_em = 0.0;
  double eta_sum_cell = eta_sum;
  const bool has_emission =
      (T_eval > temperature_floor_eV) && (eta_sum > kNlteEtaFloor) &&
      (T4 > 0.0);
  if (has_emission) {
    sigma_p_em = eta_sum / (core::constants::a_eV *
                            core::constants::c_light * T4);
  } else {
    eta_sum_cell = 0.0;
  }
  if (out_sigma_p_em != nullptr) {
    out_sigma_p_em[c] = sigma_p_em;
  }

  double f = 1.0;
  if (sigma_p_em > 0.0) {
    const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
    f = 1.0 / (1.0 + alpha_safe * beta * core::constants::c_light *
                         fmax(dt, 0.0) * sigma_p_em);
  }
  if (!isfinite(f)) {
    f = 0.0;
  }
  f = fmin(fmax(f, 0.0), 1.0);
  (void)sigma_p_abs;

  double cdf_accum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = base + g;
    out_f[idx] = f;
    out_sigma_a_eff[idx] = f * out_sigma_pa[idx];
    out_sigma_s_eff[idx] = (1.0 - f) * out_sigma_pa[idx];
    if (eta_sum_cell > kNlteEtaFloor) {
      cdf_accum += fmax(out_eta[idx], 0.0) / eta_sum_cell;
      out_eta_cdf[idx] = fmin(fmax(cdf_accum, 0.0), 1.0);
    } else {
      out_eta[idx] = 0.0;
      out_eta_cdf[idx] =
          (n_groups == 1) ? 1.0 : (static_cast<double>(g + 1) /
                                   static_cast<double>(n_groups));
    }
    if (g == n_groups - 1) {
      out_eta_cdf[idx] = 1.0;
    }
  }
}

__device__ inline bool has_electron_eos_table(
    const materials::DeviceEOSTableView& electron_eos) {
  return electron_eos.n_rho > 0 && electron_eos.n_T > 0 &&
         electron_eos.P_table != nullptr && electron_eos.e_table != nullptr &&
         electron_eos.cv_table != nullptr;
}

__device__ inline double eos_log_temperature(const double T,
                                             const double temperature_floor_eV) {
  return log(fmax(finite_or_zero(T), fmax(temperature_floor_eV, 1.0e-30)));
}

__device__ inline double flux_limiter_lambda(const double R, const int limiter) {
  const double r = fmax(finite_or_zero(R), 0.0);
  if (limiter == 2) {
    return 1.0 / 3.0;
  }
  if (limiter == 1) {
    return 1.0 / sqrt(9.0 + r * r);
  }
  if (r < 1.0e-3) {
    // Levermore-Pomraning series (coth r = 1/r + r/3 - r^3/45 + 2r^5/945):
    // the raw form below loses up to ~1e-4 relative accuracy to cancellation
    // for r near 1e-6; the omitted series term is O(r^6/4725) < 3e-22 here.
    const double r2 = r * r;
    return 1.0 / 3.0 - r2 / 45.0 + 2.0 * r2 * r2 / 945.0;
  }
  if (r > 50.0) {
    // coth(r) - 1 < 4e-44 for r > 50: (1 - 1/r)/r is exact to double
    // precision and continuous at the crossover (the bare 1/r tail had a
    // 2% jump at r = 50).
    return (1.0 - 1.0 / r) / r;
  }
  return (1.0 / tanh(r) - 1.0 / r) / r;
}

__host__ __device__ inline double fld_1d_outer_leak_coeff(const int bc) {
  if (bc == kFld1dOuterVacuum) {
    return 0.5 * core::constants::c_light;
  }
  if (bc == kFld1dOuterMarshak) {
    return 0.25 * core::constants::c_light;
  }
  return 0.0;
}

__device__ inline void build_eta_from_planck_kernel_body(
    const int idx,
    const double* __restrict__ Te,
    const double* __restrict__ sigma_a,
    PlanckTableDeviceView planck,
    double* __restrict__ eta,
    int n_cells,
    int n_groups,
    double temperature_floor_eV,
    const std::uint8_t* __restrict__ cell_eval_skip) {
  const int c = idx / n_groups;
  if (cell_eval_skip != nullptr && cell_eval_skip[c] != 0U) {
    return;
  }
  const int g = idx - c * n_groups;
  (void)n_cells;
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T2 = T * T;
  const double B = core::constants::a_eV * T2 * T2 *
                   fmax(planck.interpolate_b(g, T), 0.0);
  eta[idx] = core::constants::c_light *
             fmax(finite_or_zero(sigma_a[idx]), 0.0) * B;
}

__device__ inline double fld_face_diffusion_coeff(
    const double* __restrict__ x_r,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_R,
    const int f,
    const int g,
    const int n_groups,
    const double sigma_floor,
    const int limiter) {
  const int lcg = (f - 1) * n_groups + g;
  const int rcg = f * n_groups + g;
  const double El = fmax(finite_or_zero(rad_E[lcg]), 0.0);
  const double Er = fmax(finite_or_zero(rad_E[rcg]), 0.0);
  const double xl = 0.5 * (x_r[f - 1] + x_r[f]);
  const double xr = 0.5 * (x_r[f] + x_r[f + 1]);
  const double dist = fmax(xr - xl, 1.0e-300);
  const double sl = fmax(finite_or_zero(sigma_R[lcg]), sigma_floor);
  const double sr = fmax(finite_or_zero(sigma_R[rcg]), sigma_floor);
  const double sigma_face = 0.5 * (sl + sr);
  const double E_face = fmax(0.5 * (El + Er), 1.0e-300);
  const double R = fabs(Er - El) / (dist * sigma_face * E_face);
  const double lambda = flux_limiter_lambda(R, limiter);
  return core::constants::c_light * lambda / sigma_face;
}

// Exponential-Rosenbrock Fleck-factor form: f = phi_1(-z) = (1 - exp(-z))/z.
// Exact retention for the fixed-radiation scalar relaxation (design doc
// docs/design/fleck_exp_source_20260716.md section 2). 0 < f <= 1 for z >= 0
// and z*f -> 1 as z -> inf, so stiff cells still equilibrate (unlike the
// retired exp(-z) blend). Series below the crossover avoids the 0/0; the
// z^3/24 term at z = 1e-6 is ~4e-20 relative.
__device__ inline double fleck_form_phi1(const double z) {
  if (z < 1.0e-6) {
    return 1.0 - 0.5 * z + z * z / 6.0;
  }
  return (1.0 - exp(-z)) / z;
}

__device__ inline void compute_fleck_for_fld_kernel_body(
    const int c,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    const double* __restrict__ state_cv_e,
    double* __restrict__ f_fleck,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double alpha,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const int use_table_cv,
    const double temperature_floor_eV,
    const double* __restrict__ rad_E_old,
    const PlanckTableDeviceView planck,
    const int fleck_beta_secant,
    const int fleck_form_exp,
    const std::uint8_t* __restrict__ cell_eval_skip) {
  if (cell_eval_skip != nullptr && cell_eval_skip[c] != 0U) {
    return;
  }
  const int base = c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    for (int g = 0; g < n_groups; ++g) {
      f_fleck[base + g] = 1.0;
    }
    return;
  }

  const double rho_c = rho[c];
  const double Te_c = fmax(Te[c], 1.0e-12);
  double Cv_e = -1.0;
  if (use_table_cv != 0 && has_electron_eos_table(electron_eos)) {
    const materials::RhoBracket br = materials::find_rho_bracket(electron_eos, rho_c);
    const double cv_mass = materials::device_eos_cv(
        electron_eos, br, eos_log_temperature(Te_c, temperature_floor_eV));
    if (isfinite(cv_mass) && cv_mass > 0.0) {
      Cv_e = fmax(rho_c, 0.0) * cv_mass;
    }
  }
  if (Cv_e <= 0.0) {
    if (cv_e_override > 0.0) {
      Cv_e = cv_e_override;
    } else if (state_cv_e != nullptr && state_cv_e[c] > 0.0) {
      Cv_e = fmax(rho_c, 0.0) * state_cv_e[c];
    } else {
      // Shared per-cell effective properties (multi-material fix).
      const double z_atom = fmax(zbar[c], 0.0);
      const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
      const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
      const double cv_mass_e =
          z_atom * core::constants::eV_to_erg /
          (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
      Cv_e = fmax(rho_c, 0.0) * cv_mass_e;
    }
  }
  Cv_e = fmax(Cv_e, 1.0e-30);

  double beta =
      4.0 * core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
  const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
  if (fleck_beta_secant != 0 && use_table_cv != 0 &&
      has_electron_eos_table(electron_eos) && rad_E_old != nullptr) {
    // beta_sec = [B(T_pred)-B(T^n)]/[U_e(T_pred)-U_e(T^n)], grey, with a
    // 0-D local predictor for T_pred; U_e via the SAME table accessor the
    // matter Newton linearizes (consistency by construction). Every guard
    // falls back to the tangent beta above. Design:
    // docs/design/fleck_beta_secant_20260714.md section 3.1.
    double E_grey = 0.0;
    double sp_num = 0.0;
    double sp_den = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      E_grey += fmax(rad_E_old[base + g], 0.0);
      const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
      sp_num += fmax(sigma_a[base + g], 0.0) * b_g;
      sp_den += b_g;
    }
    const double T2n = Te_c * Te_c;
    const double B_n = core::constants::a_eV * T2n * T2n;
    const double sigma_p = (sp_den > 0.0) ? (sp_num / sp_den) : 0.0;
    const double z_tan =
        alpha_safe * beta * core::constants::c_light * sigma_p * dt;
    const double f_tan = 1.0 / (1.0 + fmax(z_tan, 0.0));
    double dT_est = f_tan * core::constants::c_light * sigma_p *
                    (E_grey - B_n) * dt / Cv_e;
    const double dT_cap = 0.5 * Te_c;
    dT_est = fmin(fmax(dT_est, -dT_cap), dT_cap);
    constexpr double kSecantMinDT = 1.0e-6;
    if (isfinite(dT_est) && fabs(dT_est) > kSecantMinDT * Te_c) {
      const double T_pred =
          fmax(Te_c + dT_est, fmax(temperature_floor_eV, 1.0e-12));
      if (T_pred != Te_c) {
        const materials::RhoBracket br_sec =
            materials::find_rho_bracket(electron_eos, rho_c);
        const double e_n = materials::device_eos_energy(
            electron_eos, br_sec,
            eos_log_temperature(Te_c, temperature_floor_eV));
        const double e_p = materials::device_eos_energy(
            electron_eos, br_sec,
            eos_log_temperature(T_pred, temperature_floor_eV));
        const double dU = fmax(rho_c, 0.0) * (e_p - e_n);
        const double Tp2 = T_pred * T_pred;
        const double dB = core::constants::a_eV * (Tp2 * Tp2 - T2n * T2n);
        const double beta_sec = dB / dU;
        if (isfinite(beta_sec) && beta_sec > 0.0) {
          // 1 = secant (adopt), 2 = guard (one-sided: max beta => min f).
          beta = (fleck_beta_secant == 2) ? fmax(beta, beta_sec) : beta_sec;
        }
      }
    }
  }
  for (int g = 0; g < n_groups; ++g) {
    const double sigma_g = fmax(sigma_a[base + g], 0.0);
    const double z =
        alpha_safe * beta * core::constants::c_light * sigma_g * dt;
    // Standard Fleck-Cummings factor. The old exp(-z) blend for z > 2 has
    // the wrong stiff limit (z*f -> 0 kills the matter-radiation exchange
    // entirely, freezing T away from equilibrium); f = 1/(1+z) keeps
    // z*f -> 1/alpha so stiff cells still equilibrate. The blend was
    // invisible before the W-B Fleck-consistent matter update because the
    // matter kernel ignored f altogether.
    double f = 1.0 / (1.0 + z);
    if (fleck_form_exp != 0) {
      f = fleck_form_phi1(z);
    }
    if (!isfinite(f) || f < 0.0) {
      f = 0.0;
    }
    if (f > 1.0) {
      f = 1.0;
    }
    f_fleck[base + g] = f;
  }
}

__device__ inline void compute_marshak_finc_kernel_body(
    const int g,
    const PlanckTableDeviceView planck,
    const double T_r_eV,
    const double const_flux,
    double* __restrict__ finc,
    const int n_groups) {
  if (T_r_eV > 0.0) {
    const double T2 = T_r_eV * T_r_eV;
    const double b = fmax(planck.interpolate_b(g, T_r_eV), 0.0);
    finc[g] =
        0.25 * core::constants::c_light * core::constants::a_eV * T2 * T2 * b;
    return;
  }
  finc[g] = (g == 0) ? fmax(const_flux, 0.0) : 0.0;
  (void)n_groups;
}

template <int GEOM>
__device__ inline void assemble_fld_tridiag_kernel_body(
    const int idx,
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ sigma_pa,
    const double* __restrict__ fleck,
    const double* __restrict__ eta,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ rad_E_iter,
    const double* __restrict__ sigma_R,
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    int n_cells,
    int n_groups,
    double dt,
    int outer_bc,
    const double* __restrict__ marshak_finc,
    double volume_source_rate,
    double volume_source_r_max,
    double sigma_floor,
    int limiter) {
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  const double sigma = fmax(finite_or_zero(sigma_removal[cg]), 0.0);
  double d = V + dt * core::constants::c_light * sigma * V;
  double l = 0.0;
  double u = 0.0;
  if (c > 0) {
    const double Df = fld_face_diffusion_coeff(
        x_r, rad_E_iter, sigma_R, c, g, n_groups, sigma_floor, limiter);
    const double xf = x_r[c];
    const double area = (GEOM == 0) ? 4.0 * kPi * xf * xf
                                    : mesh::geometry_1d_face_area(GEOM, xf);
    const double xl = 0.5 * (x_r[c - 1] + x_r[c]);
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double dist = fmax(xc - xl, 1.0e-300);
    const double coef = dt * area * Df / dist;
    d += coef;
    l = -coef;
  }
  if (c + 1 < n_cells) {
    const double Df = fld_face_diffusion_coeff(
        x_r, rad_E_iter, sigma_R, c + 1, g, n_groups, sigma_floor, limiter);
    const double xf = x_r[c + 1];
    const double area = (GEOM == 0) ? 4.0 * kPi * xf * xf
                                    : mesh::geometry_1d_face_area(GEOM, xf);
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double xr = 0.5 * (x_r[c + 1] + x_r[c + 2]);
    const double dist = fmax(xr - xc, 1.0e-300);
    const double coef = dt * area * Df / dist;
    d += coef;
    u = -coef;
  } else {
    const double r_outer = x_r[n_cells];
    const double area =
        (GEOM == 0) ? 4.0 * kPi * r_outer * r_outer
                    : mesh::geometry_1d_face_area(GEOM, r_outer);
    d += dt * area * fld_1d_outer_leak_coeff(outer_bc);
  }
  lower[idx] = l;
  diag[idx] = d;
  upper[idx] = u;
  const double E_old = fmax(finite_or_zero(rad_E_old[cg]), 0.0);
  double source = fmax(finite_or_zero(eta[cg]), 0.0);
  if (fleck != nullptr && sigma_pa != nullptr) {
    const double f = fmin(fmax(finite_or_zero(fleck[cg]), 0.0), 1.0);
    const double sigma_raw = fmax(finite_or_zero(sigma_pa[cg]), 0.0);
    source = f * source +
             (1.0 - f) * core::constants::c_light * sigma_raw * E_old;
  }
  double rhs_val = V * E_old + dt * V * source;
  if (g == 0 && volume_source_rate > 0.0 && volume_source_r_max > 0.0) {
    const double r_c = 0.5 * (x_r[c] + x_r[c + 1]);
    if (r_c <= volume_source_r_max) {
      rhs_val += dt * V * volume_source_rate;
    }
  }
  if (outer_bc == kFld1dOuterMarshak && marshak_finc != nullptr &&
      c + 1 == n_cells) {
    const double r_outer = x_r[n_cells];
    const double area =
        (GEOM == 0) ? 4.0 * kPi * r_outer * r_outer
                    : mesh::geometry_1d_face_area(GEOM, r_outer);
    rhs_val += dt * area * fmax(finite_or_zero(marshak_finc[g]), 0.0);
  }
  rhs[idx] = rhs_val;
}

__device__ inline void publish_solution_kernel_body(
    const int idx,
    const double* __restrict__ rhs,
    double* __restrict__ rad_E,
    int n_cells,
    int n_groups) {
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  rad_E[c * n_groups + g] = fmax(finite_or_zero(rhs[idx]), 0.0);
}

__device__ inline void snapshot_Te_kernel_body(const int c,
                                               const double* __restrict__ Te,
                                               double* __restrict__ Te_old,
                                               int n_cells) {
  (void)n_cells;
  Te_old[c] = finite_or_zero(Te[c]);
}

// PERSISTENT-PATH-ONLY duplicate (design §8 fallback): the multi-kernel path's update_matter lives inline in fld_1d_gpu.cu because sharing this body broke the bit-exact GXII golden (FMA-contraction shift). Keep the two in sync manually when physics changes.
__device__ inline void update_matter_body_persistent(
    const int cell,
    const int lane,
    double* __restrict__ s_F_g,
    double* __restrict__ s_dF_g,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ Te_old,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_pe,
    const double* __restrict__ rad_E,
    const double* __restrict__ fleck,
    const double* __restrict__ rad_E_old,
    PlanckTableDeviceView planck,
    materials::DeviceEOSTableView electron_eos,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    int n_cells,
    int n_groups,
    double dt,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    double cv_e_override,
    double temperature_floor_eV,
    int has_cv_e,
    int max_newton_iterations,
    double newton_tolerance) {
  const int c = cell;
  const int tid = lane;
  (void)n_cells;

  double T_lane0 = temperature_floor_eV;
  int break_lane0 = 0;
  double step_T_ref = temperature_floor_eV;
  double iter_T_prev = temperature_floor_eV;
  double rho_c = 1.0e-300;
  double cv_mass = 1.0e-300;
  double dt_safe = 1.0e-300;
  double Cv = 1.0e-300;
  double e_e_ref = 0.0;
  int use_table_eos = 0;
  materials::RhoBracket eos_rho_bracket{0, 0, 0.0};
  if (tid == 0) {
    rho_c = fmax(finite_or_zero(rho[c]), 1.0e-300);
    step_T_ref = fmax(finite_or_zero(Te_old[c]), temperature_floor_eV);
    iter_T_prev = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
    T_lane0 = iter_T_prev;
    double cv_mass_raw = 0.0;
    if (has_cv_e != 0 && cv_e != nullptr && cv_e[c] > 0.0 && isfinite(cv_e[c])) {
      cv_mass_raw = cv_e[c];
    } else if (cv_e_override > 0.0) {
      cv_mass_raw = cv_e_override / rho_c;
    } else {
      const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
      const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
      const double z = fmax(finite_or_zero(zbar[c]), 0.0);
      cv_mass_raw =
          z * core::constants::eV_to_erg /
          (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
    }
    cv_mass = fmax(finite_or_zero(cv_mass_raw), 1.0e-300);
    dt_safe = fmax(dt, 1.0e-300);
    Cv = fmax(rho_c * cv_mass, 1.0e-300);
    use_table_eos = has_electron_eos_table(electron_eos) ? 1 : 0;
    if (use_table_eos != 0) {
      eos_rho_bracket = materials::find_rho_bracket(electron_eos, rho_c);
      e_e_ref = materials::device_eos_energy(
          electron_eos, eos_rho_bracket,
          eos_log_temperature(step_T_ref, temperature_floor_eV));
    } else {
      e_e_ref = 0.0;
    }
    break_lane0 = 0;
  }
  __syncwarp();

  const int max_iter = (max_newton_iterations > 1) ? max_newton_iterations : 1;
  const double tol = fmax(newton_tolerance, 0.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    const double T = __shfl_sync(0xffffffffu, T_lane0, 0);
    const int brk = __shfl_sync(0xffffffffu, break_lane0, 0);
    if (brk != 0) {
      break;
    }
    const double T4 = safe_pow4(T);
    const double T3 = (T > 0.0) ? (T4 / T) : 0.0;

    for (int g = tid; g < n_groups; g += kUpdateMatterWarpSize) {
      const int idx = c * n_groups + g;
      const double sigma_pa = nonnegative_finite(sigma_a[idx]);
      const double sigma_pe_g =
          (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
      const double E = nonnegative_finite(rad_E[idx]);
      const double b = fmax(planck.interpolate_b(g, T), 0.0);
      const double B = core::constants::a_eV * T4 * b;
      double f = 1.0;
      double E_old_g = 0.0;
      if (fleck != nullptr) {
        f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
        E_old_g = (rad_E_old != nullptr)
                      ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                      : 0.0;
      }
      const double emit_g = f * core::constants::c_light * sigma_pe_g * B +
                            (1.0 - f) * core::constants::c_light *
                                sigma_pe_g * E_old_g;
      s_F_g[g] = -(core::constants::c_light * sigma_pa * E - emit_g);
      s_dF_g[g] = f * core::constants::c_light * sigma_pe_g *
                  4.0 * core::constants::a_eV * T3 * b;
    }
    __syncwarp();

    if (tid == 0) {
      double F = 0.0;
      double dF = 0.0;
      if (use_table_eos != 0) {
        const double logT = eos_log_temperature(T, temperature_floor_eV);
        const double e_e_T =
            materials::device_eos_energy(electron_eos, eos_rho_bracket, logT);
        const double cv_e_T =
            fmax(nonnegative_finite(materials::device_eos_cv(
                     electron_eos, eos_rho_bracket, logT)),
                 1.0e-300);
        F = rho_c * (e_e_T - e_e_ref) / dt_safe;
        dF = rho_c * cv_e_T / dt_safe;
      } else {
        F = Cv * (T - step_T_ref) / dt_safe;
        dF = Cv / dt_safe;
      }
      for (int g = 0; g < n_groups; ++g) {
        F += s_F_g[g];
        dF += s_dF_g[g];
      }
      if (!(isfinite(F) && isfinite(dF)) || !(dF > 0.0)) {
        break_lane0 = 1;
      } else {
        double dT = -F / dF;
        const double max_step = 0.5 * fmax(T, temperature_floor_eV);
        dT = fmin(fmax(dT, -max_step), max_step);
        const double T_next = fmax(T + dT, temperature_floor_eV);
        const double rel = fabs(T_next - T) / fmax(T_next, temperature_floor_eV);
        T_lane0 = T_next;
        if (rel <= tol) {
          break_lane0 = 1;
        }
      }
    }
    __syncwarp();
  }

  const double T_final = __shfl_sync(0xffffffffu, T_lane0, 0);
  if (tid == 0) {
    Te[c] = T_final;
    if (use_table_eos != 0) {
      const double logT = eos_log_temperature(T_final, temperature_floor_eV);
      ee[c] =
          materials::device_eos_energy(electron_eos, eos_rho_bracket, logT);
      Pe[c] =
          materials::device_eos_pressure(electron_eos, eos_rho_bracket, logT);
    } else {
      ee[c] = finite_or_zero(ee[c]) + cv_mass * (T_final - iter_T_prev);
      Pe[c] = fmax(fmax(gamma_eff[c], 1.0 + 1.0e-12) - 1.0, 1.0e-12) *
              rho_c * finite_or_zero(ee[c]);
    }
    delta_T_rel[c] =
        fabs(T_final - iter_T_prev) / fmax(T_final, temperature_floor_eV);
  }
  __syncwarp();

  const double Tn4 = safe_pow4(T_final);
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  for (int g = tid; g < n_groups; g += kUpdateMatterWarpSize) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double sigma_pe_g =
        (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
    const double E = nonnegative_finite(rad_E[idx]);
    const double b = fmax(planck.interpolate_b(g, T_final), 0.0);
    double f = 1.0;
    double E_old_g = 0.0;
    if (fleck != nullptr) {
      f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
      E_old_g =
          (rad_E_old != nullptr) ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                                 : 0.0;
    }
    rad_dep[idx] = dt * V * core::constants::c_light * sigma_pa * E;
    rad_emit[idx] =
        dt * V *
        (f * core::constants::c_light * sigma_pe_g * core::constants::a_eV *
             Tn4 * b +
         (1.0 - f) * core::constants::c_light * sigma_pe_g * E_old_g);
  }
}

}  // namespace tenryu::radiation::fld_1d_bodies
