#include "radiation/sn_material_newton_gpu.cuh"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "materials/eos_device_table.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::radiation {
namespace {

constexpr int kNewtonThreads = 128;
constexpr double kTiny = 1.0e-300;
constexpr int kRetryFloorRoot = 1;
constexpr int kRetryBracketExpansion = 2;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__host__ __device__ inline double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

__host__ __device__ inline double safe_pow4(const double T) {
  const double Tc = fmin(fmax(finite_or_zero(T), 0.0), 1.0e6);
  const double T2 = Tc * Tc;
  return T2 * T2;
}

__host__ __device__ inline double sn_unfloored_streaming_state(
    const double E_sweep,
    const double lambda_pa,
    const double lambda_pe,
    const double B_prev) {
  return E_sweep * (1.0 + lambda_pa) - lambda_pe * B_prev;
}

__host__ __device__ inline double sn_conservative_E_plus(
    const double E_sweep,
    const double r_g,
    const double B,
    const double B_prev) {
  return fmax(E_sweep + r_g * (B - B_prev), 0.0);
}

__host__ __device__ inline double sn_conservative_E_plus_from_star(
    const double E_star,
    const double lambda_pa,
    const double lambda_pe,
    const double B,
    const double S_dt) {
  return fmax((E_star + lambda_pe * B + S_dt) / (1.0 + lambda_pa), 0.0);
}

class SnElectronEOSTableCache {
 public:
  materials::DeviceEOSTableView view_for(const materials::EOSTableTriplet& tables) {
    if (tables_ != &tables) {
      electron_.upload(tables.electron);
      tables_ = &tables;
    }
    return electron_.view();
  }

 private:
  const materials::EOSTableTriplet* tables_ = nullptr;
  materials::DeviceEOSTable electron_;
};

SnElectronEOSTableCache& sn_electron_eos_table_cache() {
  static SnElectronEOSTableCache cache;
  return cache;
}

__device__ inline bool has_electron_eos_table(
    const materials::DeviceEOSTableView& electron_eos) {
  return electron_eos.n_rho > 0 && electron_eos.n_T > 0 &&
         electron_eos.P_table != nullptr &&
         electron_eos.e_table != nullptr &&
         electron_eos.cv_table != nullptr;
}

__device__ inline double eos_log_temperature(const double T,
                                             const double temperature_floor_eV) {
  return log(fmax(finite_or_zero(T), fmax(temperature_floor_eV, 1.0e-30)));
}

__device__ inline double sn_tail_T_top(
    const materials::DeviceEOSTableView& tab) {
  return exp(tab.log_T_max);
}

// e/cv/P with a linear-cv ideal tail above the table temperature ceiling;
// below/at the ceiling these are exactly the existing extrap evaluations.
__device__ inline double sn_eval_e_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double e_top = materials::device_eos_energy_extrap(
        tab, rb, rho_c, T_top, Zbar, A_c, low_density_extrap);
    const double cv_top = fmax(
        materials::device_eos_cv_extrap(tab, rb, rho_c, T_top, Zbar, A_c,
                                        low_density_extrap),
        0.0);
    if (isfinite(e_top) && cv_top > 0.0) {
      return e_top + cv_top * (T - T_top);
    }
  }
  return materials::device_eos_energy_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                             low_density_extrap);
}

__device__ inline double sn_eval_cv_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double cv_top = fmax(
        materials::device_eos_cv_extrap(tab, rb, rho_c, T_top, Zbar, A_c,
                                        low_density_extrap),
        0.0);
    if (cv_top > 0.0) {
      return cv_top;
    }
  }
  return materials::device_eos_cv_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                         low_density_extrap);
}

__device__ inline double sn_eval_P_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double P_top = materials::device_eos_pressure_extrap(
        tab, rb, rho_c, T_top, Zbar, A_c, low_density_extrap);
    if (isfinite(P_top)) {
      return P_top * (T / T_top);
    }
  }
  return materials::device_eos_pressure_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                               low_density_extrap);
}

template <bool EOS_TAIL>
__device__ inline double invert_e_e_via_bisection(
    const materials::DeviceEOSTableView& electron_eos,
    const materials::RhoBracket& eos_rho_bracket,
    double e_target,
    double T_floor,
    const double rho_c,
    const double Zbar,
    const double A_amu,
    const bool low_density_extrap) {
  if constexpr (EOS_TAIL) {
    const double T_top = sn_tail_T_top(electron_eos);
    const double e_top = materials::device_eos_energy_extrap(
        electron_eos, eos_rho_bracket, rho_c, T_top, Zbar, A_amu,
        low_density_extrap);
    const double cv_top = fmax(materials::device_eos_cv_extrap(
        electron_eos, eos_rho_bracket, rho_c, T_top, Zbar, A_amu,
        low_density_extrap), 0.0);
    if (isfinite(e_top) && cv_top > 0.0 && e_target > e_top) {
      return T_top + (e_target - e_top) / cv_top;
    }
  }
  // Bisect on log-T to find T such that device_eos_energy(rho, T) ≈ e_target.
  // Bracket: [log(T_floor), log(T_max)] where log(T_max) = electron_eos.log_T_max.
  // Iterate up to 60 times until xhi - xlo < 1e-14.
  double xlo = log(fmax(T_floor, 1.0e-30));
  double xhi = fmax(electron_eos.log_T_max, log(fmax(T_floor * 10.0, 1.0e-30)));
  for (int k = 0; k < 60; ++k) {
    const double xm = 0.5 * (xlo + xhi);
    if constexpr (EOS_TAIL) {
      const double e_m = sn_eval_e_tail(
          electron_eos, eos_rho_bracket, rho_c, exp(xm), Zbar, A_amu,
          low_density_extrap);
      if (e_m < e_target) xlo = xm;
      else                xhi = xm;
    } else {
      const double e_m = materials::device_eos_energy(electron_eos, eos_rho_bracket, xm);
      if (e_m < e_target) xlo = xm;
      else                xhi = xm;
    }
    if (xhi - xlo < 1.0e-14) break;
  }
  return exp(0.5 * (xlo + xhi));
}

struct ResidualEval {
  double R;       // residual, [erg/cm^3]
  double Rprime;  // dR/dT = rho * cv_e + dP/dT, [erg/cm^3/eV]
  double A;       // bracket energy/source scale, [erg/cm^3]
  double P;       // emission/final-radiation scale, [erg/cm^3]
  double U_T;     // matter energy density at T, [erg/cm^3]
};

template <bool EOS_TAIL>
__device__ inline ResidualEval compute_residual_R_2d_legacy(
    int c,
    int n_groups,
    double T,
    double rho_c,
    double cv_mass,        // for ideal-gas path
    double e_e_ref,        // U_n / rho for TMAT
    double T_ref,          // for ideal-gas path
    double dt_safe,
    bool use_table_eos,
    const materials::DeviceEOSTableView& electron_eos,
    const materials::RhoBracket& eos_rho_bracket,
    const double* sigma_a,
    const double* sigma_pe,
    const double* rad_E,
    PlanckTableDeviceView planck,
    double temperature_floor_eV,
    bool low_density_extrap,
    double Zbar,
    double A_amu) {
  ResidualEval out;
  const double T4 = safe_pow4(T);
  const double T3 = (T > 0.0) ? T4 / T : 0.0;
  double cv_T;
  double U_n;
  if (use_table_eos) {
    if constexpr (EOS_TAIL) {
      out.U_T = rho_c * sn_eval_e_tail(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu,
          low_density_extrap);
      cv_T = nonnegative_finite(sn_eval_cv_tail(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu,
          low_density_extrap));
    } else {
      out.U_T = rho_c * materials::device_eos_energy_extrap(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu, low_density_extrap);
      cv_T = nonnegative_finite(materials::device_eos_cv_extrap(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu, low_density_extrap));
    }
    U_n = rho_c * e_e_ref;
  } else {
    out.U_T = rho_c * cv_mass * T;
    cv_T = cv_mass;
    U_n = rho_c * cv_mass * T_ref;
  }
  double A = 0.0, P = 0.0, dPdT = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double sigma_pe_g = (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
    const double E = nonnegative_finite(rad_E[idx]);
    const double b = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
    const double B = core::constants::a_eV * T4 * b;
    A    += dt_safe * core::constants::c_light * sigma_pa * E;
    P    += dt_safe * core::constants::c_light * sigma_pe_g * B;
    dPdT += dt_safe * core::constants::c_light * sigma_pe_g * 4.0 * core::constants::a_eV * T3 * b;
  }
  out.A = A;
  out.P = P;
  out.R = out.U_T - U_n - A + P;
  out.Rprime = rho_c * cv_T + dPdT;
  return out;
}

template <bool EOS_TAIL>
__device__ inline ResidualEval compute_residual_R_production(
    int c,
    int n_groups,
    double T,
    double rho_c,
    double cv_mass,
    double e_e_ref,
    double T_ref,
    double dt_safe,
    bool use_table_eos,
    const materials::DeviceEOSTableView& electron_eos,
    const materials::RhoBracket& eos_rho_bracket,
    const double* sigma_a,
    const double* sigma_pe,
    const double* rad_E,
    const double* E_star_override,
    const double* source_ext,
    double T_prev_eV,
    PlanckTableDeviceView planck,
    double temperature_floor_eV,
    bool low_density_extrap,
    double Zbar,
    double A_amu) {
  ResidualEval out;
  const double T4 = safe_pow4(T);
  const double T3 = (T > 0.0) ? T4 / T : 0.0;
  double cv_T;
  double U_n;
  if (use_table_eos) {
    if constexpr (EOS_TAIL) {
      out.U_T = rho_c * sn_eval_e_tail(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu,
          low_density_extrap);
      cv_T = nonnegative_finite(sn_eval_cv_tail(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu,
          low_density_extrap));
    } else {
      out.U_T = rho_c * materials::device_eos_energy_extrap(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu, low_density_extrap);
      cv_T = nonnegative_finite(materials::device_eos_cv_extrap(
          electron_eos, eos_rho_bracket, rho_c, T, Zbar, A_amu, low_density_extrap));
    }
    U_n = rho_c * e_e_ref;
  } else {
    out.U_T = rho_c * cv_mass * T;
    cv_T = cv_mass;
    U_n = rho_c * cv_mass * T_ref;
  }
  out.R = 0.0;
  double A_eff = 0.0, P_eff = 0.0, dPdT = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double sigma_pe_g = (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
    const double lambda_pa = dt_safe * core::constants::c_light * sigma_pa;
    const double lambda_pe = dt_safe * core::constants::c_light * sigma_pe_g;
    const double T_prev4 = safe_pow4(T_prev_eV);
    const double b_prev = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_prev_eV), 0.0);
    const double B_prev = core::constants::a_eV * T_prev4 * b_prev;
    const double E_post = nonnegative_finite(rad_E[idx]);
    const double b = (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
    const double B = core::constants::a_eV * T4 * b;
    const double denom = 1.0 + lambda_pa;
    const double r_g = lambda_pe / denom;
    const bool use_E_star_override = E_star_override != nullptr;
    const double E_star =
        use_E_star_override
            ? finite_or_zero(E_star_override[idx])
            : sn_unfloored_streaming_state(E_post, lambda_pa, lambda_pe, B_prev);
    const double S_dt = (source_ext != nullptr && g == 0)
                            ? dt_safe * source_ext[c]
                            : 0.0;
    const double E_plus =
        use_E_star_override
            ? sn_conservative_E_plus_from_star(E_star, lambda_pa, lambda_pe, B, S_dt)
            : sn_conservative_E_plus(E_post, r_g, B, B_prev);
    A_eff += fmax(E_star, 0.0);
    P_eff += E_plus;
    if (E_plus > 0.0) {
      dPdT += r_g * 4.0 * core::constants::a_eV * T3 * b;
    }
    out.R += E_plus - E_star - S_dt;
  }
  out.A = A_eff;
  out.P = P_eff;
  out.R += out.U_T - U_n;
  out.Rprime = rho_c * cv_T + dPdT;
  return out;
}

__device__ double mass_heat_capacity(const double rho,
                                     const double zbar,
                                     const double cv_e_value,
                                     const double cv_e_const,
                                     const double Cv_e_const,
                                     const double A,
                                     const double gamma) {
  if (isfinite(cv_e_value) && cv_e_value > 0.0) {
    return cv_e_value;
  }
  const double rho_c = fmax(nonnegative_finite(rho), kTiny);
  if (isfinite(Cv_e_const) && Cv_e_const > 0.0) {
    return Cv_e_const / rho_c;
  }
  if (isfinite(cv_e_const) && cv_e_const > 0.0) {
    return cv_e_const;
  }
  const double gm1 = fmax(finite_or_zero(gamma) - 1.0, 1.0e-12);
  const double z = fmax(finite_or_zero(zbar), 0.0);
  return fmax(z * core::constants::eV_to_erg /
                  (fmax(finite_or_zero(A), 1.0e-12) *
                   core::constants::proton_mass * gm1),
              kTiny);
}

template <bool EOS_TAIL>
__global__ void sn_material_newton_kernel(
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_pe,
    const double* __restrict__ rad_E,
    const double* __restrict__ E_star_override,
    const double* __restrict__ source_ext,
    double* __restrict__ rad_E_out,
    int legacy_2d_closure,
    int eos_low_density_extrap,
    const double* __restrict__ rho,
    const double* __restrict__ cv_e,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ Te_old,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ diag_rad_emission_at_Tnp1,
    double* __restrict__ diag_rad_absorption,
    double* __restrict__ diag_clip_energy,
    double* __restrict__ diag_clip_full_deficit,
    double* __restrict__ delta_T_rel,
    PlanckTableDeviceView planck,
    materials::DeviceEOSTableView electron_eos,
    int n_cells,
    int n_groups,
    double dt,
    double cv_e_const,
    double Cv_e_const,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    double temperature_floor_eV,
    int max_iterations,
    double tolerance,
    int* __restrict__ retry_flag,
    int c_begin) {
  const int c = c_begin + blockIdx.x;
  (void)n_cells;
  (void)max_iterations;
  (void)tolerance;

  const double rho_c = fmax(nonnegative_finite(rho[c]), kTiny);
  const double Zbar = (zbar != nullptr) ? zbar[c] : 1.0;
  // Shared per-cell effective ideal-gas properties (multi-material fix).
  const double A_c = (A_eff != nullptr) ? fmax(A_eff[c], 1.0e-12) : 1.0;
  const double gamma_c =
      (gamma_eff != nullptr) ? fmax(gamma_eff[c], 1.0 + 1.0e-12) : 5.0 / 3.0;
  const double cv_mass = mass_heat_capacity(
      rho_c,
      Zbar,
      (cv_e != nullptr) ? cv_e[c] : 0.0,
      cv_e_const,
      Cv_e_const,
      A_c,
      gamma_c);
  const double T_ref = fmax(finite_or_zero(Te_old[c]), temperature_floor_eV);
  const double T_prev = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double dt_safe = fmax(dt, kTiny);
  const bool use_2d_legacy_closure = legacy_2d_closure != 0;
  const bool use_E_star_override = !use_2d_legacy_closure && E_star_override != nullptr;
  const bool use_table_eos = has_electron_eos_table(electron_eos);
  const bool low_density_extrap = eos_low_density_extrap != 0;
  materials::RhoBracket eos_rho_bracket{};
  double e_e_ref = 0.0;
  double T = 0.0;
  if (threadIdx.x == 0) {
    T = T_prev;
    if (!(dt > 0.0)) {
      T = T_ref;
    }
  }
  if (threadIdx.x == 0 && use_table_eos) {
    eos_rho_bracket = materials::find_rho_bracket(electron_eos, rho_c);
    if constexpr (EOS_TAIL) {
      e_e_ref = sn_eval_e_tail(electron_eos, eos_rho_bracket, rho_c, T_ref,
                               Zbar, A_c, low_density_extrap);
    } else {
      e_e_ref = materials::device_eos_energy_extrap(
          electron_eos, eos_rho_bracket, rho_c, T_ref, Zbar, A_c,
          low_density_extrap);
    }
  }

  __shared__ ResidualEval s_ev;
  __shared__ double s_eval_T;
  __shared__ double s_final_T;
  __shared__ double s_scalar_R_tail;
  __shared__ double s_rho_cv;
  __shared__ double s_T_trial;
  __shared__ double s_Rg[kNewtonThreads];
  __shared__ double s_Ag[kNewtonThreads];
  __shared__ double s_Pg[kNewtonThreads];
  __shared__ double s_Rprimeg[kNewtonThreads];
  __shared__ int s_floor_root;
  __shared__ int s_expand_enabled;
  __shared__ int s_expand_active;
  __shared__ int s_bracket_valid;
  __shared__ int s_newton_active;
  __shared__ int s_newton_eval;
  __shared__ int s_trial_eval;

  auto eval = [&]() {
    __syncthreads();
    const double T_eval = s_eval_T;
    const double T4 = safe_pow4(T_eval);
    const double T3 = (T_eval > 0.0) ? T4 / T_eval : 0.0;
    if (threadIdx.x == 0) {
      ResidualEval out{};
      double cv_T;
      double U_n;
      if (use_table_eos) {
        if constexpr (EOS_TAIL) {
          out.U_T = rho_c * sn_eval_e_tail(
              electron_eos, eos_rho_bracket, rho_c, T_eval, Zbar, A_c,
              low_density_extrap);
          cv_T = nonnegative_finite(sn_eval_cv_tail(
              electron_eos, eos_rho_bracket, rho_c, T_eval, Zbar, A_c,
              low_density_extrap));
        } else {
          out.U_T = rho_c * materials::device_eos_energy_extrap(
              electron_eos, eos_rho_bracket, rho_c, T_eval, Zbar, A_c,
              low_density_extrap);
          cv_T = nonnegative_finite(materials::device_eos_cv_extrap(
              electron_eos, eos_rho_bracket, rho_c, T_eval, Zbar, A_c,
              low_density_extrap));
        }
        U_n = rho_c * e_e_ref;
      } else {
        out.U_T = rho_c * cv_mass * T_eval;
        cv_T = cv_mass;
        U_n = rho_c * cv_mass * T_ref;
      }
      out.R = 0.0;
      out.Rprime = 0.0;
      out.A = 0.0;
      out.P = 0.0;
      s_scalar_R_tail = out.U_T - U_n;
      s_rho_cv = rho_c * cv_T;
      s_ev = out;
    }
    __syncthreads();
    for (int base = 0; base < n_groups; base += blockDim.x) {
      const int g = base + threadIdx.x;
      double Rg = 0.0;
      double Ag = 0.0;
      double Pg = 0.0;
      double Rprimeg = 0.0;
      if (g < n_groups) {
        const int idx = c * n_groups + g;
        const double sigma_pa = nonnegative_finite(sigma_a[idx]);
        const double sigma_pe_g =
            (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
        if (!use_2d_legacy_closure) {
          const double lambda_pa = dt_safe * core::constants::c_light * sigma_pa;
          const double lambda_pe = dt_safe * core::constants::c_light * sigma_pe_g;
          const double T_prev_eV = T_ref;
          const double T_prev4 = safe_pow4(T_prev_eV);
          const double b_prev =
              (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_prev_eV), 0.0);
          const double B_prev = core::constants::a_eV * T_prev4 * b_prev;
          const double E_post = nonnegative_finite(rad_E[idx]);
          const double b =
              (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_eval), 0.0);
          const double B = core::constants::a_eV * T4 * b;
          const double denom = 1.0 + lambda_pa;
          const double r_g = lambda_pe / denom;
          const double E_star =
              use_E_star_override
                  ? finite_or_zero(E_star_override[idx])
                  : sn_unfloored_streaming_state(E_post, lambda_pa, lambda_pe, B_prev);
          const double S_dt = (source_ext != nullptr && g == 0)
                                  ? dt_safe * source_ext[c]
                                  : 0.0;
          const double E_plus =
              use_E_star_override
                  ? sn_conservative_E_plus_from_star(E_star, lambda_pa, lambda_pe, B,
                                                     S_dt)
                  : sn_conservative_E_plus(E_post, r_g, B, B_prev);
          Ag = fmax(E_star, 0.0);
          Pg = E_plus;
          if (E_plus > 0.0) {
            Rprimeg = r_g * 4.0 * core::constants::a_eV * T3 * b;
          }
          Rg = E_plus - E_star - S_dt;
        } else {
          const double E = nonnegative_finite(rad_E[idx]);
          const double b =
              (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_eval), 0.0);
          const double B = core::constants::a_eV * T4 * b;
          Ag = dt_safe * core::constants::c_light * sigma_pa * E;
          Pg = dt_safe * core::constants::c_light * sigma_pe_g * B;
          Rprimeg = dt_safe * core::constants::c_light * sigma_pe_g * 4.0 *
                    core::constants::a_eV * T3 * b;
        }
      }
      s_Rg[threadIdx.x] = Rg;
      s_Ag[threadIdx.x] = Ag;
      s_Pg[threadIdx.x] = Pg;
      s_Rprimeg[threadIdx.x] = Rprimeg;
      __syncthreads();
      if (threadIdx.x == 0) {
        const int g_end =
            (base + blockDim.x < n_groups) ? base + blockDim.x : n_groups;
        for (int gg = base; gg < g_end; ++gg) {
          const int slot = gg - base;
          s_ev.R += s_Rg[slot];
          s_ev.A += s_Ag[slot];
          s_ev.P += s_Pg[slot];
          s_ev.Rprime += s_Rprimeg[slot];
        }
      }
      __syncthreads();
    }
    if (threadIdx.x == 0) {
      if (use_2d_legacy_closure) {
        s_ev.R = s_scalar_R_tail - s_ev.A + s_ev.P;
      } else {
        s_ev.R += s_scalar_R_tail;
      }
      s_ev.Rprime = s_rho_cv + s_ev.Rprime;
    }
    __syncthreads();
    return s_ev;
  };

  // === Energy-variable bracketed Newton ===
  // Compute initial residual to obtain A_total (laser absorption energy density)
  if (threadIdx.x == 0) {
    s_eval_T = T;
  }
  ResidualEval ev_init = eval();
  double A_total = 0.0;
  double U_n_const = 0.0;
  if (threadIdx.x == 0) {
    A_total = ev_init.A;
    U_n_const = use_table_eos ? rho_c * e_e_ref : rho_c * cv_mass * T_ref;
  }

  // Energy-conservation upper bracket. For conservative_active_set, A_total
  // is sum_g max(E_star_g, 0), the tight local-conservation bound.
  auto temperature_for_energy = [&](double U_target) {
    if (use_table_eos) {
      return invert_e_e_via_bisection<EOS_TAIL>(
          electron_eos, eos_rho_bracket,
          U_target / fmax(rho_c, 1.0e-300), temperature_floor_eV, rho_c,
          Zbar, A_c, low_density_extrap);
    }
    return (cv_mass > 0.0)
        ? U_target / fmax(rho_c * cv_mass, 1.0e-300)
        : temperature_floor_eV;
  };
  const double T_lo = temperature_floor_eV;
  double U_hi = 0.0;
  double T_hi = 0.0;
  if (threadIdx.x == 0) {
    U_hi = U_n_const + A_total;
    T_hi = temperature_for_energy(U_hi);
    T_hi = fmax(T_hi, T_lo);
  }

  // Validate bracket: evaluate R at endpoints
  if (threadIdx.x == 0) {
    s_eval_T = T_lo;
  }
  ResidualEval ev_lo = eval();
  if (threadIdx.x == 0) {
    s_eval_T = T_hi;
  }
  ResidualEval ev_hi = eval();

  bool bracket_valid = false;
  if (threadIdx.x == 0) {
    if (ev_lo.R > 0.0) {
      // The wrapper returns this flag to the SN stage, which forwards it to
      // State::sn_material_retry_flag for the driver retry (NUMERICS §6.8).
      if (!use_2d_legacy_closure && retry_flag != nullptr) {
        atomicOr(retry_flag, kRetryFloorRoot);
      }
      T = T_lo;
      s_floor_root = 1;
      s_bracket_valid = 0;
    } else {
      bracket_valid = ev_hi.R >= 0.0;
      s_floor_root = 0;
      s_bracket_valid = bracket_valid ? 1 : 0;
    }
  }
  __syncthreads();
  if (s_floor_root == 0) {
    double U_span = 0.0;
    if (threadIdx.x == 0) {
      s_expand_enabled = (!bracket_valid && !use_2d_legacy_closure) ? 1 : 0;
      if (s_expand_enabled != 0) {
        U_span = U_hi - U_n_const;
        if (!(U_span > 0.0) || !isfinite(U_span)) {
          U_span = fmax(fabs(U_n_const), 1.0) * 1.0e-12;
        }
      }
    }
    __syncthreads();
    if (s_expand_enabled != 0) {
      for (int expand = 0; expand < 20; ++expand) {
        if (threadIdx.x == 0) {
          if (ev_hi.R < 0.0) {
            U_span *= 2.0;
            U_hi = U_n_const + U_span;
            T_hi = fmax(temperature_for_energy(U_hi), T_lo);
            s_eval_T = T_hi;
            s_expand_active = 1;
          } else {
            s_expand_active = 0;
          }
        }
        __syncthreads();
        if (s_expand_active != 0) {
          ev_hi = eval();
        }
        __syncthreads();
      }
      if (threadIdx.x == 0) {
        bracket_valid = ev_hi.R >= 0.0;
        s_bracket_valid = bracket_valid ? 1 : 0;
        if (!bracket_valid && retry_flag != nullptr) {
          atomicOr(retry_flag, kRetryBracketExpansion);
        }
      }
    }
    __syncthreads();
    double Tlo = 0.0;
    double Thi = 0.0;
    double E_scale = 0.0;
    const double mach_eps = 2.220446049250313e-16;
    const int max_iter_local = 60;
    if (threadIdx.x == 0) {
      if (!bracket_valid) {
        // Legacy mixed-time behavior clamps to the energy bound. The
        // conservative path flags this for future global timestep rejection.
        T = T_hi;
      } else {
        // Valid bracket. Solve.
        Tlo = T_lo;
        Thi = T_hi;
        T = (T > Tlo && T < Thi) ? T : 0.5 * (Tlo + Thi);
        E_scale = fmax(fabs(A_total) + fabs(ev_init.P) + fabs(U_n_const), 1.0e-30);
        s_newton_active = 1;
      }
      s_bracket_valid = bracket_valid ? 1 : 0;
    }
    __syncthreads();
    if (s_bracket_valid != 0) {
      for (int iter = 0; iter < max_iter_local; ++iter) {
        if (threadIdx.x == 0) {
          s_newton_eval = (s_newton_active != 0) ? 1 : 0;
          if (s_newton_eval != 0) {
            s_eval_T = T;
          }
        }
        __syncthreads();
        if (s_newton_eval != 0) {
          ResidualEval ev = eval();
          if (threadIdx.x == 0) {
            s_trial_eval = 0;
            // Convergence: energy-residual or bracket adjacency
            if (fabs(ev.R) < 64.0 * mach_eps * E_scale) {
              s_newton_active = 0;
            } else if (Thi - Tlo < mach_eps * fmax(fabs(Thi), temperature_floor_eV)) {
              s_newton_active = 0;
            } else {
              // Newton step with bracket safeguard
              double T_trial;
              if (ev.Rprime > 0.0 && isfinite(ev.R / ev.Rprime)) {
                T_trial = T - ev.R / ev.Rprime;
                if (!(T_trial > Tlo && T_trial < Thi)) {
                  T_trial = 0.5 * (Tlo + Thi);
                }
              } else {
                T_trial = 0.5 * (Tlo + Thi);
              }
              s_T_trial = T_trial;
              s_eval_T = T_trial;
              s_trial_eval = 1;
            }
          }
          __syncthreads();
          if (s_trial_eval != 0) {
            ResidualEval ev_trial = eval();
            if (threadIdx.x == 0) {
              const double T_trial = s_T_trial;
              // Update bracket by sign of R(T_trial)
              if (ev_trial.R < 0.0) Tlo = T_trial;
              else                  Thi = T_trial;
              T = T_trial;
            }
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }
  __syncthreads();
  // === End Phase A replacement ===

  if (threadIdx.x == 0) {
    s_final_T = T;
    Te[c] = T;
  }
  __syncthreads();
  const double T_final = s_final_T;
  if (!use_2d_legacy_closure && rad_E_out != nullptr) {
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
      const int idx = c * n_groups + g;
      const double sigma_pa = nonnegative_finite(sigma_a[idx]);
      const double sigma_pe_g =
          (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
      const double lambda_pa = dt_safe * core::constants::c_light * sigma_pa;
      const double lambda_pe = dt_safe * core::constants::c_light * sigma_pe_g;
      const double b_prev =
          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_ref), 0.0);
      const double B_prev = core::constants::a_eV * safe_pow4(T_ref) * b_prev;
      const double E_post_old = nonnegative_finite(rad_E[idx]);
      const double E_star =
          use_E_star_override ? finite_or_zero(E_star_override[idx]) : 0.0;
      const double b_new =
          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_final), 0.0);
      const double B_new = core::constants::a_eV * safe_pow4(T_final) * b_new;
      const double S_dt = (source_ext != nullptr && g == 0)
                              ? dt_safe * source_ext[c]
                              : 0.0;
      if (diag_rad_absorption != nullptr) {
        diag_rad_absorption[idx] = lambda_pa * E_post_old;
      }
      if (diag_rad_emission_at_Tnp1 != nullptr) {
        diag_rad_emission_at_Tnp1[idx] = lambda_pe * B_new;
      }
      if (diag_clip_energy != nullptr) {
        const double E_pre_local_unclipped =
            E_post_old * (1.0 + lambda_pa) - lambda_pe * B_prev;
        diag_clip_energy[idx] =
            use_E_star_override
                ? 0.0
                : fmax(0.0, -E_pre_local_unclipped) / (1.0 + lambda_pa);
      }
      if (diag_clip_full_deficit != nullptr) {
        const double E_pre_local_unclipped =
            E_post_old * (1.0 + lambda_pa) - lambda_pe * B_prev;
        diag_clip_full_deficit[idx] =
            use_E_star_override ? 0.0 : fmax(0.0, -E_pre_local_unclipped);
      }
      if (use_E_star_override) {
        rad_E_out[idx] =
            sn_conservative_E_plus_from_star(E_star, lambda_pa, lambda_pe, B_new,
                                             S_dt);
      } else {
        const double r_g = lambda_pe / (1.0 + lambda_pa);
        rad_E_out[idx] =
            sn_conservative_E_plus(E_post_old, r_g, B_new, B_prev);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    if (use_table_eos) {
      if (ee != nullptr) {
        if constexpr (EOS_TAIL) {
          ee[c] = sn_eval_e_tail(electron_eos, eos_rho_bracket, rho_c,
                                 T_final, Zbar, A_c, low_density_extrap);
        } else {
          ee[c] = materials::device_eos_energy_extrap(
              electron_eos, eos_rho_bracket, rho_c, T_final, Zbar, A_c,
              low_density_extrap);
        }
      }
      if (Pe != nullptr) {
        if constexpr (EOS_TAIL) {
          Pe[c] = sn_eval_P_tail(electron_eos, eos_rho_bracket, rho_c,
                                 T_final, Zbar, A_c, low_density_extrap);
        } else {
          Pe[c] = materials::device_eos_pressure_extrap(
              electron_eos, eos_rho_bracket, rho_c, T_final, Zbar, A_c,
              low_density_extrap);
        }
      }
    } else {
      if (ee != nullptr && cv_mass > 0.0) {
        ee[c] = finite_or_zero(ee[c]) + cv_mass * (T_final - T_prev);
      }
      if (Pe != nullptr) {
        Pe[c] = fmax(gamma_c - 1.0, 1.0e-12) * rho_c *
                ((ee != nullptr) ? finite_or_zero(ee[c]) : cv_mass * T_final);
      }
    }
    if (delta_T_rel != nullptr) {
      delta_T_rel[c] = fabs(T_final - T_prev) / fmax(fabs(T_final), temperature_floor_eV);
    }
  }

  const double V = nonnegative_finite(vol[c]);
  const double T4 = safe_pow4(T_final);
  for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double sigma_pe_g =
        (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
    const double E = nonnegative_finite(rad_E[idx]);
    const double b =
        (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_final), 0.0);
    const double eta =
        core::constants::c_light * sigma_pe_g * core::constants::a_eV * T4 * b;
    if (use_2d_legacy_closure) {
      if (diag_rad_absorption != nullptr) {
        diag_rad_absorption[idx] =
            fmax(dt, 0.0) * core::constants::c_light * sigma_pa * E;
      }
      if (diag_rad_emission_at_Tnp1 != nullptr) {
        diag_rad_emission_at_Tnp1[idx] = fmax(dt, 0.0) * eta;
      }
      if (diag_clip_energy != nullptr) {
        diag_clip_energy[idx] = 0.0;
      }
      if (diag_clip_full_deficit != nullptr) {
        diag_clip_full_deficit[idx] = 0.0;
      }
    }
    if (rad_dep != nullptr) {
      rad_dep[idx] = fmax(dt, 0.0) * V * core::constants::c_light * sigma_pa * E;
    }
    if (rad_emit != nullptr) {
      rad_emit[idx] = fmax(dt, 0.0) * V * eta;
    }
  }
}

}  // namespace

materials::DeviceEOSTableView sn_electron_eos_device_view(
    const materials::EOSTableTriplet* tables) {
  if (tables == nullptr) {
    return materials::DeviceEOSTableView{};
  }
  return sn_electron_eos_table_cache().view_for(*tables);
}

int solve_sn_material_temperature_newton_gpu(
    const SnMaterialNewton1DInputs& in) {
  TENRYU_ASSERT(in.sigma_a != nullptr, "SN Newton requires sigma_a");
  TENRYU_ASSERT(in.rad_E != nullptr, "SN Newton requires rad_E");
  TENRYU_ASSERT(in.rad_E_out != nullptr,
                "SN Newton production closure requires rad_E_out");
  TENRYU_ASSERT(in.rho != nullptr, "SN Newton requires rho");
  TENRYU_ASSERT(in.vol != nullptr, "SN Newton requires vol");
  TENRYU_ASSERT(in.Te_old != nullptr, "SN Newton requires Te_old");
  TENRYU_ASSERT(in.Te != nullptr, "SN Newton requires Te");
  TENRYU_ASSERT(in.n_cells >= 0, "SN Newton requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 0, "SN Newton requires n_groups >= 0");
  if (in.n_cells == 0 || in.n_groups == 0) {
    return 0;
  }
  // Owned-window launch under MPI (spec §6i): default (0, 0) keeps the
  // full range for serial and 1D-replicated callers.
  const bool windowed = in.c_end > in.c_begin;
  const int win_begin = windowed ? in.c_begin : 0;
  const int blocks = windowed ? (in.c_end - in.c_begin) : in.n_cells;
  int* d_retry_flag = static_cast<int*>(
      tenryu::core::device_scratch_acquire("sn_newton:retry_flag", sizeof(int)));
  cuda_check(cudaMemset(d_retry_flag, 0, sizeof(int)),
             "SN Newton cudaMemset retry flag failed");
  if (in.energy_authoritative) {
    sn_material_newton_kernel<true><<<blocks, kNewtonThreads>>>(
        in.sigma_a,
        in.sigma_pe,
        in.rad_E,
        in.E_star_override,
        in.source_ext,
        in.rad_E_out,
        0,
        in.eos_low_density_extrap ? 1 : 0,
        in.rho,
        in.cv_e,
        in.vol,
        in.zbar,
        in.Te_old,
        in.Te,
        in.ee,
        in.Pe,
        in.rad_dep,
        in.rad_emit,
        in.diag_rad_emission_at_Tnp1,
        in.diag_rad_absorption,
        in.diag_clip_energy,
        in.diag_clip_full_deficit,
        in.delta_T_rel,
        in.planck,
        in.electron_eos,
        in.n_cells,
        in.n_groups,
        in.dt,
        in.cv_e_const,
        in.Cv_e_const,
        in.A_eff,
        in.gamma_eff,
        in.temperature_floor_eV,
        in.max_iterations,
        in.tolerance,
        d_retry_flag,
        win_begin);
  } else {
    sn_material_newton_kernel<false><<<blocks, kNewtonThreads>>>(
        in.sigma_a,
        in.sigma_pe,
        in.rad_E,
        in.E_star_override,
        in.source_ext,
        in.rad_E_out,
        0,
        in.eos_low_density_extrap ? 1 : 0,
        in.rho,
        in.cv_e,
        in.vol,
        in.zbar,
        in.Te_old,
        in.Te,
        in.ee,
        in.Pe,
        in.rad_dep,
        in.rad_emit,
        in.diag_rad_emission_at_Tnp1,
        in.diag_rad_absorption,
        in.diag_clip_energy,
        in.diag_clip_full_deficit,
        in.delta_T_rel,
        in.planck,
        in.electron_eos,
        in.n_cells,
        in.n_groups,
        in.dt,
        in.cv_e_const,
        in.Cv_e_const,
        in.A_eff,
        in.gamma_eff,
        in.temperature_floor_eV,
        in.max_iterations,
        in.tolerance,
        d_retry_flag,
        win_begin);
  }
  cuda_check(cudaGetLastError(), "SN Newton launch failed");
  int retry_flag = 0;
  cuda_check(cudaMemcpy(&retry_flag,
                        d_retry_flag,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "SN Newton copy retry flag failed");
  if ((retry_flag & kRetryFloorRoot) != 0) {
    core::log_warning(
        "SN conservative_active_set material coupling found R(T_floor)>0; "
        "requesting global timestep rejection (state was clamped to T_floor "
        "for this attempt).");
  }
  if ((retry_flag & kRetryBracketExpansion) != 0) {
    core::log_warning(
        "SN conservative_active_set material coupling failed adaptive bracket "
        "expansion; requesting global timestep rejection (local clamp applied "
        "for this attempt).");
  }
  return retry_flag;
}

void solve_sn_material_temperature_newton_2d_legacy_gpu(
    const SnMaterialNewton1DInputs& in) {
  TENRYU_ASSERT(in.sigma_a != nullptr, "SN Newton 2D legacy requires sigma_a");
  TENRYU_ASSERT(in.rad_E != nullptr, "SN Newton 2D legacy requires rad_E");
  TENRYU_ASSERT(in.rho != nullptr, "SN Newton 2D legacy requires rho");
  TENRYU_ASSERT(in.vol != nullptr, "SN Newton 2D legacy requires vol");
  TENRYU_ASSERT(in.Te_old != nullptr, "SN Newton 2D legacy requires Te_old");
  TENRYU_ASSERT(in.Te != nullptr, "SN Newton 2D legacy requires Te");
  TENRYU_ASSERT(in.n_cells >= 0, "SN Newton 2D legacy requires n_cells >= 0");
  TENRYU_ASSERT(in.n_groups >= 0, "SN Newton 2D legacy requires n_groups >= 0");
  if (in.n_cells == 0 || in.n_groups == 0) {
    return;
  }
  const bool windowed = in.c_end > in.c_begin;
  const int win_begin = windowed ? in.c_begin : 0;
  const int blocks = windowed ? (in.c_end - in.c_begin) : in.n_cells;
  if (in.energy_authoritative) {
    sn_material_newton_kernel<true><<<blocks, kNewtonThreads>>>(
        in.sigma_a,
        in.sigma_pe,
        in.rad_E,
        nullptr,
        in.source_ext,
        nullptr,
        1,
        in.eos_low_density_extrap ? 1 : 0,
        in.rho,
        in.cv_e,
        in.vol,
        in.zbar,
        in.Te_old,
        in.Te,
        in.ee,
        in.Pe,
        in.rad_dep,
        in.rad_emit,
        in.diag_rad_emission_at_Tnp1,
        in.diag_rad_absorption,
        in.diag_clip_energy,
        in.diag_clip_full_deficit,
        in.delta_T_rel,
        in.planck,
        in.electron_eos,
        in.n_cells,
        in.n_groups,
        in.dt,
        in.cv_e_const,
        in.Cv_e_const,
        in.A_eff,
        in.gamma_eff,
        in.temperature_floor_eV,
        in.max_iterations,
        in.tolerance,
        nullptr,
        win_begin);
  } else {
    sn_material_newton_kernel<false><<<blocks, kNewtonThreads>>>(
        in.sigma_a,
        in.sigma_pe,
        in.rad_E,
        nullptr,
        in.source_ext,
        nullptr,
        1,
        in.eos_low_density_extrap ? 1 : 0,
        in.rho,
        in.cv_e,
        in.vol,
        in.zbar,
        in.Te_old,
        in.Te,
        in.ee,
        in.Pe,
        in.rad_dep,
        in.rad_emit,
        in.diag_rad_emission_at_Tnp1,
        in.diag_rad_absorption,
        in.diag_clip_energy,
        in.diag_clip_full_deficit,
        in.delta_T_rel,
        in.planck,
        in.electron_eos,
        in.n_cells,
        in.n_groups,
        in.dt,
        in.cv_e_const,
        in.Cv_e_const,
        in.A_eff,
        in.gamma_eff,
        in.temperature_floor_eV,
        in.max_iterations,
        in.tolerance,
        nullptr,
        win_begin);
  }
  cuda_check(cudaGetLastError(), "SN Newton 2D legacy launch failed");
}

void solve_sn_material_temperature_newton_gpu(
    const SNMaterialCouplingGPUInputs& in,
    const int n_cells,
    const int n_groups,
    const double temperature_floor_eV) {
  SnMaterialNewton1DInputs wrapped{};
  wrapped.sigma_a = in.sigma_a;
  wrapped.rad_E = in.E_out;
  wrapped.rad_E_out = in.E_out;
  wrapped.rho = in.rho;
  wrapped.cv_e = in.cv_e;
  wrapped.vol = in.vol;
  wrapped.Te_old = in.Te_old;
  wrapped.Te = in.Te;
  wrapped.ee = in.ee;
  wrapped.rad_dep = in.rad_dep;
  wrapped.rad_emit = in.rad_emit;
  wrapped.planck = in.planck;
  wrapped.electron_eos = in.electron_eos;
  wrapped.n_cells = n_cells;
  wrapped.n_groups = n_groups;
  wrapped.dt = in.dt;
  wrapped.cv_e_const = in.cv_e_const;
  wrapped.Cv_e_const = in.Cv_e_const;
  wrapped.temperature_floor_eV = temperature_floor_eV;
  wrapped.c_begin = in.mpi_c_begin;
  wrapped.c_end = in.mpi_c_end;
  solve_sn_material_temperature_newton_gpu(wrapped);
}

}  // namespace tenryu::radiation
