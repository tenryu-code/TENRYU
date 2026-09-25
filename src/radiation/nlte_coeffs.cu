#include "radiation/nlte_coeffs.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>
#include <string>
#include <type_traits>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "materials/eos_cell_table_selector.cuh"
#include "materials/ionmix_reader.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

constexpr int kWarpSize = 32;
constexpr int kMaxGroupsPerBlock = 128;
constexpr double kEtaFloor = 1.0e-300;
constexpr double kKappaFloor = 1.0e-100;
constexpr double kSigmaDiagEps = 1.0e-30;
constexpr double kFleckTol = 1.0e-12;
constexpr double kCorrectedFleckDeltaRel = 1.0e-3;
constexpr double kCorrectedFleckSigmaFloor = 1.0e-30;
constexpr double kCorrectedFleckOnePlusXiMin = 0.1;
constexpr double kCorrectedFleckOnePlusXiMax = 10.0;

__device__ constexpr double kCLight = 2.99792458e10;
__device__ constexpr double kAeV = 1.3720e2;
__device__ constexpr double kEvToErg = 1.6022e-12;
__device__ constexpr double kProtonMass = 1.6726219e-24;

__device__ inline double safe_temperature_pow4_device(double temperature_eV) {
  if (!isfinite(temperature_eV) || !(temperature_eV > 0.0)) {
    return 0.0;
  }
  const double t = fmin(temperature_eV, 1.0e6);
  return t * t * t * t;
}

__device__ inline double safe_log_kappa_device(double kappa) {
  return log(fmax(kappa, kKappaFloor));
}

__device__ inline double planck_weight_device(const PlanckTableDeviceView& planck,
                                              int g,
                                              double temperature_eV,
                                              int n_groups,
                                              bool* invalid) {
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

__device__ inline double edge_log_slope_temperature_upper_device(
    const materials::IonmixOpacityDeviceView& table,
    const double* kappa_table,
    int group,
    double log_ni) {
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
  return (safe_log_kappa_device(k_hi) - safe_log_kappa_device(k_lo)) /
         (log_T_hi - log_T_lo);
}

__device__ inline double edge_log_slope_density_lower_device(
    const materials::IonmixOpacityDeviceView& table,
    const double* kappa_table,
    int group,
    double log_T) {
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
  return (safe_log_kappa_device(k_hi) - safe_log_kappa_device(k_lo)) /
         (log_ni_hi - log_ni_lo);
}

// Evaluation point of interpolate_kappa_with_smooth_edges: everything that
// depends on (ni, T) but not on the table or group, so the three opacity
// tables of one thread share one set of logs (2026-09-23; each evaluation
// used to recompute them). Values and order of operations are unchanged.
struct KappaEvalPoint {
  double log_ni_eval;
  double log_T_eval;
  double log_ni_edge;
  double log_T_edge;
  double log_kappa_min;
  double log_kappa_max;
  bool rarefied_extrapolation;
  bool hot_extrapolation;
};

__device__ inline KappaEvalPoint make_kappa_eval_point(
    const materials::IonmixOpacityDeviceView& table,
    double ni_cm3,
    double temperature_eV) {
  const double ni_min = exp(table.log_ni_min);
  double ni_eval = ni_cm3;
  if (!isfinite(ni_eval) || !(ni_eval > 0.0)) {
    ni_eval = ni_min;
  }
  double T_eval = temperature_eV;
  if (!isfinite(T_eval) || !(T_eval > 0.0)) {
    T_eval = table.T_min;
  }
  KappaEvalPoint pt;
  pt.log_ni_eval = log(ni_eval);
  pt.log_T_eval = log(T_eval);
  pt.rarefied_extrapolation = ni_eval < ni_min;
  pt.hot_extrapolation = T_eval > table.T_max;
  pt.log_ni_edge = fmin(fmax(pt.log_ni_eval, table.log_ni_min), table.log_ni_max);
  pt.log_T_edge = fmin(fmax(pt.log_T_eval, table.log_T_min), table.log_T_max);
  pt.log_kappa_min = log(kKappaFloor);
  pt.log_kappa_max = log(DBL_MAX);
  return pt;
}

__device__ inline double interpolate_kappa_at_point_device(
    const materials::IonmixOpacityDeviceView& table,
    const double* kappa_table,
    int group,
    const KappaEvalPoint& pt) {
  double log_kappa = safe_log_kappa_device(
      table.interpolate(kappa_table, group, pt.log_ni_edge, pt.log_T_edge));
  if (pt.rarefied_extrapolation) {
    const double slope_ni =
        edge_log_slope_density_lower_device(table, kappa_table, group, pt.log_T_edge);
    log_kappa += slope_ni * (pt.log_ni_eval - table.log_ni_min);
  }
  if (pt.hot_extrapolation) {
    const double slope_T =
        edge_log_slope_temperature_upper_device(table, kappa_table, group, pt.log_ni_edge);
    log_kappa += slope_T * (pt.log_T_eval - table.log_T_max);
  }
  log_kappa = fmin(fmax(log_kappa, pt.log_kappa_min), pt.log_kappa_max);
  return exp(log_kappa);
}

__device__ inline double interpolate_kappa_with_smooth_edges_device(
    const materials::IonmixOpacityDeviceView& table,
    const double* kappa_table,
    int group,
    double ni_cm3,
    double temperature_eV) {
  return interpolate_kappa_at_point_device(
      table, kappa_table, group, make_kappa_eval_point(table, ni_cm3, temperature_eV));
}

__device__ inline double warp_reduce_sum_device(double value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ inline double block_reduce_sum_device(double value, double* shared) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  value = warp_reduce_sum_device(value);
  if (lane == 0) {
    shared[warp] = value;
  }
  __syncthreads();

  const int n_warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
  value = (threadIdx.x < n_warps) ? shared[lane] : 0.0;
  if (warp == 0) {
    value = warp_reduce_sum_device(value);
  }
  if (threadIdx.x == 0) {
    shared[0] = value;
  }
  __syncthreads();
  // Read before the barrier: the next reduction's warp-0 partial goes to
  // shared[0] (compute-sanitizer racecheck, 2026-09-24).
  const double total = shared[0];
  __syncthreads();
  return total;
}

__device__ inline double block_inclusive_scan_device(double value, double* shared) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int offset = 1; offset < blockDim.x; offset <<= 1) {
    const double addend = (tid >= offset) ? shared[tid - offset] : 0.0;
    __syncthreads();
    if (tid >= offset) {
      shared[tid] += addend;
    }
    __syncthreads();
  }
  return shared[tid];
}

// Exponential-Rosenbrock form of the Fleck factor, f = (1 - exp(-z))/z
// (fld_1d_bodies.cuh fleck_form_phi1).
__device__ inline double nlte_fleck_form_phi1(const double z) {
  if (z < 1.0e-6) {
    return 1.0 - 0.5 * z + z * z / 6.0;
  }
  return (1.0 - exp(-z)) / z;
}

// kBetaMode: 0 tangent beta = 4 a T^3 / Cv_e, 1 secant beta over a local 0-D
// predictor, 2 guard (the larger of the two); kFormExp: 0 f = 1/(1+z), 1 the
// exponential form. Radiation.multigroup_diffusion.fleck_beta / fleck_form for
// table opacities (2026-09-24; the tangent 1/(1+z) instantiation is the
// historic kernel).
template <int kBetaMode, int kFormExp>
__global__ void compute_nlte_coefficients_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e_table,
    const std::uint8_t* __restrict__ cell_is_void,
    materials::IonmixOpacityDeviceView table,
    PlanckTableDeviceView planck,
    int n_cells,
    int n_groups,
    double dt,
    double A,
    double alpha,
    double /*f_min*/,
    double /*f_max*/,
    double /*fd_delta_rel*/,
    double /*fd_abs_min*/,
    double sigma_cap,
    double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1,
    bool linearized_planck,
    bool /*use_freeze_opacity*/,
    bool corrected_fleck,
    double* __restrict__ out_f,
    double* __restrict__ out_sigma_pa,
    double* __restrict__ out_sigma_pe,
    double* __restrict__ out_sigma_R,
    double* __restrict__ out_sigma_a_eff,
    double* __restrict__ out_sigma_s_eff,
    double* __restrict__ out_eta_cdf,
    double* __restrict__ out_eta,
    double* __restrict__ out_sigma_p_em,
    bool bypass_fleck,
    int low_density_extrap,
    int* __restrict__ d_negative_alpha_count,
    int* __restrict__ d_negative_eta_count,
    int* __restrict__ d_nan_inf_count,
    const int* __restrict__ cell_material_index,
    int material_filter,
    const materials::DeviceEOSTableView cv_electron_eos,
    const materials::CellEOSTableSelector cv_cell_tables,
    const int use_table_cv,
    const double* __restrict__ rad_E_old) {
  const int c = blockIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (cell_material_index != nullptr &&
      cell_material_index[c] != material_filter) {
    return;
  }

  __shared__ double s_reduce[kMaxGroupsPerBlock];
  __shared__ double s_scan[kMaxGroupsPerBlock];
  __shared__ double s_rho_c;
  __shared__ double s_T_eval;
  __shared__ double s_ni_c;
  __shared__ double s_T4;
  __shared__ double s_b_sum;
  __shared__ double s_eta_sum;
  __shared__ double s_sigma_p_abs;
  __shared__ double s_sigma_p_em;
  __shared__ double s_beta;
  __shared__ double s_f;
  __shared__ double s_T_plus;
  __shared__ int s_use_uniform_b;
  __shared__ int s_use_uniform_b_plus;
  __shared__ int s_has_emission;
  __shared__ int s_do_corrected_fleck;
  __shared__ int s_is_void;

  const int g = threadIdx.x;
  const bool active = g < n_groups;
  const int base = c * n_groups;

  if (threadIdx.x == 0) {
    s_is_void = (cell_is_void[c] != 0U) ? 1 : 0;
  }
  __syncthreads();

  if (s_is_void != 0) {
    if (threadIdx.x == 0) {
      out_sigma_p_em[c] = 0.0;
    }
    if (active) {
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
      out_eta_cdf[idx] = (n_groups == 1) ? 1.0 : (double(g + 1) / double(n_groups));
    }
    return;
  }

  if (threadIdx.x == 0) {
    s_rho_c = fmax(rho[c], 0.0);
    double T_eval = Te[c];
    if (!isfinite(T_eval) || !(T_eval > 0.0)) {
      T_eval = table.T_min;
    }
    s_T_eval = T_eval;

    double ni_c = exp(table.log_ni_min);
    if (s_rho_c > 0.0) {
      ni_c = s_rho_c / (A * kProtonMass);
      if (!isfinite(ni_c) || !(ni_c > 0.0)) {
        ni_c = exp(table.log_ni_min);
      }
    }
    s_ni_c = ni_c;
    s_T4 = safe_temperature_pow4_device(T_eval);
  }
  __syncthreads();

  const double rho_c = s_rho_c;
  const double T_eval = s_T_eval;
  const double ni_c = s_ni_c;
  const double T4 = s_T4;

  bool b_invalid = false;
  const double b_raw =
      active ? planck_weight_device(planck, g, T_eval, n_groups, &b_invalid) : 0.0;
  const double b_sum = block_reduce_sum_device(b_raw, s_reduce);
  const double b_invalid_count =
      block_reduce_sum_device((active && b_invalid) ? 1.0 : 0.0, s_reduce);
  if (threadIdx.x == 0) {
    s_b_sum = b_sum;
    s_use_uniform_b =
        (n_groups > 1) && (b_invalid_count > 0.0 || !(b_sum > 0.0) || !isfinite(b_sum));
  }
  __syncthreads();

  double sigma_p_abs_g = 0.0;
  double eta_g = 0.0;
  if (active) {
    const int idx = base + g;
    double b_g = 1.0;
    if (n_groups > 1) {
      b_g = (s_use_uniform_b != 0) ? (1.0 / double(n_groups)) : (b_raw / s_b_sum);
    }

    double sigma_pa_raw = 0.0;
    double sigma_pe_raw = 0.0;
    double sigma_R_raw = 0.0;
    if (rho_c > 0.0) {
      const KappaEvalPoint pt = make_kappa_eval_point(table, ni_c, T_eval);
      sigma_pa_raw = rho_c * interpolate_kappa_at_point_device(table, table.kappa_PA, g, pt);
      sigma_pe_raw = rho_c * interpolate_kappa_at_point_device(table, table.kappa_PE, g, pt);
      sigma_R_raw = rho_c * interpolate_kappa_at_point_device(table, table.kappa_R, g, pt);
    }

    if (!isfinite(sigma_pa_raw) || sigma_pa_raw < 0.0) {
      atomicAdd(d_negative_alpha_count, 1);
      if (!isfinite(sigma_pa_raw)) {
        atomicAdd(d_nan_inf_count, 1);
      }
    }
    if (!isfinite(sigma_pe_raw) || sigma_pe_raw < 0.0) {
      atomicAdd(d_negative_eta_count, 1);
      if (!isfinite(sigma_pe_raw)) {
        atomicAdd(d_nan_inf_count, 1);
      }
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
      atomicAdd(d_nan_inf_count, 1);
    }
    if (sigma_cap > 0.0) {
      sigma_pa = fmin(sigma_pa, sigma_cap);
      sigma_pe = fmin(sigma_pe, sigma_cap);
      sigma_R = fmin(sigma_R, sigma_cap);
    }
    if (low_density_extrap != 0) {
      const double rho_table_min = exp(table.log_ni_min) * A * kProtonMass;
      if (rho_c < rho_table_min && rho_table_min > 0.0) {
        const double rho_scale = rho_c / rho_table_min;
        sigma_pa *= rho_scale;
        sigma_pe *= rho_scale;
        sigma_R *= rho_scale;
      }
    }

    out_sigma_pa[idx] = sigma_pa;
    if (out_sigma_pe != nullptr) {
      out_sigma_pe[idx] = sigma_pe;
    }
    out_sigma_R[idx] = sigma_R;
    sigma_p_abs_g = b_g * sigma_pa;

    if (T_eval > temperature_floor_eV) {
      eta_g = sigma_pe * kCLight * kAeV * T4 * b_g;
    }
    if (!isfinite(eta_g) || eta_g < 0.0) {
      eta_g = 0.0;
      atomicAdd(d_negative_eta_count, 1);
      atomicAdd(d_nan_inf_count, 1);
    }
    out_eta[idx] = eta_g;
  }

  const double sigma_p_abs = block_reduce_sum_device(sigma_p_abs_g, s_reduce);
  const double eta_sum = block_reduce_sum_device(eta_g, s_reduce);

  if (threadIdx.x == 0) {
    double Cv_e = 0.0;
    // fleck_cv_source="table" (NUMERICS §6.7): the electron table's cv at the
    // linearization temperature, i.e. dU_e/dT_e of the energy function the
    // matter update advances; the same rule and priority as the constant-
    // opacity Fleck kernel (compute_fleck_for_fld_kernel_body).
    if (use_table_cv != 0) {
      const materials::DeviceEOSTableView eos =
          cv_cell_tables.electron(c, cv_electron_eos);
      if (eos.n_rho > 0 && eos.n_T > 0 && eos.P_table != nullptr &&
          eos.e_table != nullptr && eos.cv_table != nullptr) {
        const materials::RhoBracket br = materials::find_rho_bracket(eos, rho_c);
        const double cv_mass = materials::device_eos_cv(
            eos, br, log(fmax(T_eval, fmax(temperature_floor_eV, 1.0e-30))));
        if (isfinite(cv_mass) && cv_mass > 0.0) {
          Cv_e = fmax(rho_c, 0.0) * cv_mass;
        }
      }
    }
    if (!(Cv_e > 0.0)) {
      if (cv_e_override > 0.0) {
        Cv_e = cv_e_override;
      } else if (cv_e_table != nullptr && cv_e_table[c] > 0.0) {
        Cv_e = fmax(rho_c, 0.0) * cv_e_table[c];
      } else {
        Cv_e = rho_c * fmax(zbar[c], 0.0) * kEvToErg /
               (fmax(A, 1.0e-12) * kProtonMass * fmax(gamma_m1, 1.0e-12));
      }
    }
    Cv_e = fmax(Cv_e, 1.0e-30);

    double beta = 0.0;
    if (linearized_planck && cv_e_override > 0.0) {
      beta = 1.0;
    } else {
      beta = 4.0 * kAeV * T_eval * T_eval * T_eval / Cv_e;
    }
    // No beta <= 1 cap on the ideal-gas fallback (NUMERICS §6.7, 2026-07-26):
    // f = 1/(1+z) stays in (0, 1] for any z >= 0, and the cap only raised f
    // for hot low-Cv cells, weakening the Fleck linearization (2026-09-23:
    // the cap had survived in this kernel).
    if (!isfinite(beta) || beta < 0.0) {
      beta = 0.0;
    }

    double sigma_p_em = 0.0;
    double eta_sum_cell = eta_sum;
    const bool has_emission =
        (T_eval > temperature_floor_eV) && (eta_sum > kEtaFloor) && (T4 > 0.0);
    if (has_emission) {
      sigma_p_em = eta_sum / (kAeV * kCLight * T4);
    } else {
      eta_sum_cell = 0.0;
    }
    out_sigma_p_em[c] = sigma_p_em;

    double f = 1.0;
    if (!bypass_fleck && sigma_p_em > 0.0) {
      const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
      if constexpr (kBetaMode == 0 && kFormExp == 0) {
        f = 1.0 / (1.0 + alpha_safe * beta * kCLight * fmax(dt, 0.0) * sigma_p_em);
      } else {
        if constexpr (kBetaMode != 0) {
          // Secant beta = [B(T_pred) - B(T^n)] / [U_e(T_pred) - U_e(T^n)] with
          // a local 0-D predictor for T_pred and U_e from the electron table
          // the heat capacity comes from; any failing guard keeps the tangent
          // beta (the constant-opacity kernel's rule, fld_1d_bodies.cuh
          // compute_fleck_for_fld_kernel_body). Net heating of the predictor:
          // c (sigma_P,abs E - sigma_P,em a T^4).
          const materials::DeviceEOSTableView eos =
              (use_table_cv != 0) ? cv_cell_tables.electron(c, cv_electron_eos)
                                  : materials::DeviceEOSTableView{};
          if (rad_E_old != nullptr && eos.n_rho > 0 && eos.n_T > 0 &&
              eos.e_table != nullptr) {
            double E_grey = 0.0;
            for (int gg = 0; gg < n_groups; ++gg) {
              E_grey += fmax(rad_E_old[base + gg], 0.0);
            }
            const double T_n = fmax(T_eval, 1.0e-12);
            const double T2n = T_n * T_n;
            const double B_n = kAeV * T2n * T2n;
            const double z_tan = alpha_safe * beta * kCLight * fmax(dt, 0.0) * sigma_p_em;
            const double f_tan = 1.0 / (1.0 + fmax(z_tan, 0.0));
            double dT_est = f_tan * kCLight * (sigma_p_abs * E_grey - sigma_p_em * B_n) *
                            fmax(dt, 0.0) / Cv_e;
            const double dT_cap = 0.5 * T_n;
            dT_est = fmin(fmax(dT_est, -dT_cap), dT_cap);
            constexpr double kSecantMinDT = 1.0e-6;
            if (isfinite(dT_est) && fabs(dT_est) > kSecantMinDT * T_n) {
              const double T_floor = fmax(temperature_floor_eV, 1.0e-12);
              const double T_pred = fmax(T_n + dT_est, T_floor);
              if (T_pred != T_n) {
                const materials::RhoBracket br = materials::find_rho_bracket(eos, rho_c);
                const double e_n =
                    materials::device_eos_energy(eos, br, log(fmax(T_n, T_floor)));
                const double e_p =
                    materials::device_eos_energy(eos, br, log(fmax(T_pred, T_floor)));
                const double dU = fmax(rho_c, 0.0) * (e_p - e_n);
                const double Tp2 = T_pred * T_pred;
                const double dB = kAeV * (Tp2 * Tp2 - T2n * T2n);
                const double beta_sec = dB / dU;
                if (isfinite(beta_sec) && beta_sec > 0.0) {
                  beta = (kBetaMode == 2) ? fmax(beta, beta_sec) : beta_sec;
                }
              }
            }
          }
        }
        const double z = alpha_safe * beta * kCLight * fmax(dt, 0.0) * sigma_p_em;
        f = (kFormExp != 0) ? nlte_fleck_form_phi1(z) : 1.0 / (1.0 + z);
        if (!isfinite(f) || f < 0.0) {
          f = 0.0;
        }
        if (f > 1.0) {
          f = 1.0;
        }
      }
    }

    s_sigma_p_abs = sigma_p_abs;
    s_eta_sum = eta_sum_cell;
    s_sigma_p_em = sigma_p_em;
    s_beta = beta;
    s_f = f;
    s_has_emission = has_emission ? 1 : 0;
    s_do_corrected_fleck =
        (!bypass_fleck && corrected_fleck && sigma_p_em > kCorrectedFleckSigmaFloor)
            ? 1
            : 0;
    s_T_plus = T_eval * (1.0 + kCorrectedFleckDeltaRel);
  }
  __syncthreads();

  if (active && s_has_emission == 0) {
    out_eta[base + g] = 0.0;
  }

  if (s_do_corrected_fleck != 0) {
    bool b_plus_invalid = false;
    const double b_plus_raw =
        active ? planck_weight_device(planck, g, s_T_plus, n_groups, &b_plus_invalid) : 0.0;
    const double b_plus_sum = block_reduce_sum_device(b_plus_raw, s_reduce);
    const double b_plus_invalid_count =
        block_reduce_sum_device((active && b_plus_invalid) ? 1.0 : 0.0, s_reduce);
    if (threadIdx.x == 0) {
      s_b_sum = b_plus_sum;
      s_use_uniform_b_plus =
          (n_groups > 1) &&
          (b_plus_invalid_count > 0.0 || !(b_plus_sum > 0.0) || !isfinite(b_plus_sum));
    }
    __syncthreads();

    double spe_plus_g = 0.0;
    if (active) {
      double b_plus_g = 1.0;
      if (n_groups > 1) {
        b_plus_g =
            (s_use_uniform_b_plus != 0) ? (1.0 / double(n_groups)) : (b_plus_raw / s_b_sum);
      }
      double sigma_pe_plus = 0.0;
      if (rho_c > 0.0) {
        sigma_pe_plus = rho_c * interpolate_kappa_with_smooth_edges_device(
                                    table, table.kappa_PE, g, ni_c, s_T_plus);
      }
      if (!isfinite(sigma_pe_plus) || sigma_pe_plus < 0.0) {
        sigma_pe_plus = 0.0;
      }
      if (sigma_cap > 0.0) {
        sigma_pe_plus = fmin(sigma_pe_plus, sigma_cap);
      }
      spe_plus_g = sigma_pe_plus * b_plus_g;
    }

    const double spe_plus = block_reduce_sum_device(spe_plus_g, s_reduce);
    if (threadIdx.x == 0) {
      const double corrected_fleck_log_delta = log(1.0 + kCorrectedFleckDeltaRel);
      double xi = 0.0;
      if (spe_plus > kCorrectedFleckSigmaFloor) {
        xi = (log(spe_plus) - log(s_sigma_p_em)) / (4.0 * corrected_fleck_log_delta);
      }
      if (!isfinite(xi)) {
        xi = 0.0;
      }
      const double one_plus_xi =
          fmin(fmax(1.0 + xi, kCorrectedFleckOnePlusXiMin), kCorrectedFleckOnePlusXiMax);
      const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
      s_f = 1.0 / (1.0 + alpha_safe * s_beta * kCLight * fmax(dt, 0.0) *
                               s_sigma_p_em * one_plus_xi);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    double f = s_f;
    if (!isfinite(f) || f < -kFleckTol || f > 1.0 + kFleckTol) {
      atomicAdd(d_nan_inf_count, 1);
    }
    f = fmin(fmax(f, 0.0), 1.0);
    // Fleck-array layout contract: out_f is a [n_cells x n_groups] cg
    // array like every other per-group FLD field. The gray Fleck factor is
    // broadcast to all groups; the old per-cell write left the g > 0 slots
    // of the cg-indexed consumers stale for G > 1 (silent wrong blend).
    for (int g = 0; g < n_groups; ++g) {
      out_f[c * n_groups + g] = f;
    }
    s_f = f;

    const double gamma_diag = s_sigma_p_em / fmax(s_sigma_p_abs, kSigmaDiagEps);
    if (!isfinite(gamma_diag)) {
      atomicAdd(d_nan_inf_count, 1);
    }
  }
  __syncthreads();

  double cdf_weight = 0.0;
  if (active) {
    const int idx = base + g;
    if (bypass_fleck) {
      out_sigma_a_eff[idx] = out_sigma_pa[idx];
      out_sigma_s_eff[idx] = 0.0;
    } else {
      out_sigma_a_eff[idx] = s_f * out_sigma_pa[idx];
      out_sigma_s_eff[idx] = (1.0 - s_f) * out_sigma_pa[idx];
    }
    if (s_eta_sum > kEtaFloor) {
      cdf_weight = fmax(out_eta[idx], 0.0) / s_eta_sum;
    }
  }

  const double cdf_value = block_inclusive_scan_device(cdf_weight, s_scan);
  if (active) {
    const int idx = base + g;
    if (s_eta_sum > kEtaFloor) {
      out_eta_cdf[idx] = fmin(fmax(cdf_value, 0.0), 1.0);
    } else {
      out_eta_cdf[idx] = (n_groups == 1) ? 1.0 : (double(g + 1) / double(n_groups));
    }
    if (g == n_groups - 1) {
      out_eta_cdf[idx] = 1.0;
    }
  }
}

inline void nlte_cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    TENRYU_ASSERT(false, std::string("NLTE CUDA error: ") + msg + " (" +
                              cudaGetErrorString(err) + ")");
  }
}

}  // namespace

namespace {

NlteCoeffsDeviceResult compute_nlte_coefficients_cuda_impl(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells,
    int n_groups,
    double dt,
    double A,
    double alpha,
    double f_min,
    double f_max,
    double fd_delta_rel,
    double fd_abs_min,
    double sigma_cap,
    double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1,
    bool linearized_planck,
    bool use_freeze_opacity,
    bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa,
    double* d_sigma_pe,
    double* d_sigma_R,
    double* d_sigma_a_eff,
    double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_sigma_p_em,
    bool bypass_fleck,
    cudaStream_t stream,
    bool low_density_extrap = false,
    int* pinned_clamp_counts = nullptr,
    bool reuse_device_void_mask = false,
    const int* cell_material_index = nullptr,
    int material_filter = -1,
    const materials::DeviceEOSTableView* table_cv_electron_eos = nullptr,
    const materials::CellEOSTableSelector* table_cv_cell_tables = nullptr,
    bool accumulate_clamp_counts = false,
    const double* d_rad_E_old = nullptr,
    int fleck_beta_mode = 0,
    int fleck_form_exp = 0) {
  (void)f_min;
  (void)f_max;
  (void)fd_delta_rel;
  (void)fd_abs_min;
  (void)use_freeze_opacity;
  TENRYU_ASSERT(n_cells >= 0, "compute_nlte_coefficients_cuda: n_cells < 0");
  TENRYU_ASSERT(n_groups >= 1, "compute_nlte_coefficients_cuda: n_groups < 1");
  NlteCoeffsDeviceResult result{};
  if (n_cells == 0) {
    return result;
  }
  TENRYU_ASSERT(n_groups <= kMaxGroupsPerBlock,
                "compute_nlte_coefficients_cuda: n_groups > 128 unsupported");
  TENRYU_ASSERT(cell_is_void_host != nullptr,
                "compute_nlte_coefficients_cuda: cell_is_void_host null");
  TENRYU_ASSERT(cell_is_void_size >= static_cast<std::size_t>(n_cells),
                "compute_nlte_coefficients_cuda: cell_is_void_size mismatch");
  TENRYU_ASSERT(table_view.ngroups == n_groups,
                "compute_nlte_coefficients_cuda: IONMIX/group mismatch");
  TENRYU_ASSERT(planck_view.n_groups == n_groups,
                "compute_nlte_coefficients_cuda: Planck/group mismatch");

  std::uint8_t* d_void = nullptr;
  int* d_clamp_counts = nullptr;

  d_void = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("nlte:cell_is_void",
                                   sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)));
  if (!reuse_device_void_mask) {
    nlte_cuda_check(cudaMemcpyAsync(d_void,
                                    cell_is_void_host,
                                    sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells),
                                    cudaMemcpyHostToDevice,
                                    stream),
                    "copy cell_is_void H2D");
  }
  d_clamp_counts = static_cast<int*>(
      core::device_scratch_acquire("nlte:clamp_counts", 3 * sizeof(int)));
  int* const d_negative_alpha = &d_clamp_counts[0];
  int* const d_negative_eta = &d_clamp_counts[1];
  int* const d_nan_inf = &d_clamp_counts[2];
  if (!accumulate_clamp_counts) {
    nlte_cuda_check(cudaMemsetAsync(d_clamp_counts, 0, 3 * sizeof(int), stream),
                    "memset clamp_counts");
  }

  const int groups_for_block = std::min(n_groups, kMaxGroupsPerBlock);
  const int groups_per_block =
      ((groups_for_block + kWarpSize - 1) / kWarpSize) * kWarpSize;
  const auto launch = [&](auto beta_mode, auto form_exp) {
  compute_nlte_coefficients_kernel<decltype(beta_mode)::value, decltype(form_exp)::value>
      <<<n_cells, groups_per_block, 0, stream>>>(
      d_rho,
      d_Te,
      d_zbar,
      d_cv_e,
      d_void,
      table_view,
      planck_view,
      n_cells,
      n_groups,
      dt,
      A,
      alpha,
      f_min,
      f_max,
      fd_delta_rel,
      fd_abs_min,
      sigma_cap,
      cv_e_override,
      temperature_floor_eV,
      gamma_m1,
      linearized_planck,
      use_freeze_opacity,
      corrected_fleck,
      d_f,
      d_sigma_pa,
      d_sigma_pe,
      d_sigma_R,
      d_sigma_a_eff,
      d_sigma_s_eff,
      d_eta_cdf,
      d_eta,
      d_sigma_p_em,
      bypass_fleck,
      low_density_extrap ? 1 : 0,
      d_negative_alpha,
      d_negative_eta,
      d_nan_inf,
      cell_material_index,
      material_filter,
      (table_cv_electron_eos != nullptr) ? *table_cv_electron_eos
                                         : materials::DeviceEOSTableView{},
      (table_cv_cell_tables != nullptr) ? *table_cv_cell_tables
                                        : materials::CellEOSTableSelector{},
      (table_cv_electron_eos != nullptr) ? 1 : 0,
      d_rad_E_old);
  };
  using I0 = std::integral_constant<int, 0>;
  using I1 = std::integral_constant<int, 1>;
  using I2 = std::integral_constant<int, 2>;
  const int beta_mode = (d_rad_E_old != nullptr) ? fleck_beta_mode : 0;
  if (beta_mode == 1) {
    fleck_form_exp != 0 ? launch(I1{}, I1{}) : launch(I1{}, I0{});
  } else if (beta_mode == 2) {
    fleck_form_exp != 0 ? launch(I2{}, I1{}) : launch(I2{}, I0{});
  } else {
    fleck_form_exp != 0 ? launch(I0{}, I1{}) : launch(I0{}, I0{});
  }
  nlte_cuda_check(cudaGetLastError(), "kernel launch");

  if (pinned_clamp_counts != nullptr) {
    nlte_cuda_check(cudaMemcpyAsync(pinned_clamp_counts, d_clamp_counts,
                                    3 * sizeof(int), cudaMemcpyDeviceToHost, stream),
                    "copy clamp_counts D2H");
    result.negative_alpha_clamp_count = 0;
    result.negative_eta_clamp_count = 0;
    result.nan_inf_count = 0;
  } else {
    nlte_cuda_check(cudaMemcpyAsync(&result.negative_alpha_clamp_count,
                                    d_negative_alpha,
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "copy negative_alpha D2H");
    nlte_cuda_check(cudaMemcpyAsync(&result.negative_eta_clamp_count,
                                    d_negative_eta,
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "copy negative_eta D2H");
    nlte_cuda_check(cudaMemcpyAsync(&result.nan_inf_count,
                                    d_nan_inf,
                                    sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    stream),
                    "copy nan_inf D2H");
    nlte_cuda_check(cudaStreamSynchronize(stream), "stream sync");
  }

  return result;
}

}  // namespace

NlteCoeffsDeviceResult compute_nlte_coefficients_cuda(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells,
    int n_groups,
    double dt,
    double A,
    double alpha,
    double f_min,
    double f_max,
    double fd_delta_rel,
    double fd_abs_min,
    double sigma_cap,
    double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1,
    bool linearized_planck,
    bool use_freeze_opacity,
    bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa,
    double* d_sigma_R,
    double* d_sigma_a_eff,
    double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_sigma_p_em,
    cudaStream_t stream,
    bool low_density_extrap,
    const int* cell_material_index,
    int material_filter) {
  return compute_nlte_coefficients_cuda_impl(
      d_rho,
      d_Te,
      d_zbar,
      d_cv_e,
      cell_is_void_host,
      cell_is_void_size,
      table_view,
      planck_view,
      n_cells,
      n_groups,
      dt,
      A,
      alpha,
      f_min,
      f_max,
      fd_delta_rel,
      fd_abs_min,
      sigma_cap,
      cv_e_override,
      temperature_floor_eV,
      gamma_m1,
      linearized_planck,
      use_freeze_opacity,
      corrected_fleck,
      d_f,
      d_sigma_pa,
      nullptr,
      d_sigma_R,
      d_sigma_a_eff,
      d_sigma_s_eff,
      d_eta_cdf,
      d_eta,
      d_sigma_p_em,
      false,
      stream,
      low_density_extrap,
      nullptr,
      false,
      cell_material_index,
      material_filter);
}

NlteCoeffsDeviceResult compute_nlte_coefficients_cuda_with_pe(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells,
    int n_groups,
    double dt,
    double A,
    double alpha,
    double f_min,
    double f_max,
    double fd_delta_rel,
    double fd_abs_min,
    double sigma_cap,
    double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1,
    bool linearized_planck,
    bool use_freeze_opacity,
    bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa,
    double* d_sigma_pe,
    double* d_sigma_R,
    double* d_sigma_a_eff,
    double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_sigma_p_em,
    cudaStream_t stream,
    bool low_density_extrap,
    int* pinned_clamp_counts,
    bool reuse_device_void_mask,
    const int* cell_material_index,
    int material_filter,
    const materials::DeviceEOSTableView* table_cv_electron_eos,
    const materials::CellEOSTableSelector* table_cv_cell_tables,
    bool accumulate_clamp_counts,
    const double* d_rad_E_old,
    int fleck_beta_mode,
    int fleck_form_exp) {
  return compute_nlte_coefficients_cuda_impl(
      d_rho,
      d_Te,
      d_zbar,
      d_cv_e,
      cell_is_void_host,
      cell_is_void_size,
      table_view,
      planck_view,
      n_cells,
      n_groups,
      dt,
      A,
      alpha,
      f_min,
      f_max,
      fd_delta_rel,
      fd_abs_min,
      sigma_cap,
      cv_e_override,
      temperature_floor_eV,
      gamma_m1,
      linearized_planck,
      use_freeze_opacity,
      corrected_fleck,
      d_f,
      d_sigma_pa,
      d_sigma_pe,
      d_sigma_R,
      d_sigma_a_eff,
      d_sigma_s_eff,
      d_eta_cdf,
      d_eta,
      d_sigma_p_em,
      false,
      stream,
      low_density_extrap,
      pinned_clamp_counts,
      reuse_device_void_mask,
      cell_material_index,
      material_filter,
      table_cv_electron_eos,
      table_cv_cell_tables,
      accumulate_clamp_counts,
      d_rad_E_old,
      fleck_beta_mode,
      fleck_form_exp);
}

NlteCoeffsDeviceResult compute_nlte_coefficients_cuda_pure_sn(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells,
    int n_groups,
    double dt,
    double A,
    double alpha,
    double f_min,
    double f_max,
    double fd_delta_rel,
    double fd_abs_min,
    double sigma_cap,
    double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1,
    bool linearized_planck,
    bool use_freeze_opacity,
    bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa,
    double* d_sigma_pe,
    double* d_sigma_R,
    double* d_sigma_a_eff,
    double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_sigma_p_em,
    cudaStream_t stream,
    bool low_density_extrap,
    const int* cell_material_index,
    int material_filter) {
  TENRYU_ASSERT(d_sigma_pe != nullptr,
                "compute_nlte_coefficients_cuda_pure_sn: sigma_pe output null");
  return compute_nlte_coefficients_cuda_impl(
      d_rho,
      d_Te,
      d_zbar,
      d_cv_e,
      cell_is_void_host,
      cell_is_void_size,
      table_view,
      planck_view,
      n_cells,
      n_groups,
      dt,
      A,
      alpha,
      f_min,
      f_max,
      fd_delta_rel,
      fd_abs_min,
      sigma_cap,
      cv_e_override,
      temperature_floor_eV,
      gamma_m1,
      linearized_planck,
      use_freeze_opacity,
      corrected_fleck,
      d_f,
      d_sigma_pa,
      d_sigma_pe,
      d_sigma_R,
      d_sigma_a_eff,
      d_sigma_s_eff,
      d_eta_cdf,
      d_eta,
      d_sigma_p_em,
      true,
      stream,
      low_density_extrap,
      nullptr,
      false,
      cell_material_index,
      material_filter);
}

}  // namespace tenryu::radiation
