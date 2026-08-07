#pragma once

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "materials/ionmix_reader.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation::fleck_bodies {

constexpr int kDtRadOpacityConstant = 0;
constexpr int kDtRadOpacityFreqDepMarshak = 1;
constexpr int kDtRadOpacityTableNLTE = 2;
constexpr int kReduceBlock = 256;
constexpr double kDtRadKappaFloor = 1.0e-100;

struct DtRadCellCandidate {
  double dt_rad = 0.0;
  double sigma_P = 0.0;
  double beta = 0.0;
  double Cv_e = 0.0;
  double Te = 0.0;
  double rho = 0.0;
  int cell = -1;
};

struct DtRadLimitDeviceResult {
  double dt_rad = 0.0;
  double max_sigma_P = 0.0;
  double sigma_P = 0.0;
  double beta = 0.0;
  double Cv_e = 0.0;
  double Te = 0.0;
  double rho = 0.0;
  int limiting_cell = -1;
};

__device__ inline bool dt_rad_limit_better(
    const DtRadLimitDeviceResult& candidate,
    const DtRadLimitDeviceResult& current) {
  return candidate.dt_rad < current.dt_rad ||
         (candidate.dt_rad == current.dt_rad &&
          candidate.limiting_cell >= 0 &&
          current.limiting_cell >= 0 &&
          candidate.limiting_cell < current.limiting_cell);
}

__device__ inline double dt_rad_std_max(const double a, const double b) {
  return (a < b) ? b : a;
}

__device__ inline double dt_rad_std_min(const double a, const double b) {
  return (b < a) ? b : a;
}

__device__ inline double dt_rad_planck_weight(const double E_eV,
                                              const double T_eV) {
  if (!(E_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = E_eV / T_eV;
  if (x > 700.0) {
    return E_eV * E_eV * E_eV * exp(-x);
  }
  const double denom = expm1(x);
  if (!(denom > 0.0)) {
    return 0.0;
  }
  return (E_eV * E_eV * E_eV) / denom;
}

__device__ inline double dt_rad_freq_dep_sigma_nu(const double E_eV,
                                                  const double T_eV) {
  if (!(E_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = E_eV / T_eV;
  const double one_minus_exp = (x > 700.0) ? 1.0 : (-expm1(-x));
  return (27.0e9 / (E_eV * E_eV * E_eV)) * one_minus_exp;
}

__device__ inline double dt_rad_freq_dep_sigma_p(const double E_min_eV,
                                                 const double E_max_eV,
                                                 const double T_eV) {
  if (!(T_eV > 0.0) || !(E_max_eV > E_min_eV) || !(E_min_eV > 0.0)) {
    return 0.0;
  }
  constexpr int n = 128;
  const double log_x0 = log(E_min_eV);
  const double log_x1 = log(E_max_eV);
  const double h = (log_x1 - log_x0) / static_cast<double>(n);
  double num = 0.0;
  double den = 0.0;
  for (int i = 0; i <= n; ++i) {
    const double z = log_x0 + h * static_cast<double>(i);
    const double E = exp(z);
    const double w = (i == 0 || i == n) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
    const double planck_w = dt_rad_planck_weight(E, T_eV);
    num += w * dt_rad_freq_dep_sigma_nu(E, T_eV) * planck_w * E;
    den += w * planck_w * E;
  }
  num *= h / 3.0;
  den *= h / 3.0;
  return (den > 0.0) ? dt_rad_std_max(0.0, num / den) : 0.0;
}

__device__ inline double dt_rad_safe_log_kappa(const double kappa) {
  return log(dt_rad_std_max(kappa, kDtRadKappaFloor));
}

__device__ inline double dt_rad_interpolate_kappa_smooth(
    const materials::IonmixOpacityDeviceView& table,
    const double* __restrict__ kappa_table,
    const int group,
    const double ni_cm3,
    const double T_eV) {
  if (kappa_table == nullptr || table.ntemp <= 0 || table.ndens <= 0) {
    return 0.0;
  }
  const double ni_min = exp(table.log_ni_min);
  const double ni_max = exp(table.log_ni_max);
  const double T_min = table.T_min;
  const double T_max = table.T_max;

  double ni_eval = ni_cm3;
  if (!isfinite(ni_eval) || !(ni_eval > 0.0)) {
    ni_eval = ni_min;
  }
  double T_eval = T_eV;
  if (!isfinite(T_eval) || !(T_eval > 0.0)) {
    T_eval = T_min;
  }

  const bool rarefied_extrapolation = ni_eval < ni_min;
  const bool hot_extrapolation = T_eval > T_max;
  const double ni_edge = fmin(fmax(ni_eval, ni_min), ni_max);
  const double T_edge = fmin(fmax(T_eval, T_min), T_max);

  double log_kappa = dt_rad_safe_log_kappa(
      table.interpolate(kappa_table, group, log(ni_edge), log(T_edge)));
  if (rarefied_extrapolation && table.ndens >= 2) {
    const double ni_lo = exp(table.log_numdens[0]);
    const double ni_hi = exp(table.log_numdens[1]);
    if (ni_hi > ni_lo) {
      const double k_lo =
          table.interpolate(kappa_table, group, log(ni_lo), log(T_edge));
      const double k_hi =
          table.interpolate(kappa_table, group, log(ni_hi), log(T_edge));
      const double slope =
          (dt_rad_safe_log_kappa(k_hi) - dt_rad_safe_log_kappa(k_lo)) /
          (log(ni_hi) - log(ni_lo));
      log_kappa += slope * (log(ni_eval) - log(ni_min));
    }
  }
  if (hot_extrapolation && table.ntemp >= 2) {
    const double T_hi = exp(table.log_temps[table.ntemp - 1]);
    const double T_lo = exp(table.log_temps[table.ntemp - 2]);
    if (T_hi > T_lo) {
      const double k_hi =
          table.interpolate(kappa_table, group, log(ni_edge), log(T_hi));
      const double k_lo =
          table.interpolate(kappa_table, group, log(ni_edge), log(T_lo));
      const double slope =
          (dt_rad_safe_log_kappa(k_hi) - dt_rad_safe_log_kappa(k_lo)) /
          (log(T_hi) - log(T_lo));
      log_kappa += slope * (log(T_eval) - log(T_max));
    }
  }

  const double log_kappa_min = log(kDtRadKappaFloor);
  const double log_kappa_max = log(DBL_MAX);
  if (log_kappa < log_kappa_min) {
    log_kappa = log_kappa_min;
  } else if (log_kappa > log_kappa_max) {
    log_kappa = log_kappa_max;
  }
  return exp(log_kappa);
}

__device__ inline double dt_rad_nlte_sigma_p_em_block(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const std::uint8_t* __restrict__ cell_is_void,
    const materials::IonmixOpacityDeviceView table,
    const PlanckTableDeviceView planck,
    const int c,
    const int n_groups,
    const double A,
    const double sigma_cap,
    const double T_floor,
    const int tid,
    const int block_size,
    double* __restrict__ s_b_g,
    double* __restrict__ s_sigma_pe_g,
    double* __restrict__ s_rho_c,
    double* __restrict__ s_T_eval,
    double* __restrict__ s_ni_c,
    double* __restrict__ s_b_sum,
    double* __restrict__ s_inv_b_sum,
    double* __restrict__ s_sigma_p_em,
    int* __restrict__ s_use_uniform,
    int* __restrict__ s_skip) {
  if (tid == 0) {
    *s_sigma_p_em = 0.0;
    *s_skip = 0;
    if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
      *s_skip = 1;
    } else {
      const double rho_c = dt_rad_std_max(rho[c], 0.0);
      if (!(rho_c > 0.0)) {
        *s_skip = 1;
      } else {
        double T_eval = Te[c];
        if (!isfinite(T_eval) || !(T_eval > 0.0)) {
          T_eval = table.T_min;
        }
        if (!(T_eval > T_floor)) {
          *s_skip = 1;
        } else {
          double ni_c = rho_c / (A * tenryu::core::constants::proton_mass);
          if (!isfinite(ni_c) || !(ni_c > 0.0)) {
            ni_c = exp(table.log_ni_min);
          }
          *s_rho_c = rho_c;
          *s_T_eval = T_eval;
          *s_ni_c = ni_c;
        }
      }
    }
  }
  __syncthreads();
  if (*s_skip != 0) {
    return 0.0;
  }

  const double T_eval = *s_T_eval;
  for (int g = tid; g < n_groups; g += block_size) {
    s_b_g[g] = (n_groups == 1) ? 1.0
                               : dt_rad_std_max(planck.interpolate_b(g, T_eval),
                                                 0.0);
  }
  __syncthreads();

  if (tid == 0) {
    double b_sum = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      b_sum += s_b_g[g];
    }
    const bool use_uniform_b = !(b_sum > 0.0) || !isfinite(b_sum);
    *s_b_sum = b_sum;
    *s_inv_b_sum = use_uniform_b ? 0.0 : (1.0 / b_sum);
    *s_use_uniform = use_uniform_b ? 1 : 0;
  }
  __syncthreads();

  const double rho_c = *s_rho_c;
  const double ni_c = *s_ni_c;
  const double uniform_b = 1.0 / static_cast<double>(n_groups);
  for (int g = tid; g < n_groups; g += block_size) {
    const double b_g =
        (*s_use_uniform != 0) ? uniform_b : s_b_g[g] * (*s_inv_b_sum);
    double sigma_pe =
        rho_c * dt_rad_interpolate_kappa_smooth(table, table.kappa_PE, g, ni_c,
                                                T_eval);
    if (!isfinite(sigma_pe) || sigma_pe < 0.0) {
      sigma_pe = 0.0;
    }
    if (sigma_cap > 0.0) {
      sigma_pe = fmin(sigma_pe, sigma_cap);
    }
    s_sigma_pe_g[g] = sigma_pe * b_g;
  }
  __syncthreads();

  if (tid == 0) {
    double sigma_p_em = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      sigma_p_em += s_sigma_pe_g[g];
    }
    *s_sigma_p_em = sigma_p_em;
  }
  __syncthreads();
  return *s_sigma_p_em;
}

__device__ inline void compute_dt_rad_candidates_kernel_body(
    const int c,
    const int tid,
    double* smem,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const int state_cv_e_size,
    const std::uint8_t* __restrict__ cell_is_void,
    const materials::IonmixOpacityDeviceView table,
    const PlanckTableDeviceView planck,
    DtRadCellCandidate* __restrict__ candidates,
    const int n_cells,
    const int n_groups,
    const int opacity_model,
    const double kappa_a,
    const double E_min_eV,
    const double E_max_eV,
    const double sigma_cap,
    const double T_floor,
    const double rho_floor,
    const double alpha,
    const double f_min,
    const double cv_e_override,
    const double gm1,
    const double A,
    const int linearized_planck,
    const double infinity) {
  double* s_b_g = smem;
  double* s_sigma_pe_g = smem + n_groups;

  __shared__ DtRadCellCandidate s_candidate;
  __shared__ double s_rho_raw;
  __shared__ double s_Te_c;
  __shared__ double s_sigma_P;
  __shared__ double s_nlte_rho_c;
  __shared__ double s_nlte_T_eval;
  __shared__ double s_nlte_ni_c;
  __shared__ double s_b_sum;
  __shared__ double s_inv_b_sum;
  __shared__ double s_sigma_p_em;
  __shared__ int s_use_uniform;
  __shared__ int s_nlte_skip;
  __shared__ int s_done;

  if (tid == 0) {
    s_candidate.dt_rad = infinity;
    s_candidate.sigma_P = 0.0;
    s_candidate.beta = 0.0;
    s_candidate.Cv_e = 0.0;
    s_candidate.Te = 0.0;
    s_candidate.rho = 0.0;
    s_candidate.cell = c;
    s_done = 0;

    const double rho_raw = rho[c];
    s_rho_raw = rho_raw;
    if (rho_raw < rho_floor) {
      candidates[c] = s_candidate;
      s_done = 1;
    }
  }
  __syncthreads();
  if (s_done != 0) {
    return;
  }

  if (opacity_model == kDtRadOpacityFreqDepMarshak) {
    if (tid == 0) {
      const double Te_c = dt_rad_std_max(Te[c], T_floor);
      s_Te_c = Te_c;
      double sigma_P = 0.0;
      if (Te_c > 0.0) {
        sigma_P = dt_rad_freq_dep_sigma_p(E_min_eV, E_max_eV, Te_c);
      }
      s_sigma_P = sigma_P;
    }
  } else if (opacity_model == kDtRadOpacityTableNLTE) {
    if (tid == 0) {
      s_Te_c = dt_rad_std_max(Te[c], T_floor);
    }
    const double sigma_P = dt_rad_std_max(
        dt_rad_nlte_sigma_p_em_block(rho, Te, cell_is_void, table, planck, c,
                                     n_groups, A, sigma_cap, T_floor, tid,
                                     static_cast<int>(blockDim.x), s_b_g,
                                     s_sigma_pe_g, &s_nlte_rho_c,
                                     &s_nlte_T_eval, &s_nlte_ni_c, &s_b_sum,
                                     &s_inv_b_sum, &s_sigma_p_em,
                                     &s_use_uniform, &s_nlte_skip),
        0.0);
    if (tid == 0) {
      s_sigma_P = sigma_P;
    }
  } else {
    if (tid == 0) {
      const double Te_c = dt_rad_std_max(Te[c], T_floor);
      s_Te_c = Te_c;
      s_sigma_P = dt_rad_std_max(0.0, s_rho_raw) * kappa_a;
    }
  }
  __syncthreads();

  if (tid == 0) {
    const double sigma_P = s_sigma_P;
    s_candidate.sigma_P = sigma_P;
    if (sigma_P <= 0.0) {
      candidates[c] = s_candidate;
      s_done = 1;
    } else {
      s_done = 0;
    }
  }
  __syncthreads();
  if (s_done != 0) {
    return;
  }

  if (tid == 0) {
    const double rho_raw = s_rho_raw;
    const double Te_c = s_Te_c;
    const double sigma_P = s_sigma_P;
    const bool has_state_cv_e_cell =
        state_cv_e != nullptr && c < state_cv_e_size && state_cv_e[c] > 0.0;
    double Cv_e = -1.0;
    if (cv_e_override > 0.0) {
      Cv_e = cv_e_override;
    } else if (has_state_cv_e_cell) {
      Cv_e = dt_rad_std_max(rho_raw, 0.0) * state_cv_e[c];
    } else {
      const double z = dt_rad_std_max(zbar[c], 0.0);
      const double cv_mass_e = z * tenryu::core::constants::eV_to_erg /
                               (A * tenryu::core::constants::proton_mass * gm1);
      Cv_e = dt_rad_std_max(rho_raw, 0.0) * cv_mass_e;
    }
    Cv_e = dt_rad_std_max(Cv_e, 1.0e-30);

    double beta = 0.0;
    if (linearized_planck) {
      beta = 1.0;
    } else {
      beta = 4.0 * tenryu::core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
    }
    if (cv_e_override <= 0.0 && !has_state_cv_e_cell) {
      beta = dt_rad_std_min(beta, 1.0);
    }
    const double denom =
        f_min * alpha * tenryu::core::constants::c_light * beta * sigma_P;
    if (denom <= 0.0) {
      candidates[c] = s_candidate;
      return;
    }

    const double dt_c = (1.0 - f_min) / denom;
    if (dt_c < infinity) {
      s_candidate.dt_rad = dt_c;
      s_candidate.beta = beta;
      s_candidate.Cv_e = Cv_e;
      s_candidate.Te = Te_c;
      s_candidate.rho = rho_raw;
    }
    candidates[c] = s_candidate;
  }
}

__device__ inline void reduce_dt_rad_candidates_kernel_body(
    const int tid,
    const DtRadCellCandidate* __restrict__ candidates,
    const int n_cells,
    DtRadLimitDeviceResult* __restrict__ result,
    const double infinity) {
  DtRadLimitDeviceResult best{};
  best.dt_rad = infinity;
  best.limiting_cell = -1;
  double local_max_sigma_P = 0.0;
  for (int c = tid; c < n_cells; c += blockDim.x) {
    const DtRadCellCandidate candidate = candidates[c];
    if (isfinite(candidate.sigma_P)) {
      local_max_sigma_P = fmax(local_max_sigma_P, fmax(candidate.sigma_P, 0.0));
    }
    DtRadLimitDeviceResult candidate_result{};
    candidate_result.dt_rad = candidate.dt_rad;
    candidate_result.limiting_cell = candidate.cell;
    candidate_result.sigma_P = candidate.sigma_P;
    candidate_result.beta = candidate.beta;
    candidate_result.Cv_e = candidate.Cv_e;
    candidate_result.Te = candidate.Te;
    candidate_result.rho = candidate.rho;
    if (dt_rad_limit_better(candidate_result, best)) {
      best = candidate_result;
    }
  }

  __shared__ DtRadLimitDeviceResult s_best[kReduceBlock];
  __shared__ double s_max_sigma_P[kReduceBlock];
  s_best[tid] = best;
  s_max_sigma_P[tid] = local_max_sigma_P;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (tid < offset) {
      const int other = tid + offset;
      if (dt_rad_limit_better(s_best[other], s_best[tid])) {
        s_best[tid] = s_best[other];
      }
      s_max_sigma_P[tid] = fmax(s_max_sigma_P[tid], s_max_sigma_P[other]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    DtRadLimitDeviceResult out = s_best[0];
    out.max_sigma_P = s_max_sigma_P[0];
    *result = out;
  }
}

}  // namespace tenryu::radiation::fleck_bodies
