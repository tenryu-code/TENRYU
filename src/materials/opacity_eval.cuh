#pragma once

#include <cuda_runtime.h>

#include "core/device_error_flags.cuh"

namespace tenryu::materials {

__device__ inline double sigma_nu_marshak(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  if (x > 700.0) {
    return 27.0e9 / (nu_eV * nu_eV * nu_eV);
  }
  const double one_minus_exp = -expm1(-x);
  return (27.0e9 / (nu_eV * nu_eV * nu_eV)) * one_minus_exp;
}

__device__ inline double planck_weight(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  if (x > 700.0) {
    return nu_eV * nu_eV * nu_eV * exp(-x);
  }
  const double denom = expm1(x);
  if (!(denom > 0.0)) {
    return 0.0;
  }
  return (nu_eV * nu_eV * nu_eV) / denom;
}

__device__ inline double rosseland_weight(const double nu_eV, const double T_eV) {
  if (!(nu_eV > 0.0) || !(T_eV > 0.0)) {
    return 0.0;
  }
  const double x = nu_eV / T_eV;
  const double T2 = T_eV * T_eV;
  if (x > 80.0) {
    return (nu_eV * nu_eV * nu_eV * nu_eV) * exp(-x) / T2;
  }
  const double ex = exp(x);
  const double em1 = expm1(x);
  if (!(em1 > 0.0)) {
    return 0.0;
  }
  return (nu_eV * nu_eV * nu_eV * nu_eV) * ex / (T2 * em1 * em1);
}

__device__ inline double apply_sigma_cap(const double sigma,
                                         const double sigma_max) {
  return fmin(fmax(sigma, 0.0), fmax(sigma_max, 0.0));
}

__device__ inline double apply_sigma_floor(const double sigma,
                                           const double sigma_min) {
  return fmax(fmax(sigma, 0.0), fmax(sigma_min, 0.0));
}

__device__ inline double apply_sigma_bounds(const double sigma,
                                            const double sigma_min,
                                            const double sigma_max) {
  return apply_sigma_cap(apply_sigma_floor(sigma, sigma_min), sigma_max);
}

__device__ inline void eval_opacity_constant_kernel_body(
    const int c,
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_R,
    const double kappa_planck_const,
    const double kappa_rosseland_const,
    const double* __restrict__ kappa_planck_cell,
    const double* __restrict__ kappa_rosseland_cell,
    const double kappa_floor,
    const double kappa_cap,
    const int n_cells,
    const int n_groups,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const double rho_c = rho[c];
  (void)Te;  // Constant-opacity backend ignores temperature.
  if (!(rho_c >= 0.0)) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->opacity_out_of_range, 1);
    }
  }

  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const double sigma_min = rho_safe * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_safe * fmax(kappa_cap, 0.0);
  const double kappa_planck_c =
      (kappa_planck_cell != nullptr) ? kappa_planck_cell[c] : kappa_planck_const;
  const double kappa_rosseland_c =
      (kappa_rosseland_cell != nullptr) ? kappa_rosseland_cell[c] : kappa_rosseland_const;
  const double sigma_a_c = apply_sigma_bounds(rho_safe * kappa_planck_c, sigma_min, sigma_max);
  const double sigma_R_c = apply_sigma_bounds(rho_safe * kappa_rosseland_c, sigma_min, sigma_max);
  const int base = c * n_groups;
  for (int g = 0; g < n_groups; ++g) {
    sigma_a[base + g] = sigma_a_c;
    sigma_R[base + g] = sigma_R_c;
  }
}

__global__ inline void eval_opacity_constant_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_R,
    const double kappa_planck_const,
    const double kappa_rosseland_const,
    const double* __restrict__ kappa_planck_cell,
    const double* __restrict__ kappa_rosseland_cell,
    const double kappa_floor,
    const double kappa_cap,
    const int n_cells,
    const int n_groups,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  eval_opacity_constant_kernel_body(c, rho, Te, sigma_a, sigma_R,
                                    kappa_planck_const,
                                    kappa_rosseland_const,
                                    kappa_planck_cell, kappa_rosseland_cell,
                                    kappa_floor,
                                    kappa_cap, n_cells, n_groups,
                                    error_flags);
}

__global__ inline void eval_opacity_freq_dep_marshak_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ group_bounds_eV,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_R,
    const double kappa_floor,
    const double kappa_cap,
    const int n_cells,
    const int n_groups,
    tenryu::core::DeviceErrorFlags* __restrict__ error_flags) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const double rho_c = rho[c];
  if (!(rho_c > 0.0)) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->opacity_out_of_range, 1);
    }
    const int base = c * n_groups;
    for (int g = 0; g < n_groups; ++g) {
      sigma_a[base + g] = 0.0;
      sigma_R[base + g] = 0.0;
    }
    return;
  }

  const double T_c = fmax(Te[c], 1.0e-6);
  const double sigma_min = rho_c * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_c * fmax(kappa_cap, 0.0);
  const int base = c * n_groups;
  constexpr int n_simpson = 128;

  for (int g = 0; g < n_groups; ++g) {
    const double lo = group_bounds_eV[g];
    const double hi = group_bounds_eV[g + 1];
    if (!isfinite(lo) || !isfinite(hi) || !(lo >= 0.0) || !(hi > lo)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->opacity_out_of_range, 1);
      }
      sigma_a[base + g] = 0.0;
      sigma_R[base + g] = 0.0;
      continue;
    }

    const double lo_pos = fmax(lo, 1.0e-300);
    const double log_lo = log(lo_pos);
    const double log_hi = log(hi);
    const double h_log = (log_hi - log_lo) / static_cast<double>(n_simpson);

    double sum_wp = 0.0;
    double sum_sigma_wp = 0.0;
    double sum_wr = 0.0;
    double sum_inv_sigma_wr = 0.0;

    for (int i = 0; i <= n_simpson; ++i) {
      const double z = log_lo + h_log * static_cast<double>(i);
      const double E = exp(z);
      const double w = (i == 0 || i == n_simpson) ? 1.0 : ((i & 1) ? 4.0 : 2.0);
      const double jac = E;

      const double sigma = sigma_nu_marshak(E, T_c);
      const double wp = planck_weight(E, T_c) * jac;
      const double wr = rosseland_weight(E, T_c) * jac;

      sum_wp += w * wp;
      sum_sigma_wp += w * sigma * wp;
      sum_wr += w * wr;
      if (sigma > 0.0) {
        sum_inv_sigma_wr += w * wr / sigma;
      }
    }

    sum_wp *= h_log / 3.0;
    sum_sigma_wp *= h_log / 3.0;
    sum_wr *= h_log / 3.0;
    sum_inv_sigma_wr *= h_log / 3.0;

    const double sigmaP = (sum_wp > 0.0) ? (sum_sigma_wp / sum_wp) : 0.0;
    const double sigmaR =
        (sum_inv_sigma_wr > 0.0) ? (sum_wr / sum_inv_sigma_wr) : 0.0;

    sigma_a[base + g] = apply_sigma_bounds(sigmaP, sigma_min, sigma_max);
    sigma_R[base + g] = apply_sigma_bounds(sigmaR, sigma_min, sigma_max);
  }
}

}  // namespace tenryu::materials
