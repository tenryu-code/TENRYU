#pragma once

#include <algorithm>
#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::laser {

TENRYU_HOST_DEVICE inline double compute_n_hat(const double n_e,
                                               const double n_crit) {
  if (!(n_crit > 0.0) || !(n_e > 0.0)) {
    return 0.0;
  }
  const double n_hat = n_e / n_crit;
  return ::fmin(1.0, ::fmax(0.0, n_hat));
}

TENRYU_HOST_DEVICE inline double compute_refractive_index(const double n_hat,
                                                          const double eps_n) {
  const double clamped_n = ::fmin(1.0, ::fmax(0.0, n_hat));
  return ::sqrt(::fmax(eps_n, 1.0 - clamped_n));
}

TENRYU_HOST_DEVICE inline bool is_near_critical(const double n_hat,
                                                const double eps_crit) {
  return n_hat >= (1.0 - eps_crit);
}

double compute_critical_density_from_wavelength_cm(double lambda_cm);

}  // namespace tenryu::laser

