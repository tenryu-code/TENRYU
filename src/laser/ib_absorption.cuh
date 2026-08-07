#pragma once

#include <cfloat>
#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::laser {

TENRYU_HOST_DEVICE inline double compute_coulomb_log(const double n_hat,
                                                     const double Te_eV,
                                                     const double Zbar,
                                                     const double lambda_cm,
                                                     const double floor_value = 2.0) {
  const double effective_floor = ::fmax(2.0, floor_value);
  if (!(Te_eV > 0.0) || !(Zbar > 0.0) || !(lambda_cm > 0.0)) {
    return effective_floor;
  }

  const double n_hat_clamped = ::fmin(1.0, ::fmax(0.0, n_hat));
  const double lambda_um = lambda_cm * 1.0e4;
  const double Te_keV = Te_eV * 1.0e-3;
  const double Te_32 = Te_keV * ::sqrt(::fmax(Te_keV, 0.0));

  if (!(lambda_um > 0.0) || !(Te_32 > 0.0)) {
    return effective_floor;
  }

  const double nh_sqrt = ::sqrt(::fmax(1.0e-16, n_hat_clamped));
  const double arg = 1.5e4 * lambda_um * Te_32 / (Zbar * nh_sqrt);
  const double ln_lambda = (arg > 0.0) ? ::log(arg) : effective_floor;
  return ::fmax(effective_floor, ln_lambda);
}

TENRYU_HOST_DEVICE inline double compute_kappa_smooth_factor(const double n_hat,
                                                             const double Te_eV,
                                                             const double Zbar,
                                                             const double lambda_cm,
                                                             const double eps_n,
                                                             const double coulomb_log_floor);

TENRYU_HOST_DEVICE inline double compute_kappa_from_smooth(const double smooth_factor,
                                                           const double n_hat,
                                                           const double eps_n);

TENRYU_HOST_DEVICE inline double compute_kappa_ib(
    const double n_hat,
    const double Te_eV,
    const double Zbar,
    const double lambda_cm,
    const double eps_n = 1.0e-4,
    const double coulomb_log_floor = 2.0) {
  const double smooth_factor =
      compute_kappa_smooth_factor(n_hat, Te_eV, Zbar, lambda_cm, eps_n, coulomb_log_floor);
  return compute_kappa_from_smooth(smooth_factor, n_hat, eps_n);
}

TENRYU_HOST_DEVICE inline double compute_kappa_smooth_factor(
    const double n_hat,
    const double Te_eV,
    const double Zbar,
    const double lambda_cm,
    const double eps_n = 1.0e-4,
    const double coulomb_log_floor = 2.0) {
  (void)eps_n;
  if (!(Te_eV > 0.0) || !(Zbar > 0.0) || !(lambda_cm > 0.0)) {
    return 0.0;
  }

  const double n_hat_clamped = ::fmin(1.0, ::fmax(0.0, n_hat));
  const double lambda_um = lambda_cm * 1.0e4;
  const double Te_eV_32 = Te_eV * ::sqrt(::fmax(Te_eV, 0.0));
  const double Te_32 = Te_eV_32 * (1.0e-3 * ::sqrt(1.0e-3));

  if (!(lambda_um > 0.0) || !(Te_32 > 0.0)) {
    return 0.0;
  }

  const double ln_lambda =
      compute_coulomb_log(n_hat_clamped, Te_eV, Zbar, lambda_cm, coulomb_log_floor);
  return 3.4 * Zbar * ln_lambda / (lambda_um * lambda_um * Te_32);
}

TENRYU_HOST_DEVICE inline double tenryu_rsqrt(const double x) {
#ifdef __CUDA_ARCH__
  return ::rsqrt(x);
#else
  return 1.0 / std::sqrt(x);
#endif
}

TENRYU_HOST_DEVICE inline double compute_kappa_from_smooth(const double smooth_factor,
                                                           const double n_hat,
                                                           const double eps_n = 1.0e-4) {
  if (!(smooth_factor > 0.0)) {
    return 0.0;
  }
  const double n_hat_clamped = ::fmin(1.0, ::fmax(0.0, n_hat));
  return smooth_factor * n_hat_clamped * n_hat_clamped *
         tenryu_rsqrt(::fmax(eps_n, 1.0 - n_hat_clamped));
}

TENRYU_HOST_DEVICE inline double compute_optical_depth(const double kappa_n,
                                                       const double kappa_n1,
                                                       const double ds) {
  if (!(ds > 0.0)) {
    return 0.0;
  }
  const double k0 = ::fmax(0.0, kappa_n);
  const double k1 = ::fmax(0.0, kappa_n1);
  const double kmax = ::fmax(k0, k1);
  if (!(kmax > 0.0)) {
    return 0.0;
  }

  const double kavg = 0.5 * (k0 + k1);
  if (!::isfinite(kavg) || ds > DBL_MAX / kavg) {
    return DBL_MAX;
  }
  return kavg * ds;
}

TENRYU_HOST_DEVICE inline double absorbed_power_expm1(const double I_old,
                                                      const double optical_depth,
                                                      double& I_new) {
  if (!(I_old > 0.0) || !(optical_depth > 0.0)) {
    I_new = I_old;
    return 0.0;
  }

  double dP = -I_old * ::expm1(-optical_depth);
  dP = ::fmin(::fmax(dP, 0.0), I_old);
  I_new = I_old - dP;
  return dP;
}

}  // namespace tenryu::laser
