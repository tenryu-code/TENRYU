#pragma once

#include <cfloat>
#include <cmath>

#include "core/axis_tolerance.hpp"

namespace tenryu::radiation {

constexpr double kMilneLambda = 0.7104;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct InterfaceProbabilityResult {
  double probability = 0.0;              // Final directional probability (mu-weighted).
  double c_hat = 0.0;                    // Scalar C-hat before directional weighting.
  bool used_standard_fallback = false;   // Safety fallback (denom<=0 or c_hat<0).
  bool clamped_high = false;             // C-hat clamped to 4/5.
};

#if defined(__CUDACC__)
#define TENRYU_INTERFACE_HD __host__ __device__
#else
#define TENRYU_INTERFACE_HD
#endif

TENRYU_INTERFACE_HD inline double clamp01(const double x) {
  if (!(x > 0.0)) {
    return 0.0;
  }
  if (x >= 1.0) {
    return 1.0;
  }
  return x;
}

TENRYU_INTERFACE_HD inline double compute_tau_saturated(const double sigma_R,
                                                        const double delta_x) {
  const double sigma = fmax(sigma_R, 0.0);
  const double dx = fmax(delta_x, 0.0);
  if (sigma <= 0.0 || dx <= 0.0) {
    return 0.0;
  }
  return fmin(sigma * dx, DBL_MAX);
}

TENRYU_INTERFACE_HD inline double compute_standard_p_mu(const double sigma_R,
                                                        const double delta_x,
                                                        const double mu) {
  const double tau = compute_tau_saturated(sigma_R, delta_x);
  const double mu_clamped = clamp01(mu);
  const double denom = 3.0 * tau + 6.0 * kMilneLambda;
  if (!(denom > 0.0) || !isfinite(denom)) {
    return 0.0;
  }
  const double p = (4.0 / denom) * (1.0 + 1.5 * mu_clamped);
  if (!isfinite(p)) {
    return 0.0;
  }
  return clamp01(p);
}

// Emissivity-preserving C-hat(mu), Densmore 2006 Eq.48 + NUMERICS §7.7.3 safety.
TENRYU_INTERFACE_HD inline InterfaceProbabilityResult compute_P_hat_emissivity(
    const double sigma_R,
    const double delta_x,
    const double omega,
    const double mu) {
  InterfaceProbabilityResult out{};
  const double mu_clamped = clamp01(mu);

  const double omega_clamped = clamp01(omega);
  const double one_minus_omega = fmax(1.0 - omega_clamped, 0.0);

  // Pure-scattering limit: epsilon'->0 and beta->0 => C-hat->0 (not 1).
  if (one_minus_omega <= 1.0e-12) {
    out.probability = 0.0;
    out.c_hat = 0.0;
    return out;
  }

  const double tau = compute_tau_saturated(sigma_R, delta_x);
  const double sqrt_arg = 3.0 * one_minus_omega;
  const double sqrt_term = sqrt(fmax(sqrt_arg, 0.0));
  const double eps_prime =
      (4.0 / 3.0) * sqrt_term / (1.0 + kMilneLambda * sqrt_term);

  const double tau_limit = sqrt(DBL_MAX);
  if (tau > tau_limit) {
    double c_hat = eps_prime;
    if (!(c_hat >= 0.0) || !isfinite(c_hat)) {
      out.used_standard_fallback = true;
      out.probability = compute_standard_p_mu(sigma_R, delta_x, mu_clamped);
      out.c_hat = 0.0;
      return out;
    }
    if (c_hat > 0.8) {
      c_hat = 0.8;
      out.clamped_high = true;
    }
    out.c_hat = c_hat;
    out.probability = clamp01(0.5 * c_hat * (1.0 + 1.5 * mu_clamped));
    if (!isfinite(out.probability)) {
      out.used_standard_fallback = true;
      out.probability = compute_standard_p_mu(sigma_R, delta_x, mu_clamped);
      out.c_hat = 0.0;
    }
    return out;
  }

  const double tau2 = tau * tau;
  const double beta_inner = 3.0 * one_minus_omega * tau2 +
                            2.25 * one_minus_omega * one_minus_omega * tau2 * tau2;
  const double beta = 1.5 * one_minus_omega * tau2 + sqrt(fmax(beta_inner, 0.0));

  const double denom = beta - (4.0 / 3.0) * eps_prime * tau;
  if (!(denom > 0.0) || !isfinite(denom)) {
    out.used_standard_fallback = true;
    out.probability = compute_standard_p_mu(sigma_R, delta_x, mu_clamped);
    out.c_hat = 0.0;
    return out;
  }

  double c_hat = eps_prime * beta / denom;
  if (!(c_hat >= 0.0) || !isfinite(c_hat)) {
    out.used_standard_fallback = true;
    out.probability = compute_standard_p_mu(sigma_R, delta_x, mu_clamped);
    out.c_hat = 0.0;
    return out;
  }

  if (c_hat > 0.8) {
    c_hat = 0.8;
    out.clamped_high = true;
  }

  out.c_hat = c_hat;
  out.probability = clamp01(0.5 * c_hat * (1.0 + 1.5 * mu_clamped));
  if (!isfinite(out.probability)) {
    out.used_standard_fallback = true;
    out.probability = compute_standard_p_mu(sigma_R, delta_x, mu_clamped);
    out.c_hat = 0.0;
  }
  return out;
}

TENRYU_INTERFACE_HD inline double compute_delta_x_m_1d(const double r_inner,
                                                       const double r_outer) {
  return fmax(r_outer - r_inner, 0.0);
}

TENRYU_INTERFACE_HD inline double compute_delta_x_m_2d_rz(const double cell_volume,
                                                          const double face_r_bar,
                                                          const double face_length,
                                                          const double r_max) {
  const double V = fmax(cell_volume, 0.0);
  const double L = fmax(face_length, 0.0);
  if (V <= 0.0 || L <= 0.0) {
    return 0.0;
  }

  // Axis special-case: A_m = pi * R_max * L_m.
  if (face_r_bar <= core::axis_tolerance(r_max) && r_max > 0.0) {
    const double area_axis = kPi * r_max * L;
    if (area_axis <= 0.0) {
      return 0.0;
    }
    return V / area_axis;
  }

  // General 2D_RZ face area: A_m = 2*pi*R_bar*L_m.
  const double area = 2.0 * kPi * fmax(face_r_bar, 0.0) * L;
  if (area <= 0.0) {
    return 0.0;
  }
  return V / area;
}

#undef TENRYU_INTERFACE_HD

}  // namespace tenryu::radiation
