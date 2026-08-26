#pragma once

#include <cmath>

namespace tenryu::materials {

struct EOSParams {
  double gamma;   // gamma (same for electrons and ions in v1.0)
  double A;       // atomic mass number
  double Z_bar;   // average ionization
  double Cv_e;    // electron specific heat [erg/(g*eV)]
  double Cv_i;    // ion specific heat [erg/(g*eV)]
};

__device__ inline double eos_pressure_ion(const double rho,
                                          const double e_i,
                                          const double gamma_i) {
  return (gamma_i - 1.0) * rho * e_i;
}

__device__ inline double eos_pressure_electron(const double rho,
                                               const double e_e,
                                               const double gamma_e) {
  return (gamma_e - 1.0) * rho * e_e;
}

__device__ inline double eos_energy_ion(const double /*rho*/,
                                        const double T_i,
                                        const double Cv_i) {
  return Cv_i * T_i;
}

__device__ inline double eos_energy_electron(const double /*rho*/,
                                             const double T_e,
                                             const double Cv_e) {
  return Cv_e * T_e;
}

__device__ inline double eos_temperature_ion(const double e_i,
                                             const double Cv_i) {
  return (Cv_i > 0.0) ? (e_i / Cv_i) : 0.0;
}

__device__ inline double eos_temperature_electron(const double e_e,
                                                  const double Cv_e) {
  return (Cv_e > 0.0) ? (e_e / Cv_e) : 0.0;
}

__device__ inline double eos_sound_speed_2t(const double gamma_e,
                                            const double P_e,
                                            const double gamma_i,
                                            const double P_i,
                                            const double rho) {
  if (rho <= 0.0) {
    return 0.0;
  }
  const double cs2 = (gamma_e * P_e + gamma_i * P_i) / rho;
  return (cs2 > 0.0) ? sqrt(cs2) : 0.0;
}

__device__ inline double compute_Qei(const double rho,
                                     const double Te,
                                     const double Ti,
                                     const double Zbar,
                                     const double A) {
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);
  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) / (Zbar * Zbar * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  // v1.0 ideal-gas electron specific heat with gamma=5/3:
  // c_v,e = 3/2 * Zbar * k_B / (A * m_p) [erg/(g*eV)].
  const double cv_e_mass = 1.5 * Zbar * kEvToErg / (A * kProtonMass);
  const double Cv_e = rho * cv_e_mass;  // volumetric [erg/(cm^3*eV)]
  return Cv_e * (te_safe - Ti) / tau_eq;
}

__host__ __device__ inline double compute_Qei_ext(const double rho,
                                                  const double Te,
                                                  const double Ti,
                                                  const double Zbar,
                                                  const double zmom_r2,
                                                  const double A) {
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);
  // Only <Z^2> enters tau_eq; the Coulomb log and cv_e keep plain Zbar.
  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) /
      ((Zbar * Zbar * zmom_r2) * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  // v1.0 ideal-gas electron specific heat with gamma=5/3:
  // c_v,e = 3/2 * Zbar * k_B / (A * m_p) [erg/(g*eV)].
  const double cv_e_mass = 1.5 * Zbar * kEvToErg / (A * kProtonMass);
  const double Cv_e = rho * cv_e_mass;  // volumetric [erg/(cm^3*eV)]
  return Cv_e * (te_safe - Ti) / tau_eq;
}

__device__ inline double compute_qei_term_analytical(const double rho,
                                                     const double Te,
                                                     const double Ti,
                                                     const double Zbar,
                                                     const double A,
                                                     const double gamma,
                                                     const double dt,
                                                     const double qei_multiplier = 1.0) {
  // Returns energy per gram [erg/g] transferred from electrons to ions.
  // Positive means electrons lose energy and ions gain energy.
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0 || dt <= 0.0 ||
      qei_multiplier <= 0.0 || !isfinite(qei_multiplier)) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  // Coulomb logarithm (same form as compute_Qei()).
  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);

  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) / (Zbar * Zbar * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }

  // Specific heats [erg/(g*eV)].
  const double gm1 = gamma - 1.0;
  if (!(gm1 > 0.0) || !isfinite(gm1)) {
    return 0.0;
  }
  const double cv_e = Zbar * kEvToErg / (A * kProtonMass * gm1);
  const double cv_i = kEvToErg / (A * kProtonMass * gm1);
  const double cv_sum = cv_e + cv_i;
  if (cv_sum <= 0.0 || !isfinite(cv_sum)) {
    return 0.0;
  }

  // Effective relaxation timescale and exact finite-dt relaxation factor.
  const double tau_eff = tau_eq * cv_i / cv_sum;
  if (!(tau_eff > 0.0) || !isfinite(tau_eff)) {
    return 0.0;
  }
  const double ratio = dt * qei_multiplier / tau_eff;
  // -expm1(-x) == 1 - exp(-x) exactly, but keeps full relative precision for
  // ratio << 1 (the raw form loses ~eps/ratio relative accuracy to
  // cancellation; 2026-07-26 review).
  const double f_relax = (ratio > 500.0) ? 1.0 : (-expm1(-ratio));

  return cv_e * cv_i / cv_sum * (te_safe - Ti) * f_relax;
}

__host__ __device__ inline double compute_qei_term_analytical_ext(
    const double rho,
    const double Te,
    const double Ti,
    const double Zbar,
    const double zmom_r2,
    const double A,
    const double gamma,
    const double dt,
    const double qei_multiplier = 1.0) {
  // Returns energy per gram [erg/g] transferred from electrons to ions.
  // Positive means electrons lose energy and ions gain energy.
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0 || dt <= 0.0 ||
      qei_multiplier <= 0.0 || !isfinite(qei_multiplier)) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  // Coulomb logarithm (same form as compute_Qei()).
  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);

  // Only <Z^2> enters tau_eq; the Coulomb log and cv_e keep plain Zbar.
  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) /
      ((Zbar * Zbar * zmom_r2) * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }

  // Specific heats [erg/(g*eV)].
  const double gm1 = gamma - 1.0;
  if (!(gm1 > 0.0) || !isfinite(gm1)) {
    return 0.0;
  }
  const double cv_e = Zbar * kEvToErg / (A * kProtonMass * gm1);
  const double cv_i = kEvToErg / (A * kProtonMass * gm1);
  const double cv_sum = cv_e + cv_i;
  if (cv_sum <= 0.0 || !isfinite(cv_sum)) {
    return 0.0;
  }

  // Effective relaxation timescale and exact finite-dt relaxation factor.
  const double tau_eff = tau_eq * cv_i / cv_sum;
  if (!(tau_eff > 0.0) || !isfinite(tau_eff)) {
    return 0.0;
  }
  const double ratio = dt * qei_multiplier / tau_eff;
  // -expm1(-x) == 1 - exp(-x) exactly, but keeps full relative precision for
  // ratio << 1 (the raw form loses ~eps/ratio relative accuracy to
  // cancellation; 2026-07-26 review).
  const double f_relax = (ratio > 500.0) ? 1.0 : (-expm1(-ratio));

  return cv_e * cv_i / cv_sum * (te_safe - Ti) * f_relax;
}

/// Q_ei using explicit cv_e/cv_i (for table EOS).
/// cv_e_mass: [erg/(g*eV)], cv_i_mass: [erg/(g*eV)].
/// Returns energy per gram transferred from electrons to ions [erg/g].
__device__ inline double compute_qei_term_with_cv(const double rho,
                                                   const double Te,
                                                   const double Ti,
                                                   const double Zbar,
                                                   const double A,
                                                   const double cv_e_mass,
                                                   const double cv_i_mass,
                                                   const double dt,
                                                   const double qei_multiplier = 1.0) {
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0 || dt <= 0.0 ||
      qei_multiplier <= 0.0 || !isfinite(qei_multiplier)) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);

  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) / (Zbar * Zbar * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }

  if (!(cv_e_mass > 0.0) || !isfinite(cv_e_mass) ||
      !(cv_i_mass > 0.0) || !isfinite(cv_i_mass)) {
    return 0.0;
  }
  const double cv_sum = cv_e_mass + cv_i_mass;
  if (cv_sum <= 0.0 || !isfinite(cv_sum)) {
    return 0.0;
  }

  const double tau_eff = tau_eq * cv_i_mass / cv_sum;
  if (!(tau_eff > 0.0) || !isfinite(tau_eff)) {
    return 0.0;
  }
  const double ratio = dt * qei_multiplier / tau_eff;
  // -expm1(-x) == 1 - exp(-x) exactly, but keeps full relative precision for
  // ratio << 1 (the raw form loses ~eps/ratio relative accuracy to
  // cancellation; 2026-07-26 review).
  const double f_relax = (ratio > 500.0) ? 1.0 : (-expm1(-ratio));

  return cv_e_mass * cv_i_mass / cv_sum * (te_safe - Ti) * f_relax;
}

__host__ __device__ inline double compute_qei_term_with_cv_ext(
    const double rho,
    const double Te,
    const double Ti,
    const double Zbar,
    const double zmom_r2,
    const double A,
    const double cv_e_mass,
    const double cv_i_mass,
    const double dt,
    const double qei_multiplier = 1.0) {
  constexpr double kProtonMass = 1.6726219e-24;  // must match core::constants::proton_mass
  constexpr double kMaxTeForPow = 1.0e8;

  if (rho <= 0.0 || Te <= 0.0 || A <= 0.0 || Zbar <= 0.0 || dt <= 0.0 ||
      qei_multiplier <= 0.0 || !isfinite(qei_multiplier)) {
    return 0.0;
  }

  const double te_safe = (isfinite(Te) && Te > 0.0) ? fmin(Te, kMaxTeForPow) : 0.0;
  if (!(te_safe > 0.0)) {
    return 0.0;
  }
  if (!isfinite(Ti)) {
    return 0.0;
  }

  const double n_i = rho / (A * kProtonMass);
  if (n_i <= 0.0) {
    return 0.0;
  }
  const double n_e = Zbar * n_i;
  if (n_e <= 0.0) {
    return 0.0;
  }

  double ln_lambda_raw = 0.0;
  if (te_safe >= 10.0 * Zbar * Zbar) {
    ln_lambda_raw = 24.0 - 0.5 * log(n_e) + log(te_safe);
  } else {
    ln_lambda_raw = 23.0 - 0.5 * log(n_e) - log(Zbar) + 1.5 * log(te_safe);
  }
  const double ln_lambda = fmax(2.0, ln_lambda_raw);

  // Only <Z^2> enters tau_eq; the Coulomb log and cv_e keep plain Zbar.
  const double tau_eq =
      3.16e8 * A * pow(te_safe, 1.5) /
      ((Zbar * Zbar * zmom_r2) * n_i * ln_lambda);
  if (!(tau_eq > 0.0) || !isfinite(tau_eq)) {
    return 0.0;
  }

  if (!(cv_e_mass > 0.0) || !isfinite(cv_e_mass) ||
      !(cv_i_mass > 0.0) || !isfinite(cv_i_mass)) {
    return 0.0;
  }
  const double cv_sum = cv_e_mass + cv_i_mass;
  if (cv_sum <= 0.0 || !isfinite(cv_sum)) {
    return 0.0;
  }

  const double tau_eff = tau_eq * cv_i_mass / cv_sum;
  if (!(tau_eff > 0.0) || !isfinite(tau_eff)) {
    return 0.0;
  }
  const double ratio = dt * qei_multiplier / tau_eff;
  // -expm1(-x) == 1 - exp(-x) exactly, but keeps full relative precision for
  // ratio << 1 (the raw form loses ~eps/ratio relative accuracy to
  // cancellation; 2026-07-26 review).
  const double f_relax = (ratio > 500.0) ? 1.0 : (-expm1(-ratio));

  return cv_e_mass * cv_i_mass / cv_sum * (te_safe - Ti) * f_relax;
}

}  // namespace tenryu::materials
