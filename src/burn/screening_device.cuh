#pragma once

#include <cassert>
#include <cmath>
#include <limits>

#include "burn/burn_constants.hpp"
#include "burn/deposition.cuh"
#include "burn/screening.hpp"
#include "core/macros.hpp"

namespace tenryu::burn {

constexpr unsigned int kScreeningWarningInvalid = 1U << 0U;
constexpr unsigned int kScreeningWarningStrong = 1U << 1U;

namespace screening_device_detail {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kElectronChargeEsu = 4.80320425e-10;
constexpr double kEVToErg = 1.602176634e-12;
constexpr double kHbarErgS = 1.054571817e-27;
constexpr double kProtonMassG = 1.6726219e-24;

// Salpeter weak-screening validity: h << 1. Beyond h ~ 2 the linearized
// Debye picture is meaningless; cap the enhancement (e^2 ~ 7.4x) and warn
// instead of letting exp(h) run away (2026-07-26 review).
constexpr double kSalpeterHMax = 2.0;

struct ReactionPair {
  int i;
  int j;
};

TENRYU_HOST_DEVICE inline ReactionPair reaction_pair(const int k) {
  constexpr ReactionPair pairs[kNumReactions] = {
      {kD, kT}, {kD, kD}, {kD, kD}, {kD, kHe3}};
  return pairs[k];
}

TENRYU_HOST_DEVICE inline void fill_identity(double F[kNumReactions]) {
  for (int k = 0; k < kNumReactions; ++k) {
    F[k] = 1.0;
  }
}

TENRYU_HOST_DEVICE inline double chugunov_dewitt_f0(const double G) {
  constexpr double A1 = -0.907;
  constexpr double A2 = 0.62954;
  const double A3 = -std::sqrt(3.0) / 2.0 - A1 / std::sqrt(A2);
  constexpr double B1 = 0.00456;
  constexpr double B2 = 211.6;
  constexpr double B3 = -1.0e-4;
  constexpr double B4 = 0.00462;
  const double sqrt_G = std::sqrt(G);
  return A1 * (std::sqrt(G * (A2 + G)) -
               A2 * std::log(std::sqrt(G / A2) +
                              std::sqrt(1.0 + G / A2))) +
         2.0 * A3 * (sqrt_G - std::atan(sqrt_G)) +
         B1 * (G - B2 * std::log(1.0 + G / B2)) +
         0.5 * B3 * std::log(1.0 + G * G / B4);
}

TENRYU_HOST_DEVICE inline void salpeter_factors(
    const double Ti_eV, const double Te_eV, const double ne_cm3,
    const double n_species[], double F[kNumReactions],
    unsigned int* warning_flags) {
  if (!(Ti_eV > 0.0) || !(Te_eV > 0.0) || !(ne_cm3 > 0.0) ||
      !std::isfinite(Ti_eV) || !std::isfinite(Te_eV) ||
      !std::isfinite(ne_cm3)) {
    for (int k = 0; k < kNumReactions; ++k) {
      F[k] = 1.0;
    }
    *warning_flags |= kScreeningWarningInvalid;
    return;
  }
  const double kTi = Ti_eV * kEVToErg;
  const double kTe = Te_eV * kEVToErg;
  double ion_sum = 0.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    const double Z = species_Z(s);
    ion_sum += n_species[s] * Z * Z;
  }
  const double inv_lambda2 =
      4.0 * kPi * kElectronChargeEsu * kElectronChargeEsu *
      (ion_sum / kTi + ne_cm3 / kTe);
  const double lambda_inv = std::sqrt(inv_lambda2);
  for (int k = 0; k < kNumReactions; ++k) {
    const ReactionPair p = reaction_pair(k);
    const double Zi = species_Z(p.i);
    const double Zj = species_Z(p.j);
    const double h =
        Zi * Zj * kElectronChargeEsu * kElectronChargeEsu * lambda_inv / kTi;
    double h_eff = h;
    if (!std::isfinite(h) || h < 0.0) {
      h_eff = 0.0;
      *warning_flags |= kScreeningWarningInvalid;
    } else if (h > kSalpeterHMax) {
      h_eff = kSalpeterHMax;
      *warning_flags |= kScreeningWarningStrong;
    }
    F[k] = std::exp(h_eff);
  }
}

TENRYU_HOST_DEVICE inline void chugunov_dewitt_factors(
    const double Ti_eV, const double ne_cm3, const double Zbar,
    const double Z2bar, double F[kNumReactions]) {
  const double kTi = Ti_eV * kEVToErg;
  const double a_e = std::pow(3.0 / (4.0 * kPi * ne_cm3), 1.0 / 3.0);
  const double Ge = kElectronChargeEsu * kElectronChargeEsu / (a_e * kTi);
  for (int k = 0; k < kNumReactions; ++k) {
    const ReactionPair p = reaction_pair(k);
    const double Zi = species_Z(p.i);
    const double Zj = species_Z(p.j);
    const double Ai = species_A(p.i);
    const double Aj = species_A(p.j);
    const double mu = Ai * Aj / (Ai + Aj) * kProtonMassG;
    const double tau =
        std::pow(27.0 * kPi * kPi * mu * Zi * Zi * Zj * Zj *
                     std::pow(kElectronChargeEsu, 4.0) /
                     (2.0 * kTi * kHbarErgS * kHbarErgS),
                 1.0 / 3.0);
    const double a_ij =
        0.5 * (std::pow(Zi, 1.0 / 3.0) + std::pow(Zj, 1.0 / 3.0)) * a_e;
    const double Gij =
        Zi * Zj * kElectronChargeEsu * kElectronChargeEsu / (a_ij * kTi);
    const double zeta = 3.0 * Gij / tau;
    const double y = 4.0 * Zi * Zj / ((Zi + Zj) * (Zi + Zj));
    const double c1 = 0.013 * y * y;
    const double c2 = 0.406 * std::pow(y, 0.14);
    const double c3 = 0.062 * std::pow(y, 0.19) + 1.8 / Gij;
    const double t =
        std::pow(1.0 + c1 * zeta + c2 * zeta * zeta +
                     c3 * zeta * zeta * zeta,
                 1.0 / 3.0);
    const double Gi = Ge * std::pow(Zi, 5.0 / 3.0);
    const double Gj = Ge * std::pow(Zj, 5.0 / 3.0);
    const double Gcomp = Ge * std::pow(Zi + Zj, 5.0 / 3.0);
    const double hsum = chugunov_dewitt_f0(Gi / t) +
                        chugunov_dewitt_f0(Gj / t) -
                        chugunov_dewitt_f0(Gcomp / t);
    const double C =
        3.0 * Zi * Zj * std::sqrt(Z2bar / Zbar) /
        (std::pow(Zi + Zj, 2.5) - std::pow(Zi, 2.5) -
         std::pow(Zj, 2.5));
    double h = (C + Gij * Gij) / (1.0 + Gij * Gij) * hsum;
    if (h < 0.0) {
      assert(h >= -std::numeric_limits<double>::epsilon());
      h = 0.0;
    }
    F[k] = std::exp(h);
  }
}

}  // namespace screening_device_detail

TENRYU_HOST_DEVICE inline void burn_screening_factors_core(
    const ScreeningMode mode, const double Ti_eV, const double Te_eV,
    const double ne_cm3, const double n_species[], double F[],
    unsigned int* warning_flags) {
  screening_device_detail::fill_identity(F);
  if (mode == ScreeningMode::kNone) {
    return;
  }

  if (!(ne_cm3 > 0.0) || !(Ti_eV > 0.0) || !(Te_eV > 0.0)) {
    return;
  }

  double n_tot = 0.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    n_tot += n_species[s];
  }
  if (!(n_tot > 0.0)) {
    return;
  }

  double Zbar = 0.0;
  double Z2bar = 0.0;
  for (int s = 0; s < kNumSpecies; ++s) {
    const double x = n_species[s] / n_tot;
    const double Z = species_Z(s);
    Zbar += x * Z;
    Z2bar += x * Z * Z;
  }

  if (mode == ScreeningMode::kSalpeter) {
    screening_device_detail::salpeter_factors(
        Ti_eV, Te_eV, ne_cm3, n_species, F, warning_flags);
  } else if (mode == ScreeningMode::kChugunovDeWitt) {
    screening_device_detail::chugunov_dewitt_factors(Ti_eV, ne_cm3, Zbar,
                                                      Z2bar, F);
  }
}

}  // namespace tenryu::burn
