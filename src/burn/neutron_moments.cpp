#include "burn/neutron_moments.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"

namespace tenryu::burn {
namespace {

constexpr double kNeutronA = 1.0087;
constexpr double kProtonMassG = 1.6726219e-24;
constexpr double kKeVToErg = 1.602176634e-9;

constexpr int kBryskN = 11;
constexpr double kMed[kBryskN] = {
    1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0, 50.0, 70.0, 100.0};
constexpr double kDTReac[kBryskN] = {
    3.13, 5.29, 7.30, 10.97, 14.21, 18.41, 29.14, 37.78, 53.85, 69.25, 87.53};
constexpr double kDDnReac[kBryskN] = {
    2.94, 5.00, 6.84, 10.22, 13.38, 17.93, 32.27, 46.06, 72.83, 98.88, 136.80};

double interpolated_T_reac(const int channel, const double theta_keV) {
  const double* table = (channel == 0) ? kDTReac : kDDnReac;
  const double theta = std::clamp(theta_keV, kMed[0], kMed[kBryskN - 1]);
  if (theta <= kMed[0]) {
    return table[0];
  }
  if (theta >= kMed[kBryskN - 1]) {
    return table[kBryskN - 1];
  }

  const double log_theta = std::log(theta);
  for (int i = 0; i < kBryskN - 1; ++i) {
    if (theta <= kMed[i + 1]) {
      const double lo = std::log(kMed[i]);
      const double hi = std::log(kMed[i + 1]);
      const double f = (log_theta - lo) / (hi - lo);
      return table[i] + f * (table[i + 1] - table[i]);
    }
  }
  return table[kBryskN - 1];
}

}  // namespace

BryskMoments brysk_moments(const int channel, const double Ti_burn_keV,
                           const double vr2_burn_cm2s2) {
  BryskMoments moments;
  if ((channel != 0 && channel != 1) || !(Ti_burn_keV > 0.0)) {
    return moments;
  }

  const double theta = Ti_burn_keV;
  const double T_reac = interpolated_T_reac(channel, theta);
  const double K_mean = 3.0 * T_reac - 1.5 * theta;

  const double A_D = species_A(kD);
  const double E_n0_keV = (channel == 0) ? 14049.0 : 2449.0;
  double A_partner = 0.0;
  double A_reactants = 0.0;
  if (channel == 0) {
    A_partner = species_A(kHe4);
    A_reactants = A_D + species_A(kT);
  } else {
    A_partner = species_A(kHe3);
    A_reactants = 2.0 * A_D;
  }

  moments.mean_shift_keV =
      (kNeutronA / A_reactants) * 1.5 * theta +
      (A_partner / (kNeutronA + A_partner)) * K_mean;
  const double E_n_keV = E_n0_keV + moments.mean_shift_keV;

  // Convert Brysk's full 1/e width convention to Gaussian sigma.
  moments.sigma_thermal_keV =
      0.5 * std::sqrt(2.0 * kNeutronA * theta * E_n_keV /
                      (kNeutronA + A_partner));

  double sigma_fluid_keV = 0.0;
  if (vr2_burn_cm2s2 > 0.0) {
    const double m_n_g = kNeutronA * kProtonMassG;
    const double E_n0_erg = E_n0_keV * kKeVToErg;
    sigma_fluid_keV =
        std::sqrt(2.0 * m_n_g * E_n0_erg * (vr2_burn_cm2s2 / 3.0)) /
        kKeVToErg;
  }

  moments.sigma_total_keV =
      std::sqrt(moments.sigma_thermal_keV * moments.sigma_thermal_keV +
                sigma_fluid_keV * sigma_fluid_keV);
  return moments;
}

}  // namespace tenryu::burn
