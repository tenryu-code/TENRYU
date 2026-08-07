#include "materials/zbar_tf.hpp"

#include <algorithm>
#include <cmath>

namespace tenryu::materials {
namespace {

constexpr double kRhoMin = 1.0e-3;
constexpr double kRhoMax = 1.0e4;
constexpr double kTeMin = 1.0e-2;
constexpr double kTeMax = 1.0e4;
constexpr double kEtaRhoNorm = 0.148;
constexpr double kT0Prefactor = 0.0327;
constexpr double kT0ExpFactor = 6.98;
constexpr double kT0RhoPower = 0.075;
constexpr double kFiniteDiffEps = 1.0e-4;

[[nodiscard]] double zbar_tf_impl(const double rho,
                                  const double Te_eV,
                                  const double Z_nuc,
                                  const double A_amu) {
  const double Z = std::max(Z_nuc, 0.0);
  const double A = std::max(A_amu, 1.0e-30);
  if (Z <= 0.0) {
    return 0.0;
  }

  const double rho_c = std::clamp(rho, kRhoMin, kRhoMax);
  const double Te_c = std::clamp(Te_eV, kTeMin, kTeMax);

  const double rho_H = rho_c / A;
  const double Te_H = Te_c / std::pow(Z, 4.0 / 3.0);

  const double eta = std::sqrt(std::max(rho_H / kEtaRhoNorm, 0.0));
  const double z0_H = eta / (1.0 + eta);

  double T0_exp_arg =
      kT0ExpFactor * std::pow(std::max(rho_H, 0.0), kT0RhoPower);
  if (!std::isfinite(T0_exp_arg)) {
    T0_exp_arg = 700.0;
  }
  T0_exp_arg = std::min(T0_exp_arg, 700.0);
  const double T0_H = kT0Prefactor * std::exp(T0_exp_arg);
  const double Y = 1.0 / (1.0 + std::sqrt(std::max(T0_H / std::max(Te_H, 1.0e-30), 0.0)));
  const double zth_H = (1.0 - z0_H) * Y;

  const double zbar = Z * (z0_H + zth_H);
  return std::clamp(zbar, 0.0, Z);
}

}  // namespace

double compute_zbar_tf(const double rho,
                       const double Te_eV,
                       const double Z_nuc,
                       const double A_amu) {
  return zbar_tf_impl(rho, Te_eV, Z_nuc, A_amu);
}

// NOTE: dZbar/dTe is not used in the current runtime path. It is kept for
// future cv_e / thermodynamic-consistency corrections when tabular/TF zbar is
// coupled into EOS derivatives.
double compute_dzbar_dTe(const double rho,
                         const double Te_eV,
                         const double Z_nuc,
                         const double A_amu,
                         const double temperature_floor_eV) {
  const double Te_c = std::clamp(Te_eV, kTeMin, kTeMax);
  const double dTe = std::max(kFiniteDiffEps * Te_c, std::max(temperature_floor_eV, 1.0e-12));

  const double Tm = std::max(kTeMin, Te_c - dTe);
  const double Tp = std::min(kTeMax, Te_c + dTe);

  if (Tp <= Tm) {
    return 0.0;
  }

  const double zm = compute_zbar_tf(rho, Tm, Z_nuc, A_amu);
  const double zp = compute_zbar_tf(rho, Tp, Z_nuc, A_amu);

  if (Te_c <= kTeMin + 1.0e-12) {
    const double z0 = compute_zbar_tf(rho, Te_c, Z_nuc, A_amu);
    return (zp - z0) / (Tp - Te_c);
  }
  if (Te_c >= kTeMax - 1.0e-12) {
    const double z0 = compute_zbar_tf(rho, Te_c, Z_nuc, A_amu);
    return (z0 - zm) / (Te_c - Tm);
  }

  return (zp - zm) / (Tp - Tm);
}

}  // namespace tenryu::materials
