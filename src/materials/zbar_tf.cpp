#include "materials/zbar_tf.hpp"
#include "materials/zbar_math.hpp"

#include <algorithm>
#include <cmath>

namespace tenryu::materials {
namespace {

constexpr double kTeMin = 1.0e-2;
constexpr double kTeMax = 1.0e4;
constexpr double kFiniteDiffEps = 1.0e-4;

[[nodiscard]] double zbar_tf_impl(const double rho,
                                  const double Te_eV,
                                  const double Z_nuc,
                                  const double A_amu) {
  return zbar_tf_value(rho, Te_eV, Z_nuc, A_amu);
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
