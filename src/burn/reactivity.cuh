#pragma once

#include <cmath>

#include "burn/burn_constants.hpp"
#include "core/macros.hpp"

namespace tenryu::burn {

// Bosch-Hale thermal reactivity [cm^3/s]; T in keV (convert eV->keV at the
// CALLER'S kernel entry, exactly once -- repo temperatures are eV).
TENRYU_HOST_DEVICE inline double bosch_hale_sv(int reaction, double T_keV) {
  const BoschHaleFit f = bosch_hale_fit(reaction);
  if (!(T_keV >= f.T_min_keV)) return 0.0;         // inclusive fit floor (also rejects NaN)
  const double T = (T_keV < f.T_max_keV) ? T_keV : f.T_max_keV;  // documented ceiling clamp
  const double num = T * (f.C2 + T * (f.C4 + T * f.C6));
  const double den = 1.0 + T * (f.C3 + T * (f.C5 + T * f.C7));
  const double theta = T / (1.0 - num / den);
  const double xi = cbrt(f.B_G * f.B_G / (4.0 * theta));
  return f.C1 * theta * sqrt(xi / (f.mrc2_keV * T * T * T)) * exp(-3.0 * xi);
}

}  // namespace tenryu::burn
