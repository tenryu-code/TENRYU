#include "verification/noh_analytic.hpp"

#include <cmath>

#include "core/error.hpp"

namespace tenryu::verification {

double noh_shock_radius(const double t, const double v0) {
  TENRYU_ASSERT(v0 > 0.0, "Noh analytic requires v0 > 0");
  if (t <= 0.0) {
    return 0.0;
  }
  return v0 * t / 3.0;
}

double noh_density_plateau(const double rho0, const double gamma) {
  TENRYU_ASSERT(rho0 > 0.0, "Noh analytic requires rho0 > 0");
  TENRYU_ASSERT(gamma > 1.0, "Noh analytic requires gamma > 1");

  const double compression = (gamma + 1.0) / (gamma - 1.0);
  return rho0 * compression * compression * compression;
}

double noh_density_plateau_planar(const double rho0, const double gamma) {
  TENRYU_ASSERT(rho0 > 0.0, "Noh analytic requires rho0 > 0");
  TENRYU_ASSERT(gamma > 1.0, "Noh analytic requires gamma > 1");
  return rho0 * (gamma + 1.0) / (gamma - 1.0);
}

double noh_density_plateau_cylindrical(const double rho0, const double gamma) {
  TENRYU_ASSERT(rho0 > 0.0, "Noh analytic requires rho0 > 0");
  TENRYU_ASSERT(gamma > 1.0, "Noh analytic requires gamma > 1");
  const double compression = (gamma + 1.0) / (gamma - 1.0);
  return rho0 * compression * compression;
}

}  // namespace tenryu::verification
