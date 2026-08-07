#include "verification/sedov_analytic.hpp"

#include <cmath>

#include "core/error.hpp"

namespace tenryu::verification {

double sedov_shock_radius(const double t,
                          const double E0,
                          const double rho0,
                          const double xi0) {
  TENRYU_ASSERT(E0 > 0.0, "Sedov analytic requires E0 > 0");
  TENRYU_ASSERT(rho0 > 0.0, "Sedov analytic requires rho0 > 0");
  TENRYU_ASSERT(xi0 > 0.0, "Sedov analytic requires xi0 > 0");

  if (t <= 0.0) {
    return 0.0;
  }

  return xi0 * std::pow(E0 * t * t / rho0, 0.2);
}

}  // namespace tenryu::verification
