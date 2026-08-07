#include "verification/laser_analytic.cuh"

#include <cmath>

namespace tenryu::verification {

double laser_turning_point_linear(const double L_cm, const double cos_theta0) {
  return L_cm * cos_theta0 * cos_theta0;
}

double laser_beer_lambert_intensity(const double I0,
                                    const double kappa_cm_inv,
                                    const double s_cm) {
  return I0 * std::exp(-kappa_cm_inv * s_cm);
}

double laser_beer_lambert_absorbed(const double I0,
                                   const double kappa_cm_inv,
                                   const double s0_cm,
                                   const double s1_cm) {
  const double I0s = laser_beer_lambert_intensity(I0, kappa_cm_inv, s0_cm);
  const double I1s = laser_beer_lambert_intensity(I0, kappa_cm_inv, s1_cm);
  return I0s - I1s;
}

}  // namespace tenryu::verification

