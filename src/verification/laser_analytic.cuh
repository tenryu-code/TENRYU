#pragma once

namespace tenryu::verification {

[[nodiscard]] double laser_turning_point_linear(double L_cm, double cos_theta0);
[[nodiscard]] double laser_beer_lambert_intensity(double I0, double kappa_cm_inv, double s_cm);
[[nodiscard]] double laser_beer_lambert_absorbed(double I0,
                                                 double kappa_cm_inv,
                                                 double s0_cm,
                                                 double s1_cm);

}  // namespace tenryu::verification

