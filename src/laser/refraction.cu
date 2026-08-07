#include "laser/refraction.cuh"

#include <cmath>

#include "core/constants.hpp"

namespace tenryu::laser {

namespace {

constexpr double kElectronMass = 9.1094e-28;      // g
constexpr double kElementaryCharge = 4.8032e-10;  // statC
constexpr double kPi = 3.14159265358979323846;

}  // namespace

double compute_critical_density_from_wavelength_cm(const double lambda_cm) {
  if (!(lambda_cm > 0.0)) {
    return 0.0;
  }

  const double c = tenryu::core::constants::c_light;
  const double omega = 2.0 * kPi * c / lambda_cm;
  return kElectronMass * omega * omega /
         (4.0 * kPi * kElementaryCharge * kElementaryCharge);
}

}  // namespace tenryu::laser

