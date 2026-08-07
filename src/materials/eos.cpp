#include "materials/eos.hpp"

#include <cmath>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::materials {
namespace {

constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = tenryu::core::constants::proton_mass;

}  // namespace

IdealGasEOS::IdealGasEOS(const double gamma,
                         const double A,
                         const double Z_bar)
    : gamma_(gamma), A_(A), Z_bar_(Z_bar) {
  TENRYU_ASSERT(gamma_ > 1.0, "IdealGasEOS requires gamma > 1");
  TENRYU_ASSERT(A_ > 0.0, "IdealGasEOS requires A > 0");
  TENRYU_ASSERT(Z_bar_ >= 0.0, "IdealGasEOS requires Z_bar >= 0");

  cv_i_ = kEvToErg / (A_ * kProtonMass * (gamma_ - 1.0));
  cv_e_ = Z_bar_ * kEvToErg / (A_ * kProtonMass * (gamma_ - 1.0));
  cv_total_ = cv_i_ + cv_e_;

  TENRYU_ASSERT(cv_i_ > 0.0, "IdealGasEOS computed non-positive Cv_i");
  TENRYU_ASSERT(cv_e_ >= 0.0, "IdealGasEOS computed negative Cv_e");
  TENRYU_ASSERT(cv_total_ > 0.0, "IdealGasEOS computed non-positive Cv_total");
}

double IdealGasEOS::pressure(const double rho, const double e) const {
  TENRYU_ASSERT(rho >= 0.0, "IdealGasEOS::pressure requires rho >= 0");
  TENRYU_ASSERT(e >= 0.0, "IdealGasEOS::pressure requires e >= 0");
  return (gamma_ - 1.0) * rho * e;
}

double IdealGasEOS::energy(const double, const double T) const {
  TENRYU_ASSERT(T >= 0.0, "IdealGasEOS::energy requires T >= 0");
  return cv_total_ * T;
}

double IdealGasEOS::temperature(const double, const double e) const {
  TENRYU_ASSERT(e >= 0.0, "IdealGasEOS::temperature requires e >= 0");
  return e / cv_total_;
}

double IdealGasEOS::sound_speed(const double, const double e) const {
  TENRYU_ASSERT(e >= 0.0, "IdealGasEOS::sound_speed requires e >= 0");
  return std::sqrt(gamma_ * (gamma_ - 1.0) * e);
}

double IdealGasEOS::cv(const double, const double) const {
  return cv_total_;
}

double IdealGasEOS::pressure_ion(const double rho, const double e_i) const {
  TENRYU_ASSERT(rho >= 0.0, "IdealGasEOS::pressure_ion requires rho >= 0");
  TENRYU_ASSERT(e_i >= 0.0, "IdealGasEOS::pressure_ion requires e_i >= 0");
  return (gamma_ - 1.0) * rho * e_i;
}

double IdealGasEOS::pressure_electron(const double rho, const double e_e) const {
  TENRYU_ASSERT(rho >= 0.0, "IdealGasEOS::pressure_electron requires rho >= 0");
  TENRYU_ASSERT(e_e >= 0.0, "IdealGasEOS::pressure_electron requires e_e >= 0");
  return (gamma_ - 1.0) * rho * e_e;
}

double IdealGasEOS::energy_ion(const double, const double T_i) const {
  TENRYU_ASSERT(T_i >= 0.0, "IdealGasEOS::energy_ion requires T_i >= 0");
  return cv_i_ * T_i;
}

double IdealGasEOS::energy_electron(const double, const double T_e) const {
  TENRYU_ASSERT(T_e >= 0.0, "IdealGasEOS::energy_electron requires T_e >= 0");
  return cv_e_ * T_e;
}

}  // namespace tenryu::materials
