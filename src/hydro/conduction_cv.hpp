#pragma once

#include <cfloat>
#include <cmath>

#include "core/macros.hpp"

namespace tenryu::hydro {

// Electron heat capacity per unit mass [erg/(g eV)] that the effective-
// diffusivity conduction solve uses in a cell (NUMERICS §3.4.1): the EOS
// value when it is positive, otherwise the ideal-gas value
// zbar e / (A m_p (gamma - 1)), so a cell whose table heat capacity is <= 0
// still conducts. The energy booked after the solve must use this same value,
// or the energy the solve moved through such a cell is not the energy that is
// booked. The constants and floors are the conduction kernels' own.
TENRYU_HOST_DEVICE inline double conduction_solve_cv_e(const double state_cv_e,
                                                       const double zbar,
                                                       const double gamma_eff,
                                                       const double A_eff) {
  if (state_cv_e > 0.0) {
    return state_cv_e;
  }
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kProtonMass = 1.6726219e-24;
  constexpr double kMinEffectiveA = 1.0e-12;
  constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;
  const double z = fmax(zbar, 0.0);
  const double gamma = fmax(gamma_eff, kMinEffectiveGamma);
  const double A = fmax(A_eff, kMinEffectiveA);
  return (gamma > 1.0 && A > 0.0) ? (z * kEvToErg / (A * kProtonMass * (gamma - 1.0))) : 0.0;
}

// Ion heat capacity per unit mass [erg/(g eV)] of the ion conduction solve
// (NUMERICS §4.6), by the rule of conduction_solve_cv_e: the EOS value when it
// is finite and positive, otherwise the ideal-gas value e / (A m_p (gamma - 1)).
// The energy booked after the solve uses this same value.
TENRYU_HOST_DEVICE inline double conduction_solve_cv_i(const double state_cv_i,
                                                       const double gamma_eff,
                                                       const double A_eff) {
  if (state_cv_i > 0.0 && state_cv_i <= DBL_MAX) {  // finite and positive
    return state_cv_i;
  }
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kProtonMass = 1.6726219e-24;
  constexpr double kMinEffectiveA = 1.0e-12;
  constexpr double kMinEffectiveGamma = 1.0 + 1.0e-12;
  const double gamma = fmax(gamma_eff, kMinEffectiveGamma);
  const double A = fmax(A_eff, kMinEffectiveA);
  return (gamma > 1.0 && A > 0.0) ? (kEvToErg / (A * kProtonMass * (gamma - 1.0))) : 0.0;
}

// Ideal-gas closure of a 1D cell whose material has no EOS table, in the form
// of the 1D hydro closure (hydro_1d_bodies.cuh ideal_gas_cv_i_mass /
// ideal_gas_cv_e_mass in 2T, the ideal-gas branch of enforce_1t_closure_kernel
// in 1T): the cell's effective gamma and A (State::ensure_cell_material_props)
// and zbar, with the same floors. cv_e is the electron heat capacity per unit
// mass in 2T and the total one in 1T, cv_i the ion one in 2T and 0 in 1T
// [erg/(g eV)]; the pressures are P_s = gm1 rho e_s.
struct IdealGasCellCv {
  double cv_e;
  double cv_i;
  double gm1;
};

TENRYU_HOST_DEVICE inline IdealGasCellCv ideal_gas_cell_cv(const bool two_temp,
                                                           const double gamma_eff,
                                                           const double A_eff,
                                                           const double zbar,
                                                           const double cv_e_override,
                                                           const double rho) {
  constexpr double kEvToErg = 1.6022e-12;
  constexpr double kProtonMass = 1.6726219e-24;
  const double gamma = fmax(gamma_eff, 1.0 + 1.0e-12);
  const double A = fmax(A_eff, 1.0e-12);
  const double z = fmax(zbar, 0.0);
  const double rho_safe = fmax(rho, 1.0e-30);
  IdealGasCellCv out{};
  out.gm1 = gamma - 1.0;
  if (two_temp) {
    out.cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
    out.cv_e = (cv_e_override > 0.0) ? cv_e_override / rho_safe : z * out.cv_i;
  } else {
    out.cv_i = 0.0;
    out.cv_e = (cv_e_override > 0.0)
                   ? cv_e_override / rho_safe
                   : (1.0 + z) * kEvToErg / (A * kProtonMass * (gamma - 1.0));
  }
  return out;
}

}  // namespace tenryu::hydro
