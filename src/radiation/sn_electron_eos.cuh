#pragma once

// Electron EOS evaluations shared by the 1D S_N matter updates (the
// cell-average Newton of radiation/sn_material_newton_gpu and the nodal
// update of radiation/sn_ld_1d_gpu): table energy, heat capacity and
// pressure with a linear-cv ideal tail above the table's temperature
// ceiling, the table inversion by bisection in ln T, and the mass heat
// capacity of the ideal-gas / cv_e / cv_e_override paths.

#include <cmath>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "materials/eos_device_table.hpp"

namespace tenryu::radiation::sn_electron_eos {

__host__ __device__ inline double finite_or_zero(const double value) {
  return isfinite(value) ? value : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double value) {
  return isfinite(value) ? fmax(value, 0.0) : 0.0;
}

constexpr double kTiny = 1.0e-300;

__device__ inline bool has_electron_eos_table(
    const materials::DeviceEOSTableView& electron_eos) {
  return electron_eos.n_rho > 0 && electron_eos.n_T > 0 &&
         electron_eos.P_table != nullptr &&
         electron_eos.e_table != nullptr &&
         electron_eos.cv_table != nullptr;
}

__device__ inline double eos_log_temperature(const double T,
                                             const double temperature_floor_eV) {
  return log(fmax(finite_or_zero(T), fmax(temperature_floor_eV, 1.0e-30)));
}

__device__ inline double sn_tail_T_top(
    const materials::DeviceEOSTableView& tab) {
  return exp(tab.log_T_max);
}

// e/cv/P with a linear-cv ideal tail above the table temperature ceiling;
// below/at the ceiling these are exactly the existing extrap evaluations.
__device__ inline double sn_eval_e_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double e_top = materials::device_eos_energy_extrap(
        tab, rb, rho_c, T_top, Zbar, A_c, low_density_extrap);
    const double cv_top = fmax(
        materials::device_eos_cv_extrap(tab, rb, rho_c, T_top, Zbar, A_c,
                                        low_density_extrap),
        0.0);
    if (isfinite(e_top) && cv_top > 0.0) {
      return e_top + cv_top * (T - T_top);
    }
  }
  return materials::device_eos_energy_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                             low_density_extrap);
}

__device__ inline double sn_eval_cv_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double cv_top = fmax(
        materials::device_eos_cv_extrap(tab, rb, rho_c, T_top, Zbar, A_c,
                                        low_density_extrap),
        0.0);
    if (cv_top > 0.0) {
      return cv_top;
    }
  }
  return materials::device_eos_cv_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                         low_density_extrap);
}

__device__ inline double sn_eval_P_tail(
    const materials::DeviceEOSTableView& tab,
    const materials::RhoBracket& rb,
    const double rho_c, const double T, const double Zbar, const double A_c,
    const bool low_density_extrap) {
  const double T_top = sn_tail_T_top(tab);
  if (isfinite(T_top) && T_top > 0.0 && T > T_top) {
    const double P_top = materials::device_eos_pressure_extrap(
        tab, rb, rho_c, T_top, Zbar, A_c, low_density_extrap);
    if (isfinite(P_top)) {
      return P_top * (T / T_top);
    }
  }
  return materials::device_eos_pressure_extrap(tab, rb, rho_c, T, Zbar, A_c,
                                               low_density_extrap);
}

template <bool EOS_TAIL>
__device__ inline double invert_e_e_via_bisection(
    const materials::DeviceEOSTableView& electron_eos,
    const materials::RhoBracket& eos_rho_bracket,
    double e_target,
    double T_floor,
    const double rho_c,
    const double Zbar,
    const double A_amu,
    const bool low_density_extrap) {
  if constexpr (EOS_TAIL) {
    const double T_top = sn_tail_T_top(electron_eos);
    const double e_top = materials::device_eos_energy_extrap(
        electron_eos, eos_rho_bracket, rho_c, T_top, Zbar, A_amu,
        low_density_extrap);
    const double cv_top = fmax(materials::device_eos_cv_extrap(
        electron_eos, eos_rho_bracket, rho_c, T_top, Zbar, A_amu,
        low_density_extrap), 0.0);
    if (isfinite(e_top) && cv_top > 0.0 && e_target > e_top) {
      return T_top + (e_target - e_top) / cv_top;
    }
  }
  // Bisect on log-T to find T such that device_eos_energy(rho, T) ≈ e_target.
  // Bracket: [log(T_floor), log(T_max)] where log(T_max) = electron_eos.log_T_max.
  // Iterate up to 60 times until xhi - xlo < 1e-14.
  double xlo = log(fmax(T_floor, 1.0e-30));
  double xhi = fmax(electron_eos.log_T_max, log(fmax(T_floor * 10.0, 1.0e-30)));
  for (int k = 0; k < 60; ++k) {
    const double xm = 0.5 * (xlo + xhi);
    if constexpr (EOS_TAIL) {
      const double e_m = sn_eval_e_tail(
          electron_eos, eos_rho_bracket, rho_c, exp(xm), Zbar, A_amu,
          low_density_extrap);
      if (e_m < e_target) xlo = xm;
      else                xhi = xm;
    } else {
      const double e_m = materials::device_eos_energy(electron_eos, eos_rho_bracket, xm);
      if (e_m < e_target) xlo = xm;
      else                xhi = xm;
    }
    if (xhi - xlo < 1.0e-14) break;
  }
  return exp(0.5 * (xlo + xhi));
}

__device__ inline double mass_heat_capacity(const double rho,
                                     const double zbar,
                                     const double cv_e_value,
                                     const double cv_e_const,
                                     const double Cv_e_const,
                                     const double A,
                                     const double gamma) {
  if (isfinite(cv_e_value) && cv_e_value > 0.0) {
    return cv_e_value;
  }
  const double rho_c = fmax(nonnegative_finite(rho), kTiny);
  if (isfinite(Cv_e_const) && Cv_e_const > 0.0) {
    return Cv_e_const / rho_c;
  }
  if (isfinite(cv_e_const) && cv_e_const > 0.0) {
    return cv_e_const;
  }
  const double gm1 = fmax(finite_or_zero(gamma) - 1.0, 1.0e-12);
  const double z = fmax(finite_or_zero(zbar), 0.0);
  return fmax(z * core::constants::eV_to_erg /
                  (fmax(finite_or_zero(A), 1.0e-12) *
                   core::constants::proton_mass * gm1),
              kTiny);
}

}  // namespace tenryu::radiation::sn_electron_eos
