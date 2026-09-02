#pragma once

#include "materials/eos_device_table.cuh"

namespace tenryu::hydro::remap_eos {

inline constexpr int kClosureStatusBracketFailure = 1;
inline constexpr int kClosureStatusBelowTable = 2;  // lower_clamp NOT explained by the floor
inline constexpr int kClosureStatusAboveTable = 4;
inline constexpr int kClosureStatusNonfinite = 8;

struct SpeciesClosureResult {
  double T;
  double P;
  double cv;
  int status;
};

__device__ inline SpeciesClosureResult close_species_energy_authoritative(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho,
    const double mass,
    double* e_s,
    const double T_floor,
    const double zbar,
    const double A_amu,
    const bool low_density_extrap,
    double* ledger_dE) {
  const double e_raw = *e_s;
  if (!isfinite(e_raw) || !(mass > 0.0)) {
    return SpeciesClosureResult{
        fmax(T_floor, 1.0e-30), 0.0, 0.0, kClosureStatusNonfinite};
  }

  const auto inv =
      tenryu::materials::device_inverse_reclose_with_low_density_extrap(
          tab,
          rho,
          e_raw,
          T_floor,
          zbar,
          A_amu,
          low_density_extrap);

  int status = 0;
  if (inv.bracket_failure != 0) {
    status |= kClosureStatusBracketFailure;
  }
  if (inv.upper_clamp != 0) {
    status |= kClosureStatusAboveTable;
  }

  const double accepted_floor = fmax(T_floor, 1.0e-30);
  if (inv.T <= accepted_floor && inv.energy > e_raw) {
    *e_s = inv.energy;
    *ledger_dE += mass * (*e_s - e_raw);
  } else if (inv.lower_clamp != 0) {
    status |= kClosureStatusBelowTable;
  }
  // NOTE: in-range inversion residual is NEVER erased by overwriting e_s with
  // inv.energy -- design §19 item 1 (E authoritative).

  SpeciesClosureResult result{inv.T, inv.pressure, inv.cv, status};
  if (!isfinite(result.P)) {
    result.status |= kClosureStatusNonfinite;
  }
  return result;
}

// A separate alias keeps the one-temperature semantic mode explicit at call sites.
__device__ inline SpeciesClosureResult close_total_1t_energy_authoritative(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho,
    const double mass,
    double* e_s,
    const double T_floor,
    const double zbar,
    const double A_amu,
    const bool low_density_extrap,
    double* ledger_dE) {
  return close_species_energy_authoritative(tab,
                                            rho,
                                            mass,
                                            e_s,
                                            T_floor,
                                            zbar,
                                            A_amu,
                                            low_density_extrap,
                                            ledger_dE);
}

}  // namespace tenryu::hydro::remap_eos
