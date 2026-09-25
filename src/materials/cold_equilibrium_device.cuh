#pragma once

#include "materials/cold_equilibrium.hpp"
#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

struct DeviceColdInverseResult {
  double T;
  double Pe;       // corrected electron pressure at T
  double qe;       // conserved coordinate consistent with T (== target when status == 0)
  double ee_phys;
  double cv;       // corrected heat capacity at T
  int status;      // ColdInverseResult::status
  int iterations;
};

#ifdef __CUDACC__
// Electron closure with the cold-equilibrium branch: inverts qe_target to T on
// the base electron table `tab` (fixed rho bracket), then evaluates the
// corrected pressure/heat capacity at that T. T_floor is the runtime floor.
__device__ inline DeviceColdInverseResult device_cold_inverse_reclose(
    const DeviceEOSTableView& tab, const ColdEquilibriumView& cold, const double rho,
    const double qe_target, const double T_floor) {
  DeviceColdInverseResult out{};
  const RhoBracket rb = find_rho_bracket(tab, rho);
  const double T_lo = fmax(fmax(T_floor, 1.0e-30), exp(tab.log_T_min));
  const double T_hi = exp(tab.log_T_max);
  // Base evaluators: `tab` may itself carry the cold branch, and the
  // correction is applied once, by cold_inverse_Te / cold_electron_forward.
  const auto ee_base = [&](const double T) { return device_eos_energy_base(tab, rb, log(T)); };
  const auto cv_base = [&](const double T) { return device_eos_cv_base(tab, rb, log(T)); };
  const ColdInverseResult inv =
      cold_inverse_Te(cold, rho, qe_target, T_lo, T_hi, 0.0, ee_base, cv_base, tab.log_T_grid,
                      tab.n_T);
  const double T = inv.T;
  const double logT = log(fmax(T, 1.0e-30));
  const ColdElectronState s = cold_electron_forward(
      cold, rho, T, device_eos_energy_base(tab, rb, logT), device_eos_pressure_base(tab, rb, logT),
      fmax(device_eos_cv_base(tab, rb, logT), 0.0));
  out.T = T;
  out.Pe = s.Pe;
  out.qe = (inv.status == 0) ? qe_target : s.qe;
  out.ee_phys = out.qe + (s.ee_phys - s.qe);
  out.cv = s.cv;
  out.status = inv.status;
  out.iterations = inv.iterations;
  return out;
}
#endif  // __CUDACC__

}  // namespace tenryu::materials
