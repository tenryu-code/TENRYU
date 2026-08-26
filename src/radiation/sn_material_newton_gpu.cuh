#pragma once

#include "materials/eos_device_table.cuh"
#include "radiation/planck_table.cuh"
#include "radiation/sn_transport_gpu.cuh"

namespace tenryu::materials {
struct EOSTableTriplet;
}

namespace tenryu::radiation {

struct SnMaterialNewton1DInputs {
  const double* sigma_a = nullptr;  // [n_cells * n_groups], [1/cm]
  const double* sigma_pe = nullptr; // optional [n_cells * n_groups], [1/cm]
  const double* rad_E = nullptr;    // [n_cells * n_groups], [erg/cm^3]
  const double* E_star_override = nullptr; // optional [n_cells * n_groups], face-flux E* [erg/cm^3]
  double* rad_E_out = nullptr;      // where to write E_g^{n+1}; required by 1D production closure.
  bool eos_low_density_extrap = false;  // If true, use analytic-ideal EOS below table rho_min.
  bool energy_authoritative = false;
  const double* rho = nullptr;      // [n_cells], [g/cm^3]
  const double* cv_e = nullptr;     // optional [n_cells], [erg/(g*eV)]
  const double* vol = nullptr;      // [n_cells], [cm^3]
  const double* zbar = nullptr;     // optional [n_cells]
  const double* Te_old = nullptr;   // [n_cells], [eV]
  double* Te = nullptr;             // [n_cells], [eV]
  double* ee = nullptr;             // [n_cells], [erg/g]
  double* Pe = nullptr;             // optional [n_cells], [erg/cm^3]
  double* rad_dep = nullptr;        // [n_cells * n_groups], [erg]
  double* rad_emit = nullptr;       // [n_cells * n_groups], [erg]
  double* diag_rad_emission_at_Tnp1 = nullptr;  // [n_cells * n_groups], [erg/cm^3]
  double* diag_rad_absorption = nullptr;        // [n_cells * n_groups], [erg/cm^3]
  double* diag_clip_energy = nullptr;           // [n_cells * n_groups], [erg/cm^3]
  double* diag_clip_full_deficit = nullptr;     // [n_cells * n_groups], [erg/cm^3]
  double* delta_T_rel = nullptr;    // [n_cells]
  PlanckTableDeviceView planck{};
  materials::DeviceEOSTableView electron_eos{}; // optional TMAT electron EOS
  int n_cells = 0;
  int n_groups = 0;
  double dt = 0.0;
  double cv_e_const = 0.0;          // [erg/(g*eV)]
  double Cv_e_const = 0.0;          // [erg/(cm^3*eV)]
  // Shared per-cell effective ideal-gas properties (multi-material fix; see
  // State::ensure_cell_material_props). Null pointers fall back to A=1,
  // gamma=5/3 for direct-constructed test inputs.
  const double* A_eff = nullptr;      // [n_cells]
  const double* gamma_eff = nullptr;  // [n_cells]
  double temperature_floor_eV = 1.0e-12;
  int max_iterations = 10;
  double tolerance = 1.0e-6;
  // MPI (Option C, M18c slice-3): owned cell window. Default (0, 0) means
  // full range — the 1D replicated path and serial callers keep the full
  // -line Newton; the 2D distributed path windows to owned cells so
  // stale-finite far cells cannot raise spurious retry flags.
  int c_begin = 0;
  int c_end = 0;
};

materials::DeviceEOSTableView sn_electron_eos_device_view(
    const materials::EOSTableTriplet* tables);

// Returns the timestep-rejection request flags (0 = converged everywhere;
// bit 1 = R(T_floor) > 0 floor-root, bit 2 = adaptive bracket-expansion
// failure). The caller (SN radiation stage) forwards a nonzero value to
// State::sn_material_retry_flag so the driver can reject and retry the global
// step (NUMERICS §6.8).
int solve_sn_material_temperature_newton_gpu(
    const SnMaterialNewton1DInputs& in);

void solve_sn_material_temperature_newton_2d_legacy_gpu(
    const SnMaterialNewton1DInputs& in);

void solve_sn_material_temperature_newton_gpu(
    const SNMaterialCouplingGPUInputs& in,
    int n_cells,
    int n_groups,
    double temperature_floor_eV);

}  // namespace tenryu::radiation
