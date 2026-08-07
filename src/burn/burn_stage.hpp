#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "burn/network.cuh"
#include "burn/partition.hpp"

namespace tenryu::burn {

struct BurnStageParams {
  BurnChannels channels{false, false, false};
  int scheme = 0;  // 0=fraley deposition, 1=birth-source transport
  bool use_fraley_partition = false;  // else LP table
  int screening_mode = 0;  // ScreeningMode enum: 0=none,1=salpeter,2=chugunov_dewitt
  double T_floor_keV = 0.2;
  double eps_deplete = 0.1;
  int subcycle_max = 64;
  double vf_threshold = 1.0e-3;
  double explicit_source_limit = 0.2;
  double dt_s = 0.0;
  bool neutron_heating = false;    // v2-E two-line first-collision heating
  int neutron_heating_n_mu = 16;   // even Gauss-Legendre order
};

struct BurnStageInputs {           // all size n_cells unless noted; host
  int n_cells = 0;
  int n_mat = 0;
  const double* r_node = nullptr;  // size n_cells+1, cell edges [cm]
  const double* rho = nullptr;     // g/cm^3
  const double* vol = nullptr;     // cm^3
  const double* Te_eV = nullptr;
  const double* Ti_eV = nullptr;
  const double* v_r = nullptr;      // [n_cells] cell radial velocity, cm/s (nullptr -> zero)
  const double* zbar = nullptr;
  const double* A_eff = nullptr;   // proton-mass units
  const double* ee = nullptr;      // erg/g (specific electron IE)
  const double* ei = nullptr;      // erg/g
  const double* volFrac = nullptr; // [n_cells*n_mat], cell-major
  const int* fuel_mat = nullptr;   // n_fuel_mat material indices
  int n_fuel_mat = 0;
};

struct BurnStageResult {           // scalars in erg (already x vol, x dt)
  double released_charged = 0.0;
  double released_neutron = 0.0;
  double dep_e = 0.0;
  double dep_i = 0.0;
  double esc_charged = 0.0;        // = released_charged - dep_e - dep_i (FP)
  double esc_neutron = 0.0;        // = released_neutron (v1 free-streaming)
  double n_neutrons_dt = 0.0;      // counts (absolute number)
  double n_neutrons_dd = 0.0;
  double w_dt = 0.0;               // sum r1*vol (DT neutron rate, 1/s)
  double w_dd = 0.0;               // sum r3*vol
  double wTi_dt = 0.0;             // sum r1*vol*Ti_eV
  double wTi_dd = 0.0;
  double wvr2_dt = 0.0;            // sum r1*vol*v_r^2 (cm^2/s^2 weighted)
  double wvr2_dd = 0.0;
  double dt_limit_s = 0.0;         // +inf when no deposition
  int max_substeps = 0;
  int max_substeps_required = 0;   // largest ceil(dt*rate/eps) over cells
  int subcycle_saturated_cells = 0;  // cells where required > subcycle_max
  int burn_region_first = -1;      // fuel-region cell index range (diag)
  int burn_region_last = -1;
  double nh_dep_e = 0.0;           // v2-E neutron-heating ledgers (erg)
  double nh_dep_i = 0.0;
  double nh_degraded = 0.0;        // scattered remainder (inside esc_neutron)
  double nh_escaped = 0.0;         // uncollided
  double nh_conservation_resid = 0.0;
};

// Advance species inventories over dt and produce per-cell deposition.
// burn_y: [n_cells*kNumSpecies] cell-major specific inventories Y_s [particles/gram], mutated in place.
// dE_e/dE_i: [n_cells] OUTPUT, erg added to electrons/ions per cell.
// rate_diag: [n_cells] OUTPUT, total reactions/cm^3/s this step.
// Qe_diag/Qi_diag: [n_cells] OUTPUT, erg/cm^3/s.
BurnStageResult compute_burn_step_1d(const BurnStageInputs& in,
                                     const BurnStageParams& p,
                                     const PartitionTable& table,
                                     std::vector<double>& burn_y,
                                     std::vector<double>& dE_e,
                                     std::vector<double>& dE_i,
                                     std::vector<double>& rate_diag,
                                     std::vector<double>& Qe_diag,
                                     std::vector<double>& Qi_diag,
                                     std::vector<double>* S_birth = nullptr);

}  // namespace tenryu::burn
