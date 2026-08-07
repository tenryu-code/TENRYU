#pragma once

#include <vector>

#include <cuda_runtime.h>

#include "burn/network.cuh"
#include "burn/partition.hpp"

namespace tenryu::burn {

struct BurnStage2DParams {
  BurnChannels channels{false, false, false};
  int scheme = 0;  // 0=local deposition, 1=birth-source transport (S_birth fill)
  bool use_fraley_partition = false;  // else LP table
  int screening_mode = 0;  // ScreeningMode enum: 0=none,1=salpeter,2=chugunov_dewitt
  double T_floor_keV = 0.2;
  double eps_deplete = 0.1;
  int subcycle_max = 64;
  double vf_threshold = 1.0e-3;
  double explicit_source_limit = 0.2;
  double dt_s = 0.0;
};

struct BurnStage2DInputs {           // all size n_cells unless noted; host
  int n_cells = 0;
  int n_mat = 0;
  const double* rho = nullptr;     // g/cm^3
  const double* vol = nullptr;     // cm^3 (true RZ cell volumes)
  const double* Te_eV = nullptr;
  const double* Ti_eV = nullptr;
  const double* zbar = nullptr;
  const double* A_eff = nullptr;   // proton-mass units
  const double* ee = nullptr;      // erg/g (specific electron IE)
  const double* ei = nullptr;      // erg/g
  const double* volFrac = nullptr; // [n_cells*n_mat], cell-major
  const int* fuel_mat = nullptr;   // n_fuel_mat material indices
  int n_fuel_mat = 0;
  // MPI (Option C, M18d d2): owned cell window for the network loop.
  // Default (0, 0) means full range — 1D replication and serial callers.
  // Windowed 2D stages produce owned-partial tallies (dE arrays are
  // full-size zero-initialized, so the dep sums stay correct) that the
  // driver budget Allreduce completes.
  int c_begin = 0;
  int c_end = 0;
};

struct BurnStage2DResult {           // scalars in erg (already x vol, x dt)
  double released_charged = 0.0;
  double released_neutron = 0.0;
  double dep_e = 0.0;
  double dep_i = 0.0;
  double esc_charged = 0.0;        // = released_charged - dep_e - dep_i (FP)
  double esc_neutron = 0.0;        // = released_neutron (free-streaming)
  double n_neutrons_dt = 0.0;      // counts (absolute number)
  double n_neutrons_dd = 0.0;
  double dt_limit_s = 0.0;         // +inf when no deposition
  int max_substeps = 0;
};

// Advance species inventories over dt and produce per-cell deposition (2D RZ:
// per-cell network + local deposit or birth-source fill; no fuel-region
// geometry — the point-sphere kernel is 1D_SPH-only).
// burn_y: [n_cells*kNumSpecies] cell-major specific inventories Y_s
// [particles/gram], mutated in place.
// dE_e/dE_i: [n_cells] OUTPUT, erg added to electrons/ions per cell.
// rate_diag: [n_cells] OUTPUT, total reactions/cm^3/s this step.
// Qe_diag/Qi_diag: [n_cells] OUTPUT, erg/cm^3/s.
BurnStage2DResult compute_burn_step_2d(const BurnStage2DInputs& in,
                                       const BurnStage2DParams& p,
                                       const PartitionTable& table,
                                       std::vector<double>& burn_y,
                                       std::vector<double>& dE_e,
                                       std::vector<double>& dE_i,
                                       std::vector<double>& rate_diag,
                                       std::vector<double>& Qe_diag,
                                       std::vector<double>& Qi_diag,
                                       std::vector<double>* S_birth = nullptr);

}  // namespace tenryu::burn
