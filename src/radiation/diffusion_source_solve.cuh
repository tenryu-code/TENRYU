#pragma once

#include <cstdint>

#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct DiffusionSourceSolveInputs {
  double* diff_E = nullptr;          // [n_cells * G] in/out [erg/cm^3]
  double* ee = nullptr;              // [n_cells] in/out electron specific energy [erg/g]
  double* Te = nullptr;              // [n_cells] in/out [eV]
  double* Pe = nullptr;              // [n_cells] out [erg/cm^3]
  const double* sigma_P = nullptr;   // [n_cells * G] Planck absorption opacity [1/cm]
  const double* vol = nullptr;       // [n_cells] [cm^3]
  const double* rho = nullptr;       // [n_cells] [g/cm^3]
  const double* mass = nullptr;      // [n_cells] optional, rho * vol [g]
  const double* cv_e = nullptr;      // [n_cells] optional mass cv_e [erg/(g*eV)]
  const std::uint8_t* diff_cell = nullptr;  // [n_cells] mask
  double* rad_dep = nullptr;         // [n_cells * G] diagnostic out, accumulated [erg]
  double* rad_emit = nullptr;        // [n_cells * G] diagnostic out, accumulated [erg]
  int n_cells = 0;
  int n_groups = 1;
  double dt_s = 0.0;
  PlanckTableDeviceView planck{};
  double cv_e_const = 0.0;           // fallback mass cv_e [erg/(g*eV)]
  double pressure_gamma_minus_one = 1.0;
  double temperature_floor_eV = 1.0e-3;
  bool use_table_eos = false;
};

struct DiffusionSourceSolveResult {
  int max_newton_iter = 0;
  int n_failures = 0;
  double matter_delta = 0.0;  // [erg], positive when matter gains energy
  double rad_delta = 0.0;     // [erg], positive when diffusion radiation gains energy
};

DiffusionSourceSolveResult diffusion_source_solve_cuda(
    const DiffusionSourceSolveInputs& in);

}  // namespace tenryu::radiation
