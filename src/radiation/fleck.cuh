#pragma once

#include <cstdint>
#include <limits>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct FleckView {
  const double* rho = nullptr;          // [n_cells]
  const double* Te = nullptr;           // [n_cells]
  const double* zbar = nullptr;         // [n_cells]
  const double* state_cv_e = nullptr;   // [n_cells] mass cv_e [erg/(g*eV)] from EOS state
  const std::uint8_t* cell_is_void = nullptr;  // [n_cells] host mask
  const double* sigma_a = nullptr;      // [n_cells * G]
  PlanckTableDeviceView planck{};       // b_g(T) interpolation table
  double* f_fleck = nullptr;            // [n_cells]
  double* sigma_a_eff = nullptr;        // [n_cells * G]
  double* sigma_s_eff = nullptr;        // [n_cells * G]
  int n_cells = 0;
  int n_groups = 1;
  double dt = 0.0;
};

struct DtRadDiagnostics {
  double dt_rad = std::numeric_limits<double>::infinity();
  int limiting_cell = -1;
  double limiting_sigma_P = 0.0;
  double limiting_beta = 0.0;
  double limiting_Cv_e = 0.0;
  double limiting_Te = 0.0;
  double limiting_rho = 0.0;
};

void compute_fleck_and_sigma_eff_cuda(const FleckView& view,
                                      const core::Config& cfg);

double compute_dt_rad_limit(const core::State& state,
                            const core::Config& cfg);
double compute_dt_rad_limit(const core::State& state,
                            const core::Config& cfg,
                            DtRadDiagnostics* diag);

}  // namespace tenryu::radiation
