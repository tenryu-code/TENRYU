#pragma once

#include <cstdint>

#include "numerics/rkl2_sts.hpp"

namespace tenryu::radiation {

struct DiffusionStepInputs {
  double* diff_E = nullptr;                 // [n_cells * G] in/out [erg/cm^3]
  const double* sigma_R = nullptr;          // [n_cells * G] [1/cm]
  const double* vol = nullptr;              // [n_cells] [cm^3]
  const double* node_r = nullptr;           // [n_cells + 1] [cm]
  const std::uint8_t* diff_cell = nullptr;  // [n_cells]
  double* face_current_in = nullptr;        // [(n_cells + 1) * G] [erg], consumed
  double face_current_dt = 0.0;             // legacy tally interval; energy is deposited directly
  int n_cells = 0;
  int n_groups = 1;
  double dt = 0.0;
  int bc_inner = 0;  // 0=reflect, 1=vacuum
  int bc_outer = 0;  // 0=reflect, 1=vacuum
};

struct DiffusionStepResult {
  int rkl2_stages = 0;
  int rkl2_subcycles = 0;
  double E_before = 0.0;
  double E_after = 0.0;
  double E_positive_before = 0.0;
  double E_positive_after = 0.0;
  double E_negative_before = 0.0;
  double E_negative_after = 0.0;
  double E_limiter_removed = 0.0;
  double E_leaked = 0.0;
  double dt_explicit = 0.0;
  bool rkl2_skipped = false;
  bool positivity_limited = false;
};

struct DiffusionStepPlan {
  int rkl2_stages = 0;
  int rkl2_subcycles = 0;
  double dt_explicit = 0.0;
  bool skip = false;
};

DiffusionStepResult deterministic_diffusion_step_1d(
    const DiffusionStepInputs& in,
    const tenryu::numerics::RKL2Coefficients& coeff);

DiffusionStepPlan deterministic_diffusion_plan_1d(
    const DiffusionStepInputs& in,
    int sts_max_stages,
    double safety);

DiffusionStepResult deterministic_diffusion_step_1d(
    const DiffusionStepInputs& in,
    int sts_max_stages,
    double damping,
    double safety);

}  // namespace tenryu::radiation
