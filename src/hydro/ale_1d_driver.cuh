#pragma once

#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_types.cuh"

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::hydro::ale1d {

// Public entry point. Returns Ale1dStepResult describing what, if anything, was done.
// Per V3 Section 4 data flow: 21 steps, two-phase commit, no state mutation until commit.
//
// V1 scope: 1D_SPH only, deterministic radiation (FLD/SN) only.
// IMC/DDMC must be rejected at ConfigValidation time, not silently.
Ale1dStepResult apply_ale_1d(core::State& state,
                              const core::Config& cfg,
                              const HydroEOSContext* eos_ctx = nullptr);

// Acoustic time-step bounds of the candidate gates (NUMERICS §3.4.1):
// min_i (r_{i+1} - r_i) / c_i over the cells with c_i > 0 of the current mesh,
// and of a candidate mesh whose cell takes the largest sound speed of the
// current cells it overlaps (the remap fills it from them). Infinite when no
// cell has c_i > 0.
struct Ale1dAcousticDtBounds {
  double current = std::numeric_limits<double>::infinity();
  double candidate = std::numeric_limits<double>::infinity();
};

Ale1dAcousticDtBounds acoustic_dt_bounds(const std::vector<double>& r_current,
                                         const std::vector<double>& cs_current,
                                         const std::vector<double>& r_candidate);

}  // namespace tenryu::hydro::ale1d
