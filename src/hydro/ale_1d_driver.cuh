#pragma once

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

}  // namespace tenryu::hydro::ale1d
