#pragma once

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "hydro/shock_tracker.hpp"

namespace tenryu::hydro {

enum class AdaptiveAVMode {
  Base,
  PrimaryFull,
  PrimaryTaper,
  Rebound,
};

struct AdaptiveAVFields {
  core::CellField1D c1;
  core::CellField1D c2;
  core::CellField1D heat_C;
  core::CellField1D Cpsv;
  core::CellField1D cbulk;
};

// dt: current hydro step [s]; used only when
// Numerics.hydro.adaptive_av.hysteresis_tau > 0 to form the
// timestep-invariant gate blend w(dt) = 1 - exp(-dt/tau) (k01 §8.1,
// AI review 2026-07-26). hysteresis_tau == 0 keeps the legacy fixed
// per-step hysteresis_w blend.
void build_adaptive_av_fields_1d(core::State& state,
                                 const core::Config& cfg,
                                 const LeadingShockState& shock,
                                 double dt,
                                 AdaptiveAVFields& out);

}  // namespace tenryu::hydro
