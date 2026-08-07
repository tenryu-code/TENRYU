#pragma once

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

enum class HydroBoundaryType {
  FREE,
  FIXED,
  REFLECT,
  PRESSURE,
};

HydroBoundaryType parse_boundary_type_1d(const core::Config& cfg);

double pressure_ghost_1d(const core::State& state,
                         const core::Config& cfg,
                         HydroBoundaryType bc_type,
                         double eval_time);

void apply_boundary_1d(core::State& state, const core::Config& cfg);

}  // namespace tenryu::hydro
