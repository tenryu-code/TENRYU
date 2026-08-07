#pragma once

#include <string>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

enum class Boundary2DType {
  FREE,
  FIXED,
  REFLECT,
  PRESSURE,
  AXIS,
  STATE_SUPPLY,
  PINNED,
};

[[nodiscard]] Boundary2DType parse_boundary_2d_type(const std::string& value);

void validate_boundary_2d(const core::Config& cfg);
void apply_boundary_2d(core::State& state, const core::Config& cfg, double t);

}  // namespace tenryu::hydro
