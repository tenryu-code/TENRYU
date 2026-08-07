#pragma once

#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

struct HllcZFlux2DRZResult {
  bool applied = false;
  bool reconstructed_momentum = false;
  double E_floor_injected = 0.0;
  int clamp_count = 0;
  int rho_clamp_count = 0;
  unsigned long long hlle_fallback_count = 0;
  unsigned long long invalid_state_count = 0;
  double min_star_rho = 0.0;
  double min_star_p = 0.0;
};

HllcZFlux2DRZResult apply_hllc_z_flux_2d_rz(
    core::State& state,
    const core::Config& cfg,
    double dt,
    const std::int8_t* d_hydro_active,
    int rank,
    const char* hydro_half);

}  // namespace tenryu::hydro
