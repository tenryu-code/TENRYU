#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {
struct HydroEOSContext;
}  // namespace tenryu::hydro

namespace tenryu::hydro::per_material {

void refresh_per_material_derived_cell_fields(core::State& state,
                                              const core::Config& cfg,
                                              const HydroEOSContext* eos_ctx,
                                              bool force_invalidate_all = false);

void invalidate_per_material_caches(core::State& state, int c, int m);
void invalidate_per_material_caches_all(core::State& state);

}  // namespace tenryu::hydro::per_material
