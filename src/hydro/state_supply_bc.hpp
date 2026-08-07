#pragma once

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {

[[nodiscard]] bool has_state_supply_bc(const core::Config& cfg);

void init_state_supply_mask(core::State& state, const core::Config& cfg);
void reset_state_supply_step_tallies(core::State& state);
void reset_state_supply_all_tallies(core::State& state);

void apply_state_supply_zonal_override(core::State& state,
                                       const core::Config& cfg,
                                       bool accumulate_tally = true);
void restore_state_supply_material_velocity(core::State& state,
                                            const core::Config& cfg);
void tally_state_supply_delta_E(core::State& state,
                                const core::Config& cfg,
                                bool accumulate_tally = true);

}  // namespace tenryu::hydro
