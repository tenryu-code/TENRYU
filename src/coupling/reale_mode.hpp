#pragma once

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::coupling::reale_mode {

void validate_runtime_scope(const core::Config& cfg, int n_ranks);

// Returns true only when a new mesh and its remapped hydro fields commit.
bool rezone_step(core::State& state, const core::Config& cfg, double dt,
                 double dt_hydro_bound);

}  // namespace tenryu::coupling::reale_mode
