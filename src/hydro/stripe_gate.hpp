#pragma once

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

void stripe_gate_begin(core::State& state, const core::Config& cfg);

void stripe_gate_end(core::State& state,
                     const core::Config& cfg,
                     double dt,
                     double t_after);

void stripe_gate_finalize();

}  // namespace tenryu::hydro
