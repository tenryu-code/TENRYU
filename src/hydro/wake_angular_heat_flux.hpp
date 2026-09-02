#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro::wake_angular_heat_flux {

// Returns true when the energy field was edited and requires EOS reclosure.
bool apply(core::State& state, const core::Config& cfg, double dt);

void destroy();

}  // namespace tenryu::hydro::wake_angular_heat_flux
