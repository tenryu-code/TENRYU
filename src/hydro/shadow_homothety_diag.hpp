#pragma once

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

void shadow_homothety_run_start(const core::State& state,
                                const core::Config& cfg,
                                int rank);

void shadow_homothety_record_step(const core::State& state,
                                  int step,
                                  double t);

void shadow_homothety_record_post_lagrange(const core::State& state,
                                           int step,
                                           double t);

void shadow_homothety_flush();

}  // namespace tenryu::hydro
