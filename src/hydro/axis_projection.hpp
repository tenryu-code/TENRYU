#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro::axis_projection {

void run_start(const core::Config& cfg);

void observe_precommit(const core::State& state,
                       const core::Config& cfg,
                       const double* x_r_old,
                       const double* x_z_old,
                       double* pos_v_r,
                       double* pos_v_z,
                       double* ion_energy_old,
                       double dt,
                       double predicted_time,
                       const parallel::Reduction* reduction,
                       int rank);

void run_end(const core::Config& cfg, int rank);

}  // namespace tenryu::hydro::axis_projection
