#pragma once

#include <vector>

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

enum class CoreClearanceControllerEventKind {
  FOUR_HALVINGS,
  BF_FIRST_REJECT,
  GEOMETRY_RETRY,
};

void core_clearance_controller_run_start(const core::State& state,
                                         const core::Config& cfg,
                                         int rank);

void core_clearance_controller_record_step(const core::State& state,
                                           int step,
                                           double t);

void core_clearance_controller_pre_lagrange_rezone(core::State& state,
                                                   const core::Config& cfg);

bool core_clearance_controller_bcr_rezone_active();

bool core_clearance_controller_promote_cell(int cell_id);

bool core_clearance_controller_request_feasible_seed(int cell_id);

void core_clearance_controller_note_event(
    CoreClearanceControllerEventKind kind);

// G3.1: true when the transactional controller is active (variant env set AND
// the BCR rezone machinery is enabled/initialized for this run).
bool core_clearance_controller_g31_active();

// G3.1 Phase-B reject flag: returns true exactly once after a trial remap was
// applied and then failed post-conditions (caller must restore the pre-rezone
// snapshot); clears the flag.
bool core_clearance_controller_g31_take_trial_reject();

void core_clearance_controller_build_active_target(
    const core::State& state,
    int step,
    double t,
    double dt,
    std::vector<double>& lagrangian_r,
    std::vector<double>& lagrangian_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z);

void core_clearance_controller_record_active_accept(
    const core::State& state,
    int step,
    double t,
    double dt,
    double sigma);

void core_clearance_controller_flush();

}  // namespace tenryu::hydro
