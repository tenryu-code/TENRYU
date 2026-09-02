#pragma once

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

// work_p_per_cell is a corrector-stage power [erg/s].  Therefore
// Vdot_work = -work_p_per_cell / max(p_half, tiny): multiplying the work
// bucket by dt and dividing the deposited work by dt leaves the same power.
void axis_cell_ledger_step_begin(core::State& state, const core::Config& cfg);

void axis_cell_ledger_step_end(core::State& state,
                               const core::Config& cfg,
                               double dt_op);

}  // namespace tenryu::hydro
