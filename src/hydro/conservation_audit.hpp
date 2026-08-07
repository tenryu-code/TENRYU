#pragma once

#include "core/state.hpp"

namespace tenryu::hydro::conservation_audit {

bool enabled();

void emit_stage(const core::State& state,
                const char* stage,
                double boundary_work_delta = 0.0);

}  // namespace tenryu::hydro::conservation_audit
