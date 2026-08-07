#pragma once

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::core {

void populate_per_material_conserved_from_cell_mean(State& state,
                                                    const Config& cfg);

}  // namespace tenryu::core
