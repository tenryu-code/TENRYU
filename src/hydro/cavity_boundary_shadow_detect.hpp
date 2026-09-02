#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

void shadow_detect_cavity_contacts(
    core::State& state,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<std::int32_t>& prev_nodes,
    const std::vector<std::int32_t>& prev_best_ids,
    bool topology_changed);

}  // namespace tenryu::hydro
