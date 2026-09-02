#pragma once

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

// Rebuilds state.cavity_boundary from the current void-cell set. Read-only with
// respect to all physics state; logs only on topology-epoch change. 2D RZ only
// (no-op when the mesh is not a structured 2D grid).
void rebuild_cavity_boundary_graph(core::State& state,
                                   const core::Config& cfg);

}  // namespace tenryu::hydro
