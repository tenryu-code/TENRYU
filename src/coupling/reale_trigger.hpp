#pragma once

namespace tenryu::core {
struct State;
}

namespace tenryu::coupling::reale_trigger {

// True once a post-commit reference snapshot exists (and matches the current
// node count).
bool reference_valid(const core::State& state);

// Snapshot the CURRENT node coordinates (device-to-device) as the trigger
// reference. Called at every rezone COMMIT, after the new mesh is installed.
void snapshot_reference(const core::State& state);

// Max over cells of (max corner displacement since the reference) /
// (sqrt(planar cell area) / 2). Fires the rezone at >= 1. Returns -1.0 if the
// reference is absent or the node count changed. GPU-resident: one kernel +
// one 8-byte D2H readback.
double max_displacement_ratio(const core::State& state);

}  // namespace tenryu::coupling::reale_trigger
