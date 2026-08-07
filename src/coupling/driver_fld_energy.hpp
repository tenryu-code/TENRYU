#pragma once

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::coupling {

double compute_fld_rad_energy_total_device(const core::State& state,
                                           const core::Config& cfg);

// W-R1 slot variants: identical contribution kernel, block sums written
// into d_slot (fld_rad_energy_reduce_blocks(...) doubles), NO device
// sync, NO readback. Returns the block count used (0 on the same
// early-return conditions where the eager function returns 0.0).
int compute_fld_rad_energy_total_device_to_slot(const core::State& state,
                                                const core::Config& cfg,
                                                double* d_slot);
int fld_rad_energy_reduce_blocks(const core::State& state,
                                 const core::Config& cfg);
// Host materialization (ascending long-double over blocks; 0 -> 0.0).
double materialize_fld_rad_energy(const double* h_slot, int blocks);

double compute_fld_rad_energy_total_host(const core::State& state,
                                         const core::Config& cfg);

}  // namespace tenryu::coupling
