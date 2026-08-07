#pragma once

#include <cstdint>

namespace tenryu::core {

// Device-side soft error flags/counters.
// Most entries are boolean latches set via atomicExch; `infinite_loop` is a per-step counter.
struct DeviceErrorFlags {
  int32_t nan_particle = 0;
  int32_t invalid_cell = 0;
  int32_t invalid_boundary = 0;
  int32_t pool_overflow = 0;
  int32_t opacity_out_of_range = 0;
  int32_t infinite_loop = 0;
  int32_t ddmc_sigma_tot_zero = 0;
  int32_t roulette_kill = 0;
};

}  // namespace tenryu::core
