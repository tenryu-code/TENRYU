#pragma once

#include <string>

namespace tenryu::coupling {

inline bool is_dt_floor_stall_step(const double dt_committed,
                                   const double dt_min_s,
                                   const double t_remaining,
                                   const std::string& limiter) {
  if (dt_committed > (1.0 + 1.0e-6) * dt_min_s) {
    return false;
  }
  if (t_remaining <= 1.0e3 * dt_min_s) {
    return false;
  }
  if (limiter == "output" || limiter == "t_end") {
    return false;
  }
  return true;
}

}  // namespace tenryu::coupling
