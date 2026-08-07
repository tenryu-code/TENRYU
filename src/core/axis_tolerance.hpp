#pragma once

#include <cmath>

namespace tenryu::core {

#if defined(__CUDACC__)
#define TENRYU_AXIS_TOL_HD __host__ __device__
#else
#define TENRYU_AXIS_TOL_HD
#endif

TENRYU_AXIS_TOL_HD inline double axis_tolerance(const double r_max) {
  return 1.0e-10 * fmax(r_max, 1.0);
}

#undef TENRYU_AXIS_TOL_HD

}  // namespace tenryu::core
