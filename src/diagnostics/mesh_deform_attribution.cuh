#pragma once

#include "diagnostics/mesh_deform_attribution.hpp"

namespace tenryu::diagnostics::mesh_attribution::device {

__host__ __device__ inline double cross2(const double ar,
                                         const double az,
                                         const double br,
                                         const double bz) {
  return ar * bz - az * br;
}

}  // namespace tenryu::diagnostics::mesh_attribution::device
