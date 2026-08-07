#pragma once

#include "hydro/ale_rezone.cuh"

namespace tenryu::hydro::ale {

inline constexpr double kQualityGateTol = 1.0e-12;

inline bool ale_escalation_quality_gate_accepts(
    const RezoneResult& interior,
    const double pre_request_min_quality) {
  return interior.triggered && interior.converged && !interior.mesh_tangle &&
         (interior.min_quality + kQualityGateTol >= pre_request_min_quality);
}

}  // namespace tenryu::hydro::ale
