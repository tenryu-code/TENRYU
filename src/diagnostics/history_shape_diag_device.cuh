#pragma once

#include "core/config.hpp"
#include "core/state.hpp"
#include "diagnostics/diagnostics.hpp"

namespace tenryu::diagnostics {

// Device-resident computation of the areal-density and sphericity history
// diagnostics. Returns true and fills the requested outputs when the device
// path ran (2D structured mesh, fields present); returns false (outputs
// untouched) when the caller must use the host implementations.
// Default-on: the device path runs unless TENRYU_HISTORY_DIAG_NO_DEVICE=1
// forces the host implementations.
bool compute_shape_history_device(const core::State& state,
                                  const core::Config& cfg,
                                  ArealDensityDiagnostics* areal,
                                  SphericityDiagnostics* sphericity);

}  // namespace tenryu::diagnostics
