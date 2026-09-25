#pragma once

#include "core/config.hpp"

namespace tenryu::core {

// True when the radiation operator runs its thermal subcycle this step
// (Numerics.radiation_thermal_subcycle with radiation enabled on the
// single-stage path). The subcycle then carries the electron-ion exchange and
// electron conduction ([radiation -> exchange -> conduction] per substep), so
// the hydro energy update applies no exchange and the Strang conduction slot is
// skipped: each operator is applied exactly once per step (NUMERICS §2.1).
inline bool thermal_subcycle_active(const Config& cfg) {
  return cfg.numerics.radiation_thermal_subcycle && cfg.radiation.enabled &&
         !cfg.radiation.imc.two_stage;
}

}  // namespace tenryu::core
