#pragma once

#include <string>

namespace tenryu::drivers {

// G2: local-limit identity (mode ladder + tanh ramp, 2D RZ).
bool run_snb_local_limit_2d(const std::string& label);

// G3: Epperlein-Short dispersion, Tier A (discrete-analytic) + Tier B (VFP band).
bool run_snb_dispersion_2d(const std::string& label);

// G4: machine-precision conservation ledger with SNB active.
bool run_snb_conservation_2d(const std::string& label);

}  // namespace tenryu::drivers
