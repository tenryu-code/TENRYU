#pragma once

#include <string>

namespace tenryu::drivers {

// G2: local-limit identity (mode ladder + tanh ramp, planar).
bool run_snb_local_limit_1d(const std::string& label);

// G3: Epperlein-Short dispersion, Tier A (discrete-analytic) + Tier B (VFP band).
bool run_snb_dispersion_1d(const std::string& label);

// G4: machine-precision conservation ledger with SNB active.
bool run_snb_conservation_1d(const std::string& label, const std::string& geometry);

bool run_snb_max_principle_1d(const std::string& label);

}  // namespace tenryu::drivers
