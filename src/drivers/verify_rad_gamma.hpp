#pragma once

// W-K gamma_r=4/3 coupling verify gates (bodies). cmd_verify registration
// ships as diff proposal tmp/diff_proposals/wk-gammar/02; the Catch2 test
// tests/verification/test_rad_gamma_gates.cpp invokes these directly.

namespace tenryu::drivers {

bool run_rad_gamma_adiabat_verify();
bool run_rad_gamma_ceff_verify();

}  // namespace tenryu::drivers
