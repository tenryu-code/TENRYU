#pragma once

// Braginskii viscosity verify gates (bodies), registered in cmd_verify.cpp;
// the Catch2 battery in tests/verification/test_braginskii_gates.cpp invokes
// these directly so the gates also run under ctest. Electron-channel and
// regime gates: docs/design/electron_viscosity_1d_20260712.md section 12.

namespace tenryu::drivers {

bool run_braginskii_planar_wave_decay_verify();
bool run_braginskii_cylindrical_wave_decay_verify();
bool run_braginskii_spherical_wave_decay_verify();
bool run_braginskii_hubble_null_verify();
bool run_braginskii_electron_wave_decay_verify();
bool run_braginskii_regime_map_verify();

}  // namespace tenryu::drivers
