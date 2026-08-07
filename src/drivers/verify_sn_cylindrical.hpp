#pragma once

namespace tenryu::drivers {

// W-G3 gate: 1D cylindrical S_N marshak equilibration.
// Phase A — uniform blackbody fixed point on the full cylinder (r0 = 0,
// axis cell included), the decisive discrete-operator probe (BUG-8 class);
// Phase B — cold-start equilibration to the Tr plateau (parity with the
// spherical/planar gates). Design:
// docs/design/wg3_sn_cylindrical_quadrature_design.md.
bool run_sn_1d_cylindrical_marshak_equilibration_verify();

}  // namespace tenryu::drivers
