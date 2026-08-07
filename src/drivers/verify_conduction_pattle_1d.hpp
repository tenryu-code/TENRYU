#pragma once

#include <string>

namespace tenryu::drivers {

// W-G2 nlheat: Pattle (1959) self-similar nonlinear thermal wave gates.
//
// kappa = kappa0 * Te^n (n = 5/2, Spitzer-like) via the default-inert env hook
// TENRYU_CONDUCTION_TEST_KAPPA_POWER on the fixed test_kappa path. Reference
// (Pattle QJMAM 12, 407 (1959), eqs. (4a),(6); transcription verified against
// the PDE symbolically and re-verified in-gate by finite differences):
//   T(r,t) = C0 (t/t0)^{-s/(sn+2)} (1 - r^2/r1^2)^{1/n},  r <= r1,  else 0
//   r1(t)  = r0 (t/t0)^{1/(sn+2)}
//   t0     = r0^2 n rho c_v / (2 kappa0 C0^n (sn+2))
// s = 1 (planar), 2 (cylindrical), 3 (spherical). The compact-support front
// makes the reflect outer wall exact until arrival (no far-field tail).
//
// Gate per geometry: in-gate PDE-residual transcription self-check, then
// nr = {200, 400}: evolve t0 -> 6 t0 with STS, measure (a) front-position
// power-law slope over 5 log-spaced checkpoints vs 1/(sn+2), (b) profile L2
// vs the analytic reference at t1, (c) discrete conservation of Sum(Te V),
// (d) clamp count == 0; then the pairwise convergence order log2(L2_200/L2_400).
bool run_conduction_pattle_1d(const std::string& label,
                              const std::string& geometry);

// n -> 0 linear-connection check (transcription protocol): the same smooth
// initial state stepped with the hook OFF (historic constant test_kappa) and
// with the hook ON at power 0 must produce BITWISE-identical Te trajectories
// (the deff hook kernel mirrors the historic branch with pow(x,0) == 1, and
// the secant stage kernel takes an exact constant-kappa shortcut at p == 0).
bool run_conduction_pattle_kappa_power_zero_connection(const std::string& label);

}  // namespace tenryu::drivers
