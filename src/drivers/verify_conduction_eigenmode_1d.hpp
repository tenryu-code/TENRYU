#pragma once

#include <string>

namespace tenryu::drivers {

// W-G2: 1D electron-conduction eigenmode-decay verification (analytic gates).
//
// Fixed-kappa (Numerics.conduction.test_kappa) linear conduction on the first
// zero-flux eigenmode of the 1D operator in the given coordinate geometry:
//   T(r,t) = T0 + A * phi(k r) * exp(-chi k^2 t),  chi = kappa / (rho cv_e)
//   planar      phi = cos(x),      k R = pi
//   cylindrical phi = J0(x),       k R = j_{1,1} (first positive zero of J1)
//   spherical   phi = sin(x)/x,    k R = alpha   (first positive root of tan x = x)
// The mode satisfies the operator's built-in zero-flux boundaries exactly, so
// the discrete total electron energy is conserved to roundoff and the L2 error
// against the exact solution converges at second order in dx (dt = 0.2 dx^2/chi
// scales the O(dt) time error like the O(dx^2) spatial error).
//
// Configs/states are built programmatically (no namelist), so the cylindrical
// case runs independently of the namelist builder validation.

// Multi-resolution (nr = 50,100,200,400) STS order study:
// PASS = loglog L2 slope in [1.8, 2.2] AND per-resolution energy identity
// |E_final - E_initial|/E_initial <= 1e-14 AND eigenvalue self-check.
bool run_conduction_eigenmode_1d_order_study(const std::string& label,
                                             const std::string& geometry);

// Single-resolution (nr = 100) solver pair: runs solver="sts" and
// solver="implicit" on the same problem; PASS = both energy identities AND
// L2(implicit) <= 2 * L2(sts) with L2(sts) > 0.
bool run_conduction_eigenmode_1d_solver_pair(const std::string& label,
                                             const std::string& geometry);

// Single-resolution (nr = 100) per-material pair: runs the single-material STS
// path and the per-material STS path (two identical materials, volfrac
// 0.5/0.5, per_material_conservation_enabled); PASS = both energy identities
// AND 0.5 <= L2(per_material)/L2(single) <= 1.5.
bool run_conduction_eigenmode_1d_per_material_pair(const std::string& label,
                                                   const std::string& geometry);

}  // namespace tenryu::drivers
