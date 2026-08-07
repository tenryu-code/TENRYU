#pragma once

#include <vector>

namespace tenryu::radiation {

// W-G3: product angular quadrature for 1D cylindrical S_N.
// Design: docs/design/wg3_sn_cylindrical_quadrature_design.md; authority:
// Morel & Montry 1984 (TTSP 13(5) 615) appendix Eqs. A1-A4.
//
// Flattened layout: level-major, mu ascending within each level. Level l
// occupies flat ordinates [l*M, (l+1)*M) and alpha edges [l*M, l*M + M];
// the shared edge at every level boundary is exactly 0.0 (chain delimiter,
// same convention the spherical sweep kernels key on).
struct SnCylQuadrature1D {
  int n_levels = 0;                // L: xi > 0 polar levels (GL(2L) positive nodes)
  int n_azim = 0;                  // M = 2L azimuthal ordinates per level
  std::vector<double> mu;          // L*M: radial cosine sin(theta_l)*cos(omega_m)
  std::vector<double> weight;      // L*M: v_l * (2/M); total sum == 2
  std::vector<double> alpha_half;  // L*M+1: per-level Carlson recursion, level edges 0
  std::vector<double> tau_wd;      // L*M: M&M A1-A4 weighted-diamond factor (M-periodic)
  std::vector<double> sd_mu;       // L: sin(theta_l) = starting-direction |mu| per level
};

// Level count L for a flattened ordinate count n == 2*L*L (L >= 2); 0 if n is
// not of that form.
int sn_cyl_levels_for(int n_angles);

// Build the product quadrature. TENRYU_ASSERTs n_angles == 2*L*L with L >= 2.
SnCylQuadrature1D build_sn_cyl_quadrature_1d(int n_angles);

}  // namespace tenryu::radiation
