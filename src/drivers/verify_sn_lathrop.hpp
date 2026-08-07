#pragma once

namespace tenryu::drivers {

// W-D gate: 1D spherical S_N vs Lathrop (NSE 134, 2000) two-region purely
// absorbing sphere analytic solutions (paper Eqs. 3 / 5a-d; Eq. 5a term-2
// exponent corrected per the NOTE-1 resolution in
// docs/design/wd_sn_spherical_lathrop_design.md). Part 0 re-runs the
// transcription-verification protocol (continuity + reduction) at startup;
// Part A compares the swept per-ordinate flux and scalar flux against the
// analytic solution on the reference case S1=4, S2=1, sigma1=1, sigma2=0.5,
// a=1, b=4 at S16/nr=400 with zone-resolved error metrics and an S8->S16
// refinement check; Part B runs the single-region Eq. (3) case.
bool run_sn_1d_spherical_lathrop_two_region_verify();

}  // namespace tenryu::drivers
