#pragma once

namespace tenryu::core {
struct State;
struct Config;
}

namespace tenryu::hydro {

// Test seam: orientation-invariant metric factor (1/|d|; 0 when
// degenerate or non-finite).
double lohner_metric_factor(double signed_denominator);

double lohner_ratio_1d(double um2, double u0, double up2, double eps);

// 3x3 patch second-pass evaluation on a uniform Cartesian metric (fx=fy=1):
// takes the 5x5 field values and returns E of the center cell per the
// FLASH4.7 four-term form.
double lohner_e_2d_flat(const double u[5][5], double eps);

// FLASH-style modified Lohner error estimator (design doc §18 C1).
// Host-side, read-only: computes per-cell E on the per-block structured
// (n_i radial layers x n_j angular columns) grids, updates
// state.refine_error (host vector), and logs aggregates. No-op unless
// cfg.numerics.diagnostics.refinement_estimator.enabled and
// step % every == 0.
void refinement_estimator_step(core::State& state, const core::Config& cfg);

}  // namespace tenryu::hydro
