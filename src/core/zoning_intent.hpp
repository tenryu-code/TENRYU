#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace tenryu::core {

// Measure that the solver equidistributes / constrains. Geometry is implied by
// the measure itself; there is no separate geometry code.
enum class ZoningMeasure : std::uint8_t {
  kWidth = 0,                // d(mu) = dr                      [cm]
  kArealMass = 1,            // d(mu) = rho0(r) dr              [g/cm^2]
  kCylindricalLineMass = 2,  // d(mu) = rho0(r) * 2*pi*r dr     [g/cm]
  kSphericalCellMass = 3,    // d(mu) = rho0(r) * 4*pi*r^2 dr   [g]
};

struct ZoningIntentPin {
  double r = 0.0;                  // node pinned exactly at this radius [cm]
  bool ratio_jump_allowed = false; // true: the hard ratio cap is not applied
                                   // across the two cells adjacent to this pin
};

// Piecewise log-linear preferred cell-size weight profile w(r) > 0 (relative
// shape only; absolute scale cancels in the normalized equidistribution).
struct ZoningProfilePoint {
  double r = 0.0;  // [cm]
  double w = 0.0;  // > 0
};

// Compact-support multiplicative refinement/coarsening kernel, additive in
// ln w: w(r) gains a term log_amplitude * K((r - r_center)/half_width) with
// K(u) = 0.5*(1 + cos(pi*u)) for |u| < 1 and 0 outside (C^1 at the edges).
// Negative log_amplitude refines (smaller cells near the center).
struct ZoningAnchor {
  double r = 0.0;             // center [cm]
  double half_width = 0.0;    // support half-width [cm], > 0
  double log_amplitude = 0.0; // additive amplitude in ln w
};

// Per-cell measure bounds restricted to a band of the GLOBAL cumulative
// measure, addressed by fractions of the domain-total measure (a Lagrangian-
// invariant selector: for mass measures the band tracks the same material
// regardless of where the radii lie). Bounds compose with the global
// cell_measure_min/max by intersection (max of lower bounds, min of upper
// bounds) on every cell the band covers.
struct ZoningBand {
  double measure_frac_begin = 0.0;  // in [0, 1), < measure_frac_end
  double measure_frac_end = 0.0;    // in (0, 1]
  double cell_measure_min = 0.0;    // lower bound inside the band; 0 disables
  double cell_measure_max = 0.0;    // upper bound inside the band; 0 disables
};

struct ZoningIntentConfig {
  int n_cells = 0;                          // total cell budget (required, >= 1)
  ZoningMeasure measure = ZoningMeasure::kWidth;
  std::vector<ZoningIntentPin> pins;        // strictly increasing, strictly inside (r_min, r_max)
  std::vector<ZoningProfilePoint> profile;  // optional; strictly increasing r; empty => w == 1
  std::vector<ZoningAnchor> anchors;        // optional refinement/coarsening kernels
  std::vector<ZoningBand> bands;            // optional banded measure bounds
  std::vector<double> extra_events;         // known rho0 breakpoints (quadrature panel edges)
  double dr_min = 0.0;                      // hard width floor [cm]; 0 disables
  double cell_measure_min = 0.0;            // per-cell measure lower bound; 0 disables
  double cell_measure_max = 0.0;            // per-cell measure upper bound; 0 disables
  double preferred_ratio = 1.3;             // soft adjacent-cell-measure ratio (reported only)
  double ratio_hard_max = 2.0;              // hard adjacent-cell-measure ratio cap, in (1.0, 2.0]
  int min_cells_per_segment = 1;            // >= 1, applied to every pin-delimited segment
};

enum class ZoningStatus : std::uint8_t {
  kOk = 0,
  kInvalidInput = 1,      // the intent itself is malformed
  kInfeasible = 2,        // intent is well-formed but provably unsatisfiable
  kNumericalFailure = 3,  // solver/verifier failed; do NOT retry with changed physics intent
};

struct ZoningDiagnostics {
  ZoningStatus status = ZoningStatus::kOk;
  std::string code;     // stable machine-readable code, e.g. "MESH_PIN_OUT_OF_DOMAIN"
  std::string message;  // human-readable detail including the offending numbers
  // Achieved statistics, filled only on kOk (from the independent verification grid):
  double ratio_max_achieved = 1.0;   // max adjacent cell-measure ratio (>= 1 form)
  double ratio_mean_achieved = 1.0;  // mean of the >= 1 form over interior edges
  int n_ratio_soft_exceed = 0;       // edges above preferred_ratio (hard cap still met)
  double width_min_achieved = 0.0;   // [cm]
  double cell_measure_min_achieved = 0.0;   // from the verification grid
  double cell_measure_max_achieved = 0.0;   // from the verification grid
  // Per band (same order as config.bands), from the verification grid:
  // the max and min cell measure among cells whose ACTUAL measure span
  // overlaps the band (0.0 when the band covers no cell).
  std::vector<double> band_cell_measure_max_achieved;
  std::vector<double> band_cell_measure_min_achieved;
  double quadrature_rel_residual = 0.0;  // verification-grid measure closure residual
  std::vector<int> cells_per_segment;    // one entry per pin-delimited segment
  std::vector<std::string> warnings;
};

struct ZoningResult {
  bool ok = false;
  std::vector<double> nodes;  // size n_cells + 1 when ok
  ZoningDiagnostics diag;
};

// rho0(r) [g/cc] is required for the three mass measures (may be nullptr for
// kWidth). It must be finite and >= 0, and smooth within each quadrature panel;
// discontinuities must be declared via pins or extra_events.
ZoningResult compute_zoning_intent_nodes(
    double r_min, double r_max,
    const ZoningIntentConfig& cfg,
    const std::function<double(double)>& rho0);

}  // namespace tenryu::core
