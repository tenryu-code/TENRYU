#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro::band_ale {

// Compact quintic smoothstep: 0 for y<=0, 1 for y>=1, C2 at both ends.
inline double band_smoothstep5(const double y) {
  if (y <= 0.0) { return 0.0; }
  if (y >= 1.0) { return 1.0; }
  return ((6.0 * y - 15.0) * y + 10.0) * y * y * y;
}

// Rising activation: 0 at x<=arm, 1 at x>=full (arm < full).
inline double band_activation_up(const double x, const double arm,
                                 const double full) {
  return band_smoothstep5((x - arm) / (full - arm));
}

inline double pole_theta_routine_activation(const double E,
                                            const double noise_floor,
                                            const double noise_ceiling) {
  const double a_low = band_activation_up(E / noise_floor, 2.0, 4.0);
  const double a_high = band_activation_up(E / noise_ceiling, 1.0, 2.0);
  return a_low * (1.0 - a_high);
}

// Falling activation: 0 at x>=arm, 1 at x<=full (full < arm).
inline double band_activation_down(const double x, const double arm,
                                   const double full) {
  return band_smoothstep5((arm - x) / (arm - full));
}

struct ActiveBand {
  std::size_t index = 0;
  std::string name;
  double min_ratio = 0.0;
  int argmin_cell = -1;
};

void run_start(core::State& state, const core::Config& cfg);

std::vector<ActiveBand> begin_step(
    core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction,
    int rank);

void build_target(const core::State& state,
                  const core::Config& cfg,
                  std::size_t band_index,
                  std::vector<double>& target_r,
                  std::vector<double>& target_z);

bool is_closure_catchment(std::size_t band_index);

// True when the named band's last prepared target was prescaled down by
// the per-transaction swept-fraction cap and another capped bite in the
// same step would make further progress (closure catchment only).
bool wants_reapply(std::size_t band_index);

// Pole-theta maintenance: fold the current step's delta-theta map (if one
// was armed by begin_step and not yet consumed) into a prescribed target.
// No-op when the controller did not fire this step.
void compose_pole_theta_target(const core::State& state,
                               const core::Config& cfg,
                               std::vector<double>& target_r,
                               std::vector<double>& target_z);

// Mark the current step's pole-theta map as consumed (call after a
// prescribed-target transaction that included the composed map was
// ACCEPTED, so later bites in the same step do not re-apply it).
void note_pole_theta_consumed();

bool is_pole_theta(std::size_t band_index);

bool pole_theta_composed_last_build();

void record_inadmissible(std::size_t band_index);

void record_partial_sigma(std::size_t band_index, double sigma);

// Redistribute interval [0, L] into birth_spacings.size() spacings while
// preserving the birth proportions subject to floor and adjacent-ratio
// constraints. This is a host-only test seam.
bool closure_catchment_redistribute(
    const std::vector<double>& birth_spacings,
    double L,
    double floor_cm,
    double ratio_max,
    std::vector<double>* out_spacings);

// Return the applied radial respace displacement after the local-gap cap.
double band_respace_capped_move(double radius,
                                double target_radius,
                                double chi,
                                double local_gap,
                                double cap_frac);

// Return the interior cumulative fractions induced by interval widths.
std::vector<double> closure_catchment_xi(
    const std::vector<double>& Hstar);

// --- Pole-theta maintenance pure host seams (State-free, unit-tested) ---

// Smooth two-decade logarithmic significance channel: exactly 0 at or
// below y0*reference, exactly 1 at or above y1*reference, smoothstep5 in
// log space.
double band_significance_channel(double value,
                                 double reference,
                                 double y0,
                                 double y1);

// Multi-channel smooth OR: a cell is negligible only when every channel
// is negligible.
double band_significance_weight(double w_rho,
                                double w_m,
                                double w_z,
                                double w_pv);

struct PoleThetaCellMetrics {
  double h_theta_min = 0.0;   // min over the 4 corners of the angular altitude
  double min_altitude_abs = 0.0;  // unnormalized minimum subzonal altitude
  double kappa_max = 0.0;     // max over the 4 corners of the target-relative
                              // Jacobian condition number
};

// Quad node order is the multiblock CSR order (inner_j, outer_j, outer_j+1,
// inner_j+1). h_theta_star > 0 is the qualified target angular height.
PoleThetaCellMetrics pole_theta_cell_metrics(const double r[4],
                                             const double z[4],
                                             double h_theta_star);

// Unsaturated logarithmic hard-threshold excess of one cell.
double pole_theta_cell_excess(double H,
                              double kappa,
                              double h_hard,
                              double kappa_hard);

// Continuous hard-lane audit gates (consult 7): quality benefit arms over
// 0.02..0.08, mesh-dt benefit over 0.05..0.20, and capability fades over
// debt 1.5..3.0.
inline double pole_theta_hard_benefit(const double R_Q,
                                      const double R_dt) {
  const double b_Q = band_activation_up(R_Q, 0.02, 0.08);
  const double b_dt = band_activation_up(R_dt, 0.05, 0.20);
  return 1.0 - (1.0 - b_Q) * (1.0 - b_dt);
}

inline double pole_theta_capability(const double debt) {
  return 1.0 - band_activation_up(debt, 1.5, 3.0);
}

inline double pole_theta_damage_gate(const double D_rho) {
  return 1.0 - band_activation_up(D_rho, 5.0e-4, 2.0e-3);
}

// Synthetic concentric-row seam for the same frozen-weight L8 trial audit
// used by the runtime. weights has one entry per angular cell.
std::pair<double, double> pole_theta_row_trial_excess(
    const std::vector<double>& theta_row,
    const std::vector<double>& applied,
    const std::vector<double>& weights,
    double h_hard,
    double kappa_hard,
    double h_theta_star);

// Grow a pole-anchored sector over per-column worst-case metrics.
// H[j], kappa[j] are the per-column minima/maxima over the scanned rows
// (H = h_theta/h_theta_star). from_north=true anchors at column 0, false
// anchors at column n_columns-1. Marking rule: H < mark_h || kappa > mark_k.
// Growth stops after two consecutive columns with H > stop_h && kappa <
// stop_k. halo_columns extra columns are appended on the equator side.
// Returns {col_begin, col_end} inclusive, clamped to [0, n_columns-1];
// returns {-1, -1} when the pole-adjacent column itself is unmarked.
struct PoleThetaSector {
  int col_begin = -1;
  int col_end = -1;
};
PoleThetaSector pole_theta_grow_sector(const std::vector<double>& H,
                                       const std::vector<double>& kappa,
                                       int n_columns,
                                       bool from_north,
                                       double mark_h,
                                       double mark_k,
                                       double stop_h,
                                       double stop_k,
                                       int halo_columns);

// Grow a pole-anchored sector over a weighted excess half-field. Columns
// are ordered pole-to-equator. The violating run is q > q_stop.
PoleThetaSector pole_theta_grow_sector_q(
    const std::vector<double>& q_half,
    int half_size,
    double q_stop,
    int halo_columns);

// Build the per-node-row delta-theta map. theta[j] are the current node
// angles of one node row, strictly monotonic over [j0, j1] (j0 < j1, both
// anchors held fixed). dtheta_star is the qualified pitch. taper_w[j] in
// [0, 1] scales the relaxation per node. Returns dtheta (same size as
// theta, zero outside (j0, j1)); after application theta + dtheta is
// strictly monotonic on [j0, j1] with adjacent gaps >= 0.02 * dtheta_star
// (deterministic forward projection, pole-side anchor first).
std::vector<double> pole_theta_delta_map(const std::vector<double>& theta,
                                         int j0,
                                         int j1,
                                         double alpha,
                                         double deadband_frac,
                                         double move_limit_frac,
                                         double dtheta_star,
                                         const std::vector<double>& taper_w);

// Couple north/south delta-theta maps under the global-theta mirror. The
// south entry paired with north spoke s is ntheta-s. The mirror-consistent
// component is retained exactly below the activation dead zone, while the
// raw per-pole maps are retained exactly at full activation. Returns the
// per-row parity-breaking activation.
double pole_theta_parity_gate(
    const std::vector<double>& d_north,
    const std::vector<double>& d_south,
    int ntheta,
    double deadband_frac,
    double dtheta_star,
    std::vector<double>* out_north_applied,
    std::vector<double>* out_south_applied);

// Fit Legendre coefficients a_l (l = 0..order) of samples y(theta) with
// solid-angle weights w_j = |cos(theta_{j+1/2}) - cos(theta_{j-1/2})|
// (half-points from neighbor midpoints, one-sided at the ends). theta
// strictly increasing in [0, pi]. Least squares via the normal equations
// with partial-pivot Gaussian elimination (deterministic). Returns false
// on a singular system.
bool pole_theta_legendre_fit(const std::vector<double>& theta,
                             const std::vector<double>& y,
                             int order,
                             std::vector<double>* coeffs);

// Evaluate a Legendre series at theta.
double pole_theta_legendre_eval(const std::vector<double>& coeffs,
                                double theta);

// Fit coefficients b_l (l = 1..lc_eff) of samples on the pinned basis
// P_l(cos(theta)) minus the line through that mode's endpoint values,
// using node weights max(sin(theta), 1e-12). lc_eff is min(lc, ntheta/4);
// element zero of the returned vector is always zero. Least squares uses
// fixed-order Cholesky with a deterministic numerical-degeneracy guard.
std::vector<double> pole_theta_pinned_band_fit(
    const std::vector<double>& theta,
    int lc,
    const std::vector<double>& correction);

// Return the solid-angle L8 high-pass pitch defect of one strictly
// increasing pole-theta node row. When requested, return the unscaled
// routine correction after the continuous displacement cap and pinned-basis
// retained-band projection. Its endpoint corrections are exactly zero.
double pole_theta_row_defect(const std::vector<double>& theta,
                             int phys_lp,
                             int phys_lc,
                             std::vector<double>* dtheta_correction);

// Reconstruct the row curve: monotone piecewise-cubic interpolation of
// s(theta) through the nodes (Fritsch-Carlson limited tangents, exact at
// the nodes, no overshoot), then add the protected-mode correction
//   sum_{l in protected} (a_l[data] - a_l[interp]) P_l(cos theta)
// where both coefficient sets come from pole_theta_legendre_fit at
// fit_order. protected_modes entries beyond fit_order are ignored.
// The result evaluates s_rec(theta) for theta in [theta.front(),
// theta.back()].
struct PoleThetaRowCurve {
  std::vector<double> theta;      // node angles (copy)
  std::vector<double> s;          // node radii (copy)
  std::vector<double> tangents;   // limited d s/d theta at nodes
  std::vector<double> mode_correction;  // Legendre coeffs of the additive
                                        // correction (index l), possibly
                                        // all zero
};
bool pole_theta_build_row_curve(const std::vector<double>& theta,
                                const std::vector<double>& s,
                                int fit_order,
                                const std::vector<int>& protected_modes,
                                PoleThetaRowCurve* out);
double pole_theta_eval_row_curve(const PoleThetaRowCurve& curve,
                                 double theta);

// Return whether a reconstructed row radius lies within the current row
// envelope allowance.
bool pole_theta_s_target_admissible(double s_target,
                                    double s_min_row,
                                    double s_max_row);

struct PerColumnStateForTest {
  std::vector<int> i0, i1;  // per column; -1 when ill-formed
  std::vector<std::uint8_t> well_formed;
  std::vector<std::uint8_t> reason;  // per column; 0 when well-formed
  std::vector<std::uint8_t> spoke_active;  // per spoke (n_columns+1)
  std::vector<std::uint8_t> classification;
  std::vector<int> tube_i0, tube_i1;  // per column; -1 when no tube rows
  int patch_count = 0;
  std::uint8_t escalation_reason = 0;
  bool hold = false;
  bool engaged = false;
  double eta_p = 0.0;
  double coverage = 0.0;
};

// Test seam: host cell indices of the named band's current
// membership (nullptr if no such band). Read-only; not for
// production logic.
const std::vector<int>* band_ale_band_cells_for_test(
    const char* band_name);
bool band_ale_band_engaged_for_test(const char* band_name);
const PerColumnStateForTest* band_ale_per_column_state_for_test();

void run_end(const core::Config& cfg, int rank);

}  // namespace tenryu::hydro::band_ale
