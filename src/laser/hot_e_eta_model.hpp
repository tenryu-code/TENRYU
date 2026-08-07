#pragma once

#include <cstddef>

namespace tenryu::laser::hot_e_eta {

// Mechanism selector for the threshold family.
enum class Mechanism { kTpd, kSrs };

// Per-channel model parameters (all SI-free: microns, keV, 1e14 W/cm2, seconds
// where stated). Defaults follow the design doc §6.3/§11/§12.
struct ChannelParams {
  Mechanism mechanism = Mechanism::kTpd;
  double threshold_multiplier = 1.0;   // C_k (TPD 1.0, SRS 8.0)
  double eta_inf = 0.01;               // saturation ceiling (TPD 0.01, SRS 0.08)
  double eta_hard_cap = 0.03;          // hard cap (TPD 0.03, SRS 0.08)
  double shape_coefficient = 1.0;      // a_k
  bool tau_vu2012 = true;              // TPD: tau = clip(1/(0.05 g + 0.06), tau_min, tau_max) [ps]
  double relaxation_tau_s = 6.0e-12;   // fixed tau (used when !tau_vu2012; SRS default)
  double relaxation_tau_min_s = 3.0e-12;
  double relaxation_tau_max_s = 1.0e-11;
};

// Shared model parameters (design doc §12).
struct ModelParams {
  double ln_filter_tau_s = 5.0e-12;    // EMA time constant for kappa = 1/L_n
  double eta_total_cap = 0.08;         // global cap on the channel sum
  double L_n_min_um = 10.0;            // clamp + out-of-range counting
  double L_n_max_um = 1000.0;
  double Te_min_keV = 0.5;
  double Te_max_keV = 10.0;
  double I14_min = 1.0e-1;             // 1e13 W/cm2 in units of 1e14
  double I14_max = 2.0e2;              // 2e16 W/cm2
};

// Per-channel dynamic state (persists across steps; init = all zero).
struct ChannelState {
  double eta = 0.0;         // relaxed efficiency in [0, eta_hard_cap]
  double kappa_bar = 0.0;   // filtered |d ln n_e / dr| [1/um]; 0 = no sample yet
};

// Instantaneous per-channel inputs for one step.
struct ChannelInputs {
  double I14 = 0.0;         // overlap intensity at the capture surface [1e14 W/cm2]
  double Te_keV = 0.0;      // electron temperature at the evaluation surface
  double kappa_um = 0.0;    // instantaneous |d ln n_e/dr| [1/um] at the eval surface
  bool valid = false;       // false => relax toward eta_eq = 0 (target_zero_relax)
};

struct ChannelDiagnostics {
  double g = 0.0;           // normalized drive
  double eta_eq = 0.0;      // equilibrium target before relaxation
  double tau_s = 0.0;       // relaxation time actually used [s]
  double L_n_eff_um = 0.0;  // filtered, clamped scale length
  int clamped = 0;          // bitmask: 1=Ln, 2=Te, 4=I14 clamped this step
};

// Weighted least-squares slope of ln(n_e) vs r over a window.
// r_um[i], ne[i] (any consistent density unit; only the log-slope matters),
// n points (>= 2), center_um = capture-surface radius. Weights
// w_i = exp(-(d_i / w0)^2) with d_i = |r_i - center| and
// w0 = 2 * mean(|r_{i+1} - r_i|) over the window (w0 > 0 guaranteed by caller
// passing distinct radii; if the weighted variance of r is zero, return 0).
// Returns |slope| in 1/um; non-finite or non-positive densities make the fit
// skip that point; fewer than 2 usable points => 0.
double fit_kappa_abs_um(const double* r_um, const double* ne, std::size_t n,
                        double center_um);

// Equilibrium efficiency for one channel (design doc §6.1/§6.2).
// lambda_um: laser wavelength in microns. Uses the 351-nm NIF-compatible SRS
// absolute threshold I_abs,14 = 2377 / L^{4/3} when |lambda_um - 0.351| < 5e-3,
// else the general I_abs,14 = 995 / (L^2 lambda)^{2/3}.
// TPD: xi = I14 * L_um * lambda_um / (82 * Te_keV); g = xi / C.
// SRS: g = I14 / (C * I_abs14).
// g <= 1 => 0; else min(eta_hard_cap, eta_inf * (1 - exp(-a * sqrt(g - 1)))).
// Outputs g via diag.
double eta_equilibrium(const ChannelParams& p, double I14, double Te_keV,
                       double L_n_um, double lambda_um, ChannelDiagnostics& diag);

// Relaxation time [s]: TPD vu2012 form tau_ps = 1 / (0.05 g + 0.06) clipped to
// [tau_min, tau_max] (params carry seconds); fixed tau otherwise.
double relaxation_tau_s(const ChannelParams& p, double g);

// One full per-channel update (design doc §11.1/§12.2):
//   alpha_L = -expm1(-dt / ln_filter_tau);  kappa_bar += alpha_L (kappa - kappa_bar)
//     (only when inputs.valid and kappa_um > 0; a fresh state (kappa_bar == 0)
//      snaps directly to kappa_um instead of filtering)
//   L_n_eff = 1 / max(kappa_bar, 1 / L_n_max_um), then clamped to
//     [L_n_min_um, L_n_max_um] with diag.clamped |= 1 when it clips
//   Te/I14 clamped to their ranges with diag.clamped |= 2 / 4
//   eta_eq = inputs.valid ? eta_equilibrium(clamped values) : 0
//   tau = relaxation_tau_s(p, diag.g)   (vu2012 uses the g just computed;
//         when !inputs.valid use the fixed relaxation_tau_s)
//   alpha = -expm1(-dt_s / tau); state.eta += alpha * (eta_eq - state.eta)
// Returns the updated eta (also stored in state).
double update_channel(const ChannelParams& p, const ModelParams& mp,
                      ChannelState& state, const ChannelInputs& inputs,
                      double dt_s, double lambda_um, ChannelDiagnostics& diag);

// Global cap (design doc §9): if the sum of etas exceeds mp.eta_total_cap,
// scale every channel eta by cap/sum (states updated in place). Returns the
// applied scale (1.0 when no scaling).
double apply_total_cap(const ModelParams& mp, ChannelState* states,
                       std::size_t n_channels);

}  // namespace tenryu::laser::hot_e_eta
