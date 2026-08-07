#include "laser/hot_e_eta_model.hpp"

#include <algorithm>
#include <cmath>

namespace tenryu::laser::hot_e_eta {

double fit_kappa_abs_um(const double* r_um, const double* ne, const std::size_t n,
                        const double center_um) {
  if (n < 2) {
    return 0.0;
  }

  double spacing_sum = 0.0;
  for (std::size_t i = 1; i < n; ++i) {
    spacing_sum += std::abs(r_um[i] - r_um[i - 1]);
  }
  const double w0 = 2.0 * spacing_sum / static_cast<double>(n - 1);

  std::size_t usable = 0;
  double weight_sum = 0.0;
  double weighted_r_sum = 0.0;
  double weighted_log_ne_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(ne[i]) || ne[i] <= 0.0) {
      continue;
    }
    const double d_over_w0 = std::abs(r_um[i] - center_um) / w0;
    const double weight = std::exp(-(d_over_w0 * d_over_w0));
    const double log_ne = std::log(ne[i]);
    ++usable;
    weight_sum += weight;
    weighted_r_sum += weight * r_um[i];
    weighted_log_ne_sum += weight * log_ne;
  }
  if (usable < 2 || weight_sum == 0.0) {
    return 0.0;
  }

  const double mean_r = weighted_r_sum / weight_sum;
  const double mean_log_ne = weighted_log_ne_sum / weight_sum;
  double variance_r = 0.0;
  double covariance = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(ne[i]) || ne[i] <= 0.0) {
      continue;
    }
    const double d_over_w0 = std::abs(r_um[i] - center_um) / w0;
    const double weight = std::exp(-(d_over_w0 * d_over_w0));
    const double delta_r = r_um[i] - mean_r;
    variance_r += weight * delta_r * delta_r;
    covariance += weight * delta_r * (std::log(ne[i]) - mean_log_ne);
  }
  if (variance_r == 0.0) {
    return 0.0;
  }
  return std::abs(covariance / variance_r);
}

double eta_equilibrium(const ChannelParams& p, const double I14,
                       const double Te_keV, const double L_n_um,
                       const double lambda_um, ChannelDiagnostics& diag) {
  double g = 0.0;
  if (p.mechanism == Mechanism::kTpd) {
    const double xi = I14 * L_n_um * lambda_um / (82.0 * Te_keV);
    g = xi / p.threshold_multiplier;
  } else {
    const double I_abs14 =
        (std::abs(lambda_um - 0.351) < 5.0e-3)
            ? 2377.0 / std::pow(L_n_um, 4.0 / 3.0)
            : 995.0 / std::pow(L_n_um * L_n_um * lambda_um, 2.0 / 3.0);
    g = I14 / (p.threshold_multiplier * I_abs14);
  }
  diag.g = g;

  if (g <= 1.0) {
    return 0.0;
  }
  const double eta =
      p.eta_inf *
      (1.0 - std::exp(-p.shape_coefficient * std::sqrt(g - 1.0)));
  return std::min(p.eta_hard_cap, eta);
}

double relaxation_tau_s(const ChannelParams& p, const double g) {
  if (p.mechanism == Mechanism::kTpd && p.tau_vu2012) {
    const double tau_s = 1.0e-12 / (0.05 * g + 0.06);
    return std::min(p.relaxation_tau_max_s,
                    std::max(p.relaxation_tau_min_s, tau_s));
  }
  return p.relaxation_tau_s;
}

double update_channel(const ChannelParams& p, const ModelParams& mp,
                      ChannelState& state, const ChannelInputs& inputs,
                      const double dt_s, const double lambda_um,
                      ChannelDiagnostics& diag) {
  diag = ChannelDiagnostics{};

  if (inputs.valid && inputs.kappa_um > 0.0) {
    if (state.kappa_bar == 0.0) {
      state.kappa_bar = inputs.kappa_um;
    } else {
      const double alpha_L = -std::expm1(-dt_s / mp.ln_filter_tau_s);
      state.kappa_bar += alpha_L * (inputs.kappa_um - state.kappa_bar);
    }
  }

  const double kappa_floor = 1.0 / mp.L_n_max_um;
  if (state.kappa_bar < kappa_floor) {
    diag.clamped |= 1;
  }
  const double L_n_unclamped = 1.0 / std::max(state.kappa_bar, kappa_floor);
  diag.L_n_eff_um = L_n_unclamped;
  if (diag.L_n_eff_um < mp.L_n_min_um) {
    diag.L_n_eff_um = mp.L_n_min_um;
    diag.clamped |= 1;
  } else if (diag.L_n_eff_um > mp.L_n_max_um) {
    diag.L_n_eff_um = mp.L_n_max_um;
    diag.clamped |= 1;
  }

  double Te_keV = inputs.Te_keV;
  if (Te_keV < mp.Te_min_keV) {
    Te_keV = mp.Te_min_keV;
    diag.clamped |= 2;
  } else if (Te_keV > mp.Te_max_keV) {
    Te_keV = mp.Te_max_keV;
    diag.clamped |= 2;
  }

  double I14 = inputs.I14;
  if (I14 < mp.I14_min) {
    I14 = mp.I14_min;
    diag.clamped |= 4;
  } else if (I14 > mp.I14_max) {
    I14 = mp.I14_max;
    diag.clamped |= 4;
  }

  if (inputs.valid) {
    diag.eta_eq =
        eta_equilibrium(p, I14, Te_keV, diag.L_n_eff_um, lambda_um, diag);
    diag.tau_s = relaxation_tau_s(p, diag.g);
  } else {
    diag.eta_eq = 0.0;
    diag.tau_s = p.relaxation_tau_s;
  }

  const double alpha = -std::expm1(-dt_s / diag.tau_s);
  state.eta += alpha * (diag.eta_eq - state.eta);
  return state.eta;
}

double apply_total_cap(const ModelParams& mp, ChannelState* states,
                       const std::size_t n_channels) {
  double eta_sum = 0.0;
  for (std::size_t i = 0; i < n_channels; ++i) {
    eta_sum += states[i].eta;
  }
  if (eta_sum <= mp.eta_total_cap) {
    return 1.0;
  }

  const double scale = mp.eta_total_cap / eta_sum;
  for (std::size_t i = 0; i < n_channels; ++i) {
    states[i].eta *= scale;
  }
  return scale;
}

}  // namespace tenryu::laser::hot_e_eta
