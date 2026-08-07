#include "radiation/mode_selector.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "core/constants.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kMilneLambda = 0.7104;
constexpr double kSmallSigma = 1.0e-30;
constexpr double kAlwaysDiffusiveTau = 5.0;
constexpr double kTemperatureFloor = 1.0e-12;
constexpr double kDdmcMinTemperature = 1.0;

struct ResidenceSupport {
  double t_ae = DBL_MAX;
  bool valid = false;
};

bool omega_gate_enabled(const ModeSelectorConfig& config) {
  return config.omega_ddmc > 0.0;
}

ResidenceSupport compute_material_coupling_time_from_fleck(const double fleck_f,
                                                           const double dt,
                                                           const double alpha) {
  ResidenceSupport out{};
  const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
  const double dt_safe = std::max(dt, 0.0);
  const double one_minus_f = std::max(1.0 - fleck_f, 0.0);
  if (dt_safe <= 0.0 || one_minus_f <= kSmallSigma) {
    return out;
  }
  out.t_ae = alpha_safe * dt_safe * std::max(fleck_f, 0.0) / one_minus_f;
  out.valid = std::isfinite(out.t_ae) && out.t_ae > 0.0;
  return out;
}

ResidenceSupport compute_material_coupling_time_from_coeffs(const CellRadiationCoeffs& coeffs,
                                                            const std::size_t cell) {
  ResidenceSupport out{};
  if (cell < coeffs.cv_e.size() && cell < coeffs.Te_eval.size() &&
      cell < coeffs.sigma_p_em.size()) {
    const double cv_e = coeffs.cv_e[cell];
    const double Te = std::max(coeffs.Te_eval[cell], kTemperatureFloor);
    const double sigma_p_em = coeffs.sigma_p_em[cell];
    const double denom = 4.0 * core::constants::a_eV * core::constants::c_light * Te * Te * Te *
                         sigma_p_em;
    if (std::isfinite(cv_e) && cv_e > 0.0 && std::isfinite(Te) && Te >= kDdmcMinTemperature &&
        std::isfinite(denom) && denom > kSmallSigma && std::isfinite(sigma_p_em) &&
        sigma_p_em > kSmallSigma) {
      out.t_ae = cv_e / denom;
      out.valid = std::isfinite(out.t_ae) && out.t_ae > 0.0;
      if (out.valid) {
        return out;
      }
    }
  }
  return out;
}

bool coeffs_match_cell_layout(const CellRadiationCoeffs* coeffs, const std::int64_t n_cells) {
  if (coeffs == nullptr) {
    return false;
  }
  const std::size_t n_cells_us = static_cast<std::size_t>(std::max<std::int64_t>(n_cells, 0));
  return coeffs->cv_e.size() == n_cells_us && coeffs->Te_eval.size() == n_cells_us &&
         coeffs->sigma_p_em.size() == n_cells_us;
}

double compute_tau_saturated(const double sigma_tr, const double dx) {
  if (!std::isfinite(sigma_tr) || !std::isfinite(dx)) {
    return DBL_MAX;
  }
  const double sigma = std::max(sigma_tr, 0.0);
  const double length = std::max(dx, 0.0);
  if (sigma <= 0.0 || length <= 0.0) {
    return 0.0;
  }
  const double tau = sigma * length;
  if (!std::isfinite(tau)) {
    return DBL_MAX;
  }
  return std::min(tau, DBL_MAX);
}

double compute_standard_p_mu1_unclamped(const double sigma_R, const double dx) {
  const double tau = compute_tau_saturated(sigma_R, dx);
  const double denom = 3.0 * tau + 6.0 * kMilneLambda;
  if (denom <= 0.0) {
    return 0.0;
  }
  return 10.0 / denom;
}

double compute_emissivity_preserving_p_scalar(const double tau,
                                              const double omega,
                                              bool* used_standard_fallback) {
  const double one_minus_omega = std::max(1.0 - omega, 0.0);
  const double sqrt_term = std::sqrt(3.0 * one_minus_omega);
  const double eps_prime = (4.0 / 3.0) * sqrt_term / (1.0 + kMilneLambda * sqrt_term);

  const double tau2 = tau * tau;
  const double beta_inner = 3.0 * one_minus_omega * tau2 +
                            2.25 * one_minus_omega * one_minus_omega * tau2 * tau2;
  const double beta = 1.5 * one_minus_omega * tau2 + std::sqrt(std::max(beta_inner, 0.0));

  const double denom = beta - (4.0 / 3.0) * eps_prime * tau;
  if (denom <= 0.0) {
    *used_standard_fallback = true;
    const double standard_p = 4.0 / (3.0 * tau + 6.0 * kMilneLambda);
    return std::max(0.0, standard_p);
  }

  double p_hat = eps_prime * beta / denom;
  if (!std::isfinite(p_hat) || p_hat < 0.0) {
    *used_standard_fallback = true;
    const double standard_p = 4.0 / (3.0 * tau + 6.0 * kMilneLambda);
    return std::max(0.0, standard_p);
  }

  // p_hat(mu=1) <= 1 requires p_hat <= 4/5.
  p_hat = std::clamp(p_hat, 0.0, 0.8);
  *used_standard_fallback = false;
  return p_hat;
}

int node_index_2d_rz(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

double point_line_distance(const double px,
                           const double pz,
                           const double ax,
                           const double az,
                           const double bx,
                           const double bz) {
  const double ex = bx - ax;
  const double ez = bz - az;
  const double len = std::sqrt(ex * ex + ez * ez);
  if (len <= 0.0) {
    return 0.0;
  }
  const double cross = std::abs(ex * (az - pz) - (ax - px) * ez);
  return cross / len;
}

double compute_cell_length_2d_rz(const std::vector<double>& node_r,
                                 const std::vector<double>& node_z,
                                 const int nr,
                                 const int nz,
                                 const int cell) {
  if (nr <= 0 || nz <= 0 || cell < 0 || cell >= nr * nz) {
    return 0.0;
  }
  const int i = cell / nz;
  const int j = cell - i * nz;

  const int n00 = node_index_2d_rz(i, j, nz);
  const int n10 = node_index_2d_rz(i + 1, j, nz);
  const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz(i, j + 1, nz);

  const int n_nodes = static_cast<int>(node_r.size());
  if (n00 < 0 || n10 < 0 || n11 < 0 || n01 < 0 || n00 >= n_nodes || n10 >= n_nodes ||
      n11 >= n_nodes || n01 >= n_nodes || node_z.size() != node_r.size()) {
    return 0.0;
  }

  const double r0 = node_r[static_cast<std::size_t>(n00)];
  const double z0 = node_z[static_cast<std::size_t>(n00)];
  const double r1 = node_r[static_cast<std::size_t>(n10)];
  const double z1 = node_z[static_cast<std::size_t>(n10)];
  const double r2 = node_r[static_cast<std::size_t>(n11)];
  const double z2 = node_z[static_cast<std::size_t>(n11)];
  const double r3 = node_r[static_cast<std::size_t>(n01)];
  const double z3 = node_z[static_cast<std::size_t>(n01)];

  const double rc = 0.25 * (r0 + r1 + r2 + r3);
  const double zc = 0.25 * (z0 + z1 + z2 + z3);
  const double d0 = point_line_distance(rc, zc, r0, z0, r3, z3);  // face 0
  const double d1 = point_line_distance(rc, zc, r1, z1, r2, z2);  // face 1
  const double d2 = point_line_distance(rc, zc, r0, z0, r1, z1);  // face 2
  const double d3 = point_line_distance(rc, zc, r3, z3, r2, z2);  // face 3
  const double d_min = std::max(0.0, std::min(std::min(d0, d1), std::min(d2, d3)));
  return 2.0 * d_min;
}

}  // namespace

ModeSelector::ModeSelector(const std::int64_t n_cells,
                           const int n_groups,
                           const ModeSelectorConfig& config)
    : n_cells_(n_cells),
      n_groups_(n_groups),
      config_(config),
      mode_(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups),
            TransportMode::IMC),
      tau_(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups), 0.0),
      omega_(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups), 0.0),
      cell_length_(static_cast<std::size_t>(n_cells), 0.0) {}

std::size_t ModeSelector::index(const std::int64_t cell, const int group) const {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

double ModeSelector::compute_optical_depth(const double sigma_tr,
                                           const double cell_length_cm) {
  return compute_tau_saturated(sigma_tr, cell_length_cm);
}

double ModeSelector::compute_scattering_ratio(const double fleck_f,
                                              const double sigma_a,
                                              const double sigma_s_phys) {
  const double denom = sigma_a + sigma_s_phys;
  if (denom < kSmallSigma) {
    return std::clamp(1.0 - fleck_f, 0.0, 1.0);
  }
  const double ratio = ((1.0 - fleck_f) * sigma_a + sigma_s_phys) / denom;
  return std::clamp(ratio, 0.0, 1.0);
}

double ModeSelector::compute_cell_length(const double r_inner,
                                         const double r_outer) {
  return std::max(r_outer - r_inner, 0.0);
}

bool ModeSelector::check_conversion_probability_constraint(const double sigma_R,
                                                           const double dx,
                                                           const double omega) const {
  const double tau = compute_tau_saturated(sigma_R, dx);
  if (tau <= 0.0) {
    return false;
  }

  if (!config_.emissivity_preserving) {
    return compute_standard_p_mu1_unclamped(sigma_R, dx) <= 1.0 + 1.0e-12;
  }

  bool used_standard_fallback = false;
  const double p_scalar =
      compute_emissivity_preserving_p_scalar(tau, std::clamp(omega, 0.0, 1.0),
                                             &used_standard_fallback);

  const double p_mu1 = used_standard_fallback
                           ? compute_standard_p_mu1_unclamped(sigma_R, dx)
                           : (1.25 * p_scalar);
  return p_mu1 <= 1.0 + 1.0e-12;
}

void ModeSelector::compute_modes(const std::vector<double>& node_r,
                                 const std::vector<double>& sigma_R,
                                 const std::vector<double>& fleck_f,
                                 const std::vector<double>& sigma_a,
                                 const std::vector<double>& sigma_s_phys,
                                 const CellRadiationCoeffs* coeffs,
                                 const double dt,
                                 const double alpha) {
  for (std::int64_t c = 0; c < n_cells_; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    cell_length_[c_us] = compute_cell_length(node_r[c_us], node_r[c_us + 1]);
  }
  classify_modes_from_lengths(
      sigma_R, fleck_f, sigma_a, sigma_s_phys, coeffs, dt, alpha, true);
}

void ModeSelector::compute_modes_2d_rz(const std::vector<double>& node_r,
                                       const std::vector<double>& node_z,
                                       const int nr,
                                       const int nz,
                                       const std::vector<double>& sigma_R,
                                       const std::vector<double>& fleck_f,
                                       const std::vector<double>& sigma_a,
                                       const std::vector<double>& sigma_s_phys,
                                       const CellRadiationCoeffs* coeffs,
                                       const double dt,
                                       const double alpha) {
  for (std::int64_t c = 0; c < n_cells_; ++c) {
    cell_length_[static_cast<std::size_t>(c)] =
        compute_cell_length_2d_rz(node_r,
                                  node_z,
                                  nr,
                                  nz,
                                  static_cast<int>(c));
  }
  classify_modes_from_lengths(
      sigma_R, fleck_f, sigma_a, sigma_s_phys, coeffs, dt, alpha, false);
}

void ModeSelector::classify_modes_from_lengths(const std::vector<double>& sigma_R,
                                               const std::vector<double>& fleck_f,
                                               const std::vector<double>& sigma_a,
                                               const std::vector<double>& sigma_s_phys,
                                               const CellRadiationCoeffs* /*coeffs*/,
                                               const double /*dt*/,
                                               const double /*alpha*/,
                                               const bool allow_rw) {
  omega_below_threshold_ = 0;
  const bool use_legacy_omega_gate = omega_gate_enabled(config_);
  (void)allow_rw;
  const bool rw_enabled = false;  // Legacy RW mode is disabled; PGRW runs inside IMC.
  for (std::int64_t c = 0; c < n_cells_; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const double dx = cell_length_[c_us];
    const double f = fleck_f[c_us];
    for (int g = 0; g < n_groups_; ++g) {
      const std::size_t idx = index(c, g);
      const double sigma_R_cg = std::max(sigma_R[idx], 0.0);
      const double sigma_a_cg = std::max(sigma_a[idx], 0.0);
      const double sigma_s_cg = sigma_s_phys.empty() ? 0.0 : std::max(sigma_s_phys[idx], 0.0);

      const double tau = compute_optical_depth(sigma_R_cg, dx);
      const double omega = compute_scattering_ratio(f, sigma_a_cg, sigma_s_cg);
      tau_[idx] = tau;
      omega_[idx] = omega;

      if (sigma_R_cg < config_.sigma_floor) {
        mode_[idx] = TransportMode::IMC;
        if (use_legacy_omega_gate && omega < config_.omega_ddmc) {
          ++omega_below_threshold_;
        }
        continue;
      }

      const bool omega_ok = !use_legacy_omega_gate || (omega >= config_.omega_ddmc);
      const bool tau_ok = (tau >= config_.tau_ddmc);
      const bool p_ok = check_conversion_probability_constraint(sigma_R_cg, dx, omega);
      const bool rw_ok = rw_enabled && (tau >= config_.tau_rw);

      if (use_legacy_omega_gate && !omega_ok) {
        ++omega_below_threshold_;
      }

      if (omega_ok && tau_ok && p_ok) {
        mode_[idx] = TransportMode::DDMC;
      } else if (rw_ok) {
        mode_[idx] = TransportMode::RW;
      } else {
        mode_[idx] = TransportMode::IMC;
      }
    }
  }
}

TransportMode ModeSelector::get_mode(const std::int64_t cell, const int group) const {
  if (cell < 0 || cell >= n_cells_ || group < 0 || group >= n_groups_) {
    return TransportMode::IMC;
  }
  return mode_[index(cell, group)];
}

double ModeSelector::get_tau(const std::int64_t cell, const int group) const {
  if (cell < 0 || cell >= n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return tau_[index(cell, group)];
}

double ModeSelector::get_omega(const std::int64_t cell, const int group) const {
  if (cell < 0 || cell >= n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return omega_[index(cell, group)];
}

void ModeSelector::force_imc(const std::int64_t cell, const int group) {
  if (cell < 0 || cell >= n_cells_ || group < 0 || group >= n_groups_) {
    return;
  }
  mode_[index(cell, group)] = TransportMode::IMC;
}

ModeSelector::HysteresisResult ModeSelector::apply_hysteresis(
    const std::vector<TransportMode>& prev_mode,
    const std::vector<double>& prev_tau,
    std::vector<std::uint8_t>& hold_count) {
  const std::size_t total = mode_.size();
  const auto recount_omega_below = [this]() {
    omega_below_threshold_ = 0;
    if (!omega_gate_enabled(config_)) {
      return;
    }
    for (const double omega : omega_) {
      if (omega < config_.omega_ddmc) {
        ++omega_below_threshold_;
      }
    }
  };

  if (prev_mode.size() != total || prev_tau.size() != total ||
      hold_count.size() != total) {
    if (hold_count.size() != total) {
      hold_count.assign(total, 0U);
    }
    recount_omega_below();
    return {};
  }

  const double tau_on = config_.tau_ddmc;
  const double effective_tau_off =
      (config_.tau_ddmc_off >= 0.0) ? config_.tau_ddmc_off : config_.tau_ddmc;
  const bool use_legacy_omega_gate = omega_gate_enabled(config_);
  const double effective_omega_off = use_legacy_omega_gate
                                         ? ((config_.omega_ddmc_off >= 0.0) ? config_.omega_ddmc_off
                                                                            : config_.omega_ddmc)
                                         : -1.0;

  HysteresisResult result{};
  for (std::int64_t c = 0; c < n_cells_; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const double dx = cell_length_[c_us];
    for (int g = 0; g < n_groups_; ++g) {
      const std::size_t idx = index(c, g);
      const TransportMode baseline = mode_[idx];
      const TransportMode prev = prev_mode[idx];
      const double curr_tau = tau_[idx];
      const double prev_tau_val = prev_tau[idx];
      const double curr_omega = omega_[idx];
      const double sigma_R_approx = (dx > 0.0) ? (curr_tau / dx) : 0.0;
      const std::uint8_t hold = hold_count[idx];

      const auto hold_incremented = [hold]() -> std::uint8_t {
        return (hold < static_cast<std::uint8_t>(255))
                   ? static_cast<std::uint8_t>(hold + 1U)
                   : static_cast<std::uint8_t>(255);
      };

      if (sigma_R_approx < config_.sigma_floor) {
        mode_[idx] = TransportMode::IMC;
        hold_count[idx] = hold_incremented();
        if (prev == TransportMode::DDMC) {
          ++result.switches_ddmc_to_imc;
        }
        continue;
      }

      if (prev != TransportMode::DDMC) {
        const double rate =
            (prev_tau_val > 1.0e-30) ? std::abs(curr_tau - prev_tau_val) / prev_tau_val
                                     : 0.0;
        const bool baseline_entry =
            (baseline == TransportMode::DDMC) && (curr_tau >= tau_on);
        const bool entry_ok = baseline_entry && (hold >= config_.mode_hold) &&
                              (rate <= config_.rate_max);
        if (entry_ok) {
          mode_[idx] = TransportMode::DDMC;
          hold_count[idx] = 0U;
          ++result.switches_imc_to_ddmc;
        } else {
          mode_[idx] = baseline_entry ? prev : baseline;
          hold_count[idx] = hold_incremented();
        }
      } else {
        const bool exit_tau = (curr_tau < effective_tau_off);
        const bool exit_omega = use_legacy_omega_gate && (curr_omega < effective_omega_off);
        const bool exit_p =
            !check_conversion_probability_constraint(sigma_R_approx, dx, curr_omega);
        if (exit_tau || exit_omega || exit_p) {
          mode_[idx] = baseline;
          hold_count[idx] = 0U;
          ++result.switches_ddmc_to_imc;
        } else {
          mode_[idx] = TransportMode::DDMC;
          hold_count[idx] = hold_incremented();
        }
      }
    }
  }

  recount_omega_below();
  return result;
}

std::int64_t ModeSelector::count_ddmc() const {
  std::int64_t count = 0;
  for (const TransportMode mode : mode_) {
    if (mode == TransportMode::DDMC) {
      ++count;
    }
  }
  return count;
}

std::int64_t ModeSelector::count_rw() const {
  std::int64_t count = 0;
  for (const TransportMode mode : mode_) {
    if (mode == TransportMode::RW) {
      ++count;
    }
  }
  return count;
}

std::int64_t ModeSelector::count_imc() const {
  std::int64_t count = 0;
  for (const TransportMode mode : mode_) {
    if (mode == TransportMode::IMC) {
      ++count;
    }
  }
  return count;
}

std::int64_t ModeSelector::count_omega_below_threshold() const {
  return omega_below_threshold_;
}

}  // namespace tenryu::radiation
