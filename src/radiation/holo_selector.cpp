#include "radiation/holo_selector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaFloor = 1.0e-30;
constexpr double kMaxTemperatureForT4 = 1.0e6;

bool has_required_input(const HoloSelectorInputs& in) {
  const std::size_t n_cells = static_cast<std::size_t>(std::max(in.n_cells, 0));
  const std::size_t n_groups = static_cast<std::size_t>(std::max(in.n_groups, 0));
  const std::size_t n_cell_groups = n_cells * n_groups;
  return in.dimension == HoloGeometryDimension::Spherical1D &&
         in.n_cells >= 0 && in.n_groups > 0 &&
         in.node_r != nullptr && in.node_r->size() == n_cells + 1U &&
         in.mass != nullptr && in.mass->size() == n_cells &&
         in.Te != nullptr && in.Te->size() == n_cells &&
         in.sigma_R != nullptr && in.sigma_R->size() == n_cell_groups &&
         in.planck != nullptr &&
         (in.n_groups == 1 || in.planck->n_groups() == in.n_groups) &&
         in.cell_is_void != nullptr && in.cell_is_void->size() == n_cells;
}

void ensure_state_size(HoloSelectorStateView state,
                       const std::size_t n_cells,
                       const bool keep_existing) {
  if (!keep_existing || state.core_mask.size() != n_cells) {
    state.core_mask.assign(n_cells, 0U);
  }
  state.patch_mask.assign(n_cells, 0U);
  state.prev_core_mask.assign(n_cells, 0U);
  if (!keep_existing || state.hold_count.size() != n_cells) {
    state.hold_count.assign(n_cells, 0);
  }
  if (!keep_existing || state.dwell_count.size() != n_cells) {
    state.dwell_count.assign(n_cells, 0);
  }
  state.tau_R.assign(n_cells, 0.0);
  state.reduced_flux.assign(n_cells, 0.0);
  state.mass_q.assign(n_cells, 0.0);
  state.lo_weight.assign(n_cells, 0.0);
}

double finite_nonnegative(const double value) {
  return std::isfinite(value) ? std::max(value, 0.0) : 0.0;
}

double clamped_temperature_for_t4_host(const double temperature_eV) {
  return std::min(std::max(temperature_eV, 1.0e-12), kMaxTemperatureForT4);
}

void build_patch_and_weights(const std::vector<std::uint8_t>& core,
                             const std::vector<std::uint8_t>& cell_is_void,
                             const int guard_cells,
                             const int blend_cells,
                             std::vector<std::uint8_t>& patch,
                             std::vector<double>& lo_weight) {
  const int n_cells = static_cast<int>(core.size());
  patch.assign(core.size(), 0U);
  lo_weight.assign(core.size(), 0.0);
  std::vector<int> distance(core.size(), n_cells + 1);
  const int patch_radius = std::max(std::max(guard_cells, 0), std::max(blend_cells, 0));
  for (int c = 0; c < n_cells; ++c) {
    if (core[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int first = std::max(0, c - patch_radius);
    const int last = std::min(n_cells - 1, c + patch_radius);
    for (int j = first; j <= last; ++j) {
      if (cell_is_void[static_cast<std::size_t>(j)] == 0U) {
        patch[static_cast<std::size_t>(j)] = 1U;
        distance[static_cast<std::size_t>(j)] =
            std::min(distance[static_cast<std::size_t>(j)], std::abs(j - c));
      }
    }
  }
  const int blend = std::max(blend_cells, 0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    if (core[c_us] != 0U) {
      patch[c_us] = 1U;
      lo_weight[c_us] = 1.0;
    } else if (patch[c_us] != 0U && blend > 0 && distance[c_us] > 0 &&
               distance[c_us] < blend) {
      lo_weight[c_us] =
          static_cast<double>(blend - distance[c_us]) / static_cast<double>(blend);
    }
  }
}

}  // namespace

double temperature_weighted_rosseland_sigma_R(const PlanckTable& planck,
                                              const std::vector<double>& sigma_R,
                                              const std::size_t base,
                                              const int n_groups,
                                              const double temperature_eV,
                                              const double temperature_floor_eV) {
  if (n_groups <= 1) {
    return finite_nonnegative(sigma_R[base]);
  }

  const double T =
      clamped_temperature_for_t4_host(std::max(temperature_eV, temperature_floor_eV));
  const double dT = std::max(T * 0.01, 1.0e-3);
  const double T_hi = T + dT;
  const double T_lo = std::max(T - dT, 1.0e-6);
  double weight_sum = 0.0;
  double denom = 0.0;
  double sigma_max = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const std::size_t idx = base + static_cast<std::size_t>(g);
    const double sigma_g = finite_nonnegative(sigma_R[idx]);
    sigma_max = std::max(sigma_max, sigma_g);
    const double t4_hi = T_hi * T_hi * T_hi * T_hi;
    const double t4_lo = T_lo * T_lo * T_lo * T_lo;
    const double b_hi = std::max(planck.interpolate_b_host(g, T_hi), 0.0);
    const double b_lo = std::max(planck.interpolate_b_host(g, T_lo), 0.0);
    const double dT_eff = std::max(T_hi - T_lo, 1.0e-30);
    const double w_g = std::max((t4_hi * b_hi - t4_lo * b_lo) / dT_eff, 0.0);
    weight_sum += w_g;
    denom += w_g / std::max(sigma_g, kSigmaFloor);
  }
  return (weight_sum > 0.0 && denom > 0.0) ? (weight_sum / denom) : sigma_max;
}

HoloSelectorDiagnostics update_holo_core_mask(const HoloSelectorConfig& cfg,
                                              const HoloSelectorInputs& in,
                                              HoloSelectorStateView state) {
  HoloSelectorDiagnostics diag{};
  const std::size_t n_cells = static_cast<std::size_t>(std::max(in.n_cells, 0));
  const std::size_t n_groups = static_cast<std::size_t>(std::max(in.n_groups, 0));

  if (!cfg.enabled || !has_required_input(in)) {
    ensure_state_size(state, n_cells, false);
    state.valid = false;
    return diag;
  }

  diag.active = true;
  const bool had_valid_state = state.valid && state.core_mask.size() == n_cells;
  std::vector<std::uint8_t> previous(n_cells, 0U);
  if (had_valid_state) {
    previous = state.core_mask;
  }
  ensure_state_size(state, n_cells, had_valid_state);
  state.prev_core_mask = previous;

  const auto& node_r = *in.node_r;
  const auto& mass = *in.mass;
  const auto& Te = *in.Te;
  const auto& sigma_R = *in.sigma_R;
  const auto& planck = *in.planck;
  const auto& cell_is_void = *in.cell_is_void;

  double shell_mass = 0.0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (cell_is_void[c] == 0U) {
      shell_mass += finite_nonnegative(mass[c]);
    }
  }

  double mass_prefix = 0.0;
  bool have_tau = false;
  double tau_min = std::numeric_limits<double>::infinity();
  double tau_max = 0.0;
  std::vector<std::uint8_t> base_mask(n_cells, 0U);
  const bool use_hysteresis = cfg.tau_on > 0.0 && cfg.tau_off > 0.0;
  const double coupling_tau = std::max(cfg.coupling_tau, 0.0);
  const double tau_on = use_hysteresis ? std::max(cfg.tau_on, 0.0) : coupling_tau;
  const double tau_off =
      use_hysteresis ? std::min(std::max(cfg.tau_off, 0.0), tau_on) : coupling_tau;
  const int min_dwell_steps = use_hysteresis ? std::max(cfg.min_dwell_steps, 0) : 0;
  for (std::size_t c = 0; c < n_cells; ++c) {
    const double m = (cell_is_void[c] == 0U) ? finite_nonnegative(mass[c]) : 0.0;
    const double q = (shell_mass > 0.0) ? ((mass_prefix + 0.5 * m) / shell_mass) : 0.0;
    state.mass_q[c] = std::clamp(q, 0.0, 1.0);
    mass_prefix += m;

    const double r_lo = node_r[c];
    const double r_hi = node_r[c + 1U];
    const double dx = (std::isfinite(r_lo) && std::isfinite(r_hi))
                          ? std::max(r_hi - r_lo, 0.0)
                          : 0.0;
    const std::size_t base = c * n_groups;
    const double sigma_mean =
        temperature_weighted_rosseland_sigma_R(
            planck, sigma_R, base, in.n_groups, Te[c], cfg.temperature_floor);
    const double tau = sigma_mean * dx;
    state.tau_R[c] = std::isfinite(tau) ? std::max(tau, 0.0) : 0.0;
    state.reduced_flux[c] = 0.0;

    if (cell_is_void[c] == 0U && dx > 0.0) {
      have_tau = true;
      tau_min = std::min(tau_min, state.tau_R[c]);
      tau_max = std::max(tau_max, state.tau_R[c]);
      if (!use_hysteresis) {
        if (state.tau_R[c] >= coupling_tau) {
          base_mask[c] = 1U;
        }
      } else if (previous[c] != 0U) {
        if (state.tau_R[c] >= tau_off ||
            state.dwell_count[c] < min_dwell_steps) {
          base_mask[c] = 1U;
        }
      } else if (state.tau_R[c] >= tau_on) {
        base_mask[c] = 1U;
      }
    }
  }

  std::int64_t n_base_core = 0;
  for (const auto value : base_mask) {
    if (value != 0U) {
      ++n_base_core;
    }
  }
  if (n_base_core < static_cast<std::int64_t>(std::max(cfg.min_lo_cells, 0))) {
    base_mask.assign(n_cells, 0U);
  }

  for (std::size_t c = 0; c < n_cells; ++c) {
    if (base_mask[c] != 0U) {
      ++diag.n_core_cells;
    }
    if (previous[c] == 0U && base_mask[c] != 0U) {
      ++diag.n_entered;
    } else if (previous[c] != 0U && base_mask[c] == 0U) {
      ++diag.n_exited;
    }
  }

  state.core_mask = std::move(base_mask);
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.core_mask[c] != 0U) {
      if (previous[c] != 0U) {
        state.dwell_count[c] =
            (state.dwell_count[c] < std::numeric_limits<std::int32_t>::max())
                ? state.dwell_count[c] + 1
                : state.dwell_count[c];
      } else {
        state.dwell_count[c] = 1;
      }
    } else {
      state.dwell_count[c] = 0;
    }
  }
  build_patch_and_weights(state.core_mask,
                          cell_is_void,
                          cfg.guard_cells,
                          cfg.blend_cells,
                          state.patch_mask,
                          state.lo_weight);
  for (std::size_t c = 0; c < n_cells; ++c) {
    if (state.patch_mask[c] != 0U) {
      ++diag.n_patch_cells;
    }
    if (state.patch_mask[c] != 0U && state.core_mask[c] == 0U &&
        state.lo_weight[c] > 0.0) {
      ++diag.n_blend_cells;
    }
  }
  state.valid = true;
  diag.tau_R_min = have_tau ? tau_min : 0.0;
  diag.tau_R_max = have_tau ? tau_max : 0.0;
  diag.reduced_flux_max = 0.0;
  return diag;
}

}  // namespace tenryu::radiation
