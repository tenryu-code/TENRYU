#include "hydro/adaptive_av_gate.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tenryu::hydro {
namespace {

using AdaptiveAVCoeff =
    core::Config::NumericsConfig::HydroConfig::AdaptiveAVCoeff;

double clamp01(const double x) {
  return std::min(1.0, std::max(0.0, x));
}

double smoothstep01(const double x) {
  const double y = clamp01(x);
  return y * y * (3.0 - 2.0 * y);
}

AdaptiveAVCoeff blend_coeff(const AdaptiveAVCoeff& a,
                            const AdaptiveAVCoeff& b,
                            const double w) {
  const double x = clamp01(w);
  AdaptiveAVCoeff out;
  out.c1 = a.c1 + x * (b.c1 - a.c1);
  out.c2 = a.c2 + x * (b.c2 - a.c2);
  out.heat_C = a.heat_C + x * (b.heat_C - a.heat_C);
  out.Cpsv = a.Cpsv + x * (b.Cpsv - a.Cpsv);
  out.cbulk = a.cbulk + x * (b.cbulk - a.cbulk);
  return out;
}

AdaptiveAVMode select_mode(const core::State& state,
                           const core::Config& cfg,
                           const LeadingShockState& shock) {
  if (!shock.valid || !(state.adaptive_av_r0 > 0.0)) {
    return AdaptiveAVMode::Base;
  }
  const auto& av = cfg.numerics.hydro.adaptive_av;
  const double r_frac = shock.r_s / state.adaptive_av_r0;
  if (!shock.bounce_seen && shock.u_s < 0.0) {
    if (r_frac > av.taper_r_start) {
      return AdaptiveAVMode::PrimaryFull;
    }
    if (r_frac > av.taper_r_end) {
      return AdaptiveAVMode::PrimaryTaper;
    }
    return AdaptiveAVMode::Base;
  }
  if (shock.bounce_seen && shock.u_s > 0.0 && shock.shell_mean_div_u > 0.0) {
    return AdaptiveAVMode::Rebound;
  }
  return AdaptiveAVMode::Base;
}

AdaptiveAVCoeff mode_coeff(const core::State& state,
                           const core::Config& cfg,
                           const LeadingShockState& shock,
                           const AdaptiveAVMode mode) {
  const auto& av = cfg.numerics.hydro.adaptive_av;
  if (mode == AdaptiveAVMode::PrimaryFull) {
    return av.primary;
  }
  if (mode == AdaptiveAVMode::Rebound) {
    return av.rebound;
  }
  if (mode == AdaptiveAVMode::PrimaryTaper && state.adaptive_av_r0 > 0.0) {
    const double r_frac = shock.r_s / state.adaptive_av_r0;
    const double w = (r_frac - av.taper_r_end) /
                     std::max(av.taper_r_start - av.taper_r_end, 1.0e-30);
    return blend_coeff(av.base, av.primary, smoothstep01(w));
  }
  return av.base;
}

double support_window(const double xi,
                      const double u_s,
                      const int support_ahead,
                      const int support_behind) {
  const double ahead = std::max(0, support_ahead);
  const double behind = std::max(0, support_behind);
  const double left = (u_s < 0.0) ? ahead : behind;
  const double right = (u_s < 0.0) ? behind : ahead;
  if (xi < 0.0) {
    if (!(left > 0.0) || xi < -left) {
      return 0.0;
    }
    return smoothstep01((xi + left) / left);
  }
  if (!(right > 0.0) || xi > right) {
    return 0.0;
  }
  return smoothstep01((right - xi) / right);
}

}  // namespace

void build_adaptive_av_fields_1d(core::State& state,
                                 const core::Config& cfg,
                                 const LeadingShockState& shock,
                                 const double dt,
                                 AdaptiveAVFields& out) {
  const auto& av = cfg.numerics.hydro.adaptive_av;
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 || state.x_r.size() != static_cast<std::size_t>(n_cells + 1)) {
    state.adaptive_av_mode = static_cast<int>(AdaptiveAVMode::Base);
    return;
  }
  if (state.adaptive_av_gate.size() != state.rho.size()) {
    state.adaptive_av_gate.reset(state.rho.size());
    state.adaptive_av_gate.fill(0.0);
  }

  out.c1.reset(state.rho.size());
  out.c2.reset(state.rho.size());
  out.heat_C.reset(state.rho.size());
  out.Cpsv.reset(state.rho.size());
  out.cbulk.reset(state.rho.size());

  std::vector<double> node_r;
  std::vector<double> gate_old;
  state.x_r.copy_to_host(node_r);
  state.adaptive_av_gate.copy_to_host(gate_old);

  const AdaptiveAVMode mode = select_mode(state, cfg, shock);
  state.adaptive_av_mode = static_cast<int>(mode);
  const AdaptiveAVCoeff coeff = mode_coeff(state, cfg, shock, mode);
  // k01 §8.1 (AI review 2026-07-26): a fixed per-step blend weight makes the
  // gate relaxation rate depend on the timestep count, breaking dt-refinement
  // convergence. hysteresis_tau > 0 opts into the physical-time-constant
  // blend w(dt) = 1 - exp(-dt/tau); tau == 0 keeps the legacy fixed w.
  const double w =
      (av.hysteresis_tau > 0.0 && dt > 0.0)
          ? clamp01(1.0 - std::exp(-dt / av.hysteresis_tau))
          : clamp01(av.hysteresis_w);

  std::vector<double> gate_new(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> c1(static_cast<std::size_t>(n_cells), av.base.c1);
  std::vector<double> c2(static_cast<std::size_t>(n_cells), av.base.c2);
  std::vector<double> heat_C(static_cast<std::size_t>(n_cells), av.base.heat_C);
  std::vector<double> Cpsv(static_cast<std::size_t>(n_cells), av.base.Cpsv);
  std::vector<double> cbulk(static_cast<std::size_t>(n_cells), av.base.cbulk);

  for (int i = 0; i < n_cells; ++i) {
    double target = 0.0;
    if (mode != AdaptiveAVMode::Base && shock.valid && shock.dr_s > 0.0) {
      const int left_support =
          shock.u_s < 0.0 ? av.support_ahead : av.support_behind;
      const int right_support =
          shock.u_s < 0.0 ? av.support_behind : av.support_ahead;
      const bool in_cluster_support =
          i >= shock.i0 - left_support && i <= shock.i1 + right_support;
      if (in_cluster_support) {
        const double rc = 0.5 * (node_r[static_cast<std::size_t>(i)] +
                                 node_r[static_cast<std::size_t>(i + 1)]);
        const double xi = (rc - shock.r_s) / shock.dr_s;
        target = support_window(xi, shock.u_s, av.support_ahead, av.support_behind);
      }
    }
    const double g_old = gate_old[static_cast<std::size_t>(i)];
    const double g = clamp01((1.0 - w) * g_old + w * target);
    gate_new[static_cast<std::size_t>(i)] = g;
    c1[static_cast<std::size_t>(i)] = av.base.c1 + g * (coeff.c1 - av.base.c1);
    c2[static_cast<std::size_t>(i)] = av.base.c2 + g * (coeff.c2 - av.base.c2);
    heat_C[static_cast<std::size_t>(i)] =
        av.base.heat_C + g * (coeff.heat_C - av.base.heat_C);
    Cpsv[static_cast<std::size_t>(i)] =
        av.base.Cpsv + g * (coeff.Cpsv - av.base.Cpsv);
    cbulk[static_cast<std::size_t>(i)] =
        av.base.cbulk + g * (coeff.cbulk - av.base.cbulk);
  }

  state.adaptive_av_gate.copy_from_host(gate_new);
  out.c1.copy_from_host(c1);
  out.c2.copy_from_host(c2);
  out.heat_C.copy_from_host(heat_C);
  out.Cpsv.copy_from_host(Cpsv);
  out.cbulk.copy_from_host(cbulk);
}

}  // namespace tenryu::hydro
