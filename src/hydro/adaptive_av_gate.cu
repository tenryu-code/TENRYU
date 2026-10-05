#include "hydro/adaptive_av_gate.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "core/error.hpp"

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

// On the device: clamp01 and smoothstep01 with the host's arithmetic (std::min / std::max rules,
// every product and sum rounded on its own), the support window of the gate around the shock.
__device__ inline double clamp01_device(const double x) {
  const double lo = (0.0 < x) ? x : 0.0;  // std::max(0.0, x)
  return (lo < 1.0) ? lo : 1.0;           // std::min(1.0, lo)
}

__device__ inline double smoothstep01_device(const double x) {
  const double y = clamp01_device(x);
  return __dmul_rn(__dmul_rn(y, y), __dsub_rn(3.0, __dmul_rn(2.0, y)));
}

__device__ inline double support_window_device(const double xi, const double u_s,
                                               const int support_ahead,
                                               const int support_behind) {
  const double ahead = static_cast<double>(support_ahead > 0 ? support_ahead : 0);
  const double behind = static_cast<double>(support_behind > 0 ? support_behind : 0);
  const double left = (u_s < 0.0) ? ahead : behind;
  const double right = (u_s < 0.0) ? behind : ahead;
  if (xi < 0.0) {
    if (!(left > 0.0) || xi < -left) {
      return 0.0;
    }
    return smoothstep01_device((xi + left) / left);
  }
  if (!(right > 0.0) || xi > right) {
    return 0.0;
  }
  return smoothstep01_device((right - xi) / right);
}

__device__ inline double blend_device(const double base, const double target, const double g) {
  return __dadd_rn(base, __dmul_rn(g, __dsub_rn(target, base)));
}

// Each cell's gate g = clamp01((1 - w) g_old + w target) (in place) and the coefficients
// base + g (coeff - base).
__global__ void adaptive_av_gate_kernel(const double* __restrict__ node_r, const int n_cells,
                                        const int windowed, const int i0, const int i1,
                                        const double r_s, const double u_s, const double dr_s,
                                        const int support_ahead, const int support_behind,
                                        const double w, const AdaptiveAVCoeff base,
                                        const AdaptiveAVCoeff coeff, double* __restrict__ gate,
                                        double* __restrict__ c1, double* __restrict__ c2,
                                        double* __restrict__ heat_C, double* __restrict__ Cpsv,
                                        double* __restrict__ cbulk) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  double target = 0.0;
  if (windowed != 0) {
    const int left_support = u_s < 0.0 ? support_ahead : support_behind;
    const int right_support = u_s < 0.0 ? support_behind : support_ahead;
    if (i >= i0 - left_support && i <= i1 + right_support) {
      const double rc = 0.5 * (node_r[i] + node_r[i + 1]);
      const double xi = (rc - r_s) / dr_s;
      target = support_window_device(xi, u_s, support_ahead, support_behind);
    }
  }
  const double g =
      clamp01_device(__dadd_rn(__dmul_rn(1.0 - w, gate[i]), __dmul_rn(w, target)));
  gate[i] = g;
  c1[i] = blend_device(base.c1, coeff.c1, g);
  c2[i] = blend_device(base.c2, coeff.c2, g);
  heat_C[i] = blend_device(base.heat_C, coeff.heat_C, g);
  Cpsv[i] = blend_device(base.Cpsv, coeff.Cpsv, g);
  cbulk[i] = blend_device(base.cbulk, coeff.cbulk, g);
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
    // no coefficient fields (the caller then uses the base coefficients)
    out.c1.reset(0);
    out.c2.reset(0);
    out.heat_C.reset(0);
    out.Cpsv.reset(0);
    out.cbulk.reset(0);
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

  const AdaptiveAVMode mode = select_mode(state, cfg, shock);
  state.adaptive_av_mode = static_cast<int>(mode);
  const AdaptiveAVCoeff coeff = mode_coeff(state, cfg, shock, mode);
  // 2026-07-26 review: a fixed per-step blend weight makes the
  // gate relaxation rate depend on the timestep count, breaking dt-refinement
  // convergence. hysteresis_tau > 0 opts into the physical-time-constant
  // blend w(dt) = 1 - exp(-dt/tau); tau == 0 keeps the legacy fixed w.
  const double w =
      (av.hysteresis_tau > 0.0 && dt > 0.0)
          ? clamp01(1.0 - std::exp(-dt / av.hysteresis_tau))
          : clamp01(av.hysteresis_w);

  // The gate and the coefficients per cell on the device.
  const bool windowed = mode != AdaptiveAVMode::Base && shock.valid && shock.dr_s > 0.0;
  constexpr int kBlock = 256;
  adaptive_av_gate_kernel<<<(n_cells + kBlock - 1) / kBlock, kBlock>>>(
      state.x_r.data(), n_cells, windowed ? 1 : 0, shock.i0, shock.i1, shock.r_s, shock.u_s,
      shock.dr_s, av.support_ahead, av.support_behind, w, av.base, coeff,
      state.adaptive_av_gate.data(), out.c1.data(), out.c2.data(), out.heat_C.data(),
      out.Cpsv.data(), out.cbulk.data());
  TENRYU_ASSERT(cudaGetLastError() == cudaSuccess, "adaptive AV gate kernel launch failed");
}

}  // namespace tenryu::hydro
