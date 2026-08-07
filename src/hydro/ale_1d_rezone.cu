#include "hydro/ale_1d_rezone.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace tenryu::hydro::ale1d {
namespace {

constexpr double kTiny = 1.0e-30;
constexpr double kKernelIntegralFloor = 1.0e-14;

struct MassMap {
  std::vector<double> node_x;
  std::vector<double> cell_x;
  std::vector<double> dx;
};

struct MonitorKernel {
  std::vector<double> values;
  double integral = 0.0;
  double active_cells = 0.0;
};

int effective_cell_count(const core::State& state, const core::Config& cfg) {
  if (state.mesh.topo.n_cells > 0) {
    return state.mesh.topo.n_cells;
  }
  return cfg.mesh.nr;
}

double clamp_value(const double x, const double lo, const double hi) {
  return std::min(std::max(x, lo), hi);
}

bool is_positive_finite(const double x) {
  return std::isfinite(x) && x > 0.0;
}

std::vector<double> copy_nodes_or_uniform(const core::State& state,
                                          const core::Config& cfg,
                                          const int n) {
  std::vector<double> r(static_cast<std::size_t>(n + 1), 0.0);
  if (state.x_r.size() == static_cast<std::size_t>(n + 1)) {
    state.x_r.copy_to_host(r);
    return r;
  }

  const double r_min = cfg.mesh.r_min;
  const double r_max = cfg.mesh.r_max;
  const double dr = (n > 0) ? (r_max - r_min) / static_cast<double>(n) : 0.0;
  for (int j = 0; j <= n; ++j) {
    r[static_cast<std::size_t>(j)] = r_min + static_cast<double>(j) * dr;
  }
  return r;
}

std::vector<double> copy_mass_or_uniform(const core::State& state, const int n) {
  std::vector<double> mass(static_cast<std::size_t>(n), 1.0);
  if (state.mass.size() == static_cast<std::size_t>(n)) {
    state.mass.copy_to_host(mass);
    bool valid = true;
    for (double& m : mass) {
      if (!std::isfinite(m) || m < 0.0) {
        valid = false;
        break;
      }
    }
    const double total = std::accumulate(mass.begin(), mass.end(), 0.0);
    if (valid && total > 0.0) {
      return mass;
    }
  }
  std::fill(mass.begin(), mass.end(), 1.0);
  return mass;
}

MassMap build_mass_map(const core::State& state, const int n) {
  const std::vector<double> mass = copy_mass_or_uniform(state, n);
  const double total_mass = std::accumulate(mass.begin(), mass.end(), 0.0);
  MassMap map;
  map.node_x.assign(static_cast<std::size_t>(n + 1), 0.0);
  map.cell_x.assign(static_cast<std::size_t>(n), 0.0);
  map.dx.assign(static_cast<std::size_t>(n), 0.0);

  double prefix = 0.0;
  for (int i = 0; i < n; ++i) {
    const double dx = (total_mass > 0.0)
                          ? mass[static_cast<std::size_t>(i)] / total_mass
                          : 1.0 / static_cast<double>(n);
    map.dx[static_cast<std::size_t>(i)] = dx;
    map.node_x[static_cast<std::size_t>(i)] = prefix;
    map.cell_x[static_cast<std::size_t>(i)] = prefix + 0.5 * dx;
    prefix += dx;
  }
  map.node_x[static_cast<std::size_t>(n)] = 1.0;
  return map;
}

double cell_dr(const std::vector<double>& r, const int i) {
  const double outer = std::max(std::abs(r.back()), kTiny);
  return std::max(r[static_cast<std::size_t>(i + 1)] -
                      r[static_cast<std::size_t>(i)],
                  1.0e-14 * outer);
}

int nearest_face(const std::vector<double>& node_x, const double x_center) {
  const auto it = std::min_element(
      node_x.begin(), node_x.end(), [x_center](const double a, const double b) {
        return std::abs(a - x_center) < std::abs(b - x_center);
      });
  return static_cast<int>(it - node_x.begin());
}

NodeConstraintMask build_node_mask(const core::State& state,
                                   const core::Config& cfg,
                                   const std::vector<Ale1dFeature>& features,
                                   const MassMap& mass_map) {
  const int n = effective_cell_count(state, cfg);
  NodeConstraintMask mask;
  mask.pinned.assign(static_cast<std::size_t>(n + 1), false);
  if (n < 0) {
    return mask;
  }

  mask.pinned.front() = true;
  if (cfg.numerics.hydro.boundary_1d == "fixed") {
    mask.pinned.back() = true;
  }

  for (const Ale1dFeature& feature : features) {
    if (feature.kind != FeatureKind::MaterialInterface || !feature.pinned_face) {
      continue;
    }
    int face = feature.peak_cell_or_face;
    if (face < 0 || face > n) {
      face = nearest_face(mass_map.node_x, feature.x_center);
    }
    face = std::clamp(face, 0, n);
    mask.pinned[static_cast<std::size_t>(face)] = true;
  }

  mask.n_protected_nodes =
      static_cast<int>(std::count(mask.pinned.begin(), mask.pinned.end(), true));
  return mask;
}

bool is_spatial_feature(const FeatureKind kind) {
  return kind == FeatureKind::LaserAbsorption ||
         kind == FeatureKind::AblationFront || kind == FeatureKind::Shock;
}

double spatial_target_dr(const Ale1dRezoneConfig& cfg,
                         const Ale1dFeature& feature) {
  double lo = cfg.shock_spatial_dr_min_cm;
  double hi = cfg.shock_spatial_dr_max_cm;
  if (feature.kind == FeatureKind::LaserAbsorption) {
    lo = cfg.laser_spatial_dr_min_cm;
    hi = cfg.laser_spatial_dr_max_cm;
  } else if (feature.kind == FeatureKind::AblationFront) {
    lo = cfg.ablation_spatial_dr_min_cm;
    hi = cfg.ablation_spatial_dr_max_cm;
  }
  return clamp_value(feature.sigma_r / 3.0, lo, hi);
}

double radius_from_mass_coordinate(const std::vector<double>& node_x,
                                   const std::vector<double>& r,
                                   const double x) {
  if (x <= node_x.front()) {
    return r.front();
  }
  if (x >= node_x.back()) {
    return r.back();
  }

  const auto upper = std::upper_bound(node_x.begin(), node_x.end(), x);
  int i = static_cast<int>(upper - node_x.begin()) - 1;
  i = std::clamp(i, 0, static_cast<int>(node_x.size()) - 2);
  const double x0 = node_x[static_cast<std::size_t>(i)];
  const double x1 = node_x[static_cast<std::size_t>(i + 1)];
  if (!(x1 > x0)) {
    return r[static_cast<std::size_t>(i)];
  }
  const double t = (x - x0) / (x1 - x0);
  return (1.0 - t) * r[static_cast<std::size_t>(i)] +
         t * r[static_cast<std::size_t>(i + 1)];
}

double local_node_spacing(const std::vector<double>& x, const int j) {
  const int n = static_cast<int>(x.size()) - 1;
  double scale = std::numeric_limits<double>::infinity();
  if (j > 0) {
    scale = std::min(scale, std::abs(x[static_cast<std::size_t>(j)] -
                                    x[static_cast<std::size_t>(j - 1)]));
  }
  if (j < n) {
    scale = std::min(scale, std::abs(x[static_cast<std::size_t>(j + 1)] -
                                    x[static_cast<std::size_t>(j)]));
  }
  return std::isfinite(scale) && scale > 0.0 ? scale : 1.0;
}

void compute_displacement_diagnostics(const std::vector<double>& old_x,
                                      const std::vector<double>& old_r,
                                      const std::vector<double>& new_x,
                                      const std::vector<double>& new_r,
                                      double& max_mu,
                                      double& max_r) {
  max_mu = 0.0;
  max_r = 0.0;
  const int n = static_cast<int>(old_x.size()) - 1;
  for (int j = 0; j <= n; ++j) {
    const double mu_scale = local_node_spacing(old_x, j);
    const double r_scale = local_node_spacing(old_r, j);
    max_mu = std::max(max_mu, std::abs(new_x[static_cast<std::size_t>(j)] -
                                       old_x[static_cast<std::size_t>(j)]) /
                                  mu_scale);
    max_r = std::max(max_r, std::abs(new_r[static_cast<std::size_t>(j)] -
                                     old_r[static_cast<std::size_t>(j)]) /
                                r_scale);
  }
}

std::vector<int> segment_breakpoints(const NodeConstraintMask& mask) {
  const int n = static_cast<int>(mask.pinned.size()) - 1;
  std::vector<int> breaks;
  breaks.push_back(0);
  for (int j = 1; j < n; ++j) {
    if (mask.pinned[static_cast<std::size_t>(j)]) {
      breaks.push_back(j);
    }
  }
  breaks.push_back(n);
  return breaks;
}

bool candidate_is_strictly_ordered(const std::vector<double>& r) {
  for (std::size_t j = 1; j < r.size(); ++j) {
    if (!(r[j] > r[j - 1])) {
      return false;
    }
  }
  return true;
}

bool monitor_is_uniform(const std::vector<double>& W) {
  if (W.empty()) {
    return true;
  }
  const auto [lo, hi] = std::minmax_element(W.begin(), W.end());
  const double scale = std::max(std::abs(*hi), 1.0);
  return (*hi - *lo) <= 16.0 * std::numeric_limits<double>::epsilon() * scale;
}

}  // namespace

std::vector<double> build_monitor(const core::State& state,
                                  const core::Config& cfg,
                                  const std::vector<Ale1dFeature>& features) {
  const int n = effective_cell_count(state, cfg);
  if (n <= 0) {
    return {};
  }

  const auto& rezone_cfg = cfg.numerics.ale1d.rezone;
  const MassMap mass_map = build_mass_map(state, n);
  const std::vector<double> r = copy_nodes_or_uniform(state, cfg, n);
  const NodeConstraintMask mask = build_node_mask(state, cfg, features, mass_map);

  const double w0 = rezone_cfg.monitor_floor;
  std::vector<double> W(static_cast<std::size_t>(n), w0);
  std::vector<MonitorKernel> kernels;
  kernels.reserve(features.size());

  double active_budget = 0.0;
  for (const Ale1dFeature& feature : features) {
    if (!(feature.confidence > 0.0) || !(feature.target_cells > 0.0) ||
        !is_positive_finite(feature.sigma_x)) {
      continue;
    }

    MonitorKernel kernel;
    kernel.values.assign(static_cast<std::size_t>(n), 0.0);
    const double truncation =
        rezone_cfg.gaussian_truncation_sigma * feature.sigma_x;
    for (int i = 0; i < n; ++i) {
      const double dx = mass_map.cell_x[static_cast<std::size_t>(i)] -
                        feature.x_center;
      if (std::abs(dx) > truncation) {
        continue;
      }
      const double q = dx / feature.sigma_x;
      const double g = std::exp(-0.5 * q * q);
      kernel.values[static_cast<std::size_t>(i)] = g;
      kernel.integral += g * mass_map.dx[static_cast<std::size_t>(i)];
    }
    if (kernel.integral < kKernelIntegralFloor) {
      continue;
    }
    kernel.active_cells = feature.confidence * feature.target_cells;
    active_budget += kernel.active_cells;
    kernels.push_back(std::move(kernel));
  }

  std::vector<double> spatial_u(static_cast<std::size_t>(n), 0.0);
  double spatial_integral = 0.0;
  double spatial_confidence = 0.0;
  if (rezone_cfg.spatial_monitor_enabled &&
      rezone_cfg.spatial_target_cells_fraction > 0.0) {
    const double p = rezone_cfg.spatial_power;
    for (const Ale1dFeature& feature : features) {
      if (!is_spatial_feature(feature.kind) || !(feature.confidence > 0.0) ||
          !is_positive_finite(feature.sigma_x)) {
        continue;
      }
      const double sigma = 2.0 * feature.sigma_x;
      const double target_dr = spatial_target_dr(rezone_cfg, feature);
      spatial_confidence = std::max(spatial_confidence, feature.confidence);
      for (int i = 0; i < n; ++i) {
        const double dx = (mass_map.cell_x[static_cast<std::size_t>(i)] -
                           feature.x_center) /
                          sigma;
        const double g = std::exp(-0.5 * dx * dx);
        const double ratio = std::max(1.0, cell_dr(r, i) / target_dr);
        spatial_u[static_cast<std::size_t>(i)] +=
            feature.confidence * g * (std::pow(ratio, p) - 1.0);
      }
    }
    for (int i = 0; i < n; ++i) {
      spatial_integral += spatial_u[static_cast<std::size_t>(i)] *
                          mass_map.dx[static_cast<std::size_t>(i)];
    }
    if (spatial_integral >= kKernelIntegralFloor) {
      active_budget += rezone_cfg.spatial_target_cells_fraction *
                       static_cast<double>(n) * spatial_confidence;
    }
  }

  const double max_active_budget =
      (1.0 - rezone_cfg.min_floor_fraction) * static_cast<double>(n);
  const double alpha_budget =
      (active_budget > max_active_budget && active_budget > 0.0)
          ? max_active_budget / active_budget
          : 1.0;

  double active_after_cap = 0.0;
  for (const MonitorKernel& kernel : kernels) {
    active_after_cap += alpha_budget * kernel.active_cells;
  }
  const double spatial_cells =
      (spatial_integral >= kKernelIntegralFloor)
          ? alpha_budget * rezone_cfg.spatial_target_cells_fraction *
                static_cast<double>(n) * spatial_confidence
          : 0.0;
  active_after_cap += spatial_cells;
  const double n_floor =
      std::max(kTiny, static_cast<double>(n) - active_after_cap);

  for (const MonitorKernel& kernel : kernels) {
    const double n_star = alpha_budget * kernel.active_cells;
    const double amplitude = n_star / (n_floor * kernel.integral);
    for (int i = 0; i < n; ++i) {
      W[static_cast<std::size_t>(i)] +=
          amplitude * kernel.values[static_cast<std::size_t>(i)];
    }
  }
  if (spatial_cells > 0.0) {
    const double amplitude = spatial_cells / (n_floor * spatial_integral);
    for (int i = 0; i < n; ++i) {
      W[static_cast<std::size_t>(i)] +=
          amplitude * spatial_u[static_cast<std::size_t>(i)];
    }
  }

  for (int iter = 0; iter < rezone_cfg.monitor_smoothing_iterations; ++iter) {
    std::vector<double> next = W;
    for (int i = 0; i < n; ++i) {
      double numerator = 2.0 * W[static_cast<std::size_t>(i)];
      double denominator = 2.0;
      if (i > 0 &&
          (rezone_cfg.monitor_smooth_across_protected_faces ||
           !mask.pinned[static_cast<std::size_t>(i)])) {
        numerator += W[static_cast<std::size_t>(i - 1)];
        denominator += 1.0;
      }
      if (i + 1 < n &&
          (rezone_cfg.monitor_smooth_across_protected_faces ||
           !mask.pinned[static_cast<std::size_t>(i + 1)])) {
        numerator += W[static_cast<std::size_t>(i + 1)];
        denominator += 1.0;
      }
      next[static_cast<std::size_t>(i)] = numerator / denominator;
    }
    W = std::move(next);
  }

  const double w_max = w0 * rezone_cfg.monitor_wmax_ratio;
  for (double& w : W) {
    w = clamp_value(w, w0, w_max);
  }
  return W;
}

Ale1dRezoneResult rezone(const core::State& state,
                         const core::Config& cfg,
                         const std::vector<Ale1dFeature>& features,
                         const std::vector<double>& W_smooth_clipped) {
  Ale1dRezoneResult out;
  const int n = effective_cell_count(state, cfg);
  if (n <= 0) {
    out.skip_reason = Ale1dSkipReason::NTooSmall;
    return out;
  }

  const MassMap mass_map = build_mass_map(state, n);
  const std::vector<double> r_old = copy_nodes_or_uniform(state, cfg, n);
  out.r_candidate = r_old;
  out.node_mask = build_node_mask(state, cfg, features, mass_map);
  out.protected_fraction =
      static_cast<double>(out.node_mask.n_protected_nodes) /
      static_cast<double>(n + 1);

  if (out.protected_fraction > cfg.numerics.ale1d.protected_fraction_max) {
    out.skip_reason = Ale1dSkipReason::ProtectedFractionTooHigh;
    return out;
  }
  if (W_smooth_clipped.size() != static_cast<std::size_t>(n)) {
    out.skip_reason = Ale1dSkipReason::CandidateInvalid;
    return out;
  }
  for (const double w : W_smooth_clipped) {
    if (!is_positive_finite(w)) {
      out.skip_reason = Ale1dSkipReason::CandidateInvalid;
      return out;
    }
  }

  const std::vector<int> breaks = segment_breakpoints(out.node_mask);
  out.min_movable_segment_size = n;
  for (std::size_t b = 1; b < breaks.size(); ++b) {
    const int segment_cells = breaks[b] - breaks[b - 1];
    out.min_movable_segment_size =
        std::min(out.min_movable_segment_size, segment_cells);
  }
  if (out.min_movable_segment_size < cfg.numerics.ale1d.min_movable_segment_hard) {
    out.skip_reason = Ale1dSkipReason::MovableSegmentTooSmall;
    return out;
  }
  if (n < cfg.numerics.ale1d.min_cells) {
    out.skip_reason = Ale1dSkipReason::NTooSmall;
    return out;
  }
  if (monitor_is_uniform(W_smooth_clipped)) {
    if (!candidate_is_strictly_ordered(out.r_candidate)) {
      out.skip_reason = Ale1dSkipReason::CandidateInvalid;
      return out;
    }
    out.success = true;
    out.skip_reason = Ale1dSkipReason::None;
    return out;
  }

  std::vector<double> x_new = mass_map.node_x;
  for (std::size_t b = 1; b < breaks.size(); ++b) {
    const int j_lo = breaks[b - 1];
    const int j_hi = breaks[b];
    const int segment_cells = j_hi - j_lo;
    if (segment_cells <= 0) {
      continue;
    }

    double segment_integral = 0.0;
    for (int i = j_lo; i < j_hi; ++i) {
      segment_integral += W_smooth_clipped[static_cast<std::size_t>(i)] *
                          mass_map.dx[static_cast<std::size_t>(i)];
    }
    if (!(segment_integral > 0.0)) {
      out.skip_reason = Ale1dSkipReason::CandidateInvalid;
      return out;
    }

    int cell = j_lo;
    double cumulative = 0.0;
    for (int j = j_lo + 1; j < j_hi; ++j) {
      const double target =
          static_cast<double>(j - j_lo) * segment_integral /
          static_cast<double>(segment_cells);
      while (cell < j_hi - 1) {
        const double cell_integral =
            W_smooth_clipped[static_cast<std::size_t>(cell)] *
            mass_map.dx[static_cast<std::size_t>(cell)];
        if (cumulative + cell_integral >= target) {
          break;
        }
        cumulative += cell_integral;
        ++cell;
      }
      const double w = W_smooth_clipped[static_cast<std::size_t>(cell)];
      const double local_dx =
          clamp_value((target - cumulative) / w, 0.0,
                      mass_map.dx[static_cast<std::size_t>(cell)]);
      x_new[static_cast<std::size_t>(j)] =
          mass_map.node_x[static_cast<std::size_t>(cell)] + local_dx;
    }
  }

  for (int j = 0; j <= n; ++j) {
    out.r_candidate[static_cast<std::size_t>(j)] =
        radius_from_mass_coordinate(mass_map.node_x, r_old,
                                    x_new[static_cast<std::size_t>(j)]);
  }

  compute_displacement_diagnostics(mass_map.node_x, r_old, x_new, out.r_candidate,
                                   out.max_node_displacement_mu,
                                   out.max_node_displacement_r);

  double displacement_scale = 1.0;
  if (out.max_node_displacement_mu >
      cfg.numerics.ale1d.max_node_displacement_fraction_mu) {
    displacement_scale = std::min(
        displacement_scale,
        cfg.numerics.ale1d.max_node_displacement_fraction_mu /
            out.max_node_displacement_mu);
  }
  if (out.max_node_displacement_r >
      cfg.numerics.ale1d.max_node_displacement_fraction_r) {
    displacement_scale = std::min(
        displacement_scale,
        cfg.numerics.ale1d.max_node_displacement_fraction_r /
            out.max_node_displacement_r);
  }

  if (displacement_scale < 1.0) {
    for (int j = 0; j <= n; ++j) {
      x_new[static_cast<std::size_t>(j)] =
          mass_map.node_x[static_cast<std::size_t>(j)] +
          displacement_scale *
              (x_new[static_cast<std::size_t>(j)] -
               mass_map.node_x[static_cast<std::size_t>(j)]);
      out.r_candidate[static_cast<std::size_t>(j)] =
          radius_from_mass_coordinate(mass_map.node_x, r_old,
                                      x_new[static_cast<std::size_t>(j)]);
    }
    compute_displacement_diagnostics(mass_map.node_x, r_old, x_new,
                                     out.r_candidate,
                                     out.max_node_displacement_mu,
                                     out.max_node_displacement_r);
  }

  if (!candidate_is_strictly_ordered(out.r_candidate)) {
    out.skip_reason = Ale1dSkipReason::CandidateInvalid;
    out.success = false;
    return out;
  }

  out.success = true;
  out.skip_reason = Ale1dSkipReason::None;
  return out;
}

Ale1dRezoneResult rezone(const core::State& state,
                         const core::Config& cfg,
                         const std::vector<Ale1dFeature>& features) {
  return rezone(state, cfg, features, build_monitor(state, cfg, features));
}

}  // namespace tenryu::hydro::ale1d
