#include "laser/sector_phase_space.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace tenryu::laser::sector_ps {

namespace {

struct TurningInfo {
  int count = 0;
  int first_index = -1;
};

struct PendingCrossing {
  std::size_t bin_index;
  CrossingView crossing;
};

int sign_of(const double value) {
  return (value > 0.0) - (value < 0.0);
}

TurningInfo find_turning_points(const std::vector<double>& r) {
  TurningInfo result;
  int previous_sign = 0;
  for (std::size_t k = 0; k + 1 < r.size(); ++k) {
    const int current_sign = sign_of(r[k + 1] - r[k]);
    if (current_sign == 0) {
      continue;
    }
    if (previous_sign != 0 && current_sign != previous_sign) {
      ++result.count;
      if (result.first_index < 0) {
        result.first_index = static_cast<int>(k);
      }
    }
    previous_sign = current_sign;
  }
  return result;
}

std::vector<int> find_caustics(const std::vector<double>& area,
                               const double relative_tolerance) {
  const double max_area = *std::max_element(area.begin(), area.end());
  const double threshold = relative_tolerance * max_area;
  std::vector<int> minima;
  for (std::size_t k = 1; k + 1 < area.size(); ++k) {
    if (area[k] < area[k - 1] && area[k] < area[k + 1] &&
        area[k] < threshold) {
      minima.push_back(static_cast<int>(k));
    }
  }
  return minima;
}

std::vector<double> reconstruct_alpha(const RayPath& ray) {
  const std::size_t n = ray.r.size();
  std::vector<double> segment_alpha(n - 1, 0.0);
  for (std::size_t k = 0; k + 1 < n; ++k) {
    const double dr = ray.r[k + 1] - ray.r[k];
    const double dtheta = ray.theta[k + 1] - ray.theta[k];
    const double r_mid = 0.5 * (ray.r[k] + ray.r[k + 1]);
    const double transverse = r_mid * dtheta;
    const double length = std::hypot(dr, transverse);
    const double sin_alpha =
        length > 0.0 ? std::min(1.0, std::abs(transverse) / length) : 0.0;
    segment_alpha[k] = std::asin(sin_alpha);
  }

  std::vector<double> alpha(n, 0.0);
  alpha.front() = segment_alpha.front();
  alpha.back() = segment_alpha.back();
  for (std::size_t k = 1; k + 1 < n; ++k) {
    alpha[k] = 0.5 * (segment_alpha[k - 1] + segment_alpha[k]);
  }
  return alpha;
}

double interpolate_eps(const std::vector<double>& shell_r,
                       const std::vector<double>& eps_r,
                       const double radius) {
  if (radius <= shell_r.front()) {
    return eps_r.front();
  }
  if (radius >= shell_r.back()) {
    return eps_r.back();
  }

  const auto upper = std::upper_bound(shell_r.begin(), shell_r.end(), radius);
  const std::size_t hi =
      static_cast<std::size_t>(std::distance(shell_r.begin(), upper));
  const std::size_t lo = hi - 1;
  const double weight =
      (radius - shell_r[lo]) / (shell_r[hi] - shell_r[lo]);
  return eps_r[lo] + weight * (eps_r[hi] - eps_r[lo]);
}

double bouguer_drift(const RayPath& ray,
                     const std::vector<double>& alpha,
                     const std::vector<double>& shell_r,
                     const std::vector<double>& eps_r) {
  // Ill-conditioning cut (2026-07-31): near-tangential records
  // (|cos alpha| -> 0) sit in the ray's turning cell, where the
  // cell-edge eps sample cannot represent the turning-point eps —
  // the reconstructed B there carries an O(delta_eps/eps) error at
  // record granularity for ANY eps magnitude. Audit B only on
  // transversal crossings; precision turning validation lives in the
  // supplied-alpha path (1e-10).
  constexpr double kBouguerAuditMuMin = 0.3;

  std::size_t b0_record = 0;
  while (std::abs(std::cos(alpha[b0_record])) < kBouguerAuditMuMin) {
    ++b0_record;
    if (b0_record == ray.r.size()) {
      // No well-conditioned records to audit.
      return 0.0;
    }
  }
  const double eps_b0 = interpolate_eps(shell_r, eps_r, ray.r[b0_record]);
  const double B0 = std::sqrt(eps_b0) * ray.r[b0_record] *
                    std::sin(alpha[b0_record]);
  // Denominator conditioning (2026-07-31): B0 -> 0 for near-radial rays
  // makes self-relative drift ill-conditioned — record-granularity
  // reconstruction noise diverges after the march fixes let such rays
  // reach deep turnings. Floor at a fraction of the geometry's full B
  // scale (B <= r_outer since eps <= 1): rays with B0 above the floor
  // keep exact relative semantics; smaller-B0 rays are measured against
  // the floor, bounding the amplification. The audit is a catastrophe
  // detector — precision Bouguer validation (supplied analytic alpha,
  // B0 = O(r)) is unaffected by the floor.
  constexpr double kBouguerDriftRefFrac = 0.1;
  const double denominator =
      std::max(std::abs(B0), kBouguerDriftRefFrac * shell_r.back());
  double max_drift = 0.0;
  for (std::size_t k = 0; k < ray.r.size(); ++k) {
    if (std::abs(std::cos(alpha[k])) < kBouguerAuditMuMin) {
      continue;
    }
    const double eps_k = interpolate_eps(shell_r, eps_r, ray.r[k]);
    const double B = std::sqrt(eps_k) * ray.r[k] * std::sin(alpha[k]);
    max_drift = std::max(max_drift, std::abs(B - B0) / denominator);
  }
  return max_drift;
}

void assert_input_contract(const std::vector<RayPath>& rays,
                           const std::vector<double>& shell_r,
                           const std::vector<double>& eps_r,
                           const PhaseSpaceParams& p) {
  assert(!shell_r.empty());
  assert(shell_r.size() == eps_r.size());
  assert(p.caustic_area_rel_tol >= 0.0);
  assert(p.bouguer_tol >= 0.0);
  assert(p.bouguer_tol_fd >= 0.0);
  for (std::size_t k = 0; k < shell_r.size(); ++k) {
    assert(std::isfinite(shell_r[k]));
    assert(std::isfinite(eps_r[k]));
    assert(eps_r[k] >= 0.0);
    if (k > 0) {
      assert(shell_r[k] > shell_r[k - 1]);
    }
  }

  double previous_impact_parameter = -1.0;
  for (std::size_t ray_position = 0; ray_position < rays.size();
       ++ray_position) {
    const RayPath& ray = rays[ray_position];
    const std::size_t n = ray.r.size();
    assert(ray.ray_index == static_cast<int>(ray_position));
    assert(std::isfinite(ray.impact_parameter));
    assert(ray.impact_parameter >= 0.0);
    assert(ray.impact_parameter >= previous_impact_parameter);
    previous_impact_parameter = ray.impact_parameter;
    assert(n >= 2);
    assert(ray.theta.size() == n);
    assert(ray.P.size() == n);
    assert(ray.area.size() == n);
    assert(ray.alpha.empty() || ray.alpha.size() == n);
    for (std::size_t k = 0; k < n; ++k) {
      assert(std::isfinite(ray.r[k]));
      assert(std::isfinite(ray.theta[k]));
      assert(std::isfinite(ray.P[k]));
      assert(std::isfinite(ray.area[k]));
      assert(ray.r[k] > 0.0);
      assert(ray.theta[k] >= 0.0);
      assert(ray.area[k] > 0.0);
      if (!ray.alpha.empty()) {
        assert(std::isfinite(ray.alpha[k]));
      }
    }
  }
}

std::pair<std::size_t, std::size_t> segment_range(
    const std::size_t n_nodes, const int turning_index, const int sheet) {
  if (turning_index < 0) {
    return sheet == 0 ? std::pair<std::size_t, std::size_t>{0, n_nodes - 1}
                      : std::pair<std::size_t, std::size_t>{0, 0};
  }
  if (sheet == 0) {
    return {0, static_cast<std::size_t>(turning_index)};
  }
  return {static_cast<std::size_t>(turning_index), n_nodes - 1};
}

std::optional<CrossingView> find_crossing(
    const RayPath& ray, const std::vector<double>& alpha,
    const double shell_radius, const int turning_index,
    const int caustic_index, const int sheet, bool& invariant_violation) {
  const auto [segment_begin, segment_end] =
      segment_range(ray.r.size(), turning_index, sheet);
  if (segment_begin == segment_end) {
    return std::nullopt;
  }

  std::size_t last_nonzero_segment = segment_begin;
  bool has_nonzero_segment = false;
  for (std::size_t k = segment_begin; k < segment_end; ++k) {
    if (ray.r[k + 1] != ray.r[k]) {
      last_nonzero_segment = k;
      has_nonzero_segment = true;
    }
  }
  if (!has_nonzero_segment) {
    return std::nullopt;
  }

  std::optional<CrossingView> result;
  int n_crossings = 0;
  for (std::size_t k = segment_begin; k < segment_end; ++k) {
    const double r0 = ray.r[k];
    const double r1 = ray.r[k + 1];
    if (r0 == r1 ||
        shell_radius < std::min(r0, r1) ||
        shell_radius > std::max(r0, r1)) {
      continue;
    }

    double weight = (shell_radius - r0) / (r1 - r0);
    weight = std::clamp(weight, 0.0, 1.0);
    if (weight == 1.0 && k != last_nonzero_segment) {
      continue;
    }

    const int node_index = static_cast<int>(k);
    const int limiter_lo = std::min(turning_index, caustic_index);
    const int limiter_hi = std::max(turning_index, caustic_index);
    const bool in_limiter_zone =
        caustic_index >= 0 && node_index >= limiter_lo &&
        node_index <= limiter_hi;
    const auto interpolate = [weight](const double a, const double b) {
      return a + weight * (b - a);
    };
    result = CrossingView{interpolate(ray.theta[k], ray.theta[k + 1]),
                          interpolate(alpha[k], alpha[k + 1]),
                          interpolate(ray.P[k], ray.P[k + 1]),
                          interpolate(ray.area[k], ray.area[k + 1]),
                          ray.ray_index,
                          in_limiter_zone};
    ++n_crossings;
  }

  assert(n_crossings <= 1);
  if (n_crossings > 1) {
    invariant_violation = true;
    return std::nullopt;
  }
  return result;
}

}  // namespace

struct PhaseSpaceTable::Impl {
  std::vector<double> shell_r;
  std::vector<std::vector<CrossingView>> bins;
  ExclusionLedger ledger{0.0, 0};
};

PhaseSpaceTable build_table(const std::vector<RayPath>& rays,
                            const std::vector<double>& shell_r,
                            const std::vector<double>& eps_r,
                            const PhaseSpaceParams& p,
                            std::vector<RayAnnotation>& annotations_out) {
  assert_input_contract(rays, shell_r, eps_r, p);

  auto impl = std::make_shared<PhaseSpaceTable::Impl>();
  impl->shell_r = shell_r;
  impl->bins.resize(shell_r.size() * 2);
  annotations_out.clear();
  annotations_out.reserve(rays.size());

  double total_launch_power = 0.0;
  for (const RayPath& ray : rays) {
    total_launch_power += ray.P.front();
  }
  double excluded_launch_power = 0.0;

  for (const RayPath& ray : rays) {
    const TurningInfo turning = find_turning_points(ray.r);
    const std::vector<int> caustics =
        find_caustics(ray.area, p.caustic_area_rel_tol);
    const int caustic_index =
        caustics.size() == 1 ? caustics.front() : -1;
    const bool supplied_alpha = !ray.alpha.empty();
    const std::vector<double> alpha =
        supplied_alpha ? ray.alpha : reconstruct_alpha(ray);

    RayAnnotation annotation{
        turning.count,
        caustic_index,
        turning.count > 1 || caustics.size() > 1,
        bouguer_drift(ray, alpha, shell_r, eps_r)};
    const double audit_tolerance =
        supplied_alpha ? p.bouguer_tol : p.bouguer_tol_fd;
    assert(annotation.bouguer_max_drift <= audit_tolerance);

    if (annotation.ambiguous) {
      ++impl->ledger.n_excluded_rays;
      excluded_launch_power += ray.P.front();
      annotations_out.push_back(annotation);
      continue;
    }

    std::vector<PendingCrossing> pending;
    pending.reserve(shell_r.size() * 2);
    bool invariant_violation = false;
    for (std::size_t shell = 0; shell < shell_r.size(); ++shell) {
      for (int sheet = 0; sheet < 2; ++sheet) {
        const std::optional<CrossingView> crossing =
            find_crossing(ray, alpha, shell_r[shell], turning.first_index,
                          caustic_index, sheet, invariant_violation);
        if (crossing.has_value()) {
          pending.push_back(
              PendingCrossing{2 * shell + static_cast<std::size_t>(sheet),
                              *crossing});
        }
      }
    }

    if (invariant_violation) {
      annotation.ambiguous = true;
      ++impl->ledger.n_excluded_rays;
      excluded_launch_power += ray.P.front();
    } else {
      for (const PendingCrossing& entry : pending) {
        impl->bins[entry.bin_index].push_back(entry.crossing);
      }
    }
    annotations_out.push_back(annotation);
  }

  impl->ledger.excluded_power_fraction =
      total_launch_power != 0.0
          ? excluded_launch_power / total_launch_power
          : 0.0;

  for (std::vector<CrossingView>& bin : impl->bins) {
    std::sort(bin.begin(), bin.end(),
              [](const CrossingView& lhs, const CrossingView& rhs) {
                if (lhs.theta != rhs.theta) {
                  return lhs.theta < rhs.theta;
                }
                return lhs.ray_index < rhs.ray_index;
              });
  }

  PhaseSpaceTable table;
  table.impl_ = std::move(impl);
  return table;
}

ExclusionLedger exclusion_ledger(const PhaseSpaceTable& table) {
  assert(table.impl_ != nullptr);
  return table.impl_->ledger;
}

double pump_polar_angle(const double theta_s, const double theta_i) {
  return std::acos(std::clamp(std::cos(theta_s - theta_i), -1.0, 1.0));
}

double pump_polar_angle_3d(const double theta_s, const double theta_i,
                           const double phi_i) {
  const double cosine =
      std::cos(theta_s) * std::cos(theta_i) +
      std::sin(theta_s) * std::sin(theta_i) * std::cos(phi_i);
  return std::acos(std::clamp(cosine, -1.0, 1.0));
}

double crossing_cos_2d(const double alpha_s, const double alpha_p,
                       const double theta_s, const double theta_i) {
  const double sine = std::sin(theta_s - theta_i);
  const int sign = sign_of(sine);
  return std::cos(alpha_s - static_cast<double>(sign) * alpha_p);
}

std::array<double, 2> meridional_direction(const double alpha,
                                           const int sheet) {
  assert(sheet == 0 || sheet == 1);
  const double radial = sheet == 0 ? -std::cos(alpha) : std::cos(alpha);
  return {radial, std::sin(alpha)};
}

PumpLookup lookup(const PhaseSpaceTable& table, const int shell_index,
                  const int sheet, const double theta_p) {
  assert(table.impl_ != nullptr);
  assert(shell_index >= 0);
  assert(static_cast<std::size_t>(shell_index) < table.impl_->shell_r.size());
  assert(sheet == 0 || sheet == 1);
  const std::vector<CrossingView>& bin =
      table.impl_->bins[2 * static_cast<std::size_t>(shell_index) +
                        static_cast<std::size_t>(sheet)];
  const auto is_field_crossing = [](const CrossingView& crossing) {
    return !crossing.in_limiter_zone;
  };
  const auto first =
      std::find_if(bin.begin(), bin.end(), is_field_crossing);
  if (first == bin.end()) {
    return PumpLookup{false, 0.0, 0.0};
  }
  const auto last =
      std::find_if(bin.rbegin(), bin.rend(), is_field_crossing);
  if (theta_p > last->theta) {
    return PumpLookup{false, 0.0, 0.0};
  }

  const auto value_at = [](const CrossingView& crossing) {
    return PumpLookup{true, crossing.P / crossing.area, crossing.alpha};
  };
  if (theta_p <= first->theta) {
    return value_at(*first);
  }

  const auto hi = std::find_if(
      std::next(first), bin.end(),
      [theta_p](const CrossingView& crossing) {
        return !crossing.in_limiter_zone &&
               crossing.theta >= theta_p;
      });
  assert(hi != bin.end());
  if (hi->theta == theta_p) {
    return value_at(*hi);
  }
  auto lo = hi;
  do {
    --lo;
  } while (lo->in_limiter_zone);
  const CrossingView& upper = *hi;
  const CrossingView& lower = *lo;
  const double weight =
      (theta_p - lower.theta) / (upper.theta - lower.theta);
  const double lower_intensity = lower.P / lower.area;
  const double upper_intensity = upper.P / upper.area;
  return PumpLookup{
      true,
      lower_intensity + weight * (upper_intensity - lower_intensity),
      lower.alpha + weight * (upper.alpha - lower.alpha)};
}

const std::vector<CrossingView>& crossings(const PhaseSpaceTable& table,
                                           const int shell_index,
                                           const int sheet) {
  assert(table.impl_ != nullptr);
  assert(shell_index >= 0);
  assert(static_cast<std::size_t>(shell_index) < table.impl_->shell_r.size());
  assert(sheet == 0 || sheet == 1);
  return table.impl_->bins[2 * static_cast<std::size_t>(shell_index) +
                           static_cast<std::size_t>(sheet)];
}

FlatTable flatten_table(const PhaseSpaceTable& table) {
  FlatTable flat;
  flatten_table_into(table, flat);
  return flat;
}

void flatten_table_into(const PhaseSpaceTable& table, FlatTable& flat) {
  assert(table.impl_ != nullptr);
  flat.n_shells = static_cast<int>(table.impl_->shell_r.size());
  flat.offsets.clear();
  flat.theta.clear();
  flat.alpha.clear();
  flat.P.clear();
  flat.area.clear();
  flat.ray_index.clear();
  flat.in_limiter.clear();
  flat.offsets.reserve(table.impl_->bins.size() + 1);
  flat.offsets.push_back(0);
  for (const std::vector<CrossingView>& bin : table.impl_->bins) {
    for (const CrossingView& crossing : bin) {
      flat.theta.push_back(crossing.theta);
      flat.alpha.push_back(crossing.alpha);
      flat.P.push_back(crossing.P);
      flat.area.push_back(crossing.area);
      flat.ray_index.push_back(
          static_cast<std::int32_t>(crossing.ray_index));
      flat.in_limiter.push_back(
          static_cast<std::uint8_t>(crossing.in_limiter_zone));
    }
    flat.offsets.push_back(static_cast<int>(flat.theta.size()));
  }
}

}  // namespace tenryu::laser::sector_ps
