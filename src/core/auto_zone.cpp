#include "core/auto_zone.hpp"
#include "mesh/geometry_1d.cuh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace tenryu::core {
namespace {

constexpr double kFourPiOver3 = 4.188790204786390984616857844372670512262892532500141;
constexpr double kTiny = 1.0e-300;

struct RegionState {
  double r_in = 0.0;
  double r_out = 0.0;
  int nz = 0;
  double rho = 0.0;
  bool is_void = false;
  std::string material_group;
  double total_mass = 0.0;
  double avg_dr = 0.0;
};

struct InterfaceCandidate {
  bool valid = false;
  int n_left = 0;
  int n_right = 0;
  double q = 1.0;
  double m_int = 0.0;
  double iface_mass_left = 0.0;
  double iface_mass_right = 0.0;
  bool two_sided = false;
  int symmetry = 0;
  double dr_margin = 0.0;
};

double shell_volume(const double r0, const double r1, const int geom) {
  return (geom == 0)
             ? (kFourPiOver3 * (r1 * r1 * r1 - r0 * r0 * r0))
             : tenryu::mesh::geometry_1d_shell_volume_cubes(geom, r0, r1);
}

double interface_mass_floor_left(const double r_int,
                                 const double rho,
                                 const double dr_min,
                                 const int geom) {
  if (!(rho > 0.0) || !(dr_min > 0.0)) {
    return 0.0;
  }
  const double r_inner = std::max(0.0, r_int - dr_min);
  return rho * shell_volume(r_inner, r_int, geom);
}

double interface_mass_floor_right(const double r_int,
                                  const double rho,
                                  const double dr_min,
                                  const int geom) {
  if (!(rho > 0.0) || !(dr_min > 0.0)) {
    return 0.0;
  }
  return rho * shell_volume(r_int, r_int + dr_min, geom);
}

int bridge_cap(const int nz, const AutoZoneConfig& cfg) {
  const int frac_cap = static_cast<int>(std::floor(cfg.bridge_frac_max * static_cast<double>(nz)));
  return std::max(0, std::min(cfg.n_bridge_max, frac_cap));
}

void trim_bridge_for_overlap(const int nz,
                             std::vector<double>& c_left,
                             std::vector<double>& c_right) {
  const int min_bulk = (nz <= 6) ? 1 : 2;
  const int max_bridge = std::max(0, nz - min_bulk);
  while (static_cast<int>(c_left.size() + c_right.size()) > max_bridge) {
    if (c_left.size() > c_right.size()) {
      c_left.erase(c_left.begin());
    } else if (!c_right.empty()) {
      c_right.erase(c_right.begin());
    } else if (!c_left.empty()) {
      c_left.erase(c_left.begin());
    } else {
      break;
    }
  }
}

std::vector<double> build_region_zone_masses(const int nz,
                                             const double m_bulk,
                                             const std::vector<double>& c_left,
                                             const std::vector<double>& c_right) {
  std::vector<double> masses(static_cast<std::size_t>(nz), m_bulk);
  for (std::size_t i = 0; i < c_left.size(); ++i) {
    masses[i] = m_bulk * c_left[c_left.size() - 1U - i];
  }
  for (std::size_t i = 0; i < c_right.size(); ++i) {
    masses[static_cast<std::size_t>(nz - 1) - i] =
        m_bulk * c_right[c_right.size() - 1U - i];
  }
  return masses;
}

bool better_candidate(const InterfaceCandidate& a, const InterfaceCandidate& b) {
  if (!b.valid) {
    return true;
  }
  if (a.two_sided != b.two_sided) {
    return a.two_sided;
  }
  if (a.symmetry != b.symmetry) {
    return a.symmetry > b.symmetry;
  }
  if (std::abs(a.dr_margin - b.dr_margin) > 1.0e-14) {
    return a.dr_margin > b.dr_margin;
  }
  return false;
}

std::vector<RegionState> build_regions(const double r_min,
                                       const std::vector<AutoZoneRegion>& regions,
                                       const AutoZoneConfig& cfg) {
  if (!(r_min >= 0.0) || !std::isfinite(r_min)) {
    throw std::runtime_error("auto-zone requires finite r_min >= 0");
  }
  if (regions.empty()) {
    throw std::runtime_error("auto-zone requires at least one region");
  }

  std::vector<RegionState> out;
  out.reserve(regions.size());
  double r_in = r_min;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& src = regions[i];
    if (!std::isfinite(src.r_end) || !(src.r_end > r_in)) {
      throw std::runtime_error("auto-zone region[" + std::to_string(i) +
                               "] requires finite r_end > previous boundary");
    }
    if (src.nz <= 0) {
      throw std::runtime_error("auto-zone region[" + std::to_string(i) + "] requires nz > 0");
    }
    if (!std::isfinite(src.rho_ref) || src.rho_ref < 0.0) {
      throw std::runtime_error("auto-zone region[" + std::to_string(i) +
                               "] requires finite rho_ref >= 0");
    }

    RegionState dst;
    dst.r_in = r_in;
    dst.r_out = src.r_end;
    dst.nz = src.nz;
    dst.rho = src.rho_ref;
    dst.is_void = src.is_void || (src.rho_ref <= cfg.rho_void_cut);
    dst.material_group = src.material_group;
    dst.total_mass = dst.rho * shell_volume(dst.r_in, dst.r_out, cfg.geometry_code);
    dst.avg_dr = (dst.r_out - dst.r_in) / static_cast<double>(dst.nz);

    if (!dst.is_void && !(dst.rho > 0.0)) {
      throw std::runtime_error("auto-zone non-void region requires rho_ref > 0");
    }

    out.push_back(std::move(dst));
    r_in = src.r_end;
  }
  return out;
}

std::vector<double> alpha_sweep(double alpha_start, double alpha_hard) {
  std::vector<double> values;
  alpha_start = std::max(alpha_start, 1.0 + 1.0e-12);
  alpha_hard = std::max(alpha_hard, alpha_start);
  values.push_back(alpha_start);
  if (alpha_hard <= alpha_start + 1.0e-14) {
    return values;
  }
  double alpha = alpha_start;
  for (int k = 0; k < 32 && alpha < alpha_hard - 1.0e-14; ++k) {
    alpha = std::min(alpha_hard, alpha * 1.08);
    if (alpha > values.back() + 1.0e-14) {
      values.push_back(alpha);
    }
  }
  if (values.back() < alpha_hard - 1.0e-14) {
    values.push_back(alpha_hard);
  }
  return values;
}

void push_warning_once(std::vector<std::string>& warnings, const std::string& text) {
  if (std::find(warnings.begin(), warnings.end(), text) == warnings.end()) {
    warnings.push_back(text);
  }
}

int compute_base_side_count(const int n_required,
                            const int cap_this,
                            const int cap_other) {
  int n_this = std::clamp(n_required / 2, 0, cap_this);
  int n_other = n_required - n_this;
  if (n_other > cap_other) {
    n_other = cap_other;
    n_this = std::min(cap_this, n_required - n_other);
  }
  return n_this;
}

InterfaceCandidate choose_violation_candidate(const double mass_ratio,
                                              const double m_left,
                                              const double m_right,
                                              const double floor_left,
                                              const double floor_right,
                                              const int cap_left,
                                              const int cap_right) {
  InterfaceCandidate best;
  double best_q = std::numeric_limits<double>::infinity();

  const int cap_sum = cap_left + cap_right;
  for (int n_total = 0; n_total <= cap_sum; ++n_total) {
    const int n_left_min = std::max(0, n_total - cap_right);
    const int n_left_max = std::min(cap_left, n_total);
    for (int n_left = n_left_min; n_left <= n_left_max; ++n_left) {
      const int n_right = n_total - n_left;

      const double q = (n_total > 0)
                           ? std::pow(mass_ratio, 1.0 / static_cast<double>(n_total))
                           : mass_ratio;
      const double m_int =
          (n_total > 0)
              ? std::pow(m_left, static_cast<double>(n_right) / static_cast<double>(n_total)) *
                    std::pow(m_right, static_cast<double>(n_left) / static_cast<double>(n_total))
              : std::sqrt(m_left * m_right);

      const double iface_mass_left = (n_left > 0) ? m_int : m_left;
      const double iface_mass_right = (n_right > 0) ? m_int : m_right;
      if (iface_mass_left + 1.0e-14 < floor_left ||
          iface_mass_right + 1.0e-14 < floor_right) {
        continue;
      }

      InterfaceCandidate cand;
      cand.valid = true;
      cand.n_left = n_left;
      cand.n_right = n_right;
      cand.q = q;
      cand.m_int = m_int;
      cand.iface_mass_left = iface_mass_left;
      cand.iface_mass_right = iface_mass_right;
      cand.two_sided = (n_left > 0 && n_right > 0);
      cand.symmetry = -std::abs(n_left - n_right);
      const double margin_left = floor_left > 0.0 ? (iface_mass_left / floor_left)
                                                  : std::numeric_limits<double>::infinity();
      const double margin_right = floor_right > 0.0 ? (iface_mass_right / floor_right)
                                                     : std::numeric_limits<double>::infinity();
      cand.dr_margin = std::min(margin_left, margin_right);

      if (!best.valid || q < best_q - 1.0e-14 ||
          (std::abs(q - best_q) <= 1.0e-14 && better_candidate(cand, best))) {
        best = cand;
        best_q = q;
      }
    }
  }
  return best;
}

double coeff_sum_increasing(const double q, const int n) {
  if (n <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  double qk = 1.0;
  for (int k = 0; k < n; ++k) {
    qk *= q;
    sum += qk;
  }
  return sum;
}

double coeff_sum_decreasing(const double q, const int n) {
  if (n <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  double qk = 1.0;
  for (int k = 0; k < n; ++k) {
    qk /= q;
    sum += qk;
  }
  return sum;
}

double interface_g(const double q,
                   const int n_left,
                   const int n_right,
                   const double m_left,
                   const double m_right,
                   const int n_bulk_left,
                   const int n_bulk_right,
                   const double other_sum_left,
                   const double other_sum_right) {
  const int n_total = n_left + n_right;
  const double s_left = static_cast<double>(n_bulk_left) +
                        coeff_sum_increasing(q, n_left) + other_sum_left;
  const double s_right = static_cast<double>(n_bulk_right) +
                         coeff_sum_decreasing(q, n_right) + other_sum_right;
  const double q_pow = std::pow(q, static_cast<double>(n_total));
  return q_pow * m_left * s_right - m_right * s_left;
}

double solve_q_bisection(const int n_left,
                         const int n_right,
                         const double m_left,
                         const double m_right,
                         const int n_bulk_left,
                         const int n_bulk_right,
                         const double other_sum_left,
                         const double other_sum_right) {
  if (n_left + n_right <= 0) {
    return 1.0;
  }
  if (!(m_left > 0.0) || !(m_right > 0.0)) {
    return 1.0;
  }

  const double g_lo = interface_g(1.0,
                                  n_left,
                                  n_right,
                                  m_left,
                                  m_right,
                                  n_bulk_left,
                                  n_bulk_right,
                                  other_sum_left,
                                  other_sum_right);
  if (std::abs(g_lo) <= 1.0e-30) {
    return 1.0;
  }
  if (!(g_lo < 0.0)) {
    return 1.0;
  }

  double q_hi = 2.0;
  double g_hi = interface_g(q_hi,
                            n_left,
                            n_right,
                            m_left,
                            m_right,
                            n_bulk_left,
                            n_bulk_right,
                            other_sum_left,
                            other_sum_right);
  while (!(g_hi > 0.0) && q_hi < 1.0e6) {
    q_hi = std::min(1.0e6, q_hi * 2.0);
    g_hi = interface_g(q_hi,
                       n_left,
                       n_right,
                       m_left,
                       m_right,
                       n_bulk_left,
                       n_bulk_right,
                       other_sum_left,
                       other_sum_right);
  }
  if (!(g_hi > 0.0)) {
    return q_hi;
  }

  double lo = 1.0;
  double hi = q_hi;
  for (int iter = 0; iter < 60; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double g_mid = interface_g(mid,
                                     n_left,
                                     n_right,
                                     m_left,
                                     m_right,
                                     n_bulk_left,
                                     n_bulk_right,
                                     other_sum_left,
                                     other_sum_right);
    if (g_mid > 0.0) {
      hi = mid;
    } else {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
}

}  // namespace

std::vector<double> compute_auto_zone_nodes(
    const double r_min,
    const std::vector<AutoZoneRegion>& regions,
    const AutoZoneConfig& cfg,
    AutoZoneDiagnostics* diag) {
  if (!(cfg.mass_ratio_max > 1.0)) {
    throw std::runtime_error("auto-zone requires mass_ratio_max > 1");
  }
  if (!(cfg.mass_ratio_hard_max >= cfg.mass_ratio_max)) {
    throw std::runtime_error("auto-zone requires mass_ratio_hard_max >= mass_ratio_max");
  }
  if (cfg.n_bridge_min < 0 || cfg.n_bridge_max < 0) {
    throw std::runtime_error("auto-zone requires n_bridge_min/n_bridge_max >= 0");
  }
  if (!(cfg.bridge_frac_max >= 0.0)) {
    throw std::runtime_error("auto-zone requires bridge_frac_max >= 0");
  }
  if (!(cfg.dr_min >= 0.0)) {
    throw std::runtime_error("auto-zone requires dr_min >= 0");
  }
  if (cfg.max_iter <= 0) {
    throw std::runtime_error("auto-zone requires max_iter > 0");
  }
  if (!(cfg.bulk_mass_tol > 0.0)) {
    throw std::runtime_error("auto-zone requires bulk_mass_tol > 0");
  }

  const std::vector<RegionState> rs = build_regions(r_min, regions, cfg);
  const std::size_t n_regions = rs.size();

  std::int64_t total_zones_ll = 0;
  for (const auto& r : rs) {
    total_zones_ll += r.nz;
  }
  if (total_zones_ll <= 0) {
    throw std::runtime_error("auto-zone total zone count must be positive");
  }
  const int total_zones = static_cast<int>(total_zones_ll);

  std::vector<double> m_bulk(n_regions, 0.0);
  std::vector<double> m_bulk_new(n_regions, 0.0);
  std::vector<double> prev_m_bulk(n_regions, 0.0);
  for (std::size_t i = 0; i < n_regions; ++i) {
    if (!rs[i].is_void) {
      m_bulk[i] = rs[i].total_mass / static_cast<double>(rs[i].nz);
    }
  }

  std::vector<std::vector<double>> left_coeff(n_regions);
  std::vector<std::vector<double>> right_coeff(n_regions);
  std::vector<std::string> warnings;

  struct BridgePlan {
    int n_left = 0;
    int n_right = 0;
    bool active = false;
  };
  std::vector<BridgePlan> plans(n_regions > 1 ? n_regions - 1 : 0);

  // Phase 1: bridge planning from initial uniform per-region bulk masses.
  for (std::size_t iface = 0; iface + 1 < n_regions; ++iface) {
    const auto& left = rs[iface];
    const auto& right = rs[iface + 1];
    if (left.is_void || right.is_void) {
      continue;
    }

    const double m_left = m_bulk[iface];
    const double m_right = m_bulk[iface + 1];
    if (!(m_left > 0.0) || !(m_right > 0.0)) {
      continue;
    }

    const double ratio =
        std::max(m_left, m_right) / std::max(std::min(m_left, m_right), kTiny);
    if (!(ratio >= 1.0)) {
      continue;
    }

    const double r_int = left.r_out;
    const double floor_left = interface_mass_floor_left(r_int, left.rho, cfg.dr_min, cfg.geometry_code);
    const double floor_right = interface_mass_floor_right(r_int, right.rho, cfg.dr_min, cfg.geometry_code);

    const int cap_left = bridge_cap(left.nz, cfg);
    const int cap_right = bridge_cap(right.nz, cfg);

    const bool bind_left = left.avg_dr <= 1.01 * cfg.dr_min;
    const bool bind_right = right.avg_dr <= 1.01 * cfg.dr_min;

    InterfaceCandidate chosen;
    double used_alpha = cfg.mass_ratio_max;
    bool relaxed = false;

    const std::vector<double> alphas = alpha_sweep(cfg.mass_ratio_max, cfg.mass_ratio_hard_max);
    for (const double alpha : alphas) {
      used_alpha = alpha;

      const int n_required_ratio =
          (ratio <= 1.0 + 1.0e-14)
              ? 0
              : static_cast<int>(std::ceil(std::log(ratio) / std::log(alpha)));
      const int n_required =
          (ratio <= 1.0 + 1.0e-14) ? 0 : std::max(cfg.n_bridge_min, n_required_ratio);

      const int base_left = compute_base_side_count(n_required, cap_left, cap_right);
      const int base_right = n_required - base_left;

      int cap_left_eff = cap_left;
      int cap_right_eff = cap_right;
      if (bind_left) {
        cap_left_eff = std::min(cap_left_eff, base_left);
      }
      if (bind_right) {
        cap_right_eff = std::min(cap_right_eff, base_right);
      }

      const int cap_sum = cap_left_eff + cap_right_eff;
      if (n_required > cap_sum) {
        continue;
      }

      for (int n_total = n_required; n_total <= cap_sum; ++n_total) {
        InterfaceCandidate best_for_total;
        const int n_left_min = std::max(0, n_total - cap_right_eff);
        const int n_left_max = std::min(cap_left_eff, n_total);
        for (int n_left = n_left_min; n_left <= n_left_max; ++n_left) {
          const int n_right = n_total - n_left;
          const double q = (n_total > 0)
                               ? std::pow(ratio, 1.0 / static_cast<double>(n_total))
                               : 1.0;
          if (n_total > 0 && q > alpha * (1.0 + 1.0e-12)) {
            continue;
          }

          const double m_int =
              (n_total > 0)
                  ? std::pow(m_left, static_cast<double>(n_right) / static_cast<double>(n_total)) *
                        std::pow(m_right, static_cast<double>(n_left) /
                                             static_cast<double>(n_total))
                  : std::sqrt(m_left * m_right);
          const double iface_mass_left = (n_left > 0) ? m_int : m_left;
          const double iface_mass_right = (n_right > 0) ? m_int : m_right;

          if (iface_mass_left + 1.0e-14 < floor_left ||
              iface_mass_right + 1.0e-14 < floor_right) {
            continue;
          }

          InterfaceCandidate cand;
          cand.valid = true;
          cand.n_left = n_left;
          cand.n_right = n_right;
          cand.q = q;
          cand.m_int = m_int;
          cand.iface_mass_left = iface_mass_left;
          cand.iface_mass_right = iface_mass_right;
          cand.two_sided = (n_left > 0 && n_right > 0);
          cand.symmetry = -std::abs(n_left - n_right);
          const double margin_left = floor_left > 0.0
                                         ? (iface_mass_left / floor_left)
                                         : std::numeric_limits<double>::infinity();
          const double margin_right = floor_right > 0.0
                                          ? (iface_mass_right / floor_right)
                                          : std::numeric_limits<double>::infinity();
          cand.dr_margin = std::min(margin_left, margin_right);

          if (better_candidate(cand, best_for_total)) {
            best_for_total = cand;
          }
        }

        if (best_for_total.valid) {
          chosen = best_for_total;
          relaxed = alpha > cfg.mass_ratio_max + 1.0e-14;
          break;
        }
      }

      if (chosen.valid) {
        break;
      }
    }

    if (!chosen.valid) {
      const int n_required_hard =
          (ratio <= 1.0 + 1.0e-14)
              ? 0
              : std::max(cfg.n_bridge_min,
                         static_cast<int>(std::ceil(std::log(ratio) /
                                                    std::log(cfg.mass_ratio_hard_max))));
      const int base_left = compute_base_side_count(n_required_hard, cap_left, cap_right);
      const int base_right = n_required_hard - base_left;

      int cap_left_eff = cap_left;
      int cap_right_eff = cap_right;
      if (bind_left) {
        cap_left_eff = std::min(cap_left_eff, base_left);
      }
      if (bind_right) {
        cap_right_eff = std::min(cap_right_eff, base_right);
      }

      chosen = choose_violation_candidate(ratio,
                                          m_left,
                                          m_right,
                                          floor_left,
                                          floor_right,
                                          cap_left_eff,
                                          cap_right_eff);
      if (!chosen.valid) {
        std::ostringstream oss;
        oss << "auto-zone infeasible at interface " << iface
            << ": dr_min hard floor cannot be satisfied";
        throw std::runtime_error(oss.str());
      }

      std::ostringstream oss;
      oss << "auto-zone interface " << iface << " accepted mass-ratio violation: q="
          << chosen.q << " exceeds hard_max=" << cfg.mass_ratio_hard_max;
      push_warning_once(warnings, oss.str());
    }

    if (relaxed) {
      std::ostringstream oss;
      oss << "auto-zone interface " << iface << " relaxed mass_ratio_max from "
          << cfg.mass_ratio_max << " to " << used_alpha;
      push_warning_once(warnings, oss.str());
    }
    if (bind_left || bind_right) {
      std::ostringstream oss;
      oss << "auto-zone interface " << iface << " dr_min binding"
          << " (left=" << (bind_left ? "on" : "off")
          << ", right=" << (bind_right ? "on" : "off") << ")";
      push_warning_once(warnings, oss.str());
    }

    plans[iface] = BridgePlan{chosen.n_left, chosen.n_right, chosen.valid};
  }

  // Trim planned bridge counts so each region retains minimum bulk zones.
  for (std::size_t i = 0; i < n_regions; ++i) {
    int left_count = 0;
    int right_count = 0;
    if (i > 0 && plans[i - 1].active) {
      left_count = plans[i - 1].n_right;
    }
    if (i + 1 < n_regions && plans[i].active) {
      right_count = plans[i].n_left;
    }

    std::vector<double> c_left(static_cast<std::size_t>(left_count), 1.0);
    std::vector<double> c_right(static_cast<std::size_t>(right_count), 1.0);
    trim_bridge_for_overlap(rs[i].nz, c_left, c_right);
    left_count = static_cast<int>(c_left.size());
    right_count = static_cast<int>(c_right.size());

    if (i > 0 && plans[i - 1].active) {
      plans[i - 1].n_right = left_count;
    }
    if (i + 1 < n_regions && plans[i].active) {
      plans[i].n_left = right_count;
    }
  }

  std::vector<double> other_sum_left(n_regions, 0.0);
  std::vector<double> other_sum_right(n_regions, 0.0);

  // Phase 2+3: bisection q-solve with a few coupling passes.
  for (int coupling_pass = 0; coupling_pass < 3; ++coupling_pass) {
    prev_m_bulk = m_bulk;
    m_bulk_new = prev_m_bulk;

    for (std::size_t iface = 0; iface + 1 < n_regions; ++iface) {
      if (!plans[iface].active) {
        right_coeff[iface].clear();
        left_coeff[iface + 1].clear();
        other_sum_right[iface] = 0.0;
        other_sum_left[iface + 1] = 0.0;
        continue;
      }
      if (rs[iface].is_void || rs[iface + 1].is_void) {
        right_coeff[iface].clear();
        left_coeff[iface + 1].clear();
        other_sum_right[iface] = 0.0;
        other_sum_left[iface + 1] = 0.0;
        continue;
      }

      int n_left = plans[iface].n_left;
      int n_right = plans[iface].n_right;

      const int cap_left = bridge_cap(rs[iface].nz, cfg);
      const int cap_right = bridge_cap(rs[iface + 1].nz, cfg);
      const double r_int = rs[iface].r_out;
      const double floor_left_val = interface_mass_floor_left(r_int, rs[iface].rho, cfg.dr_min, cfg.geometry_code);
      const double floor_right_val =
          interface_mass_floor_right(r_int, rs[iface + 1].rho, cfg.dr_min, cfg.geometry_code);

      double q = 1.0;

      for (int adj_iter = 0; adj_iter < 20; ++adj_iter) {
        const int left_bridge_of_left =
            (iface > 0 && plans[iface - 1].active) ? plans[iface - 1].n_right : 0;
        const int right_bridge_of_right =
            (iface + 2 < n_regions && plans[iface + 1].active) ? plans[iface + 1].n_left : 0;
        const int n_bulk_left = rs[iface].nz - left_bridge_of_left - n_left;
        const int n_bulk_right = rs[iface + 1].nz - n_right - right_bridge_of_right;
        if (n_bulk_left < 2 || n_bulk_right < 2) break;

        q = solve_q_bisection(n_left,
                              n_right,
                              rs[iface].total_mass,
                              rs[iface + 1].total_mass,
                              n_bulk_left,
                              n_bulk_right,
                              other_sum_left[iface],
                              other_sum_right[iface + 1]);

        // Check dr_min floor constraint (HARD - takes priority)
        const double S_L_val =
            static_cast<double>(n_bulk_left) + coeff_sum_increasing(q, n_left) + other_sum_left[iface];
        const double S_R_val = static_cast<double>(n_bulk_right) +
                               coeff_sum_decreasing(q, n_right) + other_sum_right[iface + 1];
        const double m_iface_L =
            (n_left > 0 && S_L_val > 0.0)
                ? (rs[iface].total_mass / S_L_val) * std::pow(q, static_cast<double>(n_left))
                : 0.0;
        const double m_iface_R =
            (n_right > 0 && S_R_val > 0.0)
                ? (rs[iface + 1].total_mass / S_R_val) * std::pow(q, -static_cast<double>(n_right))
                : 0.0;

        bool reduced = false;
        if (n_right > 0 && m_iface_R + 1.0e-14 < floor_right_val) {
          --n_right;
          reduced = true;
        }
        if (n_left > 0 && m_iface_L + 1.0e-14 < floor_left_val) {
          --n_left;
          reduced = true;
        }
        if (reduced) continue;

        // Check mass_ratio_max (SOFT - try to add zones if possible)
        if (q > cfg.mass_ratio_max * (1.0 + 1.0e-12)) {
          const int eff_cap_left =
              std::min(cap_left, rs[iface].nz - left_bridge_of_left - 2);
          const int eff_cap_right =
              std::min(cap_right, rs[iface + 1].nz - right_bridge_of_right - 2);
          bool added = false;
          if (n_left < eff_cap_left && n_left <= n_right) {
            ++n_left;
            added = true;
          } else if (n_right < eff_cap_right) {
            ++n_right;
            added = true;
          } else if (n_left < eff_cap_left) {
            ++n_left;
            added = true;
          }
          if (added) continue;
          // Cannot add more: accept the mass-ratio violation.
        }

        break;  // Constraints satisfied or accepted.
      }

      // Update the plan with adjusted counts.
      plans[iface].n_left = n_left;
      plans[iface].n_right = n_right;

      // Recompute n_bulk with final plan.
      const int left_bridge_of_left_final =
          (iface > 0 && plans[iface - 1].active) ? plans[iface - 1].n_right : 0;
      const int right_bridge_of_right_final =
          (iface + 2 < n_regions && plans[iface + 1].active) ? plans[iface + 1].n_left : 0;
      const int n_bulk_left_final = rs[iface].nz - left_bridge_of_left_final - n_left;
      const int n_bulk_right_final = rs[iface + 1].nz - n_right - right_bridge_of_right_final;

      right_coeff[iface].clear();
      right_coeff[iface].reserve(static_cast<std::size_t>(n_left));
      for (int k = 0; k < n_left; ++k) {
        right_coeff[iface].push_back(std::pow(q, static_cast<double>(k + 1)));
      }

      left_coeff[iface + 1].clear();
      left_coeff[iface + 1].reserve(static_cast<std::size_t>(n_right));
      for (int k = 0; k < n_right; ++k) {
        left_coeff[iface + 1].push_back(std::pow(q, -static_cast<double>(k + 1)));
      }

      other_sum_right[iface] =
          std::accumulate(right_coeff[iface].begin(), right_coeff[iface].end(), 0.0);
      other_sum_left[iface + 1] =
          std::accumulate(left_coeff[iface + 1].begin(), left_coeff[iface + 1].end(), 0.0);

      const double s_left =
          static_cast<double>(n_bulk_left_final) + other_sum_left[iface] + other_sum_right[iface];
      const double s_right = static_cast<double>(n_bulk_right_final) + other_sum_left[iface + 1] +
                             other_sum_right[iface + 1];
      if (s_left > 0.0) {
        m_bulk_new[iface] = rs[iface].total_mass / s_left;
      }
      if (s_right > 0.0) {
        m_bulk_new[iface + 1] = rs[iface + 1].total_mass / s_right;
      }
    }

    for (std::size_t i = 0; i < n_regions; ++i) {
      if (rs[i].is_void) {
        m_bulk_new[i] = 0.0;
      }
    }
    m_bulk.swap(m_bulk_new);
  }

  // Finalize bulk masses from final coefficients for exact consistency.
  for (std::size_t i = 0; i < n_regions; ++i) {
    if (rs[i].is_void) {
      m_bulk[i] = 0.0;
      continue;
    }
    const int n_bulk = rs[i].nz - static_cast<int>(left_coeff[i].size()) -
                       static_cast<int>(right_coeff[i].size());
    if (n_bulk <= 0) {
      continue;
    }
    const double sum_left = std::accumulate(left_coeff[i].begin(), left_coeff[i].end(), 0.0);
    const double sum_right = std::accumulate(right_coeff[i].begin(), right_coeff[i].end(), 0.0);
    const double denom = static_cast<double>(n_bulk) + sum_left + sum_right;
    if (denom > 0.0) {
      m_bulk[i] = rs[i].total_mass / denom;
    }
  }

  std::vector<double> nodes;
  nodes.reserve(static_cast<std::size_t>(total_zones + 1));
  nodes.push_back(r_min);

  std::vector<double> zone_masses;
  zone_masses.reserve(static_cast<std::size_t>(total_zones));
  std::vector<bool> zone_nonvoid;
  zone_nonvoid.reserve(static_cast<std::size_t>(total_zones));

  for (std::size_t i = 0; i < n_regions; ++i) {
    const auto& r = rs[i];
    const double r_in = nodes.back();
    if (r.is_void) {
      for (int k = 0; k < r.nz; ++k) {
        const double u = static_cast<double>(k + 1) / static_cast<double>(r.nz);
        double r_next = r_in + (r.r_out - r_in) * u;
        if (k + 1 == r.nz) {
          r_next = r.r_out;
        }
        const double m_zone = r.rho * shell_volume(nodes.back(), r_next, cfg.geometry_code);
        zone_masses.push_back(m_zone);
        zone_nonvoid.push_back(false);
        nodes.push_back(r_next);
      }
      continue;
    }

    std::vector<double> m_zone =
        build_region_zone_masses(r.nz, m_bulk[i], left_coeff[i], right_coeff[i]);
    // W-G: mass -> radius inversion per geometry. Spherical keeps the
    // historic cube-accumulator arithmetic verbatim.
    double r_cubed = r_in * r_in * r_in;
    double r_sq = r_in * r_in;
    double x_lin = r_in;
    for (int k = 0; k < r.nz; ++k) {
      const double m = m_zone[static_cast<std::size_t>(k)];
      double r_next;
      if (cfg.geometry_code == 1) {
        r_sq += m / (std::max(r.rho, kTiny) *
                     3.141592653589793238462643383279502884);
        r_next = std::sqrt(std::max(r_sq, 0.0));
      } else if (cfg.geometry_code == 2) {
        x_lin += m / std::max(r.rho, kTiny);
        r_next = x_lin;
      } else {
        const double delta = m / (std::max(r.rho, kTiny) * kFourPiOver3);
        r_cubed += delta;
        r_next = std::cbrt(std::max(r_cubed, 0.0));
      }
      if (k + 1 == r.nz) {
        r_next = r.r_out;
      }
      zone_masses.push_back(m);
      zone_nonvoid.push_back(true);
      nodes.push_back(r_next);
    }
  }

  if (static_cast<int>(nodes.size()) != total_zones + 1) {
    throw std::runtime_error("auto-zone internal error: node count mismatch");
  }

  for (int i = 0; i < total_zones; ++i) {
    if (!(nodes[static_cast<std::size_t>(i + 1)] >
          nodes[static_cast<std::size_t>(i)])) {
      throw std::runtime_error("auto-zone produced non-monotonic nodes");
    }
  }

  int iface_zone = 0;
  for (std::size_t i = 0; i + 1 < n_regions; ++i) {
    iface_zone += rs[i].nz;
    if (rs[i].is_void || rs[i + 1].is_void) {
      continue;
    }
    const double dr_left =
        nodes[static_cast<std::size_t>(iface_zone)] -
        nodes[static_cast<std::size_t>(iface_zone - 1)];
    const double dr_right =
        nodes[static_cast<std::size_t>(iface_zone + 1)] -
        nodes[static_cast<std::size_t>(iface_zone)];
    if (dr_left + 1.0e-14 < cfg.dr_min || dr_right + 1.0e-14 < cfg.dr_min) {
      std::ostringstream oss;
      oss << "auto-zone dr_min hard constraint violated at interface " << i
          << " (dr_left=" << dr_left << ", dr_right=" << dr_right
          << ", dr_min=" << cfg.dr_min << ")";
      throw std::runtime_error(oss.str());
    }
  }

  AutoZoneDiagnostics local_diag;
  local_diag.warnings = std::move(warnings);

  double dr_min_actual = std::numeric_limits<double>::infinity();
  for (int i = 0; i < total_zones; ++i) {
    dr_min_actual = std::min(
        dr_min_actual,
        nodes[static_cast<std::size_t>(i + 1)] - nodes[static_cast<std::size_t>(i)]);
  }
  local_diag.dr_min_actual = std::isfinite(dr_min_actual) ? dr_min_actual : 0.0;

  double ratio_min = std::numeric_limits<double>::infinity();
  double ratio_max = 1.0;
  double ratio_sum = 0.0;
  int ratio_count = 0;
  int n_viol = 0;
  for (int i = 0; i + 1 < total_zones; ++i) {
    if (!zone_nonvoid[static_cast<std::size_t>(i)] ||
        !zone_nonvoid[static_cast<std::size_t>(i + 1)]) {
      continue;
    }
    const double m0 = zone_masses[static_cast<std::size_t>(i)];
    const double m1 = zone_masses[static_cast<std::size_t>(i + 1)];
    if (!(m0 > 0.0) || !(m1 > 0.0)) {
      continue;
    }
    const double ratio = std::max(m0, m1) / std::min(m0, m1);
    ratio_min = std::min(ratio_min, ratio);
    ratio_max = std::max(ratio_max, ratio);
    ratio_sum += ratio;
    ++ratio_count;
    if (ratio > cfg.mass_ratio_max * (1.0 + 1.0e-12)) {
      ++n_viol;
    }
  }

  if (ratio_count > 0) {
    local_diag.mass_ratio_min = ratio_min;
    local_diag.mass_ratio_max = ratio_max;
    local_diag.mass_ratio_mean = ratio_sum / static_cast<double>(ratio_count);
  }
  local_diag.n_ratio_violations = n_viol;

  if (diag != nullptr) {
    *diag = std::move(local_diag);
  }

  return nodes;
}

}  // namespace tenryu::core
