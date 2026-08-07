#pragma once

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "radiation/groups.cuh"

namespace tenryu::core {
struct Config;
}

namespace tenryu::radiation {

#if defined(__CUDACC__)
#define TENRYU_PLANCK_DEVICE __device__
#else
#define TENRYU_PLANCK_DEVICE
#endif

struct PlanckTableDeviceView {
  const double* T_grid = nullptr;   // [n_T]
  const double* b_g = nullptr;      // [n_T * n_groups]
  const double* cdf_g = nullptr;    // [n_T * n_groups]
  int n_T = 0;
  int n_groups = 1;

  TENRYU_PLANCK_DEVICE inline double interpolate_b(const int g, const double T_eV) const {
    if (n_groups <= 1) {
      return 1.0;
    }
    if (n_T <= 1) {
      return b_g[g];
    }
    if (T_eV <= T_grid[0]) {
      return b_g[g];
    }
    if (T_eV >= T_grid[n_T - 1]) {
      return b_g[(n_T - 1) * n_groups + g];
    }

    int lo = 0;
    int hi = n_T - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (T_grid[mid] <= T_eV) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const double T0 = T_grid[lo];
    const double T1 = T_grid[hi];
    const double b0 = b_g[lo * n_groups + g];
    const double b1 = b_g[hi * n_groups + g];
    const double dT = T1 - T0;
    if (!(dT > 1.0e-30)) {
      return b0;
    }
    const double w = (T_eV - T0) / dT;
    return (1.0 - w) * b0 + w * b1;
  }

  TENRYU_PLANCK_DEVICE inline double interpolate_cdf(const int g, const double T_eV) const {
    if (n_groups <= 1) {
      return 1.0;
    }
    if (n_T <= 1) {
      return cdf_g[g];
    }
    if (T_eV <= T_grid[0]) {
      return cdf_g[g];
    }
    if (T_eV >= T_grid[n_T - 1]) {
      return cdf_g[(n_T - 1) * n_groups + g];
    }

    int lo = 0;
    int hi = n_T - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (T_grid[mid] <= T_eV) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const double T0 = T_grid[lo];
    const double T1 = T_grid[hi];
    const double c0 = cdf_g[lo * n_groups + g];
    const double c1 = cdf_g[hi * n_groups + g];
    const double dT = T1 - T0;
    if (!(dT > 1.0e-30)) {
      return c0;
    }
    const double w = (T_eV - T0) / dT;
    return (1.0 - w) * c0 + w * c1;
  }

  TENRYU_PLANCK_DEVICE inline int sample_group(const double xi, const double T_eV) const {
    if (n_groups <= 1) {
      return 0;
    }
    const double u = fmin(fmax(xi, 0.0), 1.0 - 1.0e-15);
    for (int g = 0; g < n_groups; ++g) {
      const double cdf = interpolate_cdf(g, T_eV);
      if (u <= cdf) {
        return g;
      }
    }
    return n_groups - 1;
  }
};

#undef TENRYU_PLANCK_DEVICE

class PlanckTable {
 public:
  PlanckTable() = default;
  ~PlanckTable();

  PlanckTable(const PlanckTable&) = delete;
  PlanckTable& operator=(const PlanckTable&) = delete;

  PlanckTable(PlanckTable&& other) noexcept;
  PlanckTable& operator=(PlanckTable&& other) noexcept;

  void build(const Groups& groups,
             int n_T = 200,
             double T_min_eV = 1.0e-2,
             double T_max_eV = 1.0e2);

  // Fixed-measure (picket-fence) table: b_g == fractions[g] at EVERY
  // temperature, exactly. Implemented as an n_T = 2 table whose two rows are
  // identical, so the linear-in-T interpolation (and the out-of-range clamps)
  // reproduce the constants without error. Additive helper for verification
  // gates (Su & Olson 1999 picket-fence benchmark); no existing caller
  // changes. fractions must be non-negative and sum to 1 within 1e-12.
  void build_constant_fractions(const Groups& groups,
                                const std::vector<double>& fractions,
                                double T_min_eV = 1.0,
                                double T_max_eV = 1.0e3);

  [[nodiscard]] int n_T() const noexcept {
    return n_T_;
  }

  [[nodiscard]] int n_groups() const noexcept {
    return n_groups_;
  }

  [[nodiscard]] const std::vector<double>& host_T_grid() const noexcept {
    return T_grid_host_;
  }

  [[nodiscard]] const std::vector<double>& host_b_g() const noexcept {
    return b_g_host_;
  }

  [[nodiscard]] const std::vector<double>& host_cdf_g() const noexcept {
    return cdf_g_host_;
  }

  [[nodiscard]] double interpolate_b_host(const int g, const double T_eV) const {
    if (n_groups_ <= 1) {
      return 1.0;
    }
    if (g < 0 || g >= n_groups_) {
      return 0.0;
    }
    if (n_T_ <= 1) {
      return b_g_host_[static_cast<std::size_t>(g)];
    }
    warn_if_temperature_clamped(T_eV);
    if (T_eV <= T_grid_host_.front()) {
      return b_g_host_[static_cast<std::size_t>(g)];
    }
    if (T_eV >= T_grid_host_.back()) {
      return b_g_host_[static_cast<std::size_t>((n_T_ - 1) * n_groups_ + g)];
    }

    int lo = 0;
    int hi = n_T_ - 1;
    while (hi - lo > 1) {
      const int mid = (lo + hi) / 2;
      if (T_grid_host_[static_cast<std::size_t>(mid)] <= T_eV) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    const double T0 = T_grid_host_[static_cast<std::size_t>(lo)];
    const double T1 = T_grid_host_[static_cast<std::size_t>(hi)];
    const double b0 =
        b_g_host_[static_cast<std::size_t>(lo * n_groups_ + g)];
    const double b1 =
        b_g_host_[static_cast<std::size_t>(hi * n_groups_ + g)];
    const double dT = T1 - T0;
    if (!(dT > 1.0e-30)) {
      return b0;
    }
    const double w = (T_eV - T0) / dT;
    return (1.0 - w) * b0 + w * b1;
  }

  [[nodiscard]] PlanckTableDeviceView device_view() const noexcept {
    PlanckTableDeviceView v;
    v.T_grid = d_T_grid_;
    v.b_g = d_b_g_;
    v.cdf_g = d_cdf_g_;
    v.n_T = n_T_;
    v.n_groups = n_groups_;
    return v;
  }

 private:
  void warn_if_temperature_clamped(const double T_eV) const {
    if (n_T_ <= 1 || T_grid_host_.empty() || !std::isfinite(T_eV)) {
      return;
    }
    if (T_eV < T_grid_host_.front()) {
      static bool warned_low = false;
      if (!warned_low) {
        warned_low = true;
        core::log_warning("Planck table temperature below tabulated range; clamping to Tmin "
                          "(T=" + std::to_string(T_eV) +
                          ", Tmin=" + std::to_string(T_grid_host_.front()) + ").");
      }
      return;
    }
    if (T_eV > T_grid_host_.back()) {
      static bool warned_high = false;
      if (!warned_high) {
        warned_high = true;
        core::log_warning("Planck table temperature above tabulated range; clamping to Tmax "
                          "(T=" + std::to_string(T_eV) +
                          ", Tmax=" + std::to_string(T_grid_host_.back()) + ").");
      }
    }
  }

  void release();

  int n_T_ = 0;
  int n_groups_ = 1;

  std::vector<double> T_grid_host_;
  std::vector<double> b_g_host_;
  std::vector<double> cdf_g_host_;

  double* d_T_grid_ = nullptr;
  double* d_b_g_ = nullptr;
  double* d_cdf_g_ = nullptr;
};

std::vector<double> resolve_compute_T_range_eV(const core::Config& cfg,
                                               bool log_auto_derivation);
PlanckTable build_planck_table_from_config(const core::Config& cfg);

}  // namespace tenryu::radiation
