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
#define TENRYU_PLANCK_HD __host__ __device__
#else
#define TENRYU_PLANCK_DEVICE
#define TENRYU_PLANCK_HD
#endif

// Group Planck fraction b_g(T) at a temperature strictly inside table
// interval [lo, hi] (NUMERICS §6.1). The cumulative fraction
// C_g = sum_{g' <= g} b_g' (low-energy side) and the tail
// D_g = 1 - C_g = sum_{g' > g} b_g' (high-energy side) are interpolated in
// u = ln T by cubic Hermite polynomials of ln C_g or ln D_g — C where
// C_g < 1/2 at both table temperatures, D elsewhere — with the exact node
// slopes d ln C/du, d ln D/du from the Planck density at the group bounds;
// b_g is the difference of consecutive values, so the fractions sum to 1 by
// construction. The error is O(du^4); interpolating b_g linearly in T
// overestimated the convex exp(-h nu/T) Wien fractions between nodes by about
// (dT/T)^2 (x^2 - 2x)/8 at mid-interval (x = h nu/T). A negative difference
// (possible only far below the interpolation error) is clamped to 0
// (2026-09-23).
TENRYU_PLANCK_HD inline double planck_log_hermite(const double* ln_x,
                                                  const double* dln_x,
                                                  const int i0,
                                                  const int i1,
                                                  const double du,
                                                  const double w) {
  const double w2 = w * w;
  const double w3 = w2 * w;
  const double h00 = 2.0 * w3 - 3.0 * w2 + 1.0;
  const double h10 = w3 - 2.0 * w2 + w;
  const double h01 = -2.0 * w3 + 3.0 * w2;
  const double h11 = w3 - w2;
  return exp(h00 * ln_x[i0] + h10 * du * dln_x[i0] + h01 * ln_x[i1] + h11 * du * dln_x[i1]);
}

TENRYU_PLANCK_HD inline double planck_fraction_in_interval(const double* cdf,
                                                           const double* tail,
                                                           const double* ln_cdf,
                                                           const double* ln_tail,
                                                           const double* dln_cdf,
                                                           const double* dln_tail,
                                                           const int n_groups,
                                                           const int lo,
                                                           const int hi,
                                                           const double du,
                                                           const double w,
                                                           const int g) {
  const int base0 = lo * n_groups;
  const int base1 = hi * n_groups;
  const auto value = [&](const double* x, const double* ln_x, const double* dln_x,
                         const int k) {
    if (w <= 0.0) {
      return x[base0 + k];
    }
    if (w >= 1.0) {
      return x[base1 + k];
    }
    return planck_log_hermite(ln_x, dln_x, base0 + k, base1 + k, du, w);
  };
  // Group k in [0, n_groups - 2] takes the cumulative form while C_k < 1/2 at
  // both temperatures; k = -1 is the empty cumulative (0), k = n_groups - 1
  // the empty tail (0).
  const auto cumulative_side = [&](const int k) {
    return k < 0 || (k < n_groups - 1 && cdf[base0 + k] < 0.5 && cdf[base1 + k] < 0.5);
  };
  const bool g_cum = cumulative_side(g);
  const bool gm1_cum = cumulative_side(g - 1);
  double b = 0.0;
  if (g_cum) {
    const double C_g = value(cdf, ln_cdf, dln_cdf, g);
    const double C_gm1 = (g == 0) ? 0.0 : value(cdf, ln_cdf, dln_cdf, g - 1);
    b = C_g - C_gm1;
  } else if (!gm1_cum) {
    const double D_g = (g == n_groups - 1) ? 0.0 : value(tail, ln_tail, dln_tail, g);
    const double D_gm1 = value(tail, ln_tail, dln_tail, g - 1);
    b = D_gm1 - D_g;
  } else {
    const double D_g = (g == n_groups - 1) ? 0.0 : value(tail, ln_tail, dln_tail, g);
    const double C_gm1 = (g == 0) ? 0.0 : value(cdf, ln_cdf, dln_cdf, g - 1);
    b = (1.0 - D_g) - C_gm1;
  }
  return fmax(b, 0.0);
}

// d/dT of planck_log_hermite at the same arguments: exp(H) H'(w) / (du T).
TENRYU_PLANCK_HD inline double planck_log_hermite_dT(const double* ln_x,
                                                     const double* dln_x,
                                                     const int i0,
                                                     const int i1,
                                                     const double du,
                                                     const double w,
                                                     const double T) {
  const double w2 = w * w;
  const double w3 = w2 * w;
  const double h00 = 2.0 * w3 - 3.0 * w2 + 1.0;
  const double h10 = w3 - 2.0 * w2 + w;
  const double h01 = -2.0 * w3 + 3.0 * w2;
  const double h11 = w3 - w2;
  const double d00 = 6.0 * w2 - 6.0 * w;
  const double d10 = 3.0 * w2 - 4.0 * w + 1.0;
  const double d01 = -6.0 * w2 + 6.0 * w;
  const double d11 = 3.0 * w2 - 2.0 * w;
  const double H = h00 * ln_x[i0] + h10 * du * dln_x[i0] + h01 * ln_x[i1] + h11 * du * dln_x[i1];
  const double dH = d00 * ln_x[i0] + d10 * du * dln_x[i0] + d01 * ln_x[i1] + d11 * du * dln_x[i1];
  return exp(H) * dH / (du * T);
}

// d b_g / dT of planck_fraction_in_interval where the fraction is positive
// (same branches). w is taken in [0, 1] (the node values' slopes at the ends).
TENRYU_PLANCK_HD inline double planck_fraction_slope_in_interval(const double* cdf,
                                                                const double* ln_cdf,
                                                                const double* ln_tail,
                                                                const double* dln_cdf,
                                                                const double* dln_tail,
                                                                const int n_groups,
                                                                const int lo,
                                                                const int hi,
                                                                const double du,
                                                                const double w,
                                                                const int g,
                                                                const double T) {
  const int base0 = lo * n_groups;
  const int base1 = hi * n_groups;
  const double wc = fmin(fmax(w, 0.0), 1.0);
  const auto slope = [&](const double* ln_x, const double* dln_x, const int k) {
    return planck_log_hermite_dT(ln_x, dln_x, base0 + k, base1 + k, du, wc, T);
  };
  const auto cumulative_side = [&](const int k) {
    return k < 0 || (k < n_groups - 1 && cdf[base0 + k] < 0.5 && cdf[base1 + k] < 0.5);
  };
  const bool g_cum = cumulative_side(g);
  const bool gm1_cum = cumulative_side(g - 1);
  if (g_cum) {
    const double dC_g = slope(ln_cdf, dln_cdf, g);
    const double dC_gm1 = (g == 0) ? 0.0 : slope(ln_cdf, dln_cdf, g - 1);
    return dC_g - dC_gm1;
  }
  if (!gm1_cum) {
    const double dD_g = (g == n_groups - 1) ? 0.0 : slope(ln_tail, dln_tail, g);
    const double dD_gm1 = slope(ln_tail, dln_tail, g - 1);
    return dD_gm1 - dD_g;
  }
  const double dD_g = (g == n_groups - 1) ? 0.0 : slope(ln_tail, dln_tail, g);
  const double dC_gm1 = (g == 0) ? 0.0 : slope(ln_cdf, dln_cdf, g - 1);
  return -dD_g - dC_gm1;
}

// d b_g / dT of planck_fraction_in_interval (same branches; 0 where the
// fraction is clamped at 0). w is taken in [0, 1] (the node values' slopes at
// the ends).
TENRYU_PLANCK_HD inline double planck_fraction_dT_in_interval(const double* cdf,
                                                             const double* tail,
                                                             const double* ln_cdf,
                                                             const double* ln_tail,
                                                             const double* dln_cdf,
                                                             const double* dln_tail,
                                                             const int n_groups,
                                                             const int lo,
                                                             const int hi,
                                                             const double du,
                                                             const double w,
                                                             const int g,
                                                             const double T) {
  if (planck_fraction_in_interval(cdf, tail, ln_cdf, ln_tail, dln_cdf, dln_tail, n_groups, lo,
                                  hi, du, w, g) <= 0.0) {
    return 0.0;
  }
  return planck_fraction_slope_in_interval(cdf, ln_cdf, ln_tail, dln_cdf, dln_tail, n_groups, lo,
                                           hi, du, w, g, T);
}

// b_g and d b_g / dT with the fraction evaluated once: the values of
// planck_fraction_in_interval and planck_fraction_dT_in_interval.
TENRYU_PLANCK_HD inline void planck_fraction_and_dT_in_interval(const double* cdf,
                                                               const double* tail,
                                                               const double* ln_cdf,
                                                               const double* ln_tail,
                                                               const double* dln_cdf,
                                                               const double* dln_tail,
                                                               const int n_groups,
                                                               const int lo,
                                                               const int hi,
                                                               const double du,
                                                               const double w,
                                                               const int g,
                                                               const double T,
                                                               double& b,
                                                               double& db) {
  b = planck_fraction_in_interval(cdf, tail, ln_cdf, ln_tail, dln_cdf, dln_tail, n_groups, lo, hi,
                                  du, w, g);
  db = (b <= 0.0) ? 0.0
                  : planck_fraction_slope_in_interval(cdf, ln_cdf, ln_tail, dln_cdf, dln_tail,
                                                      n_groups, lo, hi, du, w, g, T);
}

// Interpolated cumulative fraction C_g(T) inside interval [lo, hi], the
// same forms as planck_fraction_in_interval.
TENRYU_PLANCK_HD inline double planck_cumulative_in_interval(const double* cdf,
                                                             const double* tail,
                                                             const double* ln_cdf,
                                                             const double* ln_tail,
                                                             const double* dln_cdf,
                                                             const double* dln_tail,
                                                             const int n_groups,
                                                             const int lo,
                                                             const int hi,
                                                             const double du,
                                                             const double w,
                                                             const int g) {
  if (g >= n_groups - 1) {
    return 1.0;
  }
  const int base0 = lo * n_groups;
  const int base1 = hi * n_groups;
  const bool cum = cdf[base0 + g] < 0.5 && cdf[base1 + g] < 0.5;
  const double* x = cum ? cdf : tail;
  double v = 0.0;
  if (w <= 0.0) {
    v = x[base0 + g];
  } else if (w >= 1.0) {
    v = x[base1 + g];
  } else {
    v = cum ? planck_log_hermite(ln_cdf, dln_cdf, base0 + g, base1 + g, du, w)
            : planck_log_hermite(ln_tail, dln_tail, base0 + g, base1 + g, du, w);
  }
  return cum ? fmin(v, 1.0) : fmax(1.0 - v, 0.0);
}

struct PlanckTableDeviceView {
  const double* T_grid = nullptr;   // [n_T]
  const double* b_g = nullptr;      // [n_T * n_groups]
  const double* cdf_g = nullptr;    // [n_T * n_groups] C_g = sum_{g'<=g} b_g'
  const double* tail_g = nullptr;   // [n_T * n_groups] D_g = sum_{g'>g} b_g'
  const double* ln_cdf_g = nullptr;   // [n_T * n_groups] ln max(C_g, 1e-300)
  const double* ln_tail_g = nullptr;  // [n_T * n_groups] ln max(D_g, 1e-300)
  const double* dln_cdf_g = nullptr;   // [n_T * n_groups] d ln C_g / d ln T
  const double* dln_tail_g = nullptr;  // [n_T * n_groups] d ln D_g / d ln T
  const double* ln_T_grid = nullptr;  // [n_T]
  int n_T = 0;
  int n_groups = 1;
  int constant_in_T = 0;  // every row equal (picket-fence tables): b_g, C_g exact

  // Where a temperature falls in the table, for interpolate_b(g, location)
  // over many groups at one temperature: the interval search and ln T once
  // instead of per group (2026-09-25). interpolate_b(g, T) is
  // interpolate_b(g, locate_b(T)), the same operations.
  struct Location {
    int kind = 0;  // 0: 1.0 (one group); 1: row `lo`; 2: inside [lo, hi] at weight w
    int lo = 0;
    int hi = 0;
    double du = 0.0;
    double w = 0.0;
  };

  TENRYU_PLANCK_DEVICE inline Location locate_b(const double T_eV) const {
    Location loc;
    if (n_groups <= 1) {
      loc.kind = 0;
      return loc;
    }
    loc.kind = 1;
    if (n_T <= 1 || constant_in_T != 0) {
      loc.lo = 0;
      return loc;
    }
    if (T_eV <= T_grid[0]) {
      loc.lo = 0;
      return loc;
    }
    if (T_eV >= T_grid[n_T - 1]) {
      loc.lo = n_T - 1;
      return loc;
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

    const double dl = ln_T_grid[hi] - ln_T_grid[lo];
    loc.lo = lo;
    if (!(dl > 0.0)) {
      return loc;
    }
    loc.kind = 2;
    loc.hi = hi;
    loc.du = dl;
    loc.w = (log(T_eV) - ln_T_grid[lo]) / dl;
    return loc;
  }

  TENRYU_PLANCK_DEVICE inline double interpolate_b(const int g, const Location& loc) const {
    if (loc.kind == 0) {
      return 1.0;
    }
    if (loc.kind == 1) {
      return b_g[loc.lo * n_groups + g];
    }
    return planck_fraction_in_interval(cdf_g, tail_g, ln_cdf_g, ln_tail_g, dln_cdf_g, dln_tail_g,
                                       n_groups, loc.lo, loc.hi, loc.du, loc.w, g);
  }

  // d b_g / dT at the location of T_eV (0 outside the table and for
  // temperature-independent tables).
  TENRYU_PLANCK_DEVICE inline double interpolate_b_dT(const int g, const Location& loc,
                                                     const double T_eV) const {
    if (loc.kind != 2) {
      return 0.0;
    }
    return planck_fraction_dT_in_interval(cdf_g, tail_g, ln_cdf_g, ln_tail_g, dln_cdf_g,
                                          dln_tail_g, n_groups, loc.lo, loc.hi, loc.du, loc.w, g,
                                          T_eV);
  }

  // interpolate_b(g, loc) and interpolate_b_dT(g, loc, T_eV) with the
  // fraction evaluated once (the same values).
  TENRYU_PLANCK_DEVICE inline void interpolate_b_and_dT(const int g, const Location& loc,
                                                       const double T_eV, double& b,
                                                       double& db) const {
    if (loc.kind != 2) {
      b = interpolate_b(g, loc);
      db = 0.0;
      return;
    }
    planck_fraction_and_dT_in_interval(cdf_g, tail_g, ln_cdf_g, ln_tail_g, dln_cdf_g, dln_tail_g,
                                       n_groups, loc.lo, loc.hi, loc.du, loc.w, g, T_eV, b, db);
  }

  TENRYU_PLANCK_DEVICE inline double interpolate_b(const int g, const double T_eV) const {
    return interpolate_b(g, locate_b(T_eV));
  }

  TENRYU_PLANCK_DEVICE inline double interpolate_cdf(const int g, const double T_eV) const {
    if (n_groups <= 1) {
      return 1.0;
    }
    if (n_T <= 1 || constant_in_T != 0) {
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

    const double dl = ln_T_grid[hi] - ln_T_grid[lo];
    if (!(dl > 0.0)) {
      return cdf_g[lo * n_groups + g];
    }
    const double w = (log(T_eV) - ln_T_grid[lo]) / dl;
    return planck_cumulative_in_interval(cdf_g, tail_g, ln_cdf_g, ln_tail_g, dln_cdf_g,
                                         dln_tail_g, n_groups, lo, hi, dl, w, g);
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
#undef TENRYU_PLANCK_HD

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
  // identical; the interpolation returns the stored row for such a table
  // (constant_in_T), so the constants are reproduced without error.
  // Additive helper for verification gates (Su & Olson 1999 picket-fence
  // benchmark); no existing caller changes. fractions must be non-negative
  // and sum to 1 within 1e-12.
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
    if (n_T_ <= 1 || constant_in_T_) {
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

    const double dl = ln_T_grid_host_[static_cast<std::size_t>(hi)] -
                      ln_T_grid_host_[static_cast<std::size_t>(lo)];
    if (!(dl > 0.0)) {
      return b_g_host_[static_cast<std::size_t>(lo * n_groups_ + g)];
    }
    const double w =
        (std::log(T_eV) - ln_T_grid_host_[static_cast<std::size_t>(lo)]) / dl;
    return planck_fraction_in_interval(cdf_g_host_.data(), tail_g_host_.data(),
                                       ln_cdf_g_host_.data(), ln_tail_g_host_.data(),
                                       dln_cdf_g_host_.data(), dln_tail_g_host_.data(),
                                       n_groups_, lo, hi, dl, w, g);
  }

  [[nodiscard]] PlanckTableDeviceView device_view() const noexcept {
    PlanckTableDeviceView v;
    v.T_grid = d_T_grid_;
    v.b_g = d_b_g_;
    v.cdf_g = d_cdf_g_;
    v.tail_g = d_tail_g_;
    v.ln_cdf_g = d_ln_cdf_g_;
    v.ln_tail_g = d_ln_tail_g_;
    v.dln_cdf_g = d_dln_cdf_g_;
    v.dln_tail_g = d_dln_tail_g_;
    v.ln_T_grid = d_ln_T_grid_;
    v.constant_in_T = constant_in_T_ ? 1 : 0;
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
  // From T_grid_host_, b_g_host_, cdf_g_host_ and the node slopes
  // (dln_cdf_g_host_, dln_tail_g_host_, filled by the builder): fills the
  // tail and log tables and uploads all arrays to the device.
  void finalize_and_upload(const char* who);

  int n_T_ = 0;
  int n_groups_ = 1;
  bool constant_in_T_ = false;

  std::vector<double> T_grid_host_;
  std::vector<double> b_g_host_;
  std::vector<double> cdf_g_host_;
  std::vector<double> tail_g_host_;
  std::vector<double> ln_cdf_g_host_;
  std::vector<double> ln_tail_g_host_;
  std::vector<double> dln_cdf_g_host_;
  std::vector<double> dln_tail_g_host_;
  std::vector<double> ln_T_grid_host_;

  double* d_T_grid_ = nullptr;
  double* d_b_g_ = nullptr;
  double* d_cdf_g_ = nullptr;
  double* d_tail_g_ = nullptr;
  double* d_ln_cdf_g_ = nullptr;
  double* d_ln_tail_g_ = nullptr;
  double* d_dln_cdf_g_ = nullptr;
  double* d_dln_tail_g_ = nullptr;
  double* d_ln_T_grid_ = nullptr;
};

std::vector<double> resolve_compute_T_range_eV(const core::Config& cfg,
                                               bool log_auto_derivation);
PlanckTable build_planck_table_from_config(const core::Config& cfg);

}  // namespace tenryu::radiation
