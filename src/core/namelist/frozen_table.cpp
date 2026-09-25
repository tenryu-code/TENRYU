#include "core/namelist/frozen_table.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <string>

#include "core/error.hpp"
#include "core/namelist/errors.hpp"

#if TENRYU_ENABLE_PYTHON
#include <pybind11/pybind11.h>
#endif

namespace tenryu::core::namelist {

double FrozenTable1D::eval(const double xi) const {
  TENRYU_ASSERT(n_points == static_cast<int>(x.size()),
                "FrozenTable1D.x size must match n_points");
  TENRYU_ASSERT(n_points == static_cast<int>(y.size()),
                "FrozenTable1D.y size must match n_points");

  if (n_points <= 0) {
    static std::atomic<bool> warned_empty_table{false};
    bool expected = false;
    if (warned_empty_table.compare_exchange_strong(expected, true)) {
      tenryu::core::log_warning("FrozenTable1D::eval called with empty table; returning 0.0");
    }
    return 0.0;
  }
  if (n_points == 1) {
    if (zero_outside && (xi < x_min || xi > x_max)) {
      return 0.0;
    }
    return y.front();
  }

  if (xi < x_min) {
    return zero_outside ? 0.0 : y.front();
  }
  if (xi > x_max) {
    return zero_outside ? 0.0 : y.back();
  }

  auto it = std::upper_bound(x.begin(), x.end(), xi);
  if (it == x.begin()) {
    return y.front();
  }
  if (it == x.end()) {
    return y.back();
  }

  const int hi = static_cast<int>(std::distance(x.begin(), it));
  const int lo = hi - 1;
  const double x0 = x[static_cast<std::size_t>(lo)];
  const double x1 = x[static_cast<std::size_t>(hi)];
  const double y0 = y[static_cast<std::size_t>(lo)];
  const double y1 = y[static_cast<std::size_t>(hi)];

  const double dx = x1 - x0;
  if (dx <= std::numeric_limits<double>::epsilon()) {
    return y0;
  }
  const double s = (xi - x0) / dx;
  return y0 + s * (y1 - y0);
}

FrozenTable1D create_frozen_table_from_sampler(
    const std::function<double(double)>& sampler,
    const double t_min,
    const double t_max,
    const int n_samples) {
  TENRYU_ASSERT(sampler != nullptr,
                "create_frozen_table_from_sampler requires a valid sampler");
  TENRYU_ASSERT(n_samples >= 2,
                "create_frozen_table requires n_samples >= 2");
  TENRYU_ASSERT(t_max >= t_min,
                "create_frozen_table requires t_max >= t_min");

  FrozenTable1D table;
  table.n_points = n_samples;
  table.x_min = t_min;
  table.x_max = t_max;
  table.x.resize(static_cast<std::size_t>(n_samples), 0.0);
  table.y.resize(static_cast<std::size_t>(n_samples), 0.0);

  const double denom = static_cast<double>(n_samples - 1);
  for (int k = 0; k < n_samples; ++k) {
    const double alpha = static_cast<double>(k) / denom;
    const double t = t_min + alpha * (t_max - t_min);
    table.x[static_cast<std::size_t>(k)] = t;
    const double y = sampler(t);
    if (!std::isfinite(y)) {
      throw ConfigError(
          "Frozen table sampling produced non-finite value (NaN/Inf) during callable freeze");
    }
    table.y[static_cast<std::size_t>(k)] = y;
  }

  return table;
}

namespace {

struct TimeSampler {
  const std::function<double(double)>& sampler;
  const std::string& name;
  bool require_non_negative;

  double operator()(const double t) const {
    const double y = sampler(t);
    if (!std::isfinite(y)) {
      throw ConfigError("Callable " + name + " returned non-finite value at t=" +
                        std::to_string(t));
    }
    if (require_non_negative && y < 0.0) {
      throw ConfigError("Callable " + name + " returned negative value at t=" +
                        std::to_string(t));
    }
    return y;
  }
};

// Appends the dyadic refinement points strictly inside (a, b), in ascending
// order, while the chord misses the midpoint by more than the tolerance.
void refine_time_interval(const TimeSampler& sampler,
                          const double a,
                          const double fa,
                          const double b,
                          const double fb,
                          const int level,
                          std::vector<double>& x,
                          std::vector<double>& y) {
  if (level >= kFrozenTimeMaxRefine) {
    return;
  }
  const double m = 0.5 * (a + b);
  const double fm = sampler(m);
  const double chord_error = std::abs(fm - 0.5 * (fa + fb));
  const double scale = std::max({std::abs(fa), std::abs(fb), std::abs(fm)});
  if (!(chord_error > kFrozenTimeRelTol * scale)) {
    return;
  }
  refine_time_interval(sampler, a, fa, m, fm, level + 1, x, y);
  x.push_back(m);
  y.push_back(fm);
  refine_time_interval(sampler, m, fm, b, fb, level + 1, x, y);
}

}  // namespace

FrozenTable1D create_frozen_time_table_from_sampler(
    const std::function<double(double)>& raw_sampler, const double t_end,
    const std::string& name, const bool require_non_negative) {
  TENRYU_ASSERT(raw_sampler != nullptr,
                "create_frozen_time_table_from_sampler requires a valid sampler");
  const TimeSampler sampler{raw_sampler, name, require_non_negative};
  TENRYU_ASSERT(std::isfinite(t_end) && t_end >= 0.0,
                "create_frozen_time_table requires a finite t_end >= 0");
  // Base step 2^-40 s, doubled only while more than kFrozenTimeMaxBaseIntervals
  // base intervals would be needed (runs longer than about 0.95 us), so the
  // grid is independent of t_end below that length. k * h is exact (h is a
  // power of two and k < 2^53).
  double h = kFrozenTimeBaseStep_s;
  while (std::ceil(t_end / h) > static_cast<double>(kFrozenTimeMaxBaseIntervals)) {
    h *= 2.0;
  }
  const long long n_intervals =
      static_cast<long long>(std::max(std::ceil(t_end / h), 1.0));

  FrozenTable1D table;
  table.x.reserve(static_cast<std::size_t>(n_intervals) + 1U);
  table.y.reserve(static_cast<std::size_t>(n_intervals) + 1U);
  double t_prev = 0.0;
  double f_prev = sampler(t_prev);
  table.x.push_back(t_prev);
  table.y.push_back(f_prev);
  for (long long k = 1; k <= n_intervals; ++k) {
    const double t_next = static_cast<double>(k) * h;
    const double f_next = sampler(t_next);
    refine_time_interval(sampler, t_prev, f_prev, t_next, f_next, 0, table.x, table.y);
    table.x.push_back(t_next);
    table.y.push_back(f_next);
    t_prev = t_next;
    f_prev = f_next;
  }
  table.n_points = static_cast<int>(table.x.size());
  table.x_min = table.x.front();
  table.x_max = table.x.back();
  return table;
}

FrozenTable1D sum_frozen_tables(const std::vector<FrozenTable1D>& tables) {
  FrozenTable1D total;
  total.zero_outside = true;
  for (const auto& table : tables) {
    total.x.insert(total.x.end(), table.x.begin(), table.x.end());
  }
  std::sort(total.x.begin(), total.x.end());
  total.x.erase(std::unique(total.x.begin(), total.x.end()), total.x.end());
  total.y.assign(total.x.size(), 0.0);
  for (const auto& table : tables) {
    for (std::size_t k = 0; k < total.x.size(); ++k) {
      total.y[k] += table.eval(total.x[k]);
    }
  }
  total.n_points = static_cast<int>(total.x.size());
  if (total.n_points > 0) {
    total.x_min = total.x.front();
    total.x_max = total.x.back();
  }
  return total;
}

double integrate_frozen_table(const FrozenTable1D& table, const double t0, const double t1) {
  if (!(t1 > t0) || table.n_points <= 0) {
    return 0.0;
  }
  // Breakpoints inside (t0, t1) plus the ends; eval() is linear in between.
  std::vector<double> knots;
  knots.push_back(t0);
  for (const double x : table.x) {
    if (x > t0 && x < t1) {
      knots.push_back(x);
    }
  }
  knots.push_back(t1);
  double integral = 0.0;
  for (std::size_t k = 0; k + 1 < knots.size(); ++k) {
    const double a = knots[k];
    const double b = knots[k + 1];
    integral += 0.5 * (b - a) * (table.eval(a) + table.eval(b));
  }
  return integral;
}

void normalize_beam_power_table(FrozenTable1D& table, const double t_end, const double energy_J,
                                const char* path) {
  if (!(energy_J > 0.0)) {
    return;
  }
  const double computed_J = integrate_frozen_table(table, 0.0, t_end);
  if (!(std::isfinite(computed_J) && computed_J >= 1.0e-30)) {
    throw ConfigError(std::string(path) +
                      ".energy_J: zero-integral waveform cannot be normalized");
  }
  const double scale = energy_J / computed_J;
  for (double& y : table.y) {
    y *= scale;
  }
}

FrozenTableSummary summarize_frozen_table(const FrozenTable1D& table) {
  FrozenTableSummary summary;
  summary.n_points = table.n_points;
  TENRYU_ASSERT(table.n_points == static_cast<int>(table.x.size()),
                "summarize_frozen_table requires x.size()==n_points");
  TENRYU_ASSERT(table.n_points == static_cast<int>(table.y.size()),
                "summarize_frozen_table requires y.size()==n_points");
  if (table.n_points <= 0) {
    return summary;
  }

  summary.t_min = table.x.front();
  summary.t_max = table.x.back();
  summary.peak_value = *std::max_element(table.y.begin(), table.y.end());

  double integral = 0.0;
  for (int i = 0; i + 1 < table.n_points; ++i) {
    const std::size_t i0 = static_cast<std::size_t>(i);
    const std::size_t i1 = static_cast<std::size_t>(i + 1);
    const double dt = table.x[i1] - table.x[i0];
    integral += 0.5 * (table.y[i0] + table.y[i1]) * dt;
  }
  summary.integrated_value = integral;
  return summary;
}

#if TENRYU_ENABLE_PYTHON

namespace py = pybind11;

FrozenTable1D create_frozen_table(py::object callable,
                                  const double t_min,
                                  const double t_max,
                                  const int n_samples) {
  auto sampler = [callable = std::move(callable)](const double t) -> double {
    return py::cast<double>(callable(t));
  };
  return create_frozen_table_from_sampler(sampler, t_min, t_max, n_samples);
}

FrozenTable1D create_frozen_time_table(py::object callable, const double t_end,
                                       const std::string& name,
                                       const bool require_non_negative) {
  auto sampler = [callable = std::move(callable)](const double t) -> double {
    return py::cast<double>(callable(t));
  };
  return create_frozen_time_table_from_sampler(sampler, t_end, name, require_non_negative);
}

#endif

}  // namespace tenryu::core::namelist
