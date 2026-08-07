#include "core/namelist/frozen_table.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>

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

#endif

}  // namespace tenryu::core::namelist
