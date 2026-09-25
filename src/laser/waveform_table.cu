#include "laser/waveform_table.cuh"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <limits>

namespace tenryu::laser {

double waveform_power_at_time(const core::namelist::FrozenTable1D& table,
                              const double t) {
  return table.eval(t);
}

double waveform_integral(const core::namelist::FrozenTable1D& table,
                         const double t0,
                         const double t1) {
  if (!(t1 > t0) || table.n_points <= 0) {
    return 0.0;
  }
  const int n = table.n_points;
  if (n == 1) {
    // eval() returns y for every argument unless zero_outside, in which case
    // only the single abscissa (measure zero) carries the value.
    return table.zero_outside ? 0.0 : table.y.front() * (t1 - t0);
  }
  double sum = 0.0;
  double a = t0;
  double b = t1;
  // Constant (or zero) extension left of the table.
  if (a < table.x_min) {
    const double edge = std::min(b, table.x_min);
    if (!table.zero_outside) {
      sum += table.y.front() * (edge - a);
    }
    a = edge;
  }
  // Constant (or zero) extension right of the table.
  if (b > table.x_max) {
    const double edge = std::max(a, table.x_max);
    if (!table.zero_outside) {
      sum += table.y.back() * (b - edge);
    }
    b = edge;
  }
  if (!(b > a)) {
    return sum;
  }
  // Interior: exact trapezoids over the linear pieces overlapping [a, b].
  const auto first = std::upper_bound(table.x.begin(), table.x.end(), a);
  int lo = static_cast<int>(std::distance(table.x.begin(), first)) - 1;
  lo = std::clamp(lo, 0, n - 2);
  double x_start = a;
  while (x_start < b && lo <= n - 2) {
    const double x0 = table.x[static_cast<std::size_t>(lo)];
    const double x1 = table.x[static_cast<std::size_t>(lo + 1)];
    const double y0 = table.y[static_cast<std::size_t>(lo)];
    const double y1 = table.y[static_cast<std::size_t>(lo + 1)];
    const double seg_end = std::min(x1, b);
    if (seg_end > x_start) {
      const double dx = x1 - x0;
      const auto value = [&](const double xx) {
        return (dx > std::numeric_limits<double>::epsilon())
                   ? y0 + (xx - x0) / dx * (y1 - y0)
                   : y0;
      };
      sum += 0.5 * (value(x_start) + value(seg_end)) * (seg_end - x_start);
    }
    x_start = std::max(x_start, seg_end);
    ++lo;
  }
  return sum;
}

double waveform_average_power(const core::namelist::FrozenTable1D& table,
                              const double t0,
                              const double t1) {
  if (!(t1 > t0)) {
    return table.eval(t0);
  }
  return waveform_integral(table, t0, t1) / (t1 - t0);
}

}  // namespace tenryu::laser
