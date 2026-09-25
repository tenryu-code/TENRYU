#pragma once

#include <functional>
#include <string>
#include <vector>

#if TENRYU_ENABLE_PYTHON
namespace pybind11 {
class object;
}
#endif

namespace tenryu::core::namelist {

struct FrozenTable1D {
  std::vector<double> x;
  std::vector<double> y;
  int n_points = 0;
  double x_min = 0.0;
  double x_max = 0.0;
  bool zero_outside = false;

  [[nodiscard]] double eval(double xi) const;
};

struct FrozenTableSummary {
  double t_min = 0.0;
  double t_max = 0.0;
  int n_points = 0;
  double peak_value = 0.0;
  double integrated_value = 0.0;
};

FrozenTable1D create_frozen_table_from_sampler(
    const std::function<double(double)>& sampler,
    double t_min,
    double t_max,
    int n_samples);

FrozenTableSummary summarize_frozen_table(const FrozenTable1D& table);

// Time callables (beam power, Marshak T_r(t), boundary pressure, hot-electron
// eta): samples at the absolute times t_k = k * h (h = kFrozenTimeBaseStep_s
// up to t_end of about 0.95 us, see kFrozenTimeMaxBaseIntervals) on
// [0, ceil(t_end / h) * h], plus dyadic midpoints wherever the chord misses
// the midpoint value by more than kFrozenTimeRelTol of the local magnitude,
// down to kFrozenTimeBaseStep_s / 2^kFrozenTimeMaxRefine. Every sample time
// and every refinement decision depends only on the callable near that time,
// so runs with different t_end get the same table where their ranges overlap.
inline constexpr double kFrozenTimeBaseStep_s = 9.094947017729282379150390625e-13;  // 2^-40 s
inline constexpr int kFrozenTimeMaxRefine = 7;                                      // to 2^-47 s
inline constexpr double kFrozenTimeRelTol = 1.0e-6;
// Beyond this many base intervals (t_end > 2^20 * 2^-40 s, about 0.95 us) the
// base step doubles until the count fits, and the grid then depends on t_end
// through that power of two.
inline constexpr long long kFrozenTimeMaxBaseIntervals = 1LL << 20;

// name labels the ConfigError of a non-finite (or, with require_non_negative,
// negative) sample: "Callable {name} returned non-finite value at t=...".
FrozenTable1D create_frozen_time_table_from_sampler(
    const std::function<double(double)>& sampler, double t_end,
    const std::string& name = "time callable", bool require_non_negative = false);

// Sum of tables on the sorted union of their abscissae (zero_outside = true),
// so each table's own refinement points are kept. Empty input -> empty table.
FrozenTable1D sum_frozen_tables(const std::vector<FrozenTable1D>& tables);

// Exact integral of the piecewise-linear table over [t0, t1] (t0 <= t1),
// following eval() outside the tabulated range.
double integrate_frozen_table(const FrozenTable1D& table, double t0, double t1);

// Beam power table [W]: rescaled by energy_J / integral over [0, t_end] when
// energy_J > 0 (LaserBeam.energy_J [J]); ConfigError when that integral is
// below 1e-30 J. energy_J <= 0 leaves the table unchanged.
void normalize_beam_power_table(FrozenTable1D& table, double t_end, double energy_J,
                                const char* path);

#if TENRYU_ENABLE_PYTHON
FrozenTable1D create_frozen_table(pybind11::object callable,
                                  double t_min,
                                  double t_max,
                                  int n_samples);

FrozenTable1D create_frozen_time_table(pybind11::object callable, double t_end,
                                       const std::string& name = "time callable",
                                       bool require_non_negative = false);
#endif

}  // namespace tenryu::core::namelist
