#pragma once

#include <functional>
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

#if TENRYU_ENABLE_PYTHON
FrozenTable1D create_frozen_table(pybind11::object callable,
                                  double t_min,
                                  double t_max,
                                  int n_samples);
#endif

}  // namespace tenryu::core::namelist
