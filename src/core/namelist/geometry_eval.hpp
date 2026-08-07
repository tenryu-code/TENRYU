#pragma once

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::core::namelist {

class Builder;

struct ScalarRangeSummary {
  double min = 0.0;
  double max = 0.0;
  double mean = 0.0;
  bool valid = false;
};

struct GeometrySummary {
  ScalarRangeSummary rho;
  ScalarRangeSummary Te;
  ScalarRangeSummary Ti;
  std::map<std::string, double> material_volume;
  int interface_cells_observed = 0;
};

struct GeometryCallables {
  std::function<double(double, double)> rho;
  std::function<double(double, double)> Te;
  std::function<double(double, double)> Ti;
  std::function<std::array<double, 2>(double, double)> velocity;
  std::vector<std::function<double(double, double)>> volfrac;
};

GeometrySummary evaluate_geometry_from_callables(
    const Config& cfg,
    State& state,
    const GeometryCallables& callables);

#if TENRYU_ENABLE_PYTHON
GeometrySummary evaluate_geometry(const Config& cfg,
                                  const Builder& builder,
                                  State& state);
#endif

void evaluate_geometry(const Config& cfg, State& state);

}  // namespace tenryu::core::namelist
