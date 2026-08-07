#pragma once

#include <limits>
#include <string>
#include <vector>

#include "core/state.hpp"

namespace tenryu::diagnostics {

struct OperatorResidualSnapshot {
  double E_thermal_total = 0.0;
  double E_kinetic_total = 0.0;
  double E_kinetic_total_nodal = 0.0;
  double timestamp = 0.0;
  bool valid = false;
};

OperatorResidualSnapshot capture_total_energy(const core::State& state);

struct OperatorResidualEntry {
  std::string operator_name;
  double residual = std::numeric_limits<double>::quiet_NaN();
  double E_before = 0.0;
  double E_after = 0.0;
  double delta_E_ext = 0.0;
  bool valid = false;
  int step = -1;
  double time = 0.0;
};

class OperatorEnergyTracker {
 public:
  void enable(bool yes) { enabled_ = yes; }
  [[nodiscard]] bool enabled() const { return enabled_; }

  void record(const std::string& op_name,
              const OperatorResidualSnapshot& before,
              const OperatorResidualSnapshot& after,
              double delta_E_ext,
              int step,
              double time);

  [[nodiscard]] const std::vector<OperatorResidualEntry>& entries() const {
    return entries_;
  }
  void clear() { entries_.clear(); }

 private:
  bool enabled_ = false;
  std::vector<OperatorResidualEntry> entries_;
};

}  // namespace tenryu::diagnostics
