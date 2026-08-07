#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include "core/config.hpp"

namespace tenryu::core {
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::verification {

inline constexpr std::size_t kPositivityFieldCount = 6;

enum class PositivityField : std::uint8_t {
  Rho = 0,
  Pressure = 1,
  ElectronTemperature = 2,
  IonTemperature = 3,
  RadiationEnergy = 4,
  Kappa = 5,
};

inline constexpr std::array<const char*, kPositivityFieldCount>
    kPositivityFieldNames = {
        "rho",
        "p",
        "T_e",
        "T_i",
        "E_r",
        "kappa",
};

inline constexpr std::array<const char*, kPositivityFieldCount>
    kPositivityMinDatasetNames = {
        "rho_min",
        "p_min",
        "T_e_min",
        "T_i_min",
        "E_r_min",
        "kappa_min",
};

[[nodiscard]] const char* positivity_field_name(PositivityField field);
[[nodiscard]] const char* positivity_min_dataset_name(PositivityField field);

struct PositivityCellState {
  bool valid = false;
  int cell_id = -1;
  int cell_i = -1;
  int cell_j = -1;
  double rho = 0.0;
  double p = 0.0;
  double T_e = 0.0;
  double T_i = 0.0;
  double E_r = 0.0;
  double kappa = 0.0;
};

struct PositivityViolation {
  bool valid = false;
  PositivityField field = PositivityField::Rho;
  int cell_id = -1;
  int cell_i = -1;
  int cell_j = -1;
  double value = 0.0;
  PositivityCellState cell_state;
  std::array<PositivityCellState, 4> neighbors{};
};

struct PositivityScanResult {
  PositivityScanResult() {
    minimum.fill(std::numeric_limits<double>::infinity());
  }

  bool enabled = false;
  bool has_negative = false;
  bool local_has_negative = false;
  int step = -1;
  double time = 0.0;
  std::array<double, kPositivityFieldCount> minimum;
  std::array<bool, kPositivityFieldCount> has_violation{};
  std::array<PositivityViolation, kPositivityFieldCount> violations{};
};

struct PositivityAuditDecision {
  bool fatal = false;
  std::string audit_status = "PASS";
  std::string reason;
};

class PositivityTracker {
 public:
  using ProductionAuditConfig =
      tenryu::core::Config::NumericsConfig::DiagnosticsConfig::
          ProductionAuditConfig;

  void initialize(const ProductionAuditConfig& cfg, bool emit_logs = true);

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] const std::string& audit_status() const noexcept {
    return audit_status_;
  }
  [[nodiscard]] std::int64_t negative_event_count() const noexcept {
    return negative_event_count_;
  }
  [[nodiscard]] int kernel_launch_count() const noexcept {
    return kernel_launch_count_;
  }
  [[nodiscard]] int allocation_count() const noexcept {
    return allocation_count_;
  }

  [[nodiscard]] PositivityScanResult scan(
      const tenryu::core::State& state,
      int step,
      double time,
      int n_groups,
      const tenryu::parallel::Reduction* reduction = nullptr);

  PositivityAuditDecision record_scan(const PositivityScanResult& result);
  PositivityAuditDecision finalize(int step, double time);

 private:
  bool enabled_ = false;
  bool fatal_on_neg_ = false;
  bool emit_logs_ = true;
  std::string tier_ = "none";
  std::string audit_status_ = "NOT_REQUESTED";
  std::int64_t negative_event_count_ = 0;
  int kernel_launch_count_ = 0;
  int allocation_count_ = 0;
};

}  // namespace tenryu::verification
