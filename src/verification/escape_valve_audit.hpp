#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"

namespace tenryu::verification {

inline constexpr std::size_t kEscapeValveFlagCount = 6;

enum class EscapeValveFlag : std::uint8_t {
  EmergencyCellDeactivation = 0,
  MultiNodeBoundaryRepair = 1,
  MultiNodeInteriorRepair = 2,
  AxisVariationalProjection = 3,
  LocalBoundaryRepair = 4,
  RetryActiveMeshRepair = 5,
};

inline constexpr std::array<const char*, kEscapeValveFlagCount>
    kEscapeValveFlagNames = {
        "emergency_cell_deactivation_count",
        "multi_node_boundary_repair_count",
        "multi_node_interior_repair_count",
        "axis_variational_projection_count",
        "local_boundary_repair_count",
        "retry_active_mesh_repair_count",
};

struct EscapeValveAuditCellState {
  double rho = 0.0;
  double p = 0.0;
  double Te = 0.0;
  double Ti = 0.0;
};

struct EscapeValveAuditEvent {
  EscapeValveFlag flag = EscapeValveFlag::EmergencyCellDeactivation;
  std::string flag_name;
  std::string reason;
  int cell_id = -1;
  int cell_i = -1;
  int cell_j = -1;
  int step = -1;
  double time = 0.0;
  double mass_delta = 0.0;
  double momentum_delta_r = 0.0;
  double momentum_delta_z = 0.0;
  double energy_delta = 0.0;
  EscapeValveAuditCellState before_state;
  EscapeValveAuditCellState after_state;
};

struct EscapeValveAuditDecision {
  bool fatal = false;
  std::string audit_status = "PASS";
  std::string reason;
};

struct EscapeValveAuditSnapshot {
  bool enabled = false;
  std::string tier = "none";
  std::string audit_status = "NOT_REQUESTED";
  int step = -1;
  double time = 0.0;
  std::array<std::int64_t, kEscapeValveFlagCount> counts{};
  double mass_delta_abs_cumulative = 0.0;
  double energy_delta_abs_cumulative = 0.0;

  [[nodiscard]] std::int64_t total_count() const {
    std::int64_t total = 0;
    for (const std::int64_t count : counts) {
      total += count;
    }
    return total;
  }
};

[[nodiscard]] inline const char* escape_valve_flag_name(
    const EscapeValveFlag flag) {
  const auto idx = static_cast<std::size_t>(flag);
  if (idx >= kEscapeValveFlagNames.size()) {
    return "unknown_escape_valve_count";
  }
  return kEscapeValveFlagNames[idx];
}

[[nodiscard]] inline std::optional<EscapeValveFlag> escape_valve_flag_from_name(
    const std::string_view value) {
  for (std::size_t i = 0; i < kEscapeValveFlagNames.size(); ++i) {
    if (value == kEscapeValveFlagNames[i]) {
      return static_cast<EscapeValveFlag>(i);
    }
  }
  if (value == "EmergencyCellDeactivation") {
    return EscapeValveFlag::EmergencyCellDeactivation;
  }
  if (value == "MultiNodeBoundaryRepair" ||
      value == "BoundaryPatchProjection") {
    return EscapeValveFlag::MultiNodeBoundaryRepair;
  }
  if (value == "InteriorMultiNodeProjection") {
    return EscapeValveFlag::MultiNodeInteriorRepair;
  }
  if (value == "AxisVariationalProjection") {
    return EscapeValveFlag::AxisVariationalProjection;
  }
  if (value == "LocalBoundaryRepair") {
    return EscapeValveFlag::LocalBoundaryRepair;
  }
  if (value == "HydroDriverRetryActiveMeshRepair") {
    return EscapeValveFlag::RetryActiveMeshRepair;
  }
  return std::nullopt;
}

class EscapeValveAudit {
 public:
  using ProductionAuditConfig =
      tenryu::core::Config::NumericsConfig::DiagnosticsConfig::
          ProductionAuditConfig;

  void initialize(const ProductionAuditConfig& cfg,
                  const std::string& output_dir,
                  int mesh_nz,
                  bool emit_files = true) {
    *this = EscapeValveAudit{};
    if (!cfg.enabled) {
      return;
    }

    enabled_ = true;
    tier_ = cfg.tier;
    mass_budget_max_ = cfg.escape_valve_budget.mass_max;
    energy_budget_max_ = cfg.escape_valve_budget.energy_max;
    mesh_nz_ = mesh_nz;
    regions_.reserve(cfg.region_of_interest.size());
    for (const auto& region : cfg.region_of_interest) {
      regions_.push_back(
          Region{region.i_min, region.i_max, region.j_min, region.j_max});
    }

    snapshot_.enabled = true;
    snapshot_.tier = tier_;
    snapshot_.audit_status = "RUNNING";

    if (emit_files) {
      jsonl_path_ =
          (std::filesystem::path(output_dir) / "escape_valve_events.jsonl")
              .string();
    }
    if (!jsonl_path_.empty()) {
      const std::filesystem::path path(jsonl_path_);
      if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
      }
      std::ofstream os(path, std::ios::out | std::ios::trunc);
      TENRYU_ASSERT(os.good(),
                    "Failed to initialize escape_valve_events.jsonl");
    }
  }

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }

  [[nodiscard]] const EscapeValveAuditSnapshot& snapshot() const noexcept {
    return snapshot_;
  }

  EscapeValveAuditDecision record_event(EscapeValveAuditEvent event) {
    EscapeValveAuditDecision decision;
    if (!enabled_) {
      return decision;
    }

    if (const auto parsed = escape_valve_flag_from_name(event.flag_name)) {
      event.flag = *parsed;
    }
    if (event.flag_name.empty()) {
      event.flag_name = escape_valve_flag_name(event.flag);
    }
    if (event.cell_i < 0 || event.cell_j < 0) {
      derive_cell_ij(event);
    }

    const auto idx = static_cast<std::size_t>(event.flag);
    TENRYU_ASSERT(idx < snapshot_.counts.size(),
                  "Invalid escape-valve audit flag");
    snapshot_.counts[idx] += 1;
    snapshot_.step = event.step;
    snapshot_.time = event.time;
    if (std::isfinite(event.mass_delta)) {
      snapshot_.mass_delta_abs_cumulative += std::abs(event.mass_delta);
    }
    if (std::isfinite(event.energy_delta)) {
      snapshot_.energy_delta_abs_cumulative += std::abs(event.energy_delta);
    }

    append_jsonl(event);

    if (tier_ == "B") {
      if (event_in_region_of_interest(event)) {
        decision.fatal = true;
        decision.audit_status = "TIER_B_FAIL_ESCAPE_VALVE_ROI";
        decision.reason =
            "escape-valve firing inside production_audit.region_of_interest";
      } else if (snapshot_.mass_delta_abs_cumulative > mass_budget_max_) {
        decision.fatal = true;
        decision.audit_status =
            "TIER_B_FAIL_ESCAPE_VALVE_MASS_BUDGET";
        decision.reason =
            "escape-valve cumulative absolute mass delta exceeded budget";
      } else if (snapshot_.energy_delta_abs_cumulative > energy_budget_max_) {
        decision.fatal = true;
        decision.audit_status =
            "TIER_B_FAIL_ESCAPE_VALVE_ENERGY_BUDGET";
        decision.reason =
            "escape-valve cumulative absolute energy delta exceeded budget";
      }
    }

    snapshot_.audit_status = decision.fatal ? decision.audit_status : "RUNNING";
    return decision;
  }

  EscapeValveAuditDecision finalize(const int step, const double time) {
    EscapeValveAuditDecision decision;
    if (!enabled_) {
      return decision;
    }
    snapshot_.step = step;
    snapshot_.time = time;
    if (tier_ == "A" && snapshot_.total_count() > 0) {
      decision.fatal = true;
      decision.audit_status = "TIER_A_FAIL_ESCAPE_VALVE_FIRED";
      decision.reason = "Tier-A production audit forbids escape-valve firing";
    }
    snapshot_.audit_status = decision.fatal ? decision.audit_status : "PASS";
    return decision;
  }

 private:
  struct Region {
    int i_min = 0;
    int i_max = 0;
    int j_min = 0;
    int j_max = 0;
  };

  static void append_json_number(std::ostream& os, const double value) {
    if (std::isfinite(value)) {
      os << std::setprecision(17) << value;
    } else {
      os << "null";
    }
  }

  static std::string json_escape(const std::string& input) {
    std::string out;
    out.reserve(input.size());
    for (const char ch : input) {
      switch (ch) {
        case '\\':
          out += "\\\\";
          break;
        case '"':
          out += "\\\"";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        default:
          out += ch;
          break;
      }
    }
    return out;
  }

  void derive_cell_ij(EscapeValveAuditEvent& event) const {
    if (event.cell_id < 0 || mesh_nz_ <= 0) {
      return;
    }
    event.cell_i = event.cell_id / mesh_nz_;
    event.cell_j = event.cell_id - event.cell_i * mesh_nz_;
  }

  [[nodiscard]] bool event_in_region_of_interest(
      const EscapeValveAuditEvent& event) const {
    if (event.cell_i < 0 || event.cell_j < 0) {
      return false;
    }
    for (const Region& region : regions_) {
      if (event.cell_i >= region.i_min && event.cell_i <= region.i_max &&
          event.cell_j >= region.j_min && event.cell_j <= region.j_max) {
        return true;
      }
    }
    return false;
  }

  void append_state_json(std::ostream& os,
                         const EscapeValveAuditCellState& state) const {
    os << "{\"rho\":";
    append_json_number(os, state.rho);
    os << ",\"p\":";
    append_json_number(os, state.p);
    os << ",\"Te\":";
    append_json_number(os, state.Te);
    os << ",\"Ti\":";
    append_json_number(os, state.Ti);
    os << "}";
  }

  void append_jsonl(const EscapeValveAuditEvent& event) const {
    if (jsonl_path_.empty()) {
      return;
    }
    std::ofstream os(jsonl_path_, std::ios::out | std::ios::app);
    TENRYU_ASSERT(os.good(), "Failed to open escape_valve_events.jsonl");
    os << "{\"cell_id\":" << event.cell_id;
    os << ",\"time\":";
    append_json_number(os, event.time);
    os << ",\"flag_name\":\"" << json_escape(event.flag_name) << "\"";
    os << ",\"reason\":\"" << json_escape(event.reason) << "\"";
    os << ",\"mass_delta\":";
    append_json_number(os, event.mass_delta);
    os << ",\"momentum_delta_r\":";
    append_json_number(os, event.momentum_delta_r);
    os << ",\"momentum_delta_z\":";
    append_json_number(os, event.momentum_delta_z);
    os << ",\"energy_delta\":";
    append_json_number(os, event.energy_delta);
    os << ",\"before_state\":";
    append_state_json(os, event.before_state);
    os << ",\"after_state\":";
    append_state_json(os, event.after_state);
    os << "}\n";
    TENRYU_ASSERT(os.good(), "Failed while writing escape_valve_events.jsonl");
  }

  bool enabled_ = false;
  std::string tier_ = "none";
  double mass_budget_max_ = 0.0;
  double energy_budget_max_ = 0.0;
  int mesh_nz_ = -1;
  std::string jsonl_path_;
  std::vector<Region> regions_;
  EscapeValveAuditSnapshot snapshot_{};
};

}  // namespace tenryu::verification
