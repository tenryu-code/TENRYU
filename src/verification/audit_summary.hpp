#pragma once

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tenryu::verification {

struct AuditScalarResiduals {
  std::optional<double> mass;
  std::optional<double> r_momentum;
  std::optional<double> z_momentum;
  std::optional<double> energy;
};

struct AuditPositivityMinimum {
  std::optional<double> rho;
  std::optional<double> p;
  std::optional<double> T_e;
  std::optional<double> T_i;
  std::optional<double> E_r;
  std::optional<double> kappa;
};

struct AuditSummary {
  std::string schema_version = "tenryu.audit_summary.v1";
  bool production_audit_enabled = false;
  std::string tier = "none";
  std::string audit_status = "NOT_REQUESTED";
  std::string termination_reason = "unknown";
  AuditScalarResiduals conservation_residual_max;
  std::optional<AuditPositivityMinimum> positivity_min;
  std::map<std::string, double> solver_stats;
  std::map<std::string, bool> escape_valve_firings;
  std::map<std::string, double> profile_l2_error;
  std::map<std::string, double> symmetry_residual;
  std::optional<double> vol_closure_residual;
  std::vector<std::string> missing_required_data;
};

[[nodiscard]] bool is_valid_audit_tier(const std::string& tier);
[[nodiscard]] std::string to_json(const AuditSummary& summary);

}  // namespace tenryu::verification
