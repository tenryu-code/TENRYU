#include "verification/audit_summary.hpp"

#include <cmath>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace tenryu::verification {
namespace {

std::string escape_json_string(const std::string& input) {
  std::ostringstream out;
  for (const unsigned char ch : input) {
    switch (ch) {
      case '"':
        out << "\\\"";
        break;
      case '\\':
        out << "\\\\";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
              << static_cast<int>(ch) << std::dec << std::setfill(' ');
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

void write_string(std::ostream& os, const std::string& value) {
  os << '"' << escape_json_string(value) << '"';
}

void write_double_or_null(std::ostream& os, const std::optional<double> value) {
  if (!value.has_value() || !std::isfinite(*value)) {
    os << "null";
    return;
  }
  os << std::setprecision(17) << *value;
}

void write_double_value_or_null(std::ostream& os, const double value) {
  if (!std::isfinite(value)) {
    os << "null";
    return;
  }
  os << std::setprecision(17) << value;
}

void write_string_double_map(std::ostream& os,
                             const std::map<std::string, double>& values,
                             const char* indent) {
  os << "{";
  if (!values.empty()) {
    bool first = true;
    for (const auto& [key, value] : values) {
      os << (first ? "\n" : ",\n") << indent << "  ";
      write_string(os, key);
      os << ": ";
      write_double_value_or_null(os, value);
      first = false;
    }
    os << "\n" << indent;
  }
  os << "}";
}

void write_string_bool_map(std::ostream& os,
                           const std::map<std::string, bool>& values,
                           const char* indent) {
  os << "{";
  if (!values.empty()) {
    bool first = true;
    for (const auto& [key, value] : values) {
      os << (first ? "\n" : ",\n") << indent << "  ";
      write_string(os, key);
      os << ": " << (value ? "true" : "false");
      first = false;
    }
    os << "\n" << indent;
  }
  os << "}";
}

void write_string_vector(std::ostream& os,
                         const std::vector<std::string>& values,
                         const char* indent) {
  os << "[";
  if (!values.empty()) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      os << (i == 0 ? "\n" : ",\n") << indent << "  ";
      write_string(os, values[i]);
    }
    os << "\n" << indent;
  }
  os << "]";
}

}  // namespace

bool is_valid_audit_tier(const std::string& tier) {
  return tier == "A" || tier == "B" || tier == "none";
}

std::string to_json(const AuditSummary& summary) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"schema_version\": ";
  write_string(os, summary.schema_version);
  os << ",\n  \"production_audit_enabled\": "
     << (summary.production_audit_enabled ? "true" : "false");
  os << ",\n  \"tier\": ";
  write_string(os, summary.tier);
  os << ",\n  \"termination_reason\": ";
  write_string(os, summary.termination_reason);
  os << ",\n  \"conservation_residual_max\": {\n";
  os << "    \"mass\": ";
  write_double_or_null(os, summary.conservation_residual_max.mass);
  os << ",\n    \"R-momentum\": ";
  write_double_or_null(os, summary.conservation_residual_max.r_momentum);
  os << ",\n    \"Z-momentum\": ";
  write_double_or_null(os, summary.conservation_residual_max.z_momentum);
  os << ",\n    \"energy\": ";
  write_double_or_null(os, summary.conservation_residual_max.energy);
  os << "\n  },\n  \"positivity_min\": ";
  if (summary.positivity_min.has_value()) {
    const AuditPositivityMinimum& min = *summary.positivity_min;
    os << "{\n";
    os << "    \"rho\": ";
    write_double_or_null(os, min.rho);
    os << ",\n    \"p\": ";
    write_double_or_null(os, min.p);
    os << ",\n    \"T_e\": ";
    write_double_or_null(os, min.T_e);
    os << ",\n    \"T_i\": ";
    write_double_or_null(os, min.T_i);
    os << ",\n    \"E_r\": ";
    write_double_or_null(os, min.E_r);
    os << ",\n    \"kappa\": ";
    write_double_or_null(os, min.kappa);
    os << "\n  }";
  } else {
    os << "null";
  }
  os << ",\n  \"solver_stats\": ";
  write_string_double_map(os, summary.solver_stats, "  ");
  os << ",\n  \"escape_valve_firings\": ";
  write_string_bool_map(os, summary.escape_valve_firings, "  ");
  os << ",\n  \"profile_l2_error\": ";
  write_string_double_map(os, summary.profile_l2_error, "  ");
  os << ",\n  \"symmetry_residual\": ";
  write_string_double_map(os, summary.symmetry_residual, "  ");
  os << ",\n  \"gcl_residual\": ";
  write_double_or_null(os, summary.vol_closure_residual);
  os << ",\n  \"audit_status\": ";
  write_string(os, summary.audit_status);
  os << ",\n  \"missing_required_data\": ";
  write_string_vector(os, summary.missing_required_data, "  ");
  os << "\n}\n";
  return os.str();
}

}  // namespace tenryu::verification
