#pragma once

#include <string>

#include "verification/escape_valve_audit.hpp"

namespace tenryu::diagnostics {

void write_escape_valve_audit_history_file(
    const std::string& history_path,
    const tenryu::verification::EscapeValveAuditSnapshot& snapshot);

}  // namespace tenryu::diagnostics
