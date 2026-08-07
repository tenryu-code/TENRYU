#pragma once

#include <string>

#include "verification/positivity_tracker.hpp"

namespace tenryu::diagnostics {

void write_positivity_history_file(
    const std::string& history_path,
    const tenryu::verification::PositivityScanResult& result);

}  // namespace tenryu::diagnostics
