#include "verification/escape_valve_audit.hpp"

namespace tenryu::verification {

static_assert(kEscapeValveFlagCount == kEscapeValveFlagNames.size(),
              "escape-valve flag name table size mismatch");

}  // namespace tenryu::verification
