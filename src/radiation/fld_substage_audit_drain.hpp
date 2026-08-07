#pragma once

#include <vector>

#include "diagnostics/radial_fourier_audit.hpp"

namespace tenryu::radiation {

std::vector<diagnostics::FldSubstageAuditRecord>& fld_substage_audit_batch();

}  // namespace tenryu::radiation
