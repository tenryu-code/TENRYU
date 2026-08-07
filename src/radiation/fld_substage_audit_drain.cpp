#include "radiation/fld_substage_audit_drain.hpp"

#include "radiation/fld_2d_rz_gpu.cuh"

namespace tenryu::radiation {

std::vector<diagnostics::FldSubstageAuditRecord>& fld_substage_audit_batch() {
  thread_local std::vector<diagnostics::FldSubstageAuditRecord> records;
  return records;
}

std::vector<diagnostics::FldSubstageAuditRecord>
drain_fld_substage_audit_records() {
  auto& batch = fld_substage_audit_batch();
  std::vector<diagnostics::FldSubstageAuditRecord> drained;
  drained.swap(batch);
  return drained;
}

}  // namespace tenryu::radiation
