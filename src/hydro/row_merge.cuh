#pragma once

#include <cstdint>

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro {

struct RowMergeResult {
  bool engaged = false;      // preconditions met, transaction attempted
  bool committed = false;    // topology + state atomically installed
  int merged_pairs = 0;
  int skipped_pairs = 0;
  int first_bad_cell = -1;
  double delta_kinetic_total = 0.0;  // K_new - K_old [erg]
  double q_total = 0.0;                // kinetic variance deposited [erg]
  double mass_residual = 0.0;          // M_new - M_old [g]
  double momentum_r_residual = 0.0;    // P_r,new - P_r,old [g cm s^-1]
  double momentum_z_residual = 0.0;    // P_z,new - P_z,old [g cm s^-1]
  double total_energy_residual = 0.0;  // E_new - E_old [erg]
  const char* reject_reason = nullptr;  // static string; nullptr when committed
};

// A178/consult-22 §4: conservative tier-row merge. Attempts every cell of the
// failing cell's tier-row orbit (both N and S mirror belts at the same radius
// band) with its radially inner face-neighbor, at zero physical time,
// host-authoritative and deterministic. Non-subject pairs may be skipped by
// energy/momentum gates. Opt-in via TENRYU_I1B_ROW_MERGE. Returns
// committed=false with the state untouched on a transaction-wide rejection.
RowMergeResult apply_tier_row_merge(core::State& state,
                                    const core::Config& cfg,
                                    int failing_cell);

bool row_merge_enabled();  // env TENRYU_I1B_ROW_MERGE (non-empty, != "0")

}  // namespace tenryu::hydro
