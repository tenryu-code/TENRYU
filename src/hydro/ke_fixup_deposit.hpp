#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro::ale {

// PR3 (corner-mass basis verdict, interim rank-2): incremental KE-fixup
// deposit. Per velocity remap, the budget/dynamics basis (frozen
// state.corner_mass nodal sums M^F) and the remap's own first-moment
// basis (M^{B,-} pre / M^{B,+} post) disagree, so each velocity change
// carries a basis-weighted KE residual no ledger compensates (the
// adjudicated -1%..+21% terminal-phase drift). This deposits that residual,
// SIGNED, into cell internal energy so the energy budget closes. The
// momentum defect remains — this is the verdict's interim measure, not
// the basis-contract fix (PR5/6).
//
// Node residual. The verdict's compact form
//   R_n = 1/2 M_n^F (|v+|^2-|v-|^2) - 1/2 (M_n^{B,+}|v+|^2 - M_n^{B,-}|v-|^2)
// localizes the global leak only up to the remap's INTERNAL KE transport
// (for a uniform velocity field it is nonzero node-by-node, summing to
// zero). A deposit must not spray that zero-sum transport noise into
// cells: folding in the convected KE at the midpoint (1/4(|v-|^2+|v+|^2)
// per unit transported mass) collapses the formula to
//   R_n = 1/2 [ M_n^F - 1/2 (M_n^{B,+} + M_n^{B,-}) ] (|v+|^2 - |v-|^2),
// which has the same global content up to second-order transport terms,
// vanishes identically for identity remaps AND uniform flow (the DeBar
// consistency the verdict itself demands of remap machinery), and is
// nonzero exactly where a velocity actually changed against a basis
// mismatch. R_global below is the sum of THIS R_n.
struct KeFixupInput {
  int n_cells = 0;
  int n_nodes = 0;
  const int* cell_node_csr_offsets = nullptr;  // 4 slots per cell
  const int* cell_node_csr_indices = nullptr;
  // Frozen subzonal basis (state.corner_mass, n_cells*4): INCLUDES macro
  // member mirrors, exactly as the dynamics' nodal masses do.
  const double* corner_mass_frozen = nullptr;
  // First-moment basis, n_cells*4 each: pre = host-built on the pre-remap
  // mesh with inactive members EXCLUDED (basis_defect_capture_pre
  // convention); post = the Option-B remap's transported buffer (zeros on
  // inactive members).
  const double* corner_mass_b_pre = nullptr;
  const double* corner_mass_b_post = nullptr;
  const double* v_r_pre = nullptr;   // n_nodes
  const double* v_z_pre = nullptr;
  const double* v_r_post = nullptr;
  const double* v_z_post = nullptr;
  // Per-cell internal energy U_c [erg] (capacity basis for the chi guard).
  const double* cell_internal_energy = nullptr;
  // Macro inactive members never receive deposits; their frozen mirror
  // corners are excluded from the deposit weights (active renormalization).
  const std::uint8_t* inactive_mask = nullptr;  // may be nullptr
  // |dU_c| <= chi * U_c hard guard (env default 0.1).
  double chi = 0.1;
};

struct KeFixupResult {
  bool valid = false;
  // Per-cell signed deposit dU_c [erg]; zeros when abandoned.
  std::vector<double> dU;
  double R_global = 0.0;          // sum_n R_n
  double deposited_total = 0.0;   // sum_c dU_c (== -R_global unless abandoned)
  long long clipped_cells = 0;    // cells that hit the chi capacity
  int redistribution_rounds = 0;
  // Residual |R_n| at nodes with no active receiving cell (expected ~0:
  // macro-interior nodes have frozen velocities).
  double undepositable_abs = 0.0;
  // True when capacity (or receivers) could not absorb the residual:
  // NOTHING is deposited (never floor-and-lose-conservation).
  bool abandoned = false;
};

KeFixupResult compute_ke_fixup_deposit(const KeFixupInput& in);

// Gap-form compensation (the corrected per-install/remap quantity from
// the 2026-06-12 naive-jump refutation): the budget's un-ledgered KE
// change per remap+install cycle is the change of the gap-weighted KE,
//   R_gap = 1/2 sum_n [ G_post(n) |v+|^2 - G_pre(n) |v-|^2 ],
//   G = M^vp - M^fm  (V-paired minus first-moment nodal sums),
// because the total-energy remap already credits the KE of its own
// (first-moment) basis transport. Globally exact-zero for uniform flow
// (all four bases sum to the same total mass). Deposit -R_gap with the
// shared PR3 machinery (weights = the V-paired post basis).
// env TENRYU_I1B_GAP_FORM_COMPENSATION: "audit" (or "2") = compute+log
// only; truthy = deposit. Returns 0 off / 1 deposit / 2 audit-only.
int gap_form_compensation_mode();

struct GapFormInput {
  int n_cells = 0;
  int n_nodes = 0;
  const int* cell_node_csr_offsets = nullptr;
  const int* cell_node_csr_indices = nullptr;
  // V-paired bases are mirror-INCLUSIVE (the budget divides by them);
  // first-moment bases exclude inactive members (zeros), matching the
  // basis-defect audit conventions.
  const double* corner_mass_vp_pre = nullptr;   // state.corner_mass at remap start
  const double* corner_mass_vp_post = nullptr;  // install candidate (composite)
  const double* corner_mass_fm_pre = nullptr;   // host-built pre-mesh first moments
  const double* corner_mass_fm_post = nullptr;  // OptionB transported buffer
  const double* v_r_pre = nullptr;
  const double* v_z_pre = nullptr;
  const double* v_r_post = nullptr;
  const double* v_z_post = nullptr;
  const double* cell_internal_energy = nullptr;
  const std::uint8_t* inactive_mask = nullptr;
  double chi = 0.1;
};

KeFixupResult compute_gap_form_deposit(const GapFormInput& in);

// env TENRYU_I1B_KE_FIXUP_DEPOSIT=1 (+ TENRYU_I1B_KE_FIXUP_DEPOSIT_CHI,
// default 0.1). Default-off: bit-identical production behavior.
bool ke_fixup_deposit_enabled();
double ke_fixup_deposit_chi();

// Hook-level applier (called from the velocity-remap bracket in
// ale_remap_2d_rz.cu with the SAME pre-captured data the basis-defect
// audit uses). Applies the deposit to state.ee/ei with the recover-stage
// ye split (ye = clamp01(e_e/(e_e+e_i)), 0.5 on degenerate), which keeps
// both species non-negative for chi <= 1 by construction. Logs one
// [ke_fixup_deposit] line. No-op (no state writes) when the kernel
// abandons.
void ke_fixup_apply_deposit(core::State& state,
                            const std::vector<double>& corner_mass_b_pre,
                            const std::vector<double>& corner_mass_b_post,
                            const std::vector<double>& v_r_pre,
                            const std::vector<double>& v_z_pre);

}  // namespace tenryu::hydro::ale
