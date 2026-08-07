#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::hydro::ale {

// PR4 (corner-mass basis verdict): LO positive subzonal corner-mass remap,
// AUDIT-ONLY. Transports a subzonal corner-mass field through one remap
// using the remap's own per-face cell-mass fluxes (LO donor density x
// outward swept volume, the production donor convention incl. the legacy
// sign quirk), distributes each face flux to the face's two corners on
// each side, then repairs every cell to the post-remap cell mass by a
// small positivity-constrained projection. It never writes state; PR5/6
// decide whether this basis replaces the frozen one.
//
// LO corner split choice: each face flux is split 1/2-1/2 onto the face's
// two endpoint corners (the task's option 1). Rationale: the split only
// shapes the SUB-cell distribution — the per-cell totals are re-reconciled
// against the actual post-remap cell masses by the projection — and the
// equal split is exactly face-antisymmetric (the same half leaves corner
// (loser,k) and enters corner (gainer,k')) and positivity-friendly.
// Sub-sweep-weighted splitting (splitting the face's swept quad at the
// face midpoint) is the PR5 refinement; exact uniform-density corner
// transport additionally needs the internal-median sweeps of the full
// staggered (Kucharik-Shashkov) remap, out of scope here.
struct CornerMassRemapLoInput {
  int n_cells = 0;
  int n_nodes = 0;
  // Cell->node CSR, 4 slots per cell (slot k = corner k); tri cells carry
  // 3 active slots per cell_nverts.
  const int* cell_node_csr_offsets = nullptr;
  const int* cell_node_csr_indices = nullptr;
  const std::uint8_t* cell_nverts = nullptr;     // nullptr = all quads
  const int* cell_orientation_sign = nullptr;    // nullptr = +1
  // Unique internal faces (each once, with both incident cells).
  int n_faces = 0;
  const int* face_cell_a = nullptr;
  const int* face_cell_b = nullptr;
  const int* face_local_a = nullptr;
  const int* face_local_b = nullptr;
  // Node coordinates before/after the remap mesh motion.
  const double* x_r_old = nullptr;
  const double* x_z_old = nullptr;
  const double* x_r_new = nullptr;
  const double* x_z_new = nullptr;
  // Transported basis (n_cells*4) and pre-remap cell mass/volume.
  const double* corner_mass_old = nullptr;
  const double* cell_mass_old = nullptr;
  const double* cell_vol_old = nullptr;
  // Post-remap cell masses: the per-cell sum targets (non-negotiable
  // cell-sum constraint sum_a m_a = m_c).
  const double* cell_mass_new = nullptr;
  // Optional macro inactive-member mask; faces touching an inactive cell
  // are zero-flux (the remap's own convention) and inactive cells are
  // excluded from transport, projection, and statistics.
  const std::uint8_t* inactive_mask = nullptr;
  // Numerics.ale.swept_volume_sign_fixed: false = legacy (sign-reversed
  // donor, bit-preserved production default), true = corrected. The audit
  // mirrors whichever convention the production remap ran with.
  bool swept_volume_sign_fixed = false;
};

struct CornerMassRemapLoResult {
  bool valid = false;
  // Transported + projected corner masses, n_cells*4; inactive cells and
  // inactive tri slots are zero.
  std::vector<double> corner_mass;
  // Sum over faces of |face mass flux| actually transported.
  double flux_abs_sum = 0.0;
  // Sum over corners of |projection correction| (cell-sum repair size).
  double projection_repair_abs_sum = 0.0;
  // Corners clipped at zero (transport overshoot); the projection then
  // re-balances the cell. Expected 0 for production-CFL remap sweeps.
  long long negative_clip_count = 0;
  // Cells whose post-remap target mass was <= 0 (anomaly; corners zeroed).
  long long nonpositive_target_cells = 0;
  // Min over nodes (with >= 1 active incident corner) of the transported
  // nodal mass sum — the M_n > 0 hard contract probe.
  double min_node_mass_sum = 0.0;
  long long n_active_cells = 0;
};

CornerMassRemapLoResult lo_corner_mass_remap(const CornerMassRemapLoInput& in);

}  // namespace tenryu::hydro::ale

namespace tenryu::core {
struct State;
struct Config;
}  // namespace tenryu::core

namespace tenryu::hydro::ale {

// env TENRYU_I1B_CORNER_MASS_REMAP_AUDIT=1 — audit-only; never writes
// state. Pre/post hook pair around the CSR conservative remap.
bool corner_mass_remap_audit_enabled();

// PR5(a): env TENRYU_I1B_INSTALL_PR4_CORNER_MASS=1 — install the PR4
// remapped+projected corner masses into state.corner_mass after each CSR
// remap (the basis-contract first connection). Unlike the adjudicated
// transported-install arm (negative-mass assert death at t=1.16 ns), the
// PR4 product is hard m_a >= 0 with exact cell sums, so it is installable.
// The OptionB velocity-remap basis switch (PR5(b)) is NOT here.
bool pr4_corner_mass_install_enabled();

// Pre-install consistency verdict over the INSTALL CANDIDATE (PR4 values
// on active cells, frozen mirror values preserved on inactive macro
// members — installing PR4's zeros there would destroy the rim
// machinery's member mirrors).
struct CornerMassInstallCheck {
  bool ok = false;
  long long negative_corners = 0;       // active corners with m_a < 0
  double max_cell_sum_rel_err = 0.0;    // vs post-remap cell mass
  long long bad_sum_cells = 0;          // cells beyond the 1e-12 gate
  double min_node_mass = 0.0;           // over active-incident nodes
};

CornerMassInstallCheck check_corner_mass_for_install(
    int n_cells,
    int n_nodes,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const double* corner_mass_candidate,  // n_cells*4 composite
    const double* cell_mass,              // post-remap targets
    const std::uint8_t* inactive_mask);   // may be nullptr

// PR6-LO: env TENRYU_I1B_INSTALL_VPAIRED_CORNER_MASS=1 — install the
// V-PAIRED corner masses m_a = rho_c * V_a after each CSR remap, where
// V_a are the EXACT SUBPOLYGON corner volumes on the post-remap geometry
// (the SAME rz::compute_quad_corner_volumes_exact_subpolygon /
// triangle-quadrant definition the subzonal-pressure force assembly
// divides by — the (m,V) pairing IS the construction). Corner density is
// then rho_c at every corner BY CONSTRUCTION: the subzonal anti-hourglass
// dp[k] = p(rho_a) - p_c vanishes structurally on uniform states, which
// is exactly what the PR5(a) naive-LO product violated (0.40 smearing ->
// 49 ps velocity-runaway death). Positivity is structural (rho_c > 0,
// V_a > 0); cell sums are enforced by a single multiplicative scale
// m_a *= m_c / sum_a m_a (scale = 1 + O(roundoff) since the quadrants
// tile the cell exactly). Takes precedence over the PR4 install env when
// both are set.
bool vpaired_corner_mass_install_enabled(const core::Config& cfg);

struct VPairedCornerMassResult {
  bool valid = false;
  std::vector<double> corner_mass;  // n_cells*4; zeros on inactive cells
  // max over active cells of |scale - 1| where scale = m_c / sum V-paired
  // masses (measures state.vol vs subpolygon-tiling mismatch).
  double max_scale_dev = 0.0;
  // cells skipped because of nonpositive mass/volume (left zero).
  long long degenerate_cells = 0;
};

// env TENRYU_I1B_INSTALL_KE_COMPENSATION=1 (default off): at V-paired
// install time, the budget's KE measure jumps by
//   dKE = sum_c sum_k 1/2 (m_new - m_old)[4c+k] |v_node(c,k)|^2
// (a pure basis change, no physical motion). Compensate by depositing
// -dKE_c into each cell's internal energy (PR3's ye split), so the
// budget total is continuous across the install to round-off.
bool install_ke_compensation_enabled();

struct InstallKeJump {
  bool valid = false;
  std::vector<double> dKE;  // per cell [erg]; zero on inactive cells
  double total = 0.0;
};

// Pure: per-cell KE jump of replacing corner_mass_old by corner_mass_new
// at fixed node velocities. Inactive cells contribute zero (their
// candidate values are the preserved mirrors).
InstallKeJump compute_install_ke_jump(int n_cells,
                                      int n_nodes,
                                      const int* cell_node_csr_offsets,
                                      const int* cell_node_csr_indices,
                                      const std::uint8_t* cell_nverts,
                                      const double* corner_mass_old,
                                      const double* corner_mass_new,
                                      const double* v_r,
                                      const double* v_z,
                                      const std::uint8_t* inactive_mask);

// Pure kernel: candidate = rho_c * V_a(post geometry), multiplicatively
// scaled to sum exactly to cell_mass per cell (the scale is folded in:
// m_a = m_c * V_a / sum V_a). Post-install corner density is m_c/sum(V_a)
// at EVERY corner, so |rho_a/rho_c - 1| == |cell_vol/sum(V_a) - 1| ==
// |scale - 1| per cell — one audit number covers both task metrics.
// cell_vol may be nullptr (skips the metric).
VPairedCornerMassResult compute_vpaired_corner_mass(
    int n_cells,
    int n_nodes,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    const double* x_r,
    const double* x_z,
    const double* cell_mass,
    const double* cell_vol,
    const std::uint8_t* inactive_mask);

struct CornerMassRemapAuditPre {
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> corner_mass;  // frozen basis at remap entry
};

void corner_mass_remap_audit_capture_pre(const core::State& state,
                                         CornerMassRemapAuditPre* pre);

// Runs the LO transported-basis remap over the remap's net mesh motion
// (pre-capture mesh -> current state mesh) reconciled to the current
// post-remap cell masses. With the audit env: compares against the frozen
// corner basis and logs one [corner_mass_remap_audit] line (no state
// writes). With the PR5(a) install env: validates the composite candidate
// (check_corner_mass_for_install) and either installs it into
// state.corner_mass (clearing corner_mass_is_lagrangian_invariant) or
// abandons with a warning — never installs a violating field.
// coherent_corner_mass (basis-coherent transport, TENRYU_I1B_OPTIONB_COHERENT):
// the OptionB component's post-projection (V-paired) corner masses; when
// non-null the V-paired install writes THIS product — literally the numbers
// the component's velocity re-recover divided by — instead of recomputing it.
void corner_mass_remap_audit_emit(
    core::State& state,
    const core::Config& cfg,
    const CornerMassRemapAuditPre& pre,
    const std::vector<double>* coherent_corner_mass = nullptr);

// Gap-form compensation, bracket-side applier (called from the OptionB
// velocity-remap bracket where the transported first-moment buffer
// exists). Recomputes the V-paired install candidate on the post-remap
// geometry (the same pure kernel the install uses), evaluates the
// gap-form residual, logs one [gap_form_comp] line, and (mode 1 only)
// deposits -R_gap into ee/ei with the PR3 ye split. audit_only (mode 2)
// never writes state.
void gap_form_compensation_bracket(
    core::State& state,
    const std::vector<double>& corner_mass_vp_pre,
    const std::vector<double>& corner_mass_fm_pre,
    const std::vector<double>& v_r_pre,
    const std::vector<double>& v_z_pre,
    const std::vector<double>& corner_mass_fm_post,
    const std::vector<double>& cell_mass_new,
    bool audit_only);

}  // namespace tenryu::hydro::ale
