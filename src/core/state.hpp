#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/device_error_flags.cuh"
#include "core/field.hpp"
#include "core/namelist/frozen_table.hpp"
#include "materials/zmoment_device.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::core {

template <typename T>
using DeviceBuffer = DeviceArray<T>;

inline int corner_stride_for_scheme(const TopologyScheme s) {
  return s == TopologyScheme::PENTAGON_BELT_SHELL ? 8 : 4;
}

struct DispatchCounters {
  std::atomic<std::uint64_t> per_material_kernel_call_count{0};
  std::atomic<std::uint64_t> eos_inverse_call_count{0};
  std::atomic<std::uint64_t> eos_table_validity_violations{0};
  std::atomic<std::uint64_t> presence_absent_events{0};
  std::atomic<std::uint64_t> conservation_drift_warnings{0};
  std::atomic<std::uint64_t> lazy_cache_te_m_invalidation_count{0};
  std::atomic<std::uint64_t> lazy_cache_te_m_hit_count{0};
  std::atomic<std::uint64_t> lazy_cache_te_m_miss_count{0};
  std::atomic<std::uint64_t> mixture_projection_call_count{0};

  DispatchCounters() = default;

  DispatchCounters(const DispatchCounters& other)
      : per_material_kernel_call_count(
            other.per_material_kernel_call_count.load(std::memory_order_relaxed)),
        eos_inverse_call_count(
            other.eos_inverse_call_count.load(std::memory_order_relaxed)),
        eos_table_validity_violations(
            other.eos_table_validity_violations.load(std::memory_order_relaxed)),
        presence_absent_events(
            other.presence_absent_events.load(std::memory_order_relaxed)),
        conservation_drift_warnings(
            other.conservation_drift_warnings.load(std::memory_order_relaxed)),
        lazy_cache_te_m_invalidation_count(
            other.lazy_cache_te_m_invalidation_count.load(std::memory_order_relaxed)),
        lazy_cache_te_m_hit_count(
            other.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed)),
        lazy_cache_te_m_miss_count(
            other.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed)),
        mixture_projection_call_count(
            other.mixture_projection_call_count.load(std::memory_order_relaxed)) {}

  DispatchCounters& operator=(const DispatchCounters& other) {
    if (this != &other) {
      per_material_kernel_call_count.store(
          other.per_material_kernel_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      eos_inverse_call_count.store(
          other.eos_inverse_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      eos_table_validity_violations.store(
          other.eos_table_validity_violations.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      presence_absent_events.store(
          other.presence_absent_events.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      conservation_drift_warnings.store(
          other.conservation_drift_warnings.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_invalidation_count.store(
          other.lazy_cache_te_m_invalidation_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_hit_count.store(
          other.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_miss_count.store(
          other.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      mixture_projection_call_count.store(
          other.mixture_projection_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
    return *this;
  }

  DispatchCounters(DispatchCounters&& other) noexcept
      : per_material_kernel_call_count(
            other.per_material_kernel_call_count.load(std::memory_order_relaxed)),
        eos_inverse_call_count(
            other.eos_inverse_call_count.load(std::memory_order_relaxed)),
        eos_table_validity_violations(
            other.eos_table_validity_violations.load(std::memory_order_relaxed)),
        presence_absent_events(
            other.presence_absent_events.load(std::memory_order_relaxed)),
        conservation_drift_warnings(
            other.conservation_drift_warnings.load(std::memory_order_relaxed)),
        lazy_cache_te_m_invalidation_count(
            other.lazy_cache_te_m_invalidation_count.load(std::memory_order_relaxed)),
        lazy_cache_te_m_hit_count(
            other.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed)),
        lazy_cache_te_m_miss_count(
            other.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed)),
        mixture_projection_call_count(
            other.mixture_projection_call_count.load(std::memory_order_relaxed)) {}

  DispatchCounters& operator=(DispatchCounters&& other) noexcept {
    if (this != &other) {
      per_material_kernel_call_count.store(
          other.per_material_kernel_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      eos_inverse_call_count.store(
          other.eos_inverse_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      eos_table_validity_violations.store(
          other.eos_table_validity_violations.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      presence_absent_events.store(
          other.presence_absent_events.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      conservation_drift_warnings.store(
          other.conservation_drift_warnings.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_invalidation_count.store(
          other.lazy_cache_te_m_invalidation_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_hit_count.store(
          other.lazy_cache_te_m_hit_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      lazy_cache_te_m_miss_count.store(
          other.lazy_cache_te_m_miss_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
      mixture_projection_call_count.store(
          other.mixture_projection_call_count.load(std::memory_order_relaxed),
          std::memory_order_relaxed);
    }
    return *this;
  }

  void reset() {
    per_material_kernel_call_count.store(0, std::memory_order_relaxed);
    eos_inverse_call_count.store(0, std::memory_order_relaxed);
    eos_table_validity_violations.store(0, std::memory_order_relaxed);
    presence_absent_events.store(0, std::memory_order_relaxed);
    conservation_drift_warnings.store(0, std::memory_order_relaxed);
    lazy_cache_te_m_invalidation_count.store(0, std::memory_order_relaxed);
    lazy_cache_te_m_hit_count.store(0, std::memory_order_relaxed);
    lazy_cache_te_m_miss_count.store(0, std::memory_order_relaxed);
    mixture_projection_call_count.store(0, std::memory_order_relaxed);
  }
};

struct TriFanCenterPerturbationDiag {
  bool valid = false;
  double Edot_prime_P = 0.0;
  double Edot_prime_Q = 0.0;
  double Eprime_k = 0.0;
  double max_q_over_p = 0.0;
  double min_corner_J = 0.0;
  double min_cell_volume = 0.0;

  void reset() noexcept { *this = TriFanCenterPerturbationDiag{}; }
};

enum class CentralPseudoCoreActiveClass : std::uint8_t {
  ACTIVE_ORDINARY = 0U,
  ACTIVE_MACRO_CORE = 1U,
  INACTIVE_CORE_MEMBER = 2U,
};

struct CentralPseudoCoreState {
  bool configured = false;
  bool built = false;
  bool valid = false;
  double s_c = 0.0;
  int representative_cell = -1;
  // Dynamic complete-ring absorption (high-AI verdict Phase-2): when the
  // first active ring is crushed below tau*V_birth, rebuild the macro cell
  // with one more ring (env TENRYU_I1B_RING_ABSORB).
  int member_ring_count = 0;
  int min_member_ring_count = 0;
  int absorbed_ring_count = 0;
  // Emergency mixed-material absorption (macro-boundary endgame verdict):
  // rows absorbed BEYOND the pure-gas prefix. > 0 marks the run as a
  // "mixed-core endgame" — the macro CV then contains dense-shell material
  // and gas-region metrics that count the whole macro volume carry a
  // documented caveat.
  int mixed_absorbed_row_count = 0;
  // Macro-band rezone-taper auto-lift: when the emergency absorption walk
  // is refused because the WOULD-BE new boundary loop is tangled, the
  // taper (which keeps the band Lagrangian and lets its angular sawtooth
  // grow untreated) is lifted until this step so the rezone can repair the
  // band and the walk can proceed. -1 = no lift pending.
  int taper_lift_until_step = -1;
  // Volume-trigger watch (pre-trigger): per absorption-unit-depth birth
  // minimum cell volume, cached when the unit first enters the watch
  // window (0 = unseen). Replaces the single-ring scalar watch: a row
  // BEYOND the first active unit can collapse to a committed-degenerate
  // volume within one accepted step (observed: cell vol=-0.0 hard assert
  // at t=2.78 ns while the watch looked only at the first unit), so the
  // window may span several units (env TENRYU_I1B_ABSORB_WATCH_ROWS).
  std::vector<double> absorb_watch_birth_by_depth;
  std::vector<std::uint8_t> member_mask;
  std::vector<std::uint8_t> passive_mask;
  std::vector<std::uint8_t> inactive_member_mask;
  std::vector<std::uint8_t> active_cell_class;
  std::vector<std::uint8_t> boundary_node_mask;
  std::vector<int> member_cells;
  std::vector<int> passive_cells;
  std::vector<int> boundary_nodes_ordered;
  std::vector<double> member_scatter_weights;
  DeviceBuffer<std::uint8_t> d_member_mask;
  DeviceBuffer<std::uint8_t> d_passive_mask;
  DeviceBuffer<std::uint8_t> d_inactive_member_mask;
  DeviceBuffer<std::uint8_t> d_active_cell_class;
  DeviceBuffer<std::uint8_t> d_boundary_node_mask;
  DeviceBuffer<int> d_boundary_nodes_ordered;
  DeviceBuffer<double> d_member_scatter_weights;
  DeviceBuffer<double> d_boundary_impulse_r;
  DeviceBuffer<double> d_boundary_impulse_z;
  double V_c = 0.0;
  double M_c = 0.0;
  double M_Y_c = 0.0;
  // Gas share of the pooled internal energy (dynamic-core metric): updated
  // only when membership changes (build/absorb); boundary work and adiabatic
  // compression preserve the fraction (same-gamma ideal gas). The
  // pressure-equilibrium partial volume of the absorbed gas is
  // V_gas = U_gas_frac_c * V_c.
  double U_gas_frac_c = std::numeric_limits<double>::quiet_NaN();
  int u_gas_frac_member_count = -1;
  // Published by the stratified 1D core sub-model when active (hydro layer
  // writes, diagnostics reads): shell-resolved gas partial volume [cm^3].
  double core1d_V_gas_c = std::numeric_limits<double>::quiet_NaN();
  // Running minimum of core1d_V_gas_c (rebound detector for the terminal
  // absorption: V_gas rising a factor past this minimum marks the post-peak
  // phase where eating the LAST shell row is legitimate).
  double core1d_V_gas_min_c = std::numeric_limits<double>::infinity();
  // Terminal absorption (I1-B-R endgame, env TENRYU_I1B_TERMINAL_ABSORB):
  // when the rebound-phase emergency walk requests the structurally
  // unabsorbable LAST shell row, the whole remaining 2D mesh is absorbed
  // into the stratified 1D sub-model and the run completes as a core1d-only
  // tail integration. The 2D mesh state is FROZEN at terminal_absorb_t.
  bool terminal_absorb_pending = false;
  bool terminal_absorbed = false;
  double terminal_absorb_t = 0.0;
  int terminal_absorb_step = -1;
  double Ue_c = 0.0;
  double Ui_c = 0.0;
  bool boundary_work_pending = false;
  bool boundary_work_valid = false;
  double boundary_work_dU = 0.0;
  double boundary_work_chi_e = 1.0;
  double boundary_work_W_old = 0.0;
  double boundary_work_W_new = 0.0;
  double boundary_work_W_mid = 0.0;
  double boundary_work_residual = 0.0;
  bool activation_ke_valid = false;
  bool activation_ke_bank_pending = false;
  double activation_ke_K_old = 0.0;
  double activation_ke_K_new = 0.0;
  double activation_ke_dU = 0.0;
  double activation_ke_chi_e = 1.0;
  double activation_ke_E_before = 0.0;
  double activation_ke_E_after_zero = 0.0;
  std::vector<int> activation_passive_nodes;
  std::vector<double> activation_saved_v_r;
  std::vector<double> activation_saved_v_z;
  double E_total_ref = std::numeric_limits<double>::quiet_NaN();
  double mass_ref = std::numeric_limits<double>::quiet_NaN();
  double tracer_mass_ref = std::numeric_limits<double>::quiet_NaN();
  double Ue_ref = std::numeric_limits<double>::quiet_NaN();
  double Ui_ref = std::numeric_limits<double>::quiet_NaN();
  double V_c_initial = std::numeric_limits<double>::quiet_NaN();
  double rho_c_initial = std::numeric_limits<double>::quiet_NaN();
};

struct PoleAngularDerefineMacroState {
  int block_id = -1;
  int representative_cell = -1;
  int local_i_begin = -1;
  int local_i_end = -1;
  int local_j_begin = -1;
  int local_j_end = -1;
  int level = 0;
  int span = 0;
  int skipped_nodes = 0;
  bool single_apex_boundary = false;
  int canonical_apex_node = -1;
  std::vector<int> member_cells;
  std::vector<int> boundary_nodes_ordered;
  std::vector<double> member_scatter_weights;
  double V_c = 0.0;
  double M_c = 0.0;
  double M_Y_c = 0.0;
  double Ue_c = 0.0;
  double Ui_c = 0.0;
  bool boundary_work_pending = false;
  bool boundary_work_valid = false;
  double boundary_work_dU = 0.0;
  double boundary_work_chi_e = 1.0;
  double boundary_work_W_old = 0.0;
  double boundary_work_W_new = 0.0;
  double boundary_work_W_mid = 0.0;
  double boundary_work_residual = 0.0;
};

struct PoleAngularDerefineState {
  bool configured = false;
  bool built = false;
  bool valid = false;
  int block_id = -1;
  int representative_cell = -1;
  int n_i_cells = 0;
  int n_j_cells = 0;
  int cell_begin = -1;
  int owned_node_begin = -1;
  int macro_count = 0;
  std::vector<PoleAngularDerefineMacroState> macros;
  std::vector<std::uint8_t> member_mask;
  std::vector<std::uint8_t> inactive_member_mask;
  std::vector<std::uint8_t> boundary_node_mask;
  std::vector<int> boundary_nodes_flat;
  std::vector<int> boundary_node_offsets;
  DeviceBuffer<std::uint8_t> d_member_mask;
  DeviceBuffer<std::uint8_t> d_inactive_member_mask;
  DeviceBuffer<std::uint8_t> d_boundary_node_mask;
  DeviceBuffer<int> d_boundary_nodes_flat;
  DeviceBuffer<int> d_boundary_node_offsets;
  DeviceBuffer<double> d_boundary_impulse_r;
  DeviceBuffer<double> d_boundary_impulse_z;
  bool activation_ke_valid = false;
  double activation_ke_K_old = 0.0;
  double activation_ke_K_new = 0.0;
  double activation_ke_dU = 0.0;
  double activation_ke_chi_e = 1.0;
  double E_total_ref = std::numeric_limits<double>::quiet_NaN();
  double mass_ref = std::numeric_limits<double>::quiet_NaN();
  double tracer_mass_ref = std::numeric_limits<double>::quiet_NaN();
  double Ue_ref = std::numeric_limits<double>::quiet_NaN();
  double Ui_ref = std::numeric_limits<double>::quiet_NaN();
};

struct State {
  // --- Mesh ---
  tenryu::mesh::Mesh mesh;

  // --- Parallel owned compute range (Option C decomposition) ---
  // Set by the driver from PartitionInfo when n_ranks > 1 (r-slab: owned
  // flat ranges are contiguous). Zero-initialized = "full range": the
  // owned_* helpers below then return [0, n), so single-rank launch
  // parameters are bit-identical to the undecomposed code.
  // (docs/design/mpi_m18_20_20260717.md §3/§4.1)
  int owned_cell_begin = 0;
  int owned_cell_end = 0;
  int owned_node_begin = 0;
  int owned_node_end = 0;

  int owned_cells_begin_or(const int) const { return owned_cell_end > 0 ? owned_cell_begin : 0; }
  int owned_cells_end_or(const int n) const { return owned_cell_end > 0 ? owned_cell_end : n; }
  int owned_nodes_begin_or(const int) const { return owned_node_end > 0 ? owned_node_begin : 0; }
  int owned_nodes_end_or(const int n) const { return owned_node_end > 0 ? owned_node_end : n; }

  // Launch window [begin, end) over flat indices. Kernels take (begin, end)
  // and compute the ABSOLUTE index c = begin + thread_idx, so boundary
  // predicates against global extents keep their meaning unchanged.
  struct LaunchWindow {
    int begin = 0;
    int end = 0;
    int count() const { return end > begin ? end - begin : 0; }
    int blocks(const int block_size = 256) const {
      return (count() + block_size - 1) / block_size;
    }
  };
  // Owned window (update/accumulate kernels: each cell/node computed by
  // exactly one rank; windowed reductions are owned-masked by construction).
  LaunchWindow owned_cell_window(const int n) const {
    return {owned_cells_begin_or(0), owned_cells_end_or(n)};
  }
  LaunchWindow owned_node_window(const int n) const {
    return {owned_nodes_begin_or(0), owned_nodes_end_or(n)};
  }
  // Owned+ghost window (derive kernels: quantities computed locally per cell
  // from halo-exchanged primitives, extended one ghost row on each interior
  // side; r-slab ownership tiles the domain, so begin > 0 iff a lower
  // neighbor exists). `stride` is the flat cells-per-r-row (nz for 2D cells,
  // nz+1 for 2D nodes, 1 for 1D).
  LaunchWindow owned_cell_window_ghost(const int n, const int stride) const {
    const int b = owned_cells_begin_or(0);
    const int e = owned_cells_end_or(n);
    return {b > 0 ? b - stride : 0, e < n ? e + stride : n};
  }
  LaunchWindow owned_node_window_ghost(const int n, const int stride) const {
    const int b = owned_nodes_begin_or(0);
    const int e = owned_nodes_end_or(n);
    return {b > 0 ? b - stride : 0, e < n ? e + stride : n};
  }

  // --- Hydro cell fields ---
  CellField1D rho;
  CellField1D mass;
  // Subzonal corner masses, Lagrangian-frozen for 2D RZ hydro.
  // Existing schemes use the legacy four-corner layout; pentagon-belt
  // reserves eight slots per cell for the staged variable-vertex path.
  int corner_stride = 4;
  CellField1D corner_mass;
  std::vector<std::int8_t> cell_nverts_host;
  bool corner_mass_initialized = false;
  bool corner_mass_is_lagrangian_invariant = false;
  CellField1D corner_volume;
  CellField1D corner_pressure;
  // Phase B Caramana-Shashkov hourglass subzonal masses. These are initialized
  // when the opt-in hourglass force is enabled and remain Lagrangian-invariant
  // between accepted ALE remaps.
  CellField1D subzonal_mass_corner0;
  CellField1D subzonal_mass_corner1;
  CellField1D subzonal_mass_corner2;
  CellField1D subzonal_mass_corner3;
  CellField1D subzonal_band_weight;
  bool subzonal_band_weight_initialized = false;
  CellField1D vol;
  CellField1D cell_vol_initial;
  // Runtime-only CP2 hysteresis bitmask; restarts re-seed from geometry.
  std::vector<std::uint8_t> center_patch_latch;
  CellField1D vol_prev_hydro;
  CellField1D zbar;
  // Per-cell effective ideal-gas material properties (A harmonic, gamma
  // linear over positive volfrac weights — numerically identical to the
  // historic conduction-local mixing; void cells carry the void material's
  // values but are masked out of physics). Single source of truth for every
  // closure site (hydro / conduction / Qei). Filled by
  // ensure_cell_material_props(); invalidated whenever volFrac changes
  // (ALE remap).
  CellField1D A_eff;
  CellField1D zmom_r2;  // per-cell <Z^2>/zbar^2 ratio (1.0 when tables absent)
  CellField1D zmom_r4;  // per-cell <Z^4>/zbar^4 ratio
  materials::ZMomentDeviceTables zmom_tables;  // null pointers when absent
  bool zmom_active = false;
  DeviceBuffer<double> zmom_r2_table_storage;
  DeviceBuffer<double> zmom_r4_table_storage;
  CellField1D gamma_eff;
  // Per-cell effective constant opacities, filled and invalidated with
  // ensure_cell_material_props() on the same lifecycle as A_eff/gamma_eff.
  CellField1D kappa_planck_eff;
  CellField1D kappa_rosseland_eff;
  CellField1D Te;
  CellField1D Ti;
  CellField1D ee;
  CellField1D ei;
  CellField1D Pe;
  CellField1D Pi;
  CellField1D Qvisc;
  CellField1D corner_force_p_r;
  CellField1D corner_force_p_z;
  CellField1D corner_force_sub_r;
  CellField1D corner_force_sub_z;
  CellField1D corner_force_q_r;
  CellField1D corner_force_q_z;
  CellField1D edge_force_av_r;
  CellField1D edge_force_av_z;
  CellField1D work_p_per_cell;
  CellField1D work_sub_per_cell;
  CellField1D work_av_per_cell;
  // 2D Braginskii viscosity corner forces / work tally
  // (docs/design/2d_visc_port_spec.md §2.3; sized n_cells*4 / n_cells,
  // allocated only when plasma_viscosity.enabled).
  CellField1D corner_force_visc_r;
  CellField1D corner_force_visc_z;
  CellField1D work_visc_per_cell;
  CellField1D visc_heat_rate_per_cell;  // [n_cells] erg/s, corrector-stage
  CellField1D visc_heat_rate_e_per_cell;  // [n_cells] erg/s, electron channel
  // Experimental 2D_RZ z-HLLC path: authoritative cell-centered z momentum
  // [g cm/s]. HLLC primitives read this field instead of reconstructing cell
  // velocity from projected nodes.
  CellField1D hllc_mom_z_cell;
  CellField1D shock_time;
  CellField1D adaptive_av_gate;
  CellField1D eta_compatible;
  CellField1D volFrac;
  CellField1D gas_tracer_Y;
  CellField1D cv_e;  // electron heat capacity [erg/(g*eV)], from table EOS or ideal gas
  CellField1D cv_i;  // ion heat capacity [erg/(g*eV)], from table EOS or ideal gas
  CellField1D cs;    // sound speed [cm/s], from table EOS or ideal gas fallback
  int av_max_cell_id = -1;
  int av_max_i = -1;
  int av_max_j = -1;
  double av_q_visc_max = 0.0;
  double av_rho_at_max = 0.0;
  double av_cs_at_max = 0.0;
  double av_delta_u_at_max = 0.0;
  int count_edge_compressive_edges_step = 0;
  double max_corner_density_spread_step = 0.0;
  double max_corner_density_spread_run = 0.0;
  double max_subzonal_merit_step = 0.0;
  int subzonal_nonzero_force_cells_step = 0;
  double max_abs_subzonal_force_step = 0.0;
  double max_abs_subzonal_work_step = 0.0;
  bool gas_tracer_initialized = false;
  double gas_tracer_mass_initial = 0.0;
  double hotspot_rho_bar_initial = std::numeric_limits<double>::quiet_NaN();
  double hotspot_cr50_initial = std::numeric_limits<double>::quiet_NaN();
  double hotspot_rho50_initial = std::numeric_limits<double>::quiet_NaN();
  double hotspot_internal_energy_initial_erg = std::numeric_limits<double>::quiet_NaN();
  double hotspot_kinetic_energy_initial_erg = std::numeric_limits<double>::quiet_NaN();
  bool diff_ref_diag_baseline_initialized = false;
  double diff_ref_diag_gas_mesh_volume_initial = 0.0;
  double diff_ref_diag_gas_rho_p50_initial = 0.0;
  int max_corner_density_spread_step_idx = -1;
  int max_corner_density_spread_cell_id = -1;
  int max_corner_density_spread_corner_id = -1;
  TriFanCenterPerturbationDiag tri_fan_center_perturbation_diag{};
  CentralPseudoCoreState central_pseudo_core{};
  PoleAngularDerefineState pole_angular_derefine{};
  bool ring7_seam_rezone_requested = false;
  int ring7_seam_rezone_request_cell = -1;
  bool ring7_pole_cap_oracle_requested = false;
  int ring7_pole_cap_request_cell = -1;
  bool ring7_last_failed_velocity_valid = false;
  int ring7_last_failed_cell = -1;
  double ring7_last_failed_sigma_safe = 1.0;
  double ring7_last_failed_dt = 0.0;
  NodeField1D ring7_last_failed_v_r;
  NodeField1D ring7_last_failed_v_z;
  bool ring7_pole_cap_validation_pending = false;
  int ring7_pole_cap_validation_cell = -1;
  double ring7_pole_cap_validation_dt = 0.0;
  double ring7_pole_cap_eta_prod_proxy = 0.0;
  // --- Per-material conservation arrays (Stage 32a Wave A) ---
  // Cell-major layout: idx_cm = c * n_mat + m. All extensive (cgs).
  // Disabled by default via numerics.materials.per_material_conservation_enabled.
  CellField1D mass_per_material;  // [n_cells * n_mat] [g]
  CellField1D Ee_per_material;    // [n_cells * n_mat] [erg]
  CellField1D Ei_per_material;    // [n_cells * n_mat] [erg]

  // --- Lazy-cache infrastructure (derived, never authoritative) ---
  CellField1D Te_per_material;  // [n_cells * n_mat] cached Te_m [eV]
  CellField1D Ti_per_material;  // [n_cells * n_mat] cached Ti_m [eV]
  std::vector<std::uint8_t> Te_per_material_valid;
  std::vector<std::uint8_t> Ti_per_material_valid;

  // --- Node fields ---
  NodeField1D x_r;
  NodeField1D x_z;
  NodeField1D x_r_initial;
  NodeField1D x_z_initial;
  NodeField1D x_r_reference;
  NodeField1D x_z_reference;
  NodeField1D x_r_shock_target;
  NodeField1D x_z_shock_target;
  std::uint64_t reference_epoch = 0;
  DeviceBuffer<double> ref_xi_node;
  DeviceBuffer<double> ref_xi_cell;
  DeviceBuffer<double> ref_dir0_r;
  DeviceBuffer<double> ref_dir0_z;
  NodeField1D v_r;
  NodeField1D v_z;
  NodeField1D node_planar_mass;
  std::uint64_t axis_lagrangian_tangential_engaged_count = 0;
  std::uint64_t axis_lagrangian_tangential_ineffective_count = 0;
  double axis_lagrangian_tangential_last_sigma = 1.0;
  int trace_mesh_outer_node = -1;
  int trace_mesh_outer_cell = -1;
  std::uint64_t c1_solver_steps_total = 0;
  double c1_solver_residual_max = 0.0;
  double c1_solver_residual_last = 0.0;
  int c1_solver_iter_max = 0;
  int c1_solver_iter_last = 0;
  double c1_solver_cond_number_max = 0.0;
  double c1_solver_cond_number_last = 0.0;
  // C1 boundary heat flux integration [erg], boundary ids:
  // 0=r_inner, 1=r_outer, 2=z_bottom, 3=z_top.
  std::array<double, 4> c1_bc_heat_flux_integrated = {0.0, 0.0, 0.0, 0.0};
  std::array<double, 4> c1_bc_heat_flux_last = {0.0, 0.0, 0.0, 0.0};
  // SNB nonlocal conduction history scalars (nonlocal_model="snb" only; the
  // steps_total gate keeps default-OFF history files byte-identical).
  std::uint64_t snb_steps_total = 0;
  int snb_picard_iters_last = 0;
  int snb_picard_iters_max = 0;
  int snb_nonconverged_steps = 0;
  double snb_picard_resid_last = 0.0;
  int snb_cap_faces_99_last = 0;
  int snb_cap_faces_50_last = 0;
  double snb_cap_theta_min_last = 1.0;
  double snb_cap_theta_min_run = 1.0;
  double snb_dq_over_qsh_max_last = 0.0;
  double snb_dq_over_qsh_max_run = 0.0;
  int snb_solver_iters = 0;
  double snb_solver_resid = 0.0;
  std::uint64_t reference_barrier_engaged_count = 0;
  std::uint64_t reference_barrier_succeeded_count = 0;
  double reference_barrier_last_lambda = 1.0;
  std::uint64_t driver_retry_reference_barrier_attempts = 0;
  std::uint64_t driver_retry_reference_barrier_successes = 0;
  std::uint64_t driver_retry_reference_barrier_lambda_collapse_events = 0;
  std::uint64_t driver_retry_reference_barrier_same_sig_aborts = 0;
  std::uint64_t driver_retry_reference_barrier_dt_collapse_aborts = 0;
  std::uint64_t driver_retry_reference_barrier_max_attempts_aborts = 0;
  std::uint64_t driver_retry_reference_barrier_quality_stagnation_aborts = 0;
  std::uint64_t multiblock_path_dt_rejection_count = 0;
  std::uint64_t multiblock_path_dt_rejection_exhausted_count = 0;
  double multiblock_path_dt_min_accepted =
      std::numeric_limits<double>::infinity();
  bool hllc_mom_z_cell_initialized = false;
  std::array<std::uint8_t, 200> driver_retry_reference_barrier_rezone_recent_steps = {};
  int driver_retry_reference_barrier_recent_idx = 0;

  // --- Radiation placeholders ---
  GroupField1D rad_E;
  GroupField1D rad_E_old;
  GroupField1D rad_dep;
  GroupField1D rad_emit;
  GroupField1D fld_sigma_a;
  GroupField1D fld_sigma_pe;
  GroupField1D fld_sigma_R;
  GroupField1D fld_eta;
  GroupField1D fld_D_cell;
  CellField1D fld_cell_rc;
  CellField1D fld_cell_zc;
  GroupField1D fld_lower;
  GroupField1D fld_diag;
  GroupField1D fld_upper;
  GroupField1D fld_rhs;
  CellField1D fld_Te_old;
  CellField1D fld_delta_T;
  CellField1D fld_reduction_work;
  GroupField1D fld_cusparse_buffer;
  GroupField1D fld_group_bounds_work;
  GroupField1D fld_nlte_f_work;
  GroupField1D fld_nlte_sigma_eff_work;
  GroupField1D fld_nlte_sigma_s_eff_work;
  GroupField1D fld_nlte_eta_cdf_work;
  GroupField1D fld_nlte_lambda_work;
  CellField1D fld_fleck;  // [n_cells], 2D FLD Fleck-factor diagnostic
  GroupField1D fld_trace_records;
  GroupField1D sn_sigma_a;
  GroupField1D sn_rad_E_pre_newton;
  GroupField1D sn_sigma_pe;
  GroupField1D sn_sigma_s;  // sweep scattering opacity; in AP face_blend this stores sigma_R.
  GroupField1D sn_eta;
  GroupField1D sn_phi_old;
  GroupField1D sn_phi_new;
  GroupField1D sn_phi_sweep;
  GroupField1D sn_theta_donor;
  GroupField1D sn_diag_rad_E_pre;
  GroupField1D sn_diag_rad_E_post;
  GroupField1D sn_diag_rad_emission_at_Tn;
  GroupField1D sn_diag_rad_emission_at_Tnp1;
  GroupField1D sn_diag_rad_absorption;
  GroupField1D sn_diag_clip_energy;
  GroupField1D sn_diag_clip_full_deficit;  // [erg/cm^3], per cell x group, full clip deficit magnitude
  GroupField1D sn_diag_chi_opacity;
  GroupField1D sn_diag_F_first_moment;
  GroupField1D sn_face_flux_raw;      // 1D: [(n_cells+1) * G]; 2D: [(n_R_faces+n_Z_faces) * G], [erg/cm^2/s]
  GroupField1D sn_face_flux_diff;     // 1D: [(n_cells+1) * G]; 2D: [(n_R_faces+n_Z_faces) * G], [erg/cm^2/s]
  GroupField1D sn_face_flux_blended;  // 1D: [(n_cells+1) * G]; 2D: [(n_R_faces+n_Z_faces) * G], [erg/cm^2/s]
  GroupField1D sn_face_flux_limited;  // 1D: [(n_cells+1) * G]; 2D: [(n_R_faces+n_Z_faces) * G], [erg/cm^2/s]
  GroupField1D sn_face_alpha;         // 1D: [(n_cells+1) * G]; 2D: [(n_R_faces+n_Z_faces) * G], dimensionless
  GroupField1D sn_stream_theta;      // [n_cells * G], donor streaming limiter theta [dimensionless]
  // BUG-21b: pass-1 donor-only theta (inflow-credit pass 2 writes the final
  // theta into sn_stream_theta).
  GroupField1D sn_stream_theta_donor;
  GroupField1D sn_E_star_flux;       // [n_cells * G], face-flux E* [erg/cm^3]
  GroupField1D sn_diag_E_star_flux;  // [n_cells * G], always-written face-flux E* diagnostic
  GroupField1D sn_psi_scratch;
  GroupField1D sn_psi_outgoing_scratch;
  // BUG-21: previous-step angular intensity psi^n (n_cells*groups*angles).
  // Persists across radiation steps so the backward-Euler time source is
  // per-angle (inv_cdt * psi_prev) instead of the isotropized
  // 0.5*inv_cdt*c*rad_E_old (which made transparent-medium fronts diffusive).
  // Reseeded isotropically (0.5*c*rad_E) whenever its size does not match
  // (first step, mesh change, restart).
  GroupField1D sn_psi_prev;
  GroupField1D sn_lc_E_scratch;
  GroupField1D sn_lc_A_scratch;
  GroupField1D sn_origin_boundary;
  // W-B2: persistent per-group incoming Marshak intensity (2*F_inc,g) at the
  // outer node for the 1D sweep; pointer joins the CUDA-graph key. Filled
  // host-side each step outside the captured region; empty => vacuum.
  GroupField1D sn_outer_marshak_psi_in;
  GroupField1D sn_radial_fixup_count;
  GroupField1D sn_radial_fixup_artificial_abs;
  GroupField1D sn_angular_fixup_count;
  GroupField1D sn_angular_fixup_artificial_abs;
  GroupField1D sn_Prr;
  GroupField1D sn_chi;
  GroupField1D sn_F_z;
  GroupField1D sn_P_zz;
  GroupField1D sn_chi_z;
  GroupField1D sn_dsa_lower;
  GroupField1D sn_dsa_diag;
  GroupField1D sn_dsa_upper;
  GroupField1D sn_dsa_rhs;
  GroupField1D sn_dsa_cusparse_buffer;
  GroupField1D sn_group_bounds_work;
  GroupField1D sn_nlte_f_work;
  GroupField1D sn_nlte_sigma_eff_work;
  GroupField1D sn_nlte_sigma_s_eff_work;
  GroupField1D sn_nlte_eta_cdf_work;
  GroupField1D sn_nlte_lambda_work;
  CellField1D sn_Te_old;
  CellField1D sn_delta_T;
  CellField1D sn_reduction_work;
  CellField1D sn_tau_R;         // [n_cells], 2D S_N AP optical-depth diagnostic
  CellField1D sn_reduced_flux;  // [n_cells], 2D S_N AP reduced-flux diagnostic
  CellField1D sn_ap_alpha;      // [n_cells], 2D S_N AP alpha diagnostic
  CellField1D difference_W;           // [n_cells], optional difference reference weight
  GroupField1D difference_E_ref;      // [n_cells * G], optional time-average reference density
  GroupField1D difference_residual_E; // [n_cells * G], optional signed residual density
  CellField1D delta_E_rad_prev;  // [n_cells], previous step applied net radiation source
  std::vector<std::int8_t> ddmc_mode_map;
  bool ddmc_mode_map_valid = false;
  bool particle_sort_cache_invalidated = false;
  std::vector<std::uint8_t> holo_core_mask;
  std::vector<std::uint8_t> holo_patch_mask;
  std::vector<std::uint8_t> holo_core_prev_mask;
  std::vector<std::int32_t> holo_hold_count;
  std::vector<std::int32_t> holo_dwell_count;
  std::vector<double> holo_tau_R;
  std::vector<double> holo_reduced_flux;
  std::vector<double> holo_mass_q;
  std::vector<double> holo_lo_weight;
  GroupField1D holo_E_LO;
  GroupField1D holo_F_LO;  // [(n_cells+1) * G], face radiation flux for QD solver
  GroupField1D holo_consistency_source;  // [n_cells * G], same-step HOLO RHS source [erg/s]
  GroupField1D holo_rad_dep;
  GroupField1D holo_rad_emit;
  GroupField1D holo_Prr;
  GroupField1D holo_chi;
  GroupField1D holo_chi_filtered;  // [n_cells * G], filtered QD closure history
  GroupField1D holo_Prr_coverage;
  bool holo_core_mask_valid = false;
  bool holo_lo_source_valid = false;
  bool holo_ale_invalidated = false;
  int fld_outer_iterations = 0;
  bool fld_converged = false;
  double fld_outer_residual = 0.0;
  std::uint64_t cg_cap_exit_unconverged = 0;
  std::uint64_t newton_cap_exit_unconverged = 0;
  // Phase 2a-1.5 diagnostics (variant A)
  int fld_clamp_hits_step = 0;                  // sum over Picard outers
  double fld_clamp_energy_delta_step = 0.0;     // sum V*(x_pub - x_raw)
  double fld_min_x_raw_step = 0.0;              // min over Picard outers
  double fld_cg_true_residual_l2_rel_step = 0.0;  // max ||r_true||_2 / ||b||_2
  double fld_cg_true_residual_max_step = 0.0;   // max |r_true_i|
  double fld_E_solver_step = 0.0;               // max |sum_rows(A*x_pub - b)|
  double fld_cg_true_residual_l2_rel_RAW_step = 0.0;
  double fld_cg_true_residual_max_RAW_step = 0.0;
  double fld_E_solver_RAW_step = 0.0;
  double fld_csr_diag_min_pos_step = std::numeric_limits<double>::infinity();
  double fld_csr_diag_max_step = 0.0;
  int fld_csr_weak_diag_dom_count_step = 0;
  int fld_csr_nonfinite_count_step = 0;
  double fld_gershgorin_lower_min_step = std::numeric_limits<double>::infinity();
  double fld_gershgorin_upper_max_step = 0.0;
  double fld_cg_pAp_min_step = std::numeric_limits<double>::infinity();
  int fld_cg_nonpos_pAp_count_step = 0;
  int fld_cg_nonfinite_count_step = 0;
  double fld_cg_recurrent_resid_last_check_max_step = 0.0;
  double fld_Dcell_min_step = std::numeric_limits<double>::infinity();
  double fld_Dcell_max_step = 0.0;
  int fld_Dcell_zero_count_step = 0;
  int fld_Dcell_nonfinite_count_step = 0;
  int fld_face_skip_D_count_step = 0;
  int fld_face_skip_nonfinite_count_step = 0;
  int fld_face_skip_dist_count_step = 0;
  int fld_diag_fallback_count_step = 0;
  double fld_pair_symmetry_max_diff_step = 0.0;
  int fld_pair_symmetry_violation_count_step = 0;
  int fld_newton_converged_count_step = 0;
  int fld_newton_cap_hit_count_step = 0;
  int fld_newton_invalid_count_step = 0;
  double fld_newton_resid_abs_max_step = 0.0;
  double fld_newton_resid_rel_max_step = 0.0;
  int fld_newton_reject_count_step = 0;
  double fld_newton_reject_resid_rel_max_step = 0.0;
  double fld_rogue_max_abs_x_raw_step = 0.0;
  int fld_rogue_max_abs_x_raw_cell_step = -1;
  int fld_rogue_max_abs_x_raw_group_step = -1;
  double fld_rogue_min_x_raw_step = 0.0;
  int fld_rogue_min_x_raw_cell_step = -1;
  int fld_rogue_min_x_raw_group_step = -1;
  double fld_rogue_max_rad_E_step = 0.0;
  int fld_rogue_max_rad_E_cell_step = -1;
  int fld_rogue_max_rad_E_group_step = -1;
  double fld_rogue_max_r_true_step = 0.0;
  int fld_rogue_max_r_true_cell_step = -1;
  int fld_rogue_max_r_true_group_step = -1;
  double fld_rogue_max_E_solver_row_step = 0.0;
  int fld_rogue_max_E_solver_row_cell_step = -1;
  int fld_rogue_max_E_solver_row_group_step = -1;
  double fld_escaped_step = 0.0;
  double fld_marshak_in_step = 0.0;
  // W-B: external radiation volume-source energy injected this step [erg]
  // (Su-Olson class drive); consumed by the driver energy ledger.
  double fld_volume_source_in_step = 0.0;
  double fld_state_supply_in_cumulative = 0.0;
  double fld_state_supply_out_cumulative = 0.0;
  double fld_state_supply_net_cumulative = 0.0;
  double fld_state_supply_in_step = 0.0;
  double fld_state_supply_out_step = 0.0;
  double fld_state_supply_net_step = 0.0;
  int sn_outer_iterations = 0;
  int sn_inner_iterations = 0;
  bool sn_converged = false;
  bool sn_outer_stagnated = false;
  double sn_outer_residual = 0.0;
  double sn_inner_residual = 0.0;
  double sn_escaped_step = 0.0;
  double sn_marshak_in_step = 0.0;
  // AI review k06 C8 (2026-07-26): per-step void-anchor energy ledger —
  // signed and absolute sums of V*(E_after - E_before) over every anchor
  // application in the step (1D SN). A validation run can require the abs
  // sum == 0 before trusting the run as an independent S_N reference.
  double sn_void_anchor_dE_step = 0.0;
  double sn_void_anchor_dE_abs_step = 0.0;
  double sn_ap_alpha_max = 0.0;
  double sn_ap_alpha_active_faces = 0.0;
  // W-B: SN counterpart of fld_volume_source_in_step (unwired until the SN
  // 1D volume source lands; stays 0 so the ledger reads it unconditionally).
  double sn_volume_source_in_step = 0.0;
  // W-C: SN material Newton global-timestep-rejection request (bit 1 =
  // floor-root, bit 2 = bracket-expansion failure; NUMERICS §6.8). Set by the
  // SN radiation stage, consumed and cleared by the driver retry logic.
  int sn_material_retry_flag = 0;

  // --- Laser placeholders ---
  CellField1D laser_dep;
  CellField1D ray_density;
  std::vector<double> laser_ray_R0;
  std::vector<double> laser_ray_Z0;
  std::vector<double> laser_ray_vR0;
  std::vector<double> laser_ray_vZ0;
  std::vector<double> laser_ray_x0;
  std::vector<double> laser_ray_y0;
  std::vector<double> laser_ray_z0;
  std::vector<double> laser_ray_vx0;
  std::vector<double> laser_ray_vy0;
  std::vector<double> laser_ray_vz0;
  std::vector<double> laser_ray_power0;
  std::vector<int> laser_ray_beam_id;
  bool laser_ray_is_3d = false;
  // Laser mesh snapshot for ray trace diagnostics
  std::vector<double> laser_mesh_node_R;
  std::vector<double> laser_mesh_node_Z;
  std::vector<double> laser_mesh_n_e_hat;
  std::vector<double> laser_mesh_T_e;
  std::vector<double> laser_mesh_Zbar;
  std::vector<double> laser_mesh_grad_R;
  std::vector<double> laser_mesh_grad_Z;
  int laser_mesh_n_nodes_r = 0;
  int laser_mesh_n_nodes_z = 0;
  double laser_mesh_n_crit = 0.0;

  // Ray trajectory output (ragged/CSR format)
  std::vector<std::int64_t> ray_traj_offsets;
  std::vector<std::int32_t> ray_traj_step_counts;
  std::vector<std::int32_t> ray_traj_beam_ids;
  std::vector<double> ray_traj_pos1;
  std::vector<double> ray_traj_pos2;
  std::vector<double> ray_traj_pos3;
  std::vector<double> ray_traj_power;
  bool ray_traj_is_3d = false;
  std::vector<tenryu::core::namelist::FrozenTable1D> laser_waveforms;

  // --- Hydro control ---
  std::vector<std::int8_t> hydro_active;
  // Device mirror of hydro_active (persistent-kernel prep, task #47). Host-side
  // writers must bump hydro_active_host_version (see note_hydro_active_host_write);
  // hydro_active_device_ptr() re-uploads lazily when the versions differ. mutable +
  // const accessor: consumers legitimately take const State&.
  mutable DeviceArray<std::int8_t> hydro_active_device;
  mutable std::uint64_t hydro_active_device_version = 0;
  std::uint64_t hydro_active_host_version = 1;  // != device_version -> first use uploads

  void note_hydro_active_host_write() { ++hydro_active_host_version; }
  const std::int8_t* hydro_active_device_ptr() const;
  std::vector<std::int8_t> state_supply_mask;
  CellField1D state_supply_pre_rho;
  CellField1D state_supply_pre_mass;
  CellField1D state_supply_pre_ee;
  CellField1D state_supply_pre_ei;
  CellField1D state_supply_pre_uz;
  double state_supply_dM_cumulative = 0.0;
  double state_supply_dE_cumulative = 0.0;
  double state_supply_dPz_cumulative = 0.0;
  double state_supply_dM_step = 0.0;
  double state_supply_dE_step = 0.0;
  double state_supply_dPz_step = 0.0;
  std::vector<std::uint8_t> cell_is_void;
  double hydro_t_start_eV = 0.0;
  double adaptive_av_r0 = 0.0;
  double adaptive_av_last_rs = 0.0;
  double adaptive_av_last_us = 0.0;
  double adaptive_av_rs_min = std::numeric_limits<double>::infinity();
  int adaptive_av_tracker_steps = 0;
  int adaptive_av_mode = 0;
  bool adaptive_av_tracker_valid = false;
  bool adaptive_av_bounce_seen = false;
  std::optional<tenryu::core::namelist::FrozenTable1D> pressure_drive_1d;
  std::optional<tenryu::core::namelist::FrozenTable1D> marshak_Tr_1d;
  std::optional<tenryu::core::namelist::FrozenTable1D> hot_e_eta_1d;
  // Per-channel frozen eta(t) tables (Laser.hot_electron.sources[i].eta_table);
  // empty unless the deck used the multi-channel sources form.
  std::vector<std::optional<tenryu::core::namelist::FrozenTable1D>> hot_e_eta_ch_1d;
  // --- hot-electron eta(t) physics model (eta_mode="model"; 1D only) ---
  // Persistent per-config-channel model state (design doc §16.2; one-step-lagged
  // intensity measurement). NOT checkpointed in v1: a restart re-spins-up eta
  // over ~tau (3-10 ps) — documented transient.
  std::vector<double> hot_e_eta_state_eta;        // current eta per channel
  std::vector<double> hot_e_eta_state_kappa_bar;  // EMA of |dln n_e/dr| [1/um]
  std::vector<double> hot_e_eta_prev_Pcross;      // prev-step raw sum P_cross [erg/s]
  std::vector<double> hot_e_eta_prev_rbar;        // prev-step P-weighted mean r_s [cm]
  std::vector<std::uint8_t> hot_e_eta_prev_valid; // prev-step had crossings
  // Diagnostics mirror (filled each model update; for HDF5/log only)
  std::vector<double> hot_e_eta_diag_g;
  std::vector<double> hot_e_eta_diag_eta_eq;
  std::vector<double> hot_e_eta_diag_tau_s;
  std::vector<double> hot_e_eta_diag_I14;
  std::vector<double> hot_e_eta_diag_I14_lower;
  std::vector<double> hot_e_eta_diag_I14_upper;
  std::vector<double> hot_e_eta_diag_n_sigma;
  std::vector<double> hot_e_eta_diag_Te_keV;
  std::vector<double> hot_e_eta_diag_Ln_um;
  std::vector<double> hot_e_eta_diag_clamped;     // Stage A bitmask as double
  std::vector<double> cbet_gross_exchange;  // [n_cells] erg/cm3/s
  std::vector<double> cbet_net_to_inbound;  // [n_cells] erg/cm3/s
  // port_section sky maps (empty unless port_section + hot-e model).
  std::vector<double> ps_sky_mu;
  std::vector<double> ps_sky_phi;
  std::vector<double> ps_sky_I_tot;
  std::vector<double> ps_sky_I_cw;
  std::vector<double> ps_sky_n_sigma;
  // port_section phase-space exports.
  std::vector<double> ps_ray_map;          // [n_shells*64*2], W/cm2
  std::vector<double> ps_ray_map_shell_r;  // [n_shells], cm
  // Path-end alive power: final record's post-solve power, per port.
  std::vector<double> ps_port_outgoing_power;
  std::vector<double> ps_port_capture_pcross;  // [n_ports*n_channels], erg/s
  double ps_f_illum2 = 0.0;
  double ps_f_union = 0.0;
  std::map<std::string, tenryu::core::namelist::FrozenTable1D> marshak_Tr_face_tables;

  // --- Time/step ---
  double t = 0.0;
  int step = 0;
  double dt = 0.0;
  double dt_prev_hydro = -1.0;
  bool ale_rezoned = false;
  int ale_rezone_invocations = 0;
  int ale_remaps_applied = 0;
  int ale_last_applied_step = -1;
  double axis_margin_initial = -1.0;
  bool plic_remap_sticky_fallback = false;
  int plic_consecutive_drift_triggers = 0;
  int plic_last_reconstruction_step = -1;
  CellField1D plic_interface_mask;
  CellField1D plic_active_mask;
  CellField1D plic_reconstruction_valid;
  CellField1D plic_normal_r;
  CellField1D plic_normal_z;
  CellField1D plic_alpha;
  CellField1D plic_centroid_r;
  CellField1D plic_centroid_z;
  CellField1D plic_last_centroid_r;
  CellField1D plic_last_centroid_z;
  CellField1D plic_face_flux_r;
  CellField1D plic_face_flux_z;
  CellField1D plic_cell_residual;
  // Phase 9 cumulative axis-inflow budget (only used when
  // Numerics.ale.remap_damage_axis_budget_enabled). Per-axis-j scalar arrays.
  // axis_mass_initial[j]: M_{0,j}^init lazily set on first ALE call requiring budgets.
  // axis_inflow_budget[j]: cumulative B_{0,j} = sum of accepted positive
  //                       remap mass imports from i=1 row into i=0 row at j.
  //                       Initialized to 0; persisted through checkpoint.
  // Length = nz (number of axis-row cells) when 2D_RZ; empty otherwise.
  std::vector<double> axis_mass_initial;
  std::vector<double> axis_inflow_budget;
  int ale_lambda_sweep_target_cell_c = -1;
  int ale_lambda_sweep_target_cell_i = -1;
  int ale_lambda_sweep_target_cell_j = -1;
  std::string ale_lambda_sweep_classification;
  std::vector<double> ale_lambda_sweep_lambda;
  std::vector<double> ale_lambda_sweep_min_gauss_j;
  std::vector<double> ale_lambda_sweep_min_corner_j;
  std::vector<double> ale_lambda_sweep_min_v_rz;
  std::vector<std::uint8_t> ale_lambda_sweep_admissible;

  // --- Cumulative diagnostics (checkpoint/restart continuity) ---
  double E_safety = 0.0;
  double E_numerical_loss = 0.0;
  double E_laser_deposited = 0.0;
  double E_laser_escaped = 0.0;
  double E_laser_incident = 0.0;
  double E_ra_deposited = 0.0;
  double E_cbet_iaw_step = 0.0;  // erg, signed port_section IAW ledger this step
  double E_cbet_iaw = 0.0;       // erg, cumulative signed port_section IAW ledger
  double E_hot_e_deposited = 0.0;  // erg, cumulative hot-electron deposition
  double E_hot_e_escaped = 0.0;    // erg, cumulative hot-electron escape

  // --- Hot-electron preheat (1D): per-step diagnostics (laser module writes,
  //     driver folds, history/snapshot writers read) ---
  bool hot_e_enabled_any = false;   // latched true when the model first activates
  double hot_e_in_step = 0.0;       // erg: eta*P_ray energy taken this step
  double hot_e_deposited_step = 0.0;  // erg (includes radial-mode inner residual)
  double hot_e_residual_step = 0.0;   // erg
  double hot_e_escaped_step = 0.0;    // erg
  double hot_e_source_r = 0.0;        // cm, power-weighted mean launch radius
  double hot_e_conservation_resid = 0.0;
  double hot_e_dt_limit_s = std::numeric_limits<double>::infinity();
  std::vector<double> hot_e_Q_host;        // erg/cm^3/s, instantaneous (empty unless enabled)
  std::vector<double> hot_e_eps_cum_host;  // erg/g, cumulative specific preheat
  // Per-config-channel step ledgers (Laser.hot_electron.sources); sized to
  // the config channel count while the model is active, zeroed every laser
  // step. History emits them only when more than one channel is configured.
  std::vector<double> hot_e_ch_in_step;
  std::vector<double> hot_e_ch_deposited_step;
  std::vector<double> hot_e_ch_escaped_step;
  // --- Nuclear burn (1D v1): cumulative ledgers (checkpoint continuity NOT
  //     wired in v1 — restart of a burning run resets these; recorded in
  //     the design doc), per-step mirrors (burn stage writes, driver folds,
  //     history/snapshot writers read), and host-resident inventories ---
  double E_burn_released = 0.0;      // erg, cumulative fusion energy release
  double E_burn_dep_e = 0.0;         // erg, cumulative electron deposition
  double E_burn_dep_i = 0.0;         // erg, cumulative ion deposition
  double E_burn_esc_charged = 0.0;   // erg, cumulative charged-product escape
  double E_burn_esc_neutron = 0.0;   // erg, cumulative neutron escape
  double E_burn_inflight = 0.0;      // erg, total in-flight CP energy
  double N_burn_neutrons_dt = 0.0;   // cumulative DT neutron count
  double N_burn_neutrons_dd = 0.0;   // cumulative D(d,n)3He neutron count
  bool burn_enabled_any = false;     // latched true when the stage first runs
  bool burn_diffusion_any = false;   // latched true when scheme="diffusion" runs
  bool burn_mc_any = false;          // latched true when scheme="mc" runs
  double burn_released_step = 0.0;   // erg
  double burn_dep_e_step = 0.0;      // erg
  double burn_dep_i_step = 0.0;      // erg
  double burn_esc_charged_step = 0.0;   // erg
  double burn_esc_neutron_step = 0.0;   // erg
  double burn_nh_dep_e_step = 0.0;      // erg, v2-E neutron heating
  double burn_nh_dep_i_step = 0.0;      // erg
  double burn_nh_degraded_step = 0.0;   // erg (scattered remainder)
  double burn_nh_escaped_step = 0.0;    // erg (uncollided)
  double burn_Ti_burn_dt_eV = 0.0;   // step burn-averaged Ti (DT-weighted)
  double burn_Ti_burn_dd_eV = 0.0;
  double burn_vr2_burn_dt = 0.0;     // cm^2/s^2
  double burn_vr2_burn_dd = 0.0;
  double burn_neutron_mean_shift_dt_keV = 0.0;
  double burn_neutron_sigma_thermal_dt_keV = 0.0;
  double burn_neutron_sigma_total_dt_keV = 0.0;
  double burn_neutron_mean_shift_dd_keV = 0.0;
  double burn_neutron_sigma_thermal_dd_keV = 0.0;
  double burn_neutron_sigma_total_dd_keV = 0.0;
  double burn_dt_limit_s = std::numeric_limits<double>::infinity();
  std::vector<double> burn_n_host;        // [n_cells*5] SPECIFIC inventories Y_s [1/g] (empty unless enabled); n_s = Y_s*rho
  std::vector<double> burn_rate_host;     // reactions/cm^3/s
  std::vector<double> burn_Q_e_host;      // erg/cm^3/s
  std::vector<double> burn_Q_i_host;      // erg/cm^3/s
  std::vector<double> burn_eps_cum_host;  // erg/g cumulative specific burn heating
  DeviceBuffer<double> burn_Ng;           // [6*G*n_cells] specific diffusion in-flight spectra Y_g [1/g]
  DeviceBuffer<double> burn_Ng_work;      // [6*G*n_cells] diffusion density view scratch [1/cm3]
  DeviceBuffer<double> burn_dep_e_dev;    // [n_cells] diffusion deposition scratch
  DeviceBuffer<double> burn_dep_i_dev;    // [n_cells] diffusion deposition scratch
  DeviceBuffer<unsigned char> burn_corman_scratch;  // module scratch pool
  DeviceBuffer<double> burn_mc_r;
  DeviceBuffer<double> burn_mc_mu;
  DeviceBuffer<double> burn_mc_E;
  DeviceBuffer<double> burn_mc_w;
  DeviceBuffer<int> burn_mc_slot;
  DeviceBuffer<unsigned char> burn_mc_alive;
  int burn_mc_live = 0;         // host-tracked live count
  double E_rad_escaped = 0.0;
  double E_floor_injected = 0.0;
  double E_pdV_bdry = 0.0;
  double E_Marshak_in = 0.0;
  double E_solver = 0.0;
  DeviceErrorFlags radiation_device_flags{};
  mutable DispatchCounters dispatch_counters;

  // --- Output timing state ---
  double t_next_plot = -1.0;
  double t_next_history = -1.0;
  double t_next_checkpoint = -1.0;

  static State allocate(const Config& cfg);
  static State allocate(const Config& cfg, double hydro_t_start_eV);

  // Fill A_eff/gamma_eff (per-cell effective ideal-gas properties) from
  // volFrac when absent or size-stale. No-op when already sized. Cheap to
  // call at every operator entry; heavy work happens only after
  // invalidate_cell_material_props() (volFrac changed, e.g. ALE remap) or on
  // first use. Falls back to materials[0] uniformly when volFrac is not
  // populated (single-material legacy states in unit tests).
  void ensure_cell_material_props(const Config& cfg);
  void invalidate_cell_material_props();

  void reset();
};

}  // namespace tenryu::core
