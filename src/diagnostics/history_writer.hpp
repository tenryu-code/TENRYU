#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <limits>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "coupling/profile_observability.hpp"
#include "diagnostics/diagnostics.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/operator_energy_residuals.hpp"
#include "diagnostics/radial_fourier_audit.hpp"

#if TENRYU_ENABLE_HDF5
#include <hdf5.h>
#endif

namespace tenryu::diagnostics {

struct DtBreakdownHistoryRecord {
  bool valid = false;
  int cycle = 0;
  double t_s = 0.0;
  double dt_chosen = 0.0;
  std::string dt_winner = "other";
  int dt_winner_code = 0;
  double dt_hydro = std::numeric_limits<double>::infinity();
  double dt_rad = std::numeric_limits<double>::infinity();
  double dt_cond = std::numeric_limits<double>::infinity();
  double dt_post_shock = std::numeric_limits<double>::infinity();
  double dt_growth = std::numeric_limits<double>::infinity();
  double dt_max = std::numeric_limits<double>::infinity();
  double dt_output = std::numeric_limits<double>::infinity();
  double dt_remaining = std::numeric_limits<double>::infinity();
  double dt_hydro_acoustic = std::numeric_limits<double>::infinity();
  double dt_hydro_axis_margin = std::numeric_limits<double>::infinity();
  double dt_hydro_volume_rate = std::numeric_limits<double>::infinity();
  double dt_hydro_tri_fan_center = std::numeric_limits<double>::infinity();
  int tri_fan_center_cfl_cell_id = -1;
  int tri_fan_center_cfl_i = -1;
  int tri_fan_center_cfl_j = -1;
  double tri_fan_center_cfl_h = 0.0;
  double tri_fan_center_cfl_c_eff = 0.0;
  double tri_fan_center_cfl_q_over_p = 0.0;
  double tri_fan_center_cfl_dt_global_over_dt_center =
      std::numeric_limits<double>::infinity();
  int hydro_winner_cell_id = -1;
  int hydro_winner_i = -1;
  int hydro_winner_j = -1;
  double hydro_winner_dt_at_cell = std::numeric_limits<double>::infinity();
  double hydro_winner_dl_at_cell = 0.0;
  double hydro_winner_cs_at_cell = 0.0;
  double hydro_winner_rho_at_cell = 0.0;
  double hydro_winner_u_z_at_cell = 0.0;
};

// Electron-viscosity lane (2026-07-12): plasma-viscosity regime diagnostics
// (NUMERICS 3.1.13). Filled by the driver at history cadence when
// Numerics.hydro.plasma_viscosity is enabled; field semantics documented at
// hydro/braginskii_viscosity.cuh::HistoryDiagnostics.
struct PlasmaViscosityHistoryRecord {
  bool valid = false;
  int cycle = 0;
  double t_s = 0.0;
  double eta_i_max = 0.0;
  double eta_e_max = 0.0;
  double eta_eff_max = 0.0;
  double ratio_min = 0.0;
  double ratio_geomean_masswt = 0.0;
  double ratio_max = 0.0;
  int n_cells_e_dom = 0;
  int n_cells_i_dom = 0;
  int n_cells_mixed = 0;
  int n_cells_active = 0;
  double heat_rate_i_tot = 0.0;
  double heat_rate_e_tot = 0.0;
};

struct HistorySnapshot {
  EnergyBudget energy{};
  ConservationResiduals conservation_residuals{};
  int clamp_count = 0;
  ArealDensityDiagnostics areal_density{};
  SphericityDiagnostics sphericity{};
  LaserPatternDiagnostics laser_pattern{};
  FldSolverDiagnostics fld_solver;
  PhaseResolvedEnergyDiagnostics phase_energy{};
  AleClosureAuditDiagnostics ale_closure_audit{};
  IcfShellDiagnostics icf_shell{};
  HotspotGasDiagnostics hotspot_gas{};
  tenryu::coupling::ProfileObservability* ale_provenance = nullptr;
  std::vector<OperatorResidualEntry> operator_residuals;
  RadiationDiagnostics mc{};
  DtBreakdownHistoryRecord dt_breakdown{};
  PlasmaViscosityHistoryRecord plasma_viscosity{};
};

#if TENRYU_ENABLE_HDF5
void write_ale_provenance_history(
    hid_t file,
    tenryu::coupling::ProfileObservability& obs,
    double time,
    int step);

void write_ale_provenance_final_attributes(hid_t file,
                                           const char* final_provenance,
                                           const char* final_claim_level,
                                           bool reached_t_end,
                                           bool profile_enabled,
                                           const char* plic_gate_status = nullptr);

void write_material_interface_history(
    hid_t file,
    tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg,
    double time,
    int step);

void write_material_interface_final_attributes(
    hid_t file,
    const tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg);

void append_ale_provenance_final_to_history_file(
    const std::string& history_path,
    tenryu::coupling::ProfileObservability& obs,
    const tenryu::core::Config& cfg,
    const char* final_provenance,
    const char* final_claim_level,
    double time,
    int step);

std::optional<double> read_icf_initial_radius_from_history_file(
    const std::string& history_path);
#endif

class HistoryWriter {
 public:
  HistoryWriter() = default;
  ~HistoryWriter() noexcept;

  void init(const core::Config& cfg, const std::string& output_dir);
  void append(const core::State& state, const HistorySnapshot& snapshot);
  void append_radial_fourier_audit(const RadialFourierAuditRecord& record);
  void append_radial_fourier_complex_audit(
      const RadialFourierComplexAuditRecord& record);
  void append_fld_substage_audit_batch(
      const std::vector<FldSubstageAuditRecord>& records);
  void flush_pending();

  [[nodiscard]] bool enabled() const noexcept { return enabled_; }
  [[nodiscard]] const std::string& path() const noexcept { return path_; }

 private:
  struct PlasmaHistoryDiagnostics {
    bool valid = false;
    double zbar_mean = 0.0;
    double zbar_max = 0.0;
  };

  struct ImplosionHistoryDiagnostics {
    bool valid = false;
    double rho_peak = 0.0;
    double shell_radius_mean = 0.0;
    double shell_radius_min = 0.0;
    double center_temperature = 0.0;
  };

  struct PerRowMassValues {
    int step = 0;
    double t = 0.0;
    std::vector<int> z_row_idx;
    std::vector<double> mass;
    std::vector<double> rho_max;
    std::vector<double> rho_axis;
    std::vector<double> rho_outer;
    std::vector<double> cs_max;
  };

  struct CornerBcAuditValues {
    std::int64_t cycle = 0;
    double t_s = 0.0;
    int cell_id = -1;
    int i = -1;
    int j = -1;
    double interior_rho = 0.0;
    double interior_u_r = 0.0;
    double interior_u_z = 0.0;
    double interior_e = 0.0;
    double interior_E_r = 0.0;
    double r_outer_ghost_rho = 0.0;
    double r_outer_ghost_u_r = 0.0;
    double r_outer_ghost_u_z = 0.0;
    double z_top_ghost_rho = 0.0;
    double z_top_ghost_u_r = 0.0;
    double z_top_ghost_u_z = 0.0;
    double diagonal_corner_ghost_rho = 0.0;
    double diagonal_corner_ghost_u_r = 0.0;
    double diagonal_corner_ghost_u_z = 0.0;
    double dt_at_cell = 0.0;
    double cs_at_cell = 0.0;
    double q_visc_at_cell = 0.0;
  };

  struct AvMaxHistoryValues {
    int step = 0;
    double t = 0.0;
    int cell_id = -1;
    int i = -1;
    int j = -1;
    double q_visc_max = 0.0;
    double rho_at_max = 0.0;
    double cs_at_max = 0.0;
    double delta_u_at_max = 0.0;
  };

  struct AleProvenanceValues {
    int forbidden_config_violations = 0;
    int escape_valve_activations = 0;
    int class_c_runtime_fires = 0;
    int mesh_geometry_failures_observed = 0;
    int public_baseline_terminal_failures = 0;
    bool emergency_cell_deactivation_fired = false;
    int last_failing_cell = -1;
    int last_failing_i = -1;
    int last_failing_j = -1;
    double last_min_cell_vol = 0.0;
    double last_min_corner_j = 0.0;
    tenryu::mesh::MeshGeometryFailureKind last_failure_kind =
        tenryu::mesh::MeshGeometryFailureKind::None;
    std::vector<tenryu::coupling::EscapeValveEvent> escape_valve_events;
    int interface_cells_observed = 0;
    int interface_reconstruction_attempt_count = 0;
    int interface_reconstruction_success_count = 0;
    double plic_max_eta_E_observed = 0.0;
    double plic_max_volume_fraction_residual_observed = 0.0;
    double plic_min_grad_F_observed = 0.0;
    std::array<std::array<int, 3>, 3> class_d_runtime_fires_matrix{};
    std::size_t plic_events_emitted_count = 0;
    std::vector<tenryu::coupling::ProfileObservability::PlicDegradationEvent>
        plic_events;
    bool plic_events_summary_mode = false;
    std::array<std::array<std::int32_t, 3>, 3> plic_event_summary_counts{};
    bool mesh_quality_min_observed = false;
    double achieved_min_corner_j_rel = std::numeric_limits<double>::infinity();
    double achieved_min_gauss_j_rel = std::numeric_limits<double>::infinity();
    double achieved_min_rz_volume_rel = std::numeric_limits<double>::infinity();
    double achieved_min_edge_length_rel = std::numeric_limits<double>::infinity();
    double achieved_min_altitude_rel = std::numeric_limits<double>::infinity();
    double achieved_max_condition_number = 0.0;
    std::uint64_t negative_rz_volume_count_total = 0;
  };

  struct PendingHistoryRecord {
    double t = 0.0;
    double dt = 0.0;
    std::int64_t step = 0;
    bool mc_group_enabled = true;
    bool mc_particle_counts_enabled = true;
    bool mc_weight_stats_enabled = true;
    bool mc_ddmc_fraction_enabled = true;
    bool phase_resolved_energy_enabled = false;
    bool ale_closure_audit_enabled = false;
    bool icf_enabled = false;
    bool ale_provenance_enabled = false;
    bool hotspot_gas_enabled = false;
    bool mesh_quality_min_enabled = false;
    bool conservation_enabled = false;
    bool dt_breakdown_history_enabled = true;
    bool center_perturbation_enabled = false;
    double E_laser_deposited = 0.0;
    double E_laser_escaped = 0.0;
    double E_rad_escaped = 0.0;
    double E_laser_incident = 0.0;
    double E_ra_deposited = 0.0;
    double E_cbet_iaw_step = 0.0;
    double E_cbet_iaw = 0.0;
    std::uint64_t c1_solver_steps_total = 0;
    double c1_solver_residual_last = 0.0;
    double c1_solver_residual_max = 0.0;
    int c1_solver_iter_last = 0;
    int c1_solver_iter_max = 0;
    double c1_solver_cond_number_last = 0.0;
    double c1_solver_cond_number_max = 0.0;
    std::array<double, 4> c1_bc_heat_flux_integrated = {0.0, 0.0, 0.0, 0.0};
    bool burn_enabled_any = false;
    bool burn_diffusion_any = false;
    bool burn_mc_any = false;
    double burn_released_step = 0.0;
    double burn_dep_e_step = 0.0;
    double burn_dep_i_step = 0.0;
    double burn_esc_charged_step = 0.0;
    double burn_esc_neutron_step = 0.0;
    double burn_nh_dep_e_step = 0.0;      // v2-E neutron heating (1D)
    double burn_nh_dep_i_step = 0.0;
    double burn_nh_degraded_step = 0.0;
    double burn_nh_escaped_step = 0.0;
    double E_burn_released = 0.0;
    double E_burn_dep_e = 0.0;
    double E_burn_dep_i = 0.0;
    double E_burn_esc_charged = 0.0;
    double E_burn_esc_neutron = 0.0;
    double E_burn_inflight = 0.0;
    double N_burn_neutrons_dt = 0.0;
    double N_burn_neutrons_dd = 0.0;
    double burn_Ti_burn_dt_eV = 0.0;
    double burn_Ti_burn_dd_eV = 0.0;
    double burn_neutron_mean_shift_dt_keV = 0.0;
    double burn_neutron_sigma_thermal_dt_keV = 0.0;
    double burn_neutron_sigma_total_dt_keV = 0.0;
    double burn_neutron_mean_shift_dd_keV = 0.0;
    double burn_neutron_sigma_thermal_dd_keV = 0.0;
    double burn_neutron_sigma_total_dd_keV = 0.0;
    double burn_dt_limit_s = std::numeric_limits<double>::infinity();
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
    bool hot_e_enabled_any = false;
    double hot_e_in_step = 0.0;
    double hot_e_deposited_step = 0.0;
    double hot_e_residual_step = 0.0;
    double hot_e_escaped_step = 0.0;
    double hot_e_source_r = 0.0;
    double hot_e_conservation_resid = 0.0;
    double E_hot_e_deposited = 0.0;
    double E_hot_e_escaped = 0.0;
    std::vector<double> hot_e_ch_in_step;
    std::vector<double> hot_e_ch_deposited_step;
    std::vector<double> hot_e_ch_escaped_step;
    int ale_rezone_invocations = 0;
    HistorySnapshot snapshot;
    bool has_ale_provenance = false;
    AleProvenanceValues ale_provenance_values;
    std::optional<PerRowMassValues> per_row_mass_values;
    std::optional<CornerBcAuditValues> corner_bc_audit_values;
    std::optional<AvMaxHistoryValues> av_max_values;
    PlasmaHistoryDiagnostics plasma_diag;
    ImplosionHistoryDiagnostics implosion_diag;
    core::TriFanCenterPerturbationDiag tri_fan_center_perturbation_diag;
  };

#if TENRYU_ENABLE_HDF5
  [[nodiscard]] PendingHistoryRecord build_pending_record(
      const core::State& state,
      const HistorySnapshot& snapshot);
  [[nodiscard]] std::optional<PerRowMassValues> compute_per_row_mass_values(
      const core::State& state) const;
  [[nodiscard]] std::optional<CornerBcAuditValues> compute_corner_bc_audit_values(
      const core::State& state,
      const DtBreakdownHistoryRecord& record) const;
  [[nodiscard]] PlasmaHistoryDiagnostics compute_plasma_history_diagnostics(
      const core::State& state) const;
  [[nodiscard]] ImplosionHistoryDiagnostics compute_implosion_history_diagnostics(
      const core::State& state) const;
  [[nodiscard]] AleProvenanceValues build_ale_provenance_values(
      tenryu::coupling::ProfileObservability& obs,
      std::int64_t step) const;
  void append_record_to_file(hid_t file, const PendingHistoryRecord& rec) const;
  void write_per_row_mass_history(hid_t file, const PerRowMassValues& values) const;
  void write_corner_bc_audit_history(hid_t file, const CornerBcAuditValues& values) const;
  void write_av_max_history(hid_t file, const AvMaxHistoryValues& values) const;
  void write_tri_fan_center_perturbation_history(
      hid_t file,
      const core::TriFanCenterPerturbationDiag& diag,
      double time,
      std::int64_t step) const;
  void write_ale_provenance_history(
      hid_t file,
      const AleProvenanceValues& values,
      double time,
      std::int64_t step) const;
  void write_material_interface_history(
      hid_t file,
      const AleProvenanceValues& values,
      double time,
      std::int64_t step) const;
  void write_mesh_quality_min_history(
      hid_t file,
      const AleProvenanceValues& values,
      double time,
      std::int64_t step) const;
#endif

  bool enabled_ = false;
  bool mc_group_enabled_ = true;
  bool mc_particle_counts_enabled_ = true;
  bool mc_weight_stats_enabled_ = true;
  bool mc_ddmc_fraction_enabled_ = true;
  bool phase_resolved_energy_enabled_ = false;
  bool ale_closure_audit_enabled_ = false;
  bool icf_enabled_ = false;
  bool ale_provenance_enabled_ = false;
  bool hotspot_gas_enabled_ = false;
  bool mesh_quality_min_enabled_ = false;
  bool conservation_enabled_ = false;
  bool dt_breakdown_history_enabled_ = true;
  bool center_perturbation_enabled_ = false;
  int append_flush_every_ = 64;
  std::chrono::steady_clock::time_point last_flush_time_{};
  std::vector<PendingHistoryRecord> pending_;
  bool history_bootstrapped_ = false;
  core::Config cfg_{};
  std::string path_;
};

}  // namespace tenryu::diagnostics
