#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "diagnostics/energy_budget.hpp"

namespace tenryu::laser {
struct LaserMesh;
}  // namespace tenryu::laser

namespace tenryu::diagnostics {

struct RadiationDiagnostics {
  // Legacy mode-map counts (cell x group), kept for compatibility.
  std::int64_t ddmc_mode_count = 0;
  std::int64_t imc_mode_count = 0;
  // SPEC §7.3 particle statistics.
  std::int64_t n_total = 0;
  std::int64_t n_imc_particles = 0;
  std::int64_t n_ddmc_particles = 0;
  std::int64_t n_census = 0;
  std::int64_t n_absorbed = 0;
  std::int64_t n_escaped = 0;
  std::int64_t n_leaked = 0;
  double ddmc_fraction = 0.0;
  double weight_min = 0.0;
  double weight_mean = 0.0;
  double weight_max = 0.0;
  std::int64_t overshoot_count = 0;
  double overshoot_max = 0.0;
  std::int64_t mmatrix_violations = 0;
  std::int64_t mmatrix_fallback_count = 0;
  std::int64_t omega_below_threshold = 0;
  std::int64_t interface_transitions = 0;
  std::int64_t interface_reflections = 0;
  std::int64_t conversion_prob_violations = 0;
  std::int64_t ddmc_to_imc_conversions = 0;
  double rad_momentum_deposition = 0.0;
  std::int64_t difference_reference_valid = 0;
  std::int64_t difference_eligible_cells = 0;
  std::int64_t difference_active_cells = 0;
  std::int64_t difference_strong_cells = 0;
  std::int64_t difference_hybrid_suppressed_cells = 0;
  double difference_W_min = 0.0;
  double difference_W_mean = 0.0;
  double difference_W_max = 0.0;
  double difference_tau_min = 0.0;
  double difference_tau_mean = 0.0;
  double difference_tau_max = 0.0;
  double difference_chi_mean = 0.0;
  double difference_chi_max = 0.0;
  double difference_reduced_flux_max = 0.0;
  double difference_knudsen_max = 0.0;
  double difference_front_grad_Te_max = 0.0;
  double difference_front_grad_rho_max = 0.0;
  double difference_E_ref_total = 0.0;
  std::int64_t holo_n_core_cells = 0;
  std::int64_t holo_n_entered = 0;
  std::int64_t holo_n_exited = 0;
  std::int64_t holo_n_hard_exited = 0;
  std::int64_t holo_n_island_rejected = 0;
  double holo_tau_R_min = 0.0;
  double holo_tau_R_max = 0.0;
  double holo_reduced_flux_max = 0.0;
  double holo_E_LO_total = 0.0;
  double holo_E_LO_boundary_in = 0.0;
  double holo_E_LO_boundary_out = 0.0;
  double holo_matter_delta = 0.0;
  double holo_source_balance_error = 0.0;
  double holo_particle_net_source_core = 0.0;
  double holo_lo_particle_source_mismatch = 0.0;
  double holo_Prr_coverage = 0.0;
  double holo_chi_min = 0.0;
  double holo_chi_mean = 0.0;
  double holo_chi_max = 0.0;
};

struct ArealDensityDiagnostics {
  std::vector<double> angles_deg;
  std::vector<double> rhoR;
  std::vector<double> rhoR_hotspot_tracer;
  std::vector<double> rhoR_fuel_tracer;
};

struct SphericityDiagnostics {
  std::vector<int> modes;
  std::vector<double> coefficients;
};

struct LaserPatternDiagnostics {
  std::vector<double> absorbed_fraction_per_beam;
  double critical_surface_r = 0.0;
  double near_critical_r = 0.0;
  double absorption_weighted_r = 0.0;
  double absorbed_total = 0.0;
  double absorbed_power_total = 0.0;
  double commanded_power_total = 0.0;
  double trace_unabsorbed_power_total = 0.0;
  double unabsorbed_power_total = 0.0;
  double trace_absorption_efficiency_total = 0.0;
  double absorption_efficiency_total = 0.0;
  double transfer_blocked_power_total = 0.0;
  std::int64_t tail_closure_count = 0;
  double tail_closure_absorbed_power_total = 0.0;
  double cbet_exchanged_power_total = 0.0;
  double cbet_ledger_residual_rel = 0.0;
  std::int64_t cbet_iterations = 0;
  std::int64_t cbet_clamp_count = 0;
  std::int64_t critical_surface_hit_count = 0;
  double corona_transition_blend = 0.0;
  std::int64_t corona_transition_resolved_cells = 0;
};

struct FldSolverDiagnostics {
  std::int64_t outer_iterations = 0;
  double outer_residual = 0.0;
  std::uint8_t outer_converged = 0;
};

struct PhaseResolvedEnergyDiagnostics {
  bool valid = false;
  double E_pre_hydro = 0.0;
  double E_post_hydro = 0.0;
  double E_post_ale = 0.0;
  double dE_hydro = 0.0;
  double dE_ale = 0.0;
  double internal_pre_hydro = 0.0;
  double internal_post_hydro = 0.0;
  double internal_post_ale = 0.0;
  double dE_hydro_internal = 0.0;
  double dE_ale_internal = 0.0;
  double kinetic_pre_hydro = 0.0;
  double kinetic_post_hydro = 0.0;
  double kinetic_post_ale = 0.0;
  double dE_hydro_kinetic = 0.0;
  double dE_ale_kinetic = 0.0;
};

struct AleClosureAuditDiagnostics {
  bool valid = false;
  double K0_cellcorner = 0.0;
  double K0_node_from_corner = 0.0;
  double K0_budget = 0.0;
  double I0 = 0.0;
  double K0_scalar_total = 0.0;
  double K_remap_total = 0.0;
  double I_raw = 0.0;
  double K_cellmom = 0.0;
  double K_node_preBC = 0.0;
  double K_node_postBC = 0.0;
  double K_post_budget = 0.0;
  double sum_dI_raw = 0.0;
  double sum_dI_after_floor = 0.0;
  double F_floor = 0.0;
  int n_cells_negative_dI = 0;
  int n_cells_floor_e = 0;
  int n_cells_floor_i = 0;
  double min_tentative_e_e = 0.0;
  double min_tentative_e_i = 0.0;
  double residual_K0_cellcorner_node = 0.0;
  double residual_K0_node_budget = 0.0;
  double residual_K_remap = 0.0;
  double residual_I_remap = 0.0;
  double residual_closure_target = 0.0;
  double dE_ale_predicted = 0.0;
  double dE_ale_total = 0.0;
  double residual_dE_ale = 0.0;
  int mechanism_code = 0;
  bool k_remap_defect_dominant = false;
  bool floor_dominant = false;
  bool diagnostic_mismatch_dominant = false;
};

struct IcfShellDiagnostics {
  bool valid = false;
  double R_shell_cm = 0.0;
  double shell_thickness_cm = 0.0;
  double IFAR = 0.0;
  double CR = 0.0;
  double R_initial_cm = 0.0;
};

struct HotspotGasDiagnostics {
  bool valid = false;
  bool Ti_valid = false;
  double R_g_cm = 0.0;
  double gas_mass_initial_g = 0.0;
  double gas_mass_g = 0.0;
  double gas_mass_rel_drift = 0.0;
  double R50_cm = 0.0;
  double R90_cm = 0.0;
  double R95_cm = 0.0;
  double R99_cm = 0.0;
  double CR50 = 0.0;
  double CR90 = 0.0;
  double CR95 = 0.0;
  double CR99 = 0.0;
  double C_R50_norm = 0.0;
  double Rrms_cm = 0.0;
  double CRrms = 0.0;
  double rho_bar_volume_gcc = 0.0;
  double rho_bar_initial_gcc = 0.0;
  double CR_V = 0.0;
  bool macro_core_valid = false;
  double macro_core_mass_g = 0.0;
  double macro_core_tracer_mass_g = 0.0;
  double macro_core_volume_cm3 = 0.0;
  double macro_core_volume_initial_cm3 = 0.0;
  double macro_core_rho_gcc = 0.0;
  double macro_core_rho_initial_gcc = 0.0;
  double macro_core_gas_energy_frac = 0.0;
  double CR_V_macro = 0.0;
  double rho_mean_mass_gcc = 0.0;
  double rho50_gcc = 0.0;
  double CR_rho50 = 0.0;
  double rho90_gcc = 0.0;
  double rho95_gcc = 0.0;
  double rho99_gcc = 0.0;
  double p_mean_dyn_cm2 = 0.0;
  double p50_dyn_cm2 = 0.0;
  double p90_dyn_cm2 = 0.0;
  double p95_dyn_cm2 = 0.0;
  double p99_dyn_cm2 = 0.0;
  double Te_mean_eV = 0.0;
  double Te_p10_eV = 0.0;
  double Te_p50_eV = 0.0;
  double Te_p90_eV = 0.0;
  double Ti_mean_eV = 0.0;
  double Ti_p10_eV = 0.0;
  double Ti_p50_eV = 0.0;
  double Ti_p90_eV = 0.0;
  double hotspot_internal_energy_erg = 0.0;
  double hotspot_kinetic_energy_erg = 0.0;
  double hotspot_total_energy_erg = 0.0;
  double hotspot_internal_energy_initial_erg = 0.0;
  double hotspot_kinetic_energy_initial_erg = 0.0;
  double hotspot_total_energy_initial_erg = 0.0;
  double hotspot_work_proxy_internal_erg = 0.0;
  double hotspot_work_proxy_kinetic_erg = 0.0;
  double hotspot_work_proxy_total_erg = 0.0;
  double K_mean = 0.0;
  double K50 = 0.0;
  double K90 = 0.0;
  double K95 = 0.0;
  double K99 = 0.0;
};

void initialize_hotspot_gas_tracer(core::State& state,
                                   const core::Config& cfg);

EnergyBudget compute_energy_budget_1d(const core::State& state);
EnergyBudget compute_energy_budget_2d(const core::State& state);

double relative_total_energy_error(const EnergyBudget& reference,
                                   const EnergyBudget& current);

bool check_energy_conservation(const EnergyBudget& reference,
                               const EnergyBudget& current,
                               double rel_tol);

ArealDensityDiagnostics compute_areal_density(const core::State& state,
                                              const core::Config& cfg);

SphericityDiagnostics compute_sphericity(const core::State& state,
                                         const core::Config& cfg);

LaserPatternDiagnostics compute_laser_pattern(const core::State& state,
                                              const core::Config& cfg,
                                              const laser::LaserMesh* laser_mesh);

IcfShellDiagnostics compute_icf_shell_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    double R_initial_cm);

HotspotGasDiagnostics compute_hotspot_gas_diagnostics(
    const core::State& state,
    const core::Config& cfg);

PhaseResolvedEnergyDiagnostics compute_phase_resolved_energy_diagnostic(
    const EnergyTotals& pre_hydro,
    const EnergyTotals& post_hydro,
    const EnergyTotals& post_ale);

void finalize_ale_closure_audit(AleClosureAuditDiagnostics& audit);

void log_energy_budget_step(const EnergyBudget& budget,
                            const core::Config& cfg,
                            int step,
                            double t);

void log_ale_closure_audit_step(const AleClosureAuditDiagnostics& audit,
                                const EnergyBudget& budget,
                                const core::Config& cfg,
                                int step,
                                double t);

}  // namespace tenryu::diagnostics
