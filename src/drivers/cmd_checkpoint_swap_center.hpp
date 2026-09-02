#pragma once

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tenryu::core {
struct Config;
struct PolarTierLayout;
struct PolarTierTruncation;
struct State;
}  // namespace tenryu::core

namespace tenryu::drivers {

struct CheckpointSwapCenterThresholds {
  double max_speed_cm_s = 1.0e4;
  double max_mach = 1.0e-4;
  double max_rho_relative = 1.0e-6;
  double max_temperature_relative = 1.0e-6;
  double seam_displacement_floor_cm = 1.0e-9;
  double mirror_z_floor_cm = 1.0e-12;
  double mirror_v_floor_cm_s = 1.0e-6;
  double mirror_tolerance = 1.0e-10;
  double legendre_tolerance = 1.0e-10;
  double ledger_kappa = 32.0;
  double conservation_v_floor_cm_s = 1.0e-6;
};

struct CheckpointSwapCenterOptions {
  std::string namelist_path;
  std::string input_checkpoint;
  std::string output_checkpoint;
  std::optional<int> assert_cut_ring;
  std::optional<double> assert_r_c_cm;
  std::optional<int> assert_core_cells_per_half;
  std::optional<int> assert_bridge_layers;
  CheckpointSwapCenterThresholds thresholds;
  bool dry_run = false;
  bool identity_noop = false;
};

struct CheckpointSwapCenterGateMeasurements {
  double max_speed_cm_s = 0.0;
  double max_mach = 0.0;
  double max_rho_relative = 0.0;
  double max_temperature_relative = 0.0;
  double max_seam_displacement_cm = 0.0;
  double max_mirror_relative = 0.0;
  double max_legendre_relative = 0.0;
};

struct CheckpointSwapCenterConservationTotals {
  long double mass = 0.0L;
  long double energy = 0.0L;
  long double momentum_r = 0.0L;
  long double momentum_z = 0.0L;
  long double abs_mass_terms = 0.0L;
  long double abs_energy_terms = 0.0L;
  long double abs_momentum_r_terms = 0.0L;
  long double abs_momentum_z_terms = 0.0L;
  std::size_t mass_terms = 0;
  std::size_t energy_terms = 0;
  std::size_t momentum_terms = 0;
};

struct CheckpointSwapCenterConservationComparison {
  double delta_mass = 0.0;
  double delta_energy = 0.0;
  double delta_momentum_r = 0.0;
  double delta_momentum_z = 0.0;
  double tolerance_mass = 0.0;
  double tolerance_energy = 0.0;
  double tolerance_momentum_r = 0.0;
  double tolerance_momentum_z = 0.0;
  double momentum_floor = 0.0;
  bool passed = false;
};

struct CheckpointSwapCenterAngularSample {
  double weight = 0.0;
  double mu = 0.0;
  double rho_relative_delta = 0.0;
  double Te_relative_delta = 0.0;
  double Ti_relative_delta = 0.0;
};

#if TENRYU_ENABLE_PYTHON
core::Config make_parent_config(
    const core::Config& segment2,
    const std::string& parent_frozen_config);
#endif

void assert_checkpoint_swap_topology_prefix(
    const std::vector<int>& parent_cell_node_csr_offsets,
    const std::vector<int>& parent_cell_node_csr_indices,
    const std::vector<unsigned char>& parent_cell_nverts,
    const std::vector<int>& target_cell_node_csr_offsets,
    const std::vector<int>& target_cell_node_csr_indices,
    const std::vector<unsigned char>& target_cell_nverts,
    std::size_t kept_cells);

bool checkpoint_swap_gates_pass(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& thresholds);

CheckpointSwapCenterGateMeasurements measure_checkpoint_swap_a_side(
    const core::State& parent,
    const core::Config& cfg,
    const core::PolarTierLayout& truncated,
    const core::PolarTierTruncation& truncation,
    int lmax,
    const CheckpointSwapCenterThresholds& thresholds);

CheckpointSwapCenterGateMeasurements measure_checkpoint_swap_b_side(
    const core::State& parent,
    const core::State& fresh_target,
    std::size_t first_parent_cell,
    std::size_t first_target_cell,
    const CheckpointSwapCenterGateMeasurements& a_side,
    const CheckpointSwapCenterThresholds& thresholds);

void assert_checkpoint_swap_a_side_gates(
    const CheckpointSwapCenterGateMeasurements& measured,
    const CheckpointSwapCenterThresholds& thresholds);

double checkpoint_swap_legendre_margin(
    const std::vector<CheckpointSwapCenterAngularSample>& samples,
    int lmax);

CheckpointSwapCenterConservationComparison
compare_checkpoint_swap_conservation(
    const CheckpointSwapCenterConservationTotals& dropped,
    const CheckpointSwapCenterConservationTotals& fresh,
    double kappa,
    double conservation_v_floor_cm_s = 1.0e-6);

void publish_checkpoint_identity_noop(
    const std::filesystem::path& input,
    const std::filesystem::path& output);

int cmd_checkpoint_swap_center(const CheckpointSwapCenterOptions& options);

}  // namespace tenryu::drivers
