#pragma once

#include <vector>

#include "laser/port_geometry.hpp"
#include "laser/sector_phase_space.hpp"

namespace tenryu::laser::port_section {

// Power-per-solid-angle profile of the reference beam at one shell:
// I1(mu_axis), where mu_axis = cos(theta). Both phase-space sheets are
// power-weighted into a fixed mu grid; empty interior bins are linearly
// interpolated, ends are clamped, and an all-empty profile is zero.
port_geom::BeamAngularProfile beam_profile_at_shell(
    const sector_ps::PhaseSpaceTable& table, int shell_index,
    int n_mu_profile);

struct IlluminationInput {
  const port_geom::PortTable* ports;
  const sector_ps::PhaseSpaceTable* table;
  int shell_index;
  int n_mu_profile = 64;
  int n_mu_grid = 128;
  int n_phi_grid = 64;
  double I_cut_fraction = 0.01;
};

port_geom::IlluminationResult illumination_at_shell(
    const IlluminationInput& input);

struct CommonWaveDriveInput {
  const port_geom::PortTable* ports;
  const sector_ps::PhaseSpaceTable* table;
  int shell_index;
  double delta_theta_deg;
  int n_mu_grid = 64;
  int n_phi_grid = 32;
  int n_mu_profile = 64;
  // Optional test/diagnostic hook replacing beam_profile_at_shell.
  // Production callers pass nullptr.
  const port_geom::BeamAngularProfile* profile_override = nullptr;
};

struct CommonWaveDriveResult {
  double I_drive;    // integral(I_cw^2 dA) / integral(I_cw dA)
  double I_lower;    // area-weighted mean of per-point max_i I_i
  double I_upper;    // area-weighted mean of per-point sum_i I_i
  int n_sigma_mode;  // mode of winning-cluster member count where I_cw > 0
};

struct CommonWaveDriveGridOutput {
  // mu-major [mu.size()*phi.size()] grids; intensities remain in erg/s/cm2.
  std::vector<double> mu;
  std::vector<double> phi;
  std::vector<double> I_tot;
  std::vector<double> I_cw;
  std::vector<double> n_sigma;
};

CommonWaveDriveResult common_wave_drive(
    const CommonWaveDriveInput& input,
    CommonWaveDriveGridOutput* grid_output = nullptr);

}  // namespace tenryu::laser::port_section
