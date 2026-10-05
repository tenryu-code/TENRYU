#pragma once

#include <vector>

#include "core/state.hpp"
#include "hydro/ale_1d_remap.cuh"

namespace tenryu::hydro::ale1d {

// Output and work arrays of project_velocity, kept across ALE attempts.
struct Ale1dVelocityProjectScratch {
  // Cell momenta of the nodal velocities, m_i (v_i + v_{i+1}) / 2, before and
  // after the remap (their sum is the nodal momentum sum_j M_j v_j with
  // M_j = (m_{j-1} + m_j) / 2), and the new mean cell velocity.
  DeviceArray<double> p_old_cell;
  DeviceArray<double> p_new_cell;
  DeviceArray<double> u_new_cell;
  DeviceArray<double> v_new_node;
  // Remapped left- and right-node velocities of each cell and their limited
  // slopes in the mass coordinate.
  DeviceArray<double> psi_left;
  DeviceArray<double> psi_right;
  DeviceArray<double> slope_left;
  DeviceArray<double> slope_right;
  // Kinetic energy closure and reductions.
  DeviceArray<double> ke_node_old;
  DeviceArray<double> ke_node_new;
  DeviceArray<double> deficit;
  DeviceArray<double> capacity;
  DeviceArray<double> deposited;
  DeviceArray<double> reduce_out;
  DeviceArray<unsigned char> reduce_temp;

  void resize(int n_cells);
  [[nodiscard]] bool size_matches(int n_cells) const;
};

struct Ale1dVelocityProjectResult {
  bool success = false;
  double kinetic_energy_drift_rel = 0.0;
  double kinetic_energy_old = 0.0;
  double kinetic_energy_new = 0.0;
  double ke_closure_deposited = 0.0;
};

// Nodal velocity remap by the half-index-shift method (Benson 1992 §3.5.5;
// NUMERICS §3.4.5): each cell carries the velocities of its two nodes,
// psi^L_i = v_i and psi^R_i = v_{i+1}, as specific quantities advected with
// the accepted face mass fluxes (limited linear reconstruction in the mass
// coordinate, the remap's limiter_theta and face taper phi); the new nodal
// velocity is the mass-weighted mean of the two adjacent cells' values,
// v_j = (m_j psi^L_j + m_{j-1} psi^R_{j-1}) / (m_j + m_{j-1}). Without
// transport it returns the old velocities exactly, a uniform velocity stays
// uniform, and sum_j M_j v_j is conserved (up to the centre node, which is
// set to zero).
//
// mass_new, mass_flux and phi_face are device arrays of n, n + 1 and n + 1
// entries (Ale1dRemapScratch::mass_new, mass_flux, phi_face).
Ale1dVelocityProjectResult project_velocity(
    const core::State& state,
    const double* mass_new,
    const double* mass_flux,
    const double* phi_face,
    double limiter_theta,
    bool ke_conservation_closure,
    bool two_temperature,
    const double* ke_remap,
    double* ee_new,
    double* ei_new,
    Ale1dVelocityProjectScratch& scratch);

}  // namespace tenryu::hydro::ale1d
