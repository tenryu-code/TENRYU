#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::hydro::corner_distribution {

struct CellDistributionResult {
  bool bounds_infeasible = false;
  double beta = 1.0;
};

// All arrays have length n_corners. out_corner_mass may alias
// ref_corner_mass. Zero-volume corners are skipped and assigned zero mass.
// This is also the general scalar scaling distributor: for one momentum
// component, pass corner masses as corner_volume, reference corner momenta as
// ref_corner_mass, and corner velocity bounds as rho_min/rho_max. The weights
// then enter through the weighted mean and alpha normalization (and convert
// the scalar bounds to extensive bounds), so the mean is the constant-velocity
// cell-momentum share.
CellDistributionResult distribute_cell_mass_scaling(
    double cell_mass,
    const double* corner_volume,
    const double* ref_corner_mass,
    const double* rho_min,
    const double* rho_max,
    int n_corners,
    double* out_corner_mass);

struct OldMeshView {
  const double* node_r = nullptr;
  const double* node_z = nullptr;
  int n_nodes = 0;
  const int* cell_node_csr_offsets = nullptr;
  const int* cell_node_csr_indices = nullptr;
  int n_cells = 0;
  const double* vel_r = nullptr;
  const double* vel_z = nullptr;
};

// Limited spherical-basis PWL reconstruction over the point control volume of
// old_node. The least-squares stencil is the sorted set of all vertices of all
// incident cells. Each spherical component is limited to the extrema on that
// same stencil, excluding the origin where the basis is undefined. A component
// whose entire stencil is signed zero bypasses the fit and returns exact zero.
// The optional spherical outputs expose the reconstructed scalars before the
// Cartesian-vector rebuild and the component extrema used by the limiter.
void reference_velocity_at(
    const OldMeshView& mesh,
    const std::vector<std::vector<int>>& node_cells,
    int old_node,
    double r,
    double z,
    double* u_r_out,
    double* u_z_out,
    double* u_spherical_r_out = nullptr,
    double* u_spherical_theta_out = nullptr,
    double* u_spherical_r_min_out = nullptr,
    double* u_spherical_r_max_out = nullptr,
    double* u_spherical_theta_min_out = nullptr,
    double* u_spherical_theta_max_out = nullptr);

struct AwAggregateProjectionInputs {
  // Per-node arrays over the caller's compact projection node set.
  const double* u_ref_s = nullptr;
  const double* u_ref_t = nullptr;
  // Spherical basis components at the target node.
  const double* basis_s_r = nullptr;
  const double* basis_s_z = nullptr;
  const double* basis_t_r = nullptr;
  const double* basis_t_z = nullptr;
  // Planar nodal inertia and aggregate physical momentum weight.
  const double* mu_node = nullptr;
  const double* a_node = nullptr;
  // Spherical-component bounds.
  const double* s_min = nullptr;
  const double* s_max = nullptr;
  const double* t_min = nullptr;
  const double* t_max = nullptr;
  // At an axis node the tangential scalar is the R direction and is fixed to
  // exactly zero.
  const std::uint8_t* axis_mask = nullptr;
  int n_nodes = 0;
  // Aggregate Cartesian RZ equality target.
  double target_r = 0.0;
  double target_z = 0.0;
};

struct AwAggregateProjectionResult {
  bool exact_reference = false;  // Arithmetic-zero branch fired.
  bool infeasible = false;
  int active_bound_nodes = 0;
};

// Projects per-node spherical scalars onto one aggregate Cartesian RZ target
// while respecting their component bounds. Axis-node tangential scalars are
// fixed to exactly zero.
AwAggregateProjectionResult project_aggregate_target_state(
    const AwAggregateProjectionInputs& in,
    double* out_u_s,
    double* out_u_t);

struct CornerRemapInputs {
  const double* corner_centroid_r = nullptr;
  const double* corner_centroid_z = nullptr;
  const double* corner_volume = nullptr;
  const double* corner_mass = nullptr;
  const int* containing_old_node = nullptr;
  const double* velocity_r_min = nullptr;
  const double* velocity_r_max = nullptr;
  const double* velocity_z_min = nullptr;
  const double* velocity_z_max = nullptr;
  int n_corners = 0;
  double cell_momentum_r = 0.0;
  double cell_momentum_z = 0.0;
};

struct CornerMomentumDistributionResult {
  CellDistributionResult momentum_r;
  CellDistributionResult momentum_z;
};

// Produces reference corner momenta from the old-mesh PWL velocity at each
// new-corner centroid, then conservatively distributes each momentum component
// using corner_mass as the scaling weight. If a component's reference momenta
// already sum bit-exactly to the zonal momentum, they are copied unchanged.
CornerMomentumDistributionResult distribute_cell_momentum_scaling(
    const OldMeshView& old_mesh,
    const std::vector<std::vector<int>>& old_node_cells,
    const CornerRemapInputs& inputs,
    double* out_corner_momentum_r,
    double* out_corner_momentum_z);

// For each node p, sum mass and momentum over its CSR corner set C(p), then
// recover velocity by division. If every contributing corner has the same
// bitwise velocity, copy that velocity directly. A node with zero mass receives
// zero velocity.
void recover_nodal_velocity(
    const int* node_corner_csr_offsets,
    const int* node_corner_csr_indices,
    int n_nodes,
    const double* corner_mass,
    const double* corner_momentum_r,
    const double* corner_momentum_z,
    double* out_nodal_mass,
    double* out_nodal_momentum_r,
    double* out_nodal_momentum_z,
    double* out_nodal_velocity_r,
    double* out_nodal_velocity_z);

struct AwCornerDistributionInputs {
  // Per-corner planar inertia mu[c,p], using the existing corner-array stride.
  const double* mu_corner = nullptr;
  // Node-common cylindrical lift radius r_p, exactly zero on the axis.
  const double* omega_node = nullptr;
  // Node id for each corner.
  const int* corner_node = nullptr;
  // One target-position reference velocity per node.
  const double* u_ref_r = nullptr;
  const double* u_ref_z = nullptr;
  // Physical conserved cell momentum in Cartesian RZ components.
  double cell_momentum_r = 0.0;
  double cell_momentum_z = 0.0;
  // Per-corner Cartesian velocity bounds, including the remapped cell mean.
  const double* velocity_r_min = nullptr;
  const double* velocity_r_max = nullptr;
  const double* velocity_z_min = nullptr;
  const double* velocity_z_max = nullptr;
  // One on an axis node, where u_R is fixed to exactly +0.0.
  const std::uint8_t* node_axis_mask = nullptr;
  int n_corners = 0;
};

struct AwMomentumDistributionResult {
  double lambda_r = 0.0;
  double lambda_z = 0.0;
  int active_bound_corners = 0;
  bool exact_reference = false;
};

// Solves min 1/2 sum mu*|u-u_ref|^2 subject to sum omega*mu*u=P and
// componentwise bounds, then writes pi=mu*u. The two Cartesian components use
// independent finite active sets. A reference residual within the 64-epsilon
// gate is copied bitwise without a correction.
AwMomentumDistributionResult distribute_cell_momentum_aw(
    const AwCornerDistributionInputs& inputs,
    double* out_pi_r,
    double* out_pi_z);

// Recovers u=u_ref+sum(pi-mu*u_ref)/sum(mu). Axis nodes recover only u_Z and
// receive u_R=+0.0; zero-planar-inertia nodes retain u_ref. The optional
// physical outputs use M_RZ=omega*mu_node and P_RZ=M_RZ*u.
void recover_nodal_velocity_aw(
    const int* node_corner_csr_offsets,
    const int* node_corner_csr_indices,
    int n_nodes,
    const double* mu_corner,
    const double* pi_r,
    const double* pi_z,
    const double* u_ref_r,
    const double* u_ref_z,
    const double* omega_node,
    const std::uint8_t* node_axis_mask,
    double* out_mu_node,
    double* out_velocity_r,
    double* out_velocity_z,
    double* out_mass_rz,
    double* out_momentum_rz_r,
    double* out_momentum_rz_z);

}  // namespace tenryu::hydro::corner_distribution
