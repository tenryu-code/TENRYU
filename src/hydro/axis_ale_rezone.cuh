#pragma once

#include <array>
#include <limits>
#include <vector>

#include "mesh/mesh.hpp"

namespace tenryu::hydro::axis_ale {

constexpr double kAxisAleCoreShearTriggerThreshold = 1.0e-2;

struct FirstOffAxisRingDiagnostics {
  bool sampled = false;
  double min_edge_length = std::numeric_limits<double>::infinity();
  double min_altitude = std::numeric_limits<double>::infinity();
  int min_edge_cell = -1;
  int min_altitude_cell = -1;
};

struct AxisAleCoreQualityTrigger {
  bool sampled = false;
  bool active = false;
  int sampled_cells = 0;
  double shear_threshold = kAxisAleCoreShearTriggerThreshold;
  double max_shear = 0.0;
  double min_corner_j = std::numeric_limits<double>::infinity();
  int max_shear_cell = -1;
  int min_corner_j_cell = -1;
};

struct AxisAleTargetVolumeGuard {
  bool sampled = false;
  bool passed = true;
  double contraction_floor_frac = 0.5;
  double aggregate_rel_tol = 1.0e-10;
  double min_ratio = std::numeric_limits<double>::infinity();
  double min_current_volume = std::numeric_limits<double>::infinity();
  double min_target_volume = std::numeric_limits<double>::infinity();
  double aggregate_current_volume = 0.0;
  double aggregate_target_volume = 0.0;
  double aggregate_ratio = std::numeric_limits<double>::infinity();
  int min_ratio_cell = -1;
};

struct AxisAlePatchTarget {
  bool active = false;
  std::vector<int> node_ids;
  std::vector<double> r_target;
  std::vector<double> z_target;
  std::vector<double> beta;
  int patch_nodes = 0;
  int core_nodes = 0;
  int seam_nodes = 0;
  int fan_transition_nodes = 0;
  int fixed_axis_nodes = 0;
  int fixed_fan_boundary_nodes = 0;
  bool diagonal_barrier_sampled = false;
  bool diagonal_barrier_passed = true;
  double diagonal_barrier_lambda = 1.0;
  int diagonal_barrier_iters = 0;
  int diagonal_barrier_limited_cell = -1;
  AxisAleTargetVolumeGuard volume_guard;
};

struct AxisAleScaledFullPatchOptions {
  int polar_shell_rings = 4;
  int winslow_iterations = 20;
  double winslow_relaxation = 0.2;
  bool diagonal_barrier_enabled = false;
  double diagonal_barrier_floor_rel = 0.25;
  int diagonal_barrier_max_halves = 24;
};

struct AxisAleCellTargetQuality {
  bool sampled = false;
  double shear = 0.0;
  double aspect_ratio = 0.0;
  double min_corner_j = std::numeric_limits<double>::infinity();
};

struct AxisAleCellTargetAudit {
  bool sampled = false;
  int cell = -1;
  std::array<int, 4> nodes{{-1, -1, -1, -1}};
  AxisAleCellTargetQuality axis_only;
  AxisAleCellTargetQuality patch;
  bool meaningful_shear = false;
  bool shear_reduced = false;
  bool corner_j_ok = false;
  bool passed = false;
  double shear_threshold = 1.0e-2;
  double shear_tolerance = 0.0;
  double corner_j_tolerance = 0.0;
};

struct AxisAleRezoneInput {
  std::vector<int> node_ids;
  std::vector<double> z_tilde;
  std::vector<double> lumped_mass;
  std::vector<double> initial_edge_length;
  std::vector<double> adjacent_cell_area;
  double eta_floor = 1.0e-2;
  double dt = 0.0;
  int patch_winslow_iterations = 20;
  double patch_winslow_relaxation = 0.2;
  double patch_volume_contraction_floor_frac = 0.5;
  double patch_volume_aggregate_rel_tol = 1.0e-10;
};

struct AxisAleRezoneResult {
  bool active = false;
  std::vector<int> node_ids;
  std::vector<double> z_target;
  FirstOffAxisRingDiagnostics first_off_axis_ring;
};

struct AxisAleDiagonalCornerMetrics {
  bool sampled = false;
  int cell = -1;
  int corner = 2;
  std::array<int, 4> nodes{{-1, -1, -1, -1}};
  double j2 = std::numeric_limits<double>::quiet_NaN();
  double b_eff = std::numeric_limits<double>::quiet_NaN();
  double psi_eff = std::numeric_limits<double>::quiet_NaN();
  double psi_target = std::numeric_limits<double>::quiet_NaN();
  double p_psi = std::numeric_limits<double>::quiet_NaN();
  double sigma2 = std::numeric_limits<double>::quiet_NaN();
  double sigma2_target = std::numeric_limits<double>::quiet_NaN();
  double s2 = std::numeric_limits<double>::quiet_NaN();
  double q2 = std::numeric_limits<double>::quiet_NaN();
  double min_altitude = std::numeric_limits<double>::quiet_NaN();
  double min_altitude_reference = std::numeric_limits<double>::quiet_NaN();
  double h2 = std::numeric_limits<double>::quiet_NaN();
  double score_min = std::numeric_limits<double>::quiet_NaN();
};

std::vector<double> project_weighted_lower_bound_pava(
    const std::vector<double>& z_tilde,
    const std::vector<double>& lumped_mass,
    const std::vector<double>& initial_edge_length,
    const std::vector<double>& adjacent_cell_area,
    double eta_floor,
    double dt);

FirstOffAxisRingDiagnostics compute_first_off_axis_ring_diagnostics(
    const mesh::Mesh& mesh,
    const std::vector<int>& axis_node_ids);

// Per-cell baseline hourglass shear of the pristine mesh. The rounded /
// D-shape core seam cells are intrinsically keystone-shaped, so the
// trigger must compare against each cell's OWN initial shear (degradation),
// not an absolute shape threshold.
struct AxisAleCoreQualityBaseline {
  bool valid = false;
  std::vector<double> shear_by_cell;
};

AxisAleCoreQualityBaseline collect_axis_ale_core_quality_baseline(
    const mesh::Mesh& mesh);

AxisAleCoreQualityTrigger compute_axis_ale_core_quality_trigger(
    const mesh::Mesh& mesh,
    double shear_threshold = kAxisAleCoreShearTriggerThreshold,
    const AxisAleCoreQualityBaseline* baseline = nullptr);

AxisAlePatchTarget compute_axis_ale_patch_winslow_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target);

AxisAlePatchTarget compute_axis_ale_scaled_full_patch_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target,
    const std::vector<double>& scaled_target_r,
    const std::vector<double>& scaled_target_z,
    const AxisAleScaledFullPatchOptions& options);

AxisAleDiagonalCornerMetrics compute_axis_ale_diagonal_corner_metrics(
    const mesh::Mesh& mesh,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<double>& r_reference,
    const std::vector<double>& z_reference,
    double scale,
    int cell);

AxisAleDiagonalCornerMetrics compute_axis_ale_worst_diagonal_corner_metrics(
    const mesh::Mesh& mesh,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<double>& r_reference,
    const std::vector<double>& z_reference,
    double scale);

AxisAleCellTargetAudit audit_axis_ale_patch_target_cell(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    const AxisAleRezoneResult& axis_target,
    const AxisAlePatchTarget& patch_target,
    int cell);

AxisAleRezoneResult compute_axis_ale_rezone_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input);

void project_polar_shell_axis_radial_order_target(
    const mesh::Mesh& mesh,
    const AxisAleRezoneInput& input,
    AxisAleRezoneResult& target);

}  // namespace tenryu::hydro::axis_ale
