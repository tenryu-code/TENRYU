#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/ale_remap.cuh"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/hydro_multiblock_topology.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {

struct AleRezoneIterStats {
  int j_floor_i1_fallback_hits = 0;
  int j_floor_ix_skip_hits = 0;
  int regular_cap_i1_hits = 0;
  int regular_cap_ix_hits = 0;
  int fallback_cap_hits = 0;
  int zero_or_tiny_motion_hits = 0;
  int local_linesearch_rejects = 0;
  int weighted_laplacian_fallback_hits = 0;
  double max_proposed_displacement = 0.0;
  double max_displacement_cap = 0.0;
  double max_proposed_over_cap = 0.0;
};

struct AleMinQualityCell {
  int c = -1;
  int i = -1;
  int j = -1;
};

struct RezoneResult {
  bool triggered = false;
  bool converged = true;
  bool axis_radial_bound_failed = false;
  int iterations = 0;
  double residual = 0.0;
  double min_quality = 1.0;
  bool mesh_tangle = false;
  double accepted_lambda = 1.0;
  int backtrack_attempts = 0;
  int gauss_tangle_rejection_count = 0;
  int corner_tangle_rejection_count = 0;
  int other_rejection_count = 0;
  double final_axis_margin = 1.0e300;
  int axis_radial_bound_fail_j = -1;
  AleRezoneIterStats stats;
  AleMinQualityCell min_quality_cell_pre;
  AleMinQualityCell min_quality_cell_post;
};

struct LambdaSweepPoint {
  double lambda = 0.0;
  double min_gauss_j = std::numeric_limits<double>::infinity();
  double min_corner_j = std::numeric_limits<double>::infinity();
  double min_v_rz = std::numeric_limits<double>::infinity();
  bool admissible = false;
};

struct LambdaSweepResult {
  int target_cell_c = -1;
  int target_cell_i = -1;
  int target_cell_j = -1;
  int max_exp = 20;
  std::vector<LambdaSweepPoint> points;
};

struct LocalBoundaryRepairResult {
  bool fired = false;
  bool applied = false;
  int repaired_node_count = 0;
  int repaired_failing_cells = 0;
  int node_id = -1;
  int node_i = -1;
  int node_j = -1;
  AleMinQualityCell failing_cell;
  int failing_corner = -1;
  double old_coord = 0.0;
  double new_coord = 0.0;
  double min_corner_j = 1.0;
  int ring_size = 0;
  int dof = 0;
  int constraint_count = 0;
  int affected_cell_count = 0;
  int solver_iterations = 0;
  int infeasible_constraint = -1;
  const char* variable = "none";
  const char* reason = "not attempted";
};

struct EmergencyCellDeactivationResult {
  bool fired = false;
  bool applied = false;
  int target_cell_c = -1;
  int neighbor_cell_c = -1;
  AleMinQualityCell target_cell;
  AleMinQualityCell neighbor_cell;
  int failing_corner = -1;
  double mass_transferred = 0.0;
  double ee_transferred = 0.0;
  double ei_transferred = 0.0;
  double mass_floor_delta = 0.0;
  double E_floor_injected = 0.0;
  double target_mass_before = 0.0;
  double target_volume_before = 0.0;
  double target_rho_before = 0.0;
  double target_mass_floor = 0.0;
  double before_rho = 0.0;
  double before_p = 0.0;
  double before_Te = 0.0;
  double before_Ti = 0.0;
  double after_rho = 0.0;
  double after_p = 0.0;
  double after_Te = 0.0;
  double after_Ti = 0.0;
  double momentum_delta_r = 0.0;
  double momentum_delta_z = 0.0;
  double energy_delta = 0.0;
  int target_hydro_active_after = -1;
  int target_cell_is_void_after = -1;
  double min_corner_j = 1.0;
  const char* reason = "not attempted";
};

struct AxisMarginResult {
  double min_margin = 1.0e300;
  int min_j = -1;
};

struct AxisRadialBoundResult {
  bool passed = true;
  int fail_j = -1;
};

namespace detail {

constexpr double kGauss = 0.577350269189625764509148780501957456;
constexpr double kJFloor = 1.0e-30;
constexpr double kAxisCapFloorFraction = 0.25;
constexpr double kTinyMotion2 = 1.0e-30;
constexpr double kPhase9AxisEmergencyFraction = 0.01;
constexpr double kPhase10SpineFallbackFraction = 0.05;
constexpr int kBacktrackMaxAttempts = 6;
constexpr double kBacktrackLambdaSchedule[kBacktrackMaxAttempts] = {
    1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125};

__host__ __device__ inline double cross2(const double ar,
                                         const double az,
                                         const double br,
                                         const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ inline void tri_corner_jacobians_canonical(
    const double* rr,
    const double* zz,
    double* j) {
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    const double raw = cross2(rr[kp] - rr[k], zz[kp] - zz[k],
                              rr[km] - rr[k], zz[km] - zz[k]);
    j[k] = -raw;
  }
}

__host__ __device__ inline void quad_corner_jacobians_canonical_polar(
    const double* rr,
    const double* zz,
    double* j) {
  for (int corner = 0; corner < 4; ++corner) {
    j[corner] = -tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
  }
}

inline void accumulate_rezone_stats(AleRezoneIterStats& total,
                                    const AleRezoneIterStats& iter) {
  total.j_floor_i1_fallback_hits += iter.j_floor_i1_fallback_hits;
  total.j_floor_ix_skip_hits += iter.j_floor_ix_skip_hits;
  total.regular_cap_i1_hits += iter.regular_cap_i1_hits;
  total.regular_cap_ix_hits += iter.regular_cap_ix_hits;
  total.fallback_cap_hits += iter.fallback_cap_hits;
  total.zero_or_tiny_motion_hits += iter.zero_or_tiny_motion_hits;
  total.local_linesearch_rejects += iter.local_linesearch_rejects;
  total.weighted_laplacian_fallback_hits += iter.weighted_laplacian_fallback_hits;
  total.max_proposed_displacement =
      std::max(total.max_proposed_displacement, iter.max_proposed_displacement);
  total.max_displacement_cap =
      std::max(total.max_displacement_cap, iter.max_displacement_cap);
  total.max_proposed_over_cap =
      std::max(total.max_proposed_over_cap, iter.max_proposed_over_cap);
}

inline std::string format_min_quality_cell(const AleMinQualityCell& cell) {
  return "(c=" + std::to_string(cell.c) + ",i=" + std::to_string(cell.i) +
         ",j=" + std::to_string(cell.j) + ")";
}

inline int state_cell_active_nverts(const core::State& state,
                                    const int c,
                                    const int n_cells) {
  return state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
             ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
             : mesh::kMeshTopoCellStorageSlots;
}

inline std::string format_ale_scalar(const double value) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(12) << value;
  return os.str();
}

inline std::string format_gate_reason(const char* gate_reason) {
  if (gate_reason == nullptr || gate_reason[0] == '\0') {
    return std::string();
  }
  return " gate_reason=" + std::string(gate_reason);
}

inline void log_ale_rezone_stats(const int iterations,
                                 const AleRezoneIterStats& stats,
                                 const AleMinQualityCell& pre_cell,
                                 const AleMinQualityCell& post_cell,
                                 const double final_axis_margin,
                                 const int rollback_count,
                                 const std::string& last_reason) {
  core::log_warning("[ale-stats] iter=" + std::to_string(iterations) +
                    " i1_fb=" + std::to_string(stats.j_floor_i1_fallback_hits) +
                    " ix_skip=" + std::to_string(stats.j_floor_ix_skip_hits) +
                    " i1_cap=" + std::to_string(stats.regular_cap_i1_hits) +
                    " ix_cap=" + std::to_string(stats.regular_cap_ix_hits) +
                    " fb_cap=" + std::to_string(stats.fallback_cap_hits) +
                    " tiny=" + std::to_string(stats.zero_or_tiny_motion_hits) +
                    " local_ls_reject=" +
                    std::to_string(stats.local_linesearch_rejects) +
                    " weighted_lap_fb=" +
                    std::to_string(stats.weighted_laplacian_fallback_hits) +
                    " max_raw_disp=" +
                    format_ale_scalar(stats.max_proposed_displacement) +
                    " max_cap=" + format_ale_scalar(stats.max_displacement_cap) +
                    " max_raw_over_cap=" +
                    format_ale_scalar(stats.max_proposed_over_cap));
  core::log_warning("[ale-stats] min_quality_cell pre=" +
                    format_min_quality_cell(pre_cell) +
                    " post=" + format_min_quality_cell(post_cell));
  core::log_warning("[ale-stats] rollback_count=" + std::to_string(rollback_count) +
                    " axis_margin_min=" + std::to_string(final_axis_margin) +
                    " last_reason=\"" + last_reason + "\"");
}

inline void log_ale_backtrack_stats(const double accepted_lambda,
                                    const int trials,
                                    const AleMinQualityCell& first_fail_cell,
                                    const int gauss_tangle_rejection_count,
                                    const int corner_tangle_rejection_count,
                                    const int other_rejection_count) {
  core::log_warning("[ale-stats] backtrack_lambda_accepted=" +
                    std::to_string(accepted_lambda) +
                    " trials=" + std::to_string(trials) +
                    " first_fail_cell=" + format_min_quality_cell(first_fail_cell) +
                    " gauss_tangle_rejection_count=" +
                    std::to_string(gauss_tangle_rejection_count) +
                    " corner_tangle_rejection_count=" +
                    std::to_string(corner_tangle_rejection_count) +
                    " other_rejection_count=" + std::to_string(other_rejection_count));
}

inline void log_local_boundary_repair_stats(
    const LocalBoundaryRepairResult& repair,
    const char* gate_reason = nullptr) {
  core::log_warning("[ale-stats] local_boundary_repositioning fired_count=" +
                    std::to_string(repair.fired ? 1 : 0) +
                    format_gate_reason(gate_reason) +
                    " applied_count=" + std::to_string(repair.applied ? 1 : 0) +
                    " repaired_node_count=" +
                    std::to_string(repair.repaired_node_count) +
                    " repaired_failing_cells=" +
                    std::to_string(repair.repaired_failing_cells) +
                    " node=(id=" + std::to_string(repair.node_id) +
                    ",i=" + std::to_string(repair.node_i) +
                    ",j=" + std::to_string(repair.node_j) + ")" +
                    " failing_cell=" + format_min_quality_cell(repair.failing_cell) +
                    " failing_corner=" + std::to_string(repair.failing_corner) +
                    " variable=\"" + std::string(repair.variable) + "\"" +
                    " old_coord=" + format_ale_scalar(repair.old_coord) +
                    " new_coord=" + format_ale_scalar(repair.new_coord) +
                    " min_corner_J=" + format_ale_scalar(repair.min_corner_j) +
                    " ring_size=" + std::to_string(repair.ring_size) +
                    " dof_count=" + std::to_string(repair.dof) +
                    " constraint_count=" + std::to_string(repair.constraint_count) +
                    " affected_cells=" + std::to_string(repair.affected_cell_count) +
                    " solver_iterations=" +
                    std::to_string(repair.solver_iterations) +
                    " infeasible_constraint=" +
                    std::to_string(repair.infeasible_constraint) +
                    " reason=\"" + std::string(repair.reason) + "\"");
}

inline void log_multi_node_boundary_repair_stats(
    const LocalBoundaryRepairResult& repair,
    const char* gate_reason = nullptr) {
  core::log_warning("[ale-stats] multi_node_boundary_repair fired_count=" +
                    std::to_string(repair.fired ? 1 : 0) +
                    format_gate_reason(gate_reason) +
                    " applied_count=" + std::to_string(repair.applied ? 1 : 0) +
                    " ring_size=" + std::to_string(repair.ring_size) +
                    " dof_count=" + std::to_string(repair.dof) +
                    " constraint_count=" + std::to_string(repair.constraint_count) +
                    " affected_cells=" + std::to_string(repair.affected_cell_count) +
                    " solver_iterations=" +
                    std::to_string(repair.solver_iterations) +
                    " failing_cell=" + format_min_quality_cell(repair.failing_cell) +
                    " failing_corner=" + std::to_string(repair.failing_corner) +
                    " infeasible_constraint=" +
                    std::to_string(repair.infeasible_constraint) +
                    " reason=\"" + std::string(repair.reason) + "\"");
}

inline void log_emergency_cell_deactivation_stats(
    const EmergencyCellDeactivationResult& deactivation,
    const char* gate_reason = nullptr) {
  core::log_warning("[ale-stats] emergency_cell_deactivation fired_count=" +
                    std::to_string(deactivation.fired ? 1 : 0) +
                    format_gate_reason(gate_reason) +
                    " applied_count=" +
                    std::to_string(deactivation.applied ? 1 : 0) +
                    " target_cell=" +
                    format_min_quality_cell(deactivation.target_cell) +
                    " neighbor_cell=" +
                    format_min_quality_cell(deactivation.neighbor_cell) +
                    " failing_corner=" +
                    std::to_string(deactivation.failing_corner) +
                    " mass_transferred=" +
                    format_ale_scalar(deactivation.mass_transferred) +
                    " ee_transferred=" +
                    format_ale_scalar(deactivation.ee_transferred) +
                    " ei_transferred=" +
                    format_ale_scalar(deactivation.ei_transferred) +
                    " mass_floor_delta=" +
                    format_ale_scalar(deactivation.mass_floor_delta) +
                    " E_floor_injected=" +
                    format_ale_scalar(deactivation.E_floor_injected) +
                    " target_mass_before=" +
                    format_ale_scalar(deactivation.target_mass_before) +
                    " target_volume_before=" +
                    format_ale_scalar(deactivation.target_volume_before) +
                    " target_rho_before=" +
                    format_ale_scalar(deactivation.target_rho_before) +
                    " target_mass_floor=" +
                    format_ale_scalar(deactivation.target_mass_floor) +
                    " target_hydro_active_after=" +
                    std::to_string(deactivation.target_hydro_active_after) +
                    " target_cell_is_void_after=" +
                    std::to_string(deactivation.target_cell_is_void_after) +
                    " min_corner_J=" +
                    format_ale_scalar(deactivation.min_corner_j) +
                    " reason=\"" + std::string(deactivation.reason) + "\"");
  core::log_warning("[ale-audit] emergency_cell_deactivation"
                    " cell_id=" +
                    std::to_string(deactivation.target_cell_c) +
                    " reason=\"" + std::string(deactivation.reason) + "\"" +
                    " mass_delta=" +
                    format_ale_scalar(deactivation.mass_floor_delta) +
                    " momentum_delta_r=" +
                    format_ale_scalar(deactivation.momentum_delta_r) +
                    " momentum_delta_z=" +
                    format_ale_scalar(deactivation.momentum_delta_z) +
                    " energy_delta=" +
                    format_ale_scalar(deactivation.energy_delta) +
                    " before_rho=" +
                    format_ale_scalar(deactivation.before_rho) +
                    " before_p=" +
                    format_ale_scalar(deactivation.before_p) +
                    " before_Te=" +
                    format_ale_scalar(deactivation.before_Te) +
                    " before_Ti=" +
                    format_ale_scalar(deactivation.before_Ti) +
                    " after_rho=" +
                    format_ale_scalar(deactivation.after_rho) +
                    " after_p=" +
                    format_ale_scalar(deactivation.after_p) +
                    " after_Te=" +
                    format_ale_scalar(deactivation.after_Te) +
                    " after_Ti=" +
                    format_ale_scalar(deactivation.after_Ti));
}

inline void log_ale_damage_stats(const RemapDamageResult& dmg) {
  core::log_warning("[ale-stats] damage D_rho_max=" + std::to_string(dmg.D_rho_max) +
                    " A_axis_max=" + std::to_string(dmg.A_axis_max) +
                    " axis_max_j=" + std::to_string(dmg.axis_max_j));
}

__device__ inline double atomic_max_double(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value > __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline void record_rezone_displacement_cap_diag(
    AleRezoneIterStats* stats,
    const double proposed_displacement,
    const double displacement_cap) {
  if (stats == nullptr) {
    return;
  }
  if (isfinite(proposed_displacement) && proposed_displacement >= 0.0) {
    atomic_max_double(&stats->max_proposed_displacement, proposed_displacement);
  }
  if (isfinite(displacement_cap) && displacement_cap >= 0.0) {
    atomic_max_double(&stats->max_displacement_cap, displacement_cap);
  }
  if (isfinite(proposed_displacement) && isfinite(displacement_cap) &&
      proposed_displacement >= 0.0 && displacement_cap > 0.0) {
    atomic_max_double(&stats->max_proposed_over_cap,
                      proposed_displacement / displacement_cap);
  }
}

// Closed-form analytic positivity margin for an axis-row cell (i=0, j).
// Returns M = s * min(r_outer_j, r_outer_jp1) + min(Q, 0), where M > 0
// implies non-tangling over the full bilinear reference cell.
__device__ inline double axis_cell_margin(const double r_outer_j,
                                          const double z_outer_j,
                                          const double r_outer_jp1,
                                          const double z_outer_jp1,
                                          const double z_axis_j,
                                          const double z_axis_jp1) {
  const double s = z_axis_jp1 - z_axis_j;
  if (s <= 0.0 || r_outer_j <= 0.0 || r_outer_jp1 <= 0.0) {
    return -1.0;
  }
  const double Q = r_outer_j * (z_outer_jp1 - z_axis_jp1) -
                   r_outer_jp1 * (z_outer_j - z_axis_j);
  return s * fmin(r_outer_j, r_outer_jp1) + fmin(Q, 0.0);
}

static __global__ void axis_margin_min_kernel(double* __restrict__ d_min_margin,
                                              int* __restrict__ d_min_j,
                                              const double* __restrict__ x_r,
                                              const double* __restrict__ x_z,
                                              const int nz,
                                              const bool has_physical_rz_axis) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (!has_physical_rz_axis || j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis_j = j;
  const int n_axis_jp1 = j + 1;
  const int n_outer_j = stride + j;
  const int n_outer_jp1 = stride + (j + 1);

  const double margin = axis_cell_margin(x_r[n_outer_j],
                                         x_z[n_outer_j],
                                         x_r[n_outer_jp1],
                                         x_z[n_outer_jp1],
                                         x_z[n_axis_j],
                                         x_z[n_axis_jp1]);

  auto* address_as_ull = reinterpret_cast<unsigned long long*>(d_min_margin);
  unsigned long long old = *address_as_ull;
  while (margin < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(margin)));
    if (assumed == old) {
      atomicExch(d_min_j, j);
      break;
    }
  }
}

static __global__ void axis_spine_target_kernel(
    double* __restrict__ z_axis_target,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_z_cand,
    const std::uint8_t* __restrict__ node_flags,
    const int nz,
    const double omega,
    const bool has_physical_rz_axis) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (!has_physical_rz_axis || j > nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis = j;
  const int n_inner = stride + j;

  const std::uint8_t flags = node_flags[n_axis];
  if ((flags & (mesh::NODE_BOUNDARY | mesh::NODE_CENTER)) != 0u) {
    z_axis_target[j] = x_z_old[n_axis];
    return;
  }

  const double z_old = x_z_old[n_axis];
  const double z_inner_cand = x_z_cand[n_inner];
  z_axis_target[j] = z_old + omega * (z_inner_cand - z_old);
}

inline bool axis_spine_fixed(const std::uint8_t flags) {
  return (flags & (mesh::NODE_BOUNDARY | mesh::NODE_CENTER)) != 0u;
}

inline double axis_spine_spacing_floor(const double z, const double dz_init) {
  return std::max(1.0e-12 * std::fabs(dz_init),
                  100.0 * std::numeric_limits<double>::epsilon() * std::fabs(z));
}

inline bool axis_spine_is_symmetric(const std::vector<double>& z_axis_old,
                                    const double dz_init) {
  const int n = static_cast<int>(z_axis_old.size());
  const double tol = 1.0e-10 * std::fabs(dz_init);
  for (int j = 0; j <= n / 2; ++j) {
    if (std::fabs(z_axis_old[static_cast<std::size_t>(j)] +
                  z_axis_old[static_cast<std::size_t>(n - 1 - j)]) > tol) {
      return false;
    }
  }
  return true;
}

inline void apply_axis_spine_symmetry(std::vector<double>& z_axis_target,
                                      const std::vector<std::uint8_t>& axis_flags) {
  const int n = static_cast<int>(z_axis_target.size());
  for (int j = 0; j <= n / 2; ++j) {
    const int j_mirror = n - 1 - j;
    const bool fixed_j = axis_spine_fixed(axis_flags[static_cast<std::size_t>(j)]);
    const bool fixed_mirror =
        axis_spine_fixed(axis_flags[static_cast<std::size_t>(j_mirror)]);
    if (j == j_mirror) {
      if (!fixed_j) {
        z_axis_target[static_cast<std::size_t>(j)] = 0.0;
      }
      continue;
    }
    if (fixed_j && fixed_mirror) {
      continue;
    }
    if (fixed_j) {
      z_axis_target[static_cast<std::size_t>(j_mirror)] =
          -z_axis_target[static_cast<std::size_t>(j)];
      continue;
    }
    if (fixed_mirror) {
      z_axis_target[static_cast<std::size_t>(j)] =
          -z_axis_target[static_cast<std::size_t>(j_mirror)];
      continue;
    }
    const double avg = 0.5 * (z_axis_target[static_cast<std::size_t>(j)] -
                              z_axis_target[static_cast<std::size_t>(j_mirror)]);
    z_axis_target[static_cast<std::size_t>(j)] = avg;
    z_axis_target[static_cast<std::size_t>(j_mirror)] = -avg;
  }
}

inline void apply_axis_spine_monotonicity(std::vector<double>& z_axis_target,
                                          const std::vector<double>& z_axis_old,
                                          const std::vector<std::uint8_t>& axis_flags,
                                          const double dz_init) {
  const int n = static_cast<int>(z_axis_target.size());
  if (n < 2) {
    return;
  }

  for (int j = 0; j < n; ++j) {
    if (axis_spine_fixed(axis_flags[static_cast<std::size_t>(j)])) {
      z_axis_target[static_cast<std::size_t>(j)] =
          z_axis_old[static_cast<std::size_t>(j)];
    }
  }

  std::vector<double> s_min(static_cast<std::size_t>(n), 0.0);
  for (int j = 1; j < n; ++j) {
    s_min[static_cast<std::size_t>(j)] = axis_spine_spacing_floor(
        z_axis_target[static_cast<std::size_t>(j)], dz_init);
  }

  std::vector<int> fixed_indices;
  fixed_indices.reserve(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    if (axis_spine_fixed(axis_flags[static_cast<std::size_t>(j)])) {
      fixed_indices.push_back(j);
    }
  }

  if (fixed_indices.empty()) {
    for (int j = 1; j < n; ++j) {
      const double s = z_axis_target[static_cast<std::size_t>(j)] -
                       z_axis_target[static_cast<std::size_t>(j - 1)];
      if (s < s_min[static_cast<std::size_t>(j)]) {
        z_axis_target[static_cast<std::size_t>(j)] =
            z_axis_target[static_cast<std::size_t>(j - 1)] +
            s_min[static_cast<std::size_t>(j)];
      }
    }
    return;
  }

  const int first_fixed = fixed_indices.front();
  for (int j = first_fixed - 1; j >= 0; --j) {
    z_axis_target[static_cast<std::size_t>(j)] =
        std::min(z_axis_target[static_cast<std::size_t>(j)],
                 z_axis_target[static_cast<std::size_t>(j + 1)] -
                     s_min[static_cast<std::size_t>(j + 1)]);
  }

  for (std::size_t seg = 0; seg + 1 < fixed_indices.size(); ++seg) {
    const int lo = fixed_indices[seg];
    const int hi = fixed_indices[seg + 1];
    if (hi <= lo + 1) {
      continue;
    }

    std::vector<double> upper(static_cast<std::size_t>(n), 0.0);
    double tail = 0.0;
    for (int j = hi - 1; j > lo; --j) {
      tail += s_min[static_cast<std::size_t>(j + 1)];
      upper[static_cast<std::size_t>(j)] =
          z_axis_target[static_cast<std::size_t>(hi)] - tail;
    }

    double lower = z_axis_target[static_cast<std::size_t>(lo)];
    for (int j = lo + 1; j < hi; ++j) {
      lower += s_min[static_cast<std::size_t>(j)];
      const double upper_j = upper[static_cast<std::size_t>(j)];
      if (lower <= upper_j) {
        z_axis_target[static_cast<std::size_t>(j)] =
            std::clamp(z_axis_target[static_cast<std::size_t>(j)], lower, upper_j);
      } else {
        z_axis_target[static_cast<std::size_t>(j)] = upper_j;
      }
      lower = z_axis_target[static_cast<std::size_t>(j)];
    }
  }

  const int last_fixed = fixed_indices.back();
  for (int j = last_fixed + 1; j < n; ++j) {
    const double s = z_axis_target[static_cast<std::size_t>(j)] -
                     z_axis_target[static_cast<std::size_t>(j - 1)];
    if (s < s_min[static_cast<std::size_t>(j)]) {
      z_axis_target[static_cast<std::size_t>(j)] =
          z_axis_target[static_cast<std::size_t>(j - 1)] +
          s_min[static_cast<std::size_t>(j)];
    }
  }

  for (int j = 0; j < n; ++j) {
    if (axis_spine_fixed(axis_flags[static_cast<std::size_t>(j)])) {
      z_axis_target[static_cast<std::size_t>(j)] =
          z_axis_old[static_cast<std::size_t>(j)];
    }
  }
}

inline void apply_axis_spine_constraints(std::vector<double>& z_axis_target,
                                         const std::vector<double>& z_axis_old,
                                         const std::vector<std::uint8_t>& axis_flags,
                                         const double dz_init) {
  const int n = static_cast<int>(z_axis_target.size());
  if (n < 2) {
    return;
  }

  if (axis_spine_is_symmetric(z_axis_old, dz_init)) {
    apply_axis_spine_symmetry(z_axis_target, axis_flags);
  }
  apply_axis_spine_monotonicity(z_axis_target, z_axis_old, axis_flags, dz_init);
}

static __global__ void apply_axis_spine_kernel(double* __restrict__ x_r_cand,
                                               double* __restrict__ x_z_cand,
                                               const double* __restrict__ z_axis_target,
                                               const int nz,
                                               const bool has_physical_rz_axis) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (!has_physical_rz_axis || j > nz) {
    return;
  }
  const int n_axis = j;
  x_r_cand[n_axis] = 0.0;
  x_z_cand[n_axis] = z_axis_target[j];
}

inline void apply_axis_spine_projection(double* x_r_cand,
                                        double* x_z_cand,
                                        double* d_z_axis_target,
                                        std::vector<double>& z_axis_target,
                                        const std::vector<double>& z_axis_old,
                                        const std::vector<std::uint8_t>& axis_flags,
                                        const int nz,
                                        const double dz_init,
                                        const bool has_physical_rz_axis) {
  if (!has_physical_rz_axis) {
    return;
  }
  const int axis_nodes = nz + 1;
  CUDA_CHECK(cudaMemcpy(z_axis_target.data(),
                        x_z_cand,
                        static_cast<std::size_t>(axis_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  apply_axis_spine_constraints(z_axis_target, z_axis_old, axis_flags, dz_init);
  CUDA_CHECK(cudaMemcpy(d_z_axis_target,
                        z_axis_target.data(),
                        static_cast<std::size_t>(axis_nodes) * sizeof(double),
                        cudaMemcpyHostToDevice));
  const int blocks_axis = (axis_nodes + 255) / 256;
  apply_axis_spine_kernel<<<blocks_axis, 256>>>(
      x_r_cand, x_z_cand, d_z_axis_target, nz, has_physical_rz_axis);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

static __global__ void axis_radial_bound_check_kernel(
    int* __restrict__ failed,
    int* __restrict__ fail_j,
    const double* __restrict__ x_r_trial,
    const double* __restrict__ x_r_lagrangian,
    const int nr,
    const int nz,
    const double kappa) {
  if (nr < 1 || !(kappa > 0.0)) {
    return;
  }
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  const int stride = nz + 1;
  const int n = stride + j;
  const double r_trial = x_r_trial[n];
  const double r_min = kappa * x_r_lagrangian[n];
  if (!isfinite(r_trial) || r_trial < r_min) {
    atomicExch(failed, 1);
    atomicMin(fail_j, j);
  }
}

static __global__ void mesh_quality_check_kernel(double* __restrict__ q_cell,
                                                 const double* __restrict__ x_r,
                                                 const double* __restrict__ x_z,
                                                 const std::uint8_t* __restrict__ cell_nverts,
                                                 int* __restrict__ mesh_tangle,
                                                 const int nr,
                                                 const int nz,
                                                 const bool reject_zero_gauss_j,
                                                 const double zero_gauss_j_floor_rel) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  if (cell_nverts != nullptr) {
    const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    const int nodes[4] = {n00, n10, n11, n01};
    double rr[4];
    double zz[4];
    for (int k = 0; k < nverts; ++k) {
      rr[k] = x_r[nodes[k]];
      zz[k] = x_z[nodes[k]];
    }
    double j_corner[4] = {};
    if (nverts == 3) {
      tri_corner_jacobians_canonical(rr, zz, j_corner);
    } else {
      quad_corner_jacobians_canonical_polar(rr, zz, j_corner);
    }

    double J_min = 1.0e300;
    double J_max = -1.0e300;
    for (int k = 0; k < nverts; ++k) {
      const double J = j_corner[k];
      J_min = fmin(J_min, J);
      J_max = fmax(J_max, J);
      if (!isfinite(J) || (!reject_zero_gauss_j && J < 0.0)) {
        atomicExch(mesh_tangle, 1);
      }
    }
    const double J_max_eff = fmax(J_max, kJFloor);
    if (reject_zero_gauss_j) {
      const double J_floor = fmax(kJFloor, zero_gauss_j_floor_rel * J_max_eff);
      if (J_min <= J_floor) {
        atomicExch(mesh_tangle, 1);
      }
    }
    if (J_max < kJFloor) {
      q_cell[c] = 0.0;
      atomicExch(mesh_tangle, 1);
      return;
    }
    q_cell[c] = J_min / J_max_eff;
    return;
  }

  const double r1 = x_r[n00];
  const double z1 = x_z[n00];
  const double r2 = x_r[n10];
  const double z2 = x_z[n10];
  const double r3 = x_r[n11];
  const double z3 = x_z[n11];
  const double r4 = x_r[n01];
  const double z4 = x_z[n01];

  const double xis[2] = {-kGauss, kGauss};
  const double etas[2] = {-kGauss, kGauss};

  double J_min = 1.0e300;
  double J_max = -1.0e300;

  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const double xi = xis[a];
      const double eta = etas[b];

      const double dN1_dxi = -0.25 * (1.0 - eta);
      const double dN2_dxi = 0.25 * (1.0 - eta);
      const double dN3_dxi = 0.25 * (1.0 + eta);
      const double dN4_dxi = -0.25 * (1.0 + eta);

      const double dN1_deta = -0.25 * (1.0 - xi);
      const double dN2_deta = -0.25 * (1.0 + xi);
      const double dN3_deta = 0.25 * (1.0 + xi);
      const double dN4_deta = 0.25 * (1.0 - xi);

      const double dr_dxi = dN1_dxi * r1 + dN2_dxi * r2 + dN3_dxi * r3 + dN4_dxi * r4;
      const double dz_dxi = dN1_dxi * z1 + dN2_dxi * z2 + dN3_dxi * z3 + dN4_dxi * z4;
      const double dr_deta =
          dN1_deta * r1 + dN2_deta * r2 + dN3_deta * r3 + dN4_deta * r4;
      const double dz_deta =
          dN1_deta * z1 + dN2_deta * z2 + dN3_deta * z3 + dN4_deta * z4;

      const double J = dr_dxi * dz_deta - dr_deta * dz_dxi;
      J_min = fmin(J_min, J);
      J_max = fmax(J_max, J);
      if (!reject_zero_gauss_j && J < 0.0) {
        atomicExch(mesh_tangle, 1);
      }
    }
  }

  const double J_max_eff = fmax(J_max, kJFloor);
  if (reject_zero_gauss_j) {
    const double J_floor = fmax(kJFloor, zero_gauss_j_floor_rel * J_max_eff);
    if (J_min <= J_floor) {
      atomicExch(mesh_tangle, 1);
    }
  }
  if (J_max < kJFloor) {
    q_cell[c] = 0.0;
    atomicExch(mesh_tangle, 1);
    return;
  }

  q_cell[c] = J_min / J_max_eff;
}

static __global__ void corner_post_tangle_kernel(
    double* __restrict__ min_corner_j_cell,
    int* __restrict__ first_failing_corner_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz,
    const bool strict_floor_enabled,
    const double strict_floor_eps) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    min_corner_j_cell[c] = 1.0e300;
    first_failing_corner_cell[c] = -1;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int nodes[4] = {
      tenryu::hydro::rz_node_index_2d(i, j, nz),
      tenryu::hydro::rz_node_index_2d(i + 1, j, nz),
      tenryu::hydro::rz_node_index_2d(i + 1, j + 1, nz),
      tenryu::hydro::rz_node_index_2d(i, j + 1, nz),
  };

  if (cell_nverts != nullptr) {
    const int nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    double rr[4];
    double zz[4];
    for (int k = 0; k < nverts; ++k) {
      const int n = nodes[k];
      rr[k] = x_r[n];
      zz[k] = x_z[n];
    }
    double corner_j[4] = {};
    if (nverts == 3) {
      tri_corner_jacobians_canonical(rr, zz, corner_j);
    } else {
      quad_corner_jacobians_canonical_polar(rr, zz, corner_j);
    }

    double min_corner_j = 1.0e300;
    int first_failing_corner = -1;
    double max_abs_corner_j = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      const double j_corner = corner_j[corner];
      if (!isfinite(j_corner)) {
        min_corner_j = -1.0e300;
        if (first_failing_corner < 0) {
          first_failing_corner = corner;
        }
        continue;
      }
      min_corner_j = fmin(min_corner_j, j_corner);
      max_abs_corner_j = fmax(max_abs_corner_j, fabs(j_corner));
      if (!strict_floor_enabled && !(j_corner > 0.0) &&
          first_failing_corner < 0) {
        first_failing_corner = corner;
      }
    }
    if (strict_floor_enabled) {
      const double floor_eps = fmax(strict_floor_eps, 0.0);
      const double j_floor = fmax(kJFloor, floor_eps * max_abs_corner_j);
      for (int corner = 0; corner < nverts; ++corner) {
        const double j_corner = corner_j[corner];
        if (!isfinite(j_corner) || !(j_corner > 0.0) || j_corner <= j_floor) {
          first_failing_corner = corner;
          break;
        }
      }
    }

    min_corner_j_cell[c] = min_corner_j;
    first_failing_corner_cell[c] = first_failing_corner;
    return;
  }

  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    const int n = nodes[k];
    rr[k] = x_r[n];
    zz[k] = x_z[n];
  }

  double min_corner_j = 1.0e300;
  int first_failing_corner = -1;
  if (!strict_floor_enabled) {
    for (int corner = 0; corner < 4; ++corner) {
      const double j_corner =
          tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
      if (!isfinite(j_corner)) {
        min_corner_j = -1.0e300;
        if (first_failing_corner < 0) {
          first_failing_corner = corner;
        }
        continue;
      }
      min_corner_j = fmin(min_corner_j, j_corner);
      if (!(j_corner > 0.0) && first_failing_corner < 0) {
        first_failing_corner = corner;
      }
    }
  } else {
    double corner_j[4];
    double max_abs_corner_j = 0.0;
    for (int corner = 0; corner < 4; ++corner) {
      const double j_corner =
          tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
      corner_j[corner] = j_corner;
      if (!isfinite(j_corner)) {
        min_corner_j = -1.0e300;
        continue;
      }
      min_corner_j = fmin(min_corner_j, j_corner);
      max_abs_corner_j = fmax(max_abs_corner_j, fabs(j_corner));
    }
    const double floor_eps = fmax(strict_floor_eps, 0.0);
    const double j_floor = fmax(kJFloor, floor_eps * max_abs_corner_j);
    for (int corner = 0; corner < 4; ++corner) {
      const double j_corner = corner_j[corner];
      if (!isfinite(j_corner) || !(j_corner > 0.0) || j_corner <= j_floor) {
        first_failing_corner = corner;
        break;
      }
    }
  }

  min_corner_j_cell[c] = min_corner_j;
  first_failing_corner_cell[c] = first_failing_corner;
}

struct RzNodeValue {
  double r;
  double z;
};

__device__ inline bool reference_mesh_target_enabled(
    const tenryu::hydro::BC2DRZAxis& axis) {
  return axis.mesh_normal == tenryu::hydro::BCMeshNormalKind::ClampedAtBoundary &&
         axis.mesh_tangential == tenryu::hydro::BCMeshTangentialKind::ReferenceTarget;
}

__device__ inline RzNodeValue apply_reference_mesh_boundary_target(
    const RzNodeValue& candidate,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int nr,
    const int nz,
    const int node_i,
    const int node_j,
    const tenryu::hydro::BC2DRZConfig& bc_config) {
  if (x_r_reference == nullptr || x_z_reference == nullptr) {
    return candidate;
  }
  const int stride = nz + 1;
  const int n = node_i * stride + node_j;
  RzNodeValue out = candidate;
  if (node_i == 0 && reference_mesh_target_enabled(bc_config.r_inner)) {
    out.r = x_r_reference[n];
    out.z = x_z_reference[n];
  }
  if (node_i == nr && reference_mesh_target_enabled(bc_config.r_outer)) {
    out.r = x_r_reference[n];
    out.z = x_z_reference[n];
  }
  if (node_j == 0 && reference_mesh_target_enabled(bc_config.z_bottom)) {
    out.r = x_r_reference[n];
    out.z = x_z_reference[n];
  }
  if (node_j == nz && reference_mesh_target_enabled(bc_config.z_top)) {
    out.r = x_r_reference[n];
    out.z = x_z_reference[n];
  }
  return out;
}

__device__ inline RzNodeValue apply_corner_cell_aspect_protection(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const int node_i,
    const int node_j,
    const RzNodeValue& center,
    const RzNodeValue& candidate,
    const bool has_physical_rz_axis,
    const bool corner_cell_aspect_protection_enabled,
    const double corner_cell_aspect_eta,
    const bool z_bottom_state_supply,
    const bool z_top_state_supply);

static __global__ void winslow_jacobi_step_kernel(double* __restrict__ x_r_new,
                                                  double* __restrict__ x_z_new,
                                                  const double* __restrict__ x_r,
                                                  const double* __restrict__ x_z,
                                                  const std::uint8_t* __restrict__ node_flags,
                                                  const int nr,
                                                  const int nz,
                                                  const double max_displacement_fraction,
                                                  const double reference_lmin,
                                                  AleRezoneIterStats* __restrict__ stats,
                                                  const bool axis_z_motion_winslow,
                                                  const bool has_physical_rz_axis,
                                                  const double* __restrict__ x_r_initial,
                                                  const double* __restrict__ x_z_initial,
                                                  const double* __restrict__ x_r_reference,
                                                  const double* __restrict__ x_z_reference,
                                                  const tenryu::hydro::BC2DRZConfig bc_config,
                                                  const bool corner_cell_aspect_protection_enabled,
                                                  const double corner_cell_aspect_eta,
                                                  const bool z_bottom_state_supply,
                                                  const bool z_top_state_supply) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  const std::uint8_t flags = node_flags[n];
  if ((flags & (mesh::NODE_BOUNDARY | mesh::NODE_CENTER)) != 0u || i == nr ||
      j == 0 || j == nz) {
    RzNodeValue candidate{x_r[n], x_z[n]};
    candidate = apply_corner_cell_aspect_protection(
        x_r,
        x_z,
        x_r_initial,
        x_z_initial,
        nr,
        nz,
        i,
        j,
        RzNodeValue{x_r[n], x_z[n]},
        candidate,
        has_physical_rz_axis,
        corner_cell_aspect_protection_enabled,
        corner_cell_aspect_eta,
        z_bottom_state_supply,
        z_top_state_supply);
    candidate = apply_reference_mesh_boundary_target(candidate,
                                                     x_r_reference,
                                                     x_z_reference,
                                                     nr,
                                                     nz,
                                                     i,
                                                     j,
                                                     bc_config);
    x_r_new[n] = candidate.r;
    x_z_new[n] = candidate.z;
    return;
  }

  if (!has_physical_rz_axis && i == 0) {
    RzNodeValue candidate{x_r[n], x_z[n]};
    candidate = apply_corner_cell_aspect_protection(
        x_r,
        x_z,
        x_r_initial,
        x_z_initial,
        nr,
        nz,
        i,
        j,
        RzNodeValue{x_r[n], x_z[n]},
        candidate,
        has_physical_rz_axis,
        corner_cell_aspect_protection_enabled,
        corner_cell_aspect_eta,
        z_bottom_state_supply,
        z_top_state_supply);
    candidate = apply_reference_mesh_boundary_target(candidate,
                                                     x_r_reference,
                                                     x_z_reference,
                                                     nr,
                                                     nz,
                                                     i,
                                                     j,
                                                     bc_config);
    x_r_new[n] = candidate.r;
    x_z_new[n] = candidate.z;
    return;
  }

  const bool on_axis =
      (flags & mesh::NODE_AXIS) != 0u || (has_physical_rz_axis && i == 0);
  if (on_axis) {
    if (!axis_z_motion_winslow) {
      x_r_new[n] = x_r[n];
      x_z_new[n] = x_z[n];
      return;
    }

    const int n_jp = j + 1;
    const int n_jm = j - 1;
    const int n_i1 = stride + j;

    const double z_jp = x_z[n_jp];
    const double z_jm = x_z[n_jm];
    const double z_i1 = x_z[n_i1];
    const double r_i1 = x_r[n_i1];

    const double zeta = 0.5 * (z_jp - z_jm);
    const double alpha = zeta * zeta;
    const double gamma = r_i1 * r_i1;
    const double denom = 2.0 * (alpha + gamma) + 1.0e-30;
    const double z_candidate =
        (2.0 * alpha * z_i1 + gamma * (z_jp + z_jm)) / denom;

    const double dz_init_cap = max_displacement_fraction * reference_lmin;
    double dz = z_candidate - x_z[n];
    record_rezone_displacement_cap_diag(stats, fabs(dz), dz_init_cap);
    if (fabs(dz) > dz_init_cap) {
      dz = (dz > 0.0) ? dz_init_cap : -dz_init_cap;
    }

    x_r_new[n] = 0.0;
    x_z_new[n] = x_z[n] + dz;
    return;
  }

  const int ip = i + 1;
  const int im = i - 1;
  const int jp = j + 1;
  const int jm = j - 1;

  const int n_ip_j = ip * stride + j;
  const int n_im_j = im * stride + j;
  const int n_i_jp = i * stride + jp;
  const int n_i_jm = i * stride + jm;

  const double dr_dxi = 0.5 * (x_r[n_ip_j] - x_r[n_im_j]);
  const double dz_dxi = 0.5 * (x_z[n_ip_j] - x_z[n_im_j]);
  const double dr_deta = 0.5 * (x_r[n_i_jp] - x_r[n_i_jm]);
  const double dz_deta = 0.5 * (x_z[n_i_jp] - x_z[n_i_jm]);
  const double J = dr_dxi * dz_deta - dr_deta * dz_dxi;

  const double dr_local =
      fmin(fabs(x_r[n_ip_j] - x_r[n]), fabs(x_r[n] - x_r[n_im_j]));
  const double dz_local =
      fmin(fabs(x_z[n_i_jp] - x_z[n]), fabs(x_z[n] - x_z[n_i_jm]));
  // NUMERICS §3.3.3 specifies max_disp ~ f_disp * sqrt(A_c). Here we use a conservative
  // local-length proxy from neighbor distances to avoid extra area reductions in-kernel.
  const double lmin = fmax(fmin(dr_local, dz_local), 1.0e-30);
  const double cap_floor = (i == 1) ? (kAxisCapFloorFraction * reference_lmin) : 0.0;
  const double max_disp = max_displacement_fraction * fmax(lmin, cap_floor);

  if (J <= kJFloor) {
    if (i == 1) {
      // Singular stencil at axis-adjacent row: use uniform 4-neighbor Laplacian fallback
      // bounded by the same max_disp cap.
      const double r_avg =
          0.25 * (x_r[n_ip_j] + x_r[n_im_j] + x_r[n_i_jp] + x_r[n_i_jm]);
      const double z_avg =
          0.25 * (x_z[n_ip_j] + x_z[n_im_j] + x_z[n_i_jp] + x_z[n_i_jm]);
      double dr_fb = r_avg - x_r[n];
      double dz_fb = z_avg - x_z[n];
      const double mag_fb = sqrt(dr_fb * dr_fb + dz_fb * dz_fb);
      record_rezone_displacement_cap_diag(stats, mag_fb, max_disp);
      if (mag_fb > max_disp) {
        if (stats != nullptr) {
          atomicAdd(&stats->fallback_cap_hits, 1);
        }
        const double scale = max_disp / mag_fb;
        dr_fb *= scale;
        dz_fb *= scale;
      }
      if (stats != nullptr) {
        atomicAdd(&stats->j_floor_i1_fallback_hits, 1);
        if (dr_fb * dr_fb + dz_fb * dz_fb < kTinyMotion2) {
          atomicAdd(&stats->zero_or_tiny_motion_hits, 1);
        }
      }
      RzNodeValue candidate{x_r[n] + dr_fb, x_z[n] + dz_fb};
      candidate = apply_corner_cell_aspect_protection(
          x_r,
          x_z,
          x_r_initial,
          x_z_initial,
          nr,
          nz,
          i,
          j,
          RzNodeValue{x_r[n], x_z[n]},
          candidate,
          has_physical_rz_axis,
          corner_cell_aspect_protection_enabled,
          corner_cell_aspect_eta,
          z_bottom_state_supply,
          z_top_state_supply);
      x_r_new[n] = candidate.r;
      x_z_new[n] = candidate.z;
      return;
    }
    // Guard singular cells away from the axis-adjacent row: preserve existing skip semantics.
    if (stats != nullptr) {
      atomicAdd(&stats->j_floor_ix_skip_hits, 1);
      atomicAdd(&stats->zero_or_tiny_motion_hits, 1);
    }
    x_r_new[n] = x_r[n];
    x_z_new[n] = x_z[n];
    return;
  }

  const double alpha = dr_deta * dr_deta + dz_deta * dz_deta;
  const double beta = dr_dxi * dr_dxi + dz_dxi * dz_dxi;
  const double denom = 2.0 * (alpha + beta) + 1.0e-30;

  double r_candidate =
      (alpha * (x_r[n_ip_j] + x_r[n_im_j]) + beta * (x_r[n_i_jp] + x_r[n_i_jm])) /
      denom;
  double z_candidate =
      (alpha * (x_z[n_ip_j] + x_z[n_im_j]) + beta * (x_z[n_i_jp] + x_z[n_i_jm])) /
      denom;

  const double d_r = r_candidate - x_r[n];
  const double d_z = z_candidate - x_z[n];
  const double disp = sqrt(d_r * d_r + d_z * d_z);
  record_rezone_displacement_cap_diag(stats, disp, max_disp);
  if (disp > max_disp) {
    if (stats != nullptr) {
      if (i == 1) {
        atomicAdd(&stats->regular_cap_i1_hits, 1);
      } else {
        atomicAdd(&stats->regular_cap_ix_hits, 1);
      }
    }
    const double s = max_disp / disp;
    r_candidate = x_r[n] + s * d_r;
    z_candidate = x_z[n] + s * d_z;
  }
  RzNodeValue candidate{r_candidate, z_candidate};
  candidate = apply_corner_cell_aspect_protection(
      x_r,
      x_z,
      x_r_initial,
      x_z_initial,
      nr,
      nz,
      i,
      j,
      RzNodeValue{x_r[n], x_z[n]},
      candidate,
      has_physical_rz_axis,
      corner_cell_aspect_protection_enabled,
      corner_cell_aspect_eta,
      z_bottom_state_supply,
      z_top_state_supply);
  r_candidate = candidate.r;
  z_candidate = candidate.z;
  if (stats != nullptr) {
    const double dr_final = r_candidate - x_r[n];
    const double dz_final = z_candidate - x_z[n];
    if (dr_final * dr_final + dz_final * dz_final < kTinyMotion2) {
      atomicAdd(&stats->zero_or_tiny_motion_hits, 1);
    }
  }

  x_r_new[n] = r_candidate;
  x_z_new[n] = z_candidate;
}

__device__ inline RzNodeValue load_node_with_axis_mirror(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int i,
    const int j,
    const int nr,
    const int nz) {
  const int stride = nz + 1;
  if (i < 0) {
    const int n_mirror = stride + j;
    return RzNodeValue{-x_r[n_mirror], x_z[n_mirror]};
  }
  const int i_clamped = (i > nr) ? nr : i;
  const int n = i_clamped * stride + j;
  return RzNodeValue{x_r[n], x_z[n]};
}

__device__ inline double gauss_jacobian_from_quad(const double* rr,
                                                  const double* zz,
                                                  const double xi,
                                                  const double eta) {
  const double dN1_dxi = -0.25 * (1.0 - eta);
  const double dN2_dxi = 0.25 * (1.0 - eta);
  const double dN3_dxi = 0.25 * (1.0 + eta);
  const double dN4_dxi = -0.25 * (1.0 + eta);

  const double dN1_deta = -0.25 * (1.0 - xi);
  const double dN2_deta = -0.25 * (1.0 + xi);
  const double dN3_deta = 0.25 * (1.0 + xi);
  const double dN4_deta = 0.25 * (1.0 - xi);

  const double dr_dxi =
      dN1_dxi * rr[0] + dN2_dxi * rr[1] + dN3_dxi * rr[2] + dN4_dxi * rr[3];
  const double dz_dxi =
      dN1_dxi * zz[0] + dN2_dxi * zz[1] + dN3_dxi * zz[2] + dN4_dxi * zz[3];
  const double dr_deta =
      dN1_deta * rr[0] + dN2_deta * rr[1] + dN3_deta * rr[2] +
      dN4_deta * rr[3];
  const double dz_deta =
      dN1_deta * zz[0] + dN2_deta * zz[1] + dN3_deta * zz[2] +
      dN4_deta * zz[3];

  return dr_dxi * dz_deta - dr_deta * dz_dxi;
}

__device__ inline void load_cell_quad_with_trial(const double* __restrict__ x_r,
                                                 const double* __restrict__ x_z,
                                                 const int nr,
                                                 const int nz,
                                                 const int cell_i,
                                                 const int cell_j,
                                                 const int trial_i,
                                                 const int trial_j,
                                                 const double trial_r,
                                                 const double trial_z,
                                                 double* rr,
                                                 double* zz,
                                                 int* node_i) {
  const int stride = nz + 1;
  const int ii[4] = {cell_i, cell_i + 1, cell_i + 1, cell_i};
  const int jj[4] = {cell_j, cell_j, cell_j + 1, cell_j + 1};
  for (int k = 0; k < 4; ++k) {
    node_i[k] = ii[k];
    if (ii[k] == trial_i && jj[k] == trial_j) {
      rr[k] = trial_r;
      zz[k] = trial_z;
    } else {
      const int n = ii[k] * stride + jj[k];
      rr[k] = x_r[n];
      zz[k] = x_z[n];
    }
  }
}

__device__ inline double signed_quad_area(const double* rr, const double* zz) {
  return 0.5 * (rr[0] * zz[1] - zz[0] * rr[1] +
                rr[1] * zz[2] - zz[1] * rr[2] +
                rr[2] * zz[3] - zz[2] * rr[3] +
                rr[3] * zz[0] - zz[3] * rr[0]);
}

__device__ inline double cell_signed_area_with_trial(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const int cell_i,
    const int cell_j,
    const int trial_i,
    const int trial_j,
    const double trial_r,
    const double trial_z) {
  double rr[4];
  double zz[4];
  int node_i[4];
  load_cell_quad_with_trial(x_r,
                            x_z,
                            nr,
                            nz,
                            cell_i,
                            cell_j,
                            trial_i,
                            trial_j,
                            trial_r,
                            trial_z,
                            rr,
                            zz,
                            node_i);
  return signed_quad_area(rr, zz);
}

__device__ inline double cell_signed_area(const double* __restrict__ x_r,
                                          const double* __restrict__ x_z,
                                          const int nr,
                                          const int nz,
                                          const int cell_i,
                                          const int cell_j) {
  return cell_signed_area_with_trial(
      x_r, x_z, nr, nz, cell_i, cell_j, -1, -1, 0.0, 0.0);
}

__device__ inline int corner_cell_node_slot(const int cell_i,
                                            const int cell_j,
                                            const int node_i,
                                            const int node_j) {
  if (node_i == cell_i && node_j == cell_j) {
    return 0;
  }
  if (node_i == cell_i + 1 && node_j == cell_j) {
    return 1;
  }
  if (node_i == cell_i + 1 && node_j == cell_j + 1) {
    return 2;
  }
  if (node_i == cell_i && node_j == cell_j + 1) {
    return 3;
  }
  return -1;
}

__device__ inline RzNodeValue corner_cell_aspect_constrain_for_cell(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const int cell_i,
    const int cell_j,
    const int node_i,
    const int node_j,
    const RzNodeValue& center,
    const RzNodeValue& candidate,
    const double eta) {
  const int slot = corner_cell_node_slot(cell_i, cell_j, node_i, node_j);
  if (slot < 0) {
    return candidate;
  }

  const double initial_signed =
      cell_signed_area(x_r_initial, x_z_initial, nr, nz, cell_i, cell_j);
  if (!isfinite(initial_signed) || !(fabs(initial_signed) > 0.0)) {
    return candidate;
  }
  const double orientation = (initial_signed >= 0.0) ? 1.0 : -1.0;
  const double initial_area = fabs(initial_signed);
  const double area_floor = eta * initial_area;
  const double current_area =
      orientation * cell_signed_area(x_r, x_z, nr, nz, cell_i, cell_j);
  const double candidate_area = cell_signed_area_with_trial(x_r,
                                                           x_z,
                                                           nr,
                                                           nz,
                                                           cell_i,
                                                           cell_j,
                                                           node_i,
                                                           node_j,
                                                           candidate.r,
                                                           candidate.z);
  const double oriented_candidate_area = orientation * candidate_area;
  if (!isfinite(current_area) || !isfinite(oriented_candidate_area) ||
      !(oriented_candidate_area < area_floor)) {
    return candidate;
  }

  RzNodeValue trial = candidate;
  if (current_area > area_floor) {
    const double denom = current_area - oriented_candidate_area;
    if (denom > 0.0 && isfinite(denom)) {
      const double lambda =
          fmin(1.0, fmax(0.0, (current_area - area_floor) / denom));
      trial.r = center.r + lambda * (candidate.r - center.r);
      trial.z = center.z + lambda * (candidate.z - center.z);
    }
  } else if (oriented_candidate_area > current_area) {
    const double denom = oriented_candidate_area - current_area;
    if (denom > 0.0 && isfinite(denom)) {
      const double lambda = (area_floor - current_area) / denom;
      if (lambda >= 0.0 && lambda <= 1.0 && isfinite(lambda)) {
        trial.r = center.r + lambda * (candidate.r - center.r);
        trial.z = center.z + lambda * (candidate.z - center.z);
      }
    }
  }

  double rr[4];
  double zz[4];
  int node_slots[4];
  load_cell_quad_with_trial(x_r,
                            x_z,
                            nr,
                            nz,
                            cell_i,
                            cell_j,
                            node_i,
                            node_j,
                            trial.r,
                            trial.z,
                            rr,
                            zz,
                            node_slots);
  const double trial_area = orientation * signed_quad_area(rr, zz);
  if (!isfinite(trial_area) || !(trial_area < area_floor)) {
    return trial;
  }

  const bool bottom_corner_cell = (cell_j == 0);
  const bool top_corner_cell = (cell_j == nz - 1);
  const bool movable_z_edge =
      (bottom_corner_cell && node_j == cell_j + 1) ||
      (top_corner_cell && node_j == cell_j);
  if (!movable_z_edge) {
    return trial;
  }

  const int prev = (slot + 3) & 3;
  const int next = (slot + 1) & 3;
  const double grad_z =
      orientation * 0.5 * (rr[prev] - rr[next]);
  if (!isfinite(grad_z) || !(fabs(grad_z) > 0.0)) {
    return trial;
  }

  const double delta_z = (area_floor - trial_area) / grad_z;
  if (!isfinite(delta_z) ||
      (bottom_corner_cell && delta_z < 0.0) ||
      (top_corner_cell && delta_z > 0.0)) {
    return trial;
  }
  const double projected_z = trial.z + delta_z;
  if (!isfinite(projected_z)) {
    return trial;
  }
  const int away_j = bottom_corner_cell ? (node_j + 1) : (node_j - 1);
  if (away_j >= 0 && away_j <= nz) {
    const int stride = nz + 1;
    const double neighbor_z = x_z[node_i * stride + away_j];
    if (isfinite(neighbor_z) &&
        ((bottom_corner_cell && projected_z >= neighbor_z) ||
         (top_corner_cell && projected_z <= neighbor_z))) {
      return trial;
    }
  }
  return RzNodeValue{trial.r, projected_z};
}

__device__ inline RzNodeValue apply_corner_cell_aspect_protection(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const int node_i,
    const int node_j,
    const RzNodeValue& center,
    const RzNodeValue& candidate,
    const bool has_physical_rz_axis,
    const bool corner_cell_aspect_protection_enabled,
    const double corner_cell_aspect_eta,
    const bool z_bottom_state_supply,
    const bool z_top_state_supply) {
  if (!corner_cell_aspect_protection_enabled || has_physical_rz_axis ||
      x_r_initial == nullptr || x_z_initial == nullptr ||
      !(corner_cell_aspect_eta > 0.0) || nr < 2 || nz < 2) {
    return candidate;
  }

  const bool near_inner = (node_i == 0 || node_i == 1);
  const bool near_outer = (node_i == nr - 1 || node_i == nr);
  const bool near_bottom = z_bottom_state_supply && (node_j == 1);
  const bool near_top = z_top_state_supply && (node_j == nz - 1);
  if (!(near_inner || near_outer) || !(near_bottom || near_top)) {
    return candidate;
  }

  const double eta = fmin(1.0, fmax(0.0, corner_cell_aspect_eta));
  RzNodeValue protected_candidate = candidate;
  for (int radial_side = 0; radial_side < 2; ++radial_side) {
    const bool use_radial_side =
        (radial_side == 0) ? near_inner : near_outer;
    if (!use_radial_side) {
      continue;
    }
    const int cell_i = (radial_side == 0) ? 0 : (nr - 1);
    for (int z_side = 0; z_side < 2; ++z_side) {
      const bool use_z_side = (z_side == 0) ? near_bottom : near_top;
      if (!use_z_side) {
        continue;
      }
      const int cell_j = (z_side == 0) ? 0 : (nz - 1);
      protected_candidate = corner_cell_aspect_constrain_for_cell(
          x_r,
          x_z,
          x_r_initial,
          x_z_initial,
          nr,
          nz,
          cell_i,
          cell_j,
          node_i,
          node_j,
          center,
          protected_candidate,
          eta);
    }
  }

  return protected_candidate;
}

static __global__ void corner_cell_aspect_floor_violated_kernel(
    int* __restrict__ violated,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const bool has_physical_rz_axis,
    const bool corner_cell_aspect_protection_enabled,
    const double corner_cell_aspect_eta,
    const bool z_bottom_state_supply,
    const bool z_top_state_supply) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  *violated = 0;
  if (!corner_cell_aspect_protection_enabled || has_physical_rz_axis ||
      x_r_initial == nullptr || x_z_initial == nullptr ||
      !(corner_cell_aspect_eta > 0.0) || nr < 2 || nz < 2 ||
      (!z_bottom_state_supply && !z_top_state_supply)) {
    return;
  }

  const double eta = fmin(1.0, fmax(0.0, corner_cell_aspect_eta));
  for (int radial_side = 0; radial_side < 2; ++radial_side) {
    const int cell_i = (radial_side == 0) ? 0 : (nr - 1);
    for (int z_side = 0; z_side < 2; ++z_side) {
      const bool use_z_side =
          (z_side == 0) ? z_bottom_state_supply : z_top_state_supply;
      if (!use_z_side) {
        continue;
      }
      const int cell_j = (z_side == 0) ? 0 : (nz - 1);
      const double initial_signed =
          cell_signed_area(x_r_initial, x_z_initial, nr, nz, cell_i, cell_j);
      if (!isfinite(initial_signed) || !(fabs(initial_signed) > 0.0)) {
        continue;
      }
      const double orientation = (initial_signed >= 0.0) ? 1.0 : -1.0;
      const double area_floor = eta * fabs(initial_signed);
      const double current_area =
          orientation * cell_signed_area(x_r, x_z, nr, nz, cell_i, cell_j);
      if (!isfinite(current_area) || current_area < area_floor) {
        *violated = 1;
        return;
      }
    }
  }
}

__device__ inline void enforce_corner_cell_aspect_floor_for_cell(
    double* __restrict__ x_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const int cell_i,
    const int cell_j,
    const double eta) {
  const double initial_signed =
      cell_signed_area(x_r_initial, x_z_initial, nr, nz, cell_i, cell_j);
  if (!isfinite(initial_signed) || !(fabs(initial_signed) > 0.0)) {
    return;
  }
  const double orientation = (initial_signed >= 0.0) ? 1.0 : -1.0;
  const double area_floor = eta * fabs(initial_signed);
  const double current_area =
      orientation * cell_signed_area(x_r, x_z, nr, nz, cell_i, cell_j);
  if (!isfinite(current_area) || !(current_area < area_floor)) {
    return;
  }

  const bool bottom_corner_cell = (cell_j == 0);
  const bool top_corner_cell = (cell_j == nz - 1);
  if (!bottom_corner_cell && !top_corner_cell) {
    return;
  }

  double rr[4];
  double zz[4];
  int node_slots[4];
  load_cell_quad_with_trial(x_r,
                            x_z,
                            nr,
                            nz,
                            cell_i,
                            cell_j,
                            -1,
                            -1,
                            0.0,
                            0.0,
                            rr,
                            zz,
                            node_slots);
  const int edge_slot_a = bottom_corner_cell ? 2 : 0;
  const int edge_slot_b = bottom_corner_cell ? 3 : 1;
  const int prev_a = (edge_slot_a + 3) & 3;
  const int next_a = (edge_slot_a + 1) & 3;
  const int prev_b = (edge_slot_b + 3) & 3;
  const int next_b = (edge_slot_b + 1) & 3;
  const double grad_z_a = orientation * 0.5 * (rr[prev_a] - rr[next_a]);
  const double grad_z_b = orientation * 0.5 * (rr[prev_b] - rr[next_b]);
  const double grad_z = grad_z_a + grad_z_b;
  if (!isfinite(grad_z) || !(fabs(grad_z) > 0.0)) {
    return;
  }

  const double delta_z = (area_floor - current_area) / grad_z;
  if (!isfinite(delta_z) ||
      (bottom_corner_cell && delta_z < 0.0) ||
      (top_corner_cell && delta_z > 0.0)) {
    return;
  }

  const int stride = nz + 1;
  const int edge_j = bottom_corner_cell ? (cell_j + 1) : cell_j;
  const int away_j = bottom_corner_cell ? (edge_j + 1) : (edge_j - 1);
  const int n_a = cell_i * stride + edge_j;
  const int n_b = (cell_i + 1) * stride + edge_j;
  double projected_a = x_z[n_a] + delta_z;
  double projected_b = x_z[n_b] + delta_z;
  if (away_j >= 0 && away_j <= nz) {
    const double neighbor_a = x_z[cell_i * stride + away_j];
    const double neighbor_b = x_z[(cell_i + 1) * stride + away_j];
    if (bottom_corner_cell) {
      if (isfinite(neighbor_a) && projected_a >= neighbor_a) {
        projected_a = x_z[n_a];
      }
      if (isfinite(neighbor_b) && projected_b >= neighbor_b) {
        projected_b = x_z[n_b];
      }
    } else {
      if (isfinite(neighbor_a) && projected_a <= neighbor_a) {
        projected_a = x_z[n_a];
      }
      if (isfinite(neighbor_b) && projected_b <= neighbor_b) {
        projected_b = x_z[n_b];
      }
    }
  }
  zz[edge_slot_a] = projected_a;
  zz[edge_slot_b] = projected_b;
  double projected_area = orientation * signed_quad_area(rr, zz);
  for (int edge_node = 0; edge_node < 2 && projected_area < area_floor; ++edge_node) {
    const int slot_k = (edge_node == 0) ? edge_slot_a : edge_slot_b;
    double* projected_k = (edge_node == 0) ? &projected_a : &projected_b;
    const int node_i_k = (edge_node == 0) ? cell_i : (cell_i + 1);
    const int prev_k = (slot_k + 3) & 3;
    const int next_k = (slot_k + 1) & 3;
    const double grad_z_k = orientation * 0.5 * (rr[prev_k] - rr[next_k]);
    if (!isfinite(grad_z_k) || !(fabs(grad_z_k) > 0.0)) {
      continue;
    }
    const double dz_k = (area_floor - projected_area) / grad_z_k;
    if (!isfinite(dz_k) ||
        (bottom_corner_cell && dz_k < 0.0) ||
        (top_corner_cell && dz_k > 0.0)) {
      continue;
    }
    const double candidate_z = *projected_k + dz_k;
    if (!isfinite(candidate_z)) {
      continue;
    }
    if (away_j >= 0 && away_j <= nz) {
      const double neighbor = x_z[node_i_k * stride + away_j];
      if (isfinite(neighbor) &&
          ((bottom_corner_cell && candidate_z >= neighbor) ||
           (top_corner_cell && candidate_z <= neighbor))) {
        continue;
      }
    }
    *projected_k = candidate_z;
    zz[slot_k] = candidate_z;
    projected_area = orientation * signed_quad_area(rr, zz);
  }
  if (isfinite(projected_a)) {
    x_z[n_a] = projected_a;
  }
  if (isfinite(projected_b)) {
    x_z[n_b] = projected_b;
  }
}

static __global__ void enforce_corner_cell_aspect_floor_kernel(
    double* __restrict__ x_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const int nr,
    const int nz,
    const bool has_physical_rz_axis,
    const bool corner_cell_aspect_protection_enabled,
    const double corner_cell_aspect_eta,
    const bool z_bottom_state_supply,
    const bool z_top_state_supply) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (!corner_cell_aspect_protection_enabled || has_physical_rz_axis ||
      x_r_initial == nullptr || x_z_initial == nullptr ||
      !(corner_cell_aspect_eta > 0.0) || nr < 2 || nz < 2 ||
      (!z_bottom_state_supply && !z_top_state_supply)) {
    return;
  }

  const double eta = fmin(1.0, fmax(0.0, corner_cell_aspect_eta));
  for (int radial_side = 0; radial_side < 2; ++radial_side) {
    const int cell_i = (radial_side == 0) ? 0 : (nr - 1);
    if (z_bottom_state_supply) {
      enforce_corner_cell_aspect_floor_for_cell(
          x_z, x_r, x_r_initial, x_z_initial, nr, nz, cell_i, 0, eta);
    }
    if (z_top_state_supply) {
      enforce_corner_cell_aspect_floor_for_cell(
          x_z, x_r, x_r_initial, x_z_initial, nr, nz, cell_i, nz - 1, eta);
    }
  }
}

inline void enforce_corner_cell_aspect_floor(
    double* x_r,
    double* x_z,
    const core::State& state,
    const core::Config& cfg) {
  if (!cfg.numerics.ale.corner_cell_aspect_protection_enabled ||
      cfg.numerics.has_physical_rz_axis ||
      !(cfg.numerics.ale.corner_cell_aspect_eta > 0.0) ||
      state.mesh.topo.nr < 2 || state.mesh.topo.nz < 2 ||
      state.x_r_initial.size() != state.x_r.size() ||
      state.x_z_initial.size() != state.x_z.size() ||
      (!cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply() &&
       !cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply())) {
    return;
  }
  enforce_corner_cell_aspect_floor_kernel<<<1, 1>>>(
      x_z,
      x_r,
      state.x_r_initial.data(),
      state.x_z_initial.data(),
      state.mesh.topo.nr,
      state.mesh.topo.nz,
      cfg.numerics.has_physical_rz_axis,
      cfg.numerics.ale.corner_cell_aspect_protection_enabled,
      cfg.numerics.ale.corner_cell_aspect_eta,
      cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
      cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

inline bool corner_cell_aspect_floor_violated(
    const core::State& state,
    const core::Config& cfg,
    const parallel::Reduction* reduction = nullptr) {
  if (!cfg.numerics.ale.corner_cell_aspect_protection_enabled ||
      cfg.numerics.has_physical_rz_axis ||
      !(cfg.numerics.ale.corner_cell_aspect_eta > 0.0) ||
      state.mesh.topo.nr < 2 || state.mesh.topo.nz < 2 ||
      state.x_r_initial.size() != state.x_r.size() ||
      state.x_z_initial.size() != state.x_z.size() ||
      (!cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply() &&
       !cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply())) {
    return false;
  }

  int* d_violated = nullptr;
  int h_violated = 0;
  d_violated = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:corner_aspect:d_violated", sizeof(int)));
  corner_cell_aspect_floor_violated_kernel<<<1, 1>>>(
      d_violated,
      state.x_r.data(),
      state.x_z.data(),
      state.x_r_initial.data(),
      state.x_z_initial.data(),
      state.mesh.topo.nr,
      state.mesh.topo.nz,
      cfg.numerics.has_physical_rz_axis,
      cfg.numerics.ale.corner_cell_aspect_protection_enabled,
      cfg.numerics.ale.corner_cell_aspect_eta,
      cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
      cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(&h_violated, d_violated, sizeof(int), cudaMemcpyDeviceToHost));
  if (reduction != nullptr) {
    h_violated =
        (reduction->allreduce_max(h_violated != 0 ? 1.0 : 0.0) > 0.5) ? 1 : 0;
  }
  return h_violated != 0;
}

__device__ inline bool trial_cell_admissible(const double* rr,
                                             const double* zz,
                                             const int* node_i,
                                             const double j_floor_rel) {
  const double xis[2] = {-kGauss, kGauss};
  const double etas[2] = {-kGauss, kGauss};
  double J_values[4];
  double J_max = -1.0e300;
  int p = 0;
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      const double J = gauss_jacobian_from_quad(rr, zz, xis[a], etas[b]);
      if (!isfinite(J)) {
        return false;
      }
      J_values[p++] = J;
      J_max = fmax(J_max, J);
    }
  }
  const double J_max_eff = fmax(J_max, kJFloor);
  const double J_floor = fmax(j_floor_rel * J_max_eff, kJFloor);
  for (int k = 0; k < 4; ++k) {
    if (!(J_values[k] > J_floor)) {
      return false;
    }
  }

  for (int corner = 0; corner < 4; ++corner) {
    const double J_corner =
        tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
    if (!isfinite(J_corner) || !(J_corner > J_floor)) {
      return false;
    }
  }

  const double v_rz = detail::rz_signed_quad_volume(
      rr[0], zz[0], rr[1], zz[1], rr[2], zz[2], rr[3], zz[3]);
  if (!isfinite(v_rz) || !(v_rz > kJFloor)) {
    return false;
  }

  for (int k = 0; k < 4; ++k) {
    if (!isfinite(rr[k]) || !isfinite(zz[k])) {
      return false;
    }
    if (node_i[k] == 0) {
      if (!(rr[k] == 0.0)) {
        return false;
      }
    } else if (!(rr[k] >= 0.0)) {
      return false;
    }
  }
  return true;
}

__device__ inline bool incident_cells_admissible_with_trial(
    const int i_node,
    const int j_node,
    const double trial_r,
    const double trial_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int nr,
    const int nz,
    const double j_floor_rel) {
  bool checked = false;
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ci = i_node + di;
      const int cj = j_node + dj;
      if (ci < 0 || ci >= nr || cj < 0 || cj >= nz) {
        continue;
      }
      checked = true;
      double rr[4];
      double zz[4];
      int node_i[4];
      load_cell_quad_with_trial(x_r,
                                x_z,
                                nr,
                                nz,
                                ci,
                                cj,
                                i_node,
                                j_node,
                                trial_r,
                                trial_z,
                                rr,
                                zz,
                                node_i);
      if (!trial_cell_admissible(rr, zz, node_i, j_floor_rel)) {
        return false;
      }
    }
  }
  return checked;
}

__device__ inline RzNodeValue weighted_laplacian_candidate(
    const RzNodeValue& center,
    const RzNodeValue& east,
    const RzNodeValue& west,
    const RzNodeValue& north,
    const RzNodeValue& south,
    const double alpha,
    const double gamma,
    const double eps_r) {
  const double wE = fmax(0.5 * (center.r + east.r), eps_r);
  const double wW = fmax(0.5 * (center.r + west.r), eps_r);
  const double wN = center.r;
  const double wS = center.r;
  const double denom_metric = alpha * (wE + wW) + gamma * (wN + wS);
  if (isfinite(denom_metric) && denom_metric > eps_r) {
    return RzNodeValue{
        (alpha * (wE * east.r + wW * west.r) +
         gamma * (wN * north.r + wS * south.r)) /
            denom_metric,
        (alpha * (wE * east.z + wW * west.z) +
         gamma * (wN * north.z + wS * south.z)) /
            denom_metric};
  }

  const double denom = wE + wW + wN + wS;
  if (isfinite(denom) && denom > eps_r) {
    return RzNodeValue{(wE * east.r + wW * west.r + wN * north.r +
                        wS * south.r) /
                           denom,
                       (wE * east.z + wW * west.z + wN * north.z +
                        wS * south.z) /
                           denom};
  }

  return RzNodeValue{0.25 * (east.r + west.r + north.r + south.r),
                     0.25 * (east.z + west.z + north.z + south.z)};
}

static __global__ void winslow_jacobi_step_rz_full_metric_kernel(
    double* __restrict__ x_r_new,
    double* __restrict__ x_z_new,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const int nr,
    const int nz,
    const double max_displacement_fraction,
    const double reference_lmin,
    AleRezoneIterStats* __restrict__ stats,
    const bool axis_z_motion_winslow,
    const bool has_physical_rz_axis,
    const bool local_admissibility_linesearch,
    const double j_floor_rel,
    const int linesearch_max_halves,
    const double* __restrict__ x_r_initial,
    const double* __restrict__ x_z_initial,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const tenryu::hydro::BC2DRZConfig bc_config,
    const bool corner_cell_aspect_protection_enabled,
    const double corner_cell_aspect_eta,
    const bool z_bottom_state_supply,
    const bool z_top_state_supply) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;
  const std::uint8_t flags = node_flags[n];

  if ((flags & (mesh::NODE_BOUNDARY | mesh::NODE_CENTER)) != 0u || i == nr ||
      j == 0 || j == nz) {
    RzNodeValue candidate{(has_physical_rz_axis && i == 0) ? 0.0 : x_r[n],
                          x_z[n]};
    candidate = apply_corner_cell_aspect_protection(
        x_r,
        x_z,
        x_r_initial,
        x_z_initial,
        nr,
        nz,
        i,
        j,
        RzNodeValue{candidate.r, x_z[n]},
        candidate,
        has_physical_rz_axis,
        corner_cell_aspect_protection_enabled,
        corner_cell_aspect_eta,
        z_bottom_state_supply,
        z_top_state_supply);
    candidate = apply_reference_mesh_boundary_target(candidate,
                                                     x_r_reference,
                                                     x_z_reference,
                                                     nr,
                                                     nz,
                                                     i,
                                                     j,
                                                     bc_config);
    x_r_new[n] = candidate.r;
    x_z_new[n] = candidate.z;
    return;
  }

  if (!has_physical_rz_axis && i == 0) {
    RzNodeValue candidate{x_r[n], x_z[n]};
    candidate = apply_corner_cell_aspect_protection(
        x_r,
        x_z,
        x_r_initial,
        x_z_initial,
        nr,
        nz,
        i,
        j,
        RzNodeValue{x_r[n], x_z[n]},
        candidate,
        has_physical_rz_axis,
        corner_cell_aspect_protection_enabled,
        corner_cell_aspect_eta,
        z_bottom_state_supply,
        z_top_state_supply);
    candidate = apply_reference_mesh_boundary_target(candidate,
                                                     x_r_reference,
                                                     x_z_reference,
                                                     nr,
                                                     nz,
                                                     i,
                                                     j,
                                                     bc_config);
    x_r_new[n] = candidate.r;
    x_z_new[n] = candidate.z;
    return;
  }

  const bool on_axis =
      (flags & mesh::NODE_AXIS) != 0u || (has_physical_rz_axis && i == 0);
  if (on_axis) {
    if (!axis_z_motion_winslow) {
      x_r_new[n] = 0.0;
      x_z_new[n] = x_z[n];
      return;
    }
  }

  const RzNodeValue center{x_r[n], x_z[n]};
  const RzNodeValue east = load_node_with_axis_mirror(x_r, x_z, i + 1, j, nr, nz);
  const RzNodeValue west = load_node_with_axis_mirror(x_r, x_z, i - 1, j, nr, nz);
  const RzNodeValue north = load_node_with_axis_mirror(x_r, x_z, i, j + 1, nr, nz);
  const RzNodeValue south = load_node_with_axis_mirror(x_r, x_z, i, j - 1, nr, nz);
  const RzNodeValue ne = load_node_with_axis_mirror(x_r, x_z, i + 1, j + 1, nr, nz);
  const RzNodeValue nw = load_node_with_axis_mirror(x_r, x_z, i - 1, j + 1, nr, nz);
  const RzNodeValue se = load_node_with_axis_mirror(x_r, x_z, i + 1, j - 1, nr, nz);
  const RzNodeValue sw = load_node_with_axis_mirror(x_r, x_z, i - 1, j - 1, nr, nz);

  const double dr_dxi = 0.5 * (east.r - west.r);
  const double dz_dxi = 0.5 * (east.z - west.z);
  const double dr_deta = 0.5 * (north.r - south.r);
  const double dz_deta = 0.5 * (north.z - south.z);
  const double J = dr_dxi * dz_deta - dr_deta * dz_dxi;

  const double alpha = dr_deta * dr_deta + dz_deta * dz_deta;
  const double gamma = dr_dxi * dr_dxi + dz_dxi * dz_dxi;
  const double beta = dr_dxi * dr_deta + dz_dxi * dz_deta;
  const double x_xieta_r = 0.25 * (ne.r - nw.r - se.r + sw.r);
  const double x_xieta_z = 0.25 * (ne.z - nw.z - se.z + sw.z);
  const double denom_cart = 2.0 * (alpha + gamma) + 1.0e-30;
  const double eps_r = fmax(1.0e-30, 1.0e-12 * reference_lmin);

  // Full metric Winslow follows the elliptic metric tensor form used in
  // Knupp (SIAM, 2001) and the TMOP lineage of Dobrev-Kolev-Mittal (2019).
  const RzNodeValue candidate_cartesian{
      (alpha * (east.r + west.r) + gamma * (north.r + south.r) -
       2.0 * beta * x_xieta_r) /
          denom_cart,
      (alpha * (east.z + west.z) + gamma * (north.z + south.z) -
       2.0 * beta * x_xieta_z) /
          denom_cart};

  const double wE = fmax(0.5 * (center.r + east.r), eps_r);
  const double wW = fmax(0.5 * (center.r + west.r), eps_r);
  const double wN = center.r;
  const double wS = center.r;
  const double denom_weighted = alpha * (wE + wW) + gamma * (wN + wS);
  RzNodeValue candidate_weighted =
      weighted_laplacian_candidate(center, east, west, north, south, alpha, gamma, eps_r);
  if (isfinite(denom_weighted) && denom_weighted > eps_r) {
    candidate_weighted.r =
        (alpha * (wE * east.r + wW * west.r) +
         gamma * (wN * north.r + wS * south.r) -
         2.0 * center.r * beta * x_xieta_r) /
        denom_weighted;
    candidate_weighted.z =
        (alpha * (wE * east.z + wW * west.z) +
         gamma * (wN * north.z + wS * south.z) -
         2.0 * center.r * beta * x_xieta_z) /
        denom_weighted;
  }

  RzNodeValue candidate;
  if (!isfinite(J) || J <= kJFloor) {
    candidate =
        weighted_laplacian_candidate(center, east, west, north, south, alpha, gamma, eps_r);
    if (stats != nullptr) {
      atomicAdd(&stats->weighted_laplacian_fallback_hits, 1);
    }
  } else {
    candidate.r = 0.5 * (candidate_cartesian.r + candidate_weighted.r);
    candidate.z = 0.5 * (candidate_cartesian.z + candidate_weighted.z);
  }

  if (on_axis) {
    candidate.r = 0.0;
  }

  const double dE_r = east.r - center.r;
  const double dE_z = east.z - center.z;
  const double dW_r = west.r - center.r;
  const double dW_z = west.z - center.z;
  const double dN_r = north.r - center.r;
  const double dN_z = north.z - center.z;
  const double dS_r = south.r - center.r;
  const double dS_z = south.z - center.z;
  const double dE = sqrt(dE_r * dE_r + dE_z * dE_z);
  const double dW = sqrt(dW_r * dW_r + dW_z * dW_z);
  const double dN = sqrt(dN_r * dN_r + dN_z * dN_z);
  const double dS = sqrt(dS_r * dS_r + dS_z * dS_z);
  const double lmin = fmax(fmin(fmin(dE, dW), fmin(dN, dS)), 1.0e-30);
  const double cap_floor =
      (i <= 1) ? (kAxisCapFloorFraction * reference_lmin) : 0.0;
  const double max_disp = max_displacement_fraction * fmax(lmin, cap_floor);

  double d_r = candidate.r - center.r;
  double d_z = candidate.z - center.z;
  double disp = sqrt(d_r * d_r + d_z * d_z);
  if (!isfinite(candidate.r) || !isfinite(candidate.z)) {
    candidate = weighted_laplacian_candidate(center, east, west, north, south, alpha, gamma, eps_r);
    if (on_axis) {
      candidate.r = 0.0;
    }
    d_r = candidate.r - center.r;
    d_z = candidate.z - center.z;
    disp = sqrt(d_r * d_r + d_z * d_z);
  }
  record_rezone_displacement_cap_diag(stats, disp, max_disp);
  if (isfinite(disp) && disp > max_disp) {
    if (stats != nullptr) {
      if (J <= kJFloor || !isfinite(J)) {
        atomicAdd(&stats->fallback_cap_hits, 1);
      } else if (i == 1) {
        atomicAdd(&stats->regular_cap_i1_hits, 1);
      } else {
        atomicAdd(&stats->regular_cap_ix_hits, 1);
      }
    }
    const double scale = max_disp / disp;
    candidate.r = center.r + scale * d_r;
    candidate.z = center.z + scale * d_z;
    if (on_axis) {
      candidate.r = 0.0;
    }
  }

  candidate = apply_corner_cell_aspect_protection(
      x_r,
      x_z,
      x_r_initial,
      x_z_initial,
      nr,
      nz,
      i,
      j,
      center,
      candidate,
      has_physical_rz_axis,
      corner_cell_aspect_protection_enabled,
      corner_cell_aspect_eta,
      z_bottom_state_supply,
      z_top_state_supply);
  if (on_axis) {
    candidate.r = 0.0;
  }

  RzNodeValue accepted = candidate;
  if (local_admissibility_linesearch) {
    bool found = false;
    const int max_halves = (linesearch_max_halves < 0)
                               ? 0
                               : ((linesearch_max_halves > 32)
                                      ? 32
                                      : linesearch_max_halves);
    for (int h = 0; h <= max_halves; ++h) {
      const double lambda = ldexp(1.0, -h);
      RzNodeValue trial{center.r + lambda * (candidate.r - center.r),
                        center.z + lambda * (candidate.z - center.z)};
      if (on_axis) {
        trial.r = 0.0;
      }
      if (incident_cells_admissible_with_trial(i,
                                               j,
                                               trial.r,
                                               trial.z,
                                               x_r,
                                               x_z,
                                               nr,
                                               nz,
                                               j_floor_rel)) {
        accepted = trial;
        found = true;
        break;
      }
      if (stats != nullptr) {
        atomicAdd(&stats->local_linesearch_rejects, 1);
      }
    }
    if (!found) {
      accepted.r = on_axis ? 0.0 : center.r;
      accepted.z = center.z;
    }
  }

  if (stats != nullptr) {
    const double dr_final = accepted.r - center.r;
    const double dz_final = accepted.z - center.z;
    if (dr_final * dr_final + dz_final * dz_final < kTinyMotion2) {
      atomicAdd(&stats->zero_or_tiny_motion_hits, 1);
    }
  }

  x_r_new[n] = accepted.r;
  x_z_new[n] = accepted.z;
}

static __global__ void max_node_delta_kernel(const double* __restrict__ x_r_new,
                                             const double* __restrict__ x_z_new,
                                             const double* __restrict__ x_r_old,
                                             const double* __restrict__ x_z_old,
                                             double* __restrict__ max_delta,
                                             const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }

  const double dr = x_r_new[n] - x_r_old[n];
  const double dz = x_z_new[n] - x_z_old[n];
  const double delta = sqrt(dr * dr + dz * dz);
  atomic_max_double(max_delta, delta);
}

__device__ inline bool multiblock_winslow_is_seam_tag(const std::int8_t tag) {
  return tag ==
             static_cast<std::int8_t>(
                 mesh::BoundaryKind::SEAM_CORE_BRIDGE) ||
         tag ==
             static_cast<std::int8_t>(
                 mesh::BoundaryKind::SEAM_BRIDGE_SHELL);
}

static __global__ void multiblock_winslow_node_delta_kernel(
    double* __restrict__ delta_r,
    double* __restrict__ delta_z,
    const double* __restrict__ x_r_new,
    const double* __restrict__ x_z_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  delta_r[n] = x_r_new[n] - x_r_old[n];
  delta_z[n] = x_z_new[n] - x_z_old[n];
}

static __global__ void multiblock_winslow_reference_add_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  x_r[n] = x_r_reference[n] + delta_r[n];
  x_z[n] = x_z_reference[n] + delta_z[n];
}

inline void ensure_multiblock_winslow_device_csr(core::State& state) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR Winslow smoother requires multiblock topology");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const bool use_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(state.mesh.cell_nverts.begin(),
                  state.mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts == 3U; });
  std::size_t reverse_payload_size = mb.cell_node_csr_indices.size();
  if (use_cell_nverts) {
    reverse_payload_size = 0U;
    for (int c = 0; c < n_cells; ++c) {
      reverse_payload_size += static_cast<std::size_t>(
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c));
    }
  }

  if (state.mesh.multiblock_cell_node_csr_offsets.size() !=
      mb.cell_node_csr_offsets.size()) {
    state.mesh.multiblock_cell_node_csr_offsets.reset(
        mb.cell_node_csr_offsets.size());
    state.mesh.multiblock_cell_node_csr_offsets.copy_from_host(
        mb.cell_node_csr_offsets);
  }
  if (state.mesh.multiblock_cell_node_csr_indices.size() !=
      mb.cell_node_csr_indices.size()) {
    state.mesh.multiblock_cell_node_csr_indices.reset(
        mb.cell_node_csr_indices.size());
    state.mesh.multiblock_cell_node_csr_indices.copy_from_host(
        mb.cell_node_csr_indices);
  }

  const bool reverse_offsets_missing =
      state.mesh.multiblock_reverse_csr_node_offsets.size() !=
      static_cast<std::size_t>(n_nodes + 1);
  const bool reverse_payload_missing =
      state.mesh.multiblock_reverse_csr_node_cells.size() !=
          reverse_payload_size ||
      state.mesh.multiblock_reverse_csr_node_corners.size() !=
          reverse_payload_size;
  if (reverse_offsets_missing || reverse_payload_missing) {
    const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
        tenryu::hydro::build_reverse_cell_node_csr(
            mb, n_nodes, use_cell_nverts ? &state.mesh.cell_nverts : nullptr,
            state.mesh.corner_stride);
    state.mesh.multiblock_reverse_csr_node_offsets.reset(
        reverse_csr.node_offsets.size());
    state.mesh.multiblock_reverse_csr_node_offsets.copy_from_host(
        reverse_csr.node_offsets);
    state.mesh.multiblock_reverse_csr_node_cells.reset(
        reverse_csr.node_cells.size());
    state.mesh.multiblock_reverse_csr_node_cells.copy_from_host(
        reverse_csr.node_cells);
    state.mesh.multiblock_reverse_csr_node_corners.reset(
        reverse_csr.node_corners.size());
    state.mesh.multiblock_reverse_csr_node_corners.copy_from_host(
        reverse_csr.node_corners);
  }
}

inline std::vector<std::int8_t> multiblock_winslow_face_tags_i8(
    const std::vector<int>& tags) {
  std::vector<std::int8_t> out(tags.size(), 0);
  for (std::size_t i = 0; i < tags.size(); ++i) {
    TENRYU_ASSERT(tags[i] >= std::numeric_limits<std::int8_t>::min() &&
                      tags[i] <= std::numeric_limits<std::int8_t>::max(),
                  "CSR Winslow face bc_tag does not fit int8_t");
    out[i] = static_cast<std::int8_t>(tags[i]);
  }
  return out;
}

inline double max_node_delta_device(const double* x_r_new,
                                    const double* x_z_new,
                                    const double* x_r_old,
                                    const double* x_z_old,
                                    const int n_nodes) {
  double* d_max_delta = nullptr;
  d_max_delta = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:max_node_delta:d_max_delta",
                                   sizeof(double)));
  const double zero = 0.0;
  CUDA_CHECK(cudaMemcpy(d_max_delta, &zero, sizeof(double), cudaMemcpyHostToDevice));
  const int blocks = (n_nodes + 255) / 256;
  max_node_delta_kernel<<<blocks, 256>>>(
      x_r_new, x_z_new, x_r_old, x_z_old, d_max_delta, n_nodes);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  double out = 0.0;
  CUDA_CHECK(cudaMemcpy(&out, d_max_delta, sizeof(double), cudaMemcpyDeviceToHost));
  return out;
}

}  // namespace detail

struct MultiblockWinslowSmoothStats {
  int iterations_attempted = 0;
  int accepted_iterations = 0;
  int omega_halving_count = 0;
  int backtrack_attempts = 0;
  double omega_final = 0.0;
  double max_displacement = 0.0;
  mesh::CandidateMeshQuality final_quality{};
  std::vector<double> min_corner_j_rel_history;
};

__device__ inline void multiblock_winslow_append_neighbor(
    int* neighbors,
    int& n_neighbors,
    bool& overflow,
    const int max_neighbors,
    const int self,
    const int nb) {
  if (nb == self) {
    return;
  }
  bool duplicate = false;
  for (int k = 0; k < n_neighbors; ++k) {
    duplicate = duplicate || neighbors[k] == nb;
  }
  if (duplicate) {
    return;
  }
  if (n_neighbors >= max_neighbors) {
    overflow = true;
    return;
  }
  neighbors[n_neighbors++] = nb;
}

__device__ inline double multiblock_winslow_physical_coord(
    const double coord,
    const double* __restrict__ reference,
    const int n,
    const bool smooth_reference_displacement) {
  return smooth_reference_displacement ? reference[n] + coord : coord;
}

__device__ inline double multiblock_winslow_axis_spacing_floor(
    const double z_a,
    const double z_b,
    const double reference_spacing) {
  constexpr double kRelSpacingFloor = 1.0e-12;
  constexpr double kDoubleEps = 2.2204460492503131e-16;
  const double spacing_scale =
      (isfinite(reference_spacing) && reference_spacing > 0.0)
          ? reference_spacing
          : fabs(z_b - z_a);
  const double coord_scale = fmax(fabs(z_a), fabs(z_b));
  return fmax(kRelSpacingFloor * spacing_scale,
              100.0 * kDoubleEps * coord_scale);
}

constexpr double kMultiblockWinslowBarrierFloorRel = 1.0e-8;
constexpr double kMultiblockWinslowBarrierQualityTrigger = 0.2;
constexpr int kMultiblockWinslowBacktrackIters = 24;

struct MultiblockWinslowLocalQuality {
  bool admissible = true;
  double min_area_rel = 1.0e300;
  double min_corner_j_rel = 1.0e300;
  double min_condition_quality = 1.0e300;
  double min_affected_condition_quality = 1.0e300;
  int worst_affected_cell = -1;
  int worst_affected_corner = -1;
  int worst_affected_moved_corner = -1;
  double worst_affected_edge_scale = 0.0;
};

struct MultiblockWinslowNodeTrial {
  double r = 0.0;
  double z = 0.0;
  double lambda = 0.0;
  MultiblockWinslowLocalQuality quality{};
};

__device__ inline double multiblock_winslow_signed_quad_area(
    const double* rr,
    const double* zz) {
  return 0.5 * (rr[0] * zz[1] - zz[0] * rr[1] +
                rr[1] * zz[2] - zz[1] * rr[2] +
                rr[2] * zz[3] - zz[2] * rr[3] +
                rr[3] * zz[0] - zz[3] * rr[0]);
}

__device__ inline double multiblock_winslow_signed_polygon_area(
    const double* rr,
    const double* zz,
    const int active_nverts) {
  if (active_nverts == 3) {
    return 0.5 * (rr[0] * zz[1] - zz[0] * rr[1] +
                  rr[1] * zz[2] - zz[1] * rr[2] +
                  rr[2] * zz[0] - zz[2] * rr[0]);
  }
  return multiblock_winslow_signed_quad_area(rr, zz);
}

__device__ inline double multiblock_winslow_corner_jacobian(
    const double* rr,
    const double* zz,
    const int corner,
    const int active_nverts) {
  if (active_nverts == 3) {
    const int kp = (corner + 1) % 3;
    const int km = (corner + 2) % 3;
    return tenryu::hydro::corner_jacobian_cross2(
        rr[kp] - rr[corner], zz[kp] - zz[corner],
        rr[km] - rr[corner], zz[km] - zz[corner]);
  }
  return tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
}

__device__ inline bool multiblock_winslow_corner_depends_on_node(
    const int corner,
    const int moved_corner,
    const int active_nverts) {
  if (active_nverts == 3) {
    return moved_corner >= 0 && moved_corner < 3 && corner >= 0 && corner < 3;
  }
  return moved_corner >= 0 && moved_corner < 4 &&
         moved_corner != ((corner + 2) & 3);
}

__device__ inline double multiblock_winslow_corner_condition_quality(
    const double* rr,
    const double* zz,
    const int corner,
    const double oriented_j,
    const int active_nverts) {
  const int kp = active_nverts == 3 ? (corner + 1) % 3 : (corner + 1) & 3;
  const int km = active_nverts == 3 ? (corner + 2) % 3 : (corner + 3) & 3;
  const double ar = rr[kp] - rr[corner];
  const double az = zz[kp] - zz[corner];
  const double br = rr[km] - rr[corner];
  const double bz = zz[km] - zz[corner];
  const double denom = ar * ar + az * az + br * br + bz * bz;
  if (!isfinite(oriented_j) || !isfinite(denom) || !(oriented_j > 0.0) ||
      !(denom > 0.0)) {
    return -1.0e300;
  }
  return 2.0 * oriented_j / denom;
}

__device__ inline double multiblock_winslow_corner_edge_scale(
    const double* rr,
    const double* zz,
    const int corner,
    const int active_nverts) {
  const int kp = active_nverts == 3 ? (corner + 1) % 3 : (corner + 1) & 3;
  const int km = active_nverts == 3 ? (corner + 2) % 3 : (corner + 3) & 3;
  const double ar = rr[kp] - rr[corner];
  const double az = zz[kp] - zz[corner];
  const double br = rr[km] - rr[corner];
  const double bz = zz[km] - zz[corner];
  const double denom = ar * ar + az * az + br * br + bz * bz;
  return (isfinite(denom) && denom > 0.0) ? sqrt(0.5 * denom) : 0.0;
}

__device__ inline MultiblockWinslowLocalQuality
multiblock_winslow_evaluate_node_quality(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n,
    const double trial_r,
    const double trial_z,
    const bool smooth_reference_displacement) {
  MultiblockWinslowLocalQuality q{};
  const int incident_begin = node_cell_csr_offsets[n];
  const int incident_end = node_cell_csr_offsets[n + 1];
  for (int p = incident_begin; p < incident_end; ++p) {
    const int c = node_cell_csr_indices[p];
    const int moved_corner = node_cell_csr_corners[p];
    const int cell_begin = cell_node_csr_offsets[c];
    const int cell_end = cell_node_csr_offsets[c + 1];
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (moved_corner < 0 || moved_corner >= active_nverts ||
        cell_end - cell_begin != mesh::kMeshTopoCellStorageSlots) {
      q.admissible = false;
      q.min_area_rel = -1.0e300;
      q.min_corner_j_rel = -1.0e300;
      q.min_condition_quality = -1.0e300;
      continue;
    }

    double r_old_cell[4];
    double z_old_cell[4];
    double r_trial_cell[4];
    double z_trial_cell[4];
    bool finite_nodes = true;
    for (int k = 0; k < active_nverts; ++k) {
      const int node = cell_node_csr_indices[cell_begin + k];
      const double r_phys = multiblock_winslow_physical_coord(
          x_r_old[node], x_r_reference, node, smooth_reference_displacement);
      const double z_phys = multiblock_winslow_physical_coord(
          x_z_old[node], x_z_reference, node, smooth_reference_displacement);
      r_old_cell[k] = r_phys;
      z_old_cell[k] = z_phys;
      r_trial_cell[k] = (node == n) ? trial_r : r_phys;
      z_trial_cell[k] = (node == n) ? trial_z : z_phys;
      finite_nodes = finite_nodes && isfinite(r_old_cell[k]) &&
                     isfinite(z_old_cell[k]) &&
                     isfinite(r_trial_cell[k]) &&
                     isfinite(z_trial_cell[k]);
    }
    if (!finite_nodes) {
      q.admissible = false;
      q.min_area_rel = -1.0e300;
      q.min_corner_j_rel = -1.0e300;
      q.min_condition_quality = -1.0e300;
      continue;
    }

    const double area_old =
        multiblock_winslow_signed_polygon_area(
            r_old_cell, z_old_cell, active_nverts);
    const double area_trial =
        multiblock_winslow_signed_polygon_area(
            r_trial_cell, z_trial_cell, active_nverts);
    const double area_ref = fabs(area_old);
    if (!isfinite(area_old) || !isfinite(area_trial) || !(area_ref > 0.0)) {
      q.admissible = false;
      q.min_area_rel = -1.0e300;
    } else {
      const double orient = area_old >= 0.0 ? 1.0 : -1.0;
      const double area_oriented = orient * area_trial;
      const double area_rel = area_oriented / area_ref;
      q.min_area_rel = fmin(q.min_area_rel, area_rel);
      if (!isfinite(area_rel) ||
          !(area_oriented > kMultiblockWinslowBarrierFloorRel * area_ref)) {
        q.admissible = false;
      }
    }

    for (int corner = 0; corner < active_nverts; ++corner) {
      const double j_old = multiblock_winslow_corner_jacobian(
          r_old_cell, z_old_cell, corner, active_nverts);
      const double j_trial = multiblock_winslow_corner_jacobian(
          r_trial_cell, z_trial_cell, corner, active_nverts);
      const double j_ref = fabs(j_old);
      const bool affected =
          multiblock_winslow_corner_depends_on_node(
              corner, moved_corner, active_nverts);
      double condition_quality = -1.0e300;
      if (!isfinite(j_old) || !isfinite(j_trial) || !(j_ref > 0.0)) {
        q.admissible = false;
        q.min_corner_j_rel = -1.0e300;
        q.min_condition_quality = -1.0e300;
      } else {
        const double orient = j_old >= 0.0 ? 1.0 : -1.0;
        const double j_oriented = orient * j_trial;
        const double j_rel = j_oriented / j_ref;
        condition_quality =
            multiblock_winslow_corner_condition_quality(
                r_trial_cell, z_trial_cell, corner, j_oriented, active_nverts);
        q.min_corner_j_rel = fmin(q.min_corner_j_rel, j_rel);
        if (!isfinite(j_rel) ||
            !(j_oriented > kMultiblockWinslowBarrierFloorRel * j_ref)) {
          q.admissible = false;
        }
      }

      q.min_condition_quality =
          fmin(q.min_condition_quality, condition_quality);
      if (affected &&
          condition_quality < q.min_affected_condition_quality) {
        q.min_affected_condition_quality = condition_quality;
        q.worst_affected_cell = c;
        q.worst_affected_corner = corner;
        q.worst_affected_moved_corner = moved_corner;
        q.worst_affected_edge_scale =
            multiblock_winslow_corner_edge_scale(
                r_trial_cell, z_trial_cell, corner, active_nverts);
      }
    }
  }
  if (q.min_area_rel == 1.0e300 ||
      q.min_corner_j_rel == 1.0e300 ||
      q.min_condition_quality == 1.0e300) {
    q.admissible = false;
    q.min_area_rel = -1.0e300;
    q.min_corner_j_rel = -1.0e300;
    q.min_condition_quality = -1.0e300;
  }
  return q;
}

__device__ inline MultiblockWinslowNodeTrial
multiblock_winslow_backtrack_node_trial(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n,
    const double old_r,
    const double old_z,
    const double dr,
    const double dz,
    const bool smooth_reference_displacement,
    const MultiblockWinslowLocalQuality& old_quality,
    const MultiblockWinslowLocalQuality& full_quality) {
  MultiblockWinslowNodeTrial out;
  out.r = old_r;
  out.z = old_z;
  out.quality = old_quality;
  if (!isfinite(dr) || !isfinite(dz) || !(dr * dr + dz * dz > 0.0) ||
      !old_quality.admissible) {
    return out;
  }
  if (full_quality.admissible) {
    out.r = old_r + dr;
    out.z = old_z + dz;
    out.lambda = 1.0;
    out.quality = full_quality;
    return out;
  }

  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < kMultiblockWinslowBacktrackIters; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const MultiblockWinslowLocalQuality mid_quality =
        multiblock_winslow_evaluate_node_quality(
            x_r_old,
            x_z_old,
            x_r_reference,
            x_z_reference,
            node_cell_csr_offsets,
            node_cell_csr_indices,
            node_cell_csr_corners,
            cell_node_csr_offsets,
            cell_node_csr_indices,
            cell_nverts,
            n,
            old_r + mid * dr,
            old_z + mid * dz,
            smooth_reference_displacement);
    if (mid_quality.admissible) {
      lo = mid;
      out.r = old_r + lo * dr;
      out.z = old_z + lo * dz;
      out.lambda = lo;
      out.quality = mid_quality;
    } else {
      hi = mid;
    }
  }
  return out;
}

__device__ inline bool multiblock_winslow_trial_better(
    const MultiblockWinslowNodeTrial& candidate,
    const MultiblockWinslowNodeTrial& current) {
  if (candidate.quality.admissible != current.quality.admissible) {
    return candidate.quality.admissible;
  }
  if (!candidate.quality.admissible) {
    return false;
  }
  constexpr double kTie = 1.0e-14;
  const double q_scale =
      fmax(1.0, fabs(current.quality.min_condition_quality));
  if (candidate.quality.min_condition_quality >
      current.quality.min_condition_quality + kTie * q_scale) {
    return true;
  }
  if (candidate.quality.min_condition_quality <
      current.quality.min_condition_quality - kTie * q_scale) {
    return false;
  }
  const double j_scale = fmax(1.0, fabs(current.quality.min_corner_j_rel));
  if (candidate.quality.min_corner_j_rel >
      current.quality.min_corner_j_rel + kTie * j_scale) {
    return true;
  }
  if (candidate.quality.min_corner_j_rel <
      current.quality.min_corner_j_rel - kTie * j_scale) {
    return false;
  }
  return candidate.lambda > current.lambda + kTie;
}

__device__ inline bool multiblock_winslow_worst_j_gradient(
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n,
    const MultiblockWinslowLocalQuality& quality,
    const bool smooth_reference_displacement,
    double& grad_r,
    double& grad_z) {
  grad_r = 0.0;
  grad_z = 0.0;
  if (quality.worst_affected_cell < 0 ||
      quality.worst_affected_corner < 0 ||
      quality.worst_affected_moved_corner < 0) {
    return false;
  }
  const int cell_begin =
      cell_node_csr_offsets[quality.worst_affected_cell];
  const int cell_end =
      cell_node_csr_offsets[quality.worst_affected_cell + 1];
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, quality.worst_affected_cell);
  if (cell_end - cell_begin != mesh::kMeshTopoCellStorageSlots) {
    return false;
  }
  double rr[4];
  double zz[4];
  bool found = false;
  for (int k = 0; k < active_nverts; ++k) {
    const int node = cell_node_csr_indices[cell_begin + k];
    rr[k] = multiblock_winslow_physical_coord(
        x_r_old[node], x_r_reference, node, smooth_reference_displacement);
    zz[k] = multiblock_winslow_physical_coord(
        x_z_old[node], x_z_reference, node, smooth_reference_displacement);
    found = found || node == n;
  }
  if (!found) {
    return false;
  }
  const int corner = quality.worst_affected_corner;
  const int moved = quality.worst_affected_moved_corner;
  if (corner < 0 || corner >= active_nverts ||
      moved < 0 || moved >= active_nverts) {
    return false;
  }
  const int kp = active_nverts == 3 ? (corner + 1) % 3 : (corner + 1) & 3;
  const int km = active_nverts == 3 ? (corner + 2) % 3 : (corner + 3) & 3;
  const double j_old =
      multiblock_winslow_corner_jacobian(rr, zz, corner, active_nverts);
  if (!isfinite(j_old) || j_old == 0.0) {
    return false;
  }
  const double orient = j_old >= 0.0 ? 1.0 : -1.0;
  if (moved == corner) {
    grad_r = zz[kp] - zz[km];
    grad_z = rr[km] - rr[kp];
  } else if (moved == kp) {
    grad_r = zz[km] - zz[corner];
    grad_z = -(rr[km] - rr[corner]);
  } else if (moved == km) {
    grad_r = -(zz[kp] - zz[corner]);
    grad_z = rr[kp] - rr[corner];
  } else {
    return false;
  }
  grad_r *= orient;
  grad_z *= orient;
  const double norm2 = grad_r * grad_r + grad_z * grad_z;
  return isfinite(norm2) && norm2 > 0.0;
}

static __global__ void multiblock_winslow_jacobi_kernel(
    double* __restrict__ x_r_new,
    double* __restrict__ x_z_new,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ x_r_reference,
    const double* __restrict__ x_z_reference,
    const int* __restrict__ node_cell_csr_offsets,
    const int* __restrict__ node_cell_csr_indices,
    const int* __restrict__ node_cell_csr_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_block_id,
    const int* __restrict__ face_adj_csr_offsets,
    const int* __restrict__ face_adj_csr_indices,
    const std::int8_t* __restrict__ face_bc_tags,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ d_rezone_active_node_mask,
    int* __restrict__ backtrack_attempts,
    const int n_nodes_total,
    const double omega,
    const bool smooth_reference_displacement,
    const bool smooth_cross_seams) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes_total) {
    return;
  }
  (void)face_adj_csr_indices;

  const double r_old = x_r_old[n];
  const double z_old = x_z_old[n];
  x_r_new[n] = r_old;
  x_z_new[n] = z_old;
  if (d_rezone_active_node_mask != nullptr &&
      d_rezone_active_node_mask[n] == 0U) {
    x_r_new[n] = r_old;
    x_z_new[n] = z_old;
    return;
  }

  const std::uint8_t flags = node_flags[n];
  const bool pole_axis_node = (flags & mesh::NODE_POLE_AXIS) != 0U;
  constexpr std::uint8_t always_fixed_mask = static_cast<std::uint8_t>(
      mesh::NODE_OUTER_PHYSICAL_BOUNDARY | mesh::NODE_CENTER);
  if ((flags & always_fixed_mask) != 0U ||
      (!pole_axis_node && (flags & mesh::NODE_AXIS) != 0U)) {
    return;
  }

  const int incident_begin = node_cell_csr_offsets[n];
  const int incident_end = node_cell_csr_offsets[n + 1];
  if (incident_end <= incident_begin) {
    return;
  }

  int incident_block = -1;
  bool seam_touching = false;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int c = node_cell_csr_indices[p];
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    const int block = cell_block_id[c];
    if (incident_block < 0) {
      incident_block = block;
    } else if (block != incident_block) {
      seam_touching = true;
    }
    const int face_begin = face_adj_csr_offsets[c];
    const int face_end = face_adj_csr_offsets[c + 1];
    for (int f = face_begin; f < face_end; ++f) {
      const int local_face = f - face_begin;
      if (!mesh::mesh_topo_local_face_is_active(active_nverts, local_face)) {
        continue;
      }
      if (detail::multiblock_winslow_is_seam_tag(face_bc_tags[f])) {
        seam_touching = true;
      }
    }
  }
  if (seam_touching && !smooth_cross_seams) {
    return;
  }

  constexpr int kMaxNeighbors = 64;
  int neighbors[kMaxNeighbors];
  int n_neighbors = 0;
  bool overflow = false;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int c = node_cell_csr_indices[p];
    const int cell_begin = cell_node_csr_offsets[c];
    const int cell_end = cell_node_csr_offsets[c + 1];
    const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (seam_touching && smooth_cross_seams) {
      const int corner = node_cell_csr_corners[p];
      if (corner < 0 || corner >= active_nverts ||
          cell_end - cell_begin != mesh::kMeshTopoCellStorageSlots) {
        overflow = true;
        continue;
      }
      const int local_a =
          active_nverts == 3 ? (corner + 1) % 3 : (corner + 1) & 3;
      const int local_b =
          active_nverts == 3 ? (corner + 2) % 3 : (corner + 3) & 3;
      multiblock_winslow_append_neighbor(neighbors,
                                         n_neighbors,
                                         overflow,
                                         kMaxNeighbors,
                                         n,
                                         cell_node_csr_indices[cell_begin + local_a]);
      multiblock_winslow_append_neighbor(neighbors,
                                         n_neighbors,
                                         overflow,
                                         kMaxNeighbors,
                                         n,
                                         cell_node_csr_indices[cell_begin + local_b]);
      continue;
    }
    for (int q = cell_begin; q < cell_begin + active_nverts; ++q) {
      multiblock_winslow_append_neighbor(neighbors,
                                         n_neighbors,
                                         overflow,
                                         kMaxNeighbors,
                                         n,
                                         cell_node_csr_indices[q]);
    }
  }
  if (overflow || n_neighbors == 0) {
    return;
  }

  if (pole_axis_node) {
    const double r_axis =
        smooth_reference_displacement ? -x_r_reference[n] : 0.0;
    x_r_new[n] = r_axis;

    const double z_phys_old = multiblock_winslow_physical_coord(
        z_old, x_z_reference, n, smooth_reference_displacement);
    if (!isfinite(z_phys_old)) {
      return;
    }

    double sum_axis_z = 0.0;
    int n_axis_neighbors = 0;
    bool has_lower = false;
    bool has_upper = false;
    bool coincident_axis_neighbor = false;
    double lower_z = 0.0;
    double upper_z = 0.0;
    double lower_reference_spacing = 0.0;
    double upper_reference_spacing = 0.0;
    bool lower_fixed = false;
    bool upper_fixed = false;
    for (int k = 0; k < n_neighbors; ++k) {
      const int nb = neighbors[k];
      const std::uint8_t nb_flags = node_flags[nb];
      if ((nb_flags & mesh::NODE_POLE_AXIS) == 0U) {
        continue;
      }
      const double nb_z = multiblock_winslow_physical_coord(
          x_z_old[nb], x_z_reference, nb, smooth_reference_displacement);
      if (!isfinite(nb_z)) {
        continue;
      }
      sum_axis_z += nb_z;
      ++n_axis_neighbors;
      const double dz = nb_z - z_phys_old;
      const double reference_spacing =
          smooth_reference_displacement
              ? fabs(x_z_reference[nb] - x_z_reference[n])
              : fabs(dz);
      const bool nb_fixed = (nb_flags & always_fixed_mask) != 0U;
      if (dz < 0.0) {
        if (!has_lower || nb_z > lower_z) {
          has_lower = true;
          lower_z = nb_z;
          lower_reference_spacing = reference_spacing;
          lower_fixed = nb_fixed;
        }
      } else if (dz > 0.0) {
        if (!has_upper || nb_z < upper_z) {
          has_upper = true;
          upper_z = nb_z;
          upper_reference_spacing = reference_spacing;
          upper_fixed = nb_fixed;
        }
      } else {
        coincident_axis_neighbor = true;
      }
    }
    if (n_axis_neighbors == 0 || coincident_axis_neighbor) {
      return;
    }

    const double target_z =
        sum_axis_z / static_cast<double>(n_axis_neighbors);
    double candidate_z = (1.0 - omega) * z_phys_old + omega * target_z;
    if (!isfinite(candidate_z)) {
      return;
    }

    bool feasible = true;
    double lower_bound = 0.0;
    double upper_bound = 0.0;
    if (has_lower) {
      const double s_min = multiblock_winslow_axis_spacing_floor(
          z_phys_old, lower_z, lower_reference_spacing);
      feasible = feasible && (s_min > 0.0);
      lower_bound = lower_fixed ? lower_z + s_min
                                : 0.5 * (lower_z + z_phys_old + s_min);
      feasible = feasible && isfinite(lower_bound);
      if (candidate_z < lower_bound) {
        candidate_z = lower_bound;
      }
    }
    if (has_upper) {
      const double s_min = multiblock_winslow_axis_spacing_floor(
          z_phys_old, upper_z, upper_reference_spacing);
      feasible = feasible && (s_min > 0.0);
      upper_bound = upper_fixed ? upper_z - s_min
                                : 0.5 * (z_phys_old + upper_z - s_min);
      feasible = feasible && isfinite(upper_bound);
      if (candidate_z > upper_bound) {
        candidate_z = upper_bound;
      }
    }
    if (has_lower && has_upper) {
      feasible = feasible && (lower_bound <= upper_bound);
    }
    if (!feasible ||
        (has_lower && candidate_z < lower_bound) ||
        (has_upper && candidate_z > upper_bound)) {
      return;
    }

    x_z_new[n] =
        smooth_reference_displacement ? candidate_z - x_z_reference[n]
                                      : candidate_z;
    return;
  }

  double sum_r = 0.0;
  double sum_z = 0.0;
  for (int k = 0; k < n_neighbors; ++k) {
    const int nb = neighbors[k];
    sum_r += x_r_old[nb];
    sum_z += x_z_old[nb];
  }
  const double inv_n = 1.0 / static_cast<double>(n_neighbors);
  const double target_r = sum_r * inv_n;
  const double target_z = sum_z * inv_n;
  const double old_r_phys = multiblock_winslow_physical_coord(
      r_old, x_r_reference, n, smooth_reference_displacement);
  const double old_z_phys = multiblock_winslow_physical_coord(
      z_old, x_z_reference, n, smooth_reference_displacement);
  const double base_r = (1.0 - omega) * r_old + omega * target_r;
  const double base_z = (1.0 - omega) * z_old + omega * target_z;
  const double base_dr = base_r - r_old;
  const double base_dz = base_z - z_old;
  if (!isfinite(old_r_phys) || !isfinite(old_z_phys) ||
      !isfinite(base_dr) || !isfinite(base_dz)) {
    return;
  }

  const MultiblockWinslowLocalQuality old_quality =
      multiblock_winslow_evaluate_node_quality(
          x_r_old,
          x_z_old,
          x_r_reference,
          x_z_reference,
          node_cell_csr_offsets,
          node_cell_csr_indices,
          node_cell_csr_corners,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          n,
          old_r_phys,
          old_z_phys,
          smooth_reference_displacement);
  const MultiblockWinslowLocalQuality base_full_quality =
      multiblock_winslow_evaluate_node_quality(
          x_r_old,
          x_z_old,
          x_r_reference,
          x_z_reference,
          node_cell_csr_offsets,
          node_cell_csr_indices,
          node_cell_csr_corners,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          n,
          old_r_phys + base_dr,
          old_z_phys + base_dz,
          smooth_reference_displacement);
  const MultiblockWinslowNodeTrial base_trial =
      multiblock_winslow_backtrack_node_trial(
          x_r_old,
          x_z_old,
          x_r_reference,
          x_z_reference,
          node_cell_csr_offsets,
          node_cell_csr_indices,
          node_cell_csr_corners,
          cell_node_csr_offsets,
          cell_node_csr_indices,
          cell_nverts,
          n,
          old_r_phys,
          old_z_phys,
          base_dr,
          base_dz,
          smooth_reference_displacement,
          old_quality,
          base_full_quality);
  int node_backtrack_attempts =
      (!base_full_quality.admissible && base_trial.quality.admissible &&
       base_trial.lambda < 1.0)
          ? 1
          : 0;

  MultiblockWinslowNodeTrial best_trial = base_trial;
  const bool barrier_active =
      old_quality.admissible &&
      (!base_full_quality.admissible ||
       base_full_quality.min_area_rel < 1.0 ||
       base_full_quality.min_corner_j_rel < 1.0 ||
       old_quality.min_condition_quality <
           kMultiblockWinslowBarrierQualityTrigger ||
       base_full_quality.min_condition_quality <
           kMultiblockWinslowBarrierQualityTrigger);
  if (barrier_active) {
    MultiblockWinslowNodeTrial old_trial;
    old_trial.r = old_r_phys;
    old_trial.z = old_z_phys;
    old_trial.lambda = 0.0;
    old_trial.quality = old_quality;
    best_trial = old_trial;
    if (multiblock_winslow_trial_better(base_trial, best_trial)) {
      best_trial = base_trial;
    }

    MultiblockWinslowLocalQuality gradient_quality = old_quality;
    if (base_full_quality.worst_affected_cell >= 0 &&
        base_full_quality.min_affected_condition_quality <
            old_quality.min_affected_condition_quality) {
      gradient_quality = base_full_quality;
    }
    double grad_r = 0.0;
    double grad_z = 0.0;
    if (multiblock_winslow_worst_j_gradient(
            x_r_old,
            x_z_old,
            x_r_reference,
            x_z_reference,
            cell_node_csr_offsets,
            cell_node_csr_indices,
            cell_nverts,
            n,
            gradient_quality,
            smooth_reference_displacement,
            grad_r,
            grad_z)) {
      const double grad_norm = sqrt(grad_r * grad_r + grad_z * grad_z);
      const double base_len = sqrt(base_dr * base_dr + base_dz * base_dz);
      const double edge_scale =
          gradient_quality.worst_affected_edge_scale > 0.0
              ? gradient_quality.worst_affected_edge_scale
              : base_len;
      if (isfinite(grad_norm) && grad_norm > 0.0 &&
          isfinite(edge_scale) && edge_scale > 0.0) {
        double barrier_len = fmax(base_len, 0.25 * omega * edge_scale);
        barrier_len = fmin(barrier_len, omega * edge_scale);
        const double barrier_dr = barrier_len * grad_r / grad_norm;
        const double barrier_dz = barrier_len * grad_z / grad_norm;

        const double trial_dr[3] = {
            barrier_dr,
            0.5 * (base_dr + barrier_dr),
            0.5 * barrier_dr,
        };
        const double trial_dz[3] = {
            barrier_dz,
            0.5 * (base_dz + barrier_dz),
            0.5 * barrier_dz,
        };
        for (int t = 0; t < 3; ++t) {
          const MultiblockWinslowLocalQuality full_quality =
              multiblock_winslow_evaluate_node_quality(
                  x_r_old,
                  x_z_old,
                  x_r_reference,
                  x_z_reference,
                  node_cell_csr_offsets,
                  node_cell_csr_indices,
                  node_cell_csr_corners,
                  cell_node_csr_offsets,
                  cell_node_csr_indices,
                  cell_nverts,
                  n,
                  old_r_phys + trial_dr[t],
                  old_z_phys + trial_dz[t],
                  smooth_reference_displacement);
          const MultiblockWinslowNodeTrial trial =
              multiblock_winslow_backtrack_node_trial(
                  x_r_old,
                  x_z_old,
                  x_r_reference,
                  x_z_reference,
                  node_cell_csr_offsets,
                  node_cell_csr_indices,
                  node_cell_csr_corners,
            cell_node_csr_offsets,
            cell_node_csr_indices,
            cell_nverts,
            n,
                  old_r_phys,
                  old_z_phys,
                  trial_dr[t],
                  trial_dz[t],
                  smooth_reference_displacement,
                  old_quality,
                  full_quality);
          if (!full_quality.admissible && trial.quality.admissible &&
              trial.lambda < 1.0) {
            ++node_backtrack_attempts;
          }
          if (multiblock_winslow_trial_better(trial, best_trial)) {
            best_trial = trial;
          }
        }
      }
    }
  }

  if (!best_trial.quality.admissible ||
      !isfinite(best_trial.r) || !isfinite(best_trial.z)) {
    return;
  }
  x_r_new[n] =
      smooth_reference_displacement ? best_trial.r - x_r_reference[n]
                                    : best_trial.r;
  x_z_new[n] =
      smooth_reference_displacement ? best_trial.z - x_z_reference[n]
                                    : best_trial.z;
  if (node_backtrack_attempts > 0 && backtrack_attempts != nullptr) {
    atomicAdd(backtrack_attempts, node_backtrack_attempts);
  }
}

inline MultiblockWinslowSmoothStats multiblock_winslow_smooth_impl(
    core::State& state,
    const core::Config& cfg,
    const int max_iterations,
    const double omega_initial,
    const std::uint8_t* d_rezone_active_node_mask,
    const bool smooth_cross_seams) {
  MultiblockWinslowSmoothStats stats;
  stats.omega_final = omega_initial;

  if (!mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return stats;
  }
  TENRYU_ASSERT(max_iterations >= 0,
                "CSR Winslow smoother max_iterations must be non-negative");
  TENRYU_ASSERT(omega_initial > 0.0 && omega_initial <= 1.0,
                "CSR Winslow smoother omega_initial must be in (0, 1]");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR Winslow smoother requires State multiblock topology");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "CSR Winslow smoother coordinate size mismatch");
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(static_cast<std::size_t>(n_nodes) == state.x_r.size(),
                "CSR Winslow smoother node count mismatch");
  if (max_iterations == 0 || n_nodes == 0) {
    return stats;
  }

  const bool masked_coordinate_smoothing =
      d_rezone_active_node_mask != nullptr;
  if (!masked_coordinate_smoothing &&
      state.x_r_reference.size() == state.x_r.size() &&
      state.x_z_reference.size() == state.x_z.size()) {
    const double max_reference_delta =
        detail::max_node_delta_device(state.x_r.data(),
                                      state.x_z.data(),
                                      state.x_r_reference.data(),
                                      state.x_z_reference.data(),
                                      n_nodes);
    if (max_reference_delta <= 0.0) {
      stats.max_displacement = 0.0;
      return stats;
    }
  }

  detail::ensure_multiblock_winslow_device_csr(state);
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(static_cast<int>(mb.cell_block_id.size()) == n_cells,
                "CSR Winslow smoother requires cell block ids");
  TENRYU_ASSERT(static_cast<int>(mb.cell_id_stable.size()) == n_cells,
                "CSR Winslow smoother requires stable cell ids");
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells + 1),
                "CSR Winslow smoother requires face adjacency offsets");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() == mb.face_bc_tags.size(),
                "CSR Winslow smoother face adjacency/tag size mismatch");
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(n_nodes),
                "CSR Winslow smoother requires node flags");

  core::DeviceArray<int> d_cell_block_id(mb.cell_block_id.size());
  d_cell_block_id.copy_from_host(mb.cell_block_id);
  core::DeviceArray<int> d_cell_id_stable(mb.cell_id_stable.size());
  d_cell_id_stable.copy_from_host(mb.cell_id_stable);
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "winslow relaxation admissibility requires cell orientation signs");
  core::DeviceArray<int> d_cell_orientation_sign(mb.cell_orientation_sign.size());
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
  core::DeviceArray<int> d_face_adj_offsets(mb.face_adj_csr_offsets.size());
  d_face_adj_offsets.copy_from_host(mb.face_adj_csr_offsets);
  core::DeviceArray<int> d_face_adj_indices(mb.face_adj_csr_indices.size());
  d_face_adj_indices.copy_from_host(mb.face_adj_csr_indices);
  const std::vector<std::int8_t> face_tags =
      detail::multiblock_winslow_face_tags_i8(mb.face_bc_tags);
  core::DeviceArray<std::int8_t> d_face_bc_tags(face_tags.size());
  d_face_bc_tags.copy_from_host(face_tags);
  core::DeviceArray<std::uint8_t> d_node_flags(state.mesh.topo.node_flags.size());
  d_node_flags.copy_from_host(state.mesh.topo.node_flags);
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const bool use_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(state.mesh.cell_nverts.begin(),
                  state.mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts == 3U; });
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (use_cell_nverts) {
    d_cell_nverts.reset(state.mesh.cell_nverts.size());
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }

  core::NodeField1D x_r_a(static_cast<std::size_t>(n_nodes));
  core::NodeField1D x_z_a(static_cast<std::size_t>(n_nodes));
  core::NodeField1D x_r_b(static_cast<std::size_t>(n_nodes));
  core::NodeField1D x_z_b(static_cast<std::size_t>(n_nodes));
  core::NodeField1D delta_r(static_cast<std::size_t>(n_nodes));
  core::NodeField1D delta_z(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<int> d_backtrack_attempts(1);

  const bool smooth_reference_displacement =
      !masked_coordinate_smoothing &&
      state.x_r_reference.size() == state.x_r.size() &&
      state.x_z_reference.size() == state.x_z.size();
  core::NodeField1D coord_r_a;
  core::NodeField1D coord_z_a;
  core::NodeField1D coord_r_b;
  core::NodeField1D coord_z_b;
  if (smooth_reference_displacement) {
    coord_r_a.reset(static_cast<std::size_t>(n_nodes));
    coord_z_a.reset(static_cast<std::size_t>(n_nodes));
    coord_r_b.reset(static_cast<std::size_t>(n_nodes));
    coord_z_b.reset(static_cast<std::size_t>(n_nodes));
    const int node_blocks = (n_nodes + 255) / 256;
    detail::multiblock_winslow_node_delta_kernel<<<node_blocks, 256>>>(
        x_r_a.data(),
        x_z_a.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.x_r_reference.data(),
        state.x_z_reference.data(),
        n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  } else {
    CUDA_CHECK(cudaMemcpy(x_r_a.data(),
                          state.x_r.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(x_z_a.data(),
                          state.x_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }

  double* cur_r = x_r_a.data();
  double* cur_z = x_z_a.data();
  double* nxt_r = x_r_b.data();
  double* nxt_z = x_z_b.data();
  double* cur_coord_r =
      smooth_reference_displacement ? coord_r_a.data() : cur_r;
  double* cur_coord_z =
      smooth_reference_displacement ? coord_z_a.data() : cur_z;
  double* nxt_coord_r =
      smooth_reference_displacement ? coord_r_b.data() : nxt_r;
  double* nxt_coord_z =
      smooth_reference_displacement ? coord_z_b.data() : nxt_z;
  double omega = omega_initial;
  constexpr double kOmegaFloor = 0.01;
  constexpr int threads = 256;
  const int node_blocks = (n_nodes + threads - 1) / threads;

  for (int iter = 0; iter < max_iterations; ++iter) {
    ++stats.iterations_attempted;
    if (smooth_reference_displacement) {
      detail::multiblock_winslow_reference_add_kernel<<<node_blocks, threads>>>(
          cur_coord_r,
          cur_coord_z,
          state.x_r_reference.data(),
          state.x_z_reference.data(),
          cur_r,
          cur_z,
          n_nodes);
      CUDA_CHECK(cudaGetLastError());
    }
    bool accepted = false;
    mesh::CandidateMeshQuality accepted_quality{};
    while (!accepted) {
      multiblock_winslow_jacobi_kernel<<<node_blocks, threads>>>(
          nxt_r,
          nxt_z,
          cur_r,
          cur_z,
          smooth_reference_displacement ? state.x_r_reference.data() : nullptr,
          smooth_reference_displacement ? state.x_z_reference.data() : nullptr,
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_nverts_ptr,
          d_cell_block_id.data(),
          d_face_adj_offsets.data(),
          d_face_adj_indices.data(),
          d_face_bc_tags.data(),
          d_node_flags.data(),
          d_rezone_active_node_mask,
          d_backtrack_attempts.data(),
          n_nodes,
          omega,
          smooth_reference_displacement,
          smooth_cross_seams);
      CUDA_CHECK(cudaGetLastError());

      if (smooth_reference_displacement) {
        detail::multiblock_winslow_reference_add_kernel<<<node_blocks, threads>>>(
            nxt_coord_r,
            nxt_coord_z,
            state.x_r_reference.data(),
            state.x_z_reference.data(),
            nxt_r,
            nxt_z,
            n_nodes);
        CUDA_CHECK(cudaGetLastError());
      }

      detail::multiblock_winslow_node_delta_kernel<<<node_blocks, threads>>>(
          delta_r.data(),
          delta_z.data(),
          nxt_coord_r,
          nxt_coord_z,
          cur_coord_r,
          cur_coord_z,
          n_nodes);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      const mesh::CandidateMeshQuality quality =
          mesh::evaluate_candidate_mesh_quality_csr(
              cur_coord_r,
              cur_coord_z,
              delta_r.data(),
              delta_z.data(),
              1.0,
              n_cells,
              state.mesh.corner_stride,
              state.mesh.multiblock_cell_node_csr_offsets.data(),
              state.mesh.multiblock_cell_node_csr_indices.data(),
              d_cell_id_stable.data(),
              d_cell_orientation_sign.data(),
              mesh::CandidateMeshAdmissibilityFloors{},
              d_cell_nverts_ptr,
              nullptr,
              nullptr,
              0,
              state.x_r_reference.size() == state.x_r.size() &&
                      state.x_z_reference.size() == state.x_z.size()
                  ? state.x_r_reference.data()
                  : nullptr,
              state.x_r_reference.size() == state.x_r.size() &&
                      state.x_z_reference.size() == state.x_z.size()
                  ? state.x_z_reference.data()
                  : nullptr);
      if (quality.admissible()) {
        accepted = true;
        accepted_quality = quality;
        break;
      }

      omega *= 0.5;
      ++stats.omega_halving_count;
      if (omega < kOmegaFloor) {
        // Reject-not-die: no admissible relaxation fraction exists for this
        // iteration (e.g. the pole-shear ceiling at deep convergence). The
        // smoother stops with the iterations accepted so far -- cur_coord
        // still holds the last ADMISSIBLE iterate -- instead of aborting
        // the run; the downstream rezone transaction and hydro gates
        // respond on the (partially smoothed or unsmoothed) mesh.
        core::log_warning(
            "CSR Winslow smoother omega floor reached: omega=" +
            std::to_string(omega) + ", first_bad_cell=" +
            std::to_string(quality.first_bad_cell) +
            ", first_bad_stable_cell=" +
            std::to_string(quality.first_bad_stable_cell) +
            "; stopping the smoother with " +
            std::to_string(stats.accepted_iterations) +
            " accepted iterations (reject-not-die)");
        break;
      }
    }
    if (!accepted) {
      break;
    }

    stats.max_displacement =
        detail::max_node_delta_device(
            nxt_coord_r, nxt_coord_z, cur_coord_r, cur_coord_z, n_nodes);
    stats.final_quality = accepted_quality;
    stats.min_corner_j_rel_history.push_back(accepted_quality.min_corner_j_rel);
    ++stats.accepted_iterations;
    stats.omega_final = omega;

    double* old_cur_r = cur_r;
    double* old_cur_z = cur_z;
    cur_r = nxt_r;
    cur_z = nxt_z;
    nxt_r = old_cur_r;
    nxt_z = old_cur_z;
    if (smooth_reference_displacement) {
      double* old_cur_coord_r = cur_coord_r;
      double* old_cur_coord_z = cur_coord_z;
      cur_coord_r = nxt_coord_r;
      cur_coord_z = nxt_coord_z;
      nxt_coord_r = old_cur_coord_r;
      nxt_coord_z = old_cur_coord_z;
    } else {
      cur_coord_r = cur_r;
      cur_coord_z = cur_z;
      nxt_coord_r = nxt_r;
      nxt_coord_z = nxt_z;
    }
  }

  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        cur_coord_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        cur_coord_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  state.mesh.recompute_geometry();
  state.vol = state.mesh.cell_vol;
  std::vector<int> backtrack_attempts_host;
  d_backtrack_attempts.copy_to_host(backtrack_attempts_host);
  stats.backtrack_attempts = backtrack_attempts_host.empty()
                                 ? 0
                                 : backtrack_attempts_host.front();
  return stats;
}

inline MultiblockWinslowSmoothStats multiblock_winslow_smooth_with_stats(
    core::State& state,
    const core::Config& cfg,
    const int max_iterations,
    const double omega_initial,
    const std::uint8_t* d_rezone_active_node_mask = nullptr) {
  return multiblock_winslow_smooth_impl(
      state, cfg, max_iterations, omega_initial, d_rezone_active_node_mask, false);
}

inline MultiblockWinslowSmoothStats multiblock_winslow_smooth_with_seams_stats(
    core::State& state,
    const core::Config& cfg,
    const int max_iterations,
    const double omega_initial,
    const std::uint8_t* d_rezone_active_node_mask = nullptr) {
  return multiblock_winslow_smooth_impl(
      state, cfg, max_iterations, omega_initial, d_rezone_active_node_mask, true);
}

inline void multiblock_winslow_smooth(core::State& state,
                                      const core::Config& cfg,
                                      const int max_iterations,
                                      const double omega_initial,
                                      const std::uint8_t* d_rezone_active_node_mask = nullptr) {
  (void)multiblock_winslow_smooth_with_stats(
      state, cfg, max_iterations, omega_initial, d_rezone_active_node_mask);
}

inline void multiblock_winslow_smooth_with_seams(core::State& state,
                                                 const core::Config& cfg,
                                                 const int max_iterations,
                                                 const double omega_initial,
                                                 const std::uint8_t* d_rezone_active_node_mask = nullptr) {
  (void)multiblock_winslow_smooth_with_seams_stats(
      state, cfg, max_iterations, omega_initial, d_rezone_active_node_mask);
}

inline double compute_min_quality(core::State& state,
                                  bool* mesh_tangle = nullptr,
                                  AleMinQualityCell* min_cell = nullptr,
                                  const bool reject_zero_gauss_j = false,
                                  const double zero_gauss_j_floor_rel = 1.0e-8) {
  TENRYU_ASSERT(state.mesh.dim == 2, "ALE quality check requires 2D mesh");
  const int n_cells = state.mesh.topo.n_cells;
  if (min_cell != nullptr) {
    *min_cell = AleMinQualityCell{};
  }
  if (n_cells <= 0) {
    if (mesh_tangle != nullptr) {
      *mesh_tangle = false;
    }
    return 1.0;
  }

  if (state.mesh.topo.multiblock.has_value()) {
    const int n_nodes = state.mesh.topo.n_nodes;
    const auto& topology = *state.mesh.topo.multiblock;
    core::DeviceArray<double> d_delta_r(static_cast<std::size_t>(n_nodes));
    core::DeviceArray<double> d_delta_z(static_cast<std::size_t>(n_nodes));
    CUDA_CHECK(cudaMemset(
        d_delta_r.data(), 0, static_cast<std::size_t>(n_nodes) * sizeof(double)));
    CUDA_CHECK(cudaMemset(
        d_delta_z.data(), 0, static_cast<std::size_t>(n_nodes) * sizeof(double)));
    core::DeviceArray<int> d_cell_id_stable(
        topology.cell_id_stable.size());
    core::DeviceArray<int> d_cell_orientation_sign(
        topology.cell_orientation_sign.size());
    core::DeviceArray<std::uint8_t> d_cell_nverts(
        state.mesh.cell_nverts.size());
    d_cell_id_stable.copy_from_host(topology.cell_id_stable);
    d_cell_orientation_sign.copy_from_host(
        topology.cell_orientation_sign);
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    mesh::CandidateMeshAdmissibilityFloors floors;
    floors.volume_rel = 0.0;
    floors.corner_j_rel = 0.0;
    floors.gauss_j_rel = 0.0;
    const mesh::CandidateMeshQuality quality =
        mesh::evaluate_candidate_mesh_quality_csr(
            state.x_r.data(),
            state.x_z.data(),
            d_delta_r.data(),
            d_delta_z.data(),
            1.0,
            n_cells,
            state.corner_stride,
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_id_stable.data(),
            d_cell_orientation_sign.data(),
            floors,
            d_cell_nverts.data(),
            nullptr,
            nullptr,
            0,
            state.x_r_reference.data(),
            state.x_z_reference.data());

    std::vector<double> x_r;
    std::vector<double> x_z;
    std::vector<double> x_r_reference;
    std::vector<double> x_z_reference;
    state.x_r.copy_to_host(x_r);
    state.x_z.copy_to_host(x_z);
    state.x_r_reference.copy_to_host(x_r_reference);
    state.x_z_reference.copy_to_host(x_z_reference);

    constexpr double kFlatCornerRelEps = 1.0e-9;
    double min_quality = 1.0;
    int worst_cell = -1;
    bool tangled = !quality.admissible();
    for (int c = 0; c < n_cells; ++c) {
      const int nverts =
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
      const int offset =
          topology.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const double orientation_sign = static_cast<double>(
          topology.cell_orientation_sign[static_cast<std::size_t>(c)]);

      double max_reference_wedge = 0.0;
      for (int k = 0; k < nverts; ++k) {
        const int kp = (k + 1) % nverts;
        const int km = (k + nverts - 1) % nverts;
        const int node =
            topology.cell_node_csr_indices[static_cast<std::size_t>(offset + k)];
        const int node_p = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + kp)];
        const int node_m = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + km)];
        const double edge_p_r =
            x_r_reference[static_cast<std::size_t>(node_p)] -
            x_r_reference[static_cast<std::size_t>(node)];
        const double edge_p_z =
            x_z_reference[static_cast<std::size_t>(node_p)] -
            x_z_reference[static_cast<std::size_t>(node)];
        const double edge_m_r =
            x_r_reference[static_cast<std::size_t>(node_m)] -
            x_r_reference[static_cast<std::size_t>(node)];
        const double edge_m_z =
            x_z_reference[static_cast<std::size_t>(node_m)] -
            x_z_reference[static_cast<std::size_t>(node)];
        const double reference_wedge =
            orientation_sign *
            (edge_p_r * edge_m_z - edge_p_z * edge_m_r);
        max_reference_wedge =
            std::max(max_reference_wedge, std::abs(reference_wedge));
      }

      double cell_quality = 1.0;
      for (int k = 0; k < nverts; ++k) {
        const int kp = (k + 1) % nverts;
        const int km = (k + nverts - 1) % nverts;
        const int node =
            topology.cell_node_csr_indices[static_cast<std::size_t>(offset + k)];
        const int node_p = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + kp)];
        const int node_m = topology.cell_node_csr_indices[
            static_cast<std::size_t>(offset + km)];
        const double reference_edge_p_r =
            x_r_reference[static_cast<std::size_t>(node_p)] -
            x_r_reference[static_cast<std::size_t>(node)];
        const double reference_edge_p_z =
            x_z_reference[static_cast<std::size_t>(node_p)] -
            x_z_reference[static_cast<std::size_t>(node)];
        const double reference_edge_m_r =
            x_r_reference[static_cast<std::size_t>(node_m)] -
            x_r_reference[static_cast<std::size_t>(node)];
        const double reference_edge_m_z =
            x_z_reference[static_cast<std::size_t>(node_m)] -
            x_z_reference[static_cast<std::size_t>(node)];
        const double reference_wedge =
            orientation_sign *
            (reference_edge_p_r * reference_edge_m_z -
             reference_edge_p_z * reference_edge_m_r);
        if (nverts > 4 &&
            std::abs(reference_wedge) <=
                kFlatCornerRelEps * max_reference_wedge) {
          continue;
        }

        const double edge_p_r = x_r[static_cast<std::size_t>(node_p)] -
                                x_r[static_cast<std::size_t>(node)];
        const double edge_p_z = x_z[static_cast<std::size_t>(node_p)] -
                                x_z[static_cast<std::size_t>(node)];
        const double edge_m_r = x_r[static_cast<std::size_t>(node_m)] -
                                x_r[static_cast<std::size_t>(node)];
        const double edge_m_z = x_z[static_cast<std::size_t>(node_m)] -
                                x_z[static_cast<std::size_t>(node)];
        const double wedge =
            orientation_sign *
            (edge_p_r * edge_m_z - edge_p_z * edge_m_r);
        const double edge_product =
            std::hypot(edge_p_r, edge_p_z) * std::hypot(edge_m_r, edge_m_z);
        if (!(wedge > 0.0) || !(edge_product > 0.0) ||
            !std::isfinite(wedge) || !std::isfinite(edge_product)) {
          tangled = true;
          cell_quality = 0.0;
          continue;
        }
        cell_quality = std::min(cell_quality, wedge / edge_product);
      }
      if (worst_cell < 0 || cell_quality < min_quality) {
        min_quality = cell_quality;
        worst_cell = c;
      }
    }

    if (mesh_tangle != nullptr) {
      *mesh_tangle = tangled;
    }
    if (min_cell != nullptr) {
      min_cell->c = worst_cell;
    }
    return min_quality;
  }

  double* d_q = nullptr;
  int* d_tangle = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  d_q = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:min_quality:d_q",
                                   n_cells * sizeof(double)));
  d_tangle = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:min_quality:d_tangle", sizeof(int)));
  CUDA_CHECK(cudaMemset(d_tangle, 0, sizeof(int)));
  const bool use_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(state.mesh.cell_nverts.begin(),
                  state.mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts != 4U; });
  if (use_cell_nverts) {
    d_cell_nverts = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("ale_rezone:min_quality:d_cell_nverts",
                                     static_cast<std::size_t>(n_cells) *
                                         sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_cell_nverts,
                          state.mesh.cell_nverts.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
  }

  const int blocks = (n_cells + 255) / 256;
  detail::mesh_quality_check_kernel<<<blocks, 256>>>(d_q,
                                                     state.x_r.data(),
                                                     state.x_z.data(),
                                                     d_cell_nverts,
                                                     d_tangle,
                                                     state.mesh.topo.nr,
                                                     state.mesh.topo.nz,
                                                     reject_zero_gauss_j,
                                                     zero_gauss_j_floor_rel);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> q_host(static_cast<std::size_t>(n_cells), 1.0);
  int tangle = 0;
  CUDA_CHECK(cudaMemcpy(q_host.data(), d_q, n_cells * sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&tangle, d_tangle, sizeof(int), cudaMemcpyDeviceToHost));

  if (mesh_tangle != nullptr) {
    *mesh_tangle = (tangle != 0);
  }

  double q_min = 1.0;
  for (int c = 0; c < n_cells; ++c) {
    const double q = q_host[static_cast<std::size_t>(c)];
    if (q < q_min) {
      q_min = q;
      if (min_cell != nullptr) {
        min_cell->c = c;
        min_cell->i = c / state.mesh.topo.nz;
        min_cell->j = c - min_cell->i * state.mesh.topo.nz;
      }
    }
  }
  return q_min;
}

inline double compute_min_quality(core::State& state,
                                  const core::Config& cfg,
                                  bool* mesh_tangle = nullptr,
                                  AleMinQualityCell* min_cell = nullptr) {
  return compute_min_quality(state,
                             mesh_tangle,
                             min_cell,
                             cfg.numerics.ale.reject_zero_gauss_j,
                             cfg.numerics.ale.zero_gauss_j_floor_rel);
}

inline bool compute_corner_post_tangle(core::State& state,
                                       AleMinQualityCell* first_corner_fail_cell,
                                       double* min_corner_j = nullptr,
                                       int* first_corner = nullptr,
                                       const bool strict_floor_enabled = false,
                                       const double strict_floor_eps = 0.0) {
  TENRYU_ASSERT(state.mesh.dim == 2, "ALE corner-J check requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  if (first_corner_fail_cell != nullptr) {
    *first_corner_fail_cell = AleMinQualityCell{};
  }
  if (first_corner != nullptr) {
    *first_corner = -1;
  }
  if (min_corner_j != nullptr) {
    *min_corner_j = 1.0;
  }
  if (n_cells <= 0) {
    return false;
  }
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "ALE corner-J check requires matching node fields");
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == n_cells,
                  "ALE corner-J check requires hydro_active size == n_cells");
  }

  double* d_min_corner_j = nullptr;
  int* d_first_corner = nullptr;
  std::int8_t* d_active = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  d_min_corner_j = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:corner_post:d_min_corner_j",
                                   static_cast<std::size_t>(n_cells) *
                                       sizeof(double)));
  d_first_corner = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:corner_post:d_first_corner",
                                   static_cast<std::size_t>(n_cells) *
                                       sizeof(int)));
  if (!state.hydro_active.empty()) {
    d_active = static_cast<std::int8_t*>(
        core::device_scratch_acquire("ale_rezone:corner_post:d_active",
                                     state.hydro_active.size() * sizeof(std::int8_t)));
    CUDA_CHECK(cudaMemcpy(d_active,
                          state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice));
  }
  const bool use_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells) &&
      std::any_of(state.mesh.cell_nverts.begin(),
                  state.mesh.cell_nverts.end(),
                  [](const std::uint8_t nverts) { return nverts != 4U; });
  if (use_cell_nverts) {
    d_cell_nverts = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("ale_rezone:corner_post:d_cell_nverts",
                                     static_cast<std::size_t>(n_cells) *
                                         sizeof(std::uint8_t)));
    CUDA_CHECK(cudaMemcpy(d_cell_nverts,
                          state.mesh.cell_nverts.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice));
  }

  const int blocks = (n_cells + 255) / 256;
  detail::corner_post_tangle_kernel<<<blocks, 256>>>(d_min_corner_j,
                                                     d_first_corner,
                                                     state.x_r.data(),
                                                     state.x_z.data(),
                                                     d_active,
                                                     d_cell_nverts,
                                                     nr,
                                                     nz,
                                                     strict_floor_enabled,
                                                     strict_floor_eps);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<double> min_corner_host(static_cast<std::size_t>(n_cells), 1.0);
  std::vector<int> first_corner_host(static_cast<std::size_t>(n_cells), -1);
  CUDA_CHECK(cudaMemcpy(min_corner_host.data(),
                        d_min_corner_j,
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(first_corner_host.data(),
                        d_first_corner,
                        static_cast<std::size_t>(n_cells) * sizeof(int),
                        cudaMemcpyDeviceToHost));

  bool tangled = false;
  double local_min_corner_j = 1.0e300;
  AleMinQualityCell local_first_fail;
  int local_first_corner = -1;
  for (int c = 0; c < n_cells; ++c) {
    const double cell_min = min_corner_host[static_cast<std::size_t>(c)];
    if (cell_min < local_min_corner_j) {
      local_min_corner_j = cell_min;
    }
    const int cell_corner = first_corner_host[static_cast<std::size_t>(c)];
    if (cell_corner >= 0 && !tangled) {
      tangled = true;
      local_first_fail.c = c;
      local_first_fail.i = c / nz;
      local_first_fail.j = c - local_first_fail.i * nz;
      local_first_corner = cell_corner;
    }
  }
  if (local_min_corner_j == 1.0e300) {
    local_min_corner_j = 1.0;
  }
  if (first_corner_fail_cell != nullptr) {
    *first_corner_fail_cell = local_first_fail;
  }
  if (first_corner != nullptr) {
    *first_corner = local_first_corner;
  }
  if (min_corner_j != nullptr) {
    *min_corner_j = local_min_corner_j;
  }
  return tangled;
}

inline bool compute_corner_post_tangle(core::State& state,
                                       const core::Config& cfg,
                                       AleMinQualityCell* first_corner_fail_cell,
                                       double* min_corner_j = nullptr,
                                       int* first_corner = nullptr) {
  return compute_corner_post_tangle(
      state,
      first_corner_fail_cell,
      min_corner_j,
      first_corner,
      cfg.numerics.ale.corner_post_tangle_strict_floor_enabled,
      cfg.numerics.hydro.corner_jacobian_floor_eps);
}

inline double compute_active_gauss_quality_host(core::State& state,
                                                bool* mesh_tangle = nullptr,
                                                AleMinQualityCell* first_fail_cell = nullptr) {
  TENRYU_ASSERT(state.mesh.dim == 2, "ALE active Gauss check requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  if (mesh_tangle != nullptr) {
    *mesh_tangle = false;
  }
  if (first_fail_cell != nullptr) {
    *first_fail_cell = AleMinQualityCell{};
  }
  if (n_cells <= 0) {
    return 1.0;
  }
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == n_cells,
                  "ALE active Gauss check requires hydro_active size == n_cells");
  }

  std::vector<double> r_host;
  std::vector<double> z_host;
  state.x_r.copy_to_host(r_host);
  state.x_z.copy_to_host(z_host);
  TENRYU_ASSERT(r_host.size() == z_host.size(),
                "ALE active Gauss check requires matching node fields");
  TENRYU_ASSERT(static_cast<int>(r_host.size()) >= (nr + 1) * (nz + 1),
                "ALE active Gauss check node field size is smaller than mesh node count");

  bool tangled = false;
  double q_min = 1.0;
  const double xis[2] = {-detail::kGauss, detail::kGauss};
  const double etas[2] = {-detail::kGauss, detail::kGauss};
  const int stride = nz + 1;
  for (int c = 0; c < n_cells; ++c) {
    if (!state.hydro_active.empty() &&
        state.hydro_active[static_cast<std::size_t>(c)] == 0) {
      continue;
    }
    const int i = c / nz;
    const int j = c - i * nz;
    const int n00 = i * stride + j;
    const int n10 = (i + 1) * stride + j;
    const int n11 = (i + 1) * stride + (j + 1);
    const int n01 = i * stride + (j + 1);

    const double r1 = r_host[static_cast<std::size_t>(n00)];
    const double z1 = z_host[static_cast<std::size_t>(n00)];
    const double r2 = r_host[static_cast<std::size_t>(n10)];
    const double z2 = z_host[static_cast<std::size_t>(n10)];
    const double r3 = r_host[static_cast<std::size_t>(n11)];
    const double z3 = z_host[static_cast<std::size_t>(n11)];
    const double r4 = r_host[static_cast<std::size_t>(n01)];
    const double z4 = z_host[static_cast<std::size_t>(n01)];

    double J_min = 1.0e300;
    double J_max = -1.0e300;
    bool cell_tangled = false;
    for (double xi : xis) {
      for (double eta : etas) {
        const double dN1_dxi = -0.25 * (1.0 - eta);
        const double dN2_dxi = 0.25 * (1.0 - eta);
        const double dN3_dxi = 0.25 * (1.0 + eta);
        const double dN4_dxi = -0.25 * (1.0 + eta);

        const double dN1_deta = -0.25 * (1.0 - xi);
        const double dN2_deta = -0.25 * (1.0 + xi);
        const double dN3_deta = 0.25 * (1.0 + xi);
        const double dN4_deta = 0.25 * (1.0 - xi);

        const double dr_dxi =
            dN1_dxi * r1 + dN2_dxi * r2 + dN3_dxi * r3 + dN4_dxi * r4;
        const double dz_dxi =
            dN1_dxi * z1 + dN2_dxi * z2 + dN3_dxi * z3 + dN4_dxi * z4;
        const double dr_deta =
            dN1_deta * r1 + dN2_deta * r2 + dN3_deta * r3 + dN4_deta * r4;
        const double dz_deta =
            dN1_deta * z1 + dN2_deta * z2 + dN3_deta * z3 + dN4_deta * z4;

        const double J = dr_dxi * dz_deta - dr_deta * dz_dxi;
        if (!std::isfinite(J) || J < 0.0) {
          cell_tangled = true;
        }
        J_min = std::min(J_min, J);
        J_max = std::max(J_max, J);
      }
    }
    const double q_cell =
        (!std::isfinite(J_min) || !std::isfinite(J_max) || J_max < detail::kJFloor)
            ? 0.0
            : J_min / std::max(J_max, detail::kJFloor);
    if (q_cell < q_min) {
      q_min = q_cell;
    }
    if ((cell_tangled || q_cell <= 0.0) && !tangled) {
      tangled = true;
      if (first_fail_cell != nullptr) {
        first_fail_cell->c = c;
        first_fail_cell->i = i;
        first_fail_cell->j = j;
      }
    }
  }

  if (mesh_tangle != nullptr) {
    *mesh_tangle = tangled;
  }
  return q_min;
}

inline const char* classify_lambda_sweep(const LambdaSweepResult& result) {
  if (result.points.empty()) {
    return "no_admissible";
  }
  if (!result.points.front().admissible) {
    return "lambda0_invalid";
  }
  bool small_lambda_admissible = false;
  for (const LambdaSweepPoint& point : result.points) {
    if (!point.admissible || !(point.lambda > 0.0)) {
      continue;
    }
    if (point.lambda >= 0.03125) {
      return "standard_lambda_admissible";
    }
    small_lambda_admissible = true;
  }
  return small_lambda_admissible ? "small_lambda_admissible" : "no_admissible";
}

inline LambdaSweepResult evaluate_lambda_sweep(const core::State& state,
                                               const double* d_x_r_old,
                                               const double* d_x_z_old,
                                               const double* d_x_r_cand,
                                               const double* d_x_z_cand,
                                               const int target_cell_c,
                                               const int max_exp) {
  TENRYU_ASSERT(state.mesh.dim == 2, "ALE lambda sweep requires 2D mesh");
  TENRYU_ASSERT(d_x_r_old != nullptr && d_x_z_old != nullptr,
                "ALE lambda sweep requires old mesh coordinates");
  TENRYU_ASSERT(d_x_r_cand != nullptr && d_x_z_cand != nullptr,
                "ALE lambda sweep requires candidate mesh coordinates");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int max_exp_clamped = std::clamp(max_exp, 0, 1022);

  LambdaSweepResult result;
  result.target_cell_c = target_cell_c;
  result.max_exp = max_exp_clamped;
  if (nz > 0 && target_cell_c >= 0 && target_cell_c < n_cells) {
    result.target_cell_i = target_cell_c / nz;
    result.target_cell_j = target_cell_c - result.target_cell_i * nz;
  }
  if (nr <= 0 || nz <= 0 || n_nodes <= 0 ||
      target_cell_c < 0 || target_cell_c >= n_cells) {
    return result;
  }
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == n_cells,
                  "ALE lambda sweep requires hydro_active size == n_cells");
  }

  std::vector<int> exps;
  exps.reserve(11);
  exps.push_back(max_exp_clamped);
  const int fixed_exps[] = {20, 15, 10, 7, 5, 4, 3, 2, 1, 0};
  for (const int e : fixed_exps) {
    if (e <= max_exp_clamped) {
      exps.push_back(e);
    }
  }
  std::sort(exps.begin(), exps.end(), std::greater<int>());
  exps.erase(std::unique(exps.begin(), exps.end()), exps.end());

  std::vector<double> lambdas;
  lambdas.reserve(exps.size() + 1);
  lambdas.push_back(0.0);
  for (const int e : exps) {
    lambdas.push_back((e == 0) ? 1.0 : std::ldexp(1.0, -e));
  }

  std::vector<double> r_old(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> z_old(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> r_cand(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> z_cand(static_cast<std::size_t>(n_nodes), 0.0);
  CUDA_CHECK(cudaMemcpy(r_old.data(),
                        d_x_r_old,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(z_old.data(),
                        d_x_z_old,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r_cand.data(),
                        d_x_r_cand,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(z_cand.data(),
                        d_x_z_cand,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost));

  const int stride = nz + 1;
  const int i0 = std::max(0, result.target_cell_i - 1);
  const int i1 = std::min(nr - 1, result.target_cell_i + 1);
  const int j0 = std::max(0, result.target_cell_j - 1);
  const int j1 = std::min(nz - 1, result.target_cell_j + 1);
  const double xis[2] = {-detail::kGauss, detail::kGauss};
  const double etas[2] = {-detail::kGauss, detail::kGauss};

  result.points.reserve(lambdas.size());
  for (const double lambda : lambdas) {
    LambdaSweepPoint point;
    point.lambda = lambda;

    for (int i = i0; i <= i1; ++i) {
      for (int j = j0; j <= j1; ++j) {
        const int c = i * nz + j;
        if (!state.hydro_active.empty() &&
            state.hydro_active[static_cast<std::size_t>(c)] == 0) {
          continue;
        }
        const int active_nverts = detail::state_cell_active_nverts(state, c, n_cells);

        const int nodes[4] = {
            i * stride + j,
            (i + 1) * stride + j,
            (i + 1) * stride + (j + 1),
            i * stride + (j + 1),
        };
        double rr[4];
        double zz[4];
        for (int k = 0; k < active_nverts; ++k) {
          const std::size_t n = static_cast<std::size_t>(nodes[k]);
          rr[k] = r_old[n] + lambda * (r_cand[n] - r_old[n]);
          zz[k] = z_old[n] + lambda * (z_cand[n] - z_old[n]);
        }

        if (active_nverts == 3) {
          double corner_j[4]{};
          detail::tri_corner_jacobians_canonical(rr, zz, corner_j);
          for (int corner = 0; corner < 3; ++corner) {
            point.min_corner_j = std::min(point.min_corner_j, corner_j[corner]);
            point.min_gauss_j = std::min(point.min_gauss_j, corner_j[corner]);
          }
          point.min_v_rz =
              std::min(point.min_v_rz,
                       -tenryu::mesh::rz_triangle_volume_exact(rr[0],
                                                               zz[0],
                                                               rr[1],
                                                               zz[1],
                                                               rr[2],
                                                               zz[2]));
        } else {
          for (const double xi : xis) {
            for (const double eta : etas) {
              const double dN1_dxi = -0.25 * (1.0 - eta);
              const double dN2_dxi = 0.25 * (1.0 - eta);
              const double dN3_dxi = 0.25 * (1.0 + eta);
              const double dN4_dxi = -0.25 * (1.0 + eta);

              const double dN1_deta = -0.25 * (1.0 - xi);
              const double dN2_deta = -0.25 * (1.0 + xi);
              const double dN3_deta = 0.25 * (1.0 + xi);
              const double dN4_deta = 0.25 * (1.0 - xi);

              const double dr_dxi =
                  dN1_dxi * rr[0] + dN2_dxi * rr[1] + dN3_dxi * rr[2] +
                  dN4_dxi * rr[3];
              const double dz_dxi =
                  dN1_dxi * zz[0] + dN2_dxi * zz[1] + dN3_dxi * zz[2] +
                  dN4_dxi * zz[3];
              const double dr_deta =
                  dN1_deta * rr[0] + dN2_deta * rr[1] + dN3_deta * rr[2] +
                  dN4_deta * rr[3];
              const double dz_deta =
                  dN1_deta * zz[0] + dN2_deta * zz[1] + dN3_deta * zz[2] +
                  dN4_deta * zz[3];

              const double J = dr_dxi * dz_deta - dr_deta * dz_dxi;
              point.min_gauss_j = std::min(point.min_gauss_j, J);
            }
          }

          for (int corner = 0; corner < 4; ++corner) {
            point.min_corner_j =
                std::min(point.min_corner_j,
                         tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner));
          }
          point.min_v_rz =
              std::min(point.min_v_rz,
                       detail::rz_signed_quad_volume(rr[0],
                                                     zz[0],
                                                     rr[1],
                                                     zz[1],
                                                     rr[2],
                                                     zz[2],
                                                     rr[3],
                                                     zz[3]));
        }
      }
    }

    point.admissible = std::isfinite(point.min_gauss_j) &&
                       std::isfinite(point.min_corner_j) &&
                       std::isfinite(point.min_v_rz) &&
                       point.min_gauss_j > 0.0 &&
                       point.min_corner_j > 0.0 &&
                       point.min_v_rz > 0.0;
    result.points.push_back(point);
  }

  return result;
}

inline LocalBoundaryRepairResult try_multi_node_boundary_repair(
    core::State& state,
    const core::Config& cfg,
    const AleMinQualityCell& failing_cell,
    const int failing_corner) {
  struct MoveNode {
    int id = -1;
    int i = -1;
    int j = -1;
    bool move_r = false;
    bool move_z = false;
    int r_dof = -1;
    int z_dof = -1;
  };
  struct LinearConstraint {
    std::vector<double> a;
    double b = 0.0;
    int cell = -1;
    int corner = -1;
  };

  LocalBoundaryRepairResult best;
  best.fired = true;
  best.failing_cell = failing_cell;
  best.failing_corner = failing_corner;
  best.reason = "multi-node repair not attempted";

  TENRYU_ASSERT(state.mesh.dim == 2, "multi-node boundary repair requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 1 || nz <= 1 || failing_corner < 0 || failing_corner > 3) {
    best.reason = "multi-node invalid failing cell/corner";
    return best;
  }

  int cell_i = failing_cell.i;
  int cell_j = failing_cell.j;
  if ((cell_i < 0 || cell_j < 0) && failing_cell.c >= 0) {
    cell_i = failing_cell.c / nz;
    cell_j = failing_cell.c - cell_i * nz;
  }
  if (cell_i < 0 || cell_i >= nr || cell_j < 0 || cell_j >= nz) {
    best.reason = "multi-node failing cell outside mesh";
    return best;
  }

  const int apex_i = cell_i + ((failing_corner == 1 || failing_corner == 2) ? 1 : 0);
  const int apex_j = cell_j + ((failing_corner == 2 || failing_corner == 3) ? 1 : 0);
  if (apex_i <= 0 || apex_i > nr || apex_j < 0 || apex_j > nz) {
    best.reason = "multi-node failing corner outside supported boundary";
    return best;
  }

  std::vector<double> r_original;
  std::vector<double> z_original;
  state.x_r.copy_to_host(r_original);
  state.x_z.copy_to_host(z_original);
  TENRYU_ASSERT(r_original.size() == z_original.size(),
                "multi-node repair requires matching node field sizes");
  TENRYU_ASSERT(static_cast<int>(r_original.size()) >= (nr + 1) * (nz + 1),
                "multi-node repair node field size is smaller than mesh node count");

  const auto node_index = [&](const int i, const int j) {
    return tenryu::hydro::rz_node_index_2d(i, j, nz);
  };
  const int structured_n_cells = nr * nz;
  const auto active_nverts_for_cell = [&](const int c) {
    return detail::state_cell_active_nverts(state, c, structured_n_cells);
  };
  const auto cell_active = [&](const int c) {
    if (state.hydro_active.empty()) {
      return true;
    }
    TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == nr * nz,
                  "multi-node repair requires hydro_active size == n_cells");
    return state.hydro_active[static_cast<std::size_t>(c)] != 0;
  };

  const bool z_boundary_mode = (apex_j == 0 || apex_j == nz) &&
                               apex_i > 0 && apex_i < nr;
  const bool r_boundary_mode = (apex_i == nr) && apex_j > 0 && apex_j < nz;
  if (!z_boundary_mode && !r_boundary_mode) {
    best.reason = "multi-node repair supports non-corner z/r boundary nodes only";
    return best;
  }

  const double domain_span =
      std::max(std::fabs(cfg.mesh.r_max - cfg.mesh.r_min),
               std::fabs(cfg.mesh.z_max - cfg.mesh.z_min));
  const double coord_scale = std::max(1.0, domain_span);
  const double coord_eps =
      64.0 * std::numeric_limits<double>::epsilon() * coord_scale;
  const int max_ring_level = 1;

  for (int ring_level = 0; ring_level <= max_ring_level; ++ring_level) {
    std::vector<MoveNode> moves;
    const auto add_move = [&](const int i,
                              const int j,
                              const bool move_r,
                              const bool move_z) {
      if (i < 0 || i > nr || j < 0 || j > nz || (!move_r && !move_z)) {
        return;
      }
      if (i == 0) {
        return;
      }
      if (i == nr && (j == 0 || j == nz)) {
        return;
      }
      const int id = node_index(i, j);
      for (MoveNode& move : moves) {
        if (move.id == id) {
          move.move_r = move.move_r || move_r;
          move.move_z = move.move_z || move_z;
          return;
        }
      }
      moves.push_back(MoveNode{id, i, j, move_r, move_z, -1, -1});
    };

    if (z_boundary_mode) {
      const int interior_j = (apex_j == nz) ? apex_j - 1 : apex_j + 1;
      add_move(apex_i, apex_j, true, false);
      add_move(apex_i - 1, apex_j, true, false);
      add_move(apex_i + 1, apex_j, true, false);
      if (interior_j > 0 && interior_j < nz) {
        add_move(apex_i, interior_j, true, true);
        if (ring_level >= 1) {
          add_move(apex_i - 1, interior_j, true, true);
          add_move(apex_i + 1, interior_j, true, true);
        }
      }
    } else {
      const int interior_i = apex_i - 1;
      add_move(apex_i, apex_j, false, true);
      add_move(apex_i, apex_j - 1, false, true);
      add_move(apex_i, apex_j + 1, false, true);
      if (interior_i > 0 && interior_i < nr) {
        add_move(interior_i, apex_j, true, true);
        if (ring_level >= 1) {
          add_move(interior_i, apex_j - 1, true, true);
          add_move(interior_i, apex_j + 1, true, true);
        }
      }
    }

    int ndof = 0;
    for (MoveNode& move : moves) {
      if (move.move_r) {
        move.r_dof = ndof++;
      }
      if (move.move_z) {
        move.z_dof = ndof++;
      }
    }
    if (ndof <= 0 || ndof > 9) {
      best.reason = "multi-node ring has unsupported dof count";
      continue;
    }

    std::vector<int> affected_cells;
    const auto add_affected_cell = [&](const int ci, const int cj) {
      if (ci < 0 || ci >= nr || cj < 0 || cj >= nz) {
        return;
      }
      const int c = ci * nz + cj;
      if (!cell_active(c)) {
        return;
      }
      if (std::find(affected_cells.begin(), affected_cells.end(), c) ==
          affected_cells.end()) {
        affected_cells.push_back(c);
      }
    };
    for (const MoveNode& move : moves) {
      for (int ci = move.i - 1; ci <= move.i; ++ci) {
        for (int cj = move.j - 1; cj <= move.j; ++cj) {
          add_affected_cell(ci, cj);
        }
      }
    }
    if (affected_cells.empty()) {
      best.reason = "multi-node ring has no affected cells";
      continue;
    }

    std::vector<double> x(static_cast<std::size_t>(ndof), 0.0);
    for (const MoveNode& move : moves) {
      if (move.r_dof >= 0) {
        x[static_cast<std::size_t>(move.r_dof)] =
            r_original[static_cast<std::size_t>(move.id)];
      }
      if (move.z_dof >= 0) {
        x[static_cast<std::size_t>(move.z_dof)] =
            z_original[static_cast<std::size_t>(move.id)];
      }
    }

    const auto find_dof = [&](const int id, const bool is_r) {
      for (const MoveNode& move : moves) {
        if (move.id == id) {
          return is_r ? move.r_dof : move.z_dof;
        }
      }
      return -1;
    };
    const auto coord_value = [&](const int id,
                                 const bool is_r,
                                 const std::vector<double>& x_values) {
      const int dof = find_dof(id, is_r);
      if (dof >= 0) {
        return x_values[static_cast<std::size_t>(dof)];
      }
      return is_r ? r_original[static_cast<std::size_t>(id)]
                  : z_original[static_cast<std::size_t>(id)];
    };
    const auto eval_corner_j = [&](const int ci,
                                   const int cj,
                                   const int corner,
                                   const std::vector<double>& x_values) {
      const int c = ci * nz + cj;
      const int active_nverts = active_nverts_for_cell(c);
      if (corner < 0 || corner >= active_nverts) {
        return std::numeric_limits<double>::quiet_NaN();
      }
      const int nodes[4] = {node_index(ci, cj),
                            node_index(ci + 1, cj),
                            node_index(ci + 1, cj + 1),
                            node_index(ci, cj + 1)};
      double rr[4];
      double zz[4];
      for (int k = 0; k < active_nverts; ++k) {
        rr[k] = coord_value(nodes[k], true, x_values);
        zz[k] = coord_value(nodes[k], false, x_values);
      }
      if (active_nverts == 3) {
        double corner_j[4]{};
        detail::tri_corner_jacobians_canonical(rr, zz, corner_j);
        return corner_j[corner];
      }
      return tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
    };
    const auto compute_max_abs_corner_j = [&](const std::vector<double>& x_ref) {
      double max_abs_j = 0.0;
      for (const int c : affected_cells) {
        const int ci = c / nz;
        const int cj = c - ci * nz;
        const int active_nverts = active_nverts_for_cell(c);
        for (int corner = 0; corner < active_nverts; ++corner) {
          const double j = eval_corner_j(ci, cj, corner, x_ref);
          if (std::isfinite(j)) {
            max_abs_j = std::max(max_abs_j, std::fabs(j));
          }
        }
      }
      return max_abs_j;
    };
    const double repair_max_abs_j = compute_max_abs_corner_j(x);
    const auto compute_repair_j_floor = [&]() {
      const double floor_eps =
          std::max(cfg.numerics.hydro.corner_jacobian_floor_eps, 0.0);
      return std::max(detail::kJFloor, floor_eps * repair_max_abs_j);
    };
    const double repair_j_floor = compute_repair_j_floor();
    const double repair_j_clearance =
        std::max(16.0 * detail::kJFloor,
                 1024.0 * std::numeric_limits<double>::epsilon() *
                     std::max(repair_max_abs_j, repair_j_floor));
    const double solver_j_floor = repair_j_floor + repair_j_clearance;
    const auto add_difference_constraint =
        [&](std::vector<LinearConstraint>& constraints,
            const int positive_node,
            const int negative_node,
            const bool is_r,
            const double min_spacing,
            const int cell,
            const int corner) {
          LinearConstraint con;
          con.a.assign(static_cast<std::size_t>(ndof), 0.0);
          con.cell = cell;
          con.corner = corner;
          double fixed = 0.0;
          const int pos_dof = find_dof(positive_node, is_r);
          if (pos_dof >= 0) {
            con.a[static_cast<std::size_t>(pos_dof)] += 1.0;
          } else {
            fixed += is_r ? r_original[static_cast<std::size_t>(positive_node)]
                          : z_original[static_cast<std::size_t>(positive_node)];
          }
          const int neg_dof = find_dof(negative_node, is_r);
          if (neg_dof >= 0) {
            con.a[static_cast<std::size_t>(neg_dof)] -= 1.0;
          } else {
            fixed -= is_r ? r_original[static_cast<std::size_t>(negative_node)]
                          : z_original[static_cast<std::size_t>(negative_node)];
          }
          con.b = min_spacing - fixed;
          constraints.push_back(con);
        };
    const auto build_constraints =
        [&](const std::vector<double>& x_ref,
            std::vector<LinearConstraint>& constraints,
            int* impossible_constraint) {
          constraints.clear();
          if (impossible_constraint != nullptr) {
            *impossible_constraint = -1;
          }
          for (const int c : affected_cells) {
            const int ci = c / nz;
            const int cj = c - ci * nz;
            const int active_nverts = active_nverts_for_cell(c);
            for (int corner = 0; corner < active_nverts; ++corner) {
              const double j0 = eval_corner_j(ci, cj, corner, x_ref);
              if (!std::isfinite(j0)) {
                if (impossible_constraint != nullptr) {
                  *impossible_constraint = static_cast<int>(constraints.size());
                }
                return false;
              }
              LinearConstraint con;
              con.a.assign(static_cast<std::size_t>(ndof), 0.0);
              con.cell = c;
              con.corner = corner;
              double dot_grad_x = 0.0;
              double norm2 = 0.0;
              for (int d = 0; d < ndof; ++d) {
                std::vector<double> x_plus = x_ref;
                const double h =
                    std::max(1.0e-8 * coord_scale,
                             64.0 * std::numeric_limits<double>::epsilon() *
                                 std::max(coord_scale,
                                          std::fabs(x_ref[static_cast<std::size_t>(d)])));
                x_plus[static_cast<std::size_t>(d)] += h;
                const double jp = eval_corner_j(ci, cj, corner, x_plus);
                if (!std::isfinite(jp)) {
                  if (impossible_constraint != nullptr) {
                    *impossible_constraint = static_cast<int>(constraints.size());
                  }
                  return false;
                }
                const double grad = (jp - j0) / h;
                con.a[static_cast<std::size_t>(d)] = grad;
                dot_grad_x += grad * x_ref[static_cast<std::size_t>(d)];
                norm2 += grad * grad;
              }
              if (norm2 <= std::numeric_limits<double>::min()) {
                if (j0 < repair_j_floor) {
                  if (impossible_constraint != nullptr) {
                    *impossible_constraint = static_cast<int>(constraints.size());
                  }
                  return false;
                }
                continue;
              }
              con.b = solver_j_floor - j0 + dot_grad_x;
              constraints.push_back(con);
            }
          }

          for (const MoveNode& move : moves) {
            if (move.move_r) {
              if (move.i > 0) {
                add_difference_constraint(constraints,
                                          move.id,
                                          node_index(move.i - 1, move.j),
                                          true,
                                          coord_eps,
                                          -1,
                                          -1);
              }
              if (move.i < nr) {
                add_difference_constraint(constraints,
                                          node_index(move.i + 1, move.j),
                                          move.id,
                                          true,
                                          coord_eps,
                                          -1,
                                          -1);
              }
            }
            if (move.move_z) {
              if (move.j > 0) {
                add_difference_constraint(constraints,
                                          move.id,
                                          node_index(move.i, move.j - 1),
                                          false,
                                          coord_eps,
                                          -1,
                                          -1);
              }
              if (move.j < nz) {
                add_difference_constraint(constraints,
                                          node_index(move.i, move.j + 1),
                                          move.id,
                                          false,
                                          coord_eps,
                                          -1,
                                          -1);
              }
            }
          }
          return true;
        };

    const auto solve_linear_feasibility =
        [&](const std::vector<LinearConstraint>& constraints,
            std::vector<double>& x_values,
            int* iterations,
            int* infeasible_constraint) {
          constexpr int kMaxIterations = 600;
          const double slack_tol = 1.0e-24;
          if (iterations != nullptr) {
            *iterations = 0;
          }
          if (infeasible_constraint != nullptr) {
            *infeasible_constraint = -1;
          }
          for (int iter = 0; iter < kMaxIterations; ++iter) {
            double worst_slack = std::numeric_limits<double>::infinity();
            int worst = -1;
            for (int cidx = 0; cidx < static_cast<int>(constraints.size()); ++cidx) {
              const LinearConstraint& con =
                  constraints[static_cast<std::size_t>(cidx)];
              double lhs = 0.0;
              double norm2 = 0.0;
              for (int d = 0; d < ndof; ++d) {
                const double a = con.a[static_cast<std::size_t>(d)];
                lhs += a * x_values[static_cast<std::size_t>(d)];
                norm2 += a * a;
              }
              const double slack = lhs - con.b;
              if (slack < worst_slack) {
                worst_slack = slack;
                worst = cidx;
              }
              if (slack < -slack_tol && norm2 <= std::numeric_limits<double>::min()) {
                if (iterations != nullptr) {
                  *iterations += iter + 1;
                }
                if (infeasible_constraint != nullptr) {
                  *infeasible_constraint = cidx;
                }
                return false;
              }
            }
            if (worst < 0 || worst_slack >= -slack_tol) {
              if (iterations != nullptr) {
                *iterations += iter + 1;
              }
              return true;
            }
            const LinearConstraint& con = constraints[static_cast<std::size_t>(worst)];
            double lhs = 0.0;
            double norm2 = 0.0;
            for (int d = 0; d < ndof; ++d) {
              const double a = con.a[static_cast<std::size_t>(d)];
              lhs += a * x_values[static_cast<std::size_t>(d)];
              norm2 += a * a;
            }
            if (norm2 <= std::numeric_limits<double>::min()) {
              if (iterations != nullptr) {
                *iterations += iter + 1;
              }
              if (infeasible_constraint != nullptr) {
                *infeasible_constraint = worst;
              }
              return false;
            }
            const double alpha = (con.b - lhs) / norm2;
            for (int d = 0; d < ndof; ++d) {
              x_values[static_cast<std::size_t>(d)] +=
                  alpha * con.a[static_cast<std::size_t>(d)];
            }
          }
          if (iterations != nullptr) {
            *iterations += kMaxIterations;
          }
          if (infeasible_constraint != nullptr) {
            *infeasible_constraint = -1;
          }
          return false;
        };
    const auto exact_affected_cells_valid = [&](const std::vector<double>& x_values,
                                                double* min_j) {
      double local_min = std::numeric_limits<double>::infinity();
      for (const int c : affected_cells) {
        const int ci = c / nz;
        const int cj = c - ci * nz;
        const int active_nverts = active_nverts_for_cell(c);
        for (int corner = 0; corner < active_nverts; ++corner) {
          const double j = eval_corner_j(ci, cj, corner, x_values);
          if (!std::isfinite(j) || j <= repair_j_floor) {
            if (min_j != nullptr) {
              *min_j = std::isfinite(j) ? j : -1.0;
            }
            return false;
          }
          local_min = std::min(local_min, j);
        }
      }
      if (min_j != nullptr) {
        *min_j = (local_min == std::numeric_limits<double>::infinity()) ? 1.0
                                                                        : local_min;
      }
      return true;
    };
    const auto exact_ordering_valid = [&](const std::vector<double>& x_values) {
      for (const MoveNode& move : moves) {
        if (move.move_r) {
          const double r_value = coord_value(move.id, true, x_values);
          if (!std::isfinite(r_value)) {
            return false;
          }
          if (move.i > 0) {
            const double r_left =
                coord_value(node_index(move.i - 1, move.j), true, x_values);
            if (!std::isfinite(r_left) || !(r_value - r_left > coord_eps)) {
              return false;
            }
          }
          if (move.i < nr) {
            const double r_right =
                coord_value(node_index(move.i + 1, move.j), true, x_values);
            if (!std::isfinite(r_right) || !(r_right - r_value > coord_eps)) {
              return false;
            }
          }
        }
        if (move.move_z) {
          const double z_value = coord_value(move.id, false, x_values);
          if (!std::isfinite(z_value)) {
            return false;
          }
          if (move.j > 0) {
            const double z_lower =
                coord_value(node_index(move.i, move.j - 1), false, x_values);
            if (!std::isfinite(z_lower) || !(z_value - z_lower > coord_eps)) {
              return false;
            }
          }
          if (move.j < nz) {
            const double z_upper =
                coord_value(node_index(move.i, move.j + 1), false, x_values);
            if (!std::isfinite(z_upper) || !(z_upper - z_value > coord_eps)) {
              return false;
            }
          }
        }
      }
      return true;
    };
    const auto exact_candidate_valid = [&](const std::vector<double>& x_values,
                                           double* min_j) {
      return exact_affected_cells_valid(x_values, min_j) &&
             exact_ordering_valid(x_values);
    };

    int total_iterations = 0;
    int infeasible_constraint = -1;
    bool solved = false;
    std::vector<LinearConstraint> constraints;
    for (int picard = 0; picard < 3; ++picard) {
      if (!build_constraints(x, constraints, &infeasible_constraint)) {
        break;
      }
      int iterations = 0;
      const bool linear_solved = solve_linear_feasibility(constraints,
                                                          x,
                                                          &iterations,
                                                          &infeasible_constraint);
      total_iterations += iterations;
      double affected_min_j = 1.0;
      if (exact_candidate_valid(x, &affected_min_j)) {
        solved = true;
        best.min_corner_j = affected_min_j;
        break;
      }
      if (!linear_solved) {
        break;
      }
    }

    best.ring_size = static_cast<int>(moves.size());
    best.dof = ndof;
    best.constraint_count = static_cast<int>(constraints.size());
    best.affected_cell_count = static_cast<int>(affected_cells.size());
    best.solver_iterations = total_iterations;
    best.infeasible_constraint = infeasible_constraint;
    if (!solved) {
      best.reason = (ring_level == 0) ? "multi-node minimal ring infeasible"
                                      : "multi-node expanded ring infeasible";
      continue;
    }

    std::vector<double> r_candidate = r_original;
    std::vector<double> z_candidate = z_original;
    for (const MoveNode& move : moves) {
      if (move.r_dof >= 0) {
        r_candidate[static_cast<std::size_t>(move.id)] =
            x[static_cast<std::size_t>(move.r_dof)];
      }
      if (move.z_dof >= 0) {
        z_candidate[static_cast<std::size_t>(move.id)] =
            x[static_cast<std::size_t>(move.z_dof)];
      }
    }
    state.x_r.copy_from_host(r_candidate);
    state.x_z.copy_from_host(z_candidate);

    AleMinQualityCell corner_fail;
    int corner = -1;
    double min_corner_j = 1.0;
    const bool corner_tangle =
        compute_corner_post_tangle(state, cfg, &corner_fail, &min_corner_j, &corner);
    bool gauss_tangle = false;
    AleMinQualityCell gauss_fail;
    (void)compute_min_quality(state, cfg, &gauss_tangle, &gauss_fail);
    if (corner_tangle || gauss_tangle) {
      state.x_r.copy_from_host(r_original);
      state.x_z.copy_from_host(z_original);
      best.min_corner_j = min_corner_j;
      best.reason = corner_tangle ? "multi-node global corner-J check failed"
                                  : "multi-node global Gauss check failed";
      continue;
    }

    best.applied = true;
    best.repaired_node_count = static_cast<int>(moves.size());
    best.repaired_failing_cells = static_cast<int>(affected_cells.size());
    best.node_i = apex_i;
    best.node_j = apex_j;
    best.node_id = node_index(apex_i, apex_j);
    best.old_coord = z_boundary_mode
                         ? r_original[static_cast<std::size_t>(best.node_id)]
                         : z_original[static_cast<std::size_t>(best.node_id)];
    const int apex_dof = z_boundary_mode
                             ? find_dof(best.node_id, true)
                             : find_dof(best.node_id, false);
    best.new_coord =
        (apex_dof >= 0) ? x[static_cast<std::size_t>(apex_dof)] : best.old_coord;
    best.variable = z_boundary_mode ? "multi_rz" : "multi_zr";
    best.min_corner_j = min_corner_j;
    best.reason = "multi-node accepted";
    return best;
  }

  return best;
}

inline LocalBoundaryRepairResult try_local_boundary_repair(
    core::State& state,
    const core::Config& cfg,
    const AleMinQualityCell& failing_cell,
    const int failing_corner,
    const char* gate_reason = nullptr) {
  LocalBoundaryRepairResult result;
  result.fired = true;
  result.failing_cell = failing_cell;
  result.failing_corner = failing_corner;

  TENRYU_ASSERT(state.mesh.dim == 2, "local boundary repair requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0 || failing_corner < 0 || failing_corner > 3) {
    result.reason = "invalid failing cell/corner";
    return result;
  }

  int cell_i = failing_cell.i;
  int cell_j = failing_cell.j;
  if ((cell_i < 0 || cell_j < 0) && failing_cell.c >= 0) {
    cell_i = failing_cell.c / nz;
    cell_j = failing_cell.c - cell_i * nz;
  }
  if (cell_i < 0 || cell_i >= nr || cell_j < 0 || cell_j >= nz) {
    result.reason = "failing cell outside mesh";
    return result;
  }

  const int node_i = cell_i + ((failing_corner == 1 || failing_corner == 2) ? 1 : 0);
  const int node_j = cell_j + ((failing_corner == 2 || failing_corner == 3) ? 1 : 0);
  if (node_i < 0 || node_i > nr || node_j < 0 || node_j > nz) {
    result.reason = "failing corner outside mesh";
    return result;
  }
  const int node_id = tenryu::hydro::rz_node_index_2d(node_i, node_j, nz);
  result.node_i = node_i;
  result.node_j = node_j;
  result.node_id = node_id;

  const bool on_axis = cfg.numerics.has_physical_rz_axis && (node_i == 0);
  const bool on_outer_r = (node_i == nr);
  const bool on_z_boundary = (node_j == 0 || node_j == nz);
  if (on_axis) {
    result.reason = "axis nodes use axis-specific repair";
    return result;
  }
  if (on_z_boundary && on_outer_r) {
    result.reason = "domain corner has no tangential repair degree of freedom";
    return result;
  }

  const bool move_r = on_z_boundary && node_i > 0 && node_i < nr;
  const bool move_z = on_outer_r && node_j > 0 && node_j < nz;
  if (!move_r && !move_z) {
    result.reason = "failing corner is not a supported boundary node";
    return result;
  }
  result.variable = move_r ? "r" : "z";

  std::vector<double> r_host;
  std::vector<double> z_host;
  state.x_r.copy_to_host(r_host);
  state.x_z.copy_to_host(z_host);
  TENRYU_ASSERT(static_cast<int>(r_host.size()) >= (nr + 1) * (nz + 1),
                "local boundary repair x_r size is smaller than mesh node count");
  TENRYU_ASSERT(r_host.size() == z_host.size(),
                "local boundary repair requires matching node field sizes");

  const auto cell_active = [&](const int c) {
    if (state.hydro_active.empty()) {
      return true;
    }
    TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == nr * nz,
                  "local boundary repair requires hydro_active size == n_cells");
    return state.hydro_active[static_cast<std::size_t>(c)] != 0;
  };
  const auto node_index = [&](const int i, const int j) {
    return tenryu::hydro::rz_node_index_2d(i, j, nz);
  };
  const int structured_n_cells = nr * nz;
  const auto active_nverts_for_cell = [&](const int c) {
    return detail::state_cell_active_nverts(state, c, structured_n_cells);
  };
  const auto corner_for_node = [&](const int ci, const int cj) {
    if (node_i == ci && node_j == cj) {
      return 0;
    }
    if (node_i == ci + 1 && node_j == cj) {
      return 1;
    }
    if (node_i == ci + 1 && node_j == cj + 1) {
      return 2;
    }
    if (node_i == ci && node_j == cj + 1) {
      return 3;
    }
    return -1;
  };
  struct CellCorner {
    int i = -1;
    int j = -1;
    int corner = -1;
  };
  std::vector<CellCorner> constraints;
  for (int ci = node_i - 1; ci <= node_i; ++ci) {
    if (ci < 0 || ci >= nr) {
      continue;
    }
    for (int cj = node_j - 1; cj <= node_j; ++cj) {
      if (cj < 0 || cj >= nz) {
        continue;
      }
      const int c = ci * nz + cj;
      if (!cell_active(c)) {
        continue;
      }
      const int corner = corner_for_node(ci, cj);
      const int active_nverts = active_nverts_for_cell(c);
      if (corner >= 0 && corner < active_nverts) {
        constraints.push_back(CellCorner{ci, cj, corner});
      }
    }
  }
  if (constraints.empty()) {
    result.reason = "no active adjacent cells";
    return result;
  }

  const double old_coord =
      move_r ? r_host[static_cast<std::size_t>(node_id)]
             : z_host[static_cast<std::size_t>(node_id)];
  result.old_coord = old_coord;
  const double domain_span =
      move_r ? std::fabs(cfg.mesh.r_max - cfg.mesh.r_min)
             : std::fabs(cfg.mesh.z_max - cfg.mesh.z_min);
  const double coord_scale =
      std::max(1.0, std::max(domain_span, std::fabs(old_coord)));
  const double coord_eps =
      64.0 * std::numeric_limits<double>::epsilon() * coord_scale;
  const double deriv_step = std::max(1.0e-8 * coord_scale, coord_eps);
  double lower = -std::numeric_limits<double>::infinity();
  double upper = std::numeric_limits<double>::infinity();

  if (move_r) {
    lower = std::max(
        lower, r_host[static_cast<std::size_t>(node_index(node_i - 1, node_j))] +
                   coord_eps);
    upper = std::min(
        upper, r_host[static_cast<std::size_t>(node_index(node_i + 1, node_j))] -
                   coord_eps);
  } else {
    lower = std::max(
        lower, z_host[static_cast<std::size_t>(node_index(node_i, node_j - 1))] +
                   coord_eps);
    upper = std::min(
        upper, z_host[static_cast<std::size_t>(node_index(node_i, node_j + 1))] -
                   coord_eps);
  }

  const auto eval_cell_corner_j = [&](const int ci,
                                      const int cj,
                                      const int corner,
                                      const double coord) {
    const int c = ci * nz + cj;
    const int active_nverts = active_nverts_for_cell(c);
    if (corner < 0 || corner >= active_nverts) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    const int n0 = node_index(ci, cj);
    const int n1 = node_index(ci + 1, cj);
    const int n2 = node_index(ci + 1, cj + 1);
    const int n3 = node_index(ci, cj + 1);
    const int nodes[4] = {n0, n1, n2, n3};
    double rr[4];
    double zz[4];
    for (int k = 0; k < active_nverts; ++k) {
      rr[k] = r_host[static_cast<std::size_t>(nodes[k])];
      zz[k] = z_host[static_cast<std::size_t>(nodes[k])];
      if (nodes[k] == node_id) {
        if (move_r) {
          rr[k] = coord;
        } else {
          zz[k] = coord;
        }
      }
    }
    if (active_nverts == 3) {
      double corner_j[4]{};
      detail::tri_corner_jacobians_canonical(rr, zz, corner_j);
      return corner_j[corner];
    }
    return tenryu::hydro::corner_jacobian_from_quad(rr, zz, corner);
  };
  const auto eval_corner_j = [&](const CellCorner& cell, const double coord) {
    return eval_cell_corner_j(cell.i, cell.j, cell.corner, coord);
  };
  const auto compute_local_repair_j_floor = [&]() {
    double max_abs_j = 0.0;
    for (const CellCorner& cell : constraints) {
      const int c = cell.i * nz + cell.j;
      const int active_nverts = active_nverts_for_cell(c);
      for (int corner = 0; corner < active_nverts; ++corner) {
        const double j = eval_cell_corner_j(cell.i, cell.j, corner, old_coord);
        if (std::isfinite(j)) {
          max_abs_j = std::max(max_abs_j, std::fabs(j));
        }
      }
    }
    const double floor_eps =
        std::max(cfg.numerics.hydro.corner_jacobian_floor_eps, 0.0);
    return std::max(detail::kJFloor, floor_eps * max_abs_j);
  };
  const double repair_j_floor = compute_local_repair_j_floor();

  for (const CellCorner& cell : constraints) {
    const double j0 = eval_corner_j(cell, old_coord);
    const double j1 = eval_corner_j(cell, old_coord + deriv_step);
    if (!std::isfinite(j0) || !std::isfinite(j1)) {
      result.reason = "non-finite local corner-J constraint";
      return result;
    }
    const double slope = (j1 - j0) / deriv_step;
    if (std::fabs(slope) <= std::numeric_limits<double>::min()) {
      if (j0 <= repair_j_floor) {
        result.reason = "local corner-J constraint has no movable degree";
        return result;
      }
      continue;
    }
    const double bound = old_coord + (repair_j_floor - j0) / slope;
    if (!std::isfinite(bound)) {
      result.reason = "non-finite local boundary repair bound";
      return result;
    }
    if (slope > 0.0) {
      lower = std::max(lower, bound);
    } else {
      upper = std::min(upper, bound);
    }
  }

  const auto escalate_to_multi_node = [&]() {
    LocalBoundaryRepairResult multi =
        try_multi_node_boundary_repair(state, cfg, failing_cell, failing_corner);
    detail::log_multi_node_boundary_repair_stats(multi, gate_reason);
    return multi;
  };
  if (!std::isfinite(lower) || !std::isfinite(upper) || !(lower < upper)) {
    result.reason = "empty local feasible interval";
    if (cfg.numerics.ale.multi_node_boundary_repair_enabled) {
      return escalate_to_multi_node();
    }
    return result;
  }

  double new_coord = std::min(std::max(old_coord, lower), upper);
  const double width = upper - lower;
  const double nudge = std::min(0.25 * width, coord_eps);
  if (new_coord <= lower + nudge) {
    new_coord = lower + nudge;
  }
  if (new_coord >= upper - nudge) {
    new_coord = upper - nudge;
  }
  if (!std::isfinite(new_coord) || !(new_coord > lower) || !(new_coord < upper)) {
    result.reason = "local repair candidate outside feasible interval";
    return result;
  }
  const bool moved = std::fabs(new_coord - old_coord) > coord_eps;

  if (move_r) {
    r_host[static_cast<std::size_t>(node_id)] = new_coord;
    state.x_r.copy_from_host(r_host);
  } else {
    z_host[static_cast<std::size_t>(node_id)] = new_coord;
    state.x_z.copy_from_host(z_host);
  }

  AleMinQualityCell corner_fail;
  int corner = -1;
  double min_corner_j = 1.0;
  const bool corner_tangle =
      compute_corner_post_tangle(state, &corner_fail, &min_corner_j, &corner);
  bool gauss_tangle = false;
  AleMinQualityCell gauss_fail;
  (void)compute_min_quality(state, cfg, &gauss_tangle, &gauss_fail);
  if (corner_tangle || gauss_tangle) {
    if (move_r) {
      r_host[static_cast<std::size_t>(node_id)] = old_coord;
      state.x_r.copy_from_host(r_host);
    } else {
      z_host[static_cast<std::size_t>(node_id)] = old_coord;
      state.x_z.copy_from_host(z_host);
    }
    result.min_corner_j = min_corner_j;
    result.reason = corner_tangle ? "post-repair corner-J check failed"
                                  : "post-repair Gauss check failed";
    return result;
  }

  double affected_min_corner_j = std::numeric_limits<double>::infinity();
  for (const CellCorner& cell : constraints) {
    for (int affected_corner = 0; affected_corner < 4; ++affected_corner) {
      const double j =
          eval_cell_corner_j(cell.i, cell.j, affected_corner, new_coord);
      if (!std::isfinite(j) || j <= repair_j_floor) {
        if (move_r) {
          r_host[static_cast<std::size_t>(node_id)] = old_coord;
          state.x_r.copy_from_host(r_host);
        } else {
          z_host[static_cast<std::size_t>(node_id)] = old_coord;
          state.x_z.copy_from_host(z_host);
        }
        result.min_corner_j = std::isfinite(j) ? j : -1.0;
        result.reason = "post-repair corner-J floor check failed";
        if (cfg.numerics.ale.multi_node_boundary_repair_enabled) {
          return escalate_to_multi_node();
        }
        return result;
      }
      affected_min_corner_j = std::min(affected_min_corner_j, j);
    }
  }

  result.applied = true;
  result.repaired_node_count = 1;
  result.repaired_failing_cells = static_cast<int>(constraints.size());
  result.new_coord = new_coord;
  result.min_corner_j =
      std::min(min_corner_j,
               (affected_min_corner_j == std::numeric_limits<double>::infinity())
                   ? min_corner_j
                   : affected_min_corner_j);
  result.reason = moved ? "accepted" : "already_admissible";
  return result;
}

inline EmergencyCellDeactivationResult try_emergency_cell_deactivation(
    core::State& state,
    const core::Config& cfg,
    const int failing_cell_c,
    const int failing_corner) {
  EmergencyCellDeactivationResult result;
  result.fired = true;
  result.target_cell_c = failing_cell_c;
  result.failing_corner = failing_corner;

  TENRYU_ASSERT(state.mesh.dim == 2, "emergency cell deactivation requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = (nr + 1) * (nz + 1);
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  if (nr <= 0 || nz <= 0 || n_cells <= 0 || n_mat <= 0) {
    result.reason = "invalid mesh or material count";
    return result;
  }
  if (failing_cell_c < 0 || failing_cell_c >= n_cells) {
    result.reason = "failing cell outside mesh";
    return result;
  }

  const int target_i = failing_cell_c / nz;
  const int target_j = failing_cell_c - target_i * nz;
  result.target_cell = AleMinQualityCell{failing_cell_c, target_i, target_j};

  const std::vector<std::int8_t> hydro_active_backup = state.hydro_active;
  const std::vector<std::uint8_t> cell_is_void_backup = state.cell_is_void;
  if (state.hydro_active.empty()) {
    state.hydro_active.assign(static_cast<std::size_t>(n_cells), static_cast<std::int8_t>(1));
    state.note_hydro_active_host_write();
  }
  if (state.cell_is_void.empty()) {
    state.cell_is_void.assign(static_cast<std::size_t>(n_cells), static_cast<std::uint8_t>(0));
  }
  TENRYU_ASSERT(static_cast<int>(state.hydro_active.size()) == n_cells,
                "emergency cell deactivation requires hydro_active size == n_cells");
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "emergency cell deactivation requires cell_is_void size == n_cells");
  if (state.cell_is_void[static_cast<std::size_t>(failing_cell_c)] != 0 ||
      state.hydro_active[static_cast<std::size_t>(failing_cell_c)] == 0) {
    result.reason = "already deactivated";
    state.hydro_active = hydro_active_backup;
    state.note_hydro_active_host_write();
    state.cell_is_void = cell_is_void_backup;
    return result;
  }

  std::vector<double> rho;
  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> Te;
  std::vector<double> Ti;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> zbar;
  std::vector<double> Qvisc;
  std::vector<double> volfrac;
  std::vector<double> cv_e;
  std::vector<double> cv_i;
  std::vector<double> cs;
  std::vector<double> corner_mass;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.Te.copy_to_host(Te);
  state.Ti.copy_to_host(Ti);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  if (state.zbar.size() == static_cast<std::size_t>(n_cells)) {
    state.zbar.copy_to_host(zbar);
  }
  if (state.Qvisc.size() == static_cast<std::size_t>(n_cells)) {
    state.Qvisc.copy_to_host(Qvisc);
  }
  state.volFrac.copy_to_host(volfrac);
  if (state.cv_e.size() == static_cast<std::size_t>(n_cells)) {
    state.cv_e.copy_to_host(cv_e);
  }
  if (state.cv_i.size() == static_cast<std::size_t>(n_cells)) {
    state.cv_i.copy_to_host(cv_i);
  }
  if (state.cs.size() == static_cast<std::size_t>(n_cells)) {
    state.cs.copy_to_host(cs);
  }
  if (state.corner_mass.size() == static_cast<std::size_t>(4 * n_cells)) {
    state.corner_mass.copy_to_host(corner_mass);
  }
  if (state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
      state.v_z.size() == static_cast<std::size_t>(n_nodes)) {
    state.v_r.copy_to_host(v_r);
    state.v_z.copy_to_host(v_z);
  }

  if (static_cast<int>(rho.size()) != n_cells ||
      static_cast<int>(mass.size()) != n_cells ||
      static_cast<int>(vol.size()) != n_cells ||
      static_cast<int>(ee.size()) != n_cells ||
      static_cast<int>(ei.size()) != n_cells ||
      static_cast<int>(Te.size()) != n_cells ||
      static_cast<int>(Ti.size()) != n_cells ||
      static_cast<int>(Pe.size()) != n_cells ||
      static_cast<int>(Pi.size()) != n_cells ||
      static_cast<int>(volfrac.size()) != n_cells * n_mat) {
    result.reason = "required cell fields are not allocated";
    state.hydro_active = hydro_active_backup;
    state.note_hydro_active_host_write();
    state.cell_is_void = cell_is_void_backup;
    return result;
  }

  const std::vector<double> rho_backup = rho;
  const std::vector<double> mass_backup = mass;
  const std::vector<double> ee_backup = ee;
  const std::vector<double> ei_backup = ei;
  const std::vector<double> Te_backup = Te;
  const std::vector<double> Ti_backup = Ti;
  const std::vector<double> Pe_backup = Pe;
  const std::vector<double> Pi_backup = Pi;
  const std::vector<double> zbar_backup = zbar;
  const std::vector<double> Qvisc_backup = Qvisc;
  const std::vector<double> volfrac_backup = volfrac;
  const std::vector<double> cs_backup = cs;
  const std::vector<double> corner_mass_backup = corner_mass;
  const std::vector<double> v_r_backup = v_r;
  const std::vector<double> v_z_backup = v_z;

  const auto restore_all = [&]() {
    state.hydro_active = hydro_active_backup;
    state.note_hydro_active_host_write();
    state.cell_is_void = cell_is_void_backup;
    state.rho.copy_from_host(rho_backup);
    state.mass.copy_from_host(mass_backup);
    state.ee.copy_from_host(ee_backup);
    state.ei.copy_from_host(ei_backup);
    state.Te.copy_from_host(Te_backup);
    state.Ti.copy_from_host(Ti_backup);
    state.Pe.copy_from_host(Pe_backup);
    state.Pi.copy_from_host(Pi_backup);
    state.volFrac.copy_from_host(volfrac_backup);
    if (!zbar_backup.empty()) {
      state.zbar.copy_from_host(zbar_backup);
    }
    if (!Qvisc_backup.empty()) {
      state.Qvisc.copy_from_host(Qvisc_backup);
    }
    if (!cs_backup.empty()) {
      state.cs.copy_from_host(cs_backup);
    }
    if (!corner_mass_backup.empty()) {
      state.corner_mass.copy_from_host(corner_mass_backup);
    }
    if (!v_r_backup.empty() && !v_z_backup.empty()) {
      state.v_r.copy_from_host(v_r_backup);
      state.v_z.copy_from_host(v_z_backup);
    }
  };

  struct Candidate {
    int c = -1;
    int i = -1;
    int j = -1;
    double score = std::numeric_limits<double>::infinity();
  };
  const auto candidate_active = [&](const int c) {
    return c >= 0 && c < n_cells &&
           state.hydro_active[static_cast<std::size_t>(c)] != 0 &&
           state.cell_is_void[static_cast<std::size_t>(c)] == 0 &&
           c != failing_cell_c;
  };
  const auto boundary_cell = [&](const int i, const int j) {
    return i == 0 || i == nr - 1 || j == 0 || j == nz - 1;
  };

  Candidate best_neighbor;
  const int dirs[4][2] = {{0, -1}, {-1, 0}, {1, 0}, {0, 1}};
  const double vol_target =
      std::max(std::fabs(vol[static_cast<std::size_t>(failing_cell_c)]),
               detail::kJFloor);
  for (int order = 0; order < 4; ++order) {
    const int ni = target_i + dirs[order][0];
    const int nj = target_j + dirs[order][1];
    if (ni < 0 || ni >= nr || nj < 0 || nj >= nz) {
      continue;
    }
    const int c = ni * nz + nj;
    if (!candidate_active(c)) {
      continue;
    }
    const double vol_c =
        std::max(std::fabs(vol[static_cast<std::size_t>(c)]), detail::kJFloor);
    const double interior_penalty = boundary_cell(ni, nj) ? 1.0e6 : 0.0;
    const double volume_penalty = 1.0e3 * std::fabs(std::log(vol_c / vol_target));
    const double score = interior_penalty + volume_penalty + static_cast<double>(order);
    if (score < best_neighbor.score) {
      best_neighbor = Candidate{c, ni, nj, score};
    }
  }
  if (best_neighbor.c < 0) {
    result.reason = "no active neighbor";
    restore_all();
    return result;
  }
  result.neighbor_cell_c = best_neighbor.c;
  result.neighbor_cell = AleMinQualityCell{best_neighbor.c, best_neighbor.i,
                                           best_neighbor.j};

  const std::size_t ct = static_cast<std::size_t>(failing_cell_c);
  const std::size_t cn = static_cast<std::size_t>(best_neighbor.c);
  const double rho_floor = cfg.numerics.floors.rho;
  const double Te_floor = cfg.numerics.floors.Te;
  const double Ti_floor = cfg.numerics.floors.Ti;
  const double vol_t = std::max(vol[ct], 0.0);
  const double vol_n = std::max(vol[cn], detail::kJFloor);
  const double m_t_old = std::max(mass[ct], 0.0);
  const double m_n_old = std::max(mass[cn], 0.0);
  const double m_floor = std::max(rho_floor * vol_t, 0.0);
  result.target_mass_before = m_t_old;
  result.target_volume_before = vol_t;
  result.target_rho_before = rho[ct];
  result.target_mass_floor = m_floor;
  result.before_rho = rho[ct];
  result.before_p = Pe[ct] + Pi[ct];
  result.before_Te = Te[ct];
  result.before_Ti = Ti[ct];
  const double m_transfer = std::max(m_t_old - m_floor, 0.0);
  const double m_n_new = m_n_old + m_transfer;
  result.mass_transferred = m_transfer;
  result.mass_floor_delta = std::max(m_floor - m_t_old, 0.0);

  const double ee_floor_specific =
      (!cv_e.empty() && cv_e[ct] > 0.0) ? cv_e[ct] * Te_floor : 0.0;
  const double ei_floor_specific =
      (!cv_i.empty() && cv_i[ct] > 0.0) ? cv_i[ct] * Ti_floor : 0.0;
  const double E_e_target_old = m_t_old * std::max(ee[ct], 0.0);
  const double E_i_target_old = m_t_old * std::max(ei[ct], 0.0);
  const double E_e_target_floor = m_floor * std::max(ee_floor_specific, 0.0);
  const double E_i_target_floor = m_floor * std::max(ei_floor_specific, 0.0);
  const double E_e_transfer = std::max(E_e_target_old - E_e_target_floor, 0.0);
  const double E_i_transfer = std::max(E_i_target_old - E_i_target_floor, 0.0);
  result.ee_transferred = E_e_transfer;
  result.ei_transferred = E_i_transfer;
  result.E_floor_injected = std::max(E_e_target_floor - E_e_target_old, 0.0) +
                            std::max(E_i_target_floor - E_i_target_old, 0.0);

  mass[cn] = m_n_new;
  rho[cn] = m_n_new / vol_n;
  if (m_n_new > 0.0) {
    ee[cn] = (m_n_old * std::max(ee[cn], 0.0) + E_e_transfer) / m_n_new;
    ei[cn] = (m_n_old * std::max(ei[cn], 0.0) + E_i_transfer) / m_n_new;
  }
  mass[ct] = m_floor;
  rho[ct] = rho_floor;
  ee[ct] = std::max(ee_floor_specific, 0.0);
  ei[ct] = std::max(ei_floor_specific, 0.0);
  Te[ct] = Te_floor;
  Ti[ct] = Ti_floor;

  const double gamma = (!cfg.materials.materials.empty() &&
                        cfg.materials.materials.front().ideal_gas_gamma > 1.0)
                           ? cfg.materials.materials.front().ideal_gas_gamma
                           : (5.0 / 3.0);
  const double gm1 = gamma - 1.0;
  const auto close_cell_ideal = [&](const std::size_t c) {
    if (!cv_e.empty() && cv_e[c] > 0.0) {
      Te[c] = std::max(Te_floor, ee[c] / cv_e[c]);
    }
    if (!cv_i.empty() && cv_i[c] > 0.0) {
      Ti[c] = std::max(Ti_floor, ei[c] / cv_i[c]);
    }
    Pe[c] = gm1 * rho[c] * std::max(ee[c], 0.0);
    Pi[c] = gm1 * rho[c] * std::max(ei[c], 0.0);
    if (!cs.empty()) {
      cs[c] = std::sqrt(std::max(gamma * (Pe[c] + Pi[c]) /
                                     std::max(rho[c], rho_floor),
                                 0.0));
    }
  };
  close_cell_ideal(cn);
  close_cell_ideal(ct);
  result.after_rho = rho[ct];
  result.after_p = Pe[ct] + Pi[ct];
  result.after_Te = Te[ct];
  result.after_Ti = Ti[ct];
  result.energy_delta = result.E_floor_injected;
  if (!Qvisc.empty()) {
    Qvisc[ct] = 0.0;
  }
  if (!zbar.empty()) {
    if (m_n_new > 0.0) {
      zbar[cn] = (m_n_old * zbar[cn] + m_transfer * zbar[ct]) / m_n_new;
    }
    zbar[ct] = 0.0;
  }

  int void_mat = -1;
  for (int mat = 0; mat < n_mat; ++mat) {
    if (cfg.materials.materials[static_cast<std::size_t>(mat)].is_void) {
      void_mat = mat;
      break;
    }
  }
  for (int mat = 0; mat < n_mat; ++mat) {
    const std::size_t idx_n = cn * static_cast<std::size_t>(n_mat) +
                              static_cast<std::size_t>(mat);
    const std::size_t idx_t = ct * static_cast<std::size_t>(n_mat) +
                              static_cast<std::size_t>(mat);
    const double vf_n_old = std::max(volfrac[idx_n], 0.0);
    const double vf_t_old = std::max(volfrac[idx_t], 0.0);
    volfrac[idx_n] =
        (m_n_new > 0.0) ? ((m_n_old * vf_n_old + m_transfer * vf_t_old) / m_n_new)
                        : vf_n_old;
    volfrac[idx_t] = (mat == void_mat) ? 1.0 : 0.0;
  }
  double vf_sum = 0.0;
  for (int mat = 0; mat < n_mat; ++mat) {
    vf_sum += volfrac[cn * static_cast<std::size_t>(n_mat) +
                      static_cast<std::size_t>(mat)];
  }
  if (vf_sum > 0.0 && std::isfinite(vf_sum)) {
    for (int mat = 0; mat < n_mat; ++mat) {
      volfrac[cn * static_cast<std::size_t>(n_mat) +
              static_cast<std::size_t>(mat)] /= vf_sum;
    }
  }

  const auto cell_nodes = [&](const int c, int nodes[4]) {
    const int i = c / nz;
    const int j = c - i * nz;
    nodes[0] = tenryu::hydro::rz_node_index_2d(i, j, nz);
    nodes[1] = tenryu::hydro::rz_node_index_2d(i + 1, j, nz);
    nodes[2] = tenryu::hydro::rz_node_index_2d(i + 1, j + 1, nz);
    nodes[3] = tenryu::hydro::rz_node_index_2d(i, j + 1, nz);
  };
  if (!corner_mass.empty()) {
    const int target_nverts =
        detail::state_cell_active_nverts(state, failing_cell_c, n_cells);
    const int neighbor_nverts =
        detail::state_cell_active_nverts(state, best_neighbor.c, n_cells);
    for (int k = 0; k < 4; ++k) {
      corner_mass[ct * 4U + static_cast<std::size_t>(k)] =
          (k < target_nverts)
              ? m_floor / static_cast<double>(target_nverts)
              : 0.0;
      corner_mass[cn * 4U + static_cast<std::size_t>(k)] =
          (k < neighbor_nverts)
              ? m_n_new / static_cast<double>(neighbor_nverts)
              : 0.0;
    }
  }
  if (!v_r.empty() && !v_z.empty() && m_n_new > 0.0) {
    int target_nodes[4];
    int neighbor_nodes[4];
    cell_nodes(failing_cell_c, target_nodes);
    cell_nodes(best_neighbor.c, neighbor_nodes);
    const int target_nverts =
        detail::state_cell_active_nverts(state, failing_cell_c, n_cells);
    const int neighbor_nverts =
        detail::state_cell_active_nverts(state, best_neighbor.c, n_cells);
    double vr_t = 0.0;
    double vz_t = 0.0;
    double vr_n = 0.0;
    double vz_n = 0.0;
    for (int k = 0; k < target_nverts; ++k) {
      vr_t += v_r[static_cast<std::size_t>(target_nodes[k])];
      vz_t += v_z[static_cast<std::size_t>(target_nodes[k])];
    }
    for (int k = 0; k < neighbor_nverts; ++k) {
      vr_n += v_r[static_cast<std::size_t>(neighbor_nodes[k])];
      vz_n += v_z[static_cast<std::size_t>(neighbor_nodes[k])];
    }
    vr_t /= static_cast<double>(target_nverts);
    vz_t /= static_cast<double>(target_nverts);
    vr_n /= static_cast<double>(neighbor_nverts);
    vz_n /= static_cast<double>(neighbor_nverts);
    const double vr_new = (m_n_old * vr_n + m_transfer * vr_t) / m_n_new;
    const double vz_new = (m_n_old * vz_n + m_transfer * vz_t) / m_n_new;
    for (int k = 0; k < neighbor_nverts; ++k) {
      v_r[static_cast<std::size_t>(neighbor_nodes[k])] = vr_new;
      v_z[static_cast<std::size_t>(neighbor_nodes[k])] = vz_new;
    }
  }

  state.hydro_active[ct] = 0;
  state.note_hydro_active_host_write();
  state.cell_is_void[ct] = 1;
  result.target_hydro_active_after = state.hydro_active[ct];
  result.target_cell_is_void_after = state.cell_is_void[ct];
  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.ee.copy_from_host(ee);
  state.ei.copy_from_host(ei);
  state.Te.copy_from_host(Te);
  state.Ti.copy_from_host(Ti);
  state.Pe.copy_from_host(Pe);
  state.Pi.copy_from_host(Pi);
  state.volFrac.copy_from_host(volfrac);
  if (!zbar.empty()) {
    state.zbar.copy_from_host(zbar);
  }
  if (!Qvisc.empty()) {
    state.Qvisc.copy_from_host(Qvisc);
  }
  if (!cs.empty()) {
    state.cs.copy_from_host(cs);
  }
  if (!corner_mass.empty()) {
    state.corner_mass.copy_from_host(corner_mass);
  }
  if (!v_r.empty() && !v_z.empty()) {
    state.v_r.copy_from_host(v_r);
    state.v_z.copy_from_host(v_z);
  }

  bool gauss_tangle = false;
  AleMinQualityCell gauss_fail;
  (void)compute_active_gauss_quality_host(state, &gauss_tangle, &gauss_fail);
  bool corner_tangle = false;
  AleMinQualityCell corner_fail;
  int corner = -1;
  double min_corner_j = 1.0;
  corner_tangle =
      compute_corner_post_tangle(state, cfg, &corner_fail, &min_corner_j, &corner);
  result.min_corner_j = min_corner_j;
  if (gauss_tangle || corner_tangle) {
    restore_all();
    result.reason = gauss_tangle ? "cascading Gauss degeneracy after deactivation"
                                 : "cascading corner-J degeneracy after deactivation";
    return result;
  }

  result.applied = true;
  result.reason = "accepted";
  return result;
}

inline AxisMarginResult compute_axis_margin_min(core::State& state,
                                                const bool has_physical_rz_axis) {
  if (!has_physical_rz_axis) {
    return AxisMarginResult{};
  }
  TENRYU_ASSERT(state.mesh.dim == 2, "ALE axis margin check requires 2D mesh");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return AxisMarginResult{};
  }
  if (mesh::has_tri_fan_center_topology(state.mesh)) {
    return AxisMarginResult{};
  }

  double* d_min_margin = nullptr;
  int* d_min_j = nullptr;
  d_min_margin = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:axis_margin:d_min_margin",
                                   sizeof(double)));
  d_min_j = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:axis_margin:d_min_j", sizeof(int)));

  const double init_margin = 1.0e300;
  const int init_j = -1;
  CUDA_CHECK(cudaMemcpy(d_min_margin, &init_margin, sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_min_j, &init_j, sizeof(int), cudaMemcpyHostToDevice));

  const int blocks = (nz + 255) / 256;
  detail::axis_margin_min_kernel<<<blocks, 256>>>(
      d_min_margin, d_min_j, state.x_r.data(), state.x_z.data(), nz,
      has_physical_rz_axis);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  AxisMarginResult result;
  CUDA_CHECK(cudaMemcpy(&result.min_margin,
                        d_min_margin,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result.min_j, d_min_j, sizeof(int), cudaMemcpyDeviceToHost));
  return result;
}

inline AxisMarginResult compute_axis_margin_min(core::State& state) {
  return compute_axis_margin_min(state, true);
}

inline AxisMarginResult compute_axis_margin_min(
    core::State& state,
    const parallel::Reduction* reduction,
    const bool has_physical_rz_axis) {
  AxisMarginResult result = compute_axis_margin_min(state, has_physical_rz_axis);
  if (reduction != nullptr) {
    result.min_margin = reduction->allreduce_min(result.min_margin);
  }
  return result;
}

inline AxisMarginResult compute_axis_margin_min(core::State& state,
                                                const parallel::Reduction* reduction) {
  return compute_axis_margin_min(state, reduction, true);
}

inline AxisRadialBoundResult check_axis_radial_bound(
    const double* d_xr_trial,
    const double* d_xr_lagrangian,
    const int nr,
    const int nz,
    const double kappa,
    const parallel::Reduction* reduction = nullptr) {
  AxisRadialBoundResult result;
  if (nr < 1 || nz < 0 || !(kappa > 0.0)) {
    return result;
  }

  int* d_failed = nullptr;
  int* d_fail_j = nullptr;
  d_failed = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:axis_bound:d_failed", sizeof(int)));
  d_fail_j = static_cast<int*>(
      core::device_scratch_acquire("ale_rezone:axis_bound:d_fail_j", sizeof(int)));

  const int zero = 0;
  const int no_fail_j = nz + 1;
  CUDA_CHECK(cudaMemcpy(d_failed, &zero, sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fail_j, &no_fail_j, sizeof(int), cudaMemcpyHostToDevice));

  const int axis_nodes = nz + 1;
  const int blocks = (axis_nodes + 255) / 256;
  detail::axis_radial_bound_check_kernel<<<blocks, 256>>>(
      d_failed, d_fail_j, d_xr_trial, d_xr_lagrangian, nr, nz, kappa);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  int failed = 0;
  int fail_j = no_fail_j;
  CUDA_CHECK(cudaMemcpy(&failed, d_failed, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&fail_j, d_fail_j, sizeof(int), cudaMemcpyDeviceToHost));

  result.passed = (failed == 0);
  result.fail_j = (fail_j >= 0 && fail_j <= nz) ? fail_j : -1;
  if (reduction != nullptr) {
    const double failed_global =
        reduction->allreduce_max(result.passed ? 0.0 : 1.0);
    double fail_j_global =
        (result.fail_j >= 0) ? static_cast<double>(result.fail_j) : 1.0e300;
    fail_j_global = reduction->allreduce_min(fail_j_global);
    result.passed = (failed_global < 0.5);
    result.fail_j = (!result.passed && fail_j_global < 1.0e299)
                        ? static_cast<int>(fail_j_global)
                        : -1;
  }
  return result;
}

// Phase 10: minimal axis-spine repair (Z-projection only, no Winslow on i>=1).
// Modifies x_z[i=0, :] in place. x_r is unchanged so the axis remains at r=0.
inline double minimal_axis_z_repair(const core::NodeField1D& d_xr,
                                    core::NodeField1D& d_xz,
                                    const int nr,
                                    const int nz,
                                    const double alpha = 0.5,
                                    const int max_sweeps = 10,
                                    const double tol_z = 1.0e-12) {
  TENRYU_ASSERT(nr >= 1, "minimal_axis_z_repair requires at least one inner row");
  TENRYU_ASSERT(nz >= 0, "minimal_axis_z_repair requires non-negative nz");
  TENRYU_ASSERT(alpha >= 0.0 && alpha <= 1.0,
                "minimal_axis_z_repair alpha must be in [0, 1]");
  TENRYU_ASSERT(max_sweeps >= 0, "minimal_axis_z_repair max_sweeps must be non-negative");

  const int stride = nz + 1;
  const std::size_t required =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(stride);
  TENRYU_ASSERT(d_xr.size() >= required,
                "minimal_axis_z_repair x_r size is smaller than mesh node count");
  TENRYU_ASSERT(d_xz.size() >= required,
                "minimal_axis_z_repair x_z size is smaller than mesh node count");

  std::vector<double> xz_axis_h(static_cast<std::size_t>(stride), 0.0);
  std::vector<double> xz_inner_row_h(static_cast<std::size_t>(stride), 0.0);

  CUDA_CHECK(cudaMemcpy(xz_axis_h.data(),
                        d_xz.data(),
                        static_cast<std::size_t>(stride) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(xz_inner_row_h.data(),
                        d_xz.data() + stride,
                        static_cast<std::size_t>(stride) * sizeof(double),
                        cudaMemcpyDeviceToHost));

  double max_dz = 0.0;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double sweep_max_dz = 0.0;
    for (int j = 0; j < stride; ++j) {
      const std::size_t idx = static_cast<std::size_t>(j);
      const double z_old = xz_axis_h[idx];
      const double z_new = alpha * z_old + (1.0 - alpha) * xz_inner_row_h[idx];
      sweep_max_dz = std::max(sweep_max_dz, std::abs(z_new - z_old));
      xz_axis_h[idx] = z_new;
    }
    max_dz = sweep_max_dz;
    if (sweep_max_dz < tol_z) {
      break;
    }
  }

  CUDA_CHECK(cudaMemcpy(d_xz.data(),
                        xz_axis_h.data(),
                        static_cast<std::size_t>(stride) * sizeof(double),
                        cudaMemcpyHostToDevice));

  return max_dz;
}

inline bool axis_feasibility_guard_trigger(core::State& state,
                                           const core::Config& cfg,
                                           const parallel::Reduction* reduction = nullptr,
                                           AxisMarginResult* current_margin = nullptr,
                                           double* threshold = nullptr) {
  const double guard_fraction = cfg.numerics.ale.preventive_axis_guard_fraction;
  if (!cfg.numerics.has_physical_rz_axis || guard_fraction <= 0.0) {
    return false;
  }
  if (mesh::has_tri_fan_center_topology(state.mesh)) {
    if (current_margin != nullptr) {
      *current_margin = AxisMarginResult{};
    }
    if (threshold != nullptr) {
      *threshold = 0.0;
    }
    return false;
  }

  if (state.axis_margin_initial < 0.0) {
    const AxisMarginResult initial =
        compute_axis_margin_min(state, reduction, cfg.numerics.has_physical_rz_axis);
    state.axis_margin_initial = (initial.min_margin > 0.0) ? initial.min_margin : 1.0;
  }

  const AxisMarginResult current =
      compute_axis_margin_min(state, reduction, cfg.numerics.has_physical_rz_axis);
  const double trigger_threshold = guard_fraction * state.axis_margin_initial;
  if (current_margin != nullptr) {
    *current_margin = current;
  }
  if (threshold != nullptr) {
    *threshold = trigger_threshold;
  }
  return current.min_margin < trigger_threshold;
}

inline RezoneResult run_winslow_rezone(core::State& state,
                                       const core::Config& cfg,
                                       const bool force_rezone = false) {
  RezoneResult out;
  out.min_quality =
      compute_min_quality(state, cfg, &out.mesh_tangle, &out.min_quality_cell_pre);

  const bool corner_floor_violated =
      detail::corner_cell_aspect_floor_violated(state, cfg);
  if (!force_rezone && !corner_floor_violated &&
      out.min_quality >= cfg.numerics.ale.quality_threshold) {
    return out;
  }

  out.triggered = true;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;

  std::vector<std::uint8_t> node_flags = state.mesh.topo.node_flags;
  std::uint8_t* d_node_flags = nullptr;
  d_node_flags = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_node_flags",
                                   static_cast<std::size_t>(n_nodes) *
                                       sizeof(std::uint8_t)));
  CUDA_CHECK(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice));

  double* d_xr_a = nullptr;
  double* d_xz_a = nullptr;
  double* d_xr_b = nullptr;
  double* d_xz_b = nullptr;
  d_xr_a = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_xr_a",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_a = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_xz_a",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xr_b = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_xr_b",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));
  d_xz_b = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_xz_b",
                                   static_cast<std::size_t>(n_nodes) * sizeof(double)));

  CUDA_CHECK(cudaMemcpy(d_xr_a,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_xz_a,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  double dl_min = 1.0;
  if (!state.mesh.cell_area.empty()) {
    dl_min = 1.0e300;
    for (double area : state.mesh.cell_area) {
      if (area > 0.0) {
        dl_min = std::min(dl_min, std::sqrt(area));
      }
    }
    if (!std::isfinite(dl_min) || !(dl_min > 0.0)) {
      dl_min = 1.0;
    }
  }
  const double tol = cfg.numerics.ale.convergence_tol * dl_min;
  const double dr_init =
      (cfg.mesh.r_max - cfg.mesh.r_min) / static_cast<double>(cfg.mesh.nr);
  const double dz_init =
      (cfg.mesh.z_max - cfg.mesh.z_min) / static_cast<double>(cfg.mesh.nz);
  const double reference_lmin = std::max(std::min(dr_init, dz_init), 1.0e-30);
  const bool axis_z_motion_winslow =
      cfg.numerics.has_physical_rz_axis &&
      (cfg.numerics.ale.axis_z_motion == "winslow");
  const bool use_rz_full_metric_winslow =
      (cfg.numerics.ale.rezone_solver == "rz_full_metric_winslow");
  const tenryu::hydro::BC2DRZConfig bc_config =
      cfg.numerics.hydro.boundary_2d.make_bc_config();
  const int axis_nodes = nz + 1;
  std::vector<double> z_axis_old_h(static_cast<std::size_t>(axis_nodes), 0.0);
  std::vector<double> z_axis_target_h(static_cast<std::size_t>(axis_nodes), 0.0);
  std::vector<std::uint8_t> axis_flags_h(static_cast<std::size_t>(axis_nodes),
                                         mesh::NODE_NONE);
  double* d_z_axis_target = nullptr;
  if (axis_z_motion_winslow) {
    for (int j = 0; j <= nz; ++j) {
      axis_flags_h[static_cast<std::size_t>(j)] =
          node_flags[static_cast<std::size_t>(j)];
    }
    CUDA_CHECK(cudaMemcpy(z_axis_old_h.data(),
                          d_xz_a,
                          static_cast<std::size_t>(axis_nodes) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    d_z_axis_target = static_cast<double*>(
        core::device_scratch_acquire("ale_rezone:winslow:d_z_axis_target",
                                     static_cast<std::size_t>(axis_nodes) *
                                         sizeof(double)));
  }

  double* d_max_delta = nullptr;
  AleRezoneIterStats* d_stats = nullptr;
  d_max_delta = static_cast<double*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_max_delta", sizeof(double)));
  d_stats = static_cast<AleRezoneIterStats*>(
      core::device_scratch_acquire("ale_rezone:winslow:d_stats",
                                   sizeof(AleRezoneIterStats)));

  double* cur_r = d_xr_a;
  double* cur_z = d_xz_a;
  double* nxt_r = d_xr_b;
  double* nxt_z = d_xz_b;

  const int blocks = (n_nodes + 255) / 256;
  out.converged = false;
  for (int iter = 0; iter < cfg.numerics.ale.max_iterations; ++iter) {
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(AleRezoneIterStats)));
    if (use_rz_full_metric_winslow) {
      detail::winslow_jacobi_step_rz_full_metric_kernel<<<blocks, 256>>>(
          nxt_r,
          nxt_z,
          cur_r,
          cur_z,
          d_node_flags,
          nr,
          nz,
          cfg.numerics.ale.max_displacement_fraction,
          reference_lmin,
          d_stats,
          axis_z_motion_winslow,
          cfg.numerics.has_physical_rz_axis,
          cfg.numerics.ale.rezone_local_admissibility_linesearch,
          cfg.numerics.ale.rezone_local_j_floor_rel,
          cfg.numerics.ale.rezone_local_linesearch_max_halves,
          state.x_r_initial.size() == state.x_r.size()
              ? state.x_r_initial.data()
              : nullptr,
          state.x_z_initial.size() == state.x_z.size()
              ? state.x_z_initial.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr,
          bc_config,
          cfg.numerics.ale.corner_cell_aspect_protection_enabled,
          cfg.numerics.ale.corner_cell_aspect_eta,
          cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
          cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
    } else {
      detail::winslow_jacobi_step_kernel<<<blocks, 256>>>(
          nxt_r,
          nxt_z,
          cur_r,
          cur_z,
          d_node_flags,
          nr,
          nz,
          cfg.numerics.ale.max_displacement_fraction,
          reference_lmin,
          d_stats,
          axis_z_motion_winslow,
          cfg.numerics.has_physical_rz_axis,
          state.x_r_initial.size() == state.x_r.size()
              ? state.x_r_initial.data()
              : nullptr,
          state.x_z_initial.size() == state.x_z.size()
              ? state.x_z_initial.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr,
          bc_config,
          cfg.numerics.ale.corner_cell_aspect_protection_enabled,
          cfg.numerics.ale.corner_cell_aspect_eta,
          cfg.numerics.hydro.boundary_2d.z_bottom_cfg.is_state_supply(),
          cfg.numerics.hydro.boundary_2d.z_top_cfg.is_state_supply());
    }
    CUDA_CHECK(cudaGetLastError());

    if (axis_z_motion_winslow && !use_rz_full_metric_winslow) {
      detail::apply_axis_spine_projection(nxt_r,
                                          nxt_z,
                                          d_z_axis_target,
                                          z_axis_target_h,
                                          z_axis_old_h,
                                          axis_flags_h,
                                          nz,
                                          dz_init,
                                          cfg.numerics.has_physical_rz_axis);
      const AxisRadialBoundResult bound = check_axis_radial_bound(
          nxt_r, d_xr_a, nr, nz, cfg.numerics.ale.winslow_axis_kappa);
      if (!bound.passed) {
        out.axis_radial_bound_failed = true;
        if (out.axis_radial_bound_fail_j < 0) {
          out.axis_radial_bound_fail_j = bound.fail_j;
        }
      }
    }

    const double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(d_max_delta, &zero, sizeof(double), cudaMemcpyHostToDevice));
    detail::max_node_delta_kernel<<<blocks, 256>>>(nxt_r,
                                                   nxt_z,
                                                   cur_r,
                                                   cur_z,
                                                   d_max_delta,
                                                   n_nodes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&out.residual, d_max_delta, sizeof(double), cudaMemcpyDeviceToHost));
    AleRezoneIterStats iter_stats;
    CUDA_CHECK(cudaMemcpy(&iter_stats,
                          d_stats,
                          sizeof(AleRezoneIterStats),
                          cudaMemcpyDeviceToHost));
    detail::accumulate_rezone_stats(out.stats, iter_stats);
    out.iterations = iter + 1;

    double* old_cur_r = cur_r;
    double* old_cur_z = cur_z;
    cur_r = nxt_r;
    cur_z = nxt_z;
    nxt_r = old_cur_r;
    nxt_z = old_cur_z;

    if (out.residual < tol) {
      out.converged = true;
      break;
    }
  }

  detail::enforce_corner_cell_aspect_floor(cur_r, cur_z, state, cfg);

  CUDA_CHECK(cudaMemcpy(state.x_r.data(),
                        cur_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.x_z.data(),
                        cur_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice));

  bool post_tangle = false;
  (void)compute_min_quality(state, cfg, &post_tangle, &out.min_quality_cell_post);
  out.final_axis_margin =
      compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis).min_margin;

  if (!out.converged) {
    core::log_warning("ALE rezone did not converge within max_iterations: iter=" +
                      std::to_string(out.iterations) +
                      ", residual=" + std::to_string(out.residual));
    detail::log_ale_rezone_stats(out.iterations,
                                 out.stats,
                                 out.min_quality_cell_pre,
                                 out.min_quality_cell_post,
                                 out.final_axis_margin,
                                 0,
                                 "non-convergence");
  }

  return out;
}

}  // namespace tenryu::hydro::ale
