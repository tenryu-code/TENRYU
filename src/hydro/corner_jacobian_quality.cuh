#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "coupling/hydro_step_result.hpp"
#include "hydro/rz_quad_volume.cuh"

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::hydro {

struct CellRegime;

struct CornerJacobianQualityResult {
  bool admissible = true;
  double min_current_j = std::numeric_limits<double>::infinity();
  double min_trial_j = std::numeric_limits<double>::infinity();
  bool gauss_j_failed = false;
  bool rz_volume_failed = false;
  double min_gauss_j_trial = std::numeric_limits<double>::infinity();
  double min_rz_volume_trial = std::numeric_limits<double>::infinity();
  double min_trial_ratio = 1.0;
  double suggested_dt = 0.0;
  int first_failing_cell = -1;
  int first_failing_corner = -1;
  // Axis-classification fields. Populated only when the telemetry
  // env gate or dispatcher state-sensitive bypass is active.
  double g0_rel = 0.0;
  double g1_rel = 0.0;
  double dt_star_est = 0.0;
  bool dt_sensitive = false;
  bool repair_sensitive = false;
};

struct CornerJacobianScaleResult {
  double scale = 1.0;
  bool trigger_recommended = false;
  double trigger_scale = 1.0;
  double trigger_threshold = 0.0;
  int first_failing_cell = -1;
  int first_failing_corner = -1;
  int first_trigger_cell = -1;
  int first_trigger_corner = -1;
};

struct CornerBalanceResult {
  bool admissible = true;
  double min_q_balance = 1.0;
  int first_failing_cell = -1;
  int first_failing_corner = -1;
  bool any_corner_non_positive = false;
  bool any_corner_non_finite = false;
};

enum class MeshQualityFailingMetric : std::uint8_t {
  CornerJ = 0,
  GaussJ,
  RzVolume,
  AxisMargin,
  None,
};

struct MeshQualityDtLimit {
  bool admissible = true;
  double sigma_safe = 1.0;
  double suggested_dt = 0.0;
  int first_cell = -1;
  int first_corner_or_gauss = -1;
  MeshQualityFailingMetric metric = MeshQualityFailingMetric::None;
};

struct MeshQualityRzVolumeCellMargin {
  int cell = -1;
  bool admissible = true;
  double sigma_raw = 1.0;
  double eta_safe = 1.0;
  double v0 = 0.0;
  double floor = 0.0;
};

const char* mesh_quality_metric_string(MeshQualityFailingMetric metric);

inline double sanitize_in_hydro_reduced_min_metric(
    const double global_min_metric,
    const double global_any_failure) {
  if (!std::isfinite(global_min_metric)) {
    return global_any_failure > 0.5
               ? -std::numeric_limits<double>::infinity()
               : 0.0;
  }
  return global_min_metric;
}

inline void apply_in_hydro_location_retry_hint(
    tenryu::coupling::HydroStepResult& result,
    const int nr,
    const int nz) {
  if (result.first_failing_i < 0 || result.first_failing_j < 0 ||
      nr <= 0 || nz <= 0) {
    return;
  }
  if (result.first_failing_i == 0) {
    result.regime = MeshFailureRegime::AxisFace;
    result.retry_action =
        tenryu::coupling::RetryActionHint::ForceAxisSpinePlusLocalAle;
  } else if (result.first_failing_i == nr - 1 ||
             result.first_failing_j == 0 ||
             result.first_failing_j == nz - 1) {
    result.regime = MeshFailureRegime::DomainBoundary;
    result.retry_action =
        tenryu::coupling::RetryActionHint::ForceBoundaryPatchRepair;
  } else {
    result.regime = MeshFailureRegime::InteriorSmooth;
    result.retry_action = tenryu::coupling::RetryActionHint::ReduceDtOnly;
  }
}

__host__ __device__ inline int rz_node_index_2d(const int i,
                                                const int j,
                                                const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ inline double corner_jacobian_cross2(const double ar,
                                                         const double az,
                                                         const double br,
                                                         const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ inline double corner_jacobian_from_quad(const double* rr,
                                                            const double* zz,
                                                            const int corner) {
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  return corner_jacobian_cross2(rr[kp] - rr[corner], zz[kp] - zz[corner],
                                rr[km] - rr[corner], zz[km] - zz[corner]);
}

__host__ __device__ inline double in_hydro_trial_quadratic_at_sigma(
    const double q0,
    const double q1,
    const double q2,
    const double sigma) {
  return q0 + sigma * q1 + sigma * sigma * q2;
}

__host__ __device__ inline bool in_hydro_isfinite(const double value) {
#if defined(__CUDA_ARCH__)
  return isfinite(value);
#else
  return std::isfinite(value);
#endif
}

__host__ __device__ inline double in_hydro_trial_quadratic_min_on_unit_interval(
    const double q0,
    const double q1,
    const double q2) {
  double min_q = fmin(q0, in_hydro_trial_quadratic_at_sigma(q0, q1, q2, 1.0));
  if (q2 > 0.0) {
    const double sigma_vertex = -q1 / (2.0 * q2);
    if (sigma_vertex > 0.0 && sigma_vertex < 1.0) {
      min_q = fmin(min_q,
                   in_hydro_trial_quadratic_at_sigma(q0, q1, q2, sigma_vertex));
    }
  }
  return min_q;
}

__host__ __device__ inline void in_hydro_bilinear_derivatives(
    const double* rr,
    const double* zz,
    const double xi,
    const double eta,
    double& r_xi,
    double& z_xi,
    double& r_eta,
    double& z_eta) {
  const double dxi[4] = {
      -0.25 * (1.0 - eta),
       0.25 * (1.0 - eta),
       0.25 * (1.0 + eta),
      -0.25 * (1.0 + eta),
  };
  const double deta[4] = {
      -0.25 * (1.0 - xi),
      -0.25 * (1.0 + xi),
       0.25 * (1.0 + xi),
       0.25 * (1.0 - xi),
  };
  r_xi = 0.0;
  z_xi = 0.0;
  r_eta = 0.0;
  z_eta = 0.0;
  for (int k = 0; k < 4; ++k) {
    r_xi += dxi[k] * rr[k];
    z_xi += dxi[k] * zz[k];
    r_eta += deta[k] * rr[k];
    z_eta += deta[k] * zz[k];
  }
}

__host__ __device__ inline void in_hydro_gauss_j_coefficients_at(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1,
    const double xi,
    const double eta,
    double& j0,
    double& j1,
    double& j2) {
  double dr[4];
  double dz[4];
  for (int k = 0; k < 4; ++k) {
    dr[k] = rr1[k] - rr0[k];
    dz[k] = zz1[k] - zz0[k];
  }

  double rxi0;
  double zxi0;
  double reta0;
  double zeta0;
  double drxi;
  double dzxi;
  double dreta;
  double dzeta;
  in_hydro_bilinear_derivatives(rr0, zz0, xi, eta,
                                rxi0, zxi0, reta0, zeta0);
  in_hydro_bilinear_derivatives(dr, dz, xi, eta,
                                drxi, dzxi, dreta, dzeta);
  j0 = corner_jacobian_cross2(rxi0, zxi0, reta0, zeta0);
  j1 = corner_jacobian_cross2(drxi, dzxi, reta0, zeta0) +
       corner_jacobian_cross2(rxi0, zxi0, dreta, dzeta);
  j2 = corner_jacobian_cross2(drxi, dzxi, dreta, dzeta);
}

__host__ __device__ inline double in_hydro_gauss_j_min_on_unit_interval(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1) {
  constexpr double g = 0.57735026918962576450914878050195745565;
  const double xi[4] = {-g, g, g, -g};
  const double eta[4] = {-g, -g, g, g};
  double min_j = INFINITY;
  for (int q = 0; q < 4; ++q) {
    double j0;
    double j1;
    double j2;
    in_hydro_gauss_j_coefficients_at(rr0, zz0, rr1, zz1, xi[q], eta[q],
                                     j0, j1, j2);
    if (!in_hydro_isfinite(j0) || !in_hydro_isfinite(j1) ||
        !in_hydro_isfinite(j2)) {
      return -INFINITY;
    }
    min_j = fmin(min_j,
                 in_hydro_trial_quadratic_min_on_unit_interval(j0, j1, j2));
  }
  return min_j;
}

__host__ __device__ inline double in_hydro_rz_volume_at_sigma(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1,
    const double sigma) {
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = rr0[k] + sigma * (rr1[k] - rr0[k]);
    zz[k] = zz0[k] + sigma * (zz1[k] - zz0[k]);
  }
  return ale::detail::rz_signed_quad_volume(
      rr[0], zz[0], rr[1], zz[1], rr[2], zz[2], rr[3], zz[3]);
}

__host__ __device__ inline void in_hydro_rz_volume_cubic_coefficients(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1,
    double& a,
    double& b,
    double& c,
    double& d) {
  a = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 0.0);
  const double v13 = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 1.0 / 3.0);
  const double v23 = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 2.0 / 3.0);
  const double v1 = in_hydro_rz_volume_at_sigma(rr0, zz0, rr1, zz1, 1.0);
  const double A = v13 - a;
  const double B = v23 - a;
  const double C = v1 - a;
  d = 0.5 * (9.0 * C + 27.0 * A - 27.0 * B);
  c = (9.0 * C - 27.0 * A - 8.0 * d) / 6.0;
  b = C - c - d;
}

__host__ __device__ inline double in_hydro_cubic_at_sigma(
    const double a,
    const double b,
    const double c,
    const double d,
    const double sigma) {
  return ((d * sigma + c) * sigma + b) * sigma + a;
}

__host__ __device__ inline double in_hydro_rz_volume_min_on_unit_interval(
    const double* rr0,
    const double* zz0,
    const double* rr1,
    const double* zz1) {
  double a;
  double b;
  double c;
  double d;
  in_hydro_rz_volume_cubic_coefficients(rr0, zz0, rr1, zz1, a, b, c, d);
  if (!in_hydro_isfinite(a) || !in_hydro_isfinite(b) ||
      !in_hydro_isfinite(c) || !in_hydro_isfinite(d)) {
    return -INFINITY;
  }

  double min_v = fmin(a, in_hydro_cubic_at_sigma(a, b, c, d, 1.0));
  const double qa = 3.0 * d;
  const double qb = 2.0 * c;
  const double qc = b;
  const double scale = fmax(fmax(fabs(qa), fabs(qb)),
                            fmax(fabs(qc), 1.0e-300));
  if (fabs(qa) <= 1.0e-12 * scale) {
    if (fabs(qb) > 1.0e-12 * scale) {
      const double root = -qc / qb;
      if (root > 0.0 && root < 1.0) {
        min_v = fmin(min_v, in_hydro_cubic_at_sigma(a, b, c, d, root));
      }
    }
    return min_v;
  }

  const double disc = qb * qb - 4.0 * qa * qc;
  const double disc_scale = fmax(qb * qb, fabs(4.0 * qa * qc));
  if (disc >= -1.0e-12 * disc_scale) {
    const double sqrt_disc = sqrt(fmax(0.0, disc));
    const double root_a = (-qb - sqrt_disc) / (2.0 * qa);
    const double root_b = (-qb + sqrt_disc) / (2.0 * qa);
    if (root_a > 0.0 && root_a < 1.0) {
      min_v = fmin(min_v, in_hydro_cubic_at_sigma(a, b, c, d, root_a));
    }
    if (root_b > 0.0 && root_b < 1.0) {
      min_v = fmin(min_v, in_hydro_cubic_at_sigma(a, b, c, d, root_b));
    }
  }
  return min_v;
}

CornerJacobianQualityResult check_trial_lagrangian_corner_jacobian_host(
    int nr,
    int nz,
    const std::vector<double>& r_base,
    const std::vector<double>& z_base,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    double dt,
    double floor_eps,
    const parallel::Reduction* reduction,
    const std::vector<std::int8_t>& hydro_active = {},
    bool axis_margin_guard_enabled = false,
    bool has_physical_rz_axis = true);

CornerJacobianQualityResult check_trial_lagrangian_corner_jacobian_fields(
    int nr,
    int nz,
    const core::NodeField1D& r_base,
    const core::NodeField1D& z_base,
    const core::NodeField1D& v_r,
    const core::NodeField1D& v_z,
    double dt,
    double floor_eps,
    const parallel::Reduction* reduction,
    const std::vector<std::int8_t>& hydro_active = {},
    bool axis_margin_guard_enabled = false,
    bool has_physical_rz_axis = true);

CornerJacobianScaleResult compute_corner_jacobian_trial_scale(
    const core::State& state,
    double dt_in,
    double floor_eps,
    const parallel::Reduction* reduction,
    const CellRegime* d_cell_regime = nullptr,
    bool regime_aware_corner_j_guard_enabled = false,
    bool axis_margin_guard_enabled = false,
    double legacy_trigger_scale = 0.5,
    bool has_physical_rz_axis = true);

tenryu::coupling::HydroStepResult compute_in_hydro_candidate_mesh_guard(
    const core::State& state,
    double dt,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_x_r_candidate,
    const double* d_x_z_candidate,
    const core::Config& cfg,
    tenryu::coupling::HydroFailureStage stage,
    const parallel::Reduction* reduction,
    const std::int8_t* d_hydro_active = nullptr,
    const CellRegime* d_cell_regime = nullptr);

MeshQualityDtLimit compute_mesh_quality_dt_limit(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    double dt,
    const parallel::Reduction* reduction);

std::vector<MeshQualityRzVolumeCellMargin>
compute_multiblock_mesh_quality_rz_volume_cell_margins(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    double dt,
    const std::vector<int>& cells);

std::vector<MeshQualityRzVolumeCellMargin>
compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double* d_cell_vol_for_floor,
    double dt,
    const std::vector<int>& cells);

CornerBalanceResult evaluate_corner_balance(
    const core::State& state,
    double balance_threshold,
    const parallel::Reduction* reduction);

}  // namespace tenryu::hydro
