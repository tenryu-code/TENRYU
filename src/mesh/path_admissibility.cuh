#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "hydro/pole_angular_coarsen.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::mesh {

inline constexpr int kCentralMacroCoreSentinelCell = -2;
inline constexpr int kPoleAngularCoarsenSentinelCell = -3;

enum class PathAdmissibilityMetricKind : int {
  UNKNOWN = 0,
  EDGE_CROSS = 1,
  GAUSS_J = 2,
  AREA_OR_RZ_VOLUME = 3,
  R_GUARD = 4,
};

enum class PathAdmissibilitySourceKind : int {
  UNKNOWN = 0,
  ACTIVE_FINE_CHILD = 1,
  CENTRAL_MACRO_BOUNDARY = 2,
  POLE_MACRO_BOUNDARY = 3,
};

inline const char* path_admissibility_metric_kind_name(const int kind) {
  switch (static_cast<PathAdmissibilityMetricKind>(kind)) {
    case PathAdmissibilityMetricKind::EDGE_CROSS:
      return "edge_cross";
    case PathAdmissibilityMetricKind::GAUSS_J:
      return "gauss_j";
    case PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME:
      return "area_or_rz_volume";
    case PathAdmissibilityMetricKind::R_GUARD:
      return "r_guard";
    case PathAdmissibilityMetricKind::UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

inline const char* path_admissibility_source_kind_name(const int kind) {
  switch (static_cast<PathAdmissibilitySourceKind>(kind)) {
    case PathAdmissibilitySourceKind::ACTIVE_FINE_CHILD:
      return "active_fine_child";
    case PathAdmissibilitySourceKind::CENTRAL_MACRO_BOUNDARY:
      return "central_macro_boundary";
    case PathAdmissibilitySourceKind::POLE_MACRO_BOUNDARY:
      return "pole_macro_boundary";
    case PathAdmissibilitySourceKind::UNKNOWN:
      return "unknown";
  }
  return "unknown";
}

struct PathAdmissibilityResult {
  __host__ __device__ PathAdmissibilityResult()
      : min_margin(HUGE_VAL),
        min_margin_cell(-1),
        first_failing_cell(-1),
        first_failing_lambda(1.0),
        min_margin_metric_kind(
            static_cast<int>(PathAdmissibilityMetricKind::UNKNOWN)),
        first_failing_metric_kind(
            static_cast<int>(PathAdmissibilityMetricKind::UNKNOWN)),
        path_source_kind(
            static_cast<int>(PathAdmissibilitySourceKind::UNKNOWN)),
        old_geometry_inadmissible(0),
        macro_repair_required(0),
        pole_axis_radial_order_inversion(0),
        first_failing_block_id(-1),
        first_failing_local_i(-1),
        first_failing_local_j(-1),
        macro_has_anatomy(0),
        macro_simple_ok(1),
        macro_viol_node_a(-1),
        macro_viol_node_b(-1),
        macro_v_old(0.0),
        macro_v_new(0.0),
        macro_floor(0.0),
        macro_viol_theta_a(0.0),
        macro_viol_theta_b(0.0),
        macro_span(0),
        macro_level(0),
        macro_q_begin(-1),
        macro_q_end(-1),
        macro_j_begin(-1),
        macro_j_end(-1),
        macro_skipped_nodes(0) {}

  double min_margin;
  int min_margin_cell;
  int first_failing_cell;
  double first_failing_lambda;
  int min_margin_metric_kind;
  int first_failing_metric_kind;
  int path_source_kind;
  int old_geometry_inadmissible;
  int macro_repair_required;
  int pole_axis_radial_order_inversion;
  int first_failing_block_id;
  int first_failing_local_i;
  int first_failing_local_j;
  // Macro-boundary failure anatomy, filled by the host macro-boundary
  // evaluator (terminal-failure forensics: volume-floor vs simple-loop is
  // otherwise not distinguishable from the rejection log). theta =
  // atan2(r_mid, z_mid) of the violating segment midpoint in [0, pi]:
  // ~0 = north pole, ~pi = south pole, ~pi/2 = equator.
  int macro_has_anatomy;
  int macro_simple_ok;
  int macro_viol_node_a;
  int macro_viol_node_b;
  double macro_v_old;
  double macro_v_new;
  double macro_floor;
  double macro_viol_theta_a;
  double macro_viol_theta_b;
  int macro_span;
  int macro_level;
  int macro_q_begin;
  int macro_q_end;
  int macro_j_begin;
  int macro_j_end;
  int macro_skipped_nodes;
};

enum MeshForecastFailureBits : std::uint32_t {
  kMeshForecastFailureQJ = 1U << 0,
  kMeshForecastFailureQV = 1U << 1,
  kMeshForecastFailureQEdge = 1U << 2,
  kMeshForecastFailureQR = 1U << 3,
  kMeshForecastFailureQPhi = 1U << 4,
};

struct MeshForecastComponents {
  double q_J = HUGE_VAL;
  double q_V = HUGE_VAL;
  double q_edge = HUGE_VAL;
  double q_R = HUGE_VAL;
  double q_phi = HUGE_VAL;
};

struct MeshForecast {
  int endpoint_valid = 1;
  double tau_zero = 1.0;
  double tau_warn = 1.0;
  double q_warn = 0.0;
  double q_min_now = HUGE_VAL;
  double q_min_end = HUGE_VAL;
  double q_min_path = HUGE_VAL;
  int first_cell = -1;
  int block_id = -1;
  int local_i = -1;
  int local_j = -1;
  std::uint32_t failure_bits = 0U;
  int seed_count = 0;
  std::vector<std::uint8_t> seed_mask;
  MeshForecastComponents worst_now;
  MeshForecastComponents worst_warn;
  MeshForecastComponents worst_end;
  double q_trace_tau0 = HUGE_VAL;
  double q_trace_tau25 = HUGE_VAL;
  double q_trace_tau50 = HUGE_VAL;
  double q_trace_tau75 = HUGE_VAL;
  double q_trace_tau100 = HUGE_VAL;
};

namespace path_admissibility_detail {

__host__ __device__ inline double cross2(const double ax,
                                         const double az,
                                         const double bx,
                                         const double bz) {
  return ax * bz - az * bx;
}

__host__ __device__ inline double quadratic_value(const double q0,
                                                  const double q1,
                                                  const double q2,
                                                  const double lambda) {
  return q0 + lambda * (q1 + lambda * q2);
}

__host__ __device__ inline void observe_unit_root_candidate(
    const double root,
    double& best_root) {
  constexpr double tol = 1.0e-14;
  if (!(root > 0.0) || root > 1.0 + tol) {
    return;
  }
  const double clamped = root > 1.0 ? 1.0 : root;
  if (clamped < best_root) {
    best_root = clamped;
  }
}

__host__ __device__ inline double quadratic_floor_root_unit(
    const double q0,
    const double q1,
    const double q2,
    const double floor) {
  if (!isfinite(q0) || !isfinite(q1) || !isfinite(q2) ||
      !isfinite(floor)) {
    return 0.0;
  }
  const double p0 = q0 - floor;
  if (!(p0 > 0.0)) {
    return 0.0;
  }
  const double p1 = q1;
  const double p2 = q2;
  const double scale =
      fmax(fmax(fabs(p0), fabs(p1)), fmax(fabs(p2), 1.0));
  constexpr double eps = 1.0e-14;
  double best_root = 1.0;
  if (fabs(p2) <= eps * scale) {
    if (fabs(p1) <= eps * scale) {
      return 1.0;
    }
    observe_unit_root_candidate(-p0 / p1, best_root);
    return best_root;
  }

  const double disc = p1 * p1 - 4.0 * p2 * p0;
  const double disc_scale = fmax(p1 * p1, fabs(4.0 * p2 * p0));
  if (disc < -eps * fmax(disc_scale, 1.0)) {
    return 1.0;
  }
  const double sqrt_disc = sqrt(fmax(0.0, disc));
  const double denom = 2.0 * p2;
  observe_unit_root_candidate((-p1 - sqrt_disc) / denom, best_root);
  observe_unit_root_candidate((-p1 + sqrt_disc) / denom, best_root);
  return best_root;
}

__host__ __device__ inline void observe_quadratic_min(const double q0,
                                                      const double q1,
                                                      const double q2,
                                                      const double lambda,
                                                      double& best_value,
                                                      double& best_lambda) {
  const double value = quadratic_value(q0, q1, q2, lambda);
  if (value < best_value) {
    best_value = value;
    best_lambda = lambda;
  }
}

__host__ __device__ inline void quadratic_min_on_unit_interval(
    const double q0,
    const double q1,
    const double q2,
    double& min_value,
    double& min_lambda) {
  min_value = q0;
  min_lambda = 0.0;
  observe_quadratic_min(q0, q1, q2, 1.0, min_value, min_lambda);
  if (q2 > 0.0) {
    const double lambda_v = -0.5 * q1 / q2;
    if (lambda_v > 0.0 && lambda_v < 1.0) {
      observe_quadratic_min(q0, q1, q2, lambda_v, min_value, min_lambda);
    }
  }
}

__host__ __device__ inline void edge_cross_coefficients(
    const double rr0[4],
    const double zz0[4],
    const double drr[4],
    const double dzz[4],
    const int u_tail,
    const int u_head,
    const int v_tail,
    const int v_head,
    double& q0,
    double& q1,
    double& q2) {
  const double ur0 = rr0[u_head] - rr0[u_tail];
  const double uz0 = zz0[u_head] - zz0[u_tail];
  const double vr0 = rr0[v_head] - rr0[v_tail];
  const double vz0 = zz0[v_head] - zz0[v_tail];
  const double dur = drr[u_head] - drr[u_tail];
  const double duz = dzz[u_head] - dzz[u_tail];
  const double dvr = drr[v_head] - drr[v_tail];
  const double dvz = dzz[v_head] - dzz[v_tail];
  q0 = cross2(ur0, uz0, vr0, vz0);
  q1 = cross2(dur, duz, vr0, vz0) + cross2(ur0, uz0, dvr, dvz);
  q2 = cross2(dur, duz, dvr, dvz);
}

__host__ __device__ inline void point_cross_coefficients(
    const double rr0[4],
    const double zz0[4],
    const double drr[4],
    const double dzz[4],
    const int a,
    const int b,
    double& q0,
    double& q1,
    double& q2) {
  q0 = cross2(rr0[a], zz0[a], rr0[b], zz0[b]);
  q1 = cross2(drr[a], dzz[a], rr0[b], zz0[b]) +
       cross2(rr0[a], zz0[a], drr[b], dzz[b]);
  q2 = cross2(drr[a], dzz[a], drr[b], dzz[b]);
}

__host__ __device__ inline void gauss_j_coefficients(
    const double rr0[4],
    const double zz0[4],
    const double drr[4],
    const double dzz[4],
    const double xi,
    const double eta,
    double& q0,
    double& q1,
    double& q2) {
  const double dxi[4] = {-0.25 * (1.0 - eta),
                         0.25 * (1.0 - eta),
                         0.25 * (1.0 + eta),
                         -0.25 * (1.0 + eta)};
  const double deta[4] = {-0.25 * (1.0 - xi),
                          -0.25 * (1.0 + xi),
                          0.25 * (1.0 + xi),
                          0.25 * (1.0 - xi)};
  double gr0 = 0.0;
  double gz0 = 0.0;
  double hr0 = 0.0;
  double hz0 = 0.0;
  double dgr = 0.0;
  double dgz = 0.0;
  double dhr = 0.0;
  double dhz = 0.0;
  for (int k = 0; k < 4; ++k) {
    gr0 += dxi[k] * rr0[k];
    gz0 += dxi[k] * zz0[k];
    hr0 += deta[k] * rr0[k];
    hz0 += deta[k] * zz0[k];
    dgr += dxi[k] * drr[k];
    dgz += dxi[k] * dzz[k];
    dhr += deta[k] * drr[k];
    dhz += deta[k] * dzz[k];
  }
  q0 = cross2(gr0, gz0, hr0, hz0);
  q1 = cross2(dgr, dgz, hr0, hz0) + cross2(gr0, gz0, dhr, dhz);
  q2 = cross2(dgr, dgz, dhr, dhz);
}

__host__ __device__ inline double gauss_j_static_value(
    const double rr[4],
    const double zz[4],
    const double xi,
    const double eta) {
  const double zero[4] = {0.0, 0.0, 0.0, 0.0};
  double q0 = 0.0;
  double q1 = 0.0;
  double q2 = 0.0;
  gauss_j_coefficients(rr, zz, zero, zero, xi, eta, q0, q1, q2);
  return q0;
}

template <bool TrackAnatomy>
__host__ __device__ inline void observe_metric_with_root(
    const double q0,
    const double q1,
    const double q2,
    const double q_reference,
    const double floor_rel,
    const int metric_kind,
    double& min_margin,
    double& first_failing_lambda,
    int& min_margin_metric_kind,
    int& first_failing_metric_kind) {
  if (!isfinite(q0) || !isfinite(q1) || !isfinite(q2) || q0 == 0.0) {
    min_margin = -HUGE_VAL;
    first_failing_lambda = 0.0;
    if (TrackAnatomy) {
      min_margin_metric_kind = metric_kind;
      first_failing_metric_kind = metric_kind;
    }
    return;
  }
  const double sign = q0 > 0.0 ? 1.0 : -1.0;
  const double q0n = sign * q0;
  const double q1n = sign * q1;
  const double q2n = sign * q2;
  const double q_ref_n = sign * q_reference;
  const double floor_baseline =
      isfinite(q_reference) && q_ref_n > 0.0 ? q_ref_n : q0n;
  const double floor = fmax(0.0, floor_rel) * floor_baseline;
  double value = HUGE_VAL;
  double lambda = 0.0;
  quadratic_min_on_unit_interval(q0n, q1n, q2n, value, lambda);
  const double margin = value - floor;
  if (margin < min_margin) {
    min_margin = margin;
    if (TrackAnatomy) {
      min_margin_metric_kind = metric_kind;
    }
  }
  const double root = quadratic_floor_root_unit(q0n, q1n, q2n, floor);
  if (root < first_failing_lambda) {
    first_failing_lambda = root;
    if (TrackAnatomy) {
      first_failing_metric_kind = metric_kind;
    }
  }
}

template <bool TrackAnatomy>
__host__ __device__ inline void observe_metric_with_oriented_root(
    const double q0,
    const double q1,
    const double q2,
    const double q_reference,
    const double orientation_sign,
    const double floor_rel,
    const int metric_kind,
    double& min_margin,
    double& first_failing_lambda,
    int& min_margin_metric_kind,
    int& first_failing_metric_kind,
    int& old_geometry_inadmissible) {
  if (orientation_sign == 0.0) {
    observe_metric_with_root<TrackAnatomy>(
        q0, q1, q2, q_reference, floor_rel, metric_kind, min_margin,
        first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind);
    return;
  }
  if (!isfinite(q0) || !isfinite(q1) || !isfinite(q2)) {
    min_margin = -HUGE_VAL;
    first_failing_lambda = 0.0;
    old_geometry_inadmissible = 1;
    if (TrackAnatomy) {
      min_margin_metric_kind = metric_kind;
      first_failing_metric_kind = metric_kind;
    }
    return;
  }
  const double sign = orientation_sign < 0.0 ? -1.0 : 1.0;
  const double q0n = sign * q0;
  const double q1n = sign * q1;
  const double q2n = sign * q2;
  if (!(q0n > 0.0)) {
    min_margin = -HUGE_VAL;
    first_failing_lambda = 0.0;
    old_geometry_inadmissible = 1;
    if (TrackAnatomy) {
      min_margin_metric_kind = metric_kind;
      first_failing_metric_kind = metric_kind;
    }
    return;
  }
  const double q_ref_n = sign * q_reference;
  const double floor_baseline =
      isfinite(q_reference) && q_ref_n > 0.0 ? q_ref_n : q0n;
  const double floor = fmax(0.0, floor_rel) * floor_baseline;
  double value = HUGE_VAL;
  double lambda = 0.0;
  quadratic_min_on_unit_interval(q0n, q1n, q2n, value, lambda);
  const double margin = value - floor;
  if (margin < min_margin) {
    min_margin = margin;
    if (TrackAnatomy) {
      min_margin_metric_kind = metric_kind;
    }
  }
  const double root = quadratic_floor_root_unit(q0n, q1n, q2n, floor);
  if (root < first_failing_lambda) {
    first_failing_lambda = root;
    if (TrackAnatomy) {
      first_failing_metric_kind = metric_kind;
    }
  }
}

template <bool TrackAnatomy,
          int SlotCap = kMeshTopoCellStorageSlotsMax>
__host__ __device__ inline PathAdmissibilityResult evaluate_cell_path_local(
    const int cell,
    const double rr0[4],
    const double zz0[4],
    const double drr[4],
    const double dzz[4],
    const int active_nverts,
    const double* rr_reference,
    const double* zz_reference,
    const double floor_rel,
    const double orientation_sign = 0.0,
    // r_guard (rebound-scope verdict PR3, env TENRYU_PATH_ADMISSIBILITY_R_GUARD):
    // a free node carried through the z-axis (r < 0) is unconditionally
    // unphysical in RZ, but only topologically axis-flagged nodes have the
    // r=0 projector — the 3.23 ns lite wall was exactly such a crossing
    // reaching the hard geometry assert. When enabled, each vertex's linear
    // radius path r(lambda) = r0 + lambda*dr joins the admissibility line
    // search (reusing the quadratic observer with q2 = 0); vertices already
    // at r <= 0 are skipped (axis-pinned or degenerate — the volume metrics
    // own those).
    const bool r_guard = false) {
  if (active_nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#else
    ::tenryu::core::tenryu_abort("active_nverts <= SlotCap",
                                 "cell exceeds slot cap",
                                 __FILE__, __LINE__);
#endif
  }
  PathAdmissibilityResult out;
  out.min_margin_cell = cell;
  out.first_failing_cell = cell;
  out.path_source_kind =
      static_cast<int>(PathAdmissibilitySourceKind::ACTIVE_FINE_CHILD);

  double min_margin = HUGE_VAL;
  double first_failing_lambda = 1.0;
  int min_margin_metric_kind =
      static_cast<int>(PathAdmissibilityMetricKind::UNKNOWN);
  int first_failing_metric_kind =
      static_cast<int>(PathAdmissibilityMetricKind::UNKNOWN);
  int old_geometry_inadmissible = 0;
  double q0 = 0.0;
  double q1 = 0.0;
  double q2 = 0.0;
  const bool reference_available =
      floor_rel > 0.0 && rr_reference != nullptr && zz_reference != nullptr;
  const double no_reference = HUGE_VAL;

  if (active_nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const int km = (k + 2) % 3;
      edge_cross_coefficients(rr0, zz0, drr, dzz, k, kp, k, km,
                              q0, q1, q2);
      observe_metric_with_oriented_root<TrackAnatomy>(
          q0, q1, q2, no_reference, orientation_sign, floor_rel,
          static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
          min_margin, first_failing_lambda, min_margin_metric_kind,
          first_failing_metric_kind, old_geometry_inadmissible);
    }
  } else if (active_nverts == 4) {
    edge_cross_coefficients(rr0, zz0, drr, dzz, 0, 1, 0, 3, q0, q1, q2);
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, no_reference, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    edge_cross_coefficients(rr0, zz0, drr, dzz, 0, 1, 1, 2, q0, q1, q2);
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, no_reference, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    edge_cross_coefficients(rr0, zz0, drr, dzz, 3, 2, 1, 2, q0, q1, q2);
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, no_reference, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    edge_cross_coefficients(rr0, zz0, drr, dzz, 3, 2, 0, 3, q0, q1, q2);
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, no_reference, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);

    const double g = 0.57735026918962576451;
    gauss_j_coefficients(rr0, zz0, drr, dzz, -g, -g, q0, q1, q2);
    const double gauss_ref_mm =
        reference_available
            ? gauss_j_static_value(rr_reference, zz_reference, -g, -g)
            : no_reference;
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, gauss_ref_mm, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::GAUSS_J),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    gauss_j_coefficients(rr0, zz0, drr, dzz, g, -g, q0, q1, q2);
    const double gauss_ref_pm =
        reference_available
            ? gauss_j_static_value(rr_reference, zz_reference, g, -g)
            : no_reference;
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, gauss_ref_pm, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::GAUSS_J),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    gauss_j_coefficients(rr0, zz0, drr, dzz, g, g, q0, q1, q2);
    const double gauss_ref_pp =
        reference_available
            ? gauss_j_static_value(rr_reference, zz_reference, g, g)
            : no_reference;
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, gauss_ref_pp, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::GAUSS_J),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
    gauss_j_coefficients(rr0, zz0, drr, dzz, -g, g, q0, q1, q2);
    const double gauss_ref_mp =
        reference_available
            ? gauss_j_static_value(rr_reference, zz_reference, -g, g)
            : no_reference;
    observe_metric_with_oriented_root<TrackAnatomy>(
        q0, q1, q2, gauss_ref_mp, orientation_sign, floor_rel,
        static_cast<int>(PathAdmissibilityMetricKind::GAUSS_J),
        min_margin, first_failing_lambda, min_margin_metric_kind,
        first_failing_metric_kind, old_geometry_inadmissible);
  } else if (active_nverts == 5) {
    double corner_q0[5];
    double corner_q1[5];
    double corner_q2[5];
    double corner_scale = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      const int kp = (k + 1) % active_nverts;
      const int km = (k + active_nverts - 1) % active_nverts;
      edge_cross_coefficients(rr0, zz0, drr, dzz, k, kp, k, km,
                              corner_q0[k], corner_q1[k], corner_q2[k]);
      corner_scale = fmax(corner_scale, fabs(corner_q0[k]));
    }
    for (int k = 0; k < active_nverts; ++k) {
      if (corner_scale > 0.0 &&
          fabs(corner_q0[k]) <= 1.0e-9 * corner_scale) {
        continue;
      }
      observe_metric_with_oriented_root<TrackAnatomy>(
          corner_q0[k], corner_q1[k], corner_q2[k], no_reference,
          orientation_sign, floor_rel,
          static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS),
          min_margin, first_failing_lambda, min_margin_metric_kind,
          first_failing_metric_kind, old_geometry_inadmissible);
    }
  } else {
    min_margin = -HUGE_VAL;
    first_failing_lambda = 0.0;
    old_geometry_inadmissible = 1;
  }

  double area0 = 0.0;
  double area1 = 0.0;
  double area2 = 0.0;
  double area_reference = 0.0;
  const double zero[SlotCap] = {0.0};
  for (int k = 0; k < active_nverts; ++k) {
    const int n = (k + 1) % active_nverts;
    point_cross_coefficients(rr0, zz0, drr, dzz, k, n, q0, q1, q2);
    area0 += q0;
    area1 += q1;
    area2 += q2;
    if (reference_available) {
      point_cross_coefficients(rr_reference, zz_reference, zero, zero, k, n,
                               q0, q1, q2);
      area_reference += q0;
    }
  }
  observe_metric_with_oriented_root<TrackAnatomy>(
      area0, area1, area2,
      reference_available ? area_reference : no_reference,
      orientation_sign, floor_rel,
      static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME),
      min_margin, first_failing_lambda, min_margin_metric_kind,
      first_failing_metric_kind, old_geometry_inadmissible);

  if (r_guard) {
    for (int k = 0; k < active_nverts; ++k) {
      if (rr0[k] > 0.0) {
        observe_metric_with_root<TrackAnatomy>(
            rr0[k], drr[k], 0.0, no_reference, 0.0,
            static_cast<int>(PathAdmissibilityMetricKind::R_GUARD),
            min_margin, first_failing_lambda, min_margin_metric_kind,
            first_failing_metric_kind);
      }
    }
  }

  out.min_margin = min_margin;
  out.first_failing_lambda = first_failing_lambda;
  out.old_geometry_inadmissible = old_geometry_inadmissible;
  if (TrackAnatomy) {
    out.min_margin_metric_kind = min_margin_metric_kind;
    out.first_failing_metric_kind = first_failing_metric_kind;
  }
  if (min_margin > 0.0 || !(first_failing_lambda < 1.0)) {
    out.first_failing_cell = -1;
  }
  return out;
}

template <bool TrackAnatomy,
          int SlotCap = kMeshTopoCellStorageSlotsMax>
__device__ inline PathAdmissibilityResult evaluate_cell_path(
    const int cell,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const double* __restrict__ xr_old,
    const double* __restrict__ xz_old,
    const double* __restrict__ xr_new,
    const double* __restrict__ xz_new,
    const double* __restrict__ xr_reference,
    const double* __restrict__ xz_reference,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const int topology_stride,
    const double floor_rel,
    const bool r_guard = false) {
  PathAdmissibilityResult out;
  out.min_margin_cell = cell;
  out.first_failing_cell = cell;
  out.path_source_kind =
      static_cast<int>(PathAdmissibilitySourceKind::ACTIVE_FINE_CHILD);
  const int off = cell_node_offsets[cell];
  const int next = cell_node_offsets[cell + 1];
  const int active_nverts = mesh_topo_cell_active_nverts(cell_nverts, cell);
  if (active_nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#else
    ::tenryu::core::tenryu_abort("active_nverts <= SlotCap",
                                 "cell exceeds slot cap",
                                 __FILE__, __LINE__);
#endif
  }
  if (next - off != topology_stride || off < 0) {
    out.min_margin = -HUGE_VAL;
    out.first_failing_lambda = 0.0;
    return out;
  }

  int nodes[SlotCap] = {};
  for (int k = 0; k < SlotCap; ++k) {
    nodes[k] = -1;
  }
  double rr0[SlotCap] = {0.0};
  double zz0[SlotCap] = {0.0};
  double drr[SlotCap] = {0.0};
  double dzz[SlotCap] = {0.0};
  double rr_reference[SlotCap] = {0.0};
  double zz_reference[SlotCap] = {0.0};
  const bool reference_available =
      floor_rel > 0.0 && xr_reference != nullptr && xz_reference != nullptr;
  for (int k = 0; k < active_nverts; ++k) {
    nodes[k] = cell_node_indices[off + k];
    if (nodes[k] < 0) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_lambda = 0.0;
      return out;
    }
    rr0[k] = xr_old[nodes[k]];
    zz0[k] = xz_old[nodes[k]];
    drr[k] = xr_new[nodes[k]] - rr0[k];
    dzz[k] = xz_new[nodes[k]] - zz0[k];
    if (reference_available) {
      rr_reference[k] = xr_reference[nodes[k]];
      zz_reference[k] = xz_reference[nodes[k]];
    }
    if (!isfinite(rr0[k]) || !isfinite(zz0[k]) ||
        !isfinite(drr[k]) || !isfinite(dzz[k])) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_lambda = 0.0;
      return out;
    }
  }

  const double orientation_sign =
      (cell_orientation_sign != nullptr)
          ? (cell_orientation_sign[cell] < 0 ? -1.0 : 1.0)
          : 0.0;
  return evaluate_cell_path_local<TrackAnatomy, SlotCap>(
      cell, rr0, zz0, drr, dzz, active_nverts,
      reference_available ? rr_reference : nullptr,
      reference_available ? zz_reference : nullptr, floor_rel,
      orientation_sign, r_guard);
}

template <bool TrackAnatomy,
          int SlotCap = kMeshTopoCellStorageSlotsMax>
__device__ inline PathAdmissibilityResult evaluate_cell_path(
    const int cell,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const double* __restrict__ xr_old,
    const double* __restrict__ xz_old,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double dt,
    const double* __restrict__ xr_reference,
    const double* __restrict__ xz_reference,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const int topology_stride,
    const double floor_rel,
    const bool r_guard = false) {
  PathAdmissibilityResult out;
  out.min_margin_cell = cell;
  out.first_failing_cell = cell;
  out.path_source_kind =
      static_cast<int>(PathAdmissibilitySourceKind::ACTIVE_FINE_CHILD);
  const int off = cell_node_offsets[cell];
  const int next = cell_node_offsets[cell + 1];
  const int active_nverts = mesh_topo_cell_active_nverts(cell_nverts, cell);
  if (active_nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#else
    ::tenryu::core::tenryu_abort("active_nverts <= SlotCap",
                                 "cell exceeds slot cap",
                                 __FILE__, __LINE__);
#endif
  }
  if (next - off != topology_stride || off < 0 || !isfinite(dt)) {
    out.min_margin = -HUGE_VAL;
    out.first_failing_lambda = 0.0;
    return out;
  }

  int nodes[SlotCap] = {};
  for (int k = 0; k < SlotCap; ++k) {
    nodes[k] = -1;
  }
  double rr0[SlotCap] = {0.0};
  double zz0[SlotCap] = {0.0};
  double drr[SlotCap] = {0.0};
  double dzz[SlotCap] = {0.0};
  double rr_reference[SlotCap] = {0.0};
  double zz_reference[SlotCap] = {0.0};
  const bool reference_available =
      floor_rel > 0.0 && xr_reference != nullptr && xz_reference != nullptr;
  for (int k = 0; k < active_nverts; ++k) {
    nodes[k] = cell_node_indices[off + k];
    if (nodes[k] < 0) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_lambda = 0.0;
      return out;
    }
    rr0[k] = xr_old[nodes[k]];
    zz0[k] = xz_old[nodes[k]];
    drr[k] = dt * v_r[nodes[k]];
    dzz[k] = dt * v_z[nodes[k]];
    if (reference_available) {
      rr_reference[k] = xr_reference[nodes[k]];
      zz_reference[k] = xz_reference[nodes[k]];
    }
    if (!isfinite(rr0[k]) || !isfinite(zz0[k]) ||
        !isfinite(drr[k]) || !isfinite(dzz[k])) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_lambda = 0.0;
      return out;
    }
  }

  const double orientation_sign =
      (cell_orientation_sign != nullptr)
          ? (cell_orientation_sign[cell] < 0 ? -1.0 : 1.0)
          : 0.0;
  return evaluate_cell_path_local<TrackAnatomy, SlotCap>(
      cell, rr0, zz0, drr, dzz, active_nverts,
      reference_available ? rr_reference : nullptr,
      reference_available ? zz_reference : nullptr, floor_rel,
      orientation_sign, r_guard);
}

template <bool TrackAnatomy,
          int SlotCap = kMeshTopoCellStorageSlotsMax>
static __global__ void evaluate_path_admissibility_csr_kernel(
    PathAdmissibilityResult* __restrict__ results,
    const int n_cells,
    const int topology_stride,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const double* __restrict__ xr_old,
    const double* __restrict__ xz_old,
    const double* __restrict__ xr_new,
    const double* __restrict__ xz_new,
    const double* __restrict__ xr_reference,
    const double* __restrict__ xz_reference,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const double floor_rel,
    const bool r_guard) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }
  if (inactive_cell_mask != nullptr && inactive_cell_mask[cell] != 0U) {
    results[cell] = PathAdmissibilityResult{};
    return;
  }
  results[cell] = evaluate_cell_path<TrackAnatomy, SlotCap>(cell,
                                                   cell_node_offsets,
                                                   cell_node_indices,
                                                   xr_old,
                                                   xz_old,
                                                   xr_new,
                                                   xz_new,
                                                   xr_reference,
                                                   xz_reference,
                                                   cell_nverts,
                                                   cell_orientation_sign,
                                                   topology_stride,
                                                   floor_rel,
                                                   r_guard);
}

template <bool TrackAnatomy,
          int SlotCap = kMeshTopoCellStorageSlotsMax>
static __global__ void evaluate_path_admissibility_csr_velocity_kernel(
    PathAdmissibilityResult* __restrict__ results,
    const int n_cells,
    const int topology_stride,
    const int* __restrict__ cell_node_offsets,
    const int* __restrict__ cell_node_indices,
    const double* __restrict__ xr_old,
    const double* __restrict__ xz_old,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double dt,
    const double* __restrict__ xr_reference,
    const double* __restrict__ xz_reference,
    const std::uint8_t* __restrict__ cell_nverts,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ inactive_cell_mask,
    const double floor_rel,
    const bool r_guard) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }
  if (inactive_cell_mask != nullptr && inactive_cell_mask[cell] != 0U) {
    results[cell] = PathAdmissibilityResult{};
    return;
  }
  results[cell] = evaluate_cell_path<TrackAnatomy, SlotCap>(cell,
                                                   cell_node_offsets,
                                                   cell_node_indices,
                                                   xr_old,
                                                   xz_old,
                                                   v_r,
                                                   v_z,
                                                   dt,
                                                   xr_reference,
                                                   xz_reference,
                                                   cell_nverts,
                                                   cell_orientation_sign,
                                                   topology_stride,
                                                   floor_rel,
                                                   r_guard);
}

inline bool path_admissibility_r_guard_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_PATH_ADMISSIBILITY_R_GUARD");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

inline bool path_admissibility_anatomy_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_PATH_ADMIS_ANATOMY");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  return enabled;
}

inline void annotate_path_admissibility_location(
    PathAdmissibilityResult& result,
    const tenryu::core::State& state) {
  if (result.first_failing_cell < 0 ||
      !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int cell = result.first_failing_cell;
  int block_id = -1;
  const BlockInfo* block = nullptr;
  if (cell >= 0 && mb.cell_block_id.size() > static_cast<std::size_t>(cell)) {
    const int mapped = mb.cell_block_id[static_cast<std::size_t>(cell)];
    if (mapped >= 0 && mapped < static_cast<int>(mb.blocks.size())) {
      const BlockInfo& candidate = mb.blocks[static_cast<std::size_t>(mapped)];
      if (cell >= candidate.cell_begin &&
          cell < candidate.cell_begin + candidate.cell_count) {
        block_id = mapped;
        block = &candidate;
      }
    }
  }
  if (block == nullptr) {
    for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
      const BlockInfo& candidate = mb.blocks[static_cast<std::size_t>(b)];
      if (cell >= candidate.cell_begin &&
          cell < candidate.cell_begin + candidate.cell_count) {
        block_id = b;
        block = &candidate;
        break;
      }
    }
  }
  if (block == nullptr || block->n_j_cells <= 0) {
    return;
  }
  const int local = cell - block->cell_begin;
  if (local < 0 || local >= block->cell_count) {
    return;
  }
  result.first_failing_block_id = block_id;
  result.first_failing_local_i = local / block->n_j_cells;
  result.first_failing_local_j =
      local - result.first_failing_local_i * block->n_j_cells;
}

inline void mark_pole_deref_repair_cell(
    const tenryu::hydro::pole_angular_coarsen::Overlay& overlay,
    const int i,
    const int j,
    std::vector<std::uint8_t>& mask) {
  if (i < 0 || i >= overlay.n_i_cells || j < 0 || j >= overlay.n_j_cells ||
      overlay.cell_begin < 0 || overlay.n_j_cells <= 0) {
    return;
  }
  const int cell = overlay.cell_begin + i * overlay.n_j_cells + j;
  if (cell >= 0 && static_cast<std::size_t>(cell) < mask.size()) {
    mask[static_cast<std::size_t>(cell)] = 1U;
  }
}

inline std::vector<std::uint8_t> build_pole_deref_repair_cell_mask(
    const tenryu::hydro::pole_angular_coarsen::Overlay* overlay,
    const int n_cells) {
  std::vector<std::uint8_t> mask;
  if (overlay == nullptr || !overlay->active ||
      !overlay->supports_deref_macro_repair || n_cells <= 0 ||
      overlay->cell_begin < 0 || overlay->n_i_cells <= 0 ||
      overlay->n_j_cells <= 0) {
    return mask;
  }
  mask.assign(static_cast<std::size_t>(n_cells), 0U);
  for (const auto& macro : overlay->macros) {
    if (macro.block_id != overlay->block_id) {
      continue;
    }
    const int i0 = std::max(0, macro.local_i_begin);
    const int i1 = std::min(overlay->n_i_cells, macro.local_i_end);
    const int j0 = std::max(0, macro.local_j_begin);
    const int j1 = std::min(overlay->n_j_cells, macro.local_j_end);
    for (int i = i0; i < i1; ++i) {
      for (int j = j0; j < j1; ++j) {
        mark_pole_deref_repair_cell(*overlay, i, j, mask);
      }
      if (macro.local_j_begin == 0 && macro.local_j_end < overlay->n_j_cells) {
        mark_pole_deref_repair_cell(*overlay, i, macro.local_j_end, mask);
      }
      if (macro.local_j_end == overlay->n_j_cells &&
          macro.local_j_begin > 0) {
        mark_pole_deref_repair_cell(*overlay, i, macro.local_j_begin - 1, mask);
      }
    }
  }
  return mask;
}

inline void gate_active_fine_child_macro_repair(
    PathAdmissibilityResult& result,
    const std::vector<std::uint8_t>& pole_deref_repair_cell_mask) {
  if (result.path_source_kind !=
          static_cast<int>(PathAdmissibilitySourceKind::ACTIVE_FINE_CHILD) ||
      result.old_geometry_inadmissible == 0) {
    return;
  }
  result.macro_repair_required = 0;
  const int cell = result.first_failing_cell;
  if (cell >= 0 &&
      static_cast<std::size_t>(cell) < pole_deref_repair_cell_mask.size() &&
      pole_deref_repair_cell_mask[static_cast<std::size_t>(cell)] != 0U) {
    result.macro_repair_required = 1;
  }
}

inline void merge_path_admissibility_result(
    PathAdmissibilityResult& result,
    const PathAdmissibilityResult& cell_result) {
  if (cell_result.min_margin < result.min_margin) {
    result.min_margin = cell_result.min_margin;
    result.min_margin_cell = cell_result.min_margin_cell;
    result.min_margin_metric_kind = cell_result.min_margin_metric_kind;
  }
  if (cell_result.first_failing_cell == -1) {
    return;
  }
  if (cell_result.first_failing_lambda < result.first_failing_lambda ||
      (cell_result.first_failing_lambda == result.first_failing_lambda &&
       (result.first_failing_cell == -1 ||
        cell_result.first_failing_cell < result.first_failing_cell))) {
    result.first_failing_cell = cell_result.first_failing_cell;
    result.first_failing_lambda = cell_result.first_failing_lambda;
    result.first_failing_metric_kind =
        cell_result.first_failing_metric_kind;
    result.path_source_kind = cell_result.path_source_kind;
    result.old_geometry_inadmissible =
        cell_result.old_geometry_inadmissible;
    result.macro_repair_required = cell_result.macro_repair_required;
    result.pole_axis_radial_order_inversion =
        cell_result.pole_axis_radial_order_inversion;
    result.first_failing_block_id = cell_result.first_failing_block_id;
    result.first_failing_local_i = cell_result.first_failing_local_i;
    result.first_failing_local_j = cell_result.first_failing_local_j;
  }
  // The macro-boundary evaluator runs once per evaluation and is the only
  // producer of the anatomy fields; carry them across the merge.
  if (cell_result.macro_has_anatomy != 0) {
    result.macro_has_anatomy = cell_result.macro_has_anatomy;
    result.macro_simple_ok = cell_result.macro_simple_ok;
    result.macro_viol_node_a = cell_result.macro_viol_node_a;
    result.macro_viol_node_b = cell_result.macro_viol_node_b;
    result.macro_v_old = cell_result.macro_v_old;
    result.macro_v_new = cell_result.macro_v_new;
    result.macro_floor = cell_result.macro_floor;
    result.macro_viol_theta_a = cell_result.macro_viol_theta_a;
    result.macro_viol_theta_b = cell_result.macro_viol_theta_b;
    result.macro_span = cell_result.macro_span;
    result.macro_level = cell_result.macro_level;
    result.macro_q_begin = cell_result.macro_q_begin;
    result.macro_q_end = cell_result.macro_q_end;
    result.macro_j_begin = cell_result.macro_j_begin;
    result.macro_j_end = cell_result.macro_j_end;
    result.macro_skipped_nodes = cell_result.macro_skipped_nodes;
    result.pole_axis_radial_order_inversion =
        cell_result.pole_axis_radial_order_inversion;
  }
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool has_nonquad_cell_nverts(
    const std::vector<std::uint8_t>& cell_nverts,
    const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts == 3U || nverts == 5U;
                     });
}

inline bool central_macro_path_active(const tenryu::core::State& state) {
  const auto& pc = state.central_pseudo_core;
  return pc.configured && pc.valid &&
         pc.boundary_nodes_ordered.size() >= 3U &&
         pc.inactive_member_mask.size() ==
             static_cast<std::size_t>(state.mesh.topo.n_cells);
}

inline void copy_device_node_field(const double* d_values,
                                   const int n_nodes,
                                   std::vector<double>& host,
                                   const char* message) {
  host.assign(static_cast<std::size_t>(n_nodes), 0.0);
  cuda_check(cudaMemcpy(host.data(),
                        d_values,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             message);
}

inline double rz_polygon_volume(const std::vector<double>& r,
                                const std::vector<double>& z,
                                const std::vector<int>& nodes) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  const int n = static_cast<int>(nodes.size());
  if (n < 3) {
    return -HUGE_VAL;
  }
  double sum = 0.0;
  for (int k = 0; k < n; ++k) {
    const int a = nodes[static_cast<std::size_t>(k)];
    const int b = nodes[static_cast<std::size_t>((k + 1) % n)];
    const double ra = r[static_cast<std::size_t>(a)];
    const double za = z[static_cast<std::size_t>(a)];
    const double rb = r[static_cast<std::size_t>(b)];
    const double zb = z[static_cast<std::size_t>(b)];
    if (!std::isfinite(ra) || !std::isfinite(za) ||
        !std::isfinite(rb) || !std::isfinite(zb)) {
      return -HUGE_VAL;
    }
    sum += (ra + rb) * (ra * zb - rb * za);
  }
  return kPi / 3.0 * sum;
}

inline double orient2d(const double ax,
                       const double az,
                       const double bx,
                       const double bz,
                       const double cx,
                       const double cz) {
  return (bx - ax) * (cz - az) - (bz - az) * (cx - ax);
}

inline long double orient2d_long_double(const double ax,
                                        const double az,
                                        const double bx,
                                        const double bz,
                                        const double cx,
                                        const double cz) {
  const long double lax = static_cast<long double>(ax);
  const long double laz = static_cast<long double>(az);
  const long double lbx = static_cast<long double>(bx);
  const long double lbz = static_cast<long double>(bz);
  const long double lcx = static_cast<long double>(cx);
  const long double lcz = static_cast<long double>(cz);
  return (lbx - lax) * (lcz - laz) - (lbz - laz) * (lcx - lax);
}

inline int orient2d_long_double_sign(const double ax,
                                     const double az,
                                     const double bx,
                                     const double bz,
                                     const double cx,
                                     const double cz) {
  const long double value =
      orient2d_long_double(ax, az, bx, bz, cx, cz);
  if (value > 0.0L) {
    return 1;
  }
  if (value < 0.0L) {
    return -1;
  }
  return 0;
}

inline double distance2d(const double ax,
                         const double az,
                         const double bx,
                         const double bz) {
  return std::hypot(bx - ax, bz - az);
}

inline bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

inline bool path_predicate_harden_enabled() {
  static const bool enabled =
      env_flag_enabled("TENRYU_I1B_PATH_PREDICATE_HARDEN") ||
      env_flag_enabled("TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE");
  return enabled;
}

inline double loop_coordinate_scale(const std::vector<double>& r,
                                    const std::vector<double>& z,
                                    const std::vector<int>& nodes) {
  if (nodes.empty()) {
    return 0.0;
  }
  double r_min = HUGE_VAL;
  double r_max = -HUGE_VAL;
  double z_min = HUGE_VAL;
  double z_max = -HUGE_VAL;
  double max_abs = 0.0;
  double max_edge = 0.0;
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    const int a = nodes[static_cast<std::size_t>(i)];
    const int b = nodes[static_cast<std::size_t>((i + 1) % nodes.size())];
    const double ra = r[static_cast<std::size_t>(a)];
    const double za = z[static_cast<std::size_t>(a)];
    const double rb = r[static_cast<std::size_t>(b)];
    const double zb = z[static_cast<std::size_t>(b)];
    if (!std::isfinite(ra) || !std::isfinite(za)) {
      continue;
    }
    r_min = std::min(r_min, ra);
    r_max = std::max(r_max, ra);
    z_min = std::min(z_min, za);
    z_max = std::max(z_max, za);
    max_abs = std::max(max_abs, std::max(std::abs(ra), std::abs(za)));
    if (std::isfinite(rb) && std::isfinite(zb)) {
      max_edge = std::max(max_edge, distance2d(ra, za, rb, zb));
    }
  }
  double bbox = 0.0;
  if (std::isfinite(r_min) && std::isfinite(r_max) &&
      std::isfinite(z_min) && std::isfinite(z_max)) {
    bbox = std::max(r_max - r_min, z_max - z_min);
  }
  return std::max({bbox, max_abs, max_edge, 0.0});
}

inline double loop_coincidence_epsilon(const std::vector<double>& r,
                                       const std::vector<double>& z,
                                       const std::vector<int>& nodes) {
  constexpr double kAbsTol = 1.0e-12;
  constexpr double kRelTol = 1.0e-9;
  const double scale = loop_coordinate_scale(r, z, nodes);
  return std::max(kAbsTol, kRelTol * scale);
}

struct LoopNodeAliasMap {
  double epsilon = 0.0;
  std::vector<int> input_representative_nodes;
  std::vector<int> loop_nodes;
  std::vector<int> source_segment_index;
};

inline bool loop_nodes_coincident(const std::vector<double>& r,
                                  const std::vector<double>& z,
                                  const int a,
                                  const int b,
                                  const double eps) {
  if (a == b) {
    return true;
  }
  if (a < 0 || b < 0 ||
      static_cast<std::size_t>(a) >= r.size() ||
      static_cast<std::size_t>(b) >= r.size() ||
      static_cast<std::size_t>(a) >= z.size() ||
      static_cast<std::size_t>(b) >= z.size()) {
    return false;
  }
  const double dr = r[static_cast<std::size_t>(a)] -
                    r[static_cast<std::size_t>(b)];
  const double dz = z[static_cast<std::size_t>(a)] -
                    z[static_cast<std::size_t>(b)];
  return std::isfinite(dr) && std::isfinite(dz) &&
         dr * dr + dz * dz <= eps * eps;
}

inline LoopNodeAliasMap canonicalize_loop_node_aliases(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes,
    const int preferred_node = -1,
    const bool keep_single_preferred = false) {
  LoopNodeAliasMap map;
  map.epsilon = loop_coincidence_epsilon(r, z, nodes);
  map.input_representative_nodes.assign(nodes.size(), -1);
  if (nodes.empty()) {
    return map;
  }

  const bool preferred_valid =
      preferred_node >= 0 &&
      static_cast<std::size_t>(preferred_node) < r.size() &&
      static_cast<std::size_t>(preferred_node) < z.size();
  std::vector<int> group_representatives;
  std::vector<int> input_group(nodes.size(), -1);
  int preferred_group = -1;
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    const int node = nodes[static_cast<std::size_t>(i)];
    int group = -1;
    for (int g = 0; g < static_cast<int>(group_representatives.size()); ++g) {
      if (loop_nodes_coincident(
              r, z, group_representatives[static_cast<std::size_t>(g)],
              node, map.epsilon)) {
        group = g;
        break;
      }
    }
    if (group < 0) {
      group = static_cast<int>(group_representatives.size());
      group_representatives.push_back(node);
    }
    input_group[static_cast<std::size_t>(i)] = group;
    if (preferred_valid &&
        (node == preferred_node ||
         loop_nodes_coincident(r, z, preferred_node, node, map.epsilon))) {
      preferred_group = group;
    }
  }
  if (preferred_group >= 0) {
    group_representatives[static_cast<std::size_t>(preferred_group)] =
        preferred_node;
  }

  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    const int group = input_group[static_cast<std::size_t>(i)];
    if (group >= 0) {
      map.input_representative_nodes[static_cast<std::size_t>(i)] =
          group_representatives[static_cast<std::size_t>(group)];
    }
  }

  bool preferred_emitted = false;
  for (int i = 0; i < static_cast<int>(nodes.size()); ++i) {
    const int representative =
        map.input_representative_nodes[static_cast<std::size_t>(i)];
    if (representative < 0) {
      continue;
    }
    if (keep_single_preferred && preferred_group >= 0 &&
        representative == preferred_node) {
      if (preferred_emitted) {
        continue;
      }
      preferred_emitted = true;
    }
    if (!map.loop_nodes.empty() &&
        map.loop_nodes.back() == representative) {
      continue;
    }
    map.loop_nodes.push_back(representative);
    map.source_segment_index.push_back(i);
  }
  while (map.loop_nodes.size() > 1U &&
         map.loop_nodes.front() == map.loop_nodes.back()) {
    map.loop_nodes.pop_back();
    map.source_segment_index.pop_back();
  }
  return map;
}

inline bool on_segment(const double ax,
                       const double az,
                       const double bx,
                       const double bz,
                       const double px,
                       const double pz,
                       const double length_eps) {
  const double ab_r = bx - ax;
  const double ab_z = bz - az;
  if (ab_r * ab_r + ab_z * ab_z <= length_eps * length_eps) {
    return false;
  }
  return px >= std::min(ax, bx) - length_eps &&
         px <= std::max(ax, bx) + length_eps &&
         pz >= std::min(az, bz) - length_eps &&
         pz <= std::max(az, bz) + length_eps &&
         orient2d_long_double_sign(ax, az, bx, bz, px, pz) == 0;
}

inline bool segments_intersect(const double ax,
                               const double az,
                               const double bx,
                               const double bz,
                               const double cx,
                               const double cz,
                               const double dx,
                               const double dz,
                               const double length_eps,
                               const double orient_eps) {
  (void)orient_eps;
  const double ab_r = bx - ax;
  const double ab_z = bz - az;
  const double cd_r = dx - cx;
  const double cd_z = dz - cz;
  if (ab_r * ab_r + ab_z * ab_z <= length_eps * length_eps ||
      cd_r * cd_r + cd_z * cd_z <= length_eps * length_eps) {
    return false;
  }
  const int s1 = orient2d_long_double_sign(ax, az, bx, bz, cx, cz);
  const int s2 = orient2d_long_double_sign(ax, az, bx, bz, dx, dz);
  const int s3 = orient2d_long_double_sign(cx, cz, dx, dz, ax, az);
  const int s4 = orient2d_long_double_sign(cx, cz, dx, dz, bx, bz);
  if (s1 * s2 < 0 && s3 * s4 < 0) {
    return true;
  }
  return on_segment(ax, az, bx, bz, cx, cz, length_eps) ||
         on_segment(ax, az, bx, bz, dx, dz, length_eps) ||
         on_segment(cx, cz, dx, dz, ax, az, length_eps) ||
         on_segment(cx, cz, dx, dz, bx, bz, length_eps);
}

inline bool simple_loop_first_violation_hardened(const std::vector<double>& r,
                                                 const std::vector<double>& z,
                                                 const std::vector<int>& nodes,
                                                 int* viol_a,
                                                 int* viol_b) {
  const LoopNodeAliasMap alias = canonicalize_loop_node_aliases(r, z, nodes);
  const double eps = alias.epsilon;
  const double orient_eps =
      eps * std::max(loop_coordinate_scale(r, z, nodes), eps);
  const auto axis_coincident = [&](const int a, const int b) {
    return loop_nodes_coincident(r, z, a, b, eps) &&
           std::abs(r[static_cast<std::size_t>(a)]) <= 10.0 * eps &&
           std::abs(r[static_cast<std::size_t>(b)]) <= 10.0 * eps;
  };
  const auto& canonical_nodes = alias.loop_nodes;
  const auto& canonical_segment_index = alias.source_segment_index;
  const int n = static_cast<int>(canonical_nodes.size());
  if (n < 3) {
    return false;
  }
  for (int a = 0; a < n; ++a) {
    const int a0 = canonical_nodes[static_cast<std::size_t>(a)];
    const int a1 = canonical_nodes[static_cast<std::size_t>((a + 1) % n)];
    for (int b = a + 1; b < n; ++b) {
      if (b == a || b == (a + 1) % n || (b + 1) % n == a) {
        continue;
      }
      const int b0 = canonical_nodes[static_cast<std::size_t>(b)];
      const int b1 = canonical_nodes[static_cast<std::size_t>((b + 1) % n)];
      if (axis_coincident(a0, b0) || axis_coincident(a0, b1) ||
          axis_coincident(a1, b0) || axis_coincident(a1, b1)) {
        continue;
      }
      if (segments_intersect(r[static_cast<std::size_t>(a0)],
                             z[static_cast<std::size_t>(a0)],
                             r[static_cast<std::size_t>(a1)],
                             z[static_cast<std::size_t>(a1)],
                             r[static_cast<std::size_t>(b0)],
                             z[static_cast<std::size_t>(b0)],
                             r[static_cast<std::size_t>(b1)],
                             z[static_cast<std::size_t>(b1)],
                             eps,
                             orient_eps)) {
        if (viol_a != nullptr) {
          *viol_a = canonical_segment_index[static_cast<std::size_t>(a)];
        }
        if (viol_b != nullptr) {
          *viol_b = canonical_segment_index[static_cast<std::size_t>(b)];
        }
        return false;
      }
    }
  }
  return true;
}

inline bool loop_node_on_axis(const std::vector<double>& r,
                              const int node,
                              const double eps) {
  return node >= 0 && static_cast<std::size_t>(node) < r.size() &&
         std::abs(r[static_cast<std::size_t>(node)]) <= 10.0 * eps;
}

inline bool loop_edge_on_axis(const std::vector<double>& r,
                              const int a,
                              const int b,
                              const double eps) {
  return loop_node_on_axis(r, a, eps) && loop_node_on_axis(r, b, eps);
}

inline bool loop_edge_longer_than(const std::vector<double>& r,
                                  const std::vector<double>& z,
                                  const int a,
                                  const int b,
                                  const double eps) {
  return distance2d(r[static_cast<std::size_t>(a)],
                    z[static_cast<std::size_t>(a)],
                    r[static_cast<std::size_t>(b)],
                    z[static_cast<std::size_t>(b)]) > eps;
}

inline void log_pole_macro_simple_diag(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes,
    const int edge_a,
    const int edge_b,
    const double length_eps,
    const double orient_eps,
    const double coord_scale) {
  const int n = static_cast<int>(nodes.size());
  if (n <= 0 || edge_a < 0 || edge_b < 0 ||
      edge_a >= n || edge_b >= n) {
    return;
  }
  const int a0 = nodes[static_cast<std::size_t>(edge_a)];
  const int a1 = nodes[static_cast<std::size_t>((edge_a + 1) % n)];
  const int b0 = nodes[static_cast<std::size_t>(edge_b)];
  const int b1 = nodes[static_cast<std::size_t>((edge_b + 1) % n)];
  const double ar0 = r[static_cast<std::size_t>(a0)];
  const double az0 = z[static_cast<std::size_t>(a0)];
  const double ar1 = r[static_cast<std::size_t>(a1)];
  const double az1 = z[static_cast<std::size_t>(a1)];
  const double br0 = r[static_cast<std::size_t>(b0)];
  const double bz0 = z[static_cast<std::size_t>(b0)];
  const double br1 = r[static_cast<std::size_t>(b1)];
  const double bz1 = z[static_cast<std::size_t>(b1)];
  const bool loop_adjacent =
      edge_b == edge_a || edge_b == (edge_a + 1) % n ||
      (edge_b + 1) % n == edge_a;
  const bool shared_node =
      a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1;
  const bool topologically_adjacent = loop_adjacent || shared_node;
  const double o1 = orient2d(ar0, az0, ar1, az1, br0, bz0);
  const double o2 = orient2d(ar0, az0, ar1, az1, br1, bz1);
  const double o3 = orient2d(br0, bz0, br1, bz1, ar0, az0);
  const double o4 = orient2d(br0, bz0, br1, bz1, ar1, az1);
  const int s1 = orient2d_long_double_sign(ar0, az0, ar1, az1, br0, bz0);
  const int s2 = orient2d_long_double_sign(ar0, az0, ar1, az1, br1, bz1);
  const int s3 = orient2d_long_double_sign(br0, bz0, br1, bz1, ar0, az0);
  const int s4 = orient2d_long_double_sign(br0, bz0, br1, bz1, ar1, az1);
  const bool long_double_proper_cross =
      s1 * s2 < 0 && s3 * s4 < 0;

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(17)
      << "[pole_macro_simple_diag]"
      << " edge_a=" << edge_a
      << " edge_b=" << edge_b
      << " nodes=" << a0 << "," << a1 << "," << b0 << "," << b1
      << " A0=(" << ar0 << "," << az0 << ")"
      << " A1=(" << ar1 << "," << az1 << ")"
      << " B0=(" << br0 << "," << bz0 << ")"
      << " B1=(" << br1 << "," << bz1 << ")"
      << " topologically_adjacent="
      << (topologically_adjacent ? 1 : 0)
      << " loop_adjacent=" << (loop_adjacent ? 1 : 0)
      << " shared_node=" << (shared_node ? 1 : 0)
      << " o_double=" << o1 << "," << o2 << "," << o3 << "," << o4
      << " orient_eps=" << orient_eps
      << " loop_coincidence_epsilon=" << length_eps
      << " coord_scale=" << coord_scale
      << " orient_sign_long_double=" << s1 << "," << s2 << "," << s3
      << "," << s4
      << " long_double_proper_cross="
      << (long_double_proper_cross ? 1 : 0);
  ::tenryu::core::log_warning(oss.str());
}

inline bool pole_macro_axis_touching_wedge(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes) {
  if (n_j_cells <= 0 || macro.local_i_end != macro.local_i_begin + 1 ||
      !(macro.local_j_begin == 0 || macro.local_j_end == n_j_cells) ||
      nodes.size() < 4U) {
    return false;
  }
  const double eps = loop_coincidence_epsilon(r, z, nodes);
  int axis_nodes = 0;
  int finite_axis_edges = 0;
  for (const int node : nodes) {
    if (loop_node_on_axis(r, node, eps)) {
      ++axis_nodes;
    }
  }
  for (int k = 0; k < static_cast<int>(nodes.size()); ++k) {
    const int a = nodes[static_cast<std::size_t>(k)];
    const int b = nodes[static_cast<std::size_t>((k + 1) % nodes.size())];
    if (loop_edge_on_axis(r, a, b, eps) &&
        loop_edge_longer_than(r, z, a, b, eps)) {
      ++finite_axis_edges;
    }
  }
  return axis_nodes >= 2 && finite_axis_edges == 1;
}

inline bool simple_loop_first_violation_axis_touching_wedge(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes,
    int* viol_a,
    int* viol_b) {
  const double eps = loop_coincidence_epsilon(r, z, nodes);
  const double coord_scale = loop_coordinate_scale(r, z, nodes);
  const double orient_eps =
      eps * std::max(coord_scale, eps);
  const int n = static_cast<int>(nodes.size());
  if (n < 3) {
    return false;
  }
  for (int a = 0; a < n; ++a) {
    const int a0 = nodes[static_cast<std::size_t>(a)];
    const int a1 = nodes[static_cast<std::size_t>((a + 1) % n)];
    for (int b = a + 1; b < n; ++b) {
      if (b == a || b == (a + 1) % n || (b + 1) % n == a) {
        continue;
      }
      const int b0 = nodes[static_cast<std::size_t>(b)];
      const int b1 = nodes[static_cast<std::size_t>((b + 1) % n)];
      if (segments_intersect(r[static_cast<std::size_t>(a0)],
                             z[static_cast<std::size_t>(a0)],
                             r[static_cast<std::size_t>(a1)],
                             z[static_cast<std::size_t>(a1)],
                             r[static_cast<std::size_t>(b0)],
                             z[static_cast<std::size_t>(b0)],
                             r[static_cast<std::size_t>(b1)],
                             z[static_cast<std::size_t>(b1)],
                             eps,
                             orient_eps)) {
        if (path_admissibility_anatomy_enabled()) {
          log_pole_macro_simple_diag(
              r, z, nodes, a, b, eps, orient_eps, coord_scale);
        }
        if (viol_a != nullptr) {
          *viol_a = a;
        }
        if (viol_b != nullptr) {
          *viol_b = b;
        }
        return false;
      }
    }
  }
  return true;
}

inline bool on_segment_legacy(const double ax,
                              const double az,
                              const double bx,
                              const double bz,
                              const double px,
                              const double pz) {
  constexpr double eps = 1.0e-14;
  return px >= std::min(ax, bx) - eps && px <= std::max(ax, bx) + eps &&
         pz >= std::min(az, bz) - eps && pz <= std::max(az, bz) + eps &&
         std::abs(orient2d(ax, az, bx, bz, px, pz)) <= eps;
}

inline bool segments_intersect_legacy(const double ax,
                                      const double az,
                                      const double bx,
                                      const double bz,
                                      const double cx,
                                      const double cz,
                                      const double dx,
                                      const double dz) {
  const double o1 = orient2d(ax, az, bx, bz, cx, cz);
  const double o2 = orient2d(ax, az, bx, bz, dx, dz);
  const double o3 = orient2d(cx, cz, dx, dz, ax, az);
  const double o4 = orient2d(cx, cz, dx, dz, bx, bz);
  constexpr double eps = 1.0e-14;
  if (((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
      ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))) {
    return true;
  }
  return on_segment_legacy(ax, az, bx, bz, cx, cz) ||
         on_segment_legacy(ax, az, bx, bz, dx, dz) ||
         on_segment_legacy(cx, cz, dx, dz, ax, az) ||
         on_segment_legacy(cx, cz, dx, dz, bx, bz);
}

inline bool simple_loop_first_violation_legacy(const std::vector<double>& r,
                                               const std::vector<double>& z,
                                               const std::vector<int>& nodes,
                                               int* viol_a,
                                               int* viol_b) {
  const int n = static_cast<int>(nodes.size());
  if (n < 3) {
    return false;
  }
  for (int a = 0; a < n; ++a) {
    const int a0 = nodes[static_cast<std::size_t>(a)];
    const int a1 = nodes[static_cast<std::size_t>((a + 1) % n)];
    for (int b = a + 1; b < n; ++b) {
      if (b == a || b == (a + 1) % n || (b + 1) % n == a) {
        continue;
      }
      const int b0 = nodes[static_cast<std::size_t>(b)];
      const int b1 = nodes[static_cast<std::size_t>((b + 1) % n)];
      if (segments_intersect_legacy(r[static_cast<std::size_t>(a0)],
                                    z[static_cast<std::size_t>(a0)],
                                    r[static_cast<std::size_t>(a1)],
                                    z[static_cast<std::size_t>(a1)],
                                    r[static_cast<std::size_t>(b0)],
                                    z[static_cast<std::size_t>(b0)],
                                    r[static_cast<std::size_t>(b1)],
                                    z[static_cast<std::size_t>(b1)])) {
        if (viol_a != nullptr) {
          *viol_a = a;
        }
        if (viol_b != nullptr) {
          *viol_b = b;
        }
        return false;
      }
    }
  }
  return true;
}

inline bool simple_loop_first_violation(const std::vector<double>& r,
                                        const std::vector<double>& z,
                                        const std::vector<int>& nodes,
                                        int* viol_a,
                                        int* viol_b,
                                        const bool harden) {
  return harden
             ? simple_loop_first_violation_hardened(r, z, nodes, viol_a, viol_b)
             : simple_loop_first_violation_legacy(r, z, nodes, viol_a, viol_b);
}

inline bool simple_loop_first_violation(const std::vector<double>& r,
                                        const std::vector<double>& z,
                                        const std::vector<int>& nodes,
                                        int* viol_a,
                                        int* viol_b) {
  return simple_loop_first_violation(
      r, z, nodes, viol_a, viol_b, path_predicate_harden_enabled());
}

inline bool simple_loop(const std::vector<double>& r,
                        const std::vector<double>& z,
                        const std::vector<int>& nodes) {
  return simple_loop_first_violation(r, z, nodes, nullptr, nullptr);
}

inline int macro_central_core_block_id(const MultiBlockTopology& mb) {
  int block_id = -1;
  for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
    if (mb.blocks[static_cast<std::size_t>(b)].role ==
        BlockRole::CENTRAL_CORE) {
      if (block_id >= 0) {
        return -1;
      }
      block_id = b;
    }
  }
  return block_id;
}

inline int macro_boundary_node_for_g(const tenryu::core::State& state,
                                     const int g) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return -1;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (!mb.has_trifan_cap || mb.n_cap <= 0) {
    return -1;
  }
  const int core_block_id = macro_central_core_block_id(mb);
  if (core_block_id < 0) {
    return -1;
  }
  const auto& core = mb.blocks[static_cast<std::size_t>(core_block_id)];
  const int ntheta = core.n_j_cells;
  if (ntheta != 4 * mb.n_cap || g < 0 || g > ntheta) {
    return -1;
  }
  const auto& north = mesh_topo_trifan_fan_block(mb, BlockRole::NORTH_FAN);
  const int D = state.central_pseudo_core.member_ring_count;
  const int n_b = north.n_i_cells;
  const int L = std::min(std::max(0, D - mb.n_cap), n_b);
  const int S = std::max(0, D - mb.n_cap - n_b);
  if (D <= 0) {
    return -1;
  }
  if (S > 0) {
    const auto& shell = mesh_topo_multiblock_polar_shell_block(mb);
    if (shell.n_j_cells != ntheta || S > shell.n_i_cells) {
      return -1;
    }
    return shell.owned_node_begin + S * (ntheta + 1) + g;
  }
  if (L == 0) {
    if (D > mb.n_cap) {
      return -1;
    }
    return mesh_topo_cap_ring_node_id(mb, D, g);
  }
  if (g <= mb.n_cap) {
    return mesh_topo_trifan_fan_node_id(mb, BlockRole::NORTH_FAN, L, g);
  }
  if (g <= 3 * mb.n_cap) {
    return mesh_topo_trifan_fan_node_id(
        mb, BlockRole::EAST_FAN, L, g - mb.n_cap);
  }
  return mesh_topo_trifan_fan_node_id(
      mb, BlockRole::SOUTH_FAN, L, g - 3 * mb.n_cap);
}

inline int macro_boundary_ntheta(const tenryu::core::State& state) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return -1;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (!mb.has_trifan_cap) {
    return -1;
  }
  const int core_block_id = macro_central_core_block_id(mb);
  if (core_block_id < 0) {
    return -1;
  }
  return mb.blocks[static_cast<std::size_t>(core_block_id)].n_j_cells;
}

inline int macro_ray_segment_crossing_count(const std::vector<int>& loop,
                                            const std::vector<double>& r,
                                            const std::vector<double>& z,
                                            const double theta,
                                            const double tol) {
  if (loop.size() < 2U) {
    return 0;
  }
  const double er = std::sin(theta);
  const double ez = std::cos(theta);
  int count = 0;
  for (std::size_t k = 0; k < loop.size(); ++k) {
    const int n0 = loop[k];
    const int n1 = loop[(k + 1U) % loop.size()];
    const double p0r = r[static_cast<std::size_t>(n0)];
    const double p0z = z[static_cast<std::size_t>(n0)];
    const double dr = r[static_cast<std::size_t>(n1)] - p0r;
    const double dz = z[static_cast<std::size_t>(n1)] - p0z;
    const double denom = dr * ez - dz * er;
    const double p_cross_e = p0r * ez - p0z * er;
    if (std::abs(denom) <= tol) {
      const double a0 = p0r * er + p0z * ez;
      const double a1 =
          r[static_cast<std::size_t>(n1)] * er +
          z[static_cast<std::size_t>(n1)] * ez;
      if (std::abs(p_cross_e) <= tol && std::max(a0, a1) > tol) {
        ++count;
      }
      continue;
    }
    const double t = -p_cross_e / denom;
    if (t < -tol || t >= 1.0 - tol) {
      continue;
    }
    const double ir = p0r + t * dr;
    const double iz = p0z + t * dz;
    const double a = ir * er + iz * ez;
    if (a > tol) {
      ++count;
    }
  }
  return count;
}

inline std::vector<int> canonicalized_loop_nodes(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes) {
  return canonicalize_loop_node_aliases(r, z, nodes).loop_nodes;
}

inline void macro_ray_segment_crossing_distances(
    const std::vector<int>& loop,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const double theta,
    const double tol,
    std::vector<double>& crossings) {
  crossings.clear();
  if (loop.size() < 2U) {
    return;
  }
  const double er = std::sin(theta);
  const double ez = std::cos(theta);
  for (std::size_t k = 0; k < loop.size(); ++k) {
    const int n0 = loop[k];
    const int n1 = loop[(k + 1U) % loop.size()];
    const double p0r = r[static_cast<std::size_t>(n0)];
    const double p0z = z[static_cast<std::size_t>(n0)];
    const double p1r = r[static_cast<std::size_t>(n1)];
    const double p1z = z[static_cast<std::size_t>(n1)];
    const double dr = p1r - p0r;
    const double dz = p1z - p0z;
    if (dr * dr + dz * dz <= tol * tol) {
      continue;
    }
    const double denom = dr * ez - dz * er;
    const double p_cross_e = p0r * ez - p0z * er;
    if (std::abs(denom) <= tol) {
      continue;
    }
    const double t = -p_cross_e / denom;
    if (t < -tol || t >= 1.0 - tol) {
      continue;
    }
    const double ir = p0r + t * dr;
    const double iz = p0z + t * dz;
    const double a = ir * er + iz * ez;
    if (a > tol && std::isfinite(a)) {
      crossings.push_back(a);
    }
  }
  std::sort(crossings.begin(), crossings.end());
  std::vector<double> unique;
  unique.reserve(crossings.size());
  for (const double a : crossings) {
    if (unique.empty() || std::abs(a - unique.back()) > tol) {
      unique.push_back(a);
    }
  }
  crossings.swap(unique);
}

inline bool pole_macro_boundary_star_simple(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes,
    int* first_bad_j) {
  if (first_bad_j != nullptr) {
    *first_bad_j = -1;
  }
  if (n_j_cells <= 0 ||
      macro.local_j_end <= macro.local_j_begin ||
      !(macro.local_j_begin == 0 || macro.local_j_end == n_j_cells)) {
    return false;
  }
  const LoopNodeAliasMap alias = canonicalize_loop_node_aliases(
      r, z, nodes,
      macro.single_apex_boundary ? macro.canonical_apex_node : -1,
      macro.single_apex_boundary);
  std::vector<int> loop = alias.loop_nodes;
  if (loop.size() < 3U) {
    return false;
  }
  if (macro.single_apex_boundary) {
    if (macro.canonical_apex_node < 0) {
      return false;
    }
    const auto apex_it =
        std::find(loop.begin(), loop.end(), macro.canonical_apex_node);
    if (apex_it == loop.end()) {
      return false;
    }
    std::rotate(loop.begin(), apex_it, loop.end());
    int apex_count = 0;
    for (const int node : loop) {
      if (node == macro.canonical_apex_node) {
        ++apex_count;
      }
    }
    if (apex_count != 1) {
      return false;
    }
  }
  const double coord_scale = loop_coordinate_scale(r, z, loop);
  const double tol =
      std::max(1.0e-14, 1.0e-10 * std::max(coord_scale, 1.0e-30));
  constexpr double kPi = 3.141592653589793238462643383279502884;
  std::vector<double> crossings;
  for (int j = macro.local_j_begin + 1; j < macro.local_j_end; ++j) {
    const double theta = kPi * static_cast<double>(j) /
                         static_cast<double>(n_j_cells);
    macro_ray_segment_crossing_distances(loop, r, z, theta, tol, crossings);
    if (crossings.size() != 2U || !(crossings[1] - crossings[0] > tol)) {
      if (first_bad_j != nullptr) {
        *first_bad_j = j;
      }
      return false;
    }
  }
  return true;
}

inline bool macro_boundary_star_simple(const tenryu::core::State& state,
                                       const std::vector<double>& r,
                                       const std::vector<double>& z,
                                       const std::vector<int>& nodes,
                                       int* first_bad_g) {
  if (first_bad_g != nullptr) {
    *first_bad_g = -1;
  }
  if (!state.mesh.topo.multiblock.has_value()) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (!mb.has_trifan_cap) {
    return false;
  }
  const int core_block_id = macro_central_core_block_id(mb);
  if (core_block_id < 0) {
    return false;
  }
  const int ntheta =
      mb.blocks[static_cast<std::size_t>(core_block_id)].n_j_cells;
  if (ntheta <= 0 || ntheta != 4 * mb.n_cap) {
    return false;
  }
  const int n_nodes = static_cast<int>(r.size());
  const LoopNodeAliasMap alias = canonicalize_loop_node_aliases(r, z, nodes);
  const std::vector<int>& loop = alias.loop_nodes;
  if (loop.size() < 3U) {
    return false;
  }
  std::vector<int> node_to_g(static_cast<std::size_t>(n_nodes), -1);
  const double dtheta = 3.141592653589793238462643383279502884 /
                        static_cast<double>(ntheta);
  double coord_scale = 0.0;
  for (const int n : nodes) {
    if (n >= 0 && n < n_nodes) {
      coord_scale =
          std::max(coord_scale,
                   std::hypot(r[static_cast<std::size_t>(n)],
                              z[static_cast<std::size_t>(n)]));
    }
  }
  const double tol =
      std::max(1.0e-14, 1.0e-10 * std::max(coord_scale, 1.0e-30));
  const auto& boundary_node_mask =
      state.central_pseudo_core.boundary_node_mask;
  for (int g = 0; g <= ntheta; ++g) {
    const int n = macro_boundary_node_for_g(state, g);
    if (n < 0 || n >= n_nodes || node_to_g[static_cast<std::size_t>(n)] >= 0) {
      if (first_bad_g != nullptr) {
        *first_bad_g = g;
      }
      return false;
    }
    node_to_g[static_cast<std::size_t>(n)] = g;
    if (boundary_node_mask.size() == static_cast<std::size_t>(n_nodes) &&
        boundary_node_mask[static_cast<std::size_t>(n)] == 0U) {
      if (first_bad_g != nullptr) {
        *first_bad_g = g;
      }
      return false;
    }
    const double theta = dtheta * static_cast<double>(g);
    const double er = std::sin(theta);
    const double ez = std::cos(theta);
    const double ag =
        r[static_cast<std::size_t>(n)] * er +
        z[static_cast<std::size_t>(n)] * ez;
    const double perp =
        r[static_cast<std::size_t>(n)] * ez -
        z[static_cast<std::size_t>(n)] * er;
    if (!(std::isfinite(ag) && ag > 0.0 && std::isfinite(perp))) {
      if (first_bad_g != nullptr) {
        *first_bad_g = g;
      }
      return false;
    }
    if (g > 0 && g < ntheta &&
        macro_ray_segment_crossing_count(loop, r, z, theta, tol) != 1) {
      if (first_bad_g != nullptr) {
        *first_bad_g = g;
      }
      return false;
    }
  }

  int prev_g = -1;
  int graph_g_order_violations = 0;
  int graph_g_wrap_jumps = 0;
  int graph_mapped_nodes = 0;
  for (const int n : loop) {
    int g = -1;
    if (n >= 0 && n < n_nodes) {
      g = node_to_g[static_cast<std::size_t>(n)];
    }
    if (g < 0) {
      continue;
    }
    ++graph_mapped_nodes;
    if (prev_g >= 0) {
      const int delta = std::abs(g - prev_g);
      if (delta == ntheta) {
        ++graph_g_wrap_jumps;
      } else if (delta != 1) {
        ++graph_g_order_violations;
      }
    }
    prev_g = g;
  }
  if (graph_mapped_nodes < ntheta || graph_mapped_nodes > ntheta + 1 ||
      graph_g_order_violations != 0 || graph_g_wrap_jumps > 1) {
    if (first_bad_g != nullptr) {
      *first_bad_g = prev_g;
    }
    return false;
  }
  return true;
}

inline double cubic_value(const double c0,
                          const double c1,
                          const double c2,
                          const double c3,
                          const double lambda) {
  return c0 + lambda * (c1 + lambda * (c2 + lambda * c3));
}

inline void observe_cubic_min_candidate(const double c0,
                                        const double c1,
                                        const double c2,
                                        const double c3,
                                        const double lambda,
                                        double& min_value,
                                        double& min_lambda) {
  if (!(lambda >= 0.0 && lambda <= 1.0)) {
    return;
  }
  const double value = cubic_value(c0, c1, c2, c3, lambda);
  if (value < min_value) {
    min_value = value;
    min_lambda = lambda;
  }
}

inline void cubic_min_on_unit_interval(const double c0,
                                       const double c1,
                                       const double c2,
                                       const double c3,
                                       double& min_value,
                                       double& min_lambda) {
  min_value = cubic_value(c0, c1, c2, c3, 0.0);
  min_lambda = 0.0;
  observe_cubic_min_candidate(c0, c1, c2, c3, 1.0, min_value, min_lambda);

  const double a = 3.0 * c3;
  const double b = 2.0 * c2;
  const double c = c1;
  const double scale =
      std::max({std::abs(a), std::abs(b), std::abs(c), 1.0});
  constexpr double eps = 1.0e-14;
  if (std::abs(a) <= eps * scale) {
    if (std::abs(b) > eps * scale) {
      observe_cubic_min_candidate(c0, c1, c2, c3, -c / b, min_value,
                                  min_lambda);
    }
    return;
  }
  const double disc = b * b - 4.0 * a * c;
  if (disc < -eps * std::max(b * b, std::abs(4.0 * a * c))) {
    return;
  }
  const double sqrt_disc = std::sqrt(std::max(0.0, disc));
  observe_cubic_min_candidate(c0, c1, c2, c3, (-b - sqrt_disc) / (2.0 * a),
                              min_value, min_lambda);
  observe_cubic_min_candidate(c0, c1, c2, c3, (-b + sqrt_disc) / (2.0 * a),
                              min_value, min_lambda);
}

inline bool rz_polygon_volume_path_coefficients(
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const std::vector<int>& nodes,
    double coeff[4]) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  coeff[0] = 0.0;
  coeff[1] = 0.0;
  coeff[2] = 0.0;
  coeff[3] = 0.0;
  const int n = static_cast<int>(nodes.size());
  if (n < 3) {
    return false;
  }
  for (int k = 0; k < n; ++k) {
    const int a = nodes[static_cast<std::size_t>(k)];
    const int b = nodes[static_cast<std::size_t>((k + 1) % n)];
    const double ra0 = xr_old[static_cast<std::size_t>(a)];
    const double za0 = xz_old[static_cast<std::size_t>(a)];
    const double rb0 = xr_old[static_cast<std::size_t>(b)];
    const double zb0 = xz_old[static_cast<std::size_t>(b)];
    const double dra = xr_new[static_cast<std::size_t>(a)] - ra0;
    const double dza = xz_new[static_cast<std::size_t>(a)] - za0;
    const double drb = xr_new[static_cast<std::size_t>(b)] - rb0;
    const double dzb = xz_new[static_cast<std::size_t>(b)] - zb0;
    if (!std::isfinite(ra0) || !std::isfinite(za0) ||
        !std::isfinite(rb0) || !std::isfinite(zb0) ||
        !std::isfinite(dra) || !std::isfinite(dza) ||
        !std::isfinite(drb) || !std::isfinite(dzb)) {
      return false;
    }
    const double cross0 = ra0 * zb0 - rb0 * za0;
    const double cross1 = dra * zb0 + ra0 * dzb - drb * za0 - rb0 * dza;
    const double cross2 = dra * dzb - drb * dza;
    const double sum0 = ra0 + rb0;
    const double sum1 = dra + drb;
    coeff[0] += sum0 * cross0;
    coeff[1] += sum0 * cross1 + sum1 * cross0;
    coeff[2] += sum0 * cross2 + sum1 * cross1;
    coeff[3] += sum1 * cross2;
  }
  for (int k = 0; k < 4; ++k) {
    coeff[k] *= kPi / 3.0;
  }
  return std::isfinite(coeff[0]) && std::isfinite(coeff[1]) &&
         std::isfinite(coeff[2]) && std::isfinite(coeff[3]);
}

inline double first_cubic_floor_root_sampled(const double c0,
                                             const double c1,
                                             const double c2,
                                             const double c3,
                                             const double floor) {
  const auto f = [&](const double lambda) {
    return cubic_value(c0, c1, c2, c3, lambda) - floor;
  };
  double prev_lambda = 0.0;
  if (!(f(prev_lambda) > 0.0)) {
    return 0.0;
  }
  constexpr int kSamples = 64;
  for (int s = 1; s <= kSamples; ++s) {
    const double lambda = static_cast<double>(s) / kSamples;
    const double value = f(lambda);
    if (!(value > 0.0)) {
      double lo = prev_lambda;
      double hi = lambda;
      for (int iter = 0; iter < 60; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return hi;
    }
    prev_lambda = lambda;
  }
  return 1.0;
}

inline bool pole_coarsen_macro_boundary_simple(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<int>& nodes,
    int* viol_a,
    int* viol_b,
    const bool harden) {
  if (harden &&
      pole_macro_axis_touching_wedge(macro, n_j_cells, r, z, nodes)) {
    return simple_loop_first_violation_axis_touching_wedge(
        r, z, nodes, viol_a, viol_b);
  }
  int first_bad_j = -1;
  const bool star_simple =
      harden && macro.single_apex_boundary && pole_macro_boundary_star_simple(
                    macro, n_j_cells, r, z, nodes, &first_bad_j);
  if (harden && macro.single_apex_boundary) {
    if (viol_a != nullptr) {
      *viol_a = -1;
    }
    if (viol_b != nullptr) {
      *viol_b = -1;
    }
    (void)first_bad_j;
    return star_simple;
  }
  const bool segment_simple =
      simple_loop_first_violation(r, z, nodes, viol_a, viol_b, harden);
  (void)first_bad_j;
  return segment_simple;
}

inline bool pole_coarsen_macro_boundary_path_first_violation(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const std::vector<int>& nodes,
    double& lambda,
    int* viol_a,
    int* viol_b,
    const bool harden) {
  constexpr int kSamples = 64;
  std::vector<double> r(xr_old.size(), 0.0);
  std::vector<double> z(xz_old.size(), 0.0);
  for (int s = 0; s <= kSamples; ++s) {
    lambda = static_cast<double>(s) / kSamples;
    for (const int node : nodes) {
      const std::size_t i = static_cast<std::size_t>(node);
      r[i] = xr_old[i] + lambda * (xr_new[i] - xr_old[i]);
      z[i] = xz_old[i] + lambda * (xz_new[i] - xz_old[i]);
    }
    if (!pole_coarsen_macro_boundary_simple(
            macro, n_j_cells, r, z, nodes, viol_a, viol_b, harden)) {
      return true;
    }
  }
  return false;
}

inline void set_pole_coarsen_macro_anatomy(
    PathAdmissibilityResult& out,
    const tenryu::hydro::pole_angular_coarsen::Macro& macro) {
  out.path_source_kind =
      static_cast<int>(PathAdmissibilitySourceKind::POLE_MACRO_BOUNDARY);
  out.macro_has_anatomy = 1;
  out.first_failing_block_id = macro.block_id;
  out.first_failing_local_i = macro.local_i_begin;
  out.first_failing_local_j = macro.local_j_begin;
  out.macro_span = macro.span;
  out.macro_level = macro.level;
  out.macro_q_begin = macro.local_i_begin;
  out.macro_q_end = macro.local_i_end - 1;
  out.macro_j_begin = macro.local_j_begin;
  out.macro_j_end = macro.local_j_end;
  out.macro_skipped_nodes = macro.skipped_nodes;
}

inline double pole_coarsen_segment_mid_theta(
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const std::vector<int>& nodes,
    const int seg,
    const double lambda,
    int& node_out) {
  const int n_loop = static_cast<int>(nodes.size());
  const int s0 = nodes[static_cast<std::size_t>(seg)];
  const int s1 = nodes[static_cast<std::size_t>((seg + 1) % n_loop)];
  node_out = s0;
  const double r0 = xr_old[static_cast<std::size_t>(s0)] +
                    lambda * (xr_new[static_cast<std::size_t>(s0)] -
                              xr_old[static_cast<std::size_t>(s0)]);
  const double r1 = xr_old[static_cast<std::size_t>(s1)] +
                    lambda * (xr_new[static_cast<std::size_t>(s1)] -
                              xr_old[static_cast<std::size_t>(s1)]);
  const double z0 = xz_old[static_cast<std::size_t>(s0)] +
                    lambda * (xz_new[static_cast<std::size_t>(s0)] -
                              xz_old[static_cast<std::size_t>(s0)]);
  const double z1 = xz_old[static_cast<std::size_t>(s1)] +
                    lambda * (xz_new[static_cast<std::size_t>(s1)] -
                              xz_old[static_cast<std::size_t>(s1)]);
  return std::atan2(0.5 * (r0 + r1), 0.5 * (z0 + z1));
}

inline bool pole_axis_radial_order_inversion_at_lambda0(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<int>& nodes) {
  if (n_j_cells <= 0 || macro.local_i_end != macro.local_i_begin + 1 ||
      !(macro.local_j_begin == 0 || macro.local_j_end == n_j_cells) ||
      nodes.size() < 4U || xr_old.size() != xz_old.size()) {
    return false;
  }
  const double eps = loop_coincidence_epsilon(xr_old, xz_old, nodes);
  std::vector<int> axis_nodes;
  axis_nodes.reserve(2U);
  for (const int node : nodes) {
    if (!loop_node_on_axis(xr_old, node, eps)) {
      continue;
    }
    if (std::find(axis_nodes.begin(), axis_nodes.end(), node) ==
        axis_nodes.end()) {
      axis_nodes.push_back(node);
    }
  }
  if (axis_nodes.size() != 2U) {
    return false;
  }
  std::sort(axis_nodes.begin(), axis_nodes.end());
  const int stride = n_j_cells + 1;
  if (axis_nodes[1] - axis_nodes[0] != stride) {
    return false;
  }
  const int inner = axis_nodes[0];
  const int outer = axis_nodes[1];
  if (inner < 0 || outer < 0 ||
      static_cast<std::size_t>(outer) >= xr_old.size()) {
    return false;
  }
  const double zi = xz_old[static_cast<std::size_t>(inner)];
  const double zo = xz_old[static_cast<std::size_t>(outer)];
  if (!(std::isfinite(zi) && std::isfinite(zo)) || zi * zo <= 0.0) {
    return false;
  }
  const double si = distance2d(0.0,
                               0.0,
                               xr_old[static_cast<std::size_t>(inner)],
                               zi);
  const double so = distance2d(0.0,
                               0.0,
                               xr_old[static_cast<std::size_t>(outer)],
                               zo);
  const double floor =
      std::max(10.0 * eps, 1.0e-14 * std::max(1.0, std::max(si, so)));
  return std::isfinite(si) && std::isfinite(so) && !(si + floor < so);
}

inline void set_pole_coarsen_simple_loop_failure(
    PathAdmissibilityResult& out,
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const std::vector<int>& nodes,
    const double lambda,
    const int viol_a,
    const int viol_b,
    const bool allow_macro_repair) {
  out.macro_simple_ok = 0;
  out.first_failing_cell = kPoleAngularCoarsenSentinelCell;
  out.first_failing_lambda = lambda;
  out.first_failing_metric_kind =
      static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS);
  if (lambda == 0.0 && path_predicate_harden_enabled()) {
    out.old_geometry_inadmissible = 1;
    if (pole_axis_radial_order_inversion_at_lambda0(
            macro, n_j_cells, xr_old, xz_old, nodes)) {
      out.pole_axis_radial_order_inversion = 1;
    } else if (allow_macro_repair) {
      out.macro_repair_required = 1;
    }
  }
  if (viol_a >= 0 && viol_b >= 0) {
    out.macro_viol_theta_a =
        pole_coarsen_segment_mid_theta(xr_old, xz_old, xr_new, xz_new, nodes,
                                       viol_a, lambda,
                                       out.macro_viol_node_a);
    out.macro_viol_theta_b =
        pole_coarsen_segment_mid_theta(xr_old, xz_old, xr_new, xz_new, nodes,
                                       viol_b, lambda,
                                       out.macro_viol_node_b);
  }
}

inline PathAdmissibilityResult evaluate_pole_coarsen_macro_boundary_path_host(
    const tenryu::hydro::pole_angular_coarsen::Macro& macro,
    const int n_j_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    const bool allow_macro_repair,
    const std::vector<double>* xr_reference,
    const std::vector<double>* xz_reference) {
  PathAdmissibilityResult out;
  if (macro.boundary_nodes_ordered.size() < 3U) {
    return out;
  }
  const auto& raw_nodes = macro.boundary_nodes_ordered;
  const int n_nodes = static_cast<int>(xr_old.size());
  const bool harden = path_predicate_harden_enabled();
  set_pole_coarsen_macro_anatomy(out, macro);
  for (const int node : raw_nodes) {
    if (node < 0 || node >= n_nodes) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_cell = kPoleAngularCoarsenSentinelCell;
      out.first_failing_lambda = 0.0;
      out.first_failing_metric_kind =
          static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
      return out;
    }
  }
  LoopNodeAliasMap volume_alias;
  const std::vector<int>* volume_nodes = &raw_nodes;
  const bool axis_touching_wedge =
      harden && pole_macro_axis_touching_wedge(
                    macro, n_j_cells, xr_old, xz_old, raw_nodes);
  if (harden) {
    if (!axis_touching_wedge) {
      volume_alias = canonicalize_loop_node_aliases(
          xr_old, xz_old, raw_nodes,
          macro.single_apex_boundary ? macro.canonical_apex_node : -1,
          macro.single_apex_boundary);
      volume_nodes = &volume_alias.loop_nodes;
    }
    if (volume_nodes->size() < 3U) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_cell = kPoleAngularCoarsenSentinelCell;
      out.first_failing_lambda = 0.0;
      out.first_failing_metric_kind =
          static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
      out.old_geometry_inadmissible = 1;
      if (allow_macro_repair) {
        out.macro_repair_required = 1;
      }
      return out;
    }
  }

  double coeff[4] = {0.0, 0.0, 0.0, 0.0};
  const bool coeff_ok = rz_polygon_volume_path_coefficients(
      xr_old, xz_old, xr_new, xz_new, *volume_nodes, coeff);
  const double v_old = coeff[0];
  const double v_new = cubic_value(coeff[0], coeff[1], coeff[2], coeff[3], 1.0);
  const double sign = axis_touching_wedge ? 1.0 : (v_old >= 0.0 ? 1.0 : -1.0);
  double floor_baseline = std::abs(v_old);
  if (floor_rel > 0.0 && xr_reference != nullptr && xz_reference != nullptr &&
      xr_reference->size() == xr_old.size() &&
      xz_reference->size() == xz_old.size()) {
    const double v_reference =
        rz_polygon_volume(*xr_reference, *xz_reference, *volume_nodes);
    if (std::isfinite(v_reference) && sign * v_reference > 0.0) {
      floor_baseline = std::abs(v_reference);
    }
  }
  const double floor = std::max(0.0, floor_rel) * floor_baseline;
  out.macro_v_old = v_old;
  out.macro_v_new = v_new;
  out.macro_floor = floor;
  out.macro_simple_ok = 1;

  double min_v = HUGE_VAL;
  double min_lambda = 0.0;
  if (coeff_ok && std::isfinite(v_old) &&
      (axis_touching_wedge ? v_old > 0.0 : v_old != 0.0)) {
    cubic_min_on_unit_interval(sign * coeff[0], sign * coeff[1],
                               sign * coeff[2], sign * coeff[3], min_v,
                               min_lambda);
    out.min_margin = min_v - floor;
  } else {
    out.min_margin = -HUGE_VAL;
    out.first_failing_cell = kPoleAngularCoarsenSentinelCell;
    out.first_failing_lambda = 0.0;
    out.first_failing_metric_kind =
        static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
    if (harden) {
      out.old_geometry_inadmissible = 1;
      if (allow_macro_repair) {
        out.macro_repair_required = 1;
      }
    }
    return out;
  }

  int viol_a = -1;
  int viol_b = -1;
  if (!pole_coarsen_macro_boundary_simple(
          macro, n_j_cells, xr_old, xz_old, raw_nodes, &viol_a, &viol_b,
          harden)) {
    set_pole_coarsen_simple_loop_failure(
        out, macro, n_j_cells, xr_old, xz_old, xr_new, xz_new, raw_nodes,
        0.0, viol_a, viol_b, allow_macro_repair);
    return out;
  }
  viol_a = -1;
  viol_b = -1;
  if (!pole_coarsen_macro_boundary_simple(
          macro, n_j_cells, xr_new, xz_new, raw_nodes, &viol_a, &viol_b,
          harden)) {
    set_pole_coarsen_simple_loop_failure(
        out, macro, n_j_cells, xr_old, xz_old, xr_new, xz_new, raw_nodes,
        1.0, viol_a, viol_b, allow_macro_repair);
    return out;
  }

  if (!(min_v > floor)) {
    out.first_failing_cell = kPoleAngularCoarsenSentinelCell;
    out.first_failing_lambda = first_cubic_floor_root_sampled(
        sign * coeff[0], sign * coeff[1], sign * coeff[2], sign * coeff[3],
        floor);
    out.first_failing_metric_kind =
        static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
  }

  viol_a = -1;
  viol_b = -1;
  double loop_lambda = 1.0;
  const bool loop_failed = pole_coarsen_macro_boundary_path_first_violation(
      macro, n_j_cells, xr_old, xz_old, xr_new, xz_new, raw_nodes, loop_lambda,
      &viol_a, &viol_b, harden);
  out.macro_simple_ok = loop_failed ? 0 : 1;
  if (loop_failed) {
    if (out.first_failing_cell == -1 ||
        loop_lambda < out.first_failing_lambda) {
      set_pole_coarsen_simple_loop_failure(
          out, macro, n_j_cells, xr_old, xz_old, xr_new, xz_new, raw_nodes,
          loop_lambda, viol_a, viol_b, allow_macro_repair);
    } else if (viol_a >= 0 && viol_b >= 0) {
      out.macro_viol_theta_a =
          pole_coarsen_segment_mid_theta(xr_old, xz_old, xr_new, xz_new, raw_nodes,
                                         viol_a, loop_lambda,
                                         out.macro_viol_node_a);
      out.macro_viol_theta_b =
          pole_coarsen_segment_mid_theta(xr_old, xz_old, xr_new, xz_new, raw_nodes,
                                         viol_b, loop_lambda,
                                         out.macro_viol_node_b);
    }
  }
  return out;
}

inline bool pole_coarsen_failure_is_simple_loop(
    const PathAdmissibilityResult& result) {
  return result.first_failing_cell == kPoleAngularCoarsenSentinelCell &&
         result.first_failing_metric_kind ==
             static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS);
}

struct PoleCoarsenCoverStatus {
  bool covered = false;
  bool retryable_simple_failure = false;
  PathAdmissibilityResult failure;
};

inline PoleCoarsenCoverStatus cover_pole_coarsen_interval_host(
    const tenryu::hydro::pole_angular_coarsen::Overlay& source,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    const int j_begin,
    const int span,
    tenryu::hydro::pole_angular_coarsen::Overlay& accepted,
    const std::vector<double>* xr_reference,
    const std::vector<double>* xz_reference) {
  PoleCoarsenCoverStatus status;
  if (span < 2 || j_begin < 0 || j_begin + span > source.n_j_cells) {
    return status;
  }

  const std::size_t macro_mark = accepted.macros.size();
  if (span > 2) {
    auto left = cover_pole_coarsen_interval_host(
        source, xr_old, xz_old, xr_new, xz_new, floor_rel, j_begin, span / 2,
        accepted, xr_reference, xz_reference);
    if (left.covered) {
      auto right = cover_pole_coarsen_interval_host(
          source, xr_old, xz_old, xr_new, xz_new, floor_rel,
          j_begin + span / 2, span / 2, accepted, xr_reference, xz_reference);
      if (right.covered) {
        status.covered = true;
        return status;
      }
      if (!right.retryable_simple_failure) {
        accepted.macros.resize(macro_mark);
        return right;
      }
    } else if (!left.retryable_simple_failure) {
      accepted.macros.resize(macro_mark);
      return left;
    }
    accepted.macros.resize(macro_mark);
  }

  auto macro =
      tenryu::hydro::pole_angular_coarsen::make_macro(source, j_begin,
                                                      j_begin + span);
  const auto trial = evaluate_pole_coarsen_macro_boundary_path_host(
      macro,
      source.n_j_cells,
      xr_old,
      xz_old,
      xr_new,
      xz_new,
      floor_rel,
      source.supports_deref_macro_repair,
      xr_reference,
      xz_reference);
  if (trial.first_failing_cell == -1) {
    accepted.macros.push_back(std::move(macro));
    status.covered = true;
    return status;
  }
  status.failure = trial;
  status.retryable_simple_failure = pole_coarsen_failure_is_simple_loop(trial);
  return status;
}

inline PathAdmissibilityResult prepare_pole_coarsen_overlay_host(
    const tenryu::hydro::pole_angular_coarsen::Overlay* source,
    const int n_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    tenryu::hydro::pole_angular_coarsen::Overlay& accepted,
    const std::vector<double>* xr_reference = nullptr,
    const std::vector<double>* xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (source == nullptr || !source->active) {
    return out;
  }
  if (!source->macros.empty() &&
      source->pole_coarsen_inactive_fine_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    accepted = *source;
    for (const auto& macro : accepted.macros) {
      TENRYU_ASSERT(macro.local_i_end == macro.local_i_begin + 1,
                    "static pole shell angular de-refine macro must be one radial row");
    }
    return out;
  }
  accepted = *source;
  accepted.macros.clear();
  accepted.pole_coarsen_inactive_fine_mask.assign(
      static_cast<std::size_t>(n_cells), 0U);

  const int max_span = tenryu::hydro::pole_angular_coarsen::span_for_level(
      accepted.level_max);
  for (int j = 0; j < accepted.n_j_cells;) {
    int span = max_span;
    while (span > accepted.n_j_cells - j) {
      span /= 2;
    }
    if (span < 2) {
      break;
    }
    const auto status = cover_pole_coarsen_interval_host(
        *source, xr_old, xz_old, xr_new, xz_new, floor_rel, j, span,
        accepted, xr_reference, xz_reference);
    if (!status.covered) {
      for (const auto& macro : accepted.macros) {
        tenryu::hydro::pole_angular_coarsen::mark_macro_cells(
            accepted, macro, accepted.pole_coarsen_inactive_fine_mask);
      }
      return status.failure;
    }
    j += span;
  }

  for (const auto& macro : accepted.macros) {
    tenryu::hydro::pole_angular_coarsen::mark_macro_cells(
        accepted, macro, accepted.pole_coarsen_inactive_fine_mask);
  }
  return out;
}

inline PathAdmissibilityResult evaluate_pole_coarsen_boundary_path_host(
    const tenryu::hydro::pole_angular_coarsen::Overlay& overlay,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    const std::vector<double>* xr_reference = nullptr,
    const std::vector<double>* xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (!overlay.active || overlay.macros.empty()) {
    return out;
  }
  for (const auto& macro : overlay.macros) {
    merge_path_admissibility_result(
        out,
        evaluate_pole_coarsen_macro_boundary_path_host(
            macro, overlay.n_j_cells, xr_old, xz_old, xr_new, xz_new,
            floor_rel, overlay.supports_deref_macro_repair,
            xr_reference, xz_reference));
  }
  return out;
}

inline PathAdmissibilityResult evaluate_macro_boundary_path_host(
    const tenryu::core::State& state,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    const std::vector<double>* xr_reference = nullptr,
    const std::vector<double>* xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (!central_macro_path_active(state)) {
    return out;
  }
  out.path_source_kind =
      static_cast<int>(PathAdmissibilitySourceKind::CENTRAL_MACRO_BOUNDARY);
  const auto& raw_nodes = state.central_pseudo_core.boundary_nodes_ordered;
  const int n_nodes = static_cast<int>(xr_old.size());
  const bool harden = path_predicate_harden_enabled();
  for (const int node : raw_nodes) {
    if (node < 0 || node >= n_nodes) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_cell = kCentralMacroCoreSentinelCell;
      out.first_failing_lambda = 0.0;
      out.first_failing_metric_kind =
          static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
      return out;
    }
  }
  LoopNodeAliasMap volume_alias;
  const std::vector<int>* nodes = &raw_nodes;
  if (harden) {
    volume_alias = canonicalize_loop_node_aliases(xr_new, xz_new, raw_nodes);
    nodes = &volume_alias.loop_nodes;
    if (nodes->size() < 3U) {
      out.min_margin = -HUGE_VAL;
      out.first_failing_cell = kCentralMacroCoreSentinelCell;
      out.first_failing_lambda = 0.0;
      out.first_failing_metric_kind =
          static_cast<int>(PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
      return out;
    }
  }
  const double v_old = rz_polygon_volume(xr_old, xz_old, *nodes);
  const double v_new = rz_polygon_volume(xr_new, xz_new, *nodes);
  double floor_baseline = std::abs(v_old);
  if (floor_rel > 0.0 && xr_reference != nullptr && xz_reference != nullptr &&
      xr_reference->size() == xr_old.size() &&
      xz_reference->size() == xz_old.size()) {
    const double v_reference =
        rz_polygon_volume(*xr_reference, *xz_reference, *nodes);
    if (std::isfinite(v_reference) && v_reference > 0.0) {
      floor_baseline = v_reference;
    }
  }
  const double floor = std::max(0.0, floor_rel) * floor_baseline;
  out.min_margin = v_new - floor;
  int viol_a = -1;
  int viol_b = -1;
  int first_bad_g = -1;
  const bool star_simple =
      harden &&
      macro_boundary_star_simple(state, xr_new, xz_new, raw_nodes, &first_bad_g);
  const bool segment_simple =
      simple_loop_first_violation(
          xr_new, xz_new, raw_nodes, &viol_a, &viol_b, harden);
  const bool is_simple = harden ? (star_simple || segment_simple) : segment_simple;
  out.macro_has_anatomy = 1;
  out.macro_v_old = v_old;
  out.macro_v_new = v_new;
  out.macro_floor = floor;
  out.macro_simple_ok = is_simple ? 1 : 0;
  if (harden && !star_simple && first_bad_g >= 0) {
    out.macro_viol_node_a = macro_boundary_node_for_g(state, first_bad_g);
    const int ntheta = macro_boundary_ntheta(state);
    out.macro_viol_theta_a =
        3.141592653589793238462643383279502884 *
        static_cast<double>(first_bad_g) /
        std::max(1, ntheta);
  } else if (!is_simple && viol_a >= 0 && viol_b >= 0) {
    const int n_loop = static_cast<int>(raw_nodes.size());
    const auto segment_mid_theta = [&](const int seg, int& node_out) {
      const int s0 = raw_nodes[static_cast<std::size_t>(seg)];
      const int s1 = raw_nodes[static_cast<std::size_t>((seg + 1) % n_loop)];
      node_out = s0;
      const double rm = 0.5 * (xr_new[static_cast<std::size_t>(s0)] +
                               xr_new[static_cast<std::size_t>(s1)]);
      const double zm = 0.5 * (xz_new[static_cast<std::size_t>(s0)] +
                               xz_new[static_cast<std::size_t>(s1)]);
      return std::atan2(rm, zm);
    };
    out.macro_viol_theta_a = segment_mid_theta(viol_a, out.macro_viol_node_a);
    out.macro_viol_theta_b = segment_mid_theta(viol_b, out.macro_viol_node_b);
  }
  const bool ok = std::isfinite(v_old) && std::isfinite(v_new) &&
                  v_old > 0.0 && v_new > floor && is_simple;
  if (!ok) {
    out.first_failing_cell = kCentralMacroCoreSentinelCell;
    out.first_failing_lambda = 0.0;
    out.first_failing_metric_kind =
        (!is_simple ? static_cast<int>(PathAdmissibilityMetricKind::EDGE_CROSS)
                    : static_cast<int>(
                          PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME));
  }
  return out;
}

inline PathAdmissibilityResult evaluate_macro_boundary_path(
    const tenryu::core::State& state,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const double floor_rel,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (!central_macro_path_active(state)) {
    return out;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> xr_new;
  std::vector<double> xz_new;
  std::vector<double> xr_reference;
  std::vector<double> xz_reference;
  copy_device_node_field(d_xr_old, n_nodes, xr_old,
                         "path admissibility macro: copy old r failed");
  copy_device_node_field(d_xz_old, n_nodes, xz_old,
                         "path admissibility macro: copy old z failed");
  copy_device_node_field(d_xr_new, n_nodes, xr_new,
                         "path admissibility macro: copy new r failed");
  copy_device_node_field(d_xz_new, n_nodes, xz_new,
                         "path admissibility macro: copy new z failed");
  if (floor_rel > 0.0 && d_xr_reference != nullptr &&
      d_xz_reference != nullptr) {
    copy_device_node_field(
        d_xr_reference, n_nodes, xr_reference,
        "path admissibility macro: copy reference r failed");
    copy_device_node_field(
        d_xz_reference, n_nodes, xz_reference,
        "path admissibility macro: copy reference z failed");
  }
  return evaluate_macro_boundary_path_host(state, xr_old, xz_old, xr_new, xz_new,
                                           floor_rel,
                                           xr_reference.empty()
                                               ? nullptr
                                               : &xr_reference,
                                           xz_reference.empty()
                                               ? nullptr
                                               : &xz_reference);
}

inline PathAdmissibilityResult evaluate_pole_coarsen_boundary_path(
    const tenryu::hydro::pole_angular_coarsen::Overlay* overlay,
    const tenryu::core::State& state,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const double floor_rel,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (overlay == nullptr || !overlay->active) {
    return out;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> xr_new;
  std::vector<double> xz_new;
  std::vector<double> xr_reference;
  std::vector<double> xz_reference;
  copy_device_node_field(d_xr_old, n_nodes, xr_old,
                         "path admissibility pole macro: copy old r failed");
  copy_device_node_field(d_xz_old, n_nodes, xz_old,
                         "path admissibility pole macro: copy old z failed");
  copy_device_node_field(d_xr_new, n_nodes, xr_new,
                         "path admissibility pole macro: copy new r failed");
  copy_device_node_field(d_xz_new, n_nodes, xz_new,
                         "path admissibility pole macro: copy new z failed");
  if (floor_rel > 0.0 && d_xr_reference != nullptr &&
      d_xz_reference != nullptr) {
    copy_device_node_field(
        d_xr_reference, n_nodes, xr_reference,
        "path admissibility pole macro: copy reference r failed");
    copy_device_node_field(
        d_xz_reference, n_nodes, xz_reference,
        "path admissibility pole macro: copy reference z failed");
  }
  return evaluate_pole_coarsen_boundary_path_host(
      *overlay, xr_old, xz_old, xr_new, xz_new, floor_rel,
      xr_reference.empty() ? nullptr : &xr_reference,
      xz_reference.empty() ? nullptr : &xz_reference);
}

inline PathAdmissibilityResult evaluate_macro_boundary_path_velocity(
    const tenryu::core::State& state,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_v_r,
    const double* d_v_z,
    const double dt,
    const double floor_rel,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (!central_macro_path_active(state)) {
    return out;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> vr;
  std::vector<double> vz;
  std::vector<double> xr_reference;
  std::vector<double> xz_reference;
  copy_device_node_field(d_xr_old, n_nodes, xr_old,
                         "path admissibility macro: copy old r failed");
  copy_device_node_field(d_xz_old, n_nodes, xz_old,
                         "path admissibility macro: copy old z failed");
  copy_device_node_field(d_v_r, n_nodes, vr,
                         "path admissibility macro: copy velocity r failed");
  copy_device_node_field(d_v_z, n_nodes, vz,
                         "path admissibility macro: copy velocity z failed");
  if (floor_rel > 0.0 && d_xr_reference != nullptr &&
      d_xz_reference != nullptr) {
    copy_device_node_field(
        d_xr_reference, n_nodes, xr_reference,
        "path admissibility macro: copy reference r failed");
    copy_device_node_field(
        d_xz_reference, n_nodes, xz_reference,
        "path admissibility macro: copy reference z failed");
  }
  std::vector<double> xr_new(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> xz_new(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    xr_new[static_cast<std::size_t>(n)] =
        xr_old[static_cast<std::size_t>(n)] + dt * vr[static_cast<std::size_t>(n)];
    xz_new[static_cast<std::size_t>(n)] =
        xz_old[static_cast<std::size_t>(n)] + dt * vz[static_cast<std::size_t>(n)];
  }
  return evaluate_macro_boundary_path_host(state, xr_old, xz_old, xr_new, xz_new,
                                           floor_rel,
                                           xr_reference.empty()
                                               ? nullptr
                                               : &xr_reference,
                                           xz_reference.empty()
                                               ? nullptr
                                               : &xz_reference);
}

inline PathAdmissibilityResult evaluate_pole_coarsen_boundary_path_velocity(
    const tenryu::hydro::pole_angular_coarsen::Overlay* overlay,
    const tenryu::core::State& state,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_v_r,
    const double* d_v_z,
    const double dt,
    const double floor_rel,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult out;
  if (overlay == nullptr || !overlay->active) {
    return out;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> vr;
  std::vector<double> vz;
  std::vector<double> xr_reference;
  std::vector<double> xz_reference;
  copy_device_node_field(d_xr_old, n_nodes, xr_old,
                         "path admissibility pole macro: copy old r failed");
  copy_device_node_field(d_xz_old, n_nodes, xz_old,
                         "path admissibility pole macro: copy old z failed");
  copy_device_node_field(d_v_r, n_nodes, vr,
                         "path admissibility pole macro: copy velocity r failed");
  copy_device_node_field(d_v_z, n_nodes, vz,
                         "path admissibility pole macro: copy velocity z failed");
  if (floor_rel > 0.0 && d_xr_reference != nullptr &&
      d_xz_reference != nullptr) {
    copy_device_node_field(
        d_xr_reference, n_nodes, xr_reference,
        "path admissibility pole macro: copy reference r failed");
    copy_device_node_field(
        d_xz_reference, n_nodes, xz_reference,
        "path admissibility pole macro: copy reference z failed");
  }
  std::vector<double> xr_new(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> xz_new(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    xr_new[static_cast<std::size_t>(n)] =
        xr_old[static_cast<std::size_t>(n)] + dt * vr[static_cast<std::size_t>(n)];
    xz_new[static_cast<std::size_t>(n)] =
        xz_old[static_cast<std::size_t>(n)] + dt * vz[static_cast<std::size_t>(n)];
  }
  return evaluate_pole_coarsen_boundary_path_host(
      *overlay, xr_old, xz_old, xr_new, xz_new, floor_rel,
      xr_reference.empty() ? nullptr : &xr_reference,
      xz_reference.empty() ? nullptr : &xz_reference);
}

inline double mesh_forecast_q_min(const MeshForecastComponents& q) {
  return std::min(std::min(q.q_J, q.q_V),
                  std::min(q.q_edge, std::min(q.q_R, q.q_phi)));
}

inline double mesh_forecast_cross_at(const double rr0[4],
                                     const double zz0[4],
                                     const double drr[4],
                                     const double dzz[4],
                                     const int a_tail,
                                     const int a_head,
                                     const int b_tail,
                                     const int b_head,
                                     const double tau) {
  const double ar_tail = rr0[a_tail] + tau * drr[a_tail];
  const double az_tail = zz0[a_tail] + tau * dzz[a_tail];
  const double ar_head = rr0[a_head] + tau * drr[a_head];
  const double az_head = zz0[a_head] + tau * dzz[a_head];
  const double br_tail = rr0[b_tail] + tau * drr[b_tail];
  const double bz_tail = zz0[b_tail] + tau * dzz[b_tail];
  const double br_head = rr0[b_head] + tau * drr[b_head];
  const double bz_head = zz0[b_head] + tau * dzz[b_head];
  return cross2(ar_head - ar_tail, az_head - az_tail,
                br_head - br_tail, bz_head - bz_tail);
}

inline double mesh_forecast_normalized_cross(const double raw0,
                                             const double raw,
                                             const double len_a,
                                             const double len_b,
                                             const double orientation_sign) {
  constexpr double kTiny = 1.0e-300;
  if (!std::isfinite(raw0) || !std::isfinite(raw) ||
      !std::isfinite(len_a) || !std::isfinite(len_b) ||
      !(len_a > kTiny) || !(len_b > kTiny) || raw0 == 0.0) {
    return -HUGE_VAL;
  }
  const double sign = orientation_sign == 0.0 ? (raw0 < 0.0 ? -1.0 : 1.0)
                                              : (orientation_sign < 0.0 ? -1.0
                                                                        : 1.0);
  return sign * raw / (len_a * len_b);
}

inline double mesh_forecast_edge_length(const double rr0[4],
                                        const double zz0[4],
                                        const double drr[4],
                                        const double dzz[4],
                                        const int tail,
                                        const int head,
                                        const double tau) {
  const double dr = (rr0[head] + tau * drr[head]) -
                    (rr0[tail] + tau * drr[tail]);
  const double dz = (zz0[head] + tau * dzz[head]) -
                    (zz0[tail] + tau * dzz[tail]);
  return std::hypot(dr, dz);
}

inline void mesh_forecast_observe_corner_j(const double rr0[4],
                                           const double zz0[4],
                                           const double drr[4],
                                           const double dzz[4],
                                           const int corner,
                                           const int active_nverts,
                                           const double tau,
                                           const double orientation_sign,
                                           double& q_min) {
  const int kp = (corner + 1) % active_nverts;
  const int km = (corner + active_nverts - 1) % active_nverts;
  const double raw0 =
      mesh_forecast_cross_at(rr0, zz0, drr, dzz, corner, kp, corner, km, 0.0);
  const double raw =
      mesh_forecast_cross_at(rr0, zz0, drr, dzz, corner, kp, corner, km, tau);
  const double len_a =
      mesh_forecast_edge_length(rr0, zz0, drr, dzz, corner, kp, tau);
  const double len_b =
      mesh_forecast_edge_length(rr0, zz0, drr, dzz, corner, km, tau);
  q_min = std::min(
      q_min,
      mesh_forecast_normalized_cross(raw0, raw, len_a, len_b,
                                     orientation_sign));
}

inline void mesh_forecast_observe_edge_cross(const double rr0[4],
                                             const double zz0[4],
                                             const double drr[4],
                                             const double dzz[4],
                                             const int u_tail,
                                             const int u_head,
                                             const int v_tail,
                                             const int v_head,
                                             const double tau,
                                             const double orientation_sign,
                                             double& q_min) {
  const double raw0 = mesh_forecast_cross_at(
      rr0, zz0, drr, dzz, u_tail, u_head, v_tail, v_head, 0.0);
  const double raw = mesh_forecast_cross_at(
      rr0, zz0, drr, dzz, u_tail, u_head, v_tail, v_head, tau);
  const double len_u =
      mesh_forecast_edge_length(rr0, zz0, drr, dzz, u_tail, u_head, tau);
  const double len_v =
      mesh_forecast_edge_length(rr0, zz0, drr, dzz, v_tail, v_head, tau);
  q_min = std::min(
      q_min,
      mesh_forecast_normalized_cross(raw0, raw, len_u, len_v,
                                     orientation_sign));
}

inline double mesh_forecast_rz_volume_local(const double rr[4],
                                            const double zz[4],
                                            const int active_nverts) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  double sum = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int n = (k + 1) % active_nverts;
    if (!std::isfinite(rr[k]) || !std::isfinite(zz[k]) ||
        !std::isfinite(rr[n]) || !std::isfinite(zz[n])) {
      return -HUGE_VAL;
    }
    sum += (rr[k] + rr[n]) * (rr[k] * zz[n] - rr[n] * zz[k]);
  }
  return kPi / 3.0 * sum;
}

inline double mesh_forecast_rz_volume_q(const double rr0[4],
                                        const double zz0[4],
                                        const double drr[4],
                                        const double dzz[4],
                                        const int active_nverts,
                                        const double tau,
                                        const double floor_rel,
                                        const double orientation_sign) {
  double rr_tau[4] = {0.0, 0.0, 0.0, 0.0};
  double zz_tau[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < active_nverts; ++k) {
    rr_tau[k] = rr0[k] + tau * drr[k];
    zz_tau[k] = zz0[k] + tau * dzz[k];
  }
  const double v0 = mesh_forecast_rz_volume_local(rr0, zz0, active_nverts);
  const double vt =
      mesh_forecast_rz_volume_local(rr_tau, zz_tau, active_nverts);
  if (!std::isfinite(v0) || !std::isfinite(vt) || v0 == 0.0) {
    return -HUGE_VAL;
  }
  const double sign = orientation_sign == 0.0 ? (v0 < 0.0 ? -1.0 : 1.0)
                                              : (orientation_sign < 0.0 ? -1.0
                                                                        : 1.0);
  const double floor = std::max(1.0e-300, std::max(0.0, floor_rel) *
                                             std::abs(v0));
  return sign * vt / floor;
}

inline const BlockInfo* mesh_forecast_locate_cell(
    const MultiBlockTopology& mb,
    const int cell,
    int& block_id,
    int& local_i,
    int& local_j) {
  block_id = -1;
  local_i = -1;
  local_j = -1;
  const BlockInfo* block = nullptr;
  if (cell >= 0 && mb.cell_block_id.size() > static_cast<std::size_t>(cell)) {
    const int mapped = mb.cell_block_id[static_cast<std::size_t>(cell)];
    if (mapped >= 0 && mapped < static_cast<int>(mb.blocks.size())) {
      const BlockInfo& candidate = mb.blocks[static_cast<std::size_t>(mapped)];
      if (cell >= candidate.cell_begin &&
          cell < candidate.cell_begin + candidate.cell_count) {
        block_id = mapped;
        block = &candidate;
      }
    }
  }
  if (block == nullptr) {
    for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
      const BlockInfo& candidate = mb.blocks[static_cast<std::size_t>(b)];
      if (cell >= candidate.cell_begin &&
          cell < candidate.cell_begin + candidate.cell_count) {
        block_id = b;
        block = &candidate;
        break;
      }
    }
  }
  if (block == nullptr || block->n_j_cells <= 0) {
    return nullptr;
  }
  const int local = cell - block->cell_begin;
  if (local < 0 || local >= block->cell_count) {
    return nullptr;
  }
  local_i = local / block->n_j_cells;
  local_j = local - local_i * block->n_j_cells;
  return block;
}

inline double mesh_forecast_node_r_at(const std::vector<double>& xr_old,
                                      const std::vector<double>& xr_new,
                                      const int node,
                                      const double tau) {
  return xr_old[static_cast<std::size_t>(node)] +
         tau * (xr_new[static_cast<std::size_t>(node)] -
                xr_old[static_cast<std::size_t>(node)]);
}

inline double mesh_forecast_node_z_at(const std::vector<double>& xz_old,
                                      const std::vector<double>& xz_new,
                                      const int node,
                                      const double tau) {
  return xz_old[static_cast<std::size_t>(node)] +
         tau * (xz_new[static_cast<std::size_t>(node)] -
                xz_old[static_cast<std::size_t>(node)]);
}

inline double mesh_forecast_radius_at(const std::vector<double>& xr_old,
                                      const std::vector<double>& xz_old,
                                      const std::vector<double>& xr_new,
                                      const std::vector<double>& xz_new,
                                      const int node,
                                      const double tau) {
  return std::hypot(mesh_forecast_node_r_at(xr_old, xr_new, node, tau),
                    mesh_forecast_node_z_at(xz_old, xz_new, node, tau));
}

inline double mesh_forecast_theta_at(const std::vector<double>& xr_old,
                                     const std::vector<double>& xz_old,
                                     const std::vector<double>& xr_new,
                                     const std::vector<double>& xz_new,
                                     const int node,
                                     const double tau) {
  return std::atan2(mesh_forecast_node_r_at(xr_old, xr_new, node, tau),
                    mesh_forecast_node_z_at(xz_old, xz_new, node, tau));
}

inline void mesh_forecast_shell_order_q(
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const BlockInfo& block,
    const int local_i,
    const int local_j,
    const double tau,
    double& q_R,
    double& q_phi) {
  q_R = HUGE_VAL;
  q_phi = HUGE_VAL;
  if (block.role != BlockRole::POLAR_SHELL || local_i < 0 || local_j < 0 ||
      local_i >= block.n_i_cells || local_j >= block.n_j_cells) {
    return;
  }
  const int stride = block.n_j_cells + 1;
  const auto node = [&](const int i, const int j) {
    return block.owned_node_begin + i * stride + j;
  };
  const int n00 = node(local_i, local_j);
  const int n01 = node(local_i, local_j + 1);
  const int n10 = node(local_i + 1, local_j);
  const int n11 = node(local_i + 1, local_j + 1);
  const int n_nodes = static_cast<int>(xr_old.size());
  if (n00 < 0 || n01 < 0 || n10 < 0 || n11 < 0 ||
      n00 >= n_nodes || n01 >= n_nodes || n10 >= n_nodes || n11 >= n_nodes) {
    q_R = -HUGE_VAL;
    q_phi = -HUGE_VAL;
    return;
  }
  const auto radial_q = [&](const int inner, const int outer) {
    const double d0 = mesh_forecast_radius_at(xr_old, xz_old, xr_new, xz_new,
                                              outer, 0.0) -
                      mesh_forecast_radius_at(xr_old, xz_old, xr_new, xz_new,
                                              inner, 0.0);
    const double dtau = mesh_forecast_radius_at(xr_old, xz_old, xr_new, xz_new,
                                                outer, tau) -
                        mesh_forecast_radius_at(xr_old, xz_old, xr_new, xz_new,
                                                inner, tau);
    if (!std::isfinite(d0) || !std::isfinite(dtau) || !(d0 > 0.0)) {
      return -HUGE_VAL;
    }
    return dtau / d0;
  };
  const auto angular_q = [&](const int left, const int right) {
    const double d0 = mesh_forecast_theta_at(xr_old, xz_old, xr_new, xz_new,
                                             right, 0.0) -
                      mesh_forecast_theta_at(xr_old, xz_old, xr_new, xz_new,
                                             left, 0.0);
    const double dtau = mesh_forecast_theta_at(xr_old, xz_old, xr_new, xz_new,
                                               right, tau) -
                        mesh_forecast_theta_at(xr_old, xz_old, xr_new, xz_new,
                                               left, tau);
    if (!std::isfinite(d0) || !std::isfinite(dtau) || !(d0 > 0.0)) {
      return -HUGE_VAL;
    }
    return dtau / d0;
  };
  q_R = std::min(radial_q(n00, n10), radial_q(n01, n11));
  q_phi = std::min(angular_q(n00, n01), angular_q(n10, n11));
}

template <int SlotCap = kMeshTopoCellStorageSlotsMax>
inline MeshForecastComponents mesh_forecast_cell_components(
    const tenryu::core::State& state,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const int cell,
    const double tau,
    const double floor_rel) {
  MeshForecastComponents q;
  if (!state.mesh.topo.multiblock.has_value()) {
    return q;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  if (cell < 0 || cell >= n_cells ||
      mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells + 1) ||
      mb.cell_node_csr_indices.size() !=
          static_cast<std::size_t>(state.mesh.corner_stride * n_cells)) {
    q.q_J = -HUGE_VAL;
    q.q_V = -HUGE_VAL;
    q.q_edge = -HUGE_VAL;
    return q;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int next = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)];
  const std::uint8_t* nverts_ptr =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? state.mesh.cell_nverts.data()
          : nullptr;
  const int active_nverts = mesh_topo_cell_active_nverts(nverts_ptr, cell);
  TENRYU_ASSERT(active_nverts <= SlotCap, "cell exceeds slot cap");
  if (next - off != state.mesh.corner_stride || off < 0) {
    q.q_J = -HUGE_VAL;
    q.q_V = -HUGE_VAL;
    q.q_edge = -HUGE_VAL;
    return q;
  }
  double rr0[SlotCap] = {0.0};
  double zz0[SlotCap] = {0.0};
  double drr[SlotCap] = {0.0};
  double dzz[SlotCap] = {0.0};
  for (int k = 0; k < active_nverts; ++k) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    if (node < 0 || static_cast<std::size_t>(node) >= xr_old.size() ||
        static_cast<std::size_t>(node) >= xz_old.size() ||
        static_cast<std::size_t>(node) >= xr_new.size() ||
        static_cast<std::size_t>(node) >= xz_new.size()) {
      q.q_J = -HUGE_VAL;
      q.q_V = -HUGE_VAL;
      q.q_edge = -HUGE_VAL;
      return q;
    }
    rr0[k] = xr_old[static_cast<std::size_t>(node)];
    zz0[k] = xz_old[static_cast<std::size_t>(node)];
    drr[k] = xr_new[static_cast<std::size_t>(node)] - rr0[k];
    dzz[k] = xz_new[static_cast<std::size_t>(node)] - zz0[k];
    if (!std::isfinite(rr0[k]) || !std::isfinite(zz0[k]) ||
        !std::isfinite(drr[k]) || !std::isfinite(dzz[k])) {
      q.q_J = -HUGE_VAL;
      q.q_V = -HUGE_VAL;
      q.q_edge = -HUGE_VAL;
      return q;
    }
  }
  const double orientation_sign =
      path_predicate_harden_enabled() &&
              mb.cell_orientation_sign.size() == static_cast<std::size_t>(n_cells)
          ? (mb.cell_orientation_sign[static_cast<std::size_t>(cell)] < 0 ? -1.0
                                                                          : 1.0)
          : 0.0;
  q.q_J = HUGE_VAL;
  for (int k = 0; k < active_nverts; ++k) {
    mesh_forecast_observe_corner_j(rr0, zz0, drr, dzz, k, active_nverts, tau,
                                   orientation_sign, q.q_J);
  }
  q.q_edge = HUGE_VAL;
  if (active_nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const int km = (k + 2) % 3;
      mesh_forecast_observe_edge_cross(rr0, zz0, drr, dzz, k, kp, k, km,
                                       tau, orientation_sign, q.q_edge);
    }
  } else {
    mesh_forecast_observe_edge_cross(rr0, zz0, drr, dzz, 0, 1, 0, 3,
                                     tau, orientation_sign, q.q_edge);
    mesh_forecast_observe_edge_cross(rr0, zz0, drr, dzz, 0, 1, 1, 2,
                                     tau, orientation_sign, q.q_edge);
    mesh_forecast_observe_edge_cross(rr0, zz0, drr, dzz, 3, 2, 1, 2,
                                     tau, orientation_sign, q.q_edge);
    mesh_forecast_observe_edge_cross(rr0, zz0, drr, dzz, 3, 2, 0, 3,
                                     tau, orientation_sign, q.q_edge);
  }
  q.q_V = mesh_forecast_rz_volume_q(rr0, zz0, drr, dzz, active_nverts, tau,
                                    floor_rel, orientation_sign);
  int block_id = -1;
  int local_i = -1;
  int local_j = -1;
  const BlockInfo* block =
      mesh_forecast_locate_cell(mb, cell, block_id, local_i, local_j);
  if (block != nullptr) {
    mesh_forecast_shell_order_q(xr_old, xz_old, xr_new, xz_new, *block,
                                local_i, local_j, tau, q.q_R, q.q_phi);
  }
  return q;
}

inline std::uint32_t mesh_forecast_component_bits(
    const MeshForecastComponents& q,
    const double threshold) {
  const auto warns = [threshold](const double value) {
    if (!std::isfinite(value)) {
      return value < 0.0;
    }
    return value <= threshold;
  };
  std::uint32_t bits = 0U;
  if (warns(q.q_J)) {
    bits |= kMeshForecastFailureQJ;
  }
  if (warns(q.q_V)) {
    bits |= kMeshForecastFailureQV;
  }
  if (warns(q.q_edge)) {
    bits |= kMeshForecastFailureQEdge;
  }
  if (warns(q.q_R)) {
    bits |= kMeshForecastFailureQR;
  }
  if (warns(q.q_phi)) {
    bits |= kMeshForecastFailureQPhi;
  }
  return bits;
}

inline std::uint32_t mesh_forecast_min_component_bits(
    const MeshForecastComponents& q) {
  const double q_min = mesh_forecast_q_min(q);
  std::uint32_t bits = 0U;
  if (!std::isfinite(q_min)) {
    return mesh_forecast_component_bits(q, q_min);
  }
  const double tol = 1.0e-12 * std::max(1.0, std::abs(q_min));
  if (std::abs(q.q_J - q_min) <= tol) {
    bits |= kMeshForecastFailureQJ;
  }
  if (std::abs(q.q_V - q_min) <= tol) {
    bits |= kMeshForecastFailureQV;
  }
  if (std::abs(q.q_edge - q_min) <= tol) {
    bits |= kMeshForecastFailureQEdge;
  }
  if (std::abs(q.q_R - q_min) <= tol) {
    bits |= kMeshForecastFailureQR;
  }
  if (std::abs(q.q_phi - q_min) <= tol) {
    bits |= kMeshForecastFailureQPhi;
  }
  return bits;
}

inline void mesh_forecast_fill_cell_summary(
    MeshForecast& out,
    const tenryu::core::State& state,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const int cell,
    const double tau_warn,
    const double floor_rel,
    const double q_warn) {
  out.first_cell = cell;
  out.worst_now = mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, 0.0, floor_rel);
  out.worst_warn = mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, tau_warn, floor_rel);
  out.worst_end = mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, 1.0, floor_rel);
  out.q_trace_tau0 = mesh_forecast_q_min(out.worst_now);
  out.q_trace_tau25 = mesh_forecast_q_min(mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, 0.25, floor_rel));
  out.q_trace_tau50 = mesh_forecast_q_min(mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, 0.50, floor_rel));
  out.q_trace_tau75 = mesh_forecast_q_min(mesh_forecast_cell_components(
      state, xr_old, xz_old, xr_new, xz_new, cell, 0.75, floor_rel));
  out.q_trace_tau100 = mesh_forecast_q_min(out.worst_end);
  out.failure_bits = mesh_forecast_component_bits(out.worst_warn, q_warn);
  if (out.failure_bits == 0U) {
    out.failure_bits = mesh_forecast_min_component_bits(out.worst_warn);
  }
  out.block_id = -1;
  out.local_i = -1;
  out.local_j = -1;
  if (state.mesh.topo.multiblock.has_value()) {
    int block_id = -1;
    int local_i = -1;
    int local_j = -1;
    mesh_forecast_locate_cell(*state.mesh.topo.multiblock, cell, block_id,
                              local_i, local_j);
    out.block_id = block_id;
    out.local_i = local_i;
    out.local_j = local_j;
  }
}

inline std::vector<std::uint8_t> mesh_forecast_inactive_mask(
    const tenryu::core::State& state,
    const tenryu::hydro::pole_angular_coarsen::Overlay* accepted_pole_overlay,
    const int n_cells) {
  std::vector<std::uint8_t> inactive_mask;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive_mask = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive_mask.empty()) {
      inactive_mask.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive_mask[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (accepted_pole_overlay != nullptr &&
      accepted_pole_overlay->skip_fine_child_paths &&
      accepted_pole_overlay->pole_coarsen_inactive_fine_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    if (inactive_mask.empty()) {
      inactive_mask.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (accepted_pole_overlay
              ->pole_coarsen_inactive_fine_mask[static_cast<std::size_t>(c)] !=
          0U) {
        inactive_mask[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  return inactive_mask;
}

inline MeshForecast evaluate_mesh_forecast_host(
    const tenryu::core::State& state,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    const double q_warn_in,
    const PathAdmissibilityResult& path_result,
    const tenryu::hydro::pole_angular_coarsen::Overlay* pole_overlay) {
  MeshForecast out;
  const double q_warn = std::max(0.0, q_warn_in);
  out.q_warn = q_warn;
  out.tau_zero = std::isfinite(path_result.first_failing_lambda)
                     ? std::min(1.0, std::max(0.0,
                                              path_result.first_failing_lambda))
                     : 0.0;
  out.endpoint_valid =
      (path_result.first_failing_cell == -1 || !(out.tau_zero < 1.0)) ? 1 : 0;
  if (!state.mesh.topo.multiblock.has_value() || state.mesh.topo.n_cells <= 0) {
    return out;
  }
  const int n_cells = state.mesh.topo.n_cells;
  tenryu::hydro::pole_angular_coarsen::Overlay accepted_pole_overlay;
  const tenryu::hydro::pole_angular_coarsen::Overlay* accepted_pole_overlay_ptr =
      nullptr;
  if (pole_overlay != nullptr && pole_overlay->active) {
    prepare_pole_coarsen_overlay_host(pole_overlay, n_cells, xr_old, xz_old,
                                      xr_new, xz_new, floor_rel,
                                      accepted_pole_overlay);
    accepted_pole_overlay_ptr = &accepted_pole_overlay;
  }
  const std::vector<std::uint8_t> inactive_mask =
      mesh_forecast_inactive_mask(state, accepted_pole_overlay_ptr, n_cells);
  out.seed_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  constexpr int kSamples = 256;
  int endpoint_worst_cell = -1;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (!inactive_mask.empty() &&
        inactive_mask[static_cast<std::size_t>(cell)] != 0U) {
      continue;
    }
    const MeshForecastComponents q0 = mesh_forecast_cell_components(
        state, xr_old, xz_old, xr_new, xz_new, cell, 0.0, floor_rel);
    const MeshForecastComponents q1 = mesh_forecast_cell_components(
        state, xr_old, xz_old, xr_new, xz_new, cell, 1.0, floor_rel);
    const double q_now = mesh_forecast_q_min(q0);
    const double q_end = mesh_forecast_q_min(q1);
    if (q_now < out.q_min_now) {
      out.q_min_now = q_now;
    }
    if (q_end < out.q_min_end) {
      out.q_min_end = q_end;
      endpoint_worst_cell = cell;
    }
    double q_cell_path_min = std::min(q_now, q_end);
    bool crossed = q_now <= q_warn;
    double tau_cross = crossed ? 0.0 : 1.0;
    double prev_tau = 0.0;
    double prev_q = q_now;
    for (int s = 1; s <= kSamples; ++s) {
      const double tau = static_cast<double>(s) / static_cast<double>(kSamples);
      const double q_tau = mesh_forecast_q_min(mesh_forecast_cell_components(
          state, xr_old, xz_old, xr_new, xz_new, cell, tau, floor_rel));
      q_cell_path_min = std::min(q_cell_path_min, q_tau);
      if (!crossed && prev_q > q_warn && q_tau <= q_warn) {
        double lo = prev_tau;
        double hi = tau;
        for (int iter = 0; iter < 48; ++iter) {
          const double mid = 0.5 * (lo + hi);
          const double q_mid =
              mesh_forecast_q_min(mesh_forecast_cell_components(
                  state, xr_old, xz_old, xr_new, xz_new, cell, mid, floor_rel));
          if (q_mid <= q_warn) {
            hi = mid;
          } else {
            lo = mid;
          }
        }
        tau_cross = hi;
        crossed = true;
      }
      prev_tau = tau;
      prev_q = q_tau;
    }
    out.q_min_path = std::min(out.q_min_path, q_cell_path_min);
    if (crossed) {
      out.seed_mask[static_cast<std::size_t>(cell)] = 1U;
      ++out.seed_count;
      if (tau_cross < out.tau_warn ||
          (tau_cross == out.tau_warn &&
           (out.first_cell < 0 || cell < out.first_cell))) {
        out.tau_warn = tau_cross;
        mesh_forecast_fill_cell_summary(out, state, xr_old, xz_old, xr_new,
                                        xz_new, cell, tau_cross, floor_rel,
                                        q_warn);
      }
    }
  }
  if (out.first_cell < 0 && endpoint_worst_cell >= 0) {
    mesh_forecast_fill_cell_summary(out, state, xr_old, xz_old, xr_new, xz_new,
                                    endpoint_worst_cell, 1.0, floor_rel,
                                    q_warn);
  }
  return out;
}

}  // namespace path_admissibility_detail

inline bool path_admissibility_anatomy_enabled() {
  return path_admissibility_detail::path_admissibility_anatomy_enabled();
}

inline bool path_predicate_harden_enabled() {
  return path_admissibility_detail::path_predicate_harden_enabled();
}

inline PathAdmissibilityResult prepare_pole_coarsen_overlay_host(
    const tenryu::hydro::pole_angular_coarsen::Overlay* source,
    const int n_cells,
    const std::vector<double>& xr_old,
    const std::vector<double>& xz_old,
    const std::vector<double>& xr_new,
    const std::vector<double>& xz_new,
    const double floor_rel,
    tenryu::hydro::pole_angular_coarsen::Overlay& accepted,
    const std::vector<double>* xr_reference = nullptr,
    const std::vector<double>* xz_reference = nullptr) {
  return path_admissibility_detail::prepare_pole_coarsen_overlay_host(
      source, n_cells, xr_old, xz_old, xr_new, xz_new, floor_rel, accepted,
      xr_reference, xz_reference);
}

inline PathAdmissibilityResult evaluate_path_admissibility(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const double J_floor,
    const tenryu::hydro::pole_angular_coarsen::Overlay* pole_overlay =
        nullptr,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult result;
  if (!mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(d_xr_old != nullptr && d_xz_old != nullptr &&
                    d_xr_new != nullptr && d_xz_new != nullptr,
                "evaluate_path_admissibility requires non-null coordinate arrays");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_cells + 1),
                "path admissibility requires multiblock CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(state.mesh.corner_stride *
                                             state.mesh.topo.n_cells),
                "path admissibility requires multiblock CSR indices");

  const int n_cells = state.mesh.topo.n_cells;
  const auto& mb = *state.mesh.topo.multiblock;
  const bool predicate_harden =
      path_admissibility_detail::path_predicate_harden_enabled();
  tenryu::hydro::pole_angular_coarsen::Overlay accepted_pole_overlay;
  const tenryu::hydro::pole_angular_coarsen::Overlay*
      accepted_pole_overlay_ptr = nullptr;
  PathAdmissibilityResult pole_prepare_result;
  std::vector<double> pole_xr_old;
  std::vector<double> pole_xz_old;
  std::vector<double> pole_xr_new;
  std::vector<double> pole_xz_new;
  std::vector<double> pole_xr_reference;
  std::vector<double> pole_xz_reference;
  if (pole_overlay != nullptr && pole_overlay->active) {
    const int n_nodes = state.mesh.topo.n_nodes;
    path_admissibility_detail::copy_device_node_field(
        d_xr_old, n_nodes, pole_xr_old,
        "path admissibility pole macro: copy old r failed");
    path_admissibility_detail::copy_device_node_field(
        d_xz_old, n_nodes, pole_xz_old,
        "path admissibility pole macro: copy old z failed");
    path_admissibility_detail::copy_device_node_field(
        d_xr_new, n_nodes, pole_xr_new,
        "path admissibility pole macro: copy new r failed");
    path_admissibility_detail::copy_device_node_field(
        d_xz_new, n_nodes, pole_xz_new,
        "path admissibility pole macro: copy new z failed");
    if (J_floor > 0.0 && d_xr_reference != nullptr &&
        d_xz_reference != nullptr) {
      path_admissibility_detail::copy_device_node_field(
          d_xr_reference, n_nodes, pole_xr_reference,
          "path admissibility pole macro: copy reference r failed");
      path_admissibility_detail::copy_device_node_field(
          d_xz_reference, n_nodes, pole_xz_reference,
          "path admissibility pole macro: copy reference z failed");
    }
    pole_prepare_result =
        path_admissibility_detail::prepare_pole_coarsen_overlay_host(
            pole_overlay,
            n_cells,
            pole_xr_old,
            pole_xz_old,
            pole_xr_new,
            pole_xz_new,
            std::max(0.0, J_floor),
            accepted_pole_overlay,
            pole_xr_reference.empty() ? nullptr : &pole_xr_reference,
            pole_xz_reference.empty() ? nullptr : &pole_xz_reference);
    accepted_pole_overlay_ptr = &accepted_pole_overlay;
  }
  tenryu::core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (path_admissibility_detail::has_nonquad_cell_nverts(
          state.mesh.cell_nverts, n_cells)) {
    d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }
  tenryu::core::DeviceArray<int> d_cell_orientation_sign;
  const int* d_cell_orientation_sign_ptr = nullptr;
  if (predicate_harden &&
      mb.cell_orientation_sign.size() == static_cast<std::size_t>(n_cells)) {
    d_cell_orientation_sign.reset(static_cast<std::size_t>(n_cells));
    d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
    d_cell_orientation_sign_ptr = d_cell_orientation_sign.data();
  }
  tenryu::core::DeviceArray<std::uint8_t> d_inactive_mask;
  const std::uint8_t* d_inactive_mask_ptr = nullptr;
  std::vector<std::uint8_t> inactive_mask_host;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive_mask_host = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive_mask_host.empty()) {
      inactive_mask_host.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive_mask_host[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (accepted_pole_overlay_ptr != nullptr &&
      accepted_pole_overlay_ptr->skip_fine_child_paths &&
      accepted_pole_overlay_ptr->pole_coarsen_inactive_fine_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    if (inactive_mask_host.empty()) {
      inactive_mask_host.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (accepted_pole_overlay_ptr
              ->pole_coarsen_inactive_fine_mask[static_cast<std::size_t>(c)] !=
          0U) {
        inactive_mask_host[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (!inactive_mask_host.empty()) {
    d_inactive_mask.reset(static_cast<std::size_t>(n_cells));
    d_inactive_mask.copy_from_host(inactive_mask_host);
    d_inactive_mask_ptr = d_inactive_mask.data();
  }
  tenryu::core::DeviceArray<PathAdmissibilityResult> d_results(
      static_cast<std::size_t>(n_cells));
  const int blocks = (n_cells + 255) / 256;
  const bool anatomy =
      path_admissibility_detail::path_admissibility_anatomy_enabled();
  if (anatomy) {
    if (state.mesh.corner_stride <= kMeshTopoCellStorageSlotsMax) {
      path_admissibility_detail::evaluate_path_admissibility_csr_kernel<true>
          <<<blocks, 256>>>(d_results.data(),
                            n_cells,
                            state.mesh.corner_stride,
                            state.mesh.multiblock_cell_node_csr_offsets.data(),
                            state.mesh.multiblock_cell_node_csr_indices.data(),
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            d_xr_reference,
                            d_xz_reference,
                            d_cell_nverts_ptr,
                            d_cell_orientation_sign_ptr,
                            d_inactive_mask_ptr,
                            std::max(0.0, J_floor),
                            path_admissibility_detail::
                                path_admissibility_r_guard_enabled());
    } else {
      path_admissibility_detail::evaluate_path_admissibility_csr_kernel<
          true, kMeshTopoCellStorageSlotsMaxGeneral>
          <<<blocks, 256>>>(d_results.data(),
                            n_cells,
                            state.mesh.corner_stride,
                            state.mesh.multiblock_cell_node_csr_offsets.data(),
                            state.mesh.multiblock_cell_node_csr_indices.data(),
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            d_xr_reference,
                            d_xz_reference,
                            d_cell_nverts_ptr,
                            d_cell_orientation_sign_ptr,
                            d_inactive_mask_ptr,
                            std::max(0.0, J_floor),
                            path_admissibility_detail::
                                path_admissibility_r_guard_enabled());
    }
  } else {
    if (state.mesh.corner_stride <= kMeshTopoCellStorageSlotsMax) {
      path_admissibility_detail::evaluate_path_admissibility_csr_kernel<false>
          <<<blocks, 256>>>(d_results.data(),
                            n_cells,
                            state.mesh.corner_stride,
                            state.mesh.multiblock_cell_node_csr_offsets.data(),
                            state.mesh.multiblock_cell_node_csr_indices.data(),
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            d_xr_reference,
                            d_xz_reference,
                            d_cell_nverts_ptr,
                            d_cell_orientation_sign_ptr,
                            d_inactive_mask_ptr,
                            std::max(0.0, J_floor),
                            path_admissibility_detail::
                                path_admissibility_r_guard_enabled());
    } else {
      path_admissibility_detail::evaluate_path_admissibility_csr_kernel<
          false, kMeshTopoCellStorageSlotsMaxGeneral>
          <<<blocks, 256>>>(d_results.data(),
                            n_cells,
                            state.mesh.corner_stride,
                            state.mesh.multiblock_cell_node_csr_offsets.data(),
                            state.mesh.multiblock_cell_node_csr_indices.data(),
                            d_xr_old,
                            d_xz_old,
                            d_xr_new,
                            d_xz_new,
                            d_xr_reference,
                            d_xz_reference,
                            d_cell_nverts_ptr,
                            d_cell_orientation_sign_ptr,
                            d_inactive_mask_ptr,
                            std::max(0.0, J_floor),
                            path_admissibility_detail::
                                path_admissibility_r_guard_enabled());
    }
  }
  path_admissibility_detail::cuda_check(
      cudaGetLastError(),
      "evaluate_path_admissibility kernel launch failed");
  path_admissibility_detail::cuda_check(
      cudaDeviceSynchronize(),
      "evaluate_path_admissibility kernel failed");

  std::vector<PathAdmissibilityResult> host;
  d_results.copy_to_host(host);
  const auto pole_deref_repair_cell_mask =
      path_admissibility_detail::build_pole_deref_repair_cell_mask(
          accepted_pole_overlay_ptr, n_cells);
  for (auto& cell_result : host) {
    if (anatomy) {
      path_admissibility_detail::annotate_path_admissibility_location(
          cell_result, state);
    }
    path_admissibility_detail::gate_active_fine_child_macro_repair(
        cell_result, pole_deref_repair_cell_mask);
    path_admissibility_detail::merge_path_admissibility_result(result,
                                                               cell_result);
  }
  path_admissibility_detail::merge_path_admissibility_result(
      result,
      path_admissibility_detail::evaluate_macro_boundary_path(
          state, d_xr_old, d_xz_old, d_xr_new, d_xz_new,
          std::max(0.0, J_floor), d_xr_reference, d_xz_reference));
  if (accepted_pole_overlay_ptr != nullptr) {
    path_admissibility_detail::merge_path_admissibility_result(
        result,
        path_admissibility_detail::evaluate_pole_coarsen_boundary_path_host(
            *accepted_pole_overlay_ptr,
            pole_xr_old,
            pole_xz_old,
            pole_xr_new,
            pole_xz_new,
            std::max(0.0, J_floor),
            pole_xr_reference.empty() ? nullptr : &pole_xr_reference,
            pole_xz_reference.empty() ? nullptr : &pole_xz_reference));
    path_admissibility_detail::merge_path_admissibility_result(
        result, pole_prepare_result);
  }
  if (path_admissibility_detail::env_flag_enabled(
          "TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE") &&
      result.first_failing_cell >= 0 &&
      result.first_failing_cell < n_cells &&
      state.pole_angular_derefine.inactive_member_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    TENRYU_ASSERT(
        state.pole_angular_derefine
                .inactive_member_mask[static_cast<std::size_t>(
                    result.first_failing_cell)] == 0U,
        "path admissibility selected an inactive polar shell de-refine child");
  }
  return result;
}

inline PathAdmissibilityResult evaluate_path_admissibility(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_v_r,
    const double* d_v_z,
    const double dt,
    const double J_floor,
    const tenryu::hydro::pole_angular_coarsen::Overlay* pole_overlay =
        nullptr,
    const double* d_xr_reference = nullptr,
    const double* d_xz_reference = nullptr) {
  PathAdmissibilityResult result;
  if (!mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(d_xr_old != nullptr && d_xz_old != nullptr &&
                    d_v_r != nullptr && d_v_z != nullptr,
                "evaluate_path_admissibility requires non-null coordinate and velocity arrays");
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "evaluate_path_admissibility requires finite dt > 0");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_cells + 1),
                "path admissibility requires multiblock CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(state.mesh.corner_stride *
                                             state.mesh.topo.n_cells),
                "path admissibility requires multiblock CSR indices");

  const int n_cells = state.mesh.topo.n_cells;
  const auto& mb = *state.mesh.topo.multiblock;
  const bool predicate_harden =
      path_admissibility_detail::path_predicate_harden_enabled();
  tenryu::hydro::pole_angular_coarsen::Overlay accepted_pole_overlay;
  const tenryu::hydro::pole_angular_coarsen::Overlay*
      accepted_pole_overlay_ptr = nullptr;
  PathAdmissibilityResult pole_prepare_result;
  std::vector<double> pole_xr_old;
  std::vector<double> pole_xz_old;
  std::vector<double> pole_vr;
  std::vector<double> pole_vz;
  std::vector<double> pole_xr_new;
  std::vector<double> pole_xz_new;
  std::vector<double> pole_xr_reference;
  std::vector<double> pole_xz_reference;
  if (pole_overlay != nullptr && pole_overlay->active) {
    const int n_nodes = state.mesh.topo.n_nodes;
    path_admissibility_detail::copy_device_node_field(
        d_xr_old, n_nodes, pole_xr_old,
        "path admissibility pole macro: copy old r failed");
    path_admissibility_detail::copy_device_node_field(
        d_xz_old, n_nodes, pole_xz_old,
        "path admissibility pole macro: copy old z failed");
    path_admissibility_detail::copy_device_node_field(
        d_v_r, n_nodes, pole_vr,
        "path admissibility pole macro: copy velocity r failed");
    path_admissibility_detail::copy_device_node_field(
        d_v_z, n_nodes, pole_vz,
        "path admissibility pole macro: copy velocity z failed");
    pole_xr_new.assign(static_cast<std::size_t>(n_nodes), 0.0);
    pole_xz_new.assign(static_cast<std::size_t>(n_nodes), 0.0);
    for (int n = 0; n < n_nodes; ++n) {
      pole_xr_new[static_cast<std::size_t>(n)] =
          pole_xr_old[static_cast<std::size_t>(n)] +
          dt * pole_vr[static_cast<std::size_t>(n)];
      pole_xz_new[static_cast<std::size_t>(n)] =
          pole_xz_old[static_cast<std::size_t>(n)] +
          dt * pole_vz[static_cast<std::size_t>(n)];
    }
    if (J_floor > 0.0 && d_xr_reference != nullptr &&
        d_xz_reference != nullptr) {
      path_admissibility_detail::copy_device_node_field(
          d_xr_reference, n_nodes, pole_xr_reference,
          "path admissibility pole macro: copy reference r failed");
      path_admissibility_detail::copy_device_node_field(
          d_xz_reference, n_nodes, pole_xz_reference,
          "path admissibility pole macro: copy reference z failed");
    }
    pole_prepare_result =
        path_admissibility_detail::prepare_pole_coarsen_overlay_host(
            pole_overlay,
            n_cells,
            pole_xr_old,
            pole_xz_old,
            pole_xr_new,
            pole_xz_new,
            std::max(0.0, J_floor),
            accepted_pole_overlay,
            pole_xr_reference.empty() ? nullptr : &pole_xr_reference,
            pole_xz_reference.empty() ? nullptr : &pole_xz_reference);
    accepted_pole_overlay_ptr = &accepted_pole_overlay;
  }
  tenryu::core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (path_admissibility_detail::has_nonquad_cell_nverts(
          state.mesh.cell_nverts, n_cells)) {
    d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }
  tenryu::core::DeviceArray<int> d_cell_orientation_sign;
  const int* d_cell_orientation_sign_ptr = nullptr;
  if (predicate_harden &&
      mb.cell_orientation_sign.size() == static_cast<std::size_t>(n_cells)) {
    d_cell_orientation_sign.reset(static_cast<std::size_t>(n_cells));
    d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
    d_cell_orientation_sign_ptr = d_cell_orientation_sign.data();
  }
  tenryu::core::DeviceArray<std::uint8_t> d_inactive_mask;
  const std::uint8_t* d_inactive_mask_ptr = nullptr;
  std::vector<std::uint8_t> inactive_mask_host;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive_mask_host = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive_mask_host.empty()) {
      inactive_mask_host.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive_mask_host[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (accepted_pole_overlay_ptr != nullptr &&
      accepted_pole_overlay_ptr->skip_fine_child_paths &&
      accepted_pole_overlay_ptr->pole_coarsen_inactive_fine_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    if (inactive_mask_host.empty()) {
      inactive_mask_host.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (accepted_pole_overlay_ptr
              ->pole_coarsen_inactive_fine_mask[static_cast<std::size_t>(c)] !=
          0U) {
        inactive_mask_host[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (!inactive_mask_host.empty()) {
    d_inactive_mask.reset(static_cast<std::size_t>(n_cells));
    d_inactive_mask.copy_from_host(inactive_mask_host);
    d_inactive_mask_ptr = d_inactive_mask.data();
  }
  tenryu::core::DeviceArray<PathAdmissibilityResult> d_results(
      static_cast<std::size_t>(n_cells));
  const int blocks = (n_cells + 255) / 256;
  const bool anatomy =
      path_admissibility_detail::path_admissibility_anatomy_enabled();
  if (anatomy) {
    if (state.mesh.corner_stride <= kMeshTopoCellStorageSlotsMax) {
      path_admissibility_detail::
          evaluate_path_admissibility_csr_velocity_kernel<true>
              <<<blocks, 256>>>(
                  d_results.data(),
                  n_cells,
                  state.mesh.corner_stride,
                  state.mesh.multiblock_cell_node_csr_offsets.data(),
                  state.mesh.multiblock_cell_node_csr_indices.data(),
                  d_xr_old,
                  d_xz_old,
                  d_v_r,
                  d_v_z,
                  dt,
                  d_xr_reference,
                  d_xz_reference,
                  d_cell_nverts_ptr,
                  d_cell_orientation_sign_ptr,
                  d_inactive_mask_ptr,
                  std::max(0.0, J_floor),
                  path_admissibility_detail::path_admissibility_r_guard_enabled());
    } else {
      path_admissibility_detail::
          evaluate_path_admissibility_csr_velocity_kernel<
              true, kMeshTopoCellStorageSlotsMaxGeneral>
              <<<blocks, 256>>>(
                  d_results.data(),
                  n_cells,
                  state.mesh.corner_stride,
                  state.mesh.multiblock_cell_node_csr_offsets.data(),
                  state.mesh.multiblock_cell_node_csr_indices.data(),
                  d_xr_old,
                  d_xz_old,
                  d_v_r,
                  d_v_z,
                  dt,
                  d_xr_reference,
                  d_xz_reference,
                  d_cell_nverts_ptr,
                  d_cell_orientation_sign_ptr,
                  d_inactive_mask_ptr,
                  std::max(0.0, J_floor),
                  path_admissibility_detail::path_admissibility_r_guard_enabled());
    }
  } else {
    if (state.mesh.corner_stride <= kMeshTopoCellStorageSlotsMax) {
      path_admissibility_detail::
          evaluate_path_admissibility_csr_velocity_kernel<false>
              <<<blocks, 256>>>(
                  d_results.data(),
                  n_cells,
                  state.mesh.corner_stride,
                  state.mesh.multiblock_cell_node_csr_offsets.data(),
                  state.mesh.multiblock_cell_node_csr_indices.data(),
                  d_xr_old,
                  d_xz_old,
                  d_v_r,
                  d_v_z,
                  dt,
                  d_xr_reference,
                  d_xz_reference,
                  d_cell_nverts_ptr,
                  d_cell_orientation_sign_ptr,
                  d_inactive_mask_ptr,
                  std::max(0.0, J_floor),
                  path_admissibility_detail::path_admissibility_r_guard_enabled());
    } else {
      path_admissibility_detail::
          evaluate_path_admissibility_csr_velocity_kernel<
              false, kMeshTopoCellStorageSlotsMaxGeneral>
              <<<blocks, 256>>>(
                  d_results.data(),
                  n_cells,
                  state.mesh.corner_stride,
                  state.mesh.multiblock_cell_node_csr_offsets.data(),
                  state.mesh.multiblock_cell_node_csr_indices.data(),
                  d_xr_old,
                  d_xz_old,
                  d_v_r,
                  d_v_z,
                  dt,
                  d_xr_reference,
                  d_xz_reference,
                  d_cell_nverts_ptr,
                  d_cell_orientation_sign_ptr,
                  d_inactive_mask_ptr,
                  std::max(0.0, J_floor),
                  path_admissibility_detail::path_admissibility_r_guard_enabled());
    }
  }
  path_admissibility_detail::cuda_check(
      cudaGetLastError(),
      "evaluate_path_admissibility velocity kernel launch failed");
  path_admissibility_detail::cuda_check(
      cudaDeviceSynchronize(),
      "evaluate_path_admissibility velocity kernel failed");

  std::vector<PathAdmissibilityResult> host;
  d_results.copy_to_host(host);
  const auto pole_deref_repair_cell_mask =
      path_admissibility_detail::build_pole_deref_repair_cell_mask(
          accepted_pole_overlay_ptr, n_cells);
  for (auto& cell_result : host) {
    if (anatomy) {
      path_admissibility_detail::annotate_path_admissibility_location(
          cell_result, state);
    }
    path_admissibility_detail::gate_active_fine_child_macro_repair(
        cell_result, pole_deref_repair_cell_mask);
    path_admissibility_detail::merge_path_admissibility_result(result,
                                                               cell_result);
  }
  path_admissibility_detail::merge_path_admissibility_result(
      result,
      path_admissibility_detail::evaluate_macro_boundary_path_velocity(
          state, d_xr_old, d_xz_old, d_v_r, d_v_z, dt,
          std::max(0.0, J_floor), d_xr_reference, d_xz_reference));
  if (accepted_pole_overlay_ptr != nullptr) {
    path_admissibility_detail::merge_path_admissibility_result(
        result,
        path_admissibility_detail::evaluate_pole_coarsen_boundary_path_host(
            *accepted_pole_overlay_ptr,
            pole_xr_old,
            pole_xz_old,
            pole_xr_new,
            pole_xz_new,
            std::max(0.0, J_floor),
            pole_xr_reference.empty() ? nullptr : &pole_xr_reference,
            pole_xz_reference.empty() ? nullptr : &pole_xz_reference));
    path_admissibility_detail::merge_path_admissibility_result(
        result, pole_prepare_result);
  }
  if (path_admissibility_detail::env_flag_enabled(
          "TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE") &&
      result.first_failing_cell >= 0 &&
      result.first_failing_cell < n_cells &&
      state.pole_angular_derefine.inactive_member_mask.size() ==
          static_cast<std::size_t>(n_cells)) {
    TENRYU_ASSERT(
        state.pole_angular_derefine
                .inactive_member_mask[static_cast<std::size_t>(
                    result.first_failing_cell)] == 0U,
        "path admissibility velocity check selected an inactive polar shell de-refine child");
  }
  return result;
}

inline MeshForecast evaluate_mesh_forecast(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    const double J_floor,
    const double q_warn,
    const PathAdmissibilityResult& path_result,
    const tenryu::hydro::pole_angular_coarsen::Overlay* pole_overlay =
        nullptr) {
  MeshForecast forecast;
  if (!mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.n_cells <= 0) {
    return forecast;
  }
  TENRYU_ASSERT(d_xr_old != nullptr && d_xz_old != nullptr &&
                    d_xr_new != nullptr && d_xz_new != nullptr,
                "evaluate_mesh_forecast requires non-null coordinate arrays");
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> xr_new;
  std::vector<double> xz_new;
  path_admissibility_detail::copy_device_node_field(
      d_xr_old, n_nodes, xr_old, "mesh forecast: copy old r failed");
  path_admissibility_detail::copy_device_node_field(
      d_xz_old, n_nodes, xz_old, "mesh forecast: copy old z failed");
  path_admissibility_detail::copy_device_node_field(
      d_xr_new, n_nodes, xr_new, "mesh forecast: copy new r failed");
  path_admissibility_detail::copy_device_node_field(
      d_xz_new, n_nodes, xz_new, "mesh forecast: copy new z failed");
  return path_admissibility_detail::evaluate_mesh_forecast_host(
      state, xr_old, xz_old, xr_new, xz_new, std::max(0.0, J_floor), q_warn,
      path_result, pole_overlay);
}

inline MeshForecast evaluate_mesh_forecast(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_v_r,
    const double* d_v_z,
    const double dt,
    const double J_floor,
    const double q_warn,
    const PathAdmissibilityResult& path_result,
    const tenryu::hydro::pole_angular_coarsen::Overlay* pole_overlay =
        nullptr) {
  MeshForecast forecast;
  if (!mesh_topo_is_multiblock(cfg.mesh) ||
      !state.mesh.topo.multiblock.has_value() ||
      state.mesh.topo.n_cells <= 0) {
    return forecast;
  }
  TENRYU_ASSERT(d_xr_old != nullptr && d_xz_old != nullptr &&
                    d_v_r != nullptr && d_v_z != nullptr,
                "evaluate_mesh_forecast requires non-null coordinate and velocity arrays");
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "evaluate_mesh_forecast requires finite dt > 0");
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> xr_old;
  std::vector<double> xz_old;
  std::vector<double> vr;
  std::vector<double> vz;
  path_admissibility_detail::copy_device_node_field(
      d_xr_old, n_nodes, xr_old, "mesh forecast: copy old r failed");
  path_admissibility_detail::copy_device_node_field(
      d_xz_old, n_nodes, xz_old, "mesh forecast: copy old z failed");
  path_admissibility_detail::copy_device_node_field(
      d_v_r, n_nodes, vr, "mesh forecast: copy velocity r failed");
  path_admissibility_detail::copy_device_node_field(
      d_v_z, n_nodes, vz, "mesh forecast: copy velocity z failed");
  std::vector<double> xr_new(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> xz_new(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    xr_new[static_cast<std::size_t>(n)] =
        xr_old[static_cast<std::size_t>(n)] +
        dt * vr[static_cast<std::size_t>(n)];
    xz_new[static_cast<std::size_t>(n)] =
        xz_old[static_cast<std::size_t>(n)] +
        dt * vz[static_cast<std::size_t>(n)];
  }
  return path_admissibility_detail::evaluate_mesh_forecast_host(
      state, xr_old, xz_old, xr_new, xz_new, std::max(0.0, J_floor), q_warn,
      path_result, pole_overlay);
}

}  // namespace tenryu::mesh
