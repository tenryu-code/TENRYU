#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::hydro {

struct HydroEOSContext;
struct MeshQualityDtLimit;

#ifdef __CUDACC__
#define TENRYU_RING7_HD __host__ __device__
#else
#define TENRYU_RING7_HD
#endif

struct Ring7RzPoint {
  double r = 0.0;
  double z = 0.0;
};

struct Ring7RzVector {
  double r = 0.0;
  double z = 0.0;
};

TENRYU_RING7_HD inline Ring7RzVector axisymmetric_face_area_vector(
    const Ring7RzPoint a,
    const Ring7RzPoint b) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double scale = pi * (a.r + b.r);
  return {scale * (b.z - a.z), -scale * (b.r - a.r)};
}

TENRYU_RING7_HD inline double axisymmetric_swept_volume_polygon(
    const Ring7RzPoint* points,
    const int n_points) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  double sum = 0.0;
  for (int k = 0; k < n_points; ++k) {
    const Ring7RzPoint a = points[k];
    const Ring7RzPoint b = points[(k + 1) % n_points];
    sum += (a.r + b.r) * (a.r * b.z - b.r * a.z);
  }
  return (pi / 3.0) * sum;
}

struct Ring7CompensatedSum {
  double sum = 0.0;
  double compensation = 0.0;

  TENRYU_RING7_HD void add(const double value) {
    const double y = value - compensation;
    const double t = sum + y;
    compensation = (t - sum) - y;
    sum = t;
  }

  TENRYU_RING7_HD double value() const {
    return sum;
  }
};

struct Ring7PacketExtensives {
  double mass = 0.0;
  double momentum_r = 0.0;
  double momentum_z = 0.0;
  double total_energy = 0.0;
};

struct Ring7PacketAccumulator {
  Ring7CompensatedSum mass;
  Ring7CompensatedSum momentum_r;
  Ring7CompensatedSum momentum_z;
  Ring7CompensatedSum total_energy;

  TENRYU_RING7_HD void add(const Ring7PacketExtensives packet) {
    mass.add(packet.mass);
    momentum_r.add(packet.momentum_r);
    momentum_z.add(packet.momentum_z);
    total_energy.add(packet.total_energy);
  }

  TENRYU_RING7_HD Ring7PacketExtensives value() const {
    return {mass.value(),
            momentum_r.value(),
            momentum_z.value(),
            total_energy.value()};
  }
};

TENRYU_RING7_HD inline Ring7PacketExtensives
accumulate_packets_deterministic(const Ring7PacketExtensives* packets,
                                 const int n_packets) {
  Ring7PacketAccumulator accumulator;
  for (int i = 0; i < n_packets; ++i) {
    accumulator.add(packets[i]);
  }
  return accumulator.value();
}

struct Ring7SeamPatch {
  bool valid = false;
  int shell_ring_i = -1;
  int shell_guard_i = -1;
  int shell_nj = 0;
  int shell_cell_begin = -1;
  std::vector<int> ring7_cells;
  std::vector<int> inner_guard_cells;
  std::vector<int> butterfly_adjacent_cells;
  std::vector<int> trifan_end_cells;
  std::vector<int> p_seam_cells;
  std::vector<int> node_ids;
};

struct Ring7SeamRemapResult {
  bool attempted = false;
  bool requested = false;
  bool fired = false;
  bool dt_reduction_required = false;
  double suggested_dt = 0.0;
  double min_q_J = 0.0;
  double min_q_edge = 0.0;
  double min_eta_V = 0.0;
  double max_swept_fraction = 0.0;
  int moved_node_count = 0;
  int request_cell = -1;
};

Ring7SeamPatch discover_ring7_seam_patch(const core::State& state,
                                         const core::Config& cfg);

bool ring7_seam_retry_candidate_cell(const core::State& state,
                                     const core::Config& cfg,
                                     int cell);

// Interior-patch generalization (env TENRYU_I1B_INTERIOR_PATCH_REMAP; needs
// TENRYU_I1B_RING7_QUOTIENT=1): true when the failing cell is a POLAR_SHELL
// interior cell eligible for an on-demand conservative patch rezone+remap.
bool interior_patch_retry_candidate_cell(const core::State& state,
                                         const core::Config& cfg,
                                         int cell);

bool ring7_driven_pole_cap_retry_candidate_cell(const core::State& state,
                                                const core::Config& cfg,
                                                int cell);

void request_ring7_seam_rezone(core::State& state, int failing_cell);

void request_ring7_pole_cap_oracle(core::State& state, int failing_cell);

void record_ring7_failed_mesh_velocity(core::State& state,
                                       const core::Config& cfg,
                                       int failing_cell,
                                       double sigma_safe,
                                       double dt_trial,
                                       const double* d_v_r_half,
                                       const double* d_v_z_half);

void emit_ring7_pole_cap_recomputed_limiter_validation(
    core::State& state,
    const core::Config& cfg,
    const MeshQualityDtLimit& limit,
    double dt_trial,
    const char* stage);

Ring7SeamRemapResult apply_ring7_seam_quotient_remap(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const double* d_v_r_nominal,
    const double* d_v_z_nominal,
    double dt_nominal,
    const parallel::Reduction* reduction,
    const char* stage);

void emit_ring7_seam_scaffold_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    double dt,
    const parallel::Reduction* reduction,
    const char* stage);

#undef TENRYU_RING7_HD

}  // namespace tenryu::hydro
