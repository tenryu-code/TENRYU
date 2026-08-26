#pragma once

#include <cstddef>
#include <cstdint>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::diagnostics::mesh_attribution {

enum class MeshDeformSource : std::uint8_t {
  KinematicCarry = 0,
  HydroPressure,
  ArtificialViscosity,
  LaserHeating,
  SpitzerConduction,
  FLDEnergyCoupling,
  FLDRadiationPressure,
  SNTransport,
  RaytraceDeposition,
  ALERezone,
  Remap,
  Other,
  Count
};

inline constexpr int kMeshDeformSourceCount =
    static_cast<int>(MeshDeformSource::Count);

struct MeshFailureAttributionRecord {
  std::uint64_t step = 0;
  double t_simulation = 0.0;
  std::uint64_t mesh_failure_id = 0;
  int failing_cell = -1;
  int failing_corner = -1;
  double dJ_source[kMeshDeformSourceCount] = {};
  std::uint8_t dominant_source = static_cast<std::uint8_t>(MeshDeformSource::Other);
  double dominant_fraction = 0.0;
  double nonlinear_residual = 0.0;
  // Axis-failure telemetry. Zero-initialized here; attribution kernels may
  // populate these only when TENRYU_STAGE24_TELEMETRY=1.
  double d_area_dt_by_source[kMeshDeformSourceCount] = {};
  double d_rzvol_dt_by_source[kMeshDeformSourceCount] = {};
  double d_radial_gap_dt_by_source[kMeshDeformSourceCount] = {};
  double d_z_order_dt_by_source[kMeshDeformSourceCount] = {};
  double d_remap_intermediate_dt_by_source[kMeshDeformSourceCount] = {};
  double dt_star_area = 0.0;
  double dt_star_rzvol = 0.0;
  double dt_star_radial_gap = 0.0;
  double dt_star_z_order = 0.0;
  double dt_star_remap_intermediate = 0.0;
};

struct AxisProjectionRecord {
  std::uint64_t step = 0;
  double t_simulation = 0.0;
  const char* operator_name = "AxisVariationalProjection";
  const char* phase_tag = "accepted";
  int patch_i0 = 0;
  int patch_i1 = 0;
  int patch_j0 = 0;
  int patch_j1 = 0;
  double min_local_j = 0.0;
  int post_failing_cell = -1;
  int post_failing_corner = -1;
  double pre_request_min_quality = 0.0;
  double post_min_quality = 0.0;
};

struct LeaveOneOutReplayResult {
  bool available = false;
  const char* reason = "not_run";
};

class MeshDeformAttributionWorkspace {
 public:
  MeshDeformAttributionWorkspace() = default;
  ~MeshDeformAttributionWorkspace();

  MeshDeformAttributionWorkspace(const MeshDeformAttributionWorkspace&) = delete;
  MeshDeformAttributionWorkspace& operator=(const MeshDeformAttributionWorkspace&) =
      delete;

  bool begin_step(const core::State& state,
                  const core::Config& cfg,
                  double t_simulation,
                  const char* hydro_half);
  void end_step_success(const core::State& state, const core::Config& cfg);
  void discard();

  [[nodiscard]] bool active() const noexcept {
    return active_;
  }

  [[nodiscard]] std::size_t allocated_node_count() const noexcept {
    return n_nodes_;
  }

  void clear_lagrangian_sources();
  void record_displacement_delta(MeshDeformSource source,
                                 const double* d_delta_r,
                                 const double* d_delta_z,
                                 int n_nodes);
  void record_kinematic_carry(const double* d_v_r,
                              const double* d_v_z,
                              const std::uint8_t* d_node_active,
                              int n_nodes,
                              double dt_scale);
  void record_acceleration_displacement(MeshDeformSource source,
                                        const double* d_a_r,
                                        const double* d_a_z,
                                        const std::uint8_t* d_node_active,
                                        int n_nodes,
                                        double coefficient);
  void record_direct_displacement(MeshDeformSource source,
                                  const double* d_r_before,
                                  const double* d_z_before,
                                  const double* d_r_after,
                                  const double* d_z_after,
                                  int n_nodes);

  MeshFailureAttributionRecord compute_failure_record(
      const core::State& state,
      const core::Config& cfg,
      int failing_cell,
      int failing_corner,
      const double* d_candidate_r = nullptr,
      const double* d_candidate_z = nullptr);

  MeshFailureAttributionRecord compute_aggregate_record(const core::State& state,
                                                        const core::Config& cfg);

  void emit_failure_record(const core::State& state,
                           const core::Config& cfg,
                           int failing_cell,
                           int failing_corner,
                           const double* d_candidate_r = nullptr,
                           const double* d_candidate_z = nullptr);

  [[nodiscard]] LeaveOneOutReplayResult leave_one_out_replay_stub(
      MeshDeformSource source) const;

 private:
  void release();
  void ensure_capacity(std::size_t n_nodes, std::size_t n_cells);
  double* source_delta_r(MeshDeformSource source);
  double* source_delta_z(MeshDeformSource source);
  const double* source_delta_r(MeshDeformSource source) const;
  const double* source_delta_z(MeshDeformSource source) const;

  bool active_ = false;
  bool dump_on_failure_only_ = true;
  std::size_t n_nodes_ = 0;
  std::size_t n_cells_ = 0;
  std::uint64_t step_ = 0;
  double t_simulation_ = 0.0;
  double* d_start_r_ = nullptr;
  double* d_start_z_ = nullptr;
  double* d_delta_r_ = nullptr;
  double* d_delta_z_ = nullptr;
  double* d_dj_source_ = nullptr;
  double* d_actual_delta_j_ = nullptr;
  std::int8_t* d_hydro_active_ = nullptr;
};

MeshDeformAttributionWorkspace& global_workspace();

bool begin_step(const core::State& state,
                const core::Config& cfg,
                double t_simulation,
                const char* hydro_half);
void end_step_success(const core::State& state, const core::Config& cfg);
void discard();
void emit_failure_record(const core::State& state,
                         const core::Config& cfg,
                         int failing_cell,
                         int failing_corner,
                         const double* d_candidate_r = nullptr,
                         const double* d_candidate_z = nullptr);
void emit_axis_projection_record(const core::State& state,
                                 const core::Config& cfg,
                                 const AxisProjectionRecord& record);
void record_zero_source(MeshDeformSource source);

}  // namespace tenryu::diagnostics::mesh_attribution
