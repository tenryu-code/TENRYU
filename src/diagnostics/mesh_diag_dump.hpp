#pragma once

#include <string>

#include "core/config.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::diagnostics::mesh_diag {

bool enabled_for_step(const core::State& state, const core::Config& cfg);

struct I1BPolarPoleAleCounters {
  const char* phase = "ale";
  const char* force_reason = "";
  bool axis_guard_trigger = false;
  bool force_rezone_input = false;
  bool forced_rezone_fired = false;
  bool rezone_triggered = false;
  bool retry_request_path = false;
  bool escape_valve_or_repair_fired = false;
};

void dump_i1b_polar_pole_diag(const core::State& state,
                              const core::Config& cfg,
                              const I1BPolarPoleAleCounters& counters);

void dump_hydro_cells(const core::State& state,
                      const core::Config& cfg,
                      const char* phase,
                      double dt,
                      const core::NodeField1D& x_r,
                      const core::NodeField1D& x_z,
                      const core::NodeField1D& v_r,
                      const core::NodeField1D& v_z,
                      const core::CellField1D* cs,
                      const core::CellField1D* div_u,
                      const char* dt_limiter_source,
                      const char* hydro_half);

void dump_hydro_trial(const core::State& state,
                      const core::Config& cfg,
                      const char* phase,
                      double dt,
                      const core::NodeField1D& r_base,
                      const core::NodeField1D& z_base,
                      const core::NodeField1D& pos_v_r,
                      const core::NodeField1D& pos_v_z,
                      const core::CellField1D* cs,
                      const core::CellField1D* div_u,
                      const char* dt_limiter_source,
                      const char* hydro_half);

void dump_trial_volume_cfl(const core::State& state,
                           const core::Config& cfg,
                           bool enabled,
                           bool admissible,
                           double min_trial_vol_ratio,
                           double suggested_dt,
                           int first_failing_cell);

void dump_ale_predictive_acceptance(const core::State& state,
                                    const core::Config& cfg,
                                    bool active,
                                    bool feasible,
                                    const char* failure_class,
                                    int axis_failure_count,
                                    int cell_vol_failure_count,
                                    int first_axis_failing_j,
                                    int first_vol_failing_c,
                                    double candidate_axis_margin_min,
                                    double trial_axis_margin_min,
                                    double candidate_cell_vol_min,
                                    double trial_cell_vol_min);

void dump_ale_backtrack_iter(const core::State& state,
                             const core::Config& cfg,
                             int attempt,
                             double lambda,
                             bool post_tangle,
                             bool post_corner_tangle,
                             double trial_quality,
                             double min_corner_j,
                             int post_min_quality_cell_c,
                             int post_min_quality_cell_i,
                             int post_min_quality_cell_j,
                             int post_corner_fail_cell_c,
                             int post_corner_fail_cell_i,
                             int post_corner_fail_cell_j,
                             int post_corner_fail_corner,
                             bool accepted,
                             const char* rejection_reason);

void dump_ale_rezone_end(const core::State& state,
                         const core::Config& cfg,
                         int rollback_count,
                         const char* last_reason,
                         double axis_margin_min,
                         bool accepted_axis_inflow_this_event,
                         int accepted_axis_inflow_count,
                         double accepted_axis_inflow_max,
                         int rezone_iterations,
                         int min_quality_cell_pre_c,
                         int min_quality_cell_pre_i,
                         int min_quality_cell_pre_j,
                         int min_quality_cell_post_c,
                         int min_quality_cell_post_i,
                         int min_quality_cell_post_j);

}  // namespace tenryu::diagnostics::mesh_diag
