#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
class Reduction;
}  // namespace tenryu::parallel

namespace tenryu::hydro {

struct EulerWindowSpec {
  enum class Shape : std::uint8_t {
    Rectangle = 0,
    Annulus = 1,
  };

  Shape shape;
  double r0;
  double r1;
  double z0;
  double z1;
  double cr;
  double cz;
  double rad_in;
  double rad_out;
  double transition_width;
};

void euler_window_cell_weights(const EulerWindowSpec& spec,
                               const double* cell_cr,
                               const double* cell_cz,
                               int n_cells,
                               double* w_cell);

void euler_window_node_weights(const int* cell_node_csr_offsets,
                               const int* cell_node_csr_indices,
                               const std::uint8_t* cell_nverts,
                               int n_cells,
                               int corner_stride,
                               int n_nodes,
                               const double* w_cell,
                               double* w_node);

void euler_window_node_weights(
    const std::vector<EulerWindowSpec>& specs,
    const double* cell_cr,
    const double* cell_cz,
    const int* cell_node_csr_offsets,
    const int* cell_node_csr_indices,
    const std::uint8_t* cell_nverts,
    int n_cells,
    int corner_stride,
    int n_nodes,
    double* w_node);

void euler_window_blend_targets(const double* w_node,
                                int n_nodes,
                                const double* x_lagr_r,
                                const double* x_lagr_z,
                                const double* x_euler_r,
                                const double* x_euler_z,
                                double* x_target_r,
                                double* x_target_z);

struct AxisCoreDiskObservation {
  double realized_s_p = 0.0;
  double realized_s_f = 0.0;
  std::vector<double> initial_radius;
  std::vector<double> chi;
  std::vector<double> captured_target_r;
  std::vector<double> captured_target_z;
  std::vector<int> plateau_cells;
  std::vector<int> feather_cells;
  std::vector<int> guard_cells;
};

struct AxisCoreFollowRepairDecision {
  bool repair = false;
  long long follow_count = 0;
  long long repair_count = 0;
};

struct AxisCoreJunctionLedgerCell {
  int belt_block_id = -1;
  int cell = -1;
};

struct AxisCoreTransitionPassagePatchStep {
  int patch_index = -1;
  int belt_block_id = -1;
  double shock_radius = 0.0;
};

struct AxisCoreTransitionPassageStep {
  bool enabled = false;
  std::vector<AxisCoreTransitionPassagePatchStep> passage_patches;
};

bool axis_core_ring_release_enabled(const core::Config& cfg);

int axis_core_released_units();

void axis_core_set_released_units(int n);

void build_axis_core_disk(core::State& state, const core::Config& cfg);

// Demand-driven ring release: shift the axis-core plateau boundary inward by
// one structural unit and rebuild the disk runtime. Returns false (with the
// counter rolled back and the previous runtime restored) when the unit
// ladder cannot support a further release. State fields are untouched: only
// the transaction TARGET for subsequent steps changes.
bool axis_core_disk_release_one_unit(core::State& state,
                                     const core::Config& cfg);

void destroy_axis_core_disk();

AxisCoreDiskObservation axis_core_disk_observe(const core::State& state);

std::vector<AxisCoreJunctionLedgerCell>
axis_core_disk_junction_ledger_cells(const core::State& state);

const std::uint8_t*
axis_core_disk_seam_donor_cell_mask(const core::State& state);

void axis_core_disk_build_target(const core::State& state,
                                 std::vector<double>& target_r,
                                 std::vector<double>& target_z);

double axis_core_disk_build_moving_target(
    const core::State& state,
    std::vector<double>& lagrangian_r,
    std::vector<double>& lagrangian_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z);

AxisCoreTransitionPassageStep axis_core_transition_passage_update(
    core::State& state,
    double time,
    bool emit_logs,
    const parallel::Reduction* reduction);

void axis_core_transition_passage_prepare_optimal_targets(
    const core::State& state);

void axis_core_transition_passage_compose_patch_target(
    int patch_index,
    double alpha,
    const std::vector<double>& lagrangian_r,
    const std::vector<double>& lagrangian_z,
    const std::vector<double>& normal_target_r,
    const std::vector<double>& normal_target_z,
    std::vector<double>& target_r,
    std::vector<double>& target_z);

const std::uint8_t* axis_core_transition_passage_inactive_cell_mask(
    const core::State& state,
    int patch_index);

double axis_core_transition_passage_max_swept_fraction(
    const core::State& state,
    int patch_index,
    const std::vector<double>& candidate_r,
    const std::vector<double>& candidate_z);

void axis_core_transition_passage_record_alpha(
    int patch_index,
    double alpha,
    long long step,
    double time,
    bool emit_logs);

AxisCoreFollowRepairDecision axis_core_disk_select_always_moving();

void axis_core_disk_assert_plateau(const core::State& state);

}  // namespace tenryu::hydro
