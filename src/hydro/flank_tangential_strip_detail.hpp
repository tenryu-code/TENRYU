#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro::detail {

struct FlankStripRowPolyline {
  int i = -1;
  std::vector<int> nodes;
  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> s;
  std::vector<double> target_s;
  std::vector<double> delta_s;
  std::vector<std::uint8_t> fixed;
  bool targets_valid = false;
};

struct FlankStripWindowPolylines {
  int slot_cell = -1;
  int slot_j = -1;
  int i_last = -1;
  int j_first = -1;
  int j_last = -1;
  int window_size = 0;
  std::vector<std::uint8_t> fixed_node;
  std::vector<FlankStripRowPolyline> rows;
  bool valid = false;
};

int active_flank_strip_slot_cell(const core::State& state);

void flank_strip_seam_window(const core::State& state,
                             int slot_cell,
                             int band_halfwidth_j,
                             int nz,
                             int& j_first,
                             int& j_last,
                             int focus_j = -1);

void flank_strip_extend_reference_for_window(core::State& state,
                                             const core::Config& cfg,
                                             int slot_cell,
                                             int band_layers,
                                             int band_halfwidth_j,
                                             int focus_j);

bool flank_strip_cell_is_geometry_policy_exempt(const core::State& state,
                                                int cell);

bool flank_strip_node_is_contact_owned(const core::State& state, int node);

double flank_strip_slip_mismatch_signed_ratio(const core::State& state,
                                              int j_first,
                                              int j_last);

double flank_strip_slip_mismatch_signed_ratio(
    const core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    int j_first,
    int j_last);

FlankStripWindowPolylines build_flank_strip_window_polylines(
    const core::State& state,
    const core::Config& cfg,
    int slot_cell,
    const std::vector<double>& original_r,
    const std::vector<double>& original_z,
    int focus_j);

bool build_flank_strip_row_targets(
    const core::State& state,
    FlankStripRowPolyline& row);

double flank_strip_row_blend(int i, int band_layers);

bool interpolate_flank_strip_polyline(
    const FlankStripRowPolyline& row,
    double candidate_s,
    double& candidate_r,
    double& candidate_z);

double flank_strip_band_minimum_quality_ratio(
    const core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    int slot_cell,
    int band_layers,
    int band_halfwidth_j,
    int& argmin_cell,
    int focus_j);

void run_local_jacobian_untangler(
    const core::State& state,
    int target_cell,
    int i_first,
    int i_last,
    int j_first,
    int j_last,
    const std::vector<std::uint8_t>& fixed_node,
    std::vector<double>& candidate_r,
    std::vector<double>& candidate_z);

}  // namespace tenryu::hydro::detail
