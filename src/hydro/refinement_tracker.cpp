#include "hydro/refinement_tracker.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kArrivalSpeedFloorCmPerS = 1.0e3;

enum class TrackerState { SEARCH, TRACKED, ARMED };

struct TrackedBlock {
  std::size_t block_index = 0;
  bool layers_ascending = true;
  double mean_mid_column_s = 0.0;
};

struct TrackerLatticeView {
  bool single_block = false;
  bool layer_axis_is_j = false;
  int n_layers = 0;
  int n_across = 0;
};

struct TrackerCache {
  std::size_t built_cell_count = 0;
  bool table_built = false;
  bool built_single_block = false;
  bool built_layer_axis_is_j = false;
  int built_n_layers = 0;
  int built_n_across = 0;
  int n_ref = 0;
  std::vector<std::vector<int>> columns;
  std::vector<int> severity_path;

  TrackerState state = TrackerState::SEARCH;
  int search_promote_count = 0;
  int tracked_demote_count = 0;
  int arm_promote_count = 0;
  int armed_demote_count = 0;
  std::deque<double> observation_t;
  std::deque<double> observation_s;
  std::size_t would_fire_count = 0;
  AutopilotFireRecord fire_record;
  bool front_shell_row_valid = false;
  int front_shell_row = -1;
  double front_coverage = 0.0;
};

TrackerCache& tracker_cache() {
  static TrackerCache cache;
  return cache;
}

struct TrackerFrontTestOverride {
  bool enabled = false;
  int shell_row = -1;
  double coverage = 0.0;
};

TrackerFrontTestOverride& tracker_front_test_override() {
  static TrackerFrontTestOverride override;
  return override;
}

template <typename Tag>
std::vector<double> copy_field(const core::Field1D<Tag>& field) {
  std::vector<double> out;
  field.copy_to_host(out);
  return out;
}

bool make_tracker_lattice_view(const core::State& state,
                               const core::Config& cfg,
                               TrackerLatticeView& view) {
  view = TrackerLatticeView{};
  if (state.mesh.topo.multiblock.has_value()) {
    return true;
  }
  const std::size_t n_cells = state.rho.size();
  if (state.mesh.dim != 2 || state.mesh.topo.nr < 5 ||
      state.mesh.topo.nz < 5 ||
      n_cells != static_cast<std::size_t>(state.mesh.topo.nr) *
                     static_cast<std::size_t>(state.mesh.topo.nz)) {
    return false;
  }

  view.single_block = true;
  const bool auto_axis_is_i =
      cfg.mesh.logical_mesh_2d.rfind("spherical_polar", 0) == 0;
  if (cfg.numerics.ale.band_ale.enabled &&
      cfg.numerics.ale.band_ale.bands == "estimator") {
    const std::string& configured_axis =
        cfg.numerics.ale.band_ale.estimator_band_axis;
    TENRYU_ASSERT(
        configured_axis == "auto" || configured_axis == "i" ||
            configured_axis == "j",
        "refinement tracker lattice axis must be auto, i, or j");
    // The tracker and band must agree because the front hold compares rows.
    view.layer_axis_is_j =
        configured_axis == "j" ||
        (configured_axis == "auto" && !auto_axis_is_i);
  } else {
    view.layer_axis_is_j = !auto_axis_is_i;
  }
  view.n_layers =
      view.layer_axis_is_j ? state.mesh.topo.nz : state.mesh.topo.nr;
  view.n_across =
      view.layer_axis_is_j ? state.mesh.topo.nr : state.mesh.topo.nz;
  return true;
}

void cell_centroid(const mesh::MultiBlockTopology& mb,
                   const int cell,
                   const std::vector<double>& node_r,
                   const std::vector<double>& node_z,
                   double& centroid_r,
                   double& centroid_z) {
  // Match refinement_estimator.cpp's four-node centroid convention.
  const int offset =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  centroid_r = 0.0;
  centroid_z = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int node =
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + k)];
    centroid_r += node_r[static_cast<std::size_t>(node)];
    centroid_z += node_z[static_cast<std::size_t>(node)];
  }
  centroid_r *= 0.25;
  centroid_z *= 0.25;
}

void structured_cell_centroid(const mesh::MeshTopology& topo,
                              const int cell,
                              const std::vector<double>& node_r,
                              const std::vector<double>& node_z,
                              double& centroid_r,
                              double& centroid_z) {
  const int ci = cell / topo.nz;
  const int cj = cell - ci * topo.nz;
  centroid_r = 0.0;
  centroid_z = 0.0;
  for (int di = 0; di < 2; ++di) {
    for (int dj = 0; dj < 2; ++dj) {
      const int node = topo.node_index(ci + di, cj + dj);
      centroid_r += node_r[static_cast<std::size_t>(node)];
      centroid_z += node_z[static_cast<std::size_t>(node)];
    }
  }
  centroid_r *= 0.25;
  centroid_z *= 0.25;
}

double median_sorted(const std::vector<double>& sorted) {
  if (sorted.empty()) {
    return 0.0;
  }
  const std::size_t n = sorted.size();
  if ((n & 1U) != 0U) {
    return sorted[n / 2U];
  }
  return 0.5 * (sorted[n / 2U - 1U] + sorted[n / 2U]);
}

double sorted_median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return median_sorted(values);
}

void build_column_table(TrackerCache& cache,
                        const mesh::MultiBlockTopology& mb,
                        const std::size_t n_cells,
                        const std::vector<double>& centroid_s,
                        const std::vector<double>& centroid_theta) {
  cache.table_built = true;
  cache.built_cell_count = n_cells;
  cache.n_ref = 0;
  cache.columns.clear();
  cache.severity_path.clear();

  std::vector<TrackedBlock> severity_blocks;
  severity_blocks.reserve(mb.blocks.size());
  for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
    const auto& block = mb.blocks[b];
    if (block.n_i_cells < 1 || block.n_j_cells < 1 ||
        block.cell_count != block.n_i_cells * block.n_j_cells) {
      continue;
    }
    const int mid_column = block.n_j_cells / 2;
    const int first_cell = block.cell_begin + mid_column;
    const int last_cell = block.cell_begin +
                          (block.n_i_cells - 1) * block.n_j_cells +
                          mid_column;
    const double radial_delta =
        centroid_s[static_cast<std::size_t>(last_cell)] -
        centroid_s[static_cast<std::size_t>(first_cell)];
    double mean_s = 0.0;
    for (int layer = 0; layer < block.n_i_cells; ++layer) {
      const int cell =
          block.cell_begin + layer * block.n_j_cells + mid_column;
      mean_s += centroid_s[static_cast<std::size_t>(cell)];
    }
    mean_s /= static_cast<double>(block.n_i_cells);
    severity_blocks.push_back({b, radial_delta >= 0.0, mean_s});
  }
  std::stable_sort(
      severity_blocks.begin(),
      severity_blocks.end(),
      [](const TrackedBlock& lhs, const TrackedBlock& rhs) {
        return lhs.mean_mid_column_s < rhs.mean_mid_column_s;
      });
  for (const TrackedBlock& tracked : severity_blocks) {
    const auto& block = mb.blocks[tracked.block_index];
    const int mid_column = block.n_j_cells / 2;
    if (tracked.layers_ascending) {
      for (int layer = 0; layer < block.n_i_cells; ++layer) {
        cache.severity_path.push_back(
            block.cell_begin + layer * block.n_j_cells + mid_column);
      }
    } else {
      for (int layer = block.n_i_cells - 1; layer >= 0; --layer) {
        cache.severity_path.push_back(
            block.cell_begin + layer * block.n_j_cells + mid_column);
      }
    }
  }

  std::size_t reference_block = mb.blocks.size();
  int largest_cell_count = -1;
  for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
    const auto& block = mb.blocks[b];
    if (block.cell_count == block.n_i_cells * block.n_j_cells &&
        block.cell_count > largest_cell_count) {
      reference_block = b;
      largest_cell_count = block.cell_count;
      cache.n_ref = block.n_j_cells;
    }
  }
  if (reference_block == mb.blocks.size() || cache.n_ref <= 0) {
    return;
  }

  std::vector<std::size_t> participants;
  for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
    const auto& block = mb.blocks[b];
    if (block.cell_count == block.n_i_cells * block.n_j_cells &&
        block.n_j_cells == cache.n_ref && block.n_i_cells >= 1) {
      participants.push_back(b);
    }
  }
  if (participants.empty()) {
    return;
  }

  const auto& alignment_reference = mb.blocks[participants.front()];
  const int reference_mid_layer = alignment_reference.n_i_cells / 2;
  std::vector<double> reference_theta(
      static_cast<std::size_t>(cache.n_ref), 0.0);
  for (int column = 0; column < cache.n_ref; ++column) {
    const int cell = alignment_reference.cell_begin +
                     reference_mid_layer * cache.n_ref + column;
    reference_theta[static_cast<std::size_t>(column)] =
        centroid_theta[static_cast<std::size_t>(cell)];
  }
  std::vector<double> angular_pitch;
  angular_pitch.reserve(static_cast<std::size_t>(cache.n_ref - 1));
  for (int column = 1; column < cache.n_ref; ++column) {
    const double pitch = std::abs(
        reference_theta[static_cast<std::size_t>(column)] -
        reference_theta[static_cast<std::size_t>(column - 1)]);
    if (std::isfinite(pitch) && pitch > 0.0) {
      angular_pitch.push_back(pitch);
    }
  }
  const double median_angular_pitch = sorted_median(angular_pitch);
  const double alignment_tolerance = 0.51 * median_angular_pitch;

  std::vector<TrackedBlock> tracked_blocks;
  tracked_blocks.reserve(participants.size());
  const int mid_column = cache.n_ref / 2;
  for (const std::size_t b : participants) {
    const auto& block = mb.blocks[b];
    const int mid_layer = block.n_i_cells / 2;
    bool aligned = true;
    for (int column = 0; column < cache.n_ref; ++column) {
      const int cell =
          block.cell_begin + mid_layer * cache.n_ref + column;
      const double theta = centroid_theta[static_cast<std::size_t>(cell)];
      const double delta =
          std::abs(theta - reference_theta[static_cast<std::size_t>(column)]);
      if (!std::isfinite(delta) || delta > alignment_tolerance) {
        aligned = false;
        break;
      }
    }
    if (!aligned) {
      core::log_warning("[refine_trk] block " + std::to_string(b) +
                        " angular columns misaligned; block dropped");
      continue;
    }

    const int first_cell = block.cell_begin + mid_column;
    const int last_cell = block.cell_begin +
                          (block.n_i_cells - 1) * cache.n_ref + mid_column;
    const double radial_delta =
        centroid_s[static_cast<std::size_t>(last_cell)] -
        centroid_s[static_cast<std::size_t>(first_cell)];
    double mean_s = 0.0;
    for (int layer = 0; layer < block.n_i_cells; ++layer) {
      const int cell =
          block.cell_begin + layer * cache.n_ref + mid_column;
      mean_s += centroid_s[static_cast<std::size_t>(cell)];
    }
    mean_s /= static_cast<double>(block.n_i_cells);
    tracked_blocks.push_back({b, radial_delta >= 0.0, mean_s});
  }

  std::stable_sort(
      tracked_blocks.begin(),
      tracked_blocks.end(),
      [](const TrackedBlock& lhs, const TrackedBlock& rhs) {
        return lhs.mean_mid_column_s < rhs.mean_mid_column_s;
      });

  cache.columns.assign(static_cast<std::size_t>(cache.n_ref), {});
  for (int column = 0; column < cache.n_ref; ++column) {
    auto& output = cache.columns[static_cast<std::size_t>(column)];
    for (const TrackedBlock& tracked : tracked_blocks) {
      const auto& block = mb.blocks[tracked.block_index];
      if (tracked.layers_ascending) {
        for (int layer = 0; layer < block.n_i_cells; ++layer) {
          output.push_back(
              block.cell_begin + layer * cache.n_ref + column);
        }
      } else {
        for (int layer = block.n_i_cells - 1; layer >= 0; --layer) {
          output.push_back(
              block.cell_begin + layer * cache.n_ref + column);
        }
      }
    }
  }
}

void build_column_table(TrackerCache& cache,
                        const TrackerLatticeView& view,
                        const mesh::MeshTopology& topo,
                        const std::size_t n_cells) {
  cache.table_built = true;
  cache.built_cell_count = n_cells;
  cache.n_ref = view.n_across;
  cache.columns.assign(static_cast<std::size_t>(view.n_across), {});
  cache.severity_path.clear();

  const int severity_across = view.n_across / 2;
  cache.severity_path.reserve(static_cast<std::size_t>(view.n_layers));
  for (int layer = 0; layer < view.n_layers; ++layer) {
    cache.severity_path.push_back(
        view.layer_axis_is_j ? severity_across * topo.nz + layer
                             : layer * topo.nz + severity_across);
  }
  for (int across = 0; across < view.n_across; ++across) {
    auto& output = cache.columns[static_cast<std::size_t>(across)];
    output.reserve(static_cast<std::size_t>(view.n_layers));
    for (int layer = 0; layer < view.n_layers; ++layer) {
      output.push_back(view.layer_axis_is_j ? across * topo.nz + layer
                                            : layer * topo.nz + across);
    }
  }
}

const char* tracker_state_name(const TrackerState state) {
  switch (state) {
    case TrackerState::SEARCH:
      return "SEARCH";
    case TrackerState::TRACKED:
      return "TRACKED";
    case TrackerState::ARMED:
      return "ARMED";
  }
  return "SEARCH";
}

double severity_a_old(
    const std::vector<int>& severity_path,
    const std::vector<double>& centroid_s,
    const double s50,
    const core::Config::NumericsConfig::DiagnosticsConfig::
        RefinementAutopilot& autopilot) {
  // One median-column profile suffices because these meshes are
  // near-spherical.
  double a_old = 0.0;
  for (std::size_t i = 1; i + 1U < severity_path.size(); ++i) {
    const int cell = severity_path[i];
    const double s = centroid_s[static_cast<std::size_t>(cell)];
    if (!std::isfinite(s) || s < autopilot.handoff_cm || s > s50 ||
        s <= 0.0) {
      continue;
    }
    const double spacing = std::abs(
        centroid_s[static_cast<std::size_t>(severity_path[i + 1U])] -
        centroid_s[static_cast<std::size_t>(severity_path[i - 1U])]);
    if (!std::isfinite(spacing) || spacing <= 0.0) {
      continue;
    }
    const double h_local = 0.5 * spacing;
    const double chi =
        autopilot.n_q_plan * h_local / (autopilot.chi_design * s);
    if (std::isfinite(chi)) {
      a_old = std::max(a_old, chi);
    }
  }
  return a_old;
}

double parent_spacing(
    const std::vector<int>& mid_column,
    const std::vector<double>& centroid_s,
    const double h_med,
    const core::Config::NumericsConfig::DiagnosticsConfig::
        RefinementAutopilot& autopilot) {
  std::vector<double> spacings;
  const double upper = autopilot.s_rep_cm + 8.0 * h_med;
  for (std::size_t i = 1; i + 1U < mid_column.size(); ++i) {
    const double s =
        centroid_s[static_cast<std::size_t>(mid_column[i])];
    if (!std::isfinite(s) || s < autopilot.s_rep_cm || s > upper) {
      continue;
    }
    const double spacing = std::abs(
        centroid_s[static_cast<std::size_t>(mid_column[i + 1U])] -
        centroid_s[static_cast<std::size_t>(mid_column[i - 1U])]);
    if (std::isfinite(spacing) && spacing > 0.0) {
      spacings.push_back(0.5 * spacing);
    }
  }
  return spacings.empty() ? h_med : sorted_median(std::move(spacings));
}

}  // namespace

const AutopilotFireRecord* refinement_tracker_pending_fire() {
  const TrackerCache& cache = tracker_cache();
  return cache.fire_record.pending ? &cache.fire_record : nullptr;
}

void refinement_tracker_clear_fire() {
  tracker_cache().fire_record = AutopilotFireRecord{};
}

bool refinement_tracker_front_shell_row(int* shell_row, double* coverage) {
  const TrackerFrontTestOverride& override = tracker_front_test_override();
  if (override.enabled) {
    *shell_row = override.shell_row;
    *coverage = override.coverage;
    return true;
  }
  const TrackerCache& cache = tracker_cache();
  if (!cache.front_shell_row_valid) {
    return false;
  }
  *shell_row = cache.front_shell_row;
  *coverage = cache.front_coverage;
  return true;
}

void refinement_tracker_set_front_for_test(const int shell_row,
                                           const double coverage) {
  TrackerFrontTestOverride& override = tracker_front_test_override();
  override.enabled = true;
  override.shell_row = shell_row;
  override.coverage = coverage;
}

void refinement_tracker_clear_front_for_test() {
  tracker_front_test_override() = TrackerFrontTestOverride{};
}

bool tracker_should_request_checkpoint(const double d_clear_h,
                                       const double window_hi_h,
                                       const double ckpt_lead_h) {
  return std::isfinite(d_clear_h) &&
         d_clear_h <= window_hi_h + ckpt_lead_h;
}

std::vector<std::pair<int, int>> tracker_components(
    const std::vector<double>& e,
    const double cut,
    const int gap_bridge) {
  std::vector<std::pair<int, int>> components;
  int begin = -1;
  int last_flagged = -1;
  for (int i = 0; i < static_cast<int>(e.size()); ++i) {
    if (e[static_cast<std::size_t>(i)] >= cut) {
      if (begin < 0) {
        begin = i;
      } else if (i - last_flagged - 1 > gap_bridge) {
        components.emplace_back(begin, last_flagged);
        begin = i;
      }
      last_flagged = i;
    }
  }
  if (begin >= 0) {
    components.emplace_back(begin, last_flagged);
  }
  return components;
}

bool tracker_fit_models(const std::vector<double>& t,
                        const std::vector<double>& s,
                        const double t_ref,
                        double linear_out[2],
                        double& linear_rms,
                        double quad_out[3],
                        double& quad_rms) {
  linear_out[0] = 0.0;
  linear_out[1] = 0.0;
  quad_out[0] = 0.0;
  quad_out[1] = 0.0;
  quad_out[2] = 0.0;
  linear_rms = 0.0;
  quad_rms = 0.0;
  if (t.size() != s.size() || t.size() < 3U || !std::isfinite(t_ref)) {
    return false;
  }

  double sx0 = 0.0;
  double sx1 = 0.0;
  double sx2 = 0.0;
  double sx3 = 0.0;
  double sx4 = 0.0;
  double sy0 = 0.0;
  double sy1 = 0.0;
  double sy2 = 0.0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    const double x = t[i] - t_ref;
    const double y = s[i];
    if (!std::isfinite(x) || !std::isfinite(y)) {
      return false;
    }
    const double x2 = x * x;
    sx0 += 1.0;
    sx1 += x;
    sx2 += x2;
    sx3 += x2 * x;
    sx4 += x2 * x2;
    sy0 += y;
    sy1 += x * y;
    sy2 += x2 * y;
  }

  const double linear_det = sx0 * sx2 - sx1 * sx1;
  const double quad_det =
      sx0 * (sx2 * sx4 - sx3 * sx3) -
      sx1 * (sx1 * sx4 - sx3 * sx2) +
      sx2 * (sx1 * sx3 - sx2 * sx2);
  if (!std::isfinite(linear_det) || linear_det == 0.0 ||
      !std::isfinite(quad_det) || quad_det == 0.0) {
    return false;
  }

  linear_out[0] = (sy0 * sx2 - sx1 * sy1) / linear_det;
  linear_out[1] = (sx0 * sy1 - sx1 * sy0) / linear_det;

  const double quad_det_0 =
      sy0 * (sx2 * sx4 - sx3 * sx3) -
      sx1 * (sy1 * sx4 - sx3 * sy2) +
      sx2 * (sy1 * sx3 - sx2 * sy2);
  const double quad_det_1 =
      sx0 * (sy1 * sx4 - sx3 * sy2) -
      sy0 * (sx1 * sx4 - sx3 * sx2) +
      sx2 * (sx1 * sy2 - sy1 * sx2);
  const double quad_det_2 =
      sx0 * (sx2 * sy2 - sy1 * sx3) -
      sx1 * (sx1 * sy2 - sy1 * sx2) +
      sy0 * (sx1 * sx3 - sx2 * sx2);
  quad_out[0] = quad_det_0 / quad_det;
  quad_out[1] = quad_det_1 / quad_det;
  quad_out[2] = quad_det_2 / quad_det;

  double linear_residual_sum = 0.0;
  double quad_residual_sum = 0.0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    const double x = t[i] - t_ref;
    const double linear_residual =
        s[i] - (linear_out[0] + linear_out[1] * x);
    const double quad_residual =
        s[i] - (quad_out[0] + quad_out[1] * x + quad_out[2] * x * x);
    linear_residual_sum += linear_residual * linear_residual;
    quad_residual_sum += quad_residual * quad_residual;
  }
  linear_rms =
      std::sqrt(linear_residual_sum / static_cast<double>(t.size()));
  quad_rms =
      std::sqrt(quad_residual_sum / static_cast<double>(t.size()));
  return std::isfinite(linear_out[0]) && std::isfinite(linear_out[1]) &&
         std::isfinite(quad_out[0]) && std::isfinite(quad_out[1]) &&
         std::isfinite(quad_out[2]) && std::isfinite(linear_rms) &&
         std::isfinite(quad_rms);
}

bool tracker_earliest_arrival(const double linear_coeffs[2],
                              const double linear_rms,
                              const double quad_coeffs[3],
                              const double quad_rms,
                              const double s_target,
                              double& dt_arrival_early) {
  dt_arrival_early = 0.0;
  double earliest = std::numeric_limits<double>::infinity();

  const double linear_speed = linear_coeffs[1];
  if (std::isfinite(linear_coeffs[0]) && std::isfinite(linear_speed) &&
      std::isfinite(linear_rms) && std::isfinite(s_target) &&
      linear_rms >= 0.0 && linear_speed < 0.0) {
    const double arrival =
        (s_target - linear_coeffs[0]) / linear_speed;
    if (std::isfinite(arrival) && arrival >= 0.0) {
      const double sigma_t = linear_rms /
                             std::max(std::abs(linear_speed),
                                      kArrivalSpeedFloorCmPerS);
      earliest = std::min(earliest, arrival - 3.0 * sigma_t);
    }
  }

  const double b0 = quad_coeffs[0];
  const double b1 = quad_coeffs[1];
  const double b2 = quad_coeffs[2];
  if (std::isfinite(b0) && std::isfinite(b1) && std::isfinite(b2) &&
      std::isfinite(quad_rms) && std::isfinite(s_target) &&
      quad_rms >= 0.0 && b1 < 0.0) {
    std::vector<double> roots;
    if (b2 == 0.0) {
      roots.push_back((s_target - b0) / b1);
    } else {
      const double discriminant =
          b1 * b1 - 4.0 * b2 * (b0 - s_target);
      if (std::isfinite(discriminant) && discriminant >= 0.0) {
        const double root_term = std::sqrt(discriminant);
        roots.push_back((-b1 - root_term) / (2.0 * b2));
        roots.push_back((-b1 + root_term) / (2.0 * b2));
      }
    }
    double quad_arrival = std::numeric_limits<double>::infinity();
    for (const double root : roots) {
      const double derivative = b1 + 2.0 * b2 * root;
      if (std::isfinite(root) && root >= 0.0 &&
          std::isfinite(derivative) && derivative < 0.0) {
        quad_arrival = std::min(quad_arrival, root);
      }
    }
    if (std::isfinite(quad_arrival)) {
      const double sigma_t =
          quad_rms /
          std::max(std::abs(b1), kArrivalSpeedFloorCmPerS);
      earliest = std::min(earliest, quad_arrival - 3.0 * sigma_t);
    }
  }

  if (!std::isfinite(earliest)) {
    return false;
  }
  dt_arrival_early = earliest;
  return true;
}

void refinement_tracker_step(core::State& state, const core::Config& cfg) {
  const auto& estimator = cfg.numerics.diagnostics.refinement_estimator;
  const auto& autopilot = cfg.numerics.diagnostics.refinement_autopilot;
  const std::size_t n_cells = state.rho.size();
  if (!estimator.enabled || !autopilot.enabled ||
      state.step % estimator.every != 0 ||
      state.refine_error.size() != n_cells) {
    return;
  }
  TrackerLatticeView lattice;
  if (!make_tracker_lattice_view(state, cfg, lattice)) {
    return;
  }

  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  std::vector<double> centroid_s(n_cells, 0.0);
  std::vector<double> centroid_theta(n_cells, 0.0);
  if (lattice.single_block) {
    for (std::size_t c = 0; c < n_cells; ++c) {
      double centroid_r = 0.0;
      double centroid_z = 0.0;
      structured_cell_centroid(state.mesh.topo,
                               static_cast<int>(c),
                               node_r,
                               node_z,
                               centroid_r,
                               centroid_z);
      centroid_s[c] = lattice.layer_axis_is_j ? centroid_z : centroid_r;
    }
  } else {
    const auto& mb = *state.mesh.topo.multiblock;
    for (std::size_t c = 0; c < n_cells; ++c) {
      double centroid_r = 0.0;
      double centroid_z = 0.0;
      cell_centroid(mb,
                    static_cast<int>(c),
                    node_r,
                    node_z,
                    centroid_r,
                    centroid_z);
      centroid_s[c] = std::hypot(centroid_r, centroid_z);
      centroid_theta[c] = std::atan2(centroid_r, centroid_z);
    }
  }

  TrackerCache& cache = tracker_cache();
  const bool lattice_changed =
      cache.table_built &&
      (cache.built_single_block != lattice.single_block ||
       cache.built_layer_axis_is_j != lattice.layer_axis_is_j ||
       cache.built_n_layers != lattice.n_layers ||
       cache.built_n_across != lattice.n_across);
  if (!cache.table_built || cache.built_cell_count != n_cells ||
      lattice_changed) {
    if (lattice.single_block) {
      build_column_table(cache, lattice, state.mesh.topo, n_cells);
    } else {
      const auto& mb = *state.mesh.topo.multiblock;
      build_column_table(
          cache, mb, n_cells, centroid_s, centroid_theta);
    }
    cache.built_single_block = lattice.single_block;
    cache.built_layer_axis_is_j = lattice.layer_axis_is_j;
    cache.built_n_layers = lattice.n_layers;
    cache.built_n_across = lattice.n_across;
  }

  const std::vector<double> pe = copy_field(state.Pe);
  const std::vector<double> pi = copy_field(state.Pi);
  std::vector<double> pressure(n_cells, 0.0);
  for (std::size_t c = 0; c < n_cells; ++c) {
    pressure[c] = pe[c] + pi[c];
  }

  std::vector<double> front_radii;
  std::vector<double> front_spacings;
  std::vector<double> component_peaks;
  std::vector<int> front_shell_rows;
  front_radii.reserve(cache.columns.size());
  front_spacings.reserve(cache.columns.size());
  component_peaks.reserve(cache.columns.size());
  front_shell_rows.reserve(cache.columns.size());
  for (const auto& column_cells : cache.columns) {
    std::vector<double> radial_s;
    std::vector<double> radial_pressure;
    std::vector<double> radial_error;
    radial_s.reserve(column_cells.size());
    radial_pressure.reserve(column_cells.size());
    radial_error.reserve(column_cells.size());
    for (const int cell : column_cells) {
      radial_s.push_back(centroid_s[static_cast<std::size_t>(cell)]);
      radial_pressure.push_back(pressure[static_cast<std::size_t>(cell)]);
      radial_error.push_back(
          state.refine_error[static_cast<std::size_t>(cell)]);
    }

    const auto components = tracker_components(
        radial_error, autopilot.assoc_cut, autopilot.gap_bridge);
    int component_begin = -1;
    int component_end = -1;
    double component_peak = 0.0;
    for (const auto& component : components) {
      double peak = 0.0;
      for (int i = component.first; i <= component.second; ++i) {
        const double e = radial_error[static_cast<std::size_t>(i)];
        if (std::isfinite(e)) {
          peak = std::max(peak, e);
        }
      }
      if (peak >= autopilot.strong_cut) {
        component_begin = component.first;
        component_end = component.second;
        component_peak = peak;
        break;
      }
    }
    if (component_begin < 0 || radial_s.size() < 3U) {
      continue;
    }

    const int search_begin = std::max(1, component_begin - 2);
    const int search_end = std::min(
        static_cast<int>(radial_s.size()) - 2, component_end + 2);
    int front_index = -1;
    double largest_gradient = -1.0;
    double front_spacing = 0.0;
    for (int i = search_begin; i <= search_end; ++i) {
      const double denominator = std::abs(
          radial_s[static_cast<std::size_t>(i + 1)] -
          radial_s[static_cast<std::size_t>(i - 1)]);
      const double numerator = std::abs(
          radial_pressure[static_cast<std::size_t>(i + 1)] -
          radial_pressure[static_cast<std::size_t>(i - 1)]);
      if (!std::isfinite(denominator) || denominator <= 0.0 ||
          !std::isfinite(numerator)) {
        continue;
      }
      const double gradient = numerator / denominator;
      if (std::isfinite(gradient) && gradient > largest_gradient) {
        largest_gradient = gradient;
        front_index = i;
        front_spacing = 0.5 * denominator;
      }
    }
    if (front_index >= 0 &&
        std::isfinite(radial_s[static_cast<std::size_t>(front_index)])) {
      front_radii.push_back(
          radial_s[static_cast<std::size_t>(front_index)]);
      front_spacings.push_back(front_spacing);
      component_peaks.push_back(component_peak);
      const int front_cell =
          column_cells[static_cast<std::size_t>(front_index)];
      if (lattice.single_block) {
        front_shell_rows.push_back(
            lattice.layer_axis_is_j
                ? front_cell % state.mesh.topo.nz
                : front_cell / state.mesh.topo.nz);
      } else {
        const auto& mb = *state.mesh.topo.multiblock;
        const int block_id =
            mb.cell_block_id[static_cast<std::size_t>(front_cell)];
        if (block_id >= 0 &&
            block_id < static_cast<int>(mb.blocks.size())) {
          const auto& front_block =
              mb.blocks[static_cast<std::size_t>(block_id)];
          if (front_block.role == mesh::BlockRole::POLAR_SHELL &&
              front_block.n_j_cells > 0 &&
              front_cell >= front_block.cell_begin &&
              front_cell < front_block.cell_begin + front_block.cell_count) {
            front_shell_rows.push_back(
                (front_cell - front_block.cell_begin) /
                front_block.n_j_cells);
          }
        }
      }
    }
  }

  const std::size_t n_tracked = front_radii.size();
  const double coverage =
      cache.columns.empty()
          ? 0.0
          : static_cast<double>(n_tracked) /
                static_cast<double>(cache.columns.size());
  double s50 = 0.0;
  double s_lead = 0.0;
  double e_peak_med = 0.0;
  double h_med = 0.0;
  if (n_tracked > 0U) {
    std::sort(front_radii.begin(), front_radii.end());
    s50 = median_sorted(front_radii);
    const std::size_t lead_index = std::min(
        static_cast<std::size_t>(0.01 * static_cast<double>(n_tracked)),
        n_tracked - 1U);
    s_lead = front_radii[lead_index];
    e_peak_med = sorted_median(std::move(component_peaks));
    h_med = sorted_median(std::move(front_spacings));
  }

  std::ostringstream tracker_line;
  tracker_line << "[refine_trk] step=" << state.step << " t="
               << std::scientific << std::setprecision(17) << state.t
               << " cov=" << std::fixed << std::setprecision(4) << coverage
               << " s50_um=" << 1.0e4 * s50
               << " slead_um=" << 1.0e4 * s_lead
               << " epeak_med=" << e_peak_med
               << " h_med_um=" << 1.0e4 * h_med;
  core::log_info(tracker_line.str());

  while (cache.observation_t.size() >
         static_cast<std::size_t>(autopilot.history)) {
    cache.observation_t.pop_front();
    cache.observation_s.pop_front();
  }
  if (coverage >= autopilot.cov_min) {
    cache.observation_t.push_back(state.t);
    cache.observation_s.push_back(s50);
    while (cache.observation_t.size() >
           static_cast<std::size_t>(autopilot.history)) {
      cache.observation_t.pop_front();
      cache.observation_s.pop_front();
    }
  }

  const TrackerState state_at_start = cache.state;
  if (cache.state == TrackerState::SEARCH) {
    if (e_peak_med >= autopilot.e_on && coverage >= autopilot.cov_min) {
      ++cache.search_promote_count;
    } else {
      cache.search_promote_count = 0;
    }
    if (cache.search_promote_count >= autopilot.persist) {
      cache.state = TrackerState::TRACKED;
      cache.search_promote_count = 0;
      cache.tracked_demote_count = 0;
      cache.arm_promote_count = 0;
      cache.armed_demote_count = 0;
      cache.observation_t.clear();
      cache.observation_s.clear();
    }
  }

  const bool have_mid_column =
      cache.n_ref > 0 &&
      static_cast<std::size_t>(cache.n_ref / 2) < cache.columns.size();
  const auto* mid_column =
      have_mid_column
          ? &cache.columns[static_cast<std::size_t>(cache.n_ref / 2)]
          : nullptr;
  const bool a_old_valid =
      cache.state != TrackerState::SEARCH &&
      cache.observation_t.size() >= 3U && !cache.severity_path.empty();
  const double a_old =
      a_old_valid
          ? severity_a_old(cache.severity_path, centroid_s, s50, autopilot)
          : 0.0;

  if (cache.state == TrackerState::TRACKED) {
    if (e_peak_med <= autopilot.e_off) {
      ++cache.tracked_demote_count;
    } else {
      cache.tracked_demote_count = 0;
    }
    if (cache.tracked_demote_count >= autopilot.persist) {
      cache.state = TrackerState::SEARCH;
      cache.search_promote_count = 0;
      cache.tracked_demote_count = 0;
      cache.arm_promote_count = 0;
      cache.armed_demote_count = 0;
    } else if (a_old_valid) {
      if (a_old >= 1.10) {
        ++cache.arm_promote_count;
      } else {
        cache.arm_promote_count = 0;
      }
      if (cache.arm_promote_count >= autopilot.persist) {
        cache.state = TrackerState::ARMED;
        cache.arm_promote_count = 0;
        cache.armed_demote_count = 0;
      }
    } else {
      cache.arm_promote_count = 0;
    }
  } else if (cache.state == TrackerState::ARMED) {
    if (a_old_valid) {
      if (a_old <= 0.90) {
        ++cache.armed_demote_count;
      } else {
        cache.armed_demote_count = 0;
      }
      if (cache.armed_demote_count >= autopilot.persist) {
        cache.state = TrackerState::TRACKED;
        cache.armed_demote_count = 0;
        cache.arm_promote_count = 0;
      }
    } else {
      cache.armed_demote_count = 0;
    }
  }

  double d_clear_h = 0.0;
  double dt_arrival_early = 0.0;
  double q99_e = 0.0;
  double max_e = 0.0;
  std::string decision = "HOLD";
  if (cache.state == TrackerState::ARMED) {
    std::vector<double> observation_t(cache.observation_t.begin(),
                                      cache.observation_t.end());
    std::vector<double> observation_s(cache.observation_s.begin(),
                                      cache.observation_s.end());
    double linear_coeffs[2] = {};
    double quad_coeffs[3] = {};
    double linear_rms = 0.0;
    double quad_rms = 0.0;
    bool prediction_ok = false;
    if (!observation_t.empty() &&
        tracker_fit_models(observation_t,
                           observation_s,
                           observation_t.back(),
                           linear_coeffs,
                           linear_rms,
                           quad_coeffs,
                           quad_rms)) {
      prediction_ok = tracker_earliest_arrival(linear_coeffs,
                                               linear_rms,
                                               quad_coeffs,
                                               quad_rms,
                                               autopilot.s_rep_cm,
                                               dt_arrival_early);
    }

    const double h_parent =
        mid_column != nullptr
            ? parent_spacing(*mid_column, centroid_s, h_med, autopilot)
            : h_med;
    if (std::isfinite(h_parent) && h_parent > 0.0) {
      d_clear_h = (s_lead - autopilot.s_rep_cm) / h_parent;
    }
    if (autopilot.mode == "arm_exit" &&
        tracker_should_request_checkpoint(d_clear_h,
                                          autopilot.window_hi_h,
                                          autopilot.ckpt_lead_h)) {
      state.checkpoint_request = true;
    }

    std::vector<double> quiescent_error;
    const double quiescent_limit = autopilot.s_rep_cm + 2.0 * h_parent;
    for (std::size_t c = 0; c < n_cells; ++c) {
      if (centroid_s[c] <= quiescent_limit) {
        const double e = std::isfinite(state.refine_error[c])
                             ? state.refine_error[c]
                             : 1.0;
        quiescent_error.push_back(e);
        max_e = std::max(max_e, e);
      }
    }
    if (!quiescent_error.empty()) {
      std::sort(quiescent_error.begin(), quiescent_error.end());
      const std::size_t q99_index = std::min(
          static_cast<std::size_t>(
              0.99 * static_cast<double>(quiescent_error.size())),
          quiescent_error.size() - 1U);
      q99_e = quiescent_error[q99_index];
    }

    const bool window_ok =
        std::isfinite(d_clear_h) && d_clear_h >= autopilot.window_lo_h &&
        d_clear_h <= autopilot.window_hi_h;
    const bool arrival_ok = prediction_ok && dt_arrival_early > 0.0;
    const bool quiescence_ok = q99_e < 0.10 && max_e < 0.20;
    if (!window_ok) {
      decision = "BLOCKED_WINDOW";
    } else if (!arrival_ok) {
      decision = "BLOCKED_PRED";
    } else if (!quiescence_ok) {
      decision = "BLOCKED_QUIES";
    } else {
      decision = "WOULD_FIRE";
      ++cache.would_fire_count;
      if (autopilot.mode == "arm_exit") {
        cache.fire_record.pending = true;
        cache.fire_record.step = static_cast<long long>(state.step);
        cache.fire_record.t = state.t;
        cache.fire_record.s50_cm = s50;
        cache.fire_record.s_lead_cm = s_lead;
        cache.fire_record.d_clear_h = d_clear_h;
        cache.fire_record.a_old = a_old;
        cache.fire_record.dt_arrival_early_s = dt_arrival_early;
        cache.fire_record.q99_e = q99_e;
      }
    }
  }

  const bool state_changed = cache.state != state_at_start;
  if (cache.state != TrackerState::SEARCH || state_changed) {
    std::ostringstream autopilot_line;
    // Single-block decision fields have lattice-coordinate semantics; only
    // the tracked front location consumed by the §18.5 front hold is
    // contractual.
    autopilot_line << "[refine_apilot] step=" << state.step << " t="
                   << std::scientific << std::setprecision(17) << state.t
                   << " state=" << tracker_state_name(cache.state)
                   << " epeak=" << std::fixed << std::setprecision(4)
                   << e_peak_med << " A_old=" << a_old
                   << " dclear_h=" << d_clear_h
                   << " tarr_early_ps=" << 1.0e12 * dt_arrival_early
                   << " q99E=" << std::scientific << std::setprecision(3)
                   << q99_e << " decision=" << decision;
    core::log_info(autopilot_line.str());
  }

  std::sort(front_shell_rows.begin(), front_shell_rows.end());
  cache.front_shell_row = front_shell_rows.empty()
                              ? -1
                              : front_shell_rows[front_shell_rows.size() / 2U];
  cache.front_coverage = coverage;
  cache.front_shell_row_valid =
      cache.state != TrackerState::SEARCH && !front_shell_rows.empty() &&
      2U * front_shell_rows.size() >= n_tracked;
}

}  // namespace tenryu::hydro
