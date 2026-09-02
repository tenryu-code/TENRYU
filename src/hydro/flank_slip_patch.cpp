#include "hydro/flank_slip_patch.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/flank_tangential_strip_detail.hpp"
#include "hydro/intersection_remap_patch.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"

namespace tenryu::hydro {
namespace {

using PackedQuad = std::array<double, 8>;

std::array<int, 4> cell_nodes(const core::State& state, const int cell) {
  const int nz = state.mesh.topo.nz;
  const int i = cell / nz;
  const int j = cell % nz;
  return {
      state.mesh.topo.node_index(i, j),
      state.mesh.topo.node_index(i + 1, j),
      state.mesh.topo.node_index(i + 1, j + 1),
      state.mesh.topo.node_index(i, j + 1),
  };
}

PackedQuad pack_cell_quad(const core::State& state,
                          const int cell,
                          const std::vector<double>& x_r,
                          const std::vector<double>& x_z) {
  PackedQuad quad{};
  const std::array<int, 4> nodes = cell_nodes(state, cell);
  for (int corner = 0; corner < 4; ++corner) {
    const std::size_t node =
        static_cast<std::size_t>(nodes[static_cast<std::size_t>(corner)]);
    quad[static_cast<std::size_t>(corner)] = x_r[node];
    quad[static_cast<std::size_t>(corner + 4)] = x_z[node];
  }
  return quad;
}

void unpack_quad(const PackedQuad& quad, double r[4], double z[4]) {
  for (int corner = 0; corner < 4; ++corner) {
    r[corner] = quad[static_cast<std::size_t>(corner)];
    z[corner] = quad[static_cast<std::size_t>(corner + 4)];
  }
}

double planar_quad_area(const PackedQuad& quad) {
  double twice_area = 0.0;
  for (int corner = 0; corner < 4; ++corner) {
    const int next = (corner + 1) & 3;
    twice_area += quad[static_cast<std::size_t>(corner)] *
                      quad[static_cast<std::size_t>(next + 4)] -
                  quad[static_cast<std::size_t>(next)] *
                      quad[static_cast<std::size_t>(corner + 4)];
  }
  return 0.5 * std::abs(twice_area);
}

double relative_defect(const double before,
                       const double after,
                       const double scale) {
  const double denominator = std::max(
      {std::abs(before), std::abs(after), scale,
       std::numeric_limits<double>::min()});
  return std::abs(after - before) / denominator;
}

bool valid_reference_band(const core::State& state,
                          const detail::FlankStripWindowPolylines& window) {
  const int nz = state.mesh.topo.nz;
  const auto& q_reference =
      state.evacuated_cells.flank_strip_q_theta_ref;
  if (q_reference.size() !=
      4U * static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return false;
  }
  for (int i = 1; i <= window.i_last; ++i) {
    for (int j = window.j_first; j <= window.j_last; ++j) {
      const int cell = state.mesh.topo.cell_index(i, j);
      if (detail::flank_strip_cell_is_geometry_policy_exempt(state, cell)) {
        continue;
      }
      const std::size_t base = 4U * static_cast<std::size_t>(cell);
      for (int corner = 0; corner < 4; ++corner) {
        const double q_ref =
            q_reference[base + static_cast<std::size_t>(corner)];
        if (!(q_ref > 0.0) || !std::isfinite(q_ref)) {
          return false;
        }
      }
    }
  }
  return nz > 0;
}

bool patch_is_single_material(const core::State& state,
                              const core::Config& cfg,
                              const std::vector<int>& patch_cells) {
  if (state.volFrac.empty()) {
    return true;
  }
  const std::size_t n_cells =
      static_cast<std::size_t>(state.mesh.topo.n_cells);
  if (n_cells == 0U || state.volFrac.size() % n_cells != 0U) {
    return false;
  }
  const std::size_t n_materials = state.volFrac.size() / n_cells;
  if (n_materials == 0U ||
      (!cfg.materials.materials.empty() &&
       n_materials != cfg.materials.materials.size())) {
    return false;
  }
  std::vector<double> volume_fraction;
  state.volFrac.copy_to_host(volume_fraction);
  int patch_material = -1;
  for (const int cell : patch_cells) {
    int present_count = 0;
    int present_material = -1;
    for (std::size_t material = 0; material < n_materials; ++material) {
      const double fraction =
          volume_fraction[static_cast<std::size_t>(cell) * n_materials +
                          material];
      if (!std::isfinite(fraction) || fraction < 0.0) {
        return false;
      }
      if (fraction > 0.0) {
        ++present_count;
        present_material = static_cast<int>(material);
      }
    }
    if (present_count > 1) {
      return false;
    }
    if (present_count == 1) {
      if (patch_material < 0) {
        patch_material = present_material;
      } else if (patch_material != present_material) {
        return false;
      }
    }
  }
  return true;
}

const char* candidate_cell_reject_check(const PackedQuad& quad,
                                        const int cell_i) {
  double r[4];
  double z[4];
  unpack_quad(quad, r, z);
  for (int corner = 0; corner < 4; ++corner) {
    if (!(corner_jacobian_from_quad(r, z, corner) > 0.0) ||
        !std::isfinite(corner_jacobian_from_quad(r, z, corner))) {
      return "corner_j";
    }
  }
  const bool physical_axis = cell_i == 0 && r[0] == 0.0 && r[3] == 0.0;
  const int inner_i = physical_axis ? 0 : std::max(cell_i, 1);
  const int node_i[4] = {inner_i, cell_i + 1, cell_i + 1, inner_i};
  const double j_floor_rel =
      mesh::CandidateMeshAdmissibilityFloors{}.gauss_j_rel;
  if (!trial_cell_admissible(r, z, node_i, j_floor_rel)) {
    const double xis[2] = {-kGauss, kGauss};
    const double etas[2] = {-kGauss, kGauss};
    double gauss_j[4];
    double gauss_j_max = -1.0e300;
    int gauss = 0;
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        const double value =
            gauss_jacobian_from_quad(r, z, xis[a], etas[b]);
        if (!std::isfinite(value)) {
          return "gauss_j";
        }
        gauss_j[gauss++] = value;
        gauss_j_max = std::fmax(gauss_j_max, value);
      }
    }
    const double j_floor =
        std::fmax(j_floor_rel * std::fmax(gauss_j_max, kJFloor), kJFloor);
    for (const double value : gauss_j) {
      if (!(value > j_floor)) {
        return "gauss_j";
      }
    }
    for (int corner = 0; corner < 4; ++corner) {
      const double value = corner_jacobian_from_quad(r, z, corner);
      if (!std::isfinite(value) || !(value > j_floor)) {
        return "corner_j";
      }
    }
    const double rz_volume = ale::detail::rz_signed_quad_volume(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3]);
    if (!std::isfinite(rz_volume) || !(rz_volume > kJFloor)) {
      return "volume";
    }
    for (int corner = 0; corner < 4; ++corner) {
      if (!std::isfinite(r[corner]) || !std::isfinite(z[corner]) ||
          (node_i[corner] == 0 ? r[corner] != 0.0 : r[corner] < 0.0)) {
        return "volume";
      }
    }
    return "gauss_j";
  }
  const ixremap::Polygon polygon{{r[0], r[1], r[2], r[3]},
                                 {z[0], z[1], z[2], z[3]}};
  return ixremap::polygon_rz_volume(polygon) > 0.0 ? "" : "volume";
}

detail::FlankStripRowPolyline build_slip_row_zero_polyline(
    const core::State& state,
    const detail::FlankStripWindowPolylines& window,
    const std::vector<double>& original_r,
    const std::vector<double>& original_z,
    const std::vector<std::uint8_t>& fixed_node) {
  detail::FlankStripRowPolyline row;
  row.i = 0;
  row.nodes.reserve(static_cast<std::size_t>(window.window_size));
  row.r.reserve(static_cast<std::size_t>(window.window_size));
  row.z.reserve(static_cast<std::size_t>(window.window_size));
  row.s.assign(static_cast<std::size_t>(window.window_size), 0.0);
  row.fixed.assign(static_cast<std::size_t>(window.window_size), 0U);
  for (int local_j = 0; local_j < window.window_size; ++local_j) {
    const int node =
        state.mesh.topo.node_index(0, window.j_first + local_j);
    row.nodes.push_back(node);
    row.r.push_back(original_r[static_cast<std::size_t>(node)]);
    row.z.push_back(original_z[static_cast<std::size_t>(node)]);
    row.fixed[static_cast<std::size_t>(local_j)] =
        fixed_node[static_cast<std::size_t>(node)];
    if (local_j > 0) {
      row.s[static_cast<std::size_t>(local_j)] =
          row.s[static_cast<std::size_t>(local_j - 1)] +
          std::hypot(
              row.r[static_cast<std::size_t>(local_j)] -
                  row.r[static_cast<std::size_t>(local_j - 1)],
              row.z[static_cast<std::size_t>(local_j)] -
                  row.z[static_cast<std::size_t>(local_j - 1)]);
    }
  }
  if (!row.fixed.empty()) {
    row.fixed.front() = 1U;
    row.fixed.back() = 1U;
  }
  detail::build_flank_strip_row_targets(state, row);
  return row;
}

void clamp_slip_row_deltas_no_fold(
    const detail::FlankStripRowPolyline& row,
    std::vector<double>& delta) {
  for (std::size_t j = 0; j + 1U < delta.size(); ++j) {
    const std::size_t next = j + 1U;
    if (row.fixed[next] != 0U) {
      continue;
    }
    const double half_spacing = 0.5 * (row.s[next] - row.s[j]);
    delta[next] = std::max(delta[next], delta[j] - half_spacing);
  }
  for (std::size_t next = delta.size(); next > 1U; --next) {
    const std::size_t j = next - 2U;
    const std::size_t right = j + 1U;
    if (row.fixed[j] != 0U) {
      continue;
    }
    const double half_spacing = 0.5 * (row.s[right] - row.s[j]);
    delta[j] = std::min(delta[j], delta[right] + half_spacing);
  }
}

void transfer_extensive(const ixremap::PatchIntersection& intersection,
                        const std::vector<double>& source,
                        std::vector<double>& destination,
                        std::vector<std::uint8_t>* orphaned = nullptr) {
  ixremap::transfer_renormalized(
      intersection, source, destination, orphaned);
}

}  // namespace

FlankSlipPatchResult flank_slip_patch_execute(
    core::State& state,
    const core::Config& cfg,
    const int target_cell,
    const int direction) {
  FlankSlipPatchResult result;
  result.attempted = true;
  result.direction = direction;
  const auto reject = [&](const char* const reason) {
    result.reject_reason = reason;
    return result;
  };

  const auto& strip_cfg = cfg.numerics.ale.evacuated_cell.closure_contact
                              .flank_tangential_strip;
  if (!strip_cfg.slip_patch_enabled) {
    return reject("config_disabled");
  }
  if (state.mesh.dim != 2) {
    return reject("not_2d");
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (target_cell < 0 || target_cell >= n_cells || nz <= 0) {
    return reject("invalid_target_cell");
  }
  const int target_i = target_cell / nz;
  if (target_i < 1 ||
      target_i > std::min(nr - 1, strip_cfg.band_layers)) {
    return reject("not_flank_row");
  }
  if (direction == 0) {
    return reject("zero_direction");
  }
  if (nr <= 1 || n_nodes <= 0) {
    return reject("invalid_window");
  }
  if (state.corner_stride != 4) {
    return reject("field_size_mismatch");
  }

  const int slot_cell = detail::active_flank_strip_slot_cell(state);
  if (slot_cell < 0) {
    return reject("invalid_window");
  }
  const int focus_j = target_cell % nz;
  detail::flank_strip_extend_reference_for_window(
      state,
      cfg,
      slot_cell,
      strip_cfg.band_layers,
      strip_cfg.band_halfwidth_j,
      focus_j);

  const std::size_t node_count = static_cast<std::size_t>(n_nodes);
  const std::size_t cell_count = static_cast<std::size_t>(n_cells);
  std::vector<double> original_r;
  std::vector<double> original_z;
  state.x_r.copy_to_host(original_r);
  state.x_z.copy_to_host(original_z);
  if (original_r.size() != node_count || original_z.size() != node_count) {
    return reject("field_size_mismatch");
  }
  const detail::FlankStripWindowPolylines window =
      detail::build_flank_strip_window_polylines(
          state, cfg, slot_cell, original_r, original_z, focus_j);
  if (!window.valid || window.rows.empty()) {
    return reject("invalid_window");
  }
  if (!valid_reference_band(state, window)) {
    return reject("invalid_reference");
  }
  result.slip_before = detail::flank_strip_slip_mismatch_signed_ratio(
      state,
      original_r,
      original_z,
      window.j_first,
      window.j_last);

  int q_argmin = -1;
  result.q_before = detail::flank_strip_band_minimum_quality_ratio(
      state,
      original_r,
      original_z,
      slot_cell,
      strip_cfg.band_layers,
      strip_cfg.band_halfwidth_j,
      q_argmin,
      focus_j);
  if (!std::isfinite(result.q_before)) {
    return reject("invalid_reference");
  }

  std::vector<std::uint8_t> row_zero_fixed_node = window.fixed_node;
  const int node_stride = nz + 1;
  const auto fix_row_zero_node = [&](const int node) {
    if (node >= 0 && node < n_nodes && node / node_stride == 0) {
      row_zero_fixed_node[static_cast<std::size_t>(node)] = 1U;
    }
  };
  const auto& node_axis_alias = state.evacuated_cells.node_axis_alias;
  for (std::size_t source = 0;
       source < node_axis_alias.size() && source < node_count;
       ++source) {
    const int target = node_axis_alias[source];
    if (target < 0) {
      continue;
    }
    fix_row_zero_node(static_cast<int>(source));
    fix_row_zero_node(target);
  }
  for (int cell = 0; cell < n_cells; ++cell) {
    if (!detail::flank_strip_cell_is_geometry_policy_exempt(state, cell)) {
      continue;
    }
    for (const int node : cell_nodes(state, cell)) {
      fix_row_zero_node(node);
    }
  }
  fix_row_zero_node(state.mesh.topo.node_index(0, window.j_first));
  fix_row_zero_node(state.mesh.topo.node_index(0, window.j_last));
  const detail::FlankStripRowPolyline row_zero =
      build_slip_row_zero_polyline(state,
                                   window,
                                   original_r,
                                   original_z,
                                   row_zero_fixed_node);
  if (!row_zero.targets_valid) {
    return reject("invalid_reference");
  }
  std::vector<double> candidate_r = original_r;
  std::vector<double> candidate_z = original_z;
  std::vector<std::uint8_t> moved_node(node_count, 0U);
  bool candidate_constructed = true;
  const auto shift_row = [&](const detail::FlankStripRowPolyline& row) {
    std::vector<double> delta(
        static_cast<std::size_t>(window.window_size), 0.0);
    for (int local_j = 1; local_j < window.window_size - 1; ++local_j) {
      const std::size_t local = static_cast<std::size_t>(local_j);
      if (row.fixed[local] != 0U) {
        continue;
      }
      const int distance_to_endpoint =
          std::min(local_j, window.window_size - 1 - local_j);
      const double taper =
          std::min(1.0, 0.5 * static_cast<double>(distance_to_endpoint));
      delta[local] = (row.target_s[local] - row.s[local]) * taper;
    }
    clamp_slip_row_deltas_no_fold(row, delta);
    for (int local_j = 1; local_j < window.window_size - 1; ++local_j) {
      const std::size_t local = static_cast<std::size_t>(local_j);
      if (row.fixed[local] != 0U) {
        continue;
      }
      const double candidate_s = row.s[local] + delta[local];
      double interpolated_r = 0.0;
      double interpolated_z = 0.0;
      if (!detail::interpolate_flank_strip_polyline(
              row, candidate_s, interpolated_r, interpolated_z)) {
        candidate_constructed = false;
        return;
      }
      const std::size_t node =
          static_cast<std::size_t>(row.nodes[local]);
      candidate_r[node] = interpolated_r;
      candidate_z[node] = interpolated_z;
      if (candidate_r[node] != original_r[node] ||
          candidate_z[node] != original_z[node]) {
        moved_node[node] = 1U;
      }
    }
  };
  shift_row(row_zero);
  for (const detail::FlankStripRowPolyline& row : window.rows) {
    if (!candidate_constructed) {
      break;
    }
    shift_row(row);
  }
  if (!candidate_constructed) {
    return reject("candidate_inadmissible");
  }
  if (std::none_of(moved_node.begin(), moved_node.end(),
                   [](const std::uint8_t moved) { return moved != 0U; })) {
    result.q_after = result.q_before;
    result.slip_after = result.slip_before;
    result.reject_check = "slip_reduction";
    return reject("candidate_inadmissible");
  }

  std::vector<std::uint8_t> untangler_fixed_node = window.fixed_node;
  for (std::size_t node = 0; node < node_count; ++node) {
    if (row_zero_fixed_node[node] != 0U) {
      untangler_fixed_node[node] = 1U;
    }
  }
  // Preserve the shifted row-0 nodes on their source polyline: the untangler
  // otherwise treats them as unconstrained in both r and z.
  for (const int node : row_zero.nodes) {
    untangler_fixed_node[static_cast<std::size_t>(node)] = 1U;
  }
  for (int j = window.j_first; j <= window.j_last; ++j) {
    const int seam_node = state.mesh.topo.node_index(1, j);
    untangler_fixed_node[static_cast<std::size_t>(seam_node)] = 1U;
  }
  for (int i = 1; i <= window.i_last; ++i) {
    const int first_node = state.mesh.topo.node_index(i, window.j_first);
    const int last_node = state.mesh.topo.node_index(i, window.j_last);
    untangler_fixed_node[static_cast<std::size_t>(first_node)] = 1U;
    untangler_fixed_node[static_cast<std::size_t>(last_node)] = 1U;
  }
  for (int cell = 0; cell < n_cells; ++cell) {
    if (!detail::flank_strip_cell_is_geometry_policy_exempt(state, cell)) {
      continue;
    }
    for (const int node : cell_nodes(state, cell)) {
      untangler_fixed_node[static_cast<std::size_t>(node)] = 1U;
    }
  }
  detail::run_local_jacobian_untangler(state,
                                       target_cell,
                                       0,
                                       window.i_last,
                                       window.j_first,
                                       window.j_last,
                                       untangler_fixed_node,
                                       candidate_r,
                                       candidate_z);
  for (std::size_t node = 0; node < node_count; ++node) {
    moved_node[node] =
        (candidate_r[node] != original_r[node] ||
         candidate_z[node] != original_z[node])
            ? 1U
            : 0U;
  }
  result.q_after = detail::flank_strip_band_minimum_quality_ratio(
      state,
      candidate_r,
      candidate_z,
      slot_cell,
      strip_cfg.band_layers,
      strip_cfg.band_halfwidth_j,
      q_argmin,
      focus_j);
  result.slip_after = detail::flank_strip_slip_mismatch_signed_ratio(
      state,
      candidate_r,
      candidate_z,
      window.j_first,
      window.j_last);

  std::vector<int> patch_cells;
  std::vector<std::uint8_t> patch_node(node_count, 0U);
  for (int cell = 0; cell < n_cells; ++cell) {
    if (detail::flank_strip_cell_is_geometry_policy_exempt(state, cell)) {
      continue;
    }
    const std::array<int, 4> nodes = cell_nodes(state, cell);
    const bool affected = std::any_of(
        nodes.begin(), nodes.end(), [&](const int node) {
          return moved_node[static_cast<std::size_t>(node)] != 0U;
        });
    if (!affected) {
      continue;
    }
    patch_cells.push_back(cell);
    for (const int node : nodes) {
      patch_node[static_cast<std::size_t>(node)] = 1U;
    }
  }
  if (patch_cells.empty()) {
    return reject("candidate_inadmissible");
  }
  if (!patch_is_single_material(state, cfg, patch_cells)) {
    return reject("multi_material_patch");
  }

  std::vector<PackedQuad> source_quads;
  std::vector<PackedQuad> destination_quads;
  source_quads.reserve(patch_cells.size());
  destination_quads.reserve(patch_cells.size());
  for (const int cell : patch_cells) {
    source_quads.push_back(
        pack_cell_quad(state, cell, original_r, original_z));
    destination_quads.push_back(
        pack_cell_quad(state, cell, candidate_r, candidate_z));
    const char* const reject_check =
        candidate_cell_reject_check(destination_quads.back(), cell / nz);
    if (reject_check[0] != '\0') {
      result.reject_cell = cell;
      result.reject_check = reject_check;
      unpack_quad(destination_quads.back(),
                  result.reject_quad_r,
                  result.reject_quad_z);
      return reject("candidate_inadmissible");
    }
  }
  // The mismatch-driven target reduces slip by construction up to the no-fold
  // clamps and window-end taper; acceptance requires consuming half a pitch or
  // halving a sub-pitch mismatch, while retaining quality.
  if (!std::isfinite(result.slip_before) ||
      !std::isfinite(result.slip_after) ||
      std::abs(result.slip_after) >
          std::abs(result.slip_before) -
              std::min(0.5, 0.5 * std::abs(result.slip_before))) {
    result.reject_check = "slip_reduction";
    return reject("candidate_inadmissible");
  }
  const double retained_quality =
      std::min(strip_cfg.release_quality_ratio, 0.5 * result.q_before);
  if (!std::isfinite(result.q_after) ||
      result.q_after < retained_quality) {
    result.reject_cell = q_argmin;
    result.reject_check = "quality_retention";
    if (q_argmin >= 0 && q_argmin < n_cells) {
      const PackedQuad reject_quad =
          pack_cell_quad(state, q_argmin, candidate_r, candidate_z);
      unpack_quad(reject_quad, result.reject_quad_r, result.reject_quad_z);
    }
    return reject("candidate_inadmissible");
  }

  const ixremap::PatchIntersection intersection =
      ixremap::build_patch_intersection(source_quads, destination_quads);
  result.partition_row_defect = intersection.max_row_defect;
  result.partition_col_defect = intersection.max_col_defect;
  if (!std::isfinite(intersection.max_row_defect) ||
      !std::isfinite(intersection.max_col_defect) ||
      intersection.max_row_defect > 1.0e-12 ||
      intersection.max_col_defect > 1.0e-12) {
    char partition_detail[256];
    std::snprintf(
        partition_detail, sizeof(partition_detail),
        "[flank-slip] partition detail row_defect=%.3e col_defect=%.3e "
        "split_failures=%d worst_row_cell=%d worst_col_cell=%d "
        "patch_cells=%zu",
        intersection.max_row_defect, intersection.max_col_defect,
        intersection.split_failures,
        patch_cells[static_cast<std::size_t>(intersection.worst_row_index)],
        patch_cells[static_cast<std::size_t>(intersection.worst_col_index)],
        patch_cells.size());
    core::log_info(partition_detail);
  }

  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> volume;
  std::vector<double> corner_mass;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  state.mass.copy_to_host(mass);
  state.rho.copy_to_host(rho);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.vol.copy_to_host(volume);
  state.corner_mass.copy_to_host(corner_mass);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  if (mass.size() != cell_count || rho.size() != cell_count ||
      ee.size() != cell_count || ei.size() != cell_count ||
      volume.size() != cell_count || velocity_r.size() != node_count ||
      velocity_z.size() != node_count ||
      (!state.hydro_active.empty() &&
       state.hydro_active.size() != cell_count) ||
      (state.corner_mass_initialized &&
       corner_mass.size() < 4U * cell_count)) {
    return reject("field_size_mismatch");
  }

  const std::size_t patch_size = patch_cells.size();
  std::vector<double> source_mass(patch_size, 0.0);
  std::vector<double> source_ie(patch_size, 0.0);
  std::vector<double> source_ii(patch_size, 0.0);
  std::vector<double> source_pr(patch_size, 0.0);
  std::vector<double> source_pz(patch_size, 0.0);
  std::vector<double> source_ke(patch_size, 0.0);
  std::vector<double> nodal_mass_before(node_count, 0.0);
  std::vector<std::uint8_t> nodal_hydro_active(
      node_count, state.hydro_active.empty() ? 1U : 0U);
  if (!state.hydro_active.empty()) {
    for (int cell = 0; cell < n_cells; ++cell) {
      if (state.hydro_active[static_cast<std::size_t>(cell)] == 0) {
        continue;
      }
      for (const int node : cell_nodes(state, cell)) {
        nodal_hydro_active[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }
  double mass_before = 0.0;
  double ie_before = 0.0;
  double ii_before = 0.0;
  double pr_before = 0.0;
  double pz_before = 0.0;
  double ke_before = 0.0;
  double pr_scale = 0.0;
  double pz_scale = 0.0;
  for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
    const int cell = patch_cells[local_cell];
    const std::size_t c = static_cast<std::size_t>(cell);
    const double cell_mass = rho[c] * intersection.src_vol[local_cell];
    if (!(cell_mass > 0.0) || !std::isfinite(cell_mass) ||
        !std::isfinite(ee[c]) || !std::isfinite(ei[c])) {
      return reject("audit_failed");
    }
    source_mass[local_cell] = cell_mass;
    source_ie[local_cell] = cell_mass * ee[c];
    source_ii[local_cell] = cell_mass * ei[c];
    double local_corner_mass[4];
    double local_vr[4];
    double local_vz[4];
    const std::array<int, 4> nodes = cell_nodes(state, cell);
    for (int corner = 0; corner < 4; ++corner) {
      local_corner_mass[corner] =
          state.corner_mass_initialized
              ? corner_mass[4U * c + static_cast<std::size_t>(corner)]
              : 0.25 * cell_mass;
      const std::size_t node = static_cast<std::size_t>(
          nodes[static_cast<std::size_t>(corner)]);
      nodal_mass_before[node] += local_corner_mass[corner];
      local_vr[corner] = velocity_r[node];
      local_vz[corner] = velocity_z[node];
    }
    ixremap::zonal_momentum_and_ke(local_corner_mass,
                                   local_vr,
                                   local_vz,
                                   source_pr[local_cell],
                                   source_pz[local_cell],
                                   source_ke[local_cell]);
    mass_before += source_mass[local_cell];
    ie_before += source_ie[local_cell];
    ii_before += source_ii[local_cell];
    pr_before += source_pr[local_cell];
    pz_before += source_pz[local_cell];
    ke_before += source_ke[local_cell];
    pr_scale += std::abs(source_pr[local_cell]);
    pz_scale += std::abs(source_pz[local_cell]);
  }

  std::vector<double> new_mass;
  std::vector<double> new_ie;
  std::vector<double> new_ii;
  std::vector<double> new_pr;
  std::vector<double> new_pz;
  std::vector<double> new_ke;
  // Exempt footprints are holes in the material partition. Volume exchanged
  // with a hole is contact-face compression/expansion; mass never crosses.
  std::vector<std::uint8_t> orphaned_source;
  transfer_extensive(
      intersection, source_mass, new_mass, &orphaned_source);
  for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
    if (orphaned_source[local_cell] != 0U &&
        source_mass[local_cell] != 0.0) {
      result.reject_cell = patch_cells[local_cell];
      return reject("orphaned_source_mass");
    }
  }
  transfer_extensive(intersection, source_ie, new_ie);
  transfer_extensive(intersection, source_ii, new_ii);
  transfer_extensive(intersection, source_pr, new_pr);
  transfer_extensive(intersection, source_pz, new_pz);
  transfer_extensive(intersection, source_ke, new_ke);

  std::vector<double> radiation;
  std::vector<double> radiation_new;
  std::size_t radiation_groups = 0U;
  if (!state.rad_E.empty()) {
    if (state.rad_E.size() % cell_count != 0U) {
      return reject("field_size_mismatch");
    }
    radiation_groups = state.rad_E.size() / cell_count;
    state.rad_E.copy_to_host(radiation);
    radiation_new = radiation;
    for (std::size_t group = 0; group < radiation_groups; ++group) {
      std::vector<double> source_radiation(patch_size, 0.0);
      for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
        const std::size_t c =
            static_cast<std::size_t>(patch_cells[local_cell]);
        source_radiation[local_cell] =
            radiation[c * radiation_groups + group] *
            intersection.src_vol[local_cell];
      }
      std::vector<double> destination_radiation;
      transfer_extensive(
          intersection, source_radiation, destination_radiation);
      for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
        const std::size_t c =
            static_cast<std::size_t>(patch_cells[local_cell]);
        radiation_new[c * radiation_groups + group] =
            destination_radiation[local_cell] /
            intersection.dst_vol[local_cell];
      }
    }
  }

  std::vector<double> corner_mass_new = corner_mass;
  if (corner_mass_new.size() < 4U * cell_count) {
    corner_mass_new.assign(4U * cell_count, 0.0);
  }
  std::vector<double> nodal_mass(node_count, 0.0);
  std::vector<double> nodal_pr(node_count, 0.0);
  std::vector<double> nodal_pz(node_count, 0.0);
  for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
    const int cell = patch_cells[local_cell];
    const std::size_t c = static_cast<std::size_t>(cell);
    double r[4];
    double z[4];
    unpack_quad(destination_quads[local_cell], r, z);
    double corner_volume[4];
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], corner_volume);
    const double corner_volume_sum =
        corner_volume[0] + corner_volume[1] +
        corner_volume[2] + corner_volume[3];
    double weights[4] = {0.25, 0.25, 0.25, 0.25};
    if (corner_volume_sum > 0.0 && std::isfinite(corner_volume_sum) &&
        std::all_of(corner_volume, corner_volume + 4,
                    [](const double value) {
                      return value >= 0.0 && std::isfinite(value);
                    })) {
      for (int corner = 0; corner < 4; ++corner) {
        weights[corner] = corner_volume[corner] / corner_volume_sum;
      }
    }
    constexpr double w_floor = 1.0e-6;
    double weight_sum = 0.0;
    for (double& weight : weights) {
      weight = std::max(weight, w_floor);
      weight_sum += weight;
    }
    for (double& weight : weights) {
      weight /= weight_sum;
    }
    // Active nodes require strictly positive gathered mass at the force stage;
    // a floored partition keeps the distribution conservative and the velocity
    // recovery finite.
    double local_corner_mass[4];
    double local_corner_pr[4];
    double local_corner_pz[4];
    ixremap::distribute_corner_masses_and_momenta(
        new_mass[local_cell],
        new_pr[local_cell],
        new_pz[local_cell],
        weights,
        local_corner_mass,
        local_corner_pr,
        local_corner_pz);
    const std::array<int, 4> nodes = cell_nodes(state, cell);
    for (int corner = 0; corner < 4; ++corner) {
      corner_mass_new[4U * c + static_cast<std::size_t>(corner)] =
          local_corner_mass[corner];
      const std::size_t node = static_cast<std::size_t>(
          nodes[static_cast<std::size_t>(corner)]);
      nodal_mass[node] += local_corner_mass[corner];
      nodal_pr[node] += local_corner_pr[corner];
      nodal_pz[node] += local_corner_pz[corner];
    }
  }

  for (std::size_t node = 0; node < node_count; ++node) {
    if (patch_node[node] != 0U && nodal_hydro_active[node] != 0U &&
        nodal_mass[node] <= 1.0e-12 * nodal_mass_before[node]) {
      result.reject_node = static_cast<int>(node);
      return reject("nodal_mass_collapse");
    }
  }

  std::vector<double> velocity_r_new = velocity_r;
  std::vector<double> velocity_z_new = velocity_z;
  for (std::size_t node = 0; node < node_count; ++node) {
    if (patch_node[node] != 0U && nodal_mass[node] > 0.0 &&
        std::isfinite(nodal_mass[node])) {
      velocity_r_new[node] = nodal_pr[node] / nodal_mass[node];
      velocity_z_new[node] = nodal_pz[node] / nodal_mass[node];
    }
  }

  double closure_shortfall = 0.0;
  double mass_after = 0.0;
  double ie_after = 0.0;
  double ii_after = 0.0;
  double pr_after = 0.0;
  double pz_after = 0.0;
  double ke_after = 0.0;
  for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
    const int cell = patch_cells[local_cell];
    const std::size_t c = static_cast<std::size_t>(cell);
    const std::array<int, 4> nodes = cell_nodes(state, cell);
    double local_corner_mass[4];
    double local_vr[4];
    double local_vz[4];
    for (int corner = 0; corner < 4; ++corner) {
      local_corner_mass[corner] =
          corner_mass_new[4U * c + static_cast<std::size_t>(corner)];
      const std::size_t node = static_cast<std::size_t>(
          nodes[static_cast<std::size_t>(corner)]);
      local_vr[corner] = velocity_r_new[node];
      local_vz[corner] = velocity_z_new[node];
    }
    double actual_pr = 0.0;
    double actual_pz = 0.0;
    double actual_ke = 0.0;
    ixremap::zonal_momentum_and_ke(local_corner_mass,
                                   local_vr,
                                   local_vz,
                                   actual_pr,
                                   actual_pz,
                                   actual_ke);
    const double delta_ke = new_ke[local_cell] - actual_ke;
    if (delta_ke < 0.0 && new_ii[local_cell] < -delta_ke) {
      closure_shortfall += -delta_ke - new_ii[local_cell];
      new_ii[local_cell] = 0.0;
    } else {
      new_ii[local_cell] += delta_ke;
    }
    mass_after += new_mass[local_cell];
    ie_after += new_ie[local_cell];
    ii_after += new_ii[local_cell];
    pr_after += actual_pr;
    pz_after += actual_pz;
    ke_after += actual_ke;
  }

  const double energy_before = ie_before + ii_before + ke_before;
  const double energy_after = ie_after + ii_after + ke_after;
  result.mass_defect_rel = relative_defect(
      mass_before, mass_after, mass_before);
  const double pr_defect = relative_defect(pr_before, pr_after, pr_scale);
  const double pz_defect = relative_defect(pz_before, pz_after, pz_scale);
  result.momentum_defect_rel = std::max(pr_defect, pz_defect);
  result.energy_defect_rel = relative_defect(
      energy_before, energy_after, std::abs(energy_before));
  if (closure_shortfall > 0.0) {
    result.energy_defect_rel = std::max(
        result.energy_defect_rel,
        closure_shortfall /
            std::max(std::abs(energy_before),
                     std::numeric_limits<double>::min()));
  }
  const double audit_tolerance =
      64.0 * DBL_EPSILON * std::max(1.0, static_cast<double>(patch_size));
  const bool positive =
      std::all_of(new_mass.begin(), new_mass.end(),
                  [](const double value) {
                    return value > 0.0 && std::isfinite(value);
                  }) &&
      std::all_of(intersection.dst_vol.begin(), intersection.dst_vol.end(),
                  [](const double value) {
                    return value > 0.0 && std::isfinite(value);
                  });
  if (!positive || result.mass_defect_rel > audit_tolerance ||
      result.energy_defect_rel > audit_tolerance ||
      result.momentum_defect_rel > audit_tolerance) {
    return reject("audit_failed");
  }

  std::vector<double> mass_new = mass;
  std::vector<double> rho_new = rho;
  std::vector<double> ee_new = ee;
  std::vector<double> ei_new = ei;
  std::vector<double> volume_new = volume;
  std::vector<double> mesh_volume_new = state.mesh.cell_vol;
  std::vector<double> mesh_area_new = state.mesh.cell_area;
  if (mesh_volume_new.size() != cell_count) {
    mesh_volume_new = volume;
  }
  if (mesh_area_new.size() != cell_count) {
    mesh_area_new.assign(cell_count, 0.0);
  }
  for (std::size_t local_cell = 0; local_cell < patch_size; ++local_cell) {
    const std::size_t c =
        static_cast<std::size_t>(patch_cells[local_cell]);
    mass_new[c] = new_mass[local_cell];
    volume_new[c] = intersection.dst_vol[local_cell];
    rho_new[c] = new_mass[local_cell] / intersection.dst_vol[local_cell];
    ee_new[c] = new_ie[local_cell] / new_mass[local_cell];
    ei_new[c] = new_ii[local_cell] / new_mass[local_cell];
    mesh_volume_new[c] = intersection.dst_vol[local_cell];
    mesh_area_new[c] = planar_quad_area(destination_quads[local_cell]);
  }

  state.x_r.copy_from_host(candidate_r);
  state.x_z.copy_from_host(candidate_z);
  state.mass.copy_from_host(mass_new);
  state.rho.copy_from_host(rho_new);
  state.ee.copy_from_host(ee_new);
  state.ei.copy_from_host(ei_new);
  state.vol.copy_from_host(volume_new);
  if (state.corner_mass.size() != corner_mass_new.size()) {
    state.corner_mass.reset(corner_mass_new.size());
  }
  state.corner_mass.copy_from_host(corner_mass_new);
  state.v_r.copy_from_host(velocity_r_new);
  state.v_z.copy_from_host(velocity_z_new);
  if (radiation_groups > 0U) {
    state.rad_E.copy_from_host(radiation_new);
  }
  state.mesh.cell_vol = std::move(mesh_volume_new);
  state.mesh.cell_area = std::move(mesh_area_new);
  result.applied = true;
  return result;
}

}  // namespace tenryu::hydro
