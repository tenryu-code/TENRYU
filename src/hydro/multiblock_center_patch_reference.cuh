#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/ale_tracking_reference.cuh"
#include "hydro/hydro_multiblock_topology.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale {

struct MultiblockCenterPatchResult {
  bool applicable = false;
  std::vector<std::uint8_t> cell_in_patch;
  std::vector<std::uint8_t> node_rezone_active;
  std::vector<std::uint8_t> node_patch_boundary;
  int n_patch_cells = 0;
  int n_active_nodes = 0;
  int n_boundary_nodes = 0;
  int n_quality_seed_cells = 0;
  int n_cap_cells = 0;
};

namespace multiblock_center_patch_detail {

constexpr std::uint8_t kLatchVol = 1U << 0;
constexpr std::uint8_t kLatchCornerJ = 1U << 1;
constexpr std::uint8_t kLatchGaussJ = 1U << 2;
constexpr double kGauss = 0.577350269189625764509148780501957456;

inline bool diffref_diag_enabled() {
  const char* const raw = std::getenv("TENRYU_I1B_DIFFREF_DIAG");
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

inline int active_nverts(const tenryu::core::State& state, const int cell) {
  if (state.mesh.cell_nverts.empty()) {
    return tenryu::mesh::kMeshTopoCellStorageSlots;
  }
  return tenryu::mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts,
                                                    cell);
}

inline double orientation_sign(const tenryu::mesh::MultiBlockTopology& mb,
                               const int cell) {
  TENRYU_ASSERT(static_cast<std::size_t>(cell) < mb.cell_orientation_sign.size(),
                "center patch requires multiblock cell orientation signs");
  const int sign = mb.cell_orientation_sign[static_cast<std::size_t>(cell)];
  TENRYU_ASSERT(sign == 1 || sign == -1,
                "center patch cell orientation sign must be +/-1");
  return static_cast<double>(sign);
}

inline double cross2(const double ar,
                     const double az,
                     const double br,
                     const double bz) {
  return ar * bz - az * br;
}

inline void load_cell_nodes(const tenryu::core::State& state,
                            const std::vector<double>& r,
                            const std::vector<double>& z,
                            const int cell,
                            double* rr,
                            double* zz) {
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(static_cast<std::size_t>(cell) + 1U <
                    mb.cell_node_csr_offsets.size(),
                "center patch requires cell-node CSR offsets");
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int next =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  TENRYU_ASSERT(next - off == tenryu::mesh::kMeshTopoCellStorageSlots,
                "center patch requires four cell-node storage slots");
  const int nverts = active_nverts(state, cell);
  for (int k = 0; k < nverts; ++k) {
    const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    TENRYU_ASSERT(n >= 0 && n < topo.n_nodes,
                  "center patch cell-node CSR node id out of range");
    rr[k] = r[static_cast<std::size_t>(n)];
    zz[k] = z[static_cast<std::size_t>(n)];
  }
}

inline double quality_ratio_from_positive_values(const double* values,
                                                 const int count) {
  double min_v = std::numeric_limits<double>::infinity();
  double max_v = 0.0;
  for (int k = 0; k < count; ++k) {
    const double v = values[k];
    if (!std::isfinite(v) || !(v > 0.0)) {
      return -std::numeric_limits<double>::infinity();
    }
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }
  if (!std::isfinite(min_v) || !std::isfinite(max_v) || !(max_v > 0.0)) {
    return -std::numeric_limits<double>::infinity();
  }
  return min_v / max_v;
}

inline void corner_jacobians(const double* r,
                             const double* z,
                             const int nverts,
                             const double orient,
                             double* j) {
  if (nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const int km = (k + 2) % 3;
      j[k] = orient * cross2(r[kp] - r[k], z[kp] - z[k],
                             r[km] - r[k], z[km] - z[k]);
    }
    return;
  }
  j[0] = orient * cross2(r[1] - r[0], z[1] - z[0],
                         r[3] - r[0], z[3] - z[0]);
  j[1] = orient * cross2(r[1] - r[0], z[1] - z[0],
                         r[2] - r[1], z[2] - z[1]);
  j[2] = orient * cross2(r[2] - r[3], z[2] - z[3],
                         r[2] - r[1], z[2] - z[1]);
  j[3] = orient * cross2(r[2] - r[3], z[2] - z[3],
                         r[3] - r[0], z[3] - z[0]);
}

inline double cell_corner_j_ratio(const double* r,
                                  const double* z,
                                  const int nverts,
                                  const double orient) {
  double j[tenryu::mesh::kMeshTopoCellStorageSlots]{};
  corner_jacobians(r, z, nverts, orient, j);
  return quality_ratio_from_positive_values(j, nverts);
}

inline double cell_gauss_j_ratio(const double* r,
                                 const double* z,
                                 const int nverts,
                                 const double orient) {
  if (nverts == 3) {
    const double j =
        orient * cross2(r[1] - r[0], z[1] - z[0],
                        r[2] - r[0], z[2] - z[0]);
    return (std::isfinite(j) && j > 0.0)
               ? 1.0
               : -std::numeric_limits<double>::infinity();
  }

  double j[4]{};
  int slot = 0;
  const double xis[2] = {-kGauss, kGauss};
  const double etas[2] = {-kGauss, kGauss};
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
      const double dr_dxi = dN1_dxi * r[0] + dN2_dxi * r[1] +
                            dN3_dxi * r[2] + dN4_dxi * r[3];
      const double dz_dxi = dN1_dxi * z[0] + dN2_dxi * z[1] +
                            dN3_dxi * z[2] + dN4_dxi * z[3];
      const double dr_deta = dN1_deta * r[0] + dN2_deta * r[1] +
                             dN3_deta * r[2] + dN4_deta * r[3];
      const double dz_deta = dN1_deta * z[0] + dN2_deta * z[1] +
                             dN3_deta * z[2] + dN4_deta * z[3];
      j[slot++] = orient * (dr_dxi * dz_deta - dr_deta * dz_dxi);
    }
  }
  return quality_ratio_from_positive_values(j, 4);
}

inline double cell_current_volume(const double* r,
                                  const double* z,
                                  const int nverts,
                                  const double orient) {
  const double signed_volume =
      nverts == 3
          ? tenryu::hydro::rz::rz_polygon_volume_exact(r, z, 3)
          : tenryu::mesh::rz_quad_volume_exact(r[0], z[0],
                                               r[1], z[1],
                                               r[2], z[2],
                                               r[3], z[3]);
  return orient * signed_volume;
}

struct CellQuality {
  double volume_ratio = -std::numeric_limits<double>::infinity();
  double corner_j_ratio = -std::numeric_limits<double>::infinity();
  double gauss_j_ratio = -std::numeric_limits<double>::infinity();
};

inline CellQuality evaluate_cell_quality(const tenryu::core::State& state,
                                         const std::vector<double>& r,
                                         const std::vector<double>& z,
                                         const std::vector<double>& vol0,
                                         const int cell) {
  const auto& mb = *state.mesh.topo.multiblock;
  double rr[tenryu::mesh::kMeshTopoCellStorageSlots]{};
  double zz[tenryu::mesh::kMeshTopoCellStorageSlots]{};
  load_cell_nodes(state, r, z, cell, rr, zz);
  const int nverts = active_nverts(state, cell);
  const double orient = orientation_sign(mb, cell);

  CellQuality q;
  const double v = cell_current_volume(rr, zz, nverts, orient);
  const double v0 = vol0[static_cast<std::size_t>(cell)];
  if (std::isfinite(v) && std::isfinite(v0) && v0 > 0.0) {
    q.volume_ratio = v / v0;
  }
  q.corner_j_ratio = cell_corner_j_ratio(rr, zz, nverts, orient);
  q.gauss_j_ratio = cell_gauss_j_ratio(rr, zz, nverts, orient);
  return q;
}

inline bool hysteresis_on(const bool was_latched,
                          const double value,
                          const double on,
                          const double off) {
  if (!std::isfinite(value)) {
    return true;
  }
  if (value < on) {
    return true;
  }
  return was_latched && !(value > off);
}

inline std::uint8_t update_latch_bits(
    const std::uint8_t previous,
    const CellQuality& q,
    const tenryu::core::Config::NumericsConfig::AleConfig& ale) {
  std::uint8_t next = 0U;
  if (hysteresis_on((previous & kLatchVol) != 0U,
                    q.volume_ratio,
                    ale.multiblock_center_patch_vol_on,
                    ale.multiblock_center_patch_vol_off)) {
    next |= kLatchVol;
  }
  if (hysteresis_on((previous & kLatchCornerJ) != 0U,
                    q.corner_j_ratio,
                    ale.multiblock_center_patch_cornerj_on,
                    ale.multiblock_center_patch_cornerj_off)) {
    next |= kLatchCornerJ;
  }
  if (hysteresis_on((previous & kLatchGaussJ) != 0U,
                    q.gauss_j_ratio,
                    ale.multiblock_center_patch_gaussj_on,
                    ale.multiblock_center_patch_gaussj_off)) {
    next |= kLatchGaussJ;
  }
  return next;
}

inline const tenryu::mesh::BlockInfo* central_core_block(
    const tenryu::mesh::MultiBlockTopology& mb) {
  for (const auto& block : mb.blocks) {
    if (block.role == tenryu::mesh::BlockRole::CENTRAL_CORE) {
      return &block;
    }
  }
  return nullptr;
}

inline double ring_fallback_xi_cutoff(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int ring_max) {
  // Without cap-layer metadata, fold logical rings into cell-centered xi bands.
  if (ring_max < 0 || mb.has_trifan_cap) {
    return 0.0;
  }
  const tenryu::mesh::BlockInfo* core = central_core_block(mb);
  if (core == nullptr || core->n_i_cells <= 0) {
    return 0.0;
  }
  const double bands = static_cast<double>(core->n_i_cells);
  return std::clamp((static_cast<double>(ring_max) + 1.0) / bands,
                    0.0,
                    1.0);
}

inline bool trifan_cap_layer(const tenryu::mesh::MultiBlockTopology& mb,
                             const int cell,
                             int* layer) {
  // For tri-fan cap meshes, low cap_cell_id ordering gives logical layer l.
  if (!mb.has_trifan_cap || cell < 0 || cell >= mb.n_cells_cap) {
    return false;
  }
  TENRYU_ASSERT(mb.n_cap > 0, "center patch requires positive cap ring count");
  const int ntheta = 4 * mb.n_cap;
  TENRYU_ASSERT(ntheta > 0 && mb.n_cells_cap == mb.n_cap * ntheta,
                "center patch cap metadata is inconsistent");
  *layer = cell / ntheta;
  return true;
}

inline void dilate_multiblock_cells(
    const tenryu::core::State& state,
    const std::vector<std::uint8_t>& in,
    std::vector<std::uint8_t>& out) {
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  TENRYU_ASSERT(mb.face_adj_csr_offsets.size() ==
                    static_cast<std::size_t>(topo.n_cells + 1),
                "center patch requires face-adjacency CSR offsets");
  TENRYU_ASSERT(mb.face_adj_csr_indices.size() ==
                    static_cast<std::size_t>(topo.n_cells) *
                        tenryu::mesh::kMeshTopoCellStorageSlots,
                "center patch requires face-adjacency CSR indices");
  for (int c = 0; c < topo.n_cells; ++c) {
    if (in[static_cast<std::size_t>(c)] == 0U) {
      continue;
    }
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts = active_nverts(state, c);
    for (int local = 0; local < tenryu::mesh::kMeshTopoCellStorageSlots;
         ++local) {
      if (!tenryu::mesh::mesh_topo_local_face_is_active(nverts, local)) {
        continue;
      }
      const int nb =
          mb.face_adj_csr_indices[static_cast<std::size_t>(off + local)];
      if (nb >= 0 && nb < topo.n_cells) {
        out[static_cast<std::size_t>(nb)] = 1U;
      }
    }
  }
}

inline bool node_pinned(const tenryu::mesh::MeshTopology& topo,
                        const tenryu::mesh::MultiBlockTopology& mb,
                        const int node) {
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(topo.n_nodes),
                "center patch requires node flags");
  const std::uint8_t flags = topo.node_flags[static_cast<std::size_t>(node)];
  // NODE_OUTER_PHYSICAL_BOUNDARY: a rezone-active outer-boundary node lets
  // the CSR remap's boundary faces sweep, and the boundary flux kernel
  // donates the cell's own state across the domain boundary — fabricating
  // (outward sweep) or destroying (inward) mass at rho*dV. The per-block
  // Winslow smoother already pins these; the center patch must too. The
  // audited fabrication event entered through exactly this gap.
  if ((flags & (tenryu::mesh::NODE_CENTER |
                tenryu::mesh::NODE_AXIS |
                tenryu::mesh::NODE_POLE_AXIS |
                tenryu::mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) != 0U) {
    return true;
  }
  return mb.has_trifan_cap &&
         node == tenryu::mesh::mesh_topo_cap_apex_node_id(mb);
}

inline void log_counts(const tenryu::core::State& state,
                       const MultiblockCenterPatchResult& result) {
  if (!diffref_diag_enabled()) {
    return;
  }
  std::ostringstream oss;
  oss << "[multiblock_center_patch_builder] step=" << state.step
      << " patch_cells=" << result.n_patch_cells
      << " active_nodes=" << result.n_active_nodes
      << " boundary_nodes=" << result.n_boundary_nodes
      << " quality_seed_cells=" << result.n_quality_seed_cells
      << " cap_cells=" << result.n_cap_cells;
  tenryu::core::log_info(oss.str());
}

}  // namespace multiblock_center_patch_detail

inline bool multiblock_center_patch_reference_applicable(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  return cfg.numerics.ale
             .multiblock_lagrangian_bulk_center_patch_reference_enabled &&
         cfg.numerics.ale.conservative_remap_enabled &&
         cfg.numerics.ale.conservative_remap_target == "reference" &&
         cfg.main.dim == 2 &&
         cfg.main.dimension == "2D_RZ" &&
         tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh) &&
         state.mesh.dim == 2 &&
         state.mesh.topo.multiblock.has_value();
}

inline MultiblockCenterPatchResult build_multiblock_center_quality_patch(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  MultiblockCenterPatchResult result;
  if (!multiblock_center_patch_reference_applicable(state, cfg)) {
    return result;
  }
  result.applicable = true;

  namespace detail = multiblock_center_patch_detail;
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  const int n_cells = topo.n_cells;
  const int n_nodes = topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "center patch requires non-empty multiblock topology");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes),
                "center patch requires current node coordinates");
  TENRYU_ASSERT(state.cell_vol_initial.size() ==
                    static_cast<std::size_t>(n_cells),
                "center patch requires initial cell volumes");
  TENRYU_ASSERT(state.mesh.cell_nverts.empty() ||
                    state.mesh.cell_nverts.size() ==
                        static_cast<std::size_t>(n_cells),
                "center patch requires empty or per-cell cell_nverts");

  if (!multiblock_reference_xi_built(state)) {
    build_multiblock_reference_xi_initial(state, cfg);
  }

  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> vol0;
  std::vector<double> xi_cell;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  state.cell_vol_initial.copy_to_host(vol0);
  state.ref_xi_cell.copy_to_host(xi_cell);
  TENRYU_ASSERT(xi_cell.size() == static_cast<std::size_t>(n_cells),
                "center patch requires ref_xi_cell per cell");

  if (state.center_patch_latch.size() !=
      static_cast<std::size_t>(n_cells)) {
    state.center_patch_latch.assign(static_cast<std::size_t>(n_cells), 0U);
  }
  std::vector<std::uint8_t> next_latch = state.center_patch_latch;

  result.cell_in_patch.assign(static_cast<std::size_t>(n_cells), 0U);
  const auto& ale = cfg.numerics.ale;
  const double xi_center = ale.multiblock_center_patch_xi_center;
  const int ring_max = ale.multiblock_center_patch_ring_max;
  const double ring_xi_cutoff =
      detail::ring_fallback_xi_cutoff(mb, ring_max);

  for (int c = 0; c < n_cells; ++c) {
    bool permanent = false;
    if (xi_center > 0.0 &&
        xi_cell[static_cast<std::size_t>(c)] < xi_center) {
      permanent = true;
    }
    if (!mb.has_trifan_cap && ring_xi_cutoff > 0.0 &&
        xi_cell[static_cast<std::size_t>(c)] <= ring_xi_cutoff) {
      permanent = true;
    }

    int cap_layer = -1;
    const bool cap_cell = detail::trifan_cap_layer(mb, c, &cap_layer);
    if (cap_cell && (cap_layer == 0 || cap_layer <= ring_max)) {
      permanent = true;
    }
    if (cap_cell && permanent) {
      ++result.n_cap_cells;
    }

    const detail::CellQuality q =
        detail::evaluate_cell_quality(state, r, z, vol0, c);
    next_latch[static_cast<std::size_t>(c)] =
        detail::update_latch_bits(
            state.center_patch_latch[static_cast<std::size_t>(c)],
            q,
            ale);
    if (next_latch[static_cast<std::size_t>(c)] != 0U) {
      ++result.n_quality_seed_cells;
    }
    if (permanent || next_latch[static_cast<std::size_t>(c)] != 0U) {
      result.cell_in_patch[static_cast<std::size_t>(c)] = 1U;
    }
  }
  state.center_patch_latch.swap(next_latch);

  for (int layer = 0; layer < ale.multiblock_center_patch_halo_layers;
       ++layer) {
    std::vector<std::uint8_t> next = result.cell_in_patch;
    detail::dilate_multiblock_cells(state, result.cell_in_patch, next);
    result.cell_in_patch.swap(next);
  }

  for (const std::uint8_t in_patch : result.cell_in_patch) {
    if (in_patch != 0U) {
      ++result.n_patch_cells;
    }
  }

  const std::vector<std::uint8_t>* cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? &state.mesh.cell_nverts
          : nullptr;
  const tenryu::hydro::ReverseCellNodeCSR reverse_csr =
      tenryu::hydro::build_reverse_cell_node_csr(
          mb, n_nodes, cell_nverts, state.mesh.corner_stride);
  result.node_rezone_active.assign(static_cast<std::size_t>(n_nodes), 0U);
  result.node_patch_boundary.assign(static_cast<std::size_t>(n_nodes), 0U);
  for (int n = 0; n < n_nodes; ++n) {
    const int begin = reverse_csr.node_offsets[static_cast<std::size_t>(n)];
    const int end = reverse_csr.node_offsets[static_cast<std::size_t>(n) + 1U];
    bool touches_patch = false;
    bool touches_outside = false;
    for (int p = begin; p < end; ++p) {
      const int c = reverse_csr.node_cells[static_cast<std::size_t>(p)];
      TENRYU_ASSERT(c >= 0 && c < n_cells,
                    "center patch reverse CSR cell id out of range");
      if (result.cell_in_patch[static_cast<std::size_t>(c)] != 0U) {
        touches_patch = true;
      } else {
        touches_outside = true;
      }
    }
    if (touches_patch && touches_outside) {
      result.node_patch_boundary[static_cast<std::size_t>(n)] = 1U;
    } else if (touches_patch && !detail::node_pinned(topo, mb, n)) {
      result.node_rezone_active[static_cast<std::size_t>(n)] = 1U;
    }
  }

  for (int n = 0; n < n_nodes; ++n) {
    const bool active =
        result.node_rezone_active[static_cast<std::size_t>(n)] != 0U;
    const bool boundary =
        result.node_patch_boundary[static_cast<std::size_t>(n)] != 0U;
    TENRYU_ASSERT(!(active && boundary),
                  "center patch active and boundary node masks overlap");
    if (active) {
      ++result.n_active_nodes;
    }
    if (boundary) {
      ++result.n_boundary_nodes;
    }
  }

  detail::log_counts(state, result);
  return result;
}

}  // namespace tenryu::hydro::ale
