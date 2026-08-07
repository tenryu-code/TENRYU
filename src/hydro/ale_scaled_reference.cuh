#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/mesh_motion_trace.hpp"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale {

namespace detail {

inline void scaled_reference_cuda_check(const cudaError_t err,
                                        const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline double multiblock_cell_orientation_sign(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int cell) {
  const auto idx = static_cast<std::size_t>(cell);
  TENRYU_ASSERT(idx < mb.cell_orientation_sign.size(),
                "scaled gamma-MVP reference orientation sign index out of range");
  const int sign = mb.cell_orientation_sign[idx];
  TENRYU_ASSERT(sign == 1 || sign == -1,
                "scaled gamma-MVP reference orientation sign must be +/-1");
  return static_cast<double>(sign);
}

inline std::vector<double> scaled_initial_cell_volumes_host(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double alpha) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "scaled gamma-MVP reference requires multiblock topology");
  TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size() &&
                    state.x_z_initial.size() == state.x_z.size(),
                "scaled gamma-MVP reference requires initial node storage");
  const int n_cells = state.mesh.topo.n_cells;
  std::vector<double> r0;
  std::vector<double> z0;
  state.x_r_initial.copy_to_host(r0);
  state.x_z_initial.copy_to_host(z0);

  const auto& mb = *state.mesh.topo.multiblock;
  const double alpha3 = alpha * alpha * alpha;
  std::vector<double> vol(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const auto nodes =
        tenryu::mesh::mesh_topo_cell_corner_nodes(state.mesh.topo, c, cfg.mesh);
    const double v_signed =
        tenryu::mesh::rz_quad_volume_exact(
            r0[static_cast<std::size_t>(nodes[0])],
            z0[static_cast<std::size_t>(nodes[0])],
            r0[static_cast<std::size_t>(nodes[1])],
            z0[static_cast<std::size_t>(nodes[1])],
            r0[static_cast<std::size_t>(nodes[2])],
            z0[static_cast<std::size_t>(nodes[2])],
            r0[static_cast<std::size_t>(nodes[3])],
            z0[static_cast<std::size_t>(nodes[3])]);
    const double v =
        multiblock_cell_orientation_sign(mb, c) * v_signed * alpha3;
    TENRYU_ASSERT(std::isfinite(v) && v > 0.0,
                  "scaled gamma-MVP reference produced invalid cell volume");
    vol[static_cast<std::size_t>(c)] = v;
  }
  return vol;
}

}  // namespace detail

inline double compute_scale_factor_alpha(const tenryu::core::State& state,
                                         const tenryu::core::Config& cfg) {
  TENRYU_ASSERT(tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh),
                "scaled gamma-MVP alpha requires multiblock config");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "scaled gamma-MVP alpha requires multiblock topology");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "scaled gamma-MVP alpha requires paired node coordinates");
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() == state.x_r.size(),
                "scaled gamma-MVP alpha requires node flags");
  const double s_max_initial = cfg.mesh.spherical_polar_s_max;
  TENRYU_ASSERT(std::isfinite(s_max_initial) && s_max_initial > 0.0,
                "scaled gamma-MVP alpha requires positive initial s_max");

  std::vector<double> r;
  std::vector<double> z;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);

  double sum = 0.0;
  int count = 0;
  for (std::size_t n = 0; n < r.size(); ++n) {
    const std::uint8_t flags = state.mesh.topo.node_flags[n];
    if ((flags & tenryu::mesh::NODE_OUTER_PHYSICAL_BOUNDARY) == 0U) {
      continue;
    }
    const double s = std::hypot(r[n], z[n]);
    TENRYU_ASSERT(std::isfinite(s),
                  "scaled gamma-MVP alpha saw non-finite outer radius");
    sum += s / s_max_initial;
    ++count;
  }
  TENRYU_ASSERT(count > 0,
                "scaled gamma-MVP alpha requires outer physical boundary nodes");
  return sum / static_cast<double>(count);
}

inline void build_scaled_gamma_mvp_target(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double alpha,
    double* d_target_r,
    double* d_target_z) {
  TENRYU_ASSERT(tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh),
                "scaled gamma-MVP target requires multiblock config");
  TENRYU_ASSERT(d_target_r != nullptr && d_target_z != nullptr,
                "scaled gamma-MVP target output buffer is null");
  TENRYU_ASSERT(std::isfinite(alpha) && alpha > 0.0,
                "scaled gamma-MVP target requires positive finite alpha");
  TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size() &&
                    state.x_z_initial.size() == state.x_z.size(),
                "scaled gamma-MVP target requires initial node storage");

  std::vector<double> r0;
  std::vector<double> z0;
  state.x_r_initial.copy_to_host(r0);
  state.x_z_initial.copy_to_host(z0);
  for (std::size_t n = 0; n < r0.size(); ++n) {
    r0[n] *= alpha;
    z0[n] *= alpha;
  }
  const std::size_t node_bytes = r0.size() * sizeof(double);
  detail::scaled_reference_cuda_check(
      cudaMemcpy(d_target_r, r0.data(), node_bytes, cudaMemcpyHostToDevice),
      "scaled gamma-MVP target r copy failed");
  detail::scaled_reference_cuda_check(
      cudaMemcpy(d_target_z, z0.data(), node_bytes, cudaMemcpyHostToDevice),
      "scaled gamma-MVP target z copy failed");
}

inline void install_scaled_gamma_mvp_reference(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double alpha) {
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "scaled gamma-MVP reference requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size(),
                "scaled gamma-MVP reference requires target volume storage");
  build_scaled_gamma_mvp_target(
      state, cfg, alpha, state.x_r_reference.data(), state.x_z_reference.data());
  state.cell_vol_initial.copy_from_host(
      detail::scaled_initial_cell_volumes_host(state, cfg, alpha));
}

inline bool prepare_scaled_gamma_mvp_reference_if_enabled(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  if (!cfg.numerics.ale.multiblock_scaled_reference_enabled ||
      !tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return false;
  }
  const double alpha = compute_scale_factor_alpha(state, cfg);
  tenryu::hydro::mesh_trace::trace_alpha_source(state, cfg, alpha);
  install_scaled_gamma_mvp_reference(state, cfg, alpha);
  tenryu::hydro::mesh_trace::trace_post_target(
      state, cfg, state.x_r_reference.data(), state.x_z_reference.data());
  return true;
}

}  // namespace tenryu::hydro::ale
