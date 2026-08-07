#include "hydro/cd_local_rezone.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/local_rezone.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"

namespace tenryu::hydro::ale {
namespace {

constexpr double kCdMonitorAlpha = 1.0;
constexpr double kCdMonitorBeta = 1.0;
constexpr int kCdLocalMaxIterations = 20;

struct CellPatch {
  int i0 = 0;
  int i1 = 0;
  int j0 = 0;
  int j1 = 0;
};

int node_index(const int i, const int j, const int nz) {
  return rz_node_index_2d(i, j, nz);
}

CellPatch patch_from_focus(const int focus_cell,
                           const int nr,
                           const int nz,
                           const int radius_i,
                           const int radius_j) {
  int ci = (nr > 0) ? nr / 2 : 0;
  int cj = (nz > 0) ? nz / 2 : 0;
  if (focus_cell >= 0 && nz > 0) {
    ci = focus_cell / nz;
    cj = focus_cell - ci * nz;
  }
  ci = std::clamp(ci, 0, std::max(nr - 1, 0));
  cj = std::clamp(cj, 0, std::max(nz - 1, 0));
  return CellPatch{
      std::clamp(ci - std::max(radius_i, 0), 0, std::max(nr - 1, 0)),
      std::clamp(ci + std::max(radius_i, 0), 0, std::max(nr - 1, 0)),
      std::clamp(cj - std::max(radius_j, 0), 0, std::max(nz - 1, 0)),
      std::clamp(cj + std::max(radius_j, 0), 0, std::max(nz - 1, 0))};
}

double reference_r(const core::Config& cfg, const int i, const int nr) {
  const double dr = (cfg.mesh.r_max - cfg.mesh.r_min) /
                    static_cast<double>(std::max(nr, 1));
  return cfg.mesh.r_min + dr * static_cast<double>(i);
}

double reference_z(const core::Config& cfg, const int j, const int nz) {
  const double dz = (cfg.mesh.z_max - cfg.mesh.z_min) /
                    static_cast<double>(std::max(nz, 1));
  return cfg.mesh.z_min + dz * static_cast<double>(j);
}

double cell_divergence(const std::vector<double>& r,
                       const std::vector<double>& z,
                       const std::vector<double>& vr,
                       const std::vector<double>& vz,
                       const int nr,
                       const int nz,
                       const int ci,
                       const int cj) {
  (void)nr;
  const int n00 = node_index(ci, cj, nz);
  const int n10 = node_index(ci + 1, cj, nz);
  const int n11 = node_index(ci + 1, cj + 1, nz);
  const int n01 = node_index(ci, cj + 1, nz);
  const double dr = std::max(1.0e-300,
                             0.5 * ((r[static_cast<std::size_t>(n10)] +
                                     r[static_cast<std::size_t>(n11)]) -
                                    (r[static_cast<std::size_t>(n00)] +
                                     r[static_cast<std::size_t>(n01)])));
  const double dz = std::max(1.0e-300,
                             0.5 * ((z[static_cast<std::size_t>(n01)] +
                                     z[static_cast<std::size_t>(n11)]) -
                                    (z[static_cast<std::size_t>(n00)] +
                                     z[static_cast<std::size_t>(n10)])));
  const double vr_l = 0.5 * (vr[static_cast<std::size_t>(n00)] +
                             vr[static_cast<std::size_t>(n01)]);
  const double vr_r = 0.5 * (vr[static_cast<std::size_t>(n10)] +
                             vr[static_cast<std::size_t>(n11)]);
  const double vz_b = 0.5 * (vz[static_cast<std::size_t>(n00)] +
                             vz[static_cast<std::size_t>(n10)]);
  const double vz_t = 0.5 * (vz[static_cast<std::size_t>(n01)] +
                             vz[static_cast<std::size_t>(n11)]);
  return (vr_r - vr_l) / dr + (vz_t - vz_b) / dz;
}

double cd_score_for_cell(const CellRegime* d_cell_regime, const int cell) {
  if (d_cell_regime == nullptr || cell < 0) {
    return 0.0;
  }
  CellRegime regime;
  CUDA_CHECK(cudaMemcpy(&regime,
                        d_cell_regime + cell,
                        sizeof(CellRegime),
                        cudaMemcpyDeviceToHost));
  return static_cast<double>(regime.cd_score);
}

double eval_corner_j(const std::vector<double>& r,
                     const std::vector<double>& z,
                     const int nz,
                     const int ci,
                     const int cj,
                     const int corner) {
  const int nodes[4] = {node_index(ci, cj, nz),
                        node_index(ci + 1, cj, nz),
                        node_index(ci + 1, cj + 1, nz),
                        node_index(ci, cj + 1, nz)};
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = r[static_cast<std::size_t>(nodes[k])];
    zz[k] = z[static_cast<std::size_t>(nodes[k])];
  }
  return corner_jacobian_from_quad(rr, zz, corner);
}

double min_patch_corner_j(const std::vector<double>& r,
                          const std::vector<double>& z,
                          const int nz,
                          const CellPatch& patch) {
  double min_j = std::numeric_limits<double>::infinity();
  for (int ci = patch.i0; ci <= patch.i1; ++ci) {
    for (int cj = patch.j0; cj <= patch.j1; ++cj) {
      for (int corner = 0; corner < 4; ++corner) {
        const double j = eval_corner_j(r, z, nz, ci, cj, corner);
        if (!std::isfinite(j)) {
          return -std::numeric_limits<double>::infinity();
        }
        min_j = std::min(min_j, j);
      }
    }
  }
  return min_j == std::numeric_limits<double>::infinity() ? 1.0 : min_j;
}

void exchange_nodes_if_needed(core::State& state,
                              const parallel::PartitionInfo& part,
                              parallel::CommBuffers* bufs) {
  if (part.n_ranks > 1 && bufs != nullptr) {
    double* node_ptrs[2] = {state.x_r.data(), state.x_z.data()};
    parallel::exchange_node_fields(
        part, *bufs, node_ptrs, 2, state.mesh.topo.n_nodes, nullptr, 6);
  }
}

}  // namespace

RezoneResult run_cd_local_winslow_rezone(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime* d_cell_regime) {
  RezoneResult out;
  out.min_quality =
      compute_min_quality(state, cfg, &out.mesh_tangle, &out.min_quality_cell_pre);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const CellPatch patch = patch_from_focus(request.focus_cell,
                                           nr,
                                           nz,
                                           request.patch_radius_i,
                                           request.patch_radius_j);
  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> vr;
  std::vector<double> vz;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);
  const std::vector<double> r_original = r;
  const std::vector<double> z_original = z;

  std::vector<double> monitor(static_cast<std::size_t>(nr * nz), 1.0);
  for (int ci = patch.i0; ci <= patch.i1; ++ci) {
    for (int cj = patch.j0; cj <= patch.j1; ++cj) {
      const int c = ci * nz + cj;
      const double s_cd = cd_score_for_cell(d_cell_regime, c);
      const double div_u = cell_divergence(r, z, vr, vz, nr, nz, ci, cj);
      const double dt_scale = state.dt > 0.0 ? state.dt : 1.0;
      monitor[static_cast<std::size_t>(c)] =
          1.0 + kCdMonitorAlpha * s_cd +
          kCdMonitorBeta * std::max(0.0, -dt_scale * div_u);
    }
  }

  const int max_iter =
      std::min(std::max(cfg.numerics.ale.max_iterations, 1), kCdLocalMaxIterations);
  for (int iter = 0; iter < max_iter; ++iter) {
    std::vector<double> next_r = r;
    std::vector<double> next_z = z;
    for (int i = patch.i0 + 1; i <= patch.i1; ++i) {
      for (int j = patch.j0 + 1; j <= patch.j1; ++j) {
        if (i <= 0 || i >= nr || j <= 0 || j >= nz) {
          continue;
        }
        const int id = node_index(i, j, nz);
        double w = 0.0;
        for (int ci = i - 1; ci <= i; ++ci) {
          for (int cj = j - 1; cj <= j; ++cj) {
            if (ci >= patch.i0 && ci <= patch.i1 && cj >= patch.j0 &&
                cj <= patch.j1) {
              w += monitor[static_cast<std::size_t>(ci * nz + cj)];
            }
          }
        }
        const double theta = std::min(0.5, 0.125 * std::max(w, 1.0));
        next_r[static_cast<std::size_t>(id)] =
            (1.0 - theta) * r[static_cast<std::size_t>(id)] +
            theta * reference_r(cfg, i, nr);
        next_z[static_cast<std::size_t>(id)] =
            (1.0 - theta) * z[static_cast<std::size_t>(id)] +
            theta * reference_z(cfg, j, nz);
      }
    }
    r.swap(next_r);
    z.swap(next_z);
  }

  const double min_j = min_patch_corner_j(r, z, nz, patch);
  if (min_j > detail::kJFloor) {
    state.x_r.copy_from_host(r);
    state.x_z.copy_from_host(z);
    exchange_nodes_if_needed(state, part, bufs);
    bool post_tangle = false;
    out.min_quality =
        compute_min_quality(state, cfg, &post_tangle, &out.min_quality_cell_post);
    out.mesh_tangle = post_tangle;
    if (reduction != nullptr) {
      out.min_quality = reduction->allreduce_min(out.min_quality);
      out.mesh_tangle =
          reduction->allreduce_max(out.mesh_tangle ? 1.0 : 0.0) > 0.5;
    }
    out.triggered = true;
    out.converged = !out.mesh_tangle;
    out.iterations = max_iter;
    out.residual = min_j;
    return out;
  }

  state.x_r.copy_from_host(r_original);
  state.x_z.copy_from_host(z_original);
  if (cfg.numerics.ale.multi_node_interior_repair_enabled) {
    return run_interior_multi_node_projection(
        state, cfg, part, bufs, reduction, request);
  }
  out.triggered = false;
  out.converged = false;
  out.residual = min_j;
  return out;
}

}  // namespace tenryu::hydro::ale
