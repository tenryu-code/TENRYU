#pragma once

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "hydro/compatible_av_csw.cuh"
#include "mesh/polygon_forces.cuh"
#include "mesh/polygon_rz_work.cuh"
#include "mesh/rz_moments.cuh"

namespace tenryu::mesh::pstep {

struct PolyMeshView {
  const int* cell_offsets;
  const int* cell_nodes;
  int n_cells;
  int n_nodes;
};

enum class NodeBc : int {
  kFree = 0,
  kFixed = 1,
  kPrescribed = 2,
};

struct MeshStepConfig {
  double dt;
  double gamma;
  const int* node_bc;
  const double* v_presc_r;
  const double* v_presc_z;
  double av_c1;
  double av_c2;
};

struct MeshStepLedger {
  double work_pressure;
  double work_av;
  double ke_before;
  double ke_after;
  double ie_before;
  double ie_after;
};

namespace detail {

__host__ __device__ inline int mesh_cell_vertex_count(
    const PolyMeshView& view, const int cell) {
  const int nverts =
      view.cell_offsets[cell + 1] - view.cell_offsets[cell];
  if (!(nverts >= 3 &&
        nverts <= hydro::compatible::kCsw98MaxSideVecs)) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "nverts >= 3 && nverts <= kCsw98MaxSideVecs",
        "polygon mesh step: cell vertex count outside supported range "
        "must not overrun local polygon storage",
        __FILE__, __LINE__);
#endif
  }
  return nverts;
}

__host__ __device__ inline int mesh_cell_vertices(
    const PolyMeshView& view, const double* node_r, const double* node_z,
    const int cell, double* r, double* z) {
  const int begin = view.cell_offsets[cell];
  const int nverts = mesh_cell_vertex_count(view, cell);
  for (int k = 0; k < nverts; ++k) {
    const int node = view.cell_nodes[begin + k];
    r[k] = node_r[node];
    z[k] = node_z[node];
  }
  return nverts;
}

__host__ __device__ inline NodeBc mesh_node_bc(
    const MeshStepConfig& cfg, const int node) {
  return static_cast<NodeBc>(cfg.node_bc[node]);
}

}  // namespace detail

/**
 * Returns the per-radian RZ volume moment V1 = integral(r)dA for one cell.
 */
__host__ __device__ inline double mesh_cell_v1(
    const PolyMeshView& view, const double* node_r, const double* node_z,
    const int cell) {
  double r[hydro::compatible::kCsw98MaxSideVecs]{};
  double z[hydro::compatible::kCsw98MaxSideVecs]{};
  const int nverts =
      detail::mesh_cell_vertices(view, node_r, node_z, cell, r, z);
  return moments::poly_rz_moments_fan(r, z, nverts).mr;
}

/**
 * Builds trajectory-constant per-radian star-P1 nodal masses.
 */
__host__ __device__ inline void mesh_node_masses(
    const PolyMeshView& view, const double* node_r, const double* node_z,
    const double* rho, double* m_node) {
  for (int node = 0; node < view.n_nodes; ++node) {
    m_node[node] = 0.0;
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    double r[hydro::compatible::kCsw98MaxSideVecs]{};
    double z[hydro::compatible::kCsw98MaxSideVecs]{};
    double weight[hydro::compatible::kCsw98MaxSideVecs]{};
    const int nverts =
        detail::mesh_cell_vertices(view, node_r, node_z, cell, r, z);
    moments::star_p1_vertex_r_moments(r, z, nverts, weight);
    const int begin = view.cell_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const int node = view.cell_nodes[begin + k];
      m_node[node] = fma(rho[cell], weight[k], m_node[node]);
    }
  }
}

/**
 * Assembles the R-weighted conjugate pressure force F = -P*d(V1)/dx.
 */
__host__ __device__ inline void mesh_node_forces_rz(
    const PolyMeshView& view, const double* node_r, const double* node_z,
    const double* pressure, double* f_r, double* f_z) {
  for (int node = 0; node < view.n_nodes; ++node) {
    f_r[node] = 0.0;
    f_z[node] = 0.0;
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    double r[hydro::compatible::kCsw98MaxSideVecs]{};
    double z[hydro::compatible::kCsw98MaxSideVecs]{};
    const int nverts =
        detail::mesh_cell_vertices(view, node_r, node_z, cell, r, z);
    const int begin = view.cell_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      double t_r = 0.0;
      double t_z = 0.0;
      forces::polygon_rz_work_surface(r, z, nverts, k, t_r, t_z);
      const int node = view.cell_nodes[begin + k];
      f_r[node] = fma(-pressure[cell], t_r, f_r[node]);
      f_z[node] = fma(-pressure[cell], t_z, f_z[node]);
    }
  }
}

/**
 * Assembles a CSW98-geometry edge-AV structure.
 *
 * The c1/c2 coefficients are structure-only placeholders. This shadow
 * evaluator does not claim the production CSW98 magnitudes or limiters.
 * Compressive edge pairs use the dissipative orientation
 * f_a -= mu*S, f_b += mu*S.
 */
__host__ __device__ inline void mesh_edge_av_forces(
    const PolyMeshView& view, const double* node_r, const double* node_z,
    const double* v_r, const double* v_z, const double* pressure,
    const double* rho, const double gamma, const double c1,
    const double c2, double* f_r, double* f_z) {
  for (int node = 0; node < view.n_nodes; ++node) {
    f_r[node] = 0.0;
    f_z[node] = 0.0;
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    double r[hydro::compatible::kCsw98MaxSideVecs]{};
    double z[hydro::compatible::kCsw98MaxSideVecs]{};
    const int nverts =
        detail::mesh_cell_vertices(view, node_r, node_z, cell, r, z);
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    hydro::compatible::csw98_cell_centroid(
        r, z, nverts, &centroid_r, &centroid_z);

    const int begin = view.cell_offsets[cell];
    for (int k = 0; k < nverts; ++k) {
      const int next = (k + 1 == nverts) ? 0 : k + 1;
      double s_r = 0.0;
      double s_z = 0.0;
      if (!hydro::compatible::csw98_median_svec_cart(
              r[k], z[k], r[next], z[next], centroid_r, centroid_z,
              &s_r, &s_z)) {
        continue;
      }

      const int node_a = view.cell_nodes[begin + k];
      const int node_b = view.cell_nodes[begin + next];
      const double comp =
          fma(s_r, v_r[node_b] - v_r[node_a],
              s_z * (v_z[node_b] - v_z[node_a]));
      if (comp < 0.0) {
        const double smag = sqrt(fma(s_r, s_r, s_z * s_z));
        const double du = comp / smag;
        const double cs = sqrt(gamma * pressure[cell] / rho[cell]);
        const double mu =
            rho[cell] *
            (fma(c1 * cs, fabs(du), fma(c2, du * du, 0.0)));
        f_r[node_a] = fma(-mu, s_r, f_r[node_a]);
        f_z[node_a] = fma(-mu, s_z, f_z[node_a]);
        f_r[node_b] = fma(mu, s_r, f_r[node_b]);
        f_z[node_b] = fma(mu, s_z, f_z[node_b]);
      }
    }
  }
}

/**
 * Advances an arbitrary CSR polygon mesh by one fixed-one-corrector midpoint
 * shadow step.
 *
 * Nodal star-P1 masses are fixed at t=0. Cell density used by the AV scaffold
 * is mass_c/V1 at each force state. The AV ledger convention is midpoint
 * mechanical work, dt*sum_nodes(f_av_half dot v_half), and does not include
 * the first force evaluation.
 */
__host__ __device__ inline void mesh_midpoint_step(
    const PolyMeshView& view, double* node_r, double* node_z, double* v_r,
    double* v_z, double* pressure, const double* mass_c,
    const double* m_node, const MeshStepConfig& cfg,
    MeshStepLedger* ledger) {
  const std::size_t n_nodes = static_cast<std::size_t>(view.n_nodes);
  const std::size_t n_cells = static_cast<std::size_t>(view.n_cells);
  double* scratch = new double[8 * n_nodes + 4 * n_cells];
  if (scratch == nullptr) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "scratch != nullptr",
        "polygon mesh midpoint step scratch allocation failed",
        __FILE__, __LINE__);
#endif
  }

  double* half_r = scratch;
  double* half_z = half_r + n_nodes;
  double* half_v_r = half_z + n_nodes;
  double* half_v_z = half_v_r + n_nodes;
  double* force_r = half_v_z + n_nodes;
  double* force_z = force_r + n_nodes;
  double* av_force_r = force_z + n_nodes;
  double* av_force_z = av_force_r + n_nodes;
  double* v1_start = av_force_z + n_nodes;
  double* v1_half = v1_start + n_cells;
  double* pressure_half = v1_half + n_cells;
  double* rho = pressure_half + n_cells;

  for (int node = 0; node < view.n_nodes; ++node) {
    const NodeBc bc = detail::mesh_node_bc(cfg, node);
    if (bc == NodeBc::kFixed) {
      v_r[node] = 0.0;
      v_z[node] = 0.0;
    } else if (bc == NodeBc::kPrescribed) {
      v_r[node] = cfg.v_presc_r[node];
      v_z[node] = cfg.v_presc_z[node];
    }
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    v1_start[cell] =
        mesh_cell_v1(view, node_r, node_z, cell);
    rho[cell] = mass_c[cell] / v1_start[cell];
  }

  if (ledger != nullptr) {
    ledger->work_pressure = 0.0;
    ledger->work_av = 0.0;
    ledger->ke_before = 0.0;
    ledger->ke_after = 0.0;
    ledger->ie_before = 0.0;
    ledger->ie_after = 0.0;
    for (int node = 0; node < view.n_nodes; ++node) {
      const double speed_squared =
          fma(v_r[node], v_r[node],
              fma(v_z[node], v_z[node], 0.0));
      ledger->ke_before =
          fma(0.5 * m_node[node], speed_squared, ledger->ke_before);
    }
    const double gamma_minus_one = cfg.gamma - 1.0;
    for (int cell = 0; cell < view.n_cells; ++cell) {
      ledger->ie_before =
          fma(pressure[cell], v1_start[cell] / gamma_minus_one,
              ledger->ie_before);
    }
  }

  mesh_node_forces_rz(
      view, node_r, node_z, pressure, force_r, force_z);
  mesh_edge_av_forces(
      view, node_r, node_z, v_r, v_z, pressure, rho, cfg.gamma,
      cfg.av_c1, cfg.av_c2, av_force_r, av_force_z);
  for (int node = 0; node < view.n_nodes; ++node) {
    force_r[node] += av_force_r[node];
    force_z[node] += av_force_z[node];
  }

  const double half_dt = 0.5 * cfg.dt;
  for (int node = 0; node < view.n_nodes; ++node) {
    const NodeBc bc = detail::mesh_node_bc(cfg, node);
    if (bc == NodeBc::kFixed) {
      half_r[node] = node_r[node];
      half_z[node] = node_z[node];
      half_v_r[node] = 0.0;
      half_v_z[node] = 0.0;
    } else if (bc == NodeBc::kPrescribed) {
      half_r[node] =
          fma(half_dt, cfg.v_presc_r[node], node_r[node]);
      half_z[node] =
          fma(half_dt, cfg.v_presc_z[node], node_z[node]);
      half_v_r[node] = cfg.v_presc_r[node];
      half_v_z[node] = cfg.v_presc_z[node];
    } else {
      half_r[node] = fma(half_dt, v_r[node], node_r[node]);
      half_z[node] = fma(half_dt, v_z[node], node_z[node]);
      half_v_r[node] =
          fma(half_dt, force_r[node] / m_node[node], v_r[node]);
      half_v_z[node] =
          fma(half_dt, force_z[node] / m_node[node], v_z[node]);
    }
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    v1_half[cell] =
        mesh_cell_v1(view, half_r, half_z, cell);
    pressure_half[cell] =
        fma(pressure[cell],
            pow(v1_start[cell] / v1_half[cell], cfg.gamma), 0.0);
    rho[cell] = mass_c[cell] / v1_half[cell];
  }

  mesh_node_forces_rz(
      view, half_r, half_z, pressure_half, force_r, force_z);
  mesh_edge_av_forces(
      view, half_r, half_z, half_v_r, half_v_z, pressure_half, rho,
      cfg.gamma, cfg.av_c1, cfg.av_c2, av_force_r, av_force_z);
  for (int node = 0; node < view.n_nodes; ++node) {
    force_r[node] += av_force_r[node];
    force_z[node] += av_force_z[node];
  }

  for (int node = 0; node < view.n_nodes; ++node) {
    const NodeBc bc = detail::mesh_node_bc(cfg, node);
    if (bc == NodeBc::kFixed) {
      v_r[node] = 0.0;
      v_z[node] = 0.0;
    } else if (bc == NodeBc::kPrescribed) {
      node_r[node] =
          fma(cfg.dt, cfg.v_presc_r[node], node_r[node]);
      node_z[node] =
          fma(cfg.dt, cfg.v_presc_z[node], node_z[node]);
      v_r[node] = cfg.v_presc_r[node];
      v_z[node] = cfg.v_presc_z[node];
    } else {
      node_r[node] = fma(cfg.dt, half_v_r[node], node_r[node]);
      node_z[node] = fma(cfg.dt, half_v_z[node], node_z[node]);
      v_r[node] =
          fma(cfg.dt, force_r[node] / m_node[node], v_r[node]);
      v_z[node] =
          fma(cfg.dt, force_z[node] / m_node[node], v_z[node]);
    }
  }

  const double gamma_minus_one = cfg.gamma - 1.0;
  for (int cell = 0; cell < view.n_cells; ++cell) {
    const double v1_new =
        mesh_cell_v1(view, node_r, node_z, cell);
    pressure[cell] =
        fma(pressure[cell],
            pow(v1_start[cell] / v1_new, cfg.gamma), 0.0);
    if (ledger != nullptr) {
      ledger->ie_after =
          fma(pressure[cell], v1_new / gamma_minus_one,
              ledger->ie_after);
      ledger->work_pressure =
          fma(-pressure_half[cell], v1_new - v1_start[cell],
              ledger->work_pressure);
    }
  }

  if (ledger != nullptr) {
    double av_power = 0.0;
    for (int node = 0; node < view.n_nodes; ++node) {
      const double speed_squared =
          fma(v_r[node], v_r[node],
              fma(v_z[node], v_z[node], 0.0));
      ledger->ke_after =
          fma(0.5 * m_node[node], speed_squared, ledger->ke_after);
      av_power =
          fma(av_force_r[node], half_v_r[node],
              fma(av_force_z[node], half_v_z[node], av_power));
    }
    ledger->work_av = fma(cfg.dt, av_power, 0.0);
  }

  delete[] scratch;
}

}  // namespace tenryu::mesh::pstep
