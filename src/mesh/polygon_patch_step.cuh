#pragma once

#include <cmath>

#include <cuda_runtime.h>

#include "mesh/polygon_patch.cuh"
#include "mesh/polygon_rz_work.cuh"

namespace tenryu::mesh::patch {

struct PatchStepConfig {
  double dt;
  double gamma;
  const bool* node_pinned;
};

struct PatchStepLedger {
  double work_done;
  double kinetic_energy_before;
  double kinetic_energy_after;
  double internal_energy_before;
  double internal_energy_after;
};

namespace detail {

__host__ __device__ inline double patch_cell_v1(
    const PolygonPatch& p, const int cell) {
  double r[kPatchMaxVerts]{};
  double z[kPatchMaxVerts]{};
  patch_cell_vertices(p, cell, r, z);
  return moments::poly_rz_moments_fan(r, z, p.cell_n[cell]).mr;
}

__host__ __device__ inline bool patch_node_is_pinned(
    const PatchStepConfig& cfg, const int node) {
  return cfg.node_pinned != nullptr && cfg.node_pinned[node];
}

}  // namespace detail

/**
 * Assembles the R-WEIGHTED pressure force F = -P*d(V1)/dx.
 *
 * This is the work/energy-conjugate force used by the patch shadow step.
 * It is a different FIX-2 measure object from the planar patch_node_forces
 * assembly in polygon_patch.cuh; the two assemblies are not interchangeable.
 */
__host__ __device__ inline void patch_node_forces_rz(
    const PolygonPatch& p, const double* pressure, double* f_r,
    double* f_z) {
  for (int node = 0; node < kPatchNodes; ++node) {
    f_r[node] = 0.0;
    f_z[node] = 0.0;
  }

  for (int cell = 0; cell < kPatchCells; ++cell) {
    double r[kPatchMaxVerts]{};
    double z[kPatchMaxVerts]{};
    detail::patch_cell_vertices(p, cell, r, z);
    for (int i = 0; i < p.cell_n[cell]; ++i) {
      double t_r = 0.0;
      double t_z = 0.0;
      forces::polygon_rz_work_surface(
          r, z, p.cell_n[cell], i, t_r, t_z);
      const int node = p.cell_nodes[cell][i];
      f_r[node] = fma(-pressure[cell], t_r, f_r[node]);
      f_z[node] = fma(-pressure[cell], t_z, f_z[node]);
    }
  }
}

/**
 * Builds the trajectory-constant star-P1 nodal masses at t=0.
 */
__host__ __device__ inline void patch_initial_node_masses(
    const PolygonPatch& p0, const double* rho0, double* m_node) {
  patch_node_masses(p0, rho0, m_node);
}

/**
 * Advances the mixed patch with one fixed-one-corrector midpoint shadow step.
 *
 * The initial star-P1 nodal masses are Lagrangian invariants between events.
 * They remain constant for the whole trajectory and are used for both
 * force-to-acceleration conversions and both kinetic-energy evaluations.
 */
__host__ __device__ inline void patch_midpoint_step(
    PolygonPatch& p, double* v_r, double* v_z, double* pressure,
    const double* m_node, const PatchStepConfig& cfg,
    PatchStepLedger* ledger) {
  double v1_start[kPatchCells]{};
  double pressure_start[kPatchCells]{};
  for (int cell = 0; cell < kPatchCells; ++cell) {
    v1_start[cell] = detail::patch_cell_v1(p, cell);
    pressure_start[cell] = pressure[cell];
  }

  if (ledger != nullptr) {
    ledger->work_done = 0.0;
    ledger->kinetic_energy_before = 0.0;
    ledger->kinetic_energy_after = 0.0;
    ledger->internal_energy_before = 0.0;
    ledger->internal_energy_after = 0.0;
    for (int node = 0; node < kPatchNodes; ++node) {
      const double speed_squared =
          fma(v_r[node], v_r[node],
              fma(v_z[node], v_z[node], 0.0));
      ledger->kinetic_energy_before =
          fma(0.5 * m_node[node], speed_squared,
              ledger->kinetic_energy_before);
    }
    const double gamma_minus_one = cfg.gamma - 1.0;
    for (int cell = 0; cell < kPatchCells; ++cell) {
      ledger->internal_energy_before =
          fma(pressure_start[cell],
              v1_start[cell] / gamma_minus_one,
              ledger->internal_energy_before);
    }
  }

  double a_start_r[kPatchNodes]{};
  double a_start_z[kPatchNodes]{};
  patch_node_forces_rz(
      p, pressure_start, a_start_r, a_start_z);
  for (int node = 0; node < kPatchNodes; ++node) {
    if (detail::patch_node_is_pinned(cfg, node)) {
      a_start_r[node] = 0.0;
      a_start_z[node] = 0.0;
    } else {
      a_start_r[node] /= m_node[node];
      a_start_z[node] /= m_node[node];
    }
  }

  const double half_dt = 0.5 * cfg.dt;
  PolygonPatch p_half = p;
  double v_half_r[kPatchNodes]{};
  double v_half_z[kPatchNodes]{};
  for (int node = 0; node < kPatchNodes; ++node) {
    if (detail::patch_node_is_pinned(cfg, node)) {
      v_half_r[node] = 0.0;
      v_half_z[node] = 0.0;
    } else {
      p_half.node_r[node] =
          fma(half_dt, v_r[node], p.node_r[node]);
      p_half.node_z[node] =
          fma(half_dt, v_z[node], p.node_z[node]);
      v_half_r[node] =
          fma(half_dt, a_start_r[node], v_r[node]);
      v_half_z[node] =
          fma(half_dt, a_start_z[node], v_z[node]);
    }
  }

  double v1_half[kPatchCells]{};
  double pressure_half[kPatchCells]{};
  for (int cell = 0; cell < kPatchCells; ++cell) {
    v1_half[cell] = detail::patch_cell_v1(p_half, cell);
    pressure_half[cell] =
        fma(pressure_start[cell],
            pow(v1_start[cell] / v1_half[cell], cfg.gamma), 0.0);
  }

  double a_half_r[kPatchNodes]{};
  double a_half_z[kPatchNodes]{};
  patch_node_forces_rz(
      p_half, pressure_half, a_half_r, a_half_z);
  for (int node = 0; node < kPatchNodes; ++node) {
    if (detail::patch_node_is_pinned(cfg, node)) {
      a_half_r[node] = 0.0;
      a_half_z[node] = 0.0;
    } else {
      a_half_r[node] /= m_node[node];
      a_half_z[node] /= m_node[node];
    }
  }

  for (int node = 0; node < kPatchNodes; ++node) {
    if (!detail::patch_node_is_pinned(cfg, node)) {
      p.node_r[node] =
          fma(cfg.dt, v_half_r[node], p.node_r[node]);
      p.node_z[node] =
          fma(cfg.dt, v_half_z[node], p.node_z[node]);
      v_r[node] = fma(cfg.dt, a_half_r[node], v_r[node]);
      v_z[node] = fma(cfg.dt, a_half_z[node], v_z[node]);
    }
  }

  double v1_new[kPatchCells]{};
  for (int cell = 0; cell < kPatchCells; ++cell) {
    v1_new[cell] = detail::patch_cell_v1(p, cell);
    pressure[cell] =
        fma(pressure_start[cell],
            pow(v1_start[cell] / v1_new[cell], cfg.gamma), 0.0);
  }

  if (ledger != nullptr) {
    for (int node = 0; node < kPatchNodes; ++node) {
      const double speed_squared =
          fma(v_r[node], v_r[node],
              fma(v_z[node], v_z[node], 0.0));
      ledger->kinetic_energy_after =
          fma(0.5 * m_node[node], speed_squared,
              ledger->kinetic_energy_after);
    }
    const double gamma_minus_one = cfg.gamma - 1.0;
    for (int cell = 0; cell < kPatchCells; ++cell) {
      ledger->internal_energy_after =
          fma(pressure[cell], v1_new[cell] / gamma_minus_one,
              ledger->internal_energy_after);
      ledger->work_done =
          fma(-pressure_half[cell],
              v1_new[cell] - v1_start[cell],
              ledger->work_done);
    }
  }
}

}  // namespace tenryu::mesh::patch
