#include "hydro/reference_barrier_ale.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/ale_remap.cuh"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/ale_scaled_reference.cuh"
#include "hydro/anti_hourglass.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/button_morph_indexing.hpp"
#include "hydro/core_freeze_ale.cuh"
#include "hydro/mesh_motion_trace.hpp"
#include "hydro/pole_axis_constraints.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"
#include "mesh/multiblock_theta_ladder.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kPi = 3.14159265358979323846;

void cuda_check(const cudaError_t err, const char* msg) {
  TENRYU_ASSERT(err == cudaSuccess, msg);
}

std::uint8_t* upload_node_flags_if_constraints(const tenryu::core::State& state) {
  const auto& node_flags = state.mesh.topo.node_flags;
  if (node_flags.size() != state.x_r.size()) {
    return nullptr;
  }
  const bool has_constraints = std::any_of(
      node_flags.begin(), node_flags.end(), [](const std::uint8_t flags) {
        return (flags & (pole_axis::kNodeCenterFlag |
                         pole_axis::kNodePoleAxisFlag)) != 0U;
      });
  if (!has_constraints) {
    return nullptr;
  }
  std::uint8_t* d_node_flags = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_node_flags),
                        node_flags.size() * sizeof(std::uint8_t)),
             "reference barrier cudaMalloc node_flags failed");
  cuda_check(cudaMemcpy(d_node_flags,
                        node_flags.data(),
                        node_flags.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "reference barrier cudaMemcpy node_flags failed");
  return d_node_flags;
}

int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ double cross2(const double ar,
                                  const double az,
                                  const double br,
                                  const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ void corner_jacobians(const double* r,
                                          const double* z,
                                          double* j) {
  j[0] = cross2(r[1] - r[0], z[1] - z[0], r[3] - r[0], z[3] - z[0]);
  j[1] = cross2(r[1] - r[0], z[1] - z[0], r[2] - r[1], z[2] - z[1]);
  j[2] = cross2(r[2] - r[3], z[2] - z[3], r[2] - r[1], z[2] - z[1]);
  j[3] = cross2(r[2] - r[3], z[2] - z[3], r[3] - r[0], z[3] - z[0]);
}

void triangle_corner_jacobians(const double* r, const double* z, double* j) {
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    j[k] = cross2(r[kp] - r[k], z[kp] - z[k],
                  r[km] - r[k], z[km] - z[k]);
  }
}

double diagonal_corner_b_eff(const double* r, const double* z) {
  return 0.5 * (std::hypot(r[1] - r[0], z[1] - z[0]) +
                std::hypot(r[3] - r[0], z[3] - z[0]));
}

int state_cell_active_nverts(const tenryu::core::State& state,
                             const int c,
                             const int n_cells) {
  return state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
             ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
             : mesh::kMeshTopoCellStorageSlots;
}

bool state_has_tri_cell_nverts(const tenryu::core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  if (state.mesh.cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(state.mesh.cell_nverts.begin(),
                     state.mesh.cell_nverts.end(),
                     [](const std::uint8_t nverts) { return nverts == 3U; });
}

std::uint8_t* upload_cell_nverts_if_active(const tenryu::core::State& state) {
  if (!state_has_tri_cell_nverts(state)) {
    return nullptr;
  }
  std::uint8_t* d_cell_nverts = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                        state.mesh.cell_nverts.size() * sizeof(std::uint8_t)),
             "reference barrier cudaMalloc cell_nverts failed");
  cuda_check(cudaMemcpy(d_cell_nverts,
                        state.mesh.cell_nverts.data(),
                        state.mesh.cell_nverts.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "reference barrier cudaMemcpy cell_nverts failed");
  return d_cell_nverts;
}

double compute_axis_margin_min_host(const tenryu::core::State& state) {
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    if (n_cells <= 0 ||
        mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells) + 1U ||
        mb.cell_node_csr_indices.size() != static_cast<std::size_t>(n_cells) * 4U) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<double> r;
    std::vector<double> z;
    state.x_r.copy_to_host(r);
    state.x_z.copy_to_host(z);

    double out = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double rc = 0.0;
      double zc = 0.0;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        rc += r[static_cast<std::size_t>(n)];
        zc += z[static_cast<std::size_t>(n)];
      }
      if (active_nverts == 3) {
        rc *= (1.0 / 3.0);
        zc *= (1.0 / 3.0);
      } else {
        rc *= 0.25;
        zc *= 0.25;
      }
      const double s = std::hypot(rc, zc);
      if (s > 0.0 && std::isfinite(s)) {
        out = std::min(out, std::abs(rc) / s);
      }
    }
    return out;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> r;
  std::vector<double> z;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);

  double out = std::numeric_limits<double>::infinity();
  const int i_max = std::min(nr - 1, 2);
  for (int i = 0; i <= i_max; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int n0 = node_index(i, j, nz);
      const int n1 = node_index(i + 1, j, nz);
      const int n2 = node_index(i + 1, j + 1, nz);
      const int n3 = node_index(i, j + 1, nz);
      const double rc = 0.25 * (r[n0] + r[n1] + r[n2] + r[n3]);
      const double zc = 0.25 * (z[n0] + z[n1] + z[n2] + z[n3]);
      const double s = std::hypot(rc, zc);
      if (s > 0.0 && std::isfinite(s)) {
        out = std::min(out, std::abs(rc) / s);
      }
    }
  }
  return out;
}

double compute_corner_j_ratio_min_host(const tenryu::core::State& state) {
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    if (n_cells <= 0 ||
        mb.cell_node_csr_offsets.size() != static_cast<std::size_t>(n_cells) + 1U ||
        mb.cell_node_csr_indices.size() != static_cast<std::size_t>(n_cells) * 4U ||
        state.x_r_initial.size() != state.x_r.size() ||
        state.x_z_initial.size() != state.x_z.size()) {
      return std::numeric_limits<double>::infinity();
    }
    std::vector<double> r_old;
    std::vector<double> z_old;
    std::vector<double> r_new;
    std::vector<double> z_new;
    state.x_r_initial.copy_to_host(r_old);
    state.x_z_initial.copy_to_host(z_old);
    state.x_r.copy_to_host(r_new);
    state.x_z.copy_to_host(z_new);

    double out = std::numeric_limits<double>::infinity();
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double r0[4]{};
      double z0[4]{};
      double r1[4]{};
      double z1[4]{};
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        r0[k] = r_old[static_cast<std::size_t>(n)];
        z0[k] = z_old[static_cast<std::size_t>(n)];
        r1[k] = r_new[static_cast<std::size_t>(n)];
        z1[k] = z_new[static_cast<std::size_t>(n)];
      }
      double j0[4]{};
      double j1[4]{};
      if (active_nverts == 3) {
        triangle_corner_jacobians(r0, z0, j0);
        triangle_corner_jacobians(r1, z1, j1);
      } else {
        corner_jacobians(r0, z0, j0);
        corner_jacobians(r1, z1, j1);
      }
      for (int k = 0; k < active_nverts; ++k) {
        if (j0[k] != 0.0 && std::isfinite(j0[k]) && std::isfinite(j1[k])) {
          out = std::min(out, j1[k] / j0[k]);
        }
      }
      if (active_nverts != 3) {
        const double b0 = diagonal_corner_b_eff(r0, z0);
        const double b1 = diagonal_corner_b_eff(r1, z1);
        if (b0 > 0.0 && std::isfinite(b0) && std::isfinite(b1)) {
          out = std::min(out, b1 / b0);
        }
      }
    }
    return out;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0 || state.x_r_initial.size() != state.x_r.size() ||
      state.x_z_initial.size() != state.x_z.size()) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> r_old;
  std::vector<double> z_old;
  std::vector<double> r_new;
  std::vector<double> z_new;
  state.x_r_initial.copy_to_host(r_old);
  state.x_z_initial.copy_to_host(z_old);
  state.x_r.copy_to_host(r_new);
  state.x_z.copy_to_host(z_new);

  double out = std::numeric_limits<double>::infinity();
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int nodes[4] = {
          node_index(i, j, nz),
          node_index(i + 1, j, nz),
          node_index(i + 1, j + 1, nz),
          node_index(i, j + 1, nz),
      };
      double r0[4]{};
      double z0[4]{};
      double r1[4]{};
      double z1[4]{};
      for (int k = 0; k < 4; ++k) {
        r0[k] = r_old[nodes[k]];
        z0[k] = z_old[nodes[k]];
        r1[k] = r_new[nodes[k]];
        z1[k] = z_new[nodes[k]];
      }
      double j0[4]{};
      double j1[4]{};
      corner_jacobians(r0, z0, j0);
      corner_jacobians(r1, z1, j1);
      for (int k = 0; k < 4; ++k) {
        if (j0[k] != 0.0 && std::isfinite(j0[k]) && std::isfinite(j1[k])) {
          out = std::min(out, j1[k] / j0[k]);
        }
      }
      const double b0 = diagonal_corner_b_eff(r0, z0);
      const double b1 = diagonal_corner_b_eff(r1, z1);
      if (b0 > 0.0 && std::isfinite(b0) && std::isfinite(b1)) {
        out = std::min(out, b1 / b0);
      }
    }
  }
  return out;
}

struct QuadAdmissibilityTableValues {
  double planar_area;
  double rz_volume;
  double corner_j[4];
  double edge_center_area[4];
  double bbsw_corner_area[4];
  double tau_a;
  double tau_v;
  bool scales_valid;
};

__device__ QuadAdmissibilityTableValues evaluate_quad_admissibility_table(
    const double* r_old,
    const double* z_old,
    const double* r,
    const double* z) {
  QuadAdmissibilityTableValues values{};
  const double old_area2 = rz::rz_polygon_area2_exact(r_old, z_old, 4);
  const double orientation_sign = old_area2 > 0.0 ? 1.0 : -1.0;

  double length_scale = 0.0;
  double r_max = 0.0;
  for (int k = 0; k < 4; ++k) {
    length_scale = fmax(length_scale, fmax(fabs(r[k]), fabs(z[k])));
    r_max = fmax(r_max, fabs(r[k]));
  }
  values.tau_a = 64.0 * DBL_EPSILON * length_scale * length_scale;
  values.tau_v = values.tau_a * r_max;
  values.scales_valid = rz::finite_double(old_area2) && old_area2 != 0.0 &&
                        rz::finite_double(values.tau_a) &&
                        rz::finite_double(values.tau_v);

  values.planar_area =
      0.5 * orientation_sign * rz::rz_polygon_area2_exact(r, z, 4);
  values.rz_volume =
      orientation_sign * rz::rz_polygon_volume_exact(r, z, 4);

  corner_jacobians(r, z, values.corner_j);
  for (int k = 0; k < 4; ++k) {
    values.corner_j[k] *= orientation_sign;
  }

  const double centroid_r = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double centroid_z = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    values.edge_center_area[k] =
        0.5 * orientation_sign *
        cross2(r[kp1] - r[k], z[kp1] - z[k],
               centroid_r - r[k], centroid_z - z[k]);
  }

  // Contract (A124(b) census follow-up; ruling 2026-08-17): this compare
  // metric keeps the legacy bbsw-radial corner-area basis on BOTH sides of
  // the candidate-vs-baseline comparison — the basis choice cancels in the
  // metric's intended use, and re-basing would silently move recorded
  // rollback audit values. Deliberately NOT wired to
  // corner_mass_convention (same frozen shelf as the P-C CFL node mass).
  const double a12 = values.edge_center_area[0];
  const double a23 = values.edge_center_area[1];
  const double a34 = values.edge_center_area[2];
  const double a41 = values.edge_center_area[3];
  values.bbsw_corner_area[0] =
      (5.0 * a41 + 5.0 * a12 + a23 + a34) / 12.0;
  values.bbsw_corner_area[1] =
      (a41 + 5.0 * a12 + 5.0 * a23 + a34) / 12.0;
  values.bbsw_corner_area[2] =
      (a41 + a12 + 5.0 * a23 + 5.0 * a34) / 12.0;
  values.bbsw_corner_area[3] =
      (5.0 * a41 + a12 + a23 + 5.0 * a34) / 12.0;
  return values;
}

__device__ bool admissibility_quantity_passes_relative(
    const double candidate,
    const double baseline,
    const double tau) {
  // Immaterial relative drift on already-violated quantities passes; real
  // degradation still rejects; bounded cumulative drift is ~1e-6 per event.
  return rz::finite_double(candidate) &&
         ((candidate > -tau) ||
          (candidate >= baseline - fmax(tau, 1.0e-6 * fabs(baseline))));
}

__device__ bool quad_candidate_passes_admissibility_table(
    const QuadAdmissibilityTableValues& baseline,
    const QuadAdmissibilityTableValues& candidate) {
  if (!candidate.scales_valid ||
      !admissibility_quantity_passes_relative(
          candidate.planar_area, baseline.planar_area, candidate.tau_a) ||
      !admissibility_quantity_passes_relative(
          candidate.rz_volume, baseline.rz_volume, candidate.tau_v)) {
    return false;
  }
  // This lets the controller improve or work around pre-existing degenerate
  // slivers instead of deadlocking; crossing the band from healthy to
  // inadmissible still rejects.
  for (int k = 0; k < 4; ++k) {
    if (!admissibility_quantity_passes_relative(
            candidate.corner_j[k], baseline.corner_j[k], candidate.tau_a) ||
        !admissibility_quantity_passes_relative(
            candidate.edge_center_area[k], baseline.edge_center_area[k],
            candidate.tau_a) ||
        !admissibility_quantity_passes_relative(
            candidate.bbsw_corner_area[k], baseline.bbsw_corner_area[k],
            candidate.tau_a)) {
      return false;
    }
  }
  return true;
}

void log_runtime_ale_admissibility_dump(
    const int cell,
    const QuadAdmissibilityTableValues& baseline,
    const QuadAdmissibilityTableValues& candidate) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6)
      << "[runtime-ale-admiss-dump] cell=" << cell;
  const auto append_quantity = [&oss](const char* name,
                                      const double baseline_value,
                                      const double candidate_value,
                                      const double tau) {
    oss << " " << name << "=" << baseline_value << "/" << candidate_value
        << "/" << tau;
  };
  append_quantity("A_z", baseline.planar_area, candidate.planar_area,
                  candidate.tau_a);
  append_quantity("V_z", baseline.rz_volume, candidate.rz_volume,
                  candidate.tau_v);
  for (int k = 0; k < 4; ++k) {
    const std::string suffix = std::to_string(k);
    append_quantity(("J_c" + suffix).c_str(), baseline.corner_j[k],
                    candidate.corner_j[k], candidate.tau_a);
  }
  constexpr const char* edge_names[4] = {"A_12", "A_23", "A_34", "A_41"};
  for (int k = 0; k < 4; ++k) {
    append_quantity(edge_names[k], baseline.edge_center_area[k],
                    candidate.edge_center_area[k], candidate.tau_a);
  }
  for (int k = 0; k < 4; ++k) {
    const std::string suffix = std::to_string(k);
    append_quantity(("Abbsw" + suffix).c_str(), baseline.bbsw_corner_area[k],
                    candidate.bbsw_corner_area[k], candidate.tau_a);
  }
  core::log_info(oss.str());
}

bool admissibility_quantity_passes_relative_host(const double candidate,
                                                  const double baseline,
                                                  const double tau) {
  // The commit gate rejects catastrophic next-step worsening (healthy sign
  // flips or order-of-magnitude collapses), not sub-percent Lagrangian drift.
  return std::isfinite(candidate) &&
         ((candidate > -tau) ||
          (candidate >=
           baseline - std::max(tau, 0.10 * std::abs(baseline))));
}

struct CommitGateFailure {
  int cell = std::numeric_limits<int>::max();
  int corner = -1;
  const char* quantity = nullptr;
  double baseline = 0.0;
  double predicted = 0.0;
  double tau = 0.0;
  double dt_predict = 0.0;
};

struct RuntimeAleRollbackField {
  double* data = nullptr;
  std::size_t size = 0;
  bool captured = false;
};

template <typename Field>
RuntimeAleRollbackField capture_runtime_ale_device_field(
    const Field& field,
    const char* const tag,
    const char* const error_message) {
  RuntimeAleRollbackField snapshot;
  snapshot.size = field.size();
  snapshot.captured = true;
  if (snapshot.size == 0U) {
    return snapshot;
  }
  const std::size_t bytes = snapshot.size * sizeof(double);
  snapshot.data = static_cast<double*>(
      core::device_scratch_acquire(tag, bytes));
  cuda_check(cudaMemcpy(snapshot.data, field.data(), bytes,
                        cudaMemcpyDeviceToDevice),
             error_message);
  return snapshot;
}

RuntimeAleRollbackField capture_runtime_ale_host_field(
    const std::vector<double>& field,
    const char* const tag,
    const char* const error_message) {
  RuntimeAleRollbackField snapshot;
  snapshot.size = field.size();
  snapshot.captured = true;
  if (snapshot.size == 0U) {
    return snapshot;
  }
  const std::size_t bytes = snapshot.size * sizeof(double);
  snapshot.data = static_cast<double*>(
      core::device_scratch_acquire(tag, bytes));
  cuda_check(cudaMemcpy(snapshot.data, field.data(), bytes,
                        cudaMemcpyHostToDevice),
             error_message);
  return snapshot;
}

template <typename Field>
void restore_runtime_ale_device_field(
    const RuntimeAleRollbackField& snapshot,
    Field& field,
    const char* const error_message) {
  if (!snapshot.captured) {
    return;
  }
  if (field.size() != snapshot.size) {
    field.reset(snapshot.size);
  }
  if (snapshot.size == 0U) {
    return;
  }
  cuda_check(cudaMemcpy(field.data(), snapshot.data,
                        snapshot.size * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             error_message);
}

void restore_runtime_ale_host_field(
    const RuntimeAleRollbackField& snapshot,
    std::vector<double>& field,
    const char* const error_message) {
  if (!snapshot.captured) {
    return;
  }
  field.resize(snapshot.size);
  if (snapshot.size == 0U) {
    return;
  }
  cuda_check(cudaMemcpy(field.data(), snapshot.data,
                        snapshot.size * sizeof(double),
                        cudaMemcpyDeviceToHost),
             error_message);
}

struct RuntimeAleRollbackSnapshot {
  RuntimeAleRollbackField x_r;
  RuntimeAleRollbackField x_z;
  RuntimeAleRollbackField x_r_reference;
  RuntimeAleRollbackField x_z_reference;
  RuntimeAleRollbackField vol;
  RuntimeAleRollbackField cell_vol_initial;
  RuntimeAleRollbackField rho;
  RuntimeAleRollbackField mass;
  RuntimeAleRollbackField ee;
  RuntimeAleRollbackField ei;
  RuntimeAleRollbackField Te;
  RuntimeAleRollbackField Ti;
  RuntimeAleRollbackField Pe;
  RuntimeAleRollbackField Pi;
  RuntimeAleRollbackField zbar;
  RuntimeAleRollbackField Qvisc;
  RuntimeAleRollbackField v_r;
  RuntimeAleRollbackField v_z;
  RuntimeAleRollbackField hllc_mom_z_cell;
  RuntimeAleRollbackField volFrac;
  RuntimeAleRollbackField mass_per_material;
  RuntimeAleRollbackField Ee_per_material;
  RuntimeAleRollbackField Ei_per_material;
  RuntimeAleRollbackField rad_E;
  RuntimeAleRollbackField gas_tracer_Y;
  RuntimeAleRollbackField burn_Ng;
  RuntimeAleRollbackField burn_n_host;
  RuntimeAleRollbackField hot_e_eps_cum_host;
  RuntimeAleRollbackField burn_eps_cum_host;
  RuntimeAleRollbackField corner_mass;
  RuntimeAleRollbackField corner_volume;
  RuntimeAleRollbackField subzonal_mass_corner0;
  RuntimeAleRollbackField subzonal_mass_corner1;
  RuntimeAleRollbackField subzonal_mass_corner2;
  RuntimeAleRollbackField subzonal_mass_corner3;
  RuntimeAleRollbackField vol_prev_hydro;
  bool corner_mass_initialized = false;
  bool corner_mass_is_lagrangian_invariant = false;
  bool hllc_mom_z_cell_initialized = false;
  bool holo_ale_invalidated = false;
  int ale_remaps_applied = 0;
  bool captured = false;

  void capture(tenryu::core::State& state,
               const tenryu::core::Config& cfg,
               const bool is_multiblock,
               const int n_cells) {
    x_r = capture_runtime_ale_device_field(
        state.x_r, "runtime_ale:rollback:x_r",
        "runtime ALE rollback x_r snapshot failed");
    x_z = capture_runtime_ale_device_field(
        state.x_z, "runtime_ale:rollback:x_z",
        "runtime ALE rollback x_z snapshot failed");
    x_r_reference = capture_runtime_ale_device_field(
        state.x_r_reference, "runtime_ale:rollback:x_r_reference",
        "runtime ALE rollback x_r_reference snapshot failed");
    x_z_reference = capture_runtime_ale_device_field(
        state.x_z_reference, "runtime_ale:rollback:x_z_reference",
        "runtime ALE rollback x_z_reference snapshot failed");
    vol = capture_runtime_ale_device_field(
        state.vol, "runtime_ale:rollback:vol",
        "runtime ALE rollback vol snapshot failed");
    cell_vol_initial = capture_runtime_ale_device_field(
        state.cell_vol_initial, "runtime_ale:rollback:cell_vol_initial",
        "runtime ALE rollback cell_vol_initial snapshot failed");
    rho = capture_runtime_ale_device_field(
        state.rho, "runtime_ale:rollback:rho",
        "runtime ALE rollback rho snapshot failed");
    mass = capture_runtime_ale_device_field(
        state.mass, "runtime_ale:rollback:mass",
        "runtime ALE rollback mass snapshot failed");
    ee = capture_runtime_ale_device_field(
        state.ee, "runtime_ale:rollback:ee",
        "runtime ALE rollback ee snapshot failed");
    ei = capture_runtime_ale_device_field(
        state.ei, "runtime_ale:rollback:ei",
        "runtime ALE rollback ei snapshot failed");
    Te = capture_runtime_ale_device_field(
        state.Te, "runtime_ale:rollback:Te",
        "runtime ALE rollback Te snapshot failed");
    Ti = capture_runtime_ale_device_field(
        state.Ti, "runtime_ale:rollback:Ti",
        "runtime ALE rollback Ti snapshot failed");
    Pe = capture_runtime_ale_device_field(
        state.Pe, "runtime_ale:rollback:Pe",
        "runtime ALE rollback Pe snapshot failed");
    Pi = capture_runtime_ale_device_field(
        state.Pi, "runtime_ale:rollback:Pi",
        "runtime ALE rollback Pi snapshot failed");
    zbar = capture_runtime_ale_device_field(
        state.zbar, "runtime_ale:rollback:zbar",
        "runtime ALE rollback zbar snapshot failed");
    Qvisc = capture_runtime_ale_device_field(
        state.Qvisc, "runtime_ale:rollback:Qvisc",
        "runtime ALE rollback Qvisc snapshot failed");
    v_r = capture_runtime_ale_device_field(
        state.v_r, "runtime_ale:rollback:v_r",
        "runtime ALE rollback v_r snapshot failed");
    v_z = capture_runtime_ale_device_field(
        state.v_z, "runtime_ale:rollback:v_z",
        "runtime ALE rollback v_z snapshot failed");
    hllc_mom_z_cell = capture_runtime_ale_device_field(
        state.hllc_mom_z_cell, "runtime_ale:rollback:hllc_mom_z_cell",
        "runtime ALE rollback hllc_mom_z_cell snapshot failed");
    volFrac = capture_runtime_ale_device_field(
        state.volFrac, "runtime_ale:rollback:volFrac",
        "runtime ALE rollback volFrac snapshot failed");
    mass_per_material = capture_runtime_ale_device_field(
        state.mass_per_material, "runtime_ale:rollback:mass_per_material",
        "runtime ALE rollback mass_per_material snapshot failed");
    Ee_per_material = capture_runtime_ale_device_field(
        state.Ee_per_material, "runtime_ale:rollback:Ee_per_material",
        "runtime ALE rollback Ee_per_material snapshot failed");
    Ei_per_material = capture_runtime_ale_device_field(
        state.Ei_per_material, "runtime_ale:rollback:Ei_per_material",
        "runtime ALE rollback Ei_per_material snapshot failed");
    const int n_groups = std::max(cfg.radiation.groups, 1);
    const std::size_t expected_rad_size =
        static_cast<std::size_t>(n_cells) *
        static_cast<std::size_t>(n_groups);
    const bool remap_radiation =
        state.rad_E.size() == expected_rad_size &&
        (!is_multiblock ||
         cfg.numerics.ale.conservative_remap_radiation_enabled);
    if (remap_radiation) {
      rad_E = capture_runtime_ale_device_field(
          state.rad_E, "runtime_ale:rollback:rad_E",
          "runtime ALE rollback rad_E snapshot failed");
    }
    gas_tracer_Y = capture_runtime_ale_device_field(
        state.gas_tracer_Y, "runtime_ale:rollback:gas_tracer_Y",
        "runtime ALE rollback gas_tracer_Y snapshot failed");
    burn_Ng = capture_runtime_ale_device_field(
        state.burn_Ng, "runtime_ale:rollback:burn_Ng",
        "runtime ALE rollback burn_Ng snapshot failed");
    burn_n_host = capture_runtime_ale_host_field(
        state.burn_n_host, "runtime_ale:rollback:burn_n_host",
        "runtime ALE rollback burn_n_host snapshot failed");
    hot_e_eps_cum_host = capture_runtime_ale_host_field(
        state.hot_e_eps_cum_host,
        "runtime_ale:rollback:hot_e_eps_cum_host",
        "runtime ALE rollback hot_e_eps_cum_host snapshot failed");
    burn_eps_cum_host = capture_runtime_ale_host_field(
        state.burn_eps_cum_host,
        "runtime_ale:rollback:burn_eps_cum_host",
        "runtime ALE rollback burn_eps_cum_host snapshot failed");
    corner_mass = capture_runtime_ale_device_field(
        state.corner_mass, "runtime_ale:rollback:corner_mass",
        "runtime ALE rollback corner_mass snapshot failed");
    corner_volume = capture_runtime_ale_device_field(
        state.corner_volume, "runtime_ale:rollback:corner_volume",
        "runtime ALE rollback corner_volume snapshot failed");
    subzonal_mass_corner0 = capture_runtime_ale_device_field(
        state.subzonal_mass_corner0,
        "runtime_ale:rollback:subzonal_mass_corner0",
        "runtime ALE rollback subzonal_mass_corner0 snapshot failed");
    subzonal_mass_corner1 = capture_runtime_ale_device_field(
        state.subzonal_mass_corner1,
        "runtime_ale:rollback:subzonal_mass_corner1",
        "runtime ALE rollback subzonal_mass_corner1 snapshot failed");
    subzonal_mass_corner2 = capture_runtime_ale_device_field(
        state.subzonal_mass_corner2,
        "runtime_ale:rollback:subzonal_mass_corner2",
        "runtime ALE rollback subzonal_mass_corner2 snapshot failed");
    subzonal_mass_corner3 = capture_runtime_ale_device_field(
        state.subzonal_mass_corner3,
        "runtime_ale:rollback:subzonal_mass_corner3",
        "runtime ALE rollback subzonal_mass_corner3 snapshot failed");
    vol_prev_hydro = capture_runtime_ale_device_field(
        state.vol_prev_hydro, "runtime_ale:rollback:vol_prev_hydro",
        "runtime ALE rollback vol_prev_hydro snapshot failed");
    corner_mass_initialized = state.corner_mass_initialized;
    corner_mass_is_lagrangian_invariant =
        state.corner_mass_is_lagrangian_invariant;
    hllc_mom_z_cell_initialized = state.hllc_mom_z_cell_initialized;
    holo_ale_invalidated = state.holo_ale_invalidated;
    ale_remaps_applied = state.ale_remaps_applied;
    captured = true;
  }

  void restore(tenryu::core::State& state) const {
    TENRYU_ASSERT(captured, "runtime ALE rollback snapshot was not captured");
    restore_runtime_ale_device_field(
        x_r, state.x_r, "runtime ALE rollback x_r restore failed");
    restore_runtime_ale_device_field(
        x_z, state.x_z, "runtime ALE rollback x_z restore failed");
    restore_runtime_ale_device_field(
        x_r_reference, state.x_r_reference,
        "runtime ALE rollback x_r_reference restore failed");
    restore_runtime_ale_device_field(
        x_z_reference, state.x_z_reference,
        "runtime ALE rollback x_z_reference restore failed");
    restore_runtime_ale_device_field(
        vol, state.vol, "runtime ALE rollback vol restore failed");
    restore_runtime_ale_device_field(
        cell_vol_initial, state.cell_vol_initial,
        "runtime ALE rollback cell_vol_initial restore failed");
    restore_runtime_ale_device_field(
        rho, state.rho, "runtime ALE rollback rho restore failed");
    restore_runtime_ale_device_field(
        mass, state.mass, "runtime ALE rollback mass restore failed");
    restore_runtime_ale_device_field(
        ee, state.ee, "runtime ALE rollback ee restore failed");
    restore_runtime_ale_device_field(
        ei, state.ei, "runtime ALE rollback ei restore failed");
    restore_runtime_ale_device_field(
        Te, state.Te, "runtime ALE rollback Te restore failed");
    restore_runtime_ale_device_field(
        Ti, state.Ti, "runtime ALE rollback Ti restore failed");
    restore_runtime_ale_device_field(
        Pe, state.Pe, "runtime ALE rollback Pe restore failed");
    restore_runtime_ale_device_field(
        Pi, state.Pi, "runtime ALE rollback Pi restore failed");
    restore_runtime_ale_device_field(
        zbar, state.zbar, "runtime ALE rollback zbar restore failed");
    restore_runtime_ale_device_field(
        Qvisc, state.Qvisc, "runtime ALE rollback Qvisc restore failed");
    restore_runtime_ale_device_field(
        v_r, state.v_r, "runtime ALE rollback v_r restore failed");
    restore_runtime_ale_device_field(
        v_z, state.v_z, "runtime ALE rollback v_z restore failed");
    restore_runtime_ale_device_field(
        hllc_mom_z_cell, state.hllc_mom_z_cell,
        "runtime ALE rollback hllc_mom_z_cell restore failed");
    restore_runtime_ale_device_field(
        volFrac, state.volFrac,
        "runtime ALE rollback volFrac restore failed");
    restore_runtime_ale_device_field(
        mass_per_material, state.mass_per_material,
        "runtime ALE rollback mass_per_material restore failed");
    restore_runtime_ale_device_field(
        Ee_per_material, state.Ee_per_material,
        "runtime ALE rollback Ee_per_material restore failed");
    restore_runtime_ale_device_field(
        Ei_per_material, state.Ei_per_material,
        "runtime ALE rollback Ei_per_material restore failed");
    restore_runtime_ale_device_field(
        rad_E, state.rad_E, "runtime ALE rollback rad_E restore failed");
    restore_runtime_ale_device_field(
        gas_tracer_Y, state.gas_tracer_Y,
        "runtime ALE rollback gas_tracer_Y restore failed");
    restore_runtime_ale_device_field(
        burn_Ng, state.burn_Ng,
        "runtime ALE rollback burn_Ng restore failed");
    restore_runtime_ale_host_field(
        burn_n_host, state.burn_n_host,
        "runtime ALE rollback burn_n_host restore failed");
    restore_runtime_ale_host_field(
        hot_e_eps_cum_host, state.hot_e_eps_cum_host,
        "runtime ALE rollback hot_e_eps_cum_host restore failed");
    restore_runtime_ale_host_field(
        burn_eps_cum_host, state.burn_eps_cum_host,
        "runtime ALE rollback burn_eps_cum_host restore failed");
    restore_runtime_ale_device_field(
        corner_mass, state.corner_mass,
        "runtime ALE rollback corner_mass restore failed");
    restore_runtime_ale_device_field(
        corner_volume, state.corner_volume,
        "runtime ALE rollback corner_volume restore failed");
    restore_runtime_ale_device_field(
        subzonal_mass_corner0, state.subzonal_mass_corner0,
        "runtime ALE rollback subzonal_mass_corner0 restore failed");
    restore_runtime_ale_device_field(
        subzonal_mass_corner1, state.subzonal_mass_corner1,
        "runtime ALE rollback subzonal_mass_corner1 restore failed");
    restore_runtime_ale_device_field(
        subzonal_mass_corner2, state.subzonal_mass_corner2,
        "runtime ALE rollback subzonal_mass_corner2 restore failed");
    restore_runtime_ale_device_field(
        subzonal_mass_corner3, state.subzonal_mass_corner3,
        "runtime ALE rollback subzonal_mass_corner3 restore failed");
    restore_runtime_ale_device_field(
        vol_prev_hydro, state.vol_prev_hydro,
        "runtime ALE rollback vol_prev_hydro restore failed");
    state.corner_mass_initialized = corner_mass_initialized;
    state.corner_mass_is_lagrangian_invariant =
        corner_mass_is_lagrangian_invariant;
    state.hllc_mom_z_cell_initialized = hllc_mom_z_cell_initialized;
    state.holo_ale_invalidated = holo_ale_invalidated;
    state.ale_remaps_applied = ale_remaps_applied;
  }
};

int runtime_cap_boundary_node_id(
    const button_morph::ButtonIndexing& idx,
    const int k) {
  if (k <= idx.n_c) {
    return button_morph::core_node_id(idx, k, 2 * idx.n_c);
  }
  if (k <= 3 * idx.n_c) {
    return button_morph::core_node_id(idx, idx.n_c, 3 * idx.n_c - k);
  }
  return button_morph::core_node_id(idx, 4 * idx.n_c - k, 0);
}

int runtime_cap_chain_node_id(const button_morph::ButtonIndexing& idx,
                              const int row,
                              const int k) {
  if (row == 0) {
    return runtime_cap_boundary_node_id(idx, k);
  }
  if (row < idx.n_b) {
    return button_morph::bridge_interior_node_id(idx, row, k);
  }
  return button_morph::shell_node_id(idx, row - idx.n_b, k);
}

int runtime_cap_chain_cell_id(const button_morph::ButtonIndexing& idx,
                              const int row,
                              const int column) {
  if (row < idx.n_b) {
    return idx.bridge_cell_offset + row * idx.ntheta + column;
  }
  return idx.shell_cell_offset + (row - idx.n_b) * idx.ntheta + column;
}

void note_commit_gate_failure(const int cell,
                              const int corner,
                              const char* const quantity,
                              const double baseline,
                              const double predicted,
                              const double tau,
                              CommitGateFailure* const failure) {
  if (cell >= failure->cell) {
    return;
  }
  failure->cell = cell;
  failure->corner = corner;
  failure->quantity = quantity;
  failure->baseline = baseline;
  failure->predicted = predicted;
  failure->tau = tau;
}

CommitGateFailure evaluate_runtime_ale_commit_gate(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const ReferenceBarrierScope& scope) {
  CommitGateFailure failure;
  if (!scope.runtime_ale || !state.mesh.topo.multiblock.has_value()) {
    return failure;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "runtime ALE commit gate requires CSR offsets");
  TENRYU_ASSERT(state.v_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.v_z.size() == static_cast<std::size_t>(n_nodes),
                "runtime ALE commit gate velocity size mismatch");

  double dt_predict = state.dt;
  if (!(std::isfinite(dt_predict) && dt_predict > 0.0)) {
    dt_predict = cfg.numerics.dt.max_s;
  }
  if (!(std::isfinite(dt_predict) && dt_predict > 0.0)) {
    return failure;
  }

  std::vector<double> current_r;
  std::vector<double> current_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  state.x_r.copy_to_host(current_r);
  state.x_z.copy_to_host(current_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  std::vector<double> predicted_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> predicted_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t n = static_cast<std::size_t>(node);
    predicted_r[n] = current_r[n] + dt_predict * velocity_r[n];
    predicted_z[n] = current_z[n] + dt_predict * velocity_z[n];
  }

  std::vector<std::uint8_t> active_cell(
      static_cast<std::size_t>(n_cells), 1U);
  if (scope.d_active_cell_mask != nullptr) {
    cuda_check(cudaMemcpy(active_cell.data(), scope.d_active_cell_mask,
                          active_cell.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "runtime ALE commit gate active-cell copy failed");
  }

  for (int cell = 0; cell < n_cells; ++cell) {
    if (active_cell[static_cast<std::size_t>(cell)] == 0U) {
      continue;
    }
    const int begin = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = state_cell_active_nverts(state, cell, n_cells);
    if (nverts != 3 && nverts != 4) {
      continue;
    }
    double r_current[4]{};
    double z_current[4]{};
    double r_predicted[4]{};
    double z_predicted[4]{};
    double length_scale = 0.0;
    double r_max = 0.0;
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + corner)];
      const std::size_t n = static_cast<std::size_t>(node);
      r_current[corner] = current_r[n];
      z_current[corner] = current_z[n];
      r_predicted[corner] = predicted_r[n];
      z_predicted[corner] = predicted_z[n];
      length_scale =
          std::max(length_scale,
                   std::max(std::abs(r_predicted[corner]),
                            std::abs(z_predicted[corner])));
      r_max = std::max(r_max, std::abs(r_predicted[corner]));
    }
    const double area2_current =
        rz::rz_polygon_area2_exact(r_current, z_current, nverts);
    const double orientation = area2_current > 0.0 ? 1.0 : -1.0;
    const double tau_a = 64.0 * std::numeric_limits<double>::epsilon() *
                         length_scale * length_scale;
    const double tau_v = tau_a * r_max;

    double j_current[4]{};
    double j_predicted[4]{};
    if (nverts == 3) {
      triangle_corner_jacobians(r_current, z_current, j_current);
      triangle_corner_jacobians(r_predicted, z_predicted, j_predicted);
    } else {
      corner_jacobians(r_current, z_current, j_current);
      corner_jacobians(r_predicted, z_predicted, j_predicted);
    }
    for (int corner = 0; corner < nverts; ++corner) {
      const double baseline = orientation * j_current[corner];
      const double predicted = orientation * j_predicted[corner];
      if (!admissibility_quantity_passes_relative_host(
              predicted, baseline, tau_a)) {
        note_commit_gate_failure(cell, corner, "corner_J", baseline,
                                 predicted, tau_a, &failure);
        break;
      }
    }

    const double volume_current =
        orientation *
        rz::rz_polygon_volume_exact(r_current, z_current, nverts);
    const double volume_predicted =
        orientation *
        rz::rz_polygon_volume_exact(r_predicted, z_predicted, nverts);
    if (!admissibility_quantity_passes_relative_host(
            volume_predicted, volume_current, tau_v)) {
      note_commit_gate_failure(cell, -1, "V_z", volume_current,
                               volume_predicted, tau_v, &failure);
    }
  }

  const button_morph::ButtonIndexing idx =
      button_morph::make_button_indexing(cfg.mesh);
  const int radial_intervals =
      idx.n_b + cfg.numerics.ale.runtime_controller.controller_shell_rows;
  for (int axis = 0; axis < 2; ++axis) {
    for (int ray = 0; ray <= std::min(1, idx.ntheta); ++ray) {
      const int k = axis == 0 ? ray : idx.ntheta - ray;
      const int column = axis == 0 ? 0 : idx.ntheta - 1;
      const double theta = tenryu::mesh::multiblock_theta_node(
          k, idx.ntheta, cfg.mesh.multiblock_theta_cap_widen_factor);
      const double ray_r = std::sin(theta);
      const double ray_z = std::cos(theta);
      for (int row = 0; row < radial_intervals; ++row) {
        const int cell = runtime_cap_chain_cell_id(idx, row, column);
        if (cell < 0 || cell >= n_cells ||
            active_cell[static_cast<std::size_t>(cell)] == 0U) {
          continue;
        }
        const int inner_node = runtime_cap_chain_node_id(idx, row, k);
        const int outer_node = runtime_cap_chain_node_id(idx, row + 1, k);
        const std::size_t inner = static_cast<std::size_t>(inner_node);
        const std::size_t outer = static_cast<std::size_t>(outer_node);
        const double gap_current =
            current_r[outer] * ray_r + current_z[outer] * ray_z -
            current_r[inner] * ray_r - current_z[inner] * ray_z;
        const double gap_predicted =
            predicted_r[outer] * ray_r + predicted_z[outer] * ray_z -
            predicted_r[inner] * ray_r - predicted_z[inner] * ray_z;
        const double length_scale =
            std::max({std::abs(predicted_r[inner]),
                      std::abs(predicted_z[inner]),
                      std::abs(predicted_r[outer]),
                      std::abs(predicted_z[outer])});
        const double tau_g = 64.0 * std::numeric_limits<double>::epsilon() *
                             length_scale;
        if (!admissibility_quantity_passes_relative_host(
                gap_predicted, gap_current, tau_g)) {
          note_commit_gate_failure(cell, -1, "radial_gap", gap_current,
                                   gap_predicted, tau_g, &failure);
        }
      }
    }
  }

  if (failure.quantity == nullptr) {
    return failure;
  }
  failure.dt_predict = dt_predict;
  return failure;
}

void log_runtime_ale_commit_rollback(
    const CommitGateFailure& failure) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6)
      << "[runtime-ale] commit-rollback cell=" << failure.cell
      << " quantity=" << failure.quantity;
  if (failure.corner >= 0) {
    oss << " corner=" << failure.corner;
  }
  oss << " current=" << failure.baseline
      << " predicted=" << failure.predicted << " tau=" << failure.tau
      << " dt=" << failure.dt_predict;
  core::log_warning(oss.str());
}

__global__ void evaluate_structured_candidate_admissibility_table_kernel(
    const double* __restrict__ node_r_old,
    const double* __restrict__ node_z_old,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const double sigma,
    const int nr,
    const int nz,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ active_cell_mask,
    const bool button_cell_zero,
    const bool write_baseline,
    QuadAdmissibilityTableValues* __restrict__ baseline,
    int* __restrict__ smallest_bad_cell,
    const int diagnostic_cell,
    QuadAdmissibilityTableValues* __restrict__ diagnostic_values) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if ((active_cell_mask != nullptr && active_cell_mask[c] == 0U) ||
      (button_cell_zero && c == 0) ||
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c) == 3) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const double r_old[4] = {
      node_r_old[n00],
      node_r_old[n10],
      node_r_old[n11],
      node_r_old[n01],
  };
  const double z_old[4] = {
      node_z_old[n00],
      node_z_old[n10],
      node_z_old[n11],
      node_z_old[n01],
  };
  const double r[4] = {
      r_old[0] + sigma * delta_r[n00],
      r_old[1] + sigma * delta_r[n10],
      r_old[2] + sigma * delta_r[n11],
      r_old[3] + sigma * delta_r[n01],
  };
  const double z[4] = {
      z_old[0] + sigma * delta_z[n00],
      z_old[1] + sigma * delta_z[n10],
      z_old[2] + sigma * delta_z[n11],
      z_old[3] + sigma * delta_z[n01],
  };
  const QuadAdmissibilityTableValues values =
      write_baseline
          ? evaluate_quad_admissibility_table(r_old, z_old, r_old, z_old)
          : evaluate_quad_admissibility_table(r_old, z_old, r, z);
  if (diagnostic_values != nullptr && c == diagnostic_cell) {
    *diagnostic_values = values;
  }
  if (write_baseline) {
    baseline[c] = values;
    return;
  }
  if (smallest_bad_cell != nullptr &&
      !quad_candidate_passes_admissibility_table(baseline[c], values)) {
    atomicMin(smallest_bad_cell, c);
  }
}

__global__ void evaluate_csr_candidate_admissibility_table_kernel(
    const double* __restrict__ node_r_old,
    const double* __restrict__ node_z_old,
    const double* __restrict__ delta_r,
    const double* __restrict__ delta_z,
    const double sigma,
    const int n_cells,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::uint8_t* __restrict__ active_cell_mask,
    const bool button_cell_zero,
    const bool write_baseline,
    QuadAdmissibilityTableValues* __restrict__ baseline,
    int* __restrict__ smallest_bad_cell,
    const int diagnostic_cell,
    QuadAdmissibilityTableValues* __restrict__ diagnostic_values) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if ((active_cell_mask != nullptr && active_cell_mask[c] == 0U) ||
      (button_cell_zero && c == 0) ||
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c) == 3) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int n00 = cell_node_csr_indices[off + 0];
  const int n10 = cell_node_csr_indices[off + 1];
  const int n11 = cell_node_csr_indices[off + 2];
  const int n01 = cell_node_csr_indices[off + 3];
  const double r_old[4] = {
      node_r_old[n00],
      node_r_old[n10],
      node_r_old[n11],
      node_r_old[n01],
  };
  const double z_old[4] = {
      node_z_old[n00],
      node_z_old[n10],
      node_z_old[n11],
      node_z_old[n01],
  };
  const double r[4] = {
      r_old[0] + sigma * delta_r[n00],
      r_old[1] + sigma * delta_r[n10],
      r_old[2] + sigma * delta_r[n11],
      r_old[3] + sigma * delta_r[n01],
  };
  const double z[4] = {
      z_old[0] + sigma * delta_z[n00],
      z_old[1] + sigma * delta_z[n10],
      z_old[2] + sigma * delta_z[n11],
      z_old[3] + sigma * delta_z[n01],
  };
  const QuadAdmissibilityTableValues values =
      write_baseline
          ? evaluate_quad_admissibility_table(r_old, z_old, r_old, z_old)
          : evaluate_quad_admissibility_table(r_old, z_old, r, z);
  if (diagnostic_values != nullptr && c == diagnostic_cell) {
    *diagnostic_values = values;
  }
  if (write_baseline) {
    baseline[c] = values;
    return;
  }
  if (smallest_bad_cell != nullptr &&
      !quad_candidate_passes_admissibility_table(baseline[c], values)) {
    atomicMin(smallest_bad_cell, c);
  }
}

bool candidate_rejection_is_nonpositive_corner_j(
    const tenryu::mesh::CandidateMeshQuality& quality) {
  return quality.kind ==
             tenryu::mesh::MeshGeometryFailureKind::NonPositiveCellArea &&
         quality.first_bad_cell >= 0 && quality.first_bad_corner >= 0;
}

bool candidate_rejection_is_table_covered_corner_j(
    const tenryu::core::State& state,
    const std::uint8_t* d_active_cell_mask,
    const bool button_cell_zero,
    const tenryu::mesh::CandidateMeshQuality& quality) {
  if (!candidate_rejection_is_nonpositive_corner_j(quality)) {
    return false;
  }
  const int cell = quality.first_bad_cell;
  const int n_cells = state.mesh.topo.multiblock.has_value()
                          ? state.mesh.topo.n_cells
                          : state.mesh.topo.nr * state.mesh.topo.nz;
  if (cell >= n_cells || (button_cell_zero && cell == 0) ||
      state_cell_active_nverts(state, cell, n_cells) != 4) {
    return false;
  }
  if (d_active_cell_mask != nullptr) {
    std::uint8_t active = 0U;
    cuda_check(cudaMemcpy(&active, d_active_cell_mask + cell,
                          sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
               "reference barrier active-cell mask copy failed");
    return active != 0U;
  }
  return true;
}

__global__ void build_spherical_equal_angle_kernel(
    const double* __restrict__ r0,
    const double* __restrict__ z0,
    double* __restrict__ target_r,
    double* __restrict__ target_z,
    const int nr,
    const int nz) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const int i = n / (nz + 1);
  const int j = n - i * (nz + 1);
  const int shell = i * (nz + 1);
  const double s = hypot(r0[shell], z0[shell]);
  const double theta = (nz > 0) ? kPi * static_cast<double>(j) / static_cast<double>(nz)
                                : 0.0;
  target_r[n] = s * sin(theta);
  target_z[n] = s * cos(theta);
}

__global__ void delta_kernel(const double* __restrict__ x_r,
                             const double* __restrict__ x_z,
                             const double* __restrict__ target_r,
                             const double* __restrict__ target_z,
                             double* __restrict__ delta_r,
                             double* __restrict__ delta_z,
                             const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  delta_r[n] = target_r[n] - x_r[n];
  delta_z[n] = target_z[n] - x_z[n];
}

__global__ void blend_kernel(double* __restrict__ x_r,
                             double* __restrict__ x_z,
                             const double* __restrict__ delta_r,
                             const double* __restrict__ delta_z,
                             const double sigma,
                             const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  x_r[n] += sigma * delta_r[n];
  x_z[n] += sigma * delta_z[n];
}

__global__ void blend_to_output_kernel(double* __restrict__ out_r,
                                       double* __restrict__ out_z,
                                       const double* __restrict__ x_r_old,
                                       const double* __restrict__ x_z_old,
                                       const double* __restrict__ delta_r,
                                       const double* __restrict__ delta_z,
                                       const double sigma,
                                       const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  out_r[n] = x_r_old[n] + sigma * delta_r[n];
  out_z[n] = x_z_old[n] + sigma * delta_z[n];
}

__global__ void pack_cell_conserved_kernel(double* __restrict__ mom_r,
                                           double* __restrict__ mom_z,
                                           double* __restrict__ e_e,
                                           double* __restrict__ e_i,
                                           double* __restrict__ v_r_cell,
                                           double* __restrict__ v_z_cell,
                                           const double* __restrict__ rho,
                                           const double* __restrict__ ee,
                                           const double* __restrict__ ei,
                                           const double* __restrict__ v_r_node,
                                           const double* __restrict__ v_z_node,
                                           const int nr,
                                           const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const double vr = 0.25 * (v_r_node[n00] + v_r_node[n10] + v_r_node[n11] + v_r_node[n01]);
  const double vz = 0.25 * (v_z_node[n00] + v_z_node[n10] + v_z_node[n11] + v_z_node[n01]);
  const double rho_c = fmax(rho[c], 0.0);
  v_r_cell[c] = vr;
  v_z_cell[c] = vz;
  mom_r[c] = rho_c * vr;
  mom_z[c] = rho_c * vz;
  e_e[c] = rho_c * fmax(ee[c], 0.0);
  e_i[c] = rho_c * fmax(ei[c], 0.0);
}

__global__ void recover_cell_primitives_kernel(double* __restrict__ rho,
                                               double* __restrict__ mass,
                                               double* __restrict__ ee,
                                               double* __restrict__ ei,
                                               double* __restrict__ v_r_cell,
                                               double* __restrict__ v_z_cell,
                                               const double* __restrict__ mom_r,
                                               const double* __restrict__ mom_z,
                                               const double* __restrict__ e_e,
                                               const double* __restrict__ e_i,
                                               const double* __restrict__ vol,
                                               const double rho_floor,
                                               const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const double V = fmax(vol[c], 1.0e-30);
  const double rho_c = fmax(rho[c], rho_floor);
  rho[c] = rho_c;
  mass[c] = rho_c * V;
  if (rho_c > 0.0) {
    v_r_cell[c] = mom_r[c] / rho_c;
    v_z_cell[c] = mom_z[c] / rho_c;
    ee[c] = fmax(e_e[c], 0.0) / rho_c;
    ei[c] = fmax(e_i[c], 0.0) / rho_c;
  } else {
    v_r_cell[c] = 0.0;
    v_z_cell[c] = 0.0;
    ee[c] = 0.0;
    ei[c] = 0.0;
  }
}

__global__ void gather_material_field_kernel(double* __restrict__ out,
                                             const double* __restrict__ in,
                                             const int n_cells,
                                             const int n_mat,
                                             const int mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_mat + mat];
}

__global__ void scatter_material_field_kernel(double* __restrict__ out,
                                              const double* __restrict__ in,
                                              const int n_cells,
                                              const int n_mat,
                                              const int mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_mat + mat] = fmax(in[c], 0.0);
}

__global__ void gather_group_field_kernel(double* __restrict__ out,
                                          const double* __restrict__ in,
                                          const int n_cells,
                                          const int n_groups,
                                          const int group) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c] = in[c * n_groups + group];
}

__global__ void scatter_group_field_kernel(double* __restrict__ out,
                                           const double* __restrict__ in,
                                           const int n_cells,
                                           const int n_groups,
                                           const int group) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  out[c * n_groups + group] = fmax(in[c], 0.0);
}

__global__ void normalize_reference_volfrac_kernel(double* __restrict__ volfrac,
                                                   const int n_cells,
                                                   const int n_mat) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || n_mat <= 0) {
    return;
  }
  if (n_mat == 1) {
    volfrac[c * n_mat] = 1.0;
    return;
  }
  double sum = 0.0;
  int imax = 0;
  double vmax = volfrac[c * n_mat];
  for (int m = 0; m < n_mat; ++m) {
    const int idx = c * n_mat + m;
    const double clamped = fmax(volfrac[idx], 0.0);
    volfrac[idx] = clamped;
    sum += clamped;
    if (clamped > vmax) {
      vmax = clamped;
      imax = m;
    }
  }
  if (sum > 1.0e-30) {
    for (int m = 0; m < n_mat; ++m) {
      volfrac[c * n_mat + m] /= sum;
    }
  } else {
    for (int m = 0; m < n_mat; ++m) {
      volfrac[c * n_mat + m] = (m == imax) ? 1.0 : 0.0;
    }
  }
}

__global__ void recache_structured_corner_mass_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ cell_nverts,
    const int nr,
    const int nz,
    const int button_outer_node_ring) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (button_outer_node_ring >= 1 && c == 0) {
    corner_mass[0] = 0.0;
    corner_mass[1] = 0.0;
    corner_mass[2] = 0.0;
    corner_mass[3] = 0.0;
    return;
  }
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  rz::compute_rz_corner_masses_from_nodes(
      c, nz, mass[c], x_r, x_z, cell_nverts, m_corner);
  // Clamped partitions lose exact closure; renormalization restores the hard
  // invariant sum==m_z while keeping positivity; the distribution quality
  // degrades gracefully.
  const double s =
      m_corner[0] + m_corner[1] + m_corner[2] + m_corner[3];
  if (s > 0.0 && isfinite(s)) {
    const double scale = mass[c] / s;
    m_corner[0] *= scale;
    m_corner[1] *= scale;
    m_corner[2] *= scale;
    m_corner[3] *= scale;
  } else {
    m_corner[0] = 0.25 * mass[c];
    m_corner[1] = 0.25 * mass[c];
    m_corner[2] = 0.25 * mass[c];
    m_corner[3] = 0.25 * mass[c];
  }
  corner_mass[c * 4 + 0] = m_corner[0];
  corner_mass[c * 4 + 1] = m_corner[1];
  corner_mass[c * 4 + 2] = m_corner[2];
  corner_mass[c * 4 + 3] = m_corner[3];
}

__global__ void recache_multiblock_corner_mass_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int n_cells,
    const bool partition_normalized) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    rz::compute_triangle_corner_masses_exact(
        mass[c], x_r[n0], x_z[n0], x_r[n1], x_z[n1], x_r[n2], x_z[n2],
        m_corner);
  } else {
    const int n00 = cell_node_csr_indices[off + 0];
    const int n10 = cell_node_csr_indices[off + 1];
    const int n11 = cell_node_csr_indices[off + 2];
    const int n01 = cell_node_csr_indices[off + 3];
    // Mirror compute_corner_mass_2d_multiblock_kernel in hydro_2d.cu.
    if (partition_normalized) {
      rz::compute_quad_corner_masses_partitioned_subpolygon(
          mass[c], x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11],
          x_z[n11], x_r[n01], x_z[n01], m_corner);
    } else {
      rz::compute_quad_corner_masses_exact_subpolygon(
          mass[c], x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11],
          x_z[n11], x_r[n01], x_z[n01], m_corner);
    }
  }
  // Clamped partitions lose exact closure; renormalization restores the hard
  // invariant sum==m_z while keeping positivity; the distribution quality
  // degrades gracefully.
  const double s =
      m_corner[0] + m_corner[1] + m_corner[2] + m_corner[3];
  if (s > 0.0 && isfinite(s)) {
    const double scale = mass[c] / s;
    m_corner[0] *= scale;
    m_corner[1] *= scale;
    m_corner[2] *= scale;
    m_corner[3] *= scale;
  } else {
    m_corner[0] = 0.25 * mass[c];
    m_corner[1] = 0.25 * mass[c];
    m_corner[2] = 0.25 * mass[c];
    m_corner[3] = 0.25 * mass[c];
  }
  corner_mass[c * 4 + 0] = m_corner[0];
  corner_mass[c * 4 + 1] = m_corner[1];
  corner_mass[c * 4 + 2] = m_corner[2];
  corner_mass[c * 4 + 3] = m_corner[3];
}

void recache_corner_mass_after_reference_remap(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const std::uint8_t* d_cell_nverts) {
  const int n_cells = static_cast<int>(state.mass.size());
  const std::size_t expected_size = static_cast<std::size_t>(n_cells) * 4U;
  if (!state.corner_mass_initialized) {
    // No compatible corner-mass cache exists on this state (minimal/unit-test
    // states, or pre-first-hydro-step): the epoch reset is vacuous — any later
    // cache build starts from the post-remap geometry.
    return;
  }
  TENRYU_ASSERT(state.corner_mass.size() == expected_size,
                "reference barrier corner-mass recache size mismatch");
  if (n_cells == 0) {
    return;
  }

  const int blocks_cells = (n_cells + 255) / 256;
  // Recompute with the init-time formula dispatch before enforcing closure.
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "reference barrier corner-mass recache requires CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      expected_size,
                  "reference barrier corner-mass recache requires CSR indices");
    recache_multiblock_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(), d_cell_nverts,
        n_cells, corner_mass_lagrangian_invariant_enabled(cfg));
  } else {
    const int button_outer_node_ring =
        (state.mesh.button_center && state.mesh.button_center->enabled)
            ? state.mesh.button_center->outer_node_ring
            : -1;
    recache_structured_corner_mass_kernel<<<blocks_cells, 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), d_cell_nverts, state.mesh.topo.nr,
        state.mesh.topo.nz, button_outer_node_ring);
  }
  cuda_check(cudaGetLastError(),
             "reference barrier corner-mass recache kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier corner-mass recache synchronize failed");
}

__global__ void project_reference_cell_velocity_to_nodes_kernel(
    double* __restrict__ v_r_node,
    double* __restrict__ v_z_node,
    const double* __restrict__ v_r_cell,
    const double* __restrict__ v_z_cell,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const std::uint8_t* __restrict__ node_flags,
    const int nr,
    const int nz,
    const int r_outer_bc_mode,
    const int z_bottom_bc_mode,
    const int z_top_bc_mode) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_nodes) {
    return;
  }
  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;
  double w_sum = 0.0;
  double vr_sum = 0.0;
  double vz_sum = 0.0;
  for (int di = -1; di <= 0; ++di) {
    for (int dj = -1; dj <= 0; ++dj) {
      const int ic = i + di;
      const int jc = j + dj;
      if (ic < 0 || ic >= nr || jc < 0 || jc >= nz) {
        continue;
      }
      const int c = ic * nz + jc;
      const double w = fmax(rho[c] * vol[c], 0.0);
      w_sum += w;
      vr_sum += w * v_r_cell[c];
      vz_sum += w * v_z_cell[c];
    }
  }
  if (w_sum > 0.0) {
    v_r_node[n] = vr_sum / w_sum;
    v_z_node[n] = vz_sum / w_sum;
  } else {
    v_r_node[n] = 0.0;
    v_z_node[n] = 0.0;
  }
  pole_axis::apply_2d_boundary_vector_constraints(
      v_r_node[n],
      v_z_node[n],
      node_flags,
      n,
      i,
      j,
      nr,
      nz,
      r_outer_bc_mode,
      z_bottom_bc_mode,
      z_top_bc_mode,
      false);
}

double relative_residual(const double before, const double after) {
  const double denom = std::max(std::abs(before), 1.0);
  return std::abs(after - before) / denom;
}

struct ConservedSums {
  double mass = 0.0;
  double internal = 0.0;
  double kinetic = 0.0;
};

ConservedSums compute_conserved_sums_host(const tenryu::core::State& state) {
  std::vector<double> rho;
  std::vector<double> mass_field;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> vol;
  std::vector<double> vr;
  std::vector<double> vz;
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass_field);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.vol.copy_to_host(vol);
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);

  ConservedSums sums;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    const int n_cells = state.mesh.topo.n_cells;
    for (int c = 0; c < n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int active_nverts = state_cell_active_nverts(state, c, n_cells);
      double vr_cell = 0.0;
      double vz_cell = 0.0;
      for (int k = 0; k < active_nverts; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        vr_cell += vr[static_cast<std::size_t>(n)];
        vz_cell += vz[static_cast<std::size_t>(n)];
      }
      if (active_nverts == 3) {
        vr_cell *= (1.0 / 3.0);
        vz_cell *= (1.0 / 3.0);
      } else {
        vr_cell *= 0.25;
        vz_cell *= 0.25;
      }
      const double mass =
          mass_field.empty()
              ? rho[static_cast<std::size_t>(c)] * vol[static_cast<std::size_t>(c)]
              : mass_field[static_cast<std::size_t>(c)];
      sums.mass += mass;
      sums.internal += mass * (ee[static_cast<std::size_t>(c)] +
                               ei[static_cast<std::size_t>(c)]);
      sums.kinetic += 0.5 * mass * (vr_cell * vr_cell + vz_cell * vz_cell);
    }
    return sums;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const int n00 = node_index(i, j, nz);
      const int n10 = node_index(i + 1, j, nz);
      const int n11 = node_index(i + 1, j + 1, nz);
      const int n01 = node_index(i, j + 1, nz);
      const double mass = rho[c] * vol[c];
      const double vr_cell = 0.25 * (vr[n00] + vr[n10] + vr[n11] + vr[n01]);
      const double vz_cell = 0.25 * (vz[n00] + vz[n10] + vz[n11] + vz[n01]);
      sums.mass += mass;
      sums.internal += mass * (ee[c] + ei[c]);
      sums.kinetic += 0.5 * mass * (vr_cell * vr_cell + vz_cell * vz_cell);
    }
  }
  return sums;
}

void log_reference_conservation(const int step,
                                const ConservedSums& before,
                                const ConservedSums& after) {
  const double mass_residual = relative_residual(before.mass, after.mass);
  const double energy_residual =
      relative_residual(before.internal + before.kinetic, after.internal + after.kinetic);
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6)
      << "[ref-barrier-ale-conservation] step " << step
      << ": mass_residual=" << mass_residual
      << " energy_residual=" << energy_residual;
  if (mass_residual > 1.0e-10 || energy_residual > 1.0e-10) {
    core::log_warning(oss.str());
  } else {
    core::log_info(oss.str());
  }
}

int velocity_bc_mode(const Boundary2DType bc) {
  if (bc == Boundary2DType::FIXED) {
    return 2;
  }
  if (bc == Boundary2DType::REFLECT || bc == Boundary2DType::STATE_SUPPLY) {
    return 1;
  }
  return 0;
}

int* upload_multiblock_stable_cell_ids(const tenryu::core::State& state) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR reference barrier requires multiblock topology");
  const auto& stable = state.mesh.topo.multiblock->cell_id_stable;
  TENRYU_ASSERT(stable.size() == static_cast<std::size_t>(state.mesh.topo.n_cells),
                "CSR reference barrier requires stable cell ids");
  int* d_stable = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_stable),
                        stable.size() * sizeof(int)),
             "reference barrier stable-cell allocation failed");
  cuda_check(cudaMemcpy(d_stable,
                        stable.data(),
                        stable.size() * sizeof(int),
                        cudaMemcpyHostToDevice),
             "reference barrier stable-cell upload failed");
  return d_stable;
}

ale::AleRemap2DRZResult apply_multiblock_csr_reference_remap(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_vol_old,
    const double* d_delta_r,
    const double* d_delta_z,
    const double sigma,
    const ReferenceBarrierScope* scope = nullptr) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "CSR reference barrier remap requires multiblock topology");
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  TENRYU_ASSERT(state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_reference.size() == static_cast<std::size_t>(n_nodes),
                "CSR reference barrier remap requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == static_cast<std::size_t>(n_cells),
                "CSR reference barrier remap requires reference volume storage");

  double* d_ref_r_old = nullptr;
  double* d_ref_z_old = nullptr;
  double* d_cell_vol_initial_old = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ref_r_old), node_bytes),
             "reference barrier old reference r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ref_z_old), node_bytes),
             "reference barrier old reference z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_vol_initial_old), cell_bytes),
             "reference barrier old reference volume allocation failed");
  cuda_check(cudaMemcpy(d_ref_r_old,
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference r copy failed");
  cuda_check(cudaMemcpy(d_ref_z_old,
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference z copy failed");
  cuda_check(cudaMemcpy(d_cell_vol_initial_old,
                        state.cell_vol_initial.data(),
                        cell_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier old reference volume copy failed");

  const auto restore_reference = [&]() {
    cuda_check(cudaMemcpy(state.x_r_reference.data(),
                          d_ref_r_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference r failed");
    cuda_check(cudaMemcpy(state.x_z_reference.data(),
                          d_ref_z_old,
                          node_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference z failed");
    cuda_check(cudaMemcpy(state.cell_vol_initial.data(),
                          d_cell_vol_initial_old,
                          cell_bytes,
                          cudaMemcpyDeviceToDevice),
               "reference barrier restore reference volume failed");
  };

  const int blocks_nodes = (n_nodes + 255) / 256;
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_pre_csr_remap");
  blend_to_output_kernel<<<blocks_nodes, 256>>>(state.x_r_reference.data(),
                                                state.x_z_reference.data(),
                                                d_xr_old,
                                                d_xz_old,
                                                d_delta_r,
                                                d_delta_z,
                                                sigma,
                                                n_nodes);
  cuda_check(cudaGetLastError(),
             "reference barrier CSR candidate blend kernel launch failed");
  tenryu::core::DeviceArray<std::uint8_t> core_freeze_frozen_nodes;
  tenryu::hydro::ale::core_freeze::restore_target_if_enabled(
      state,
      cfg,
      state.x_r_reference.data(),
      state.x_z_reference.data(),
      d_xr_old,
      d_xz_old,
      false,
      "reference_barrier_csr",
      cfg.numerics.ale.core_freeze_skip_velocity_projection
          ? &core_freeze_frozen_nodes
          : nullptr);

  cuda_check(cudaMemcpy(state.x_r.data(),
                        state.x_r_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier CSR candidate r copy failed");
  cuda_check(cudaMemcpy(state.x_z.data(),
                        state.x_z_reference.data(),
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier CSR candidate z copy failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier CSR candidate synchronize failed");
  state.mesh.node_r = state.x_r.data();
  state.mesh.node_z = state.x_z.data();
  state.mesh.recompute_geometry();
  state.cell_vol_initial = state.mesh.cell_vol;
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_candidate");
  tenryu::hydro::ale::ale_velcoherence::sample(
      state, cfg, "s1_post_rezone");

  cuda_check(cudaMemcpy(state.x_r.data(),
                        d_xr_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old r before CSR remap failed");
  cuda_check(cudaMemcpy(state.x_z.data(),
                        d_xz_old,
                        node_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old z before CSR remap failed");
  cuda_check(cudaMemcpy(state.vol.data(),
                        d_vol_old,
                        cell_bytes,
                        cudaMemcpyDeviceToDevice),
             "reference barrier restore old volume before CSR remap failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier restore old mesh before CSR remap failed");

  tenryu::core::Config remap_cfg = cfg;
  remap_cfg.numerics.ale.conservative_remap_enabled = true;
  remap_cfg.numerics.ale.conservative_remap_target = "reference";
  remap_cfg.numerics.ale.multiblock_scaled_reference_enabled = false;
  const std::uint8_t* core_freeze_velocity_mask =
      core_freeze_frozen_nodes.size() == static_cast<std::size_t>(n_nodes)
          ? core_freeze_frozen_nodes.data()
          : nullptr;
  ale::AleRemap2DRZOverrides overrides;
  if (scope != nullptr) {
    overrides.active_cell_mask = scope->d_active_cell_mask;
    overrides.velocity_projection_frozen_node_mask =
        scope->d_frozen_velocity_node_mask;
  }
  const auto remap_result =
      tenryu::hydro::ale::ale_remap_2d_rz(
          state,
          remap_cfg,
          nullptr,
          0.0,
          core_freeze_velocity_mask,
          overrides);
  restore_reference();
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_post_remap");
  cuda_check(cudaFree(d_cell_vol_initial_old),
             "reference barrier old reference volume free failed");
  cuda_check(cudaFree(d_ref_z_old), "reference barrier old reference z free failed");
  cuda_check(cudaFree(d_ref_r_old), "reference barrier old reference r free failed");
  return remap_result;
}

}  // namespace

std::uint64_t evaluate_reference_barrier_trigger(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  std::uint64_t mask = 0;
  if (!cfg.numerics.ale.reference_barrier_enabled) {
    return mask;
  }
  if (cfg.numerics.ale.reference_force_engage_every_step) {
    return static_cast<std::uint64_t>(
        ReferenceBarrierAleResult::TriggerReason::AxisMargin);
  }
  if (cfg.numerics.ale.reference_trigger_axis_margin_enabled) {
    const double margin = cfg.numerics.has_physical_rz_axis
                              ? compute_axis_margin_min_host(state)
                              : std::numeric_limits<double>::infinity();
    if (margin < cfg.numerics.ale.reference_trigger_axis_margin_threshold) {
      mask |= static_cast<std::uint64_t>(
          ReferenceBarrierAleResult::TriggerReason::AxisMargin);
    }
  }
  if (cfg.numerics.ale.reference_trigger_corner_j_ratio_enabled) {
    const double ratio = compute_corner_j_ratio_min_host(state);
    if (ratio < cfg.numerics.ale.reference_trigger_corner_j_ratio_threshold) {
      mask |= static_cast<std::uint64_t>(
          ReferenceBarrierAleResult::TriggerReason::CornerJRatio);
    }
  }
  return mask;
}

void build_reference_target_mesh(const tenryu::core::State& state,
                                 const tenryu::core::Config& cfg,
                                 double* d_target_r,
                                 double* d_target_z) {
  TENRYU_ASSERT(d_target_r != nullptr, "reference target r buffer is null");
  TENRYU_ASSERT(d_target_z != nullptr, "reference target z buffer is null");
  const int n_nodes = state.mesh.topo.n_nodes;
  if (cfg.numerics.ale.multiblock_scaled_reference_enabled &&
      tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    const double alpha =
        tenryu::hydro::ale::compute_scale_factor_alpha(state, cfg);
    mesh_trace::trace_alpha_source(state, cfg, alpha);
    tenryu::hydro::ale::build_scaled_gamma_mvp_target(
        state, cfg, alpha, d_target_r, d_target_z);
    mesh_trace::trace_post_target(state, cfg, d_target_r, d_target_z);
    mesh_trace::trace_cell0_target(
        state, cfg, "refbar_scaled_reference", d_target_r, d_target_z);
    return;
  }
  if (cfg.numerics.ale.reference_target == "eulerian_initial") {
    TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size(),
                  "reference target requires x_r_initial");
    TENRYU_ASSERT(state.x_z_initial.size() == state.x_z.size(),
                  "reference target requires x_z_initial");
    cuda_check(cudaMemcpy(d_target_r,
                          state.x_r_initial.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "reference target r copy failed");
    cuda_check(cudaMemcpy(d_target_z,
                          state.x_z_initial.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "reference target z copy failed");
    return;
  }
  if (cfg.numerics.ale.reference_target == "spherical_equal_angle") {
    const int blocks = (n_nodes + 255) / 256;
    build_spherical_equal_angle_kernel<<<blocks, 256>>>(state.x_r_initial.data(),
                                                        state.x_z_initial.data(),
                                                        d_target_r,
                                                        d_target_z,
                                                        state.mesh.topo.nr,
                                                        state.mesh.topo.nz);
    cuda_check(cudaGetLastError(), "reference spherical target kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "reference spherical target kernel synchronize failed");
    return;
  }
  cuda_check(cudaMemcpy(d_target_r,
                        state.x_r.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "reference none target r copy failed");
  cuda_check(cudaMemcpy(d_target_z,
                        state.x_z.data(),
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "reference none target z copy failed");
}

ReferenceBarrierAleResult apply_reference_barrier_rezone(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_target_r,
    const double* d_target_z,
    const ReferenceBarrierScope* scope) {
  ReferenceBarrierAleResult result;
  // The trigger path gates on reference_barrier_enabled alone; the apply path also serves the button morph (S-C), which supplies its own targets.
  if (!cfg.numerics.ale.reference_barrier_enabled &&
      !cfg.numerics.ale.button_morph.enabled &&
      !cfg.numerics.ale.runtime_controller.controller_enabled) {
    return result;
  }
  result.engaged = true;
  tenryu::hydro::ale::ale_velcoherence::sample(
      state, cfg, "s0_post_hydro");

  const bool is_multiblock = mesh::mesh_topo_is_multiblock(cfg.mesh);
  const int n_nodes = state.mesh.topo.n_nodes;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = is_multiblock ? state.mesh.topo.n_cells : nr * nz;
  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const std::size_t cell_bytes = static_cast<std::size_t>(n_cells) * sizeof(double);
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  double* d_delta_r = nullptr;
  double* d_delta_z = nullptr;
  double* d_xr_old = nullptr;
  double* d_xz_old = nullptr;
  double* d_vol_old = nullptr;
  double* d_tmp = nullptr;
  double* d_vol_mid = nullptr;
  double* d_mom_r = nullptr;
  double* d_mom_z = nullptr;
  double* d_e_e = nullptr;
  double* d_e_i = nullptr;
  double* d_vr_cell = nullptr;
  double* d_vz_cell = nullptr;
  double* d_group = nullptr;
  std::uint8_t* d_node_flags = upload_node_flags_if_constraints(state);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_r),
                        node_bytes),
             "reference barrier delta_r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_delta_z),
                        node_bytes),
             "reference barrier delta_z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xr_old), node_bytes),
             "reference barrier old r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_xz_old), node_bytes),
             "reference barrier old z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vol_old), cell_bytes),
             "reference barrier old volume allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_tmp), cell_bytes),
             "reference barrier remap tmp allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vol_mid), cell_bytes),
             "reference barrier remap volume tmp allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mom_r), cell_bytes),
             "reference barrier mom_r allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mom_z), cell_bytes),
             "reference barrier mom_z allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_e), cell_bytes),
             "reference barrier e_e allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_i), cell_bytes),
             "reference barrier e_i allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vr_cell), cell_bytes),
             "reference barrier v_r_cell allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_vz_cell), cell_bytes),
             "reference barrier v_z_cell allocation failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_group), cell_bytes),
             "reference barrier group scratch allocation failed");
  cuda_check(cudaMemcpy(d_xr_old, state.x_r.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old r copy failed");
  cuda_check(cudaMemcpy(d_xz_old, state.x_z.data(), node_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old z copy failed");
  cuda_check(cudaMemcpy(d_vol_old, state.vol.data(), cell_bytes, cudaMemcpyDeviceToDevice),
             "reference barrier old volume copy failed");
  const ConservedSums sums_before = compute_conserved_sums_host(state);
  mesh_trace::trace_cell0_geometry(state, cfg, "refbar_pre_rezone");

  delta_kernel<<<blocks_nodes, 256>>>(state.x_r.data(),
                                      state.x_z.data(),
                                      d_target_r,
                                      d_target_z,
                                      d_delta_r,
                                      d_delta_z,
                                      n_nodes);
  cuda_check(cudaGetLastError(), "reference barrier delta kernel launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "reference barrier delta kernel synchronize failed");

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
  int* d_cell_id_stable = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  tenryu::mesh::LineSearchResult ls;
  if (is_multiblock) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CSR reference barrier requires device cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CSR reference barrier requires device cell-node CSR indices");
    d_cell_id_stable = upload_multiblock_stable_cell_ids(state);
    d_cell_nverts = upload_cell_nverts_if_active(state);
    ls = tenryu::mesh::linesearch_largest_admissible_sigma_csr(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        cfg.numerics.ale.reference_blend_default,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        n_cells,
        state.mesh.corner_stride,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_id_stable,
        floors,
        d_cell_nverts,
        nullptr,
        nullptr,
        0,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
  } else {
    ls = tenryu::mesh::linesearch_largest_admissible_sigma(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        cfg.numerics.ale.reference_blend_default,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        nr,
        nz,
        floors,
        nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
  }
  result.lambda_accepted = ls.sigma_accepted;
  result.linesearch_iters = ls.iters_used;
  result.final_quality = ls.quality;

  RuntimeAleRollbackSnapshot runtime_rollback_snapshot;
  if (ls.sigma_accepted > 0.0) {
    if (scope != nullptr && scope->runtime_ale) {
      runtime_rollback_snapshot.capture(state, cfg, is_multiblock, n_cells);
    }
    if (is_multiblock) {
      const auto remap_result =
          apply_multiblock_csr_reference_remap(state,
                                               cfg,
                                               d_xr_old,
                                               d_xz_old,
                                               d_vol_old,
                                               d_delta_r,
                                               d_delta_z,
                                               ls.sigma_accepted,
                                               scope);
      result.mass_floor_delta += remap_result.mass_floor_delta;
      result.E_floor_injected += remap_result.E_floor_injected;
      result.E_redistribution_unresolved +=
          remap_result.E_redistribution_unresolved;
      const bool remap_ok = remap_result.applied;
      result.succeeded = remap_ok;
      if (remap_ok) {
        recache_corner_mass_after_reference_remap(state, cfg, d_cell_nverts);
        const ConservedSums sums_after = compute_conserved_sums_host(state);
        log_reference_conservation(state.step, sums_before, sums_after);
      }
    } else {
    pack_cell_conserved_kernel<<<blocks_cells, 256>>>(d_mom_r,
                                                       d_mom_z,
                                                       d_e_e,
                                                       d_e_i,
                                                       d_vr_cell,
                                                       d_vz_cell,
                                                       state.rho.data(),
                                                       state.ee.data(),
                                                       state.ei.data(),
                                                       state.v_r.data(),
                                                       state.v_z.data(),
                                                       nr,
                                                       nz);
    cuda_check(cudaGetLastError(), "reference barrier pack conserved kernel launch failed");

    blend_kernel<<<blocks_nodes, 256>>>(state.x_r.data(),
                                        state.x_z.data(),
                                        d_delta_r,
                                        d_delta_z,
                                        ls.sigma_accepted,
                                        n_nodes);
    cuda_check(cudaGetLastError(), "reference barrier blend kernel launch failed");
    cuda_check(cudaDeviceSynchronize(),
               "reference barrier blend kernel synchronize failed");
    tenryu::hydro::ale::core_freeze::restore_target_if_enabled(
        state,
        cfg,
        state.x_r.data(),
        state.x_z.data(),
        d_xr_old,
        d_xz_old,
        false,
        "reference_barrier_structured");
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;

    const bool donor_sign_fixed = cfg.numerics.ale.swept_volume_sign_fixed;
    const auto remap_scalar = [&](double* field) {
      return ale::launch_remap_strang(field,
                                      d_tmp,
                                      d_vol_mid,
                                      d_vol_old,
                                      state.vol.data(),
                                      d_xr_old,
                                      d_xz_old,
                                      state.x_r.data(),
                                      state.x_z.data(),
                                      nr,
                                      nz,
                                      state.step,
                                      donor_sign_fixed,
                                      state.evacuated_cells.d_geometry_policy_exempt_cells.empty()
                                          ? nullptr
                                          : state.evacuated_cells
                                                .d_geometry_policy_exempt_cells.data());
    };

    bool remap_ok = remap_scalar(state.rho.data());
    remap_ok = remap_ok && remap_scalar(d_mom_r);
    remap_ok = remap_ok && remap_scalar(d_mom_z);
    remap_ok = remap_ok && remap_scalar(d_e_e);
    remap_ok = remap_ok && remap_scalar(d_e_i);

    const int n_mat = static_cast<int>(cfg.materials.materials.size());
    if (remap_ok && n_mat > 0 &&
        state.volFrac.size() == static_cast<std::size_t>(n_cells * n_mat)) {
      for (int m = 0; m < n_mat && remap_ok; ++m) {
        gather_material_field_kernel<<<blocks_cells, 256>>>(
            d_group, state.volFrac.data(), n_cells, n_mat, m);
        cuda_check(cudaGetLastError(),
                   "reference barrier gather volFrac kernel launch failed");
        remap_ok = remap_scalar(d_group);
        if (remap_ok) {
          scatter_material_field_kernel<<<blocks_cells, 256>>>(
              state.volFrac.data(), d_group, n_cells, n_mat, m);
          cuda_check(cudaGetLastError(),
                     "reference barrier scatter volFrac kernel launch failed");
        }
      }
      if (remap_ok) {
        normalize_reference_volfrac_kernel<<<blocks_cells, 256>>>(
            state.volFrac.data(), n_cells, n_mat);
        cuda_check(cudaGetLastError(),
                   "reference barrier normalize volFrac kernel launch failed");
      }
    }

    const int n_groups = cfg.radiation.groups;
    if (remap_ok && n_groups > 0 &&
        state.rad_E.size() == static_cast<std::size_t>(n_cells * n_groups)) {
      for (int g = 0; g < n_groups && remap_ok; ++g) {
        gather_group_field_kernel<<<blocks_cells, 256>>>(
            d_group, state.rad_E.data(), n_cells, n_groups, g);
        cuda_check(cudaGetLastError(),
                   "reference barrier gather rad_E kernel launch failed");
        remap_ok = remap_scalar(d_group);
        if (remap_ok) {
          scatter_group_field_kernel<<<blocks_cells, 256>>>(
              state.rad_E.data(), d_group, n_cells, n_groups, g);
          cuda_check(cudaGetLastError(),
                     "reference barrier scatter rad_E kernel launch failed");
        }
      }
    }

    if (remap_ok) {
      recover_cell_primitives_kernel<<<blocks_cells, 256>>>(state.rho.data(),
                                                             state.mass.data(),
                                                             state.ee.data(),
                                                             state.ei.data(),
                                                             d_vr_cell,
                                                             d_vz_cell,
                                                             d_mom_r,
                                                             d_mom_z,
                                                             d_e_e,
                                                             d_e_i,
                                                             state.vol.data(),
                                                             cfg.numerics.floors.rho,
                                                             n_cells);
      cuda_check(cudaGetLastError(),
                 "reference barrier recover primitive kernel launch failed");

      const auto r_outer_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
      const auto z_bottom_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
      const auto z_top_type =
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
      project_reference_cell_velocity_to_nodes_kernel<<<blocks_nodes, 256>>>(
          state.v_r.data(),
          state.v_z.data(),
          d_vr_cell,
          d_vz_cell,
          state.rho.data(),
          state.vol.data(),
          d_node_flags,
          nr,
          nz,
          velocity_bc_mode(r_outer_type),
          velocity_bc_mode(z_bottom_type),
          velocity_bc_mode(z_top_type));
      cuda_check(cudaGetLastError(),
                 "reference barrier velocity projection kernel launch failed");
      cuda_check(cudaDeviceSynchronize(),
                 "reference barrier remap synchronize failed");
      recache_corner_mass_after_reference_remap(state, cfg, d_cell_nverts);
      const ConservedSums sums_after = compute_conserved_sums_host(state);
      log_reference_conservation(state.step, sums_before, sums_after);
    } else {
      core::log_warning(
          "[ref-barrier-ale] remap aborted: non-positive intermediate volume");
    }
    result.succeeded = remap_ok;
    }
  } else if (is_multiblock) {
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s1_post_rezone");
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s2_post_remap");
    tenryu::hydro::ale::ale_velcoherence::sample(
        state, cfg, "s3_post_velproj");
  }

  if (result.succeeded && scope != nullptr && scope->runtime_ale) {
    const CommitGateFailure failure =
        evaluate_runtime_ale_commit_gate(state, cfg, *scope);
    if (failure.quantity != nullptr) {
      if (cfg.numerics.ale.runtime_controller.commit_rollback_enabled) {
        TENRYU_ASSERT(runtime_rollback_snapshot.captured,
                      "runtime ALE commit rollback requires a snapshot");
        runtime_rollback_snapshot.restore(state);
        state.mesh.node_r = state.x_r.data();
        state.mesh.node_z = state.x_z.data();
        state.mesh.recompute_geometry();
        state.vol = state.mesh.cell_vol;
        recache_corner_mass_after_reference_remap(state, cfg, d_cell_nverts);
        result.succeeded = false;
        result.rolled_back = true;
        result.last_reject_reason = 3;
        result.last_reject_cell = failure.cell;
        result.mass_floor_delta = 0.0;
        result.E_floor_injected = 0.0;
        result.E_redistribution_unresolved = 0.0;
      }
      log_runtime_ale_commit_rollback(failure);
    }
  }

  if (d_cell_id_stable != nullptr) {
    cuda_check(cudaFree(d_cell_id_stable),
               "reference barrier stable-cell free failed");
  }
  if (d_cell_nverts != nullptr) {
    cuda_check(cudaFree(d_cell_nverts),
               "reference barrier cell_nverts free failed");
  }
  cuda_check(cudaFree(d_group), "reference barrier group scratch free failed");
  if (d_node_flags != nullptr) {
    cuda_check(cudaFree(d_node_flags), "reference barrier node_flags free failed");
  }
  cuda_check(cudaFree(d_vz_cell), "reference barrier v_z_cell free failed");
  cuda_check(cudaFree(d_vr_cell), "reference barrier v_r_cell free failed");
  cuda_check(cudaFree(d_e_i), "reference barrier e_i free failed");
  cuda_check(cudaFree(d_e_e), "reference barrier e_e free failed");
  cuda_check(cudaFree(d_mom_z), "reference barrier mom_z free failed");
  cuda_check(cudaFree(d_mom_r), "reference barrier mom_r free failed");
  cuda_check(cudaFree(d_vol_mid), "reference barrier remap volume tmp free failed");
  cuda_check(cudaFree(d_tmp), "reference barrier remap tmp free failed");
  cuda_check(cudaFree(d_vol_old), "reference barrier old volume free failed");
  cuda_check(cudaFree(d_xz_old), "reference barrier old z free failed");
  cuda_check(cudaFree(d_xr_old), "reference barrier old r free failed");
  cuda_check(cudaFree(d_delta_z), "reference barrier delta_z free failed");
  cuda_check(cudaFree(d_delta_r), "reference barrier delta_r free failed");
  return result;
}

}  // namespace tenryu::hydro
