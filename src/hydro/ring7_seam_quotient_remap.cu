#include "hydro/ring7_seam_quotient_remap.cuh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/eos_context.hpp"
#include "mesh/path_admissibility.cuh"
#include "parallel/reduction.hpp"

namespace tenryu::hydro {
namespace {

bool env_flag_value(const char* name, bool* present = nullptr) {
  const char* raw = std::getenv(name);
  if (present != nullptr) {
    *present = raw != nullptr;
  }
  if (raw == nullptr) {
    return false;
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return !(value.empty() || value == "0" || value == "false" ||
           value == "no" || value == "off");
}

int env_int_value(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed <= 0 ||
      parsed > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

double env_double_value(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || !std::isfinite(parsed)) {
    return fallback;
  }
  return parsed;
}

bool ring7_feature_enabled(const core::Config& cfg) {
  bool present = false;
  const bool env_enabled =
      env_flag_value("TENRYU_I1B_RING7_QUOTIENT", &present);
  const bool enabled =
      present ? env_enabled : cfg.numerics.hydro.ring7_quotient_enabled;
  if (enabled && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "ring-7 seam repair is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  return enabled;
}

bool ring7_diag_due(const core::State& state) {
  if (!env_flag_value("TENRYU_I1B_RING7_QUOTIENT_DIAG")) {
    return false;
  }
  const int every = env_int_value("TENRYU_I1B_RING7_QUOTIENT_DIAG_EVERY", 1);
  return every > 0 && state.step >= 0 && (state.step % every) == 0;
}

std::string sci(const double value) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(17) << value;
  return os.str();
}

std::string hex_float(const double value) {
  std::ostringstream os;
  os << std::hexfloat << value;
  return os.str();
}

void sort_unique(std::vector<int>& values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

int cell_block_id(const mesh::MultiBlockTopology& mb, const int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= mb.cell_block_id.size()) {
    return -1;
  }
  return mb.cell_block_id[static_cast<std::size_t>(cell)];
}

mesh::BlockRole cell_role(const mesh::MultiBlockTopology& mb, const int cell) {
  const int block = cell_block_id(mb, cell);
  if (block < 0 || static_cast<std::size_t>(block) >= mb.blocks.size()) {
    return mesh::BlockRole::CENTRAL_CORE;
  }
  return mb.blocks[static_cast<std::size_t>(block)].role;
}

bool is_shell_cell(const mesh::MultiBlockTopology& mb, const int cell) {
  return cell_role(mb, cell) == mesh::BlockRole::POLAR_SHELL;
}

bool is_trifan_role(const mesh::BlockRole role) {
  return role == mesh::BlockRole::NORTH_FAN ||
         role == mesh::BlockRole::EAST_FAN ||
         role == mesh::BlockRole::SOUTH_FAN ||
         role == mesh::BlockRole::CENTRAL_CORE;
}

void mark_cell(std::vector<std::uint8_t>& mask,
               std::vector<int>& list,
               const int cell) {
  if (cell < 0 || static_cast<std::size_t>(cell) >= mask.size()) {
    return;
  }
  if (mask[static_cast<std::size_t>(cell)] == 0U) {
    mask[static_cast<std::size_t>(cell)] = 1U;
    list.push_back(cell);
  }
}

int active_nverts(const core::State& state, const int cell) {
  const int n_cells = state.mesh.topo.n_cells;
  const std::uint8_t* nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? state.mesh.cell_nverts.data()
          : nullptr;
  return mesh::mesh_topo_cell_active_nverts(nverts, cell);
}

void mark_cell_nodes(const core::State& state,
                     const int cell,
                     std::vector<std::uint8_t>& node_mask,
                     std::vector<int>& nodes) {
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (cell < 0 || cell >= state.mesh.topo.n_cells) {
    return;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(state, cell);
  for (int k = 0; k < nverts; ++k) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    if (node >= 0 && node < n_nodes &&
        node_mask[static_cast<std::size_t>(node)] == 0U) {
      node_mask[static_cast<std::size_t>(node)] = 1U;
      nodes.push_back(node);
    }
  }
}

bool cell_has_any_node(const core::State& state,
                       const int cell,
                       const std::vector<int>& target_nodes) {
  const auto& mb = *state.mesh.topo.multiblock;
  if (target_nodes.empty() || cell < 0 || cell >= state.mesh.topo.n_cells) {
    return false;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(state, cell);
  for (int k = 0; k < nverts; ++k) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    if (std::find(target_nodes.begin(), target_nodes.end(), node) !=
        target_nodes.end()) {
      return true;
    }
  }
  return false;
}

std::vector<double> copy_device_doubles(const double* device,
                                        const int count,
                                        const char* label) {
  std::vector<double> host(static_cast<std::size_t>(count), 0.0);
  if (count <= 0) {
    return host;
  }
  (void)label;
  CUDA_CHECK(cudaMemcpy(host.data(),
                        device,
                        static_cast<std::size_t>(count) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  return host;
}

mesh::MeshForecastComponents cell_components(
    const core::State& state,
    const std::vector<double>& x_r_start,
    const std::vector<double>& x_z_start,
    const std::vector<double>& x_r_trial,
    const std::vector<double>& x_z_trial,
    const int cell,
    const core::Config& cfg) {
  return mesh::path_admissibility_detail::mesh_forecast_cell_components(
      state,
      x_r_start,
      x_z_start,
      x_r_trial,
      x_z_trial,
      cell,
      1.0,
      cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel);
}

const MeshQualityRzVolumeCellMargin* find_margin(
    const std::vector<MeshQualityRzVolumeCellMargin>& margins,
    const int cell) {
  for (const auto& margin : margins) {
    if (margin.cell == cell) {
      return &margin;
    }
  }
  return nullptr;
}

struct Ring7QualityMetrics {
  double min_q_J = std::numeric_limits<double>::infinity();
  double min_q_edge = std::numeric_limits<double>::infinity();
  double min_q_V = std::numeric_limits<double>::infinity();
  double min_eta_V = std::numeric_limits<double>::infinity();
  int min_q_J_cell = -1;
  int min_q_edge_cell = -1;
  int min_q_V_cell = -1;
  int min_eta_V_cell = -1;
};

struct SeamTotals {
  long double mass = 0.0L;
  long double internal_energy = 0.0L;
  long double kinetic_energy = 0.0L;
  long double total_energy = 0.0L;
  long double momentum_r = 0.0L;
  long double momentum_z = 0.0L;
};

struct Ring7PacketRemapDiagnostics {
  bool ok = false;
  bool positivity_failed = false;
  bool conservation_failed = false;
  bool central_failed = false;
  int subcycles = 0;
  double dM_rel = 0.0;
  double dE_rel = 0.0;
  double dPr_paired_rel = 0.0;
  double dPz_paired_rel = 0.0;
  double q_mix = 0.0;
  double boundary_work = 0.0;
  double max_volume_error = 0.0;
  std::uint64_t central_hash_before = 0ULL;
  std::uint64_t central_hash_after = 0ULL;
  SeamTotals central_before;
  SeamTotals central_after;
};

double rel_delta(const long double after, const long double before) {
  const long double denom =
      std::max(std::abs(before), static_cast<long double>(1.0e-300));
  return static_cast<double>(std::abs(after - before) / denom);
}

double rel_delta_scale(const long double delta, const long double scale) {
  const long double denom =
      std::max(std::abs(scale), static_cast<long double>(1.0e-300));
  return static_cast<double>(std::abs(delta) / denom);
}

double log_hinge_barrier(const double value, const double target) {
  if (!(value > 0.0) || !std::isfinite(value) || !(target > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  const double h = std::max(0.0, std::log(target / value));
  return h * h;
}

double ring7_objective(const Ring7QualityMetrics& metrics,
                       const double q_target,
                       const double eta_target) {
  return log_hinge_barrier(metrics.min_q_J, q_target) +
         log_hinge_barrier(metrics.min_q_edge, q_target) +
         4.0 * log_hinge_barrier(metrics.min_eta_V, eta_target);
}

int shell_node_id(const mesh::BlockInfo& shell, const int i, const int j) {
  return shell.owned_node_begin + i * (shell.n_j_cells + 1) + j;
}

double signed_cell_volume_from_coords(const core::State& state,
                                      const std::vector<double>& x_r,
                                      const std::vector<double>& x_z,
                                      const int cell) {
  if (!state.mesh.topo.multiblock.has_value() || cell < 0 ||
      cell >= state.mesh.topo.n_cells) {
    return 0.0;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(state, cell);
  Ring7RzPoint points[4];
  for (int k = 0; k < nverts; ++k) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    points[k] = {x_r[static_cast<std::size_t>(node)],
                 x_z[static_cast<std::size_t>(node)]};
  }
  return axisymmetric_swept_volume_polygon(points, nverts);
}

std::vector<double> candidate_cell_volumes(const core::State& state,
                                           const Ring7SeamPatch& patch,
                                           const std::vector<double>& x_r,
                                           const std::vector<double>& x_z) {
  std::vector<double> vol;
  state.vol.copy_to_host(vol);
  for (const int cell : patch.p_seam_cells) {
    if (cell >= 0 && cell < static_cast<int>(vol.size())) {
      vol[static_cast<std::size_t>(cell)] =
          std::abs(signed_cell_volume_from_coords(state, x_r, x_z, cell));
    }
  }
  return vol;
}

Ring7QualityMetrics evaluate_ring7_candidate_metrics(
    const core::State& state,
    const core::Config& cfg,
    const Ring7SeamPatch& patch,
    const std::vector<double>& x_r_start,
    const std::vector<double>& x_z_start,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    const double* d_v_r_nominal,
    const double* d_v_z_nominal,
    const double dt_nominal,
    const double eta_probe) {
  Ring7QualityMetrics metrics;
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> x_r_trial(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> x_z_trial(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    x_r_trial[static_cast<std::size_t>(n)] =
        x_r_start[static_cast<std::size_t>(n)] +
        dt_nominal * v_r[static_cast<std::size_t>(n)];
    x_z_trial[static_cast<std::size_t>(n)] =
        x_z_start[static_cast<std::size_t>(n)] +
        dt_nominal * v_z[static_cast<std::size_t>(n)];
  }

  for (const int cell : patch.p_seam_cells) {
    const mesh::MeshForecastComponents q =
        cell_components(state, x_r_start, x_z_start, x_r_trial, x_z_trial,
                        cell, cfg);
    if (q.q_J < metrics.min_q_J) {
      metrics.min_q_J = q.q_J;
      metrics.min_q_J_cell = cell;
    }
    if (q.q_edge < metrics.min_q_edge) {
      metrics.min_q_edge = q.q_edge;
      metrics.min_q_edge_cell = cell;
    }
    if (q.q_V < metrics.min_q_V) {
      metrics.min_q_V = q.q_V;
      metrics.min_q_V_cell = cell;
    }
  }

  core::DeviceArray<double> d_x_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_x_z(static_cast<std::size_t>(n_nodes));
  d_x_r.copy_from_host(x_r_start);
  d_x_z.copy_from_host(x_z_start);
  const std::vector<double> vol =
      candidate_cell_volumes(state, patch, x_r_start, x_z_start);
  core::DeviceArray<double> d_vol(vol.size());
  d_vol.copy_from_host(vol);
  const double probe = std::max(1.0, eta_probe);
  const std::vector<MeshQualityRzVolumeCellMargin> margins_nominal =
      compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume(
          state, cfg, d_x_r.data(), d_x_z.data(), d_v_r_nominal,
          d_v_z_nominal, d_vol.data(), dt_nominal, patch.ring7_cells);
  const std::vector<MeshQualityRzVolumeCellMargin> margins_probe =
      compute_multiblock_mesh_quality_rz_volume_cell_margins_with_volume(
          state, cfg, d_x_r.data(), d_x_z.data(), d_v_r_nominal,
          d_v_z_nominal, d_vol.data(), probe * dt_nominal,
          patch.ring7_cells);
  for (std::size_t i = 0; i < margins_nominal.size(); ++i) {
    const auto& nominal = margins_nominal[i];
    double eta = nominal.eta_safe;
    if (nominal.eta_safe >= 1.0 && i < margins_probe.size()) {
      eta = probe * margins_probe[i].eta_safe;
    }
    if (eta < metrics.min_eta_V) {
      metrics.min_eta_V = eta;
      metrics.min_eta_V_cell = nominal.cell;
    }
  }
  return metrics;
}

bool path_admissible_between(const core::State& state,
                             const core::Config& cfg,
                             const std::vector<double>& x_r_old,
                             const std::vector<double>& x_z_old,
                             const std::vector<double>& x_r_new,
                             const std::vector<double>& x_z_new) {
  const int n_nodes = state.mesh.topo.n_nodes;
  core::DeviceArray<double> d_old_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_old_z(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_new_r(static_cast<std::size_t>(n_nodes));
  core::DeviceArray<double> d_new_z(static_cast<std::size_t>(n_nodes));
  d_old_r.copy_from_host(x_r_old);
  d_old_z.copy_from_host(x_z_old);
  d_new_r.copy_from_host(x_r_new);
  d_new_z.copy_from_host(x_z_new);
  const mesh::PathAdmissibilityResult path =
      mesh::evaluate_path_admissibility(
          state, cfg, d_old_r.data(), d_old_z.data(), d_new_r.data(),
          d_new_z.data(),
          cfg.numerics.hydro.mesh_quality_dt_corner_j_floor_rel,
          nullptr,
          state.x_r_reference.size() == state.x_r.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr);
  return path.first_failing_cell < 0;
}

double max_patch_swept_fraction(const core::State& state,
                                const Ring7SeamPatch& patch,
                                const std::vector<double>& x_r_old,
                                const std::vector<double>& x_z_old,
                                const std::vector<double>& x_r_new,
                                const std::vector<double>& x_z_new) {
  double max_fraction = 0.0;
  for (const int cell : patch.p_seam_cells) {
    const double v_old =
        std::abs(signed_cell_volume_from_coords(state, x_r_old, x_z_old, cell));
    const double v_new =
        std::abs(signed_cell_volume_from_coords(state, x_r_new, x_z_new, cell));
    const double denom = std::max(v_old, 1.0e-300);
    max_fraction = std::max(max_fraction, std::abs(v_new - v_old) / denom);
  }
  return max_fraction;
}

void build_tangential_displacement(const core::State& state,
                                   const Ring7SeamPatch& patch,
                                   const std::vector<double>& x_r,
                                   const std::vector<double>& x_z,
                                   const double move_scale,
                                   std::vector<double>& d_r,
                                   std::vector<double>& d_z) {
  const auto& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  const int n_nodes = state.mesh.topo.n_nodes;
  d_r.assign(static_cast<std::size_t>(n_nodes), 0.0);
  d_z.assign(static_cast<std::size_t>(n_nodes), 0.0);
  for (int i = patch.shell_ring_i; i <= patch.shell_ring_i + 1; ++i) {
    for (int j = 1; j < shell.n_j_cells; ++j) {
      const int node = shell_node_id(shell, i, j);
      const int prev = shell_node_id(shell, i, j - 1);
      const int next = shell_node_id(shell, i, j + 1);
      int radial_ref = -1;
      if (i == patch.shell_ring_i && i + 1 <= shell.n_i_cells) {
        radial_ref = shell_node_id(shell, i + 1, j);
      } else if (i > 0) {
        radial_ref = shell_node_id(shell, i - 1, j);
      }
      if (node < 0 || node >= n_nodes || radial_ref < 0 ||
          radial_ref >= n_nodes) {
        continue;
      }
      double nr = x_r[static_cast<std::size_t>(radial_ref)] -
                  x_r[static_cast<std::size_t>(node)];
      double nz = x_z[static_cast<std::size_t>(radial_ref)] -
                  x_z[static_cast<std::size_t>(node)];
      if (i != patch.shell_ring_i) {
        nr = -nr;
        nz = -nz;
      }
      const double nlen = std::hypot(nr, nz);
      if (!(nlen > 0.0) || !std::isfinite(nlen)) {
        continue;
      }
      nr /= nlen;
      nz /= nlen;
      const double tr = -nz;
      const double tz = nr;
      const double lap_r =
          0.5 * (x_r[static_cast<std::size_t>(prev)] +
                 x_r[static_cast<std::size_t>(next)]) -
          x_r[static_cast<std::size_t>(node)];
      const double lap_z =
          0.5 * (x_z[static_cast<std::size_t>(prev)] +
                 x_z[static_cast<std::size_t>(next)]) -
          x_z[static_cast<std::size_t>(node)];
      const double tangential = lap_r * tr + lap_z * tz;
      d_r[static_cast<std::size_t>(node)] = move_scale * tangential * tr;
      d_z[static_cast<std::size_t>(node)] = move_scale * tangential * tz;
    }
  }

  for (int i = patch.shell_ring_i; i <= patch.shell_ring_i + 1; ++i) {
    for (int j = 1; j < shell.n_j_cells; ++j) {
      const int jp = shell.n_j_cells - j;
      if (j > jp) {
        continue;
      }
      const int n0 = shell_node_id(shell, i, j);
      const int n1 = shell_node_id(shell, i, jp);
      if (n0 < 0 || n0 >= n_nodes || n1 < 0 || n1 >= n_nodes) {
        continue;
      }
      if (n0 == n1) {
        d_z[static_cast<std::size_t>(n0)] = 0.0;
        continue;
      }
      const double dr =
          0.5 * (d_r[static_cast<std::size_t>(n0)] +
                 d_r[static_cast<std::size_t>(n1)]);
      const double dz =
          0.5 * (d_z[static_cast<std::size_t>(n0)] -
                 d_z[static_cast<std::size_t>(n1)]);
      d_r[static_cast<std::size_t>(n0)] = dr;
      d_z[static_cast<std::size_t>(n0)] = dz;
      d_r[static_cast<std::size_t>(n1)] = dr;
      d_z[static_cast<std::size_t>(n1)] = -dz;
    }
  }
}

SeamTotals cell_totals(const core::State& state,
                       const std::vector<int>& cells,
                       const std::vector<double>& mass,
                       const std::vector<double>& ee,
                       const std::vector<double>& ei,
                       const std::vector<double>& v_r,
                       const std::vector<double>& v_z,
                       const std::vector<double>& corner_mass) {
  SeamTotals totals;
  const auto& mb = *state.mesh.topo.multiblock;
  const bool use_corner_mass =
      corner_mass.size() >= static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U;
  for (const int cell : cells) {
    if (cell < 0 || cell >= state.mesh.topo.n_cells) {
      continue;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(state, cell);
    long double pr = 0.0L;
    long double pz = 0.0L;
    long double ke = 0.0L;
    if (use_corner_mass) {
      for (int k = 0; k < nverts; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        const long double cm =
            corner_mass[static_cast<std::size_t>(cell) * 4U +
                        static_cast<std::size_t>(k)];
        const long double vr = v_r[static_cast<std::size_t>(node)];
        const long double vz = v_z[static_cast<std::size_t>(node)];
        pr += cm * vr;
        pz += cm * vz;
        ke += 0.5L * cm * (vr * vr + vz * vz);
      }
    }
    const long double m = mass[static_cast<std::size_t>(cell)];
    totals.mass += m;
    totals.momentum_r += pr;
    totals.momentum_z += pz;
    const long double internal =
        m * (ee[static_cast<std::size_t>(cell)] +
             ei[static_cast<std::size_t>(cell)]);
    totals.internal_energy += internal;
    totals.kinetic_energy += ke;
    totals.total_energy += internal + ke;
  }
  return totals;
}

struct Ring7PacketRecord {
  int donor_cell = -1;
  int acceptor_cell = -1;
  int donor_corner = -1;
  int acceptor_corner = -1;
  int owner_cell = -1;
  int owner_face = -1;
  int packet_id = -1;
  double volume = 0.0;
};

struct Ring7PacketGeometryDiagnostics {
  int patch_cells = 0;
  int patch_faces = 0;
  int boundary_faces = 0;
  int moved_nodes = 0;
  int seam_physical_faces = 0;
  int seam_shared_node_faces = 0;
  int seam_duplicate_node_faces = 0;
  int seam_unmatched_node_faces = 0;
  int seam_cloned_nodes = 0;
  int packets = 0;
  int invariant_violations = 0;
  int central_patch_cells = 0;
  int central_patch_nodes = 0;
  int boundary_nonzero_faces = 0;
  int first_bad_boundary_cell = -1;
  int first_bad_boundary_face = -1;
  double max_geom_epsilon = 0.0;
  double boundary_swept_sum = 0.0;
  double boundary_swept_abs = 0.0;
  std::uint64_t central_hash_before = 0ULL;
  std::uint64_t central_hash_after = 0ULL;
  SeamTotals central_before;
  SeamTotals central_after;
};

struct Ring7SeamNodeSharing {
  int shared_faces = 0;
  int duplicate_faces = 0;
  int unmatched_faces = 0;
  int cloned_nodes = 0;
};

bool coordinate_same(const double r0,
                     const double z0,
                     const double r1,
                     const double z1) {
  const double scale =
      std::max({1.0, std::abs(r0), std::abs(z0), std::abs(r1), std::abs(z1)});
  const double tol = 16.0 * std::numeric_limits<double>::epsilon() * scale;
  return std::abs(r0 - r1) <= tol && std::abs(z0 - z1) <= tol;
}

void clone_duplicate_coordinate_targets(const core::State& state,
                                        const std::vector<double>& x_r_old,
                                        const std::vector<double>& x_z_old,
                                        const std::vector<int>& seed_nodes,
                                        std::vector<double>& x_r_new,
                                        std::vector<double>& x_z_new) {
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<int> moved_seeds;
  for (const int node : seed_nodes) {
    if (node < 0 || node >= n_nodes) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(node);
    if (std::hypot(x_r_new[idx] - x_r_old[idx],
                   x_z_new[idx] - x_z_old[idx]) > 1.0e-30) {
      moved_seeds.push_back(node);
    }
  }
  sort_unique(moved_seeds);
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (std::hypot(x_r_new[idx] - x_r_old[idx],
                   x_z_new[idx] - x_z_old[idx]) > 1.0e-30) {
      continue;
    }
    for (const int seed : moved_seeds) {
      const std::size_t sidx = static_cast<std::size_t>(seed);
      if (!coordinate_same(x_r_old[idx], x_z_old[idx],
                           x_r_old[sidx], x_z_old[sidx])) {
        continue;
      }
      x_r_new[idx] = x_r_new[sidx];
      x_z_new[idx] = x_z_new[sidx];
      break;
    }
  }
}

std::array<int, 2> face_node_ids(const core::State& state,
                                 const int cell,
                                 const int local_face) {
  std::array<int, 2> nodes = {-1, -1};
  if (cell < 0 || cell >= state.mesh.topo.n_cells ||
      local_face < 0 || local_face >= mesh::kMeshTopoCellStorageSlots) {
    return nodes;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  int c0 = -1;
  int c1 = -1;
  if (!mesh::mesh_topo_active_local_face_corners(
          active_nverts(state, cell), local_face, &c0, &c1)) {
    return nodes;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  nodes[0] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c0)];
  nodes[1] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c1)];
  return nodes;
}

int cell_corner_node_id(const core::State& state,
                        const int cell,
                        const int corner) {
  if (cell < 0 || cell >= state.mesh.topo.n_cells || corner < 0 ||
      corner >= active_nverts(state, cell) ||
      !state.mesh.topo.multiblock.has_value()) {
    return -1;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  return mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner)];
}

int matching_cell_corner(const core::State& state,
                         const int cell,
                         const int target_node,
                         const std::vector<double>& x_r,
                         const std::vector<double>& x_z) {
  if (cell < 0 || cell >= state.mesh.topo.n_cells ||
      !state.mesh.topo.multiblock.has_value()) {
    return -1;
  }
  const int nverts = active_nverts(state, cell);
  for (int k = 0; k < nverts; ++k) {
    if (cell_corner_node_id(state, cell, k) == target_node) {
      return k;
    }
  }
  if (target_node < 0 || static_cast<std::size_t>(target_node) >= x_r.size()) {
    return -1;
  }
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_corner_node_id(state, cell, k);
    if (node >= 0 && static_cast<std::size_t>(node) < x_r.size() &&
        coordinate_same(x_r[static_cast<std::size_t>(target_node)],
                        x_z[static_cast<std::size_t>(target_node)],
                        x_r[static_cast<std::size_t>(node)],
                        x_z[static_cast<std::size_t>(node)])) {
      return k;
    }
  }
  return -1;
}

std::vector<std::uint8_t> mesh_boundary_node_mask(const core::State& state) {
  std::vector<std::uint8_t> mask(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  if (!state.mesh.topo.multiblock.has_value()) {
    return mask;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  for (const auto& face : mb.boundary_faces) {
    const std::array<int, 2> nodes =
        face_node_ids(state, face.cell_a, face.local_a);
    for (const int node : nodes) {
      if (node >= 0 && static_cast<std::size_t>(node) < mask.size()) {
        mask[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }
  return mask;
}

void freeze_boundary_node_targets(const std::vector<std::uint8_t>& boundary_mask,
                                  const std::vector<double>& x_r_old,
                                  const std::vector<double>& x_z_old,
                                  std::vector<double>& x_r_new,
                                  std::vector<double>& x_z_new) {
  const std::size_t n =
      std::min({boundary_mask.size(), x_r_old.size(), x_z_old.size(),
                x_r_new.size(), x_z_new.size()});
  for (std::size_t node = 0; node < n; ++node) {
    if (boundary_mask[node] != 0U) {
      x_r_new[node] = x_r_old[node];
      x_z_new[node] = x_z_old[node];
    }
  }
}

bool node_target_moved(const std::vector<double>& x_r_old,
                       const std::vector<double>& x_z_old,
                       const std::vector<double>& x_r_new,
                       const std::vector<double>& x_z_new,
                       const int node) {
  if (node < 0 || static_cast<std::size_t>(node) >= x_r_old.size()) {
    return false;
  }
  const std::size_t idx = static_cast<std::size_t>(node);
  return std::hypot(x_r_new[idx] - x_r_old[idx],
                    x_z_new[idx] - x_z_old[idx]) > 1.0e-30;
}

bool clone_duplicate_node_target(const std::vector<double>& x_r_old,
                                 const std::vector<double>& x_z_old,
                                 std::vector<double>& x_r_new,
                                 std::vector<double>& x_z_new,
                                 const int a,
                                 const int b) {
  const bool a_moved =
      node_target_moved(x_r_old, x_z_old, x_r_new, x_z_new, a);
  const bool b_moved =
      node_target_moved(x_r_old, x_z_old, x_r_new, x_z_new, b);
  if (a_moved == b_moved) {
    return false;
  }
  const int src = a_moved ? a : b;
  const int dst = a_moved ? b : a;
  if (src < 0 || dst < 0 ||
      static_cast<std::size_t>(src) >= x_r_new.size() ||
      static_cast<std::size_t>(dst) >= x_r_new.size()) {
    return false;
  }
  x_r_new[static_cast<std::size_t>(dst)] =
      x_r_new[static_cast<std::size_t>(src)];
  x_z_new[static_cast<std::size_t>(dst)] =
      x_z_new[static_cast<std::size_t>(src)];
  return true;
}

double node_pair_distance2(const std::vector<double>& x_r,
                           const std::vector<double>& x_z,
                           const int a,
                           const int b) {
  const double dr =
      x_r[static_cast<std::size_t>(a)] - x_r[static_cast<std::size_t>(b)];
  const double dz =
      x_z[static_cast<std::size_t>(a)] - x_z[static_cast<std::size_t>(b)];
  return dr * dr + dz * dz;
}

bool configured_seam_block_pair(const mesh::MultiBlockTopology& mb,
                                const int block_a,
                                const int block_b) {
  for (const auto& seam : mb.seams) {
    if ((seam.block_a == block_a && seam.block_b == block_b) ||
        (seam.block_a == block_b && seam.block_b == block_a)) {
      return true;
    }
  }
  return false;
}

Ring7SeamNodeSharing clone_unique_face_duplicate_targets(
    const core::State& state,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    std::vector<double>& x_r_new,
    std::vector<double>& x_z_new) {
  Ring7SeamNodeSharing stats;
  if (!state.mesh.topo.multiblock.has_value()) {
    return stats;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  for (const auto& face : mb.unique_internal_faces) {
    const int block_a = cell_block_id(mb, face.cell_a);
    const int block_b = cell_block_id(mb, face.cell_b);
    if (face.cell_a < 0 || face.cell_b < 0 ||
        face.cell_a >= state.mesh.topo.n_cells ||
        face.cell_b >= state.mesh.topo.n_cells || block_a == block_b ||
        !configured_seam_block_pair(mb, block_a, block_b) ||
        is_shell_cell(mb, face.cell_a) == is_shell_cell(mb, face.cell_b)) {
      continue;
    }
    const std::array<int, 2> a_nodes =
        face_node_ids(state, face.cell_a, face.local_a);
    const std::array<int, 2> b_nodes =
        face_node_ids(state, face.cell_b, face.local_b);
    if (a_nodes[0] < 0 || a_nodes[1] < 0 || b_nodes[0] < 0 ||
        b_nodes[1] < 0 ||
        static_cast<std::size_t>(a_nodes[0]) >= x_r_old.size() ||
        static_cast<std::size_t>(a_nodes[1]) >= x_r_old.size() ||
        static_cast<std::size_t>(b_nodes[0]) >= x_r_old.size() ||
        static_cast<std::size_t>(b_nodes[1]) >= x_r_old.size()) {
      ++stats.unmatched_faces;
      continue;
    }
    std::array<int, 2> a_sorted = a_nodes;
    std::array<int, 2> b_sorted = b_nodes;
    std::sort(a_sorted.begin(), a_sorted.end());
    std::sort(b_sorted.begin(), b_sorted.end());
    if (a_sorted == b_sorted) {
      ++stats.shared_faces;
      continue;
    }
    const double direct_distance =
        node_pair_distance2(x_r_old, x_z_old, a_nodes[0], b_nodes[0]) +
        node_pair_distance2(x_r_old, x_z_old, a_nodes[1], b_nodes[1]);
    const double reversed_distance =
        node_pair_distance2(x_r_old, x_z_old, a_nodes[0], b_nodes[1]) +
        node_pair_distance2(x_r_old, x_z_old, a_nodes[1], b_nodes[0]);
    const bool direct = direct_distance <= reversed_distance;
    ++stats.duplicate_faces;
    const int b0 = direct ? b_nodes[0] : b_nodes[1];
    const int b1 = direct ? b_nodes[1] : b_nodes[0];
    stats.cloned_nodes +=
        clone_duplicate_node_target(x_r_old, x_z_old, x_r_new, x_z_new,
                                    a_nodes[0], b0)
            ? 1
            : 0;
    stats.cloned_nodes +=
        clone_duplicate_node_target(x_r_old, x_z_old, x_r_new, x_z_new,
                                    a_nodes[1], b1)
            ? 1
            : 0;
  }
  return stats;
}

std::vector<int> moved_nodes_from_target(const std::vector<double>& x_r_old,
                                         const std::vector<double>& x_z_old,
                                         const std::vector<double>& x_r_new,
                                         const std::vector<double>& x_z_new) {
  std::vector<int> nodes;
  const int n_nodes = static_cast<int>(x_r_old.size());
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (std::hypot(x_r_new[idx] - x_r_old[idx],
                   x_z_new[idx] - x_z_old[idx]) > 1.0e-30) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

bool cell_touches_node_mask(const core::State& state,
                            const int cell,
                            const std::vector<std::uint8_t>& node_mask) {
  const auto& mb = *state.mesh.topo.multiblock;
  if (cell < 0 || cell >= state.mesh.topo.n_cells) {
    return false;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int next =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  for (int p = off; p < next; ++p) {
    const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(p)];
    if (node >= 0 && static_cast<std::size_t>(node) < node_mask.size() &&
        node_mask[static_cast<std::size_t>(node)] != 0U) {
      return true;
    }
  }
  return false;
}

bool cell_touches_node_id(const core::State& state,
                          const int cell,
                          const int target_node) {
  const auto& mb = *state.mesh.topo.multiblock;
  if (cell < 0 || cell >= state.mesh.topo.n_cells || target_node < 0) {
    return false;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int next =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  for (int p = off; p < next; ++p) {
    if (mb.cell_node_csr_indices[static_cast<std::size_t>(p)] ==
        target_node) {
      return true;
    }
  }
  return false;
}

double node_displacement(const std::vector<double>& x_r_old,
                         const std::vector<double>& x_z_old,
                         const std::vector<double>& x_r_new,
                         const std::vector<double>& x_z_new,
                         const int node) {
  if (node < 0 || static_cast<std::size_t>(node) >= x_r_old.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const std::size_t idx = static_cast<std::size_t>(node);
  return std::hypot(x_r_new[idx] - x_r_old[idx],
                    x_z_new[idx] - x_z_old[idx]);
}

bool moved_mask_contains(const std::vector<std::uint8_t>& moved_mask,
                         const int node) {
  return node >= 0 && static_cast<std::size_t>(node) < moved_mask.size() &&
         moved_mask[static_cast<std::size_t>(node)] != 0U;
}

bool coords_match_abs_1e12(const std::vector<double>& x_r,
                           const std::vector<double>& x_z,
                           const int a,
                           const int b) {
  if (a < 0 || b < 0 || static_cast<std::size_t>(a) >= x_r.size() ||
      static_cast<std::size_t>(b) >= x_r.size()) {
    return false;
  }
  return std::abs(x_r[static_cast<std::size_t>(a)] -
                  x_r[static_cast<std::size_t>(b)]) <= 1.0e-12 &&
         std::abs(x_z[static_cast<std::size_t>(a)] -
                  x_z[static_cast<std::size_t>(b)]) <= 1.0e-12;
}

std::string node_probe_fields(const char* prefix,
                              const int node,
                              const std::vector<std::uint8_t>& moved_mask,
                              const std::vector<double>& x_r_old,
                              const std::vector<double>& x_z_old,
                              const std::vector<double>& x_r_new,
                              const std::vector<double>& x_z_new) {
  std::ostringstream os;
  os << " " << prefix << "_id=" << node;
  if (node >= 0 && static_cast<std::size_t>(node) < x_r_old.size()) {
    os << " " << prefix << "_r=" << sci(x_r_old[static_cast<std::size_t>(node)])
       << " " << prefix << "_z=" << sci(x_z_old[static_cast<std::size_t>(node)]);
  } else {
    os << " " << prefix << "_r=nan"
       << " " << prefix << "_z=nan";
  }
  os << " " << prefix << "_moved="
     << (moved_mask_contains(moved_mask, node) ? 1 : 0)
     << " " << prefix << "_disp="
     << sci(node_displacement(x_r_old, x_z_old, x_r_new, x_z_new, node));
  return os.str();
}

void log_ring7_probe_face(
    const core::State& state,
    const int face_index,
    const char* face_source,
    const int cell_a,
    const int local_a,
    const int cell_b,
    const int local_b,
    const std::vector<std::uint8_t>& moved_mask,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    const std::vector<double>& x_r_new,
    const std::vector<double>& x_z_new) {
  const auto& mb = *state.mesh.topo.multiblock;
  const int block_a = cell_block_id(mb, cell_a);
  const int block_b = cell_block_id(mb, cell_b);
  const std::array<int, 2> a_nodes = face_node_ids(state, cell_a, local_a);
  const std::array<int, 2> b_nodes = face_node_ids(state, cell_b, local_b);
  bool direct_pairing = true;
  if (a_nodes[0] >= 0 && a_nodes[1] >= 0 && b_nodes[0] >= 0 &&
      b_nodes[1] >= 0) {
    const double direct =
        node_pair_distance2(x_r_old, x_z_old, a_nodes[0], b_nodes[0]) +
        node_pair_distance2(x_r_old, x_z_old, a_nodes[1], b_nodes[1]);
    const double reversed =
        node_pair_distance2(x_r_old, x_z_old, a_nodes[0], b_nodes[1]) +
        node_pair_distance2(x_r_old, x_z_old, a_nodes[1], b_nodes[0]);
    direct_pairing = direct <= reversed;
  }
  const int b_pair0 = direct_pairing ? b_nodes[0] : b_nodes[1];
  const int b_pair1 = direct_pairing ? b_nodes[1] : b_nodes[0];
  std::ostringstream os;
  os << "[ring7_seam_probe]"
     << " face_index=" << face_index
     << " face_source=" << (face_source != nullptr ? face_source : "?")
     << " cell_a=" << cell_a
     << " block_a=" << block_a
     << " local_a=" << local_a
     << " cell_b=" << cell_b
     << " block_b=" << block_b
     << " local_b=" << local_b
     << " a_nodes=" << a_nodes[0] << "," << a_nodes[1]
     << " b_nodes=" << b_nodes[0] << "," << b_nodes[1]
     << " pair_order=" << (direct_pairing ? "direct" : "reversed")
     << " pair0_same_id=" << (a_nodes[0] == b_pair0 ? 1 : 0)
     << " pair0_duplicate_id=" << (a_nodes[0] != b_pair0 ? 1 : 0)
     << " pair0_coords_match_1e12="
     << (coords_match_abs_1e12(x_r_old, x_z_old, a_nodes[0], b_pair0) ? 1 : 0)
     << " pair1_same_id=" << (a_nodes[1] == b_pair1 ? 1 : 0)
     << " pair1_duplicate_id=" << (a_nodes[1] != b_pair1 ? 1 : 0)
     << " pair1_coords_match_1e12="
     << (coords_match_abs_1e12(x_r_old, x_z_old, a_nodes[1], b_pair1) ? 1 : 0)
     << node_probe_fields("a0", a_nodes[0], moved_mask, x_r_old, x_z_old,
                          x_r_new, x_z_new)
     << node_probe_fields("a1", a_nodes[1], moved_mask, x_r_old, x_z_old,
                          x_r_new, x_z_new)
     << node_probe_fields("b0", b_nodes[0], moved_mask, x_r_old, x_z_old,
                          x_r_new, x_z_new)
     << node_probe_fields("b1", b_nodes[1], moved_mask, x_r_old, x_z_old,
                          x_r_new, x_z_new);
  core::log_info(os.str());
}

bool local_face_corners(const core::State& state,
                        const int cell,
                        const int local_face,
                        int* corner0,
                        int* corner1) {
  if (local_face < 0 || local_face >= mesh::kMeshTopoCellStorageSlots) {
    return false;
  }
  return mesh::mesh_topo_active_local_face_corners(
      active_nverts(state, cell), local_face, corner0, corner1);
}

double face_swept_volume(const core::State& state,
                         const int cell,
                         const int local_face,
                         const std::vector<double>& x_r_old,
                         const std::vector<double>& x_z_old,
                         const std::vector<double>& x_r_new,
                         const std::vector<double>& x_z_new) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const auto& mb = *state.mesh.topo.multiblock;
  int c0 = -1;
  int c1 = -1;
  if (!local_face_corners(state, cell, local_face, &c0, &c1)) {
    return 0.0;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int n0 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c0)];
  const int n1 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c1)];
  if (n0 < 0 || n1 < 0 ||
      static_cast<std::size_t>(n0) >= x_r_old.size() ||
      static_cast<std::size_t>(n1) >= x_r_old.size()) {
    return 0.0;
  }
  const auto edge = [](const double r0,
                       const double z0,
                       const double r1,
                       const double z1) {
    return (r0 + r1) * (r0 * z1 - r1 * z0);
  };
  const std::size_t i0 = static_cast<std::size_t>(n0);
  const std::size_t i1 = static_cast<std::size_t>(n1);
  const double old_edge = edge(x_r_old[i0], x_z_old[i0],
                              x_r_old[i1], x_z_old[i1]);
  const double new_edge = edge(x_r_new[i0], x_z_new[i0],
                              x_r_new[i1], x_z_new[i1]);
  const int orientation =
      static_cast<std::size_t>(cell) < mb.cell_orientation_sign.size()
          ? mb.cell_orientation_sign[static_cast<std::size_t>(cell)]
          : 1;
  return static_cast<double>(orientation) * (pi / 3.0) *
         (new_edge - old_edge);
}

double face_swept_quad_polygon_delta(
    const core::State& state,
    const int cell,
    const int local_face,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    const std::vector<double>& x_r_new,
    const std::vector<double>& x_z_new) {
  const auto& mb = *state.mesh.topo.multiblock;
  int c0 = -1;
  int c1 = -1;
  if (!local_face_corners(state, cell, local_face, &c0, &c1)) {
    return 0.0;
  }
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int n0 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c0)];
  const int n1 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c1)];
  if (n0 < 0 || n1 < 0 ||
      static_cast<std::size_t>(n0) >= x_r_old.size() ||
      static_cast<std::size_t>(n1) >= x_r_old.size()) {
    return 0.0;
  }
  const std::size_t i0 = static_cast<std::size_t>(n0);
  const std::size_t i1 = static_cast<std::size_t>(n1);
  Ring7RzPoint swept[4] = {
      {x_r_old[i0], x_z_old[i0]},
      {x_r_old[i1], x_z_old[i1]},
      {x_r_new[i1], x_z_new[i1]},
      {x_r_new[i0], x_z_new[i0]},
  };
  const int orientation =
      static_cast<std::size_t>(cell) < mb.cell_orientation_sign.size()
          ? mb.cell_orientation_sign[static_cast<std::size_t>(cell)]
          : 1;
  return -static_cast<double>(orientation) *
         axisymmetric_swept_volume_polygon(swept, 4);
}

void corner_split_weights(const core::State& state,
                          const int cell,
                          const int local_face,
                          const std::vector<double>& corner_volume,
                          int* corner0,
                          int* corner1,
                          double* weight0,
                          double* weight1) {
  *corner0 = 0;
  *corner1 = 1;
  *weight0 = 0.5;
  *weight1 = 0.5;
  if (!local_face_corners(state, cell, local_face, corner0, corner1)) {
    return;
  }
  const std::size_t base = static_cast<std::size_t>(cell) * 4U;
  if (cell >= 0 && base + static_cast<std::size_t>(*corner1) <
                       corner_volume.size()) {
    const double v0 =
        std::abs(corner_volume[base + static_cast<std::size_t>(*corner0)]);
    const double v1 =
        std::abs(corner_volume[base + static_cast<std::size_t>(*corner1)]);
    const double sum = v0 + v1;
    if (sum > 0.0 && std::isfinite(sum)) {
      *weight0 = v0 / sum;
      *weight1 = v1 / sum;
    }
  }
}

bool cell_is_central_excluded(const core::State& state,
                              const mesh::MultiBlockTopology& mb,
                              const int cell) {
  if (cell < 0 || cell >= state.mesh.topo.n_cells) {
    return true;
  }
  if (cell_role(mb, cell) == mesh::BlockRole::CENTRAL_CORE) {
    return true;
  }
  const auto& pc = state.central_pseudo_core;
  const std::size_t idx = static_cast<std::size_t>(cell);
  return (idx < pc.member_mask.size() && pc.member_mask[idx] != 0U) ||
         (idx < pc.passive_mask.size() && pc.passive_mask[idx] != 0U) ||
         (idx < pc.inactive_member_mask.size() &&
          pc.inactive_member_mask[idx] != 0U) ||
         cell == pc.representative_cell;
}

std::vector<int> central_excluded_cells(const core::State& state) {
  std::vector<int> cells;
  if (!state.mesh.topo.multiblock.has_value()) {
    return cells;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  for (int cell = 0; cell < state.mesh.topo.n_cells; ++cell) {
    if (cell_is_central_excluded(state, mb, cell)) {
      cells.push_back(cell);
    }
  }
  sort_unique(cells);
  return cells;
}

std::vector<std::uint8_t> nodes_of_cells_mask(const core::State& state,
                                              const std::vector<int>& cells) {
  std::vector<std::uint8_t> node_mask(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  if (!state.mesh.topo.multiblock.has_value()) {
    return node_mask;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  for (const int cell : cells) {
    if (cell < 0 || cell >= state.mesh.topo.n_cells) {
      continue;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(state, cell);
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (node >= 0 && node < state.mesh.topo.n_nodes) {
        node_mask[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }
  return node_mask;
}

void hash_mix(std::uint64_t* hash, const std::uint64_t value) {
  *hash ^= value;
  *hash *= 1099511628211ULL;
}

std::uint64_t double_bits(const long double value) {
  const double narrowed = static_cast<double>(value);
  std::uint64_t bits = 0ULL;
  static_assert(sizeof(bits) == sizeof(narrowed), "double bit hash size");
  std::memcpy(&bits, &narrowed, sizeof(bits));
  return bits;
}

std::uint64_t central_core_hash(const core::State& state,
                                const std::vector<int>& cells,
                                const SeamTotals& totals) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash_mix(&hash, static_cast<std::uint64_t>(cells.size()));
  for (const int cell : cells) {
    hash_mix(&hash, static_cast<std::uint64_t>(cell + 1));
  }
  const auto& pc = state.central_pseudo_core;
  hash_mix(&hash, static_cast<std::uint64_t>(pc.configured ? 1U : 0U));
  hash_mix(&hash, static_cast<std::uint64_t>(pc.built ? 1U : 0U));
  hash_mix(&hash, static_cast<std::uint64_t>(pc.valid ? 1U : 0U));
  hash_mix(&hash, static_cast<std::uint64_t>(pc.representative_cell + 1));
  hash_mix(&hash, static_cast<std::uint64_t>(pc.member_ring_count + 1));
  hash_mix(&hash, double_bits(totals.mass));
  hash_mix(&hash, double_bits(totals.total_energy));
  hash_mix(&hash, double_bits(totals.momentum_r));
  hash_mix(&hash, double_bits(totals.momentum_z));
  return hash;
}

Ring7PacketGeometryDiagnostics build_ring7_packet_geometry_diagnostics(
    const core::State& state,
    const Ring7SeamPatch& seam_patch,
    const Ring7SeamNodeSharing& seam_node_sharing,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    const std::vector<double>& x_r_new,
    const std::vector<double>& x_z_new,
    const std::vector<double>& mass,
    const std::vector<double>& ee,
    const std::vector<double>& ei,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    const std::vector<double>& corner_mass,
    const std::vector<double>& corner_volume,
    std::vector<Ring7PacketRecord>* out_packets,
    const bool emit_probe = true) {
  (void)seam_patch;
  Ring7PacketGeometryDiagnostics diag;
  if (!state.mesh.topo.multiblock.has_value()) {
    ++diag.invariant_violations;
    return diag;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  diag.seam_shared_node_faces = seam_node_sharing.shared_faces;
  diag.seam_duplicate_node_faces = seam_node_sharing.duplicate_faces;
  diag.seam_unmatched_node_faces = seam_node_sharing.unmatched_faces;
  diag.seam_cloned_nodes = seam_node_sharing.cloned_nodes;
  const std::vector<int> moved_nodes =
      moved_nodes_from_target(x_r_old, x_z_old, x_r_new, x_z_new);
  diag.moved_nodes = static_cast<int>(moved_nodes.size());
  std::vector<std::uint8_t> moved_mask(static_cast<std::size_t>(n_nodes), 0U);
  for (const int node : moved_nodes) {
    if (node >= 0 && node < n_nodes) {
      moved_mask[static_cast<std::size_t>(node)] = 1U;
    }
  }

  std::vector<std::uint8_t> patch_mask(static_cast<std::size_t>(n_cells), 0U);
  std::vector<int> patch_cells;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (cell_touches_node_mask(state, cell, moved_mask)) {
      TENRYU_ASSERT(!cell_is_central_excluded(state, mb, cell),
                    "Ring7 moved-node incident patch reached central core cell");
      patch_cells.push_back(cell);
    }
  }
  sort_unique(patch_cells);
  for (const int cell : patch_cells) {
    if (cell >= 0 && cell < n_cells) {
      patch_mask[static_cast<std::size_t>(cell)] = 1U;
    }
  }
  diag.patch_cells = static_cast<int>(patch_cells.size());

  for (const int node : moved_nodes) {
    bool found_incident = false;
    for (int cell = 0; cell < n_cells; ++cell) {
      if (!cell_touches_node_id(state, cell, node)) {
        continue;
      }
      found_incident = true;
      if (patch_mask[static_cast<std::size_t>(cell)] == 0U) {
        ++diag.invariant_violations;
      }
    }
    if (!found_incident) {
      ++diag.invariant_violations;
    }
  }

  const std::vector<int> central_cells = central_excluded_cells(state);
  const std::vector<std::uint8_t> central_node_mask =
      nodes_of_cells_mask(state, central_cells);
  diag.central_before =
      cell_totals(state, central_cells, mass, ee, ei, v_r, v_z, corner_mass);
  diag.central_after = diag.central_before;
  diag.central_hash_before =
      central_core_hash(state, central_cells, diag.central_before);
  diag.central_hash_after =
      central_core_hash(state, central_cells, diag.central_after);
  for (const int cell : patch_cells) {
    if (cell_is_central_excluded(state, mb, cell)) {
      ++diag.central_patch_cells;
      ++diag.invariant_violations;
    }
  }
  for (const int node : moved_nodes) {
    if (node >= 0 && static_cast<std::size_t>(node) < central_node_mask.size() &&
        central_node_mask[static_cast<std::size_t>(node)] != 0U) {
      ++diag.central_patch_nodes;
      ++diag.invariant_violations;
    }
  }

  std::vector<long double> packet_delta(static_cast<std::size_t>(n_cells),
                                        0.0L);
  std::vector<Ring7PacketRecord> packets;
  int packet_id = 0;
  int cross_block_unique_faces = 0;
  for (const auto& face : mb.unique_internal_faces) {
    const int block_a = cell_block_id(mb, face.cell_a);
    const int block_b = cell_block_id(mb, face.cell_b);
    if (block_a >= 0 && block_b >= 0 && block_a != block_b) {
      ++cross_block_unique_faces;
    }
  }
  if (emit_probe) {
    core::log_info("[ring7_seam_probe] n_unique_internal_faces=" +
                   std::to_string(mb.unique_internal_faces.size()) +
                   " n_seam_faces_block_a!=block_b=" +
                   std::to_string(cross_block_unique_faces));
  }
  int leak_probe_faces_logged = 0;
  for (std::size_t face_index = 0;
       face_index < mb.unique_internal_faces.size();
       ++face_index) {
    const auto& face = mb.unique_internal_faces[face_index];
    const int a = face.cell_a;
    const int b = face.cell_b;
    if (a < 0 || b < 0 || a >= n_cells || b >= n_cells) {
      ++diag.invariant_violations;
      continue;
    }
    const bool a_in = patch_mask[static_cast<std::size_t>(a)] != 0U;
    const bool b_in = patch_mask[static_cast<std::size_t>(b)] != 0U;
    if (a_in && b_in) {
      const double swept_a =
          face_swept_volume(state, a, face.local_a, x_r_old, x_z_old,
                            x_r_new, x_z_new);
      const double swept_b =
          face_swept_volume(state, b, face.local_b, x_r_old, x_z_old,
                            x_r_new, x_z_new);
      packet_delta[static_cast<std::size_t>(a)] += swept_a;
      packet_delta[static_cast<std::size_t>(b)] += swept_b;
      ++diag.patch_faces;
      if (is_shell_cell(mb, a) != is_shell_cell(mb, b) &&
          configured_seam_block_pair(mb, cell_block_id(mb, a),
                                     cell_block_id(mb, b))) {
        ++diag.seam_physical_faces;
      }
      const double canonical_swept = -swept_a;
      const bool a_to_b = canonical_swept >= 0.0;
      const int donor = a_to_b ? a : b;
      const int acceptor = a_to_b ? b : a;
      const int split_cell = donor;
      const int split_face = a_to_b ? face.local_a : face.local_b;
      int split0 = 0;
      int split1 = 1;
      double w0 = 0.5;
      double w1 = 0.5;
      corner_split_weights(state, split_cell, split_face, corner_volume,
                           &split0, &split1, &w0, &w1);
      const int donor_node0 = cell_corner_node_id(state, donor, split0);
      const int donor_node1 = cell_corner_node_id(state, donor, split1);
      const int accept0 =
          matching_cell_corner(state, acceptor, donor_node0, x_r_old, x_z_old);
      const int accept1 =
          matching_cell_corner(state, acceptor, donor_node1, x_r_old, x_z_old);
      const double transfer_volume = std::abs(canonical_swept);
      if (accept0 < 0 || accept1 < 0) {
        ++diag.invariant_violations;
      } else {
        packets.push_back({donor, acceptor, split0, accept0, a, face.local_a,
                           packet_id++, transfer_volume * w0});
        packets.push_back({donor, acceptor, split1, accept1, a, face.local_a,
                           packet_id++, transfer_volume * w1});
      }
      continue;
    }
    if (a_in != b_in) {
      const double swept =
          a_in ? face_swept_volume(state, a, face.local_a, x_r_old, x_z_old,
                                   x_r_new, x_z_new)
               : face_swept_volume(state, b, face.local_b, x_r_old, x_z_old,
                                   x_r_new, x_z_new);
      ++diag.boundary_faces;
      diag.boundary_swept_sum += swept;
      diag.boundary_swept_abs += std::abs(swept);
      if (std::abs(swept) > 1.0e-250) {
        ++diag.boundary_nonzero_faces;
        ++diag.invariant_violations;
        if (emit_probe && leak_probe_faces_logged < 4) {
          log_ring7_probe_face(state,
                               static_cast<int>(face_index),
                               "unique_internal",
                               a,
                               face.local_a,
                               b,
                               face.local_b,
                               moved_mask,
                               x_r_old,
                               x_z_old,
                               x_r_new,
                               x_z_new);
          ++leak_probe_faces_logged;
        }
        if (diag.first_bad_boundary_cell < 0) {
          diag.first_bad_boundary_cell = a_in ? a : b;
          diag.first_bad_boundary_face = a_in ? face.local_a : face.local_b;
        }
      }
    }
  }
  for (std::size_t face_index = 0; face_index < mb.boundary_faces.size();
       ++face_index) {
    const auto& face = mb.boundary_faces[face_index];
    const int cell = face.cell_a;
    if (cell < 0 || cell >= n_cells ||
        patch_mask[static_cast<std::size_t>(cell)] == 0U) {
      continue;
    }
    const double swept =
        face_swept_volume(state, cell, face.local_a, x_r_old, x_z_old,
                          x_r_new, x_z_new);
    ++diag.boundary_faces;
    diag.boundary_swept_sum += swept;
    diag.boundary_swept_abs += std::abs(swept);
    if (std::abs(swept) > 1.0e-250) {
      ++diag.boundary_nonzero_faces;
      ++diag.invariant_violations;
      if (emit_probe && leak_probe_faces_logged < 4) {
        log_ring7_probe_face(state,
                             static_cast<int>(face_index),
                             "boundary",
                             cell,
                             face.local_a,
                             -1,
                             -1,
                             moved_mask,
                             x_r_old,
                             x_z_old,
                             x_r_new,
                             x_z_new);
        ++leak_probe_faces_logged;
      }
      if (diag.first_bad_boundary_cell < 0) {
        diag.first_bad_boundary_cell = cell;
        diag.first_bad_boundary_face = face.local_a;
      }
    }
  }
  std::sort(packets.begin(), packets.end(), [](const auto& a, const auto& b) {
    return std::array<int, 7>{a.donor_cell,
                              a.acceptor_cell,
                              a.donor_corner,
                              a.acceptor_corner,
                              a.owner_cell,
                              a.owner_face,
                              a.packet_id} <
           std::array<int, 7>{b.donor_cell,
                              b.acceptor_cell,
                              b.donor_corner,
                              b.acceptor_corner,
                              b.owner_cell,
                              b.owner_face,
                              b.packet_id};
  });
  diag.packets = static_cast<int>(packets.size());
  if (out_packets != nullptr) {
    *out_packets = packets;
  }

  for (const int cell : patch_cells) {
    const long double source =
        std::abs(signed_cell_volume_from_coords(state, x_r_old, x_z_old, cell));
    const long double target =
        std::abs(signed_cell_volume_from_coords(state, x_r_new, x_z_new, cell));
    const long double epsilon =
        (target - source) - packet_delta[static_cast<std::size_t>(cell)];
    diag.max_geom_epsilon =
        std::max(diag.max_geom_epsilon,
                 static_cast<double>(std::abs(epsilon)));
  }
  return diag;
}

std::vector<int> all_cell_ids(const int n_cells) {
  std::vector<int> cells;
  cells.reserve(static_cast<std::size_t>(std::max(n_cells, 0)));
  for (int c = 0; c < n_cells; ++c) {
    cells.push_back(c);
  }
  return cells;
}

double clamp01_host(const double x) {
  if (!std::isfinite(x)) {
    return 0.0;
  }
  return std::min(1.0, std::max(0.0, x));
}

double ideal_cv_i_mass(const core::Config& cfg) {
  if (cfg.materials.materials.empty()) {
    return 0.0;
  }
  const auto& mat = cfg.materials.materials.front();
  const double gamma = std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-30);
  const double A = std::max(mat.A, 1.0e-30);
  return core::constants::eV_to_erg /
         (A * core::constants::proton_mass * (gamma - 1.0));
}

double ideal_cv_e_mass(const core::Config& cfg, const double zbar) {
  const double cv_i = ideal_cv_i_mass(cfg);
  return std::max(zbar, 0.0) * cv_i;
}

double cell_cv_e(const core::Config& cfg,
                 const std::vector<double>& cv_e,
                 const std::vector<double>& zbar,
                 const int cell) {
  if (cell >= 0 && static_cast<std::size_t>(cell) < cv_e.size() &&
      cv_e[static_cast<std::size_t>(cell)] > 0.0) {
    return cv_e[static_cast<std::size_t>(cell)];
  }
  const double z =
      (cell >= 0 && static_cast<std::size_t>(cell) < zbar.size())
          ? zbar[static_cast<std::size_t>(cell)]
          : 1.0;
  return ideal_cv_e_mass(cfg, z);
}

double cell_cv_i(const core::Config& cfg,
                 const std::vector<double>& cv_i,
                 const int cell) {
  if (cell >= 0 && static_cast<std::size_t>(cell) < cv_i.size() &&
      cv_i[static_cast<std::size_t>(cell)] > 0.0) {
    return cv_i[static_cast<std::size_t>(cell)];
  }
  return ideal_cv_i_mass(cfg);
}

double source_corner_volume(const core::State& state,
                            const std::vector<double>& corner_volume,
                            const std::vector<double>& x_r,
                            const std::vector<double>& x_z,
                            const int cell,
                            const int corner) {
  const std::size_t idx =
      static_cast<std::size_t>(cell) * 4U + static_cast<std::size_t>(corner);
  if (cell >= 0 && corner >= 0 && idx < corner_volume.size() &&
      corner_volume[idx] > 0.0 && std::isfinite(corner_volume[idx])) {
    return corner_volume[idx];
  }
  const int nverts = active_nverts(state, cell);
  if (nverts <= 0) {
    return 0.0;
  }
  const double v =
      std::abs(signed_cell_volume_from_coords(state, x_r, x_z, cell));
  return v / static_cast<double>(nverts);
}

std::vector<double> corner_volumes_for_coords(
    const core::State& state,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const std::vector<double>& reference_corner_volume) {
  const int n_cells = state.mesh.topo.n_cells;
  std::vector<double> corner_volume(static_cast<std::size_t>(n_cells) * 4U,
                                    0.0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts = active_nverts(state, cell);
    if (nverts <= 0) {
      continue;
    }
    const double cell_volume =
        std::abs(signed_cell_volume_from_coords(state, x_r, x_z, cell));
    const std::size_t base = static_cast<std::size_t>(cell) * 4U;
    double ref_sum = 0.0;
    for (int k = 0; k < nverts; ++k) {
      if (base + static_cast<std::size_t>(k) <
          reference_corner_volume.size()) {
        ref_sum += std::abs(
            reference_corner_volume[base + static_cast<std::size_t>(k)]);
      }
    }
    if (ref_sum > 0.0 && std::isfinite(ref_sum)) {
      for (int k = 0; k < nverts; ++k) {
        const std::size_t idx = base + static_cast<std::size_t>(k);
        const double fraction =
            idx < reference_corner_volume.size()
                ? std::abs(reference_corner_volume[idx]) / ref_sum
                : 0.0;
        corner_volume[idx] = cell_volume * fraction;
      }
    } else {
      const double share = cell_volume / static_cast<double>(nverts);
      for (int k = 0; k < nverts; ++k) {
        corner_volume[base + static_cast<std::size_t>(k)] = share;
      }
    }
  }
  return corner_volume;
}

bool packets_pass_geometric_positivity(
    const core::State& state,
    const std::vector<Ring7PacketRecord>& packets,
    const std::vector<double>& corner_volume,
    const std::vector<double>& x_r,
    const std::vector<double>& x_z,
    const double eps) {
  std::vector<long double> outgoing(
      static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U, 0.0L);
  for (const auto& packet : packets) {
    if (packet.donor_cell < 0 || packet.donor_corner < 0 ||
        packet.donor_corner >= 4 || !(packet.volume >= 0.0) ||
        !std::isfinite(packet.volume)) {
      return false;
    }
    const std::size_t idx = static_cast<std::size_t>(packet.donor_cell) * 4U +
                            static_cast<std::size_t>(packet.donor_corner);
    if (idx >= outgoing.size()) {
      return false;
    }
    outgoing[idx] += static_cast<long double>(packet.volume);
  }
  const long double keep = std::max(0.0, 1.0 - eps);
  for (const auto& packet : packets) {
    const std::size_t idx = static_cast<std::size_t>(packet.donor_cell) * 4U +
                            static_cast<std::size_t>(packet.donor_corner);
    const long double available = source_corner_volume(
        state, corner_volume, x_r, x_z, packet.donor_cell,
        packet.donor_corner);
    if (!(available > 0.0L) || outgoing[idx] > keep * available) {
      return false;
    }
  }
  return true;
}

void ensure_corner_mass_work(const core::State& state,
                             const std::vector<double>& mass,
                             std::vector<double>& corner_mass) {
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t expected = static_cast<std::size_t>(n_cells) * 4U;
  if (corner_mass.size() == expected) {
    return;
  }
  corner_mass.assign(expected, 0.0);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts = active_nverts(state, cell);
    if (nverts <= 0 || static_cast<std::size_t>(cell) >= mass.size()) {
      continue;
    }
    const double share = mass[static_cast<std::size_t>(cell)] /
                         static_cast<double>(nverts);
    for (int k = 0; k < nverts; ++k) {
      corner_mass[static_cast<std::size_t>(cell) * 4U +
                  static_cast<std::size_t>(k)] = share;
    }
  }
}

void build_node_mass_momentum(const core::State& state,
                              const std::vector<double>& corner_mass,
                              const std::vector<double>& v_r,
                              const std::vector<double>& v_z,
                              std::vector<long double>& node_mass,
                              std::vector<long double>& node_pr,
                              std::vector<long double>& node_pz) {
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  node_mass.assign(static_cast<std::size_t>(n_nodes), 0.0L);
  node_pr.assign(static_cast<std::size_t>(n_nodes), 0.0L);
  node_pz.assign(static_cast<std::size_t>(n_nodes), 0.0L);
  for (int cell = 0; cell < n_cells; ++cell) {
    const int nverts = active_nverts(state, cell);
    for (int k = 0; k < nverts; ++k) {
      const int node = cell_corner_node_id(state, cell, k);
      const std::size_t cidx =
          static_cast<std::size_t>(cell) * 4U + static_cast<std::size_t>(k);
      if (node < 0 || node >= n_nodes || cidx >= corner_mass.size()) {
        continue;
      }
      const std::size_t nidx = static_cast<std::size_t>(node);
      const long double cm = corner_mass[cidx];
      node_mass[nidx] += cm;
      node_pr[nidx] += cm * v_r[nidx];
      node_pz[nidx] += cm * v_z[nidx];
    }
  }
}

long double cell_kinetic_energy(const core::State& state,
                                const std::vector<double>& corner_mass,
                                const std::vector<double>& v_r,
                                const std::vector<double>& v_z,
                                const int cell) {
  long double ke = 0.0L;
  const int nverts = active_nverts(state, cell);
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_corner_node_id(state, cell, k);
    const std::size_t cidx =
        static_cast<std::size_t>(cell) * 4U + static_cast<std::size_t>(k);
    if (node < 0 || static_cast<std::size_t>(node) >= v_r.size() ||
        cidx >= corner_mass.size()) {
      continue;
    }
    const long double cm = corner_mass[cidx];
    const long double vr = v_r[static_cast<std::size_t>(node)];
    const long double vz = v_z[static_cast<std::size_t>(node)];
    ke += 0.5L * cm * (vr * vr + vz * vz);
  }
  return ke;
}

Ring7PacketRemapDiagnostics apply_ring7_packet_remap_transaction(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const Ring7SeamPatch& patch,
    const Ring7SeamNodeSharing& seam_node_sharing,
    const Ring7PacketGeometryDiagnostics& packet_diag,
    const std::vector<Ring7PacketRecord>& packets,
    const std::vector<double>& x_r_old,
    const std::vector<double>& x_z_old,
    const std::vector<double>& x_r_new,
    const std::vector<double>& x_z_new,
    std::vector<double> rho,
    std::vector<double> mass,
    std::vector<double> ee,
    std::vector<double> ei,
    std::vector<double> Te,
    std::vector<double> Ti,
    std::vector<double> Pe,
    std::vector<double> Pi,
    std::vector<double> v_r,
    std::vector<double> v_z,
    std::vector<double> corner_mass,
    const std::vector<double>& corner_volume,
    std::vector<double> gas_tracer_Y,
    std::vector<double> zbar,
    const std::vector<double>& cv_e,
    const std::vector<double>& cv_i,
    std::vector<double> cs,
    std::vector<double> hllc_mom_z_cell) {
  Ring7PacketRemapDiagnostics remap;
  (void)packets;
  if (eos_ctx != nullptr && eos_ctx->any_table) {
    core::log_info("[ring7_seam_remap] status=rejected_table_eos");
    return remap;
  }
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    core::log_info("[ring7_seam_remap] status=rejected_per_material_mode");
    return remap;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (packet_diag.invariant_violations != 0 ||
      packet_diag.boundary_nonzero_faces != 0 ||
      !(packet_diag.max_geom_epsilon <= 1.0e-12) ||
      packet_diag.packets <= 0 ||
      static_cast<int>(mass.size()) < n_cells ||
      static_cast<int>(rho.size()) < n_cells ||
      static_cast<int>(ee.size()) < n_cells ||
      static_cast<int>(ei.size()) < n_cells ||
      static_cast<int>(v_r.size()) < n_nodes ||
      static_cast<int>(v_z.size()) < n_nodes) {
    return remap;
  }

  ensure_corner_mass_work(state, mass, corner_mass);
  const std::vector<int> all_cells = all_cell_ids(n_cells);
  const std::vector<int> central_cells = central_excluded_cells(state);
  remap.central_before =
      cell_totals(state, central_cells, mass, ee, ei, v_r, v_z, corner_mass);
  remap.central_hash_before =
      central_core_hash(state, central_cells, remap.central_before);
  const SeamTotals totals_before =
      cell_totals(state, all_cells, mass, ee, ei, v_r, v_z, corner_mass);

  const int kmax =
      env_int_value("TENRYU_I1B_RING7_PACKET_SUBCYCLE_KMAX", 8);
  const double pos_eps =
      env_double_value("TENRYU_I1B_RING7_PACKET_POS_EPS", 1.0e-12);
  for (int subcycles = 1; subcycles <= kmax; ++subcycles) {
    Ring7PacketRemapDiagnostics trial;
    trial.subcycles = subcycles;
    trial.central_before = remap.central_before;
    trial.central_hash_before = remap.central_hash_before;

    std::vector<double> rho_out = rho;
    std::vector<double> mass_out = mass;
    std::vector<double> ee_out = ee;
    std::vector<double> ei_out = ei;
    std::vector<double> Te_out = Te;
    std::vector<double> Ti_out = Ti;
    std::vector<double> Pe_out = Pe;
    std::vector<double> Pi_out = Pi;
    std::vector<double> v_r_work = v_r;
    std::vector<double> v_z_work = v_z;
    std::vector<double> corner_mass_work = corner_mass;
    std::vector<double> gas_tracer_out = gas_tracer_Y;
    const bool remap_burn_species =
        cfg.burn.enabled &&
        state.burn_n_host.size() ==
            static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies;
    std::vector<double> burn_n_out = state.burn_n_host;
    const bool remap_hot_e_eps =
        !state.hot_e_eps_cum_host.empty() &&
        state.hot_e_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
    std::vector<double> hot_e_eps_out = state.hot_e_eps_cum_host;
    const bool remap_burn_eps =
        !state.burn_eps_cum_host.empty() &&
        state.burn_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
    std::vector<double> burn_eps_out = state.burn_eps_cum_host;
    std::vector<double> zbar_out = zbar;
    std::vector<double> cs_out = cs;
    std::vector<double> hllc_mom_z_out = hllc_mom_z_cell;
    std::vector<long double> mass_work(mass.begin(), mass.end());
    std::vector<long double> volume_work(static_cast<std::size_t>(n_cells),
                                         0.0L);
    std::vector<long double> total_energy_work(
        static_cast<std::size_t>(n_cells), 0.0L);
    std::vector<long double> ye_mass(static_cast<std::size_t>(n_cells),
                                     0.0L);
    std::vector<long double> tracer_mass(static_cast<std::size_t>(n_cells),
                                         0.0L);
    std::vector<long double> burn_species_mass(
        static_cast<std::size_t>(n_cells) * tenryu::burn::kNumSpecies, 0.0L);
    std::vector<long double> hot_e_eps_mass(static_cast<std::size_t>(n_cells),
                                            0.0L);
    std::vector<long double> burn_eps_mass(static_cast<std::size_t>(n_cells),
                                           0.0L);
    std::vector<long double> zbar_mass(static_cast<std::size_t>(n_cells),
                                       0.0L);
    for (int cell = 0; cell < n_cells; ++cell) {
      const std::size_t idx = static_cast<std::size_t>(cell);
      volume_work[idx] = std::abs(
          signed_cell_volume_from_coords(state, x_r_old, x_z_old, cell));
      const long double internal =
          mass_work[idx] * std::max(ee[idx] + ei[idx], 0.0);
      total_energy_work[idx] =
          internal +
          cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                              cell);
      const double denom = std::max(ee[idx] + ei[idx], 0.0);
      const double ye = denom > 0.0 ? clamp01_host(ee[idx] / denom) : 0.5;
      ye_mass[idx] = mass_work[idx] * ye;
      if (idx < gas_tracer_Y.size() && state.gas_tracer_initialized) {
        tracer_mass[idx] =
            mass_work[idx] * clamp01_host(gas_tracer_Y[idx]);
      }
      if (remap_burn_species) {
        for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
          const std::size_t species_idx =
              idx * tenryu::burn::kNumSpecies + static_cast<std::size_t>(s);
          burn_species_mass[species_idx] =
              mass_work[idx] * std::fmax(state.burn_n_host[species_idx], 0.0);
        }
      }
      if (remap_hot_e_eps) {
        hot_e_eps_mass[idx] =
            mass_work[idx] * std::max(state.hot_e_eps_cum_host[idx], 0.0);
      }
      if (remap_burn_eps) {
        burn_eps_mass[idx] =
            mass_work[idx] * std::max(state.burn_eps_cum_host[idx], 0.0);
      }
      if (idx < zbar.size()) {
        zbar_mass[idx] = mass_work[idx] * std::max(zbar[idx], 0.0);
      }
    }

    std::vector<long double> node_mass;
    std::vector<long double> node_pr;
    std::vector<long double> node_pz;
    build_node_mass_momentum(state, corner_mass_work, v_r_work, v_z_work,
                             node_mass, node_pr, node_pz);

    std::vector<int> remap_cells = patch.p_seam_cells;
    std::vector<double> segment_r0 = x_r_old;
    std::vector<double> segment_z0 = x_z_old;
    bool retry_larger_k = false;
    bool hard_failure = false;

    const auto refresh_node_velocity = [&](const int node) {
      if (node < 0 || static_cast<std::size_t>(node) >= node_mass.size()) {
        return;
      }
      const std::size_t idx = static_cast<std::size_t>(node);
      if (node_mass[idx] > 0.0L) {
        v_r_work[idx] = static_cast<double>(node_pr[idx] / node_mass[idx]);
        v_z_work[idx] = static_cast<double>(node_pz[idx] / node_mass[idx]);
      }
    };

    for (int sub = 0; sub < subcycles; ++sub) {
      const double alpha =
          static_cast<double>(sub + 1) / static_cast<double>(subcycles);
      std::vector<double> segment_r1 = x_r_old;
      std::vector<double> segment_z1 = x_z_old;
      for (int node = 0; node < n_nodes; ++node) {
        const std::size_t idx = static_cast<std::size_t>(node);
        segment_r1[idx] =
            x_r_old[idx] + alpha * (x_r_new[idx] - x_r_old[idx]);
        segment_z1[idx] =
            x_z_old[idx] + alpha * (x_z_new[idx] - x_z_old[idx]);
      }

      const std::vector<double> segment_corner_volume =
          corner_volumes_for_coords(state, segment_r0, segment_z0,
                                    corner_volume);
      std::vector<Ring7PacketRecord> segment_packets;
      const Ring7PacketGeometryDiagnostics segment_diag =
          build_ring7_packet_geometry_diagnostics(state,
                                                  patch,
                                                  seam_node_sharing,
                                                  segment_r0,
                                                  segment_z0,
                                                  segment_r1,
                                                  segment_z1,
                                                  mass,
                                                  ee,
                                                  ei,
                                                  v_r_work,
                                                  v_z_work,
                                                  corner_mass_work,
                                                  segment_corner_volume,
                                                  &segment_packets,
                                                  false);
      if (segment_diag.invariant_violations != 0 ||
          segment_diag.boundary_nonzero_faces != 0 ||
          !(segment_diag.max_geom_epsilon <= 1.0e-12) ||
          segment_packets.empty()) {
        trial.positivity_failed = true;
        hard_failure = true;
        break;
      }
      if (!packets_pass_geometric_positivity(state,
                                             segment_packets,
                                             segment_corner_volume,
                                             segment_r0,
                                             segment_z0,
                                             pos_eps)) {
        trial.positivity_failed = true;
        retry_larger_k = true;
        break;
      }
      for (const auto& packet : segment_packets) {
        const int donor = packet.donor_cell;
        const int acceptor = packet.acceptor_cell;
        const int donor_corner = packet.donor_corner;
        const int acceptor_corner = packet.acceptor_corner;
        remap_cells.push_back(donor);
        remap_cells.push_back(acceptor);
        if (donor < 0 || donor >= n_cells || acceptor < 0 ||
            acceptor >= n_cells || donor_corner < 0 || donor_corner >= 4 ||
            acceptor_corner < 0 || acceptor_corner >= 4) {
          trial.positivity_failed = true;
          hard_failure = true;
          break;
        }
        const std::size_t d = static_cast<std::size_t>(donor);
        const std::size_t a = static_cast<std::size_t>(acceptor);
        const std::size_t dc =
            d * 4U + static_cast<std::size_t>(donor_corner);
        const std::size_t ac =
            a * 4U + static_cast<std::size_t>(acceptor_corner);
        const int donor_node =
            cell_corner_node_id(state, donor, donor_corner);
        const int acceptor_node =
            cell_corner_node_id(state, acceptor, acceptor_corner);
        if (dc >= corner_mass_work.size() || ac >= corner_mass_work.size() ||
            donor_node < 0 || donor_node >= n_nodes || acceptor_node < 0 ||
            acceptor_node >= n_nodes) {
          trial.positivity_failed = true;
          hard_failure = true;
          break;
        }
        const long double dV = static_cast<long double>(packet.volume);
        const long double rho_d =
            mass_work[d] / std::max(volume_work[d], 1.0e-300L);
        const long double dm = rho_d * dV;
        if (!(dm >= 0.0L) || !std::isfinite(static_cast<double>(dm)) ||
            corner_mass_work[dc] + 1.0e-30 < static_cast<double>(dm) ||
            mass_work[d] + 1.0e-30L < dm) {
          trial.positivity_failed = true;
          retry_larger_k = true;
          break;
        }
        const std::size_t dn = static_cast<std::size_t>(donor_node);
        const std::size_t an = static_cast<std::size_t>(acceptor_node);
        const long double udr = v_r_work[dn];
        const long double udz = v_z_work[dn];
        const long double uar = v_r_work[an];
        const long double uaz = v_z_work[an];
        const long double ke_specific = 0.5L * (udr * udr + udz * udz);
        const long double ke_mix =
            0.5L * dm * ((udr - uar) * (udr - uar) +
                         (udz - uaz) * (udz - uaz));
        trial.q_mix += static_cast<double>(std::max(ke_mix, 0.0L));
        const long double kinetic_d =
            cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                                donor);
        const long double internal_d =
            std::max(total_energy_work[d] - kinetic_d, 0.0L);
        const long double e_int_d =
            internal_d / std::max(mass_work[d], 1.0e-300L);
        const long double ye = clamp01_host(static_cast<double>(
            ye_mass[d] / std::max(mass_work[d], 1.0e-300L)));
        const long double tracer =
            state.gas_tracer_initialized
                ? clamp01_host(static_cast<double>(
                      tracer_mass[d] / std::max(mass_work[d], 1.0e-300L)))
                : 0.0L;
        std::array<long double, tenryu::burn::kNumSpecies>
            burn_species_Y_d{};
        if (remap_burn_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            const std::size_t species_idx =
                d * tenryu::burn::kNumSpecies + static_cast<std::size_t>(s);
            burn_species_Y_d[static_cast<std::size_t>(s)] = std::fmax(
                burn_species_mass[species_idx] /
                    std::max(mass_work[d], 1.0e-300L),
                0.0L);
          }
        }
        const long double hot_e_eps_d =
            remap_hot_e_eps
                ? std::max(hot_e_eps_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        const long double burn_eps_d =
            remap_burn_eps
                ? std::max(burn_eps_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        const long double z =
            (d < zbar_mass.size())
                ? std::max(zbar_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        mass_work[d] -= dm;
        mass_work[a] += dm;
        volume_work[d] -= dV;
        volume_work[a] += dV;
        total_energy_work[d] -= dm * (e_int_d + ke_specific);
        total_energy_work[a] += dm * (e_int_d + ke_specific);
        ye_mass[d] -= dm * ye;
        ye_mass[a] += dm * ye;
        if (state.gas_tracer_initialized) {
          tracer_mass[d] -= dm * tracer;
          tracer_mass[a] += dm * tracer;
        }
        if (remap_burn_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            const std::size_t donor_species_idx =
                d * tenryu::burn::kNumSpecies + static_cast<std::size_t>(s);
            const std::size_t acceptor_species_idx =
                a * tenryu::burn::kNumSpecies + static_cast<std::size_t>(s);
            const long double species_transfer =
                dm * burn_species_Y_d[static_cast<std::size_t>(s)];
            burn_species_mass[donor_species_idx] -= species_transfer;
            burn_species_mass[acceptor_species_idx] += species_transfer;
          }
        }
        if (remap_hot_e_eps) {
          hot_e_eps_mass[d] -= dm * hot_e_eps_d;
          hot_e_eps_mass[a] += dm * hot_e_eps_d;
        }
        if (remap_burn_eps) {
          burn_eps_mass[d] -= dm * burn_eps_d;
          burn_eps_mass[a] += dm * burn_eps_d;
        }
        if (!zbar.empty()) {
          zbar_mass[d] -= dm * z;
          zbar_mass[a] += dm * z;
        }
        corner_mass_work[dc] -= static_cast<double>(dm);
        corner_mass_work[ac] += static_cast<double>(dm);
        node_mass[dn] -= dm;
        node_mass[an] += dm;
        node_pr[dn] -= dm * udr;
        node_pz[dn] -= dm * udz;
        node_pr[an] += dm * udr;
        node_pz[an] += dm * udz;
        refresh_node_velocity(donor_node);
        refresh_node_velocity(acceptor_node);
      }
      if (trial.positivity_failed) {
        break;
      }
      segment_r0 = std::move(segment_r1);
      segment_z0 = std::move(segment_z1);
    }
    if (hard_failure) {
      return trial;
    }
    if (retry_larger_k || trial.positivity_failed) {
      remap = trial;
      continue;
    }

    sort_unique(remap_cells);
    const double gamma =
        (!cfg.materials.materials.empty() &&
         cfg.materials.materials.front().ideal_gas_gamma > 1.0)
            ? cfg.materials.materials.front().ideal_gas_gamma
            : (5.0 / 3.0);
    const double gm1 = std::max(gamma - 1.0, 1.0e-30);
    for (const int cell : remap_cells) {
      const std::size_t idx = static_cast<std::size_t>(cell);
      const long double target_v = std::abs(
          signed_cell_volume_from_coords(state, x_r_new, x_z_new, cell));
      trial.max_volume_error = std::max(
          trial.max_volume_error,
          static_cast<double>(std::abs(volume_work[idx] - target_v)));
      if (!(mass_work[idx] > 0.0L) || !(target_v > 0.0L)) {
        trial.positivity_failed = true;
        break;
      }
      const long double kinetic =
          cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                              cell);
      const long double internal = total_energy_work[idx] - kinetic;
      if (idx < zbar_out.size()) {
        zbar_out[idx] = static_cast<double>(
            std::max(zbar_mass[idx] / std::max(mass_work[idx], 1.0e-300L),
                     0.0L));
      }
      const double cv_e_c = cell_cv_e(cfg, cv_e, zbar_out, cell);
      const double cv_i_c = cell_cv_i(cfg, cv_i, cell);
      const long double e_floor =
          static_cast<long double>(std::max(cv_e_c, 0.0) *
                                   std::max(cfg.numerics.floors.Te, 0.0) +
                                   std::max(cv_i_c, 0.0) *
                                   std::max(cfg.numerics.floors.Ti, 0.0));
      if (!(internal >= e_floor * mass_work[idx]) ||
          !std::isfinite(static_cast<double>(internal))) {
        trial.positivity_failed = true;
        break;
      }
      const double ye = clamp01_host(static_cast<double>(
          ye_mass[idx] / std::max(mass_work[idx], 1.0e-300L)));
      const long double e_specific = internal / mass_work[idx];
      const long double ee_new = ye * e_specific;
      const long double ei_new = (1.0L - ye) * e_specific;
      const long double ee_floor =
          static_cast<long double>(std::max(cv_e_c, 0.0) *
                                   std::max(cfg.numerics.floors.Te, 0.0));
      const long double ei_floor =
          static_cast<long double>(std::max(cv_i_c, 0.0) *
                                   std::max(cfg.numerics.floors.Ti, 0.0));
      if (ee_new + 1.0e-20L < ee_floor || ei_new + 1.0e-20L < ei_floor) {
        trial.positivity_failed = true;
        break;
      }
      mass_out[idx] = static_cast<double>(mass_work[idx]);
      rho_out[idx] = static_cast<double>(mass_work[idx] / target_v);
      ee_out[idx] = static_cast<double>(ee_new);
      ei_out[idx] = static_cast<double>(ei_new);
      if (state.gas_tracer_initialized && idx < gas_tracer_out.size()) {
        if (tracer_mass[idx] < -1.0e-20L ||
            tracer_mass[idx] > mass_work[idx] + 1.0e-20L) {
          trial.positivity_failed = true;
          break;
        }
        gas_tracer_out[idx] = clamp01_host(static_cast<double>(
            tracer_mass[idx] / std::max(mass_work[idx], 1.0e-300L)));
      }
      if (remap_burn_species) {
        for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
          const std::size_t species_idx =
              idx * tenryu::burn::kNumSpecies + static_cast<std::size_t>(s);
          burn_n_out[species_idx] = static_cast<double>(std::fmax(
              burn_species_mass[species_idx] /
                  std::max(mass_work[idx], 1.0e-300L),
              0.0L));
        }
      }
      if (remap_hot_e_eps && idx < hot_e_eps_out.size()) {
        hot_e_eps_out[idx] = std::max(
            static_cast<double>(hot_e_eps_mass[idx] /
                                std::max(mass_work[idx], 1.0e-300L)),
            0.0);
      }
      if (remap_burn_eps && idx < burn_eps_out.size()) {
        burn_eps_out[idx] = std::max(
            static_cast<double>(burn_eps_mass[idx] /
                                std::max(mass_work[idx], 1.0e-300L)),
            0.0);
      }
      if (idx < Te_out.size()) {
        Te_out[idx] = cv_e_c > 0.0
                          ? std::max(cfg.numerics.floors.Te,
                                     ee_out[idx] / cv_e_c)
                          : cfg.numerics.floors.Te;
      }
      if (idx < Ti_out.size()) {
        Ti_out[idx] = cv_i_c > 0.0
                          ? std::max(cfg.numerics.floors.Ti,
                                     ei_out[idx] / cv_i_c)
                          : cfg.numerics.floors.Ti;
      }
      if (idx < Pe_out.size()) {
        Pe_out[idx] = gm1 * std::max(rho_out[idx], cfg.numerics.floors.rho) *
                      std::max(ee_out[idx], 0.0);
      }
      if (idx < Pi_out.size()) {
        Pi_out[idx] = gm1 * std::max(rho_out[idx], cfg.numerics.floors.rho) *
                      std::max(ei_out[idx], 0.0);
      }
      if (idx < cs_out.size() && idx < Pe_out.size() && idx < Pi_out.size()) {
        cs_out[idx] = std::sqrt(std::max(
            gamma * (Pe_out[idx] + Pi_out[idx]) /
                std::max(rho_out[idx], cfg.numerics.floors.rho),
            0.0));
      }
      if (idx < hllc_mom_z_out.size()) {
        long double pz = 0.0L;
        const int nverts = active_nverts(state, cell);
        for (int k = 0; k < nverts; ++k) {
          const int node = cell_corner_node_id(state, cell, k);
          const std::size_t cidx = idx * 4U + static_cast<std::size_t>(k);
          if (node >= 0 && static_cast<std::size_t>(node) < v_z_work.size() &&
              cidx < corner_mass_work.size()) {
            pz += static_cast<long double>(corner_mass_work[cidx]) *
                  v_z_work[static_cast<std::size_t>(node)];
          }
        }
        hllc_mom_z_out[idx] = static_cast<double>(pz);
      }
    }
    if (trial.positivity_failed) {
      remap = trial;
      continue;
    }
    if (trial.max_volume_error > 1.0e-12) {
      trial.positivity_failed = true;
      return trial;
    }

    trial.central_after =
        cell_totals(state, central_cells, mass_out, ee_out, ei_out,
                    v_r_work, v_z_work, corner_mass_work);
    trial.central_hash_after =
        central_core_hash(state, central_cells, trial.central_after);
    const SeamTotals totals_after =
        cell_totals(state, all_cells, mass_out, ee_out, ei_out, v_r_work,
                    v_z_work, corner_mass_work);
    trial.dM_rel = rel_delta(totals_after.mass, totals_before.mass);
    trial.dE_rel =
        rel_delta(totals_after.total_energy, totals_before.total_energy);
    const long double momentum_scale = std::max(
        std::sqrt(std::max(2.0L * std::abs(totals_before.mass) *
                               std::abs(totals_before.kinetic_energy),
                           0.0L)),
        1.0e-300L);
    trial.dPr_paired_rel =
        rel_delta_scale(totals_after.momentum_r - totals_before.momentum_r,
                        std::max(std::abs(totals_before.momentum_r),
                                 momentum_scale));
    trial.dPz_paired_rel =
        rel_delta_scale(totals_after.momentum_z - totals_before.momentum_z,
                        std::max(std::abs(totals_before.momentum_z),
                                 momentum_scale));
    trial.central_failed =
        trial.central_hash_before != trial.central_hash_after ||
        rel_delta(trial.central_after.mass, trial.central_before.mass) >
            1.0e-14 ||
        rel_delta(trial.central_after.total_energy,
                  trial.central_before.total_energy) > 1.0e-14 ||
        rel_delta(trial.central_after.momentum_r,
                  trial.central_before.momentum_r) > 1.0e-14 ||
        rel_delta(trial.central_after.momentum_z,
                  trial.central_before.momentum_z) > 1.0e-14;
    trial.conservation_failed =
        trial.dM_rel > 1.0e-10 || trial.dE_rel > 1.0e-10 ||
        trial.dPr_paired_rel > 1.0e-8 || trial.dPz_paired_rel > 1.0e-8;
    if (trial.central_failed || trial.conservation_failed) {
      return trial;
    }

    state.x_r.copy_from_host(x_r_new);
    state.x_z.copy_from_host(x_z_new);
    state.mass.copy_from_host(mass_out);
    state.rho.copy_from_host(rho_out);
    state.ee.copy_from_host(ee_out);
    state.ei.copy_from_host(ei_out);
    if (state.Te.size() == Te_out.size()) {
      state.Te.copy_from_host(Te_out);
    }
    if (state.Ti.size() == Ti_out.size()) {
      state.Ti.copy_from_host(Ti_out);
    }
    if (state.Pe.size() == Pe_out.size()) {
      state.Pe.copy_from_host(Pe_out);
    }
    if (state.Pi.size() == Pi_out.size()) {
      state.Pi.copy_from_host(Pi_out);
    }
    if (state.cs.size() == cs_out.size()) {
      state.cs.copy_from_host(cs_out);
    }
    if (state.zbar.size() == zbar_out.size()) {
      state.zbar.copy_from_host(zbar_out);
    }
    if (state.gas_tracer_initialized &&
        state.gas_tracer_Y.size() == gas_tracer_out.size()) {
      state.gas_tracer_Y.copy_from_host(gas_tracer_out);
    }
    if (remap_burn_species &&
        state.burn_n_host.size() == burn_n_out.size()) {
      state.burn_n_host = burn_n_out;
    }
    if (remap_hot_e_eps &&
        state.hot_e_eps_cum_host.size() == hot_e_eps_out.size()) {
      state.hot_e_eps_cum_host = hot_e_eps_out;
    }
    if (remap_burn_eps &&
        state.burn_eps_cum_host.size() == burn_eps_out.size()) {
      state.burn_eps_cum_host = burn_eps_out;
    }
    if (state.hllc_mom_z_cell.size() == hllc_mom_z_out.size()) {
      state.hllc_mom_z_cell.copy_from_host(hllc_mom_z_out);
    }
    state.v_r.copy_from_host(v_r_work);
    state.v_z.copy_from_host(v_z_work);
    if (state.corner_mass.size() != corner_mass_work.size()) {
      state.corner_mass.reset(corner_mass_work.size());
    }
    state.corner_mass.copy_from_host(corner_mass_work);
    state.corner_mass_initialized = true;
    state.corner_mass_is_lagrangian_invariant = false;
    state.mesh.node_r = state.x_r.data();
    state.mesh.node_z = state.x_z.data();
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
    trial.ok = true;
    return trial;
  }
  remap.positivity_failed = true;
  remap.subcycles = kmax;
  return remap;
}

bool polar_shell_cell_local_ij(const core::State& state,
                               const int cell,
                               int* local_i,
                               int* local_j) {
  if (!state.mesh.topo.multiblock.has_value() || cell < 0) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  const int rel = cell - shell.cell_begin;
  if (rel < 0 || rel >= shell.cell_count || shell.n_j_cells <= 0) {
    return false;
  }
  if (local_i != nullptr) {
    *local_i = rel / shell.n_j_cells;
  }
  if (local_j != nullptr) {
    *local_j = rel % shell.n_j_cells;
  }
  return true;
}

bool cell_has_domain_boundary_face(const core::State& state, const int cell) {
  if (!state.mesh.topo.multiblock.has_value() || cell < 0) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  for (const auto& face : mb.boundary_faces) {
    if (face.cell_a == cell) {
      return true;
    }
  }
  return false;
}

bool is_driven_pole_cap_cell(const core::State& state, const int cell) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  int local_i = -1;
  int local_j = -1;
  if (!polar_shell_cell_local_ij(state, cell, &local_i, &local_j)) {
    return false;
  }
  return local_i == shell.n_i_cells - 1 && local_j == 0 &&
         cell_has_domain_boundary_face(state, cell);
}

struct Ring7PoleCap {
  bool valid = false;
  int seed_cell = -1;
  int layers = 0;
  int central_violations = 0;
  std::vector<int> cells;
  std::vector<int> nodes;
  std::vector<int> node_distance;
};

Ring7PoleCap build_ring7_pole_cap(const core::State& state,
                                  const int seed_cell,
                                  const int layers) {
  Ring7PoleCap cap;
  cap.seed_cell = seed_cell;
  cap.layers = std::max(1, layers);
  if (!state.mesh.topo.multiblock.has_value()) {
    return cap;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  int seed_i = -1;
  int seed_j = -1;
  if (!polar_shell_cell_local_ij(state, seed_cell, &seed_i, &seed_j) ||
      seed_j != 0) {
    return cap;
  }
  std::vector<std::uint8_t> node_mask(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  cap.node_distance.assign(static_cast<std::size_t>(state.mesh.topo.n_nodes),
                           cap.layers + 1);
  for (int di = 0; di <= cap.layers; ++di) {
    const int i = seed_i - di;
    if (i < 0 || i >= shell.n_i_cells) {
      continue;
    }
    for (int j = 0; j <= cap.layers; ++j) {
      if (j >= shell.n_j_cells) {
        continue;
      }
      const int cell = shell.cell_begin + i * shell.n_j_cells + j;
      if (cell < 0 || cell >= state.mesh.topo.n_cells) {
        continue;
      }
      if (cell_is_central_excluded(state, mb, cell)) {
        ++cap.central_violations;
        continue;
      }
      cap.cells.push_back(cell);
      const int distance = std::max(di, j);
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      const int nverts = active_nverts(state, cell);
      for (int k = 0; k < nverts; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (node < 0 || node >= state.mesh.topo.n_nodes) {
          continue;
        }
        const std::size_t nidx = static_cast<std::size_t>(node);
        cap.node_distance[nidx] = std::min(cap.node_distance[nidx], distance);
        if (node_mask[nidx] == 0U) {
          node_mask[nidx] = 1U;
          cap.nodes.push_back(node);
        }
      }
    }
  }
  sort_unique(cap.cells);
  sort_unique(cap.nodes);
  cap.valid = !cap.cells.empty() && !cap.nodes.empty() &&
              cap.central_violations == 0;
  return cap;
}

struct PoleCapPacketGeometry {
  std::vector<Ring7PacketRecord> packets;
  std::vector<int> patch_cells;
  std::vector<long double> boundary_volume_delta;
  std::vector<long double> signed_source_volume;
  std::vector<long double> signed_target_volume;
  std::vector<int> reference_orientation_sign;
  std::vector<std::uint8_t> source_valid;
  std::vector<std::uint8_t> target_valid;
  int patch_faces = 0;
  int cap_boundary_faces = 0;
  int cap_boundary_nonzero_faces = 0;
  int domain_boundary_faces = 0;
  int central_violations = 0;
  int invariant_violations = 0;
  int orientation_invalid_cells = 0;
  double max_geom_epsilon = 0.0;
  double domain_boundary_swept_abs = 0.0;
  double cap_boundary_swept_abs = 0.0;
};

PoleCapPacketGeometry build_pole_cap_packet_geometry(
    const core::State& state,
    const std::vector<double>& x_r_current,
    const std::vector<double>& x_z_current,
    const std::vector<double>& x_r_src,
    const std::vector<double>& x_z_src,
    const std::vector<double>& x_r_dst,
    const std::vector<double>& x_z_dst,
    const std::vector<double>& corner_volume) {
  PoleCapPacketGeometry geom;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  geom.boundary_volume_delta.assign(static_cast<std::size_t>(n_cells), 0.0L);
  geom.signed_source_volume.assign(static_cast<std::size_t>(n_cells), 0.0L);
  geom.signed_target_volume.assign(static_cast<std::size_t>(n_cells), 0.0L);
  geom.reference_orientation_sign.assign(static_cast<std::size_t>(n_cells), 1);
  geom.source_valid.assign(static_cast<std::size_t>(n_cells), 0U);
  geom.target_valid.assign(static_cast<std::size_t>(n_cells), 0U);
  if (!state.mesh.topo.multiblock.has_value()) {
    ++geom.invariant_violations;
    return geom;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const std::vector<int> moved_nodes =
      moved_nodes_from_target(x_r_src, x_z_src, x_r_dst, x_z_dst);
  std::vector<std::uint8_t> moved_mask(static_cast<std::size_t>(n_nodes), 0U);
  for (const int node : moved_nodes) {
    if (node >= 0 && node < n_nodes) {
      moved_mask[static_cast<std::size_t>(node)] = 1U;
    }
  }
  std::vector<std::uint8_t> patch_mask(static_cast<std::size_t>(n_cells), 0U);
  for (int cell = 0; cell < n_cells; ++cell) {
    if (!cell_touches_node_mask(state, cell, moved_mask)) {
      continue;
    }
    if (cell_is_central_excluded(state, mb, cell)) {
      ++geom.central_violations;
      ++geom.invariant_violations;
      continue;
    }
    patch_mask[static_cast<std::size_t>(cell)] = 1U;
    geom.patch_cells.push_back(cell);
  }
  sort_unique(geom.patch_cells);

  std::vector<long double> packet_delta(static_cast<std::size_t>(n_cells),
                                        0.0L);
  int packet_id = 0;
  for (const auto& face : mb.unique_internal_faces) {
    const int a = face.cell_a;
    const int b = face.cell_b;
    if (a < 0 || b < 0 || a >= n_cells || b >= n_cells) {
      ++geom.invariant_violations;
      continue;
    }
    const bool a_in = patch_mask[static_cast<std::size_t>(a)] != 0U;
    const bool b_in = patch_mask[static_cast<std::size_t>(b)] != 0U;
    if (a_in && b_in) {
      const double swept_a =
          face_swept_quad_polygon_delta(state, a, face.local_a, x_r_src,
                                        x_z_src, x_r_dst, x_z_dst);
      const double swept_b =
          face_swept_quad_polygon_delta(state, b, face.local_b, x_r_src,
                                        x_z_src, x_r_dst, x_z_dst);
      packet_delta[static_cast<std::size_t>(a)] += swept_a;
      packet_delta[static_cast<std::size_t>(b)] += swept_b;
      ++geom.patch_faces;
      const double canonical_swept = -swept_a;
      const bool a_to_b = canonical_swept >= 0.0;
      const int donor = a_to_b ? a : b;
      const int acceptor = a_to_b ? b : a;
      const int split_cell = donor;
      const int split_face = a_to_b ? face.local_a : face.local_b;
      int split0 = 0;
      int split1 = 1;
      double w0 = 0.5;
      double w1 = 0.5;
      corner_split_weights(state, split_cell, split_face, corner_volume,
                           &split0, &split1, &w0, &w1);
      const int donor_node0 = cell_corner_node_id(state, donor, split0);
      const int donor_node1 = cell_corner_node_id(state, donor, split1);
      const int accept0 =
          matching_cell_corner(state, acceptor, donor_node0, x_r_src, x_z_src);
      const int accept1 =
          matching_cell_corner(state, acceptor, donor_node1, x_r_src, x_z_src);
      const double transfer_volume = std::abs(canonical_swept);
      if (accept0 < 0 || accept1 < 0) {
        ++geom.invariant_violations;
      } else {
        geom.packets.push_back({donor, acceptor, split0, accept0, a,
                                face.local_a, packet_id++,
                                transfer_volume * w0});
        geom.packets.push_back({donor, acceptor, split1, accept1, a,
                                face.local_a, packet_id++,
                                transfer_volume * w1});
      }
    } else if (a_in != b_in) {
      const double swept =
          a_in ? face_swept_quad_polygon_delta(state, a, face.local_a,
                                               x_r_src, x_z_src,
                                               x_r_dst, x_z_dst)
               : face_swept_quad_polygon_delta(state, b, face.local_b,
                                               x_r_src, x_z_src,
                                               x_r_dst, x_z_dst);
      ++geom.cap_boundary_faces;
      geom.cap_boundary_swept_abs += std::abs(swept);
      if (std::abs(swept) > 1.0e-250) {
        ++geom.cap_boundary_nonzero_faces;
        ++geom.invariant_violations;
      }
    }
  }
  for (const auto& face : mb.boundary_faces) {
    const int cell = face.cell_a;
    if (cell < 0 || cell >= n_cells ||
        patch_mask[static_cast<std::size_t>(cell)] == 0U) {
      continue;
    }
    const double swept =
        face_swept_quad_polygon_delta(state, cell, face.local_a, x_r_src,
                                      x_z_src, x_r_dst, x_z_dst);
    ++geom.domain_boundary_faces;
    geom.domain_boundary_swept_abs += std::abs(swept);
    geom.boundary_volume_delta[static_cast<std::size_t>(cell)] -= swept;
  }
  std::sort(geom.packets.begin(), geom.packets.end(), [](const auto& a,
                                                         const auto& b) {
    return std::array<int, 7>{a.donor_cell,
                              a.acceptor_cell,
                              a.donor_corner,
                              a.acceptor_corner,
                              a.owner_cell,
                              a.owner_face,
                              a.packet_id} <
           std::array<int, 7>{b.donor_cell,
                              b.acceptor_cell,
                              b.donor_corner,
                              b.acceptor_corner,
                              b.owner_cell,
                              b.owner_face,
                              b.packet_id};
  });
  for (const int cell : geom.patch_cells) {
    const std::size_t idx = static_cast<std::size_t>(cell);
    const long double reference =
        signed_cell_volume_from_coords(state, x_r_current, x_z_current, cell);
    const int s_ref = reference >= 0.0L ? 1 : -1;
    const long double v_min =
        std::max(1.0e-30L, 1.0e-6L * std::abs(reference));
    const long double source =
        signed_cell_volume_from_coords(state, x_r_src, x_z_src, cell);
    const long double target =
        signed_cell_volume_from_coords(state, x_r_dst, x_z_dst, cell);
    const bool src_valid = static_cast<long double>(s_ref) * source > v_min;
    const bool dst_valid = static_cast<long double>(s_ref) * target > v_min;
    geom.signed_source_volume[idx] = source;
    geom.signed_target_volume[idx] = target;
    geom.reference_orientation_sign[idx] = s_ref;
    geom.source_valid[idx] = src_valid ? 1U : 0U;
    geom.target_valid[idx] = dst_valid ? 1U : 0U;
    if (!src_valid || !dst_valid) {
      ++geom.orientation_invalid_cells;
    }
    const long double epsilon =
        (target - source) - packet_delta[idx] - geom.boundary_volume_delta[idx];
    geom.max_geom_epsilon =
        std::max(geom.max_geom_epsilon,
                 static_cast<double>(std::abs(epsilon)));
  }
  return geom;
}

void log_pole_cap_closure_probe(const core::State& state,
                                const PoleCapPacketGeometry& geom,
                                const int cell,
                                const char* stage,
                                const std::vector<double>& x_r_lag,
                                const std::vector<double>& x_z_lag,
                                const std::vector<double>& x_r_cap,
                                const std::vector<double>& x_z_cap) {
  if (!state.mesh.topo.multiblock.has_value() || cell < 0 ||
      cell >= state.mesh.topo.n_cells) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  std::vector<std::uint8_t> patch_mask(
      static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
  for (const int patch_cell : geom.patch_cells) {
    if (patch_cell >= 0 && patch_cell < state.mesh.topo.n_cells) {
      patch_mask[static_cast<std::size_t>(patch_cell)] = 1U;
    }
  }

  const long double v_lag_signed =
      signed_cell_volume_from_coords(state, x_r_lag, x_z_lag, cell);
  const long double v_cap_signed =
      signed_cell_volume_from_coords(state, x_r_cap, x_z_cap, cell);
  const long double v_lag_abs = std::abs(v_lag_signed);
  const long double v_cap_abs = std::abs(v_cap_signed);
  long double sum_internal_swept = 0.0L;
  long double sum_domain_swept = 0.0L;
  int internal_faces = 0;
  for (const auto& face : mb.unique_internal_faces) {
    const int a = face.cell_a;
    const int b = face.cell_b;
    if (a < 0 || b < 0 || a >= state.mesh.topo.n_cells ||
        b >= state.mesh.topo.n_cells ||
        patch_mask[static_cast<std::size_t>(a)] == 0U ||
        patch_mask[static_cast<std::size_t>(b)] == 0U) {
      continue;
    }
    if (a == cell) {
      sum_internal_swept += static_cast<long double>(
          face_swept_quad_polygon_delta(state, cell, face.local_a,
                                        x_r_lag, x_z_lag, x_r_cap, x_z_cap));
      ++internal_faces;
    } else if (b == cell) {
      sum_internal_swept += static_cast<long double>(
          face_swept_quad_polygon_delta(state, cell, face.local_b,
                                        x_r_lag, x_z_lag, x_r_cap, x_z_cap));
      ++internal_faces;
    }
  }

  int domain_faces = 0;
  for (std::size_t face_index = 0; face_index < mb.boundary_faces.size();
       ++face_index) {
    const auto& face = mb.boundary_faces[face_index];
    if (face.cell_a != cell) {
      continue;
    }
    ++domain_faces;
    const double edge_raw =
        face_swept_volume(state, cell, face.local_a, x_r_lag, x_z_lag,
                          x_r_cap, x_z_cap);
    const double edge_closure = -edge_raw;

    int c0 = -1;
    int c1 = -1;
    int n0 = -1;
    int n1 = -1;
    if (local_face_corners(state, cell, face.local_a, &c0, &c1)) {
      const int off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      n0 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c0)];
      n1 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + c1)];
    }
    const bool n0_valid =
        n0 >= 0 && static_cast<std::size_t>(n0) < x_r_lag.size();
    const bool n1_valid =
        n1 >= 0 && static_cast<std::size_t>(n1) < x_r_lag.size();
    double polygon_raw = 0.0;
    double polygon_cell_delta = 0.0;
    if (n0_valid && n1_valid) {
      const std::size_t i0 = static_cast<std::size_t>(n0);
      const std::size_t i1 = static_cast<std::size_t>(n1);
      Ring7RzPoint swept_quad[4] = {
          {x_r_lag[i0], x_z_lag[i0]},
          {x_r_lag[i1], x_z_lag[i1]},
          {x_r_cap[i1], x_z_cap[i1]},
          {x_r_cap[i0], x_z_cap[i0]},
      };
      polygon_raw = axisymmetric_swept_volume_polygon(swept_quad, 4);
      polygon_cell_delta = face_swept_quad_polygon_delta(
          state, cell, face.local_a, x_r_lag, x_z_lag, x_r_cap, x_z_cap);
    }
    const double polygon_closure = -polygon_cell_delta;
    sum_domain_swept += static_cast<long double>(polygon_closure);
    std::ostringstream os;
    os << "[ring7_pole_cap_closure] step=" << state.step
       << " stage=" << (stage != nullptr ? stage : "?")
       << " cell=" << cell
       << " boundary_face_index=" << face_index
       << " local_face=" << face.local_a
       << " nodes=" << n0 << "," << n1
       << " n0_lag_r="
       << sci(n0_valid ? x_r_lag[static_cast<std::size_t>(n0)] : 0.0)
       << " n0_lag_z="
       << sci(n0_valid ? x_z_lag[static_cast<std::size_t>(n0)] : 0.0)
       << " n0_cap_r="
       << sci(n0_valid ? x_r_cap[static_cast<std::size_t>(n0)] : 0.0)
       << " n0_cap_z="
       << sci(n0_valid ? x_z_cap[static_cast<std::size_t>(n0)] : 0.0)
       << " n1_lag_r="
       << sci(n1_valid ? x_r_lag[static_cast<std::size_t>(n1)] : 0.0)
       << " n1_lag_z="
       << sci(n1_valid ? x_z_lag[static_cast<std::size_t>(n1)] : 0.0)
       << " n1_cap_r="
       << sci(n1_valid ? x_r_cap[static_cast<std::size_t>(n1)] : 0.0)
       << " n1_cap_z="
       << sci(n1_valid ? x_z_cap[static_cast<std::size_t>(n1)] : 0.0)
       << " edge_swept_raw=" << sci(edge_raw)
       << " edge_swept_closure=" << sci(edge_closure)
       << " polygon_quad_raw=" << sci(polygon_raw)
       << " polygon_cell_delta=" << sci(polygon_cell_delta)
       << " polygon_swept_closure=" << sci(polygon_closure);
    core::log_info(os.str());
  }

  const std::size_t cell_idx = static_cast<std::size_t>(cell);
  const int s_ref = cell_idx < geom.reference_orientation_sign.size()
                        ? geom.reference_orientation_sign[cell_idx]
                        : 1;
  const int src_valid = cell_idx < geom.source_valid.size()
                            ? static_cast<int>(geom.source_valid[cell_idx])
                            : 0;
  const int dst_valid = cell_idx < geom.target_valid.size()
                            ? static_cast<int>(geom.target_valid[cell_idx])
                            : 0;
  const long double signed_v_src =
      cell_idx < geom.signed_source_volume.size()
          ? geom.signed_source_volume[cell_idx]
          : v_lag_signed;
  const long double signed_v_dst =
      cell_idx < geom.signed_target_volume.size()
          ? geom.signed_target_volume[cell_idx]
          : v_cap_signed;
  const long double dV_cell = signed_v_dst - signed_v_src;
  const long double swept_total = sum_internal_swept + sum_domain_swept;
  const long double residual = dV_cell - swept_total;
  core::log_info("[ring7_pole_cap_closure] step=" +
                 std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " cell=" + std::to_string(cell) +
                 " V_lag=" + sci(static_cast<double>(v_lag_abs)) +
                 " V_cap=" + sci(static_cast<double>(v_cap_abs)) +
                 " signed_V_src=" + sci(static_cast<double>(signed_v_src)) +
                 " signed_V_dst=" + sci(static_cast<double>(signed_v_dst)) +
                 " s_ref=" + std::to_string(s_ref) +
                 " src_valid=" + std::to_string(src_valid) +
                 " dst_valid=" + std::to_string(dst_valid) +
                 " dV_cell=" + sci(static_cast<double>(dV_cell)) +
                 " sum_internal_swept=" +
                 sci(static_cast<double>(sum_internal_swept)) +
                 " sum_domain_swept=" +
                 sci(static_cast<double>(sum_domain_swept)) +
                 " closure_residual=" +
                 sci(static_cast<double>(residual)) +
                 " V_cap_from_swept=" +
                 sci(static_cast<double>(signed_v_src + swept_total)) +
                 " V_cap_exact=" + sci(static_cast<double>(signed_v_dst)) +
                 " internal_faces=" + std::to_string(internal_faces) +
                 " domain_faces=" + std::to_string(domain_faces) +
                 " max_geom_epsilon=" + sci(geom.max_geom_epsilon));
}

Ring7PacketRemapDiagnostics apply_pole_cap_packet_remap_transaction(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const Ring7PoleCap& cap,
    const std::vector<double>& x_r_current,
    const std::vector<double>& x_z_current,
    const std::vector<double>& x_r_lag,
    const std::vector<double>& x_z_lag,
    const std::vector<double>& x_r_cap,
    const std::vector<double>& x_z_cap,
    const std::vector<double>& x_r_commit,
    const std::vector<double>& x_z_commit,
    const std::vector<double>& material_u_r,
    const std::vector<double>& material_u_z,
    const double boundary_pressure,
    std::vector<double> rho,
    std::vector<double> mass,
    std::vector<double> ee,
    std::vector<double> ei,
    std::vector<double> Te,
    std::vector<double> Ti,
    std::vector<double> Pe,
    std::vector<double> Pi,
    std::vector<double> v_r,
    std::vector<double> v_z,
    std::vector<double> corner_mass,
    const std::vector<double>& corner_volume,
    std::vector<double> gas_tracer_Y,
    std::vector<double> zbar,
    const std::vector<double>& cv_e,
    const std::vector<double>& cv_i,
    std::vector<double> cs,
    std::vector<double> hllc_mom_z_cell) {
  Ring7PacketRemapDiagnostics remap;
  if (eos_ctx != nullptr && eos_ctx->any_table) {
    core::log_info("[ring7_pole_cap_remap] status=rejected_table_eos");
    return remap;
  }
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    core::log_info("[ring7_pole_cap_remap] status=rejected_per_material_mode");
    return remap;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (!cap.valid || cap.central_violations != 0 ||
      static_cast<int>(mass.size()) < n_cells ||
      static_cast<int>(rho.size()) < n_cells ||
      static_cast<int>(ee.size()) < n_cells ||
      static_cast<int>(ei.size()) < n_cells ||
      static_cast<int>(v_r.size()) < n_nodes ||
      static_cast<int>(v_z.size()) < n_nodes ||
      static_cast<int>(material_u_r.size()) < n_nodes ||
      static_cast<int>(material_u_z.size()) < n_nodes) {
    return remap;
  }

  ensure_corner_mass_work(state, mass, corner_mass);
  const std::vector<int> all_cells = all_cell_ids(n_cells);
  const std::vector<int> central_cells = central_excluded_cells(state);
  remap.central_before =
      cell_totals(state, central_cells, mass, ee, ei, v_r, v_z, corner_mass);
  remap.central_hash_before =
      central_core_hash(state, central_cells, remap.central_before);
  const SeamTotals totals_before =
      cell_totals(state, all_cells, mass, ee, ei, v_r, v_z, corner_mass);

  const int kmax =
      env_int_value("TENRYU_I1B_POLE_CAP_PACKET_SUBCYCLE_KMAX", 8);
  const double pos_eps =
      env_double_value("TENRYU_I1B_POLE_CAP_PACKET_POS_EPS", 1.0e-12);
  for (int subcycles = 1; subcycles <= kmax; ++subcycles) {
    Ring7PacketRemapDiagnostics trial;
    trial.subcycles = subcycles;
    trial.central_before = remap.central_before;
    trial.central_hash_before = remap.central_hash_before;

    std::vector<double> rho_out = rho;
    std::vector<double> mass_out = mass;
    std::vector<double> ee_out = ee;
    std::vector<double> ei_out = ei;
    std::vector<double> Te_out = Te;
    std::vector<double> Ti_out = Ti;
    std::vector<double> Pe_out = Pe;
    std::vector<double> Pi_out = Pi;
    std::vector<double> v_r_work = v_r;
    std::vector<double> v_z_work = v_z;
    std::vector<double> corner_mass_work = corner_mass;
    std::vector<double> gas_tracer_out = gas_tracer_Y;
    const bool remap_hot_e_eps =
        !state.hot_e_eps_cum_host.empty() &&
        state.hot_e_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
    std::vector<double> hot_e_eps_out = state.hot_e_eps_cum_host;
    const bool remap_burn_eps =
        !state.burn_eps_cum_host.empty() &&
        state.burn_eps_cum_host.size() == static_cast<std::size_t>(n_cells);
    std::vector<double> burn_eps_out = state.burn_eps_cum_host;
    std::vector<double> zbar_out = zbar;
    std::vector<double> cs_out = cs;
    std::vector<double> hllc_mom_z_out = hllc_mom_z_cell;
    std::vector<long double> mass_work(mass.begin(), mass.end());
    std::vector<long double> volume_work(static_cast<std::size_t>(n_cells),
                                         0.0L);
    std::vector<long double> total_energy_work(
        static_cast<std::size_t>(n_cells), 0.0L);
    std::vector<long double> ye_mass(static_cast<std::size_t>(n_cells),
                                     0.0L);
    std::vector<long double> tracer_mass(static_cast<std::size_t>(n_cells),
                                         0.0L);
    std::vector<long double> hot_e_eps_mass(static_cast<std::size_t>(n_cells),
                                            0.0L);
    std::vector<long double> burn_eps_mass(static_cast<std::size_t>(n_cells),
                                           0.0L);
    std::vector<long double> zbar_mass(static_cast<std::size_t>(n_cells),
                                       0.0L);
    for (int cell = 0; cell < n_cells; ++cell) {
      const std::size_t idx = static_cast<std::size_t>(cell);
      volume_work[idx] = std::abs(
          signed_cell_volume_from_coords(state, x_r_lag, x_z_lag, cell));
      const long double internal =
          mass_work[idx] * std::max(ee[idx] + ei[idx], 0.0);
      total_energy_work[idx] =
          internal +
          cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                              cell);
      const double denom = std::max(ee[idx] + ei[idx], 0.0);
      const double ye = denom > 0.0 ? clamp01_host(ee[idx] / denom) : 0.5;
      ye_mass[idx] = mass_work[idx] * ye;
      if (idx < gas_tracer_Y.size() && state.gas_tracer_initialized) {
        tracer_mass[idx] =
            mass_work[idx] * clamp01_host(gas_tracer_Y[idx]);
      }
      if (remap_hot_e_eps) {
        hot_e_eps_mass[idx] =
            mass_work[idx] * std::max(state.hot_e_eps_cum_host[idx], 0.0);
      }
      if (remap_burn_eps) {
        burn_eps_mass[idx] =
            mass_work[idx] * std::max(state.burn_eps_cum_host[idx], 0.0);
      }
      if (idx < zbar.size()) {
        zbar_mass[idx] = mass_work[idx] * std::max(zbar[idx], 0.0);
      }
    }

    std::vector<long double> node_mass;
    std::vector<long double> node_pr;
    std::vector<long double> node_pz;
    build_node_mass_momentum(state, corner_mass_work, v_r_work, v_z_work,
                             node_mass, node_pr, node_pz);

    std::vector<int> remap_cells = cap.cells;
    std::vector<double> segment_r0 = x_r_lag;
    std::vector<double> segment_z0 = x_z_lag;
    bool retry_larger_k = false;
    bool hard_failure = false;

    const auto refresh_node_velocity = [&](const int node) {
      if (node < 0 || static_cast<std::size_t>(node) >= node_mass.size()) {
        return;
      }
      const std::size_t idx = static_cast<std::size_t>(node);
      if (node_mass[idx] > 0.0L) {
        v_r_work[idx] = static_cast<double>(node_pr[idx] / node_mass[idx]);
        v_z_work[idx] = static_cast<double>(node_pz[idx] / node_mass[idx]);
      }
    };

    for (int sub = 0; sub < subcycles; ++sub) {
      const double alpha =
          static_cast<double>(sub + 1) / static_cast<double>(subcycles);
      std::vector<double> segment_r1 = x_r_lag;
      std::vector<double> segment_z1 = x_z_lag;
      for (int node = 0; node < n_nodes; ++node) {
        const std::size_t idx = static_cast<std::size_t>(node);
        segment_r1[idx] =
            x_r_lag[idx] + alpha * (x_r_cap[idx] - x_r_lag[idx]);
        segment_z1[idx] =
            x_z_lag[idx] + alpha * (x_z_cap[idx] - x_z_lag[idx]);
      }

      const std::vector<double> segment_corner_volume =
          corner_volumes_for_coords(state, segment_r0, segment_z0,
                                    corner_volume);
      const PoleCapPacketGeometry segment_geom =
          build_pole_cap_packet_geometry(state,
                                         x_r_current,
                                         x_z_current,
                                         segment_r0,
                                         segment_z0,
                                         segment_r1,
                                         segment_z1,
                                         segment_corner_volume);
      trial.max_volume_error = std::max(trial.max_volume_error,
                                        segment_geom.max_geom_epsilon);
      if (segment_geom.invariant_violations != 0 ||
          segment_geom.cap_boundary_nonzero_faces != 0 ||
          segment_geom.orientation_invalid_cells != 0 ||
          !(segment_geom.max_geom_epsilon <= 1.0e-12)) {
        trial.positivity_failed = true;
        hard_failure = true;
        break;
      }
      if (!packets_pass_geometric_positivity(state,
                                             segment_geom.packets,
                                             segment_corner_volume,
                                             segment_r0,
                                             segment_z0,
                                             pos_eps)) {
        trial.positivity_failed = true;
        retry_larger_k = true;
        break;
      }
      for (const auto& packet : segment_geom.packets) {
        const int donor = packet.donor_cell;
        const int acceptor = packet.acceptor_cell;
        const int donor_corner = packet.donor_corner;
        const int acceptor_corner = packet.acceptor_corner;
        remap_cells.push_back(donor);
        remap_cells.push_back(acceptor);
        if (donor < 0 || donor >= n_cells || acceptor < 0 ||
            acceptor >= n_cells || donor_corner < 0 || donor_corner >= 4 ||
            acceptor_corner < 0 || acceptor_corner >= 4) {
          trial.positivity_failed = true;
          hard_failure = true;
          break;
        }
        const std::size_t d = static_cast<std::size_t>(donor);
        const std::size_t a = static_cast<std::size_t>(acceptor);
        const std::size_t dc =
            d * 4U + static_cast<std::size_t>(donor_corner);
        const std::size_t ac =
            a * 4U + static_cast<std::size_t>(acceptor_corner);
        const int donor_node =
            cell_corner_node_id(state, donor, donor_corner);
        const int acceptor_node =
            cell_corner_node_id(state, acceptor, acceptor_corner);
        if (dc >= corner_mass_work.size() || ac >= corner_mass_work.size() ||
            donor_node < 0 || donor_node >= n_nodes || acceptor_node < 0 ||
            acceptor_node >= n_nodes) {
          trial.positivity_failed = true;
          hard_failure = true;
          break;
        }
        const long double dV = static_cast<long double>(packet.volume);
        const long double rho_d =
            mass_work[d] / std::max(volume_work[d], 1.0e-300L);
        const long double dm = rho_d * dV;
        if (!(dm >= 0.0L) || !std::isfinite(static_cast<double>(dm)) ||
            corner_mass_work[dc] + 1.0e-30 < static_cast<double>(dm) ||
            mass_work[d] + 1.0e-30L < dm) {
          trial.positivity_failed = true;
          retry_larger_k = true;
          break;
        }
        const std::size_t dn = static_cast<std::size_t>(donor_node);
        const std::size_t an = static_cast<std::size_t>(acceptor_node);
        const long double udr = material_u_r[dn];
        const long double udz = material_u_z[dn];
        const long double uar = v_r_work[an];
        const long double uaz = v_z_work[an];
        const long double ke_specific = 0.5L * (udr * udr + udz * udz);
        const long double ke_mix =
            0.5L * dm * ((udr - uar) * (udr - uar) +
                         (udz - uaz) * (udz - uaz));
        trial.q_mix += static_cast<double>(std::max(ke_mix, 0.0L));
        const long double kinetic_d =
            cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                                donor);
        const long double internal_d =
            std::max(total_energy_work[d] - kinetic_d, 0.0L);
        const long double e_int_d =
            internal_d / std::max(mass_work[d], 1.0e-300L);
        const long double ye = clamp01_host(static_cast<double>(
            ye_mass[d] / std::max(mass_work[d], 1.0e-300L)));
        const long double tracer =
            state.gas_tracer_initialized
                ? clamp01_host(static_cast<double>(
                      tracer_mass[d] / std::max(mass_work[d], 1.0e-300L)))
                : 0.0L;
        const long double hot_e_eps_d =
            remap_hot_e_eps
                ? std::max(hot_e_eps_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        const long double burn_eps_d =
            remap_burn_eps
                ? std::max(burn_eps_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        const long double z =
            (d < zbar_mass.size())
                ? std::max(zbar_mass[d] /
                               std::max(mass_work[d], 1.0e-300L),
                           0.0L)
                : 0.0L;
        mass_work[d] -= dm;
        mass_work[a] += dm;
        volume_work[d] -= dV;
        volume_work[a] += dV;
        total_energy_work[d] -= dm * (e_int_d + ke_specific);
        total_energy_work[a] += dm * (e_int_d + ke_specific);
        ye_mass[d] -= dm * ye;
        ye_mass[a] += dm * ye;
        if (state.gas_tracer_initialized) {
          tracer_mass[d] -= dm * tracer;
          tracer_mass[a] += dm * tracer;
        }
        if (remap_hot_e_eps) {
          hot_e_eps_mass[d] -= dm * hot_e_eps_d;
          hot_e_eps_mass[a] += dm * hot_e_eps_d;
        }
        if (remap_burn_eps) {
          burn_eps_mass[d] -= dm * burn_eps_d;
          burn_eps_mass[a] += dm * burn_eps_d;
        }
        if (!zbar.empty()) {
          zbar_mass[d] -= dm * z;
          zbar_mass[a] += dm * z;
        }
        corner_mass_work[dc] -= static_cast<double>(dm);
        corner_mass_work[ac] += static_cast<double>(dm);
        node_mass[dn] -= dm;
        node_mass[an] += dm;
        node_pr[dn] -= dm * udr;
        node_pz[dn] -= dm * udz;
        node_pr[an] += dm * udr;
        node_pz[an] += dm * udz;
        refresh_node_velocity(donor_node);
        refresh_node_velocity(acceptor_node);
      }
      if (trial.positivity_failed) {
        break;
      }
      for (const int cell : segment_geom.patch_cells) {
        if (cell < 0 || cell >= n_cells) {
          trial.positivity_failed = true;
          hard_failure = true;
          break;
        }
        const std::size_t idx = static_cast<std::size_t>(cell);
        const long double dV_boundary =
            segment_geom.boundary_volume_delta[idx];
        const long double dU_boundary =
            -static_cast<long double>(boundary_pressure) * dV_boundary;
        volume_work[idx] += dV_boundary;
        total_energy_work[idx] += dU_boundary;
        trial.boundary_work += static_cast<double>(dU_boundary);
        remap_cells.push_back(cell);
        if (!(volume_work[idx] > 0.0L) ||
            !std::isfinite(static_cast<double>(volume_work[idx]))) {
          trial.positivity_failed = true;
          retry_larger_k = true;
          break;
        }
      }
      if (trial.positivity_failed) {
        break;
      }
      segment_r0 = std::move(segment_r1);
      segment_z0 = std::move(segment_z1);
    }
    if (hard_failure) {
      return trial;
    }
    if (retry_larger_k || trial.positivity_failed) {
      remap = trial;
      continue;
    }

    sort_unique(remap_cells);
    const double gamma =
        (!cfg.materials.materials.empty() &&
         cfg.materials.materials.front().ideal_gas_gamma > 1.0)
            ? cfg.materials.materials.front().ideal_gas_gamma
            : (5.0 / 3.0);
    const double gm1 = std::max(gamma - 1.0, 1.0e-30);
    for (const int cell : remap_cells) {
      const std::size_t idx = static_cast<std::size_t>(cell);
      const long double target_v = std::abs(
          signed_cell_volume_from_coords(state, x_r_cap, x_z_cap, cell));
      trial.max_volume_error = std::max(
          trial.max_volume_error,
          static_cast<double>(std::abs(volume_work[idx] - target_v)));
      if (!(mass_work[idx] > 0.0L) || !(target_v > 0.0L)) {
        trial.positivity_failed = true;
        break;
      }
      const long double kinetic =
          cell_kinetic_energy(state, corner_mass_work, v_r_work, v_z_work,
                              cell);
      const long double internal = total_energy_work[idx] - kinetic;
      if (idx < zbar_out.size()) {
        zbar_out[idx] = static_cast<double>(
            std::max(zbar_mass[idx] / std::max(mass_work[idx], 1.0e-300L),
                     0.0L));
      }
      const double cv_e_c = cell_cv_e(cfg, cv_e, zbar_out, cell);
      const double cv_i_c = cell_cv_i(cfg, cv_i, cell);
      const long double e_floor =
          static_cast<long double>(std::max(cv_e_c, 0.0) *
                                   std::max(cfg.numerics.floors.Te, 0.0) +
                                   std::max(cv_i_c, 0.0) *
                                   std::max(cfg.numerics.floors.Ti, 0.0));
      if (!(internal >= e_floor * mass_work[idx]) ||
          !std::isfinite(static_cast<double>(internal))) {
        trial.positivity_failed = true;
        break;
      }
      const double ye = clamp01_host(static_cast<double>(
          ye_mass[idx] / std::max(mass_work[idx], 1.0e-300L)));
      const long double e_specific = internal / mass_work[idx];
      const long double ee_new = ye * e_specific;
      const long double ei_new = (1.0L - ye) * e_specific;
      const long double ee_floor =
          static_cast<long double>(std::max(cv_e_c, 0.0) *
                                   std::max(cfg.numerics.floors.Te, 0.0));
      const long double ei_floor =
          static_cast<long double>(std::max(cv_i_c, 0.0) *
                                   std::max(cfg.numerics.floors.Ti, 0.0));
      if (ee_new + 1.0e-20L < ee_floor || ei_new + 1.0e-20L < ei_floor) {
        trial.positivity_failed = true;
        break;
      }
      mass_out[idx] = static_cast<double>(mass_work[idx]);
      rho_out[idx] = static_cast<double>(mass_work[idx] / target_v);
      ee_out[idx] = static_cast<double>(ee_new);
      ei_out[idx] = static_cast<double>(ei_new);
      if (state.gas_tracer_initialized && idx < gas_tracer_out.size()) {
        if (tracer_mass[idx] < -1.0e-20L ||
            tracer_mass[idx] > mass_work[idx] + 1.0e-20L) {
          trial.positivity_failed = true;
          break;
        }
        gas_tracer_out[idx] = clamp01_host(static_cast<double>(
            tracer_mass[idx] / std::max(mass_work[idx], 1.0e-300L)));
      }
      if (remap_hot_e_eps && idx < hot_e_eps_out.size()) {
        hot_e_eps_out[idx] = std::max(
            static_cast<double>(hot_e_eps_mass[idx] /
                                std::max(mass_work[idx], 1.0e-300L)),
            0.0);
      }
      if (remap_burn_eps && idx < burn_eps_out.size()) {
        burn_eps_out[idx] = std::max(
            static_cast<double>(burn_eps_mass[idx] /
                                std::max(mass_work[idx], 1.0e-300L)),
            0.0);
      }
      if (idx < Te_out.size()) {
        Te_out[idx] = cv_e_c > 0.0
                          ? std::max(cfg.numerics.floors.Te,
                                     ee_out[idx] / cv_e_c)
                          : cfg.numerics.floors.Te;
      }
      if (idx < Ti_out.size()) {
        Ti_out[idx] = cv_i_c > 0.0
                          ? std::max(cfg.numerics.floors.Ti,
                                     ei_out[idx] / cv_i_c)
                          : cfg.numerics.floors.Ti;
      }
      if (idx < Pe_out.size()) {
        Pe_out[idx] = gm1 * std::max(rho_out[idx], cfg.numerics.floors.rho) *
                      std::max(ee_out[idx], 0.0);
      }
      if (idx < Pi_out.size()) {
        Pi_out[idx] = gm1 * std::max(rho_out[idx], cfg.numerics.floors.rho) *
                      std::max(ei_out[idx], 0.0);
      }
      if (idx < cs_out.size() && idx < Pe_out.size() && idx < Pi_out.size()) {
        cs_out[idx] = std::sqrt(std::max(
            gamma * (Pe_out[idx] + Pi_out[idx]) /
                std::max(rho_out[idx], cfg.numerics.floors.rho),
            0.0));
      }
      if (idx < hllc_mom_z_out.size()) {
        long double pz = 0.0L;
        const int nverts = active_nverts(state, cell);
        for (int k = 0; k < nverts; ++k) {
          const int node = cell_corner_node_id(state, cell, k);
          const std::size_t cidx = idx * 4U + static_cast<std::size_t>(k);
          if (node >= 0 && static_cast<std::size_t>(node) < v_z_work.size() &&
              cidx < corner_mass_work.size()) {
            pz += static_cast<long double>(corner_mass_work[cidx]) *
                  v_z_work[static_cast<std::size_t>(node)];
          }
        }
        hllc_mom_z_out[idx] = static_cast<double>(pz);
      }
    }
    if (trial.positivity_failed) {
      remap = trial;
      continue;
    }
    if (trial.max_volume_error > 1.0e-12) {
      trial.positivity_failed = true;
      return trial;
    }

    trial.central_after =
        cell_totals(state, central_cells, mass_out, ee_out, ei_out,
                    v_r_work, v_z_work, corner_mass_work);
    trial.central_hash_after =
        central_core_hash(state, central_cells, trial.central_after);
    const SeamTotals totals_after =
        cell_totals(state, all_cells, mass_out, ee_out, ei_out, v_r_work,
                    v_z_work, corner_mass_work);
    trial.dM_rel = rel_delta(totals_after.mass, totals_before.mass);
    trial.dE_rel = rel_delta(
        totals_after.total_energy,
        totals_before.total_energy + static_cast<long double>(trial.boundary_work));
    const long double momentum_scale = std::max(
        std::sqrt(std::max(2.0L * std::abs(totals_before.mass) *
                               std::abs(totals_before.kinetic_energy),
                           0.0L)),
        1.0e-300L);
    trial.dPr_paired_rel =
        rel_delta_scale(totals_after.momentum_r - totals_before.momentum_r,
                        std::max(std::abs(totals_before.momentum_r),
                                 momentum_scale));
    trial.dPz_paired_rel =
        rel_delta_scale(totals_after.momentum_z - totals_before.momentum_z,
                        std::max(std::abs(totals_before.momentum_z),
                                 momentum_scale));
    trial.central_failed =
        trial.central_hash_before != trial.central_hash_after ||
        rel_delta(trial.central_after.mass, trial.central_before.mass) >
            1.0e-14 ||
        rel_delta(trial.central_after.total_energy,
                  trial.central_before.total_energy) > 1.0e-14 ||
        rel_delta(trial.central_after.momentum_r,
                  trial.central_before.momentum_r) > 1.0e-14 ||
        rel_delta(trial.central_after.momentum_z,
                  trial.central_before.momentum_z) > 1.0e-14;
    trial.conservation_failed =
        trial.dM_rel > 1.0e-10 || trial.dE_rel > 1.0e-10 ||
        trial.dPr_paired_rel > 1.0e-8 || trial.dPz_paired_rel > 1.0e-8;
    if (trial.central_failed || trial.conservation_failed) {
      return trial;
    }

    state.x_r.copy_from_host(x_r_commit);
    state.x_z.copy_from_host(x_z_commit);
    state.mass.copy_from_host(mass_out);
    state.rho.copy_from_host(rho_out);
    state.ee.copy_from_host(ee_out);
    state.ei.copy_from_host(ei_out);
    if (state.Te.size() == Te_out.size()) {
      state.Te.copy_from_host(Te_out);
    }
    if (state.Ti.size() == Ti_out.size()) {
      state.Ti.copy_from_host(Ti_out);
    }
    if (state.Pe.size() == Pe_out.size()) {
      state.Pe.copy_from_host(Pe_out);
    }
    if (state.Pi.size() == Pi_out.size()) {
      state.Pi.copy_from_host(Pi_out);
    }
    if (state.cs.size() == cs_out.size()) {
      state.cs.copy_from_host(cs_out);
    }
    if (state.zbar.size() == zbar_out.size()) {
      state.zbar.copy_from_host(zbar_out);
    }
    if (state.gas_tracer_initialized &&
        state.gas_tracer_Y.size() == gas_tracer_out.size()) {
      state.gas_tracer_Y.copy_from_host(gas_tracer_out);
    }
    if (remap_hot_e_eps &&
        state.hot_e_eps_cum_host.size() == hot_e_eps_out.size()) {
      state.hot_e_eps_cum_host = hot_e_eps_out;
    }
    if (remap_burn_eps &&
        state.burn_eps_cum_host.size() == burn_eps_out.size()) {
      state.burn_eps_cum_host = burn_eps_out;
    }
    if (state.hllc_mom_z_cell.size() == hllc_mom_z_out.size()) {
      state.hllc_mom_z_cell.copy_from_host(hllc_mom_z_out);
    }
    state.v_r.copy_from_host(v_r_work);
    state.v_z.copy_from_host(v_z_work);
    if (state.corner_mass.size() != corner_mass_work.size()) {
      state.corner_mass.reset(corner_mass_work.size());
    }
    state.corner_mass.copy_from_host(corner_mass_work);
    state.corner_mass_initialized = true;
    state.corner_mass_is_lagrangian_invariant = false;
    state.mesh.node_r = state.x_r.data();
    state.mesh.node_z = state.x_z.data();
    state.mesh.recompute_geometry();
    state.vol = state.mesh.cell_vol;
    trial.ok = true;
    return trial;
  }
  remap.positivity_failed = true;
  remap.subcycles = kmax;
  return remap;
}

double smoothstep01(const double x) {
  const double t = std::min(1.0, std::max(0.0, x));
  return t * t * (3.0 - 2.0 * t);
}

bool ring7_node_is_axis_like(const core::State& state,
                             const std::vector<double>& x_r,
                             const int node) {
  if (node < 0 || node >= state.mesh.topo.n_nodes) {
    return false;
  }
  const std::size_t idx = static_cast<std::size_t>(node);
  const std::uint8_t flags =
      idx < state.mesh.topo.node_flags.size() ? state.mesh.topo.node_flags[idx]
                                              : 0U;
  return (flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U ||
         (idx < x_r.size() && std::abs(x_r[idx]) <= 1.0e-12);
}

void build_pole_cap_effective_velocity(const core::State& state,
                                       const std::vector<double>& x_r_current,
                                       const std::vector<double>& raw_u_r,
                                       const std::vector<double>& raw_u_z,
                                       const std::vector<double>& state_v_r,
                                       const std::vector<double>& state_v_z,
                                       std::vector<double>& eff_u_r,
                                       std::vector<double>& eff_u_z,
                                       int& projected_axis_nodes) {
  eff_u_r = raw_u_r;
  eff_u_z = raw_u_z;
  projected_axis_nodes = 0;
  const int n_nodes = state.mesh.topo.n_nodes;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (idx >= eff_u_r.size() || idx >= eff_u_z.size()) {
      continue;
    }
    if (!ring7_node_is_axis_like(state, x_r_current, node)) {
      continue;
    }
    eff_u_r[idx] = 0.0;
    if (idx < state_v_z.size()) {
      eff_u_z[idx] = state_v_z[idx];
    }
    if (idx < state_v_r.size()) {
      eff_u_r[idx] = 0.0;
    }
    ++projected_axis_nodes;
  }
}

double pole_cap_node_edge_scale(const core::State& state,
                                const std::vector<double>& x_r,
                                const std::vector<double>& x_z,
                                const int node) {
  if (!state.mesh.topo.multiblock.has_value() || node < 0 ||
      node >= state.mesh.topo.n_nodes) {
    return 0.0;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  double h = std::numeric_limits<double>::infinity();
  for (int cell = 0; cell < state.mesh.topo.n_cells; ++cell) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(state, cell);
    int corner = -1;
    for (int k = 0; k < nverts; ++k) {
      if (mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)] ==
          node) {
        corner = k;
        break;
      }
    }
    if (corner < 0) {
      continue;
    }
    for (const int dk : {-1, 1}) {
      const int other_corner = (corner + dk + nverts) % nverts;
      const int other =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + other_corner)];
      if (other < 0 || other >= state.mesh.topo.n_nodes) {
        continue;
      }
      const std::size_t a = static_cast<std::size_t>(node);
      const std::size_t b = static_cast<std::size_t>(other);
      if (a >= x_r.size() || a >= x_z.size() || b >= x_r.size() ||
          b >= x_z.size()) {
        continue;
      }
      const double dr = x_r[a] - x_r[b];
      const double dz = x_z[a] - x_z[b];
      const double len = std::hypot(dr, dz);
      if (len > 0.0 && std::isfinite(len)) {
        h = std::min(h, len);
      }
    }
  }
  return std::isfinite(h) ? h : 0.0;
}

struct PoleCapSourceVelocityGuard {
  bool valid = true;
  int violating_node = -1;
  double max_ratio = 0.0;
};

PoleCapSourceVelocityGuard check_pole_cap_source_velocity_guard(
    const core::State& state,
    const Ring7PoleCap& cap,
    const std::vector<double>& x_r_current,
    const std::vector<double>& x_z_current,
    const std::vector<double>& u_eff_r,
    const std::vector<double>& u_eff_z,
    const double dt_trial) {
  PoleCapSourceVelocityGuard guard;
  constexpr double chi = 0.25;
  for (const int node : cap.nodes) {
    if (node < 0 || node >= state.mesh.topo.n_nodes) {
      continue;
    }
    const std::size_t idx = static_cast<std::size_t>(node);
    if (idx >= u_eff_r.size() || idx >= u_eff_z.size()) {
      guard.valid = false;
      guard.violating_node = node;
      guard.max_ratio = std::numeric_limits<double>::infinity();
      continue;
    }
    const double h_node =
        pole_cap_node_edge_scale(state, x_r_current, x_z_current, node);
    const double displacement =
        std::hypot(dt_trial * u_eff_r[idx], dt_trial * u_eff_z[idx]);
    const double ratio =
        h_node > 0.0 ? displacement / h_node
                     : std::numeric_limits<double>::infinity();
    if (ratio > guard.max_ratio) {
      guard.max_ratio = ratio;
      guard.violating_node = node;
    }
    if (!(ratio <= chi) || !std::isfinite(ratio)) {
      guard.valid = false;
    }
  }
  return guard;
}

void build_pole_cap_candidate_velocity(const core::State& state,
                                       const Ring7PoleCap& cap,
                                       const std::vector<double>& u_r,
                                       const std::vector<double>& u_z,
                                       const double alpha_center,
                                       std::vector<double>& w_r,
                                       std::vector<double>& w_z) {
  w_r = u_r;
  w_z = u_z;
  if (!cap.valid || cap.layers <= 0) {
    return;
  }
  const double alpha0 = std::min(1.0, std::max(0.0, alpha_center));
  for (const int node : cap.nodes) {
    if (node < 0 || static_cast<std::size_t>(node) >= w_z.size()) {
      continue;
    }
    const int distance =
        static_cast<std::size_t>(node) < cap.node_distance.size()
            ? cap.node_distance[static_cast<std::size_t>(node)]
            : cap.layers;
    const double s =
        cap.layers > 0 ? static_cast<double>(distance) /
                             static_cast<double>(cap.layers)
                       : 1.0;
    const double alpha = alpha0 + (1.0 - alpha0) * smoothstep01(s);
    w_z[static_cast<std::size_t>(node)] =
        alpha * u_z[static_cast<std::size_t>(node)];
  }
}

const MeshQualityRzVolumeCellMargin* margin_for_cell(
    const std::vector<MeshQualityRzVolumeCellMargin>& margins,
    const int cell) {
  for (const auto& margin : margins) {
    if (margin.cell == cell) {
      return &margin;
    }
  }
  return nullptr;
}

void log_pole_axis_velocity_audit(
    const core::State& state,
    const int request_cell,
    const double dt_trial,
    const std::vector<std::uint8_t>& cap_node_mask,
    const std::vector<std::uint8_t>& central_node_mask,
    const std::vector<double>& x_r_current,
    const std::vector<double>& x_z_current,
    const std::vector<double>& state_v_z,
    const std::vector<double>& captured_u_half_z,
    const std::vector<double>& effective_u_half_z,
    const std::vector<double>& mass,
    std::vector<double> corner_mass) {
  if (!env_flag_value("TENRYU_I1B_POLE_AXIS_VEL_AUDIT") ||
      request_cell != 608) {
    return;
  }
  const int n_nodes = state.mesh.topo.n_nodes;
  if (static_cast<int>(x_r_current.size()) < n_nodes ||
      static_cast<int>(x_z_current.size()) < n_nodes ||
      static_cast<int>(state_v_z.size()) < n_nodes ||
      static_cast<int>(captured_u_half_z.size()) < n_nodes ||
      static_cast<int>(effective_u_half_z.size()) < n_nodes) {
    return;
  }

  ensure_corner_mass_work(state, mass, corner_mass);
  std::vector<long double> node_mass;
  std::vector<long double> node_pr;
  std::vector<long double> node_pz;
  std::vector<double> zero_v(static_cast<std::size_t>(n_nodes), 0.0);
  build_node_mass_momentum(state, corner_mass, zero_v, zero_v, node_mass,
                           node_pr, node_pz);

  std::vector<int> audit_nodes;
  const int nverts = active_nverts(state, request_cell);
  for (int k = 0; k < nverts; ++k) {
    const int node = cell_corner_node_id(state, request_cell, k);
    if (node >= 0 && node < n_nodes) {
      audit_nodes.push_back(node);
    }
  }

  std::vector<int> axis_chain_nodes;
  for (int node = 0; node < n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    const std::uint8_t flags =
        idx < state.mesh.topo.node_flags.size() ? state.mesh.topo.node_flags[idx]
                                                : 0U;
    if ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U ||
        std::abs(x_r_current[idx]) <= 1.0e-12) {
      axis_chain_nodes.push_back(node);
    }
  }
  std::sort(axis_chain_nodes.begin(), axis_chain_nodes.end(),
            [&](const int a, const int b) {
              const double za = x_z_current[static_cast<std::size_t>(a)];
              const double zb = x_z_current[static_cast<std::size_t>(b)];
              return za == zb ? a < b : za < zb;
            });

  const std::vector<int> request_corner_nodes = audit_nodes;
  for (const int node : request_corner_nodes) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (std::abs(x_r_current[idx]) > 1.0e-12) {
      continue;
    }
    const double z0 = x_z_current[idx];
    const double same_z_tol = 1.0e-10 * std::max(1.0, std::abs(z0));
    for (std::size_t i = 0; i < axis_chain_nodes.size(); ++i) {
      const int axis_node = axis_chain_nodes[i];
      const double z_axis =
          x_z_current[static_cast<std::size_t>(axis_node)];
      if (std::abs(z_axis - z0) <= same_z_tol) {
        audit_nodes.push_back(axis_node);
        if (i > 0) {
          audit_nodes.push_back(axis_chain_nodes[i - 1]);
        }
        if (i + 1U < axis_chain_nodes.size()) {
          audit_nodes.push_back(axis_chain_nodes[i + 1U]);
        }
      }
    }
  }

  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    std::vector<std::uint8_t> seed(static_cast<std::size_t>(n_nodes), 0U);
    for (const int node : audit_nodes) {
      if (node >= 0 && node < n_nodes) {
        seed[static_cast<std::size_t>(node)] = 1U;
      }
    }
    for (int cell = 0; cell < state.mesh.topo.n_cells; ++cell) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      const int cnverts = active_nverts(state, cell);
      bool touches_seed = false;
      for (int k = 0; k < cnverts; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        touches_seed = touches_seed ||
                       (node >= 0 && node < n_nodes &&
                        seed[static_cast<std::size_t>(node)] != 0U);
      }
      if (!touches_seed) {
        continue;
      }
      for (int k = 0; k < cnverts; ++k) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (node < 0 || node >= n_nodes) {
          continue;
        }
        const std::size_t idx = static_cast<std::size_t>(node);
        const std::uint8_t flags =
            idx < state.mesh.topo.node_flags.size()
                ? state.mesh.topo.node_flags[idx]
                : 0U;
        if ((flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U ||
            std::abs(x_r_current[idx]) <= 1.0e-12) {
          audit_nodes.push_back(node);
        }
      }
    }
  }

  sort_unique(audit_nodes);

  for (const int node : audit_nodes) {
    const std::size_t idx = static_cast<std::size_t>(node);
    const std::uint8_t flags =
        idx < state.mesh.topo.node_flags.size() ? state.mesh.topo.node_flags[idx]
                                                : 0U;
    const bool is_axis =
        std::abs(x_r_current[idx]) <= 1.0e-12 ||
        (flags & (mesh::NODE_AXIS | mesh::NODE_POLE_AXIS)) != 0U;
    const double state_uz = state_v_z[idx];
    const double captured_uz = captured_u_half_z[idx];
    const double effective_uz = effective_u_half_z[idx];
    const double raw_dt_uz = dt_trial * captured_uz;
    const double raw_x_z_lag = x_z_current[idx] + raw_dt_uz;
    const double eff_dt_uz = dt_trial * effective_uz;
    const double eff_x_z_lag = x_z_current[idx] + eff_dt_uz;
    core::log_info("[ring7_pole_axis_vel_audit] step=" +
                   std::to_string(state.step) +
                   " t=" + sci(state.t) +
                   " t_hex=" + hex_float(state.t) +
                   " dt_trial=" + sci(dt_trial) +
                   " dt_trial_hex=" + hex_float(dt_trial) +
                   " request_cell=" + std::to_string(request_cell) +
                   " node=" + std::to_string(node) +
                   " is_axis=" + std::to_string(is_axis ? 1 : 0) +
                   " is_bbsw_owned=" +
                   std::to_string((flags & mesh::NODE_POLE_AXIS) != 0U ? 1
                                                                        : 0) +
                   " bbsw_owned_basis=node_pole_axis_flag" +
                   " cap_node_mask=" +
                   std::to_string(idx < cap_node_mask.size()
                                      ? static_cast<int>(cap_node_mask[idx])
                                      : 0) +
                   " central_pseudo_core=" +
                   std::to_string(idx < central_node_mask.size()
                                      ? static_cast<int>(central_node_mask[idx])
                                      : 0) +
                   " x_r_current=" + sci(x_r_current[idx]) +
                   " x_r_current_hex=" + hex_float(x_r_current[idx]) +
                   " x_z_current=" + sci(x_z_current[idx]) +
                   " x_z_current_hex=" + hex_float(x_z_current[idx]) +
                   " state_u_z=" + sci(state_uz) +
                   " state_u_z_hex=" + hex_float(state_uz) +
                   " captured_u_half_z=" + sci(captured_uz) +
                   " captured_u_half_z_hex=" + hex_float(captured_uz) +
                   " effective_u_half_z=" + sci(effective_uz) +
                   " effective_u_half_z_hex=" + hex_float(effective_uz) +
                   " node_mass=" +
                   sci(static_cast<double>(idx < node_mass.size()
                                               ? node_mass[idx]
                                               : 0.0L)) +
                   " node_mass_hex=" +
                   hex_float(static_cast<double>(idx < node_mass.size()
                                                     ? node_mass[idx]
                                                     : 0.0L)) +
                   " force_unavailable=1"
                   " bbsw_proj_route=state_committed_axis_velocity"
                   " dt_trial_times_captured_u_half_z=" + sci(raw_dt_uz) +
                   " dt_trial_times_captured_u_half_z_hex=" +
                   hex_float(raw_dt_uz) +
                   " raw_x_z_lag=" + sci(raw_x_z_lag) +
                   " raw_x_z_lag_hex=" + hex_float(raw_x_z_lag) +
                   " dt_trial_times_effective_u_half_z=" + sci(eff_dt_uz) +
                   " dt_trial_times_effective_u_half_z_hex=" +
                   hex_float(eff_dt_uz) +
                   " x_z_lag=" + sci(eff_x_z_lag) +
                   " x_z_lag_hex=" + hex_float(eff_x_z_lag));
  }
}

void run_ring7_pole_cap_oracle(core::State& state,
                               const core::Config& cfg,
                               const HydroEOSContext* eos_ctx,
                               const parallel::Reduction* reduction,
                               const char* stage) {
  const int request_cell = state.ring7_pole_cap_request_cell;
  state.ring7_pole_cap_oracle_requested = false;
  state.ring7_pole_cap_request_cell = -1;
  if (!is_driven_pole_cap_cell(state, request_cell)) {
    core::log_info("[ring7_pole_cap] step=" + std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " status=not_driven_pole request_cell=" +
                   std::to_string(request_cell));
    return;
  }
  if (!state.ring7_last_failed_velocity_valid ||
      state.ring7_last_failed_cell != request_cell ||
      state.ring7_last_failed_v_r.size() != state.x_r.size() ||
      state.ring7_last_failed_v_z.size() != state.x_z.size() ||
      !(state.ring7_last_failed_dt > 0.0)) {
    core::log_info("[ring7_pole_cap] step=" + std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " status=no_failed_u_half request_cell=" +
                   std::to_string(request_cell));
    return;
  }
  const int min_layers =
      std::max(1, env_int_value("TENRYU_I1B_POLE_CAP_LAYERS_MIN", 1));
  const int max_layers = std::max(
      min_layers, env_int_value("TENRYU_I1B_POLE_CAP_LAYERS_MAX", 2));
  const double alpha0 =
      env_double_value("TENRYU_I1B_POLE_CAP_ALPHA_CENTER", 0.0);
  const double alpha1 =
      env_double_value("TENRYU_I1B_POLE_CAP_ALPHA_SECOND", 0.25);
  const double eta_accept =
      env_double_value("TENRYU_I1B_POLE_CAP_ETA_ACCEPT", 1.0);
  const double dt_trial = state.ring7_last_failed_dt;
  double boundary_pressure = 0.0;
  if (cfg.numerics.hydro.boundary_2d.r_outer == "pressure" &&
      state.pressure_drive_1d.has_value()) {
    boundary_pressure = state.pressure_drive_1d->eval(state.t);
  }

  std::vector<double> u_r;
  std::vector<double> u_z;
  state.ring7_last_failed_v_r.copy_to_host(u_r);
  state.ring7_last_failed_v_z.copy_to_host(u_z);
  std::vector<double> x_r_current;
  std::vector<double> x_z_current;
  std::vector<double> state_v_r;
  std::vector<double> state_v_z;
  state.x_r.copy_to_host(x_r_current);
  state.x_z.copy_to_host(x_z_current);
  state.v_r.copy_to_host(state_v_r);
  state.v_z.copy_to_host(state_v_z);
  std::vector<double> u_eff_r;
  std::vector<double> u_eff_z;
  int projected_axis_nodes = 0;
  build_pole_cap_effective_velocity(state,
                                    x_r_current,
                                    u_r,
                                    u_z,
                                    state_v_r,
                                    state_v_z,
                                    u_eff_r,
                                    u_eff_z,
                                    projected_axis_nodes);
  core::DeviceArray<double> d_u_eff_r(u_eff_r.size());
  core::DeviceArray<double> d_u_eff_z(u_eff_z.size());
  d_u_eff_r.copy_from_host(u_eff_r);
  d_u_eff_z.copy_from_host(u_eff_z);

  const MeshQualityDtLimit baseline_limit =
      compute_mesh_quality_dt_limit(state,
                                    cfg,
                                    state.x_r.data(),
                                    state.x_z.data(),
                                    d_u_eff_r.data(),
                                    d_u_eff_z.data(),
                                    dt_trial,
                                    reduction);
  const std::vector<int> probe_cells = {request_cell};
  const std::vector<MeshQualityRzVolumeCellMargin> baseline_margins =
      compute_multiblock_mesh_quality_rz_volume_cell_margins(
          state,
          cfg,
          state.x_r.data(),
          state.x_z.data(),
          d_u_eff_r.data(),
          d_u_eff_z.data(),
          dt_trial,
          probe_cells);
  const MeshQualityRzVolumeCellMargin* baseline_margin =
      margin_for_cell(baseline_margins, request_cell);

  bool any_pass = false;
  int best_layers = -1;
  double best_alpha = 0.0;
  double best_sigma = -1.0;
  double best_eta = -1.0;
  bool selected_pass = false;
  double selected_eta = -1.0;
  double selected_sigma = -1.0;
  double selected_alpha = 0.0;
  Ring7PoleCap selected_cap;
  std::vector<double> selected_w_r;
  std::vector<double> selected_w_z;
  int option = 0;
  const std::array<double, 2> alpha_options = {alpha0, alpha1};
  for (int layers = min_layers; layers <= max_layers; ++layers) {
    const Ring7PoleCap cap = build_ring7_pole_cap(state, request_cell, layers);
    for (const double alpha_center : alpha_options) {
      std::vector<double> w_r;
      std::vector<double> w_z;
      build_pole_cap_candidate_velocity(state, cap, u_eff_r, u_eff_z, alpha_center,
                                        w_r, w_z);
      core::DeviceArray<double> d_w_r(w_r.size());
      core::DeviceArray<double> d_w_z(w_z.size());
      d_w_r.copy_from_host(w_r);
      d_w_z.copy_from_host(w_z);
      const MeshQualityDtLimit candidate_limit =
          compute_mesh_quality_dt_limit(state,
                                        cfg,
                                        state.x_r.data(),
                                        state.x_z.data(),
                                        d_w_r.data(),
                                        d_w_z.data(),
                                        dt_trial,
                                        reduction);
      const std::vector<MeshQualityRzVolumeCellMargin> candidate_margins =
          compute_multiblock_mesh_quality_rz_volume_cell_margins(
              state,
              cfg,
              state.x_r.data(),
              state.x_z.data(),
              d_w_r.data(),
              d_w_z.data(),
              dt_trial,
              probe_cells);
      const MeshQualityRzVolumeCellMargin* candidate_margin =
          margin_for_cell(candidate_margins, request_cell);
      const double candidate_eta =
          candidate_margin != nullptr ? candidate_margin->eta_safe : -1.0;
      const double candidate_sigma =
          candidate_margin != nullptr ? candidate_margin->sigma_raw : -1.0;
      const bool pass =
          candidate_limit.admissible ||
          (candidate_limit.sigma_safe >= eta_accept &&
           candidate_eta >= eta_accept);
      if (candidate_eta > best_eta ||
          (candidate_eta == best_eta && candidate_sigma > best_sigma)) {
        best_eta = candidate_eta;
        best_sigma = candidate_sigma;
        best_layers = layers;
        best_alpha = alpha_center;
      }
      if (pass &&
          (!selected_pass || candidate_eta > selected_eta ||
           (candidate_eta == selected_eta && layers < selected_cap.layers))) {
        selected_pass = true;
        selected_eta = candidate_eta;
        selected_sigma = candidate_sigma;
        selected_alpha = alpha_center;
        selected_cap = cap;
        selected_w_r = w_r;
        selected_w_z = w_z;
      }
      any_pass = any_pass || pass;
      core::log_info(
          "[ring7_pole_cap] step=" + std::to_string(state.step) +
          " stage=" + std::string(stage != nullptr ? stage : "?") +
          " request_cell=" + std::to_string(request_cell) +
          " option=" + std::to_string(option++) +
          " diagnostic_only=1 commit=0 state_mutation=0 coord_commit=0" +
          " layers=" + std::to_string(layers) +
          " alpha_center=" + sci(alpha_center) +
          " cap_cells=" + std::to_string(cap.cells.size()) +
          " cap_nodes=" + std::to_string(cap.nodes.size()) +
          " central_violations=" + std::to_string(cap.central_violations) +
          " baseline_admissible=" +
          std::to_string(baseline_limit.admissible ? 1 : 0) +
          " baseline_first_cell=" +
          std::to_string(baseline_limit.first_cell) +
          " baseline_sigma_safe=" + sci(baseline_limit.sigma_safe) +
          " baseline_cell_sigma=" +
          sci(baseline_margin != nullptr ? baseline_margin->sigma_raw : -1.0) +
          " baseline_cell_eta=" +
          sci(baseline_margin != nullptr ? baseline_margin->eta_safe : -1.0) +
          " candidate_admissible=" +
          std::to_string(candidate_limit.admissible ? 1 : 0) +
          " candidate_first_cell=" +
          std::to_string(candidate_limit.first_cell) +
          " candidate_sigma_safe=" + sci(candidate_limit.sigma_safe) +
          " candidate_cell_sigma=" + sci(candidate_sigma) +
          " candidate_cell_eta=" + sci(candidate_eta) +
          " pass=" + std::to_string(pass ? 1 : 0));
    }
  }
  core::log_info("[ring7_pole_cap] step=" + std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " summary=1 request_cell=" + std::to_string(request_cell) +
                 " any_pass=" + std::to_string(any_pass ? 1 : 0) +
                 " best_layers=" + std::to_string(best_layers) +
                 " best_alpha_center=" + sci(best_alpha) +
                 " best_cell_sigma=" + sci(best_sigma) +
                 " best_cell_eta=" + sci(best_eta) +
                 " dt_trial=" + sci(dt_trial) +
                 " retained_failed_sigma=" +
                 sci(state.ring7_last_failed_sigma_safe));
  if (!selected_pass || selected_w_r.empty() || selected_w_z.empty()) {
    core::log_info("[ring7_pole_cap_remap] step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " fired=1 commit=0 status=no_passing_candidate" +
                   " request_cell=" + std::to_string(request_cell) +
                   " eta_prod_proxy=" + sci(best_eta));
    return;
  }

  std::vector<double> x_r_lag = x_r_current;
  std::vector<double> x_z_lag = x_z_current;
  std::vector<double> x_r_cap = x_r_current;
  std::vector<double> x_z_cap = x_z_current;
  std::vector<double> x_r_commit = x_r_current;
  std::vector<double> x_z_commit = x_z_current;
  std::vector<std::uint8_t> cap_node_mask(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const int node : selected_cap.nodes) {
    if (node >= 0 && node < state.mesh.topo.n_nodes) {
      cap_node_mask[static_cast<std::size_t>(node)] = 1U;
    }
  }
  const PoleCapSourceVelocityGuard source_velocity_guard =
      check_pole_cap_source_velocity_guard(state,
                                           selected_cap,
                                           x_r_current,
                                           x_z_current,
                                           u_eff_r,
                                           u_eff_z,
                                           dt_trial);
  if (!source_velocity_guard.valid) {
    core::log_info("[ring7_pole_cap_remap] step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " fired=1 commit=0 status=rejected_invalid_source_velocity" +
                   " request_cell=" + std::to_string(request_cell) +
                   " layers=" + std::to_string(selected_cap.layers) +
                   " alpha_center=" + sci(selected_alpha) +
                   " max_source_displacement_ratio=" +
                   sci(source_velocity_guard.max_ratio) +
                   " violating_node=" +
                   std::to_string(source_velocity_guard.violating_node) +
                   " chi=2.50000000000000000e-01" +
                   " projected_axis_nodes=" +
                   std::to_string(projected_axis_nodes) +
                   " bbsw_proj_route=state_committed_axis_velocity" +
                   " eta_prod_proxy=" + sci(selected_eta));
    return;
  }
  bool finite_target = true;
  for (int node = 0; node < state.mesh.topo.n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    x_r_lag[idx] = x_r_current[idx] + dt_trial * u_eff_r[idx];
    x_z_lag[idx] = x_z_current[idx] + dt_trial * u_eff_z[idx];
    if (cap_node_mask[idx] != 0U) {
      x_r_cap[idx] = x_r_current[idx] + dt_trial * selected_w_r[idx];
      x_z_cap[idx] = x_z_current[idx] + dt_trial * selected_w_z[idx];
    } else {
      x_r_cap[idx] = x_r_lag[idx];
      x_z_cap[idx] = x_z_lag[idx];
    }
    if (x_r_cap[idx] < 0.0 && x_r_cap[idx] > -1.0e-30) {
      x_r_cap[idx] = 0.0;
    }
    if (x_r_lag[idx] < 0.0 && x_r_lag[idx] > -1.0e-30) {
      x_r_lag[idx] = 0.0;
    }
    if (cap_node_mask[idx] != 0U) {
      x_r_commit[idx] = x_r_cap[idx];
      x_z_commit[idx] = x_z_cap[idx];
    }
    finite_target = finite_target && std::isfinite(x_r_lag[idx]) &&
                    std::isfinite(x_z_lag[idx]) &&
                    std::isfinite(x_r_cap[idx]) &&
                    std::isfinite(x_z_cap[idx]) &&
                    std::isfinite(x_r_commit[idx]) &&
                    std::isfinite(x_z_commit[idx]) &&
                    x_r_lag[idx] >= 0.0 && x_r_cap[idx] >= 0.0 &&
                    x_r_commit[idx] >= 0.0;
  }
  int central_target_violations = 0;
  const std::vector<int> central_cells = central_excluded_cells(state);
  std::vector<std::uint8_t> central_node_mask(
      static_cast<std::size_t>(state.mesh.topo.n_nodes), 0U);
  for (const int cell : central_cells) {
    const int nverts = active_nverts(state, cell);
    for (int k = 0; k < nverts; ++k) {
      const int node = cell_corner_node_id(state, cell, k);
      if (node >= 0 && node < state.mesh.topo.n_nodes) {
        central_node_mask[static_cast<std::size_t>(node)] = 1U;
      }
    }
  }
  for (int node = 0; node < state.mesh.topo.n_nodes; ++node) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (central_node_mask[idx] != 0U &&
        std::hypot(x_r_cap[idx] - x_r_lag[idx],
                   x_z_cap[idx] - x_z_lag[idx]) > 1.0e-30) {
      ++central_target_violations;
    }
  }
  if (!finite_target || central_target_violations != 0) {
    core::log_info("[ring7_pole_cap_remap] step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " fired=1 commit=0 status=rejected_target" +
                   " request_cell=" + std::to_string(request_cell) +
                   " layers=" + std::to_string(selected_cap.layers) +
                   " alpha_center=" + sci(selected_alpha) +
                   " finite_target=" +
                   std::to_string(finite_target ? 1 : 0) +
                   " central_target_violations=" +
                   std::to_string(central_target_violations) +
                   " eta_prod_proxy=" + sci(selected_eta));
    return;
  }

  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> Te;
  std::vector<double> Ti;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> corner_mass;
  std::vector<double> corner_volume;
  std::vector<double> gas_tracer_Y;
  std::vector<double> zbar;
  std::vector<double> cv_e;
  std::vector<double> cv_i;
  std::vector<double> cs;
  std::vector<double> hllc_mom_z_cell;
  state.mass.copy_to_host(mass);
  state.rho.copy_to_host(rho);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.Te.copy_to_host(Te);
  state.Ti.copy_to_host(Ti);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  if (state.gas_tracer_Y.size() ==
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.gas_tracer_Y.copy_to_host(gas_tracer_Y);
  }
  if (state.zbar.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.zbar.copy_to_host(zbar);
  }
  if (state.cv_e.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cv_e.copy_to_host(cv_e);
  }
  if (state.cv_i.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cv_i.copy_to_host(cv_i);
  }
  if (state.cs.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cs.copy_to_host(cs);
  }
  if (state.hllc_mom_z_cell.size() ==
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.hllc_mom_z_cell.copy_to_host(hllc_mom_z_cell);
  }
  if (state.corner_mass.size() >=
      static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U) {
    state.corner_mass.copy_to_host(corner_mass);
  }
  if (state.corner_volume.size() >=
      static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U) {
    state.corner_volume.copy_to_host(corner_volume);
  }

  log_pole_axis_velocity_audit(state,
                               request_cell,
                               dt_trial,
                               cap_node_mask,
                               central_node_mask,
                               x_r_current,
                               x_z_current,
                               state_v_z,
                               u_z,
                               u_eff_z,
                               mass,
                               corner_mass);

  const std::vector<double> lag_corner_volume =
      corner_volumes_for_coords(state, x_r_lag, x_z_lag, corner_volume);
  const PoleCapPacketGeometry geom =
      build_pole_cap_packet_geometry(state,
                                     x_r_current,
                                     x_z_current,
                                     x_r_lag,
                                     x_z_lag,
                                     x_r_cap,
                                     x_z_cap,
                                     lag_corner_volume);
  const int moved_nodes =
      static_cast<int>(
          moved_nodes_from_target(x_r_lag, x_z_lag, x_r_cap, x_z_cap).size());
  if (request_cell == 608) {
    log_pole_cap_closure_probe(state,
                               geom,
                               request_cell,
                               stage,
                               x_r_lag,
                               x_z_lag,
                               x_r_cap,
                               x_z_cap);
  }
  if (geom.invariant_violations != 0 ||
      geom.cap_boundary_nonzero_faces != 0 ||
      geom.orientation_invalid_cells != 0 ||
      !(geom.max_geom_epsilon <= 1.0e-12)) {
    const bool orientation_rejected = geom.orientation_invalid_cells != 0;
    core::log_info("[ring7_pole_cap_remap] step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " fired=1 commit=0 status=" +
                   std::string(orientation_rejected ? "rejected_orientation"
                                                    : "rejected_geometry") +
                   " request_cell=" + std::to_string(request_cell) +
                   " layers=" + std::to_string(selected_cap.layers) +
                   " alpha_center=" + sci(selected_alpha) +
                   " cap_cells=" + std::to_string(geom.patch_cells.size()) +
                   " cap_faces=" + std::to_string(geom.patch_faces) +
                   " cap_boundary_faces=" +
                   std::to_string(geom.cap_boundary_faces) +
                   " cap_boundary_nonzero_faces=" +
                   std::to_string(geom.cap_boundary_nonzero_faces) +
                   " domain_boundary_faces=" +
                   std::to_string(geom.domain_boundary_faces) +
                   " domain_boundary_swept_abs=" +
                   sci(geom.domain_boundary_swept_abs) +
                   " moved_nodes=" + std::to_string(moved_nodes) +
                   " invariant_violations=" +
                   std::to_string(geom.invariant_violations) +
                   " orientation_invalid_cells=" +
                   std::to_string(geom.orientation_invalid_cells) +
                   " central_violations=" +
                   std::to_string(geom.central_violations) +
                   " central_target_violations=" +
                   std::to_string(central_target_violations) +
                   " max_geom_epsilon=" + sci(geom.max_geom_epsilon) +
                   " eta_prod_proxy=" + sci(selected_eta));
    return;
  }

  const Ring7PacketRemapDiagnostics remap_diag =
      apply_pole_cap_packet_remap_transaction(state,
                                              cfg,
                                              eos_ctx,
                                              selected_cap,
                                              x_r_current,
                                              x_z_current,
                                              x_r_lag,
                                              x_z_lag,
                                              x_r_cap,
                                              x_z_cap,
                                              x_r_commit,
                                              x_z_commit,
                                              u_eff_r,
                                              u_eff_z,
                                              boundary_pressure,
                                              rho,
                                              mass,
                                              ee,
                                              ei,
                                              Te,
                                              Ti,
                                              Pe,
                                              Pi,
                                              state_v_r,
                                              state_v_z,
                                              corner_mass,
                                              corner_volume,
                                              gas_tracer_Y,
                                              zbar,
                                              cv_e,
                                              cv_i,
                                              cs,
                                              hllc_mom_z_cell);
  if (!remap_diag.ok) {
    core::log_info("[ring7_pole_cap_remap] step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " fired=1 commit=0 status=rejected" +
                   " request_cell=" + std::to_string(request_cell) +
                   " layers=" + std::to_string(selected_cap.layers) +
                   " alpha_center=" + sci(selected_alpha) +
                   " positivity_failed=" +
                   std::to_string(remap_diag.positivity_failed ? 1 : 0) +
                   " conservation_failed=" +
                   std::to_string(remap_diag.conservation_failed ? 1 : 0) +
                   " central_failed=" +
                   std::to_string(remap_diag.central_failed ? 1 : 0) +
                   " dM_rel=" + sci(remap_diag.dM_rel) +
                   " dE_rel=" + sci(remap_diag.dE_rel) +
                   " dPr_paired_rel=" + sci(remap_diag.dPr_paired_rel) +
                   " dPz_paired_rel=" + sci(remap_diag.dPz_paired_rel) +
                   " Q_mix=" + sci(remap_diag.q_mix) +
                   " boundary_work=" + sci(remap_diag.boundary_work) +
                   " boundary_pressure=" + sci(boundary_pressure) +
                   " subcycle_K=" + std::to_string(remap_diag.subcycles) +
                   " max_volume_error=" +
                   sci(remap_diag.max_volume_error) +
                   " cap_cells=" + std::to_string(geom.patch_cells.size()) +
                   " moved_nodes=" + std::to_string(moved_nodes) +
                   " eta_prod_proxy=" + sci(selected_eta));
    return;
  }
  state.ring7_pole_cap_validation_pending = true;
  state.ring7_pole_cap_validation_cell = request_cell;
  state.ring7_pole_cap_validation_dt = dt_trial;
  state.ring7_pole_cap_eta_prod_proxy = selected_eta;
  core::log_info("[ring7_pole_cap_remap] step=" +
                 std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " fired=1 commit=1 state_mutation=1 coord_commit=1" +
                 " request_cell=" + std::to_string(request_cell) +
                 " layers=" + std::to_string(selected_cap.layers) +
                 " alpha_center=" + sci(selected_alpha) +
                 " dM_rel=" + sci(remap_diag.dM_rel) +
                 " dE_rel=" + sci(remap_diag.dE_rel) +
                 " dPr_paired_rel=" + sci(remap_diag.dPr_paired_rel) +
                 " dPz_paired_rel=" + sci(remap_diag.dPz_paired_rel) +
                 " Q_mix=" + sci(remap_diag.q_mix) +
                 " boundary_work=" + sci(remap_diag.boundary_work) +
                 " boundary_pressure=" + sci(boundary_pressure) +
                 " subcycle_K=" + std::to_string(remap_diag.subcycles) +
                 " cap_cells=" + std::to_string(geom.patch_cells.size()) +
                 " cap_faces=" + std::to_string(geom.patch_faces) +
                 " cap_boundary_faces=" +
                 std::to_string(geom.cap_boundary_faces) +
                 " domain_boundary_faces=" +
                 std::to_string(geom.domain_boundary_faces) +
                 " domain_boundary_swept_abs=" +
                 sci(geom.domain_boundary_swept_abs) +
                 " moved_nodes=" + std::to_string(moved_nodes) +
                 " central_target_violations=" +
                 std::to_string(central_target_violations) +
                 " max_geom_epsilon=" + sci(geom.max_geom_epsilon) +
                 " max_volume_error=" + sci(remap_diag.max_volume_error) +
                 " central_hash_before=" +
                 std::to_string(remap_diag.central_hash_before) +
                 " central_hash_after=" +
                 std::to_string(remap_diag.central_hash_after) +
                 " central_dM_rel=" +
                 sci(rel_delta(remap_diag.central_after.mass,
                               remap_diag.central_before.mass)) +
                 " central_dE_rel=" +
                 sci(rel_delta(remap_diag.central_after.total_energy,
                               remap_diag.central_before.total_energy)) +
                 " eta_prod_proxy=" + sci(selected_eta) +
                 " sigma_prod_proxy=" + sci(selected_sigma));
}

void log_ring7_packet_diagnostics(
    const core::State& state,
    const Ring7QualityMetrics& metrics,
    const Ring7PacketGeometryDiagnostics& diag,
    const int request_cell,
    const bool requested,
    const char* stage) {
  core::log_info("[ring7_seam_packet] step=" + std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " requested=" + std::to_string(requested ? 1 : 0) +
                 " request_cell=" + std::to_string(request_cell) +
                 " fired=0 commit=0 state_mutation=0 coord_commit=0" +
                 " patch_cells=" + std::to_string(diag.patch_cells) +
                 " patch_faces=" + std::to_string(diag.patch_faces) +
                 " boundary_faces=" + std::to_string(diag.boundary_faces) +
                 " moved_nodes=" + std::to_string(diag.moved_nodes) +
                 " seam_physical_faces=" +
                 std::to_string(diag.seam_physical_faces) +
                 " seam_shared_node_faces=" +
                 std::to_string(diag.seam_shared_node_faces) +
                 " seam_duplicate_node_faces=" +
                 std::to_string(diag.seam_duplicate_node_faces) +
                 " seam_unmatched_node_faces=" +
                 std::to_string(diag.seam_unmatched_node_faces) +
                 " seam_cloned_nodes=" +
                 std::to_string(diag.seam_cloned_nodes) +
                 " packets=" + std::to_string(diag.packets) +
                 " invariant_violations=" +
                 std::to_string(diag.invariant_violations) +
                 " central_patch_cells=" +
                 std::to_string(diag.central_patch_cells) +
                 " central_patch_nodes=" +
                 std::to_string(diag.central_patch_nodes) +
                 " boundary_nonzero_faces=" +
                 std::to_string(diag.boundary_nonzero_faces) +
                 " first_bad_boundary_cell=" +
                 std::to_string(diag.first_bad_boundary_cell) +
                 " first_bad_boundary_face=" +
                 std::to_string(diag.first_bad_boundary_face) +
                 " max_geom_epsilon=" + sci(diag.max_geom_epsilon) +
                 " boundary_swept_sum=" + sci(diag.boundary_swept_sum) +
                 " boundary_swept_abs=" + sci(diag.boundary_swept_abs) +
                 " central_hash_before=" +
                 std::to_string(diag.central_hash_before) +
                 " central_hash_after=" +
                 std::to_string(diag.central_hash_after) +
                 " central_dM_rel=" +
                 sci(rel_delta(diag.central_after.mass,
                               diag.central_before.mass)) +
                 " central_dE_rel=" +
                 sci(rel_delta(diag.central_after.total_energy,
                               diag.central_before.total_energy)) +
                 " central_dPr_rel=" +
                 sci(rel_delta(diag.central_after.momentum_r,
                               diag.central_before.momentum_r)) +
                 " central_dPz_rel=" +
                 sci(rel_delta(diag.central_after.momentum_z,
                               diag.central_before.momentum_z)) +
                 " min_q_J=" + sci(metrics.min_q_J) +
                 " min_q_J_cell=" + std::to_string(metrics.min_q_J_cell) +
                 " min_q_edge=" + sci(metrics.min_q_edge) +
                 " min_q_edge_cell=" +
                 std::to_string(metrics.min_q_edge_cell) +
                 " min_eta_V=" + sci(metrics.min_eta_V) +
                 " min_eta_V_cell=" +
                 std::to_string(metrics.min_eta_V_cell));
}

}  // namespace

Ring7SeamPatch discover_ring7_seam_patch(const core::State& state,
                                         const core::Config& cfg) {
  Ring7SeamPatch patch;
  if (cfg.main.dimension != "2D_RZ" || state.mesh.dim != 2 ||
      !state.mesh.topo.multiblock.has_value()) {
    return patch;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const auto& shell = mesh::mesh_topo_multiblock_polar_shell_block(mb);
  if (shell.n_i_cells <= 0 || shell.n_j_cells <= 0 ||
      state.mesh.topo.n_cells <= 0 || state.mesh.topo.n_nodes <= 0) {
    return patch;
  }

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<std::uint8_t> p_mask(static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> node_mask(static_cast<std::size_t>(n_nodes), 0U);

  patch.valid = true;
  patch.shell_ring_i = shell.n_i_cells - 1;
  patch.shell_guard_i = patch.shell_ring_i - 1;
  patch.shell_nj = shell.n_j_cells;
  patch.shell_cell_begin = shell.cell_begin;

  const auto shell_cell = [&](const int i, const int j) {
    return shell.cell_begin + i * shell.n_j_cells + j;
  };
  const auto shell_node = [&](const int i, const int j) {
    return shell.owned_node_begin + i * (shell.n_j_cells + 1) + j;
  };

  for (int j = 0; j < shell.n_j_cells; ++j) {
    const int cell = shell_cell(patch.shell_ring_i, j);
    mark_cell(p_mask, patch.ring7_cells, cell);
    patch.p_seam_cells.push_back(cell);
    if (patch.shell_guard_i >= 0) {
      const int guard = shell_cell(patch.shell_guard_i, j);
      mark_cell(p_mask, patch.inner_guard_cells, guard);
      patch.p_seam_cells.push_back(guard);
    }
  }

  for (const int cell : patch.ring7_cells) {
    const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
    for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
      const int adj = mb.face_adj_csr_indices[static_cast<std::size_t>(off + f)];
      if (adj >= 0 && adj < n_cells && !is_shell_cell(mb, adj)) {
        mark_cell(p_mask, patch.butterfly_adjacent_cells, adj);
        patch.p_seam_cells.push_back(adj);
      }
    }
  }

  std::vector<int> endpoint_nodes;
  for (const int j : {0, shell.n_j_cells}) {
    endpoint_nodes.push_back(shell_node(patch.shell_ring_i, j));
    endpoint_nodes.push_back(shell_node(patch.shell_ring_i + 1, j));
  }
  if (mb.has_trifan_cap) {
    for (int cell = 0; cell < n_cells; ++cell) {
      const mesh::BlockRole role = cell_role(mb, cell);
      if (is_shell_cell(mb, cell) || !is_trifan_role(role)) {
        continue;
      }
      if (cell_has_any_node(state, cell, endpoint_nodes)) {
        mark_cell(p_mask, patch.trifan_end_cells, cell);
        patch.p_seam_cells.push_back(cell);
      }
    }
  }

  for (int i = patch.shell_ring_i; i <= patch.shell_ring_i + 1; ++i) {
    for (int j = 0; j <= shell.n_j_cells; ++j) {
      const int node = shell_node(i, j);
      if (node >= 0 && node < n_nodes &&
          node_mask[static_cast<std::size_t>(node)] == 0U) {
        node_mask[static_cast<std::size_t>(node)] = 1U;
        patch.node_ids.push_back(node);
      }
    }
  }
  for (const int cell : patch.butterfly_adjacent_cells) {
    mark_cell_nodes(state, cell, node_mask, patch.node_ids);
  }
  for (const int cell : patch.trifan_end_cells) {
    mark_cell_nodes(state, cell, node_mask, patch.node_ids);
  }

  sort_unique(patch.ring7_cells);
  sort_unique(patch.inner_guard_cells);
  sort_unique(patch.butterfly_adjacent_cells);
  sort_unique(patch.trifan_end_cells);
  sort_unique(patch.p_seam_cells);
  sort_unique(patch.node_ids);
  return patch;
}

// --- Interior-patch generalization (verdict #5 Q1 order 2+3; env
// TENRYU_I1B_INTERIOR_PATCH_REMAP, default OFF = byte-identical). Reuses the
// PROVEN conservative packet-remap transaction over an on-demand BFS patch
// around any failing POLAR_SHELL interior cell, with a Laplacian (untangling)
// displacement and a REAL quality target (default q>=0.3, not the 3e-8
// degeneracy floor). Requires TENRYU_I1B_RING7_QUOTIENT=1 (shares the request/
// transaction plumbing; the no-request path is a verified no-op).
bool interior_patch_remap_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_INTERIOR_PATCH_REMAP");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_present && raw[0] != '0';
  return env_present ? env_enabled
                     : cfg.numerics.ale.interior_patch_remap_enabled;
}

int interior_patch_layers() {
  static const int layers = [] {
    const char* raw = std::getenv("TENRYU_I1B_INTERIOR_PATCH_LAYERS");
    const int v = raw != nullptr ? std::atoi(raw) : 2;
    return v > 0 ? v : 2;
  }();
  return layers;
}

Ring7SeamPatch discover_interior_patch(const core::State& state,
                                       const core::Config& cfg,
                                       const int seed_cell,
                                       const int n_layers) {
  Ring7SeamPatch patch;
  if (cfg.main.dimension != "2D_RZ" || state.mesh.dim != 2 ||
      !state.mesh.topo.multiblock.has_value()) {
    return patch;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (seed_cell < 0 || seed_cell >= n_cells || !is_shell_cell(mb, seed_cell) ||
      cell_is_central_excluded(state, mb, seed_cell) ||
      is_driven_pole_cap_cell(state, seed_cell)) {
    return patch;
  }
  std::vector<std::uint8_t> cell_mask(static_cast<std::size_t>(n_cells), 0U);
  std::vector<int> frontier{seed_cell};
  cell_mask[static_cast<std::size_t>(seed_cell)] = 1U;
  patch.p_seam_cells.push_back(seed_cell);
  for (int layer = 0; layer < n_layers; ++layer) {
    std::vector<int> next;
    for (const int cell : frontier) {
      const int off =
          mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
      for (int f = 0; f < mesh::kMeshTopoCellStorageSlots; ++f) {
        const int adj =
            mb.face_adj_csr_indices[static_cast<std::size_t>(off + f)];
        if (adj < 0 || adj >= n_cells ||
            cell_mask[static_cast<std::size_t>(adj)] != 0U) {
          continue;
        }
        // v1 scope: stay inside the POLAR_SHELL block (no cross-seam clones)
        // and never touch the central core or the driven pole cap.
        if (!is_shell_cell(mb, adj) ||
            cell_is_central_excluded(state, mb, adj) ||
            is_driven_pole_cap_cell(state, adj)) {
          continue;
        }
        cell_mask[static_cast<std::size_t>(adj)] = 1U;
        patch.p_seam_cells.push_back(adj);
        next.push_back(adj);
      }
    }
    frontier = std::move(next);
  }
  std::sort(patch.p_seam_cells.begin(), patch.p_seam_cells.end());
  patch.ring7_cells = patch.p_seam_cells;
  std::vector<std::uint8_t> node_mask(static_cast<std::size_t>(n_nodes), 0U);
  for (const int cell : patch.p_seam_cells) {
    mark_cell_nodes(state, cell, node_mask, patch.node_ids);
  }
  std::sort(patch.node_ids.begin(), patch.node_ids.end());
  patch.shell_ring_i = -1;  // marks a general interior patch
  patch.valid = !patch.p_seam_cells.empty() && !patch.node_ids.empty();
  return patch;
}

// Freeze every node incident to any NON-patch cell (the Inc3a closure
// contract: only strictly-interior nodes may move; all patch-boundary faces
// stay static so the packet partition conserves by construction).
void augment_mask_with_nonpatch_incidence(const core::State& state,
                                          const Ring7SeamPatch& patch,
                                          std::vector<std::uint8_t>& mask) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (static_cast<int>(mask.size()) != n_nodes) {
    mask.assign(static_cast<std::size_t>(n_nodes), 0U);
  }
  std::vector<std::uint8_t> scratch_mask(static_cast<std::size_t>(n_nodes),
                                         0U);
  std::vector<int> scratch_nodes;
  for (int cell = 0; cell < n_cells; ++cell) {
    if (std::binary_search(patch.p_seam_cells.begin(),
                           patch.p_seam_cells.end(), cell)) {
      continue;
    }
    scratch_nodes.clear();
    mark_cell_nodes(state, cell, scratch_mask, scratch_nodes);
    for (const int node : scratch_nodes) {
      mask[static_cast<std::size_t>(node)] = 1U;
    }
  }
}

// Laplacian (edge-neighbor average) displacement over the patch: the standard
// local untangler for sheared quads. Downstream gates (lambda backtracking,
// sweep cap, path admissibility, eta_V) bound the actual committed motion.
void build_interior_laplacian_displacement(const core::State& state,
                                           const Ring7SeamPatch& patch,
                                           const std::vector<double>& x_r,
                                           const std::vector<double>& x_z,
                                           const double move_scale,
                                           std::vector<double>& d_r,
                                           std::vector<double>& d_z) {
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_nodes = state.mesh.topo.n_nodes;
  d_r.assign(static_cast<std::size_t>(n_nodes), 0.0);
  d_z.assign(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> sum_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> sum_z(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<int> count(static_cast<std::size_t>(n_nodes), 0);
  std::vector<int> cell_nodes;
  for (const int cell : patch.p_seam_cells) {
    cell_nodes.clear();
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(state, cell);
    for (int k = 0; k < nverts; ++k) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (node >= 0 && node < n_nodes) {
        cell_nodes.push_back(node);
      }
    }
    const std::size_t m = cell_nodes.size();
    if (m < 3U) {
      continue;
    }
    for (std::size_t k = 0; k < m; ++k) {
      const int a = cell_nodes[k];
      const int b = cell_nodes[(k + 1U) % m];
      const std::size_t ia = static_cast<std::size_t>(a);
      const std::size_t ib = static_cast<std::size_t>(b);
      sum_r[ia] += x_r[ib];
      sum_z[ia] += x_z[ib];
      ++count[ia];
      sum_r[ib] += x_r[ia];
      sum_z[ib] += x_z[ia];
      ++count[ib];
    }
  }
  for (const int node : patch.node_ids) {
    const std::size_t idx = static_cast<std::size_t>(node);
    if (count[idx] < 2) {
      continue;
    }
    const double inv = 1.0 / static_cast<double>(count[idx]);
    d_r[idx] = move_scale * (sum_r[idx] * inv - x_r[idx]);
    d_z[idx] = move_scale * (sum_z[idx] * inv - x_z[idx]);
  }
}

bool interior_patch_retry_candidate_cell(const core::State& state,
                                         const core::Config& cfg,
                                         const int cell) {
  if (!interior_patch_remap_enabled(cfg) || !ring7_feature_enabled(cfg) ||
      cell < 0 || cell >= state.mesh.topo.n_cells ||
      !state.mesh.topo.multiblock.has_value()) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  return is_shell_cell(mb, cell) &&
         !cell_is_central_excluded(state, mb, cell) &&
         !is_driven_pole_cap_cell(state, cell);
}

bool ring7_seam_retry_candidate_cell(const core::State& state,
                                     const core::Config& cfg,
                                     const int cell) {
  if (!ring7_feature_enabled(cfg) || cell < 0) {
    return false;
  }
  if (is_driven_pole_cap_cell(state, cell)) {
    return false;
  }
  const Ring7SeamPatch patch = discover_ring7_seam_patch(state, cfg);
  if (!patch.valid) {
    return false;
  }
  return std::binary_search(patch.p_seam_cells.begin(),
                            patch.p_seam_cells.end(),
                            cell);
}

bool ring7_driven_pole_cap_retry_candidate_cell(const core::State& state,
                                                const core::Config& cfg,
                                                const int cell) {
  return ring7_feature_enabled(cfg) && is_driven_pole_cap_cell(state, cell);
}

void request_ring7_seam_rezone(core::State& state, const int failing_cell) {
  state.ring7_seam_rezone_requested = true;
  state.ring7_seam_rezone_request_cell = failing_cell;
}

void request_ring7_pole_cap_oracle(core::State& state,
                                   const int failing_cell) {
  state.ring7_pole_cap_oracle_requested = true;
  state.ring7_pole_cap_request_cell = failing_cell;
}

void record_ring7_failed_mesh_velocity(core::State& state,
                                       const core::Config& cfg,
                                       const int failing_cell,
                                       const double sigma_safe,
                                       const double dt_trial,
                                       const double* d_v_r_half,
                                       const double* d_v_z_half) {
  if (!ring7_driven_pole_cap_retry_candidate_cell(state, cfg, failing_cell) ||
      d_v_r_half == nullptr || d_v_z_half == nullptr ||
      !(dt_trial > 0.0)) {
    return;
  }
  const std::size_t n_nodes = state.x_r.size();
  if (state.ring7_last_failed_v_r.size() != n_nodes) {
    state.ring7_last_failed_v_r.reset(n_nodes);
  }
  if (state.ring7_last_failed_v_z.size() != n_nodes) {
    state.ring7_last_failed_v_z.reset(n_nodes);
  }
  CUDA_CHECK(cudaMemcpy(state.ring7_last_failed_v_r.data(),
                        d_v_r_half,
                        n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(state.ring7_last_failed_v_z.data(),
                        d_v_z_half,
                        n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice));
  state.ring7_last_failed_velocity_valid = true;
  state.ring7_last_failed_cell = failing_cell;
  state.ring7_last_failed_sigma_safe = sigma_safe;
  state.ring7_last_failed_dt = dt_trial;
}

void emit_ring7_pole_cap_recomputed_limiter_validation(
    core::State& state,
    const core::Config& cfg,
    const MeshQualityDtLimit& limit,
    const double dt_trial,
    const char* stage) {
  if (!ring7_feature_enabled(cfg) ||
      !state.ring7_pole_cap_validation_pending) {
    return;
  }
  const double expected_dt = state.ring7_pole_cap_validation_dt;
  const double scale = std::max(std::abs(expected_dt), 1.0e-300);
  if (!(expected_dt > 0.0) ||
      std::abs(dt_trial - expected_dt) > 1.0e-10 * scale) {
    return;
  }
  const int request_cell = state.ring7_pole_cap_validation_cell;
  const bool pass = limit.admissible || limit.first_cell != request_cell ||
                    limit.metric != MeshQualityFailingMetric::RzVolume;
  const double eta_true =
      pass ? 1.0 : std::max(0.0, limit.sigma_safe);
  core::log_info("[ring7_pole_cap_remap] step=" +
                 std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " validation=production_recomputed request_cell=" +
                 std::to_string(request_cell) +
                 " pass=" + std::to_string(pass ? 1 : 0) +
                 " eta_prod_true=" + sci(eta_true) +
                 " eta_prod_proxy=" +
                 sci(state.ring7_pole_cap_eta_prod_proxy) +
                 " admissible=" +
                 std::to_string(limit.admissible ? 1 : 0) +
                 " first_cell=" + std::to_string(limit.first_cell) +
                 " sigma_safe=" + sci(limit.sigma_safe) +
                 " dt_trial=" + sci(dt_trial));
  state.ring7_pole_cap_validation_pending = false;
  state.ring7_pole_cap_validation_cell = -1;
  state.ring7_pole_cap_validation_dt = 0.0;
  state.ring7_pole_cap_eta_prod_proxy = 0.0;
}

Ring7SeamRemapResult apply_ring7_seam_quotient_remap(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const double* d_v_r_nominal,
    const double* d_v_z_nominal,
    const double dt_nominal,
    const parallel::Reduction* reduction,
    const char* stage) {
  Ring7SeamRemapResult result;
  result.suggested_dt = dt_nominal;
  if (!ring7_feature_enabled(cfg)) {
    return result;
  }
  if (state.ring7_pole_cap_oracle_requested) {
    result.attempted = true;
    result.requested = true;
    result.request_cell = state.ring7_pole_cap_request_cell;
    run_ring7_pole_cap_oracle(state, cfg, eos_ctx, reduction, stage);
    return result;
  }
  const bool rezone_requested = state.ring7_seam_rezone_requested;
  const int request_cell = state.ring7_seam_rezone_request_cell;
  if (rezone_requested) {
    state.ring7_seam_rezone_requested = false;
    state.ring7_seam_rezone_request_cell = -1;
  }
  result.requested = rezone_requested;
  result.request_cell = request_cell;
  if (!rezone_requested) {
    return result;
  }
  result.attempted = true;
  if (cfg.main.dimension != "2D_RZ" || state.mesh.dim != 2 ||
      !state.mesh.topo.multiblock.has_value() || !(dt_nominal > 0.0) ||
      d_v_r_nominal == nullptr || d_v_z_nominal == nullptr) {
    return result;
  }

  Ring7SeamPatch patch = discover_ring7_seam_patch(state, cfg);
  // Core-growth guard: once ring absorption has grown the pseudo-core into
  // the seam rings, the seam machinery's invariants (central exclusion) no
  // longer hold — decline instead of asserting mid-terminal phase.
  if (patch.valid && state.mesh.topo.multiblock.has_value()) {
    const auto& mb_guard = *state.mesh.topo.multiblock;
    for (const int cell : patch.p_seam_cells) {
      if (cell_is_central_excluded(state, mb_guard, cell)) {
        patch.valid = false;
        break;
      }
    }
  }
  bool interior_mode = false;
  if (interior_patch_remap_enabled(cfg) && request_cell >= 0) {
    // Interior mode for ANY request cell that is not a ring7 (outermost-ring)
    // cell proper: the tangential seam optimizer only makes sense on the
    // outer seam; guard-ring/interior cells need the Laplacian patch relief.
    const bool in_ring7_ring =
        patch.valid && std::find(patch.ring7_cells.begin(),
                                 patch.ring7_cells.end(),
                                 request_cell) != patch.ring7_cells.end();
    if (!in_ring7_ring) {
      Ring7SeamPatch interior = discover_interior_patch(
          state, cfg, request_cell, interior_patch_layers());
      if (interior.valid) {
        patch = std::move(interior);
        interior_mode = true;
      }
    }
  }
  if (!patch.valid || patch.ring7_cells.empty()) {
    return result;
  }

  const int n_nodes = state.mesh.topo.n_nodes;
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  v_r = copy_device_doubles(d_v_r_nominal, n_nodes, "ring7_nominal_v_r");
  v_z = copy_device_doubles(d_v_z_nominal, n_nodes, "ring7_nominal_v_z");
  std::vector<std::uint8_t> boundary_node_mask =
      mesh_boundary_node_mask(state);
  if (interior_mode) {
    augment_mask_with_nonpatch_incidence(state, patch, boundary_node_mask);
  }

  const double q_floor =
      std::max(cfg.numerics.hydro.mesh_quality_dt_corner_j_floor_rel,
               cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel);
  const double q_target =
      interior_mode
          ? env_double_value("TENRYU_I1B_INTERIOR_PATCH_Q_TARGET", 0.3)
          : env_double_value("TENRYU_I1B_RING7_QUOTIENT_Q_TARGET",
                             3.0 * q_floor);
  const double eta_target = env_double_value(
      "TENRYU_I1B_RING7_QUOTIENT_ETA_TARGET", 2.0);
  const double sweep_cap = env_double_value(
      "TENRYU_I1B_RING7_QUOTIENT_SWEEP_CAP", 0.20);
  const double move_scale = env_double_value(
      "TENRYU_I1B_RING7_QUOTIENT_MOVE_SCALE", 0.35);
  const int max_submoves = env_int_value(
      "TENRYU_I1B_RING7_QUOTIENT_MAX_SUBMOVES", 4);
  const int max_backtracks = env_int_value(
      "TENRYU_I1B_RING7_QUOTIENT_BACKTRACKS", 8);
  const double dt_shrink = env_double_value(
      "TENRYU_I1B_RING7_QUOTIENT_DT_SHRINK", 0.5);
  const int packet_lambda_backtracks = env_int_value(
      "TENRYU_I1B_RING7_PACKET_LAMBDA_BACKTRACKS", 5);
  const double packet_lambda_min = env_double_value(
      "TENRYU_I1B_RING7_PACKET_LAMBDA_MIN", 0.1);
  const double packet_lambda_floor =
      std::min(1.0, std::max(packet_lambda_min, 1.0e-6));

  Ring7QualityMetrics metrics =
      evaluate_ring7_candidate_metrics(state, cfg, patch, x_r, x_z, v_r, v_z,
                                       d_v_r_nominal, d_v_z_nominal,
                                       dt_nominal, eta_target);
  result.min_q_J = metrics.min_q_J;
  result.min_q_edge = metrics.min_q_edge;
  result.min_eta_V = metrics.min_eta_V;
  const bool needs_rezone =
      rezone_requested || metrics.min_q_J < q_target ||
      metrics.min_q_edge < q_target ||
      metrics.min_eta_V < eta_target;
  if (!needs_rezone) {
    return result;
  }

  std::vector<double> mass;
  std::vector<double> rho;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> Te;
  std::vector<double> Ti;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> corner_mass;
  std::vector<double> corner_volume;
  std::vector<double> state_v_r;
  std::vector<double> state_v_z;
  std::vector<double> gas_tracer_Y;
  std::vector<double> zbar;
  std::vector<double> cv_e;
  std::vector<double> cv_i;
  std::vector<double> cs;
  std::vector<double> hllc_mom_z_cell;
  state.mass.copy_to_host(mass);
  state.rho.copy_to_host(rho);
  state.ee.copy_to_host(ee);
  state.ei.copy_to_host(ei);
  state.Te.copy_to_host(Te);
  state.Ti.copy_to_host(Ti);
  state.Pe.copy_to_host(Pe);
  state.Pi.copy_to_host(Pi);
  state.v_r.copy_to_host(state_v_r);
  state.v_z.copy_to_host(state_v_z);
  if (state.gas_tracer_Y.size() ==
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.gas_tracer_Y.copy_to_host(gas_tracer_Y);
  }
  if (state.zbar.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.zbar.copy_to_host(zbar);
  }
  if (state.cv_e.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cv_e.copy_to_host(cv_e);
  }
  if (state.cv_i.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cv_i.copy_to_host(cv_i);
  }
  if (state.cs.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.cs.copy_to_host(cs);
  }
  if (state.hllc_mom_z_cell.size() ==
      static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    state.hllc_mom_z_cell.copy_to_host(hllc_mom_z_cell);
  }
  if (state.corner_mass.size() >=
      static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U) {
    state.corner_mass.copy_to_host(corner_mass);
  }
  if (state.corner_volume.size() >=
      static_cast<std::size_t>(state.mesh.topo.n_cells) * 4U) {
    state.corner_volume.copy_to_host(corner_volume);
  }

  std::vector<double> current_r = x_r;
  std::vector<double> current_z = x_z;
  double best_score = ring7_objective(metrics, q_target, eta_target);
  double accepted_max_swept = 0.0;
  bool accepted_any = false;

  for (int submove = 0; submove < max_submoves; ++submove) {
    std::vector<double> d_r;
    std::vector<double> d_z;
    if (interior_mode) {
      build_interior_laplacian_displacement(state, patch, current_r,
                                            current_z, move_scale, d_r, d_z);
    } else {
      build_tangential_displacement(state, patch, current_r, current_z,
                                    move_scale, d_r, d_z);
    }
    bool accepted_submove = false;
    std::vector<double> best_r = current_r;
    std::vector<double> best_z = current_z;
    Ring7QualityMetrics best_metrics = metrics;
    double best_swept = 0.0;

    double lambda = 1.0;
    for (int bt = 0; bt < max_backtracks; ++bt, lambda *= 0.5) {
      std::vector<double> trial_r = current_r;
      std::vector<double> trial_z = current_z;
      bool finite_candidate = true;
      for (const int node : patch.node_ids) {
        const std::size_t idx = static_cast<std::size_t>(node);
        if (idx < boundary_node_mask.size() &&
            boundary_node_mask[idx] != 0U) {
          trial_r[idx] = x_r[idx];
          trial_z[idx] = x_z[idx];
          continue;
        }
        trial_r[idx] += lambda * d_r[idx];
        trial_z[idx] += lambda * d_z[idx];
        if (!std::isfinite(trial_r[idx]) || !std::isfinite(trial_z[idx]) ||
            trial_r[idx] < -1.0e-30) {
          finite_candidate = false;
          break;
        }
        if (trial_r[idx] < 0.0) {
          trial_r[idx] = 0.0;
        }
      }
      if (!finite_candidate) {
        continue;
      }
      const double swept =
          max_patch_swept_fraction(state, patch, current_r, current_z,
                                   trial_r, trial_z);
      if (!(swept <= sweep_cap) || !std::isfinite(swept)) {
        continue;
      }
      if (!path_admissible_between(state, cfg, current_r, current_z,
                                   trial_r, trial_z)) {
        continue;
      }
      Ring7QualityMetrics trial_metrics =
          evaluate_ring7_candidate_metrics(state, cfg, patch, trial_r, trial_z,
                                           v_r, v_z, d_v_r_nominal,
                                           d_v_z_nominal, dt_nominal,
                                           eta_target);
      if (trial_metrics.min_eta_V <= 1.0) {
        continue;
      }
      const double score =
          ring7_objective(trial_metrics, q_target, eta_target);
      const bool accept_requested_candidate =
          rezone_requested && !accepted_submove && std::isfinite(score);
      if (score < best_score || accept_requested_candidate) {
        best_score = score;
        best_r = std::move(trial_r);
        best_z = std::move(trial_z);
        best_metrics = trial_metrics;
        best_swept = swept;
        accepted_submove = true;
      }
      if (trial_metrics.min_q_J >= q_target &&
          trial_metrics.min_q_edge >= q_target &&
          trial_metrics.min_eta_V >= eta_target) {
        break;
      }
    }

    if (!accepted_submove) {
      break;
    }
    current_r = std::move(best_r);
    current_z = std::move(best_z);
    metrics = best_metrics;
    accepted_max_swept = std::max(accepted_max_swept, best_swept);
    accepted_any = true;
    if (metrics.min_q_J >= q_target && metrics.min_q_edge >= q_target &&
        metrics.min_eta_V >= eta_target) {
      break;
    }
  }

  result.min_q_J = metrics.min_q_J;
  result.min_q_edge = metrics.min_q_edge;
  result.min_eta_V = metrics.min_eta_V;
  result.max_swept_fraction = accepted_max_swept;
  if (!accepted_any || metrics.min_eta_V < eta_target ||
      metrics.min_q_J < q_target || metrics.min_q_edge < q_target) {
    result.dt_reduction_required = true;
    result.suggested_dt =
        std::max(cfg.numerics.dt.min_s, dt_shrink * dt_nominal);
    const int moved_after_boundary_freeze =
        static_cast<int>(
            moved_nodes_from_target(x_r, x_z, current_r, current_z).size());
    core::log_info("[ring7_seam_packet] step=" + std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " requested=" + std::to_string(rezone_requested ? 1 : 0) +
                   " request_cell=" + std::to_string(request_cell) +
                   " fired=0 commit=0 state_mutation=0 coord_commit=0"
                   " status=no_admissible_target suggested_dt=" +
                   sci(result.suggested_dt) +
                   " moved_nodes_after_boundary_freeze=" +
                   std::to_string(moved_after_boundary_freeze) +
                   " min_q_J=" + sci(metrics.min_q_J) +
                   " min_q_edge=" + sci(metrics.min_q_edge) +
                   " min_eta_V=" + sci(metrics.min_eta_V));
    return result;
  }

  const std::vector<double> optimized_r = current_r;
  const std::vector<double> optimized_z = current_z;
  Ring7PacketRemapDiagnostics remap_diag;
  Ring7PacketGeometryDiagnostics packet_diag;
  double committed_lambda = 0.0;
  double last_lambda = 1.0;
  bool tried_lambda_min = false;
  for (int lambda_bt = 0; lambda_bt <= packet_lambda_backtracks;
       ++lambda_bt) {
    double packet_lambda = std::ldexp(1.0, -lambda_bt);
    if (packet_lambda < packet_lambda_floor) {
      if (tried_lambda_min) {
        break;
      }
      packet_lambda = packet_lambda_floor;
      tried_lambda_min = true;
    }
    last_lambda = packet_lambda;
    current_r = x_r;
    current_z = x_z;
    for (int node = 0; node < state.mesh.topo.n_nodes; ++node) {
      const std::size_t idx = static_cast<std::size_t>(node);
      current_r[idx] =
          x_r[idx] + packet_lambda * (optimized_r[idx] - x_r[idx]);
      current_z[idx] =
          x_z[idx] + packet_lambda * (optimized_z[idx] - x_z[idx]);
    }
    clone_duplicate_coordinate_targets(state, x_r, x_z, patch.node_ids,
                                       current_r, current_z);
    freeze_boundary_node_targets(boundary_node_mask, x_r, x_z, current_r,
                                 current_z);
    const Ring7SeamNodeSharing seam_node_sharing =
        clone_unique_face_duplicate_targets(state, x_r, x_z, current_r,
                                            current_z);
    freeze_boundary_node_targets(boundary_node_mask, x_r, x_z, current_r,
                                 current_z);
    metrics = evaluate_ring7_candidate_metrics(state,
                                               cfg,
                                               patch,
                                               current_r,
                                               current_z,
                                               v_r,
                                               v_z,
                                               d_v_r_nominal,
                                               d_v_z_nominal,
                                               dt_nominal,
                                               eta_target);
    if (metrics.min_eta_V <= 1.0) {
      remap_diag.positivity_failed = true;
      break;
    }
    std::vector<Ring7PacketRecord> packets;
    packet_diag = build_ring7_packet_geometry_diagnostics(state,
                                                          patch,
                                                          seam_node_sharing,
                                                          x_r,
                                                          x_z,
                                                          current_r,
                                                          current_z,
                                                          mass,
                                                          ee,
                                                          ei,
                                                          state_v_r,
                                                          state_v_z,
                                                          corner_mass,
                                                          corner_volume,
                                                          &packets);
    result.moved_node_count = packet_diag.moved_nodes;
    log_ring7_packet_diagnostics(state,
                                 metrics,
                                 packet_diag,
                                 request_cell,
                                 rezone_requested,
                                 stage);
    remap_diag = apply_ring7_packet_remap_transaction(state,
                                                      cfg,
                                                      eos_ctx,
                                                      patch,
                                                      seam_node_sharing,
                                                      packet_diag,
                                                      packets,
                                                      x_r,
                                                      x_z,
                                                      current_r,
                                                      current_z,
                                                      rho,
                                                      mass,
                                                      ee,
                                                      ei,
                                                      Te,
                                                      Ti,
                                                      Pe,
                                                      Pi,
                                                      state_v_r,
                                                      state_v_z,
                                                      corner_mass,
                                                      corner_volume,
                                                      gas_tracer_Y,
                                                      zbar,
                                                      cv_e,
                                                      cv_i,
                                                      cs,
                                                      hllc_mom_z_cell);
    if (remap_diag.ok) {
      committed_lambda = packet_lambda;
      break;
    }
    if (!remap_diag.positivity_failed) {
      break;
    }
  }
  result.min_q_J = metrics.min_q_J;
  result.min_q_edge = metrics.min_q_edge;
  result.min_eta_V = metrics.min_eta_V;
  result.max_swept_fraction =
      max_patch_swept_fraction(state, patch, x_r, x_z, current_r, current_z);
  if (!remap_diag.ok) {
    result.dt_reduction_required = true;
    result.suggested_dt =
        std::max(cfg.numerics.dt.min_s, dt_shrink * dt_nominal);
    core::log_info("[ring7_seam_remap] step=" + std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " requested=" + std::to_string(rezone_requested ? 1 : 0) +
                   " request_cell=" + std::to_string(request_cell) +
                   " fired=0 commit=0 status=rejected suggested_dt=" +
                   sci(result.suggested_dt) +
                   " positivity_failed=" +
                   std::to_string(remap_diag.positivity_failed ? 1 : 0) +
                   " conservation_failed=" +
                   std::to_string(remap_diag.conservation_failed ? 1 : 0) +
                   " central_failed=" +
                   std::to_string(remap_diag.central_failed ? 1 : 0) +
                   " subcycle_K=" + std::to_string(remap_diag.subcycles) +
                   " dM_rel=" + sci(remap_diag.dM_rel) +
                   " dE_rel=" + sci(remap_diag.dE_rel) +
                   " dPr_paired_rel=" + sci(remap_diag.dPr_paired_rel) +
                   " dPz_paired_rel=" + sci(remap_diag.dPz_paired_rel) +
                   " max_volume_error=" + sci(remap_diag.max_volume_error) +
                   " lambda_last=" + sci(last_lambda) +
                   " lambda_reduced=" +
                   std::to_string(last_lambda < 1.0 ? 1 : 0));
    return result;
  }
  result.fired = true;
  core::log_info("[ring7_seam_remap] step=" + std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " requested=" + std::to_string(rezone_requested ? 1 : 0) +
                 " request_cell=" + std::to_string(request_cell) +
                 " fired=1 commit=1 state_mutation=1 coord_commit=1" +
                 " dM_rel=" + sci(remap_diag.dM_rel) +
                 " dE_rel=" + sci(remap_diag.dE_rel) +
                 " dPr_paired_rel=" + sci(remap_diag.dPr_paired_rel) +
                 " dPz_paired_rel=" + sci(remap_diag.dPz_paired_rel) +
                 " Q_mix=" + sci(remap_diag.q_mix) +
                 " subcycle_K=" + std::to_string(remap_diag.subcycles) +
                 " lambda=" + sci(committed_lambda) +
                 " lambda_reduced=" +
                 std::to_string(committed_lambda < 1.0 ? 1 : 0) +
                 " moved_nodes=" + std::to_string(packet_diag.moved_nodes) +
                 " max_volume_error=" + sci(remap_diag.max_volume_error) +
                 " central_hash_before=" +
                 std::to_string(remap_diag.central_hash_before) +
                 " central_hash_after=" +
                 std::to_string(remap_diag.central_hash_after) +
                 " central_dM_rel=" +
                 sci(rel_delta(remap_diag.central_after.mass,
                               remap_diag.central_before.mass)) +
                 " central_dE_rel=" +
                 sci(rel_delta(remap_diag.central_after.total_energy,
                               remap_diag.central_before.total_energy)) +
                 " min_q_J=" + sci(metrics.min_q_J) +
                 " min_q_edge=" + sci(metrics.min_q_edge) +
                 " min_eta_V=" + sci(metrics.min_eta_V));
  return result;
}

void emit_ring7_seam_scaffold_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const double* d_x_r_start,
    const double* d_x_z_start,
    const double* d_v_r_half,
    const double* d_v_z_half,
    const double dt,
    const parallel::Reduction* reduction,
    const char* stage) {
  if (!ring7_feature_enabled(cfg) || !ring7_diag_due(state)) {
    return;
  }
  const Ring7SeamPatch patch = discover_ring7_seam_patch(state, cfg);
  if (!patch.valid || patch.ring7_cells.empty()) {
    core::log_info("[ring7_seam] step=" + std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " status=no_patch");
    return;
  }

  (void)reduction;
  const std::vector<MeshQualityRzVolumeCellMargin> margins =
      compute_multiblock_mesh_quality_rz_volume_cell_margins(state,
                                                             cfg,
                                                             d_x_r_start,
                                                             d_x_z_start,
                                                             d_v_r_half,
                                                             d_v_z_half,
                                                             dt,
                                                             patch.ring7_cells);

  const int n_nodes = state.mesh.topo.n_nodes;
  const std::vector<double> x_r_start =
      copy_device_doubles(d_x_r_start, n_nodes, "x_r_start");
  const std::vector<double> x_z_start =
      copy_device_doubles(d_x_z_start, n_nodes, "x_z_start");
  const std::vector<double> v_r =
      copy_device_doubles(d_v_r_half, n_nodes, "v_r_half");
  const std::vector<double> v_z =
      copy_device_doubles(d_v_z_half, n_nodes, "v_z_half");
  std::vector<double> x_r_trial(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> x_z_trial(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    x_r_trial[static_cast<std::size_t>(n)] =
        x_r_start[static_cast<std::size_t>(n)] + dt * v_r[static_cast<std::size_t>(n)];
    x_z_trial[static_cast<std::size_t>(n)] =
        x_z_start[static_cast<std::size_t>(n)] + dt * v_z[static_cast<std::size_t>(n)];
  }

  double min_qj = std::numeric_limits<double>::infinity();
  double min_qedge = std::numeric_limits<double>::infinity();
  double min_qv = std::numeric_limits<double>::infinity();
  double min_eta = std::numeric_limits<double>::infinity();
  int min_qj_cell = -1;
  int min_qedge_cell = -1;
  int min_qv_cell = -1;
  int min_eta_cell = -1;
  for (const int cell : patch.p_seam_cells) {
    const mesh::MeshForecastComponents q =
        cell_components(state, x_r_start, x_z_start, x_r_trial, x_z_trial,
                        cell, cfg);
    if (q.q_J < min_qj) {
      min_qj = q.q_J;
      min_qj_cell = cell;
    }
    if (q.q_edge < min_qedge) {
      min_qedge = q.q_edge;
      min_qedge_cell = cell;
    }
    if (q.q_V < min_qv) {
      min_qv = q.q_V;
      min_qv_cell = cell;
    }
  }
  for (const auto& margin : margins) {
    if (margin.eta_safe < min_eta) {
      min_eta = margin.eta_safe;
      min_eta_cell = margin.cell;
    }
  }

  core::log_info("[ring7_seam] step=" + std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " dt=" + sci(dt) +
                 " shell_ring_i=" + std::to_string(patch.shell_ring_i) +
                 " ring7_cells=" + std::to_string(patch.ring7_cells.size()) +
                 " inner_guard_cells=" +
                 std::to_string(patch.inner_guard_cells.size()) +
                 " butterfly_adjacent_cells=" +
                 std::to_string(patch.butterfly_adjacent_cells.size()) +
                 " trifan_end_cells=" +
                 std::to_string(patch.trifan_end_cells.size()) +
                 " p_seam_cells=" + std::to_string(patch.p_seam_cells.size()) +
                 " nodes=" + std::to_string(patch.node_ids.size()));
  core::log_info("[ring7_seam] scan step=" + std::to_string(state.step) +
                 " stage=" + std::string(stage != nullptr ? stage : "?") +
                 " min_q_J=" + sci(min_qj) +
                 " min_q_J_cell=" + std::to_string(min_qj_cell) +
                 " min_q_edge=" + sci(min_qedge) +
                 " min_q_edge_cell=" + std::to_string(min_qedge_cell) +
                 " min_q_V=" + sci(min_qv) +
                 " min_q_V_cell=" + std::to_string(min_qv_cell) +
                 " min_eta_V_safe=" + sci(min_eta) +
                 " min_eta_V_cell=" + std::to_string(min_eta_cell));

  for (const int cell : patch.ring7_cells) {
    const int local = cell - patch.shell_cell_begin;
    const int local_i = patch.shell_nj > 0 ? local / patch.shell_nj : -1;
    const int local_j = patch.shell_nj > 0 ? local - local_i * patch.shell_nj : -1;
    const mesh::MeshForecastComponents q =
        cell_components(state, x_r_start, x_z_start, x_r_trial, x_z_trial,
                        cell, cfg);
    const MeshQualityRzVolumeCellMargin* margin = find_margin(margins, cell);
    const double eta = margin != nullptr ? margin->eta_safe : 1.0;
    const double sigma = margin != nullptr ? margin->sigma_raw : 1.0;
    core::log_info("[ring7_seam] ring7_cell step=" +
                   std::to_string(state.step) +
                   " stage=" + std::string(stage != nullptr ? stage : "?") +
                   " cell=" + std::to_string(cell) +
                   " local_i=" + std::to_string(local_i) +
                   " local_j=" + std::to_string(local_j) +
                   " q_J=" + sci(q.q_J) +
                   " q_edge=" + sci(q.q_edge) +
                   " q_V=" + sci(q.q_V) +
                   " eta_V_safe=" + sci(eta) +
                   " sigma_V_raw=" + sci(sigma));
  }
}

}  // namespace tenryu::hydro
