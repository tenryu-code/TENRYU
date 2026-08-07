#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"
#include "hydro/corner_jacobian_quality.cuh"

namespace tenryu::hydro::mesh_trace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool enabled(const core::State& state, const core::Config& cfg) {
  return cfg.numerics.debug.trace_mesh_motion &&
         state.step < cfg.numerics.debug.trace_max_steps;
}

template <typename Tag>
std::vector<double> copy_field(const core::Field1D<Tag>& field) {
  std::vector<double> out;
  field.copy_to_host(out);
  return out;
}

template <typename T>
std::vector<T> copy_device_vector(const T* ptr, const std::size_t n) {
  std::vector<T> out(n);
  if (ptr != nullptr && n > 0U) {
    cuda_check(cudaMemcpy(out.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost),
               "mesh motion trace device vector copy failed");
  }
  return out;
}

template <typename T>
T copy_device_scalar(const T* ptr, const int idx, const T fallback = T{}) {
  if (ptr == nullptr || idx < 0) {
    return fallback;
  }
  T out = fallback;
  cuda_check(cudaMemcpy(&out,
                        ptr + idx,
                        sizeof(T),
                        cudaMemcpyDeviceToHost),
             "mesh motion trace device scalar copy failed");
  return out;
}

inline double node_value(const core::NodeField1D& field, const int node) {
  return copy_device_scalar(field.data(), node, 0.0);
}

inline double max_abs_pair(const core::NodeField1D& a,
                           const core::NodeField1D& b) {
  const std::vector<double> ha = copy_field(a);
  const std::vector<double> hb = copy_field(b);
  const std::size_t n = std::min(ha.size(), hb.size());
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    out = std::max(out, std::abs(ha[i]));
    out = std::max(out, std::abs(hb[i]));
  }
  return out;
}

inline double max_abs_diff_pair(const core::NodeField1D& ar,
                                const core::NodeField1D& az,
                                const core::NodeField1D& br,
                                const core::NodeField1D& bz) {
  const std::vector<double> har = copy_field(ar);
  const std::vector<double> haz = copy_field(az);
  const std::vector<double> hbr = copy_field(br);
  const std::vector<double> hbz = copy_field(bz);
  const std::size_t n =
      std::min(std::min(har.size(), haz.size()), std::min(hbr.size(), hbz.size()));
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    out = std::max(out, std::abs(har[i] - hbr[i]));
    out = std::max(out, std::abs(haz[i] - hbz[i]));
  }
  return out;
}

inline double max_hypot(const core::NodeField1D& r,
                        const core::NodeField1D& z) {
  const std::vector<double> hr = copy_field(r);
  const std::vector<double> hz = copy_field(z);
  const std::size_t n = std::min(hr.size(), hz.size());
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    out = std::max(out, std::hypot(hr[i], hz[i]));
  }
  return out;
}

inline double max_abs_state_minus_initial(const core::State& state) {
  return max_abs_diff_pair(state.x_r,
                           state.x_z,
                           state.x_r_initial,
                           state.x_z_initial);
}

inline double sum_abs_pair(const core::NodeField1D& r,
                           const core::NodeField1D& z) {
  const std::vector<double> hr = copy_field(r);
  const std::vector<double> hz = copy_field(z);
  const std::size_t n = std::min(hr.size(), hz.size());
  double out = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    out += std::abs(hr[i]) + std::abs(hz[i]);
  }
  return out;
}

inline std::uint64_t field_hash(const core::NodeField1D& field) {
  const std::vector<double> host = copy_field(field);
  std::uint64_t h = 1469598103934665603ULL;
  for (const double value : host) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "double hash size mismatch");
    std::memcpy(&bits, &value, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ULL;
  }
  return h;
}

inline std::uint64_t coordinate_hash(const core::State& state) {
  return field_hash(state.x_r) ^ field_hash(state.x_z);
}

inline int find_outer_equator_node(const core::State& state) {
  std::vector<double> xr;
  std::vector<double> xz;
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  int best = 0;
  double best_s = -1.0;
  for (int n = 0; n < static_cast<int>(xr.size()); ++n) {
    const double s = std::hypot(xr[static_cast<std::size_t>(n)],
                                xz[static_cast<std::size_t>(n)]);
    const double weighted = s - std::abs(xz[static_cast<std::size_t>(n)]) * 0.01;
    if (weighted > best_s) {
      best_s = weighted;
      best = n;
    }
  }
  return best;
}

inline int find_outer_equator_cell(const core::State& state) {
  std::vector<double> xr;
  std::vector<double> xz;
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  int best = -1;
  double best_s = -1.0;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    for (int c = 0; c < state.mesh.topo.n_cells; ++c) {
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      double rc = 0.0;
      double zc = 0.0;
      for (int k = 0; k < 4; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        rc += xr[static_cast<std::size_t>(n)];
        zc += xz[static_cast<std::size_t>(n)];
      }
      rc *= 0.25;
      zc *= 0.25;
      const double weighted = std::hypot(rc, zc) - std::abs(zc) * 0.01;
      if (weighted > best_s) {
        best_s = weighted;
        best = c;
      }
    }
    return best;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int nodes[4] = {
          i * (nz + 1) + j,
          (i + 1) * (nz + 1) + j,
          (i + 1) * (nz + 1) + (j + 1),
          i * (nz + 1) + (j + 1),
      };
      double rc = 0.0;
      double zc = 0.0;
      for (const int n : nodes) {
        rc += xr[static_cast<std::size_t>(n)];
        zc += xz[static_cast<std::size_t>(n)];
      }
      rc *= 0.25;
      zc *= 0.25;
      const double weighted = std::hypot(rc, zc) - std::abs(zc) * 0.01;
      if (weighted > best_s) {
        best_s = weighted;
        best = i * nz + j;
      }
    }
  }
  return best;
}

inline void ensure_trace_indices(core::State& state, const core::Config& cfg) {
  if (!enabled(state, cfg)) {
    return;
  }
  if (state.trace_mesh_outer_node < 0 ||
      state.trace_mesh_outer_node >= static_cast<int>(state.x_r.size())) {
    state.trace_mesh_outer_node = find_outer_equator_node(state);
  }
  if (state.trace_mesh_outer_cell < 0 ||
      state.trace_mesh_outer_cell >= static_cast<int>(state.rho.size())) {
    state.trace_mesh_outer_cell = find_outer_equator_cell(state);
  }
}

inline int outer_node(const core::State& state) {
  if (state.trace_mesh_outer_node >= 0 &&
      state.trace_mesh_outer_node < static_cast<int>(state.x_r.size())) {
    return state.trace_mesh_outer_node;
  }
  return find_outer_equator_node(state);
}

inline int outer_cell(const core::State& state) {
  if (state.trace_mesh_outer_cell >= 0 &&
      state.trace_mesh_outer_cell < static_cast<int>(state.rho.size())) {
    return state.trace_mesh_outer_cell;
  }
  return find_outer_equator_cell(state);
}

inline std::int8_t hydro_active_at(const core::State& state,
                                   const std::int8_t* d_hydro_active,
                                   const int c) {
  if (c < 0 || c >= static_cast<int>(state.rho.size())) {
    return 0;
  }
  if (d_hydro_active != nullptr) {
    return copy_device_scalar(d_hydro_active, c, static_cast<std::int8_t>(0));
  }
  if (static_cast<std::size_t>(c) < state.hydro_active.size()) {
    return state.hydro_active[static_cast<std::size_t>(c)];
  }
  return 0;
}

inline std::uint8_t node_flag_at(const core::State& state, const int n) {
  if (n < 0 || static_cast<std::size_t>(n) >= state.mesh.topo.node_flags.size()) {
    return 0U;
  }
  return state.mesh.topo.node_flags[static_cast<std::size_t>(n)];
}

inline double min_field(const core::NodeField1D& field) {
  const std::vector<double> h = copy_field(field);
  if (h.empty()) {
    return 0.0;
  }
  return *std::min_element(h.begin(), h.end());
}

inline double max_field(const core::NodeField1D& field) {
  const std::vector<double> h = copy_field(field);
  if (h.empty()) {
    return 0.0;
  }
  return *std::max_element(h.begin(), h.end());
}

inline double planar_area_host(const double* r, const double* z) {
  double cross_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    cross_sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return 0.5 * std::abs(cross_sum);
}

inline double signed_planar_area_host(const double* r, const double* z) {
  double cross_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    cross_sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return 0.5 * cross_sum;
}

inline double signed_point_line_distance(const double ar,
                                         const double az,
                                         const double br,
                                         const double bz,
                                         const double pr,
                                         const double pz) {
  const double dr = br - ar;
  const double dz = bz - az;
  const double len = std::hypot(dr, dz);
  if (!(len > std::numeric_limits<double>::min())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return (dr * (pz - az) - dz * (pr - ar)) / len;
}

inline int origin_adjacent_corner(const double* r, const double* z) {
  int best = 0;
  double best_s = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const double s = std::hypot(r[k], z[k]);
    if (s < best_s) {
      best_s = s;
      best = k;
    }
  }
  return best;
}

inline double min_diagonal_altitude_host(const double* r, const double* z) {
  double out = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const int km = (k + 3) & 3;
    const int kp = (k + 1) & 3;
    const double altitude =
        std::abs(signed_point_line_distance(r[km], z[km], r[kp], z[kp],
                                            r[k], z[k]));
    if (std::isfinite(altitude)) {
      out = std::min(out, altitude);
    }
  }
  return std::isfinite(out) ? out : std::numeric_limits<double>::quiet_NaN();
}

inline int count_anti_hourglass_active(const core::State& state,
                                       const core::Config& cfg,
                                       const core::CellField1D& cell_pressure,
                                       const std::int8_t* d_hydro_active,
                                       const double dt) {
  std::vector<double> xr;
  std::vector<double> xz;
  std::vector<double> rho;
  std::vector<double> pressure;
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  state.rho.copy_to_host(rho);
  cell_pressure.copy_to_host(pressure);
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr && n_cells > 0) {
    hydro_active = copy_device_vector(d_hydro_active,
                                      static_cast<std::size_t>(n_cells));
  } else {
    hydro_active = state.hydro_active;
  }

  const auto& hg = cfg.numerics.hydro.hourglass;
  const bool stiffness_mode = cfg.numerics.hydro.subzonal_mass_enabled;
  if (stiffness_mode && !(dt > 0.0)) {
    return 0;
  }

  int count = 0;
  for (int c = 0; c < n_cells; ++c) {
    if (!hydro_active.empty() &&
        hydro_active[static_cast<std::size_t>(c)] == 0) {
      continue;
    }
    int nodes[4] = {};
    if (state.mesh.topo.multiblock.has_value()) {
      const auto& mb = *state.mesh.topo.multiblock;
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      for (int k = 0; k < 4; ++k) {
        nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      }
    } else {
      const int nz = state.mesh.topo.nz;
      const int i = c / nz;
      const int j = c - i * nz;
      nodes[0] = i * (nz + 1) + j;
      nodes[1] = (i + 1) * (nz + 1) + j;
      nodes[2] = (i + 1) * (nz + 1) + (j + 1);
      nodes[3] = i * (nz + 1) + (j + 1);
    }
    double r[4] = {};
    double z[4] = {};
    for (int k = 0; k < 4; ++k) {
      r[k] = xr[static_cast<std::size_t>(nodes[k])];
      z[k] = xz[static_cast<std::size_t>(nodes[k])];
    }
    double j_min = state.mesh.topo.multiblock.has_value()
                       ? std::abs(corner_jacobian_from_quad(r, z, 0))
                       : corner_jacobian_from_quad(r, z, 0);
    double j_max = j_min;
    for (int k = 1; k < 4; ++k) {
      const double jk = state.mesh.topo.multiblock.has_value()
                            ? std::abs(corner_jacobian_from_quad(r, z, k))
                            : corner_jacobian_from_quad(r, z, k);
      j_min = std::min(j_min, jk);
      j_max = std::max(j_max, jk);
    }
    if (!(std::isfinite(j_min) && std::isfinite(j_max) &&
          j_max > std::numeric_limits<double>::min())) {
      continue;
    }
    if (j_min / j_max >= hg.activation_corner_j_ratio_threshold) {
      continue;
    }
    const double a_r = 0.25 * (r[0] - r[1] + r[2] - r[3]);
    const double a_z = 0.25 * (z[0] - z[1] + z[2] - z[3]);
    const double cell_size =
        std::sqrt(std::max(planar_area_host(r, z), 0.0));
    if (!(cell_size > std::numeric_limits<double>::min())) {
      continue;
    }
    if (std::hypot(a_r, a_z) / cell_size <
        hg.activation_hourglass_amplitude_threshold) {
      continue;
    }
    const double rho_c =
        std::max(rho[static_cast<std::size_t>(c)],
                 std::numeric_limits<double>::min());
    const double p = std::max(pressure[static_cast<std::size_t>(c)], 0.0);
    if (!stiffness_mode && !(p / rho_c > 0.0)) {
      continue;
    }
    ++count;
  }
  return count;
}

inline void trace_start(core::State& state,
                        const core::Config& cfg,
                        const double dt) {
  ensure_trace_indices(state, cfg);
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=start] n_outer=%d c_outer=%d "
               "x_r=%.17e x_z=%.17e v_r=%.17e v_z=%.17e dt=%.17e\n",
               state.step,
               n,
               outer_cell(state),
               node_value(state.x_r, n),
               node_value(state.x_z, n),
               node_value(state.v_r, n),
               node_value(state.v_z, n),
               dt);
  std::fflush(stderr);
}

inline void trace_post_pressure_accel(const core::State& state,
                                      const core::Config& cfg,
                                      const core::NodeField1D& accel_r,
                                      const core::NodeField1D& accel_z,
                                      const core::NodeField1D& node_mass,
                                      const std::int8_t* d_hydro_active) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  const int c = outer_cell(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_pressure_accel] n_outer=%d "
               "c_outer=%d a_n_r=%.17e a_n_z=%.17e max_abs_a_n=%.17e "
               "node_mass=%.17e hydro_active_c_outer=%d node_flags=%u\n",
               state.step,
               n,
               c,
               node_value(accel_r, n),
               node_value(accel_z, n),
               max_abs_pair(accel_r, accel_z),
               node_value(node_mass, n),
               static_cast<int>(hydro_active_at(state, d_hydro_active, c)),
               static_cast<unsigned>(node_flag_at(state, n)));
  std::fflush(stderr);
}

inline void trace_post_hg_accel(const core::State& state,
                                const core::Config& cfg,
                                const core::NodeField1D& accel_r,
                                const core::NodeField1D& accel_z,
                                const core::CellField1D& cell_pressure,
                                const std::int8_t* d_hydro_active,
                                const double dt) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  const int active =
      count_anti_hourglass_active(state, cfg, cell_pressure, d_hydro_active, dt);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_hg_accel] n_outer=%d "
               "a_hg_r=%.17e a_hg_z=%.17e max_abs_a_hg=%.17e "
               "count_anti_hg_active=%d\n",
               state.step,
               n,
               node_value(accel_r, n),
               node_value(accel_z, n),
               max_abs_pair(accel_r, accel_z),
               active);
  std::fflush(stderr);
}

inline void trace_post_total_accel(const core::State& state,
                                   const core::Config& cfg,
                                   const core::NodeField1D& accel_r,
                                   const core::NodeField1D& accel_z) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_total_accel] n_outer=%d "
               "a_total_r=%.17e a_total_z=%.17e max_abs_a_total=%.17e\n",
               state.step,
               n,
               node_value(accel_r, n),
               node_value(accel_z, n),
               max_abs_pair(accel_r, accel_z));
  std::fflush(stderr);
}

inline void trace_post_velocity(const core::State& state,
                                const core::Config& cfg) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_velocity] n_outer=%d "
               "v_r=%.17e v_z=%.17e max_abs_v=%.17e\n",
               state.step,
               n,
               node_value(state.v_r, n),
               node_value(state.v_z, n),
               max_abs_pair(state.v_r, state.v_z));
  std::fflush(stderr);
}

inline void trace_post_lagrange_position(const core::State& state,
                                         const core::Config& cfg,
                                         const core::NodeField1D& r_old,
                                         const core::NodeField1D& z_old) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  const double xr = node_value(state.x_r, n);
  const double xz = node_value(state.x_z, n);
  const double xr_old = node_value(r_old, n);
  const double xz_old = node_value(z_old, n);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_lagrange_position] n_outer=%d "
               "x_r=%.17e x_z=%.17e dr=%.17e dz=%.17e max_abs_disp=%.17e\n",
               state.step,
               n,
               xr,
               xz,
               xr - xr_old,
               xz - xz_old,
               max_abs_diff_pair(state.x_r, state.x_z, r_old, z_old));
  std::fflush(stderr);
}

inline void trace_alpha_source(const core::State& state,
                               const core::Config& cfg,
                               const double alpha) {
  if (!enabled(state, cfg)) {
    return;
  }
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=alpha_source] alpha=%.17e "
               "s_current=%.17e s_initial=%.17e "
               "max_abs_state_x_minus_initial=%.17e\n",
               state.step,
               alpha,
               max_hypot(state.x_r, state.x_z),
               max_hypot(state.x_r_initial, state.x_z_initial),
               max_abs_state_minus_initial(state));
  std::fflush(stderr);
}

inline void trace_post_target(const core::State& state,
                              const core::Config& cfg,
                              const double* d_target_r,
                              const double* d_target_z) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  const std::size_t n_nodes = state.x_r.size();
  const std::vector<double> target_r = copy_device_vector(d_target_r, n_nodes);
  const std::vector<double> target_z = copy_device_vector(d_target_z, n_nodes);
  const std::vector<double> initial_r = copy_field(state.x_r_initial);
  const std::vector<double> initial_z = copy_field(state.x_z_initial);
  double max_target_minus_initial = 0.0;
  for (std::size_t i = 0; i < n_nodes; ++i) {
    max_target_minus_initial =
        std::max(max_target_minus_initial, std::abs(target_r[i] - initial_r[i]));
    max_target_minus_initial =
        std::max(max_target_minus_initial, std::abs(target_z[i] - initial_z[i]));
  }
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_target] n_outer=%d "
               "target_r=%.17e target_z=%.17e "
               "target_r_minus_x_initial=%.17e "
               "target_z_minus_x_initial=%.17e "
               "max_abs_target_minus_initial=%.17e\n",
               state.step,
               n,
               target_r[static_cast<std::size_t>(n)],
               target_z[static_cast<std::size_t>(n)],
               target_r[static_cast<std::size_t>(n)] -
                   initial_r[static_cast<std::size_t>(n)],
               target_z[static_cast<std::size_t>(n)] -
                   initial_z[static_cast<std::size_t>(n)],
               max_target_minus_initial);
  std::fflush(stderr);
}

inline bool cell0_nodes(const core::State& state, int* nodes) {
  if (nodes == nullptr || state.rho.empty()) {
    return false;
  }
  const int c = 0;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    if (mb.cell_node_csr_offsets.size() <= 1U ||
        mb.cell_node_csr_indices.size() < 4U) {
      return false;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    for (int k = 0; k < 4; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    }
    return true;
  }
  const int nz = state.mesh.topo.nz;
  if (state.mesh.topo.nr <= 0 || nz <= 0) {
    return false;
  }
  nodes[0] = 0;
  nodes[1] = nz + 1;
  nodes[2] = nz + 2;
  nodes[3] = 1;
  return true;
}

inline void trace_cell0_geometry(const core::State& state,
                                 const core::Config& cfg,
                                 const char* stage) {
  if (!enabled(state, cfg)) {
    return;
  }
  int nodes[4] = {-1, -1, -1, -1};
  if (!cell0_nodes(state, nodes)) {
    return;
  }
  double r[4] = {};
  double z[4] = {};
  double vr[4] = {};
  double vz[4] = {};
  for (int k = 0; k < 4; ++k) {
    r[k] = node_value(state.x_r, nodes[k]);
    z[k] = node_value(state.x_z, nodes[k]);
    vr[k] = node_value(state.v_r, nodes[k]);
    vz[k] = node_value(state.v_z, nodes[k]);
  }
  double corner_j[4] = {};
  double min_corner_j = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    corner_j[k] = corner_jacobian_from_quad(r, z, k);
    min_corner_j = std::min(min_corner_j, corner_j[k]);
  }
  const int origin_corner = origin_adjacent_corner(r, z);
  const int side_a = (origin_corner + 1) & 3;
  const int outboard_corner = (origin_corner + 2) & 3;
  const int side_b = (origin_corner + 3) & 3;
  const double baseline_length =
      std::hypot(r[side_b] - r[side_a], z[side_b] - z[side_a]);
  const double origin_side =
      signed_point_line_distance(r[side_a], z[side_a], r[side_b], z[side_b],
                                 r[origin_corner], z[origin_corner]);
  const double outboard_side =
      signed_point_line_distance(r[side_a], z[side_a], r[side_b], z[side_b],
                                 r[outboard_corner], z[outboard_corner]);
  const double keystone_altitude =
      (std::isfinite(origin_side) && std::isfinite(outboard_side))
          ? -std::copysign(1.0, origin_side) * outboard_side
          : std::numeric_limits<double>::quiet_NaN();
  const double keystone_psi_over_b =
      (baseline_length > std::numeric_limits<double>::min())
          ? keystone_altitude / (0.5 * baseline_length)
          : std::numeric_limits<double>::quiet_NaN();
  const double signed_area = signed_planar_area_host(r, z);
  const double sqrt_abs_area = std::sqrt(std::max(std::abs(signed_area), 0.0));
  const double keystone_altitude_over_sqrt_area =
      (sqrt_abs_area > std::numeric_limits<double>::min())
          ? keystone_altitude / sqrt_abs_area
          : std::numeric_limits<double>::quiet_NaN();
  int block_id = -1;
  int stable_cell_id = 0;
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    if (!mb.cell_block_id.empty()) {
      block_id = mb.cell_block_id[0];
    }
    if (!mb.cell_id_stable.empty()) {
      stable_cell_id = mb.cell_id_stable[0];
    }
  }
  const double state_vol =
      state.vol.empty() ? 0.0 : copy_device_scalar(state.vol.data(), 0, 0.0);
  const double mesh_vol =
      state.mesh.cell_vol.empty() ? 0.0 : state.mesh.cell_vol[0];
  const double mass =
      state.mass.empty() ? 0.0 : copy_device_scalar(state.mass.data(), 0, 0.0);
  double corner_mass_sum = 0.0;
  if (state.corner_mass.size() >= 4U) {
    for (int k = 0; k < 4; ++k) {
      corner_mass_sum += copy_device_scalar(state.corner_mass.data(), k, 0.0);
    }
  }
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=%s_cell0] cell=0 "
               "block_id=%d stable_cell_id=%d "
               "state_vol=%.17e mesh_vol=%.17e mass=%.17e "
               "signed_rz_area=%.17e min_altitude=%.17e "
               "corner_J0=%.17e corner_J1=%.17e corner_J2=%.17e corner_J3=%.17e "
               "min_corner_J=%.17e "
               "keystone_origin_corner=%d keystone_side_corner_a=%d "
               "keystone_side_corner_b=%d keystone_outboard_corner=%d "
               "keystone_baseline_length=%.17e keystone_signed_altitude=%.17e "
               "keystone_psi_over_b=%.17e "
               "keystone_altitude_over_sqrt_area=%.17e "
               "corner_mass_sum=%.17e "
               "n0=%d r0=%.17e z0=%.17e vr0=%.17e vz0=%.17e "
               "n1=%d r1=%.17e z1=%.17e vr1=%.17e vz1=%.17e "
               "n2=%d r2=%.17e z2=%.17e vr2=%.17e vz2=%.17e "
               "n3=%d r3=%.17e z3=%.17e vr3=%.17e vz3=%.17e\n",
               state.step,
               stage,
               block_id,
               stable_cell_id,
               state_vol,
               mesh_vol,
               mass,
               signed_area,
               min_diagonal_altitude_host(r, z),
               corner_j[0],
               corner_j[1],
               corner_j[2],
               corner_j[3],
               min_corner_j,
               origin_corner,
               side_a,
               side_b,
               outboard_corner,
               baseline_length,
               keystone_altitude,
               keystone_psi_over_b,
               keystone_altitude_over_sqrt_area,
               corner_mass_sum,
               nodes[0],
               r[0],
               z[0],
               vr[0],
               vz[0],
               nodes[1],
               r[1],
               z[1],
               vr[1],
               vz[1],
               nodes[2],
               r[2],
               z[2],
               vr[2],
               vz[2],
               nodes[3],
               r[3],
               z[3],
               vr[3],
               vz[3]);
  std::fflush(stderr);
}

inline void trace_cell0_target(const core::State& state,
                               const core::Config& cfg,
                               const char* stage,
                               const double* d_target_r,
                               const double* d_target_z) {
  if (!enabled(state, cfg)) {
    return;
  }
  int nodes[4] = {-1, -1, -1, -1};
  if (!cell0_nodes(state, nodes)) {
    return;
  }
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=%s_cell0_target] cell=0 "
               "n0=%d r0=%.17e z0=%.17e "
               "n1=%d r1=%.17e z1=%.17e "
               "n2=%d r2=%.17e z2=%.17e "
               "n3=%d r3=%.17e z3=%.17e\n",
               state.step,
               stage,
               nodes[0],
               copy_device_scalar(d_target_r, nodes[0], 0.0),
               copy_device_scalar(d_target_z, nodes[0], 0.0),
               nodes[1],
               copy_device_scalar(d_target_r, nodes[1], 0.0),
               copy_device_scalar(d_target_z, nodes[1], 0.0),
               nodes[2],
               copy_device_scalar(d_target_r, nodes[2], 0.0),
               copy_device_scalar(d_target_z, nodes[2], 0.0),
               nodes[3],
               copy_device_scalar(d_target_r, nodes[3], 0.0),
               copy_device_scalar(d_target_z, nodes[3], 0.0));
  std::fflush(stderr);
}

inline void trace_post_remap(const core::State& state,
                             const core::Config& cfg) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=post_remap] n_outer=%d "
               "x_r=%.17e x_z=%.17e v_r=%.17e v_z=%.17e "
               "max_abs_x_minus_initial=%.17e max_abs_v=%.17e\n",
               state.step,
               n,
               node_value(state.x_r, n),
               node_value(state.x_z, n),
               node_value(state.v_r, n),
               node_value(state.v_z, n),
               max_abs_state_minus_initial(state),
               max_abs_pair(state.v_r, state.v_z));
  std::fflush(stderr);
}

inline void trace_pressure_bc(const core::State& state,
                              const core::Config& cfg,
                              const double p_ext,
                              const int count_pressure_faces,
                              const core::NodeField1D& force_r,
                              const core::NodeField1D& force_z) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=pressure_bc] n_outer=%d "
               "P_ext=%.17e count_pressure_faces=%d sum_abs_F_bc=%.17e "
               "F_bc_r_at_n_outer=%.17e F_bc_z_at_n_outer=%.17e\n",
               state.step,
               n,
               p_ext,
               count_pressure_faces,
               sum_abs_pair(force_r, force_z),
               node_value(force_r, n),
               node_value(force_z, n));
  std::fflush(stderr);
}

inline void trace_activity(const core::State& state,
                           const core::Config& cfg,
                           const std::uint8_t* d_node_active,
                           const core::NodeField1D& node_mass,
                           const std::int8_t* d_hydro_active) {
  if (!enabled(state, cfg)) {
    return;
  }
  const int n = outer_node(state);
  const int c_outer = outer_cell(state);
  const int c_bridge = cfg.numerics.debug.trace_mesh_cell;
  const std::uint8_t node_active =
      copy_device_scalar(d_node_active, n, static_cast<std::uint8_t>(0));
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=activity] n_outer=%d c_outer=%d "
               "c_bridge=%d hydro_active_c_outer=%d hydro_active_c_bridge=%d "
               "node_active_n_outer=%u node_flags_n_outer=%u "
               "node_mass_n_outer=%.17e node_mass_min=%.17e "
               "node_mass_max=%.17e\n",
               state.step,
               n,
               c_outer,
               c_bridge,
               static_cast<int>(hydro_active_at(state, d_hydro_active, c_outer)),
               static_cast<int>(hydro_active_at(state, d_hydro_active, c_bridge)),
               static_cast<unsigned>(node_active),
               static_cast<unsigned>(node_flag_at(state, n)),
               node_value(node_mass, n),
               min_field(node_mass),
               max_field(node_mass));
  std::fflush(stderr);
}

inline void trace_corner_mass_canary(const core::State& state,
                                     const core::Config& cfg,
                                     const std::uint64_t hash_before,
                                     const std::uint64_t hash_after) {
  if (!enabled(state, cfg)) {
    return;
  }
  std::fprintf(stderr,
               "[mesh-trace step=%d stage=corner_mass_canary] "
               "hash_before=%llu hash_after=%llu canary_pass=%d\n",
               state.step,
               static_cast<unsigned long long>(hash_before),
               static_cast<unsigned long long>(hash_after),
               hash_before == hash_after ? 1 : 0);
  std::fflush(stderr);
}

inline void trace_gate_metrics(const core::State& state,
                               const core::Config& cfg) {
  if (!enabled(state, cfg)) {
    return;
  }
  std::vector<double> xr;
  std::vector<double> xz;
  std::vector<double> xr0;
  std::vector<double> xz0;
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  state.x_r_initial.copy_to_host(xr0);
  state.x_z_initial.copy_to_host(xz0);
  const std::size_t n_nodes =
      std::min(std::min(xr.size(), xz.size()), std::min(xr0.size(), xz0.size()));
  double s_initial = 0.0;
  double s_current = 0.0;
  for (std::size_t n = 0; n < n_nodes; ++n) {
    s_initial = std::max(s_initial, std::hypot(xr0[n], xz0[n]));
    s_current = std::max(s_current, std::hypot(xr[n], xz[n]));
  }
  const double c_meas = (s_current > 0.0) ? (s_initial / s_current) : 0.0;

  double corner_mass_closure_residual = 0.0;
  if (!state.corner_mass.empty() && !state.mass.empty() &&
      state.corner_mass.size() >= 4U * state.mass.size()) {
    const auto corner_mass = copy_field(state.corner_mass);
    const auto mass = copy_field(state.mass);
    for (std::size_t c = 0; c < mass.size(); ++c) {
      const double m = mass[c];
      if (!(m > 0.0) || !std::isfinite(m)) {
        continue;
      }
      const std::size_t base = 4U * c;
      const double sum = corner_mass[base + 0U] + corner_mass[base + 1U] +
                         corner_mass[base + 2U] + corner_mass[base + 3U];
      corner_mass_closure_residual =
          std::max(corner_mass_closure_residual, std::abs(sum - m) / m);
    }
  }

  std::fprintf(stderr,
               "[mesh-trace step=%d stage=gate_metrics] "
               "s_initial_cm=%.17e s_current_cm=%.17e C_meas=%.17e "
               "corner_mass_closure_residual=%.17e\n",
               state.step,
               s_initial,
               s_current,
               c_meas,
               corner_mass_closure_residual);
  std::fflush(stderr);
}

}  // namespace tenryu::hydro::mesh_trace
