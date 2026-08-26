#include "hydro/compatible_subzonal_pressure.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "core/error.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.cuh"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kTiny = 1.0e-300;

struct SubzonalEOSViews {
  tenryu::materials::DeviceEOSTableView ion{};
  tenryu::materials::DeviceEOSTableView electron{};
  tenryu::materials::DeviceEOSTableView total{};
};

struct SubzonalStepDiagnosticsDevice {
  double max_merit;
  double max_abs_force;
  double max_abs_work;
  int nonzero_force_cells;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void sync_kernel(const char* message) {
  cuda_check(cudaGetLastError(), message);
  cuda_check(cudaDeviceSynchronize(), message);
}

void zero_array(double* ptr, const std::size_t n, const char* message) {
  if (ptr == nullptr || n == 0U) {
    return;
  }
  cuda_check(cudaMemset(ptr, 0, n * sizeof(double)), message);
}

__device__ inline double atomic_max_double(double* address, const double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (__longlong_as_double(static_cast<long long>(old)) < value) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ inline double atomic_min_double(double* address, const double value) {
  auto* bits = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *bits;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(bits, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (old == assumed) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

bool has_tri_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
                         const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts == 3U;
                     });
}

const std::uint8_t* upload_cell_nverts_if_tri(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state,
    const int n_cells) {
  if (!has_tri_cell_nverts(state.mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
  return d_cell_nverts.data();
}

__device__ inline int structured_node_index(const int i,
                                            const int j,
                                            const int nz) {
  return i * (nz + 1) + j;
}

__device__ inline bool table_valid_at_rho(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho) {
  return tab.n_rho > 0 && rho >= exp(tab.log_rho_min);
}

__device__ inline double table_pressure_from_rho_e(
    const tenryu::materials::DeviceEOSTableView& tab,
    const double rho,
    const double e,
    const double t_floor) {
  const auto rb = tenryu::materials::find_rho_bracket(tab, rho);
  const double t =
      fmax(tenryu::materials::device_eos_T_from_e_monotone(tab, rb, e), t_floor);
  return tenryu::materials::device_eos_pressure(tab, rb, log(fmax(t, 1.0e-30)));
}

__device__ inline double subzonal_eos_pressure(
    const SubzonalEOSViews views,
    const bool use_two_temp,
    const double rho,
    const double ee,
    const double ei,
    const double gamma,
    const double te_floor,
    const double ti_floor) {
  const double rho_safe = fmax(rho, 1.0e-30);
  if (use_two_temp) {
    if (table_valid_at_rho(views.ion, rho_safe) &&
        table_valid_at_rho(views.electron, rho_safe)) {
      return table_pressure_from_rho_e(views.ion, rho_safe, fmax(ei, 0.0),
                                       ti_floor) +
             table_pressure_from_rho_e(views.electron, rho_safe, fmax(ee, 0.0),
                                       te_floor);
    }
    return tenryu::materials::eos_pressure_ion(rho_safe, fmax(ei, 0.0), gamma) +
           tenryu::materials::eos_pressure_electron(rho_safe, fmax(ee, 0.0),
                                                    gamma);
  }

  if (table_valid_at_rho(views.total, rho_safe)) {
    return table_pressure_from_rho_e(views.total, rho_safe, fmax(ee, 0.0),
                                     te_floor);
  }
  return (gamma - 1.0) * rho_safe * fmax(ee, 0.0);
}

__device__ inline void median_vector_rz(const double rc,
                                        const double zc,
                                        const double rm,
                                        const double zm,
                                        double* sr,
                                        double* sz) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double dr = rm - rc;
  const double dz = zm - zc;
  const double rsum = rc + rm;
  *sr = pi * rsum * dz;
  *sz = -pi * rsum * dr;
}

__device__ inline void triangle_corner_volumes(const double* r,
                                               const double* z,
                                               double* v_corner) {
  const double rc = (r[0] + r[1] + r[2]) / 3.0;
  const double zc = (z[0] + z[1] + z[2]) / 3.0;
  for (int k = 0; k < 3; ++k) {
    const int kp = (k + 1) % 3;
    const int km = (k + 2) % 3;
    const double r_sub[4] = {
        r[k],
        0.5 * (r[k] + r[kp]),
        rc,
        0.5 * (r[km] + r[k]),
    };
    const double z_sub[4] = {
        z[k],
        0.5 * (z[k] + z[kp]),
        zc,
        0.5 * (z[km] + z[k]),
    };
    v_corner[k] = fabs(rz::rz_polygon_volume_exact(r_sub, z_sub, 4));
  }
  v_corner[3] = 0.0;
}

__device__ inline void get_structured_cell_nodes(const int c,
                                                 const int nz,
                                                 int* nodes) {
  const int i = c / nz;
  const int j = c - i * nz;
  nodes[0] = structured_node_index(i, j, nz);
  nodes[1] = structured_node_index(i + 1, j, nz);
  nodes[2] = structured_node_index(i + 1, j + 1, nz);
  nodes[3] = structured_node_index(i, j + 1, nz);
}

__device__ inline void get_multiblock_cell_nodes(
    const int c,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    int* nodes) {
  const int off = cell_node_csr_offsets[c];
  nodes[0] = cell_node_csr_indices[off + 0];
  nodes[1] = cell_node_csr_indices[off + 1];
  nodes[2] = cell_node_csr_indices[off + 2];
  nodes[3] = cell_node_csr_indices[off + 3];
}

__device__ inline double subzonal_pressure_frequency_upper_bound(
    const double* v_corner,
    const double* corner_mass,
    const double* s_r,
    const double* s_z,
    const int nverts,
    const double dp_drho,
    const double merit) {
  if (!(dp_drho > 0.0) || !(merit > 0.0) || !isfinite(dp_drho) ||
      !isfinite(merit)) {
    return 0.0;
  }
  double omega2 = 0.0;
  for (int j = 0; j < nverts; ++j) {
    const int jm = (j + nverts - 1) % nverts;
    const int jp = (j + 1) % nverts;
    const double v = v_corner[j];
    const double m_sub = corner_mass[j];
    if (!(v > kTiny) || !(m_sub > kTiny) || !isfinite(v) ||
        !isfinite(m_sub)) {
      return INFINITY;
    }

    double b_r[4] = {};
    double b_z[4] = {};
    b_r[j] += s_r[j] - s_r[jm];
    b_z[j] += s_z[j] - s_z[jm];
    b_r[jm] += s_r[jm];
    b_z[jm] += s_z[jm];
    b_r[jp] -= s_r[j];
    b_z[jp] -= s_z[j];

    double inverse_mass_norm = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const double m = corner_mass[k];
      if (!(m > kTiny) || !isfinite(m)) {
        return INFINITY;
      }
      inverse_mass_norm +=
          (b_r[k] * b_r[k] + b_z[k] * b_z[k]) / m;
    }
    const double rho_sub = m_sub / v;
    omega2 += merit * (rho_sub * dp_drho / v) * inverse_mass_norm;
  }
  return omega2;
}

__device__ inline void record_subzonal_step_diagnostics(
    SubzonalStepDiagnosticsDevice* diagnostics,
    const double merit,
    const double max_abs_force,
    const double work) {
  if (diagnostics == nullptr) {
    return;
  }
  if (isfinite(merit) && merit >= 0.0) {
    atomic_max_double(&diagnostics->max_merit, merit);
  }
  if (isfinite(max_abs_force) && max_abs_force >= 0.0) {
    atomic_max_double(&diagnostics->max_abs_force, max_abs_force);
    if (max_abs_force > 0.0) {
      atomicAdd(&diagnostics->nonzero_force_cells, 1);
    }
  }
  const double abs_work = fabs(work);
  if (isfinite(abs_work)) {
    atomic_max_double(&diagnostics->max_abs_work, abs_work);
  }
}

__global__ void compute_subzonal_pressure_kernel(
    double* __restrict__ corner_force_sub_r,
    double* __restrict__ corner_force_sub_z,
    double* __restrict__ work_sub,
    double* __restrict__ spread,
    double* __restrict__ spread_corner,
    const double* __restrict__ corner_mass,
    const double* __restrict__ rho,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ pe,
    const double* __restrict__ pi,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int* __restrict__ cell_orientation_sign,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const double* __restrict__ subzonal_band_weight,
    const int n_cells,
    const int nz,
    const bool multiblock,
    const bool use_two_temp,
    const double gamma,
    const double te_floor,
    const double ti_floor,
    const double rho_floor,
    const SubzonalEOSViews eos_views,
    const int merit_mode,
    const double alpha1,
    const double alpha2,
    const int merit_power,
    const double merit_constant,
    SubzonalStepDiagnosticsDevice* diagnostics) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int base = c * 4;
  for (int k = 0; k < 4; ++k) {
    corner_force_sub_r[base + k] = 0.0;
    corner_force_sub_z[base + k] = 0.0;
  }
  work_sub[c] = 0.0;
  spread[c] = 0.0;
  spread_corner[c] = 0.0;

  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  int nodes[4] = {0, 0, 0, 0};
  if (multiblock && active_nverts == 3) {
    const int off = cell_node_csr_offsets[c];
    nodes[0] = cell_node_csr_indices[off + 0];
    nodes[1] = cell_node_csr_indices[off + 1];
    nodes[2] = cell_node_csr_indices[off + 2];
  } else if (multiblock) {
    get_multiblock_cell_nodes(c, cell_node_csr_offsets, cell_node_csr_indices,
                              nodes);
  } else {
    get_structured_cell_nodes(c, nz, nodes);
  }

  const double orientation_sign =
      (multiblock && cell_orientation_sign != nullptr)
          ? static_cast<double>(cell_orientation_sign[c])
          : 1.0;

  if (active_nverts == 3) {
    const double r[4] = {x_r[nodes[0]], x_r[nodes[1]], x_r[nodes[2]], 0.0};
    const double z[4] = {x_z[nodes[0]], x_z[nodes[1]], x_z[nodes[2]], 0.0};

    double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
    triangle_corner_volumes(r, z, v_corner);

    double rho_corner[4] = {0.0, 0.0, 0.0, 0.0};
    double dp[4] = {0.0, 0.0, 0.0, 0.0};
    const double rho_c = fmax(rho[c], rho_floor);
    const double p_cell = pe[c] + pi[c];
    double rho_max = 0.0;
    double max_abs_spread = 0.0;
    int max_spread_corner = 0;
    for (int k = 0; k < 3; ++k) {
      const double v = v_corner[k];
      if (!(isfinite(v) && v > kTiny)) {
        return;
      }
      rho_corner[k] = fmax(corner_mass[base + k], 0.0) / v;
      rho_max = fmax(rho_max, rho_corner[k]);
      const double rel =
          fabs(rho_corner[k] - rho_c) / fmax(rho_c, rho_floor);
      if (rel > max_abs_spread) {
        max_abs_spread = rel;
        max_spread_corner = k;
      }
      const double p_sub = subzonal_eos_pressure(
          eos_views, use_two_temp, rho_corner[k], ee[c], ei[c], gamma, te_floor,
          ti_floor);
      dp[k] = p_sub - p_cell;
    }
    spread[c] = max_abs_spread;
    spread_corner[c] = static_cast<double>(max_spread_corner);

    const double x_merit =
        fmax(0.0, (rho_max - rho_c) / fmax(rho_c, rho_floor));
    const double merit = compatible_subzonal_pressure_merit_factor(
        x_merit, static_cast<SubzonalMeritMode>(merit_mode), alpha1, alpha2,
        merit_power, merit_constant);

    const double rc = (r[0] + r[1] + r[2]) / 3.0;
    const double zc = (z[0] + z[1] + z[2]) / 3.0;
    double s_r[4] = {0.0, 0.0, 0.0, 0.0};
    double s_z[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const double rm = 0.5 * (r[k] + r[kp]);
      const double zm = 0.5 * (z[k] + z[kp]);
      median_vector_rz(rc, zc, rm, zm, &s_r[k], &s_z[k]);
      s_r[k] *= orientation_sign;
      s_z[k] *= orientation_sign;
    }

    double fr[4] = {0.0, 0.0, 0.0, 0.0};
    double fz[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 3; ++k) {
      const int km = (k + 2) % 3;
      const int kp = (k + 1) % 3;
      fr[k] = merit * ((dp[k] + dp[kp]) * s_r[k] -
                       (dp[k] + dp[km]) * s_r[km]);
      fz[k] = merit * ((dp[k] + dp[kp]) * s_z[k] -
                       (dp[k] + dp[km]) * s_z[km]);
    }

    const double sum_fr = fr[0] + fr[1] + fr[2];
    const double sum_fz = fz[0] + fz[1] + fz[2];
    double force_scale = 0.0;
    for (int k = 0; k < 3; ++k) {
      force_scale += hypot(fr[k], fz[k]);
    }
    const double residual = hypot(sum_fr, sum_fz);
    if (force_scale > 0.0 && residual > 1.0e-12 * force_scale) {
      printf("compatible_subzonal_pressure: momentum correction cell=%d rel=%.17e\n",
             c, residual / force_scale);
    }
    const double mean_fr = sum_fr / 3.0;
    const double mean_fz = sum_fz / 3.0;
    double ws = 0.0;
    double max_abs_force = 0.0;
    for (int k = 0; k < 3; ++k) {
      double fkr = fr[k] - mean_fr;
      double fkz = fz[k] - mean_fz;
      if (subzonal_band_weight != nullptr) {
        fkr *= subzonal_band_weight[c];
        fkz *= subzonal_band_weight[c];
      }
      corner_force_sub_r[base + k] = fkr;
      corner_force_sub_z[base + k] = fkz;
      const int n = nodes[k];
      ws -= fkr * v_r[n] + fkz * v_z[n];
      max_abs_force = fmax(max_abs_force, hypot(fkr, fkz));
    }
    work_sub[c] = ws;
    record_subzonal_step_diagnostics(diagnostics, merit, max_abs_force, ws);
    return;
  }

  const double r[4] = {x_r[nodes[0]], x_r[nodes[1]], x_r[nodes[2]],
                       x_r[nodes[3]]};
  const double z[4] = {x_z[nodes[0]], x_z[nodes[1]], x_z[nodes[2]],
                       x_z[nodes[3]]};

  double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
  rz::compute_quad_corner_volumes_exact_subpolygon(r[0], z[0], r[1], z[1],
                                                   r[2], z[2], r[3], z[3],
                                                   v_corner);

  double rho_corner[4] = {0.0, 0.0, 0.0, 0.0};
  double dp[4] = {0.0, 0.0, 0.0, 0.0};
  const double rho_c = fmax(rho[c], rho_floor);
  const double p_cell = pe[c] + pi[c];
  double rho_max = 0.0;
  double max_abs_spread = 0.0;
  int max_spread_corner = 0;
  for (int k = 0; k < 4; ++k) {
    const double v = v_corner[k];
    if (!(isfinite(v) && v > kTiny)) {
      return;
    }
    rho_corner[k] = fmax(corner_mass[base + k], 0.0) / v;
    rho_max = fmax(rho_max, rho_corner[k]);
    const double rel =
        fabs(rho_corner[k] - rho_c) / fmax(rho_c, rho_floor);
    if (rel > max_abs_spread) {
      max_abs_spread = rel;
      max_spread_corner = k;
    }
    const double p_sub = subzonal_eos_pressure(
        eos_views, use_two_temp, rho_corner[k], ee[c], ei[c], gamma, te_floor,
        ti_floor);
    dp[k] = p_sub - p_cell;
  }
  spread[c] = max_abs_spread;
  spread_corner[c] = static_cast<double>(max_spread_corner);

  const double x_merit = fmax(0.0, (rho_max - rho_c) / fmax(rho_c, rho_floor));
  const double merit = compatible_subzonal_pressure_merit_factor(
      x_merit, static_cast<SubzonalMeritMode>(merit_mode), alpha1, alpha2,
      merit_power, merit_constant);

  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  double s_r[4] = {0.0, 0.0, 0.0, 0.0};
  double s_z[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const double rm = 0.5 * (r[k] + r[kp]);
    const double zm = 0.5 * (z[k] + z[kp]);
    median_vector_rz(rc, zc, rm, zm, &s_r[k], &s_z[k]);
    s_r[k] *= orientation_sign;
    s_z[k] *= orientation_sign;
  }

  double fr[4] = {0.0, 0.0, 0.0, 0.0};
  double fz[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 4; ++k) {
    const int km = (k + 3) & 3;
    const int kp = (k + 1) & 3;
    fr[k] = merit * ((dp[k] + dp[kp]) * s_r[k] -
                     (dp[k] + dp[km]) * s_r[km]);
    fz[k] = merit * ((dp[k] + dp[kp]) * s_z[k] -
                     (dp[k] + dp[km]) * s_z[km]);
  }

  const double sum_fr = fr[0] + fr[1] + fr[2] + fr[3];
  const double sum_fz = fz[0] + fz[1] + fz[2] + fz[3];
  double force_scale = 0.0;
  for (int k = 0; k < 4; ++k) {
    force_scale += hypot(fr[k], fz[k]);
  }
  const double residual = hypot(sum_fr, sum_fz);
  if (force_scale > 0.0 && residual > 1.0e-12 * force_scale) {
    printf("compatible_subzonal_pressure: momentum correction cell=%d rel=%.17e\n",
           c, residual / force_scale);
  }
  const double mean_fr = 0.25 * sum_fr;
  const double mean_fz = 0.25 * sum_fz;
  double ws = 0.0;
  double max_abs_force = 0.0;
  for (int k = 0; k < 4; ++k) {
    double fkr = fr[k] - mean_fr;
    double fkz = fz[k] - mean_fz;
    if (subzonal_band_weight != nullptr) {
      fkr *= subzonal_band_weight[c];
      fkz *= subzonal_band_weight[c];
    }
    corner_force_sub_r[base + k] = fkr;
    corner_force_sub_z[base + k] = fkz;
    const int n = nodes[k];
    ws -= fkr * v_r[n] + fkz * v_z[n];
    max_abs_force = fmax(max_abs_force, hypot(fkr, fkz));
  }
  work_sub[c] = ws;
  record_subzonal_step_diagnostics(diagnostics, merit, max_abs_force, ws);
}

__global__ void compute_subzonal_pressure_dt_kernel(
    double* min_dt,
    const double* __restrict__ corner_mass,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int nz,
    const bool multiblock,
    const double rho_floor,
    const double cfl_hydro,
    const int merit_mode,
    const double alpha1,
    const double alpha2,
    const int merit_power,
    const double merit_constant) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells ||
      (hydro_active != nullptr && hydro_active[c] == 0)) {
    return;
  }

  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  if (active_nverts != 3 && active_nverts != 4) {
    return;
  }
  int nodes[4] = {0, 0, 0, 0};
  if (multiblock && active_nverts == 3) {
    const int off = cell_node_csr_offsets[c];
    nodes[0] = cell_node_csr_indices[off + 0];
    nodes[1] = cell_node_csr_indices[off + 1];
    nodes[2] = cell_node_csr_indices[off + 2];
  } else if (multiblock) {
    get_multiblock_cell_nodes(c, cell_node_csr_offsets, cell_node_csr_indices,
                              nodes);
  } else {
    get_structured_cell_nodes(c, nz, nodes);
  }

  double r[4] = {};
  double z[4] = {};
  for (int k = 0; k < active_nverts; ++k) {
    r[k] = x_r[nodes[k]];
    z[k] = x_z[nodes[k]];
  }

  double v_corner[4] = {};
  if (active_nverts == 3) {
    triangle_corner_volumes(r, z, v_corner);
  } else {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], v_corner);
  }

  const double* cell_corner_mass = corner_mass + 4 * c;
  const double rho_c = fmax(rho[c], rho_floor);
  double rho_max = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    if (!(v_corner[k] > kTiny) || !isfinite(v_corner[k])) {
      return;
    }
    rho_max = fmax(rho_max, fmax(cell_corner_mass[k], 0.0) / v_corner[k]);
  }
  const double x_merit =
      fmax(0.0, (rho_max - rho_c) / fmax(rho_c, rho_floor));
  const double merit = compatible_subzonal_pressure_merit_factor(
      x_merit, static_cast<SubzonalMeritMode>(merit_mode), alpha1, alpha2,
      merit_power, merit_constant);

  double s_r[4] = {};
  double s_z[4] = {};
  double rc = 0.0;
  double zc = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    rc += r[k] / static_cast<double>(active_nverts);
    zc += z[k] / static_cast<double>(active_nverts);
  }
  for (int k = 0; k < active_nverts; ++k) {
    const int kp = (k + 1) % active_nverts;
    median_vector_rz(rc, zc, 0.5 * (r[k] + r[kp]),
                     0.5 * (z[k] + z[kp]), &s_r[k], &s_z[k]);
  }

  const double dp_drho = cs[c] * cs[c];
  const double omega2 = subzonal_pressure_frequency_upper_bound(
      v_corner, cell_corner_mass, s_r, s_z, active_nverts, dp_drho, merit);
  if (isinf(omega2)) {
    atomic_min_double(min_dt, 0.0);
  } else if (omega2 > 0.0 && isfinite(omega2)) {
    atomic_min_double(min_dt, cfl_hydro / sqrt(omega2));
  }
}

__global__ void compute_subzonal_buffer_diagnostics_kernel(
    SubzonalStepDiagnosticsDevice* diagnostics,
    const double* __restrict__ corner_force_sub_r,
    const double* __restrict__ corner_force_sub_z,
    const double* __restrict__ work_sub,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double max_abs_force = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int idx = 4 * c + k;
    max_abs_force =
        fmax(max_abs_force,
             hypot(corner_force_sub_r[idx], corner_force_sub_z[idx]));
  }
  if (isfinite(max_abs_force)) {
    atomic_max_double(&diagnostics->max_abs_force, max_abs_force);
    if (max_abs_force > 0.0) {
      atomicAdd(&diagnostics->nonzero_force_cells, 1);
    }
  }
  const double abs_work = fabs(work_sub[c]);
  if (isfinite(abs_work)) {
    atomic_max_double(&diagnostics->max_abs_work, abs_work);
  }
}

SubzonalEOSViews select_eos_views(const HydroEOSContext* eos_ctx) {
  SubzonalEOSViews views{};
  if (eos_ctx == nullptr || eos_ctx->n_materials <= 0) {
    return views;
  }
  views.ion = eos_ctx->ion_view(0);
  views.electron = eos_ctx->electron_view(0);
  views.total = eos_ctx->total_view(0);
  return views;
}

int merit_mode_id(const core::Config& cfg) {
  const std::string& mode = cfg.numerics.hydro.subzonal_merit_mode;
  if (mode == "constant") {
    return static_cast<int>(SubzonalMeritMode::Constant);
  }
  if (mode == "off") {
    return static_cast<int>(SubzonalMeritMode::Off);
  }
  return static_cast<int>(SubzonalMeritMode::CaramanaAuto);
}

void ensure_subzonal_buffers(core::State& state, const std::size_t n_cells) {
  const std::size_t n_corners = n_cells * 4U;
  if (state.corner_force_sub_r.size() != n_corners) {
    state.corner_force_sub_r.reset(n_corners);
  }
  if (state.corner_force_sub_z.size() != n_corners) {
    state.corner_force_sub_z.reset(n_corners);
  }
  if (state.work_sub_per_cell.size() != n_cells) {
    state.work_sub_per_cell.reset(n_cells);
  }
}

void update_spread_diagnostics(core::State& state,
                               const core::CellField1D& spread,
                               const core::CellField1D& spread_corner) {
  const int n_cells = static_cast<int>(spread.size());
  if (n_cells <= 0) {
    state.max_corner_density_spread_step = 0.0;
    return;
  }
  auto begin = thrust::device_pointer_cast(spread.data());
  auto end = begin + n_cells;
  auto iter = thrust::max_element(begin, end);
  const int cell = static_cast<int>(iter - begin);
  double value = 0.0;
  double corner_value = 0.0;
  cuda_check(cudaMemcpy(&value, spread.data() + cell, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "compatible subzonal spread copy max value failed");
  cuda_check(cudaMemcpy(&corner_value, spread_corner.data() + cell,
                        sizeof(double), cudaMemcpyDeviceToHost),
             "compatible subzonal spread copy corner failed");
  state.max_corner_density_spread_step = value;
  if (value > state.max_corner_density_spread_run) {
    state.max_corner_density_spread_run = value;
    state.max_corner_density_spread_step_idx = state.step;
    state.max_corner_density_spread_cell_id = cell;
    state.max_corner_density_spread_corner_id =
        static_cast<int>(corner_value + 0.5);
  }
}

}  // namespace

double compute_compatible_subzonal_pressure_dt_2d(
    const core::State& state,
    const core::Config& cfg,
    const std::int8_t* d_hydro_active) {
  if (!cfg.numerics.hydro.subzonal_pressure_enabled || cfg.main.dim != 2 ||
      state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const std::size_t n_corners = static_cast<std::size_t>(n_cells) * 4U;
  TENRYU_ASSERT(state.corner_stride == 4 &&
                    state.corner_mass.size() == n_corners &&
                    state.corner_mass_initialized &&
                    state.corner_mass_is_lagrangian_invariant,
                "subzonal pressure dt requires initialized fixed corner masses");
  TENRYU_ASSERT(state.cs.size() == state.rho.size(),
                "subzonal pressure dt requires EOS sound speed");

  core::DeviceArray<double> d_min_dt;
  d_min_dt.reset(1U);
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt.data(), &inf, sizeof(double),
                        cudaMemcpyHostToDevice),
             "subzonal pressure dt init failed");

  const bool multiblock = state.mesh.topo.multiblock.has_value();
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_tri(d_cell_nverts, state, n_cells);
  const int blocks = (n_cells + 255) / 256;
  compute_subzonal_pressure_dt_kernel<<<blocks, 256>>>(
      d_min_dt.data(), state.corner_mass.data(), state.rho.data(),
      state.cs.data(), state.x_r.data(), state.x_z.data(),
      multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data() : nullptr,
      multiblock ? state.mesh.multiblock_cell_node_csr_indices.data() : nullptr,
      d_cell_nverts_ptr, d_hydro_active, n_cells, state.mesh.topo.nz,
      multiblock, cfg.numerics.floors.rho, cfg.numerics.dt.cfl_hydro,
      merit_mode_id(cfg), cfg.numerics.hydro.subzonal_alpha1,
      cfg.numerics.hydro.subzonal_alpha2,
      cfg.numerics.hydro.subzonal_merit_power,
      cfg.numerics.hydro.subzonal_merit_constant);
  sync_kernel("Hydro2D subzonal pressure dt kernel failed");

  double dt = inf;
  cuda_check(cudaMemcpy(&dt, d_min_dt.data(), sizeof(double),
                        cudaMemcpyDeviceToHost),
             "subzonal pressure dt copy failed");
  return dt;
}

void update_compatible_subzonal_pressure_observability_2d(
    core::State& state,
    const core::Config& cfg) {
  if (!cfg.numerics.hydro.subzonal_pressure_enabled) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    state.subzonal_nonzero_force_cells_step = 0;
    state.max_abs_subzonal_force_step = 0.0;
    state.max_abs_subzonal_work_step = 0.0;
    return;
  }
  TENRYU_ASSERT(
      state.corner_force_sub_r.size() == static_cast<std::size_t>(n_cells) * 4U &&
          state.corner_force_sub_z.size() ==
              static_cast<std::size_t>(n_cells) * 4U &&
          state.work_sub_per_cell.size() == static_cast<std::size_t>(n_cells),
      "subzonal observability requires compatible force/work buffers");
  core::DeviceArray<SubzonalStepDiagnosticsDevice> d_diagnostics;
  d_diagnostics.reset(1U);
  cuda_check(cudaMemset(d_diagnostics.data(), 0,
                        sizeof(SubzonalStepDiagnosticsDevice)),
             "compatible subzonal buffer diagnostics zero failed");
  const int blocks = (n_cells + 255) / 256;
  compute_subzonal_buffer_diagnostics_kernel<<<blocks, 256>>>(
      d_diagnostics.data(), state.corner_force_sub_r.data(),
      state.corner_force_sub_z.data(), state.work_sub_per_cell.data(),
      n_cells);
  sync_kernel("compatible subzonal buffer diagnostics kernel failed");
  SubzonalStepDiagnosticsDevice diagnostics{};
  cuda_check(cudaMemcpy(&diagnostics, d_diagnostics.data(),
                        sizeof(SubzonalStepDiagnosticsDevice),
                        cudaMemcpyDeviceToHost),
             "compatible subzonal buffer diagnostics copy failed");
  state.subzonal_nonzero_force_cells_step = diagnostics.nonzero_force_cells;
  state.max_abs_subzonal_force_step = diagnostics.max_abs_force;
  state.max_abs_subzonal_work_step = diagnostics.max_abs_work;
}

void compute_compatible_subzonal_pressure_force_2d(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const std::int8_t* d_hydro_active) {
  if (!cfg.numerics.hydro.subzonal_pressure_enabled || cfg.main.dim != 2) {
    return;
  }
  TENRYU_ASSERT(
      state.corner_stride == 4,
      "corner_stride != 4: subzonal-pressure corner path is staged for a later revision");
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    state.max_corner_density_spread_step = 0.0;
    state.max_subzonal_merit_step = 0.0;
    state.subzonal_nonzero_force_cells_step = 0;
    state.max_abs_subzonal_force_step = 0.0;
    state.max_abs_subzonal_work_step = 0.0;
    return;
  }
  const std::size_t n_corners = static_cast<std::size_t>(n_cells) * 4U;
  TENRYU_ASSERT(state.corner_mass.size() == n_corners &&
                    state.corner_mass_initialized,
                "compatible subzonal pressure requires initialized invariant corner_mass");
  TENRYU_ASSERT(state.ee.size() == static_cast<std::size_t>(n_cells) &&
                    state.ei.size() == static_cast<std::size_t>(n_cells) &&
                    state.Pe.size() == static_cast<std::size_t>(n_cells) &&
                    state.Pi.size() == static_cast<std::size_t>(n_cells),
                "compatible subzonal pressure requires EOS closure fields");

  ensure_subzonal_buffers(state, static_cast<std::size_t>(n_cells));
  zero_array(state.corner_force_sub_r.data(), n_corners,
             "compatible subzonal pressure zero corner_force_sub_r failed");
  zero_array(state.corner_force_sub_z.data(), n_corners,
             "compatible subzonal pressure zero corner_force_sub_z failed");
  zero_array(state.work_sub_per_cell.data(), static_cast<std::size_t>(n_cells),
             "compatible subzonal pressure zero work_sub_per_cell failed");
  core::CellField1D spread;
  core::CellField1D spread_corner;
  spread.reset(static_cast<std::size_t>(n_cells));
  spread_corner.reset(static_cast<std::size_t>(n_cells));
  core::DeviceArray<SubzonalStepDiagnosticsDevice> d_diagnostics;
  d_diagnostics.reset(1U);
  cuda_check(cudaMemset(d_diagnostics.data(), 0,
                        sizeof(SubzonalStepDiagnosticsDevice)),
             "compatible subzonal diagnostics zero failed");

  const bool multiblock = state.mesh.topo.multiblock.has_value();
  core::DeviceArray<int> d_cell_orientation_sign;
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_tri(d_cell_nverts, state, n_cells);
  const int* cell_orientation_sign = nullptr;
  if (multiblock) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "compatible subzonal pressure requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      n_corners,
                  "compatible subzonal pressure requires four CSR nodes per cell");
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                      static_cast<std::size_t>(n_cells),
                  "compatible subzonal pressure requires cell orientation signs");
    d_cell_orientation_sign.reset(mb.cell_orientation_sign.size());
    d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
    cell_orientation_sign = d_cell_orientation_sign.data();
  }

  const double gamma =
      cfg.materials.materials.empty()
          ? 5.0 / 3.0
          : std::max(cfg.materials.materials.front().ideal_gas_gamma,
                     1.0 + 1.0e-12);
  const double* subzonal_band_weight =
      ensure_bridge_band_subzonal_weights_2d(state, cfg);
  const int blocks = (n_cells + 255) / 256;
  compute_subzonal_pressure_kernel<<<blocks, 256>>>(
      state.corner_force_sub_r.data(),
      state.corner_force_sub_z.data(),
      state.work_sub_per_cell.data(),
      spread.data(),
      spread_corner.data(),
      state.corner_mass.data(),
      state.rho.data(),
      state.ee.data(),
      state.ei.data(),
      state.Pe.data(),
      state.Pi.data(),
      state.x_r.data(),
      state.x_z.data(),
      state.v_r.data(),
      state.v_z.data(),
      multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data() : nullptr,
      multiblock ? state.mesh.multiblock_cell_node_csr_indices.data() : nullptr,
      multiblock ? cell_orientation_sign : nullptr,
      d_cell_nverts_ptr,
      d_hydro_active,
      subzonal_band_weight,
      n_cells,
      state.mesh.topo.nz,
      multiblock,
      cfg.main.two_temperature,
      gamma,
      cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      cfg.numerics.floors.rho,
      select_eos_views(eos_ctx),
      merit_mode_id(cfg),
      cfg.numerics.hydro.subzonal_alpha1,
      cfg.numerics.hydro.subzonal_alpha2,
      cfg.numerics.hydro.subzonal_merit_power,
      cfg.numerics.hydro.subzonal_merit_constant,
      d_diagnostics.data());
  sync_kernel("Hydro2D compatible subzonal pressure kernel failed");
  SubzonalStepDiagnosticsDevice diagnostics{};
  cuda_check(cudaMemcpy(&diagnostics, d_diagnostics.data(),
                        sizeof(SubzonalStepDiagnosticsDevice),
                        cudaMemcpyDeviceToHost),
             "compatible subzonal diagnostics copy failed");
  state.max_subzonal_merit_step = diagnostics.max_merit;
  state.subzonal_nonzero_force_cells_step = diagnostics.nonzero_force_cells;
  state.max_abs_subzonal_force_step = diagnostics.max_abs_force;
  state.max_abs_subzonal_work_step = diagnostics.max_abs_work;
  update_spread_diagnostics(state, spread, spread_corner);
}

}  // namespace tenryu::hydro
