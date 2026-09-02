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

#include "core/axis_tolerance.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/pentagon_affine_null.cuh"
#include "hydro/pentagon_geometry.cuh"
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

enum class SubzonalShadowMode : int {
  Production = 0,
  AxisSource = 1,
  OffaxisSource = 2,
};

inline void cuda_check(const cudaError_t err, const char* message) {
  if (err != cudaSuccess) {
    TENRYU_ASSERT(false, std::string(message) + ": " + cudaGetErrorName(err));
  }
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

bool has_nonquad_cell_nverts(const std::vector<std::uint8_t>& cell_nverts,
                             const int n_cells) {
  if (cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(cell_nverts.begin(), cell_nverts.end(),
                     [](const std::uint8_t nverts) {
                       return nverts != 4U;
                     });
}

const std::uint8_t* upload_cell_nverts_if_nonquad(
    core::DeviceArray<std::uint8_t>& d_cell_nverts,
    const core::State& state,
    const int n_cells) {
  if (!has_nonquad_cell_nverts(state.mesh.cell_nverts, n_cells)) {
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

__device__ inline void median_vector_planar(const double rc,
                                            const double zc,
                                            const double rm,
                                            const double zm,
                                            double* sr,
                                            double* sz) {
  *sr = zm - zc;
  *sz = -(rm - rc);
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

template <int SlotCap = mesh::kMeshTopoCellStorageSlotsMax>
__device__ inline void polygon_corner_rz_volumes_slot_cap(
    const PentagonPoint* x,
    const int nverts,
    double* volumes) {
  if (nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#endif
    return;
  }
  const PentagonPoint center = polygon_center(x, nverts);
  PentagonPoint midpoints[SlotCap];
  polygon_edge_midpoints_n(x, nverts, midpoints);
  for (int k = 0; k < nverts; ++k) {
    const int km1 = (k == 0) ? nverts - 1 : k - 1;
    const PentagonPoint corner[4] = {
        x[k],
        midpoints[k],
        center,
        midpoints[km1],
    };
    volumes[k] = polygon_rz_volume(corner, 4);
  }
}

__device__ inline void validate_pentagon_corner_volumes(
    const double* v_corner) {
  for (int k = 0; k < 5; ++k) {
    if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
      printf("pentagon subzonal corner volume invalid\n");
      __trap();
    }
  }
}

__device__ inline void validate_polygon_corner_volumes_n(
    const double* v_corner,
    const int nverts) {
  for (int k = 0; k < nverts; ++k) {
    if (v_corner[k] > 0.0 && isfinite(v_corner[k])) {
      return;
    }
  }
  printf("polygon subzonal corner volume invalid\n");
  __trap();
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

template <int SlotCap = mesh::kMeshTopoCellStorageSlotsMax>
__device__ inline double subzonal_pressure_frequency_upper_bound(
    const double* v_corner,
    const double* corner_mass,
    const double* s_r,
    const double* s_z,
    const int nverts,
    const double dp_drho,
    const double merit,
    const double* corner_r,
    const double axis_tol,
    const bool exclude_invalid_polygon_corners) {
  if (nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#endif
    return INFINITY;
  }
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
    if (exclude_invalid_polygon_corners &&
        (!(v > 0.0) || !isfinite(v))) {
      continue;
    }
    if (!(v > kTiny) || !(m_sub > kTiny) || !isfinite(v) ||
        !isfinite(m_sub)) {
      if (fabs(corner_r[j]) <= axis_tol) {
        continue;  // structurally massless on-axis corner: not a stiffness DOF
      }
      return INFINITY;
    }

    double b_r[SlotCap] = {};
    double b_z[SlotCap] = {};
    b_r[j] += s_r[j] - s_r[jm];
    b_z[j] += s_z[j] - s_z[jm];
    b_r[jm] += s_r[jm];
    b_z[jm] += s_z[jm];
    b_r[jp] -= s_r[j];
    b_z[jp] -= s_z[j];

    double inverse_mass_norm = 0.0;
    for (int k = 0; k < nverts; ++k) {
      if (exclude_invalid_polygon_corners &&
          (!(v_corner[k] > 0.0) || !isfinite(v_corner[k]))) {
        continue;
      }
      const double m = corner_mass[k];
      if (!(m > kTiny) || !isfinite(m)) {
        if (fabs(corner_r[k]) <= axis_tol) {
          continue;  // on-axis massless corner carries no inertia coupling
        }
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

template <SubzonalShadowMode ShadowMode, bool CaptureProjectionDebug,
          int SlotCap = mesh::kMeshTopoCellStorageSlotsMax>
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
    const double* __restrict__ vol,
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
    const int corner_stride,
    const int nz,
    const bool multiblock,
    const bool polar_tier,
    const bool aw_planar,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
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
    const bool pentagon_affine_null_enabled,
    const double pentagon_affine_null_kappa,
    int* __restrict__ an_degenerate_count,
    SubzonalStepDiagnosticsDevice* diagnostics,
    const SubzonalPressureProjectionDebugBuffers projection_debug) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int base = c * corner_stride;
  for (int k = 0; k < corner_stride; ++k) {
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
  // class-B whole-polygon consumer: multi-page cells unsupported until C3-final
  if (active_nverts > 14) {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "active_nverts <= 14",
        "class-B whole-polygon consumer: multi-page cells unsupported until "
        "C3-final",
        __FILE__, __LINE__);
#endif
  }
  if (active_nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#endif
    return;
  }
  if constexpr (ShadowMode != SubzonalShadowMode::Production) {
    if (multiblock || !aw_planar || active_nverts != 4) {
      return;
    }
    const int structured_j = c - (c / nz) * nz;
    const bool theta0_axis_wedge =
        aw_axis_slave_theta0_active && structured_j == 0;
    const bool theta_pi_axis_wedge =
        aw_axis_slave_theta_pi_active && structured_j == nz - 1;
    if (!theta0_axis_wedge && !theta_pi_axis_wedge) {
      return;
    }
  }
  int nodes[SlotCap] = {};
  if (multiblock && active_nverts == 3) {
    const int off = cell_node_csr_offsets[c];
    nodes[0] = cell_node_csr_indices[off + 0];
    nodes[1] = cell_node_csr_indices[off + 1];
    nodes[2] = cell_node_csr_indices[off + 2];
  } else if (multiblock && active_nverts == 5) {
    const int off = cell_node_csr_offsets[c];
    for (int k = 0; k < 5; ++k) {
      nodes[k] = cell_node_csr_indices[off + k];
    }
  } else if (multiblock && active_nverts >= 6 &&
             active_nverts <= SlotCap) {
    const int off = cell_node_csr_offsets[c];
    for (int k = 0; k < active_nverts; ++k) {
      nodes[k] = cell_node_csr_indices[off + k];
    }
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
    if (polar_tier) {
      rz::compute_triangle_corner_volumes_equal_planar_area(
          r[0], z[0], r[1], z[1], r[2], z[2], v_corner);
    } else {
      triangle_corner_volumes(r, z, v_corner);
    }

    double rho_corner[4] = {0.0, 0.0, 0.0, 0.0};
    double dp[4] = {0.0, 0.0, 0.0, 0.0};
    const double rho_c = fmax(rho[c], rho_floor);
    const double p_cell = pe[c] + pi[c];
    double rho_max = 0.0;
    double max_abs_spread = 0.0;
    int max_spread_corner = -1;
    int origin_corner = -1;
    if (polar_tier) {
      for (int k = 0; k < 3; ++k) {
        if (r[k] == 0.0 && z[k] == 0.0) {
          origin_corner = k;
          break;
        }
      }
    }
    for (int k = 0; k < 3; ++k) {
      if (k == origin_corner) {
        dp[k] = 0.0;
        continue;
      }
      const double v = v_corner[k];
      if (!(isfinite(v) && v > kTiny)) {
        return;
      }
      rho_corner[k] = fmax(corner_mass[base + k], 0.0) / v;
      rho_max = fmax(rho_max, rho_corner[k]);
      const double rel =
          fabs(rho_corner[k] - rho_c) / fmax(rho_c, rho_floor);
      if (max_spread_corner < 0 || rel > max_abs_spread) {
        max_abs_spread = rel;
        max_spread_corner = k;
      }
      const double p_sub = subzonal_eos_pressure(
          eos_views, use_two_temp, rho_corner[k], ee[c], ei[c], gamma, te_floor,
          ti_floor);
      dp[k] = p_sub - p_cell;
    }
    spread[c] = max_abs_spread;
    spread_corner[c] =
        static_cast<double>((max_spread_corner >= 0) ? max_spread_corner : 0);

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
      if (aw_planar) {
        median_vector_planar(rc, zc, rm, zm, &s_r[k], &s_z[k]);
      } else {
        median_vector_rz(rc, zc, rm, zm, &s_r[k], &s_z[k]);
      }
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
    const int correction_corner_count = (origin_corner >= 0) ? 2 : 3;
    const double mean_fr =
        sum_fr / static_cast<double>(correction_corner_count);
    const double mean_fz =
        sum_fz / static_cast<double>(correction_corner_count);
    double ws = 0.0;
    double max_abs_force = 0.0;
    for (int k = 0; k < 3; ++k) {
      double fkr = fr[k];
      double fkz = fz[k];
      if (k != origin_corner) {
        fkr -= mean_fr;
        fkz -= mean_fz;
      }
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

  if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      x[k] = {x_r[nodes[k]], x_z[nodes[k]]};
    }
    double v_corner[5] = {};
    pentagon_corner_rz_volumes(x, v_corner);
    validate_pentagon_corner_volumes(v_corner);

    const double rho_c = fmax(rho[c], rho_floor);
    const double p_cell = pe[c] + pi[c];
    double rho_corner[5] = {};
    double dph[5] = {};
    double dp[5] = {};
    double rho_max = 0.0;
    double max_abs_spread = 0.0;
    int max_spread_corner = 0;
    for (int k = 0; k < 5; ++k) {
      rho_corner[k] = corner_mass[base + k] / v_corner[k];
      rho_max = fmax(rho_max, rho_corner[k]);
      const double rel =
          fabs(rho_corner[k] - rho_c) / fmax(rho_c, rho_floor);
      if (rel > max_abs_spread) {
        max_abs_spread = rel;
        max_spread_corner = k;
      }
      const double p_sub = subzonal_eos_pressure(
          eos_views, use_two_temp, rho_corner[k], ee[c], ei[c], gamma,
          te_floor, ti_floor);
      dph[k] = p_sub - p_cell;
    }
    spread[c] = max_abs_spread;
    spread_corner[c] = static_cast<double>(max_spread_corner);

    const double v_cell = vol[c];
    double weighted_dph = 0.0;
    for (int k = 0; k < 5; ++k) {
      weighted_dph += (v_corner[k] / v_cell) * dph[k];
    }
    for (int k = 0; k < 5; ++k) {
      dp[k] = dph[k] - weighted_dph;
    }

    const double x_merit = (rho_max - rho_c) / rho_c;
    const double merit = compatible_subzonal_pressure_merit_factor(
        x_merit, static_cast<SubzonalMeritMode>(merit_mode), alpha1, alpha2,
        merit_power, merit_constant);

    const PentagonPoint center = pentagon_center(x);
    PentagonPoint midpoints[5];
    pentagon_edge_midpoints(x, midpoints);
    double s_r[5] = {};
    double s_z[5] = {};
    for (int k = 0; k < 5; ++k) {
      median_vector_rz(center.r, center.z, midpoints[k].r, midpoints[k].z,
                       &s_r[k], &s_z[k]);
      s_r[k] *= orientation_sign;
      s_z[k] *= orientation_sign;
    }

    double fr[5] = {};
    double fz[5] = {};
    for (int k = 0; k < 5; ++k) {
      const int km = (k == 0) ? 4 : k - 1;
      const int kp = (k == 4) ? 0 : k + 1;
      fr[k] = merit * ((dp[k] + dp[kp]) * s_r[k] -
                       (dp[k] + dp[km]) * s_r[km]);
      fz[k] = merit * ((dp[k] + dp[kp]) * s_z[k] -
                       (dp[k] + dp[km]) * s_z[km]);
    }
    if (pentagon_affine_null_enabled &&
        pentagon_affine_null_kappa != 0.0) {
      double mass[5];
      double velocity_r[5];
      double velocity_z[5];
      for (int k = 0; k < 5; ++k) {
        mass[k] = corner_mass[base + k];
        velocity_r[k] = v_r[nodes[k]];
        velocity_z[k] = v_z[nodes[k]];
      }
      // Damping scale only, not the CFL signal speed.
      const double cs = sqrt(fmax(gamma * p_cell / rho_c, 0.0));
      double affine_null_force_r[5];
      double affine_null_force_z[5];
      const bool affine_null_nondegenerate = pentagon_affine_null_force(
          x, mass, velocity_r, velocity_z, pentagon_affine_null_kappa, cs,
          affine_null_force_r, affine_null_force_z);
      if (!affine_null_nondegenerate && an_degenerate_count != nullptr) {
        atomicAdd(an_degenerate_count, 1);
      }
      for (int k = 0; k < 5; ++k) {
        fr[k] += affine_null_force_r[k];
        fz[k] += affine_null_force_z[k];
      }
    }
    fr[2] = -((fr[0] + fr[1]) + (fr[3] + fr[4]));
    fz[2] = -((fz[0] + fz[1]) + (fz[3] + fz[4]));

    double ws = 0.0;
    double max_abs_force = 0.0;
    for (int k = 0; k < 5; ++k) {
      double fkr = fr[k];
      double fkz = fz[k];
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

  if (active_nverts >= 6 && active_nverts <= SlotCap) {
    PentagonPoint x[SlotCap];
    for (int k = 0; k < active_nverts; ++k) {
      x[k] = {x_r[nodes[k]], x_z[nodes[k]]};
    }
    double v_corner[SlotCap] = {};
    polygon_corner_rz_volumes_slot_cap<SlotCap>(x, active_nverts, v_corner);
    validate_polygon_corner_volumes_n(v_corner, active_nverts);

    const double rho_c = fmax(rho[c], rho_floor);
    const double p_cell = pe[c] + pi[c];
    double rho_corner[SlotCap] = {};
    double dph[SlotCap] = {};
    double dp[SlotCap] = {};
    double rho_max = 0.0;
    double max_abs_spread = 0.0;
    int max_spread_corner = -1;
    for (int k = 0; k < active_nverts; ++k) {
      if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
        continue;
      }
      rho_corner[k] = corner_mass[base + k] / v_corner[k];
      rho_max = fmax(rho_max, rho_corner[k]);
      const double rel =
          fabs(rho_corner[k] - rho_c) / fmax(rho_c, rho_floor);
      if (max_spread_corner < 0 || rel > max_abs_spread) {
        max_abs_spread = rel;
        max_spread_corner = k;
      }
      const double p_sub = subzonal_eos_pressure(
          eos_views, use_two_temp, rho_corner[k], ee[c], ei[c], gamma,
          te_floor, ti_floor);
      dph[k] = p_sub - p_cell;
    }
    spread[c] = max_abs_spread;
    spread_corner[c] = static_cast<double>(max_spread_corner);

    const double v_cell = vol[c];
    double weighted_dph = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
        continue;
      }
      weighted_dph += (v_corner[k] / v_cell) * dph[k];
    }
    for (int k = 0; k < active_nverts; ++k) {
      if (!(v_corner[k] > 0.0) || !isfinite(v_corner[k])) {
        dp[k] = 0.0;
        continue;
      }
      dp[k] = dph[k] - weighted_dph;
    }

    const double x_merit = (rho_max - rho_c) / rho_c;
    const double merit = compatible_subzonal_pressure_merit_factor(
        x_merit, static_cast<SubzonalMeritMode>(merit_mode), alpha1, alpha2,
        merit_power, merit_constant);

    const PentagonPoint center = polygon_center(x, active_nverts);
    PentagonPoint midpoints[SlotCap];
    polygon_edge_midpoints_n(x, active_nverts, midpoints);
    double s_r[SlotCap] = {};
    double s_z[SlotCap] = {};
    for (int k = 0; k < active_nverts; ++k) {
      median_vector_rz(center.r, center.z, midpoints[k].r, midpoints[k].z,
                       &s_r[k], &s_z[k]);
      s_r[k] *= orientation_sign;
      s_z[k] *= orientation_sign;
    }

    double fr[SlotCap] = {};
    double fz[SlotCap] = {};
    for (int k = 0; k < active_nverts; ++k) {
      const int km = (k == 0) ? active_nverts - 1 : k - 1;
      const int kp = (k + 1 == active_nverts) ? 0 : k + 1;
      fr[k] = merit * ((dp[k] + dp[kp]) * s_r[k] -
                       (dp[k] + dp[km]) * s_r[km]);
      fz[k] = merit * ((dp[k] + dp[kp]) * s_z[k] -
                       (dp[k] + dp[km]) * s_z[km]);
    }
    const int correction_corner = active_nverts / 2;
    double correction_force_r = 0.0;
    double correction_force_z = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      if (k != correction_corner) {
        correction_force_r += fr[k];
        correction_force_z += fz[k];
      }
    }
    fr[correction_corner] = -correction_force_r;
    fz[correction_corner] = -correction_force_z;

    double ws = 0.0;
    double max_abs_force = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      double fkr = fr[k];
      double fkz = fz[k];
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

  if (active_nverts != 4) {
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
  const int structured_j =
      aw_planar && !multiblock ? c - (c / nz) * nz : -1;
  const bool theta0_axis_wedge =
      aw_planar && !multiblock && aw_axis_slave_theta0_active &&
      structured_j == 0;
  const bool theta_pi_axis_wedge =
      aw_planar && !multiblock && aw_axis_slave_theta_pi_active &&
      structured_j == nz - 1;
  const int axis_corner[2] = {
      theta0_axis_wedge ? 0 : 3,
      theta0_axis_wedge ? 1 : 2,
  };
  const int partner_corner[2] = {
      theta0_axis_wedge ? 3 : 0,
      theta0_axis_wedge ? 2 : 1,
  };
  if constexpr (CaptureProjectionDebug) {
    if (theta0_axis_wedge || theta_pi_axis_wedge) {
      for (int pair = 0; pair < 2; ++pair) {
        const int axis = axis_corner[pair];
        const int partner = partner_corner[pair];
        const int debug_index = 2 * c + pair;
        projection_debug.rho_axis_raw[debug_index] =
            fmax(corner_mass[base + axis], 0.0) / v_corner[axis];
        projection_debug.rho_partner_raw[debug_index] =
            fmax(corner_mass[base + partner], 0.0) / v_corner[partner];
      }
    }
  }
  bool projected_axis_wedge = false;
  if (theta0_axis_wedge || theta_pi_axis_wedge) {
    double r_scale = 0.0;
    for (int k = 0; k < 4; ++k) {
      r_scale = fmax(r_scale, fabs(r[k]));
    }
    const double axis_tol = core::axis_tolerance(r_scale);
    const bool origin_cell =
        fabs(r[0]) <= axis_tol && fabs(z[0]) <= axis_tol &&
        fabs(r[3]) <= axis_tol && fabs(z[3]) <= axis_tol;
    if (!origin_cell && fabs(r[axis_corner[0]]) <= axis_tol &&
        fabs(r[axis_corner[1]]) <= axis_tol) {
      // Conservative even-parity projection for each same-shell pair.
      projected_axis_wedge = true;
      for (int pair = 0; pair < 2; ++pair) {
        const int axis = axis_corner[pair];
        const int partner = partner_corner[pair];
        const double axis_mass = corner_mass[base + axis];
        const double partner_mass = corner_mass[base + partner];
        const double axis_volume = v_corner[axis];
        const double partner_volume = v_corner[partner];
        if (axis_volume <= 0.0 && axis_mass > 0.0) {
          printf("compatible_subzonal_pressure: positive axis-corner mass "
                 "with non-positive volume cell=%d corner=%d mass=%.17e "
                 "volume=%.17e\n",
                 c, axis, axis_mass, axis_volume);
          return;
        }
        const double pair_volume = axis_volume + partner_volume;
        if (!(isfinite(pair_volume) && pair_volume > kTiny)) {
          return;
        }
        const double pair_density =
            (axis_mass + partner_mass) / pair_volume;
        if (!isfinite(pair_density)) {
          return;
        }
        rho_corner[axis] = pair_density;
        rho_corner[partner] = pair_density;
      }
    }
  }
  for (int k = 0; k < 4; ++k) {
    if (!projected_axis_wedge) {
      const double v = v_corner[k];
      if (!(isfinite(v) && v > kTiny)) {
        return;
      }
      rho_corner[k] = fmax(corner_mass[base + k], 0.0) / v;
    }
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
  if constexpr (CaptureProjectionDebug) {
    if (theta0_axis_wedge || theta_pi_axis_wedge) {
      for (int pair = 0; pair < 2; ++pair) {
        projection_debug.rho_projected[2 * c + pair] =
            rho_corner[axis_corner[pair]];
      }
      projection_debug.merit[c] = merit;
      projection_debug.x_merit[c] = x_merit;
    }
  }
  if constexpr (ShadowMode == SubzonalShadowMode::AxisSource) {
    dp[partner_corner[0]] = 0.0;
    dp[partner_corner[1]] = 0.0;
  } else if constexpr (ShadowMode == SubzonalShadowMode::OffaxisSource) {
    dp[axis_corner[0]] = 0.0;
    dp[axis_corner[1]] = 0.0;
  }

  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  double s_r[4] = {0.0, 0.0, 0.0, 0.0};
  double s_z[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const double rm = 0.5 * (r[k] + r[kp]);
    const double zm = 0.5 * (z[k] + z[kp]);
    if (aw_planar) {
      median_vector_planar(rc, zc, rm, zm, &s_r[k], &s_z[k]);
    } else {
      median_vector_rz(rc, zc, rm, zm, &s_r[k], &s_z[k]);
    }
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
  if constexpr (ShadowMode == SubzonalShadowMode::Production) {
    if (force_scale > 0.0 && residual > 1.0e-12 * force_scale) {
      printf("compatible_subzonal_pressure: momentum correction cell=%d rel=%.17e\n",
             c, residual / force_scale);
    }
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

template <int SlotCap = mesh::kMeshTopoCellStorageSlotsMax>
__global__ void compute_subzonal_pressure_dt_kernel(
    double* min_dt,
    int* __restrict__ winner_cell,
    const double target_dt,
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
    const int corner_stride,
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
  if (active_nverts > SlotCap) {
#ifdef __CUDA_ARCH__
    __trap();  // cell exceeds slot cap
#endif
    return;
  }
  if (active_nverts < 3) {
    return;
  }
  int nodes[SlotCap] = {};
  if (multiblock && active_nverts == 3) {
    const int off = cell_node_csr_offsets[c];
    nodes[0] = cell_node_csr_indices[off + 0];
    nodes[1] = cell_node_csr_indices[off + 1];
    nodes[2] = cell_node_csr_indices[off + 2];
  } else if (multiblock && active_nverts == 5) {
    const int off = cell_node_csr_offsets[c];
    for (int k = 0; k < 5; ++k) {
      nodes[k] = cell_node_csr_indices[off + k];
    }
  } else if (multiblock && active_nverts >= 6 &&
             active_nverts <= SlotCap) {
    const int off = cell_node_csr_offsets[c];
    for (int k = 0; k < active_nverts; ++k) {
      nodes[k] = cell_node_csr_indices[off + k];
    }
  } else if (multiblock) {
    get_multiblock_cell_nodes(c, cell_node_csr_offsets, cell_node_csr_indices,
                              nodes);
  } else {
    get_structured_cell_nodes(c, nz, nodes);
  }

  double r[SlotCap] = {};
  double z[SlotCap] = {};
  for (int k = 0; k < active_nverts; ++k) {
    r[k] = x_r[nodes[k]];
    z[k] = x_z[nodes[k]];
  }

  double v_corner[SlotCap] = {};
  if (active_nverts == 3) {
    triangle_corner_volumes(r, z, v_corner);
  } else if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      x[k] = {r[k], z[k]};
    }
    pentagon_corner_rz_volumes(x, v_corner);
    validate_pentagon_corner_volumes(v_corner);
  } else if (active_nverts >= 6 && active_nverts <= SlotCap) {
    PentagonPoint x[SlotCap];
    for (int k = 0; k < active_nverts; ++k) {
      x[k] = {r[k], z[k]};
    }
    polygon_corner_rz_volumes_slot_cap<SlotCap>(x, active_nverts, v_corner);
    validate_polygon_corner_volumes_n(v_corner, active_nverts);
  } else {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], v_corner);
  }

  const double* cell_corner_mass = corner_mass + corner_stride * c;
  const double rho_c = fmax(rho[c], rho_floor);
  double rho_max = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    if (active_nverts >= 6 &&
        (!(v_corner[k] > 0.0) || !isfinite(v_corner[k]))) {
      continue;
    }
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

  double s_r[SlotCap] = {};
  double s_z[SlotCap] = {};
  double rc = 0.0;
  double zc = 0.0;
  if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      x[k] = {r[k], z[k]};
    }
    const PentagonPoint center = pentagon_center(x);
    rc = center.r;
    zc = center.z;
  } else {
    for (int k = 0; k < active_nverts; ++k) {
      rc += r[k] / static_cast<double>(active_nverts);
      zc += z[k] / static_cast<double>(active_nverts);
    }
  }
  for (int k = 0; k < active_nverts; ++k) {
    const int kp = (k + 1) % active_nverts;
    median_vector_rz(rc, zc, 0.5 * (r[k] + r[kp]),
                     0.5 * (z[k] + z[kp]), &s_r[k], &s_z[k]);
  }

  const double dp_drho = cs[c] * cs[c];
  double r_scale = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    r_scale = fmax(r_scale, fabs(r[k]));
  }
  const double axis_tol = core::axis_tolerance(r_scale);
  const double omega2 = subzonal_pressure_frequency_upper_bound<SlotCap>(
      v_corner, cell_corner_mass, s_r, s_z, active_nverts, dp_drho, merit, r,
      axis_tol, active_nverts >= 6);
  if (isinf(omega2)) {
    atomic_min_double(min_dt, 0.0);
    if (winner_cell != nullptr && target_dt == 0.0) {
      atomicMin(winner_cell, c);
    }
  } else if (omega2 > 0.0 && isfinite(omega2)) {
    const double dt = cfl_hydro / sqrt(omega2);
    atomic_min_double(min_dt, dt);
    if (winner_cell != nullptr && dt == target_dt) {
      atomicMin(winner_cell, c);
    }
  }
}

__global__ void compute_subzonal_buffer_diagnostics_kernel(
    SubzonalStepDiagnosticsDevice* diagnostics,
    const double* __restrict__ corner_force_sub_r,
    const double* __restrict__ corner_force_sub_z,
    const double* __restrict__ work_sub,
    const int n_cells,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  double max_abs_force = 0.0;
  for (int k = 0; k < corner_stride; ++k) {
    const int idx = corner_stride * c + k;
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

template <SubzonalShadowMode ShadowMode, bool CaptureProjectionDebug>
void launch_subzonal_pressure_kernel(
    double* const corner_force_sub_r,
    double* const corner_force_sub_z,
    double* const work_sub,
    double* const spread,
    double* const spread_corner,
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* const eos_ctx,
    const std::int8_t* const d_hydro_active,
    const double* const subzonal_band_weight,
    const std::uint8_t* const d_cell_nverts,
    const int* const cell_orientation_sign,
    const int n_cells,
    const bool multiblock,
    const double gamma,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    int* const an_degenerate_count,
    SubzonalStepDiagnosticsDevice* const diagnostics,
    const SubzonalPressureProjectionDebugBuffers projection_debug) {
  const int blocks = (n_cells + 255) / 256;
  if (state.corner_stride <= mesh::kMeshTopoCellStorageSlotsMax) {
    compute_subzonal_pressure_kernel<ShadowMode, CaptureProjectionDebug>
        <<<blocks, 256>>>(
            corner_force_sub_r,
            corner_force_sub_z,
            work_sub,
            spread,
            spread_corner,
            state.corner_mass.data(),
            state.rho.data(),
            state.ee.data(),
            state.ei.data(),
            state.Pe.data(),
            state.Pi.data(),
            state.vol.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.v_r.data(),
            state.v_z.data(),
            multiblock
                ? state.mesh.multiblock_cell_node_csr_offsets.data()
                : nullptr,
            multiblock
                ? state.mesh.multiblock_cell_node_csr_indices.data()
                : nullptr,
            multiblock ? cell_orientation_sign : nullptr,
            d_cell_nverts,
            d_hydro_active,
            subzonal_band_weight,
            n_cells,
            state.corner_stride,
            state.mesh.topo.nz,
            multiblock,
            mesh::mesh_topo_polar_tier_family(cfg.mesh),
            cfg.numerics.hydro.aw_compatible_force_work,
            aw_axis_slave_theta0_active,
            aw_axis_slave_theta_pi_active,
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
            cfg.numerics.hydro.pentagon_affine_null_enabled,
            cfg.numerics.hydro.pentagon_affine_null_kappa,
            an_degenerate_count,
            diagnostics,
            projection_debug);
  } else {
    compute_subzonal_pressure_kernel<
        ShadowMode, CaptureProjectionDebug,
        mesh::kMeshTopoCellStorageSlotsMaxGeneral><<<blocks, 256>>>(
        corner_force_sub_r,
        corner_force_sub_z,
        work_sub,
        spread,
        spread_corner,
        state.corner_mass.data(),
        state.rho.data(),
        state.ee.data(),
        state.ei.data(),
        state.Pe.data(),
        state.Pi.data(),
        state.vol.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.v_r.data(),
        state.v_z.data(),
        multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                   : nullptr,
        multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                   : nullptr,
        multiblock ? cell_orientation_sign : nullptr,
        d_cell_nverts,
        d_hydro_active,
        subzonal_band_weight,
        n_cells,
        state.corner_stride,
        state.mesh.topo.nz,
        multiblock,
        mesh::mesh_topo_polar_tier_family(cfg.mesh),
        cfg.numerics.hydro.aw_compatible_force_work,
        aw_axis_slave_theta0_active,
        aw_axis_slave_theta_pi_active,
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
        cfg.numerics.hydro.pentagon_affine_null_enabled,
        cfg.numerics.hydro.pentagon_affine_null_kappa,
        an_degenerate_count,
        diagnostics,
        projection_debug);
  }
}

void ensure_subzonal_buffers(core::State& state, const std::size_t n_cells) {
  const std::size_t n_corners =
      n_cells * static_cast<std::size_t>(state.corner_stride);
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
    const std::int8_t* d_hydro_active,
    SubzonalPressureDtArgmin* argmin) {
  if (argmin != nullptr) {
    *argmin = SubzonalPressureDtArgmin{};
  }
  if (!cfg.numerics.hydro.subzonal_pressure_enabled || cfg.main.dim != 2 ||
      state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(state.corner_mass.size() == n_corners &&
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
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  const int blocks = (n_cells + 255) / 256;
  if (state.corner_stride <= mesh::kMeshTopoCellStorageSlotsMax) {
    compute_subzonal_pressure_dt_kernel<<<blocks, 256>>>(
        d_min_dt.data(), nullptr, 0.0, state.corner_mass.data(), state.rho.data(),
        state.cs.data(), state.x_r.data(), state.x_z.data(),
        multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                   : nullptr,
        multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                   : nullptr,
        d_cell_nverts_ptr, d_hydro_active, n_cells, state.corner_stride,
        state.mesh.topo.nz, multiblock, cfg.numerics.floors.rho,
        cfg.numerics.dt.cfl_hydro, merit_mode_id(cfg),
        cfg.numerics.hydro.subzonal_alpha1,
        cfg.numerics.hydro.subzonal_alpha2,
        cfg.numerics.hydro.subzonal_merit_power,
        cfg.numerics.hydro.subzonal_merit_constant);
  } else {
    compute_subzonal_pressure_dt_kernel<
        mesh::kMeshTopoCellStorageSlotsMaxGeneral><<<blocks, 256>>>(
        d_min_dt.data(), nullptr, 0.0, state.corner_mass.data(), state.rho.data(),
        state.cs.data(), state.x_r.data(), state.x_z.data(),
        multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                   : nullptr,
        multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                   : nullptr,
        d_cell_nverts_ptr, d_hydro_active, n_cells, state.corner_stride,
        state.mesh.topo.nz, multiblock, cfg.numerics.floors.rho,
        cfg.numerics.dt.cfl_hydro, merit_mode_id(cfg),
        cfg.numerics.hydro.subzonal_alpha1,
        cfg.numerics.hydro.subzonal_alpha2,
        cfg.numerics.hydro.subzonal_merit_power,
        cfg.numerics.hydro.subzonal_merit_constant);
  }
  sync_kernel("Hydro2D subzonal pressure dt kernel failed");

  double dt = inf;
  cuda_check(cudaMemcpy(&dt, d_min_dt.data(), sizeof(double),
                        cudaMemcpyDeviceToHost),
             "subzonal pressure dt copy failed");
  if (argmin != nullptr && std::isfinite(dt)) {
    core::DeviceArray<int> d_winner_cell;
    d_winner_cell.reset(1U);
    cuda_check(cudaMemcpy(d_winner_cell.data(), &n_cells, sizeof(int),
                          cudaMemcpyHostToDevice),
               "subzonal pressure dt init winner cell failed");
    if (state.corner_stride <= mesh::kMeshTopoCellStorageSlotsMax) {
      compute_subzonal_pressure_dt_kernel<<<blocks, 256>>>(
          d_min_dt.data(), d_winner_cell.data(), dt, state.corner_mass.data(),
          state.rho.data(), state.cs.data(), state.x_r.data(), state.x_z.data(),
          multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                     : nullptr,
          multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                     : nullptr,
          d_cell_nverts_ptr, d_hydro_active, n_cells, state.corner_stride,
          state.mesh.topo.nz, multiblock, cfg.numerics.floors.rho,
          cfg.numerics.dt.cfl_hydro, merit_mode_id(cfg),
          cfg.numerics.hydro.subzonal_alpha1,
          cfg.numerics.hydro.subzonal_alpha2,
          cfg.numerics.hydro.subzonal_merit_power,
          cfg.numerics.hydro.subzonal_merit_constant);
    } else {
      compute_subzonal_pressure_dt_kernel<
          mesh::kMeshTopoCellStorageSlotsMaxGeneral><<<blocks, 256>>>(
          d_min_dt.data(), d_winner_cell.data(), dt, state.corner_mass.data(),
          state.rho.data(), state.cs.data(), state.x_r.data(), state.x_z.data(),
          multiblock ? state.mesh.multiblock_cell_node_csr_offsets.data()
                     : nullptr,
          multiblock ? state.mesh.multiblock_cell_node_csr_indices.data()
                     : nullptr,
          d_cell_nverts_ptr, d_hydro_active, n_cells, state.corner_stride,
          state.mesh.topo.nz, multiblock, cfg.numerics.floors.rho,
          cfg.numerics.dt.cfl_hydro, merit_mode_id(cfg),
          cfg.numerics.hydro.subzonal_alpha1,
          cfg.numerics.hydro.subzonal_alpha2,
          cfg.numerics.hydro.subzonal_merit_power,
          cfg.numerics.hydro.subzonal_merit_constant);
    }
    sync_kernel("Hydro2D subzonal pressure dt winner kernel failed");
    int winner_cell = n_cells;
    cuda_check(cudaMemcpy(&winner_cell, d_winner_cell.data(), sizeof(int),
                          cudaMemcpyDeviceToHost),
               "subzonal pressure dt copy winner cell failed");
    if (winner_cell >= 0 && winner_cell < n_cells) {
      argmin->cell_id = winner_cell;
      argmin->dt = dt;
    }
  }
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
      state.corner_force_sub_r.size() ==
              static_cast<std::size_t>(n_cells) *
                  static_cast<std::size_t>(state.corner_stride) &&
          state.corner_force_sub_z.size() ==
              static_cast<std::size_t>(n_cells) *
                  static_cast<std::size_t>(state.corner_stride) &&
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
      n_cells, state.corner_stride);
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
    const std::int8_t* d_hydro_active,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    const SubzonalPressureProjectionDebugBuffers* const projection_debug) {
  if (!cfg.numerics.hydro.subzonal_pressure_enabled || cfg.main.dim != 2) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    state.max_corner_density_spread_step = 0.0;
    state.max_subzonal_merit_step = 0.0;
    state.subzonal_nonzero_force_cells_step = 0;
    state.max_abs_subzonal_force_step = 0.0;
    state.max_abs_subzonal_work_step = 0.0;
    return;
  }
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(state.corner_mass.size() == n_corners &&
                    state.corner_mass_initialized,
                "compatible subzonal pressure requires initialized invariant corner_mass");
  TENRYU_ASSERT(state.ee.size() == static_cast<std::size_t>(n_cells) &&
                    state.ei.size() == static_cast<std::size_t>(n_cells) &&
                    state.Pe.size() == static_cast<std::size_t>(n_cells) &&
                    state.Pi.size() == static_cast<std::size_t>(n_cells) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells),
                "compatible subzonal pressure requires EOS closure fields");

  ensure_subzonal_buffers(state, static_cast<std::size_t>(n_cells));
  zero_array(state.corner_force_sub_r.data(), n_corners,
             "compatible subzonal pressure zero corner_force_sub_r failed");
  zero_array(state.corner_force_sub_z.data(), n_corners,
             "compatible subzonal pressure zero corner_force_sub_z failed");
  zero_array(state.work_sub_per_cell.data(), static_cast<std::size_t>(n_cells),
             "compatible subzonal pressure zero work_sub_per_cell failed");
  core::CellField1D spread{"compatible_subzonal_pressure:spread"};
  core::CellField1D spread_corner{
      "compatible_subzonal_pressure:spread_corner"};
  spread.reset(static_cast<std::size_t>(n_cells));
  spread_corner.reset(static_cast<std::size_t>(n_cells));
  core::DeviceArray<SubzonalStepDiagnosticsDevice> d_diagnostics{
      "compatible_subzonal_pressure:diagnostics"};
  d_diagnostics.reset(1U);
  cuda_check(cudaMemset(d_diagnostics.data(), 0,
                        sizeof(SubzonalStepDiagnosticsDevice)),
             "compatible subzonal diagnostics zero failed");
  int* d_an_degenerate_count = nullptr;
  d_an_degenerate_count = static_cast<int*>(core::device_scratch_acquire(
      "compatible_subzonal_pressure:an_degenerate_count", sizeof(int)));
  cuda_check(cudaMemset(d_an_degenerate_count, 0, sizeof(int)),
             "compatible subzonal pressure: zero affine-null degenerate "
             "count failed");

  const bool multiblock = state.mesh.topo.multiblock.has_value();
  core::DeviceArray<int> d_cell_orientation_sign{
      "compatible_subzonal_pressure:cell_orientation_sign"};
  core::DeviceArray<std::uint8_t> d_cell_nverts{
      "compatible_subzonal_pressure:cell_nverts"};
  const std::uint8_t* d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  const int* cell_orientation_sign = nullptr;
  if (multiblock) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "compatible subzonal pressure requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      n_corners,
                  "compatible subzonal pressure requires configured CSR nodes per cell");
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
  const SubzonalPressureProjectionDebugBuffers no_projection_debug{};
  if (projection_debug != nullptr) {
    TENRYU_ASSERT(
        projection_debug->rho_axis_raw != nullptr &&
            projection_debug->rho_partner_raw != nullptr &&
            projection_debug->rho_projected != nullptr &&
            projection_debug->merit != nullptr &&
            projection_debug->x_merit != nullptr,
        "compatible subzonal pressure projection debug buffers are incomplete");
    launch_subzonal_pressure_kernel<SubzonalShadowMode::Production, true>(
        state.corner_force_sub_r.data(),
        state.corner_force_sub_z.data(),
        state.work_sub_per_cell.data(),
        spread.data(),
        spread_corner.data(),
        state,
        cfg,
        eos_ctx,
        d_hydro_active,
        subzonal_band_weight,
        d_cell_nverts_ptr,
        cell_orientation_sign,
        n_cells,
        multiblock,
        gamma,
        aw_axis_slave_theta0_active,
        aw_axis_slave_theta_pi_active,
        d_an_degenerate_count,
        d_diagnostics.data(),
        *projection_debug);
  } else {
    launch_subzonal_pressure_kernel<SubzonalShadowMode::Production, false>(
        state.corner_force_sub_r.data(),
        state.corner_force_sub_z.data(),
        state.work_sub_per_cell.data(),
        spread.data(),
        spread_corner.data(),
        state,
        cfg,
        eos_ctx,
        d_hydro_active,
        subzonal_band_weight,
        d_cell_nverts_ptr,
        cell_orientation_sign,
        n_cells,
        multiblock,
        gamma,
        aw_axis_slave_theta0_active,
        aw_axis_slave_theta_pi_active,
        d_an_degenerate_count,
        d_diagnostics.data(),
        no_projection_debug);
  }
  sync_kernel("Hydro2D compatible subzonal pressure kernel failed");
  int an_degenerate_count = 0;
  cuda_check(cudaMemcpy(&an_degenerate_count, d_an_degenerate_count,
                        sizeof(int), cudaMemcpyDeviceToHost),
             "compatible subzonal pressure: copy affine-null degenerate "
             "count failed");
  if (an_degenerate_count > 0) {
    core::log_info("[pentagon_an] an_degenerate_count=" +
                   std::to_string(an_degenerate_count));
  }
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

SubzonalPressureShadowReplay
compute_compatible_subzonal_pressure_shadow_replay_2d(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const std::int8_t* d_hydro_active,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active) {
  SubzonalPressureShadowReplay replay;
  const int n_cells = static_cast<int>(state.rho.size());
  const std::size_t n_corners =
      static_cast<std::size_t>(std::max(n_cells, 0)) *
      static_cast<std::size_t>(state.corner_stride);
  replay.axis_source_force_r.assign(n_corners, 0.0);
  replay.axis_source_force_z.assign(n_corners, 0.0);
  replay.offaxis_source_force_r.assign(n_corners, 0.0);
  replay.offaxis_source_force_z.assign(n_corners, 0.0);
  if (!cfg.numerics.hydro.subzonal_pressure_enabled || cfg.main.dim != 2 ||
      !cfg.numerics.hydro.aw_compatible_force_work ||
      state.mesh.topo.multiblock.has_value() || n_cells <= 0 ||
      n_cells != state.mesh.topo.nr * state.mesh.topo.nz) {
    return replay;
  }

  core::DeviceArray<double> d_force_r(n_corners);
  core::DeviceArray<double> d_force_z(n_corners);
  core::DeviceArray<double> d_work(static_cast<std::size_t>(n_cells));
  core::DeviceArray<double> d_spread(static_cast<std::size_t>(n_cells));
  core::DeviceArray<double> d_spread_corner(
      static_cast<std::size_t>(n_cells));
  core::DeviceArray<std::uint8_t> d_cell_nverts;
  const std::uint8_t* const d_cell_nverts_ptr =
      upload_cell_nverts_if_nonquad(d_cell_nverts, state, n_cells);
  const double gamma =
      cfg.materials.materials.empty()
          ? 5.0 / 3.0
          : std::max(cfg.materials.materials.front().ideal_gas_gamma,
                     1.0 + 1.0e-12);
  const double* const subzonal_band_weight =
      ensure_bridge_band_subzonal_weights_2d(state, cfg);
  const SubzonalPressureProjectionDebugBuffers no_projection_debug{};

  launch_subzonal_pressure_kernel<SubzonalShadowMode::AxisSource, false>(
      d_force_r.data(),
      d_force_z.data(),
      d_work.data(),
      d_spread.data(),
      d_spread_corner.data(),
      state,
      cfg,
      eos_ctx,
      d_hydro_active,
      subzonal_band_weight,
      d_cell_nverts_ptr,
      nullptr,
      n_cells,
      false,
      gamma,
      aw_axis_slave_theta0_active,
      aw_axis_slave_theta_pi_active,
      nullptr,
      nullptr,
      no_projection_debug);
  sync_kernel("Hydro2D compatible subzonal axis-source shadow kernel failed");
  d_force_r.copy_to_host(replay.axis_source_force_r);
  d_force_z.copy_to_host(replay.axis_source_force_z);

  launch_subzonal_pressure_kernel<SubzonalShadowMode::OffaxisSource, false>(
      d_force_r.data(),
      d_force_z.data(),
      d_work.data(),
      d_spread.data(),
      d_spread_corner.data(),
      state,
      cfg,
      eos_ctx,
      d_hydro_active,
      subzonal_band_weight,
      d_cell_nverts_ptr,
      nullptr,
      n_cells,
      false,
      gamma,
      aw_axis_slave_theta0_active,
      aw_axis_slave_theta_pi_active,
      nullptr,
      nullptr,
      no_projection_debug);
  sync_kernel(
      "Hydro2D compatible subzonal offaxis-source shadow kernel failed");
  d_force_r.copy_to_host(replay.offaxis_source_force_r);
  d_force_z.copy_to_host(replay.offaxis_source_force_z);
  return replay;
}

}  // namespace tenryu::hydro
