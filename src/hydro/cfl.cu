#include "hydro/cfl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/boundary_pressure_force.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/compatible_av_csw.cuh"
#include "hydro/compatible_subzonal_pressure.cuh"
#include "hydro/corner_j_predict_cfl.hpp"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/pole_axis_bbsw.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/geometry_1d.cuh"
#include "mesh/mesh.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro {
namespace {

constexpr double kCflSensorEps = 1.0e-30;
constexpr int kVolumeRateCflLogLimitPerStep = 1;
constexpr std::uint8_t kCflCellStructured = 0U;
constexpr std::uint8_t kCflCellButton = 1U;
constexpr std::uint8_t kCflCellDormant = 2U;
constexpr double kFourPiOverThree =
    4.1887902047863909846168578443726705122628925325000;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

bool pole_axis_bbsw_enabled(const core::Config& cfg) {
  static const char* const env = std::getenv(pole_axis_bbsw::kEnvEnable);
  static const bool env_present = env != nullptr && env[0] != '\0';
  static const bool env_enabled = env_present && std::string(env) != "0";
  return env_present ? env_enabled : cfg.numerics.ale.pole_axis_bbsw_enabled;
}

int button_outer_node_ring_or_zero(const mesh::Mesh& mesh) {
  return (mesh.button_center && mesh.button_center->enabled)
             ? mesh.button_center->outer_node_ring
             : 0;
}

std::vector<std::uint8_t> build_button_cfl_cell_kind(const mesh::Mesh& mesh) {
  const int outer = button_outer_node_ring_or_zero(mesh);
  if (outer <= 0 || mesh.topo.n_cells <= 0) {
    return {};
  }
  std::vector<std::uint8_t> cell_kind(static_cast<std::size_t>(mesh.topo.n_cells),
                                      kCflCellStructured);
  for (int c = 0; c < mesh.topo.n_cells; ++c) {
    if (mesh.is_button_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kCflCellButton;
    } else if (mesh.is_dormant_cell(c)) {
      cell_kind[static_cast<std::size_t>(c)] = kCflCellDormant;
    }
  }
  return cell_kind;
}

std::uint8_t* upload_button_cfl_cell_kind_if_needed(const char* pool_tag,
                                                    const mesh::Mesh& mesh) {
  const std::vector<std::uint8_t> cell_kind = build_button_cfl_cell_kind(mesh);
  if (cell_kind.empty()) {
    return nullptr;
  }
  std::uint8_t* d_cell_kind = nullptr;
  if (pool_tag != nullptr) {
    d_cell_kind = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        pool_tag, cell_kind.size() * sizeof(std::uint8_t)));
  } else {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_kind),
                          cell_kind.size() * sizeof(std::uint8_t)),
               "CFL: cudaMalloc button cell_kind failed");
  }
  cuda_check(cudaMemcpy(d_cell_kind,
                        cell_kind.data(),
                        cell_kind.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "CFL: cudaMemcpy button cell_kind failed");
  return d_cell_kind;
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
    const mesh::Mesh& mesh,
    const int n_cells) {
  if (!has_nonquad_cell_nverts(mesh.cell_nverts, n_cells)) {
    return nullptr;
  }
  d_cell_nverts.reset(static_cast<std::size_t>(n_cells));
  d_cell_nverts.copy_from_host(mesh.cell_nverts);
  return d_cell_nverts.data();
}

__host__ __device__ inline double button_polygon_characteristic_length_from_nodes(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int outer_ring,
    const int nz) {
  if (x_r == nullptr || x_z == nullptr || outer_ring <= 0 || nz < 2) {
    return 0.0;
  }
  const int nverts = nz + 1;
  const double area2 = fabs(rz::button_polygon_area2_from_nodes(
      x_r, x_z, outer_ring, nz));
  const double area = 0.5 * area2;
  double perimeter = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = rz::button_seam_node_index(outer_ring, k, nz);
    const int np = rz::button_seam_node_index(outer_ring, kp, nz);
    perimeter += hypot(x_r[np] - x_r[n], x_z[np] - x_z[n]);
  }
  if (!(area > 0.0) || !(perimeter > 0.0) ||
      !isfinite(area) || !isfinite(perimeter)) {
    return 0.0;
  }

  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz::button_polygon_area_centroid_from_nodes(
      x_r, x_z, outer_ring, nz, &centroid_r, &centroid_z);
  if (!isfinite(centroid_r) || !isfinite(centroid_z)) {
    return 0.0;
  }

  double min_altitude = INFINITY;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    const int n = rz::button_seam_node_index(outer_ring, k, nz);
    const int np = rz::button_seam_node_index(outer_ring, kp, nz);
    const double ar = x_r[n];
    const double az = x_z[n];
    const double br = x_r[np];
    const double bz = x_z[np];
    const double ab = hypot(br - ar, bz - az);
    const double bc = hypot(centroid_r - br, centroid_z - bz);
    const double ca = hypot(ar - centroid_r, az - centroid_z);
    const double tri_area2 =
        fabs((br - ar) * (centroid_z - az) -
             (bz - az) * (centroid_r - ar));
    if (!(tri_area2 > 0.0) || !(ab > 0.0) || !(bc > 0.0) || !(ca > 0.0) ||
        !isfinite(tri_area2) || !isfinite(ab) || !isfinite(bc) ||
        !isfinite(ca)) {
      return 0.0;
    }
    min_altitude = fmin(min_altitude, tri_area2 / ab);
    min_altitude = fmin(min_altitude, tri_area2 / bc);
    min_altitude = fmin(min_altitude, tri_area2 / ca);
  }

  const double h_2ap = 2.0 * area / perimeter;
  const double h = fmin(h_2ap, min_altitude);
  return (h > 0.0 && isfinite(h)) ? h : 0.0;
}

__host__ __device__ inline double button_average_node_field(
    const double* __restrict__ field,
    const int outer_ring,
    const int nz) {
  if (field == nullptr || outer_ring <= 0 || nz < 0) {
    return 0.0;
  }
  const int nverts = nz + 1;
  double sum = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = rz::button_seam_node_index(outer_ring, k, nz);
    sum += field[n];
  }
  return sum / static_cast<double>(nverts);
}

__device__ __forceinline__ double compute_node_sigma_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int j,
    const int n_nodes,
    const double J) {
  double r_left = 0.0;
  double u_left = 0.0;
  double r_right = 0.0;
  double u_right = 0.0;
  if (j == 0) {
    r_left = -node_r[1];
    u_left = -node_u[1];
  } else {
    r_left = node_r[j - 1];
    u_left = node_u[j - 1];
  }
  if (j == n_nodes - 1) {
    r_right = 2.0 * node_r[j] - node_r[j - 1];
    u_right = 2.0 * node_u[j] - node_u[j - 1];
  } else {
    r_right = node_r[j + 1];
    u_right = node_u[j + 1];
  }

  const double r_j = node_r[j];
  const double u_j = node_u[j];
  const double dr_left = r_j - r_left;
  const double dr_right = r_right - r_j;
  if (dr_left <= 0.0 || dr_right <= 0.0) {
    return 0.0;
  }

  const double SL = J * (u_j - u_left) / dr_left;
  const double SR = J * (u_right - u_j) / dr_right;
  if (SL * SR <= 0.0) {
    return 0.0;
  }

  const double SC = ((dr_right / dr_left) * (u_j - u_left) +
                     (dr_left / dr_right) * (u_right - u_j)) /
                    (r_right - r_left);
  return copysign(fmin(fmin(fabs(SL), fabs(SC)), fabs(SR)), SL);
}

__device__ __forceinline__ double compute_chi_1d(const double* __restrict__ node_r,
                                                 const double* __restrict__ node_u,
                                                 const int i,
                                                 const int n_cells,
                                                 const double J) {
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  if (dr <= 0.0) {
    return 0.0;
  }

  const int n_nodes = n_cells + 1;
  const double sigma_left = compute_node_sigma_1d(node_r, node_u, i, n_nodes, J);
  const double sigma_right = compute_node_sigma_1d(node_r, node_u, i + 1, n_nodes, J);
  const double u_right = node_u[i + 1] - 0.5 * dr * sigma_right;
  const double u_left = node_u[i] + 0.5 * dr * sigma_left;
  const double du_lim = u_right - u_left;
  return fmax(0.0, -du_lim / dr);
}

__device__ __forceinline__ bool cfl_cell_active_1d(
    const std::int8_t* __restrict__ hydro_active,
    const int i) {
  return hydro_active == nullptr || hydro_active[i] != 0;
}

__device__ __forceinline__ double cfl_minmod(const double a, const double b) {
  if (a * b <= 0.0) {
    return 0.0;
  }
  return copysign(fmin(fabs(a), fabs(b)), a);
}

__device__ __forceinline__ double cfl_cell_center_r_1d(
    const double* __restrict__ node_r,
    const int i) {
  return 0.5 * (node_r[i] + node_r[i + 1]);
}

__device__ __forceinline__ double cfl_cell_velocity_1d(
    const double* __restrict__ node_u,
    const int i) {
  return 0.5 * (node_u[i] + node_u[i + 1]);
}

__device__ __forceinline__ double cfl_riemann_velocity_slope_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int i,
    const int n_cells) {
  const double r_i = cfl_cell_center_r_1d(node_r, i);
  const double u_i = cfl_cell_velocity_1d(node_u, i);
  bool has_left = (i > 0) && cfl_cell_active_1d(hydro_active, i - 1);
  bool has_right = (i + 1 < n_cells) && cfl_cell_active_1d(hydro_active, i + 1);
  double s_left = 0.0;
  double s_right = 0.0;
  if (has_left) {
    const double r_left = cfl_cell_center_r_1d(node_r, i - 1);
    const double dr_left = r_i - r_left;
    if (dr_left > 0.0) {
      s_left = (u_i - cfl_cell_velocity_1d(node_u, i - 1)) / dr_left;
    } else {
      has_left = false;
    }
  }
  if (has_right) {
    const double r_right = cfl_cell_center_r_1d(node_r, i + 1);
    const double dr_right = r_right - r_i;
    if (dr_right > 0.0) {
      s_right = (cfl_cell_velocity_1d(node_u, i + 1) - u_i) / dr_right;
    } else {
      has_right = false;
    }
  }
  if (has_left && has_right) {
    return cfl_minmod(s_left, s_right);
  }
  if (has_left) {
    return s_left;
  }
  if (has_right) {
    return s_right;
  }
  return 0.0;
}

__device__ __forceinline__ double cfl_reconstruct_riemann_velocity_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int cell,
    const int n_cells,
    const double r_face) {
  const double r_cell = cfl_cell_center_r_1d(node_r, cell);
  const double u_cell = cfl_cell_velocity_1d(node_u, cell);
  const double slope =
      cfl_riemann_velocity_slope_1d(node_r, node_u, hydro_active, cell, n_cells);
  return u_cell + slope * (r_face - r_cell);
}

__device__ __forceinline__ double cfl_riemann_face_du_plus_1d(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int face,
    const int n_cells) {
  if (face < 0 || face > n_cells) {
    return 0.0;
  }
  if (face == n_cells) {
    return 0.0;
  }

  int left_cell = face - 1;
  const int right_cell = face;
  bool center_mirror = false;
  if (face == 0) {
    left_cell = 0;
    center_mirror = true;
  }
  if (!cfl_cell_active_1d(hydro_active, right_cell) ||
      (!center_mirror && !cfl_cell_active_1d(hydro_active, left_cell))) {
    return 0.0;
  }

  const double r_face = node_r[face];
  const double u_R = cfl_reconstruct_riemann_velocity_1d(
      node_r, node_u, hydro_active, right_cell, n_cells, r_face);
  const double u_L =
      center_mirror
          ? -u_R
          : cfl_reconstruct_riemann_velocity_1d(
                node_r, node_u, hydro_active, left_cell, n_cells, r_face);
  const double du = u_L - u_R;
  return (du > 0.0 && isfinite(du)) ? du : 0.0;
}

__device__ double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__global__ void rz_cfl_compute_node_activity_2d_kernel(
    std::uint8_t* __restrict__ node_active,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  node_active[i * stride + j] = 1u;
  node_active[(i + 1) * stride + j] = 1u;
  node_active[(i + 1) * stride + (j + 1)] = 1u;
  node_active[i * stride + (j + 1)] = 1u;
}

__global__ void rz_cfl_compute_corner_mass_2d_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const double m_cell = mass[c];
  const double R_L = 0.5 * (x_r[n00] + x_r[n01]);
  const double R_R = 0.5 * (x_r[n10] + x_r[n11]);
  const double denom = (R_L + R_R) * 6.0;
  if (denom <= 0.0) {
    const double m_quarter = 0.25 * m_cell;
    corner_mass[c * 4 + 0] = m_quarter;
    corner_mass[c * 4 + 1] = m_quarter;
    corner_mass[c * 4 + 2] = m_quarter;
    corner_mass[c * 4 + 3] = m_quarter;
    return;
  }
  const double w_L = (2.0 * R_L + R_R) / denom;
  const double w_R = (R_L + 2.0 * R_R) / denom;
  corner_mass[c * 4 + 0] = w_L * m_cell;
  corner_mass[c * 4 + 1] = w_R * m_cell;
  corner_mass[c * 4 + 2] = w_R * m_cell;
  corner_mass[c * 4 + 3] = w_L * m_cell;
}

__global__ void rz_cfl_compute_node_mass_2d_kernel(
    double* __restrict__ node_mass,
    const double* __restrict__ corner_mass,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  atomic_add_double(node_mass + i * stride + j, corner_mass[c * 4 + 0]);
  atomic_add_double(node_mass + (i + 1) * stride + j, corner_mass[c * 4 + 1]);
  atomic_add_double(node_mass + (i + 1) * stride + (j + 1),
                    corner_mass[c * 4 + 2]);
  atomic_add_double(node_mass + i * stride + (j + 1), corner_mass[c * 4 + 3]);
}

__global__ void rz_cfl_build_cell_pq_kernel(double* __restrict__ pq,
                                            const double* __restrict__ Pe,
                                            const double* __restrict__ Pi,
                                            const double* __restrict__ Qvisc,
                                            const std::int8_t* __restrict__ hydro_active,
                                            const int c_begin,
                                            const int c_end,
                                            const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = hydro_active == nullptr || hydro_active[c] != 0;
  pq[c] = active ? (Pe[c] + Pi[c] + Qvisc[c]) : 0.0;
}

__global__ void rz_cfl_compute_force_from_cells_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ cell_pq,
    const double* __restrict__ Svec_r,
    const double* __restrict__ Svec_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end || (hydro_active != nullptr && hydro_active[c] == 0)) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const int idx = c * 4;
  const double pq = cell_pq[c];
  atomic_add_double(force_r + n00, +pq * Svec_r[idx + 0]);
  atomic_add_double(force_z + n00, +pq * Svec_z[idx + 0]);
  atomic_add_double(force_r + n10, +pq * Svec_r[idx + 1]);
  atomic_add_double(force_z + n10, +pq * Svec_z[idx + 1]);
  atomic_add_double(force_r + n11, +pq * Svec_r[idx + 2]);
  atomic_add_double(force_z + n11, +pq * Svec_z[idx + 2]);
  atomic_add_double(force_r + n01, +pq * Svec_r[idx + 3]);
  atomic_add_double(force_z + n01, +pq * Svec_z[idx + 3]);
}

__global__ void rz_cfl_estimate_u_half_kernel(
    double* __restrict__ u_half_r,
    double* __restrict__ u_half_z,
    const double* __restrict__ old_v_r,
    const double* __restrict__ old_v_z,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ node_mass,
    const std::uint8_t* __restrict__ node_active,
    const int n_begin,
    const int n_end,
    const int n_nodes,
    const int nr,
    const int nz,
    const int r_outer_type,
    const int z_bottom_type,
    const int z_top_type,
    const double half_dt) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  double accel_r = 0.0;
  double accel_z = 0.0;
  if (node_active[n] != 0u && node_mass[n] > 0.0) {
    accel_r = force_r[n] / node_mass[n];
    accel_z = force_z[n] / node_mass[n];
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;
  if (i == 0) {
    accel_r = 0.0;
  }
  if (i == nr) {
    if (r_outer_type == static_cast<int>(Boundary2DType::FIXED)) {
      accel_r = 0.0;
      accel_z = 0.0;
    } else if (r_outer_type == static_cast<int>(Boundary2DType::REFLECT)) {
      accel_r = 0.0;
    }
  }
  if (j == 0) {
    if (z_bottom_type == static_cast<int>(Boundary2DType::FIXED)) {
      accel_r = 0.0;
      accel_z = 0.0;
    } else if (z_bottom_type == static_cast<int>(Boundary2DType::REFLECT) ||
               z_bottom_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
      accel_z = 0.0;
    }
  }
  if (j == nz) {
    if (z_top_type == static_cast<int>(Boundary2DType::FIXED)) {
      accel_r = 0.0;
      accel_z = 0.0;
    } else if (z_top_type == static_cast<int>(Boundary2DType::REFLECT) ||
               z_top_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
      accel_z = 0.0;
    }
  }
  if (i == 0) {
    accel_r = 0.0;
  }

  if (node_active[n] != 0u) {
    u_half_r[n] = old_v_r[n] + half_dt * accel_r;
    u_half_z[n] = old_v_z[n] + half_dt * accel_z;
  } else {
    u_half_r[n] = old_v_r[n];
    u_half_z[n] = old_v_z[n];
  }
}

__device__ __forceinline__ bool cfl_1d_cell_active(
    const std::int8_t* __restrict__ hydro_active,
    const int c,
    const int n_cells) {
  return c >= 0 && c < n_cells &&
         (hydro_active == nullptr || hydro_active[c] != 0);
}

__device__ __forceinline__ double cfl_1d_cell_cs(
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ cs_arr,
    const int c,
    const double gamma) {
  if (cs_arr != nullptr) {
    return fmax(cs_arr[c], 0.0);
  }
  const double e_e = ee[c] > 0.0 ? ee[c] : 0.0;
  const double e_i = (ei != nullptr && ei[c] > 0.0) ? ei[c] : 0.0;
  return fmax(sqrt(gamma * (gamma - 1.0) * (e_e + e_i)), 0.0);
}

__device__ __forceinline__ double cfl_1d_face_area(const int geom_code,
                                                   const double r) {
  // geometry_1d_face_area already carries the canonical spherical kFourPi
  // literal, so every geometry goes through the shared helper.
  return tenryu::mesh::geometry_1d_face_area(geom_code, r);
}

__device__ inline void cfl_1d_kernel_body(
    const int i,
    const int /*tid*/,
    double* __restrict__ /*shared*/,
    double* __restrict__ min_dt,
    double* __restrict__ min_dt_post_shock,
    double* __restrict__ min_dt_crossing,
    double* __restrict__ min_dt_art_heat,
    int* __restrict__ have_active,
    const double* __restrict__ x_r,
    const double* __restrict__ v_r,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ cs_arr,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ shock_time,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int geom_code,
    const double t_current,
    const double gamma,
    const double c1,
    const double c2,
    const int av_du_mode,
    const int post_shock_heat_enabled,
    const double post_shock_heat_c,
    const double post_shock_heat_decay,
    const double crossing_dt_safety,
    const int art_heat_bound_enabled,
    const double art_heat_c,
    const double J) {
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return;
  }

  atomicExch(have_active, 1);

  const double dr = x_r[i + 1] - x_r[i];
  if (dr <= 0.0) {
    return;
  }

  // 2026-07-26 review: node-crossing guard. The acoustic+AV
  // denominator can vanish for cold, limiter-smooth compression (cs -> 0,
  // chi -> 0) while a face still closes at u_i - u_{i+1} > 0; bound the step
  // so one face sweep stays below crossing_dt_safety of the cell width.
  // Raw geometric bound: the caller combines it un-scaled with the
  // cfl_hydro-scaled acoustic candidate.
  if (min_dt_crossing != nullptr && crossing_dt_safety > 0.0) {
    const double closing = v_r[i] - v_r[i + 1];
    if (closing > 0.0) {
      atomic_min_double(min_dt_crossing, crossing_dt_safety * dr / closing);
    }
  }

  const double cs = cfl_1d_cell_cs(ee, ei, cs_arr, i, gamma);
  double denom = cs;
  double chi = 0.0;
  if (av_du_mode == 1) {
    const double du_left =
        cfl_riemann_face_du_plus_1d(x_r, v_r, hydro_active, i, n_cells);
    const double du_right =
        cfl_riemann_face_du_plus_1d(x_r, v_r, hydro_active, i + 1, n_cells);
    denom += fmax(du_left, du_right);
  } else if (av_du_mode == 2) {
    // riemann_compatible (v1.1): raw nodal compressive jump — an upper
    // scale for the BJ-limited projected jump the operator uses;
    // conservative and mirror-free (consult 20260803-1442 §2/§9).
    denom += fmax(0.0, v_r[i] - v_r[i + 1]);
  } else {
    // Lagrangian CFL: mesh moves with fluid, only acoustic speed matters.
    // Add AV linear/quadratic corrections for shock-capture stability.
    chi = compute_chi_1d(x_r, v_r, i, n_cells, J);
    const double compression_speed = dr * chi;
    denom += c1 * cs + c2 * compression_speed;
  }
  if (denom > 0.0) {
    atomic_min_double(min_dt, dr / denom);
  }

  const double m_i = (rho != nullptr && vol != nullptr)
                         ? fmax(rho[i], 0.0) * fmax(vol[i], 0.0)
                         : 0.0;

  // 2026-07-26 review: explicit stability bound for
  // the artificial heat flux H (VNR chi only; H is disabled under riemann and
  // csw uses its own chi_lim <= raw compression, left on the legacy margin).
  // Face conductance G_f = A_f C_H rho_f l_f chi_f (flux = -C_H rho l^2 chi
  // grad e, l_f = dr_centers cancels one power), row-sum dt <= 0.5 m_i / sum G.
  if (art_heat_bound_enabled != 0 && min_dt_art_heat != nullptr &&
      av_du_mode == 0 && art_heat_c > 0.0 && m_i > 0.0) {
    double g_sum = 0.0;
    for (int side = 0; side < 2; ++side) {
      const int nb = (side == 0) ? i - 1 : i + 1;
      if (!cfl_1d_cell_active(hydro_active, nb, n_cells)) {
        continue;
      }
      const double dr_nb = x_r[nb + 1] - x_r[nb];
      if (!(dr_nb > 0.0)) {
        continue;
      }
      const double chi_nb = compute_chi_1d(x_r, v_r, nb, n_cells, J);
      const double chi_f = fmax(chi, chi_nb);
      if (!(chi_f > 0.0)) {
        continue;
      }
      const double rho_f = 0.5 * (fmax(rho[i], 0.0) + fmax(rho[nb], 0.0));
      const double l_f = 0.5 * (dr + dr_nb);
      const double r_face = (side == 0) ? x_r[i] : x_r[i + 1];
      const double area_f = cfl_1d_face_area(geom_code, r_face);
      g_sum += area_f * art_heat_c * rho_f * l_f * chi_f;
    }
    if (g_sum > 0.0) {
      atomic_min_double(min_dt_art_heat, 0.5 * m_i / g_sum);
    }
  }

  // 2026-07-26 review: post-shock heat bound as the exact
  // row-sum of the operator's face conductances G_f = A_f C_ps rho_f cs_f
  // psi_f (the operator's flux is -C_ps rho_f cs_f psi_f (e_R - e_L); the
  // face length cancels), replacing the former dr-based estimate that
  // under-restricted graded meshes and density-jump faces.
  if (post_shock_heat_enabled != 0 && min_dt_post_shock != nullptr &&
      shock_time != nullptr && post_shock_heat_c > 0.0 && m_i > 0.0 &&
      cs > 0.0) {
    const double tau_i = post_shock_heat_decay * dr / fmax(cs, kCflSensorEps);
    const double psi_i =
        (tau_i > 0.0)
            ? exp(-fmax(t_current - shock_time[i], 0.0) / tau_i)
            : 0.0;
    double g_sum = 0.0;
    for (int side = 0; side < 2; ++side) {
      const int nb = (side == 0) ? i - 1 : i + 1;
      if (!cfl_1d_cell_active(hydro_active, nb, n_cells)) {
        continue;
      }
      const double dr_nb = x_r[nb + 1] - x_r[nb];
      if (!(dr_nb > 0.0)) {
        continue;
      }
      const double cs_nb = cfl_1d_cell_cs(ee, ei, cs_arr, nb, gamma);
      const double tau_nb =
          post_shock_heat_decay * dr_nb / fmax(cs_nb, kCflSensorEps);
      const double psi_nb =
          (tau_nb > 0.0)
              ? exp(-fmax(t_current - shock_time[nb], 0.0) / tau_nb)
              : 0.0;
      const double psi_f = 0.5 * (psi_i + psi_nb);
      if (!(psi_f > 0.0)) {
        continue;
      }
      const double rho_f = 0.5 * (fmax(rho[i], 0.0) + fmax(rho[nb], 0.0));
      const double cs_f = 0.5 * (cs + cs_nb);
      const double r_face = (side == 0) ? x_r[i] : x_r[i + 1];
      const double area_f = cfl_1d_face_area(geom_code, r_face);
      g_sum += area_f * post_shock_heat_c * rho_f * cs_f * psi_f;
    }
    if (g_sum > 0.0) {
      atomic_min_double(min_dt_post_shock, 0.5 * m_i / g_sum);
    }
  }
}

__global__ void cfl_1d_kernel(double* __restrict__ min_dt,
                              double* __restrict__ min_dt_post_shock,
                              double* __restrict__ min_dt_crossing,
                              double* __restrict__ min_dt_art_heat,
                              int* __restrict__ have_active,
                              const double* __restrict__ x_r,
                              const double* __restrict__ v_r,
                              const double* __restrict__ ee,
                              const double* __restrict__ ei,
                              const double* __restrict__ cs_arr,
                              const double* __restrict__ rho,
                              const double* __restrict__ vol,
                              const double* __restrict__ shock_time,
                              const std::int8_t* __restrict__ hydro_active,
                              const int c_begin,
                              const int c_end,
                              const int n_cells,
                              const int geom_code,
                              const double t_current,
                              const double gamma,
                              const double c1,
                              const double c2,
                              const int av_du_mode,
                              const int post_shock_heat_enabled,
                              const double post_shock_heat_c,
                              const double post_shock_heat_decay,
                              const double crossing_dt_safety,
                              const int art_heat_bound_enabled,
                              const double art_heat_c,
                              const double J) {
  const int i = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ double shared[];
  if (i >= c_end) {
    return;
  }

  cfl_1d_kernel_body(i, tid, shared, min_dt, min_dt_post_shock,
                     min_dt_crossing, min_dt_art_heat, have_active,
                     x_r, v_r, ee, ei, cs_arr, rho, vol, shock_time,
                     hydro_active, n_cells, geom_code, t_current, gamma, c1,
                     c2, av_du_mode, post_shock_heat_enabled,
                     post_shock_heat_c, post_shock_heat_decay,
                     crossing_dt_safety, art_heat_bound_enabled, art_heat_c,
                     J);
}

__global__ void cfl_2d_kernel(double* __restrict__ min_dt,
                              int* __restrict__ have_active,
                              const double* __restrict__ x_r,
                              const double* __restrict__ x_z,
                              const double* __restrict__ v_r,
                              const double* __restrict__ v_z,
                              const double* __restrict__ ee,
                              const double* __restrict__ ei,
                              const double* __restrict__ cs_arr,
                              const double* __restrict__ cell_area,
                              const std::uint8_t* __restrict__ cell_kind,
                              const std::int8_t* __restrict__ hydro_active,
                              const std::uint8_t* __restrict__ pseudo_core_member,
                              const int c_begin,
                              const int c_end,
                              const int n_cells,
                              const int nz,
                              const int button_outer_node_ring,
                              const double gamma,
                              const double c1) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const std::uint8_t kind =
      (cell_kind != nullptr) ? cell_kind[c] : kCflCellStructured;
  if (kind == kCflCellDormant ||
      (hydro_active != nullptr && hydro_active[c] == 0) ||
      (pseudo_core_member != nullptr && pseudo_core_member[c] != 0U)) {
    return;
  }

  atomicExch(have_active, 1);

  double dl = 0.0;
  if (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0) {
    dl = button_polygon_characteristic_length_from_nodes(
        x_r, x_z, button_outer_node_ring, nz);
  } else {
    const double area = cell_area[c];
    if (area <= 0.0) {
      return;
    }
    dl = sqrt(area);
  }
  if (!(dl > 0.0) || !isfinite(dl)) {
    return;
  }

  // Retained in signature for launch-site compatibility.
  (void)v_r;
  (void)v_z;

  const double e_e = ee[c] > 0.0 ? ee[c] : 0.0;
  const double e_i = (ei != nullptr && ei[c] > 0.0) ? ei[c] : 0.0;
  const double cs_val =
      (cs_arr != nullptr) ? cs_arr[c] : sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
  const double cs = fmax(cs_val, 0.0);
  // Lagrangian CFL: mesh moves with fluid; include AV linear correction for
  // logical cells.  The button polygon has no quad-direction AV length, so it
  // contributes the acoustic h_c/c_s limit here; scalar VNR Q remains in the
  // hydro pressure/divergence path.
  const double denom =
      (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0)
          ? cs
          : cs * (1.0 + c1);
  if (denom <= 0.0) {
    return;
  }

  const double dt = dl / denom;
  atomic_min_double(min_dt, dt);
}

__global__ void cfl_2d_winner_kernel(int* __restrict__ winner_cell,
                                     const double min_local,
                                     const double* __restrict__ x_r,
                                     const double* __restrict__ x_z,
                                     const double* __restrict__ ee,
                                     const double* __restrict__ ei,
                                     const double* __restrict__ cs_arr,
                                     const double* __restrict__ cell_area,
                                     const std::uint8_t* __restrict__ cell_kind,
                                     const std::int8_t* __restrict__ hydro_active,
                                     const std::uint8_t* __restrict__ pseudo_core_member,
                                     const int c_begin,
                                     const int c_end,
                                     const int n_cells,
                                     const int nz,
                                     const int button_outer_node_ring,
                                     const double gamma,
                                     const double c1) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const std::uint8_t kind =
      (cell_kind != nullptr) ? cell_kind[c] : kCflCellStructured;
  if (kind == kCflCellDormant ||
      (hydro_active != nullptr && hydro_active[c] == 0) ||
      (pseudo_core_member != nullptr && pseudo_core_member[c] != 0U)) {
    return;
  }

  double dl = 0.0;
  if (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0) {
    dl = button_polygon_characteristic_length_from_nodes(
        x_r, x_z, button_outer_node_ring, nz);
  } else {
    const double area = cell_area[c];
    if (!(area > 0.0) || !isfinite(area)) {
      return;
    }
    dl = sqrt(area);
  }
  if (!(dl > 0.0) || !isfinite(dl)) {
    return;
  }
  const double e_e = ee[c] > 0.0 ? ee[c] : 0.0;
  const double e_i = (ei != nullptr && ei[c] > 0.0) ? ei[c] : 0.0;
  const double cs_val =
      (cs_arr != nullptr) ? cs_arr[c] : sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
  const double cs = fmax(cs_val, 0.0);
  const double denom =
      (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0)
          ? cs
          : cs * (1.0 + c1);
  if (!(denom > 0.0) || !isfinite(denom)) {
    return;
  }

  const double local_dt = dl / denom;
  const double tol = fmax(1.0e-12 * fmax(fabs(min_local), 1.0e-300), 1.0e-300);
  if (fabs(local_dt - min_local) > tol) {
    return;
  }

  atomicMin(winner_cell, c);
}

__global__ void cfl_2d_winner_values_kernel(double* __restrict__ winner_values,
                                            const int c,
                                            const double cfl_hydro,
                                            const double* __restrict__ x_r,
                                            const double* __restrict__ x_z,
                                            const double* __restrict__ v_z,
                                            const double* __restrict__ ee,
                                            const double* __restrict__ ei,
                                            const double* __restrict__ cs_arr,
                                            const double* __restrict__ rho,
                                            const double* __restrict__ cell_area,
                                            const std::uint8_t* __restrict__ cell_kind,
                                            const int* __restrict__ cell_node_csr_offsets,
                                            const int* __restrict__ cell_node_csr_indices,
                                            const std::uint8_t* __restrict__ cell_nverts,
                                            const int n_cells,
                                            const int nz,
                                            const int button_outer_node_ring,
                                            const double gamma,
                                            const double c1) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  if (c < 0 || c >= n_cells) {
    return;
  }

  const std::uint8_t kind =
      (cell_kind != nullptr) ? cell_kind[c] : kCflCellStructured;
  if (kind == kCflCellDormant) {
    return;
  }
  double dl = 0.0;
  if (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0) {
    dl = button_polygon_characteristic_length_from_nodes(
        x_r, x_z, button_outer_node_ring, nz);
  } else {
    const double area = cell_area[c];
    if (!(area > 0.0) || !isfinite(area)) {
      return;
    }
    dl = sqrt(area);
  }
  if (!(dl > 0.0) || !isfinite(dl)) {
    return;
  }
  const double e_e = ee[c] > 0.0 ? ee[c] : 0.0;
  const double e_i = (ei != nullptr && ei[c] > 0.0) ? ei[c] : 0.0;
  const double cs_val =
      (cs_arr != nullptr) ? cs_arr[c] : sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
  const double cs = fmax(cs_val, 0.0);
  const double denom =
      (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0)
          ? cs
          : cs * (1.0 + c1);
  if (!(denom > 0.0) || !isfinite(denom)) {
    return;
  }

  double u_z = 0.0;
  if (kind == kCflCellButton && button_outer_node_ring > 0 && c == 0) {
    u_z = button_average_node_field(v_z, button_outer_node_ring, nz);
  } else {
    int n00 = -1;
    int n10 = -1;
    int n11 = -1;
    int n01 = -1;
    if (cell_node_csr_offsets != nullptr && cell_node_csr_indices != nullptr) {
      const int off = cell_node_csr_offsets[c];
      const int active_nverts =
          mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      if (active_nverts == 3) {
        n00 = cell_node_csr_indices[off + 0];
        n10 = cell_node_csr_indices[off + 1];
        n11 = cell_node_csr_indices[off + 2];
        u_z = (v_z[n00] + v_z[n10] + v_z[n11]) / 3.0;
      } else if (active_nverts >= 5) {
        for (int k = 0; k < active_nverts; ++k) {
          u_z += v_z[cell_node_csr_indices[off + k]];
        }
        u_z /= static_cast<double>(active_nverts);
      } else {
        n00 = cell_node_csr_indices[off + 0];
        n10 = cell_node_csr_indices[off + 1];
        n11 = cell_node_csr_indices[off + 2];
        n01 = cell_node_csr_indices[off + 3];
        u_z = 0.25 * (v_z[n00] + v_z[n10] + v_z[n11] + v_z[n01]);
      }
    } else {
      const int i = c / nz;
      const int j = c - i * nz;
      const int stride = nz + 1;
      n00 = i * stride + j;
      n10 = (i + 1) * stride + j;
      n11 = (i + 1) * stride + (j + 1);
      n01 = i * stride + (j + 1);
      u_z = 0.25 * (v_z[n00] + v_z[n10] + v_z[n11] + v_z[n01]);
    }
  }
  winner_values[0] = cfl_hydro * (dl / denom);
  winner_values[1] = dl;
  winner_values[2] = cs;
  winner_values[3] = rho[c];
  winner_values[4] = u_z;
}

__device__ __forceinline__ double cfl_axis_cell_margin(
    const double r_outer_j,
    const double z_outer_j,
    const double r_outer_jp1,
    const double z_outer_jp1,
    const double z_axis_j,
    const double z_axis_jp1) {
  const double s = z_axis_jp1 - z_axis_j;
  if (s <= 0.0 || r_outer_j <= 0.0 || r_outer_jp1 <= 0.0) {
    return -1.0;
  }
  const double Q = r_outer_j * (z_outer_jp1 - z_axis_jp1) -
                   r_outer_jp1 * (z_outer_j - z_axis_j);
  return s * fmin(r_outer_j, r_outer_jp1) + fmin(Q, 0.0);
}

__global__ void axis_margin_cfl_dt_kernel(double* __restrict__ min_scale,
                                          const double* __restrict__ x_r,
                                          const double* __restrict__ x_z,
                                          const double* __restrict__ v_r,
                                          const double* __restrict__ v_z,
                                          const int nz,
                                          const double dt_proposed,
                                          const double floor_fraction) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis_j = j;
  const int n_axis_jp1 = j + 1;
  const int n_outer_j = stride + j;
  const int n_outer_jp1 = stride + j + 1;

  const double current_margin =
      cfl_axis_cell_margin(x_r[n_outer_j], x_z[n_outer_j],
                           x_r[n_outer_jp1], x_z[n_outer_jp1],
                           x_z[n_axis_j], x_z[n_axis_jp1]);
  if (!(current_margin > 0.0) || !isfinite(current_margin)) {
    atomic_min_double(min_scale, 0.0);
    return;
  }
  const double margin_floor = floor_fraction * current_margin;

  const double r_outer_j_full = x_r[n_outer_j] + dt_proposed * v_r[n_outer_j];
  const double z_outer_j_full = x_z[n_outer_j] + dt_proposed * v_z[n_outer_j];
  const double r_outer_jp1_full =
      x_r[n_outer_jp1] + dt_proposed * v_r[n_outer_jp1];
  const double z_outer_jp1_full =
      x_z[n_outer_jp1] + dt_proposed * v_z[n_outer_jp1];
  const double z_axis_j_full = x_z[n_axis_j] + dt_proposed * v_z[n_axis_j];
  const double z_axis_jp1_full =
      x_z[n_axis_jp1] + dt_proposed * v_z[n_axis_jp1];
  const double margin_full =
      cfl_axis_cell_margin(r_outer_j_full, z_outer_j_full,
                           r_outer_jp1_full, z_outer_jp1_full,
                           z_axis_j_full, z_axis_jp1_full);
  if (margin_full >= margin_floor) {
    return;
  }

  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < 48; ++iter) {
    const double scale = 0.5 * (lo + hi);
    const double tau = scale * dt_proposed;
    const double margin_mid =
        cfl_axis_cell_margin(x_r[n_outer_j] + tau * v_r[n_outer_j],
                             x_z[n_outer_j] + tau * v_z[n_outer_j],
                             x_r[n_outer_jp1] + tau * v_r[n_outer_jp1],
                             x_z[n_outer_jp1] + tau * v_z[n_outer_jp1],
                             x_z[n_axis_j] + tau * v_z[n_axis_j],
                             x_z[n_axis_jp1] + tau * v_z[n_axis_jp1]);
    if (margin_mid >= margin_floor) {
      lo = scale;
    } else {
      hi = scale;
    }
  }
  atomic_min_double(min_scale, lo);
}

__global__ void volume_rate_cfl_min_kernel(double* __restrict__ min_dt,
                                           const double* __restrict__ vol_new,
                                           const double* __restrict__ vol_old,
                                           const std::uint8_t* __restrict__ inactive_cell_mask,
                                           const int c_begin,
                                           const int c_end,
                                           const int n_cells,
                                           const double dt_used_prev,
                                           const double threshold) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (inactive_cell_mask != nullptr && inactive_cell_mask[c] != 0U) {
    return;
  }

  const double denom = fmax(vol_old[c], kCflSensorEps);
  const double frac_rate = fabs(vol_new[c] - vol_old[c]) / denom;
  if (!(frac_rate > 0.0) || !isfinite(frac_rate)) {
    return;
  }

  const double dt_limit = threshold * dt_used_prev / frac_rate;
  if (dt_limit > 0.0 && isfinite(dt_limit)) {
    atomic_min_double(min_dt, dt_limit);
  }
}

__global__ void volume_rate_cfl_detail_kernel(int* __restrict__ min_cell,
                                              double* __restrict__ min_frac_rate,
                                              const double* __restrict__ vol_new,
                                              const double* __restrict__ vol_old,
                                              const std::uint8_t* __restrict__ inactive_cell_mask,
                                              const int c_begin,
                                              const int c_end,
                                              const int n_cells,
                                              const double dt_used_prev,
                                              const double threshold,
                                              const double min_dt_limit) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (inactive_cell_mask != nullptr && inactive_cell_mask[c] != 0U) {
    return;
  }

  const double denom = fmax(vol_old[c], kCflSensorEps);
  const double frac_rate = fabs(vol_new[c] - vol_old[c]) / denom;
  if (!(frac_rate > 0.0) || !isfinite(frac_rate)) {
    return;
  }

  const double dt_limit = threshold * dt_used_prev / frac_rate;
  const double tol = fmax(1.0e-30, 1.0e-12 * fabs(min_dt_limit));
  if (!(dt_limit > 0.0) || !isfinite(dt_limit) ||
      fabs(dt_limit - min_dt_limit) > tol) {
    return;
  }

  const int old = atomicMin(min_cell, c);
  if (c < old) {
    *min_frac_rate = frac_rate;
  }
}

double copy_last_min_dt(double* d_min_dt) {
  double min_dt = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(&min_dt, d_min_dt, sizeof(double), cudaMemcpyDeviceToHost),
             "CFL: memcpy min_dt failed");
  return min_dt;
}

std::string format_cfl_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

void log_volume_rate_cfl_clamp(const core::State& state,
                               const double dt_in,
                               const double dt_out,
                               const int cell,
                               const double frac_rate,
                               const double threshold) {
  static int logged_step = std::numeric_limits<int>::min();
  static int logs_this_step = 0;
  if (state.step != logged_step) {
    logged_step = state.step;
    logs_this_step = 0;
  }
  if (logs_this_step >= kVolumeRateCflLogLimitPerStep) {
    return;
  }
  ++logs_this_step;

  core::log_info("[hydro-stats] volume_rate_cfl: dt clamped from " +
                 format_cfl_scientific(dt_in) + " to " +
                 format_cfl_scientific(dt_out) + " (cell " +
                 std::to_string(cell) + ", frac_rate=" +
                 format_cfl_scientific(frac_rate) + ", threshold=" +
                 format_cfl_scientific(threshold) + ")");
}

std::vector<std::int8_t> make_cfl_central_macro_effective_active(
    const core::State& state,
    const core::Config& cfg) {
  if (!central_pseudo_core::configured(cfg) &&
      !pole_angular_derefine::configured(cfg)) {
    return {};
  }
  core::State& mutable_state = const_cast<core::State&>(state);
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(mutable_state, cfg);
  }
  pole_angular_derefine::ensure_built(mutable_state, cfg);
  const int n_cells = static_cast<int>(state.rho.size());
  if (!central_pseudo_core::active(mutable_state) &&
      !pole_angular_derefine::active(mutable_state)) {
    return {};
  }
  std::vector<std::int8_t> active(static_cast<std::size_t>(n_cells), 1);
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "CFL central macro hydro_active size mismatch");
    active = state.hydro_active;
  }
  const auto apply_inactive = [&](const std::vector<std::uint8_t>& inactive) {
    if (inactive.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (inactive[static_cast<std::size_t>(c)] != 0U) {
        active[static_cast<std::size_t>(c)] = 0;
      }
    }
  };
  apply_inactive(mutable_state.central_pseudo_core.inactive_member_mask);
  apply_inactive(mutable_state.pole_angular_derefine.inactive_member_mask);
  return active;
}

bool compute_precise_rz_geometric_u_half(const core::State& state,
                                         const core::Config& cfg,
                                         const double dt,
                                         core::NodeField1D& u_half_r,
                                         core::NodeField1D& u_half_z) {
  if (!cfg.numerics.hydro.rz_geometric_cfl_precise_u_half_enabled ||
      state.mesh.dim != 2 ||
      !(dt > 0.0) ||
      !std::isfinite(dt)) {
    return false;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    return false;
  }
  TENRYU_ASSERT(!cfg.numerics.hydro.hourglass.enabled &&
                    !cfg.numerics.hydro.subzonal_mass_enabled,
                "RZ geometric CFL precise u_half does not include hourglass acceleration");
  TENRYU_ASSERT(cfg.numerics.hydro.axis_motion_floor_fraction <= 0.0,
                "RZ geometric CFL precise u_half does not include axis-motion preflight scaling");
  TENRYU_ASSERT(cfg.numerics.ale.axis_z_motion != "lagrangian_tangential",
                "RZ geometric CFL precise u_half does not include lagrangian_tangential axis z projection");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "RZ geometric CFL precise u_half requires matching node position arrays");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "RZ geometric CFL precise u_half requires matching node velocity arrays");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "RZ geometric CFL precise u_half requires matching node arrays");
  TENRYU_ASSERT(state.Pe.size() == state.rho.size() &&
                    state.Pi.size() == state.rho.size() &&
                    state.Qvisc.size() == state.rho.size() &&
                    state.mass.size() == state.rho.size(),
                "RZ geometric CFL precise u_half requires hydro cell fields");
  TENRYU_ASSERT(state.mesh.cell_Svec_r.size() == state.rho.size() * 4U &&
                    state.mesh.cell_Svec_z.size() == state.rho.size() * 4U,
                "RZ geometric CFL precise u_half requires current mesh S-vectors");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    state.hydro_active.size() == state.rho.size(),
                "RZ geometric CFL precise u_half hydro_active size mismatch");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_cells <= 0 || n_nodes <= 0) {
    return false;
  }
  TENRYU_ASSERT(state.mesh.topo.n_cells == n_cells,
                "RZ geometric CFL precise u_half topo cell count mismatch");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == n_nodes,
                "RZ geometric CFL precise u_half topo node count mismatch");

  const std::vector<std::int8_t> effective_active =
      make_cfl_central_macro_effective_active(state, cfg);
  const std::vector<std::int8_t>& active_source =
      effective_active.empty() ? state.hydro_active : effective_active;
  std::int8_t* d_hydro_active = nullptr;
  if (!active_source.empty()) {
    d_hydro_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cfl:compute_precise_rz_geometric_u_half:d_hydro_active",
        active_source.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_hydro_active,
                          active_source.data(),
                          active_source.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "RZ geometric CFL precise u_half: copy hydro_active failed");
  }

  core::CellField1D svec_r;
  core::CellField1D svec_z;
  core::CellField1D corner_mass;
  core::CellField1D pq;
  core::NodeField1D force_r;
  core::NodeField1D force_z;
  core::NodeField1D node_mass;
  svec_r = state.mesh.cell_Svec_r;
  svec_z = state.mesh.cell_Svec_z;
  corner_mass.reset(static_cast<std::size_t>(n_cells) * 4U);
  pq.reset(n_cells);
  force_r.reset(n_nodes);
  force_z.reset(n_nodes);
  node_mass.reset(n_nodes);
  u_half_r.reset(n_nodes);
  u_half_z.reset(n_nodes);

  std::uint8_t* d_node_active = nullptr;
  d_node_active = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "cfl:compute_precise_rz_geometric_u_half:d_node_active",
      static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)));
  cuda_check(cudaMemset(d_node_active, 0,
                        static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)),
             "RZ geometric CFL precise u_half: memset node_active failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  // Cell->node scatters (mass/force) and their cell-domain producers must
  // cover the ghost i-planes so the shared interface node plane receives
  // contributions from BOTH adjacent cell planes (Option C; same class as
  // the compute_node_mass_2d fix — owned-only windows halve interface node
  // sums). Ghost inputs (mass, Pe, Pi, Qvisc, Svec) are halo-valid.
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, nz);
  rz_cfl_compute_node_activity_2d_kernel<<<cw.blocks(), 256>>>(
      d_node_active, d_hydro_active, cw.begin, cw.end, nr, nz);
  rz_cfl_compute_corner_mass_2d_kernel<<<fw.blocks(), 256>>>(
      corner_mass.data(), state.mass.data(), state.x_r.data(), fw.begin, fw.end,
      nr, nz);
  rz_cfl_compute_node_mass_2d_kernel<<<fw.blocks(), 256>>>(
      node_mass.data(), corner_mass.data(), fw.begin, fw.end, nr, nz);
  rz_cfl_build_cell_pq_kernel<<<fw.blocks(), 256>>>(
      pq.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
      d_hydro_active, fw.begin, fw.end, n_cells);
  rz_cfl_compute_force_from_cells_kernel<<<fw.blocks(), 256>>>(
      force_r.data(), force_z.data(), pq.data(), svec_r.data(), svec_z.data(),
      d_hydro_active, fw.begin, fw.end, nr, nz);

  const auto r_outer_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (r_outer_type == Boundary2DType::PRESSURE) {
    TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                  "RZ geometric CFL precise u_half pressure boundary requires pressure_drive_1d");
    const double p_ext = state.pressure_drive_1d->eval(state.t);
    const bool rz_exact_endpoint =
        state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      mesh::mesh_topo_assert_multiblock_polar_shell_outer_boundary(
          state.mesh.topo);
      detail::launch_multiblock_polar_shell_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          mesh::mesh_topo_multiblock_polar_shell_node_offset(state.mesh.topo),
          mesh::mesh_topo_multiblock_polar_shell_nr(state.mesh.topo),
          mesh::mesh_topo_multiblock_polar_shell_nz(state.mesh.topo), p_ext,
          rz_exact_endpoint, nullptr,
          cfg.numerics.hydro.rz_momentum_scheme_id);
    } else {
      detail::launch_r_outer_boundary_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          nr, nz, p_ext, rz_exact_endpoint,
          cfg.numerics.hydro.rz_momentum_scheme_id);
    }
  }

  rz_cfl_estimate_u_half_kernel<<<nw.blocks(), 256>>>(
      u_half_r.data(), u_half_z.data(), state.v_r.data(), state.v_z.data(),
      force_r.data(), force_z.data(), node_mass.data(), d_node_active, nw.begin,
      nw.end, n_nodes, nr, nz, static_cast<int>(r_outer_type),
      static_cast<int>(parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom)),
      static_cast<int>(parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top)),
      0.5 * dt);
  cuda_check(cudaGetLastError(),
             "RZ geometric CFL precise u_half kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "RZ geometric CFL precise u_half kernel execution failed");

  return true;
}

TriFanCenterCflInfo compute_tri_fan_center_cfl_dt_legacy(
    const core::State& state,
    const core::Config& cfg);
TriFanCenterCflInfo compute_tri_fan_center_cfl_dt(const core::State& state,
                                                  const core::Config& cfg);

double compute_axis_margin_cfl_dt(const core::State& state,
                                  const double dt_proposed,
                                  const double floor_fraction,
                                  const bool has_physical_rz_axis,
                                  const core::Config& cfg,
                                  TriFanCenterCflInfo* tri_fan_center_cfl) {
  if (state.mesh.dim != 2 || !(dt_proposed > 0.0) ||
      !std::isfinite(dt_proposed)) {
    return dt_proposed;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr < 1 || nz < 1) {
    return dt_proposed;
  }
  const bool has_tri_fan_center =
      tenryu::mesh::has_tri_fan_center_topology(state.mesh);
  const bool has_multiblock = state.mesh.topo.multiblock.has_value();
  if (cfg.numerics.hydro.center_cfl_scope != core::CenterCflScope::DISABLED) {
    if (tri_fan_center_cfl != nullptr) {
      *tri_fan_center_cfl = compute_tri_fan_center_cfl_dt(state, cfg);
      if (has_tri_fan_center || has_multiblock) {
        return std::min(dt_proposed, tri_fan_center_cfl->dt);
      }
    } else {
      const TriFanCenterCflInfo center_cfl =
          compute_tri_fan_center_cfl_dt(state, cfg);
      if (has_tri_fan_center || has_multiblock) {
        return std::min(dt_proposed, center_cfl.dt);
      }
    }
  }
  if (has_tri_fan_center || has_multiblock) {
    return dt_proposed;
  }

  if (!has_physical_rz_axis || floor_fraction <= 0.0) {
    return dt_proposed;
  }

  TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                "Axis-margin CFL requires matching node position arrays");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "Axis-margin CFL requires matching node velocity arrays");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "Axis-margin CFL requires topo/state node-size consistency");

  double* d_min_scale = nullptr;
  d_min_scale = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_axis_margin_cfl_dt:d_min_scale", sizeof(double)));
  const double one = 1.0;
  cuda_check(cudaMemcpy(d_min_scale, &one, sizeof(double), cudaMemcpyHostToDevice),
             "CFL: memcpy init axis min_scale failed");

  const int blocks = (nz + 255) / 256;
  axis_margin_cfl_dt_kernel<<<blocks, 256>>>(
      d_min_scale, state.x_r.data(), state.x_z.data(), state.v_r.data(),
      state.v_z.data(), nz, dt_proposed, floor_fraction);
  cuda_check(cudaGetLastError(), "CFL axis-margin kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "CFL axis-margin kernel execution failed");

  double min_scale = 1.0;
  cuda_check(cudaMemcpy(&min_scale, d_min_scale, sizeof(double), cudaMemcpyDeviceToHost),
             "CFL: memcpy axis min_scale failed");

  if (!std::isfinite(min_scale)) {
    return 0.0;
  }
  min_scale = std::max(0.0, std::min(1.0, min_scale));
  return dt_proposed * min_scale;
}

TriFanCenterCflInfo compute_tri_fan_center_cfl_dt(const core::State& state,
                                                  const core::Config& cfg) {
  const core::CenterCflScope scope = cfg.numerics.hydro.center_cfl_scope;
  if (scope == core::CenterCflScope::DISABLED) {
    return TriFanCenterCflInfo{};
  }
  if (scope == core::CenterCflScope::TRI_FAN_RADIAL_INDEX) {
    return compute_tri_fan_center_cfl_dt_legacy(state, cfg);
  }
  if (scope == core::CenterCflScope::CENTROID_R_LE_R_MATCH) {
    return compute_centroid_r_center_cfl_dt(state, cfg);
  }
  return TriFanCenterCflInfo{};
}

TriFanCenterCflInfo compute_tri_fan_center_cfl_dt_legacy(
    const core::State& state,
    const core::Config& cfg) {
  TriFanCenterCflInfo result;
  if (state.mesh.dim != 2 ||
      !tenryu::mesh::has_tri_fan_center_topology(state.mesh)) {
    return result;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr < 1 || nz < 1) {
    return result;
  }
  TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                "tri_fan center CFL requires matching node position arrays");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "tri_fan center CFL requires topo/state node-size consistency");
  TENRYU_ASSERT(state.Pe.size() == state.rho.size() &&
                    state.Pi.size() == state.rho.size() &&
                    state.Qvisc.size() == state.rho.size(),
                "tri_fan center CFL requires cell pressure, Qvisc, and rho fields");
  TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                "tri_fan center CFL requires topo/state cell-size consistency");

  const int max_i =
      std::min(cfg.numerics.hydro.tri_fan_center_cfl_band_radial_index, nr - 1);
  const int node_rows = max_i + 2;
  const int node_count = node_rows * (nz + 1);
  const int cell_count = (max_i + 1) * nz;
  std::vector<double> node_r(static_cast<std::size_t>(node_count), 0.0);
  std::vector<double> node_z(static_cast<std::size_t>(node_count), 0.0);
  std::vector<double> rho(static_cast<std::size_t>(cell_count), 0.0);
  std::vector<double> Pe(static_cast<std::size_t>(cell_count), 0.0);
  std::vector<double> Pi(static_cast<std::size_t>(cell_count), 0.0);
  std::vector<double> Qvisc(static_cast<std::size_t>(cell_count), 0.0);

  cuda_check(cudaMemcpy(node_r.data(),
                        state.x_r.data(),
                        node_r.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy node_r failed");
  cuda_check(cudaMemcpy(node_z.data(),
                        state.x_z.data(),
                        node_z.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy node_z failed");
  cuda_check(cudaMemcpy(rho.data(),
                        state.rho.data(),
                        rho.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy rho failed");
  cuda_check(cudaMemcpy(Pe.data(),
                        state.Pe.data(),
                        Pe.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy Pe failed");
  cuda_check(cudaMemcpy(Pi.data(),
                        state.Pi.data(),
                        Pi.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy Pi failed");
  cuda_check(cudaMemcpy(Qvisc.data(),
                        state.Qvisc.data(),
                        Qvisc.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "tri_fan center CFL: memcpy Qvisc failed");

  const auto node = [nz](const int i, const int j) {
    return static_cast<std::size_t>(i * (nz + 1) + j);
  };
  std::vector<double> theta_node(static_cast<std::size_t>(nz + 1), 0.0);
  for (int j = 0; j <= nz; ++j) {
    const std::size_t n = node(1, j);
    theta_node[static_cast<std::size_t>(j)] = std::atan2(node_r[n], node_z[n]);
  }
  const double safety = cfg.numerics.hydro.tri_fan_center_cfl_safety;
  for (int i = 0; i <= max_i; ++i) {
    const std::size_t n0 = node(i, 0);
    const std::size_t n1 = node(i + 1, 0);
    const double s0 = std::hypot(node_r[n0], node_z[n0]);
    const double s1 = std::hypot(node_r[n1], node_z[n1]);
    // For tri_fan, s_mid[0] = 0.5*s_node[1] > 0 even though the
    // pinned origin row has s_node[0] = 0.
    const double s_mid = 0.5 * (s0 + s1);
    if (!(s_mid > 0.0) || !std::isfinite(s_mid)) {
      continue;
    }
    for (int j = 0; j < nz; ++j) {
      const double dtheta = theta_node[static_cast<std::size_t>(j + 1)] -
                            theta_node[static_cast<std::size_t>(j)];
      if (!(dtheta > 0.0) || !std::isfinite(dtheta)) {
        continue;
      }
      const double h = s_mid * dtheta;
      if (!(h > 0.0) || !std::isfinite(h)) {
        continue;
      }
      const int c = i * nz + j;
      const std::size_t idx = static_cast<std::size_t>(c);
      const double p = Pe[idx] + Pi[idx];
      const double q = Qvisc[idx];
      const double rho_eff = std::max(rho[idx], kCflSensorEps);
      const double c_eff_sq_raw = (p + q) / rho_eff;
      if (std::isnan(c_eff_sq_raw)) {
        continue;
      }
      const double c_eff_sq = std::max(c_eff_sq_raw, 0.0);
      const double c_eff = std::sqrt(c_eff_sq);
      const double denom = std::max(c_eff, kCflSensorEps);
      const double dt_cell = safety * h / denom;
      if (!std::isfinite(dt_cell)) {
        continue;
      }
      if (dt_cell < result.dt) {
        result.dt = dt_cell;
        result.cell_id = c;
        result.i = i;
        result.j = j;
        result.h = h;
        result.c_eff = c_eff;
        if (p > 0.0) {
          result.q_over_p = q / p;
        } else if (q > 0.0) {
          result.q_over_p = std::numeric_limits<double>::infinity();
        } else {
          result.q_over_p = 0.0;
        }
      }
    }
  }
  return result;
}

}  // namespace

TriFanCenterCflInfo compute_centroid_r_center_cfl_dt(
    const core::State& state,
    const core::Config& cfg) {
  TriFanCenterCflInfo result;
  if (cfg.numerics.hydro.center_cfl_scope !=
          core::CenterCflScope::CENTROID_R_LE_R_MATCH ||
      state.mesh.dim != 2 ||
      !state.mesh.topo.multiblock.has_value()) {
    return result;
  }
  TENRYU_ASSERT(std::isfinite(cfg.mesh.multiblock_cart_core_r_match),
                "centroid-r center CFL requires finite multiblock r_match");
  TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                "centroid-r center CFL requires matching node position arrays");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "centroid-r center CFL requires matching node velocity arrays");
  TENRYU_ASSERT(state.Pe.size() == state.rho.size() &&
                    state.Pi.size() == state.rho.size() &&
                    state.Qvisc.size() == state.rho.size(),
                "centroid-r center CFL requires pressure, Qvisc, and rho fields");
  TENRYU_ASSERT(state.ee.size() == state.rho.size(),
                "centroid-r center CFL requires electron energy field");
  TENRYU_ASSERT(state.ei.empty() || state.ei.size() == state.rho.size(),
                "centroid-r center CFL ei size mismatch");
  TENRYU_ASSERT(state.cs.empty() || state.cs.size() == state.rho.size(),
                "centroid-r center CFL cs size mismatch");
  TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                "centroid-r center CFL requires topo/state cell-size consistency");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "centroid-r center CFL requires topo/state node-size consistency");
  TENRYU_ASSERT(state.mesh.cell_centroid_r.size() == state.rho.size(),
                "centroid-r center CFL requires cell centroid_r");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "centroid-r center CFL requires at least one material");

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells == 0 || !(cfg.numerics.hydro.tri_fan_center_cfl_safety > 0.0)) {
    return result;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "centroid-r center CFL requires multiblock cell-node CSR offsets");
  TENRYU_ASSERT(mb.cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) *
                        static_cast<std::size_t>(state.mesh.corner_stride),
                "centroid-r center CFL requires multiblock cell-node CSR indices");
  std::vector<double> node_r(state.x_r.size(), 0.0);
  std::vector<double> node_z(state.x_z.size(), 0.0);
  std::vector<double> v_r(state.v_r.size(), 0.0);
  std::vector<double> v_z(state.v_z.size(), 0.0);
  std::vector<double> rho(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> Pe(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> Pi(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> Qvisc(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> ee(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> ei;
  std::vector<double> cs;

  state.x_r.copy_to_host(node_r.data());
  state.x_z.copy_to_host(node_z.data());
  state.v_r.copy_to_host(v_r.data());
  state.v_z.copy_to_host(v_z.data());
  state.rho.copy_to_host(rho.data());
  state.Pe.copy_to_host(Pe.data());
  state.Pi.copy_to_host(Pi.data());
  state.Qvisc.copy_to_host(Qvisc.data());
  state.ee.copy_to_host(ee.data());
  if (!state.ei.empty()) {
    ei.assign(static_cast<std::size_t>(n_cells), 0.0);
    state.ei.copy_to_host(ei.data());
  }
  if (!state.cs.empty()) {
    cs.assign(static_cast<std::size_t>(n_cells), 0.0);
    state.cs.copy_to_host(cs.data());
  }

  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  const double safety = cfg.numerics.hydro.tri_fan_center_cfl_safety;
  const double r_match = cfg.mesh.multiblock_cart_core_r_match;
  const bool have_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t idx = static_cast<std::size_t>(c);
    const double centroid_r = state.mesh.cell_centroid_r[idx];
    if (!(centroid_r <= r_match) || !std::isfinite(centroid_r)) {
      continue;
    }

    const int off = mb.cell_node_csr_offsets[idx];
    std::array<double, mesh::kMeshTopoCellStorageSlotsMax> r = {};
    std::array<double, mesh::kMeshTopoCellStorageSlotsMax> z = {};
    std::array<int, mesh::kMeshTopoCellStorageSlotsMax> nodes;
    nodes.fill(-1);
    const int active_nverts =
        have_cell_nverts
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    if (active_nverts == 3) {
      for (int k = 0; k < 3; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < state.mesh.topo.n_nodes,
                      "centroid-r center CFL cell-node index out of range");
        nodes[k] = n;
        r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
        z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
      }
    } else if (active_nverts == 4) {
      for (int k = 0; k < 4; ++k) {
        const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < state.mesh.topo.n_nodes,
                      "centroid-r center CFL cell-node index out of range");
        nodes[k] = n;
        r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
        z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
      }
    } else {
      for (int k = 0; k < active_nverts; ++k) {
        const int n =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        TENRYU_ASSERT(n >= 0 && n < state.mesh.topo.n_nodes,
                      "centroid-r center CFL cell-node index out of range");
        nodes[static_cast<std::size_t>(k)] = n;
        r[static_cast<std::size_t>(k)] =
            node_r[static_cast<std::size_t>(n)];
        z[static_cast<std::size_t>(k)] =
            node_z[static_cast<std::size_t>(n)];
      }
    }

    double h_2ap = 0.0;
    double min_altitude = 0.0;
    const double h =
        detail::active_polygon_center_cfl_h(r, z, active_nverts,
                                            &h_2ap, &min_altitude);
    if (!(h > 0.0) || !std::isfinite(h)) {
      continue;
    }

    const double p = Pe[idx] + Pi[idx];
    const double rho_eff = std::max(rho[idx], kCflSensorEps);
    double c_eff = 0.0;
    if (!cs.empty()) {
      c_eff = std::max(cs[idx], 0.0);
    } else if (p > 0.0) {
      c_eff = std::sqrt(std::max(gamma * p / rho_eff, 0.0));
    } else {
      const double e_e = ee[idx] > 0.0 ? ee[idx] : 0.0;
      const double e_i = (!ei.empty() && ei[idx] > 0.0) ? ei[idx] : 0.0;
      c_eff = std::sqrt(std::max(gamma * (gamma - 1.0) * (e_e + e_i), 0.0));
    }
    if (std::isnan(c_eff)) {
      continue;
    }

    double u_r = 0.0;
    double u_z = 0.0;
    if (active_nverts == 3) {
      u_r = (v_r[static_cast<std::size_t>(nodes[0])] +
             v_r[static_cast<std::size_t>(nodes[1])] +
             v_r[static_cast<std::size_t>(nodes[2])]) /
            3.0;
      u_z = (v_z[static_cast<std::size_t>(nodes[0])] +
             v_z[static_cast<std::size_t>(nodes[1])] +
             v_z[static_cast<std::size_t>(nodes[2])]) /
            3.0;
    } else if (active_nverts == 4) {
      u_r = 0.25 * (v_r[static_cast<std::size_t>(nodes[0])] +
                    v_r[static_cast<std::size_t>(nodes[1])] +
                    v_r[static_cast<std::size_t>(nodes[2])] +
                    v_r[static_cast<std::size_t>(nodes[3])]);
      u_z = 0.25 * (v_z[static_cast<std::size_t>(nodes[0])] +
                    v_z[static_cast<std::size_t>(nodes[1])] +
                    v_z[static_cast<std::size_t>(nodes[2])] +
                    v_z[static_cast<std::size_t>(nodes[3])]);
    } else {
      for (int k = 0; k < active_nverts; ++k) {
        const std::size_t node =
            static_cast<std::size_t>(nodes[static_cast<std::size_t>(k)]);
        u_r += v_r[node];
        u_z += v_z[node];
      }
      u_r /= static_cast<double>(active_nverts);
      u_z /= static_cast<double>(active_nverts);
    }
    const double velocity_proxy = std::abs(u_r) + std::abs(u_z);
    const double denom = std::max({c_eff, velocity_proxy, kCflSensorEps});
    const double dt_cell = safety * h / denom;
    if (!std::isfinite(dt_cell)) {
      continue;
    }

    if (dt_cell < result.dt) {
      result.dt = dt_cell;
      result.cell_id = (mb.cell_id_stable.size() == static_cast<std::size_t>(n_cells))
                           ? mb.cell_id_stable[idx]
                           : c;
      result.i = -1;
      result.j = -1;
      result.h = h;
      result.h_2ap = h_2ap;
      result.min_altitude = min_altitude;
      result.c_eff = c_eff;
      result.nverts = active_nverts;
      for (int k = 0; k < mesh::kMeshTopoCellStorageSlotsMax; ++k) {
        result.r8[k] = r[static_cast<std::size_t>(k)];
        result.z8[k] = z[static_cast<std::size_t>(k)];
      }
      for (int k = 0; k < 4; ++k) {
        result.r4[k] = result.r8[k];
        result.z4[k] = result.z8[k];
      }
      result.block_id =
          (mb.cell_block_id.size() == static_cast<std::size_t>(n_cells))
              ? mb.cell_block_id[idx]
              : -1;
      const double q = Qvisc[idx];
      if (p > 0.0) {
        result.q_over_p = q / p;
      } else if (q > 0.0) {
        result.q_over_p = std::numeric_limits<double>::infinity();
      } else {
        result.q_over_p = 0.0;
      }
    }
  }
  if (cfg.numerics.debug.trace_mesh_motion && std::isfinite(result.dt) &&
      result.dt < 1.0e-15 && result.cell_id >= 0) {
    std::string vertices;
    for (int k = 0; k < result.nverts; ++k) {
      vertices += " r" + std::to_string(k) + "=" +
                  format_cfl_scientific(result.r8[k]) +
                  " z" + std::to_string(k) + "=" +
                  format_cfl_scientific(result.z8[k]);
    }
    core::log_warning("[center-cfl-sliver] cell=" +
                      std::to_string(result.cell_id) +
                      " block=" + std::to_string(result.block_id) +
                      " nverts=" + std::to_string(result.nverts) +
                      " dt=" + format_cfl_scientific(result.dt) +
                      " h=" + format_cfl_scientific(result.h) +
                      " h_2ap=" + format_cfl_scientific(result.h_2ap) +
                      " min_altitude=" +
                      format_cfl_scientific(result.min_altitude) +
                      " c_eff=" + format_cfl_scientific(result.c_eff) +
                      vertices);
  }
  return result;
}

double compute_volume_rate_cfl_dt(const core::State& state,
                                  const double dt_in,
                                  const double dt_used_prev,
                                  const double threshold,
                                  const parallel::Reduction* reduction) {
  if (state.mesh.dim != 2 || !(dt_in > 0.0) || !std::isfinite(dt_in) ||
      !(dt_used_prev > 0.0) || !std::isfinite(dt_used_prev)) {
    return dt_in;
  }
  TENRYU_ASSERT(threshold > 0.0,
                "Volume-rate CFL requires threshold > 0");
  TENRYU_ASSERT(state.vol.size() == state.rho.size(),
                "Volume-rate CFL requires volume/rho size consistency");
  TENRYU_ASSERT(state.vol_prev_hydro.size() == state.vol.size(),
                "Volume-rate CFL requires previous hydro volume snapshot");

  const int n_cells = static_cast<int>(state.vol.size());
  if (n_cells == 0) {
    return dt_in;
  }

  double* d_min_dt = nullptr;
  d_min_dt = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_volume_rate_cfl_dt:d_min_dt", sizeof(double)));
  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "CFL: memcpy init volume-rate min_dt failed");

  core::State& mutable_state = const_cast<core::State&>(state);
  core::DeviceArray<std::uint8_t> d_inactive_mask;
  const std::uint8_t* d_inactive_mask_ptr =
      pole_angular_derefine::combined_inactive_mask_device(
          mutable_state, d_inactive_mask);
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  volume_rate_cfl_min_kernel<<<cw.blocks(), 256>>>(
      d_min_dt, state.vol.data(), state.vol_prev_hydro.data(),
      d_inactive_mask_ptr, cw.begin, cw.end, n_cells,
      dt_used_prev, threshold);
  cuda_check(cudaGetLastError(), "CFL volume-rate kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "CFL volume-rate kernel execution failed");

  const double local_limit = copy_last_min_dt(d_min_dt);

  double global_limit = local_limit;
  if (reduction != nullptr) {
    global_limit = reduction->allreduce_min(global_limit);
  }
  if (!std::isfinite(global_limit)) {
    return dt_in;
  }

  const double dt_out = std::min(dt_in, global_limit);
  if (dt_out < dt_in && std::isfinite(local_limit) &&
      local_limit <= global_limit * (1.0 + 1.0e-12)) {
    int* d_min_cell = nullptr;
    double* d_min_frac_rate = nullptr;
    d_min_cell = static_cast<int*>(core::device_scratch_acquire(
        "cfl:compute_volume_rate_cfl_dt:d_min_cell", sizeof(int)));
    d_min_frac_rate = static_cast<double*>(core::device_scratch_acquire(
        "cfl:compute_volume_rate_cfl_dt:d_min_frac_rate", sizeof(double)));
    cuda_check(cudaMemcpy(d_min_cell, &n_cells, sizeof(int), cudaMemcpyHostToDevice),
               "CFL: init volume-rate min_cell failed");
    const double zero = 0.0;
    cuda_check(cudaMemcpy(d_min_frac_rate, &zero, sizeof(double), cudaMemcpyHostToDevice),
               "CFL: init volume-rate min_frac_rate failed");

    volume_rate_cfl_detail_kernel<<<cw.blocks(), 256>>>(
        d_min_cell, d_min_frac_rate, state.vol.data(), state.vol_prev_hydro.data(),
        d_inactive_mask_ptr, cw.begin, cw.end, n_cells, dt_used_prev, threshold,
        local_limit);
    cuda_check(cudaGetLastError(), "CFL volume-rate detail kernel launch failed");
    cuda_check(core::debug_kernel_sync(), "CFL volume-rate detail kernel execution failed");

    int min_cell = -1;
    double min_frac_rate = 0.0;
    cuda_check(cudaMemcpy(&min_cell, d_min_cell, sizeof(int), cudaMemcpyDeviceToHost),
               "CFL: copy volume-rate min_cell failed");
    cuda_check(cudaMemcpy(&min_frac_rate, d_min_frac_rate, sizeof(double),
                          cudaMemcpyDeviceToHost),
               "CFL: copy volume-rate min_frac_rate failed");

    log_volume_rate_cfl_clamp(state, dt_in, dt_out, min_cell, min_frac_rate,
                              threshold);
  }
  return dt_out;
}

void reset_volume_rate_cfl_history_after_ale(core::State& state) {
  if (state.vol.size() == 0) {
    return;
  }
  if (state.vol_prev_hydro.size() != state.vol.size()) {
    state.vol_prev_hydro.reset(state.vol.size());
  }
  cuda_check(cudaMemcpy(state.vol_prev_hydro.data(),
                        state.vol.data(),
                        state.vol.size() * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "reset_volume_rate_cfl_history_after_ale: device-to-device copy failed");
}

double compute_dt_hydro_acoustic(const core::State& state,
                                 const core::Config& cfg) {
  if (state.mesh.dim != 2 || state.mesh.node_r == nullptr) {
    return compute_dt_hydro(state, cfg);
  }
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "CFL acoustic requires matching node position/velocity arrays");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "2D CFL acoustic requires matching node velocity arrays");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    state.hydro_active.size() == state.rho.size(),
                "CFL acoustic hydro_active size mismatch");
  TENRYU_ASSERT(state.ei.empty() || state.ei.size() == state.rho.size(),
                "CFL acoustic ei size mismatch");
  TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                "2D CFL acoustic requires topo/state cell-size consistency");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.v_r.size()),
                "2D CFL acoustic requires topo/state node-size consistency");
  TENRYU_ASSERT(state.mesh.cell_area.size() == state.rho.size(),
                "2D CFL acoustic requires cell_area for all cells");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "CFL acoustic requires at least one material");

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells == 0) {
    return std::numeric_limits<double>::infinity();
  }
  core::State& mutable_state = const_cast<core::State&>(state);
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::aggregate_state(mutable_state, cfg, "cfl_acoustic", false);
  }

  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  double av_c1 = cfg.numerics.hydro.av_linear;
  if (cfg.numerics.hydro.adaptive_av.enabled &&
      cfg.numerics.hydro.av_type == "vnr") {
    const auto& adapt = cfg.numerics.hydro.adaptive_av;
    av_c1 = std::max({av_c1, adapt.base.c1, adapt.primary.c1, adapt.rebound.c1});
  } else if (cfg.numerics.hydro.av_type == "csw") {
    av_c1 = cfg.numerics.hydro.csw_C1;
  }
  const double* ei = state.ei.empty() ? nullptr : state.ei.data();
  const double* cs = state.cs.empty() ? nullptr : state.cs.data();

  double* d_min_dt = nullptr;
  int* d_have_active = nullptr;
  double* d_area = nullptr;
  d_min_dt = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_dt_hydro_acoustic:d_min_dt", sizeof(double)));
  d_have_active = static_cast<int*>(core::device_scratch_acquire(
      "cfl:compute_dt_hydro_acoustic:d_have_active", sizeof(int)));
  d_area = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_dt_hydro_acoustic:d_area",
      state.mesh.cell_area.size() * sizeof(double)));

  const double inf = std::numeric_limits<double>::infinity();
  cuda_check(cudaMemcpy(d_min_dt, &inf, sizeof(double), cudaMemcpyHostToDevice),
             "CFL acoustic: memcpy init min_dt failed");
  const int zero = 0;
  cuda_check(cudaMemcpy(d_have_active, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "CFL acoustic: memcpy init have_active failed");
  cuda_check(cudaMemcpy(d_area, state.mesh.cell_area.data(),
                        state.mesh.cell_area.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "CFL acoustic: cudaMemcpy cell_area failed");

  const std::vector<std::int8_t> effective_active =
      make_cfl_central_macro_effective_active(state, cfg);
  const std::vector<std::int8_t>& active_source =
      effective_active.empty() ? state.hydro_active : effective_active;
  std::int8_t* d_active = nullptr;
  if (!active_source.empty()) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cfl:compute_dt_hydro_acoustic:d_active",
        active_source.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, active_source.data(),
                          active_source.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "CFL acoustic: cudaMemcpy hydro_active failed");
  }
  std::uint8_t* d_cell_kind =
      upload_button_cfl_cell_kind_if_needed("cfl2d:button_cell_kind_acoustic",
                                            state.mesh);
  const int button_outer_node_ring =
      button_outer_node_ring_or_zero(state.mesh);

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  core::DeviceArray<std::uint8_t> d_combined_inactive;
  const std::uint8_t* d_pseudo_core_member =
      pole_angular_derefine::combined_inactive_mask_device(
          mutable_state, d_combined_inactive);
  cfl_2d_kernel<<<cw.blocks(), 256>>>(
      d_min_dt, d_have_active, state.x_r.data(), state.x_z.data(),
      state.v_r.data(), state.v_z.data(), state.ee.data(), ei, cs, d_area,
      d_cell_kind, d_active, d_pseudo_core_member, cw.begin, cw.end, n_cells,
      state.mesh.topo.nz, button_outer_node_ring, gamma, av_c1);
  cuda_check(cudaGetLastError(), "CFL acoustic 2D kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "CFL acoustic 2D kernel execution failed");

  int have_active = 0;
  cuda_check(cudaMemcpy(&have_active, d_have_active, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "CFL acoustic: memcpy have_active failed");
  const double min_local = copy_last_min_dt(d_min_dt);

  double acoustic =
      (have_active != 0 && std::isfinite(min_local))
          ? cfg.numerics.dt.cfl_hydro * min_local
          : std::numeric_limits<double>::infinity();
  const double pseudo_core_dt =
      central_pseudo_core::acoustic_dt(mutable_state, cfg);
  if (std::isfinite(pseudo_core_dt)) {
    acoustic = std::min(acoustic, pseudo_core_dt);
  }
  const double pole_macro_dt =
      pole_angular_derefine::acoustic_dt(mutable_state, cfg);
  if (std::isfinite(pole_macro_dt)) {
    acoustic = std::min(acoustic, pole_macro_dt);
  }
  return acoustic;
}

AxisMarginDtArgmin compute_dt_hydro_axis_margin_argmin(
    const core::State& state,
    const core::Config& cfg) {
  AxisMarginDtArgmin result;
  const double dt_proposed = compute_dt_hydro_acoustic(state, cfg);
  result.dt = dt_proposed;
  if (state.mesh.dim != 2 || !(dt_proposed > 0.0) ||
      !std::isfinite(dt_proposed)) {
    return result;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr < 1 || nz < 1) {
    return result;
  }
  const bool has_tri_fan_center =
      tenryu::mesh::has_tri_fan_center_topology(state.mesh);
  const bool has_multiblock = state.mesh.topo.multiblock.has_value();
  if (cfg.numerics.hydro.center_cfl_scope != core::CenterCflScope::DISABLED) {
    const TriFanCenterCflInfo center_cfl =
        compute_tri_fan_center_cfl_dt(state, cfg);
    result.center_cfl = center_cfl;
    if (std::isfinite(center_cfl.dt)) {
      result.dt = std::min(result.dt, center_cfl.dt);
      result.scale = result.dt / dt_proposed;
    }
    if (has_tri_fan_center || has_multiblock) {
      return result;
    }
  }
  if (has_tri_fan_center || has_multiblock) {
    return result;
  }
  if (!cfg.numerics.has_physical_rz_axis ||
      cfg.numerics.hydro.axis_margin_dt_floor_fraction <= 0.0) {
    return result;
  }
  TENRYU_ASSERT(state.x_z.size() == state.x_r.size(),
                "Axis-margin CFL diagnostic requires matching node position arrays");
  TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                "Axis-margin CFL diagnostic requires matching node velocity arrays");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "Axis-margin CFL diagnostic requires topo/state node-size consistency");

  double* d_min_scale = nullptr;
  d_min_scale = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_dt_hydro_axis_margin_argmin:d_min_scale", sizeof(double)));
  const double one = 1.0;
  cuda_check(cudaMemcpy(d_min_scale, &one, sizeof(double), cudaMemcpyHostToDevice),
             "CFL diagnostic: memcpy init axis min_scale failed");

  const int blocks = (nz + 255) / 256;
  axis_margin_cfl_dt_kernel<<<blocks, 256>>>(
      d_min_scale, state.x_r.data(), state.x_z.data(), state.v_r.data(),
      state.v_z.data(), nz, dt_proposed,
      cfg.numerics.hydro.axis_margin_dt_floor_fraction);
  cuda_check(cudaGetLastError(), "CFL diagnostic axis-margin kernel launch failed");
  cuda_check(core::debug_kernel_sync(),
             "CFL diagnostic axis-margin kernel execution failed");

  double min_scale = 1.0;
  cuda_check(cudaMemcpy(&min_scale, d_min_scale, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "CFL diagnostic: memcpy axis min_scale failed");

  if (!std::isfinite(min_scale)) {
    min_scale = 0.0;
  }
  result.scale = std::max(0.0, std::min(1.0, min_scale));
  result.dt = dt_proposed * result.scale;
  return result;
}

double compute_pole_axis_bbsw_contact_dt(const core::State& state,
                                         const core::Config& cfg) {
  if (!pole_axis_bbsw_enabled(cfg) || cfg.main.dimension != "2D_RZ" ||
      !state.mesh.topo.multiblock.has_value() || state.x_z.size() != state.v_z.size()) {
    return std::numeric_limits<double>::infinity();
  }
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  if (shell.n_i_cells <= 0 || shell.n_j_cells <= 0 ||
      shell.owned_node_begin < 0) {
    return std::numeric_limits<double>::infinity();
  }
  std::vector<double> z;
  std::vector<double> v_z;
  state.x_z.copy_to_host(z);
  state.v_z.copy_to_host(v_z);
  const int n_nodes = static_cast<int>(z.size());
  const int stride = shell.n_j_cells + 1;
  const auto shell_node = [&](const int q, const int j) {
    return shell.owned_node_begin + q * stride + j;
  };
  const int north_ref = shell_node(shell.n_i_cells, 0);
  const int south_ref = shell_node(shell.n_i_cells, shell.n_j_cells);
  if (north_ref < 0 || south_ref < 0 || north_ref >= n_nodes ||
      south_ref >= n_nodes) {
    return std::numeric_limits<double>::infinity();
  }
  const double z_c = 0.5 * (z[static_cast<std::size_t>(north_ref)] +
                            z[static_cast<std::size_t>(south_ref)]);
  double dt_axis = std::numeric_limits<double>::infinity();
  const int columns[2] = {0, shell.n_j_cells};
  for (const int j : columns) {
    for (int q = 0; q < shell.n_i_cells; ++q) {
      const int n0 = shell_node(q, j);
      const int n1 = shell_node(q + 1, j);
      if (n0 < 0 || n1 < 0 || n0 >= n_nodes || n1 >= n_nodes) {
        continue;
      }
      const double dz0 = z[static_cast<std::size_t>(n0)] - z_c;
      const double dz1 = z[static_cast<std::size_t>(n1)] - z_c;
      const double s0 = std::fabs(dz0);
      const double s1 = std::fabs(dz1);
      const double sign0 = (dz0 >= 0.0) ? 1.0 : -1.0;
      const double sign1 = (dz1 >= 0.0) ? 1.0 : -1.0;
      const double u0 = sign0 * v_z[static_cast<std::size_t>(n0)];
      const double u1 = sign1 * v_z[static_cast<std::size_t>(n1)];
      const double closing = u0 - u1;
      if (!(closing > 0.0) || !std::isfinite(closing)) {
        continue;
      }
      const double gap = s1 - s0;
      const double delta = pole_axis_bbsw::hard_gap(s0, s1);
      const double margin = gap - delta;
      if (!std::isfinite(margin)) {
        continue;
      }
      dt_axis = std::min(dt_axis,
                         pole_axis_bbsw::kAxisContactC *
                             std::max(0.0, margin) / closing);
    }
  }
  return dt_axis;
}

VolumeRateDtArgmin compute_dt_hydro_volume_rate_argmin(
    const core::State& state,
    const core::Config& cfg) {
  VolumeRateDtArgmin result;
  if (!cfg.numerics.hydro.volume_rate_cfl_enabled ||
      state.mesh.dim != 2 ||
      !(state.dt_prev_hydro > 0.0) ||
      !std::isfinite(state.dt_prev_hydro)) {
    return result;
  }
  TENRYU_ASSERT(cfg.numerics.hydro.volume_rate_cfl_threshold > 0.0,
                "Volume-rate CFL diagnostic requires threshold > 0");
  TENRYU_ASSERT(state.vol.size() == state.rho.size(),
                "Volume-rate CFL diagnostic requires volume/rho size consistency");
  TENRYU_ASSERT(state.vol_prev_hydro.size() == state.vol.size(),
                "Volume-rate CFL diagnostic requires previous hydro volume snapshot");

  const int n_cells = static_cast<int>(state.vol.size());
  if (n_cells == 0) {
    return result;
  }

  std::vector<double> vol(static_cast<std::size_t>(n_cells));
  std::vector<double> vol_prev(static_cast<std::size_t>(n_cells));
  state.vol.copy_to_host(vol.data());
  state.vol_prev_hydro.copy_to_host(vol_prev.data());

  double max_frac_rate = 0.0;
  const double dt_used_prev = std::max(state.dt_prev_hydro, kCflSensorEps);
  std::vector<std::uint8_t> inactive_storage;
  const std::vector<std::uint8_t>* inactive_member = nullptr;
  if (central_pseudo_core::configured(cfg)) {
    core::State& mutable_state = const_cast<core::State&>(state);
    central_pseudo_core::ensure_built(mutable_state, cfg);
  }
  if (pole_angular_derefine::configured(cfg)) {
    core::State& mutable_state = const_cast<core::State&>(state);
    pole_angular_derefine::ensure_built(mutable_state, cfg);
  }
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive_storage = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive_storage.empty()) {
      inactive_storage.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive_storage[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (!inactive_storage.empty()) {
    inactive_member = &inactive_storage;
  }
  for (int c = 0; c < n_cells; ++c) {
    if (inactive_member != nullptr &&
        (*inactive_member)[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    const double denom = std::max(vol_prev[static_cast<std::size_t>(c)],
                                  kCflSensorEps);
    const double frac_rate =
        std::abs(vol[static_cast<std::size_t>(c)] -
                 vol_prev[static_cast<std::size_t>(c)]) /
        denom / dt_used_prev;
    if (!(frac_rate > 0.0) || !std::isfinite(frac_rate)) {
      continue;
    }
    if (frac_rate > max_frac_rate) {
      max_frac_rate = frac_rate;
      result.argmin_cell = c;
      pole_angular_derefine::assert_active_cell(
          state, c, "volume-rate CFL argmin");
      result.frac_rate_at_argmin = frac_rate;
    }
  }
  if (result.argmin_cell >= 0) {
    result.dt = cfg.numerics.hydro.volume_rate_cfl_threshold / max_frac_rate;
  }
  return result;
}

HydroDtDiagnostics compute_dt_hydro_diagnostics(const core::State& state,
                                                const core::Config& cfg) {
  HydroDtDiagnostics result;
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "CFL requires matching node position/velocity arrays");
  TENRYU_ASSERT(state.hydro_active.empty() || state.hydro_active.size() == state.rho.size(),
                "CFL hydro_active size mismatch");
  TENRYU_ASSERT(state.ei.empty() || state.ei.size() == state.rho.size(),
                "CFL ei size mismatch");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "CFL requires at least one material");

  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells == 0) {
    return result;
  }
  core::State& mutable_state = const_cast<core::State&>(state);
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::aggregate_state(mutable_state, cfg, "cfl", false);
  }

  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  double av_c1 = cfg.numerics.hydro.av_linear;
  double av_c2 = cfg.numerics.hydro.av_quadratic;
  if (cfg.numerics.hydro.adaptive_av.enabled &&
      cfg.numerics.hydro.av_type == "vnr") {
    const auto& adapt = cfg.numerics.hydro.adaptive_av;
    av_c1 = std::max({av_c1, adapt.base.c1, adapt.primary.c1, adapt.rebound.c1});
    av_c2 = std::max({av_c2, adapt.base.c2, adapt.primary.c2, adapt.rebound.c2});
  } else if (cfg.numerics.hydro.av_type == "csw") {
    // 2026-07-26 review: the CSW operator applies csw_C1/csw_C2,
    // so the CFL correction must use the same coefficients — the former
    // av_linear/av_quadratic values under-estimated the AV stiffness when
    // the CSW coefficients are larger (defaults: 0.5/2.0 vs 0.1/1.5).
    // riemann / riemann_compatible take the impedance branches in the
    // kernel (av_du_mode 1/2) and ignore av_c1/av_c2 there.
    av_c1 = cfg.numerics.hydro.csw_C1;
    av_c2 = cfg.numerics.hydro.csw_C2;
  }
  // Artificial-heat stability bound (VNR chi only; riemann forces H off and
  // csw's chi_lim stays on the legacy margin — see NUMERICS §3.1.9).
  double av_heat_bound_c = cfg.numerics.hydro.av_heat_C;
  if (cfg.numerics.hydro.adaptive_av.enabled &&
      cfg.numerics.hydro.av_type == "vnr") {
    const auto& adapt = cfg.numerics.hydro.adaptive_av;
    av_heat_bound_c = std::max({av_heat_bound_c, adapt.base.heat_C,
                                adapt.primary.heat_C, adapt.rebound.heat_C});
  }
  const bool art_heat_bound_enabled =
      cfg.numerics.hydro.av_type == "vnr" && av_heat_bound_c > 0.0;
  const double av_limiter_J = cfg.numerics.hydro.av_limiter_J;
  const int av_du_mode =
      cfg.numerics.hydro.av_type == "riemann"
          ? 1
          : (cfg.numerics.hydro.av_type == "riemann_compatible" ? 2 : 0);
  const double* ei = state.ei.empty() ? nullptr : state.ei.data();
  const double* cs = state.cs.empty() ? nullptr : state.cs.data();

  TENRYU_ASSERT(av_limiter_J >= 0.0, "CFL requires av_limiter_J >= 0");

  double* d_pack = static_cast<double*>(core::device_scratch_acquire(
      "cfl:compute_dt_hydro_diagnostics:reduce_pack", 5 * sizeof(double)));
  double* d_min_dt = d_pack + 0;
  double* d_min_dt_post_shock = d_pack + 1;
  double* d_min_dt_crossing = d_pack + 2;
  double* d_min_dt_art_heat = d_pack + 3;
  int* d_have_active = reinterpret_cast<int*>(d_pack + 4);

  const double inf = std::numeric_limits<double>::infinity();
  const double h_init[5] = {inf, inf, inf, inf, 0.0};
  cuda_check(cudaMemcpy(d_pack, h_init, sizeof(h_init), cudaMemcpyHostToDevice),
             "CFL: memcpy init reduce pack failed");

  const std::vector<std::int8_t> effective_active =
      make_cfl_central_macro_effective_active(state, cfg);
  std::int8_t* d_active = nullptr;
  if (effective_active.empty()) {
    d_active = const_cast<std::int8_t*>(state.hydro_active_device_ptr());
  } else {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "cfl:compute_dt_hydro_diagnostics:d_active",
        effective_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, effective_active.data(),
                          effective_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "CFL: cudaMemcpy hydro_active failed");
  }

  const bool post_shock_heat_enabled =
      (state.mesh.dim == 1) &&
      cfg.numerics.hydro.post_shock_heat &&
      cfg.numerics.hydro.post_shock_heat_C > 0.0 &&
      cfg.numerics.hydro.post_shock_heat_decay > 0.0 &&
      state.shock_time.size() == state.rho.size();
  const double* shock_time =
      post_shock_heat_enabled ? state.shock_time.data() : nullptr;

  if (state.mesh.dim == 1) {
    TENRYU_ASSERT(state.x_r.size() == state.rho.size() + 1,
                  "1D CFL requires node count = cell count + 1");

    const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
    const int blocks = cw.blocks();
    cfl_1d_kernel<<<blocks, 256>>>(
        d_min_dt, d_min_dt_post_shock, d_min_dt_crossing, d_min_dt_art_heat,
        d_have_active, state.x_r.data(),
        state.v_r.data(), state.ee.data(), ei, cs, state.rho.data(),
        state.vol.data(), shock_time, d_active, cw.begin, cw.end, n_cells,
        state.mesh.geometry_code,
        state.t, gamma, av_c1, av_c2, av_du_mode,
        post_shock_heat_enabled ? 1 : 0,
        cfg.numerics.hydro.post_shock_heat_C,
        cfg.numerics.hydro.post_shock_heat_decay,
        cfg.numerics.hydro.crossing_dt_safety,
        art_heat_bound_enabled ? 1 : 0, av_heat_bound_c, av_limiter_J);
    cuda_check(cudaGetLastError(), "CFL 1D kernel launch failed");
    cuda_check(core::debug_kernel_sync(), "CFL 1D kernel execution failed");
  } else {
    TENRYU_ASSERT(state.mesh.dim == 2, "CFL requires mesh dim = 1 or 2");
    TENRYU_ASSERT(state.v_z.size() == state.v_r.size(),
                  "2D CFL requires matching node velocity arrays");
    TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                  "2D CFL requires topo/state cell-size consistency");
    TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.v_r.size()),
                  "2D CFL requires topo/state node-size consistency");
    TENRYU_ASSERT(state.mesh.cell_area.size() == state.rho.size(),
                  "2D CFL requires cell_area for all cells");
    if (state.mesh.topo.multiblock.has_value()) {
      TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                        static_cast<std::size_t>(n_cells) + 1U,
                    "2D CFL multiblock winner requires cell-node CSR offsets");
      TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                        static_cast<std::size_t>(n_cells) *
                            static_cast<std::size_t>(state.mesh.corner_stride),
                    "2D CFL multiblock winner requires cell-node CSR indices");
    }

    double* d_area = nullptr;
    d_area = static_cast<double*>(core::device_scratch_acquire(
        "cfl:compute_dt_hydro_diagnostics:d_area",
        state.mesh.cell_area.size() * sizeof(double)));
    cuda_check(cudaMemcpy(d_area, state.mesh.cell_area.data(),
                          state.mesh.cell_area.size() * sizeof(double),
                          cudaMemcpyHostToDevice),
               "CFL: cudaMemcpy cell_area failed");
    std::uint8_t* d_cell_kind =
        upload_button_cfl_cell_kind_if_needed("cfl2d:button_cell_kind_diag",
                                              state.mesh);
    const int button_outer_node_ring =
        button_outer_node_ring_or_zero(state.mesh);
    core::DeviceArray<std::uint8_t> d_cell_nverts;
    const std::uint8_t* d_cell_nverts_ptr =
        upload_cell_nverts_if_nonquad(d_cell_nverts, state.mesh, n_cells);

    const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
    core::DeviceArray<std::uint8_t> d_combined_inactive;
    const std::uint8_t* d_pseudo_core_member =
        pole_angular_derefine::combined_inactive_mask_device(
            mutable_state, d_combined_inactive);
    cfl_2d_kernel<<<cw.blocks(), 256>>>(
        d_min_dt, d_have_active, state.x_r.data(), state.x_z.data(),
        state.v_r.data(), state.v_z.data(), state.ee.data(), ei, cs, d_area,
        d_cell_kind, d_active, d_pseudo_core_member, cw.begin, cw.end, n_cells,
        state.mesh.topo.nz, button_outer_node_ring, gamma, av_c1);
    cuda_check(cudaGetLastError(), "CFL 2D kernel launch failed");
    cuda_check(core::debug_kernel_sync(), "CFL 2D kernel execution failed");
    const double min_local_2d = copy_last_min_dt(d_min_dt);
    if (std::isfinite(min_local_2d)) {
      int* d_winner_cell = nullptr;
      double* d_winner_values = nullptr;
      d_winner_cell = static_cast<int*>(core::device_scratch_acquire(
          "cfl:compute_dt_hydro_diagnostics:d_winner_cell", sizeof(int)));
      d_winner_values = static_cast<double*>(core::device_scratch_acquire(
          "cfl:compute_dt_hydro_diagnostics:d_winner_values",
          5 * sizeof(double)));
      const int no_winner = std::numeric_limits<int>::max();
      cuda_check(cudaMemcpy(d_winner_cell,
                            &no_winner,
                            sizeof(int),
                            cudaMemcpyHostToDevice),
                 "CFL: memcpy init winner_cell failed");
      cuda_check(cudaMemset(d_winner_values, 0, 5 * sizeof(double)),
                 "CFL: memset winner_values failed");
      cfl_2d_winner_kernel<<<cw.blocks(), 256>>>(
          d_winner_cell, min_local_2d, state.x_r.data(), state.x_z.data(),
          state.ee.data(), ei, cs, d_area, d_cell_kind, d_active,
          d_pseudo_core_member, cw.begin, cw.end, n_cells, state.mesh.topo.nz,
          button_outer_node_ring, gamma, av_c1);
      cuda_check(cudaGetLastError(), "CFL 2D winner kernel launch failed");
      cuda_check(core::debug_kernel_sync(), "CFL 2D winner kernel execution failed");
      int winner_cell = no_winner;
      cuda_check(cudaMemcpy(&winner_cell,
                            d_winner_cell,
                            sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "CFL: memcpy winner_cell failed");
      if (winner_cell >= 0 && winner_cell < n_cells) {
        pole_angular_derefine::assert_active_cell(
            state, winner_cell, "CFL acoustic winner");
        cfl_2d_winner_values_kernel<<<1, 1>>>(
            d_winner_values, winner_cell, cfg.numerics.dt.cfl_hydro,
            state.x_r.data(), state.x_z.data(), state.v_z.data(),
            state.ee.data(), ei, cs, state.rho.data(), d_area, d_cell_kind,
            state.mesh.topo.multiblock.has_value()
                ? state.mesh.multiblock_cell_node_csr_offsets.data()
                : nullptr,
            state.mesh.topo.multiblock.has_value()
                ? state.mesh.multiblock_cell_node_csr_indices.data()
                : nullptr,
            d_cell_nverts_ptr,
            n_cells, state.mesh.topo.nz, button_outer_node_ring, gamma, av_c1);
        cuda_check(cudaGetLastError(), "CFL 2D winner values kernel launch failed");
        cuda_check(core::debug_kernel_sync(),
                   "CFL 2D winner values kernel execution failed");
        double winner_values[5] = {};
        cuda_check(cudaMemcpy(winner_values,
                              d_winner_values,
                              5 * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "CFL: memcpy winner_values failed");
        const bool winner_is_multiblock = state.mesh.topo.multiblock.has_value();
        int reported_cell = winner_cell;
        if (winner_is_multiblock) {
          const auto& mb = *state.mesh.topo.multiblock;
          if (mb.cell_id_stable.size() == static_cast<std::size_t>(n_cells)) {
            reported_cell =
                mb.cell_id_stable[static_cast<std::size_t>(winner_cell)];
          }
        }
        result.cfl_winner.dt = cfg.numerics.dt.cfl_hydro * min_local_2d;
        result.cfl_winner.cell_id = reported_cell;
        if (winner_is_multiblock) {
          result.cfl_winner.i = -1;
          result.cfl_winner.j = -1;
        } else {
          result.cfl_winner.i = winner_cell / state.mesh.topo.nz;
          result.cfl_winner.j =
              winner_cell - result.cfl_winner.i * state.mesh.topo.nz;
        }
        result.cfl_winner.dt_at_cell = winner_values[0];
        result.cfl_winner.dl_at_cell = winner_values[1];
        result.cfl_winner.cs_at_cell = winner_values[2];
        result.cfl_winner.rho_at_cell = winner_values[3];
        result.cfl_winner.u_z_at_cell = winner_values[4];
      }
    }
  }

  double h_pack[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(h_pack, d_pack, sizeof(h_pack), cudaMemcpyDeviceToHost),
             "CFL: memcpy reduce pack failed");
  int have_active = 0;
  std::memcpy(&have_active, &h_pack[4], sizeof(int));
  const double min_local = h_pack[0];
  const double min_local_post_shock = h_pack[1];
  const double min_local_crossing = h_pack[2];
  const double min_local_art_heat = h_pack[3];

  if (have_active == 0) {
    return result;
  }
  if (!std::isfinite(min_local)) {
    return result;
  }

  result.acoustic_dt = cfg.numerics.dt.cfl_hydro * min_local;
  const double pseudo_core_dt =
      central_pseudo_core::acoustic_dt(mutable_state, cfg);
  if (std::isfinite(pseudo_core_dt) && pseudo_core_dt < result.acoustic_dt) {
    const auto& pc = mutable_state.central_pseudo_core;
    result.acoustic_dt = pseudo_core_dt;
    result.cfl_winner.dt = pseudo_core_dt;
    result.cfl_winner.cell_id = pc.representative_cell;
    if (state.mesh.topo.multiblock.has_value()) {
      const auto& mb = *state.mesh.topo.multiblock;
      if (mb.cell_id_stable.size() == static_cast<std::size_t>(state.rho.size()) &&
          pc.representative_cell >= 0) {
        result.cfl_winner.cell_id =
            mb.cell_id_stable[static_cast<std::size_t>(pc.representative_cell)];
      }
    }
    result.cfl_winner.i = -1;
    result.cfl_winner.j = -1;
    result.cfl_winner.dt_at_cell = pseudo_core_dt;
    result.cfl_winner.dl_at_cell =
        (pc.V_c > 0.0) ? std::cbrt(pc.V_c / kFourPiOverThree) : 0.0;
    if (!state.cs.empty() && pc.representative_cell >= 0) {
      std::vector<double> cs_host;
      state.cs.copy_to_host(cs_host);
      result.cfl_winner.cs_at_cell =
          cs_host[static_cast<std::size_t>(pc.representative_cell)];
    }
    result.cfl_winner.rho_at_cell =
        (pc.V_c > 0.0) ? pc.M_c / pc.V_c : 0.0;
  }
  const double pole_macro_dt =
      pole_angular_derefine::acoustic_dt(mutable_state, cfg);
  if (std::isfinite(pole_macro_dt) && pole_macro_dt < result.acoustic_dt) {
    const auto& pc = mutable_state.pole_angular_derefine;
    result.acoustic_dt = pole_macro_dt;
    result.cfl_winner.dt = pole_macro_dt;
    result.cfl_winner.cell_id = pc.representative_cell;
    if (state.mesh.topo.multiblock.has_value() &&
        pc.representative_cell >= 0) {
      const auto& mb = *state.mesh.topo.multiblock;
      if (mb.cell_id_stable.size() == static_cast<std::size_t>(state.rho.size())) {
        result.cfl_winner.cell_id =
            mb.cell_id_stable[static_cast<std::size_t>(pc.representative_cell)];
      }
    }
    result.cfl_winner.i = -1;
    result.cfl_winner.j = -1;
    result.cfl_winner.dt_at_cell = pole_macro_dt;
    result.cfl_winner.dl_at_cell = 0.0;
    result.cfl_winner.cs_at_cell = 0.0;
    result.cfl_winner.rho_at_cell = 0.0;
    result.cfl_winner.u_z_at_cell = 0.0;
  }
  double dt = result.acoustic_dt;
  if (std::isfinite(min_local_post_shock)) {
    result.post_shock_dt = min_local_post_shock;
    dt = std::min(dt, min_local_post_shock);
  }
  if (std::isfinite(min_local_crossing)) {
    result.crossing_dt = min_local_crossing;
    dt = std::min(dt, min_local_crossing);
  }
  if (std::isfinite(min_local_art_heat)) {
    result.art_heat_dt = min_local_art_heat;
    dt = std::min(dt, min_local_art_heat);
  }
  if (state.mesh.dim == 2 &&
      (cfg.numerics.hydro.av_model == core::AvModel::CswEdge ||
       cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98)) {
    const double dt_av = compatible::compute_csw_edge_av_cfl_dt(state, cfg);
    if (std::isfinite(dt_av)) {
      result.edge_av_dt = dt_av;
      dt = std::min(dt, dt_av);
    }
  }
  if (state.mesh.dim == 2 &&
      cfg.numerics.hydro.subzonal_pressure_enabled &&
      cfg.numerics.hydro.subzonal_dt_limiter_enabled) {
    const double dt_subzonal =
        compute_compatible_subzonal_pressure_dt_2d(state, cfg, d_active);
    if (std::isfinite(dt_subzonal)) {
      result.subzonal_pressure_dt = dt_subzonal;
      dt = std::min(dt, dt_subzonal);
    }
  }
  dt = compute_axis_margin_cfl_dt(
      state, dt, cfg.numerics.hydro.axis_margin_dt_floor_fraction,
      cfg.numerics.has_physical_rz_axis, cfg, &result.tri_fan_center_cfl);
  if (cfg.numerics.hydro.corner_j_predict_cfl_enabled) {
    result.corner_j_predict_dt =
        compute_corner_j_predict_cfl_dt(state, cfg);
    result.corner_j_predict_dt = std::max(
        result.corner_j_predict_dt,
        cfg.numerics.hydro.corner_j_predict_floor_frac * result.acoustic_dt);
    dt = std::min(dt, result.corner_j_predict_dt);
  }
  result.axis_margin_dt = dt;
  const double axis_contact_dt = compute_pole_axis_bbsw_contact_dt(state, cfg);
  if (std::isfinite(axis_contact_dt)) {
    dt = std::min(dt, axis_contact_dt);
    result.axis_margin_dt = std::min(result.axis_margin_dt, axis_contact_dt);
  }
  if (cfg.numerics.hydro.rz_geometric_cfl_enabled && state.mesh.dim == 2) {
    core::NodeField1D predicted_u_half_r;
    core::NodeField1D predicted_u_half_z;
    const double* geom_v_r = state.v_r.data();
    const double* geom_v_z = state.v_z.data();
    // Baseline path uses the current state velocity as the geometric
    // displacement velocity.  The opt-in precise path recomputes
    // a_n from the current pressure+AV force and passes
    // u_half = u_old + 0.5 * dt * a_n to the geometric CFL kernel.
    if (compute_precise_rz_geometric_u_half(state,
                                            cfg,
                                            dt,
                                            predicted_u_half_r,
                                            predicted_u_half_z)) {
      geom_v_r = predicted_u_half_r.data();
      geom_v_z = predicted_u_half_z.data();
    }
    const double dt_geom =
        compute_rz_geometric_cfl_dt(state, cfg, dt, geom_v_r, geom_v_z);
    if (std::isfinite(dt_geom)) {
      dt = std::min(dt, dt_geom);
      result.rz_geometric_dt = dt;
    }
  }
  if (cfg.numerics.hydro.volume_rate_cfl_enabled) {
    const double dt_before_volume_rate = dt;
    dt = compute_volume_rate_cfl_dt(
        state, dt, state.dt_prev_hydro,
        cfg.numerics.hydro.volume_rate_cfl_threshold,
        nullptr);
    if (dt < dt_before_volume_rate) {
      result.volume_rate_dt = dt;
    }
  }
  result.dt = dt;
  return result;
}

double compute_dt_hydro(const core::State& state, const core::Config& cfg) {
  return compute_dt_hydro_diagnostics(state, cfg).dt;
}

HydroDtArgmin compute_dt_hydro_argmin(const core::State& state,
                                      const core::Config& cfg) {
  HydroDtArgmin result;
  if (state.mesh.dim != 2 || state.mesh.node_r == nullptr || state.rho.empty()) {
    result.dt = compute_dt_hydro(state, cfg);
    return result;
  }
  TENRYU_ASSERT(state.mesh.cell_area.size() == state.rho.size(),
                "2D CFL argmin requires cell_area for all cells");
  TENRYU_ASSERT(state.ei.empty() || state.ei.size() == state.rho.size(),
                "CFL argmin ei size mismatch");
  TENRYU_ASSERT(state.cs.empty() || state.cs.size() == state.rho.size(),
                "CFL argmin cs size mismatch");
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    state.hydro_active.size() == state.rho.size(),
                "CFL argmin hydro_active size mismatch");
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "CFL argmin requires at least one material");

  const int n_cells = static_cast<int>(state.rho.size());
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.topo.multiblock->cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "CFL argmin multiblock requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.topo.multiblock->cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "CFL argmin multiblock requires cell-node CSR indices");
  }
  const double gamma = cfg.materials.materials.front().ideal_gas_gamma;
  double av_c1 = cfg.numerics.hydro.av_linear;
  if (cfg.numerics.hydro.adaptive_av.enabled &&
      cfg.numerics.hydro.av_type == "vnr") {
    const auto& adapt = cfg.numerics.hydro.adaptive_av;
    av_c1 = std::max({av_c1, adapt.base.c1, adapt.primary.c1, adapt.rebound.c1});
  } else if (cfg.numerics.hydro.av_type == "csw") {
    av_c1 = cfg.numerics.hydro.csw_C1;
  }

  std::vector<double> area;
  std::vector<double> ee;
  std::vector<double> ei;
  std::vector<double> cs;
  std::vector<double> rho;
  std::vector<double> v_z;
  std::vector<double> node_r;
  std::vector<double> node_z;
  area = state.mesh.cell_area;
  state.ee.copy_to_host(ee);
  state.rho.copy_to_host(rho);
  if (!state.ei.empty()) {
    state.ei.copy_to_host(ei);
  }
  if (!state.cs.empty()) {
    state.cs.copy_to_host(cs);
  }
  if (!state.v_z.empty()) {
    state.v_z.copy_to_host(v_z);
  }
  const int button_outer_node_ring = button_outer_node_ring_or_zero(state.mesh);
  if (button_outer_node_ring > 0) {
    state.x_r.copy_to_host(node_r);
    state.x_z.copy_to_host(node_z);
  }

  double min_local = std::numeric_limits<double>::infinity();
  const bool is_multiblock = state.mesh.topo.multiblock.has_value();
  const bool have_cell_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells);
  std::vector<std::uint8_t> inactive_storage;
  const std::vector<std::uint8_t>* pseudo_core_member = nullptr;
  if (central_pseudo_core::configured(cfg)) {
    core::State& mutable_state = const_cast<core::State&>(state);
    central_pseudo_core::ensure_built(mutable_state, cfg);
    if (central_pseudo_core::active(mutable_state)) {
      pseudo_core_member = &mutable_state.central_pseudo_core.member_mask;
    }
  }
  if (pole_angular_derefine::configured(cfg)) {
    core::State& mutable_state = const_cast<core::State&>(state);
    pole_angular_derefine::ensure_built(mutable_state, cfg);
  }
  const std::vector<std::uint8_t>* inactive_member = nullptr;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    inactive_storage = state.central_pseudo_core.inactive_member_mask;
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    if (inactive_storage.empty()) {
      inactive_storage.assign(static_cast<std::size_t>(n_cells), 0U);
    }
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        inactive_storage[static_cast<std::size_t>(c)] = 1U;
      }
    }
  }
  if (!inactive_storage.empty()) {
    inactive_member = &inactive_storage;
  }
  // Option C (§6o.4k): the CFL argmin scans OWNED cells only — the far
  // regions are stale-but-finite and must not seed the dt minimum (the
  // rank-local minima are combined by the driver's Allreduce(MIN)).
  const core::State::LaunchWindow cfl_cw = state.owned_cell_window(n_cells);
  for (int c = cfl_cw.begin; c < cfl_cw.end; ++c) {
    if (state.mesh.is_dormant_cell(c)) {
      continue;
    }
    if (!state.hydro_active.empty() &&
        state.hydro_active[static_cast<std::size_t>(c)] == 0) {
      continue;
    }
    if (pseudo_core_member != nullptr &&
        (*pseudo_core_member)[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    if (inactive_member != nullptr &&
        (*inactive_member)[static_cast<std::size_t>(c)] != 0U) {
      continue;
    }
    double dl = 0.0;
    const bool is_button =
        button_outer_node_ring > 0 && state.mesh.is_button_cell(c);
    if (is_button) {
      dl = button_polygon_characteristic_length_from_nodes(
          node_r.data(), node_z.data(), button_outer_node_ring,
          state.mesh.topo.nz);
    } else {
      const double area_c = area[static_cast<std::size_t>(c)];
      if (!(area_c > 0.0) || !std::isfinite(area_c)) {
        continue;
      }
      dl = std::sqrt(area_c);
    }
    if (!(dl > 0.0) || !std::isfinite(dl)) {
      continue;
    }
    const double e_e = ee[static_cast<std::size_t>(c)] > 0.0
                           ? ee[static_cast<std::size_t>(c)]
                           : 0.0;
    const double e_i = (!ei.empty() && ei[static_cast<std::size_t>(c)] > 0.0)
                           ? ei[static_cast<std::size_t>(c)]
                           : 0.0;
    const double cs_val = !cs.empty()
                              ? cs[static_cast<std::size_t>(c)]
                              : std::sqrt(gamma * (gamma - 1.0) * (e_e + e_i));
    const double cs_c = std::max(cs_val, 0.0);
    const double denom = is_button ? cs_c : cs_c * (1.0 + av_c1);
    if (!(denom > 0.0) || !std::isfinite(denom)) {
      continue;
    }
    const double dt_local = dl / denom;
    if (dt_local < min_local) {
      min_local = dt_local;
      result.argmin_cell = c;
      pole_angular_derefine::assert_active_cell(
          state, c, "CFL acoustic argmin");
      if (is_multiblock) {
        const auto& mb = *state.mesh.topo.multiblock;
        if (mb.cell_id_stable.size() == static_cast<std::size_t>(n_cells)) {
          result.argmin_cell = mb.cell_id_stable[static_cast<std::size_t>(c)];
        }
      }
      result.sqrt_area_at_argmin = dl;
      result.cs_at_argmin = cs_c;
      if (is_multiblock) {
        result.argmin_i = -1;
        result.argmin_j = -1;
      } else {
        result.argmin_i = c / state.mesh.topo.nz;
        result.argmin_j = c - result.argmin_i * state.mesh.topo.nz;
      }
      result.rho_at_argmin = rho[static_cast<std::size_t>(c)];
      if (!v_z.empty()) {
        if (is_button) {
          result.u_z_at_argmin = button_average_node_field(
              v_z.data(), button_outer_node_ring, state.mesh.topo.nz);
        } else if (is_multiblock) {
          const auto& mb = *state.mesh.topo.multiblock;
          const int off =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
          const int active_nverts =
              have_cell_nverts
                  ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
                  : mesh::kMeshTopoCellStorageSlots;
          if (active_nverts == 3) {
            result.u_z_at_argmin =
                (v_z[static_cast<std::size_t>(
                     mb.cell_node_csr_indices[static_cast<std::size_t>(off + 0)])] +
                 v_z[static_cast<std::size_t>(
                     mb.cell_node_csr_indices[static_cast<std::size_t>(off + 1)])] +
                 v_z[static_cast<std::size_t>(
                     mb.cell_node_csr_indices[static_cast<std::size_t>(off + 2)])]) /
                3.0;
          } else if (active_nverts >= 5) {
            result.u_z_at_argmin = 0.0;
            for (int k = 0; k < active_nverts; ++k) {
              result.u_z_at_argmin +=
                  v_z[static_cast<std::size_t>(
                      mb.cell_node_csr_indices[
                          static_cast<std::size_t>(off + k)])];
            }
            result.u_z_at_argmin /= static_cast<double>(active_nverts);
          } else {
            result.u_z_at_argmin =
                0.25 * (v_z[static_cast<std::size_t>(
                            mb.cell_node_csr_indices[static_cast<std::size_t>(off + 0)])] +
                        v_z[static_cast<std::size_t>(
                            mb.cell_node_csr_indices[static_cast<std::size_t>(off + 1)])] +
                        v_z[static_cast<std::size_t>(
                            mb.cell_node_csr_indices[static_cast<std::size_t>(off + 2)])] +
                        v_z[static_cast<std::size_t>(
                            mb.cell_node_csr_indices[static_cast<std::size_t>(off + 3)])]);
          }
        } else {
          const int stride = state.mesh.topo.nz + 1;
          const int n00 = result.argmin_i * stride + result.argmin_j;
          const int n10 = (result.argmin_i + 1) * stride + result.argmin_j;
          const int n11 = (result.argmin_i + 1) * stride + (result.argmin_j + 1);
          const int n01 = result.argmin_i * stride + (result.argmin_j + 1);
          result.u_z_at_argmin =
              0.25 * (v_z[static_cast<std::size_t>(n00)] +
                      v_z[static_cast<std::size_t>(n10)] +
                      v_z[static_cast<std::size_t>(n11)] +
                      v_z[static_cast<std::size_t>(n01)]);
        }
      }
    }
  }
  if (result.argmin_cell >= 0) {
    result.dt = cfg.numerics.dt.cfl_hydro * min_local;
  }
  return result;
}

}  // namespace tenryu::hydro
