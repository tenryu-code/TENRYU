#include "hydro/artificial_viscosity.hpp"

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/state.hpp"
#include "hydro/eos_context.hpp"
#include "hydro/per_material_eos_accessors.cuh"
#include "materials/eos_device_table.cuh"
#include "mesh/geometry_1d.cuh"

namespace tenryu::hydro {
namespace {

using tenryu::mesh::geometry_1d_face_area;

constexpr double kFourPi = 12.566370614359172953850573533118;
constexpr double kShockPressureJumpThreshold = 0.3;
constexpr double kShockDensityJumpThreshold = 0.05;
constexpr double kShockRhConsistencyThreshold = 0.5;
constexpr double kCompMachScale = 0.05;
constexpr double kOscillationThreshold = 0.2;
constexpr double kShockSupportFloor = 0.25;
constexpr double kMildCompressionAlpha = 0.25;
constexpr double kSensorEps = 1.0e-30;

struct AVMaterialParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
};

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

std::vector<AVMaterialParams> make_av_material_params(const core::Config& cfg) {
  std::vector<AVMaterialParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    AVMaterialParams p{};
    p.A = std::max(mat.A, 1.0e-12);
    p.gamma = std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-12);
    if (cfg.materials.zbar.model == "fixed" && cfg.materials.zbar.fixed_value >= 0.0) {
      p.Zbar = cfg.materials.zbar.fixed_value;
    } else {
      p.Zbar = (mat.Z > 0.0) ? mat.Z : 1.0;
    }
    if (mat.is_void) {
      p.Zbar = 0.0;
    }
    params[m] = p;
  }
  return params;
}

__device__ __forceinline__ double clamp01_device(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__device__ __forceinline__ double total_pressure_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const int i) {
  return Pe[i] + Pi[i];
}

__device__ __forceinline__ double interface_developed_shock_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double Jrho =
      fabs(drho) / fmax(fmin(rho_left, rho_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  const bool same_sign = (dp * drho > 0.0);
  return (same_sign && Jp >= kShockPressureJumpThreshold &&
          Jrho >= kShockDensityJumpThreshold &&
          Z >= kShockRhConsistencyThreshold)
             ? 1.0
             : 0.0;
}

__device__ __forceinline__ double interface_shock_support_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  if (interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, left_cell, right_cell) >
      0.0) {
    return 1.0;
  }

  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double Jrho =
      fabs(drho) / fmax(fmin(rho_left, rho_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  return (Jp >= kShockPressureJumpThreshold &&
          Jrho < kShockDensityJumpThreshold &&
          Z >= kShockRhConsistencyThreshold)
             ? 1.0
             : 0.0;
}

__device__ __forceinline__ double interface_mild_compression_weight_1d_device(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ rho,
    const double* __restrict__ cs,
    const int left_cell,
    const int right_cell) {
  const double p_left = total_pressure_device(Pe, Pi, left_cell);
  const double p_right = total_pressure_device(Pe, Pi, right_cell);
  const double rho_left = rho[left_cell];
  const double rho_right = rho[right_cell];
  const double dp = p_right - p_left;
  const double drho = rho_right - rho_left;
  const double Jp = fabs(dp) / fmax(fmin(p_left, p_right), kSensorEps);
  const double cs2_face =
      0.5 * (cs[left_cell] * cs[left_cell] + cs[right_cell] * cs[right_cell]);
  const double Z = fabs(dp) / fmax(cs2_face * fabs(drho), kSensorEps);
  const bool same_sign = (dp * drho > 0.0);
  const double W_rh =
      (same_sign && Z >= kShockRhConsistencyThreshold) ? 1.0 : 0.0;
  const double W_jp_soft =
      clamp01_device(Jp / kShockPressureJumpThreshold);
  return W_rh * W_jp_soft;
}

__device__ __forceinline__ double compute_node_sigma_1d_device(
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

__device__ inline void compute_node_sigma_1d_kernel_body(
    const int j,
    double* __restrict__ sigma,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int n_nodes,
    const double J) {
  sigma[j] = compute_node_sigma_1d_device(node_r, node_u, j, n_nodes, J);
}

__global__ void compute_node_sigma_1d_kernel(
    double* __restrict__ sigma,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int n_nodes,
    const double J) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_nodes) {
    return;
  }

  compute_node_sigma_1d_kernel_body(j, sigma, node_r, node_u, n_nodes, J);
}

__device__ double compute_q_scalar_device(const double rho,
                                          const double dl,
                                          const double div_u,
                                          const double cs,
                                          const double c1,
                                          const double c2) {
  if (div_u >= 0.0) {
    return 0.0;
  }

  const double compression = fabs(div_u);
  return rho * ((c2 * c2) * dl * dl * compression * compression +
                c1 * dl * cs * compression);
}

__device__ __forceinline__ double compute_eos_aware_boost_device(
    const bool eos_aware,
    const double rho,
    const double total_pressure,
    const double cs,
    const double gamma1_ref,
    const double boost_max) {
  if (!eos_aware || !(rho > 0.0) || !(total_pressure > kSensorEps) || !(cs > 0.0)) {
    return 1.0;
  }
  const double gamma1_local = rho * cs * cs / total_pressure;
  if (!(gamma1_local > 0.0) || !isfinite(gamma1_local)) {
    return 1.0;
  }
  return fmin(fmax(gamma1_ref / gamma1_local, 1.0), boost_max);
}

__device__ __forceinline__ bool riemann_cell_active_device(
    const std::int8_t* __restrict__ hydro_active,
    const int i) {
  return hydro_active == nullptr || hydro_active[i] != 0;
}

__device__ __forceinline__ double minmod_device(const double a, const double b) {
  if (a * b <= 0.0) {
    return 0.0;
  }
  return copysign(fmin(fabs(a), fabs(b)), a);
}

__device__ __forceinline__ double csw_node_slope_van_leer_device(
    const double d_minus,
    const double d_plus) {
  if (!(d_minus * d_plus > 0.0) || !isfinite(d_minus) || !isfinite(d_plus) ||
      d_plus == 0.0) {
    return 0.0;
  }
  const double r = d_minus / d_plus;
  const double phi = (r + fabs(r)) / (1.0 + fabs(r));
  return phi * d_plus;
}

__device__ __forceinline__ double csw_node_slope_bj_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int j,
    const double d_minus,
    const double d_plus) {
  double slope = 0.5 * (d_minus + d_plus);
  if (!isfinite(slope) || slope == 0.0) {
    return 0.0;
  }

  const double u_j = node_u[j];
  const double u_min = fmin(fmin(node_u[j - 1], node_u[j]), node_u[j + 1]);
  const double u_max = fmax(fmax(node_u[j - 1], node_u[j]), node_u[j + 1]);
  double alpha = 1.0;

  const double dr_left = node_r[j] - node_r[j - 1];
  const double dr_right = node_r[j + 1] - node_r[j];
  const double du_left = -0.5 * dr_left * slope;
  const double du_right = 0.5 * dr_right * slope;
  if (du_left > 0.0) {
    alpha = fmin(alpha, (u_max - u_j) / du_left);
  } else if (du_left < 0.0) {
    alpha = fmin(alpha, (u_min - u_j) / du_left);
  }
  if (du_right > 0.0) {
    alpha = fmin(alpha, (u_max - u_j) / du_right);
  } else if (du_right < 0.0) {
    alpha = fmin(alpha, (u_min - u_j) / du_right);
  }
  return clamp01_device(alpha) * slope;
}

__device__ __forceinline__ double csw_node_slope_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const int j,
    const int n_nodes,
    const int limiter_kind) {
  if (j < 0 || j >= n_nodes) {
    return 0.0;
  }
  const bool has_left = j > 0;
  const bool has_right = j + 1 < n_nodes;
  if (!has_left && !has_right) {
    return 0.0;
  }
  if (!has_left) {
    const double dr = node_r[j + 1] - node_r[j];
    return (dr > 0.0) ? (node_u[j + 1] - node_u[j]) / dr : 0.0;
  }
  if (!has_right) {
    const double dr = node_r[j] - node_r[j - 1];
    return (dr > 0.0) ? (node_u[j] - node_u[j - 1]) / dr : 0.0;
  }

  const double dr_left = node_r[j] - node_r[j - 1];
  const double dr_right = node_r[j + 1] - node_r[j];
  if (!(dr_left > 0.0) || !(dr_right > 0.0)) {
    return 0.0;
  }
  const double d_minus = (node_u[j] - node_u[j - 1]) / dr_left;
  const double d_plus = (node_u[j + 1] - node_u[j]) / dr_right;
  if (limiter_kind == 1) {
    return csw_node_slope_bj_device(node_r, node_u, j, d_minus, d_plus);
  }
  return csw_node_slope_van_leer_device(d_minus, d_plus);
}

__device__ __forceinline__ double riemann_cell_center_r_1d_device(
    const double* __restrict__ node_r,
    const int i) {
  return 0.5 * (node_r[i] + node_r[i + 1]);
}

__device__ __forceinline__ double riemann_cell_velocity_1d_device(
    const double* __restrict__ node_u,
    const int i) {
  return 0.5 * (node_u[i] + node_u[i + 1]);
}

__device__ __forceinline__ double riemann_velocity_slope_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int i,
    const int n_cells) {
  const double r_i = riemann_cell_center_r_1d_device(node_r, i);
  const double u_i = riemann_cell_velocity_1d_device(node_u, i);

  bool has_left = (i > 0) && riemann_cell_active_device(hydro_active, i - 1);
  bool has_right =
      (i + 1 < n_cells) && riemann_cell_active_device(hydro_active, i + 1);
  double s_left = 0.0;
  double s_right = 0.0;
  if (has_left) {
    const double r_left = riemann_cell_center_r_1d_device(node_r, i - 1);
    const double dr_left = r_i - r_left;
    if (dr_left > 0.0) {
      s_left = (u_i - riemann_cell_velocity_1d_device(node_u, i - 1)) / dr_left;
    } else {
      has_left = false;
    }
  }
  if (has_right) {
    const double r_right = riemann_cell_center_r_1d_device(node_r, i + 1);
    const double dr_right = r_right - r_i;
    if (dr_right > 0.0) {
      s_right = (riemann_cell_velocity_1d_device(node_u, i + 1) - u_i) /
                dr_right;
    } else {
      has_right = false;
    }
  }
  if (has_left && has_right) {
    return minmod_device(s_left, s_right);
  }
  if (has_left) {
    return s_left;
  }
  if (has_right) {
    return s_right;
  }
  return 0.0;
}

__device__ __forceinline__ double riemann_reconstruct_velocity_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int cell,
    const int n_cells,
    const double r_face) {
  const double r_cell = riemann_cell_center_r_1d_device(node_r, cell);
  const double u_cell = riemann_cell_velocity_1d_device(node_u, cell);
  const double slope =
      riemann_velocity_slope_1d_device(node_r, node_u, hydro_active, cell, n_cells);
  return u_cell + slope * (r_face - r_cell);
}

__device__ __forceinline__ double riemann_alpha_1d_device(
    const double rho,
    const double p,
    const double cs) {
  if (!(rho > 0.0) || !(cs > 0.0) || !(p > kSensorEps)) {
    return 2.0 / 3.0;
  }
  const double gamma1 = rho * cs * cs / p;
  if (!(gamma1 > 0.0) || !isfinite(gamma1)) {
    return 2.0 / 3.0;
  }
  return 0.25 * (gamma1 + 1.0);
}

__device__ __forceinline__ double riemann_face_q_1d_device(
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
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

  if (!riemann_cell_active_device(hydro_active, right_cell) ||
      (!center_mirror && !riemann_cell_active_device(hydro_active, left_cell))) {
    return 0.0;
  }

  const double rho_R = rho[right_cell];
  const double p_R = total_pressure_device(Pe, Pi, right_cell);
  const double cs_R = cs[right_cell];
  double rho_L = rho_R;
  double p_L = p_R;
  double cs_L = cs_R;
  if (!center_mirror) {
    rho_L = rho[left_cell];
    p_L = total_pressure_device(Pe, Pi, left_cell);
    cs_L = cs[left_cell];
  }
  if (!(rho_L > 0.0) || !(rho_R > 0.0) || !(cs_L > 0.0) || !(cs_R > 0.0) ||
      !isfinite(rho_L) || !isfinite(rho_R) || !isfinite(p_L) ||
      !isfinite(p_R) || !isfinite(cs_L) || !isfinite(cs_R)) {
    return 0.0;
  }

  const double r_face = node_r[face];
  const double u_R = riemann_reconstruct_velocity_1d_device(
      node_r, node_u, hydro_active, right_cell, n_cells, r_face);
  const double u_L =
      center_mirror
          ? -u_R
          : riemann_reconstruct_velocity_1d_device(
                node_r, node_u, hydro_active, left_cell, n_cells, r_face);
  const double du = u_L - u_R;
  if (!(du > 0.0) || !isfinite(du)) {
    return 0.0;
  }

  const double alpha_L = riemann_alpha_1d_device(rho_L, p_L, cs_L);
  const double alpha_R = riemann_alpha_1d_device(rho_R, p_R, cs_R);
  const double Z_L = rho_L * (cs_L + alpha_L * du);
  const double Z_R = rho_R * (cs_R + alpha_R * du);
  const double Z_sum = Z_L + Z_R;
  if (!(Z_L > 0.0) || !(Z_R > 0.0) || !(Z_sum > 0.0) ||
      !isfinite(Z_L) || !isfinite(Z_R)) {
    return 0.0;
  }

  const double p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * du) / Z_sum;
  const double p_avg = 0.5 * (p_L + p_R);
  const double q_face = p_star - p_avg;
  return (q_face > 0.0 && isfinite(q_face)) ? q_face : 0.0;
}

__global__ void compute_q_riemann_1d_kernel(
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    double* __restrict__ chi_out,
    double* __restrict__ q2_out,
    double* __restrict__ div_u_out,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  if (!riemann_cell_active_device(hydro_active, i)) {
    Q[i] = 0.0;
  } else {
    const double q_left = riemann_face_q_1d_device(
        rho, node_r, node_u, Pe, Pi, cs, hydro_active, i, n_cells);
    const double q_right = riemann_face_q_1d_device(
        rho, node_r, node_u, Pe, Pi, cs, hydro_active, i + 1, n_cells);
    Q[i] = 0.5 * (q_left + q_right);
  }
  if (chi_out != nullptr) {
    chi_out[i] = 0.0;
  }
  if (q2_out != nullptr) {
    q2_out[i] = 0.0;
  }
  if (div_u_out != nullptr) {
    div_u_out[i] = 0.0;
  }
}

__device__ __forceinline__ double rc_cell_center_1d_device(
    const double* __restrict__ node_r,
    const int i) {
  return 0.5 * (node_r[i] + node_r[i + 1]);
}

__device__ __forceinline__ double rc_cell_average_velocity_1d_device(
    const double* __restrict__ node_u,
    const int i) {
  return 0.5 * (node_u[i] + node_u[i + 1]);
}

__device__ __forceinline__ double rc_nodal_gradient_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int j,
    const int n_cells) {
  if (j > 0 && j < n_cells &&
      riemann_cell_active_device(hydro_active, j - 1) &&
      riemann_cell_active_device(hydro_active, j)) {
    const double xc_left = rc_cell_center_1d_device(node_r, j - 1);
    const double xc_right = rc_cell_center_1d_device(node_r, j);
    const double dx = xc_right - xc_left;
    if (dx == 0.0) {
      return 0.0;
    }
    const double ubar_left = rc_cell_average_velocity_1d_device(node_u, j - 1);
    const double ubar_right = rc_cell_average_velocity_1d_device(node_u, j);
    return (ubar_right - ubar_left) / dx;
  }

  if ((j == 0 || !riemann_cell_active_device(hydro_active, j - 1)) &&
      j < n_cells && riemann_cell_active_device(hydro_active, j)) {
    const double xc = rc_cell_center_1d_device(node_r, j);
    const double dx = xc - node_r[j];
    if (dx == 0.0) {
      return 0.0;
    }
    const double ubar = rc_cell_average_velocity_1d_device(node_u, j);
    return (ubar - node_u[j]) / dx;
  }

  if ((j == n_cells || !riemann_cell_active_device(hydro_active, j)) &&
      j > 0 && riemann_cell_active_device(hydro_active, j - 1)) {
    if (j >= 2 && riemann_cell_active_device(hydro_active, j - 2)) {
      const double xc_left = rc_cell_center_1d_device(node_r, j - 2);
      const double xc_right = rc_cell_center_1d_device(node_r, j - 1);
      const double dx = xc_right - xc_left;
      if (dx != 0.0) {
        const double ubar_left =
            rc_cell_average_velocity_1d_device(node_u, j - 2);
        const double ubar_right =
            rc_cell_average_velocity_1d_device(node_u, j - 1);
        return (ubar_right - ubar_left) / dx;
      }
    }
    const double xc = rc_cell_center_1d_device(node_r, j - 1);
    const double dx = node_r[j] - xc;
    if (dx == 0.0) {
      return 0.0;
    }
    const double ubar = rc_cell_average_velocity_1d_device(node_u, j - 1);
    return (node_u[j] - ubar) / dx;
  }

  return 0.0;
}

__device__ __forceinline__ double rc_bj_phi_1d_device(
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const std::int8_t* __restrict__ hydro_active,
    const int j,
    const int n_cells,
    const double g_j) {
  constexpr double kBetaBJ = 0.5;
  double umin = node_u[j];
  double umax = node_u[j];
  if (j > 0) {
    umin = fmin(umin, node_u[j - 1]);
    umax = fmax(umax, node_u[j - 1]);
  }
  if (j < n_cells) {
    umin = fmin(umin, node_u[j + 1]);
    umax = fmax(umax, node_u[j + 1]);
  }
  if (j == 0 && n_cells > 0) {
    const double u_reflected = 2.0 * node_u[j] - node_u[j + 1];
    umin = fmin(umin, u_reflected);
    umax = fmax(umax, u_reflected);
  }

  double phi = 1.0;
  for (int i = j - 1; i <= j; ++i) {
    if (i < 0 || i >= n_cells ||
        !riemann_cell_active_device(hydro_active, i)) {
      continue;
    }
    const double d = (rc_cell_center_1d_device(node_r, i) - node_r[j]) * g_j;
    double candidate = 1.0;
    if (d > 0.0) {
      candidate = fmin(1.0, kBetaBJ * (umax - node_u[j]) / d);
    } else if (d < 0.0) {
      candidate = fmin(1.0, kBetaBJ * (umin - node_u[j]) / d);
    }
    phi = fmin(phi, candidate);
  }
  return fmin(1.0, fmax(0.0, phi));
}

// Version 1.1 (2026-08-03): Morgan-faithful staggered Godunov-like
// dissipation reduced to the 1-D cell Q slot (consult 20260803-1442,
// Morgan et al. JCP 259 (2014); Dukowicz JCP 61 (1985)).
// Nodal velocities are projected to the cell center with a nodal
// secant gradient and a Barth-Jespersen limiter (beta=0.5, one phi per
// node, min over adjacent cells); corner impedances
// mu = rho(cs + b1|u_c - ubar|), b1=(Gamma1+1)/2, close a shared
// cell-center Riemann velocity; with the symmetric first-order
// impedance mu = rho(cs + b1*du/2) this reduces to Q = (mu/2)*du,
// du = max(0, uLc-uRc), gated on actual cell-volume
// compression (Adot<0) so compatible AV work is non-negative by
// construction. Inner boundary uses a reflected (odd-velocity) state;
// outer boundary uses a one-sided gradient and zero external stress
// (the existing P+Q free-surface contract). Galilean invariant.
__global__ void compute_q_riemann_compatible_1d_kernel(
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    double* __restrict__ chi_out,
    double* __restrict__ q2_out,
    double* __restrict__ div_u_out,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int geom_code) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  if (!riemann_cell_active_device(hydro_active, i)) {
    Q[i] = 0.0;
  } else {
    const double rho_i = rho[i];
    const double p_i = total_pressure_device(Pe, Pi, i);
    const double cs_i = cs[i];
    // R3.2 (2026-08-03): cs==0 is a VALID cold state — the quadratic
    // impedance term works without sound speed. The inherited strict
    // cs>0 guard zeroed ALL viscosity in cold gas (measured Noh-2T wall
    // free-fall collapse at t=dr/|u| with Qvisc[0]==0 while compressing).
    if (!(rho_i > 0.0) || cs_i < 0.0 || !isfinite(rho_i) ||
        !isfinite(p_i) || !isfinite(cs_i)) {
      Q[i] = 0.0;
    } else {
      const double xc_i = rc_cell_center_1d_device(node_r, i);
      const double g_L =
          rc_nodal_gradient_1d_device(node_r, node_u, hydro_active, i, n_cells);
      const double phi_L =
          rc_bj_phi_1d_device(node_r, node_u, hydro_active, i, n_cells, g_L);
      const double g_R = rc_nodal_gradient_1d_device(
          node_r, node_u, hydro_active, i + 1, n_cells);
      const double phi_R = rc_bj_phi_1d_device(
          node_r, node_u, hydro_active, i + 1, n_cells, g_R);
      const double uLc = node_u[i] + phi_L * (xc_i - node_r[i]) * g_L;
      const double uRc =
          node_u[i + 1] + phi_R * (xc_i - node_r[i + 1]) * g_R;
      const double du_proj = fmax(0.0, uLc - uRc);
      const double A_L =
          tenryu::mesh::geometry_1d_face_area(geom_code, node_r[i]);
      const double A_R =
          tenryu::mesh::geometry_1d_face_area(geom_code, node_r[i + 1]);
      const double Vdot = A_R * node_u[i + 1] - A_L * node_u[i];
      if (du_proj == 0.0 || Vdot >= 0.0) {
        Q[i] = 0.0;
      } else {
        // R3.1 (2026-08-03): symmetric first-order Morgan impedance
        // (consult 20260803-1442 §2.6 equal-impedance reduction),
        // mu = rho*(cs + b1*du_proj/2), Q = (mu/2)*du_proj. The per-side
        // |u_c - ubar| surrogate degenerates (mu -> rho*cs -> 0 in cold
        // cells whenever a limited projection lands on the cell average):
        // measured Noh-2T wall free-fall collapse at t = dr/|u|.
        const double alpha = riemann_alpha_1d_device(rho_i, p_i, cs_i);
        const double b1 = 2.0 * alpha;
        const double q_val =
            0.5 * rho_i * (cs_i + 0.5 * b1 * du_proj) * du_proj;
        Q[i] = isfinite(q_val) ? q_val : 0.0;
      }
    }
  }
  if (chi_out != nullptr) {
    chi_out[i] = 0.0;
  }
  if (q2_out != nullptr) {
    q2_out[i] = 0.0;
  }
  if (div_u_out != nullptr) {
    div_u_out[i] = 0.0;
  }
}

template <int GEOM>
__device__ inline void compute_q_1d_kernel_body(
    const int i,
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ cs,
    const double* __restrict__ sigma,
    double* __restrict__ chi_out,
    double* __restrict__ q2_out,
    double* __restrict__ div_u_out,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double c1,
    const double c2,
    const double* __restrict__ c1_eff_field,
    const double* __restrict__ c2_eff_field,
    const bool eos_aware,
    const double gamma1_ref,
    const double boost_max) {
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  const double V = vol[i];
  const double r0 = node_r[i];
  const double r1 = node_r[i + 1];
  const double dr = r1 - r0;
  if (V <= 0.0 || dr <= 0.0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  double A0;
  double A1;
  if constexpr (GEOM == 0) {
    A0 = kFourPi * r0 * r0;
    A1 = kFourPi * r1 * r1;
  } else {
    A0 = geometry_1d_face_area(GEOM, r0);
    A1 = geometry_1d_face_area(GEOM, r1);
  }
  const double dVdt = A1 * node_u[i + 1] - A0 * node_u[i];
  const double div_u = dVdt / V;
  if (div_u_out != nullptr) {
    div_u_out[i] = div_u;
  }

  if (div_u >= 0.0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  const double u_right = node_u[i + 1] - 0.5 * dr * sigma[i + 1];
  const double u_left = node_u[i] + 0.5 * dr * sigma[i];
  const double du_lim = u_right - u_left;
  const double chi = fmax(0.0, -du_lim / dr);
  if (chi_out != nullptr) {
    chi_out[i] = chi;
  }
  if (chi <= 0.0) {
    Q[i] = 0.0;
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  const double boost = compute_eos_aware_boost_device(
      eos_aware, rho[i], total_pressure_device(Pe, Pi, i), cs[i], gamma1_ref, boost_max);
  const double c1_local = (c1_eff_field != nullptr) ? c1_eff_field[i] : c1;
  const double c2_local = (c2_eff_field != nullptr) ? c2_eff_field[i] : c2;
  const double c1_eff = c1_local * boost;
  const double c2_eff = c2_local * boost;
  const double q2_limited = rho[i] * (c2_eff * c2_eff) * dr * dr * chi * chi;
  if (q2_out != nullptr) {
    q2_out[i] = q2_limited;
  }

  const double left_shock = (i == 0)
                                ? 1.0
                                : interface_shock_support_weight_1d_device(
                                      Pe, Pi, rho, cs, i - 1, i);
  const double right_shock =
      (i == n_cells - 1)
          ? 1.0
          : interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);
  const double W_shock = fmax(left_shock, right_shock);
  const double M_comp = dr * chi / fmax(cs[i], kSensorEps);
  const double W_comp = clamp01_device(M_comp / kCompMachScale);
  const double Q_base =
      q2_limited + rho[i] * (c1_eff * dr * cs[i] * chi);

  if (W_shock > 0.0) {
    const double left_developed =
        (i == 0)
            ? 1.0
            : interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, i - 1, i);
    const double right_developed =
        (i == n_cells - 1)
            ? 1.0
            : interface_developed_shock_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);

    double W_osc = 1.0;
    if (i > 0 && i + 1 < n_cells) {
      const double p_left = total_pressure_device(Pe, Pi, i - 1);
      const double p_center = total_pressure_device(Pe, Pi, i);
      const double p_right = total_pressure_device(Pe, Pi, i + 1);
      const double dpL = p_center - p_left;
      const double dpR = p_right - p_center;
      if (dpL * dpR < 0.0) {
        const double osc =
            fmin(fabs(dpL), fabs(dpR)) / fmax(fmax(fabs(dpL), fabs(dpR)), kSensorEps);
        if (osc > kOscillationThreshold &&
            !(left_developed > 0.0 && right_developed > 0.0)) {
          W_osc = 0.0;
        }
      }
    }

    const double phi = W_shock * fmax(kShockSupportFloor, W_comp * W_osc);
    Q[i] = phi * Q_base;
    return;
  }

  const double left_mild =
      (i == 0)
          ? 0.0
          : interface_mild_compression_weight_1d_device(Pe, Pi, rho, cs, i - 1, i);
  const double right_mild =
      (i == n_cells - 1)
          ? 0.0
          : interface_mild_compression_weight_1d_device(Pe, Pi, rho, cs, i, i + 1);
  const double left_neighbor_shock =
      (i > 1)
          ? interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i - 2, i - 1)
          : 0.0;
  const double right_neighbor_shock =
      (i + 2 < n_cells)
          ? interface_shock_support_weight_1d_device(Pe, Pi, rho, cs, i + 1, i + 2)
          : 0.0;
  if (left_neighbor_shock > 0.0 || right_neighbor_shock > 0.0) {
    Q[i] = 0.0;
    return;
  }
  const double W_mild = fmax(left_mild, right_mild);
  Q[i] = kMildCompressionAlpha * W_mild * W_comp *
         rho[i] * (c1_eff * dr * cs[i] * chi);
}

template <int GEOM>
__global__ void compute_q_1d_kernel(double* __restrict__ Q,
                                    const double* __restrict__ rho,
                                    const double* __restrict__ vol,
                                    const double* __restrict__ node_r,
                                    const double* __restrict__ node_u,
                                    const double* __restrict__ Pe,
                                    const double* __restrict__ Pi,
                                    const double* __restrict__ cs,
                                    const double* __restrict__ sigma,
                                    double* __restrict__ chi_out,
                                    double* __restrict__ q2_out,
                                    double* __restrict__ div_u_out,
                                    const std::int8_t* __restrict__ hydro_active,
                                    const int n_cells,
                                    const double c1,
                                    const double c2,
                                    const double* __restrict__ c1_eff_field,
                                    const double* __restrict__ c2_eff_field,
                                    const bool eos_aware,
                                    const double gamma1_ref,
                                    const double boost_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  compute_q_1d_kernel_body<GEOM>(
      i, Q, rho, vol, node_r, node_u, Pe, Pi, cs, sigma, chi_out, q2_out,
      div_u_out, hydro_active, n_cells, c1, c2, c1_eff_field, c2_eff_field,
      eos_aware, gamma1_ref, boost_max);
}

__global__ void compute_q_csw_1d_kernel(
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ cs,
    double* __restrict__ chi_out,
    double* __restrict__ q2_out,
    double* __restrict__ div_u_out,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double c1,
    const double c2,
    const double* __restrict__ c1_eff_field,
    const double* __restrict__ c2_eff_field,
    const int limiter_kind,
    const double shock_limiter_floor,
    const bool zero_uniform_compression) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  if (hydro_active != nullptr && hydro_active[i] == 0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  const double dr = node_r[i + 1] - node_r[i];
  if (!(dr > 0.0)) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    if (div_u_out != nullptr) {
      div_u_out[i] = 0.0;
    }
    return;
  }

  const double du = node_u[i + 1] - node_u[i];
  const double raw_grad = du / dr;
  const double chi_raw = fmax(0.0, -raw_grad);
  if (div_u_out != nullptr) {
    div_u_out[i] = raw_grad;
  }
  if (chi_raw <= 0.0 || !(rho[i] > 0.0) || !(cs[i] >= 0.0)) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  double chi_lim = chi_raw;
  if (zero_uniform_compression) {
    const int n_nodes = n_cells + 1;
    const double s_left =
        csw_node_slope_1d_device(node_r, node_u, i, n_nodes, limiter_kind);
    const double s_right =
        csw_node_slope_1d_device(node_r, node_u, i + 1, n_nodes, limiter_kind);
    const double u_left_star = node_u[i] + 0.5 * dr * s_left;
    const double u_right_star = node_u[i + 1] - 0.5 * dr * s_right;
    const double chi_reconstructed = fmax(0.0, -(u_right_star - u_left_star) / dr);
    chi_lim = fmin(chi_raw, chi_reconstructed);
    if (chi_reconstructed > 0.0 && shock_limiter_floor > 0.0) {
      chi_lim = fmin(chi_raw, fmax(chi_lim, shock_limiter_floor * chi_raw));
    }
  }

  if (chi_lim <= 0.0) {
    Q[i] = 0.0;
    if (chi_out != nullptr) {
      chi_out[i] = 0.0;
    }
    if (q2_out != nullptr) {
      q2_out[i] = 0.0;
    }
    return;
  }

  const double c1_eff = (c1_eff_field != nullptr) ? c1_eff_field[i] : c1;
  const double c2_eff = (c2_eff_field != nullptr) ? c2_eff_field[i] : c2;
  const double q2 = rho[i] * (c2_eff * c2_eff) * dr * dr * chi_lim * chi_lim;
  Q[i] = q2 + rho[i] * (c1_eff * dr * cs[i] * chi_lim);
  if (chi_out != nullptr) {
    chi_out[i] = chi_lim;
  }
  if (q2_out != nullptr) {
    q2_out[i] = q2;
  }
}

__device__ inline void add_bulk_viscosity_1d_kernel_body(
    const int i,
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double c_bulk,
    const double* __restrict__ c_bulk_field) {
  if (hydro_active != nullptr && hydro_active[i] == 0) {
    return;
  }

  const double dr = node_r[i + 1] - node_r[i];
  if (dr <= 0.0) {
    return;
  }

  const double rho_i = fmax(rho[i], 0.0);
  const double cs_i = fmax(cs[i], 0.0);
  if (rho_i <= 0.0 || cs_i <= 0.0) {
    return;
  }

  // Compression-only: a positive scalar Q in an expanding cell removes
  // internal energy (-Q dV/dt < 0 for dV/dt > 0) instead of dissipating
  // kinetic energy, so bulk viscosity must vanish for div(u) >= 0.
  const double div_u = (node_u[i + 1] - node_u[i]) / fmax(dr, kSensorEps);
  if (div_u >= 0.0) {
    return;
  }
  const double c_bulk_i = (c_bulk_field != nullptr) ? c_bulk_field[i] : c_bulk;
  if (!(c_bulk_i > 0.0)) {
    return;
  }
  const double q_bulk = c_bulk_i * rho_i * cs_i * fabs(div_u) * dr;
  Q[i] += q_bulk;
}

__global__ void add_bulk_viscosity_1d_kernel(
    double* __restrict__ Q,
    const double* __restrict__ rho,
    const double* __restrict__ node_r,
    const double* __restrict__ node_u,
    const double* __restrict__ cs,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double c_bulk,
    const double* __restrict__ c_bulk_field) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  add_bulk_viscosity_1d_kernel_body(
      i, Q, rho, node_r, node_u, cs, hydro_active, n_cells, c_bulk,
      c_bulk_field);
}

__device__ __forceinline__ bool hydro_cell_active(
    const std::int8_t* __restrict__ hydro_active,
    const int i) {
  return hydro_active == nullptr || hydro_active[i] != 0;
}

template <int GEOM>
__device__ __forceinline__ double compute_artificial_heat_face_power_1d(
    const double* __restrict__ rho,
    const double* __restrict__ e,
    const double* __restrict__ chi,
    const double* __restrict__ node_r,
    const std::int8_t* __restrict__ hydro_active,
    const int left_cell,
    const int right_cell,
    const double heat_c,
    const double* __restrict__ heat_C_field) {
  if (!hydro_cell_active(hydro_active, left_cell) ||
      !hydro_cell_active(hydro_active, right_cell)) {
    return 0.0;
  }

  const double dr_left = node_r[left_cell + 1] - node_r[left_cell];
  const double dr_right = node_r[right_cell + 1] - node_r[right_cell];
  const double dr_centers = 0.5 * (dr_left + dr_right);
  if (dr_left <= 0.0 || dr_right <= 0.0 || dr_centers <= 0.0) {
    return 0.0;
  }

  const double rho_face = 0.5 * (rho[left_cell] + rho[right_cell]);
  const double chi_face = fmax(chi[left_cell], chi[right_cell]);
  if (rho_face <= 0.0 || chi_face <= 0.0) {
    return 0.0;
  }

  const double l_face = 0.5 * (dr_left + dr_right);
  const double de = e[right_cell] - e[left_cell];
  const double heat_c_face =
      (heat_C_field != nullptr)
          ? 0.5 * (heat_C_field[left_cell] + heat_C_field[right_cell])
          : heat_c;
  const double flux =
      -heat_c_face * rho_face * l_face * l_face * chi_face * de / dr_centers;
  const double r_face = node_r[right_cell];
  double area;
  if constexpr (GEOM == 0) {
    area = kFourPi * r_face * r_face;
  } else {
    area = geometry_1d_face_area(GEOM, r_face);
  }
  return area * flux;
}

template <int GEOM>
__device__ inline void compute_artificial_heat_1d_kernel_body(
    const int i,
    double* __restrict__ heat_rate,
    const double* __restrict__ rho,
    const double* __restrict__ e,
    const double* __restrict__ chi,
    const double* __restrict__ node_r,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double heat_c,
    const double* __restrict__ heat_C_field) {
  if (!hydro_cell_active(hydro_active, i)) {
    heat_rate[i] = 0.0;
    return;
  }

  const double left_power =
      (i > 0) ? compute_artificial_heat_face_power_1d<GEOM>(
                    rho, e, chi, node_r, hydro_active, i - 1, i, heat_c,
                    heat_C_field)
              : 0.0;
  const double right_power =
      (i + 1 < n_cells) ? compute_artificial_heat_face_power_1d<GEOM>(
                              rho, e, chi, node_r, hydro_active, i, i + 1, heat_c,
                              heat_C_field)
                        : 0.0;
  heat_rate[i] = -(right_power - left_power);
}

template <int GEOM>
__global__ void compute_artificial_heat_1d_kernel(
    double* __restrict__ heat_rate,
    const double* __restrict__ rho,
    const double* __restrict__ e,
    const double* __restrict__ chi,
    const double* __restrict__ node_r,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const double heat_c,
    const double* __restrict__ heat_C_field) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  compute_artificial_heat_1d_kernel_body<GEOM>(
      i, heat_rate, rho, e, chi, node_r, hydro_active, n_cells, heat_c,
      heat_C_field);
}

__global__ void compute_q_2d_kernel(double* __restrict__ Q,
                                    const double* __restrict__ rho,
                                    const double* __restrict__ dl,
                                    const double* __restrict__ div_u,
                                    const double* __restrict__ Pe,
                                    const double* __restrict__ Pi,
                                    const double* __restrict__ cs,
                                    const std::int8_t* __restrict__ hydro_active,
                                    const int n_cells,
                                    const double c1,
                                    const double c2,
                                    const bool eos_aware,
                                    const double gamma1_ref,
                                    const double boost_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }

  if (hydro_active != nullptr && hydro_active[i] == 0) {
    Q[i] = 0.0;
    return;
  }

  const double boost = compute_eos_aware_boost_device(
      eos_aware, rho[i], total_pressure_device(Pe, Pi, i), cs[i], gamma1_ref, boost_max);
  Q[i] = compute_q_scalar_device(
      rho[i], dl[i], div_u[i], cs[i], c1 * boost, c2 * boost);
}

__global__ void compute_q_per_material_2d_kernel(
    double* __restrict__ Q,
    double* __restrict__ Q_per_material,
    const double* __restrict__ mass_per_material,
    const double* __restrict__ Ee_per_material,
    const double* __restrict__ Ei_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ vol,
    double* __restrict__ Te_per_material,
    double* __restrict__ Ti_per_material,
    std::uint8_t* __restrict__ Te_per_material_valid,
    std::uint8_t* __restrict__ Ti_per_material_valid,
    unsigned long long* __restrict__ counts,
    const double* __restrict__ dl,
    const double* __restrict__ div_u,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const AVMaterialParams* __restrict__ params,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int n_mat,
    const double c1,
    const double c2,
    const bool eos_aware,
    const double gamma1_ref,
    const double boost_max,
    const double te_floor,
    const double ti_floor,
    const double presence_threshold_volfrac,
    const double presence_threshold_mass_density,
    const bool lazy_cache_enabled,
    const bool low_density_extrap) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_mat;
  if (idx >= total) {
    return;
  }

  const int c = idx / n_mat;
  const int m = idx - c * n_mat;
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    Q_per_material[idx] = 0.0;
    return;
  }

  const double vf = volfrac[idx];
  const double V = vol[c];
  const double mass_m = mass_per_material[idx];
  if (!(vf > presence_threshold_volfrac) || !(V > 0.0) ||
      !(mass_m > 0.0) || !isfinite(vf) || !isfinite(V) || !isfinite(mass_m)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    Q_per_material[idx] = 0.0;
    return;
  }
  const double rho_m = mass_m / (vf * V);
  if (!(rho_m > presence_threshold_mass_density) || !isfinite(rho_m)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    Q_per_material[idx] = 0.0;
    return;
  }

  tenryu::hydro::per_material::PerMaterialAccessorView view{};
  view.mass_per_material = mass_per_material;
  view.Ee_per_material = Ee_per_material;
  view.Ei_per_material = Ei_per_material;
  view.volfrac = volfrac;
  view.vol = vol;
  view.Te_per_material = lazy_cache_enabled ? Te_per_material : nullptr;
  view.Ti_per_material = lazy_cache_enabled ? Ti_per_material : nullptr;
  view.Te_per_material_valid = lazy_cache_enabled ? Te_per_material_valid : nullptr;
  view.Ti_per_material_valid = lazy_cache_enabled ? Ti_per_material_valid : nullptr;
  view.lazy_cache_te_m_enabled = lazy_cache_enabled;
  view.presence_threshold_volfrac = presence_threshold_volfrac;
  view.presence_threshold_mass_density_g_per_cc = presence_threshold_mass_density;
  view.d_counts = counts;
  view.n_cells = n_cells;
  view.n_mat = n_mat;

  const AVMaterialParams p = params[m];
  const auto electron_view =
      (electron_views != nullptr) ? electron_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto ion_view =
      (ion_views != nullptr) ? ion_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto electron = tenryu::hydro::per_material::get_electron_thermo_per_material(
      view, electron_view, c, m, p.Zbar, p.A, te_floor, low_density_extrap, p.gamma);
  const auto ion = tenryu::hydro::per_material::get_ion_thermo_per_material(
      view, ion_view, c, m, p.A, ti_floor, low_density_extrap, p.gamma);
  const double pe_m = tenryu::hydro::per_material::get_pe(electron);
  const double pi_m = tenryu::hydro::per_material::get_pi(ion);
  const double cs_m = tenryu::hydro::per_material::get_cs(electron, ion);
  const double boost = compute_eos_aware_boost_device(
      eos_aware, rho_m, pe_m + pi_m, cs_m, gamma1_ref, boost_max);
  const double q_m = compute_q_scalar_device(
      rho_m, dl[c], div_u[c], cs_m, c1 * boost, c2 * boost);
  Q_per_material[idx] = q_m;
  if (q_m > 0.0 && isfinite(q_m)) {
    atomic_add_double(Q + c, vf * q_m);
  }
}

}  // namespace

ArtificialViscosity::ArtificialViscosity(const double c1,
                                         const double c2,
                                         const double J,
                                         const double heat_C,
                                         const bool eos_aware,
                                         const double eos_gamma1_ref,
                                         const double eos_boost_max,
                                         const ArtificialViscosityType type,
                                         const double bulk_viscosity_C,
                                         const CswLimiter csw_limiter,
                                         const double csw_shock_limiter_floor,
                                         const bool csw_zero_uniform_compression,
                                         const bool csw_diagnostics)
    : c1_(c1),
      c2_(c2),
      j_(J),
      heat_c_(heat_C),
      eos_aware_(eos_aware),
      eos_gamma1_ref_(eos_gamma1_ref),
      eos_boost_max_(eos_boost_max),
      type_(type),
      bulk_viscosity_c_(bulk_viscosity_C),
      csw_limiter_(csw_limiter),
      csw_shock_limiter_floor_(csw_shock_limiter_floor),
      csw_zero_uniform_compression_(csw_zero_uniform_compression),
      csw_diagnostics_(csw_diagnostics) {
  TENRYU_ASSERT(c1_ >= 0.0, "ArtificialViscosity requires C1 >= 0");
  TENRYU_ASSERT(c2_ >= 0.0, "ArtificialViscosity requires C2 >= 0");
  TENRYU_ASSERT(j_ >= 0.0, "ArtificialViscosity requires J >= 0");
  TENRYU_ASSERT(heat_c_ >= 0.0, "ArtificialViscosity requires heat_C >= 0");
  TENRYU_ASSERT(eos_gamma1_ref_ > 0.0, "ArtificialViscosity requires eos_gamma1_ref > 0");
  TENRYU_ASSERT(eos_boost_max_ >= 1.0, "ArtificialViscosity requires eos_boost_max >= 1");
  TENRYU_ASSERT(bulk_viscosity_c_ >= 0.0,
                "ArtificialViscosity requires bulk_viscosity_C >= 0");
  TENRYU_ASSERT(csw_shock_limiter_floor_ >= 0.0 && csw_shock_limiter_floor_ <= 1.0,
                "ArtificialViscosity requires csw_shock_limiter_floor in [0, 1]");
}

double ArtificialViscosity::compute_scalar(const double rho,
                                           const double dl,
                                           const double div_u,
                                           const double cs) const {
  TENRYU_ASSERT(type_ == ArtificialViscosityType::Vnr,
                "ArtificialViscosity::compute_scalar only supports Vnr");
  TENRYU_ASSERT(rho >= 0.0, "AV requires rho >= 0");
  TENRYU_ASSERT(dl >= 0.0, "AV requires dl >= 0");
  TENRYU_ASSERT(cs >= 0.0, "AV requires cs >= 0");

  if (div_u >= 0.0) {
    return 0.0;
  }

  const double compression = std::abs(div_u);
  return rho * ((c2_ * c2_) * dl * dl * compression * compression +
                c1_ * dl * cs * compression);
}

void ArtificialViscosity::compute_Q_1d(core::CellField1DView Q,
                                       core::ConstCellField1DView rho,
                                       core::ConstCellField1DView vol,
                                       core::ConstNodeField1DView node_r,
                                       core::ConstNodeField1DView node_u,
                                       core::ConstCellField1DView Pe,
                                       core::ConstCellField1DView Pi,
                                       core::ConstCellField1DView cs,
                                       const std::int8_t* d_hydro_active,
                                       const int geom_code,
                                       core::CellField1DView chi_out,
                                       core::CellField1DView q2_out,
                                       core::CellField1DView div_u_out,
                                       core::ConstCellField1DView c1_eff,
                                       core::ConstCellField1DView c2_eff) const {
  TENRYU_ASSERT(Q.size() == rho.size(), "AV 1D: Q/rho size mismatch");
  TENRYU_ASSERT(Q.size() == vol.size(), "AV 1D: Q/vol size mismatch");
  TENRYU_ASSERT(Q.size() == Pe.size(), "AV 1D: Q/Pe size mismatch");
  TENRYU_ASSERT(Q.size() == Pi.size(), "AV 1D: Q/Pi size mismatch");
  TENRYU_ASSERT(Q.size() == cs.size(), "AV 1D: Q/cs size mismatch");
  TENRYU_ASSERT(node_r.size() == node_u.size(), "AV 1D: node fields size mismatch");
  TENRYU_ASSERT(node_r.size() == Q.size() + 1, "AV 1D: node/cell size mismatch");
  TENRYU_ASSERT(chi_out.empty() || chi_out.size() == Q.size(),
                "AV 1D: chi_out size mismatch");
  TENRYU_ASSERT(q2_out.empty() || q2_out.size() == Q.size(),
                "AV 1D: q2_out size mismatch");
  TENRYU_ASSERT(div_u_out.empty() || div_u_out.size() == Q.size(),
                "AV 1D: div_u_out size mismatch");
  TENRYU_ASSERT(c1_eff.empty() || c1_eff.size() == Q.size(),
                "AV 1D: c1_eff size mismatch");
  TENRYU_ASSERT(c2_eff.empty() || c2_eff.size() == Q.size(),
                "AV 1D: c2_eff size mismatch");

  if (Q.empty()) {
    return;
  }

  double* d_sigma = nullptr;

  const int n_cells = static_cast<int>(Q.size());
  const int blocks = (n_cells + 255) / 256;
  double* d_chi_out = !chi_out.empty() ? chi_out.data() : nullptr;
  double* d_q2_out = !q2_out.empty() ? q2_out.data() : nullptr;
  double* d_div_u_out = !div_u_out.empty() ? div_u_out.data() : nullptr;
  const double* d_c1_eff = !c1_eff.empty() ? c1_eff.data() : nullptr;
  const double* d_c2_eff = !c2_eff.empty() ? c2_eff.data() : nullptr;

  if (type_ == ArtificialViscosityType::Riemann) {
    compute_q_riemann_1d_kernel<<<blocks, 256>>>(
        Q.data(), rho.data(), node_r.data(), node_u.data(), Pe.data(), Pi.data(),
        cs.data(), d_chi_out, d_q2_out, d_div_u_out, d_hydro_active, n_cells);
    cuda_check(cudaGetLastError(), "Riemann AV 1D: kernel launch failed");
    cuda_check(core::debug_kernel_sync(), "Riemann AV 1D: kernel execution failed");
    return;
  }

  if (type_ == ArtificialViscosityType::RiemannCompatible) {
    compute_q_riemann_compatible_1d_kernel<<<blocks, 256>>>(
        Q.data(), rho.data(), node_r.data(), node_u.data(), Pe.data(), Pi.data(),
        cs.data(), d_chi_out, d_q2_out, d_div_u_out, d_hydro_active, n_cells,
        geom_code);
    cuda_check(cudaGetLastError(),
               "Riemann-compatible AV 1D: kernel launch failed");
    cuda_check(core::debug_kernel_sync(),
               "Riemann-compatible AV 1D: kernel execution failed");
    return;
  }

  if (type_ == ArtificialViscosityType::Csw) {
    const int limiter_kind =
        (csw_limiter_ == CswLimiter::BarthJespersen) ? 1 : 0;
    compute_q_csw_1d_kernel<<<blocks, 256>>>(
        Q.data(), rho.data(), node_r.data(), node_u.data(), cs.data(),
        d_chi_out, d_q2_out, d_div_u_out, d_hydro_active, n_cells, c1_, c2_,
        d_c1_eff, d_c2_eff, limiter_kind, csw_shock_limiter_floor_,
        csw_zero_uniform_compression_);
    cuda_check(cudaGetLastError(), "CSW AV 1D: kernel launch failed");
    cuda_check(core::debug_kernel_sync(), "CSW AV 1D: kernel execution failed");
    return;
  }

  TENRYU_ASSERT(type_ == ArtificialViscosityType::Vnr,
                "ArtificialViscosity::compute_Q_1d requires Vnr, Riemann, Csw, or "
                "RiemannCompatible");
  const int n_nodes = static_cast<int>(node_r.size());
  const int node_blocks = (n_nodes + 255) / 256;
  d_sigma = static_cast<double*>(
      core::device_scratch_acquire("av:sigma_1d", node_r.size() * sizeof(double)));
  compute_node_sigma_1d_kernel<<<node_blocks, 256>>>(d_sigma, node_r.data(), node_u.data(),
                                                     n_nodes, j_);
  cuda_check(cudaGetLastError(), "AV 1D: sigma kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "AV 1D: sigma kernel execution failed");
  switch (geom_code) {
    case 1:
      compute_q_1d_kernel<1><<<blocks, 256>>>(
          Q.data(), rho.data(), vol.data(), node_r.data(), node_u.data(),
          Pe.data(), Pi.data(), cs.data(), d_sigma, d_chi_out, d_q2_out,
          d_div_u_out, d_hydro_active, n_cells, c1_, c2_, d_c1_eff, d_c2_eff,
          eos_aware_, eos_gamma1_ref_, eos_boost_max_);
      break;
    case 2:
      compute_q_1d_kernel<2><<<blocks, 256>>>(
          Q.data(), rho.data(), vol.data(), node_r.data(), node_u.data(),
          Pe.data(), Pi.data(), cs.data(), d_sigma, d_chi_out, d_q2_out,
          d_div_u_out, d_hydro_active, n_cells, c1_, c2_, d_c1_eff, d_c2_eff,
          eos_aware_, eos_gamma1_ref_, eos_boost_max_);
      break;
    default:
      compute_q_1d_kernel<0><<<blocks, 256>>>(
          Q.data(), rho.data(), vol.data(), node_r.data(), node_u.data(),
          Pe.data(), Pi.data(), cs.data(), d_sigma, d_chi_out, d_q2_out,
          d_div_u_out, d_hydro_active, n_cells, c1_, c2_, d_c1_eff, d_c2_eff,
          eos_aware_, eos_gamma1_ref_, eos_boost_max_);
      break;
  }
  cuda_check(cudaGetLastError(), "AV 1D: kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "AV 1D: kernel execution failed");
}

void ArtificialViscosity::compute_H_1d(core::CellField1DView heat_rate,
                                       core::ConstCellField1DView rho,
                                       core::ConstCellField1DView e,
                                       core::ConstCellField1DView chi,
                                       core::ConstNodeField1DView node_r,
                                       const std::int8_t* d_hydro_active,
                                       const int geom_code,
                                       core::ConstCellField1DView heat_C_eff) const {
  TENRYU_ASSERT(type_ != ArtificialViscosityType::Riemann &&
                    type_ != ArtificialViscosityType::RiemannCompatible,
                "ArtificialViscosity::compute_H_1d does not implement Riemann or "
                "RiemannCompatible AV yet");
  TENRYU_ASSERT(heat_rate.size() == rho.size(), "AV heat 1D: heat/rho size mismatch");
  TENRYU_ASSERT(heat_rate.size() == e.size(), "AV heat 1D: heat/e size mismatch");
  TENRYU_ASSERT(heat_rate.size() == chi.size(), "AV heat 1D: heat/chi size mismatch");
  TENRYU_ASSERT(node_r.size() == heat_rate.size() + 1,
                "AV heat 1D: node/cell size mismatch");
  TENRYU_ASSERT(heat_C_eff.empty() || heat_C_eff.size() == heat_rate.size(),
                "AV heat 1D: heat_C_eff size mismatch");

  if (heat_rate.empty()) {
    return;
  }

  const int n_cells = static_cast<int>(heat_rate.size());
  const int blocks = (n_cells + 255) / 256;
  const double* d_heat_C_eff =
      !heat_C_eff.empty() ? heat_C_eff.data() : nullptr;
  switch (geom_code) {
    case 1:
      compute_artificial_heat_1d_kernel<1><<<blocks, 256>>>(
          heat_rate.data(), rho.data(), e.data(), chi.data(), node_r.data(),
          d_hydro_active, n_cells, heat_c_, d_heat_C_eff);
      break;
    case 2:
      compute_artificial_heat_1d_kernel<2><<<blocks, 256>>>(
          heat_rate.data(), rho.data(), e.data(), chi.data(), node_r.data(),
          d_hydro_active, n_cells, heat_c_, d_heat_C_eff);
      break;
    default:
      compute_artificial_heat_1d_kernel<0><<<blocks, 256>>>(
          heat_rate.data(), rho.data(), e.data(), chi.data(), node_r.data(),
          d_hydro_active, n_cells, heat_c_, d_heat_C_eff);
      break;
  }
  cuda_check(cudaGetLastError(), "AV heat 1D: kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "AV heat 1D: kernel execution failed");
}

void ArtificialViscosity::add_bulk_viscosity_1d(
    core::CellField1DView Q,
    core::ConstCellField1DView rho,
    core::ConstCellField1DView vol,
    core::ConstNodeField1DView node_r,
    core::ConstNodeField1DView node_u,
    core::ConstCellField1DView cs,
    const std::int8_t* d_hydro_active,
    core::ConstCellField1DView cbulk_eff) const {
  TENRYU_ASSERT(Q.size() == rho.size(), "Bulk viscosity 1D: Q/rho size mismatch");
  TENRYU_ASSERT(Q.size() == vol.size(), "Bulk viscosity 1D: Q/vol size mismatch");
  TENRYU_ASSERT(Q.size() == cs.size(), "Bulk viscosity 1D: Q/cs size mismatch");
  TENRYU_ASSERT(node_r.size() == node_u.size(),
                "Bulk viscosity 1D: node fields size mismatch");
  TENRYU_ASSERT(node_r.size() == Q.size() + 1,
                "Bulk viscosity 1D: node/cell size mismatch");
  TENRYU_ASSERT(cbulk_eff.empty() || cbulk_eff.size() == Q.size(),
                "Bulk viscosity 1D: cbulk_eff size mismatch");

  if (Q.empty() || (bulk_viscosity_c_ <= 0.0 && cbulk_eff.empty())) {
    return;
  }

  const int n_cells = static_cast<int>(Q.size());
  const int blocks = (n_cells + 255) / 256;
  const double* d_cbulk_eff = !cbulk_eff.empty() ? cbulk_eff.data() : nullptr;
  add_bulk_viscosity_1d_kernel<<<blocks, 256>>>(
      Q.data(), rho.data(), node_r.data(), node_u.data(), cs.data(), d_hydro_active,
      n_cells, bulk_viscosity_c_, d_cbulk_eff);
  cuda_check(cudaGetLastError(), "Bulk viscosity 1D: kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "Bulk viscosity 1D: kernel execution failed");
}

void ArtificialViscosity::compute_Q_2d(core::CellField1D& Q,
                                       const core::CellField1D& rho,
                                       const core::CellField1D& dl,
                                       const core::CellField1D& div_u,
                                       const core::CellField1D& Pe,
                                       const core::CellField1D& Pi,
                                       const core::CellField1D& cs,
                                       const std::vector<std::int8_t>& hydro_active) const {
  TENRYU_ASSERT(Q.size() == rho.size(), "AV 2D: Q/rho size mismatch");
  TENRYU_ASSERT(Q.size() == dl.size(), "AV 2D: Q/dl size mismatch");
  TENRYU_ASSERT(Q.size() == div_u.size(), "AV 2D: Q/div_u size mismatch");
  TENRYU_ASSERT(Q.size() == Pe.size(), "AV 2D: Q/Pe size mismatch");
  TENRYU_ASSERT(Q.size() == Pi.size(), "AV 2D: Q/Pi size mismatch");
  TENRYU_ASSERT(Q.size() == cs.size(), "AV 2D: Q/cs size mismatch");
  TENRYU_ASSERT(hydro_active.empty() || hydro_active.size() == Q.size(),
                "AV 2D: hydro_active size mismatch");
  if (type_ == ArtificialViscosityType::Csw) {
    Q.fill(0.0);
    return;
  }
  TENRYU_ASSERT(type_ == ArtificialViscosityType::Vnr,
                "ArtificialViscosity::compute_Q_2d only supports Vnr");

  if (Q.empty()) {
    return;
  }

  std::int8_t* d_active = nullptr;
  if (!hydro_active.empty()) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "av:active_2d", hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, hydro_active.data(),
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "AV 2D: cudaMemcpy hydro_active failed");
  }

  const int n_cells = static_cast<int>(Q.size());
  const int blocks = (n_cells + 255) / 256;
  compute_q_2d_kernel<<<blocks, 256>>>(Q.data(), rho.data(), dl.data(), div_u.data(),
                                       Pe.data(), Pi.data(), cs.data(), d_active, n_cells,
                                       c1_, c2_, eos_aware_, eos_gamma1_ref_,
                                       eos_boost_max_);
  cuda_check(cudaGetLastError(), "AV 2D: kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "AV 2D: kernel execution failed");
}

void ArtificialViscosity::compute_q_per_material_2d(
    core::CellField1D& Q,
    core::CellField1D& Q_per_material,
    core::State& state,
    const core::CellField1D& dl,
    const core::CellField1D& div_u,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx,
    const std::vector<std::int8_t>& hydro_active) const {
  TENRYU_ASSERT(type_ == ArtificialViscosityType::Vnr,
                "ArtificialViscosity::compute_q_per_material_2d only supports Vnr");
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  TENRYU_ASSERT(n_cells >= 0, "AV per-material 2D: invalid n_cells");
  TENRYU_ASSERT(n_mat > 0, "AV per-material 2D requires at least one material");
  TENRYU_ASSERT(Q.size() == static_cast<std::size_t>(n_cells),
                "AV per-material 2D: Q size mismatch");
  TENRYU_ASSERT(Q_per_material.size() == n_cell_mat,
                "AV per-material 2D: Q_per_material size mismatch");
  TENRYU_ASSERT(dl.size() == static_cast<std::size_t>(n_cells),
                "AV per-material 2D: dl size mismatch");
  TENRYU_ASSERT(div_u.size() == static_cast<std::size_t>(n_cells),
                "AV per-material 2D: div_u size mismatch");
  TENRYU_ASSERT(state.mass_per_material.size() == n_cell_mat,
                "AV per-material 2D: mass_per_material size mismatch");
  TENRYU_ASSERT(state.Ee_per_material.size() == n_cell_mat,
                "AV per-material 2D: Ee_per_material size mismatch");
  TENRYU_ASSERT(state.Ei_per_material.size() == n_cell_mat,
                "AV per-material 2D: Ei_per_material size mismatch");
  TENRYU_ASSERT(state.volFrac.size() == n_cell_mat,
                "AV per-material 2D: volFrac size mismatch");
  TENRYU_ASSERT(state.vol.size() == static_cast<std::size_t>(n_cells),
                "AV per-material 2D: vol size mismatch");
  TENRYU_ASSERT(hydro_active.empty() || hydro_active.size() == static_cast<std::size_t>(n_cells),
                "AV per-material 2D: hydro_active size mismatch");
  if (n_cells == 0) {
    return;
  }

  bool any_table_backed = false;
  for (const auto& mat : cfg.materials.materials) {
    any_table_backed =
        any_table_backed || mat.eos_tables != nullptr || mat.eos_model != "ideal_gas";
  }
  if (any_table_backed) {
    TENRYU_ASSERT(eos_ctx != nullptr,
                  "AV per-material 2D requires HydroEOSContext for table-backed materials");
    TENRYU_ASSERT(eos_ctx->n_materials >= n_mat,
                  "AV per-material 2D EOS context material count mismatch");
  }

  std::int8_t* d_active = nullptr;
  AVMaterialParams* d_params = nullptr;
  std::uint8_t* d_te_valid = nullptr;
  std::uint8_t* d_ti_valid = nullptr;
  unsigned long long* d_counts = nullptr;

  if (!hydro_active.empty()) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "av:active_pm_2d", hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active, hydro_active.data(),
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "AV per-material 2D: cudaMemcpy hydro_active failed");
  }

  const std::vector<AVMaterialParams> h_params = make_av_material_params(cfg);
  d_params = static_cast<AVMaterialParams*>(core::device_scratch_acquire(
      "av:params_pm_2d", h_params.size() * sizeof(AVMaterialParams)));
  cuda_check(cudaMemcpy(d_params,
                        h_params.data(),
                        h_params.size() * sizeof(AVMaterialParams),
                        cudaMemcpyHostToDevice),
             "AV per-material 2D: cudaMemcpy params failed");

  const bool lazy_cache_enabled = cfg.numerics.materials.lazy_cache_te_m_enabled;
  if (lazy_cache_enabled) {
    TENRYU_ASSERT(state.Te_per_material.size() == n_cell_mat,
                  "AV per-material 2D lazy cache requires Te_per_material");
    TENRYU_ASSERT(state.Ti_per_material.size() == n_cell_mat,
                  "AV per-material 2D lazy cache requires Ti_per_material");
    TENRYU_ASSERT(state.Te_per_material_valid.size() == n_cell_mat,
                  "AV per-material 2D lazy cache requires Te valid flags");
    TENRYU_ASSERT(state.Ti_per_material_valid.size() == n_cell_mat,
                  "AV per-material 2D lazy cache requires Ti valid flags");
    d_te_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "av:te_valid_pm_2d", n_cell_mat * sizeof(std::uint8_t)));
    d_ti_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        "av:ti_valid_pm_2d", n_cell_mat * sizeof(std::uint8_t)));
    cuda_check(cudaMemcpy(d_te_valid,
                          state.Te_per_material_valid.data(),
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "AV per-material 2D: copy Te valid failed");
    cuda_check(cudaMemcpy(d_ti_valid,
                          state.Ti_per_material_valid.data(),
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "AV per-material 2D: copy Ti valid failed");
  }
  d_counts = static_cast<unsigned long long*>(core::device_scratch_acquire(
      "av:counts_pm_2d",
      per_material::kPerMaterialCounterCount * sizeof(unsigned long long)));
  cuda_check(cudaMemset(d_counts,
                        0,
                        per_material::kPerMaterialCounterCount *
                            sizeof(unsigned long long)),
             "AV per-material 2D: cudaMemset counts failed");

  cuda_check(cudaMemset(Q.data(), 0, Q.size() * sizeof(double)),
             "AV per-material 2D: cudaMemset Q failed");
  cuda_check(cudaMemset(Q_per_material.data(), 0, Q_per_material.size() * sizeof(double)),
             "AV per-material 2D: cudaMemset Q_per_material failed");

  const auto* d_electron_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                            : nullptr;
  const auto* d_ion_views =
      (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_ion_views
                                                            : nullptr;
  const int blocks = (static_cast<int>(n_cell_mat) + 255) / 256;
  compute_q_per_material_2d_kernel<<<blocks, 256>>>(
      Q.data(),
      Q_per_material.data(),
      state.mass_per_material.data(),
      state.Ee_per_material.data(),
      state.Ei_per_material.data(),
      state.volFrac.data(),
      state.vol.data(),
      lazy_cache_enabled ? state.Te_per_material.data() : nullptr,
      lazy_cache_enabled ? state.Ti_per_material.data() : nullptr,
      d_te_valid,
      d_ti_valid,
      d_counts,
      dl.data(),
      div_u.data(),
      d_electron_views,
      d_ion_views,
      d_params,
      d_active,
      n_cells,
      n_mat,
      c1_,
      c2_,
      eos_aware_,
      eos_gamma1_ref_,
      eos_boost_max_,
      cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      cfg.numerics.materials.presence_threshold_volfrac,
      cfg.numerics.materials.presence_threshold_mass_density_g_per_cc,
      lazy_cache_enabled,
      cfg.materials.low_density_extrapolation);
  cuda_check(cudaGetLastError(), "AV per-material 2D: kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "AV per-material 2D: kernel execution failed");
  state.dispatch_counters.per_material_kernel_call_count.fetch_add(
      1, std::memory_order_relaxed);

  if (lazy_cache_enabled) {
    cuda_check(cudaMemcpy(state.Te_per_material_valid.data(),
                          d_te_valid,
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "AV per-material 2D: copy Te valid back failed");
    cuda_check(cudaMemcpy(state.Ti_per_material_valid.data(),
                          d_ti_valid,
                          n_cell_mat * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "AV per-material 2D: copy Ti valid back failed");
  }
  unsigned long long h_counts[per_material::kPerMaterialCounterCount] = {};
  cuda_check(cudaMemcpy(h_counts,
                        d_counts,
                        per_material::kPerMaterialCounterCount *
                            sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost),
             "AV per-material 2D: copy counts failed");
  state.dispatch_counters.eos_inverse_call_count.fetch_add(
      static_cast<std::uint64_t>(
          h_counts[per_material::kPerMaterialCounterEOSInverse]),
      std::memory_order_relaxed);
  state.dispatch_counters.lazy_cache_te_m_hit_count.fetch_add(
      static_cast<std::uint64_t>(
          h_counts[per_material::kPerMaterialCounterLazyCacheHit]),
      std::memory_order_relaxed);
  state.dispatch_counters.lazy_cache_te_m_miss_count.fetch_add(
      static_cast<std::uint64_t>(
          h_counts[per_material::kPerMaterialCounterLazyCacheMiss]),
      std::memory_order_relaxed);
  state.dispatch_counters.eos_table_validity_violations.fetch_add(
      static_cast<std::uint64_t>(
          h_counts[per_material::kPerMaterialCounterEOSTableValidityViolation]),
      std::memory_order_relaxed);
  state.dispatch_counters.presence_absent_events.fetch_add(
      static_cast<std::uint64_t>(
          h_counts[per_material::kPerMaterialCounterPresenceAbsent]),
      std::memory_order_relaxed);
}

}  // namespace tenryu::hydro
