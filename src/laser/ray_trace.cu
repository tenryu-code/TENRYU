#include "laser/ray_trace.cuh"
#include "laser/ray_trace_bodies.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <type_traits>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "laser/bilinear_interpolation.cuh"
#include "laser/coordinate_transform.cuh"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"

namespace tenryu::laser {
namespace {

constexpr int kMaxRayStepsGuard = 100000;
constexpr double kCritLayerHandoffNhatRaw = 0.5;
// Reachability guard for the analytic critical-layer tail closure
// (2026-07-31 audit): the closure's tail integral assumes the ray runs
// to n_hat = 1 at normal incidence and terminates there. In the local
// planar-linear model a ray turns at 1 - n_hat_t =
// (1 - n_hat_0) sin^2(theta_local), so the closure is valid only when
// that residual gap is inside the critical layer itself. Rays failing
// the guard simply keep marching and turn naturally.
constexpr double kCritReachGapMax = 4.0e-4;
constexpr double kCritLayerHandoffBeta = 1.0;
constexpr double kNearCritMSoftClampStart = 0.80;
constexpr double kNearCritMSoftClampStop = 0.95;
constexpr double kTauTailMax = 700.0;
constexpr int kRayTrace1DBlockSize = 64;
constexpr std::size_t kRayTrace1DSharedBytesCap = 96ULL * 1024ULL;

TENRYU_HOST_DEVICE inline bool outside_mesh(const double R,
                                            const double Z,
                                            const double* node_R,
                                            const double* node_Z,
                                            const int n_nodes_r,
                                            const int n_nodes_z) {
  return (R < 0.0 || R > node_R[n_nodes_r - 1] || Z < node_Z[0] || Z > node_Z[n_nodes_z - 1]);
}

TENRYU_HOST_DEVICE inline void reflect_axis_if_needed(double* R, double* vR) {
  if (*R < 0.0 || (*R == 0.0 && *vR < 0.0)) {
    *R = -*R;
    *vR = -*vR;
  }
}

TENRYU_HOST_DEVICE inline int node_index(const int i, const int j, const int stride) {
  return i * stride + j;
}

TENRYU_HOST_DEVICE inline double clamp_unit_interval(const double t) {
  if (t <= 0.0) {
    return 0.0;
  }
  if (t >= 1.0) {
    return 1.0;
  }
  return t;
}

TENRYU_HOST_DEVICE inline double clamp_adaptive_step_multiplier(const double m_raw,
                                                                const double n_hat_raw,
                                                                const double ds_adapt_max_factor) {
  double m = ::fmax(m_raw, 1.0);
  if (!(n_hat_raw > kNearCritMSoftClampStart)) {
    return m;
  }
  if (n_hat_raw >= kNearCritMSoftClampStop) {
    return 1.0;
  }

  const double ramp = (kNearCritMSoftClampStop - n_hat_raw) /
                      (kNearCritMSoftClampStop - kNearCritMSoftClampStart);
  const double m_cap = ds_adapt_max_factor * ramp;
  return ::fmax(1.0, ::fmin(m, m_cap));
}

TENRYU_HOST_DEVICE inline double reconstruct_tail_entry_smooth_factor(
    const BilinearCell& c,
    const BilinearWeights& w,
    const int n_nodes_z,
    const double* n_hat,
    const double* n_hat_raw,
    const double* T_e,
    const double* Zbar,
    const double* smooth_kappa_factor,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor) {
  const int n00 = node_index(c.i, c.j, n_nodes_z);
  const int n10 = node_index(c.i + 1, c.j, n_nodes_z);
  const int n01 = node_index(c.i, c.j + 1, n_nodes_z);
  const int n11 = node_index(c.i + 1, c.j + 1, n_nodes_z);

  const double m00 = (n_hat_raw[n00] < 1.0) ? w.w00 : 0.0;
  const double m10 = (n_hat_raw[n10] < 1.0) ? w.w10 : 0.0;
  const double m01 = (n_hat_raw[n01] < 1.0) ? w.w01 : 0.0;
  const double m11 = (n_hat_raw[n11] < 1.0) ? w.w11 : 0.0;
  const double wsum = m00 + m10 + m01 + m11;
  if (!(wsum > 0.0)) {
    return 0.0;
  }

  double A_sum = 0.0;
  if (m00 > 0.0) {
    const double A00 = (smooth_kappa_factor != nullptr)
                           ? smooth_kappa_factor[n00]
                           : compute_kappa_smooth_factor(n_hat[n00], T_e[n00], Zbar[n00],
                                                         lambda_cm, eps_n, coulomb_log_floor);
    if (!::isfinite(A00) || !(A00 > 0.0)) {
      return 0.0;
    }
    A_sum += m00 * A00;
  }
  if (m10 > 0.0) {
    const double A10 = (smooth_kappa_factor != nullptr)
                           ? smooth_kappa_factor[n10]
                           : compute_kappa_smooth_factor(n_hat[n10], T_e[n10], Zbar[n10],
                                                         lambda_cm, eps_n, coulomb_log_floor);
    if (!::isfinite(A10) || !(A10 > 0.0)) {
      return 0.0;
    }
    A_sum += m10 * A10;
  }
  if (m01 > 0.0) {
    const double A01 = (smooth_kappa_factor != nullptr)
                           ? smooth_kappa_factor[n01]
                           : compute_kappa_smooth_factor(n_hat[n01], T_e[n01], Zbar[n01],
                                                         lambda_cm, eps_n, coulomb_log_floor);
    if (!::isfinite(A01) || !(A01 > 0.0)) {
      return 0.0;
    }
    A_sum += m01 * A01;
  }
  if (m11 > 0.0) {
    const double A11 = (smooth_kappa_factor != nullptr)
                           ? smooth_kappa_factor[n11]
                           : compute_kappa_smooth_factor(n_hat[n11], T_e[n11], Zbar[n11],
                                                         lambda_cm, eps_n, coulomb_log_floor);
    if (!::isfinite(A11) || !(A11 > 0.0)) {
      return 0.0;
    }
    A_sum += m11 * A11;
  }

  const double A_entry = A_sum / wsum;
  return (::isfinite(A_entry) && A_entry > 0.0) ? A_entry : 0.0;
}

TENRYU_HOST_DEVICE inline bool should_trigger_tail_closure(const double n_hat_raw_entry,
                                                           const double A_entry,
                                                           const double g_mag,
                                                           const double v_dot_g,
                                                           const double v_mag2) {
  if (!(n_hat_raw_entry >= kCritLayerHandoffNhatRaw) || !::isfinite(A_entry) ||
      !(A_entry > 0.0) || !::isfinite(g_mag) || !(g_mag > 0.0)) {
    return false;
  }
  // Direction gate (2026-07-26 review): fire only for rays moving UP the
  // density gradient — the closure models the remaining inbound path to
  // critical. Keep in sync with ray_trace_bodies.cuh.
  if (!(v_dot_g > 0.0)) {
    return false;
  }

  // Obliqueness/reachability: local turning gap in the planar-linear
  // model. sin^2(theta) between v and the gradient direction.
  if (v_mag2 > 0.0 && g_mag > 0.0) {
    const double cos2 =
        (v_dot_g * v_dot_g) / (v_mag2 * (g_mag * g_mag));
    const double sin2 = ::fmax(0.0, 1.0 - ::fmin(1.0, cos2));
    const double reach_gap =
        ::fmax(0.0, 1.0 - n_hat_raw_entry) * sin2;
    if (!(reach_gap < kCritReachGapMax)) {
      return false;
    }
  }

  const double delta_n_to_crit = ::fmax(0.0, 1.0 - n_hat_raw_entry);
  const double handoff_band = kCritLayerHandoffBeta * A_entry / g_mag;
  return ::isfinite(handoff_band) && handoff_band > 0.0 && delta_n_to_crit < handoff_band;
}

TENRYU_HOST_DEVICE inline double first_exit_fraction_2d(const double old_R,
                                                        const double old_Z,
                                                        const double new_R_unreflected,
                                                        const double new_Z,
                                                        const double R_max,
                                                        const double Z_min,
                                                        const double Z_max) {
  const double dR = new_R_unreflected - old_R;
  const double dZ = new_Z - old_Z;
  double t_exit = 1.0;

  if (dZ > 0.0 && new_Z > Z_max) {
    const double t = (Z_max - old_Z) / dZ;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }
  if (dZ < 0.0 && new_Z < Z_min) {
    const double t = (Z_min - old_Z) / dZ;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }
  if (dR > 0.0 && (old_R + dR) > R_max) {
    const double t = (R_max - old_R) / dR;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }
  if (dR < 0.0 && (old_R + dR) < -R_max) {
    const double t = (-R_max - old_R) / dR;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }

  return clamp_unit_interval(t_exit);
}

TENRYU_HOST_DEVICE inline double first_exit_fraction_3d(const double old_x,
                                                        const double old_y,
                                                        const double old_z,
                                                        const double new_x,
                                                        const double new_y,
                                                        const double new_z,
                                                        const double R_max,
                                                        const double Z_min,
                                                        const double Z_max) {
  const double dx = new_x - old_x;
  const double dy = new_y - old_y;
  const double dz = new_z - old_z;
  double t_exit = 1.0;

  if (dz > 0.0 && new_z > Z_max) {
    const double t = (Z_max - old_z) / dz;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }
  if (dz < 0.0 && new_z < Z_min) {
    const double t = (Z_min - old_z) / dz;
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }

  const double r_new2 = new_x * new_x + new_y * new_y;
  const double r_max2 = R_max * R_max;
  if (r_new2 > r_max2) {
    const double a = dx * dx + dy * dy;
    if (a > 0.0) {
      const double b = 2.0 * (old_x * dx + old_y * dy);
      const double c = old_x * old_x + old_y * old_y - r_max2;
      const double disc = b * b - 4.0 * a * c;
      if (disc >= 0.0) {
        const double sqrt_disc = ::sqrt(disc);
        const double inv_2a = 0.5 / a;
        const double t1 = (-b - sqrt_disc) * inv_2a;
        const double t2 = (-b + sqrt_disc) * inv_2a;
        if (t1 >= 0.0 && t1 < t_exit) {
          t_exit = t1;
        }
        if (t2 >= 0.0 && t2 < t_exit) {
          t_exit = t2;
        }
      }
    }
  }

  return clamp_unit_interval(t_exit);
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(value + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

__device__ __forceinline__ void record_tail_closure(
    unsigned long long* tail_closure_count,
    double* tail_closure_absorbed_power,
    double* tail_power_per_ray_slot,
    const double absorbed_power) {
  if (tail_closure_count != nullptr) {
    atomicAdd(tail_closure_count, 1ULL);  // integer atomics are order-safe
  }
  if (absorbed_power > 0.0) {
    if (tail_power_per_ray_slot != nullptr) {
      *tail_power_per_ray_slot += absorbed_power;
    } else if (tail_closure_absorbed_power != nullptr) {
      atomic_add_double(tail_closure_absorbed_power, absorbed_power);
    }
  }
}

__device__ __forceinline__ void record_critical_surface_hit(
    unsigned long long* critical_surface_hit_count) {
  if (critical_surface_hit_count != nullptr) {
    atomicAdd(critical_surface_hit_count, 1ULL);
  }
}

__device__ __forceinline__ double local_ds(const double* node_R,
                                           const double* node_Z,
                                           const int n_nodes_r,
                                           const int n_nodes_z,
                                           const BilinearCell& c,
                                           const double cfl_ray) {
  const int iR = ::max(0, ::min(c.i, n_nodes_r - 2));
  const int jZ = ::max(0, ::min(c.j, n_nodes_z - 2));
  const double dR = node_R[iR + 1] - node_R[iR];
  const double dZ = node_Z[jZ + 1] - node_Z[jZ];
  return cfl_ray * ::fmin(::fmax(dR, 1.0e-30), ::fmax(dZ, 1.0e-30));
}

struct CriticalSurfaceContext {
  const double* node_R = nullptr;
  const double* node_Z = nullptr;
  int n_nodes_r = 0;
  int n_nodes_z = 0;
  const double* n_hat = nullptr;
  const double* n_hat_raw = nullptr;
  const double* T_e = nullptr;
  const double* Zbar = nullptr;
  BilinearCell carried_c;
  double I = 0.0;
  bool traj_on = false;
  int output_idx = -1;
  int traj_stored = 0;
  int* traj_step_count = nullptr;
  double* P_unabsorbed = nullptr;
  unsigned long long* critical_surface_hit_count = nullptr;
  core::DeviceErrorFlags* error_flags = nullptr;
};

__device__ __noinline__ void handle_critical_surface_2d(
    bool& hit_critical,
    double& t_stop,
    double& R_stop_path,
    double& R_stop,
    double& Z_stop,
    BilinearCell& c_stop,
    BilinearWeights& w_stop,
    double& nh_stop,
    double& nh_stop_raw,
    double& Te_stop,
    double& Zbar_stop,
    const double nh_old_raw,
    const double nh_crit,
    const double old_R,
    const double old_Z,
    const double dR_path,
    const double dZ_path,
    const CriticalSurfaceContext& ctx,
    bool& should_return) {
  hit_critical = true;
  record_critical_surface_hit(ctx.critical_surface_hit_count);
  const double nh_denom = nh_stop_raw - nh_old_raw;
  if (nh_denom > 0.0) {
    const double frac = clamp_unit_interval((nh_crit - nh_old_raw) / nh_denom);
    t_stop *= frac;
  } else {
    t_stop = 0.0;
  }
  if (!(t_stop > 0.0)) {
    atomic_add_double(ctx.P_unabsorbed, ctx.I);
    if (ctx.traj_on) {
      ctx.traj_step_count[ctx.output_idx] = ctx.traj_stored;
    }
    should_return = true;
    return;
  }
  R_stop_path = old_R + t_stop * dR_path;
  R_stop = ::abs(R_stop_path);
  Z_stop = old_Z + t_stop * dZ_path;
  c_stop = BilinearInterp::locate_cell_local(ctx.node_R, ctx.node_Z, ctx.n_nodes_r, ctx.n_nodes_z,
                                             R_stop, Z_stop, ctx.carried_c.i, ctx.carried_c.j);
  w_stop = BilinearInterp::compute_weights(c_stop.xi, c_stop.eta);
  nh_stop = BilinearInterp::interpolate(ctx.n_hat, ctx.n_nodes_z, c_stop, w_stop);
  nh_stop_raw = BilinearInterp::interpolate(ctx.n_hat_raw, ctx.n_nodes_z, c_stop, w_stop);
  Te_stop = BilinearInterp::interpolate(ctx.T_e, ctx.n_nodes_z, c_stop, w_stop);
  Zbar_stop = BilinearInterp::interpolate(ctx.Zbar, ctx.n_nodes_z, c_stop, w_stop);
  if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw) || !::isfinite(Te_stop) ||
      !::isfinite(Zbar_stop)) {
    if (ctx.error_flags != nullptr) {
      atomicExch(&ctx.error_flags->invalid_cell, 1);
    }
    atomic_add_double(ctx.P_unabsorbed, ctx.I);
    if (ctx.traj_on) {
      ctx.traj_step_count[ctx.output_idx] = ctx.traj_stored;
    }
    should_return = true;
  }
}

__device__ __noinline__ void handle_critical_surface_3d(
    bool& hit_critical,
    double& t_stop,
    double& x_stop,
    double& y_stop,
    double& z_stop,
    double& R_stop,
    double& Z_stop,
    BilinearCell& c_stop,
    BilinearWeights& w_stop,
    double& nh_stop,
    double& nh_stop_raw,
    double& Te_stop,
    double& Zbar_stop,
    const double nh_old_raw,
    const double nh_crit,
    const double old_x,
    const double old_y,
    const double old_z,
    const double dx_path,
    const double dy_path,
    const double dz_path,
    const CriticalSurfaceContext& ctx,
    bool& should_return) {
  hit_critical = true;
  record_critical_surface_hit(ctx.critical_surface_hit_count);
  const double nh_denom = nh_stop_raw - nh_old_raw;
  if (nh_denom > 0.0) {
    const double frac = clamp_unit_interval((nh_crit - nh_old_raw) / nh_denom);
    t_stop *= frac;
  } else {
    t_stop = 0.0;
  }
  if (!(t_stop > 0.0)) {
    atomic_add_double(ctx.P_unabsorbed, ctx.I);
    if (ctx.traj_on) {
      ctx.traj_step_count[ctx.output_idx] = ctx.traj_stored;
    }
    should_return = true;
    return;
  }
  x_stop = old_x + t_stop * dx_path;
  y_stop = old_y + t_stop * dy_path;
  z_stop = old_z + t_stop * dz_path;
  to_laser_rz(x_stop, y_stop, z_stop, R_stop, Z_stop);
  c_stop = BilinearInterp::locate_cell_local(ctx.node_R, ctx.node_Z, ctx.n_nodes_r, ctx.n_nodes_z,
                                             R_stop, Z_stop, ctx.carried_c.i, ctx.carried_c.j);
  w_stop = BilinearInterp::compute_weights(c_stop.xi, c_stop.eta);
  nh_stop = BilinearInterp::interpolate(ctx.n_hat, ctx.n_nodes_z, c_stop, w_stop);
  nh_stop_raw = BilinearInterp::interpolate(ctx.n_hat_raw, ctx.n_nodes_z, c_stop, w_stop);
  Te_stop = BilinearInterp::interpolate(ctx.T_e, ctx.n_nodes_z, c_stop, w_stop);
  Zbar_stop = BilinearInterp::interpolate(ctx.Zbar, ctx.n_nodes_z, c_stop, w_stop);
  if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw) || !::isfinite(Te_stop) ||
      !::isfinite(Zbar_stop)) {
    if (ctx.error_flags != nullptr) {
      atomicExch(&ctx.error_flags->invalid_cell, 1);
    }
    atomic_add_double(ctx.P_unabsorbed, ctx.I);
    if (ctx.traj_on) {
      ctx.traj_step_count[ctx.output_idx] = ctx.traj_stored;
    }
    should_return = true;
  }
}

struct RayStepHistogramGuard {
  int* step_histogram = nullptr;
  int* n_steps = nullptr;
  int* step_count = nullptr;
  int ray_idx = -1;
  int n_rays_capacity = 0;

  __device__ RayStepHistogramGuard(int* step_histogram_in, int* n_steps_in)
      : step_histogram(step_histogram_in), n_steps(n_steps_in) {}

  __device__ RayStepHistogramGuard(int* step_histogram_in,
                                   int* n_steps_in,
                                   int* step_count_in,
                                   int ray_idx_in,
                                   int n_rays_capacity_in)
      : step_histogram(step_histogram_in),
        n_steps(n_steps_in),
        step_count(step_count_in),
        ray_idx(ray_idx_in),
        n_rays_capacity(n_rays_capacity_in) {}

  __device__ ~RayStepHistogramGuard() {
    const int ray_steps = (n_steps != nullptr && *n_steps > 0) ? *n_steps : 0;
    if (step_histogram != nullptr && n_steps != nullptr) {
      atomicAdd(&step_histogram[0], ray_steps);
      if (ray_steps <= 100) {
        atomicAdd(&step_histogram[1], 1);
      } else if (ray_steps <= 1000) {
        atomicAdd(&step_histogram[2], 1);
      } else if (ray_steps <= 10000) {
        atomicAdd(&step_histogram[3], 1);
      } else {
        atomicAdd(&step_histogram[4], 1);
      }
    }
    if (step_count != nullptr && ray_idx >= 0 && ray_idx < n_rays_capacity) {
      step_count[ray_idx] = ray_steps;
    }
  }
};

struct DepositCacheGuard {
  double* deposit = nullptr;
  int n00 = -1;
  int n10 = -1;
  int n01 = -1;
  int n11 = -1;
  double acc00 = 0.0;
  double acc10 = 0.0;
  double acc01 = 0.0;
  double acc11 = 0.0;

  __device__ explicit DepositCacheGuard(double* deposit_in) : deposit(deposit_in) {}

  __device__ __forceinline__ void flush() {
    if (deposit == nullptr || n00 < 0) {
      return;
    }
    if (acc00 != 0.0) {
      atomic_add_double(&deposit[n00], acc00);
    }
    if (acc10 != 0.0) {
      atomic_add_double(&deposit[n10], acc10);
    }
    if (acc01 != 0.0) {
      atomic_add_double(&deposit[n01], acc01);
    }
    if (acc11 != 0.0) {
      atomic_add_double(&deposit[n11], acc11);
    }
    n00 = -1;
    n10 = -1;
    n01 = -1;
    n11 = -1;
    acc00 = 0.0;
    acc10 = 0.0;
    acc01 = 0.0;
    acc11 = 0.0;
  }

  __device__ __forceinline__ void accumulate(const int in00,
                                             const int in10,
                                             const int in01,
                                             const int in11,
                                             const double v00,
                                             const double v10,
                                             const double v01,
                                             const double v11) {
    if (n00 == in00 && n10 == in10 && n01 == in01 && n11 == in11) {
      acc00 += v00;
      acc10 += v10;
      acc01 += v01;
      acc11 += v11;
      return;
    }
    flush();
    n00 = in00;
    n10 = in10;
    n01 = in01;
    n11 = in11;
    acc00 = v00;
    acc10 = v10;
    acc01 = v01;
    acc11 = v11;
  }

  __device__ ~DepositCacheGuard() { flush(); }
};

__device__ __forceinline__ void accumulate_masked_deposit(DepositCacheGuard& deposit_cache,
                                                          const BilinearCell& c,
                                                          const BilinearWeights& w,
                                                          const int n_nodes_z,
                                                          const double* n_hat_raw,
                                                          const double absorbed_power) {
  if (!(absorbed_power > 0.0)) {
    return;
  }

  const int n00 = node_index(c.i, c.j, n_nodes_z);
  const int n10 = node_index(c.i + 1, c.j, n_nodes_z);
  const int n01 = node_index(c.i, c.j + 1, n_nodes_z);
  const int n11 = node_index(c.i + 1, c.j + 1, n_nodes_z);

  const double nh00 = n_hat_raw[n00];
  const double nh10 = n_hat_raw[n10];
  const double nh01 = n_hat_raw[n01];
  const double nh11 = n_hat_raw[n11];

  const double m00 = (nh00 < 1.0) ? w.w00 : 0.0;
  const double m10 = (nh10 < 1.0) ? w.w10 : 0.0;
  const double m01 = (nh01 < 1.0) ? w.w01 : 0.0;
  const double m11 = (nh11 < 1.0) ? w.w11 : 0.0;

  const double wsum = m00 + m10 + m01 + m11;
  if (wsum > 0.0) {
    const double inv_w = 1.0 / wsum;
    deposit_cache.accumulate(n00, n10, n01, n11, m00 * inv_w * absorbed_power,
                             m10 * inv_w * absorbed_power, m01 * inv_w * absorbed_power,
                             m11 * inv_w * absorbed_power);
    return;
  }
  // All four nodes supercritical: masked weights vanish and this power used
  // to be silently dropped (tallied absorbed, never deposited — 2026-08-30
  // I1 radial gate root cause). Deposit with the plain bilinear weights;
  // the transfer stage's blocked-cell pass re-routes supercritical deposit
  // to the nearest subcritical receiver.
  const double plain_sum = w.w00 + w.w10 + w.w01 + w.w11;
  if (plain_sum > 0.0) {
    const double inv_p = 1.0 / plain_sum;
    deposit_cache.accumulate(n00, n10, n01, n11, w.w00 * inv_p * absorbed_power,
                             w.w10 * inv_p * absorbed_power, w.w01 * inv_p * absorbed_power,
                             w.w11 * inv_p * absorbed_power);
  }
}

struct TailClosureEntryData {
  double u0 = 0.0;
  double A_entry = 0.0;
};

enum class TailClosureMode : unsigned char {
  kRequireTrigger,
  kCriticalCrossing,
};

enum class TailClosureStatus : unsigned char {
  kNoClosure,
  kClosed,
  kInvalid,
};

__device__ __forceinline__ TailClosureEntryData compute_tail_closure_entry(
    const BilinearCell& c,
    const BilinearWeights& w,
    const int n_nodes_z,
    const double* n_hat,
    const double* n_hat_raw,
    const double* T_e,
    const double* Zbar,
    const double* smooth_kappa_factor,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const double test_kappa_cm_inv,
    const double nh_entry,
    const double kappa_entry) {
  TailClosureEntryData entry;
  const double n0 = ::fmin(1.0, ::fmax(0.0, nh_entry));
  entry.u0 = ::sqrt(::fmax(1.0e-30, 1.0 - n0));
  const double n0_sq = ::fmax(n0 * n0, 1.0e-30);
  if (test_kappa_cm_inv > 0.0) {
    if (::isfinite(kappa_entry) && kappa_entry > 0.0) {
      entry.A_entry = kappa_entry * entry.u0 / n0_sq;
    }
    return entry;
  }

  entry.A_entry =
      reconstruct_tail_entry_smooth_factor(c, w, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                                           smooth_kappa_factor, lambda_cm, eps_n,
                                           coulomb_log_floor);
  return entry;
}

__device__ __forceinline__ TailClosureStatus try_tail_closure(
    DepositCacheGuard& deposit_cache,
    const BilinearCell& c,
    const BilinearWeights& w,
    const int n_nodes_z,
    const double* n_hat,
    const double* n_hat_raw,
    const double* T_e,
    const double* Zbar,
    const double* smooth_kappa_factor,
    const double lambda_cm,
    const double eps_n,
    const double coulomb_log_floor,
    const double test_kappa_cm_inv,
    const double nh_entry,
    const double nh_entry_raw,
    const double kappa_entry,
    const double g_mag,
    const double v_dot_g,
    const double v_mag2,
    const double I,
    const TailClosureMode mode,
    unsigned long long* tail_closure_count,
    double* tail_closure_absorbed_power,
    core::DeviceErrorFlags* error_flags,
    double& I_after,
    double* tau_tail_out = nullptr) {
  I_after = I;
  if (!(::isfinite(g_mag) && g_mag > 0.0)) {
    return TailClosureStatus::kNoClosure;
  }

  const TailClosureEntryData entry =
      compute_tail_closure_entry(c, w, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                                 smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                                 test_kappa_cm_inv, nh_entry, kappa_entry);
  const bool should_close = (mode == TailClosureMode::kRequireTrigger)
                                ? should_trigger_tail_closure(
                                      nh_entry_raw, entry.A_entry, g_mag, v_dot_g, v_mag2)
                                : (::isfinite(entry.A_entry) && entry.A_entry > 0.0);
  if (!should_close) {
    return TailClosureStatus::kNoClosure;
  }

  const double u0_sq = entry.u0 * entry.u0;
  const double tau_shape =
      entry.u0 * (1.0 - (2.0 / 3.0) * u0_sq + 0.2 * u0_sq * u0_sq);
  const double tau_tail =
      ::fmin(kTauTailMax, ::fmax(0.0, 2.0 * entry.A_entry / g_mag * tau_shape));
  double I_tail = I;
  const double dP_tail = absorbed_power_expm1(I, tau_tail, I_tail);
  if (!::isfinite(dP_tail) || !::isfinite(I_tail) || dP_tail < 0.0 || I_tail < 0.0) {
    if (mode == TailClosureMode::kRequireTrigger) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      return TailClosureStatus::kInvalid;
    }
    return TailClosureStatus::kNoClosure;
  }

  accumulate_masked_deposit(deposit_cache, c, w, n_nodes_z, n_hat_raw, dP_tail);
  record_tail_closure(tail_closure_count, tail_closure_absorbed_power, nullptr,
                      dP_tail);
  if (tau_tail_out != nullptr) {
    *tau_tail_out = tau_tail;
  }
  I_after = I_tail;
  return TailClosureStatus::kClosed;
}

struct RadialInterval {
  int i = 0;
  double t = 0.0;
};

TENRYU_HOST_DEVICE inline double radial_distance(const double R, const double Z) {
  return ::sqrt(R * R + Z * Z);
}

TENRYU_HOST_DEVICE inline bool outside_radial_profile(const double R,
                                                      const double Z,
                                                      const double* radial_node_r,
                                                      const int n_radial_nodes) {
  return radial_distance(R, Z) > radial_node_r[n_radial_nodes - 1];
}

TENRYU_HOST_DEVICE inline int locate_interval_1d_device(const double* nodes,
                                                        const int n_nodes,
                                                        const double x) {
  if (n_nodes <= 2) {
    return 0;
  }
  if (x <= nodes[0]) {
    return 0;
  }
  if (x >= nodes[n_nodes - 1]) {
    return n_nodes - 2;
  }

  int lo = 0;
  int hi = n_nodes - 1;
  while (hi - lo > 1) {
    const int mid = lo + (hi - lo) / 2;
    if (nodes[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

TENRYU_HOST_DEVICE inline RadialInterval locate_radial_interval(
    const double* radial_node_r,
    const int n_radial_nodes,
    const double r) {
  const int i = locate_interval_1d_device(radial_node_r, n_radial_nodes, r);
  const double r0 = radial_node_r[i];
  const double r1 = radial_node_r[i + 1];
  const double t = (r1 > r0) ? clamp_unit_interval((r - r0) / (r1 - r0)) : 0.0;
  return RadialInterval{i, t};
}

TENRYU_HOST_DEVICE inline double interpolate_radial_field(
    const double* field,
    const RadialInterval& c) {
  return field[c.i] * (1.0 - c.t) + field[c.i + 1] * c.t;
}

TENRYU_HOST_DEVICE inline double local_ds_radial(const double* radial_node_r,
                                                 const RadialInterval& c,
                                                 const double cfl_ray) {
  const double dr = radial_node_r[c.i + 1] - radial_node_r[c.i];
  return cfl_ray * ::fmax(dr, 1.0e-30);
}

TENRYU_HOST_DEVICE inline double first_exit_fraction_radial(const double old_R,
                                                            const double old_Z,
                                                            const double new_R,
                                                            const double new_Z,
                                                            const double r_max) {
  const double dR = new_R - old_R;
  const double dZ = new_Z - old_Z;
  const double a = dR * dR + dZ * dZ;
  if (!(a > 0.0)) {
    return 1.0;
  }

  const double b = 2.0 * (old_R * dR + old_Z * dZ);
  const double c = old_R * old_R + old_Z * old_Z - r_max * r_max;
  const double disc = b * b - 4.0 * a * c;
  if (!(disc >= 0.0)) {
    return 1.0;
  }

  double t_exit = 1.0;
  const double sqrt_disc = ::sqrt(disc);
  const double inv_2a = 0.5 / a;
  const double roots[2] = {(-b - sqrt_disc) * inv_2a, (-b + sqrt_disc) * inv_2a};
  for (const double t : roots) {
    if (t >= 0.0 && t < t_exit) {
      t_exit = t;
    }
  }
  return clamp_unit_interval(t_exit);
}

TENRYU_HOST_DEVICE inline bool advance_to_radial_profile_entry(double* R,
                                                               double* Z,
                                                               double* vR,
                                                               const double vZ,
                                                               const double* radial_node_r,
                                                               const int n_radial_nodes) {
  if (!outside_radial_profile(*R, *Z, radial_node_r, n_radial_nodes)) {
    return true;
  }

  const double r_max = radial_node_r[n_radial_nodes - 1];
  const double a = (*vR) * (*vR) + vZ * vZ;
  if (!(a > 0.0)) {
    return false;
  }

  const double b = 2.0 * ((*R) * (*vR) + (*Z) * vZ);
  const double c = (*R) * (*R) + (*Z) * (*Z) - r_max * r_max;
  const double disc = b * b - 4.0 * a * c;
  if (!(disc > 0.0)) {
    return false;
  }

  const double sqrt_disc = ::sqrt(disc);
  const double s_entry = (-b - sqrt_disc) * (0.5 / a);
  if (!(s_entry > 0.0)) {
    return false;
  }

  const double R_entry_unreflected = *R + (*vR) * s_entry;
  const double Z_entry = *Z + vZ * s_entry;
  if (!::isfinite(R_entry_unreflected) || !::isfinite(Z_entry)) {
    return false;
  }

  *R = R_entry_unreflected;
  *Z = Z_entry;
  reflect_axis_if_needed(R, vR);

  if (outside_radial_profile(*R, *Z, radial_node_r, n_radial_nodes)) {
    const double r_entry = radial_distance(*R, *Z);
    constexpr double kEntryTolRel = 1.0e-12;
    if (!::isfinite(r_entry) || !(r_entry > 0.0) || r_entry > r_max * (1.0 + kEntryTolRel)) {
      return false;
    }
    const double scale = r_max / r_entry;
    *R *= scale;
    *Z *= scale;
  }

  return true;
}

struct DepositCellCacheGuard {
  double* deposit = nullptr;
  int cell = -1;
  double acc = 0.0;

  __device__ explicit DepositCellCacheGuard(double* deposit_in) : deposit(deposit_in) {}

  __device__ __forceinline__ void flush() {
    if (deposit == nullptr || cell < 0 || acc == 0.0) {
      return;
    }
    atomic_add_double(&deposit[cell], acc);
    cell = -1;
    acc = 0.0;
  }

  __device__ __forceinline__ void accumulate(const int c, const double value) {
    if (deposit == nullptr || c < 0 || !(value > 0.0)) {
      return;
    }
    if (cell == c) {
      acc += value;
      return;
    }
    flush();
    cell = c;
    acc = value;
  }

  __device__ ~DepositCellCacheGuard() { flush(); }
};

__device__ __forceinline__ void accumulate_deposit_1d(
    DepositCellCacheGuard& deposit_cache,
    const double* hydro_r_edges,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double r,
    const double absorbed_power) {
  if (!(absorbed_power > 0.0) || hydro_r_edges == nullptr || n_hydro_cells <= 0) {
    return;
  }
  int c = locate_interval_1d_device(hydro_r_edges, n_hydro_cells + 1, r);
  if (critical_adjacent_subcritical_cell == c && allowed_supercritical_cell == c - 1 &&
      critical_adjacent_split_r > hydro_r_edges[c] &&
      critical_adjacent_split_r < hydro_r_edges[c + 1] && r < critical_adjacent_split_r) {
    c = allowed_supercritical_cell;
  }
  deposit_cache.accumulate(c, absorbed_power);
}

TENRYU_HOST_DEVICE inline double reconstruct_tail_entry_smooth_factor_1d(
    const RadialInterval& c,
    const double* radial_smooth_kappa,
    const double* radial_n_hat_raw) {
  if (radial_smooth_kappa == nullptr || radial_n_hat_raw == nullptr) {
    return 0.0;
  }

  const double w0 = 1.0 - c.t;
  const double w1 = c.t;
  const double m0 = (radial_n_hat_raw[c.i] < 1.0) ? w0 : 0.0;
  const double m1 = (radial_n_hat_raw[c.i + 1] < 1.0) ? w1 : 0.0;
  const double wsum = m0 + m1;
  if (!(wsum > 0.0)) {
    return 0.0;
  }

  double A_sum = 0.0;
  if (m0 > 0.0) {
    const double A0 = radial_smooth_kappa[c.i];
    if (!::isfinite(A0) || !(A0 > 0.0)) {
      return 0.0;
    }
    A_sum += m0 * A0;
  }
  if (m1 > 0.0) {
    const double A1 = radial_smooth_kappa[c.i + 1];
    if (!::isfinite(A1) || !(A1 > 0.0)) {
      return 0.0;
    }
    A_sum += m1 * A1;
  }
  const double A_entry = A_sum / wsum;
  return (::isfinite(A_entry) && A_entry > 0.0) ? A_entry : 0.0;
}

__device__ __forceinline__ TailClosureEntryData compute_tail_closure_entry_1d(
    const RadialInterval& c,
    const double* radial_smooth_kappa,
    const double* radial_n_hat_raw,
    const double test_kappa_cm_inv,
    const double nh_entry,
    const double kappa_entry) {
  TailClosureEntryData entry;
  const double n0 = ::fmin(1.0, ::fmax(0.0, nh_entry));
  entry.u0 = ::sqrt(::fmax(1.0e-30, 1.0 - n0));
  const double n0_sq = ::fmax(n0 * n0, 1.0e-30);
  if (test_kappa_cm_inv > 0.0) {
    if (::isfinite(kappa_entry) && kappa_entry > 0.0) {
      entry.A_entry = kappa_entry * entry.u0 / n0_sq;
    }
    return entry;
  }

  entry.A_entry =
      reconstruct_tail_entry_smooth_factor_1d(c, radial_smooth_kappa, radial_n_hat_raw);
  return entry;
}

__device__ __forceinline__ TailClosureStatus try_tail_closure_1d(
    DepositCellCacheGuard& deposit_cache,
    const RadialInterval& c,
    const double* radial_smooth_kappa,
    const double* radial_n_hat_raw,
    const double* hydro_r_edges,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double test_kappa_cm_inv,
    const double nh_entry,
    const double nh_entry_raw,
    const double kappa_entry,
    const double g_mag,
    const double v_dot_g,
    const double v_mag2,
    const double r_entry,
    const double I,
    const TailClosureMode mode,
    unsigned long long* tail_closure_count,
    double* tail_closure_absorbed_power,
    double* tail_power_per_ray_slot,
    core::DeviceErrorFlags* error_flags,
    double& I_after) {
  I_after = I;
  if (!(::isfinite(g_mag) && g_mag > 0.0)) {
    return TailClosureStatus::kNoClosure;
  }

  const TailClosureEntryData entry = compute_tail_closure_entry_1d(
      c, radial_smooth_kappa, radial_n_hat_raw, test_kappa_cm_inv, nh_entry, kappa_entry);
  const bool should_close = (mode == TailClosureMode::kRequireTrigger)
                                ? should_trigger_tail_closure(
                                      nh_entry_raw, entry.A_entry, g_mag, v_dot_g, v_mag2)
                                : (::isfinite(entry.A_entry) && entry.A_entry > 0.0);
  if (!should_close) {
    return TailClosureStatus::kNoClosure;
  }

  const double u0_sq = entry.u0 * entry.u0;
  const double tau_shape =
      entry.u0 * (1.0 - (2.0 / 3.0) * u0_sq + 0.2 * u0_sq * u0_sq);
  const double tau_tail =
      ::fmin(kTauTailMax, ::fmax(0.0, 2.0 * entry.A_entry / g_mag * tau_shape));
  double I_tail = I;
  const double dP_tail = absorbed_power_expm1(I, tau_tail, I_tail);
  if (!::isfinite(dP_tail) || !::isfinite(I_tail) || dP_tail < 0.0 || I_tail < 0.0) {
    if (mode == TailClosureMode::kRequireTrigger) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      return TailClosureStatus::kInvalid;
    }
    return TailClosureStatus::kNoClosure;
  }

  accumulate_deposit_1d(deposit_cache, hydro_r_edges, n_hydro_cells,
                        allowed_supercritical_cell, critical_adjacent_subcritical_cell,
                        critical_adjacent_split_r, r_entry, dP_tail);
  record_tail_closure(tail_closure_count, tail_closure_absorbed_power,
                      tail_power_per_ray_slot, dP_tail);
  I_after = I_tail;
  return TailClosureStatus::kClosed;
}

// ---- CBET v1 recorder helpers (only referenced from the kCbetRecord=true
// instantiation; see docs/design/cbet_1d_v1_design_20260707.md §4.1) ----

// Mirrors the cell-target selection of accumulate_deposit_1d (must stay in sync).
__device__ __forceinline__ int cbet_locate_deposit_cell_1d(
    const double* hydro_r_edges,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double r) {
  if (hydro_r_edges == nullptr || n_hydro_cells <= 0) {
    return -1;
  }
  int c = locate_interval_1d_device(hydro_r_edges, n_hydro_cells + 1, r);
  if (critical_adjacent_subcritical_cell == c && allowed_supercritical_cell == c - 1 &&
      critical_adjacent_split_r > hydro_r_edges[c] &&
      critical_adjacent_split_r < hydro_r_edges[c + 1] && r < critical_adjacent_split_r) {
    c = allowed_supercritical_cell;
  }
  return c;
}

// Mirrors the (cell, tau_tail) outputs of try_tail_closure_1d's kClosed branch
// without depositing (must stay in sync with try_tail_closure_1d).
__device__ __forceinline__ bool cbet_probe_tail_1d(
    const RadialInterval& c,
    const double* radial_smooth_kappa,
    const double* radial_n_hat_raw,
    const double* hydro_r_edges,
    const int n_hydro_cells,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double test_kappa_cm_inv,
    const double nh_entry,
    const double kappa_entry,
    const double g_mag,
    const double r_entry,
    int* cell_out,
    double* tau_out) {
  if (!(::isfinite(g_mag) && g_mag > 0.0)) {
    return false;
  }
  const TailClosureEntryData entry = compute_tail_closure_entry_1d(
      c, radial_smooth_kappa, radial_n_hat_raw, test_kappa_cm_inv, nh_entry, kappa_entry);
  if (!(::isfinite(entry.A_entry) && entry.A_entry > 0.0)) {
    return false;
  }
  const double u0_sq = entry.u0 * entry.u0;
  const double tau_shape =
      entry.u0 * (1.0 - (2.0 / 3.0) * u0_sq + 0.2 * u0_sq * u0_sq);
  const double tau_tail =
      ::fmin(kTauTailMax, ::fmax(0.0, 2.0 * entry.A_entry / g_mag * tau_shape));
  const int cell = cbet_locate_deposit_cell_1d(
      hydro_r_edges, n_hydro_cells, allowed_supercritical_cell,
      critical_adjacent_subcritical_cell, critical_adjacent_split_r, r_entry);
  if (cell < 0) {
    return false;
  }
  *cell_out = cell;
  *tau_out = tau_tail;
  return true;
}

struct CbetRecordCursor {
  std::int32_t* rec_cell = nullptr;
  float* rec_mu = nullptr;
  double* rec_ds = nullptr;
  double* rec_S = nullptr;
  double* rec_w = nullptr;
  std::int32_t* rec_count_out = nullptr;
  std::uint8_t* overflow_out = nullptr;
  long long base = 0;
  int cap = 0;
  int count = 0;
  int cur_cell = -1;
  double ds_acc = 0.0;
  double S_acc = 0.0;
  double muds_acc = 0.0;
  double w_entry = 0.0;
  bool overflowed = false;
  bool active = false;

  __device__ __forceinline__ void mark_overflow() {
    overflowed = true;
    if (overflow_out != nullptr) {
      *overflow_out = 1;
    }
    cur_cell = -1;
    ds_acc = 0.0;
    S_acc = 0.0;
    muds_acc = 0.0;
  }

  __device__ __forceinline__ void flush() {
    if (!active || cur_cell < 0) {
      return;
    }
    if (count >= cap) {
      mark_overflow();
      return;
    }
    const long long slot = base + count;
    rec_cell[slot] = cur_cell;
    const double mu =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(-1.0, muds_acc / ds_acc)) : 0.0;
    rec_mu[slot] = static_cast<float>(mu);
    rec_ds[slot] = ds_acc;
    rec_S[slot] = S_acc;
    rec_w[slot] = w_entry;
    ++count;
    *rec_count_out = count;
    cur_cell = -1;
    ds_acc = 0.0;
    S_acc = 0.0;
    muds_acc = 0.0;
  }

  __device__ __forceinline__ void add_segment(const int cell,
                                              const double mu,
                                              const double ds,
                                              const double S,
                                              const double I_entry) {
    if (!active || overflowed || cell < 0) {
      return;
    }
    if (cell != cur_cell) {
      flush();
      if (overflowed) {
        return;
      }
      cur_cell = cell;
      w_entry = I_entry;
    }
    ds_acc += ds;
    S_acc += S;
    muds_acc += mu * ds;
  }

  __device__ __forceinline__ void add_terminal(const int cell, const double tau) {
    if (!active) {
      return;
    }
    flush();
    if (overflowed || cell < 0) {
      return;
    }
    if (count >= cap) {
      mark_overflow();
      return;
    }
    const long long slot = base + count;
    rec_cell[slot] = cell;
    rec_mu[slot] = 0.0f;
    rec_ds[slot] = 0.0;
    rec_S[slot] = tau;
    rec_w[slot] = 0.0;
    ++count;
    *rec_count_out = count;
  }
};

struct CbetRecordCursor2D {
  std::int32_t* rec_cell = nullptr;
  float* rec_mu = nullptr;   // stores a = dR/ds (cylindrical-radial cosine)
  float* rec_c = nullptr;    // stores c = dZ/ds (axial cosine)
  float* rec_w00 = nullptr;
  float* rec_w10 = nullptr;
  float* rec_w01 = nullptr;
  double* rec_ds = nullptr;
  double* rec_S = nullptr;
  double* rec_w = nullptr;
  std::int32_t* rec_count_out = nullptr;
  std::uint8_t* overflow_out = nullptr;
  long long base = 0;
  int cap = 0;
  int count = 0;
  int cur_cell = -1;
  double ds_acc = 0.0;
  double S_acc = 0.0;
  double a_ds_acc = 0.0;
  double c_ds_acc = 0.0;
  double w00_ds_acc = 0.0;
  double w10_ds_acc = 0.0;
  double w01_ds_acc = 0.0;
  double w_entry = 0.0;
  bool overflowed = false;
  bool active = false;

  __device__ __forceinline__ void mark_overflow() {
    overflowed = true;
    if (overflow_out != nullptr) {
      *overflow_out = 1;
    }
    cur_cell = -1;
    ds_acc = 0.0;
    S_acc = 0.0;
    a_ds_acc = 0.0;
    c_ds_acc = 0.0;
    w00_ds_acc = 0.0;
    w10_ds_acc = 0.0;
    w01_ds_acc = 0.0;
  }

  __device__ __forceinline__ void flush() {
    if (!active || cur_cell < 0) {
      return;
    }
    if (count >= cap) {
      mark_overflow();
      return;
    }
    const long long slot = base + count;
    rec_cell[slot] = cur_cell;
    const double a =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(-1.0, a_ds_acc / ds_acc)) : 0.0;
    const double c =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(-1.0, c_ds_acc / ds_acc)) : 0.0;
    const double w00 =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(0.0, w00_ds_acc / ds_acc)) : 0.0;
    const double w10 =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(0.0, w10_ds_acc / ds_acc)) : 0.0;
    const double w01 =
        (ds_acc > 0.0) ? ::fmin(1.0, ::fmax(0.0, w01_ds_acc / ds_acc)) : 0.0;
    rec_mu[slot] = static_cast<float>(a);
    rec_c[slot] = static_cast<float>(c);
    rec_w00[slot] = static_cast<float>(w00);
    rec_w10[slot] = static_cast<float>(w10);
    rec_w01[slot] = static_cast<float>(w01);
    rec_ds[slot] = ds_acc;
    rec_S[slot] = S_acc;
    rec_w[slot] = w_entry;
    ++count;
    *rec_count_out = count;
    cur_cell = -1;
    ds_acc = 0.0;
    S_acc = 0.0;
    a_ds_acc = 0.0;
    c_ds_acc = 0.0;
    w00_ds_acc = 0.0;
    w10_ds_acc = 0.0;
    w01_ds_acc = 0.0;
  }

  __device__ __forceinline__ void add_segment(const int cell,
                                              const double ds,
                                              const double S,
                                              const double a,
                                              const double c,
                                              const double w00,
                                              const double w10,
                                              const double w01,
                                              const double I_entry) {
    if (!active || overflowed || cell < 0) {
      return;
    }
    if (cell != cur_cell) {
      flush();
      if (overflowed) {
        return;
      }
      cur_cell = cell;
      w_entry = I_entry;
    }
    ds_acc += ds;
    S_acc += S;
    a_ds_acc += a * ds;
    c_ds_acc += c * ds;
    w00_ds_acc += w00 * ds;
    w10_ds_acc += w10 * ds;
    w01_ds_acc += w01 * ds;
  }

  __device__ __forceinline__ void append_terminal(const int cell,
                                                  const double S_tail,
                                                  const double w00,
                                                  const double w10,
                                                  const double w01,
                                                  const double I_entry) {
    if (!active) {
      return;
    }
    flush();
    if (overflowed || cell < 0) {
      return;
    }
    if (count >= cap) {
      mark_overflow();
      return;
    }
    const long long slot = base + count;
    rec_cell[slot] = cell;
    rec_mu[slot] = 0.0f;
    rec_c[slot] = 0.0f;
    rec_w00[slot] = static_cast<float>(::fmin(1.0, ::fmax(0.0, w00)));
    rec_w10[slot] = static_cast<float>(::fmin(1.0, ::fmax(0.0, w10)));
    rec_w01[slot] = static_cast<float>(::fmin(1.0, ::fmax(0.0, w01)));
    rec_ds[slot] = 0.0;
    rec_S[slot] = S_tail;
    rec_w[slot] = I_entry;
    ++count;
    *rec_count_out = count;
  }
};

}  // namespace

template <bool kCbetRecord, bool kHotECapture, bool kPhysExt>
__global__ __launch_bounds__(kRayTrace1DBlockSize)
void ray_trace_1d_sph(double* __restrict__ deposit_1d,
                      double* __restrict__ deposit_per_ray,
                      double* __restrict__ unabsorbed_per_ray,
                      double* __restrict__ tail_power_per_ray,
                      const double* __restrict__ radial_node_r,
                      const double* __restrict__ radial_n_hat,
                      const double* __restrict__ radial_n_hat_raw,
                      const double* __restrict__ radial_smooth_kappa,
                      const double* __restrict__ radial_dn_dr,
                      const double* __restrict__ hydro_r_edges,
                      const int allowed_supercritical_cell,
                      const int critical_adjacent_subcritical_cell,
                      const double critical_adjacent_split_r,
                      const double* __restrict__ ray_R0,
                      const double* __restrict__ ray_Z0,
                      const double* __restrict__ ray_vR0,
                      const double* __restrict__ ray_vZ0,
                      const double* __restrict__ ray_power,
                      const double* __restrict__ ray_power0,
                      const double cfl_ray,
                      const double ds_adapt_g_target,
                      const double ds_adapt_tau_target,
                      const double ds_adapt_theta_target,
                      const double ds_adapt_max_factor,
                      const double eps_n,
                      const double eps_crit,
                      const double lambda_cm,
                      const double coulomb_log_floor,
                      const double test_kappa_cm_inv,
                      const double intensity_cutoff,
                      const int max_ray_steps,
                      const int n_radial_nodes,
                      const int n_hydro_cells,
                      const int n_rays,
                      const int use_shared_staging,
                      double* __restrict__ traj_pos_R,
                      double* __restrict__ traj_pos_Z,
                      double* __restrict__ traj_power,
                      int* __restrict__ traj_step_count,
                      const int n_output_rays,
                      const int output_stride,
                      const int traj_max_steps,
                      int* __restrict__ step_histogram,
                      int* __restrict__ step_count,
                      const int* __restrict__ ray_order,
                      int* __restrict__ ray_steps_out,
                      double* __restrict__ P_unabsorbed,
                      unsigned long long* __restrict__ tail_closure_count,
                      double* __restrict__ tail_closure_absorbed_power,
                      unsigned long long* __restrict__ critical_surface_hit_count,
                      core::DeviceErrorFlags* __restrict__ error_flags,
                      const CbetRecordDeviceArgs cbet_args,
                      const HotECaptureParams hot_e_params,
                      double* __restrict__ hot_e_capture,  // rows [(tid*n_channels + ch)*4 + {0:valid,1:r_s,2:mu_axis,3:P_before}]
                      const laser::LaserPhysExtOptions phys_opt,
                      const double* __restrict__ radial_T_e,
                      double* __restrict__ ra_per_ray,
                      double* __restrict__ tau_shell_out
                      ) {
  extern __shared__ double rt_smem[];
  // Stage the per-cell arrays into shared memory (bit-exact copies;
  // 2026-07-31 perf lane D). Layout: consecutive slabs of n_slab doubles in
  // this fixed order: radial_node_r, radial_n_hat, radial_n_hat_raw,
  // radial_smooth_kappa, radial_dn_dr, hydro_r_edges, radial_T_e.
  const int n_slab =
      (n_radial_nodes > n_hydro_cells + 1) ? n_radial_nodes
                                           : n_hydro_cells + 1;
  const int n_stage_arrays = (radial_T_e != nullptr) ? 7 : 6;
  const std::size_t total_stage_doubles =
      static_cast<std::size_t>(n_stage_arrays) *
      static_cast<std::size_t>(n_slab);
  const bool stage_per_cell = use_shared_staging != 0;

  const double* body_radial_node_r = radial_node_r;
  const double* body_radial_n_hat = radial_n_hat;
  const double* body_radial_n_hat_raw = radial_n_hat_raw;
  const double* body_radial_smooth_kappa = radial_smooth_kappa;
  const double* body_radial_dn_dr = radial_dn_dr;
  const double* body_hydro_r_edges = hydro_r_edges;
  const double* body_radial_T_e = radial_T_e;
  if (stage_per_cell) {
    double* s_radial_node_r = rt_smem + 0 * n_slab;
    double* s_radial_n_hat = rt_smem + 1 * n_slab;
    double* s_radial_n_hat_raw = rt_smem + 2 * n_slab;
    double* s_radial_smooth_kappa = rt_smem + 3 * n_slab;
    double* s_radial_dn_dr = rt_smem + 4 * n_slab;
    double* s_hydro_r_edges = rt_smem + 5 * n_slab;
    double* s_radial_T_e = rt_smem + 6 * n_slab;
    const double* stage_srcs[7] = {
        radial_node_r, radial_n_hat, radial_n_hat_raw, radial_smooth_kappa,
        radial_dn_dr, hydro_r_edges, radial_T_e};
    const int stage_lengths[7] = {
        n_radial_nodes, n_radial_nodes, n_radial_nodes, n_radial_nodes,
        n_radial_nodes, n_hydro_cells + 1, n_radial_nodes};
    for (std::size_t i = threadIdx.x; i < total_stage_doubles;
         i += blockDim.x) {
      const int f = static_cast<int>(i / n_slab);
      const int c = static_cast<int>(i) - f * n_slab;
      if (stage_srcs[f] != nullptr && c < stage_lengths[f]) {
        rt_smem[i] = stage_srcs[f][c];
      }
    }
    __syncthreads();
    body_radial_node_r =
        (radial_node_r != nullptr) ? s_radial_node_r : radial_node_r;
    body_radial_n_hat =
        (radial_n_hat != nullptr) ? s_radial_n_hat : radial_n_hat;
    body_radial_n_hat_raw =
        (radial_n_hat_raw != nullptr) ? s_radial_n_hat_raw
                                      : radial_n_hat_raw;
    body_radial_smooth_kappa =
        (radial_smooth_kappa != nullptr) ? s_radial_smooth_kappa
                                         : radial_smooth_kappa;
    body_radial_dn_dr =
        (radial_dn_dr != nullptr) ? s_radial_dn_dr : radial_dn_dr;
    body_hydro_r_edges =
        (hydro_r_edges != nullptr) ? s_hydro_r_edges : hydro_r_edges;
    body_radial_T_e =
        (radial_T_e != nullptr) ? s_radial_T_e : radial_T_e;
  }

  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_rays) {
    return;
  }
  const int ray = (ray_order != nullptr) ? ray_order[t] : t;
  ray_trace_bodies::ray_trace_1d_sph_body<kCbetRecord, kHotECapture, kPhysExt>(
      ray, deposit_1d, deposit_per_ray, unabsorbed_per_ray, tail_power_per_ray,
      body_radial_node_r, body_radial_n_hat, body_radial_n_hat_raw,
      body_radial_smooth_kappa, body_radial_dn_dr, body_hydro_r_edges,
      allowed_supercritical_cell,
      critical_adjacent_subcritical_cell, critical_adjacent_split_r, ray_R0,
      ray_Z0, ray_vR0, ray_vZ0, ray_power, ray_power0, cfl_ray,
      ds_adapt_g_target, ds_adapt_tau_target, ds_adapt_theta_target,
      ds_adapt_max_factor, eps_n, eps_crit, lambda_cm, coulomb_log_floor, test_kappa_cm_inv,
      intensity_cutoff, max_ray_steps, n_radial_nodes, n_hydro_cells,
      n_rays, traj_pos_R, traj_pos_Z, traj_power, traj_step_count,
      n_output_rays, output_stride, traj_max_steps, step_histogram, step_count,
      ray_steps_out, P_unabsorbed, tail_closure_count, tail_closure_absorbed_power,
      critical_surface_hit_count, error_flags, cbet_args, hot_e_params,
      hot_e_capture, phys_opt, body_radial_T_e, ra_per_ray, tau_shell_out);
}


template <bool kHotECapture>
__global__ __launch_bounds__(1, 1)
void radial_absorption_1d_kernel(double P_total,
                                 const double* __restrict__ hydro_r_edges,
                                 const double* __restrict__ radial_node_r,
                                 const double* __restrict__ radial_n_hat,
                                 const double* __restrict__ radial_n_hat_raw,
                                 const double* __restrict__ radial_smooth_kappa,
                                 const double eps_n,
                                 const double eps_crit,
                                 const double test_kappa_cm_inv,
                                 const double intensity_cutoff,
                                 const int n_hydro_cells,
                                 const int n_radial_nodes,
                                 double* __restrict__ deposit_power_cell,
                                 double* __restrict__ P_unabsorbed,
                                 unsigned long long* __restrict__ critical_surface_hit_count,
                                 core::DeviceErrorFlags* __restrict__ error_flags,
                                 const HotECaptureParams hot_e_params,
                                 double* __restrict__ hot_e_capture   // [ch*3 + {0:valid,1:r_s,2:P_before}]
                                 ) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  if (!::isfinite(P_total) || !(P_total > 0.0)) {
    if (!::isfinite(P_total) && error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    return;
  }
  if (hydro_r_edges == nullptr || radial_node_r == nullptr || radial_n_hat == nullptr ||
      radial_n_hat_raw == nullptr || radial_smooth_kappa == nullptr ||
      deposit_power_cell == nullptr || n_hydro_cells <= 0 || n_radial_nodes < 2) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->invalid_cell, 1);
    }
    if (P_unabsorbed != nullptr) {
      atomic_add_double(P_unabsorbed, P_total);
    }
    return;
  }

  double P = P_total;
  [[maybe_unused]] unsigned hot_e_captured_mask = 0u;
  const double nh_crit = 1.0 - eps_crit;
  const double cutoff_power =
      (::isfinite(intensity_cutoff) && intensity_cutoff > 0.0)
          ? intensity_cutoff * P_total
          : -1.0;

  for (int c = n_hydro_cells - 1; c >= 0; --c) {
    if (cutoff_power > 0.0 && P < cutoff_power) {
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    const double r0 = hydro_r_edges[c];
    const double r1 = hydro_r_edges[c + 1];
    const double dr = r1 - r0;
    if (!::isfinite(r0) || !::isfinite(r1) || !(dr > 0.0)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    const double r_mid = 0.5 * (r0 + r1);
    const RadialInterval radial_cell =
        locate_radial_interval(radial_node_r, n_radial_nodes, r_mid);
    const double nh = interpolate_radial_field(radial_n_hat, radial_cell);
    const double nh_raw = interpolate_radial_field(radial_n_hat_raw, radial_cell);
    if (!::isfinite(nh) || !::isfinite(nh_raw)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    if constexpr (kHotECapture) {
      for (int he_ch = 0; he_ch < hot_e_params.n_channels; ++he_ch) {
        if ((hot_e_captured_mask & (1u << he_ch)) != 0u) {
          continue;
        }
        if (!(nh_raw >= hot_e_params.threshold_nhat[he_ch])) {
          continue;
        }
        hot_e_captured_mask |= (1u << he_ch);
        hot_e_capture[he_ch * 3 + 0] = 1.0;
        hot_e_capture[he_ch * 3 + 1] = r1;  // outer edge of the crossing cell (launch radius)
        hot_e_capture[he_ch * 3 + 2] = P;   // ray power remaining at the crossing
        P *= hot_e_params.one_minus_eta[he_ch];
      }
    }

    if (nh_raw >= nh_crit) {
      record_critical_surface_hit(critical_surface_hit_count);
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    double kappa = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa = test_kappa_cm_inv;
    } else {
      const double smooth_factor = interpolate_radial_field(radial_smooth_kappa, radial_cell);
      kappa = compute_kappa_from_smooth(smooth_factor, nh, eps_n);
    }
    if (!::isfinite(kappa) || kappa < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    const double tau = compute_optical_depth(kappa, kappa, dr);
    if (!::isfinite(tau) || tau < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    double P_next = P;
    const double dP = absorbed_power_expm1(P, tau, P_next);
    if (!::isfinite(dP) || !::isfinite(P_next) || dP < 0.0 || P_next < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      if (P_unabsorbed != nullptr) {
        atomic_add_double(P_unabsorbed, P);
      }
      return;
    }

    if (dP > 0.0) {
      deposit_power_cell[c] += dP;
    }
    P = P_next;
  }

  if (P > 0.0 && P_unabsorbed != nullptr) {
    atomic_add_double(P_unabsorbed, P);
  }
}

// Explicit instantiations: keep the templated __global__ entries as STRONG
// device symbols so nvlink retains them in every target's device link
// (weak template entries are dropped from test binaries that only call the
// host launchers; dropping manifests as cudaErrorInvalidDeviceFunction).
template __global__ void radial_absorption_1d_kernel<false>(
    double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double, const double, const double, const double, const int, const int,
    double* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const HotECaptureParams, double* __restrict__);
template __global__ void radial_absorption_1d_kernel<true>(
    double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double, const double, const double, const double, const int, const int,
    double* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const HotECaptureParams, double* __restrict__);

template __global__ void ray_trace_1d_sph<false, false, false>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<false, true, false>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<true, false, false>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<true, true, false>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);

template __global__ void ray_trace_1d_sph<false, false, true>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<false, true, true>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<true, false, true>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);
template __global__ void ray_trace_1d_sph<true, true, true>(
    double* __restrict__, double* __restrict__, double* __restrict__, double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const int, const int, const double, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double* __restrict__, const double* __restrict__,
    const double* __restrict__, const double, const double, const double, const double,
    const double, const double, const double, const double, const double, const double,
    const double,
    const int, const int, const int, const int, const int, double* __restrict__, double* __restrict__,
    double* __restrict__, int* __restrict__, const int, const int, const int,
    int* __restrict__, int* __restrict__, const int* __restrict__, int* __restrict__,
    double* __restrict__,
    unsigned long long* __restrict__, double* __restrict__, unsigned long long* __restrict__,
    core::DeviceErrorFlags* __restrict__, const CbetRecordDeviceArgs,
    const HotECaptureParams, double* __restrict__, const laser::LaserPhysExtOptions,
    const double* __restrict__, double* __restrict__, double* __restrict__);

__global__ __launch_bounds__(64)
void ray_trace_2d(double* __restrict__ deposit,
                  const double* __restrict__ n_hat,
                  const double* __restrict__ n_hat_raw,
                  const double* __restrict__ grad_n_hat_R,
                  const double* __restrict__ grad_n_hat_Z,
                  const double* __restrict__ T_e,
                  const double* __restrict__ Zbar,
                  const double* __restrict__ smooth_kappa_factor,
                  const double* __restrict__ node_R,
                  const double* __restrict__ node_Z,
                  const double* __restrict__ ray_R0,
                  const double* __restrict__ ray_Z0,
                  const double* __restrict__ ray_vR0,
                  const double* __restrict__ ray_vZ0,
                  const double* __restrict__ ray_power,
                  const double* __restrict__ ray_power0,
                  const double cfl_ray,
                  const double ds_adapt_g_target,
                  const double ds_adapt_tau_target,
                  const double ds_adapt_theta_target,
                  const double ds_adapt_max_factor,
                  const double eps_n,
                  const double eps_crit,
                  const double lambda_cm,
                  const double coulomb_log_floor,
                  const double test_kappa_cm_inv,
                  const double intensity_cutoff,
                  const int max_ray_steps,
                  const int n_nodes_r,
                  const int n_nodes_z,
                  const int n_rays,
                  double* __restrict__ traj_pos_R,
                  double* __restrict__ traj_pos_Z,
                  double* __restrict__ traj_power,
                  int* __restrict__ traj_step_count,
                  const int n_output_rays,
                  const int output_stride,
                  const int traj_max_steps,
                  int* __restrict__ step_histogram,
                  int* __restrict__ step_count,
                  double* __restrict__ P_unabsorbed,
                  unsigned long long* __restrict__ tail_closure_count,
                  double* __restrict__ tail_closure_absorbed_power,
                  unsigned long long* __restrict__ critical_surface_hit_count,
                  core::DeviceErrorFlags* __restrict__ error_flags) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_idx = (output_stride > 0) ? (tid / output_stride) : -1;
  const bool traj_on = (traj_pos_R != nullptr) && (output_stride > 0) &&
                       (tid % output_stride == 0) && (output_idx < n_output_rays);
  int traj_stored = 0;
  int n_steps = 0;
  if (tid >= n_rays) {
    return;
  }
  RayStepHistogramGuard histogram_guard(step_histogram, &n_steps, step_count, tid, n_rays);
  DepositCacheGuard deposit_cache(deposit);

  double R = ray_R0[tid];
  double Z = ray_Z0[tid];
  double vR = ray_vR0[tid];
  double vZ = ray_vZ0[tid];
  double I = ray_power[tid];
  const double I0 = ray_power0[tid];

  const bool finite_init = ::isfinite(R) && ::isfinite(Z) && ::isfinite(vR) && ::isfinite(vZ) &&
                           ::isfinite(I) && ::isfinite(I0);
  if (!finite_init) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (::isfinite(I) && I > 0.0) {
      atomic_add_double(P_unabsorbed, I);
    }
    if (traj_on) {
      traj_step_count[output_idx] = 0;
    }
    return;
  }

  if (!(I > 0.0)) {
    if (traj_on) {
      traj_step_count[output_idx] = 0;
    }
    return;
  }

  reflect_axis_if_needed(&R, &vR);
  if (traj_on && traj_stored < traj_max_steps) {
    const int idx = output_idx * traj_max_steps + traj_stored;
    traj_pos_R[idx] = R;
    traj_pos_Z[idx] = Z;
    traj_power[idx] = I;
    ++traj_stored;
  }
  if (outside_mesh(R, Z, node_R, node_Z, n_nodes_r, n_nodes_z)) {
    atomic_add_double(P_unabsorbed, I);
    if (traj_on) {
      traj_step_count[output_idx] = traj_stored;
    }
    return;
  }

  BilinearCell carried_c;
  double carried_nh = 0.0;
  double carried_nh_raw = 0.0;
  double carried_Te = 0.0;
  double carried_Zbar = 0.0;
  double carried_kappa = 0.0;
  {
    carried_c = BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, R, Z);
    const double ds_local = local_ds(node_R, node_Z, n_nodes_r, n_nodes_z, carried_c, cfl_ray);
    const BilinearWeights w = BilinearInterp::compute_weights(carried_c.xi, carried_c.eta);
    const double nh0 = BilinearInterp::interpolate(n_hat, n_nodes_z, carried_c, w);
    const double nh0_raw = BilinearInterp::interpolate(n_hat_raw, n_nodes_z, carried_c, w);
    const double Te0 = BilinearInterp::interpolate(T_e, n_nodes_z, carried_c, w);
    const double Z0 = BilinearInterp::interpolate(Zbar, n_nodes_z, carried_c, w);
    if (!::isfinite(nh0)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh0 >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    double kappa0 = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa0 = test_kappa_cm_inv;
    } else if (smooth_kappa_factor != nullptr) {
      const double A0 = BilinearInterp::interpolate(smooth_kappa_factor, n_nodes_z, carried_c, w);
      kappa0 = compute_kappa_from_smooth(A0, nh0, eps_n);
    } else {
      kappa0 = compute_kappa_ib(nh0, Te0, Z0, lambda_cm, eps_n, coulomb_log_floor);
    }
    double gR = 0.0;
    double gZ = 0.0;
    BilinearInterp::interpolate_gradient(grad_n_hat_R, grad_n_hat_Z, n_nodes_z, carried_c, w, gR,
                                         gZ);
    if (!::isfinite(gR) || !::isfinite(gZ)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    // Launch speed |v| = sqrt(1 - n_hat) at the start point (H = 1;
    // 2026-07-26 review). Vacuum starts (nh0 = 0) multiply by exactly 1.0.
    {
      const double v_entry_scale = ::sqrt(::fmax(0.0, 1.0 - nh0));
      vR *= v_entry_scale;
      vZ *= v_entry_scale;
    }
    // Leapfrog seeding v^{1/2} = v^0 + (h/2) a with a = -(1/2) grad n_hat,
    // i.e. v -= (h/4) grad n_hat, matching the main-loop kick sign
    // (2026-07-31 audit; the earlier '+' was inconsistent with the loop's
    // `v -= 0.5*ds*g`).
    {
      const double g_mag0 = ::sqrt(gR * gR + gZ * gZ);
      const double g_metric0 = g_mag0 * ds_local;
      const double tau_metric0 = kappa0 * ds_local;
      double m0 = ::fmin(::fmin(ds_adapt_g_target / ::fmax(g_metric0, 1.0e-30),
                                ds_adapt_tau_target / ::fmax(tau_metric0, 1.0e-30)),
                         ds_adapt_max_factor);
      m0 = clamp_adaptive_step_multiplier(m0, nh0_raw, ds_adapt_max_factor);
      m0 = ::fmin(m0, ray_trace_bodies::theta_step_multiplier(
                           g_metric0, nh0_raw, ds_adapt_theta_target));
      const double ds_first = ds_local * m0;
      vR -= 0.25 * ds_first * gR;
      vZ -= 0.25 * ds_first * gZ;
    }
    if (!::isfinite(vR) || !::isfinite(vZ)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    carried_nh = nh0;
    carried_nh_raw = nh0_raw;
    carried_Te = Te0;
    carried_Zbar = Z0;
    carried_kappa = kappa0;
  }

  while (n_steps < max_ray_steps) {
    ++n_steps;
    const double old_R = R;
    const double old_Z = Z;

    const BilinearCell c_old = carried_c;
    const BilinearWeights w_old = BilinearInterp::compute_weights(c_old.xi, c_old.eta);
    const double ds_local = local_ds(node_R, node_Z, n_nodes_r, n_nodes_z, c_old, cfl_ray);

    double gR = 0.0;
    double gZ = 0.0;
    BilinearInterp::interpolate_gradient(grad_n_hat_R, grad_n_hat_Z, n_nodes_z, c_old, w_old,
                                         gR, gZ);
    const double g_mag = ::sqrt(gR * gR + gZ * gZ);
    const double nh_entry = carried_nh;
    const double nh_entry_raw = carried_nh_raw;
    const double kappa_entry = carried_kappa;
    if (!::isfinite(nh_entry) || !::isfinite(nh_entry_raw) || !::isfinite(carried_Te) ||
        !::isfinite(carried_Zbar)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_entry >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_entry_raw >= kCritLayerHandoffNhatRaw) {
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_entry, nh_entry_raw, kappa_entry, g_mag,
                           vR * gR + vZ * gZ, v_mag2, I,
                           TailClosureMode::kRequireTrigger, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail);
      if (tail_status == TailClosureStatus::kInvalid) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
      if (tail_status == TailClosureStatus::kClosed) {
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_R[idx] = R;
          traj_pos_Z[idx] = Z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
    }
    const double g_metric = g_mag * ds_local;
    const double tau_metric = carried_kappa * ds_local;
    double m = ::fmin(::fmin(ds_adapt_g_target / ::fmax(g_metric, 1.0e-30),
                             ds_adapt_tau_target / ::fmax(tau_metric, 1.0e-30)),
                      ds_adapt_max_factor);
    m = clamp_adaptive_step_multiplier(m, carried_nh_raw, ds_adapt_max_factor);
    m = ::fmin(m, ray_trace_bodies::theta_step_multiplier(
                         g_metric, carried_nh_raw, ds_adapt_theta_target));
    const double ds_cur = ds_local * m;

    vR -= 0.5 * ds_cur * gR;
    vZ -= 0.5 * ds_cur * gZ;

    R += ds_cur * vR;
    Z += ds_cur * vZ;
    const double R_unreflected = R;
    reflect_axis_if_needed(&R, &vR);

    if (!::isfinite(R) || !::isfinite(Z) || !::isfinite(vR) || !::isfinite(vZ)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    const bool exited_mesh = outside_mesh(R, Z, node_R, node_Z, n_nodes_r, n_nodes_z);
    double t_mesh = 1.0;
    if (exited_mesh) {
      t_mesh = first_exit_fraction_2d(old_R, old_Z, R_unreflected, Z, node_R[n_nodes_r - 1],
                                      node_Z[0], node_Z[n_nodes_z - 1]);
      if (!(t_mesh > 0.0)) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
      t_mesh = clamp_unit_interval(t_mesh);
    }

    constexpr double kSegTol = 1.0e-12;
    const double nh_old = carried_nh;
    const double nh_old_raw = carried_nh_raw;
    const double kappa_old = carried_kappa;
    if (!::isfinite(nh_old) || !::isfinite(nh_old_raw) || !::isfinite(carried_Te) ||
        !::isfinite(carried_Zbar)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_old >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_old_raw >= kCritLayerHandoffNhatRaw) {
      // Mixed hydro/ghost stencils occur exactly where tail closure should trigger.
      // Reconstruct A_entry from the subcritical nodes only so hydro-side low-Te
      // values never leak into the closure coefficient.
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag,
                           vR * gR + vZ * gZ, v_mag2, I,
                           TailClosureMode::kRequireTrigger, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail);
      if (tail_status == TailClosureStatus::kInvalid) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
      if (tail_status == TailClosureStatus::kClosed) {
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_R[idx] = R;
          traj_pos_Z[idx] = Z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
    }

    const double dR_path = R_unreflected - old_R;
    const double dZ_path = Z - old_Z;
    double t_stop = t_mesh;
    bool hit_critical = false;

    double R_stop_path = old_R + t_stop * dR_path;
    double R_stop = ::abs(R_stop_path);
    double Z_stop = old_Z + t_stop * dZ_path;
    BilinearCell c_stop =
        BilinearInterp::locate_cell_local(node_R, node_Z, n_nodes_r, n_nodes_z, R_stop, Z_stop,
                                          carried_c.i, carried_c.j);
    BilinearWeights w_stop = BilinearInterp::compute_weights(c_stop.xi, c_stop.eta);
    double nh_stop = BilinearInterp::interpolate(n_hat, n_nodes_z, c_stop, w_stop);
    double nh_stop_raw = BilinearInterp::interpolate(n_hat_raw, n_nodes_z, c_stop, w_stop);
    double Te_stop = BilinearInterp::interpolate(T_e, n_nodes_z, c_stop, w_stop);
    double Zbar_stop = BilinearInterp::interpolate(Zbar, n_nodes_z, c_stop, w_stop);
    if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw) || !::isfinite(Te_stop) ||
        !::isfinite(Zbar_stop)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    const double nh_crit = 1.0 - eps_crit;
    if (nh_old_raw >= nh_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_stop_raw >= nh_crit) {
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag, 1.0,
                           v_mag2, I,
                           TailClosureMode::kCriticalCrossing, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail);
      if (tail_status == TailClosureStatus::kClosed) {
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_R[idx] = R;
          traj_pos_Z[idx] = Z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
      CriticalSurfaceContext critical_ctx;
      critical_ctx.node_R = node_R;
      critical_ctx.node_Z = node_Z;
      critical_ctx.n_nodes_r = n_nodes_r;
      critical_ctx.n_nodes_z = n_nodes_z;
      critical_ctx.n_hat = n_hat;
      critical_ctx.n_hat_raw = n_hat_raw;
      critical_ctx.T_e = T_e;
      critical_ctx.Zbar = Zbar;
      critical_ctx.carried_c = carried_c;
      critical_ctx.I = I;
      critical_ctx.traj_on = traj_on;
      critical_ctx.output_idx = output_idx;
      critical_ctx.traj_stored = traj_stored;
      critical_ctx.traj_step_count = traj_step_count;
      critical_ctx.P_unabsorbed = P_unabsorbed;
      critical_ctx.critical_surface_hit_count = critical_surface_hit_count;
      critical_ctx.error_flags = error_flags;
      bool should_return = false;
      handle_critical_surface_2d(hit_critical, t_stop, R_stop_path, R_stop, Z_stop, c_stop, w_stop,
                                 nh_stop, nh_stop_raw, Te_stop, Zbar_stop, nh_old_raw, nh_crit,
                                 old_R, old_Z, dR_path, dZ_path, critical_ctx, should_return);
      if (should_return) {
        return;
      }
    }

    double kappa_stop = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa_stop = test_kappa_cm_inv;
    } else if (smooth_kappa_factor != nullptr) {
      const double A_stop =
          BilinearInterp::interpolate(smooth_kappa_factor, n_nodes_z, c_stop, w_stop);
      kappa_stop = compute_kappa_from_smooth(A_stop, nh_stop, eps_n);
    } else {
      kappa_stop = compute_kappa_ib(nh_stop, Te_stop, Zbar_stop, lambda_cm, eps_n,
                                    coulomb_log_floor);
    }
    if (!::isfinite(kappa_stop) || kappa_stop < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    // Measure step length from the pre-reflection radial coordinate so axis crossings
    // account for the full segment (|R_old| + |R_unreflected|).
    const bool crossed_axis = (old_R * R_unreflected) < 0.0;
    const double dR_actual =
        crossed_axis ? (::abs(old_R) + ::abs(R_unreflected)) : ::abs(R - old_R);
    const double dZ_actual = Z - old_Z;
    const double ds_actual = ::sqrt(dR_actual * dR_actual + dZ_actual * dZ_actual);
    const double segment_fraction = clamp_unit_interval(t_stop);
    const double ds_segment = ds_actual * segment_fraction;
    const double S = compute_optical_depth(kappa_old, kappa_stop, ds_segment);
    if (!::isfinite(S) || S < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    double I_next = I;
    const double dP = absorbed_power_expm1(I, S, I_next);
    if (!::isfinite(dP) || !::isfinite(I_next) || dP < 0.0 || I_next < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    if (dP > 0.0) {
      const double mid_t = 0.5 * t_stop;
      const double mid_R = ::abs(old_R + mid_t * dR_path);
      const double mid_Z = old_Z + mid_t * dZ_path;
      if (!outside_mesh(mid_R, mid_Z, node_R, node_Z, n_nodes_r, n_nodes_z)) {
        const BilinearCell c_mid =
            BilinearInterp::locate_cell_local(node_R, node_Z, n_nodes_r, n_nodes_z, mid_R, mid_Z,
                                              carried_c.i, carried_c.j);
        const BilinearWeights w_mid = BilinearInterp::compute_weights(c_mid.xi, c_mid.eta);
        accumulate_masked_deposit(deposit_cache, c_mid, w_mid, n_nodes_z, n_hat_raw, dP);
      }
    }

    I = I_next;
    if (traj_on && traj_stored < traj_max_steps) {
      const int idx = output_idx * traj_max_steps + traj_stored;
      traj_pos_R[idx] = R;
      traj_pos_Z[idx] = Z;
      traj_power[idx] = I;
      ++traj_stored;
    }
    carried_c = c_stop;
    carried_nh = nh_stop;
    carried_nh_raw = nh_stop_raw;
    carried_Te = Te_stop;
    carried_Zbar = Zbar_stop;
    carried_kappa = kappa_stop;
    if (exited_mesh || hit_critical || t_stop < 1.0 - kSegTol) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    if (I < intensity_cutoff * I0) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
  }

  atomic_add_double(P_unabsorbed, I);
  if (error_flags != nullptr) {
    atomicAdd(&error_flags->infinite_loop, 1);
  }
  if (traj_on) {
    traj_step_count[output_idx] = traj_stored;
  }
}

template <bool DebugOneRay, bool kCbetRecord, bool kHotECapture>
__global__ __launch_bounds__(64)
void ray_trace_3d(double* __restrict__ deposit,
                  const double* __restrict__ n_hat,
                  const double* __restrict__ n_hat_raw,
                  const double* __restrict__ grad_n_hat_R,
                  const double* __restrict__ grad_n_hat_Z,
                  const double* __restrict__ T_e,
                  const double* __restrict__ Zbar,
                  const double* __restrict__ smooth_kappa_factor,
                  const double* __restrict__ node_R,
                  const double* __restrict__ node_Z,
                  const double* __restrict__ ray_x0,
                  const double* __restrict__ ray_y0,
                  const double* __restrict__ ray_z0,
                  const double* __restrict__ ray_vx0,
                  const double* __restrict__ ray_vy0,
                  const double* __restrict__ ray_vz0,
                  const double* __restrict__ ray_power,
                  const double* __restrict__ ray_power0,
                  const double cfl_ray,
                  const double ds_adapt_g_target,
                  const double ds_adapt_tau_target,
                  const double ds_adapt_theta_target,
                  const double ds_adapt_max_factor,
                  const double eps_n,
                  const double eps_crit,
                  const double lambda_cm,
                  const double coulomb_log_floor,
                  const double test_kappa_cm_inv,
                  const double intensity_cutoff,
                  const int max_ray_steps,
                  const int n_nodes_r,
                  const int n_nodes_z,
                  const int n_rays,
                  const double dx_lm_min,
                  double* __restrict__ traj_pos_x,
                  double* __restrict__ traj_pos_y,
                  double* __restrict__ traj_pos_z,
                  double* __restrict__ traj_power,
                  int* __restrict__ traj_step_count,
                  const int n_output_rays,
                  const int output_stride,
                  const int traj_max_steps,
                  int* __restrict__ step_histogram,
                  int* __restrict__ step_count,
                  double* __restrict__ P_unabsorbed,
                  unsigned long long* __restrict__ tail_closure_count,
                  double* __restrict__ tail_closure_absorbed_power,
                  unsigned long long* __restrict__ critical_surface_hit_count,
                  core::DeviceErrorFlags* __restrict__ error_flags,
                  const CbetRecordDeviceArgs cbet_args,
                  const HotECaptureParams hot_e_params,
                  double* __restrict__ hot_e_capture) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_idx = (output_stride > 0) ? (tid / output_stride) : -1;
  const bool traj_on = (traj_pos_x != nullptr) && (output_stride > 0) &&
                       (tid % output_stride == 0) && (output_idx < n_output_rays);
  int traj_stored = 0;
  int n_steps = 0;
  CbetRecordCursor2D cbet_cursor;
  if constexpr (kCbetRecord) {
    if (tid < n_rays && cbet_args.rec_cell != nullptr && cbet_args.cap_per_ray > 0) {
      cbet_cursor.rec_cell = cbet_args.rec_cell;
      cbet_cursor.rec_mu = cbet_args.rec_mu;
      cbet_cursor.rec_c = cbet_args.rec_c;
      cbet_cursor.rec_w00 = cbet_args.rec_w00;
      cbet_cursor.rec_w10 = cbet_args.rec_w10;
      cbet_cursor.rec_w01 = cbet_args.rec_w01;
      cbet_cursor.rec_ds = cbet_args.rec_ds;
      cbet_cursor.rec_S = cbet_args.rec_S;
      cbet_cursor.rec_w = cbet_args.rec_w;
      const long long gray = cbet_args.ray_offset + tid;
      cbet_cursor.base = gray * static_cast<long long>(cbet_args.cap_per_ray);
      cbet_cursor.cap = cbet_args.cap_per_ray;
      cbet_cursor.rec_count_out = &cbet_args.rec_count[gray];
      cbet_cursor.overflow_out = &cbet_args.ray_overflow[gray];
      cbet_cursor.active = true;
    }
  }
  if (tid >= n_rays) {
    if constexpr (kCbetRecord) {
      cbet_cursor.flush();
    }
    return;
  }
  RayStepHistogramGuard histogram_guard(step_histogram, &n_steps, step_count, tid, n_rays);
  DepositCacheGuard deposit_cache(deposit);

  double x = ray_x0[tid];
  double y = ray_y0[tid];
  double z = ray_z0[tid];
  double vx = ray_vx0[tid];
  double vy = ray_vy0[tid];
  double vz = ray_vz0[tid];
  double I = ray_power[tid];
  const double I0 = ray_power0[tid];
  unsigned int hot_e_captured_mask = 0u;
  const double launch_I = I;
  double total_dP = 0.0;

#define DEBUG_RAY_SUMMARY(reason_value)                                                     \
  do {                                                                                      \
    if constexpr (DebugOneRay) {                                                            \
      printf("[laser_raytrace_3d_ray_summary] ray_id=%d launch_I=%.17e final_I=%.17e "     \
             "n_segments=%d final_exit_reason=%d total_dP=%.17e\n",                       \
             tid, launch_I, I, n_steps, (reason_value), total_dP);                          \
    }                                                                                       \
  } while (0)

  const bool finite_init = ::isfinite(x) && ::isfinite(y) && ::isfinite(z) && ::isfinite(vx) &&
                           ::isfinite(vy) && ::isfinite(vz) && ::isfinite(I) && ::isfinite(I0);
  if (!finite_init) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (::isfinite(I) && I > 0.0) {
      atomic_add_double(P_unabsorbed, I);
    }
    if (traj_on) {
      traj_step_count[output_idx] = 0;
    }
    DEBUG_RAY_SUMMARY(5);
    if constexpr (kCbetRecord) {
      cbet_cursor.flush();
    }
    return;
  }
  if (!(I > 0.0)) {
    if (traj_on) {
      traj_step_count[output_idx] = 0;
    }
    DEBUG_RAY_SUMMARY(5);
    if constexpr (kCbetRecord) {
      cbet_cursor.flush();
    }
    return;
  }

  double R = 0.0;
  double Z = 0.0;
  to_laser_rz(x, y, z, R, Z);
  if (traj_on && traj_stored < traj_max_steps) {
    const int idx = output_idx * traj_max_steps + traj_stored;
    traj_pos_x[idx] = x;
    traj_pos_y[idx] = y;
    traj_pos_z[idx] = z;
    traj_power[idx] = I;
    ++traj_stored;
  }
  if (outside_mesh(R, Z, node_R, node_Z, n_nodes_r, n_nodes_z)) {
    atomic_add_double(P_unabsorbed, I);
    if (traj_on) {
      traj_step_count[output_idx] = traj_stored;
    }
    DEBUG_RAY_SUMMARY(1);
    if constexpr (kCbetRecord) {
      cbet_cursor.flush();
    }
    return;
  }

  BilinearCell carried_c;
  double carried_nh = 0.0;
  double carried_nh_raw = 0.0;
  double carried_Te = 0.0;
  double carried_Zbar = 0.0;
  double carried_kappa = 0.0;
  {
    carried_c = BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, R, Z);
    const double ds_local = local_ds(node_R, node_Z, n_nodes_r, n_nodes_z, carried_c, cfl_ray);
    const BilinearWeights w = BilinearInterp::compute_weights(carried_c.xi, carried_c.eta);
    const double nh0 = BilinearInterp::interpolate(n_hat, n_nodes_z, carried_c, w);
    const double nh0_raw = BilinearInterp::interpolate(n_hat_raw, n_nodes_z, carried_c, w);
    const double Te0 = BilinearInterp::interpolate(T_e, n_nodes_z, carried_c, w);
    const double Z0 = BilinearInterp::interpolate(Zbar, n_nodes_z, carried_c, w);
    if (!::isfinite(nh0)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh0 >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(2);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    double kappa0 = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa0 = test_kappa_cm_inv;
    } else if (smooth_kappa_factor != nullptr) {
      const double A0 = BilinearInterp::interpolate(smooth_kappa_factor, n_nodes_z, carried_c, w);
      kappa0 = compute_kappa_from_smooth(A0, nh0, eps_n);
    } else {
      kappa0 = compute_kappa_ib(nh0, Te0, Z0, lambda_cm, eps_n, coulomb_log_floor);
    }

    double gR = 0.0;
    double gZ = 0.0;
    BilinearInterp::interpolate_gradient(grad_n_hat_R, grad_n_hat_Z, n_nodes_z, carried_c, w, gR,
                                         gZ);
    const Vec3 g3 = grad_2d_to_3d(gR, gZ, x, y, z, dx_lm_min);
    // 2D_RZ NOTE (2026-07-26): this kernel still uses the LEGACY leapfrog
    // seeding (wrong sign, ds_base) and |v|=1 launch. The corrected scheme
    // (NUMERICS 5.3.2: v^{-1/2} = v^0 + (h/4) grad n_hat with adaptive first
    // h, and |v| = sqrt(1 - n_hat) at the start) is applied in the 1D
    // kernels only; mirroring it here moved the 2D CBET slab detuning
    // tendency gate marginally red, so the 2D migration is owned by the 2D
    // lane together with its gate re-qualification (2026-07-26 review).
    vx -= 0.25 * ds_local * g3.x;
    vy -= 0.25 * ds_local * g3.y;
    vz -= 0.25 * ds_local * g3.z;
    if (!::isfinite(vx) || !::isfinite(vy) || !::isfinite(vz)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    carried_nh = nh0;
    carried_nh_raw = nh0_raw;
    carried_Te = Te0;
    carried_Zbar = Z0;
    carried_kappa = kappa0;
  }

  while (n_steps < max_ray_steps) {
    ++n_steps;
    const double old_x = x;
    const double old_y = y;
    const double old_z = z;

    const BilinearCell c_old = carried_c;
    const BilinearWeights w_old = BilinearInterp::compute_weights(c_old.xi, c_old.eta);
    const double ds_local = local_ds(node_R, node_Z, n_nodes_r, n_nodes_z, c_old, cfl_ray);

    double gR = 0.0;
    double gZ = 0.0;
    BilinearInterp::interpolate_gradient(grad_n_hat_R, grad_n_hat_Z, n_nodes_z, c_old, w_old,
                                         gR, gZ);
    const double g_mag = ::sqrt(gR * gR + gZ * gZ);
    const double nh_entry = carried_nh;
    const double nh_entry_raw = carried_nh_raw;
    const double kappa_entry = carried_kappa;
    if (!::isfinite(nh_entry) || !::isfinite(nh_entry_raw) || !::isfinite(carried_Te) ||
        !::isfinite(carried_Zbar)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh_entry >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(2);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh_entry_raw >= kCritLayerHandoffNhatRaw) {
      double I_tail = I;
      double tau_tail_rec = 0.0;
      const double v_mag2 = vx * vx + vy * vy + vz * vz;
      // 2D_RZ NOTE (2026-07-26): direction gate intentionally bypassed
      // (v_dot_g = 1.0) — the C4 gate is applied in the 1D kernels only;
      // 2D migration is owned by the 2D lane (see the init-block note).
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_entry, nh_entry_raw, kappa_entry, g_mag,
                           1.0, v_mag2, I,
                           TailClosureMode::kRequireTrigger, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail,
                           kCbetRecord ? &tau_tail_rec : nullptr);
      if (tail_status == TailClosureStatus::kInvalid) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(5);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
      if (tail_status == TailClosureStatus::kClosed) {
        if constexpr (DebugOneRay) {
          const double dP_tail = I - I_tail;
          if (::isfinite(dP_tail) && dP_tail > 0.0) {
            total_dP += dP_tail;
          }
        }
        if constexpr (kCbetRecord) {
          const int cell_id_old = c_old.i * (n_nodes_z - 1) + c_old.j;
          cbet_cursor.append_terminal(cell_id_old, tau_tail_rec, w_old.w00, w_old.w10,
                                      w_old.w01, I);
        }
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_x[idx] = x;
          traj_pos_y[idx] = y;
          traj_pos_z[idx] = z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(2);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
    }
    const double g_metric = g_mag * ds_local;
    const double tau_metric = carried_kappa * ds_local;
    double m = ::fmin(::fmin(ds_adapt_g_target / ::fmax(g_metric, 1.0e-30),
                             ds_adapt_tau_target / ::fmax(tau_metric, 1.0e-30)),
                      ds_adapt_max_factor);
    m = clamp_adaptive_step_multiplier(m, carried_nh_raw, ds_adapt_max_factor);
    m = ::fmin(m, ray_trace_bodies::theta_step_multiplier(
                         g_metric, carried_nh_raw, ds_adapt_theta_target));
    const double ds_cur = ds_local * m;
    const Vec3 g3 = grad_2d_to_3d(gR, gZ, old_x, old_y, old_z, dx_lm_min);
    vx -= 0.5 * ds_cur * g3.x;
    vy -= 0.5 * ds_cur * g3.y;
    vz -= 0.5 * ds_cur * g3.z;

    x += ds_cur * vx;
    y += ds_cur * vy;
    z += ds_cur * vz;
    if (!::isfinite(x) || !::isfinite(y) || !::isfinite(z) || !::isfinite(vx) ||
        !::isfinite(vy) || !::isfinite(vz)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }

    to_laser_rz(x, y, z, R, Z);
    const bool exited_mesh = outside_mesh(R, Z, node_R, node_Z, n_nodes_r, n_nodes_z);
    double t_mesh = 1.0;
    if (exited_mesh) {
      t_mesh = first_exit_fraction_3d(old_x, old_y, old_z, x, y, z, node_R[n_nodes_r - 1],
                                      node_Z[0], node_Z[n_nodes_z - 1]);
      if (!(t_mesh > 0.0)) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(1);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
      t_mesh = clamp_unit_interval(t_mesh);
    }

    constexpr double kSegTol = 1.0e-12;
    const double nh_old = carried_nh;
    const double nh_old_raw = carried_nh_raw;
    const double kappa_old = carried_kappa;
    if (!::isfinite(nh_old) || !::isfinite(nh_old_raw) || !::isfinite(carried_Te) ||
        !::isfinite(carried_Zbar)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh_old >= 1.0 - eps_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(2);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh_old_raw >= kCritLayerHandoffNhatRaw) {
      // Mixed hydro/ghost stencils occur exactly where tail closure should trigger.
      // Reconstruct A_entry from the subcritical nodes only so hydro-side low-Te
      // values never leak into the closure coefficient.
      double I_tail = I;
      double tau_tail_rec = 0.0;
      const double v_mag2 = vx * vx + vy * vy + vz * vz;
      // 2D_RZ NOTE (2026-07-26): direction gate bypassed here as well —
      // see the site-1 note.
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag,
                           1.0, v_mag2, I,
                           TailClosureMode::kRequireTrigger, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail,
                           kCbetRecord ? &tau_tail_rec : nullptr);
      if (tail_status == TailClosureStatus::kInvalid) {
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(5);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
      if (tail_status == TailClosureStatus::kClosed) {
        if constexpr (DebugOneRay) {
          const double dP_tail = I - I_tail;
          if (::isfinite(dP_tail) && dP_tail > 0.0) {
            total_dP += dP_tail;
          }
        }
        if constexpr (kCbetRecord) {
          const int cell_id_old = c_old.i * (n_nodes_z - 1) + c_old.j;
          cbet_cursor.append_terminal(cell_id_old, tau_tail_rec, w_old.w00, w_old.w10,
                                      w_old.w01, I);
        }
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_x[idx] = x;
          traj_pos_y[idx] = y;
          traj_pos_z[idx] = z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(2);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
    }

    const double dx_path = x - old_x;
    const double dy_path = y - old_y;
    const double dz_path = z - old_z;
    double t_stop = t_mesh;
    bool hit_critical = false;

    double x_stop = old_x + t_stop * dx_path;
    double y_stop = old_y + t_stop * dy_path;
    double z_stop = old_z + t_stop * dz_path;
    double R_stop = 0.0;
    double Z_stop = 0.0;
    to_laser_rz(x_stop, y_stop, z_stop, R_stop, Z_stop);
    BilinearCell c_stop =
        BilinearInterp::locate_cell_local(node_R, node_Z, n_nodes_r, n_nodes_z, R_stop, Z_stop,
                                          carried_c.i, carried_c.j);
    BilinearWeights w_stop = BilinearInterp::compute_weights(c_stop.xi, c_stop.eta);
    double nh_stop = BilinearInterp::interpolate(n_hat, n_nodes_z, c_stop, w_stop);
    double nh_stop_raw = BilinearInterp::interpolate(n_hat_raw, n_nodes_z, c_stop, w_stop);
    double Te_stop = BilinearInterp::interpolate(T_e, n_nodes_z, c_stop, w_stop);
    double Zbar_stop = BilinearInterp::interpolate(Zbar, n_nodes_z, c_stop, w_stop);
    if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw) || !::isfinite(Te_stop) ||
        !::isfinite(Zbar_stop)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }

    if constexpr (kHotECapture) {
      for (int he_ch = 0; he_ch < hot_e_params.n_channels; ++he_ch) {
        if ((hot_e_captured_mask & (1u << he_ch)) != 0u) {
          continue;
        }
        const double he_thr = hot_e_params.threshold_nhat[he_ch];
        if (!(nh_old_raw < he_thr && nh_stop_raw >= he_thr)) {
          continue;
        }
        hot_e_captured_mask |= (1u << he_ch);
        const double he_denom = nh_stop_raw - nh_old_raw;
        double he_frac = (he_denom > 0.0) ? (he_thr - nh_old_raw) / he_denom : 0.0;
        if (he_frac < 0.0) {
          he_frac = 0.0;
        }
        if (he_frac > 1.0) {
          he_frac = 1.0;
        }
        const double he_x = old_x + he_frac * t_stop * dx_path;
        const double he_y = old_y + he_frac * t_stop * dy_path;
        const double he_z = old_z + he_frac * t_stop * dz_path;
        const double he_R = sqrt(he_x * he_x + he_y * he_y);
        const double he_vn = sqrt(vx * vx + vy * vy + vz * vz);
        double he_kR = 0.0;
        double he_kphi = 0.0;
        double he_kZ = 1.0;
        if (he_vn > 0.0) {
          he_kZ = vz / he_vn;
          if (he_R > 1.0e-30) {
            he_kR = (vx * he_x + vy * he_y) / (he_R * he_vn);
            he_kphi = (vy * he_x - vx * he_y) / (he_R * he_vn);
          } else {
            he_kR = sqrt(vx * vx + vy * vy) / he_vn;
            he_kphi = 0.0;
          }
        }
        const std::size_t he_base =
            (static_cast<std::size_t>(tid) * hot_e_params.n_channels + he_ch) * 8;
        hot_e_capture[he_base + 0] = 1.0;
        hot_e_capture[he_base + 1] = he_R;
        hot_e_capture[he_base + 2] = he_z;
        hot_e_capture[he_base + 3] = he_kR;
        hot_e_capture[he_base + 4] = he_kphi;
        hot_e_capture[he_base + 5] = he_kZ;
        hot_e_capture[he_base + 6] = I;
        hot_e_capture[he_base + 7] = 0.0;
        I *= hot_e_params.one_minus_eta[he_ch];
      }
    }

    const double nh_crit = 1.0 - eps_crit;
    if (nh_old_raw >= nh_crit) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(2);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (nh_stop_raw >= nh_crit) {
      double I_tail = I;
      double tau_tail_rec = 0.0;
      const double v_mag2 = vx * vx + vy * vy + vz * vz;
      const TailClosureStatus tail_status =
          try_tail_closure(deposit_cache, c_old, w_old, n_nodes_z, n_hat, n_hat_raw, T_e, Zbar,
                           smooth_kappa_factor, lambda_cm, eps_n, coulomb_log_floor,
                           test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag, 1.0,
                           v_mag2, I,
                           TailClosureMode::kCriticalCrossing, tail_closure_count,
                           tail_closure_absorbed_power, error_flags, I_tail,
                           kCbetRecord ? &tau_tail_rec : nullptr);
      if (tail_status == TailClosureStatus::kClosed) {
        if constexpr (DebugOneRay) {
          const double dP_tail = I - I_tail;
          if (::isfinite(dP_tail) && dP_tail > 0.0) {
            total_dP += dP_tail;
          }
        }
        if constexpr (kCbetRecord) {
          const int cell_id_old = c_old.i * (n_nodes_z - 1) + c_old.j;
          cbet_cursor.append_terminal(cell_id_old, tau_tail_rec, w_old.w00, w_old.w10,
                                      w_old.w01, I);
        }
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_x[idx] = x;
          traj_pos_y[idx] = y;
          traj_pos_z[idx] = z;
          traj_power[idx] = I;
          ++traj_stored;
        }
        atomic_add_double(P_unabsorbed, I);
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        DEBUG_RAY_SUMMARY(2);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
      CriticalSurfaceContext critical_ctx;
      critical_ctx.node_R = node_R;
      critical_ctx.node_Z = node_Z;
      critical_ctx.n_nodes_r = n_nodes_r;
      critical_ctx.n_nodes_z = n_nodes_z;
      critical_ctx.n_hat = n_hat;
      critical_ctx.n_hat_raw = n_hat_raw;
      critical_ctx.T_e = T_e;
      critical_ctx.Zbar = Zbar;
      critical_ctx.carried_c = carried_c;
      critical_ctx.I = I;
      critical_ctx.traj_on = traj_on;
      critical_ctx.output_idx = output_idx;
      critical_ctx.traj_stored = traj_stored;
      critical_ctx.traj_step_count = traj_step_count;
      critical_ctx.P_unabsorbed = P_unabsorbed;
      critical_ctx.critical_surface_hit_count = critical_surface_hit_count;
      critical_ctx.error_flags = error_flags;
      bool should_return = false;
      handle_critical_surface_3d(hit_critical, t_stop, x_stop, y_stop, z_stop, R_stop, Z_stop,
                                 c_stop, w_stop, nh_stop, nh_stop_raw, Te_stop, Zbar_stop,
                                 nh_old_raw, nh_crit, old_x, old_y, old_z, dx_path, dy_path,
                                 dz_path, critical_ctx, should_return);
      if (should_return) {
        DEBUG_RAY_SUMMARY(2);
        if constexpr (kCbetRecord) {
          cbet_cursor.flush();
        }
        return;
      }
    }

    double kappa_stop = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa_stop = test_kappa_cm_inv;
    } else if (smooth_kappa_factor != nullptr) {
      const double A_stop =
          BilinearInterp::interpolate(smooth_kappa_factor, n_nodes_z, c_stop, w_stop);
      kappa_stop = compute_kappa_from_smooth(A_stop, nh_stop, eps_n);
    } else {
      kappa_stop = compute_kappa_ib(nh_stop, Te_stop, Zbar_stop, lambda_cm, eps_n,
                                    coulomb_log_floor);
    }
    if (!::isfinite(kappa_stop) || kappa_stop < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }

    const double ds_actual =
        ::sqrt((x - old_x) * (x - old_x) + (y - old_y) * (y - old_y) + (z - old_z) * (z - old_z));

    const double segment_fraction = clamp_unit_interval(t_stop);
    const double ds_segment = ds_actual * segment_fraction;

    const double S = compute_optical_depth(kappa_old, kappa_stop, ds_segment);
    if (!::isfinite(S) || S < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }

    double I_next = I;
    const double dP = absorbed_power_expm1(I, S, I_next);
    if (!::isfinite(dP) || !::isfinite(I_next) || dP < 0.0 || I_next < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(5);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }

    if constexpr (kCbetRecord) {
      if (ds_segment > 0.0) {
        double R_old_rec = 0.0;
        double Z_old_rec = 0.0;
        to_laser_rz(old_x, old_y, old_z, R_old_rec, Z_old_rec);
        const double z_stop_seg = old_z + t_stop * dz_path;
        const double R_stop_seg = R_stop;
        const double a = (R_stop_seg - R_old_rec) / ds_segment;
        const double c = (z_stop_seg - old_z) / ds_segment;
        const double mid_t = 0.5 * t_stop;
        const double mid_x = old_x + mid_t * dx_path;
        const double mid_y = old_y + mid_t * dy_path;
        const double mid_z = old_z + mid_t * dz_path;
        double mid_R = 0.0;
        double mid_Z = 0.0;
        to_laser_rz(mid_x, mid_y, mid_z, mid_R, mid_Z);
        if (!outside_mesh(mid_R, mid_Z, node_R, node_Z, n_nodes_r, n_nodes_z)) {
          const BilinearCell c_mid =
              BilinearInterp::locate_cell_local(node_R, node_Z, n_nodes_r, n_nodes_z, mid_R, mid_Z,
                                                carried_c.i, carried_c.j);
          const BilinearWeights w_mid = BilinearInterp::compute_weights(c_mid.xi, c_mid.eta);
          const int cell_id = c_mid.i * (n_nodes_z - 1) + c_mid.j;
          cbet_cursor.add_segment(cell_id, ds_segment, S, a, c, w_mid.w00, w_mid.w10,
                                  w_mid.w01, I);
        }
        (void)Z_old_rec;
      }
    }

    if constexpr (DebugOneRay) {
      if (tid < 8 && n_steps <= 4) {
        const bool has_fractional_stop = (t_stop < 1.0 - kSegTol);
        int exit_reason = 0;
        if (exited_mesh) {
          exit_reason = 1;
        } else if (hit_critical) {
          exit_reason = 2;
        } else if (has_fractional_stop) {
          exit_reason = 3;
        } else if (I_next < intensity_cutoff * I0) {
          exit_reason = 4;
        }
        printf("[laser_raytrace_3d_debug] ray_id=%d step=%d Te_stop=%.17e "
               "Zbar_stop=%.17e nh_stop=%.17e nh_stop_raw=%.17e Te_old=%.17e "
               "kappa_old=%.17e kappa_stop=%.17e t_stop=%.17e ds_actual=%.17e "
               "ds_segment=%.17e S=%.17e dP=%.17e I=%.17e I_next=%.17e "
               "has_fractional_stop=%d exit_reason=%d\n",
               tid, n_steps - 1, Te_stop, Zbar_stop, nh_stop, nh_stop_raw, carried_Te,
               kappa_old, kappa_stop, t_stop, ds_actual, ds_segment, S, dP, I, I_next,
               has_fractional_stop ? 1 : 0, exit_reason);
      }
    }
    if constexpr (DebugOneRay) {
      total_dP += dP;
    }

    if (dP > 0.0) {
      const double mid_t = 0.5 * t_stop;
      const double mid_x = old_x + mid_t * dx_path;
      const double mid_y = old_y + mid_t * dy_path;
      const double mid_z = old_z + mid_t * dz_path;
      double mid_R = 0.0;
      double mid_Z = 0.0;
      to_laser_rz(mid_x, mid_y, mid_z, mid_R, mid_Z);
      if (!outside_mesh(mid_R, mid_Z, node_R, node_Z, n_nodes_r, n_nodes_z)) {
        const BilinearCell c_mid =
            BilinearInterp::locate_cell_local(node_R, node_Z, n_nodes_r, n_nodes_z, mid_R, mid_Z,
                                              carried_c.i, carried_c.j);
        const BilinearWeights w_mid = BilinearInterp::compute_weights(c_mid.xi, c_mid.eta);
        accumulate_masked_deposit(deposit_cache, c_mid, w_mid, n_nodes_z, n_hat_raw, dP);
      }
    }

    I = I_next;
    if (traj_on && traj_stored < traj_max_steps) {
      const int idx = output_idx * traj_max_steps + traj_stored;
      traj_pos_x[idx] = x;
      traj_pos_y[idx] = y;
      traj_pos_z[idx] = z;
      traj_power[idx] = I;
      ++traj_stored;
    }
    carried_c = c_stop;
    carried_nh = nh_stop;
    carried_nh_raw = nh_stop_raw;
    carried_Te = Te_stop;
    carried_Zbar = Zbar_stop;
    carried_kappa = kappa_stop;
    if (exited_mesh || hit_critical || t_stop < 1.0 - kSegTol) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      const int final_exit_reason = exited_mesh ? 1 : (hit_critical ? 2 : 3);
      DEBUG_RAY_SUMMARY(final_exit_reason);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
    if (I < intensity_cutoff * I0) {
      atomic_add_double(P_unabsorbed, I);
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      DEBUG_RAY_SUMMARY(4);
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      return;
    }
  }

  atomic_add_double(P_unabsorbed, I);
  if (error_flags != nullptr) {
    atomicAdd(&error_flags->infinite_loop, 1);
  }
  if (traj_on) {
    traj_step_count[output_idx] = traj_stored;
  }
  if constexpr (kCbetRecord) {
    cbet_cursor.flush();
  }
  DEBUG_RAY_SUMMARY(6);
#undef DEBUG_RAY_SUMMARY
}

cudaError_t launch_ray_trace_2d(const RayArray1D& rays,
                                const LaserMesh& mesh,
                                const core::Config::LaserConfig& laser_cfg,
                                const double lambda_cm,
                                double* d_traj_pos1,
                                double* d_traj_pos2,
                                double* d_traj_pos3,
                                double* d_traj_power,
                                int* d_traj_step_count,
                                const int n_output_rays,
                                const int output_stride,
                                const int traj_max_steps,
                                double* d_unabsorbed,
                                core::DeviceErrorFlags* d_error_flags,
                                unsigned long long* d_tail_closure_count,
                                double* d_tail_closure_absorbed_power,
                                unsigned long long* d_critical_surface_hit_count,
                                cudaStream_t stream,
                                int* d_step_histogram,
                                const double* smooth_kappa_factor,
                                int* d_step_count) {
  if (rays.n_rays <= 0) {
    return cudaSuccess;
  }
  TENRYU_ASSERT(std::isfinite(laser_cfg.raytrace.cfl_ray) &&
                    laser_cfg.raytrace.cfl_ray > 0.0,
                "ray_trace_2d requires Laser.raytrace.cfl_ray > 0");
  (void)d_traj_pos3;

  const int max_ray_steps = std::min(std::max(laser_cfg.raytrace.max_steps, 1), kMaxRayStepsGuard);
  const int block = 64;
  const int grid = (rays.n_rays + block - 1) / block;
  ray_trace_2d<<<grid, block, 0, stream>>>(
      mesh.deposit, mesh.n_e_hat, mesh.n_e_hat_raw, mesh.grad_n_hat_R, mesh.grad_n_hat_Z,
      mesh.T_e, mesh.Zbar, smooth_kappa_factor, mesh.node_R, mesh.node_Z, rays.R0, rays.Z0,
      rays.vR0, rays.vZ0, rays.power, rays.power0, laser_cfg.raytrace.cfl_ray,
      laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
      laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
      laser_cfg.absorption.eps_n,
      laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
      laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
      mesh.n_nodes_r, mesh.n_nodes_z, rays.n_rays, d_traj_pos1, d_traj_pos2, d_traj_power,
      d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
      d_step_count, d_unabsorbed, d_tail_closure_count, d_tail_closure_absorbed_power,
      d_critical_surface_hit_count, d_error_flags);
  return cudaGetLastError();
}

__device__ inline void reduce_per_ray_tallies_1d_kernel_body(
    const int c,
    const double* __restrict__ deposit_per_ray,
    const double* __restrict__ unabsorbed_per_ray,
    const double* __restrict__ tail_power_per_ray,
    double* __restrict__ deposit_1d,
    double* __restrict__ P_unabsorbed,
    double* __restrict__ tail_closure_absorbed_power,
    const int n_rays,
    const int n_cells,
    const double* __restrict__ ra_per_ray,
    double* __restrict__ d_ra_total) {
  ray_trace_bodies::reduce_per_ray_tallies_1d_body(
      c, deposit_per_ray, unabsorbed_per_ray, tail_power_per_ray, deposit_1d,
      P_unabsorbed, tail_closure_absorbed_power, n_rays, n_cells, ra_per_ray,
      d_ra_total);
}


__global__ void reduce_per_ray_tallies_1d_kernel(
    const double* __restrict__ deposit_per_ray,
    const double* __restrict__ unabsorbed_per_ray,
    const double* __restrict__ tail_power_per_ray,
    double* __restrict__ deposit_1d,
    double* __restrict__ P_unabsorbed,
    double* __restrict__ tail_closure_absorbed_power,
    const int n_rays,
    const int n_cells,
    const double* __restrict__ ra_per_ray,
    double* __restrict__ d_ra_total) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  reduce_per_ray_tallies_1d_kernel_body(
      c, deposit_per_ray, unabsorbed_per_ray, tail_power_per_ray, deposit_1d,
      P_unabsorbed, tail_closure_absorbed_power, n_rays, n_cells, ra_per_ray,
      d_ra_total);
}

__global__ void sum_absorbed_power_per_ray_1d_kernel(
    const double* __restrict__ deposit_per_ray,
    const int n_rays,
    const int n_cells,
    double* __restrict__ pabs_per_ray_out) {
  const int ray = blockIdx.x * blockDim.x + threadIdx.x;
  if (ray >= n_rays) {
    return;
  }

  double absorbed = 0.0;
  const double* const row =
      deposit_per_ray + static_cast<std::size_t>(ray) * n_cells;
  for (int cell = 0; cell < n_cells; ++cell) {
    absorbed += row[cell];
  }
  // Tail closure and march-side RA are already materialized in this row.
  pabs_per_ray_out[ray] = absorbed;
}

namespace {

struct DynamicSharedBytesCacheEntry {
  bool attributes_queried = false;
  bool attributes_available = false;
  std::size_t static_shared_bytes = 0;
  std::size_t max_dynamic_shared_bytes = 0;
  std::size_t failed_requested_bytes = 0;
};

template <typename KernelFunction>
std::size_t launchable_dynamic_shared_bytes(
    KernelFunction kernel,
    const std::size_t requested) {
  if (requested == 0) {
    return 0;
  }

  static std::map<KernelFunction, DynamicSharedBytesCacheEntry> cache;
  DynamicSharedBytesCacheEntry& cached = cache[kernel];
  if (!cached.attributes_queried) {
    cudaFuncAttributes attributes{};
    const cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
    cached.attributes_queried = true;
    if (status != cudaSuccess) {
      (void)cudaGetLastError();
      return 0;
    }
    cached.attributes_available = true;
    cached.static_shared_bytes = attributes.sharedSizeBytes;
    cached.max_dynamic_shared_bytes =
        static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  }
  if (!cached.attributes_available) {
    return 0;
  }
  if (requested <= cached.max_dynamic_shared_bytes) {
    return requested;
  }
  if (requested == cached.failed_requested_bytes) {
    return 0;
  }

  cudaError_t status = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(requested));
  if (status != cudaSuccess) {
    cached.failed_requested_bytes = requested;
    (void)cudaGetLastError();
    return 0;
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    cached.attributes_available = false;
    (void)cudaGetLastError();
    return 0;
  }
  cached.static_shared_bytes = attributes.sharedSizeBytes;
  cached.max_dynamic_shared_bytes =
      static_cast<std::size_t>(attributes.maxDynamicSharedSizeBytes);
  if (requested <= cached.max_dynamic_shared_bytes) {
    return requested;
  }
  cached.failed_requested_bytes = requested;
  return 0;
}

}  // namespace

cudaError_t launch_ray_trace_1d_sph(const RayArray1D& rays,
                                    const LaserMesh& mesh,
                                    const core::Config::LaserConfig& laser_cfg,
                                    const double lambda_cm,
                                    const double* d_hydro_r_edges,
                                    const int n_hydro_cells,
                                    const int allowed_supercritical_cell,
                                    const int critical_adjacent_subcritical_cell,
                                    const double critical_adjacent_split_r,
                                    double* d_deposit_1d,
                                    double* d_traj_pos1,
                                    double* d_traj_pos2,
                                    double* d_traj_pos3,
                                    double* d_traj_power,
                                    int* d_traj_step_count,
                                    const int n_output_rays,
                                    const int output_stride,
                                    const int traj_max_steps,
                                    double* d_unabsorbed,
                                    core::DeviceErrorFlags* d_error_flags,
                                    unsigned long long* d_tail_closure_count,
                                    double* d_tail_closure_absorbed_power,
                                    unsigned long long* d_critical_surface_hit_count,
                                    cudaStream_t stream,
                                    int* d_step_histogram,
                                    int* d_step_count,
                                    const CbetRecordDeviceArgs* cbet_record,
                                    const HotECaptureParams hot_e_params,
                                    double** d_hot_e_capture_out,
                                    const laser::LaserPhysExtOptions* phys_ext,
                                    const double* d_radial_T_e,
                                    double* d_ra_power_total,
                                    const int* h_ray_order,
                                    int* h_ray_steps_out,
                                    const int max_ray_steps_override,
                                    double* d_tau_shell_out,
                                    double* d_pabs_per_ray_out) {
  if (rays.n_rays <= 0) {
    return cudaSuccess;
  }
  TENRYU_ASSERT(std::isfinite(laser_cfg.raytrace.cfl_ray) &&
                    laser_cfg.raytrace.cfl_ray > 0.0,
                "ray_trace_1d_sph requires Laser.raytrace.cfl_ray > 0");
  TENRYU_ASSERT(mesh.radial_n_nodes >= 2, "ray_trace_1d_sph requires at least 2 radial nodes");
  TENRYU_ASSERT(d_hydro_r_edges != nullptr, "ray_trace_1d_sph requires hydro_r_edges");
  TENRYU_ASSERT(d_deposit_1d != nullptr, "ray_trace_1d_sph requires deposit_1d");
  const bool hot_e_capture_enabled = hot_e_params.n_channels > 0;
  const bool cbet_record_mode =
      (cbet_record != nullptr && cbet_record->rec_cell != nullptr);
  (void)d_traj_pos3;

  int max_ray_steps = std::max(laser_cfg.raytrace.max_steps, 1);
  if (max_ray_steps_override > 0) {
    max_ray_steps = std::min(max_ray_steps, max_ray_steps_override);
  }
  max_ray_steps = std::min(max_ray_steps, kMaxRayStepsGuard);
  constexpr int block = kRayTrace1DBlockSize;
  const int grid = (rays.n_rays + block - 1) / block;
  const int n_slab =
      std::max(mesh.radial_n_nodes, n_hydro_cells + 1);
  const int n_stage_arrays =
      (phys_ext != nullptr && d_radial_T_e != nullptr) ? 7 : 6;
  const std::size_t requested_shared_bytes =
      static_cast<std::size_t>(n_stage_arrays) *
      static_cast<std::size_t>(n_slab) * sizeof(double);
  const std::size_t requested_ray_trace_shared_bytes =
      (requested_shared_bytes <= kRayTrace1DSharedBytesCap)
          ? requested_shared_bytes
          : 0;

  // Fixed-order tally-reduction statistical-reproducibility contract: per-ray private tallies +
  // fixed-order reduction give bitwise-deterministic deposition. Falls back to
  // the legacy shared-atomic path only if the scratch allocation would be
  // unreasonably large.
  double* d_per_ray_deposit = nullptr;
  double* d_per_ray_unabsorbed = nullptr;
  double* d_per_ray_tail = nullptr;
  double* d_per_ray_hot_e_capture = nullptr;
  double* d_ra_per_ray = nullptr;
  const std::size_t per_ray_doubles =
      static_cast<std::size_t>(rays.n_rays) *
      static_cast<std::size_t>(n_hydro_cells);
  const std::size_t per_ray_bytes = per_ray_doubles * sizeof(double);
  constexpr std::size_t kPerRayBytesCap = 256ULL * 1024ULL * 1024ULL;
  if (!cbet_record_mode && per_ray_bytes > 0 && per_ray_bytes <= kPerRayBytesCap) {
    const std::size_t he_channels =
        hot_e_capture_enabled ? static_cast<std::size_t>(hot_e_params.n_channels) : 1ULL;
    const std::size_t slab_doubles =
        per_ray_doubles + 2ULL * static_cast<std::size_t>(rays.n_rays) +
        4ULL * static_cast<std::size_t>(rays.n_rays) * he_channels;
    double* slab = static_cast<double*>(core::device_scratch_acquire(
        "ray_trace_1d_sph:per_ray_slab", slab_doubles * sizeof(double)));
    // Pooled scratch asserts on true OOM instead of falling back; accepted for <=cap path (1D scratch is ~1.6 MB).
    d_per_ray_deposit = slab;
    d_per_ray_unabsorbed = slab + per_ray_doubles;
    d_per_ray_tail = slab + per_ray_doubles + rays.n_rays;
    d_per_ray_hot_e_capture =
        slab + per_ray_doubles + 2ULL * static_cast<std::size_t>(rays.n_rays);
    cudaMemsetAsync(slab, 0, slab_doubles * sizeof(double), stream);
  }
  if (phys_ext != nullptr && phys_ext->ra_enable != 0 &&
      d_ra_power_total != nullptr) {
    d_ra_per_ray = static_cast<double*>(core::device_scratch_acquire(
        "ray_trace_1d_sph:ra_per_ray",
        static_cast<std::size_t>(rays.n_rays) * sizeof(double)));
    cudaMemsetAsync(
        d_ra_per_ray, 0,
        static_cast<std::size_t>(rays.n_rays) * sizeof(double), stream);
  }
  if (d_hot_e_capture_out != nullptr) {
    *d_hot_e_capture_out = hot_e_capture_enabled ? d_per_ray_hot_e_capture : nullptr;
  }

  const std::size_t ray_index_bytes =
      static_cast<std::size_t>(rays.n_rays) * sizeof(int);
  int* d_ray_steps_out = static_cast<int*>(core::device_scratch_acquire(
      "ray_trace_1d_sph:ray_steps_out", ray_index_bytes));
  int* d_ray_order = static_cast<int*>(core::device_scratch_acquire(
      "ray_trace_1d_sph:ray_order", ray_index_bytes));
  const int* kernel_ray_order = nullptr;
  if (h_ray_order != nullptr) {
    const cudaError_t order_status =
        cudaMemcpyAsync(d_ray_order, h_ray_order, ray_index_bytes,
                        cudaMemcpyHostToDevice, stream);
    if (order_status != cudaSuccess) {
      return order_status;
    }
    kernel_ray_order = d_ray_order;
  }

  if (cbet_record_mode && hot_e_capture_enabled) {
    if (phys_ext != nullptr) {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<true, true, true>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<true, true, true>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, *cbet_record,
          hot_e_params, d_per_ray_hot_e_capture, *phys_ext, d_radial_T_e,
          d_ra_per_ray, d_tau_shell_out);
    } else {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<true, true, false>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<true, true, false>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, *cbet_record,
          hot_e_params, d_per_ray_hot_e_capture, laser::LaserPhysExtOptions{},
          nullptr, nullptr, d_tau_shell_out);
    }
  } else if (cbet_record_mode) {
    if (phys_ext != nullptr) {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<true, false, true>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<true, false, true>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, *cbet_record,
          HotECaptureParams{}, nullptr, *phys_ext, d_radial_T_e,
          d_ra_per_ray, d_tau_shell_out);
    } else {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<true, false, false>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<true, false, false>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, *cbet_record,
          HotECaptureParams{}, nullptr, laser::LaserPhysExtOptions{}, nullptr,
          nullptr, d_tau_shell_out);
    }
  } else if (hot_e_capture_enabled) {
    if (phys_ext != nullptr) {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<false, true, true>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<false, true, true>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, CbetRecordDeviceArgs{},
          hot_e_params, d_per_ray_hot_e_capture, *phys_ext, d_radial_T_e,
          d_ra_per_ray, d_tau_shell_out);
    } else {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<false, true, false>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<false, true, false>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, CbetRecordDeviceArgs{},
          hot_e_params, d_per_ray_hot_e_capture, laser::LaserPhysExtOptions{},
          nullptr, nullptr, d_tau_shell_out);
    }
  } else {
    if (phys_ext != nullptr) {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<false, false, true>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<false, false, true>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, CbetRecordDeviceArgs{},
          HotECaptureParams{}, nullptr, *phys_ext, d_radial_T_e,
          d_ra_per_ray, d_tau_shell_out);
    } else {
      const std::size_t ray_trace_shared_bytes =
          launchable_dynamic_shared_bytes(ray_trace_1d_sph<false, false, false>,
                                          requested_ray_trace_shared_bytes);
      const int use_shared_staging = ray_trace_shared_bytes > 0 ? 1 : 0;
      ray_trace_1d_sph<false, false, false>
          <<<grid, block, ray_trace_shared_bytes, stream>>>(
          d_deposit_1d, d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail,
          mesh.radial_node_r, mesh.radial_n_hat, mesh.radial_n_hat_raw,
          mesh.radial_smooth_kappa, mesh.radial_dn_dr, d_hydro_r_edges,
          allowed_supercritical_cell, critical_adjacent_subcritical_cell,
          critical_adjacent_split_r, rays.R0, rays.Z0, rays.vR0, rays.vZ0, rays.power, rays.power0,
          laser_cfg.raytrace.cfl_ray,
          laser_cfg.raytrace.ds_adapt_g_target, laser_cfg.raytrace.ds_adapt_tau_target,
          laser_cfg.raytrace.ds_adapt_theta_target, laser_cfg.raytrace.ds_adapt_max_factor,
          laser_cfg.absorption.eps_n,
          laser_cfg.raytrace.eps_crit, lambda_cm, laser_cfg.absorption.coulomb_log_floor,
          laser_cfg.raytrace.test_kappa, laser_cfg.raytrace.intensity_cutoff, max_ray_steps,
          mesh.radial_n_nodes, n_hydro_cells, rays.n_rays, use_shared_staging,
          d_traj_pos1, d_traj_pos2, d_traj_power,
          d_traj_step_count, n_output_rays, output_stride, traj_max_steps, d_step_histogram,
          d_step_count, kernel_ray_order, d_ray_steps_out, d_unabsorbed,
          d_tail_closure_count, d_tail_closure_absorbed_power,
          d_critical_surface_hit_count, d_error_flags, CbetRecordDeviceArgs{},
          HotECaptureParams{}, nullptr, laser::LaserPhysExtOptions{}, nullptr,
          nullptr, d_tau_shell_out);
    }
  }
  if (d_pabs_per_ray_out != nullptr) {
    TENRYU_ASSERT(d_per_ray_deposit != nullptr,
                  "ray_trace_1d_sph pabs diagnostic requires per-ray tallies");
    sum_absorbed_power_per_ray_1d_kernel<<<grid, block, 0, stream>>>(
        d_per_ray_deposit, rays.n_rays, n_hydro_cells, d_pabs_per_ray_out);
  }
  if (h_ray_steps_out != nullptr) {
    const cudaError_t steps_status =
        cudaMemcpyAsync(h_ray_steps_out, d_ray_steps_out, ray_index_bytes,
                        cudaMemcpyDeviceToHost, stream);
    if (steps_status != cudaSuccess) {
      return steps_status;
    }
  }
  if (d_per_ray_deposit != nullptr) {
    const int red_total = n_hydro_cells + 3;
    const int red_grid = (red_total + block - 1) / block;
    reduce_per_ray_tallies_1d_kernel<<<red_grid, block, 0, stream>>>(
        d_per_ray_deposit, d_per_ray_unabsorbed, d_per_ray_tail, d_deposit_1d,
        d_unabsorbed, d_tail_closure_absorbed_power, rays.n_rays,
        n_hydro_cells, d_ra_per_ray,
        d_ra_per_ray != nullptr ? d_ra_power_total : nullptr);
  }
  return cudaGetLastError();
}

cudaError_t launch_radial_absorption_1d(double P_total,
                                        const LaserMesh& mesh,
                                        const core::Config::LaserConfig& laser_cfg,
                                        const double* d_hydro_r_edges,
                                        const int n_hydro_cells,
                                        double* d_deposit_1d,
                                        double* d_unabsorbed,
                                        core::DeviceErrorFlags* d_error_flags,
                                        unsigned long long* d_critical_surface_hit_count,
                                        cudaStream_t stream,
                                        const HotECaptureParams hot_e_params,
                                        double** d_hot_e_capture_out) {
  if (!(P_total > 0.0)) {
    return cudaSuccess;
  }
  TENRYU_ASSERT(mesh.radial_n_nodes >= 2,
                "radial_absorption_1d requires at least 2 radial nodes");
  TENRYU_ASSERT(d_hydro_r_edges != nullptr, "radial_absorption_1d requires hydro_r_edges");
  TENRYU_ASSERT(d_deposit_1d != nullptr, "radial_absorption_1d requires deposit_1d");
  const bool hot_e_capture_enabled = hot_e_params.n_channels > 0;

  double* d_hot_e_capture = nullptr;
  if (hot_e_capture_enabled) {
    d_hot_e_capture = static_cast<double*>(core::device_scratch_acquire(
        "hot_electron:radial_capture",
        static_cast<std::size_t>(3) * hot_e_params.n_channels * sizeof(double)));
    cudaMemsetAsync(d_hot_e_capture, 0,
                    static_cast<std::size_t>(3) * hot_e_params.n_channels * sizeof(double),
                    stream);
  }
  if (d_hot_e_capture_out != nullptr) {
    *d_hot_e_capture_out = d_hot_e_capture;
  }

  if (hot_e_capture_enabled) {
    radial_absorption_1d_kernel<true><<<1, 1, 0, stream>>>(
        P_total, d_hydro_r_edges, mesh.radial_node_r, mesh.radial_n_hat,
        mesh.radial_n_hat_raw, mesh.radial_smooth_kappa, laser_cfg.absorption.eps_n,
        laser_cfg.raytrace.eps_crit, laser_cfg.raytrace.test_kappa,
        laser_cfg.raytrace.intensity_cutoff, n_hydro_cells, mesh.radial_n_nodes, d_deposit_1d,
        d_unabsorbed, d_critical_surface_hit_count, d_error_flags,
        hot_e_params, d_hot_e_capture);
  } else {
    radial_absorption_1d_kernel<false><<<1, 1, 0, stream>>>(
        P_total, d_hydro_r_edges, mesh.radial_node_r, mesh.radial_n_hat,
        mesh.radial_n_hat_raw, mesh.radial_smooth_kappa, laser_cfg.absorption.eps_n,
        laser_cfg.raytrace.eps_crit, laser_cfg.raytrace.test_kappa,
        laser_cfg.raytrace.intensity_cutoff, n_hydro_cells, mesh.radial_n_nodes, d_deposit_1d,
        d_unabsorbed, d_critical_surface_hit_count, d_error_flags, HotECaptureParams{}, nullptr);
  }
  return cudaGetLastError();
}

cudaError_t launch_ray_trace_3d(const RayArray2D& rays,
                                const LaserMesh& mesh,
                                const core::Config::LaserConfig& laser_cfg,
                                const double lambda_cm,
                                double* d_traj_pos1,
                                double* d_traj_pos2,
                                double* d_traj_pos3,
                                double* d_traj_power,
                                int* d_traj_step_count,
                                const int n_output_rays,
                                const int output_stride,
                                const int traj_max_steps,
                                double* d_unabsorbed,
                                core::DeviceErrorFlags* d_error_flags,
                                const HotECaptureParams& hot_e_params,
                                double* hot_e_capture,
                                unsigned long long* d_tail_closure_count,
                                double* d_tail_closure_absorbed_power,
                                unsigned long long* d_critical_surface_hit_count,
                                cudaStream_t stream,
                                int* d_step_histogram,
                                const double* smooth_kappa_factor,
                                int* d_step_count,
                                const bool allow_debug_one_ray,
                                const CbetRecordDeviceArgs* cbet_record) {
  if (rays.n_rays <= 0) {
    return cudaSuccess;
  }
  TENRYU_ASSERT(std::isfinite(laser_cfg.raytrace.cfl_ray) &&
                    laser_cfg.raytrace.cfl_ray > 0.0,
                "ray_trace_3d requires Laser.raytrace.cfl_ray > 0");
  const bool cbet_record_mode =
      (cbet_record != nullptr && cbet_record->rec_cell != nullptr);
  TENRYU_ASSERT(!(cbet_record_mode && allow_debug_one_ray && laser_cfg.raytrace.debug_one_ray),
                "ray_trace_3d record mode cannot be combined with debug_one_ray");

  const int max_ray_steps = std::min(std::max(laser_cfg.raytrace.max_steps, 1), kMaxRayStepsGuard);
  const int block = 64;
  const int grid = (rays.n_rays + block - 1) / block;
  auto launch_kernel = [&](auto debug_tag, auto hot_e_tag) {
    constexpr bool kDebugOneRay = decltype(debug_tag)::value;
    constexpr bool kHotECapture = decltype(hot_e_tag)::value;
    ray_trace_3d<kDebugOneRay, false, kHotECapture><<<grid, block, 0, stream>>>(
        mesh.deposit, mesh.n_e_hat, mesh.n_e_hat_raw, mesh.grad_n_hat_R, mesh.grad_n_hat_Z,
        mesh.T_e, mesh.Zbar, smooth_kappa_factor, mesh.node_R, mesh.node_Z, rays.x0, rays.y0,
        rays.z0, rays.vx0, rays.vy0, rays.vz0, rays.power, rays.power0,
        laser_cfg.raytrace.cfl_ray, laser_cfg.raytrace.ds_adapt_g_target,
        laser_cfg.raytrace.ds_adapt_tau_target, laser_cfg.raytrace.ds_adapt_theta_target,
        laser_cfg.raytrace.ds_adapt_max_factor,
        laser_cfg.absorption.eps_n, laser_cfg.raytrace.eps_crit, lambda_cm,
        laser_cfg.absorption.coulomb_log_floor, laser_cfg.raytrace.test_kappa,
        laser_cfg.raytrace.intensity_cutoff, max_ray_steps, mesh.n_nodes_r, mesh.n_nodes_z,
        rays.n_rays, std::max(mesh.dx_min, 1.0e-30), d_traj_pos1, d_traj_pos2, d_traj_pos3,
        d_traj_power, d_traj_step_count, n_output_rays, output_stride, traj_max_steps,
        d_step_histogram, d_step_count, d_unabsorbed, d_tail_closure_count,
        d_tail_closure_absorbed_power, d_critical_surface_hit_count, d_error_flags,
        CbetRecordDeviceArgs{}, hot_e_params, hot_e_capture);
  };
  if (cbet_record_mode) {
    ray_trace_3d<false, true, false><<<grid, block, 0, stream>>>(
        nullptr, mesh.n_e_hat, mesh.n_e_hat_raw, mesh.grad_n_hat_R, mesh.grad_n_hat_Z,
        mesh.T_e, mesh.Zbar, smooth_kappa_factor, mesh.node_R, mesh.node_Z, rays.x0, rays.y0,
        rays.z0, rays.vx0, rays.vy0, rays.vz0, rays.power, rays.power0,
        laser_cfg.raytrace.cfl_ray, laser_cfg.raytrace.ds_adapt_g_target,
        laser_cfg.raytrace.ds_adapt_tau_target, laser_cfg.raytrace.ds_adapt_theta_target,
        laser_cfg.raytrace.ds_adapt_max_factor,
        laser_cfg.absorption.eps_n, laser_cfg.raytrace.eps_crit, lambda_cm,
        laser_cfg.absorption.coulomb_log_floor, laser_cfg.raytrace.test_kappa,
        laser_cfg.raytrace.intensity_cutoff, max_ray_steps, mesh.n_nodes_r, mesh.n_nodes_z,
        rays.n_rays, std::max(mesh.dx_min, 1.0e-30), d_traj_pos1, d_traj_pos2, d_traj_pos3,
        d_traj_power, d_traj_step_count, n_output_rays, output_stride, traj_max_steps,
        d_step_histogram, d_step_count, d_unabsorbed, d_tail_closure_count,
        d_tail_closure_absorbed_power, d_critical_surface_hit_count, d_error_flags,
        *cbet_record, HotECaptureParams{}, nullptr);
  } else if (allow_debug_one_ray && laser_cfg.raytrace.debug_one_ray) {
    if (hot_e_params.n_channels > 0) {
      launch_kernel(std::true_type{}, std::true_type{});
    } else {
      launch_kernel(std::true_type{}, std::false_type{});
    }
  } else {
    if (hot_e_params.n_channels > 0) {
      launch_kernel(std::false_type{}, std::true_type{});
    } else {
      launch_kernel(std::false_type{}, std::false_type{});
    }
  }
  return cudaGetLastError();
}

cudaError_t launch_reduce_per_ray_tallies_1d(
    const double* d_deposit_per_ray,
    const double* d_unabsorbed_per_ray,
    const double* d_tail_power_per_ray,
    double* d_deposit_1d,
    double* d_P_unabsorbed,
    double* d_tail_closure_absorbed_power,
    const int n_rays,
    const int n_cells,
    cudaStream_t stream,
    const double* d_ra_per_ray,
    double* d_ra_total) {
  if (n_rays <= 0 || n_cells < 0) {
    return cudaSuccess;
  }
  const int block = 64;
  const int red_total = n_cells + 3;
  const int red_grid = (red_total + block - 1) / block;
  reduce_per_ray_tallies_1d_kernel<<<red_grid, block, 0, stream>>>(
      d_deposit_per_ray, d_unabsorbed_per_ray, d_tail_power_per_ray, d_deposit_1d,
      d_P_unabsorbed, d_tail_closure_absorbed_power, n_rays, n_cells,
      d_ra_per_ray, d_ra_total);
  return cudaGetLastError();
}

}  // namespace tenryu::laser
