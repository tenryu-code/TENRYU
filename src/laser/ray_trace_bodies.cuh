#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "core/device_error_flags.cuh"
#include "laser/bilinear_interpolation.cuh"
#include "laser/coordinate_transform.cuh"
#include "laser/ib_absorption.cuh"
#include "laser/laser_phys_ext.cuh"
#include "laser/ray_trace.cuh"

namespace tenryu::laser::ray_trace_bodies {

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
// Floor for the turning-arc angular limiter's shrink factor: prevents a
// pathological stall if g_metric is huge on a degenerate mesh;
// raytrace.max_steps remains the hard guard (2026-07-31).
constexpr double kDsAdaptThetaMinFactor = 1.0e-4;

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

TENRYU_HOST_DEVICE inline double theta_step_multiplier(
    const double g_metric,
    const double n_hat_raw,
    const double ds_adapt_theta_target) {
  // Turning-arc angular control (2026-07-31): leapfrog rotates the ray
  // by dtheta ~= |grad n_hat| ds / (2 |v|^2) per step, |v|^2 = 1 - n_hat.
  // The g/tau targets only cap step GROWTH (m >= 1), so a coarse
  // ds_local under-resolves the arc regardless of them. Cap the
  // rotation at ds_adapt_theta_target (rad):
  //   m_theta = 2 * theta_t * (1 - n_hat) / (|g| * ds_local),
  // allowed to shrink below 1. <= 0 disables (returns no-op cap).
  if (!(ds_adapt_theta_target > 0.0)) {
    return 1.0e300;
  }
  const double one_minus = ::fmax(1.0 - n_hat_raw, 1.0e-6);
  const double m_theta = 2.0 * ds_adapt_theta_target * one_minus /
                         ::fmax(g_metric, 1.0e-30);
  return ::fmax(kDsAdaptThetaMinFactor, m_theta);
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
  // Direction gate (2026-07-26 review): the tail closure models the remaining
  // inbound path to the critical surface, so it must only fire for rays
  // moving UP the density gradient. Without this, an outbound ray (after a
  // subcritical turning) re-entering the near-critical band could absorb a
  // fictitious critical-layer tail a second time and be terminated.
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

__device__ __noinline__ inline void handle_critical_surface_2d(
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

__device__ __noinline__ inline void handle_critical_surface_3d(
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
  int* ray_steps_out = nullptr;
  int ray_idx = -1;
  int n_rays_capacity = 0;

  __device__ inline RayStepHistogramGuard(int* step_histogram_in, int* n_steps_in)
      : step_histogram(step_histogram_in), n_steps(n_steps_in) {}

  __device__ inline RayStepHistogramGuard(int* step_histogram_in,
                                          int* n_steps_in,
                                          int* step_count_in,
                                          int* ray_steps_out_in,
                                          int ray_idx_in,
                                          int n_rays_capacity_in)
      : step_histogram(step_histogram_in),
        n_steps(n_steps_in),
        step_count(step_count_in),
        ray_steps_out(ray_steps_out_in),
        ray_idx(ray_idx_in),
        n_rays_capacity(n_rays_capacity_in) {}

  __device__ inline ~RayStepHistogramGuard() {
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
    if (ray_steps_out != nullptr) {
      ray_steps_out[ray_idx] = *n_steps;
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

  __device__ inline explicit DepositCacheGuard(double* deposit_in) : deposit(deposit_in) {}

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

  __device__ inline ~DepositCacheGuard() { flush(); }
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
    double& I_after) {
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

TENRYU_HOST_DEVICE inline int walk_interval_1d(const double* nodes,
                                               const int n_nodes,
                                               const double x,
                                               const int hint) {
  if (n_nodes <= 2) {
    return 0;
  }
  if (x <= nodes[0]) {
    return 0;
  }
  if (x >= nodes[n_nodes - 1]) {
    return n_nodes - 2;
  }
  if (x != x) {
    return locate_interval_1d_device(nodes, n_nodes, x);
  }

  int i = hint;
  if (i < 0) {
    i = 0;
  } else if (i > n_nodes - 2) {
    i = n_nodes - 2;
  }

  int walk_count = 0;
  while (i < n_nodes - 2 && nodes[i + 1] <= x) {
    if (walk_count >= 32) {
      return locate_interval_1d_device(nodes, n_nodes, x);
    }
    ++i;
    ++walk_count;
  }
  while (i > 0 && nodes[i] > x) {
    if (walk_count >= 32) {
      return locate_interval_1d_device(nodes, n_nodes, x);
    }
    --i;
    ++walk_count;
  }
  return i;
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

TENRYU_HOST_DEVICE inline RadialInterval locate_radial_interval(
    const double* radial_node_r,
    const int n_radial_nodes,
    const double r,
    const int hint) {
  const int i = walk_interval_1d(radial_node_r, n_radial_nodes, r, hint);
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

// Smallest t in (0, 1] where the straight chord P(t) = P0 + t*(P1-P0)
// first crosses INTO the sphere |P| = r_s (entry from outside), or a
// negative value when the chord never enters. Used by the
// vacuum->material entry event locator (2026-07-31): in vacuum the
// march is exactly straight, so this crossing is exact.
TENRYU_HOST_DEVICE inline double first_entry_fraction_radial(
    const double R0, const double Z0,
    const double R1, const double Z1,
    const double r_s) {
  const double dR = R1 - R0;
  const double dZ = Z1 - Z0;
  const double a = dR * dR + dZ * dZ;
  const double b = R0 * dR + Z0 * dZ;                 // = 0.5 d|P|^2/dt
  const double c = R0 * R0 + Z0 * Z0 - r_s * r_s;     // > 0 outside
  if (!(a > 0.0) || !(c > 0.0)) {
    return -1.0;
  }
  const double disc = b * b - a * c;
  if (!(disc >= 0.0)) {
    return -1.0;
  }
  // smaller root, numerically stable for b < 0 (approaching):
  const double sq = ::sqrt(disc);
  const double t = (b < 0.0) ? (c / (-b + sq)) : ((-b - sq) / a);
  if (!(t > 0.0) || !(t <= 1.0)) {
    return -1.0;
  }
  return t;
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

  __device__ inline explicit DepositCellCacheGuard(double* deposit_in) : deposit(deposit_in) {}

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

  __device__ inline ~DepositCellCacheGuard() { flush(); }
};

__device__ __forceinline__ void accumulate_deposit_1d(
    DepositCellCacheGuard& deposit_cache,
    const double* hydro_r_edges,
    const int n_hydro_cells,
    int* hydro_hint,
    const int allowed_supercritical_cell,
    const int critical_adjacent_subcritical_cell,
    const double critical_adjacent_split_r,
    const double r,
    const double absorbed_power) {
  if (!(absorbed_power > 0.0) || hydro_r_edges == nullptr || n_hydro_cells <= 0) {
    return;
  }
  int c = walk_interval_1d(hydro_r_edges, n_hydro_cells + 1, r, *hydro_hint);
  *hydro_hint = c;
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
    int* hydro_hint,
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

  accumulate_deposit_1d(deposit_cache, hydro_r_edges, n_hydro_cells, hydro_hint,
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


template <bool kCbetRecord, bool kHotECapture, bool kPhysExt>
__device__ inline void ray_trace_1d_sph_body(const int ray,
                      double* __restrict__ deposit_1d,
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
                      double* __restrict__ traj_pos_R,
                      double* __restrict__ traj_pos_Z,
                      double* __restrict__ traj_power,
                      int* __restrict__ traj_step_count,
                      const int n_output_rays,
                      const int output_stride,
                      const int traj_max_steps,
                      int* __restrict__ step_histogram,
                      int* __restrict__ step_count,
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
  (void)lambda_cm;
  (void)coulomb_log_floor;

  const int tid = ray;
  const int output_idx = (output_stride > 0) ? (tid / output_stride) : -1;
  const bool traj_on = (traj_pos_R != nullptr) && (output_stride > 0) &&
                       (tid % output_stride == 0) && (output_idx < n_output_rays);
  int traj_stored = 0;
  int n_steps = 0;
  RayStepHistogramGuard histogram_guard(
      step_histogram, &n_steps, step_count, ray_steps_out, tid, n_rays);
  // Fixed-order tally-reduction reproducibility contract: when per-ray buffers are provided each
  // ray owns an exclusive row, so the cached atomic adds are contention-free
  // and the later fixed-order reduction makes the deposition bitwise
  // deterministic. Null buffers fall back to the legacy shared-atomic tally.
  double* deposit_target =
      (deposit_per_ray != nullptr)
          ? deposit_per_ray + static_cast<std::size_t>(tid) *
                                  static_cast<std::size_t>(n_hydro_cells)
          : deposit_1d;
  DepositCellCacheGuard deposit_cache(deposit_target);
  int hydro_hint = n_hydro_cells / 2;
  double* tail_slot =
      (tail_power_per_ray != nullptr) ? &tail_power_per_ray[tid] : nullptr;

  double R = ray_R0[tid];
  double Z = ray_Z0[tid];
  double vR = ray_vR0[tid];
  double vZ = ray_vZ0[tid];
  double I = ray_power[tid];
  [[maybe_unused]] unsigned hot_e_captured_mask = 0u;
  const double I0 = ray_power0[tid];
  [[maybe_unused]] double ra_r_prev = -1.0;
  [[maybe_unused]] double ra_nh_prev = 0.0;
  [[maybe_unused]] bool ra_done = false;
  CbetRecordCursor cbet_cursor;
  if constexpr (kCbetRecord) {
    if (cbet_args.rec_cell != nullptr && cbet_args.cap_per_ray > 0) {
      cbet_cursor.rec_cell = cbet_args.rec_cell;
      cbet_cursor.rec_mu = cbet_args.rec_mu;
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

  const bool finite_init = ::isfinite(R) && ::isfinite(Z) && ::isfinite(vR) && ::isfinite(vZ) &&
                           ::isfinite(I) && ::isfinite(I0);
  if (!finite_init) {
    if (error_flags != nullptr) {
      atomicExch(&error_flags->nan_particle, 1);
    }
    if (::isfinite(I) && I > 0.0) {
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
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
    if (cbet_args.traj_rec_idx != nullptr) {
      cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
    }
    ++traj_stored;
  }
  if (outside_radial_profile(R, Z, radial_node_r, n_radial_nodes)) {
    if (!advance_to_radial_profile_entry(&R, &Z, &vR, vZ, radial_node_r, n_radial_nodes) ||
        outside_radial_profile(R, Z, radial_node_r, n_radial_nodes)) {
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (traj_on && traj_stored < traj_max_steps) {
      const int idx = output_idx * traj_max_steps + traj_stored;
      traj_pos_R[idx] = R;
      traj_pos_Z[idx] = Z;
      traj_power[idx] = I;
      if (cbet_args.traj_rec_idx != nullptr) {
        cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
      }
      ++traj_stored;
    }
  }

  RadialInterval carried_c;
  double carried_nh = 0.0;
  double carried_nh_raw = 0.0;
  double carried_kappa = 0.0;
  double ds_prev = 0.0;
  {
    const double r = radial_distance(R, Z);
    carried_c = locate_radial_interval(radial_node_r, n_radial_nodes, r);
    const double nh0 = interpolate_radial_field(radial_n_hat, carried_c);
    const double nh0_raw = interpolate_radial_field(radial_n_hat_raw, carried_c);
    if (!::isfinite(nh0) || !::isfinite(nh0_raw)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh0 >= 1.0 - eps_crit) {
      if constexpr (kPhysExt) {
        if (phys_opt.crit_terminate_deposit != 0) {
          int dep_cell = critical_adjacent_subcritical_cell;
          if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
            dep_cell = allowed_supercritical_cell;
          }
          if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
            deposit_cache.accumulate(dep_cell, I);
            if (traj_on) {
              traj_step_count[output_idx] = traj_stored;
            }
            return;
          }
        }
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    double kappa0 = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa0 = test_kappa_cm_inv;
    } else {
      const double A0 = interpolate_radial_field(radial_smooth_kappa, carried_c);
      kappa0 = compute_kappa_from_smooth(A0, nh0, eps_n);
      if constexpr (kPhysExt) {
        if (phys_opt.langdon_model != 0 && kappa0 > 0.0 &&
            radial_T_e != nullptr) {
          const double Te0 = interpolate_radial_field(radial_T_e, carried_c);
          const double I_vac = vacuum_map_intensity(
              phys_opt.langdon_I0_wcm2, phys_opt.langdon_w_cm, R,
              phys_opt.langdon_profile_kind, phys_opt.langdon_sg_two_m);
          kappa0 *= compute_langdon_factor(phys_opt.langdon_model,
                                           phys_opt.langdon_zcoll, I_vac,
                                           lambda_cm, Te0,
                                           phys_opt.langdon_te_min_eV);
        }
      }
    }

    const double dn_dr0 = radial_dn_dr[carried_c.i];
    const double r_inv = 1.0 / ::fmax(r, 1.0e-30);
    const double gR = (r > 1.0e-30) ? dn_dr0 * R * r_inv : 0.0;
    const double gZ = (r > 1.0e-30) ? dn_dr0 * Z * r_inv : 0.0;
    if (!::isfinite(gR) || !::isfinite(gZ) || !::isfinite(kappa0) || kappa0 < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    // Launch speed |v| = sqrt(1 - n_hat) at the profile entry point (H = 1;
    // 2026-07-26 review). Vacuum entry (nh0 = 0) multiplies by exactly 1.0.
    {
      const double v_entry_scale = ::sqrt(::fmax(0.0, 1.0 - nh0));
      vR *= v_entry_scale;
      vZ *= v_entry_scale;
    }
    // Variable-step Verlet (2026-07-31): no seed kick. Each iteration
    // completes the previous step's second half-kick with the local gradient
    // (0.25*ds_prev*g) and applies its own first half (0.25*ds_cur*g) before
    // the drift; ds_prev = 0 starts the chain. The former staggered seeding
    // (v^{1/2} = v - 0.25*ds*g) lost O(g*dds) energy per step under adaptive
    // ds — measured 0.027 turning-depth deficit under the theta limiter.
    if (!::isfinite(vR) || !::isfinite(vZ)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    carried_nh = nh0;
    carried_nh_raw = nh0_raw;
    carried_kappa = kappa0;
  }

  // Vacuum->material entry hand-off (2026-07-31): when a pure-vacuum
  // segment (kappa = 0, g = 0, exactly straight) crosses the material
  // surface, truncate it at the exact chord-sphere crossing and set
  // the next carried interval to the material side explicitly. A
  // surface-straddling segment otherwise runs vacuum constants into
  // the shell (measured: n_hat 0 -> 0.037 unkicked and untaxed, ray
  // turns 0.027 shallow); locate() at the landed node is ambiguous, so
  // the hand-off pins the interval instead of nudging the position.
  int entry_handoff_interval = -1;

  while (n_steps < max_ray_steps) {
    ++n_steps;
    const double old_R = R;
    const double old_Z = Z;
    const double s_old = old_R * old_R + old_Z * old_Z;
    const double r_inv_old = (s_old > 1.0e-60) ? tenryu_rsqrt(s_old) : 0.0;
    const double r_old = s_old * r_inv_old;

    const RadialInterval c_old = carried_c;
    const double ds_local = local_ds_radial(radial_node_r, c_old, cfl_ray);
    const double dn_dr_entry = radial_dn_dr[c_old.i];
    const double gR = (r_old > 1.0e-30) ? dn_dr_entry * old_R * r_inv_old : 0.0;
    const double gZ = (r_old > 1.0e-30) ? dn_dr_entry * old_Z * r_inv_old : 0.0;
    const double g_mag = ::fabs(dn_dr_entry);
    const double nh_entry = carried_nh;
    const double nh_entry_raw = carried_nh_raw;
    const double kappa_entry = carried_kappa;
    if constexpr (kPhysExt) {
      if (phys_opt.ra_enable != 0 && !ra_done && ra_r_prev >= 0.0 &&
          r_old > ra_r_prev) {
        const double n_refr =
            ::sqrt(::fmax(1.0 - ra_nh_prev, 0.0));
        const double b_eff = n_refr * ra_r_prev;
        const double f_ra = compute_ra_event_fraction(phys_opt, b_eff);
        if (f_ra > 0.0 && I > 0.0 && ra_per_ray != nullptr &&
            critical_adjacent_subcritical_cell >= 0 &&
            critical_adjacent_subcritical_cell < n_hydro_cells) {
          const double dP = f_ra * I;
          I -= dP;
          ra_per_ray[tid] += dP;
          deposit_cache.accumulate(
              critical_adjacent_subcritical_cell, dP);
        }
        ra_done = true;
      }
      ra_r_prev = r_old;
      ra_nh_prev = nh_entry;
    }
    if (!::isfinite(gR) || !::isfinite(gZ) || !::isfinite(g_mag) || !::isfinite(nh_entry) ||
        !::isfinite(nh_entry_raw) || !::isfinite(kappa_entry)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_entry >= 1.0 - eps_crit) {
      if constexpr (kPhysExt) {
        if (phys_opt.crit_terminate_deposit != 0) {
          int dep_cell = critical_adjacent_subcritical_cell;
          if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
            dep_cell = allowed_supercritical_cell;
          }
          if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
            deposit_cache.accumulate(dep_cell, I);
            if (traj_on) {
              traj_step_count[output_idx] = traj_stored;
            }
            return;
          }
        }
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_entry_raw >= kCritLayerHandoffNhatRaw) {
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure_1d(deposit_cache, c_old, radial_smooth_kappa, radial_n_hat_raw,
                              hydro_r_edges, n_hydro_cells, &hydro_hint,
                              allowed_supercritical_cell,
                              critical_adjacent_subcritical_cell, critical_adjacent_split_r,
                              test_kappa_cm_inv, nh_entry, nh_entry_raw, kappa_entry, g_mag,
                              dn_dr_entry * (vR * old_R + vZ * old_Z) * r_inv_old,
                              v_mag2, r_old, I,
                              TailClosureMode::kRequireTrigger, tail_closure_count,
                              tail_closure_absorbed_power, tail_slot,
                              error_flags, I_tail);
      if constexpr (kCbetRecord) {
        if (tail_status == TailClosureStatus::kClosed) {
          int tail_cell = -1;
          double tau_tail = 0.0;
          if (cbet_probe_tail_1d(c_old, radial_smooth_kappa, radial_n_hat_raw,
                                 hydro_r_edges, n_hydro_cells,
                                 allowed_supercritical_cell,
                                 critical_adjacent_subcritical_cell,
                                 critical_adjacent_split_r, test_kappa_cm_inv,
                                 nh_entry, kappa_entry, g_mag, r_old, &tail_cell,
                                 &tau_tail)) {
            cbet_cursor.add_terminal(tail_cell, tau_tail);
          } else {
            cbet_cursor.flush();
          }
        }
      }
      if (tail_status == TailClosureStatus::kInvalid) {
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
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
          if (cbet_args.traj_rec_idx != nullptr) {
            cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
          }
          ++traj_stored;
        }
        if constexpr (kPhysExt) {
          if (phys_opt.crit_terminate_deposit != 0) {
            int dep_cell = critical_adjacent_subcritical_cell;
            if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
              dep_cell = allowed_supercritical_cell;
            }
            if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
              deposit_cache.accumulate(dep_cell, I);
              if (traj_on) {
                traj_step_count[output_idx] = traj_stored;
              }
              return;
            }
          }
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
    }

    const double g_metric = g_mag * ds_local;
    const double tau_metric = carried_kappa * ds_local;
    const double g_den = ::fmax(g_metric, 1.0e-30);
    const double tau_den = ::fmax(tau_metric, 1.0e-30);
    // Products cannot overflow for physical targets O(1e-2..1) and
    // metrics O(1e-30..1e3).
    const bool use_g_term =
        ds_adapt_g_target * tau_den <= ds_adapt_tau_target * g_den;
    double m = use_g_term ? ds_adapt_g_target / g_den
                          : ds_adapt_tau_target / tau_den;
    m = ::fmin(m, ds_adapt_max_factor);
    m = clamp_adaptive_step_multiplier(m, carried_nh_raw, ds_adapt_max_factor);
    m = ::fmin(m, theta_step_multiplier(g_metric, carried_nh_raw,
                                        ds_adapt_theta_target));

    // Complete the previous step's second half-kick with the local
    // gradient, then apply this step's first half (variable-step
    // Verlet; see seed comment).
    vR -= 0.25 * ds_prev * gR;
    vZ -= 0.25 * ds_prev * gZ;
    const double ds_cur = ds_local * m;
    vR -= 0.25 * ds_cur * gR;
    vZ -= 0.25 * ds_cur * gZ;

    R += ds_cur * vR;
    Z += ds_cur * vZ;
    const double R_unreflected = R;
    reflect_axis_if_needed(&R, &vR);

    if (!::isfinite(R) || !::isfinite(Z) || !::isfinite(vR) || !::isfinite(vZ)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->nan_particle, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    const bool exited_mesh =
        outside_radial_profile(R_unreflected, Z, radial_node_r, n_radial_nodes);
    double t_mesh = 1.0;
    if (exited_mesh) {
      t_mesh = first_exit_fraction_radial(old_R, old_Z, R_unreflected, Z,
                                          radial_node_r[n_radial_nodes - 1]);
      if (!(t_mesh > 0.0)) {
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
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
    if (!::isfinite(nh_old) || !::isfinite(nh_old_raw) || !::isfinite(kappa_old)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_old >= 1.0 - eps_crit) {
      if constexpr (kPhysExt) {
        if (phys_opt.crit_terminate_deposit != 0) {
          int dep_cell = critical_adjacent_subcritical_cell;
          if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
            dep_cell = allowed_supercritical_cell;
          }
          if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
            deposit_cache.accumulate(dep_cell, I);
            if (traj_on) {
              traj_step_count[output_idx] = traj_stored;
            }
            return;
          }
        }
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_old_raw >= kCritLayerHandoffNhatRaw) {
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure_1d(deposit_cache, c_old, radial_smooth_kappa, radial_n_hat_raw,
                              hydro_r_edges, n_hydro_cells, &hydro_hint,
                              allowed_supercritical_cell,
                              critical_adjacent_subcritical_cell, critical_adjacent_split_r,
                              test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag,
                              dn_dr_entry * (vR * old_R + vZ * old_Z) * r_inv_old,
                              v_mag2, r_old, I,
                              TailClosureMode::kRequireTrigger, tail_closure_count,
                              tail_closure_absorbed_power, tail_slot,
                              error_flags, I_tail);
      if constexpr (kCbetRecord) {
        if (tail_status == TailClosureStatus::kClosed) {
          int tail_cell = -1;
          double tau_tail = 0.0;
          if (cbet_probe_tail_1d(c_old, radial_smooth_kappa, radial_n_hat_raw,
                                 hydro_r_edges, n_hydro_cells,
                                 allowed_supercritical_cell,
                                 critical_adjacent_subcritical_cell,
                                 critical_adjacent_split_r, test_kappa_cm_inv,
                                 nh_old, kappa_old, g_mag, r_old, &tail_cell,
                                 &tau_tail)) {
            cbet_cursor.add_terminal(tail_cell, tau_tail);
          } else {
            cbet_cursor.flush();
          }
        }
      }
      if (tail_status == TailClosureStatus::kInvalid) {
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
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
          if (cbet_args.traj_rec_idx != nullptr) {
            cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
          }
          ++traj_stored;
        }
        if constexpr (kPhysExt) {
          if (phys_opt.crit_terminate_deposit != 0) {
            int dep_cell = critical_adjacent_subcritical_cell;
            if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
              dep_cell = allowed_supercritical_cell;
            }
            if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
              deposit_cache.accumulate(dep_cell, I);
              if (traj_on) {
                traj_step_count[output_idx] = traj_stored;
              }
              return;
            }
          }
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
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

    entry_handoff_interval = -1;
    if (kappa_old == 0.0 && nh_old_raw <= 0.0 && g_mag == 0.0) {
      int idx_m = -1;
      for (int k = n_radial_nodes - 1; k >= 0; --k) {
        if (radial_n_hat_raw[k] > 0.0) {
          idx_m = k;
          break;
        }
      }
      if (idx_m >= 0 && idx_m + 1 < n_radial_nodes) {
        const double r_surf = radial_node_r[idx_m + 1];
        if (r_old > r_surf) {
          const double t_surf = first_entry_fraction_radial(
              old_R, old_Z, old_R + t_stop * dR_path,
              old_Z + t_stop * dZ_path, r_surf);
          if (t_surf > 0.0) {
            t_stop = clamp_unit_interval(t_stop * t_surf);
            entry_handoff_interval = idx_m;
          }
        }
      }
    }

    double R_stop_path = old_R + t_stop * dR_path;
    double R_stop = ::abs(R_stop_path);
    double Z_stop = old_Z + t_stop * dZ_path;
    double r_stop = radial_distance(R_stop, Z_stop);
    RadialInterval c_stop =
        locate_radial_interval(radial_node_r, n_radial_nodes, r_stop, carried_c.i);
    double nh_stop = interpolate_radial_field(radial_n_hat, c_stop);
    double nh_stop_raw = interpolate_radial_field(radial_n_hat_raw, c_stop);
    if (entry_handoff_interval >= 0) {
      // Pin the stop interval to the material side of the surface node;
      // t = 1 is the upper node of interval idx_m, i.e. the surface
      // itself, so the interpolated stop values (n_hat = 0 there) are
      // unchanged while the NEXT segment's ds_local, dn_dr and kappa
      // come from the material interval.
      c_stop.i = entry_handoff_interval;
      c_stop.t = 1.0;
      nh_stop = interpolate_radial_field(radial_n_hat, c_stop);
      nh_stop_raw = interpolate_radial_field(radial_n_hat_raw, c_stop);
    }
    if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw)) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    // Phase A — detect threshold crossings in this segment and stage them.
    // The power accounting is deferred so the recorded capture power is the
    // ray power AT the crossing (P^cross, NUMERICS 5.11), i.e. after the IB
    // absorption of the pre-crossing sub-segment, not the segment-entry
    // power (2026-07-26 review). Channels are sorted by f_s ascending
    // host-side, so on a rising segment the staged fracs are ascending.
    [[maybe_unused]] int hot_e_n_pend = 0;
    [[maybe_unused]] int hot_e_pend_ch[HotECaptureParams::kMaxChannels];
    [[maybe_unused]] double hot_e_pend_t[HotECaptureParams::kMaxChannels];
    [[maybe_unused]] double hot_e_pend_r[HotECaptureParams::kMaxChannels];
    [[maybe_unused]] double hot_e_pend_mu[HotECaptureParams::kMaxChannels];
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
        const double he_frac = (he_denom > 0.0)
            ? clamp_unit_interval((he_thr - nh_old_raw) / he_denom)
            : 0.0;
        const double he_R = old_R + he_frac * t_stop * dR_path;  // path coords, NO abs
        const double he_Z = old_Z + he_frac * t_stop * dZ_path;
        const double he_r = radial_distance(he_R, he_Z);
        const double he_vnorm = sqrt(vR * vR + vZ * vZ);
        double he_mu = 0.0;
        if (he_r > 0.0 && he_vnorm > 0.0) {
          he_mu = (vR * he_R + vZ * he_Z) / (he_r * he_vnorm);
          if (he_mu < -1.0) he_mu = -1.0;
          if (he_mu > 1.0) he_mu = 1.0;
        }
        hot_e_pend_ch[hot_e_n_pend] = he_ch;
        hot_e_pend_t[hot_e_n_pend] = he_frac * t_stop;
        hot_e_pend_r[hot_e_n_pend] = he_r;
        hot_e_pend_mu[hot_e_n_pend] = he_mu;
        ++hot_e_n_pend;
      }
    }

    const double nh_crit = 1.0 - eps_crit;
    if (nh_old_raw >= nh_crit) {
      if constexpr (kPhysExt) {
        if (phys_opt.crit_terminate_deposit != 0) {
          int dep_cell = critical_adjacent_subcritical_cell;
          if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
            dep_cell = allowed_supercritical_cell;
          }
          if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
            deposit_cache.accumulate(dep_cell, I);
            if (traj_on) {
              traj_step_count[output_idx] = traj_stored;
            }
            return;
          }
        }
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (nh_stop_raw >= nh_crit) {
      if constexpr (kHotECapture) {
        // Critical-crossing segments are handed to the tail closure, whose
        // optical depth is a from-entry analytic model; apply the staged
        // captures at segment-entry power here (legacy semantics for this
        // branch) so the closure sees the depleted ray.
        for (int k = 0; k < hot_e_n_pend; ++k) {
          const int he_ch = hot_e_pend_ch[k];
          const std::size_t he_base =
              (static_cast<std::size_t>(tid) * hot_e_params.n_channels + he_ch) * 4;
          hot_e_capture[he_base + 0] = 1.0;
          hot_e_capture[he_base + 1] = hot_e_pend_r[k];
          hot_e_capture[he_base + 2] = hot_e_pend_mu[k];
          hot_e_capture[he_base + 3] = I;
          I *= hot_e_params.one_minus_eta[he_ch];
        }
        hot_e_n_pend = 0;
      }
      double I_tail = I;
      const double v_mag2 = vR * vR + vZ * vZ;
      const TailClosureStatus tail_status =
          try_tail_closure_1d(deposit_cache, c_old, radial_smooth_kappa, radial_n_hat_raw,
                              hydro_r_edges, n_hydro_cells, &hydro_hint,
                              allowed_supercritical_cell,
                              critical_adjacent_subcritical_cell, critical_adjacent_split_r,
                              test_kappa_cm_inv, nh_old, nh_old_raw, kappa_old, g_mag, 1.0,
                              v_mag2, r_old, I,
                              TailClosureMode::kCriticalCrossing, tail_closure_count,
                              tail_closure_absorbed_power, tail_slot,
                              error_flags, I_tail);
      if constexpr (kCbetRecord) {
        if (tail_status == TailClosureStatus::kClosed) {
          int tail_cell = -1;
          double tau_tail = 0.0;
          if (cbet_probe_tail_1d(c_old, radial_smooth_kappa, radial_n_hat_raw,
                                 hydro_r_edges, n_hydro_cells,
                                 allowed_supercritical_cell,
                                 critical_adjacent_subcritical_cell,
                                 critical_adjacent_split_r, test_kappa_cm_inv,
                                 nh_old, kappa_old, g_mag, r_old, &tail_cell,
                                 &tau_tail)) {
            cbet_cursor.add_terminal(tail_cell, tau_tail);
          } else {
            cbet_cursor.flush();
          }
        }
      }
      if (tail_status == TailClosureStatus::kClosed) {
        I = I_tail;
        if (traj_on && traj_stored < traj_max_steps) {
          const int idx = output_idx * traj_max_steps + traj_stored;
          traj_pos_R[idx] = R;
          traj_pos_Z[idx] = Z;
          traj_power[idx] = I;
          if (cbet_args.traj_rec_idx != nullptr) {
            cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
          }
          ++traj_stored;
        }
        if constexpr (kPhysExt) {
          if (phys_opt.crit_terminate_deposit != 0) {
            int dep_cell = critical_adjacent_subcritical_cell;
            if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
              dep_cell = allowed_supercritical_cell;
            }
            if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
              deposit_cache.accumulate(dep_cell, I);
              if (traj_on) {
                traj_step_count[output_idx] = traj_stored;
              }
              return;
            }
          }
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }

      hit_critical = true;
      record_critical_surface_hit(critical_surface_hit_count);
      const double nh_denom = nh_stop_raw - nh_old_raw;
      if (nh_denom > 0.0) {
        const double frac = clamp_unit_interval((nh_crit - nh_old_raw) / nh_denom);
        t_stop *= frac;
      } else {
        t_stop = 0.0;
      }
      if (!(t_stop > 0.0)) {
        if constexpr (kPhysExt) {
          if (phys_opt.crit_terminate_deposit != 0) {
            int dep_cell = critical_adjacent_subcritical_cell;
            if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
              dep_cell = allowed_supercritical_cell;
            }
            if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
              deposit_cache.accumulate(dep_cell, I);
              if (traj_on) {
                traj_step_count[output_idx] = traj_stored;
              }
              return;
            }
          }
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
      R_stop_path = old_R + t_stop * dR_path;
      R_stop = ::abs(R_stop_path);
      Z_stop = old_Z + t_stop * dZ_path;
      r_stop = radial_distance(R_stop, Z_stop);
      c_stop = locate_radial_interval(radial_node_r, n_radial_nodes, r_stop, c_stop.i);
      nh_stop = interpolate_radial_field(radial_n_hat, c_stop);
      nh_stop_raw = interpolate_radial_field(radial_n_hat_raw, c_stop);
      if (!::isfinite(nh_stop) || !::isfinite(nh_stop_raw)) {
        if (error_flags != nullptr) {
          atomicExch(&error_flags->invalid_cell, 1);
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }
    }

    double kappa_stop = 0.0;
    if (test_kappa_cm_inv > 0.0) {
      kappa_stop = test_kappa_cm_inv;
    } else {
      const double A_stop = interpolate_radial_field(radial_smooth_kappa, c_stop);
      kappa_stop = compute_kappa_from_smooth(A_stop, nh_stop, eps_n);
      if constexpr (kPhysExt) {
        if (phys_opt.langdon_model != 0 && kappa_stop > 0.0 &&
            radial_T_e != nullptr) {
          const double Te0 = interpolate_radial_field(radial_T_e, c_stop);
          const double I_vac = vacuum_map_intensity(
              phys_opt.langdon_I0_wcm2, phys_opt.langdon_w_cm, R,
              phys_opt.langdon_profile_kind, phys_opt.langdon_sg_two_m);
          kappa_stop *= compute_langdon_factor(phys_opt.langdon_model,
                                               phys_opt.langdon_zcoll, I_vac,
                                               lambda_cm, Te0,
                                               phys_opt.langdon_te_min_eV);
        }
      }
    }
    if (!::isfinite(kappa_stop) || kappa_stop < 0.0) {
      if (error_flags != nullptr) {
        atomicExch(&error_flags->invalid_cell, 1);
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

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
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
    if (tau_shell_out != nullptr) {
      // Attribute each marched segment to its entry radial interval c_old.i.
      tau_shell_out[static_cast<std::size_t>(tid) *
                        static_cast<std::size_t>(n_radial_nodes - 1) +
                    static_cast<std::size_t>(c_old.i)] += S;
    }

    double I_next = I;
    bool hot_e_split_done = false;
    if constexpr (kHotECapture) {
      if (hot_e_n_pend > 0) {
        // Phase C — event-split absorption: IB up to each staged crossing,
        // capture eta*P^cross there, then IB over the remainder. The optical
        // depth is split linearly in the path parameter (consistent with the
        // segment-level trapezoid S); each sub-deposit lands at its own
        // sub-midpoint. This replaces the legacy order (captures applied at
        // segment-entry power before the whole-segment IB), which biased the
        // hot-e source high and the pre-crossing IB low by O(S*frac).
        double P_run = I;
        double t_prev = 0.0;
        const double t_span = t_stop;
        for (int k = 0; k < hot_e_n_pend; ++k) {
          double t_k = hot_e_pend_t[k];
          if (t_k > t_span) {
            t_k = t_span;
          }
          if (t_k < t_prev) {
            t_k = t_prev;
          }
          const double S_k = (t_span > 0.0) ? S * ((t_k - t_prev) / t_span) : 0.0;
          double P_after = P_run;
          const double dP_k = absorbed_power_expm1(P_run, S_k, P_after);
          if (dP_k > 0.0) {
            const double sub_t = 0.5 * (t_prev + t_k);
            const double sub_R = ::abs(old_R + sub_t * dR_path);
            const double sub_Z = old_Z + sub_t * dZ_path;
            const double r_sub = radial_distance(sub_R, sub_Z);
            if (!outside_radial_profile(sub_R, sub_Z, radial_node_r, n_radial_nodes)) {
              accumulate_deposit_1d(deposit_cache, hydro_r_edges, n_hydro_cells,
                                    &hydro_hint, allowed_supercritical_cell,
                                    critical_adjacent_subcritical_cell,
                                    critical_adjacent_split_r, r_sub, dP_k);
            }
          }
          P_run = P_after;
          const int he_ch = hot_e_pend_ch[k];
          const std::size_t he_base =
              (static_cast<std::size_t>(tid) * hot_e_params.n_channels + he_ch) * 4;
          hot_e_capture[he_base + 0] = 1.0;
          hot_e_capture[he_base + 1] = hot_e_pend_r[k];
          hot_e_capture[he_base + 2] = hot_e_pend_mu[k];
          hot_e_capture[he_base + 3] = P_run;
          P_run *= hot_e_params.one_minus_eta[he_ch];
          t_prev = t_k;
        }
        const double S_rest = (t_span > 0.0) ? S * ((t_span - t_prev) / t_span) : S;
        double P_after = P_run;
        const double dP_rest = absorbed_power_expm1(P_run, S_rest, P_after);
        if (dP_rest > 0.0) {
          const double sub_t = 0.5 * (t_prev + t_span);
          const double sub_R = ::abs(old_R + sub_t * dR_path);
          const double sub_Z = old_Z + sub_t * dZ_path;
          const double r_sub = radial_distance(sub_R, sub_Z);
          if (!outside_radial_profile(sub_R, sub_Z, radial_node_r, n_radial_nodes)) {
            accumulate_deposit_1d(deposit_cache, hydro_r_edges, n_hydro_cells,
                                  &hydro_hint, allowed_supercritical_cell,
                                  critical_adjacent_subcritical_cell,
                                  critical_adjacent_split_r, r_sub, dP_rest);
          }
        }
        I_next = P_after;
        hot_e_n_pend = 0;
        hot_e_split_done = true;
        if (!::isfinite(I_next) || I_next < 0.0) {
          if (error_flags != nullptr) {
            atomicExch(&error_flags->nan_particle, 1);
          }
          if (unabsorbed_per_ray != nullptr) {
            unabsorbed_per_ray[tid] += I;
          } else {
            atomic_add_double(P_unabsorbed, I);
          }
          if (traj_on) {
            traj_step_count[output_idx] = traj_stored;
          }
          return;
        }
      }
    }
    if (!hot_e_split_done) {
      const double dP = absorbed_power_expm1(I, S, I_next);
      if (!::isfinite(dP) || !::isfinite(I_next) || dP < 0.0 || I_next < 0.0) {
        if (error_flags != nullptr) {
          atomicExch(&error_flags->nan_particle, 1);
        }
        if (unabsorbed_per_ray != nullptr) {
          unabsorbed_per_ray[tid] += I;
        } else {
          atomic_add_double(P_unabsorbed, I);
        }
        if (traj_on) {
          traj_step_count[output_idx] = traj_stored;
        }
        return;
      }

      if (dP > 0.0) {
        const double mid_t = 0.5 * t_stop;
        const double mid_R = ::abs(old_R + mid_t * dR_path);
        const double mid_Z = old_Z + mid_t * dZ_path;
        const double r_mid = radial_distance(mid_R, mid_Z);
        if (!outside_radial_profile(mid_R, mid_Z, radial_node_r, n_radial_nodes)) {
          accumulate_deposit_1d(deposit_cache, hydro_r_edges, n_hydro_cells,
                                &hydro_hint, allowed_supercritical_cell,
                                critical_adjacent_subcritical_cell,
                                critical_adjacent_split_r, r_mid, dP);
        }
      }
    }
    if constexpr (kCbetRecord) {
      if (ds_segment > 0.0) {
        const double mid_t_rec = 0.5 * t_stop;
        const double mid_R_rec = ::abs(old_R + mid_t_rec * dR_path);
        const double mid_Z_rec = old_Z + mid_t_rec * dZ_path;
        const double r_mid_rec = radial_distance(mid_R_rec, mid_Z_rec);
        if (!outside_radial_profile(mid_R_rec, mid_Z_rec, radial_node_r,
                                    n_radial_nodes)) {
          const int cell_rec = cbet_locate_deposit_cell_1d(
              hydro_r_edges, n_hydro_cells, allowed_supercritical_cell,
              critical_adjacent_subcritical_cell, critical_adjacent_split_r,
              r_mid_rec);
          const double mu_rec = (radial_distance(R_stop, Z_stop) - r_old) / ds_segment;
          cbet_cursor.add_segment(cell_rec, mu_rec, ds_segment, S, I);
        }
      }
    }

    I = I_next;
    if (traj_on && traj_stored < traj_max_steps) {
      const int idx = output_idx * traj_max_steps + traj_stored;
      traj_pos_R[idx] = R;
      traj_pos_Z[idx] = Z;
      traj_power[idx] = I;
      if (cbet_args.traj_rec_idx != nullptr) {
        cbet_args.traj_rec_idx[idx] = cbet_cursor.count;
      }
      ++traj_stored;
    }
    ds_prev = ds_cur * t_stop;
    carried_c = c_stop;
    carried_nh = nh_stop;
    carried_nh_raw = nh_stop_raw;
    carried_kappa = kappa_stop;
    if (exited_mesh || hit_critical ||
        (t_stop < 1.0 - kSegTol && entry_handoff_interval < 0)) {
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      if constexpr (kPhysExt) {
        if (hit_critical && phys_opt.crit_terminate_deposit != 0) {
          int dep_cell = critical_adjacent_subcritical_cell;
          if (dep_cell < 0 || dep_cell >= n_hydro_cells) {
            dep_cell = allowed_supercritical_cell;
          }
          if (dep_cell >= 0 && dep_cell < n_hydro_cells) {
            deposit_cache.accumulate(dep_cell, I);
            if (traj_on) {
              traj_step_count[output_idx] = traj_stored;
            }
            return;
          }
        }
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }

    if (I < intensity_cutoff * I0) {
      if constexpr (kCbetRecord) {
        cbet_cursor.flush();
      }
      if (unabsorbed_per_ray != nullptr) {
        unabsorbed_per_ray[tid] += I;
      } else {
        atomic_add_double(P_unabsorbed, I);
      }
      if (traj_on) {
        traj_step_count[output_idx] = traj_stored;
      }
      return;
    }
  }

  if constexpr (kCbetRecord) {
    cbet_cursor.flush();
  }
  if (unabsorbed_per_ray != nullptr) {
    unabsorbed_per_ray[tid] += I;
  } else {
    atomic_add_double(P_unabsorbed, I);
  }
  if (error_flags != nullptr) {
    atomicAdd(&error_flags->infinite_loop, 1);
  }
  if (traj_on) {
    traj_step_count[output_idx] = traj_stored;
  }
}


__device__ inline void reduce_per_ray_tallies_1d_body(
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
  // Fixed ascending-ray-order reduction makes the laser tallies
  // bitwise deterministic (identical seed => identical deposition). Thread c
  // owns exactly one output slot, so the trailing atomic adds have a single
  // writer per launch and beam launches accumulate in stream order.
  if (c < n_cells) {
    double acc = 0.0;
    for (int r = 0; r < n_rays; ++r) {
      acc += deposit_per_ray[static_cast<std::size_t>(r) *
                                 static_cast<std::size_t>(n_cells) +
                             static_cast<std::size_t>(c)];
    }
    if (acc != 0.0) {
      atomic_add_double(&deposit_1d[c], acc);
    }
    return;
  }
  if (c == n_cells && unabsorbed_per_ray != nullptr && P_unabsorbed != nullptr) {
    double acc = 0.0;
    for (int r = 0; r < n_rays; ++r) {
      acc += unabsorbed_per_ray[r];
    }
    if (acc != 0.0) {
      atomic_add_double(P_unabsorbed, acc);
    }
    return;
  }
  if (c == n_cells + 1 && tail_power_per_ray != nullptr &&
      tail_closure_absorbed_power != nullptr) {
    double acc = 0.0;
    for (int r = 0; r < n_rays; ++r) {
      acc += tail_power_per_ray[r];
    }
    if (acc != 0.0) {
      atomic_add_double(tail_closure_absorbed_power, acc);
    }
  }
  if (c == n_cells + 2 && ra_per_ray != nullptr && d_ra_total != nullptr) {
    double acc = 0.0;
    for (int r = 0; r < n_rays; ++r) {
      acc += ra_per_ray[r];
    }
    if (acc != 0.0) {
      atomic_add_double(d_ra_total, acc);
    }
  }
}


}  // namespace tenryu::laser::ray_trace_bodies
