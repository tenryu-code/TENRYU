#pragma once

#include <cmath>

#include <cuda_runtime.h>

namespace tenryu::coupling {

namespace detail {
__host__ __device__ inline bool dt_ctrl_finite(const double v) {
#if defined(__CUDA_ARCH__)
  return ::isfinite(v);
#else
  return std::isfinite(v);
#endif
}
}  // namespace detail

enum DtLimiterId : int {
  kDtLimiterDriverRetry = 0,
  kDtLimiterHydro = 1,
  kDtLimiterConduction = 2,
  kDtLimiterBraginskii = 3,
  kDtLimiterRadiation = 4,
  kDtLimiterGrowth = 5,
  kDtLimiterOutput = 6,
  kDtLimiterTEnd = 7,
  kDtLimiterMax = 8,
  kDtLimiterInit = 9,
  kDtLimiterUnknown = 10,
  kDtLimiterHotElectron = 11
};

__host__ __device__ inline double dt_hydro_post_1d(
    const double min_local,
    const double min_local_post_shock,
    const double cfl_hydro,
    const int have_active) {
  if (have_active == 0) {
    return INFINITY;
  }
  if (!detail::dt_ctrl_finite(min_local)) {
    return INFINITY;
  }
  const double acoustic_dt = cfl_hydro * min_local;
  double dt = acoustic_dt;
  if (detail::dt_ctrl_finite(min_local_post_shock)) {
    dt = (min_local_post_shock < dt) ? min_local_post_shock : dt;
  }
  return dt;
}

__host__ __device__ inline double dt_cond_post(const double min_ratio,
                                               const double cfl_cond,
                                               const int sts_max_stages) {
  if (detail::dt_ctrl_finite(min_ratio) && min_ratio > 0.0) {
    const double dt_exp = cfl_cond * min_ratio;
    if (!detail::dt_ctrl_finite(dt_exp) || !(dt_exp > 0.0)) {
      return INFINITY;
    }
    if (sts_max_stages <= 0) {
      return INFINITY;
    }
    const int smax = (sts_max_stages < 1) ? 1 : sts_max_stages;
    const double factor = 0.5 * smax * (smax + 1);
    return factor * dt_exp;
  }
  return INFINITY;
}

__host__ __device__ inline double dt_rad_post(const double dt_rad_reduced,
                                              const double max_sigma_P,
                                              const int use_nlte_table,
                                              const double f_min,
                                              const double alpha,
                                              const double c_light) {
  double dt_rad = dt_rad_reduced;
  if (use_nlte_table && max_sigma_P > 0.0 &&
      (!detail::dt_ctrl_finite(dt_rad) || dt_rad <= 0.0)) {
    const double safety = (1.0 - f_min) / (f_min * alpha);
    if (safety > 0.0) {
      const double bound = safety / (c_light * max_sigma_P);
      if (detail::dt_ctrl_finite(bound) && bound > 0.0) {
        dt_rad = (bound < dt_rad) ? bound : dt_rad;
      }
    }
  }
  return dt_rad;
}

struct DtLadderIn {
  double dt_hydro;
  double dt_cond;
  double dt_visc;
  double dt_rad;
  double dt_prev;
  double growth_factor;
  double dt_max;
  double dt_output;
  double dt_remaining;
};

struct DtLadderOut {
  double dt_chosen;
  int limiter;
};

__host__ __device__ inline DtLadderOut dt_ladder_eval(
    const DtLadderIn& in) {
  double dt_new = in.dt_max;
  dt_new = (in.dt_hydro < dt_new) ? in.dt_hydro : dt_new;
  dt_new = (in.dt_cond < dt_new) ? in.dt_cond : dt_new;
  dt_new = (in.dt_visc < dt_new) ? in.dt_visc : dt_new;
  dt_new = (in.dt_rad < dt_new) ? in.dt_rad : dt_new;
  const double dt_growth = in.growth_factor * in.dt_prev;
  dt_new = (dt_growth < dt_new) ? dt_growth : dt_new;
  dt_new = (in.dt_max < dt_new) ? in.dt_max : dt_new;
  dt_new = (in.dt_output < dt_new) ? in.dt_output : dt_new;
  dt_new = (in.dt_remaining < dt_new) ? in.dt_remaining : dt_new;

  int limiter = kDtLimiterUnknown;
  if (dt_new == in.dt_hydro) {
    limiter = kDtLimiterHydro;
  } else if (dt_new == in.dt_cond) {
    limiter = kDtLimiterConduction;
  } else if (dt_new == in.dt_visc) {
    limiter = kDtLimiterBraginskii;
  } else if (dt_new == in.dt_rad) {
    limiter = kDtLimiterRadiation;
  } else if (dt_new == dt_growth) {
    limiter = kDtLimiterGrowth;
  } else if (dt_new == in.dt_output) {
    limiter = kDtLimiterOutput;
  } else if (dt_new == in.dt_remaining) {
    limiter = kDtLimiterTEnd;
  } else if (dt_new == in.dt_max) {
    limiter = kDtLimiterMax;
  }
  return {dt_new, limiter};
}

void run_dt_controller_check(const DtLadderIn& in, double out[2]);

}  // namespace tenryu::coupling
