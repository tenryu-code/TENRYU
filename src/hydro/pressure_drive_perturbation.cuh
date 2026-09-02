#pragma once

#include <cmath>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define TENRYU_PRESSURE_DRIVE_PERTURBATION_HD __host__ __device__
#else
#define TENRYU_PRESSURE_DRIVE_PERTURBATION_HD
#endif

namespace tenryu::hydro {

struct PressureDrivePerturbationParams {
  static constexpr int kMaxModes = 24;
  static constexpr int kMaxSpots = 4;
  int n_modes = 0;
  int mode_l[kMaxModes] = {};
  double mode_a[kMaxModes] = {};
  int n_spots = 0;
  double spot_theta0[kMaxSpots] = {};
  double spot_sigma[kMaxSpots] = {};
  double spot_amp[kMaxSpots] = {};
};

TENRYU_PRESSURE_DRIVE_PERTURBATION_HD inline double
pressure_drive_perturbation_g(const PressureDrivePerturbationParams& p,
                              const double r,
                              const double z) {
  const double s = sqrt(r * r + z * z);
  if (!(s > 0.0)) {
    return 1.0;
  }

  double mu = z / s;
  if (mu < -1.0) {
    mu = -1.0;
  } else if (mu > 1.0) {
    mu = 1.0;
  }

  double sum = 0.0;
  for (int k = 0; k < p.n_modes; ++k) {
    const int l = p.mode_l[k];
    double p_l = 1.0;
    if (l == 1) {
      p_l = mu;
    } else if (l > 1) {
      double p_nm2 = 1.0;
      double p_nm1 = mu;
      for (int n = 2; n <= l; ++n) {
        const double p_n = ((2 * n - 1) * mu * p_nm1 -
                            (n - 1) * p_nm2) /
                           n;
        p_nm2 = p_nm1;
        p_nm1 = p_n;
      }
      p_l = p_nm1;
    }
    sum += p.mode_a[k] * p_l;
  }

  const double theta = acos(mu);
  for (int m = 0; m < p.n_spots; ++m) {
    const double d = theta - p.spot_theta0[m];
    const double sigma = p.spot_sigma[m];
    sum += p.spot_amp[m] * exp(-0.5 * d * d / (sigma * sigma));
  }

  return 1.0 + sum;
}

}  // namespace tenryu::hydro

#undef TENRYU_PRESSURE_DRIVE_PERTURBATION_HD
