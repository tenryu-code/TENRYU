#pragma once

#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#ifdef __CUDACC__
#define TENRYU_HOST_DEVICE __host__ __device__
#else
#define TENRYU_HOST_DEVICE
#endif
#endif

namespace tenryu::radiation {

struct LCWeights {
  double E;
  double A;
  double W0;
  double W1;
  double B0;
  double B1;
};

TENRYU_HOST_DEVICE inline LCWeights compute_lc_weights(const double tau) {
  LCWeights w{};

  if (tau < 1.0e-3) {
    const double t2 = tau * tau;
    const double t3 = t2 * tau;
    const double t4 = t2 * t2;
    const double t5 = t4 * tau;
    const double t6 = t3 * t3;

    w.E = 1.0 - tau + 0.5 * t2 - t3 / 6.0 + t4 / 24.0 - t5 / 120.0 +
          t6 / 720.0;
    w.A = 1.0 - 0.5 * tau + t2 / 6.0 - t3 / 24.0 + t4 / 120.0 -
          t5 / 720.0 + t6 / 5040.0;
    w.W0 = 0.5 * tau - t2 / 3.0 + t3 / 8.0 - t4 / 30.0 + t5 / 144.0 -
           t6 / 840.0;
    w.W1 = 0.5 * tau - t2 / 6.0 + t3 / 24.0 - t4 / 120.0 + t5 / 720.0 -
           t6 / 5040.0;
    w.B0 = tau / 3.0 - t2 / 8.0 + t3 / 30.0 - t4 / 144.0 + t5 / 840.0 -
           t6 / 5760.0;
    w.B1 = tau / 6.0 - t2 / 24.0 + t3 / 120.0 - t4 / 720.0 +
           t5 / 5040.0 - t6 / 40320.0;
    return w;
  }

  if (tau >= 745.0) {
    const double inv_tau = 1.0 / tau;
    const double inv_tau2 = inv_tau * inv_tau;

    w.E = 0.0;
    w.A = inv_tau;
    w.W0 = inv_tau;
    w.W1 = 1.0 - inv_tau;
    w.B0 = 0.5 - inv_tau2;
    w.B1 = 1.0 - w.A - w.B0;
    return w;
  }

  w.E = exp(-tau);
  w.A = (1.0 - w.E) / tau;
  w.W0 = w.A - w.E;
  w.W1 = 1.0 - w.E - w.W0;
  w.B0 = 0.5 - w.W0 / tau;
  w.B1 = 1.0 - w.A - w.B0;
  return w;
}

TENRYU_HOST_DEVICE inline void walters_limit_endpoints(const double q_prev,
                                                       const double q_center,
                                                       const double q_next,
                                                       double* q0,
                                                       double* q1) {
  const double dL = q_center - q_prev;
  const double dR = q_next - q_center;
  double slope = 0.0;

  if (dL * dR > 0.0) {
    const double centered = 0.5 * (q_next - q_prev);
    const double limited_mag =
        fmin(2.0 * fabs(centered), fmin(2.0 * fabs(dL), 2.0 * fabs(dR)));
    slope = (centered < 0.0 ? -limited_mag : limited_mag);
  }

  *q0 = q_center - 0.5 * slope;
  *q1 = q_center + 0.5 * slope;

  if (*q0 < 0.0 || *q1 < 0.0) {
    const double theta =
        (slope == 0.0)
            ? 1.0
            : fmin(1.0, q_center / fmax(0.5 * fabs(slope), 1.0e-300));
    slope *= theta;
    *q0 = q_center - 0.5 * slope;
    *q1 = q_center + 0.5 * slope;
  }

  if (*q0 < 0.0 || *q1 < 0.0) {
    const double q_clamped = fmax(q_center, 0.0);
    *q0 = q_clamped;
    *q1 = q_clamped;
  }
}

}  // namespace tenryu::radiation
