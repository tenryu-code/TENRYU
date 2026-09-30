#pragma once

#include <atomic>
#include <cmath>
#include <cstdint>

#include "core/constants.hpp"

#ifndef TENRYU_HOST_DEVICE
#ifdef __CUDACC__
#define TENRYU_HOST_DEVICE __host__ __device__
#else
#define TENRYU_HOST_DEVICE
#endif
#endif

namespace tenryu::radiation {

// NOTE: No DDMCEvent::Emission channel exists because DDMC particles are
// created by the source module (source.cu), not by the DDMC transport kernel.
// Emission energy accounting is handled in IMCSource::emit_thermal().
enum class DDMCEvent : std::uint8_t {
  Absorb = 0,
  Scatter = 1,
  LeakLeft = 2,
  LeakRight = 3,
  LeakBoundary = 4,
  Census = 5,
};

inline std::atomic<std::uint64_t>& ddmc_event_cdf_mismatch_counter() {
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}

inline double sample_ddmc_event_time(const double sigma_tot,
                                     const double xi) {
  // Keep xi away from exactly {0,1} to avoid log singularities while keeping
  // bias negligible relative to Monte Carlo variance.
  constexpr double eps = 1.0e-16;
  if (sigma_tot <= 0.0) {
    return 1.0e300;
  }
  const double xi_safe = fmin(fmax(xi, eps), 1.0 - eps);
  return -log(xi_safe) / (core::constants::c_light * sigma_tot);
}

TENRYU_HOST_DEVICE inline int sample_group_from_cdf(const double* cdf,
                                                    const int n_groups,
                                                    const double xi) {
  if (cdf == nullptr || n_groups <= 1) {
    return 0;
  }
  const double xi_safe = fmin(fmax(xi, 0.0), 1.0 - 1.0e-12);
  int lo = 0;
  int hi = n_groups - 1;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    const double cdf_mid = fmin(fmax(cdf[mid], 0.0), 1.0);
    if (cdf_mid >= xi_safe) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

TENRYU_HOST_DEVICE inline DDMCEvent select_ddmc_event(
    const double sigma_a_eff,
    const double sigma_s_eff,
    const double sigma_leak_left_out,
    const double sigma_leak_right_out,
    const double sigma_leak_bnd,
    const double sigma_tot,
    const double xi) {
  if (sigma_tot <= 0.0) {
    return DDMCEvent::Census;
  }

  const double sigma_a_pos = fmax(sigma_a_eff, 0.0);
  const double sigma_s_pos = fmax(sigma_s_eff, 0.0);
  const double sigma_leak_left_pos = fmax(sigma_leak_left_out, 0.0);
  const double sigma_leak_right_pos = fmax(sigma_leak_right_out, 0.0);
  const double sigma_leak_bnd_pos = fmax(sigma_leak_bnd, 0.0);
  const double sum_channels =
      sigma_a_pos + sigma_s_pos + sigma_leak_left_pos + sigma_leak_right_pos +
      sigma_leak_bnd_pos;
#ifndef __CUDA_ARCH__
  constexpr double kCdfNormTol = 1.0e-12;
  const double sigma_scale = fmax(fabs(sigma_tot), 1.0e-30);
  if (fabs(sum_channels - sigma_tot) > kCdfNormTol * sigma_scale) {
    ddmc_event_cdf_mismatch_counter().fetch_add(1, std::memory_order_relaxed);
  }
#endif

  const double xi_safe = fmin(fmax(xi, 0.0), 1.0 - 1.0e-12);
  const double threshold = xi_safe * sigma_tot;
  double cdf = sigma_a_pos;
  if (threshold < cdf) {
    return DDMCEvent::Absorb;
  }

  cdf += sigma_s_pos;
  if (threshold < cdf) {
    return DDMCEvent::Scatter;
  }

  cdf += sigma_leak_left_pos;
  if (threshold < cdf) {
    return DDMCEvent::LeakLeft;
  }

  cdf += sigma_leak_right_pos;
  if (threshold < cdf) {
    return DDMCEvent::LeakRight;
  }

  cdf += sigma_leak_bnd_pos;
  if (sigma_leak_bnd_pos > 0.0 && threshold < cdf) {
    return DDMCEvent::LeakBoundary;
  }

  // Fallback keeps transport progressing even if CDF and sigma_tot are slightly inconsistent.
  return DDMCEvent::Absorb;
}

}  // namespace tenryu::radiation
