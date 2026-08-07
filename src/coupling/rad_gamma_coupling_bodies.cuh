#pragma once

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

namespace tenryu::coupling::rad_gamma {

__device__ inline void gamma_r_43_work_update_kernel_body(
    const int c,
    double* __restrict__ rad_E,
    const double* __restrict__ W_r,
    const double* __restrict__ vol_before,
    const double* __restrict__ vol_after,
    const int n_cells,
    const int n_groups,
    double* __restrict__ floor_sum) {
  (void)n_cells;
  const double v0 = vol_before[c];
  const double v1 = vol_after[c];
  const double W = W_r[c];
  if (!(v0 > 0.0) || !(v1 > 0.0) || !isfinite(W)) {
    return;
  }

  const std::size_t base =
      static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
  double sum_old = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double e = rad_E[base + static_cast<std::size_t>(g)];
    if (isfinite(e) && e > 0.0) {
      sum_old += e;
    }
  }

  const double raw_total = (sum_old * v0 - W) / v1;
  const double new_total = fmax(raw_total, 0.0);
  if (raw_total < 0.0 && floor_sum != nullptr) {
    atomicAdd(floor_sum, -raw_total * v1);
  }

  if (sum_old > 0.0) {
    const double scale = new_total / sum_old;
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t k = base + static_cast<std::size_t>(g);
      const double e = rad_E[k];
      rad_E[k] = (isfinite(e) && e > 0.0) ? e * scale : 0.0;
    }
  } else {
    const double split = new_total / static_cast<double>(n_groups);
    for (int g = 0; g < n_groups; ++g) {
      rad_E[base + static_cast<std::size_t>(g)] = split;
    }
  }
}

__device__ inline void radiation_pressure_field_kernel_body(
    const int c,
    double* __restrict__ p_r,
    const double* __restrict__ rad_E,
    const int n_cells,
    const int n_groups) {
  (void)n_cells;
  double sum = 0.0;
  const std::size_t base =
      static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
  for (int g = 0; g < n_groups; ++g) {
    const double e = rad_E[base + static_cast<std::size_t>(g)];
    if (isfinite(e) && e > 0.0) {
      sum += e;
    }
  }
  p_r[c] = sum / 3.0;
}

}  // namespace tenryu::coupling::rad_gamma
