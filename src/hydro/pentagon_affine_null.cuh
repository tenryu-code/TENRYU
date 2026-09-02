#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "hydro/pentagon_geometry.cuh"

#if defined(__CUDACC__)
#define TENRYU_PENTAGON_AFFINE_NULL_HD __host__ __device__
#else
#define TENRYU_PENTAGON_AFFINE_NULL_HD
#endif

namespace tenryu::hydro {
namespace pentagon_affine_null_detail {

TENRYU_PENTAGON_AFFINE_NULL_HD inline bool finite_double(
    const double value) {
  return value == value && value != INFINITY && value != -INFINITY;
}

[[noreturn]] TENRYU_PENTAGON_AFFINE_NULL_HD inline void fit_degenerate() {
#if defined(__CUDA_ARCH__)
  __trap();
#else
  std::fputs("pentagon affine fit degenerate\n", stderr);
  std::abort();
#endif
}

TENRYU_PENTAGON_AFFINE_NULL_HD inline void affine_row(
    const int row,
    const double rh[5],
    const double zh[5],
    double values[6]) {
  for (int column = 0; column < 6; ++column) {
    values[column] = 0.0;
  }
  const int corner = row / 2;
  if ((row & 1) == 0) {
    values[0] = 1.0;
    values[2] = rh[corner];
    values[3] = zh[corner];
  } else {
    values[1] = 1.0;
    values[4] = rh[corner];
    values[5] = zh[corner];
  }
}

}  // namespace pentagon_affine_null_detail

TENRYU_PENTAGON_AFFINE_NULL_HD inline bool pentagon_affine_null_force(
    const PentagonPoint x[5],
    const double corner_mass[5],
    const double velocity_r[5],
    const double velocity_z[5],
    const double kappa,
    const double cs,
    double force_r[5],
    double force_z[5]) {
  const PentagonPoint x_star = pentagon_center(x);
  const double h_c = pentagon_acoustic_length(x);
  double rh[5];
  double zh[5];
  for (int k = 0; k < 5; ++k) {
    rh[k] = (x[k].r - x_star.r) / h_c;
    zh[k] = (x[k].z - x_star.z) / h_c;
  }

  double gram[6][6] = {};
  double rhs[6] = {};
  for (int row = 0; row < 10; ++row) {
    const int corner = row / 2;
    const double mass = corner_mass[corner];
    const double velocity =
        ((row & 1) == 0) ? velocity_r[corner] : velocity_z[corner];
    double a[6];
    pentagon_affine_null_detail::affine_row(row, rh, zh, a);
    for (int i = 0; i < 6; ++i) {
      rhs[i] += a[i] * mass * velocity;
      for (int j = 0; j < 6; ++j) {
        gram[i][j] += a[i] * mass * a[j];
      }
    }
  }

  double lower[6][6] = {};
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      double value = gram[i][j];
      for (int q = 0; q < j; ++q) {
        value -= lower[i][q] * lower[j][q];
      }
      if (i == j) {
        if (!(value > 0.0) ||
            !pentagon_affine_null_detail::finite_double(value)) {
          for (int k = 0; k < 5; ++k) {
            force_r[k] = 0.0;
            force_z[k] = 0.0;
          }
          return false;
        }
        lower[i][j] = sqrt(value);
      } else {
        lower[i][j] = value / lower[j][j];
      }
    }
  }

  double forward[6] = {};
  for (int i = 0; i < 6; ++i) {
    double value = rhs[i];
    for (int j = 0; j < i; ++j) {
      value -= lower[i][j] * forward[j];
    }
    forward[i] = value / lower[i][i];
  }

  double beta[6] = {};
  for (int i = 5; i >= 0; --i) {
    double value = forward[i];
    for (int j = i + 1; j < 6; ++j) {
      value -= lower[j][i] * beta[j];
    }
    beta[i] = value / lower[i][i];
  }

  const double damping_rate = kappa * (cs / h_c);
  for (int k = 0; k < 5; ++k) {
    double a_r[6];
    double a_z[6];
    pentagon_affine_null_detail::affine_row(2 * k, rh, zh, a_r);
    pentagon_affine_null_detail::affine_row(2 * k + 1, rh, zh, a_z);
    double affine_r = 0.0;
    double affine_z = 0.0;
    for (int column = 0; column < 6; ++column) {
      affine_r += a_r[column] * beta[column];
      affine_z += a_z[column] * beta[column];
    }
    force_r[k] =
        -damping_rate * corner_mass[k] * (velocity_r[k] - affine_r);
    force_z[k] =
        -damping_rate * corner_mass[k] * (velocity_z[k] - affine_z);
  }
  return true;
}

}  // namespace tenryu::hydro

#undef TENRYU_PENTAGON_AFFINE_NULL_HD
