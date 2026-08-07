#pragma once

#include <cfloat>
#include <cstddef>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/namelist/frozen_table.hpp"

namespace tenryu::core::namelist {

struct FrozenTable1DDeviceView {
  const double* x = nullptr;
  const double* y = nullptr;
  int n = 0;

  __device__ double eval(const double xi) const {
    /*
    Host FrozenTable1D::eval code mirrored below:

    double FrozenTable1D::eval(const double xi) const {
      TENRYU_ASSERT(n_points == static_cast<int>(x.size()),
                    "FrozenTable1D.x size must match n_points");
      TENRYU_ASSERT(n_points == static_cast<int>(y.size()),
                    "FrozenTable1D.y size must match n_points");

      if (n_points <= 0) {
        static std::atomic<bool> warned_empty_table{false};
        bool expected = false;
        if (warned_empty_table.compare_exchange_strong(expected, true)) {
          tenryu::core::log_warning(
              "FrozenTable1D::eval called with empty table; returning 0.0");
        }
        return 0.0;
      }
      if (n_points == 1) {
        if (zero_outside && (xi < x_min || xi > x_max)) {
          return 0.0;
        }
        return y.front();
      }

      if (xi < x_min) {
        return zero_outside ? 0.0 : y.front();
      }
      if (xi > x_max) {
        return zero_outside ? 0.0 : y.back();
      }

      auto it = std::upper_bound(x.begin(), x.end(), xi);
      if (it == x.begin()) {
        return y.front();
      }
      if (it == x.end()) {
        return y.back();
      }

      const int hi = static_cast<int>(std::distance(x.begin(), it));
      const int lo = hi - 1;
      const double x0 = x[static_cast<std::size_t>(lo)];
      const double x1 = x[static_cast<std::size_t>(hi)];
      const double y0 = y[static_cast<std::size_t>(lo)];
      const double y1 = y[static_cast<std::size_t>(hi)];

      const double dx = x1 - x0;
      if (dx <= std::numeric_limits<double>::epsilon()) {
        return y0;
      }
      const double s = (xi - x0) / dx;
      return y0 + s * (y1 - y0);
    }

    Persistent laser uploads only zero_outside=true waveform tables; the minimal
    device view intentionally stores only {x, y, n}.
    */
    if (n <= 0 || x == nullptr || y == nullptr) {
      return 0.0;
    }
    const double x_min = x[0];
    const double x_max = x[n - 1];
    if (n == 1) {
      if (xi < x_min || xi > x_max) {
        return 0.0;
      }
      return y[0];
    }

    if (xi < x_min) {
      return 0.0;
    }
    if (xi > x_max) {
      return 0.0;
    }

    int first = 0;
    int count = n;
    while (count > 0) {
      const int step = count / 2;
      const int mid = first + step;
      if (!(xi < x[mid])) {
        first = mid + 1;
        count -= step + 1;
      } else {
        count = step;
      }
    }

    if (first == 0) {
      return y[0];
    }
    if (first == n) {
      return y[n - 1];
    }

    const int hi = first;
    const int lo = hi - 1;
    const double x0 = x[lo];
    const double x1 = x[hi];
    const double y0 = y[lo];
    const double y1 = y[hi];

    const double dx = x1 - x0;
    if (dx <= DBL_EPSILON) {
      return y0;
    }
    const double s = (xi - x0) / dx;
    return y0 + s * (y1 - y0);
  }
};

inline FrozenTable1DDeviceView upload_frozen_table_1d_device_view(
    const FrozenTable1D& table) {
  TENRYU_ASSERT(table.n_points == static_cast<int>(table.x.size()),
                "upload_frozen_table_1d_device_view x size mismatch");
  TENRYU_ASSERT(table.n_points == static_cast<int>(table.y.size()),
                "upload_frozen_table_1d_device_view y size mismatch");
  TENRYU_ASSERT(table.zero_outside,
                "upload_frozen_table_1d_device_view requires zero_outside=true");
  FrozenTable1DDeviceView view{};
  view.n = table.n_points;
  if (view.n <= 0) {
    return view;
  }
  TENRYU_ASSERT(table.x.front() == table.x_min,
                "upload_frozen_table_1d_device_view x_min must equal x.front()");
  TENRYU_ASSERT(table.x.back() == table.x_max,
                "upload_frozen_table_1d_device_view x_max must equal x.back()");
  const std::size_t bytes =
      static_cast<std::size_t>(view.n) * sizeof(double);
  double* x_device = static_cast<double*>(
      tenryu::core::device_scratch_acquire("persistent_laser:waveform_x",
                                           bytes));
  double* y_device = static_cast<double*>(
      tenryu::core::device_scratch_acquire("persistent_laser:waveform_y",
                                           bytes));
  const cudaError_t err_x =
      cudaMemcpy(x_device, table.x.data(), bytes, cudaMemcpyHostToDevice);
  TENRYU_ASSERT(err_x == cudaSuccess,
                "upload_frozen_table_1d_device_view waveform_x copy failed");
  const cudaError_t err_y =
      cudaMemcpy(y_device, table.y.data(), bytes, cudaMemcpyHostToDevice);
  TENRYU_ASSERT(err_y == cudaSuccess,
                "upload_frozen_table_1d_device_view waveform_y copy failed");
  view.x = x_device;
  view.y = y_device;
  return view;
}

}  // namespace tenryu::core::namelist
