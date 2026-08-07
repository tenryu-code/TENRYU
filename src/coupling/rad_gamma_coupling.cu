// W-K gamma_r = 4/3 radiation-compression kernels. See the header and
// docs/design/wk_gamma_r_coupling_design.md.

#include "coupling/rad_gamma_coupling.cuh"
#include "coupling/rad_gamma_coupling_bodies.cuh"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"

namespace tenryu::coupling::rad_gamma {

namespace {

void rg_cuda_check(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " +
                             cudaGetErrorString(err));
  }
}

void rg_sync(const char* msg) {
  rg_cuda_check(cudaGetLastError(), msg);
  rg_cuda_check(cudaDeviceSynchronize(), msg);
}

__global__ void gamma_r_43_volume_update_kernel(
    double* __restrict__ rad_E,
    const double* __restrict__ vol_before,
    const double* __restrict__ vol_after,
    const std::size_t n_cell_groups,
    const std::size_t n_groups) {
  const std::size_t k =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k >= n_cell_groups) {
    return;
  }
  const std::size_t c = k / n_groups;
  const double v0 = vol_before[c];
  const double v1 = vol_after[c];
  if (!(v0 > 0.0) || !(v1 > 0.0)) {
    return;
  }
  const double e = rad_E[k];
  if (!isfinite(e)) {
    return;
  }
  // (v0/v1)^{4/3} = ratio * cbrt(ratio): exact E V^{4/3} adiabat.
  const double ratio = v0 / v1;
  rad_E[k] = e * (ratio * cbrt(ratio));
}

__global__ void gamma_r_43_work_update_kernel(
    double* __restrict__ rad_E,
    const double* __restrict__ W_r,
    const double* __restrict__ vol_before,
    const double* __restrict__ vol_after,
    const int c_begin,
    const int c_end,
    const int n_groups,
    double* __restrict__ floor_sum) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  // The body does not use n_cells; c_end bounds this launch window.
  gamma_r_43_work_update_kernel_body(c, rad_E, W_r, vol_before, vol_after,
                                     c_end, n_groups, floor_sum);
}

__global__ void radiation_pressure_field_kernel(
    double* __restrict__ p_r,
    const double* __restrict__ rad_E,
    const int n_cells,
    const int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  radiation_pressure_field_kernel_body(c, p_r, rad_E, n_cells, n_groups);
}

}  // namespace

bool gamma_r_43_enabled_from_env() {
  const char* s = std::getenv("TENRYU_RAD_HYDRO_COUPLING");
  return s != nullptr && std::strcmp(s, "gamma_r_43") == 0;
}

void apply_gamma_r_43_volume_update(double* rad_E,
                                    const double* vol_before,
                                    const double* vol_after,
                                    const int n_cells,
                                    const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const std::size_t blocks = (n_cell_groups + 255U) / 256U;
  gamma_r_43_volume_update_kernel<<<static_cast<unsigned>(blocks), 256>>>(
      rad_E, vol_before, vol_after, n_cell_groups,
      static_cast<std::size_t>(n_groups));
  rg_sync("rad_gamma: volume update kernel failed");
}

double apply_gamma_r_43_work_update(double* rad_E,
                                    const double* W_r,
                                    const double* vol_before,
                                    const double* vol_after,
                                    const int c_begin,
                                    const int c_end,
                                    const int n_groups) {
  const int n_window = c_end - c_begin;
  if (n_window <= 0 || n_groups <= 0) {
    return 0.0;
  }
  double* d_floor_sum = nullptr;
  d_floor_sum = static_cast<double*>(core::device_scratch_acquire(
      "rad_gamma_coupling:apply_gamma_r_43_work_update:d_floor_sum",
      sizeof(double)));
  rg_cuda_check(cudaMemset(d_floor_sum, 0, sizeof(double)),
                "rad_gamma: work update floor cudaMemset failed");
  const int blocks = (n_window + 255) / 256;
  gamma_r_43_work_update_kernel<<<blocks, 256>>>(
      rad_E, W_r, vol_before, vol_after, c_begin, c_end, n_groups,
      d_floor_sum);
  rg_sync("rad_gamma: work update kernel failed");

  double floor_sum = 0.0;
  rg_cuda_check(cudaMemcpy(&floor_sum, d_floor_sum, sizeof(double),
                           cudaMemcpyDeviceToHost),
                "rad_gamma: work update floor copy failed");
  return floor_sum;
}

void compute_radiation_pressure_field(double* p_r,
                                      const double* rad_E,
                                      const int n_cells,
                                      const int n_groups) {
  if (n_cells <= 0 || n_groups <= 0) {
    return;
  }
  const int blocks = (n_cells + 255) / 256;
  radiation_pressure_field_kernel<<<blocks, 256>>>(p_r, rad_E, n_cells,
                                                   n_groups);
  rg_sync("rad_gamma: pressure field kernel failed");
}

}  // namespace tenryu::coupling::rad_gamma
