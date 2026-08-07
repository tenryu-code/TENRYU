#include "coupling/driver_reclose.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "materials/eos_table_reclose.cuh"

namespace tenryu::coupling {
namespace {

constexpr int kRecloseBlockSize = 256;

void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void tabular_eos_reclose_kernel(
    const materials::DeviceEOSTableView electron_table,
    const materials::DeviceEOSTableView ion_table,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::size_t cell_is_void_size,
    const int n_cells,
    const double* __restrict__ rho,
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ Te,
    double* __restrict__ Ti,
    double* __restrict__ Pe,
    double* __restrict__ Pi,
    double* __restrict__ cv_e,
    double* __restrict__ cv_i,
    const double te_floor,
    const double ti_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  const std::size_t i_us = static_cast<std::size_t>(i);
  if (i_us < cell_is_void_size && cell_is_void[i_us] != 0U) {
    return;
  }

  const materials::RecloseThermo electron = materials::reclose_thermo_from_energy(
      electron_table, rho[i], ee[i], te_floor);
  Te[i] = electron.T;
  ee[i] = electron.energy;
  Pe[i] = electron.pressure;
  if (cv_e != nullptr) {
    cv_e[i] = electron.cv;
  }

  const materials::RecloseThermo ion =
      materials::reclose_thermo_from_energy(ion_table, rho[i], ei[i], ti_floor);
  Ti[i] = ion.T;
  ei[i] = ion.energy;
  Pi[i] = ion.pressure;
  if (cv_i != nullptr) {
    cv_i[i] = ion.cv;
  }
}

void refresh_cell_is_void_cache(
    DriverRecloseContext& context,
    const std::vector<std::uint8_t>& cell_is_void) {
  if (context.cached_cell_is_void == cell_is_void) {
    return;
  }
  if (cell_is_void.size() > context.cell_is_void_capacity) {
    if (context.d_cell_is_void != nullptr) {
      cuda_check(cudaFree(context.d_cell_is_void),
                 "driver reclose cell_is_void cudaFree failed");
    }
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&context.d_cell_is_void),
                          cell_is_void.size() * sizeof(std::uint8_t)),
               "driver reclose cell_is_void cudaMalloc failed");
    context.cell_is_void_capacity = cell_is_void.size();
  }
  if (!cell_is_void.empty()) {
    cuda_check(cudaMemcpy(context.d_cell_is_void,
                          cell_is_void.data(),
                          cell_is_void.size() * sizeof(std::uint8_t),
                          cudaMemcpyHostToDevice),
               "driver reclose cell_is_void H2D failed");
  }
  context.cached_cell_is_void = cell_is_void;
}

}  // namespace

DriverRecloseContext::~DriverRecloseContext() {
  if (d_cell_is_void != nullptr) {
    static_cast<void>(cudaFree(d_cell_is_void));
  }
}

void launch_tabular_eos_reclose(
    DriverRecloseContext& context,
    const materials::DeviceEOSTableView electron_table,
    const materials::DeviceEOSTableView ion_table,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::size_t n_cells,
    const double* rho,
    double* ee,
    double* ei,
    double* Te,
    double* Ti,
    double* Pe,
    double* Pi,
    double* cv_e,
    double* cv_i,
    const double te_floor,
    const double ti_floor) {
  TENRYU_ASSERT(n_cells <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "driver reclose cell count exceeds kernel limit");
  if (n_cells == 0) {
    return;
  }

  refresh_cell_is_void_cache(context, cell_is_void);
  const int n = static_cast<int>(n_cells);
  const int blocks = (n + kRecloseBlockSize - 1) / kRecloseBlockSize;
  tabular_eos_reclose_kernel<<<blocks, kRecloseBlockSize>>>(
      electron_table,
      ion_table,
      context.d_cell_is_void,
      cell_is_void.size(),
      n,
      rho,
      ee,
      ei,
      Te,
      Ti,
      Pe,
      Pi,
      cv_e,
      cv_i,
      te_floor,
      ti_floor);
  cuda_check(cudaGetLastError(), "driver tabular EOS reclose kernel launch failed");
}

}  // namespace tenryu::coupling
