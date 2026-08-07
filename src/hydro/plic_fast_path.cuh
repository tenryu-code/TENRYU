#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace tenryu::hydro::plic {

__host__ __device__ inline bool is_multi_material_cell(
    const double* volfrac,
    const std::size_t cell,
    const std::size_t n_mat,
    const double thresh_min,
    const double thresh_max) {
  for (std::size_t m = 0; m < n_mat; ++m) {
    const double f = volfrac[cell * n_mat + m];
    if (f >= thresh_min && f <= thresh_max) {
      return true;
    }
  }
  return false;
}

void build_interface_mask(const double* volfrac_host,
                          std::size_t n_cells,
                          std::size_t n_mat,
                          int nr,
                          int nz,
                          int halo_radius_cells,
                          double thresh_min,
                          double thresh_max,
                          std::vector<std::uint8_t>& out_mask);

}  // namespace tenryu::hydro::plic
