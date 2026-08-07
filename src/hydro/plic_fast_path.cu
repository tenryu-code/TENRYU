#include "hydro/plic_fast_path.cuh"

#include <algorithm>

namespace tenryu::hydro::plic {

void build_interface_mask(const double* volfrac_host,
                          const std::size_t n_cells,
                          const std::size_t n_mat,
                          const int nr,
                          const int nz,
                          const int halo_radius_cells,
                          const double thresh_min,
                          const double thresh_max,
                          std::vector<std::uint8_t>& out_mask) {
  out_mask.assign(n_cells, static_cast<std::uint8_t>(0));
  if (volfrac_host == nullptr || nr <= 0 || nz <= 0 ||
      static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz) != n_cells ||
      n_mat == 0) {
    return;
  }

  const int radius = std::max(0, halo_radius_cells);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = static_cast<std::size_t>(i * nz + j);
      if (!is_multi_material_cell(volfrac_host, c, n_mat, thresh_min, thresh_max)) {
        continue;
      }

      const int i0 = std::max(0, i - radius);
      const int i1 = std::min(nr - 1, i + radius);
      const int j0 = std::max(0, j - radius);
      const int j1 = std::min(nz - 1, j + radius);
      for (int ii = i0; ii <= i1; ++ii) {
        for (int jj = j0; jj <= j1; ++jj) {
          out_mask[static_cast<std::size_t>(ii * nz + jj)] =
              static_cast<std::uint8_t>(1);
        }
      }
    }
  }
}

}  // namespace tenryu::hydro::plic
