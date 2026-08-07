#include "materials/zmoment_state.cuh"

#include <cmath>
#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "materials/zmoment_device.cuh"

namespace tenryu::materials {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void zmoment_fill_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    double* __restrict__ r2,
    double* __restrict__ r4,
    const int n_cells,
    const ZMomentDeviceTables tables) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  if (!(rho[c] > 0.0) || !(A_eff[c] > 0.0) || !(Te[c] > 0.0)) {
    r2[c] = 1.0;
    r4[c] = 1.0;
    return;
  }

  const double ni_cm3 =
      rho[c] / (A_eff[c] * core::constants::proton_mass);
  r2[c] = zmoment_r2(tables, ni_cm3, Te[c]);
  r4[c] = zmoment_r4(tables, ni_cm3, Te[c]);
}

}  // namespace

void zmoment_upload_tables(core::State& state, const core::Config& cfg) {
  const auto& src = cfg.materials.zmoments;
  if (src.ndens <= 0) {
    return;
  }

  TENRYU_ASSERT(!state.zmom_active,
                "zmoment_upload_tables called more than once without release");
  TENRYU_ASSERT(src.ntemp > 0,
                "zmoment_upload_tables requires ntemp > 0");
  const std::size_t nd = static_cast<std::size_t>(src.ndens);
  const std::size_t nt = static_cast<std::size_t>(src.ntemp);
  TENRYU_ASSERT(nd <= std::numeric_limits<std::size_t>::max() / nt,
                "zmoment_upload_tables table size overflow");
  const std::size_t table_size = nd * nt;
  TENRYU_ASSERT(src.ni_grid.size() == nd &&
                    src.T_grid_eV.size() == nt &&
                    src.r2.size() == table_size &&
                    src.r4.size() == table_size,
                "zmoment_upload_tables config table size mismatch");

  state.zmom_r2_table_storage.reset(table_size);
  state.zmom_r4_table_storage.reset(table_size);
  state.zmom_r2_table_storage.copy_from_host(src.r2);
  state.zmom_r4_table_storage.copy_from_host(src.r4);

  state.zmom_tables.r2 = state.zmom_r2_table_storage.data();
  state.zmom_tables.r4 = state.zmom_r4_table_storage.data();
  state.zmom_tables.nd = src.ndens;
  state.zmom_tables.nt = src.ntemp;
  state.zmom_tables.l10d0 = std::log10(src.ni_grid.front());
  state.zmom_tables.dl10d =
      src.ndens > 1
          ? std::log10(src.ni_grid[1]) - state.zmom_tables.l10d0
          : 1.0;
  state.zmom_tables.l10t0 = std::log10(src.T_grid_eV.front());
  state.zmom_tables.dl10t =
      src.ntemp > 1
          ? std::log10(src.T_grid_eV[1]) - state.zmom_tables.l10t0
          : 1.0;
  state.zmom_active = true;
}

void zmoment_fill_fields(core::State& state, cudaStream_t stream) {
  if (!state.zmom_active) {
    return;
  }

  const std::size_t n = state.rho.size();
  TENRYU_ASSERT(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "zmoment_fill_fields cell count exceeds INT_MAX");
  TENRYU_ASSERT(state.Te.size() == n &&
                    state.A_eff.size() == n &&
                    state.zmom_r2.size() == n &&
                    state.zmom_r4.size() == n,
                "zmoment_fill_fields state field size mismatch");
  if (n == 0U) {
    return;
  }

  constexpr int block_size = 256;
  const int n_cells = static_cast<int>(n);
  const int n_blocks = (n_cells + block_size - 1) / block_size;
  zmoment_fill_kernel<<<n_blocks, block_size, 0, stream>>>(
      state.rho.data(),
      state.Te.data(),
      state.A_eff.data(),
      state.zmom_r2.data(),
      state.zmom_r4.data(),
      n_cells,
      state.zmom_tables);
  cuda_check(cudaGetLastError(), "zmoment_fill_fields kernel launch failed");
}

}  // namespace tenryu::materials
