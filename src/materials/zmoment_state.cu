#include "materials/zmoment_state.cuh"

#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

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

// Multi-material decks: each cell's material's tables (a material without
// them has null pointers and zmoment_r2/r4 return 1).
__global__ void zmoment_fill_by_material_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ A_eff,
    const int* __restrict__ cell_material_index,
    double* __restrict__ r2,
    double* __restrict__ r4,
    const int n_cells,
    const ZMomentDeviceTables* __restrict__ tables_by_material,
    const int n_materials) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  if (!(rho[c] > 0.0) || !(A_eff[c] > 0.0) || !(Te[c] > 0.0)) {
    r2[c] = 1.0;
    r4[c] = 1.0;
    return;
  }
  const int m_raw = cell_material_index[c];
  const int m = (m_raw < 0) ? 0 : ((m_raw >= n_materials) ? n_materials - 1 : m_raw);
  const ZMomentDeviceTables tables = tables_by_material[m];
  const double ni_cm3 =
      rho[c] / (A_eff[c] * core::constants::proton_mass);
  r2[c] = zmoment_r2(tables, ni_cm3, Te[c]);
  r4[c] = zmoment_r4(tables, ni_cm3, Te[c]);
}

ZMomentDeviceTables device_tables_view(const double* r2,
                                       const double* r4,
                                       const core::Config::MaterialsConfig::ZMomentTables& src) {
  ZMomentDeviceTables t;
  t.r2 = r2;
  t.r4 = r4;
  t.nd = src.ndens;
  t.nt = src.ntemp;
  t.l10d0 = std::log10(src.ni_grid.front());
  t.dl10d = src.ndens > 1 ? std::log10(src.ni_grid[1]) - t.l10d0 : 1.0;
  t.l10t0 = std::log10(src.T_grid_eV.front());
  t.dl10t = src.ntemp > 1 ? std::log10(src.T_grid_eV[1]) - t.l10t0 : 1.0;
  return t;
}

// Several non-void materials with tables of their own (zmoments_by_material):
// one storage for all tables and a device array of per-material views.
bool zmoment_upload_tables_by_material(core::State& state, const core::Config& cfg) {
  const auto& mats = cfg.materials.materials;
  const auto& by_mat = cfg.materials.zmoments_by_material;
  int n_nonvoid = 0;
  for (const auto& mat : mats) {
    n_nonvoid += mat.is_void ? 0 : 1;
  }
  if (n_nonvoid <= 1 || by_mat.size() != mats.size()) {
    return false;
  }
  std::size_t total = 0;
  for (const auto& t : by_mat) {
    if (t.ndens > 0) {
      TENRYU_ASSERT(t.ntemp > 0 && t.ni_grid.size() == static_cast<std::size_t>(t.ndens) &&
                        t.T_grid_eV.size() == static_cast<std::size_t>(t.ntemp) &&
                        t.r2.size() == static_cast<std::size_t>(t.ndens) *
                                           static_cast<std::size_t>(t.ntemp) &&
                        t.r4.size() == t.r2.size(),
                    "zmoment_upload_tables per-material table size mismatch");
      total += t.r2.size();
    }
  }
  if (total == 0) {
    return false;
  }
  std::vector<double> r2_host;
  std::vector<double> r4_host;
  r2_host.reserve(total);
  r4_host.reserve(total);
  std::vector<std::size_t> offsets(by_mat.size(), 0);
  for (std::size_t m = 0; m < by_mat.size(); ++m) {
    offsets[m] = r2_host.size();
    if (by_mat[m].ndens > 0) {
      r2_host.insert(r2_host.end(), by_mat[m].r2.begin(), by_mat[m].r2.end());
      r4_host.insert(r4_host.end(), by_mat[m].r4.begin(), by_mat[m].r4.end());
    }
  }
  state.zmom_r2_table_storage.reset(total);
  state.zmom_r4_table_storage.reset(total);
  state.zmom_r2_table_storage.copy_from_host(r2_host);
  state.zmom_r4_table_storage.copy_from_host(r4_host);
  std::vector<ZMomentDeviceTables> views(by_mat.size());
  for (std::size_t m = 0; m < by_mat.size(); ++m) {
    if (by_mat[m].ndens > 0) {
      views[m] = device_tables_view(state.zmom_r2_table_storage.data() + offsets[m],
                                    state.zmom_r4_table_storage.data() + offsets[m],
                                    by_mat[m]);
    }
  }
  state.zmom_tables_by_material.reset(views.size());
  state.zmom_tables_by_material.copy_from_host(views);
  state.zmom_tables = ZMomentDeviceTables{};
  state.zmom_active = true;
  return true;
}

}  // namespace

void zmoment_upload_tables(core::State& state, const core::Config& cfg) {
  const auto& src = cfg.materials.zmoments;
  if (src.ndens <= 0) {
    return;
  }
  TENRYU_ASSERT(!state.zmom_active,
                "zmoment_upload_tables called more than once without release");
  if (zmoment_upload_tables_by_material(state, cfg)) {
    return;
  }

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
  if (!state.zmom_tables_by_material.empty()) {
    TENRYU_ASSERT(state.cell_material_index.size() == n,
                  "zmoment_fill_fields per-material tables need the cell material index");
    zmoment_fill_by_material_kernel<<<n_blocks, block_size, 0, stream>>>(
        state.rho.data(),
        state.Te.data(),
        state.A_eff.data(),
        state.cell_material_index.data(),
        state.zmom_r2.data(),
        state.zmom_r4.data(),
        n_cells,
        state.zmom_tables_by_material.data(),
        static_cast<int>(state.zmom_tables_by_material.size()));
    cuda_check(cudaGetLastError(), "zmoment_fill_fields per-material kernel launch failed");
    return;
  }
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
