#pragma once

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

// Per-cell EOS table selection for the multi-material closure chain
// (2026-09-14). The 1D closure kernels historically evaluated every cell with
// the first non-void material's tables. This selector points at the
// per-material device view arrays owned by hydro::HydroEOSContext
// (d_ion_views / d_electron_views / d_total_views, indexed by the Config
// material slot) and at the per-cell dominant non-void material index owned
// by core::State (cell_material_index, maintained by
// State::ensure_cell_material_props). Selection rules:
//   * a null view array or a null index array disables the selection and the
//     caller's fallback view (the first non-void material's table) is used;
//   * a cell whose dominant material has no table of the requested kind
//     (n_rho == 0) also falls back, so table-less materials keep their
//     previous behaviour;
//   * out-of-range indices are clamped into [0, n_materials).
// Single-material runs select the first non-void slot == the fallback view,
// so their arithmetic is unchanged.
struct CellEOSTableSelector {
  const DeviceEOSTableView* ion_views = nullptr;       // [n_materials], device
  const DeviceEOSTableView* electron_views = nullptr;  // [n_materials], device
  const DeviceEOSTableView* total_views = nullptr;     // [n_materials], device
  const int* cell_material_index = nullptr;            // [n_cells], device
  int n_materials = 0;

#ifdef __CUDACC__
  // Dominant material slot of `cell`, or -1 when the selection is disabled.
  __device__ inline int material_of(const int cell) const {
    if (cell_material_index == nullptr || n_materials <= 0) {
      return -1;
    }
    const int m = cell_material_index[cell];
    return (m < 0) ? 0 : ((m >= n_materials) ? (n_materials - 1) : m);
  }

  __device__ inline DeviceEOSTableView pick(const DeviceEOSTableView* views,
                                            const int cell,
                                            const DeviceEOSTableView& fallback) const {
    const int m = material_of(cell);
    if (m < 0 || views == nullptr) {
      return fallback;
    }
    const DeviceEOSTableView view = views[m];
    return (view.n_rho > 0) ? view : fallback;
  }

  __device__ inline DeviceEOSTableView ion(const int cell,
                                           const DeviceEOSTableView& fallback) const {
    return pick(ion_views, cell, fallback);
  }
  __device__ inline DeviceEOSTableView electron(const int cell,
                                                const DeviceEOSTableView& fallback) const {
    return pick(electron_views, cell, fallback);
  }
  __device__ inline DeviceEOSTableView total(const int cell,
                                             const DeviceEOSTableView& fallback) const {
    return pick(total_views, cell, fallback);
  }
#endif  // __CUDACC__
};

// Host-side builder (all pointers are device pointers; any may be null).
inline CellEOSTableSelector make_cell_eos_table_selector(
    const DeviceEOSTableView* ion_views,
    const DeviceEOSTableView* electron_views,
    const DeviceEOSTableView* total_views,
    const int n_materials,
    const int* cell_material_index) {
  CellEOSTableSelector selector;
  selector.ion_views = ion_views;
  selector.electron_views = electron_views;
  selector.total_views = total_views;
  selector.n_materials = n_materials;
  selector.cell_material_index = cell_material_index;
  return selector;
}

}  // namespace tenryu::materials
