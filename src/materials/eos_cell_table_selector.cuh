#pragma once

#include <cstdint>

#include "materials/eos_device_table.cuh"

namespace tenryu::materials {

// Hydro EOS backend kinds of a material slot (hydro::HydroEOSContext).
enum HydroBackendKind : std::uint8_t {
  kHydroBackendLegacy = 0u,           // tables when present, else the ideal gas
  kHydroBackendHelmholtzSpline = 1u,  // total-EOS bicubic spline surrogate
  kHydroBackendHelmholtzJet = 2u,     // total-EOS projected-jet surrogate
  kHydroBackendExactIdealGas = 3u,    // ideal-gas closure, tables unused
  kHydroBackendRhoETable = 4u,        // precomputed P(rho, e), T(rho, e)
  kHydroBackendMieGruneisen = 5u,     // 2T trajectory-fitted Mie-Gruneisen
};

// Closure parameters of one material slot for the 1D per-cell closures
// (2026-09-24). The 1D closures took these from the first non-void material
// for every cell; a deck whose non-void materials differ in them now closes
// each cell with its own material's values. CellEOSTableSelector::
// closure_params is null when every non-void material has the same values
// (single-material and uniform decks), and the closures then keep their
// scalar arguments and arithmetic.
struct MaterialClosureParams {
  double Z = 0.0;
  double A = 1.0;
  double gamma = 5.0 / 3.0;
  // Volumetric electron heat capacity [erg/(cm^3 eV)]; <= 0: none.
  double cv_e_override = 0.0;
  // With cv_e_override: reference temperature of the T^3 heat capacity
  // closure (e = cv_e_override T^4 / (4 T_ref^3 rho)); <= 0: none.
  double eos_T_ref_eV = 0.0;
  std::uint8_t hydro_backend_kind = kHydroBackendLegacy;
  std::uint8_t is_void = 0u;
};

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
//   * a cell whose dominant material has no table of the requested kind gets
//     that material's empty view (n_rho == 0), and the caller closes the cell
//     with its ideal-gas branch (the cell's effective A and gamma). It used to
//     fall back to the caller's view, i.e. another material's table, and that
//     view differed between callers (2026-09-23). Launches that also serve 2D
//     set lend_fallback_to_tableless there: the 2D closures still evaluate
//     every cell with one material's tables, so a 2D cell whose material has
//     no table keeps the caller's view;
//   * out-of-range indices are clamped into [0, n_materials).
// Single-material runs select the first non-void slot == the fallback view,
// so their arithmetic is unchanged.
struct CellEOSTableSelector {
  const DeviceEOSTableView* ion_views = nullptr;       // [n_materials], device
  const DeviceEOSTableView* electron_views = nullptr;  // [n_materials], device
  const DeviceEOSTableView* total_views = nullptr;     // [n_materials], device
  const int* cell_material_index = nullptr;            // [n_cells], device
  int n_materials = 0;
  // Return the caller's view for a cell whose material has no table (see
  // above; 2D launches only).
  bool lend_fallback_to_tableless = false;
  // Per-material closure parameters ([n_materials], device), or null when
  // every non-void material has the same values (MaterialClosureParams).
  const MaterialClosureParams* closure_params = nullptr;

#ifdef __CUDACC__
  // Dominant material slot of `cell`, or -1 when the selection is disabled.
  __device__ inline int material_of(const int cell) const {
    if (cell_material_index == nullptr || n_materials <= 0) {
      return -1;
    }
    const int m = cell_material_index[cell];
    return (m < 0) ? 0 : ((m >= n_materials) ? (n_materials - 1) : m);
  }

  // Closure parameters of `cell`'s material, or null when they are uniform
  // (the caller then uses its scalar arguments).
  __device__ inline const MaterialClosureParams* closure_of(const int cell) const {
    if (closure_params == nullptr) {
      return nullptr;
    }
    const int m = material_of(cell);
    return (m < 0) ? nullptr : &closure_params[m];
  }
  __device__ inline double cv_e_override(const int cell, const double fallback) const {
    const MaterialClosureParams* p = closure_of(cell);
    return (p != nullptr) ? p->cv_e_override : fallback;
  }
  __device__ inline double eos_T_ref(const int cell, const double fallback) const {
    const MaterialClosureParams* p = closure_of(cell);
    return (p != nullptr) ? p->eos_T_ref_eV : fallback;
  }
  __device__ inline std::uint8_t hydro_backend(const int cell,
                                               const std::uint8_t fallback) const {
    const MaterialClosureParams* p = closure_of(cell);
    return (p != nullptr) ? p->hydro_backend_kind : fallback;
  }

  __device__ inline DeviceEOSTableView pick(const DeviceEOSTableView* views,
                                            const int cell,
                                            const DeviceEOSTableView& fallback) const {
    const int m = material_of(cell);
    if (m < 0 || views == nullptr) {
      return fallback;
    }
    const DeviceEOSTableView view = views[m];
    return (view.n_rho > 0 || !lend_fallback_to_tableless) ? view : fallback;
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
