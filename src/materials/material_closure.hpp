#pragma once

#include <cstdint>
#include <vector>

#include "materials/eos_cell_table_selector.cuh"

namespace tenryu::core {
struct Config;
}

namespace tenryu::materials {

// Hydro EOS backend kind of a Config material slot (HydroBackendKind), by the
// rules of hydro::HydroEOSContext: a table surrogate needs the material's EOS
// tables, the exact ideal gas does not.
std::uint8_t hydro_backend_kind_of(const core::Config& cfg, int material_index);

// Closure parameters of every Config material slot (host).
std::vector<MaterialClosureParams> material_closure_params(const core::Config& cfg);

// True when the non-void materials of a 1D run differ in a parameter that the
// 1D closures used to take from the first non-void material for every cell:
// the hydro backend kind, cv_e_override or eos_T_ref_eV. (A, gamma and Z enter
// the closures per cell through State::A_eff / gamma_eff and zbar.) The
// per-cell selectors carry the parameters (CellEOSTableSelector::
// closure_params) only then.
bool material_closure_params_vary(const core::Config& cfg);

// True when the non-void materials of a 1D run differ in their hydro backend
// kind.
bool hydro_backend_kinds_vary(const core::Config& cfg);

// True when the non-void materials of a 1D run differ in A or Z.
bool material_A_or_Z_vary(const core::Config& cfg);

// Device array of material_closure_params(cfg) for a 1D run with materials,
// else nullptr. Uploaded once per parameter set and kept for the process.
const MaterialClosureParams* material_closure_params_device(const core::Config& cfg);

// material_closure_params_device(cfg) when material_closure_params_vary(cfg),
// else nullptr: the value of CellEOSTableSelector::closure_params.
const MaterialClosureParams* selector_closure_params(const core::Config& cfg);

}  // namespace tenryu::materials
