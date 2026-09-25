#pragma once

#include "core/config.hpp"
#include "core/state.hpp"
#include "materials/eos_cell_table_selector.cuh"

namespace tenryu::radiation {

// Per-cell electron EOS table selector for the matter updates of the 1D
// radiation solvers (FLD and S_N): every cell takes the electron table and
// the closure parameters of its dominant non-void material
// (State::cell_material_index). Every material's electron table is uploaded
// once (keyed on the Config's table pointers); a material on the exact
// ideal-gas hydro backend, or without tables, gets an empty view and its
// cells close with the ideal gas. The selector is null (selection disabled:
// the kernels keep the first material's view and the run-level closure
// parameters) when the run has at most one non-void material, no material
// has a table and the materials' closure parameters are the same, or the
// dominant-material index is not sized. It carries the per-material closure
// parameters (cv_e_override) when they differ
// (materials::selector_closure_params).
materials::CellEOSTableSelector cell_electron_table_selector_1d(core::State& state,
                                                                const core::Config& cfg,
                                                                int n_cells);

}  // namespace tenryu::radiation
