#pragma once

#include "core/state.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

void initialize_radiation_field_equilibrium_gpu(
    core::State& state,
    const PlanckTable& planck);

void initialize_radiation_field_planck_gpu(
    core::State& state,
    const PlanckTable& planck,
    double Tr_eV);

// Applies the deck-declared Geometry.radiation_field initial condition
// (equilibrium at local Te, or uniform-Tr Planck). No-op when radiation is
// disabled or radiation_field=="zero". Shared by cmd_run and the verify
// loader so the two entry points cannot diverge
// (docs/design/verify_radiation_field_init_fix_20260712.md).
void apply_initial_radiation_field(core::State& state,
                                   const core::Config& cfg);

}  // namespace tenryu::radiation
