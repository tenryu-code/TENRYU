#pragma once

#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::coupling {

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     double dt,
                                     double* E_floor_injected = nullptr,
                                     int* clamp_count = nullptr,
                                     const std::vector<double>* sigma_R_max = nullptr);

double inject_radiation_source_terms(core::State& state,
                                     const core::Config& cfg,
                                     double dt,
                                     double* E_floor_injected,
                                     int* clamp_count,
                                     const std::vector<double>* sigma_R_max,
                                     const hydro::HydroEOSContext* eos_ctx);

double inject_laser_source_terms(core::State& state,
                                 const core::Config& cfg,
                                 double dt,
                                 double* E_floor_injected = nullptr,
                                 int* clamp_count = nullptr,
                                 const hydro::HydroEOSContext* eos_ctx = nullptr);

// Deposits the burn energies dE_e/dE_i [erg per cell] and closes the deposit
// cells on the device (NUMERICS §14.5); cells without a deposit are left as
// they are. Returns the deposit that fell on void or massless cells.
double inject_burn_source_terms(core::State& state,
                                const core::Config& cfg,
                                const std::vector<double>& dE_e,
                                const std::vector<double>& dE_i,
                                double* E_floor_injected,
                                int* clamp_count,
                                const hydro::HydroEOSContext* eos_ctx = nullptr);

void apply_qei_coupling_substep(core::State& state,
                                const core::Config& cfg,
                                double dt_sub,
                                const hydro::HydroEOSContext* eos_ctx = nullptr);

}  // namespace tenryu::coupling
