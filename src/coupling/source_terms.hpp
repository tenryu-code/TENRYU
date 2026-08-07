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
                                 int* clamp_count = nullptr);

// Deposit burn dE_e/dE_i [erg/cell] with laser host-mirror closure; returns skipped energy.
double inject_burn_source_terms(core::State& state,
                                const core::Config& cfg,
                                const std::vector<double>& dE_e,
                                const std::vector<double>& dE_i,
                                double* E_floor_injected,
                                int* clamp_count);

void apply_qei_coupling_substep(core::State& state,
                                const core::Config& cfg,
                                double dt_sub,
                                const hydro::HydroEOSContext* eos_ctx = nullptr);

}  // namespace tenryu::coupling
