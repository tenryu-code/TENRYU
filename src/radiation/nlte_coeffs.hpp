#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "radiation/cell_radiation_coeffs.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct SeparateEmissivityScalars {
  std::vector<double> b;
  std::vector<double> s;
  std::vector<double> cdf;
  double sigma_p_abs = 0.0;
  double sigma_p_em = 0.0;
  double gamma_diag = 0.0;
  double f = 1.0;
};

struct SpectrumReconstructionDiagnostics {
  int fallback_cell_count = 0;
  int fallback_group_count = 0;
};

SeparateEmissivityScalars compute_separate_emissivity_scalars(
    const std::vector<double>& alpha_g,
    const std::vector<double>& eta_g,
    const std::vector<double>& b_g,
    double temperature_eV,
    double beta,
    double dt,
    double alpha_fleck,
    double temperature_floor_eV);

void apply_nlte_transport_linearized_correction(
    int n_groups,
    const std::vector<double>& eta0,
    const std::vector<double>& alpha0,
    const std::vector<double>& J0,
    const std::vector<double>& deta_dJ,
    const std::vector<double>& dalpha_dJ,
    const std::vector<double>& J,
    std::vector<double>* eta,
    std::vector<double>* alpha);

SpectrumReconstructionDiagnostics reconstruct_nlte_group_spectrum(
    const core::State& state,
    const PlanckTable& planck,
    int n_cells,
    int n_groups,
    std::vector<double>* J,
    std::vector<std::uint8_t>* used_planck_fallback);

std::vector<double> compute_nlte_sigma_p_em(const core::State& state,
                                            const core::Config& cfg,
                                            const materials::IonmixOpacityData& table,
                                            const PlanckTable& planck,
                                            int n_cells,
                                            int n_groups);

CellRadiationCoeffs compute_nlte_coefficients(const core::State& state,
                                              const core::Config& cfg,
                                              const materials::IonmixOpacityData& table,
                                              const PlanckTable& planck,
                                              int n_cells,
                                              int n_groups,
                                              double dt);

}  // namespace tenryu::radiation
