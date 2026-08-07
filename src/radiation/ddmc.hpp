#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "radiation/cell_radiation_coeffs.hpp"
#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_momentum.hpp"
#include "radiation/ddmc_transport.hpp"
#include "radiation/mmatrix_check.hpp"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct HysteresisState {
  const std::vector<TransportMode>* prev_mode = nullptr;
  const std::vector<double>* prev_tau = nullptr;
  std::vector<std::uint8_t>* hold_count = nullptr;
  std::int64_t switches_imc_to_ddmc = 0;
  std::int64_t switches_ddmc_to_imc = 0;
};

struct DDMCPreparation {
  ModeSelector mode_selector;
  DDMCCoefficients coefficients;
  MMatrixDiagnostics mmatrix;
  std::int64_t ddmc_mode_count = 0;
  std::int64_t rw_mode_count = 0;
  std::int64_t imc_mode_count = 0;
  std::int64_t omega_below_threshold = 0;
  bool active = false;
};

struct NLTEDDMCCellClosureStats {
  std::int64_t cells_forced_imc = 0;
  std::int64_t ddmc_groups_forced_imc = 0;
};

NLTEDDMCCellClosureStats apply_true_nlte_ddmc_cell_closure(
    ModeSelector* mode_selector,
    const CellRadiationCoeffs& coeffs,
    double support_tol = 1.0e-12,
    double sigma_s_tol = 1.0e-30);

DDMCPreparation prepare_ddmc_step(PhotonPool& pool,
                                  const core::Config& cfg,
                                  std::uint64_t step_number,
                                  double dt,
                                  int mesh_dim,
                                  int n_cells,
                                  int n_groups,
                                  const std::vector<double>& node_r,
                                  const std::vector<double>& node_z,
                                  int nr,
                                  int nz,
                                  const std::vector<double>& cell_vol,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& Te,
                                  const std::vector<double>& sigma_R,
                                  const std::vector<double>& sigma_a,
                                  const std::vector<double>& sigma_a_eff,
                                  const std::vector<double>& fleck_f,
                                  const CellRadiationCoeffs* coeffs,
                                  DDMCBoundaryType bc_inner,
                                  DDMCBoundaryType bc_outer,
                                  DDMCBoundaryType bc_bottom_z,
                                  DDMCBoundaryType bc_top_z,
                                  HysteresisState* hysteresis = nullptr,
                                  const std::vector<std::uint8_t>* force_imc_cells = nullptr);

DDMCDiagnostics run_ddmc_transport_step(PhotonPool& pool,
                                        int start_index,
                                        int end_index,
                                        const core::Config& cfg,
                                        double dt,
                                        std::uint64_t step_number,
                                        int mesh_dim,
                                        const std::vector<double>& node_r,
                                        const std::vector<double>& node_z,
                                        int nr,
                                        int nz,
                                        const ModeSelector& mode_selector,
                                        const DDMCCoefficients& coefficients,
                                        const std::vector<double>& sigma_a_eff,
                                        const std::vector<double>* sigma_s_eff,
                                        const std::vector<double>* eta_cdf,
                                        std::vector<double>& rad_dep,
                                        std::vector<double>& rad_E_tally,
                                        std::vector<double>& E_escape,
                                        double* E_numerical_loss = nullptr,
                                        DDMCMomentumEstimator* momentum = nullptr);

}  // namespace tenryu::radiation
