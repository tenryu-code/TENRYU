#pragma once

#include <cstdint>
#include <vector>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/ddmc_event.hpp"
#include "radiation/ddmc_momentum.hpp"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::parallel {
struct PartitionInfo;
}  // namespace tenryu::parallel

namespace tenryu::radiation {

struct DDMCDiagnostics {
  std::int64_t processed = 0;
  std::int64_t absorbed = 0;
  std::int64_t scattered = 0;
  std::int64_t leak_left = 0;
  std::int64_t leak_right = 0;
  std::int64_t leak_face2 = 0;
  std::int64_t leak_face3 = 0;
  std::int64_t leak_boundary = 0;
  std::int64_t census = 0;
  std::int64_t converted_to_imc = 0;
  std::int64_t converted_to_rw = 0;
  std::int64_t sigma_tot_zero = 0;
  std::int64_t max_events_reached = 0;
};

class DDMCTransport1D {
 public:
  DDMCTransport1D() = default;
  DDMCTransport1D(std::int64_t n_cells,
                  int n_groups,
                  bool interface_exit_half_isotropic = false);

  void process_range(PhotonPool& pool,
                     int start_index,
                     int end_index,
                     double dt,
                     std::uint64_t step_number,
                     std::uint64_t user_seed,
                     const std::vector<double>& node_r,
                     const ModeSelector& mode_selector,
                     const DDMCCoefficients& coefficients,
                     const std::vector<double>& sigma_a_eff,
                     const std::vector<double>* sigma_s_eff,
                     const std::vector<double>* eta_cdf,
                     std::vector<double>& rad_dep,
                     std::vector<double>& rad_E_tally,
                     std::vector<double>& E_escape,
                     double* E_numerical_loss = nullptr,
                     DDMCMomentumEstimator* momentum = nullptr,
                     const tenryu::parallel::PartitionInfo* partition = nullptr);

  [[nodiscard]] const DDMCDiagnostics& diagnostics() const { return diagnostics_; }

 private:
  [[nodiscard]] std::size_t index(std::int64_t cell, int group) const;

  std::int64_t n_cells_ = 0;
  int n_groups_ = 0;
  bool interface_exit_half_isotropic_ = false;
  DDMCDiagnostics diagnostics_{};
};

class DDMCTransport2D {
 public:
  DDMCTransport2D() = default;
  DDMCTransport2D(std::int64_t n_cells,
                  int n_groups,
                  int nr,
                  int nz,
                  bool interface_exit_half_isotropic = false);

  void process_range(PhotonPool& pool,
                     int start_index,
                     int end_index,
                     double dt,
                     std::uint64_t step_number,
                     std::uint64_t user_seed,
                     const std::vector<double>& node_r,
                     const std::vector<double>& node_z,
                     const ModeSelector& mode_selector,
                     const DDMCCoefficients& coefficients,
                     const std::vector<double>& sigma_a_eff,
                     const std::vector<double>* sigma_s_eff,
                     const std::vector<double>* eta_cdf,
                     std::vector<double>& rad_dep,
                     std::vector<double>& rad_E_tally,
                     std::vector<double>& E_escape,
                     double* E_numerical_loss = nullptr,
                     DDMCMomentumEstimator* momentum = nullptr,
                     const tenryu::parallel::PartitionInfo* partition = nullptr);

  [[nodiscard]] const DDMCDiagnostics& diagnostics() const { return diagnostics_; }

 private:
  [[nodiscard]] std::size_t index(std::int64_t cell, int group) const;
  [[nodiscard]] int neighbor_from_face(std::int64_t cell, int face) const;
  void sample_ddmc_to_imc_direction(double xi_mu,
                                    double xi_phi,
                                    double nr,
                                    double nz,
                                    double tr,
                                    double tz,
                                    double* dir_r,
                                    double* dir_z,
                                    double* dir_phi) const;

  std::int64_t n_cells_ = 0;
  int n_groups_ = 0;
  int nr_ = 0;
  int nz_ = 0;
  bool interface_exit_half_isotropic_ = false;
  DDMCDiagnostics diagnostics_{};
};

}  // namespace tenryu::radiation
