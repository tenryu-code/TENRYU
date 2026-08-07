#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/particle_pool.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct SourceStats {
  int n_thermal = 0;
  int n_marshak = 0;
  double E_thermal = 0.0;
  double E_thermal_lost = 0.0;
  double E_marshak = 0.0;
  int pool_overflow = 0;
};

void rebin_census_particles_1d_cuda(std::int32_t* cell_id,
                                    const double* pos_r,
                                    const double* node_r,
                                    int n_particles,
                                    int n_cells);

void flip_negative_census_signs_cuda(std::int8_t* sign, int n_particles);

class IMCSource {
 public:
  // Optional NLTE source coefficients may be provided either as host vectors or
  // as device pointers for GPU-native thermal source assembly.
  static SourceStats emit_thermal(core::State& state,
                                  const core::Config& cfg,
                                  const PlanckTable& planck,
                                  const double* sigma_a_eff,
                                  PhotonPool& pool,
                                  int max_pool_size,
                                  double dt,
                                  std::uint64_t step_number,
                                  std::uint64_t user_seed,
                                  std::uint64_t step_base_gid,
                                  const std::vector<double>* nlte_eta = nullptr,
                                  const std::vector<double>* nlte_f = nullptr,
                                  const std::vector<double>* source_tilt = nullptr,
                                  const double* sloc_mean_r = nullptr,
                                  const double* sloc_sigma = nullptr,
                                  const double* sloc_alpha = nullptr,
                                  const double* sloc_prev_E = nullptr,
                                  int ppcg_override = -1,
                                  const double* emission_bias_cdf = nullptr,
                                  const std::vector<std::uint8_t>* source_skip_cell = nullptr,
                                  const double* reference_E = nullptr,
                                  const double* d_nlte_eta = nullptr,
                                  const double* d_nlte_f = nullptr);

  static SourceStats emit_marshak(core::State& state,
                                  const core::Config& cfg,
                                  const PlanckTableDeviceView& planck,
                                  PhotonPool& pool,
                                  int max_pool_size,
                                  double dt,
                                  std::uint64_t step_number,
                                  std::uint64_t user_seed,
                                  std::uint64_t step_base_gid);

  static SourceStats emit_volume_source(core::State& state,
                                        const core::Config& cfg,
                                        PhotonPool& pool,
                                        int max_pool_size,
                                        double dt,
                                        std::uint64_t step_number,
                                        std::uint64_t user_seed,
                                        std::uint64_t step_base_gid);

  static SourceStats append_census_residual_1d(core::State& state,
                                               PhotonPool& pool,
                                               int max_pool_size,
                                               double dt,
                                               std::uint64_t step_number,
                                               std::uint64_t user_seed,
                                               std::uint64_t gid_base,
                                               const std::vector<std::int32_t>& cell_id,
                                               const std::vector<std::uint16_t>& group_id,
                                               const std::vector<double>& energy,
                                               const std::vector<std::int8_t>& sign);
};

}  // namespace tenryu::radiation
