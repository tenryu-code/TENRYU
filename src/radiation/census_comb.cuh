#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"

namespace tenryu::radiation {

struct PhotonPool;
using CensusCombConfig = core::Config::RadiationConfig::CensusCombConfig;

struct CensusCombResult {
  int n_alive_out = 0;
  int n_combed_bins = 0;
  int target_count = 0;
  double E_before = 0.0;
  double E_after = 0.0;
  double E_killed_bins = 0.0;
  bool emergency = false;
  bool zero_score = false;
};

struct EssFloorResult {
  int n_alive_out = 0;
  int n_split_bins = 0;
  double E_before = 0.0;
  double E_after = 0.0;
};

CensusCombResult census_comb(PhotonPool& pool,
                             int n_alive,
                             int n_cells,
                             int n_groups,
                             const CensusCombConfig& cfg,
                             std::uint64_t step_base_gid,
                             std::uint64_t n_emit_total,
                             std::uint32_t seed,
                             int step,
                             double* d_E_numerical_loss);

CensusCombResult census_comb_gpu(PhotonPool& pool,
                                 int n_alive,
                                 int n_cells,
                                 int n_groups,
                                 const CensusCombConfig& cfg,
                                 std::uint64_t step_base_gid,
                                 std::uint64_t n_emit_total,
                                 std::uint32_t seed,
                                 int step,
                                 double* d_E_numerical_loss);

EssFloorResult census_ess_floor_gpu(PhotonPool& pool,
                                    int n_alive,
                                    int n_cells,
                                    int n_groups,
                                    const CensusCombConfig& cfg,
                                    const std::vector<double>& importance,
                                    std::uint64_t step_base_gid,
                                    std::uint64_t n_emit_total,
                                    std::uint32_t seed,
                                    int step);

}  // namespace tenryu::radiation
