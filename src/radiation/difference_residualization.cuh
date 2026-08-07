#pragma once

#include <cstdint>

namespace tenryu::core {
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
struct DeviceArray;
}  // namespace tenryu::parallel

namespace tenryu::radiation {

struct PhotonPool;

struct DifferenceResidualizationDeviceStats {
  int scaled_bins = 0;
  int rebuilt_bins = 0;
  int killed_bins = 0;
  int empty_created = 0;
  int n_before = 0;
  int n_after = 0;
};

void prepare_difference_census_reference_cuda(const PhotonPool& pool,
                                              const double* rad_E,
                                              const double* vol,
                                              const double* previous_reference_U,
                                              bool have_previous_reference,
                                              int n_cells,
                                              int n_groups,
                                              double* physical_E_density,
                                              parallel::DeviceArray& workspace);

int kill_difference_census_in_holo_core_cuda(PhotonPool& pool,
                                             const std::uint8_t* holo_core,
                                             int n_particles,
                                             int n_cells);

DifferenceResidualizationDeviceStats residualize_census_against_reference_cuda(
    core::State& state,
    PhotonPool& pool,
    int max_pool_size,
    double dt,
    std::uint64_t step_number,
    std::uint64_t user_seed,
    std::uint64_t empty_gid_base,
    const double* E_ref_start,
    const double* vol,
    double* previous_reference_U_start,
    int n_cells,
    int n_groups,
    PhotonPool& output_workspace,
    parallel::DeviceArray& workspace,
    parallel::DeviceArray& scan_workspace);

}  // namespace tenryu::radiation
