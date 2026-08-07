#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/state.hpp"
#include "laser/laser_mesh.cuh"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::laser {

struct AllowedSupercriticalCell1D {
  int allowed_cell = -1;
  int critical_adjacent_subcritical_cell = -1;
  double r_crit = -1.0;
  bool fallback_only = false;
};

AllowedSupercriticalCell1D find_allowed_supercritical_cell_1d(
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& r_edges,
    double n_crit);

void transfer_to_1d(core::State& state,
                    LaserMesh& mesh,
                    double dt,
                    double conservation_tol = 1.0e-10,
                    cudaStream_t stream = nullptr,
                    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
                    int smooth_passes = 0,
                    double smooth_alpha = 0.25);

void transfer_to_1d(core::State& state,
                    LaserMesh& mesh,
                    const HydroMirror1D& hydro,
                    const std::vector<double>& deposit_lm,
                    const std::vector<double>& node_R,
                    const std::vector<double>& node_Z,
                    double dt,
                    double conservation_tol = 1.0e-10,
                    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
                    int smooth_passes = 0,
                    double smooth_alpha = 0.25);

void apply_deposit_redistribution_1d(
    core::State& state,
    LaserMesh& mesh,
    const HydroMirror1D& hydro,
    const std::vector<double>& deposit_power_cell,
    double dt,
    double conservation_tol = 1.0e-10,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    int smooth_passes = 0,
    double smooth_alpha = 0.25,
    const std::vector<double>* hot_e_extra_power = nullptr);

void transfer_to_2d(core::State& state,
                    LaserMesh& mesh,
                    double dt,
                    double conservation_tol = 1.0e-10,
                    cudaStream_t stream = nullptr,
                    double* applied_scale = nullptr,
                    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
                    int smooth_passes = 0,
                    double smooth_alpha = 0.0,
                    const std::vector<double>* hot_e_power_cell = nullptr,
                    const parallel::Reduction* reduction = nullptr);

}  // namespace tenryu::laser
