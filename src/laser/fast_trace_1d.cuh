#pragma once

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_error_flags.cuh"
#include "laser/laser_mesh.cuh"
#include "laser/ray_init.cuh"

namespace tenryu::laser {

struct LaserPhysExtOptions;

[[nodiscard]] cudaError_t launch_fast_trace_1d(
    const RayArray1D& rays,
    const LaserMesh& mesh,
    const core::Config::LaserConfig& laser_cfg,
    double lambda_cm,
    const double* d_hydro_r_edges,
    int n_hydro_cells,
    int allowed_supercritical_cell,
    int critical_adjacent_subcritical_cell,
    double critical_adjacent_split_r,
    double* d_deposit_1d,
    double* d_unabsorbed,
    core::DeviceErrorFlags* d_error_flags,
    unsigned long long* d_critical_surface_hit_count = nullptr,
    cudaStream_t stream = nullptr,
    int* d_step_histogram = nullptr,
    int* d_step_count = nullptr,
    int* d_traj_step_count = nullptr,
    int n_output_rays = 0,
    const LaserPhysExtOptions* phys_ext = nullptr,
    const double* d_radial_T_e = nullptr,
    double beam_P_w = 0.0,
    double beam_w_cm = 0.0,
    double* d_ra_power_total = nullptr,
    double* d_tau_shell_out = nullptr,
    double* d_pabs_per_ray_out = nullptr);

}  // namespace tenryu::laser
