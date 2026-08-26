#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "core/config.hpp"
#include "core/device_error_flags.cuh"
#include "laser/laser_mesh.cuh"
#include "laser/ray_init.cuh"

namespace tenryu::laser {
struct LaserPhysExtOptions;
}

namespace tenryu::laser {


// CBET v1 recorder outputs (design docs/design/cbet_1d_v1_design_20260707.md §4.1).
// When rec_cell != nullptr the 1D_SPH trace kernel runs in record mode: it appends
// per-cell-crossing records (merged consecutive same-cell substeps) instead of
// being the deposition authority. All ledgers written by the record-mode kernel
// through the legacy deposit/unabsorbed arguments are caller-provided dummies.
struct CbetRecordDeviceArgs {
  std::int32_t* rec_cell = nullptr;
  float* rec_mu = nullptr;
  // 2D raytrace_3d record mode only; nullptr in 1D
  float* rec_c = nullptr;
  float* rec_w00 = nullptr;
  float* rec_w10 = nullptr;
  float* rec_w01 = nullptr;
  double* rec_ds = nullptr;
  double* rec_S = nullptr;
  double* rec_w = nullptr;
  std::int32_t* traj_rec_idx = nullptr;  // [n_output_rays * traj_max_steps], device-only
  std::int32_t* rec_count = nullptr;    // [n_rays_total], indexed at ray_offset + tid
  std::uint8_t* ray_overflow = nullptr; // [n_rays_total]
  long long ray_offset = 0;
  int cap_per_ray = 0;
};

// Multi-channel hot-electron capture parameters (design
// docs/design/hote_directional_sources_20260710.md §4.2). Channels are
// host-sorted by ascending threshold — the physical crossing order along the
// inward march — so a segment crossing several thresholds fires them
// lower-first, each later channel reading the ray power AFTER the upstream
// channels' (1-eta) depletion.
struct HotECaptureParams {
  static constexpr int kMaxChannels = 4;
  int n_channels = 0;                        // ACTIVE channels only; 0 = capture off
  double threshold_nhat[kMaxChannels] = {};  // f_s in n_e/n_crit units, ascending
  double one_minus_eta[kMaxChannels] = {};   // downstream power scale (1.0 = no subtraction)
};

#ifdef __CUDACC__
__global__ __launch_bounds__(64, 16)
void ray_trace_2d(double* __restrict__ deposit,
                  const double* __restrict__ n_hat,
                  const double* __restrict__ n_hat_raw,
                  const double* __restrict__ grad_n_hat_R,
                  const double* __restrict__ grad_n_hat_Z,
                  const double* __restrict__ T_e,
                  const double* __restrict__ Zbar,
                  const double* __restrict__ smooth_kappa_factor,
                  const double* __restrict__ node_R,
                  const double* __restrict__ node_Z,
                  const double* __restrict__ ray_R0,
                  const double* __restrict__ ray_Z0,
                  const double* __restrict__ ray_vR0,
                  const double* __restrict__ ray_vZ0,
                  const double* __restrict__ ray_power,
                  const double* __restrict__ ray_power0,
                  double cfl_ray,
                  double ds_adapt_g_target,
                  double ds_adapt_tau_target,
                  double ds_adapt_theta_target,
                  double ds_adapt_max_factor,
                  double eps_n,
                  double eps_crit,
                  double lambda_cm,
                  double coulomb_log_floor,
                  double test_kappa_cm_inv,
                  double intensity_cutoff,
                  int max_ray_steps,
                  int n_nodes_r,
                  int n_nodes_z,
                  int n_rays,
                  double* __restrict__ traj_pos_R,
                  double* __restrict__ traj_pos_Z,
                  double* __restrict__ traj_power,
                  int* __restrict__ traj_step_count,
                  int n_output_rays,
                  int output_stride,
                  int traj_max_steps,
                  int* __restrict__ step_histogram,
                  int* __restrict__ step_count,
                  double* __restrict__ P_unabsorbed,
                  unsigned long long* __restrict__ tail_closure_count,
                  double* __restrict__ tail_closure_absorbed_power,
                  unsigned long long* __restrict__ critical_surface_hit_count,
                  core::DeviceErrorFlags* __restrict__ error_flags);

template <bool kCbetRecord, bool kHotECapture>
__global__ __launch_bounds__(64, 16)
void ray_trace_1d_sph(double* __restrict__ deposit_1d,
                      const double* __restrict__ radial_node_r,
                      const double* __restrict__ radial_n_hat,
                      const double* __restrict__ radial_n_hat_raw,
                      const double* __restrict__ radial_smooth_kappa,
                      const double* __restrict__ radial_dn_dr,
                      const double* __restrict__ hydro_r_edges,
                      int allowed_supercritical_cell,
                      int critical_adjacent_subcritical_cell,
                      double critical_adjacent_split_r,
                      const double* __restrict__ ray_R0,
                      const double* __restrict__ ray_Z0,
                      const double* __restrict__ ray_vR0,
                      const double* __restrict__ ray_vZ0,
                      const double* __restrict__ ray_power,
                      const double* __restrict__ ray_power0,
                      double cfl_ray,
                      double ds_adapt_g_target,
                      double ds_adapt_tau_target,
                      double ds_adapt_theta_target,
                      double ds_adapt_max_factor,
                      double eps_n,
                      double eps_crit,
                      double lambda_cm,
                      double coulomb_log_floor,
                      double test_kappa_cm_inv,
                      double intensity_cutoff,
                      int max_ray_steps,
                      int n_radial_nodes,
                      int n_hydro_cells,
                      int n_rays,
                      double* __restrict__ traj_pos_R,
                      double* __restrict__ traj_pos_Z,
                      double* __restrict__ traj_power,
                      int* __restrict__ traj_step_count,
                      int n_output_rays,
                      int output_stride,
                      int traj_max_steps,
                      int* __restrict__ step_histogram,
                      int* __restrict__ step_count,
                      double* __restrict__ P_unabsorbed,
                      unsigned long long* __restrict__ tail_closure_count,
                      double* __restrict__ tail_closure_absorbed_power,
                      unsigned long long* __restrict__ critical_surface_hit_count,
                      core::DeviceErrorFlags* __restrict__ error_flags,
                      const CbetRecordDeviceArgs cbet_args,
                      const HotECaptureParams hot_e_params,
                      double* __restrict__ hot_e_capture   // rows [(tid*n_channels + ch)*4 + {0:valid,1:r_s,2:mu_axis,3:P_before}]
                      );

__global__ __launch_bounds__(64, 16)
void ray_trace_3d(double* __restrict__ deposit,
                  const double* __restrict__ n_hat,
                  const double* __restrict__ n_hat_raw,
                  const double* __restrict__ grad_n_hat_R,
                  const double* __restrict__ grad_n_hat_Z,
                  const double* __restrict__ T_e,
                  const double* __restrict__ Zbar,
                  const double* __restrict__ smooth_kappa_factor,
                  const double* __restrict__ node_R,
                  const double* __restrict__ node_Z,
                  const double* __restrict__ ray_x0,
                  const double* __restrict__ ray_y0,
                  const double* __restrict__ ray_z0,
                  const double* __restrict__ ray_vx0,
                  const double* __restrict__ ray_vy0,
                  const double* __restrict__ ray_vz0,
                  const double* __restrict__ ray_power,
                  const double* __restrict__ ray_power0,
                  double cfl_ray,
                  double ds_adapt_g_target,
                  double ds_adapt_tau_target,
                  double ds_adapt_theta_target,
                  double ds_adapt_max_factor,
                  double eps_n,
                  double eps_crit,
                  double lambda_cm,
                  double coulomb_log_floor,
                  double test_kappa_cm_inv,
                  double intensity_cutoff,
                  int max_ray_steps,
                  int n_nodes_r,
                  int n_nodes_z,
                  int n_rays,
                  double dx_lm_min,
                  double* __restrict__ traj_pos_x,
                  double* __restrict__ traj_pos_y,
                  double* __restrict__ traj_pos_z,
                  double* __restrict__ traj_power,
                  int* __restrict__ traj_step_count,
                  int n_output_rays,
                  int output_stride,
                  int traj_max_steps,
                  int* __restrict__ step_histogram,
                  int* __restrict__ step_count,
                  double* __restrict__ P_unabsorbed,
                  unsigned long long* __restrict__ tail_closure_count,
                  double* __restrict__ tail_closure_absorbed_power,
                  unsigned long long* __restrict__ critical_surface_hit_count,
                  core::DeviceErrorFlags* __restrict__ error_flags);
#endif

[[nodiscard]] cudaError_t launch_ray_trace_2d(const RayArray1D& rays,
                                              const LaserMesh& mesh,
                                              const core::Config::LaserConfig& laser_cfg,
                                              double lambda_cm,
                                              double* d_traj_pos1,
                                              double* d_traj_pos2,
                                              double* d_traj_pos3,
                                              double* d_traj_power,
                                              int* d_traj_step_count,
                                              int n_output_rays,
                                              int output_stride,
                                              int traj_max_steps,
                                              double* d_unabsorbed,
                                              core::DeviceErrorFlags* d_error_flags,
                                              unsigned long long* d_tail_closure_count = nullptr,
                                              double* d_tail_closure_absorbed_power = nullptr,
                                              unsigned long long* d_critical_surface_hit_count =
                                                  nullptr,
                                              cudaStream_t stream = nullptr,
                                              int* d_step_histogram = nullptr,
                                              const double* smooth_kappa_factor = nullptr,
                                              int* d_step_count = nullptr);

[[nodiscard]] cudaError_t launch_ray_trace_1d_sph(
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
    double* d_traj_pos1,
    double* d_traj_pos2,
    double* d_traj_pos3,
    double* d_traj_power,
    int* d_traj_step_count,
    int n_output_rays,
    int output_stride,
    int traj_max_steps,
    double* d_unabsorbed,
    core::DeviceErrorFlags* d_error_flags,
    unsigned long long* d_tail_closure_count = nullptr,
    double* d_tail_closure_absorbed_power = nullptr,
    unsigned long long* d_critical_surface_hit_count = nullptr,
    cudaStream_t stream = nullptr,
    int* d_step_histogram = nullptr,
    int* d_step_count = nullptr,
    const CbetRecordDeviceArgs* cbet_record = nullptr,
    const HotECaptureParams hot_e_params = {},
    double** d_hot_e_capture_out = nullptr,
    const laser::LaserPhysExtOptions* phys_ext = nullptr,
    const double* d_radial_T_e = nullptr,
    double* d_ra_power_total = nullptr,
    const int* h_ray_order = nullptr,
    int* h_ray_steps_out = nullptr,
    const int max_ray_steps_override = 0,
    double* d_tau_shell_out = nullptr,
    double* d_pabs_per_ray_out = nullptr);

[[nodiscard]] cudaError_t launch_radial_absorption_1d(
    double P_total,
    const LaserMesh& mesh,
    const core::Config::LaserConfig& laser_cfg,
    const double* d_hydro_r_edges,
    int n_hydro_cells,
    double* d_deposit_1d,
    double* d_unabsorbed,
    core::DeviceErrorFlags* d_error_flags,
    unsigned long long* d_critical_surface_hit_count = nullptr,
    cudaStream_t stream = nullptr,
    const HotECaptureParams hot_e_params = {},
    double** d_hot_e_capture_out = nullptr);

[[nodiscard]] cudaError_t launch_ray_trace_3d(const RayArray2D& rays,
                                              const LaserMesh& mesh,
                                              const core::Config::LaserConfig& laser_cfg,
                                              double lambda_cm,
                                              double* d_traj_pos1,
                                              double* d_traj_pos2,
                                              double* d_traj_pos3,
                                              double* d_traj_power,
                                              int* d_traj_step_count,
                                              int n_output_rays,
                                              int output_stride,
                                              int traj_max_steps,
                                              double* d_unabsorbed,
                                              core::DeviceErrorFlags* d_error_flags,
                                              const HotECaptureParams& hot_e_params,
                                              double* hot_e_capture,
                                              unsigned long long* d_tail_closure_count = nullptr,
                                              double* d_tail_closure_absorbed_power = nullptr,
                                              unsigned long long* d_critical_surface_hit_count =
                                                  nullptr,
                                              cudaStream_t stream = nullptr,
                                              int* d_step_histogram = nullptr,
                                              const double* smooth_kappa_factor = nullptr,
                                              int* d_step_count = nullptr,
                                              bool allow_debug_one_ray = true,
                                              const CbetRecordDeviceArgs* cbet_record = nullptr);

// Fixed-order tally-reduction machinery, exported so the CBET
// solve can reduce per-beam row ranges without relaunching the tracer.
[[nodiscard]] cudaError_t launch_reduce_per_ray_tallies_1d(
    const double* d_deposit_per_ray,
    const double* d_unabsorbed_per_ray,
    const double* d_tail_power_per_ray,
    double* d_deposit_1d,
    double* d_P_unabsorbed,
    double* d_tail_closure_absorbed_power,
    int n_rays,
    int n_cells,
    cudaStream_t stream = nullptr,
    const double* d_ra_per_ray = nullptr,
    double* d_ra_total = nullptr);

}  // namespace tenryu::laser
