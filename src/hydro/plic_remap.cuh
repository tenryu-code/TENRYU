#pragma once

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::coupling {
struct ProfileObservability;
}

namespace tenryu::hydro::plic {

#if defined(__CUDACC__)
#define TENRYU_PLIC_HOST_DEVICE __host__ __device__
#else
#define TENRYU_PLIC_HOST_DEVICE
#endif

TENRYU_PLIC_HOST_DEVICE inline double rho_material_aware_donor_density(
    const double cell_mean_rho,
    const double* material_volfrac,
    const double* material_rho,
    const int n_mat,
    const bool enabled) {
  if (!enabled || material_volfrac == nullptr || material_rho == nullptr ||
      n_mat <= 0) {
    return cell_mean_rho;
  }
  double rho_eff = 0.0;
  double vf_sum = 0.0;
  for (int m = 0; m < n_mat; ++m) {
    const double f = material_volfrac[m];
    const double rho_m = material_rho[m];
    if (f > 0.0 && rho_m > 0.0) {
      rho_eff += f * rho_m;
      vf_sum += f;
    }
  }
  return (rho_eff > 0.0 && vf_sum > 0.0) ? rho_eff : cell_mean_rho;
}

TENRYU_PLIC_HOST_DEVICE inline double rho_material_aware_donor_flux(
    const double swept_volume,
    const double cell_mean_rho,
    const double* material_volfrac,
    const double* material_rho,
    const int n_mat,
    const bool enabled) {
  return swept_volume *
         rho_material_aware_donor_density(
             cell_mean_rho, material_volfrac, material_rho, n_mat, enabled);
}

#undef TENRYU_PLIC_HOST_DEVICE

struct PlicRemapStatus {
  bool active = false;
  bool fallback_engaged = false;
  bool drift_triggered = false;
  bool cf6_density_remap_used = false;
  int interface_cells = 0;
  int active_cells = 0;
  int reconstruction_attempts = 0;
  int reconstruction_successes = 0;
  int axis_exempt_cells = 0;
  int class_d_events = 0;
  int repair_events = 0;
  double max_volume_fraction_residual = 0.0;
  double max_interface_centroid_drift_relative = 0.0;
  double max_swept_fraction = 0.0;
};

[[nodiscard]] bool plic_runtime_active(const tenryu::core::State& state,
                                       const tenryu::core::Config& cfg);

[[nodiscard]] std::size_t plic_cell_material_scratch_size(
    const tenryu::core::Config& cfg,
    int n_cells);

void ensure_plic_remap_scratch(tenryu::core::State& state,
                               const tenryu::core::Config& cfg,
                               int n_cells,
                               int n_mat,
                               int nr,
                               int nz);

void reset_plic_runtime(tenryu::core::State& state);

PlicRemapStatus launch_plic_reconstruction(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr,
    const double* d_xz,
    const double* d_vol,
    int nr,
    int nz,
    tenryu::coupling::ProfileObservability* observability,
    cudaStream_t stream = nullptr);

PlicRemapStatus launch_plic_drift_sensor(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const double* d_xr,
    const double* d_xz,
    const double* d_vol,
    int nr,
    int nz,
    tenryu::coupling::ProfileObservability* observability,
    cudaStream_t stream = nullptr);

PlicRemapStatus launch_plic_material_volume_remap(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double* d_volfrac_mat,
    const double* d_vol_old,
    const double* d_vol_new,
    const double* d_xr_old,
    const double* d_xz_old,
    const double* d_xr_new,
    const double* d_xz_new,
    int nr,
    int nz,
    tenryu::coupling::ProfileObservability* observability,
    const tenryu::hydro::HydroEOSContext* eos_ctx = nullptr,
    cudaStream_t stream = nullptr);

void apply_plic_fallback_policy(tenryu::core::State& state,
                                const tenryu::core::Config& cfg,
                                const PlicRemapStatus& status,
                                tenryu::coupling::ProfileObservability* observability);

}  // namespace tenryu::hydro::plic
