#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace tenryu {
namespace materials { struct IonmixOpacityDeviceView; }
namespace radiation {

struct PlanckTableDeviceView;

struct NlteCoeffsDeviceResult {
    int negative_alpha_clamp_count = 0;
    int negative_eta_clamp_count = 0;
    int nan_inf_count = 0;
};

/// GPU computation of NLTE radiation coefficients using separate emissivity.
/// Outputs are written directly to device buffers owned by the caller.
NlteCoeffsDeviceResult compute_nlte_coefficients_cuda(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells, int n_groups,
    double dt, double A, double alpha,
    double f_min, double f_max,
    double fd_delta_rel, double fd_abs_min,
    double sigma_cap, double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1, bool linearized_planck,
    bool use_freeze_opacity, bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa, double* d_sigma_R,
    double* d_sigma_a_eff, double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_lambda_raw,
    cudaStream_t stream,
    bool low_density_extrap = false,
    const int* cell_material_index = nullptr,
    int material_filter = -1);

/// When pinned_clamp_counts is non-null, it must point to page-locked host memory
/// with room for 3 ints: slot 0 = negative_alpha, slot 1 = negative_eta, and
/// slot 2 = nan_inf. The function issues the counter copies asynchronously into
/// these slots and does not synchronize; the caller must perform (or rely on) a
/// stream-ordered synchronization before reading. When reuse_device_void_mask is
/// true, the function skips re-uploading the cell_is_void mask and reuses the
/// pooled device buffer contents from the previous call; this is valid only
/// within the same step where the mask is unchanged.
NlteCoeffsDeviceResult compute_nlte_coefficients_cuda_with_pe(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells, int n_groups,
    double dt, double A, double alpha,
    double f_min, double f_max,
    double fd_delta_rel, double fd_abs_min,
    double sigma_cap, double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1, bool linearized_planck,
    bool use_freeze_opacity, bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa, double* d_sigma_pe, double* d_sigma_R,
    double* d_sigma_a_eff, double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_lambda_raw,
    cudaStream_t stream,
    bool low_density_extrap = false,
    int* pinned_clamp_counts = nullptr,
    bool reuse_device_void_mask = false,
    const int* cell_material_index = nullptr,
    int material_filter = -1);

/// Pure deterministic S_N coefficient path: raw PA/PE opacities and emissivity,
/// with Fleck linearization bypassed.
NlteCoeffsDeviceResult compute_nlte_coefficients_cuda_pure_sn(
    const double* d_rho,
    const double* d_Te,
    const double* d_zbar,
    const double* d_cv_e,
    const std::uint8_t* cell_is_void_host,
    std::size_t cell_is_void_size,
    const materials::IonmixOpacityDeviceView& table_view,
    const PlanckTableDeviceView& planck_view,
    int n_cells, int n_groups,
    double dt, double A, double alpha,
    double f_min, double f_max,
    double fd_delta_rel, double fd_abs_min,
    double sigma_cap, double cv_e_override,
    double temperature_floor_eV,
    double gamma_m1, bool linearized_planck,
    bool use_freeze_opacity, bool corrected_fleck,
    double* d_f,
    double* d_sigma_pa, double* d_sigma_pe, double* d_sigma_R,
    double* d_sigma_a_eff, double* d_sigma_s_eff,
    double* d_eta_cdf,
    double* d_eta,
    double* d_lambda_raw,
    cudaStream_t stream,
    bool low_density_extrap = false);

}  // namespace radiation
}  // namespace tenryu
