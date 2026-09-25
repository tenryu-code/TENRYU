#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace tenryu {
namespace materials {
struct IonmixOpacityDeviceView;
struct DeviceEOSTableView;
struct CellEOSTableSelector;
}  // namespace materials
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
/// When table_cv_electron_eos is non-null (Radiation.multigroup_diffusion.
/// fleck_cv_source="table"), the Fleck factor's heat capacity is the electron
/// table's cv at the cell temperature (the cell's dominant material through
/// table_cv_cell_tables when given, else this view) wherever such a table
/// exists; other cells, and all cells when it is null, keep the chain
/// cv_e_override -> d_cv_e -> ideal gas.
/// accumulate_clamp_counts keeps the device clamp counters of the previous
/// launch instead of zeroing them, so the per-material launches of one
/// multi-material evaluation copy the running total into pinned_clamp_counts
/// (the last copy holds every material's counts).
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
    int material_filter = -1,
    const materials::DeviceEOSTableView* table_cv_electron_eos = nullptr,
    const materials::CellEOSTableSelector* table_cv_cell_tables = nullptr,
    bool accumulate_clamp_counts = false,
    // Radiation.multigroup_diffusion.fleck_beta / fleck_form: beta mode 0
    // tangent, 1 secant, 2 guard (the secant needs the step-start radiation
    // d_rad_E_old and an electron table for the heat capacity, else the
    // tangent beta); form 0 f = 1/(1+z), 1 f = (1 - exp(-z))/z.
    const double* d_rad_E_old = nullptr,
    int fleck_beta_mode = 0,
    int fleck_form_exp = 0);

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
    bool low_density_extrap = false,
    // Only the cells whose cell_material_index equals material_filter (the
    // per-material launches of a multi-material S_N evaluation); null: all.
    const int* cell_material_index = nullptr,
    int material_filter = -1);

}  // namespace radiation
}  // namespace tenryu
