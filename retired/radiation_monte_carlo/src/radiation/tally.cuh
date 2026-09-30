#pragma once

#include <cstdint>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::parallel {
struct DeviceArray;
}  // namespace tenryu::parallel

namespace tenryu::radiation {

struct PlanckTableDeviceView;
struct ReferenceFieldDiagnostics;

struct ReferenceFaceTransportResult {
  double E_escape = 0.0;
  double E_source = 0.0;
};

void zero_tallies_cuda(double* rad_dep,
                       double* rad_E_tally,
                       double* E_escape,
                       int n_cells,
                       int n_groups,
                       double* holo_Prr_tally = nullptr,
                       double* holo_Prr_coverage_tally = nullptr);

void preseed_reference_absorption_cuda(double* rad_dep,
                                       const double* sigma_a_eff,
                                       const double* E_ref,
                                       const double* vol,
                                       int n_cells,
                                       int n_groups,
                                       double dt);

ReferenceFaceTransportResult reference_face_transport_1d_cuda(
    double* U_ref_end,
    double* ref_face_delta_U,
    double* E_ref_avg,
    const double* E_ref_start,
    const double* sigma_R,
    const double* node_r,
    const double* vol,
    const double* boundary_E_inner,
    const double* boundary_E_outer,
    int n_cells,
    int n_groups,
    double dt,
    int bc_inner,
    int bc_outer,
    parallel::DeviceArray* boundary_workspace = nullptr,
    double* ref_face_current = nullptr);

ReferenceFieldDiagnostics compute_reference_field_diagnostics_cuda(
    const core::State& state,
    const core::Config& cfg,
    PlanckTableDeviceView planck,
    const double* sigma_R,
    const double* physical_E_density,
    const std::uint8_t* diffusion_cell,
    int n_cells,
    int n_groups,
    double* E_ref_start,
    double* W_ref_cell,
    parallel::DeviceArray& cell_workspace,
    parallel::DeviceArray& face_workspace,
    parallel::DeviceArray& void_workspace);

void tally_finalize_cuda(double* rad_E,
                         const double* rad_E_tally,
                         const double* vol,
                         int n_cells,
                         int n_groups,
                         double dt,
                         const double* diff_E = nullptr,
                         const std::uint8_t* diff_cell = nullptr,
                         const double* E_ref_avg = nullptr,
                         double* residual_E = nullptr);

void holo_accept_radiation_cuda(double* rad_E,
                                const double* E_LO,
                                const std::uint8_t* core_mask,
                                int n_cells,
                                int n_groups,
                                const double* vol = nullptr,
                                double* previous_reference_U = nullptr);

void holo_set_reference_U_cuda(double* previous_reference_U,
                               const double* rad_E,
                               const double* vol,
                               const std::uint8_t* core_mask,
                               int n_cells,
                               int n_groups);

void compute_holo_consistency_cuda(double* consistency_source,
                                   const double* E_HO,
                                   const double* E_LO_pred,
                                   const double* E_old,
                                   const double* face_current_step,
                                   const double* reference_face_current,
                                   const double* sigma_R,
                                   const double* vol,
                                   const double* node_r,
                                   const double* lo_weight,
                                   int n_cells,
                                   int n_groups,
                                   double dt);

void difference_reproject_residual_cuda(double* residual_E,
                                        const double* rad_E,
                                        const double* E_ref,
                                        int n_cells,
                                        int n_groups,
                                        const std::uint8_t* core_mask = nullptr);

void holo_prr_finalize_cuda(double* Prr,
                            double* chi,
                            double* coverage,
                            const double* Prr_tally,
                            const double* coverage_tally,
                            const double* rad_E_tally,
                            const double* vol,
                            const std::uint8_t* holo_core,
                            int n_cells,
                            int n_groups,
                            double dt,
                            double E_floor);

}  // namespace tenryu::radiation
