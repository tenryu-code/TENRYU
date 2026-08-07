#pragma once

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

struct Sn1DDiagnostics {
  int outer_iterations = 0;
  int inner_iterations = 0;
  bool converged = false;
  double outer_residual = 0.0;
  double inner_residual = 0.0;
  double escaped_energy = 0.0;
};

void advance_radiation_step_sn_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt);

void compute_sn_E_star_flux_1d_gpu(
    const double* rad_E_old,
    const double* node_r,
    const double* vol,
    const double* face_flux,
    double* E_star,
    double* diag_E_star_flux,
    int n_cells,
    int n_groups,
    double dt,
    cudaStream_t stream,
    const int geom);

void compute_sn_donor_theta_limited_face_flux_1d_gpu(
    const double* rad_E_old,
    const double* node_r,
    const double* vol,
    const double* face_flux_raw,
    double* const stream_theta_donor,
    double* stream_theta,
    double* face_flux_limited,
    int n_cells,
    int n_groups,
    double dt,
    cudaStream_t stream,
    const int geom);

void compute_sn_diffusion_face_flux_1d_gpu(
    const double* rad_E_old,
    const double* sn_sigma_s,
    const double* node_r,
    double* face_flux_diff,
    int n_cells,
    int n_groups,
    double opacity_floor,
    cudaStream_t stream = nullptr);

void compute_sn_ap_blended_face_flux_1d_gpu(
    const double* face_flux_sn,
    const double* face_flux_diff,
    const double* rad_E_old,
    const double* Te,
    const double* sn_sigma_s,
    const double* node_r,
    PlanckTableDeviceView planck,
    double opacity_floor,
    double tau_lo,
    double tau_hi,
    double eq_full,
    double eq_off,
    double f_full,
    double f_off,
    double* face_flux_blended,
    double* alpha_face,
    int n_cells,
    int n_groups,
    cudaStream_t stream = nullptr);

}  // namespace tenryu::radiation
