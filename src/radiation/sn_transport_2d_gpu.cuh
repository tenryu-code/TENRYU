#pragma once

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

inline int sn_2d_rz_n_r_faces(const int nr, const int nz) {
  return (nr + 1) * nz;
}

inline int sn_2d_rz_n_z_faces(const int nr, const int nz) {
  return nr * (nz + 1);
}

inline int sn_2d_rz_n_faces(const int nr, const int nz) {
  return sn_2d_rz_n_r_faces(nr, nz) + sn_2d_rz_n_z_faces(nr, nz);
}

inline int sn_2d_rz_r_face_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

inline int sn_2d_rz_z_face_index(const int i,
                                 const int j,
                                 const int nr,
                                 const int nz) {
  return sn_2d_rz_n_r_faces(nr, nz) + i * (nz + 1) + j;
}

void advance_radiation_step_sn_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    double dt,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    parallel::CommBuffers* bufs = nullptr);

void compute_sn_E_star_flux_2d_rz_gpu(
    const double* rad_E_old,
    const double* node_r,
    const double* node_z,
    const double* vol,
    const double* face_flux,
    double* E_star,
    double* diag_E_star_flux,
    int nr,
    int nz,
    int n_groups,
    double dt,
    cudaStream_t stream = nullptr);

void compute_sn_donor_theta_limited_face_flux_2d_rz_gpu(
    const double* rad_E_old,
    const double* node_r,
    const double* node_z,
    const double* vol,
    const double* face_flux_raw,
    double* theta_donor,
    double* stream_theta,
    double* face_flux_limited,
    int nr,
    int nz,
    int n_groups,
    double dt,
    cudaStream_t stream = nullptr);

void compute_sn_diffusion_face_flux_2d_rz_gpu(
    const double* rad_E_old,
    const double* sn_sigma_s,
    const double* node_r,
    const double* node_z,
    const double* vol,
    double* face_flux_diff,
    int nr,
    int nz,
    int n_groups,
    double opacity_floor,
    cudaStream_t stream = nullptr);

void compute_sn_ap_blended_face_flux_2d_rz_gpu(
    const double* face_flux_sn,
    const double* face_flux_diff,
    const double* rad_E_old,
    const double* Te,
    const double* sn_sigma_s,
    const double* node_r,
    const double* node_z,
    const double* vol,
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
    int nr,
    int nz,
    int n_groups,
    cudaStream_t stream = nullptr);

}  // namespace tenryu::radiation
