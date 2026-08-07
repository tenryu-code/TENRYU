#pragma once

#include <cstdint>

#include "core/device_error_flags.cuh"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct DDMCTransport2DGPUInputs {
  PhotonPool* pool = nullptr;

  // DDMC coefficients (device pointers)
  const double* sigma_a_eff = nullptr;      // [n_cells * n_groups]
  const double* sigma_s_eff = nullptr;      // [n_cells * n_groups] or nullptr
  const double* sigma_leak_face = nullptr;  // [n_cells * 4 * n_groups]
  const std::uint8_t* bc_face = nullptr;    // [n_cells * 4 * n_groups]
  const int* neighbor_face = nullptr;       // [n_cells * 4]
  const double* eta_cdf = nullptr;          // [n_cells * n_groups] or nullptr
  const TransportMode* ddmc_mode = nullptr; // [n_cells * n_groups], mode support for g_out

  // Geometry
  const double* node_r = nullptr;  // [(nr+1)*(nz+1)]
  const double* node_z = nullptr;  // [(nr+1)*(nz+1)]

  // Tallies (device, shared with IMC)
  double* rad_dep = nullptr;
  double* rad_E_tally = nullptr;
  double* E_escape = nullptr;
  double* E_numerical_loss = nullptr;

  // Diagnostics counters (device)
  unsigned long long* ddmc_absorbed = nullptr;
  unsigned long long* ddmc_census = nullptr;
  unsigned long long* ddmc_leak_face0 = nullptr;
  unsigned long long* ddmc_leak_face1 = nullptr;
  unsigned long long* ddmc_leak_face2 = nullptr;
  unsigned long long* ddmc_leak_face3 = nullptr;
  unsigned long long* ddmc_leak_boundary = nullptr;
  unsigned long long* ddmc_converted_to_imc = nullptr;
  unsigned long long* ddmc_sigma_tot_zero = nullptr;
  unsigned long long* ddmc_max_events_reached = nullptr;

  int n_cells = 0;
  int n_groups = 0;
  int nr = 0;
  int nz = 0;
  int n_ddmc = 0;
  int ddmc_start = 0;

  // Rank boundary info (0 = single-rank, no ghost detection)
  int ghost_layers = 0;
  int nr_local = 0;
  int nz_local = 0;
  bool has_r_inner_boundary = false;
  bool has_r_outer_boundary = false;
  bool has_z_bottom_boundary = false;
  bool has_z_top_boundary = false;

  std::uint8_t interface_exit_distribution = 0;  // 0=cosine, 1=half_isotropic
  double dt = 0.0;

  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  core::DeviceErrorFlags* error_flags = nullptr;
};

void ddmc_transport_2d_gpu_cuda(const DDMCTransport2DGPUInputs& in);

}  // namespace tenryu::radiation
