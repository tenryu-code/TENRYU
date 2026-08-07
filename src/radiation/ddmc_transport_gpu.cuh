#pragma once

#include <cstdint>

#include "core/device_error_flags.cuh"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct DDMCTransportGPUInputs {
  PhotonPool* pool = nullptr;  // device SoA arrays

  // DDMC coefficients (device pointers, flat [n_cells * n_groups])
  const double* sigma_a_eff = nullptr;
  const double* sigma_s_eff = nullptr;
  const double* sigma_leak_left = nullptr;
  const double* sigma_leak_right = nullptr;
  const std::uint8_t* bc_left = nullptr;   // DDMCBoundaryType cast to uint8_t
  const std::uint8_t* bc_right = nullptr;  // DDMCBoundaryType cast to uint8_t
  const double* eta_cdf = nullptr;         // [n_cells * n_groups] or nullptr
  const TransportMode* ddmc_mode = nullptr; // [n_cells * n_groups], mode support for g_out

  // Geometry
  const double* node_r = nullptr;  // [n_cells + 1]

  // Tallies (device, shared with IMC)
  double* rad_dep = nullptr;           // [n_cells * n_groups]
  double* rad_E_tally = nullptr;       // [n_cells * n_groups]
  double* E_escape = nullptr;          // [n_groups]
  double* E_numerical_loss = nullptr;  // [1]

  // Diagnostics counters (device)
  unsigned long long* ddmc_absorbed = nullptr;
  unsigned long long* ddmc_census = nullptr;
  unsigned long long* ddmc_leak_left = nullptr;
  unsigned long long* ddmc_leak_right = nullptr;
  unsigned long long* ddmc_leak_boundary = nullptr;
  unsigned long long* ddmc_vacuum_leak_left = nullptr;
  unsigned long long* ddmc_vacuum_leak_right = nullptr;
  unsigned long long* ddmc_converted_to_imc = nullptr;
  unsigned long long* ddmc_converted_to_rw = nullptr;
  unsigned long long* ddmc_sigma_tot_zero = nullptr;
  unsigned long long* ddmc_max_events_reached = nullptr;

  int n_cells = 0;
  int n_groups = 0;
  int n_ddmc = 0;      // number of DDMC particles
  int ddmc_start = 0;  // offset into pool (= n_imc from sort result)

  // Rank boundary info (0 = single-rank, no ghost detection)
  int ghost_layers = 0;
  int nr_local = 0;  // owned cells in R direction (excluding ghosts)
  bool has_left_boundary = false;
  bool has_right_boundary = false;

  std::uint8_t interface_exit_distribution = 0;  // 0=cosine, 1=half_isotropic
  double dt = 0.0;

  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  core::DeviceErrorFlags* error_flags = nullptr;
};

void ddmc_transport_gpu_cuda(const DDMCTransportGPUInputs& in);

}  // namespace tenryu::radiation
