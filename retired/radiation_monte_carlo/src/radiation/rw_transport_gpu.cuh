#pragma once

#include <cstdint>

#include "core/device_error_flags.cuh"
#include "radiation/boundary.cuh"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {

struct RWTransportGPUInputs {
  PhotonPool* pool = nullptr;

  const double* sigma_R = nullptr;       // [n_cells * n_groups]
  const TransportMode* mode_map = nullptr;  // [n_cells * n_groups]
  const double* node_r = nullptr;        // [n_cells + 1]

  double* rad_E_tally = nullptr;         // [n_cells * n_groups]
  double* E_escape = nullptr;            // [n_groups]
  double* E_numerical_loss = nullptr;    // [1]

  unsigned long long* rw_census = nullptr;
  unsigned long long* rw_leak_left = nullptr;
  unsigned long long* rw_leak_right = nullptr;
  unsigned long long* rw_escaped = nullptr;
  unsigned long long* rw_converted_to_imc = nullptr;
  unsigned long long* rw_converted_to_ddmc = nullptr;

  int n_cells = 0;
  int n_groups = 0;
  int n_rw = 0;
  int rw_start = 0;

  int bc_inner = kBoundaryReflect;
  int bc_outer = kBoundaryVacuum;

  double dt = 0.0;
  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  core::DeviceErrorFlags* error_flags = nullptr;
};

void rw_transport_1d_gpu_cuda(const RWTransportGPUInputs& in);

}  // namespace tenryu::radiation
