#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "laser/port_geometry.hpp"
#include "laser/sector_phase_space.hpp"

namespace tenryu::laser::port_section {

struct ChiBuildInput {
  const port_geom::PortTable* ports;
  const sector_ps::PhaseSpaceTable* table;
  int n_impact_bins;
  const std::int32_t* ray_bin;
  int n_rays;
  int n_cells;
  const double* cell_chi_pref;
  const double* cell_c_a;
  const double* cell_u_r;
  const double* cell_k_bar;
  const std::uint8_t* cell_mask;
  double f_cbet;
  double alpha_iaw;
  double k_a_floor;
  int n_section_phi;
  double lambda0_nm;
};

struct ChiBuildResult {
  int G_ps;
  int n_pairs;
  long long pairs_with_seed;
  long long pairs_with_pump;
  std::vector<double> chi;
  std::vector<double> omega_state;
  std::vector<double> weight_state;
};

struct ChiDeviceAudit {
  double chi_abs_max;
  long long chi_nonzero;
};

struct ChiDeviceCellFields {  // optional device-side cell-field inputs
  const double* d_chi_pref = nullptr;
  const double* d_c_a = nullptr;
  const double* d_u_r = nullptr;
  const double* d_k_bar = nullptr;
  const std::uint8_t* d_mask = nullptr;  // all five set, or all five null
};

class ChiDeviceWorkspace {
 public:
  ChiDeviceWorkspace();
  ~ChiDeviceWorkspace();
  ChiDeviceWorkspace(const ChiDeviceWorkspace&) = delete;
  ChiDeviceWorkspace& operator=(const ChiDeviceWorkspace&) = delete;
  struct Impl;
  Impl* impl() const { return impl_.get(); }
 private:
  std::unique_ptr<Impl> impl_;
};

struct ChiBuildDeviceView {
  int G_ps;
  int n_pairs;
  long long pairs_with_seed;
  long long pairs_with_pump;
  const double* d_chi;        // device ptr [n_cells*n_pairs]; owned by ws;
                              // valid until the next build on the same ws
  ChiDeviceAudit audit;
  const double* omega_state;  // host ptr into ws cache [G_ps]
  const double* weight_state; // host ptr into ws cache [G_ps]
};

ChiBuildResult build_chi_ps(const ChiBuildInput& input);

// CUDA implementation of build_chi_ps. Bitwise-deterministic run-to-run
// (no floating-point atomics); matches the host reference to transcendental
// ULP differences. Requires an initialized CUDA context.
ChiBuildResult build_chi_ps_device(const ChiBuildInput& input);

// Persistent-workspace chi build. cuda_stream is a cudaStream_t passed as
// void* (nullptr = default stream). All device work is enqueued on that
// stream; the only synchronization is one cudaStreamSynchronize for the
// 32-byte counters/audit readback at the end. When dev_fields supplies all
// five device pointers the host cell-field pointers in `input` may be null
// and no cell-field H2D occurs.
ChiBuildDeviceView build_chi_ps_device_ws(const ChiBuildInput& input,
                                          const ChiDeviceCellFields& dev_fields,
                                          ChiDeviceWorkspace& ws,
                                          void* cuda_stream);

}  // namespace tenryu::laser::port_section
