#pragma once

// Device build of the port_section phase-space table (NUMERICS §5.10.8): the reference-beam ray
// paths of sector_adapter::build_ray_paths and the shell crossings of sector_ps::build_table,
// computed on the GPU from the CBET workspace's ray records, so that no ray record, path or
// crossing goes through the host during a step. The host functions remain the reference the unit
// tests compare against; the arithmetic follows them operation by operation (this file is compiled
// without floating-point contraction), so the results differ only by the last-place differences
// of the device transcendental functions (asin, sin, cos, expm1).

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "laser/sector_phase_space.hpp"

namespace tenryu::laser::port_section {

// Device views the build reads. Records are stored per ray in stripes of cap_per_ray entries
// (CbetWorkspace layout): record k of ray s is at index s * cap_per_ray + k for k < rec_count[s].
struct S1DeviceInput {
  int n_rays = 0;
  int cap_per_ray = 0;
  int n_cells = 0;
  int n_bins = 0;                              // CBET impact bins: ray bin = group_base % n_bins
  const std::int32_t* rec_count = nullptr;     // [n_rays]
  const std::int32_t* rec_cell = nullptr;      // [n_rays * cap_per_ray]
  const float* rec_mu = nullptr;               // [n_rays * cap_per_ray]
  const double* rec_ds = nullptr;              // [n_rays * cap_per_ray]
  const double* rec_S = nullptr;               // [n_rays * cap_per_ray] incident optical depth
  const double* ray_P0 = nullptr;              // [n_rays]
  const std::int32_t* ray_group_base = nullptr;  // [n_rays]
  const double* r_edges = nullptr;             // [n_cells + 1] hydro node radii
  const double* rho = nullptr;                 // [n_cells]
  const double* zbar = nullptr;                // [n_cells]
  const double* A_eff = nullptr;               // [n_cells]
  double n_crit = 0.0;                         // critical electron density [cm^-3]
  double impact_spacing = 0.0;                 // b_ray = impact_spacing * (ray + 0.5) [cm]
  // Caustic detection threshold (sector_ps::PhaseSpaceParams::caustic_area_rel_tol).
  double caustic_area_rel_tol = 1.0e-3;
};

// The table as flat device arrays (sector_ps::FlatTable layout: CSR over the 2 * n_shells bins
// ordered (shell, sheet), each bin sorted by (theta, ray_index)). Pointers stay valid until the
// next build on the same workspace.
struct S1DeviceTable {
  int n_shells = 0;     // n_cells + 1
  int n_paths = 0;      // rays with a geometric path (the path index is the crossing ray_index)
  const int* offsets = nullptr;          // [2 * n_shells + 1]
  const double* theta = nullptr;         // [n_crossings]
  const double* alpha = nullptr;
  const double* power = nullptr;
  const double* area = nullptr;
  const std::int32_t* ray_index = nullptr;
  const std::uint8_t* in_limiter = nullptr;
  const std::int32_t* ray_bin = nullptr;  // [n_paths]
  const double* shell_r = nullptr;        // [n_shells] (the hydro node radii)
};

// Scalars of the build read back once per step (the S1 audit and the exclusion ledger).
struct S1DeviceSummary {
  int n_paths = 0;
  int n_excluded_rays = 0;
  long long total_nodes = 0;
  long long total_crossings = 0;
  double excluded_power_fraction = 0.0;
  double bouguer_drift_max = 0.0;
  int error_flags = 0;  // bit 0: a ray bin out of range; bit 1: a record cell out of range
};

class S1DeviceWorkspace {
 public:
  S1DeviceWorkspace();
  ~S1DeviceWorkspace();
  S1DeviceWorkspace(const S1DeviceWorkspace&) = delete;
  S1DeviceWorkspace& operator=(const S1DeviceWorkspace&) = delete;
  struct Impl;
  Impl* impl() const { return impl_; }

 private:
  Impl* impl_ = nullptr;
};

// Builds the ray paths and the phase-space table on the device (all work enqueued on `stream`),
// then reads back the summary (one small copy and one stream synchronization).
S1DeviceSummary build_s1_table_device(const S1DeviceInput& input, S1DeviceWorkspace& workspace,
                                      cudaStream_t stream, S1DeviceTable* table_out);

// Intensity map of the table for the snapshot (S1HostReference::ray_map on the host): for every
// (shell, sheet) the power-weighted mean intensity of the crossings in 64 theta bins over [0, pi],
// in W/cm^2, as map[(shell * 64 + bin) * 2 + sheet]. Device output [n_shells * 64 * 2].
void build_ray_map_device(const S1DeviceTable& table, double* map_out, cudaStream_t stream);

constexpr int kRayMapThetaBins = 64;

// A host table holding only the two bins (sheets) of one shell of a device table, for the host
// functions that read a single shell (port_section::illumination_at_shell and common_wave_drive,
// the references of the device hot-electron model inputs in the tests): a copy of that shell's
// crossings and of the shell radii, synchronous on `stream`.
sector_ps::PhaseSpaceTable host_table_for_shell(const S1DeviceTable& table, int shell,
                                                cudaStream_t stream);

}  // namespace tenryu::laser::port_section
