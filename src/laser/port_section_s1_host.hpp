#pragma once

// Host reference of the port_section phase-space table build: the staging of the CBET ray records
// that laser_step ran on the host every step before the device build (port_section_s1_gpu.cuh)
// replaced it — sector_adapter::build_ray_paths and sector_ps::build_table on host copies of the
// records. Kept for the device-versus-host unit tests; the production step does not call it.

#include <cstdint>
#include <vector>

#include "laser/port_section_s1_gpu.cuh"
#include "laser/sector_phase_space.hpp"

namespace tenryu::laser::port_section {

struct S1HostReference {
  int n_paths = 0;                       // rays with a geometric path
  std::vector<std::int32_t> ray_bin;     // [n_paths]
  sector_ps::FlatTable flat;             // the table, flattened
  sector_ps::ExclusionLedger ledger{};
  long long total_nodes = 0;
  long long total_crossings = 0;
  double bouguer_drift_max = 0.0;
  // The snapshot intensity map of the table (former laser.cu fill_port_section_ray_map), in the
  // layout of build_ray_map_device: [(shell * kRayMapThetaBins + bin) * 2 + sheet], W/cm^2.
  std::vector<double> ray_map;
};

// Copies the device inputs to the host and builds the table there (synchronous).
S1HostReference build_s1_table_host_reference(const S1DeviceInput& input);

}  // namespace tenryu::laser::port_section
