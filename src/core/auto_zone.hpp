#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tenryu::core {

struct AutoZoneRegion {
  double r_end = 0.0;          // outer radius [cm]
  int nz = 0;                  // number of zones
  double rho_ref = 0.0;        // reference density [g/cc]
  bool is_void = false;        // void region flag
  std::string material_group;  // group name for same-material regions
};

struct AutoZoneConfig {
  // W-G: 1D coordinate geometry (0 spherical / 1 cylindrical / 2 planar).
  int geometry_code = 0;
  double mass_ratio_max = 1.3;
  int n_bridge_min = 2;
  int n_bridge_max = 10;
  double bridge_frac_max = 0.25;
  double rho_void_cut = 1.0e-6;    // [g/cc]
  double dr_min = 1.0e-8;          // [cm]
  double mass_ratio_hard_max = 2.0;
  int max_iter = 100;
  double bulk_mass_tol = 1.0e-3;
};

struct AutoZoneDiagnostics {
  double mass_ratio_min = 1.0;
  double mass_ratio_max = 1.0;
  double mass_ratio_mean = 1.0;
  double dr_min_actual = 0.0;
  int n_ratio_violations = 0;
  std::vector<std::string> warnings;
};

// Main entry point: compute node positions from regions + config.
// r_min is the inner boundary of the first region.
// Returns node array of size sum(nz)+1.
std::vector<double> compute_auto_zone_nodes(
    double r_min,
    const std::vector<AutoZoneRegion>& regions,
    const AutoZoneConfig& cfg,
    AutoZoneDiagnostics* diag = nullptr);

}  // namespace tenryu::core
