#pragma once

#include <cstdint>
#include <vector>

#include "laser/sector_phase_space.hpp"

namespace tenryu::laser {

struct LaserMesh;

namespace sector_adapter {

using IncidentAttenuationFn =
    double (*)(double power, std::int64_t record_index, const void* context);

struct AdapterInput {
  int n_rays;
  const std::int64_t* ray_rec_offset;
  const std::int32_t* rec_cell;
  const float* rec_mu;
  const double* rec_ds;
  const double* ray_P0;
  const double* ray_impact_parameter;
  const double* cell_r_center;
  const double* r_edges;  // [n_cells + 1]
  const double* cell_eps;
  int n_cells;

  // Applies one record's incident-pass fractional attenuation to power.
  // The callback is invoked in ascending ray and path-record order.
  IncidentAttenuationFn attenuate;
  const void* attenuation_context;
};

// Record-boundary convention: record nodes lie on cell edges selected by the
// path direction encoded by the adjacent record-cell sequence. Node alpha is
// taken from the record that terminates at that node; the path-start node uses
// the first record, with no averaging across a turning point. For record k,
// mu_mid = (mu_k + mu_{k+1})/2 and r_mid = (r_k + r_{k+1})/2, and
//
//   dtheta_k = ds_k/6 * (sqrt(1-mu_k^2)/r_k
//                        + 4*sqrt(1-mu_mid^2)/r_mid
//                        + sqrt(1-mu_{k+1}^2)/r_{k+1}).
//
// Zero-length terminal attenuation records update P at the final geometric
// node without supplying a synthetic direction or radius.
// theta_0 = asin(min(1, b/r_0)). For dtheta_eff equal to the same neighboring-
// ray angular half-width (one-sided at the ends), bundle area uses
// sin_eff = max(sin(theta), sin(dtheta_eff/2)), with dtheta_eff >= 1e-12 and
// cos(alpha) >= 1e-6.
// source_ray_indices, when non-null, is filled parallel to the returned paths.
std::vector<sector_ps::RayPath> build_ray_paths(
    const AdapterInput& input,
    std::vector<int>* source_ray_indices = nullptr);

// record-granularity audit: B is reconstructed from per-record mu and
// cell-edge radii, so the drift floor is O(per-cell delta ln eps) (~0.3 on
// this coarse fixture); this gate catches catastrophic breakage only —
// precision Bouguer validation lives in test_sector_phase_space (1e-10,
// analytic alpha).
struct S1Audit {
  int n_rays;
  long long total_nodes;
  long long total_crossings;
  double excluded_frac;
  double bouguer_drift_max;
  double chi_abs_max;
  long long chi_nonzero;
  long long pairs_with_seed;
  long long pairs_with_pump;
  double dq_abs_max;
  double dq_abs_sum;
  double L_ps_abs_max;
};

const S1Audit* last_s1_audit(const LaserMesh& mesh);

}  // namespace sector_adapter
}  // namespace tenryu::laser
