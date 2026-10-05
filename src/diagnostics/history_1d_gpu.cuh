#pragma once

// The 1D history row's per-cell reductions on the device (driver.cpp emit_due_outputs, every
// history step): the laser energy of the step and its absorption-weighted radius
// (compute_laser_pattern), the mass-weighted mean and the maximum Zbar
// (HistoryWriter::compute_plasma_history_diagnostics), the peak density, the mass-weighted shell
// radius and the central electron temperature (compute_implosion_history_diagnostics), and the
// areal density and shell radius of the shape history (compute_shape_history_1d). These read
// copies of the fields on the host every step. The maxima, minima and the central cell are exact;
// the areal density's double sum is made in the cells' order by one thread, bit for bit; the sums
// the host made in long double are double-double sums here, which can differ in the last place
// (infinite and NaN entries give inf and NaN as on the host).

#include <cuda_runtime.h>

#include "core/state.hpp"

namespace tenryu::diagnostics::history_1d {

// Σ laser_dep over the cells (compute_laser_pattern's absorbed energy).
double laser_dep_sum(const core::State& state);

// Σ q r_c / Σ q over the cells with laser_dep q > 0, r_c the midpoint of the cell's nodes; 0 when
// no cell has q > 0.
double absorption_weighted_r(const core::State& state);

struct Plasma {
  bool valid = false;     // the weights sum to more than zero
  double zbar_mean = 0.0; // Σ w max(Zbar, 0) / Σ w, w = max(mass, 0) (1 without masses)
  double zbar_max = 0.0;  // max over the cells of max(Zbar, 0)
};
Plasma plasma(const core::State& state);

struct Implosion {
  double rho_peak = 0.0;            // std::max_element of rho
  bool shell_valid = false;         // the shell cells' weights sum to more than zero
  double shell_radius_mean = 0.0;   // Σ w r / Σ w over rho >= 0.1 rho_peak, r = max(centroid r, 0)
  bool has_center = false;
  double center_temperature = 0.0;  // Te of the first cell of least centroid r^2
};
// centroid_r: the mesh's device cell centroids (Mesh::cell_centroid_r_device), nullptr when the
// mesh has none (no shell sums, no central cell).
Implosion implosion(const core::State& state, const double* centroid_r);

struct Shape {
  double rho_max = 0.0;        // std::max_element of rho
  double rhoR = 0.0;           // Σ rho dr over the cells (dr = max(r_hi - r_lo, 0) > 0)
  double rhoR_hotspot = 0.0;   // the same weighted by the clamped gas tracer
  double shell_radius = 0.0;   // max r_hi over rho >= max(rho_threshold, 0.1 rho_max)
};
// shell_only: the areal density over the cells with rho >= 0.1 rho_max only; gas_tracer: device
// [n] or nullptr; rho_threshold: the sphericity diagnostic's threshold.
Shape shape(const core::State& state, bool shell_only, const double* gas_tracer,
            double rho_threshold);

}  // namespace tenryu::diagnostics::history_1d
