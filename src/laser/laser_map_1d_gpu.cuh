#pragma once

// The host part of the 1D laser mesh mapping on the device (laser_mesh.cu map_from_hydro_1d and
// compute_dynamic_mesh_params_1d, deposit_transfer.cu find_allowed_supercritical_cell_1d and the
// scalar inputs laser_step took from a hydro mirror): per cell the electron densities of the three
// conventions the host used, the critical surface, the outer radius and the finest cell near the
// critical surface, the graded node layout, the ghost corona and its anchors, the supercritical
// cell the trace may deposit in, the resonance-absorption inputs and the outermost real cell's
// ionisation. These ran on the host every step on a copy of rho, zbar, Te and the faces. The
// arithmetic repeats the host's operation by operation without floating-point contraction (the file
// is compiled with -fmad=false); the results differ only where a device log differs from the host's
// in the last place (the log-interpolated critical radii and the ghost scale length) and through
// those in the node layout.

#include <cstdint>

#include <cuda_runtime.h>

#include "core/state.hpp"

namespace tenryu::laser::laser_map_1d {

// Host inputs of one map.
struct MapInputs {
  double n_crit = 1.0;  // LaserMesh::n_crit
  // Laser.lasermesh
  double mesh_factor = 0.1;
  double rmax_n_hat_threshold = 1.0e-3;
  double r_max_factor = 2.0;
  double target_radius = 1.0;
  int nr_max = 1;
  // ghost corona (LaserMesh fields)
  int ghost_enabled = 0;
  int ghost_n_out = 0;
  double ghost_ne_min_frac = 0.03;
  double ghost_ne_max_frac = 0.99;
  double ghost_Te_min_eV = 50.0;
  double ghost_zbar_min = 1.0;
  double ghost_zbar_max = 4.0;
  double ghost_transition_resolved_nhat = 0.9;
  int ghost_transition_resolved_cells = 3;
  double material_A = 1.0;        // LaserMesh::material_A, the fallback of the A anchor
  double t_since_turn_on = 0.0;   // state.t minus the laser turn-on time, not negative
};

// Results of one map, read back once (pinned).
struct MapScalars {
  // the node layout (compute_dynamic_mesh_params_1d, with the ghost corona's outer radius)
  double R_max = 0.0;     // the last node
  double dR_fine = 0.0;
  double R_crit = 0.0;
  int nr = 0;
  int nz = 0;
  // the map (map_from_hydro_1d)
  int fcrit_cell = -1;
  int outer_surface_cell = -1;
  int ghost_configured = 0;
  int use_ghost_corona = 0;
  double outer_n_hat = 0.0;
  double ghost_fade = 0.0;
  double ghost_ne_inner = 0.0;
  double ghost_ne_min = 0.0;
  double ghost_width = 0.0;
  double ghost_scale_length = 0.0;
  double ghost_cs = 0.0;
  double r_surface_outer = 0.0;
  double r_ghost_outer = 0.0;
  double Te_anchor = 0.0;
  double zbar_anchor = 0.0;
  double n_hat_outer_minus_1 = 0.0;  // ne_raw of the cell below the outer surface (diagnostic)
  // find_allowed_supercritical_cell_1d
  int allowed_cell = -1;
  int critical_adjacent_subcritical_cell = -1;
  int fallback_only = 0;
  double r_crit_allowed = -1.0;
  // the resonance-absorption inputs of the trace (laser_step; -1 when no crossing)
  double ra_r_crit_cm = -1.0;
  double ra_ln_cm = -1.0;
  // Zbar of the outermost real cell (the Langdon collision-charge fallback); valid when >= 0 index
  int outer_zbar_cell = -1;
  double outer_zbar = 0.0;
  // non-finite inputs found in any cell (map_from_hydro_1d's checks): 1 rho, 2 zbar, 4 Te
  int nonfinite_flags = 0;
};

class Workspace {
 public:
  Workspace();
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  struct Impl;
  Impl* impl() const { return impl_; }

 private:
  Impl* impl_ = nullptr;
};

// Runs the map on the device state (rho, zbar, Te, x_r, the void mask) and the cells' A_eff (device,
// [n_cells]); returns the scalars after one synchronisation of `stream`. The node layout is left in
// the workspace (node_R / node_Z below, valid until the next map).
MapScalars map_scalars(Workspace& ws, const core::State& state, const double* A_eff,
                       const MapInputs& in, cudaStream_t stream);

// The node arrays of the last map_scalars: node_R [nr + 1], node_Z [2 nr + 1] (mirrored).
const double* node_R(const Workspace& ws);
const double* node_Z(const Workspace& ws);

// The per-cell arrays of the last map_scalars (device, [n_cells]): n_e / n_c with the conventions
// of compute_dynamic_mesh_params_1d (clamped at zero), of map_from_hydro_1d (raw) and of
// deposit_transfer.cu compute_cell_n_hat_approx; void cells hold zero.
const double* n_hat_layout(const Workspace& ws);
const double* n_hat_raw(const Workspace& ws);
const double* n_hat_approx(const Workspace& ws);

}  // namespace tenryu::laser::laser_map_1d
