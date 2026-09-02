#pragma once

#include <string>

namespace tenryu::hydro::reale {

struct RemapMesh {
  int n_cells = 0;
  int corner_stride = 0;
  const int* csr_offsets = nullptr;
  const int* csr_indices = nullptr;
  const unsigned char* nverts = nullptr;
  const double* node_r = nullptr;
  const double* node_z = nullptr;
  // Optional target Voronoi data; null pointers mean polygon-edge clipping.
  const double* generator_r = nullptr;  // per cell
  const double* generator_z = nullptr;  // per cell
  const int* neighbor_index = nullptr;  // per corner slot, -1 = boundary
};

struct RemapFields {
  const double* mass = nullptr;
  const double* ee = nullptr;
  const double* ei = nullptr;
  const double* corner_mass = nullptr;
  const double* node_vr = nullptr;
  const double* node_vz = nullptr;
};

struct RemapResult {
  bool ok = false;
  std::string reject_reason;
  double mass_residual = 0.0;
  double momentum_residual_r = 0.0;
  double momentum_residual_z = 0.0;
  double energy_residual = 0.0;
  double q_total = 0.0;
};

RemapResult remap_conservative(
    const RemapMesh& old_mesh,
    const RemapMesh& new_mesh,
    const RemapFields& old_fields,
    double* new_mass,
    double* new_ee,
    double* new_ei,
    double* new_corner_mass,
    double* new_node_vr,
    double* new_node_vz,
    const bool target_is_exact_voronoi,
    const int dvclp_solver_rev = 0,
    const double reale_overlay_additivity_tol = 1.0e-4);

}  // namespace tenryu::hydro::reale
