#pragma once

#include <string>

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro::plic {

struct NormalEstimatorResult {
  double n_r = 0.0;
  double n_z = 0.0;
  double alpha = 0.0;
  double residual_l2 = 0.0;
  bool ill_conditioned = true;
  bool valid = false;
  bool prev_normal_invalidated = false;
  std::string fallback_used;
  double grad_F_magnitude = 0.0;
};

struct Case2FallbackResult {
  NormalEstimatorResult normal;
  bool reconstructed = false;
  bool degradation_event_required = false;
};

NormalEstimatorResult estimate_normal_youngs_seeded_lvira(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    int cell_idx,
    int material_idx,
    double prev_normal_r,
    double prev_normal_z,
    bool prev_normal_valid);

NormalEstimatorResult estimate_normal_for_test(
    double cell_r0,
    double cell_z0,
    double cell_r1,
    double cell_z1,
    double cell_r2,
    double cell_z2,
    double cell_r3,
    double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_r_centroid,
    const double* neighbor_z_centroid,
    const double* neighbor_v_cell,
    int target_i,
    int target_j,
    int stencil_size,
    const std::string& mode);

NormalEstimatorResult largest_face_jump_normal_for_test(
    double cell_r0,
    double cell_z0,
    double cell_r1,
    double cell_z1,
    double cell_r2,
    double cell_z2,
    double cell_r3,
    double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_v_cell,
    int target_i,
    int target_j,
    int stencil_size);

Case2FallbackResult resolve_case2_fallback_for_test(
    double cell_r0,
    double cell_z0,
    double cell_r1,
    double cell_z1,
    double cell_r2,
    double cell_z2,
    double cell_r3,
    double cell_z3,
    const double* neighbor_volfrac_grid,
    const double* neighbor_r_centroid,
    const double* neighbor_z_centroid,
    const double* neighbor_v_cell,
    int target_i,
    int target_j,
    int stencil_size,
    double prev_normal_r,
    double prev_normal_z,
    bool prev_normal_valid,
    double cell_volfrac_now,
    double cell_volfrac_at_last_plic,
    double freshness_volfrac_threshold);

double compute_eta_E_per_cell(const double* face_mass_flux,
                              const double* face_e_closure,
                              double e_cell,
                              double e_floor,
                              double E_floor,
                              int n_faces,
                              int n_mat);

}  // namespace tenryu::hydro::plic
