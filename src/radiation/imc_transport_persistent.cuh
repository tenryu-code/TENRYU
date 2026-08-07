#pragma once

#include <cstdint>

#include "core/device_error_flags.cuh"
#include "radiation/boundary.cuh"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::parallel {
struct PartitionInfo;
}  // namespace tenryu::parallel

namespace tenryu::radiation {

struct TransportInputs {
  PhotonPool* pool = nullptr;

  const double* sigma_a_eff = nullptr;  // [n_cells * G]
  const double* sigma_s_eff = nullptr;  // [n_cells * G]
  const double* Te = nullptr;           // [n_cells]
  const double* vol = nullptr;          // [n_cells]
  double* sloc_abs_wr = nullptr;        // [n_cells], energy-weighted absorption radius accumulator
  double* sloc_abs_wr2 = nullptr;       // [n_cells], energy-weighted absorption radius^2 accumulator
  double* sloc_abs_E = nullptr;         // [n_cells], absorbed energy accumulator
  const double* node_r = nullptr;       // [n_nodes_r]
  const double* node_z = nullptr;       // [n_nodes_z] (2D_RZ only)
  int nr = 0;
  int nz = 1;
  int mesh_dim = 1;
  const TransportMode* ddmc_mode = nullptr;  // [n_cells * G]
  bool ddmc_zero_flux_interfaces = false;
  int n_groups_for_mode = 1;
  bool emissivity_preserving = true;
  const double* sigma_R = nullptr;       // [n_cells * G]; group-dependent Rosseland opacity.
                                         // For uniform Rosseland, all groups have the same value.
                                         // Index: cell * n_groups + group.
  const double* cell_dx = nullptr;       // [n_cells]
  const double* fleck_f = nullptr;       // [n_cells]
  const double* sigma_a = nullptr;       // [n_cells * G]
  const double* eta_cdf = nullptr;       // [n_cells * G], non-null in NLTE mode
  const double* eta_cdf_hydro = nullptr; // [n_hydro * G], RadLite NLTE donor spectra
  const double* hydro_node_r = nullptr;  // [n_hydro + 1], RadLite hydro nodes for donor lookup
  const int32_t* rad_h_begin = nullptr;  // [n_cells], RadLite member hydro begin
  const int32_t* rad_h_end = nullptr;    // [n_cells], RadLite member hydro end
  const int* g_diff_end = nullptr;       // [n_cells], exclusive PGRW diffusive group cutoff
  const double* sigma_a_bar = nullptr;   // [n_cells], PGRW collapsed absorption
  const double* sigma_t_bar = nullptr;   // [n_cells], PGRW collapsed total opacity
  const double* D_pgrw = nullptr;        // [n_cells], PGRW diffusion coefficient
  const double* gamma_pgrw = nullptr;    // [n_cells], PGRW diffusive emission fraction
  const double* pgrw_leak_inv_cdf = nullptr; // [N] theta values for inverse leak CDF
  const double* pgrw_leak_cdf_xi = nullptr;  // [N] xi values paired with theta table
  const double* pgrw_pos_cdf = nullptr;      // [N_theta * N_rho] position CDF table
  int pgrw_leak_table_size = 0;
  int pgrw_pos_theta_bins = 0;
  int pgrw_pos_rho_bins = 0;
  double pgrw_theta_max = 3.0;
  double pgrw_tau_rw = 0.0;  // 0 = disabled
  const double* scatter_bias_cdf = nullptr;  // [n_cells * G], biased CDF for scatter group resampling

  double* rad_dep = nullptr;            // [n_cells * G]
  double* rad_E_tally = nullptr;        // [n_cells * G]
  double* holo_Prr_tally = nullptr;     // [n_cells * G], optional HOLO passive closure tally
  double* holo_Prr_coverage_tally = nullptr; // [n_cells * G], optional covered track length
  double* face_current_step = nullptr;  // [(n_cells + 1) * G], signed crossing energy [erg]
  const std::uint8_t* diff_cell = nullptr;       // [n_cells], deterministic diffusion mask
  double* diff_face_current_in = nullptr;        // [(n_cells + 1) * G], IMC->diffusion energy [erg]
  double* E_escape = nullptr;           // [G]
  double* E_numerical_loss = nullptr;   // [1]
  unsigned long long* imc_absorbed = nullptr;
  unsigned long long* imc_escaped = nullptr;
  unsigned long long* diffusion_interface_kills = nullptr;
  unsigned long long* interface_transitions = nullptr;
  unsigned long long* interface_reflections = nullptr;
  unsigned long long* conversion_prob_violations = nullptr;
  unsigned long long* cnt_boundary = nullptr;
  unsigned long long* cnt_scatter = nullptr;
  unsigned long long* cnt_census = nullptr;
  unsigned long long* cnt_absorb_kill = nullptr;
  unsigned long long* cnt_absorb_survive = nullptr;
  unsigned long long* cnt_roulette_kill = nullptr;

  int n_cells = 0;
  int n_groups = 1;
  int n_imc = 0;
  double dt = 0.0;
  PlanckTableDeviceView planck{};
  bool inelastic_scatter = true;

  double E_avg = 0.0;
  double w_cutoff = 1.0e-10;
  double p_survival = 0.1;
  double f_cutoff = 0.0;
  bool tail_pass = false;

  int bc_inner = kBoundaryReflect;
  int bc_outer = kBoundaryVacuum;
  int bc_bottom_z = kBoundaryVacuum;
  int bc_top_z = kBoundaryVacuum;

  std::uint64_t step_number = 0;
  std::uint64_t user_seed = 0;

  core::DeviceErrorFlags* error_flags = nullptr;
};

void imc_transport_persistent_cuda(const TransportInputs& in);
void imc_transport_persistent_cuda(const TransportInputs& in,
                                   const tenryu::parallel::PartitionInfo& part);

}  // namespace tenryu::radiation
