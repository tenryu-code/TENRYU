#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::radiation {

// Host-side structure bundling all per-cell radiation coefficients for one step.
// Coefficients are generated once and shared by source/transport/coupling paths.
// This struct is intentionally an open aggregate for compute_nlte_coefficients() to fill
// in a single pass. Consumers should treat all fields as read-only after
// compute_nlte_coefficients() returns. Phase-2 may add validation methods.
struct CellRadiationCoeffs {
  int n_cells = 0;
  int n_groups = 0;
  std::vector<double> rho_eval;     // [n_cells], density used for coefficient evaluation
  std::vector<double> Te_eval;      // [n_cells], electron temperature used for coefficient evaluation
  std::vector<double> cv_e;         // [n_cells], volumetric electron heat capacity used in beta
  std::vector<double> beta;         // [n_cells]
  std::vector<double> f;            // [n_cells]
  std::vector<double> sigma_p_abs;  // [n_cells]
  std::vector<double> sigma_p_em;   // [n_cells]
  std::vector<double> gamma_diag;   // [n_cells]
  std::vector<double> sigma_pa;     // [n_cells * n_groups]
  std::vector<double> sigma_pe;     // [n_cells * n_groups]
  std::vector<double> sigma_R;      // [n_cells * n_groups]
  std::vector<double> sigma_a_eff;  // [n_cells * n_groups]
  std::vector<double> sigma_s_eff;  // [n_cells * n_groups]
  std::vector<double> b;            // [n_cells * n_groups]
  std::vector<double> s;            // [n_cells * n_groups]
  std::vector<double> J;            // [n_cells * n_groups]
  std::vector<double> eta;          // [n_cells * n_groups]
  std::vector<double> eta_tot;      // [n_cells]
  std::vector<double> eta_cdf;      // [n_cells * n_groups]
  std::vector<std::uint8_t> used_planck_fallback;    // [n_cells * n_groups]
  std::vector<std::uint8_t> clamped_negative_alpha;  // [n_cells * n_groups]
  std::vector<std::uint8_t> clamped_negative_eta;    // [n_cells * n_groups]
  int planck_fallback_cell_count = 0;
  int planck_fallback_group_count = 0;
  int negative_alpha_clamp_count = 0;
  int negative_eta_clamp_count = 0;
  int nan_inf_count = 0;
  double min_f = 1.0;
  double max_f = 1.0;
  double mean_f = 1.0;
  double min_sigma_p_abs = 0.0;
  double max_sigma_p_abs = 0.0;
  double min_sigma_p_em = 0.0;
  double max_sigma_p_em = 0.0;
  double min_gamma_diag = 0.0;
  double max_gamma_diag = 0.0;
  bool is_nlte = false;
};

}  // namespace tenryu::radiation
