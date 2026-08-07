#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
struct ConductionResult;
}

namespace tenryu::hydro::snb2d {

// Full SNB Picard conduction super-step on the 2D_RZ STS path. Overwrites
// state.Te. Fills the snb_* fields of `result`. The T-stencil d_stencil is
// the OFF-path Kershaw stencil already built by the caller (byte-shared with
// the OFF apply); d_kappa_eff is the same mfp-capped Spitzer cell array the
// stencil was built from (used only for the theta cap's flux reconstruction).
// The clamp pack is reset every Picard iterate so it reports the accepted
// (final) solve only. Design: docs/design/2d_snb_port_spec.md §3.
void conduction_step_2d_sts_snb(core::State& state,
                                const core::Config& cfg,
                                ConductionResult& result,
                                const double* d_stencil,
                                const double* d_kappa_eff,
                                const double* d_rho_cv_e,
                                const std::uint8_t* d_cell_is_void,
                                const double* d_A_eff,
                                int nr,
                                int nz,
                                const std::vector<double>& tau,
                                int n_sub,
                                double te_floor,
                                int* d_clamp_count,
                                double* d_e_floor);

// Host helper: group reduced-energy edges (beta-hat), geometric ladder
// {0, 0.1, ..., E_max_over_Te}. Exposed for the verify drivers (the G2/G3
// discrete-analytic reference must use the same edges). Identical to the 1D
// module's ladder.
std::vector<double> group_edges_beta(int n_groups, double e_max_over_te);

// Verify-only probe: one coefficient pass + H solve on the CURRENT state.Te
// WITHOUT advancing the temperature. Host copies of the pair-power fields
// (per cell-slot, slot order kE..kSW at [c*8 + (slot-1)]), theta, and solver
// stats. The measurement backbone of the SNB-2D gates.
struct Snb2dProbe {
  std::vector<double> p_sh;     // [n*8] 4pi*G*(Te_n - Te_c) on the T-stencil
  std::vector<double> p_dq;     // [n*8] frozen nonlocal pair correction
  std::vector<double> theta;    // [n] outer-cap multiplier (1 when uncapped)
  std::vector<double> lam_tr;   // [G*n]
  std::vector<double> lam_abs;  // [G*n]
  std::vector<double> edges_erg;  // [G+1]
  double t_ref_ev = 0.0;
  int cg_iters = 0;
  double cg_resid = 0.0;   // final relative ||r||/||b||
  bool cg_converged = true;
};

// Low-level probe pass: caller supplies the OFF preamble products (T-stencil,
// kappa, rho_cv, void mask, A_eff). Defined in conduction_snb_2d.cu.
Snb2dProbe snb2d_probe_pass(core::State& state,
                            const core::Config& cfg,
                            const double* d_stencil,
                            const double* d_kappa_eff,
                            const double* d_rho_cv_e,
                            const std::uint8_t* d_cell_is_void,
                            const double* d_A_eff,
                            int nr,
                            int nz);

// High-level probe: runs the conduction_step_2d_sts preamble (kappa/rho_cv
// diagnostics, void upload, Kershaw T-stencil build, A_eff) and then the
// probe pass. DEFINED IN conduction.cu (the preamble helpers live in its
// anonymous namespace).
Snb2dProbe snb2d_probe(core::State& state, const core::Config& cfg);

}  // namespace tenryu::hydro::snb2d
