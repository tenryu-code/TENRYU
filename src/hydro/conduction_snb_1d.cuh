#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
struct ConductionResult;
}

namespace tenryu::hydro::snb {

__host__ __device__ double snb_lambda_g_ext(double eps_erg,
                                            double n_e,
                                            double zbar,
                                            double zmom_r2,
                                            double ln_lambda,
                                            int mfp_variant);

// Device workspace for one SNB conduction step (pooled scratch, acquired by
// conduction.cu's caller-side helper or internally; see conduction_snb_1d.cu).
struct SnbDeviceWork {
  double* edges = nullptr;      // [n_groups+1] group energy edges [erg]
  double* lam_tr = nullptr;     // [n_groups*n_cells] transport mfp [cm]
  double* lam_abs = nullptr;    // [n_groups*n_cells] absorption mfp [cm]
  double* lower = nullptr;      // [n_groups*n_cells]
  double* diag = nullptr;       // [n_groups*n_cells]
  double* upper = nullptr;      // [n_groups*n_cells]
  double* rhs = nullptr;        // [n_groups*n_cells] (H_g after solve)
  double* qsh_face = nullptr;   // [n_cells+1] physical SH face flux
  double* dq_face = nullptr;    // [n_cells+1] physical nonlocal correction
  double* dq_face_prev = nullptr;  // [n_cells+1]
  double* theta_face = nullptr;    // [n_cells+1] outer cap multiplier
  double* te_stash = nullptr;   // [n_cells]
  double* scalars = nullptr;    // [8] reduction pack (see kScalar* indices)
  int* counters = nullptr;      // [2] cap-firing counters
};

// Fills work_scalars[0] with max sanitized non-void Te [eV] (device reduction).
void snb_fill_t_ref(core::State& state, const std::uint8_t* d_cell_is_void,
                    int n_cells, double* work_scalars);

// Fills work.qsh_face/dq_face/theta_face for the CURRENT state.Te without
// advancing the temperature (one coefficient pass + H-solve). Returns the
// max-norm change vs dq_face_prev in *resid_out and the |dq|/|qsh| scale in
// *dq_ratio_out. Counters/theta stats are left in work.counters/work.scalars.
void snb_pass(core::State& state,
              const core::Config& cfg,
              const SnbDeviceWork& work,
              const double* d_kappa_eff,
              const double* d_A_eff,
              const std::uint8_t* d_cell_is_void,
              int n_cells,
              int geom_code,
              bool face_policy_kirchhoff,
              bool use_outer_cap,
              double t_ref_ev,
              double* resid_out,
              double* qsh_scale_out,
              double* dq_ratio_out,
              const double* zmom_r2);

// Full SNB Picard conduction super-step on the STS path. Overwrites state.Te.
// Fills the snb_* fields of `result` (declared in conduction.cuh). The clamp
// pack (d_clamp_count/d_e_floor) is reset at the start of every Picard iterate
// so it reports the accepted (final) solve only.
void conduction_step_1d_sts_snb(core::State& state,
                                const core::Config& cfg,
                                ConductionResult& result,
                                const double* d_kappa_eff,
                                const double* d_rho_cv_e,
                                const std::uint8_t* d_cell_is_void,
                                const double* d_A_eff,
                                int n_cells,
                                int geom_code,
                                bool face_policy_kirchhoff,
                                bool use_outer_cap,
                                const std::vector<double>& tau,
                                int n_sub,
                                double te_floor,
                                int* d_clamp_count,
                                double* d_e_floor,
                                const double* zmom_r2);

// Host helper: group reduced-energy edges (beta-hat), geometric ladder
// {0, 0.1, ..., E_max_over_Te} with n_groups interior spans. Exposed for the
// verify drivers (G2/G3 discrete-analytic reference must use the same edges).
std::vector<double> group_edges_beta(int n_groups, double e_max_over_te);

}  // namespace tenryu::hydro::snb
