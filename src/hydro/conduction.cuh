#pragma once

#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"

namespace tenryu::hydro {

struct HydroEOSContext;

struct ConductionResult {
  double dt_cond = 0.0;          // [s] conduction dt-control limit
  double dt_exp = 0.0;           // [s] explicit conduction stability limit
  double deff_min = 0.0;         // [cm^2/s] min effective diffusivity
  double deff_max = 0.0;         // [cm^2/s] max effective diffusivity
  int sts_stages = 0;            // STS stage count used for this step
  int sts_subcycles = 0;         // Number of STS subcycles used for this step
  int clamp_count = 0;           // floor clamp applications in this step
  double E_floor_injected = 0.0; // [erg] energy injected by floor clamps
  int mmatrix_fix_count = 0;     // Kershaw positive off-diagonal repairs
  double solver_residual = 0.0;  // relative ||Ax-b||_2/||b||_2 for linear solves
  int solver_iterations = 0;     // direct solve = 1; STS path = applied stages
  double solver_cond_number_est = 0.0; // diagonal-ratio estimate
  // SNB nonlocal conduction diagnostics (nonlocal_model="snb" only; defaults
  // inert). 2D pairs are counted where 1D counted faces (names kept for
  // cross-dimension consistency; see docs/design/2d_snb_port_spec.md §5).
  int snb_picard_iters = 0;
  bool snb_converged = true;
  double snb_picard_resid = 0.0;
  int snb_cap_faces_99 = 0;        // pairs with outer-cap theta < 0.99
  int snb_cap_faces_50 = 0;        // pairs with outer-cap theta < 0.5
  double snb_cap_theta_min = 1.0;
  double snb_dq_over_qsh_max = 0.0;
  int snb_solver_iters = 0;        // 2D-only: CG iterations of the last pass
  double snb_solver_resid = 0.0;   // 2D-only: final relative CG residual
  int snb_ceiling_clamp_count = 0;      // 1D max-principle ceiling hits (accepted iterate)
  double snb_E_ceiling_removed = 0.0;   // [erg] energy removed by the ceiling (accepted iterate)
  // STS liveness bound (sts_total_stages_max): when tripped, the
  // application returns early WITHOUT touching state and requests a
  // driver full-step retry.
  bool retry_required = false;
  const char* retry_reason = nullptr;   // static string, e.g. "sts_total_stages_overflow"
  long long retry_total_stages = 0;
  int retry_n_sub = 0;
  double retry_dt_exp = 0.0;
};

struct ConductionDiagnostics {
  double dt_exp = 0.0;      // [s] explicit conduction stability limit
  double dt_cond = 0.0;     // [s] conduction dt-control limit
  double deff_min = 0.0;    // [cm^2/s] min effective diffusivity
  double deff_max = 0.0;    // [cm^2/s] max effective diffusivity
  int limiting_cell = -1;
  double dl_at_limiting = 0.0;
  double deff_at_limiting = 0.0;
};

struct ConductionDtArgmin {
  double dt = std::numeric_limits<double>::infinity();
  int argmin_cell = -1;
  double deff_at_argmin = 0.0;
  double dl_at_argmin = 0.0;
};

// Electron heat conduction full step (Strang splitting C(dt)).
// `stream == nullptr` means CUDA default stream (stream 0), which is valid.
ConductionResult conduction_step(
    core::State& state, double dt, const core::Config& cfg,
    const parallel::PartitionInfo& part = parallel::PartitionInfo{},
    parallel::CommBuffers* bufs = nullptr, cudaStream_t stream = nullptr,
    const HydroEOSContext* eos_ctx = nullptr);

// STS conduction stability limit used by global dt control.
double compute_dt_conduction(const core::State& state,
                             const core::Config& cfg,
                             const HydroEOSContext* eos_ctx = nullptr);

ConductionDtArgmin compute_dt_conduction_argmin(const core::State& state,
                                                const core::Config& cfg,
                                                const HydroEOSContext* eos_ctx = nullptr);

// Diagnostic values computed from the frozen conduction operator.
ConductionDiagnostics compute_conduction_diagnostics(const core::State& state,
                                                     const core::Config& cfg,
                                                     const HydroEOSContext* eos_ctx = nullptr);

namespace conduction {

double coulomb_log(double n_e, double Te_eV, double Zbar);
double spitzer_conductivity(double Te_eV, double Zbar, double ln_lambda);
double electron_thermal_velocity(double Te_eV);
double flux_limited_heat_flux(double q_sh, double q_max);
double effective_diffusion(double q_limited,
                           double rho,
                           double cv_e,
                           double grad_Te,
                           double D_sh,
                           double eps_grad);
int sts_stage_count(double dt, double dt_exp, int sts_max_stages);

struct PerMaterialFaceCoefficients1D {
  int n_faces = 0;
  int n_mat = 0;
  std::vector<double> aggregate_kappa;
  std::vector<double> kappa_eff_per_material;
};

PerMaterialFaceCoefficients1D compute_per_material_face_coefficients_1d(
    core::State& state, const core::Config& cfg, const HydroEOSContext* eos_ctx);

double per_material_dirichlet_boundary_flux_1d(const core::State& state,
                                               const core::Config& cfg,
                                               int cell,
                                               double T_boundary_eV,
                                               double dx_face,
                                               double f_lim);

// Verify-only SNB flux probe (design doc §5): evaluates the SNB face fluxes for
// the current state without advancing Te. Host copies; sizes n_cells+1.
struct SnbFluxProbe {
  std::vector<double> q_sh_face;
  std::vector<double> dq_face;
  std::vector<double> theta_face;
  int picard_iters = 0;
  double dq_over_qsh_max = 0.0;
};

SnbFluxProbe snb_probe_fluxes(core::State& state, const core::Config& cfg,
                              const HydroEOSContext* eos_ctx = nullptr);

}  // namespace conduction

}  // namespace tenryu::hydro
