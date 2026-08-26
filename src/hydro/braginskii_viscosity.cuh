#pragma once

// Braginskii plasma shear viscosity for the Lagrangian hydro
// (docs/design/wh_braginskii_viscosity_design.md; 2D RZ port
// docs/design/2d_visc_port_spec.md), extended with the electron channel
// (docs/design/electron_viscosity_1d_20260712.md, ported by
// docs/design/visc_2d_parity_20260717.md).
//
// Physics: unmagnetized Braginskii viscosity, single-fluid V_e = V_i.
//   ion:      eta_i = 0.96 n_i kT_i tau_i                (Braginskii Eq. 2.22)
//   electron: eta_e = eta00_e(Z) n_e kT_e tau_e          (Braginskii Eq. 2.25
//             at Z=1; Whitney Z-dependence via Velikovich 2001 Eq. 3)
//   pi = -eta_eff W (trace-free), eta_eff per Params::species:
//   0 "ion" (legacy, default), 1 "electron", 2 "both" (eta_i + eta_e).
// NRL collision times and Coulomb logs, an optional Mason-2014
// mean-free-path cap per channel (lambda <= mfp_cap_cells * dr), viscous
// heating (eta_i/2) W:W -> ion energy and (eta_e/2) W:W -> electron energy,
// and an explicit viscous dt limit dt <= dt_safety * rho * dr^2 /
// ((8/3) * eta_eff).
//
// Default-inert: everything is gated on Params.enabled
// (Numerics.hydro.plasma_viscosity, with TENRYU_BRAG_* diagnostic overrides);
// disabled => no kernel launches, bit-identical trajectories. species=0
// reproduces the pre-electron ion trajectories bitwise (1D and 2D).

#include <cstdint>

namespace tenryu::core {
struct Config;
struct State;
}

namespace tenryu::hydro::braginskii {

struct Params {
  bool enabled = false;
  int model = 0;                 // 0 = "braginskii", 1 = "constant"
  int species = 0;               // 0 = "ion", 1 = "electron", 2 = "both"
  double eta_const = 0.0;        // [poise] model==1 uniform viscosity
  double eta0_scale = 1.0;       // multiplier on Braginskii eta0
  double mfp_cap_cells = 20.0;   // lambda_ii <= C * dr (0 = off) [Mason 2014]
  double lnlambda_fixed = 0.0;   // 0 = NRL lnLambda_ii; >0 = fixed value
  double dt_safety = 0.3;        // viscous CFL fraction
  double ti_floor_ev = 1.0e-3;   // matches Numerics.floors.Ti default
  double te_floor_ev = 1.0e-3;   // matches Numerics.floors.Te default
};

// Fail-closed parameter validation (2026-07-26 kernel review):
// throws std::runtime_error when an enabled parameter set carries a
// non-finite or out-of-range value (eta_const/eta0_scale/mfp_cap_cells/
// lnlambda_fixed < 0, dt_safety <= 0, unknown model/species code). Called by
// both factories below; a disabled Params is always accepted.
void validate_params(const Params& p);

// Reads TENRYU_BRAG_{ENABLE,MODEL,SPECIES,ETA_CONST,ETA0_SCALE,MFP_CAP_CELLS,
// LNLAMBDA_FIXED,DT_SAFETY}. Unset TENRYU_BRAG_ENABLE (or "0") => disabled
// and the remaining variables are not consulted. Numeric variables must
// parse completely as finite numbers and unknown MODEL/SPECIES strings are
// rejected (no silent fallback, no fail-open override).
Params params_from_env();

// Namelist-authoritative parameters (Numerics.hydro.plasma_viscosity),
// with TENRYU_BRAG_* env vars applied on top as diagnostic overrides.
// Unknown namelist model/species strings are rejected here as well.
Params params_from_config(const core::Config& cfg);

// Cell-centered trace-free stress + optional viscous heating power.
// Device pointers; sizes: nodes = n_cells+1 (x_r, v_r), cells = n_cells
// (all others). Nullable: heat_rate / heat_rate_e (skip that heating
// output), Ti / Te / A_eff / zbar (fallbacks Ti=ti_floor_ev,
// Te=te_floor_ev, A=1, Z=1), hydro_active (all active).
// Outputs: pi_rr, pi_tt [erg/cm^3]; heat_rate / heat_rate_e [erg/s] cell
// powers of the ion / electron channel (= (eta_s/2) W:W * vol, the
// apply_artificial_heat_kernel convention).
void compute_cell_stress_1d(double* pi_rr,
                            double* pi_tt,
                            double* heat_rate,
                            double* heat_rate_e,
                            const double* x_r,
                            const double* v_r,
                            const double* rho,
                            const double* Ti,
                            const double* Te,
                            const double* A_eff,
                            const double* zbar,
                            const double* vol,
                            const std::int8_t* hydro_active,
                            int n_cells,
                            int geom_code,
                            const Params& p,
                            const double* zmom_r4 = nullptr);

// Adds the viscous acceleration -(div pi)_r / rho to the node accel array,
// including the hoop term alpha*(pi_rr - pi_tt)/r for spherical (alpha=2)
// and cylindrical (alpha=1). Mirrors the guards of
// compute_acceleration_1d_kernel: node 0 skipped, inactive nodes skipped,
// outer node skipped when skip_outer != 0 (FIXED/REFLECT BC), otherwise the
// outside is stress-free (pi = 0).
void add_node_accel_1d(double* accel,
                       const double* mass,
                       const double* x_r,
                       const double* vol,
                       const double* pi_rr,
                       const double* pi_tt,
                       const std::uint8_t* node_active,
                       int n_cells,
                       int geom_code,
                       int skip_outer,
                       const Params& p);

// Explicit viscous stability limit over the given cells (device pointers).
// Returns +inf when p.enabled is false or no cell restricts.
double compute_dt_braginskii_raw(const double* x_r,
                                 const double* rho,
                                 const double* Ti,
                                 const double* Te,
                                 const double* A_eff,
                                 const double* zbar,
                                 const std::int8_t* hydro_active,
                                 int n_cells,
                                 const Params& p,
                                 const double* zmom_r4 = nullptr);

// State-level wrapper for the coupling dt control: config-gated
// (params_from_config), 1D meshes only; +inf when disabled.
double compute_dt_braginskii(const core::State& state, const core::Config& cfg);

// Exact Gershgorin row-sum bound lambda_G >= rho(-L) of the assembled 1D
// node-acceleration operator of add_node_accel_1d, linearized in the node
// velocities at frozen eta (the operator is linear in u for frozen
// coefficients). On uniform planar meshes 2/lambda_G reproduces the scalar
// dt formula's stability limit exactly; spherical/cylindrical rows
// additionally carry the alpha*(pi_rr - pi_tt)/r hoop stiffness that the
// scalar formula does not see (up to ~2.5x at the spherical origin), and
// graded meshes the center-distance coupling. Used by the end-of-step
// viscous stability audit in hydro_1d.cu (AI kernel review 2026-07-26,
// 2026-07-26 kernel review: dt must satisfy dt <= 2/lambda_G or the step is retried
// (soft failure) / aborted. Returns 0 when disabled or nothing contributes.
double compute_viscous_gershgorin_lambda_1d(const double* x_r,
                                            const double* rho,
                                            const double* Ti,
                                            const double* Te,
                                            const double* A_eff,
                                            const double* zbar,
                                            const double* mass,
                                            const double* vol,
                                            const std::int8_t* hydro_active,
                                            const std::uint8_t* node_active,
                                            int n_cells,
                                            int geom_code,
                                            int skip_outer,
                                            const Params& p,
                                            const double* zmom_r4 = nullptr);

// History-cadence diagnostics (design doc section 10). eta_i_max / eta_e_max
// and the ratio statistics are the PHYSICAL Braginskii channel viscosities
// evaluated from the instantaneous state — independent of model/species AND
// of the numerical knobs (eta0_scale = 1, mfp cap off, NRL Coulomb log;
// 2026-07-26 kernel review: a capped/scaled "physical" channel
// would make the regime map mesh-dependent). eta_eff_max and the heat-rate
// totals reflect the configured (species, model, cap, scale) run. Ratio statistics are
// over active cells with eta_i > 0; regime counts use R >= 10 (electron-
// dominant) and R <= 0.1 (ion-dominant) — diagnostic classification only,
// no feedback into the physics. Host-side deterministic reduction (no
// atomics): safe for the 1D bitwise reproducibility discipline. Rates are
// instantaneous (no accumulators => Strang-retry safe).
struct HistoryDiagnostics {
  bool valid = false;
  double eta_i_max = 0.0;             // [poise]
  double eta_e_max = 0.0;             // [poise]
  double eta_eff_max = 0.0;           // [poise]
  double ratio_min = 0.0;             // R = eta_e / eta_i
  double ratio_geomean_masswt = 0.0;  // exp(sum m ln R / sum m)
  double ratio_max = 0.0;
  int n_cells_e_dom = 0;              // R >= 10
  int n_cells_i_dom = 0;              // R <= 0.1
  int n_cells_mixed = 0;
  int n_cells_active = 0;
  double heat_rate_i_tot = 0.0;       // [erg/s]
  double heat_rate_e_tot = 0.0;       // [erg/s]
};

// Config-gated (params_from_config => env-aware), 1D and 2D RZ meshes
// (dim==2 dispatches to the 2D implementation below); returns valid=false
// when disabled or not applicable.
HistoryDiagnostics compute_history_diagnostics(const core::State& state,
                                               const core::Config& cfg);

// dim==2 implementation (braginskii_viscosity_2d.cu): identical field
// semantics; the strain operator is the 2D shoelace-gradient + hoop split
// of the stress kernel and the mfp-cap length is the min active cell edge
// (same L as the 2D dt kernel). Called via compute_history_diagnostics.
HistoryDiagnostics compute_history_diagnostics_2d(const core::State& state,
                                                  const core::Config& cfg);

double compute_dt_braginskii_2d(const core::State& state,
                                const core::Config& cfg);

// ---- 2D RZ (docs/design/2d_visc_port_spec.md) ----

// Mesh topology handle for the 2D kernels. Structured tensor-product
// meshes: set nr/nz and leave the pointers null. Multiblock: set n_cells /
// n_nodes and the device CSR pointers (forward cell->node CSR: 4 storage
// slots per cell in cyclic perimeter order — same contract the csw98/
// compatible kernels rely on; cell_nverts nullable => all quads; reverse
// CSR only needed by add_corner_forces_to_nodes_2d).
struct Topo2D {
  int nr = 0;
  int nz = 0;
  int n_cells = 0;
  int n_nodes = 0;
  const int* cell_node_csr_offsets = nullptr;   // [n_cells+1]
  const int* cell_node_csr_indices = nullptr;   // [n_cells*4]
  const std::uint8_t* cell_nverts = nullptr;    // [n_cells] or null
  const int* rev_node_offsets = nullptr;        // [n_nodes+1]
  const int* rev_node_cells = nullptr;
  const int* rev_node_corners = nullptr;
  bool multiblock = false;
};

// Fused cell kernel: strain -> eta_eff -> pi -> 4 corner forces (+ per-
// channel heat rates), species-templated like the 1D stress kernel.
// Outputs: corner_visc_r/z [n_cells*4] (slot k pairs with the cell's k-th
// cyclic corner node), heat_rate [n_cells] = V*(eta_i/2)*W:W [erg/s] and
// heat_rate_e [n_cells] = V*(eta_e/2)*W:W (each nullable; the ion-channel
// rate lands in heat_rate, the electron-channel rate in heat_rate_e —
// species="ion" writes heat_rate only, "electron" heat_rate_e only,
// "both" splits). All pointers device. Ti / Te / A_eff / zbar nullable
// (fallbacks ti_floor_ev / te_floor_ev / 1 / 1, as in 1D).
void compute_corner_forces_2d(double* corner_visc_r,
                              double* corner_visc_z,
                              double* heat_rate,
                              double* heat_rate_e,
                              const double* x_r,
                              const double* x_z,
                              const double* v_r,
                              const double* v_z,
                              const double* rho,
                              const double* Ti,
                              const double* Te,
                              const double* A_eff,
                              const double* zbar,
                              const double* vol,
                              const std::int8_t* hydro_active,
                              const Topo2D& topo,
                              const Params& p);

// force_r/z[n] += sum of incident viscous corner forces. Structured:
// per-cell atomicAdd scatter (same noise class as the pressure corner
// sum). Multiblock: per-node reverse-CSR gather, deterministic, ADDS (+=).
void add_corner_forces_to_nodes_2d(double* force_r,
                                   double* force_z,
                                   const double* corner_visc_r,
                                   const double* corner_visc_z,
                                   const std::int8_t* hydro_active,
                                   const Topo2D& topo);

// work_visc[c] = -sum_k F_k . v_(node_k)  (compatible-path tally; SIGNED,
// never clamped).
void compute_work_visc_2d(double* work_visc,
                          const double* corner_visc_r,
                          const double* corner_visc_z,
                          const double* v_r,
                          const double* v_z,
                          const std::int8_t* hydro_active,
                          const Topo2D& topo);

// Explicit viscous dt over 2D cells: min_c dt_safety*rho*L^2/((8/3)*eta),
// L = min active edge length. +inf when no cell restricts.
double compute_dt_braginskii_2d_raw(const double* x_r,
                                    const double* x_z,
                                    const double* rho,
                                    const double* Ti,
                                    const double* Te,
                                    const double* A_eff,
                                    const double* zbar,
                                    const std::int8_t* hydro_active,
                                    const Topo2D& topo,
                                    const Params& p);

// State-buffer helpers (idempotent; used by the W1b wiring). with_electron
// additionally sizes the electron heat-rate buffer (species != "ion").
void ensure_visc_buffers_2d(core::State& state, int n_cells,
                            bool with_electron);
void zero_visc_buffers_2d(core::State& state);

}  // namespace tenryu::hydro::braginskii
