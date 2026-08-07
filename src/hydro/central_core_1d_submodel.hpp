#pragma once

// Stratified 1D spherical Lagrangian sub-model for the central pseudo-core
// (env TENRYU_I1B_CORE_1D_SUBMODEL, default off).
//
// Replaces the pooled 0-D balloon closure as the core's DYNAMICS and METRIC
// authority: absorbed rings append outer shells preserving (m, e, u_r) —
// momentum enters as piston/shell motion instead of being thermalized by the
// KE bank — and the interior compresses as stratified same-gamma ideal-gas
// hydro (staggered von Neumann-Richtmyer, the scheme validated against the
// 2D code at the canonical drive by tmp/ref1d_caseD.py). The 2D <-> 1D
// coupling is a kinematic piston: the sub-model outer radius tracks
// r_out = (3 V_c / 4 pi)^{1/3} of the actual macro boundary loop volume, and
// the 2D boundary force consumes the sub-model's outer-face pressure via
// macro_core_pressure(). Pooled pc.Ue_c/Ui_c bookkeeping stays untouched (2D
// total-energy contracts unchanged); the sub-model keeps its own
// injected-vs-current conservation ledger.
//
// Host-only, side-map keyed by the owning core::State address; no core
// headers required.

#include <vector>

namespace tenryu::hydro::core1d {

struct Core1DParams {
  bool enabled = false;      // master switch (build/absorb/advance gates)
  int build_shells = 48;     // [4,4096]
  int split_append = 0;      // 0 = off; else max sub-shells per append [0,1024]
  double av_c1 = 0.5;        // >0, VNR linear AV coefficient
  double av_c2 = 4.0;        // >0, VNR quadratic AV coefficient
  double cfl = 0.25;         // (0,1)
  double piston_cap = 10.0;  // >0, kinematic piston speed cap factor
  int max_substeps = 20000;  // >=1 per 2D step
  // Distribution-preserving append (fidelity upgrade over pooled per-ring
  // appends): absorbed units arrive as u_r-sorted per-cell sub-shells so the
  // arrival-velocity DISTRIBUTION survives into the sub-model (the angular
  // variance then thermalizes through RESOLVED sub-shell shocks instead of
  // instant pooling). Addresses the characterized -9.4% coupling systematic.
  bool dist_append = false;
};
void set_params(const Core1DParams& p);  // called ONCE before first build

bool enabled();

struct BuildCell {
  double r = 0.0;   // cell-center radius [cm]
  double m = 0.0;   // mass [g]
  double v = 0.0;   // volume [cm^3]
  double e = 0.0;   // specific internal energy (ee+ei) [erg/g]
  double Y = 0.0;   // gas tracer mass fraction
};

void build(const void* key,
           std::vector<BuildCell> cells,
           double V_c,
           double gamma);
bool built(const void* key);
void reset(const void* key);

// One absorbed unit (ring) becomes one new outer shell. The shell's radial
// extent comes from its ACTUAL absorbed volume (robust against transient
// macro-boundary-loop volumes during failure-retry chains); the piston
// reconciles to the real V_c on the next advance.
void absorb_event(const void* key,
                  double mass,
                  double e_specific,
                  double u_radial,
                  double gas_fraction,
                  double shell_volume);

// Distribution-preserving absorption (Core1DParams::dist_append or env
// TENRYU_I1B_CORE_1D_DIST_APPEND): one absorbed unit arrives as per-cell
// samples; they are appended innermost-first in ascending u_r order (most
// negative = fastest inward = deepest), each sub-shell keeping its own
// radial momentum. Per-sample e already contains that cell's own
// non-radial-KE thermalization (caller contract).
struct ShellSample {
  double m = 0.0;    // mass [g]
  double e = 0.0;    // specific internal energy incl. own angular remainder
  double u_r = 0.0;  // radial velocity [cm/s] (negative = inward)
  double Y = 0.0;    // gas tracer fraction
  double V = 0.0;    // volume [cm^3]
};
bool dist_append_enabled();
void absorb_event_distribution(const void* key,
                               std::vector<ShellSample> samples);

// Step lifecycle: the sub-model advances EXACTLY ONCE per (state, 2D step)
// — begin_step returns true only on the first entry for a step (the
// hydro_step_start aggregate runs more than once per step and retries
// re-enter it; re-advancing with rollback was measured to leak kinetic
// energy through paired rollback/re-advance discrepancies). An absorption
// re-arms the step so the post-absorb entry advances to the new volume.
bool begin_step(const void* key, int step);
void advance_to_volume(const void* key,
                       double V_c,
                       double dt_2d,
                       double gamma);

// Dynamic (force-driven) outer boundary: the outer node evolves under the
// pressure difference between the external (2D boundary-adjacent) pressure
// and the sub-model's own outer face — no kinematic velocity override
// (measured to pump the core through override/slosh rectification after
// massive shell appends: W_1d swings +-7e4 erg/step with ~zero physical
// interface work). V_c divergence stays a diagnostic (the pair-audit dV
// column), not a constraint.
void advance_dynamic(const void* key,
                     double P_ext,
                     double dt_2d,
                     double gamma);

// Terminal (free) outer boundary: after the terminal absorption there is no
// 2D partner — the outer face becomes a standard massive VNR free surface
// (face carries half the outer cell's mass, driven by P_face - P_ext). The
// switch re-attributes the outer cell's lumped mass from the last interior
// node to the face; the kinetic-energy convention jump is booked explicitly
// into the injected-energy ledger so closure stays exact.
void set_free_outer(const void* key, double gamma);
bool free_outer(const void* key);
// Total explicit substeps taken so far (tail progress watchdog).
long long substep_count(const void* key);

double outer_face_pressure(const void* key, double gamma);
// Diagnostic split of the outer-face stress: static pressure only (the
// force/coupling path keeps P+q). Used by the A2 overlap instrument to
// compare like with like against the 2D static-pressure average.
double outer_face_pressure_static(const void* key, double gamma);

// A6 pairing-audit accessors: the sub-model's cumulative piston work and
// its current discrete volume (4pi/3 r_out^3).
double piston_work_total(const void* key);
double current_volume(const void* key);

struct GasView {
  bool valid = false;
  double V_gas = 0.0;        // stratified gas partial volume [cm^3]
  double M_gas = 0.0;        // gas mass [g]
  double U_gas = 0.0;        // gas internal energy [erg]
  double p_center = 0.0;     // innermost-shell pressure [dyn/cm^2]
  double e_center = 0.0;     // innermost-shell specific internal energy
  double rhoR_gas = 0.0;     // integral rho dr over gas shells [g/cm^2]
  double r_gas_outer = 0.0;  // outer radius of the outermost gas shell [cm]
};
GasView gas_view(const void* key, double gamma);

// Conservation ledger print (env TENRYU_I1B_CORE_1D_LEDGER_EVERY steps).
void emit_ledger(const void* key, int step, double t, double gamma);

}  // namespace tenryu::hydro::core1d
