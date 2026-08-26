#pragma once

// W-K: gamma_r = 4/3 radiation-field compression coupling for the 1D
// Lagrangian step (2026-07-04 external review,
// docs/design/external-ai-responses/20260704-conduction-kirchhoff-radiation-gammar-verdict.md;
// design docs/design/wk_gamma_r_coupling_design.md).
//
// Comoving gray zeroth moment with P_r = (E_r/3) I gives
// DE_r/Dt = -(4/3) E_r div(u), i.e. E_r V^{4/3} = const per Lagrangian
// half; the exact per-cell update is E1 = E0 * (V0/V1)^{4/3} and the exact
// radiation work ledger is W_r = Delta(E_r V) = E0 V0 [(V0/V1)^{1/3} - 1].
// Every group is scaled by the same factor (gray-work approximation, no
// inter-group frequency shift; the total pdV work is exact).
//
// Default-inert: enabled only when TENRYU_RAD_HYDRO_COUPLING equals
// "gamma_r_43" (the Radiation.multigroup_diffusion.hydro_coupling namelist
// switch ships as diff proposal tmp/diff_proposals/wk-gammar/01).

namespace tenryu::coupling::rad_gamma {

// True iff getenv("TENRYU_RAD_HYDRO_COUPLING") == "gamma_r_43".
bool gamma_r_43_enabled_from_env();

// rad_E[k] <- rad_E[k] * (vol_before[c]/vol_after[c])^{4/3}, k = c*n_groups+g
// (groups-fastest layout). Device pointers; cells with non-positive V0 or V1
// and non-finite E entries are left unchanged. No-op if sizes are zero.
void apply_gamma_r_43_volume_update(double* rad_E,
                                    const double* vol_before,
                                    const double* vol_after,
                                    int n_cells,
                                    int n_groups);

// Compatible-energy update: rad_E is adjusted so Sum_g(E_new)*V_new =
// Sum_g(E_old)*V_old - W_r[c]. Groups keep the existing gray-work split
// (common scale factor when possible). Returns floored radiation energy [erg].
// Applies to cells [c_begin, c_end) only: under MPI (Option C) W_r is
// produced by the owned-window gamma_r force-work kernel, so the payment
// window must match — full-range application would fold garbage W_r into
// non-owned rad_E. Serial callers pass (0, n_cells).
double apply_gamma_r_43_work_update(double* rad_E,
                                    const double* W_r,
                                    const double* vol_before,
                                    const double* vol_after,
                                    int c_begin,
                                    int c_end,
                                    int n_groups);

// p_r[c] = (1/3) * Sum_g max(rad_E[c*n_groups+g], 0). Device pointers.
void compute_radiation_pressure_field(double* p_r,
                                      const double* rad_E,
                                      int n_cells,
                                      int n_groups);

}  // namespace tenryu::coupling::rad_gamma
