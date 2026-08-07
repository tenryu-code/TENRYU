"""Controlled finite-CR shell-on-hotspot I1-B discriminator deck.

This is a measurement-only deck for cases A/B/C/D selected by
TENRYU_I1B_DISC_CASE.  It uses one ideal-gas material everywhere; radial
regions are density/temperature IC structure plus banner metadata for the
external harness, not runtime material fields.
TENRYU_I1B_DISC_CENTRAL_PSEUDO_CORE_* controls the default-off Exp1 virtual
central pseudo-core diagnostic.
"""

import math
import os

from tenryu_namelist import *

from i1b_polar_common import ablation_pressure_ramp, c2_smoothstep


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_float_first(names, default):
    for name in names:
        if name in os.environ:
            return float(os.environ[name])
    return default


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


PREFIX = "TENRYU_I1B_DISC_"
CASE_NAME_BASE = "2d_rz_i1b_finite_cr_shell_hotspot"
VALID_CASES = ("A", "B", "C", "D")
CASE = os.environ.get(PREFIX + "CASE", "A").strip().upper()
if CASE not in VALID_CASES:
    raise ValueError("TENRYU_I1B_DISC_CASE must be one of A, B, C, D")

# CASE D is the CERTIFIED capsule track (Tier-2 gate evidence, Addenda 1-7):
# its geometry defaults are the certified design — gas to 350 um, shell to
# 455 um, ablator to 490 um, thin exterior buffer to S_MAX=520 um (the thick
# default buffer re-opens the snowplow-grind dt collapse, measured). Envs
# still override every value.
_CASE_D = CASE == "D"

NZ = _env_int(PREFIX + "NZ", 32)
if NZ % 4 != 0:
    raise ValueError("TENRYU_I1B_DISC_NZ must be divisible by 4")
if PREFIX + "N_C" in os.environ:
    N_C = _env_int(PREFIX + "N_C", 8)
    if NZ != 4 * N_C:
        raise ValueError("TENRYU_I1B_DISC_NZ must equal 4*TENRYU_I1B_DISC_N_C")
else:
    N_C = NZ // 4
BRIDGE_LAYERS = _env_int(PREFIX + "BRIDGE_LAYERS", 4)
NR_SHELL = _env_int(PREFIX + "NR_SHELL", 8)
SEED = _env_int(PREFIX + "SEED", 12345)
S_MAX = _env_float(PREFIX + "S_MAX_CM", 0.052 if _CASE_D else 0.07)
R_C = _env_float(PREFIX + "R_C_CM", S_MAX / 12.0)
R_MATCH = _env_float(PREFIX + "R_MATCH_CM", 2.0 * R_C)

R_G = _env_float(
    PREFIX + "R_G_CM",
    0.035 if _CASE_D else _env_float(PREFIX + "R_G_FRAC", 0.50) * S_MAX,
)
R_S = _env_float(
    PREFIX + "R_S_CM",
    0.0455 if _CASE_D else _env_float(PREFIX + "R_S_FRAC", 0.65) * S_MAX,
)
R_OUTER = _env_float(
    PREFIX + "R_OUTER_CM",
    0.049 if _CASE_D else _env_float(PREFIX + "R_OUTER_FRAC", 0.70) * S_MAX,
)
R_HS = _env_float(
    PREFIX + "R_HS_CM",
    0.005 if _CASE_D else _env_float(PREFIX + "R_HS_FRAC", 50.0 / 700.0) * S_MAX,
)

if not (N_C > 0 and BRIDGE_LAYERS >= 0 and NR_SHELL > 0 and NZ > 0):
    raise ValueError("TENRYU_I1B_DISC mesh counts must be positive")
if not (0.0 < R_HS < R_G < R_S < R_OUTER < S_MAX):
    raise ValueError(
        "TENRYU_I1B_DISC radii must satisfy 0<R_HS<R_G<R_S<R_OUTER<S_MAX"
    )

A_AMU = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
EV_TO_ERG = 1.602176634e-12
AMU_G = 1.66053906660e-24

RHO_UNIFORM = _env_float(PREFIX + "RHO0_GCC", 1.0e-3)
RHO_GAS_FILL = _env_float(PREFIX + "RHO_GAS_FILL_GCC", 1.0e-4)
RHO_SHELL = _env_float(PREFIX + "RHO_SHELL_GCC", 0.25)
RHO_ABLATOR = _env_float(PREFIX + "RHO_ABLATOR_GCC", 1.05)
RHO_EXTERIOR = _env_float(PREFIX + "RHO_EXTERIOR_GCC", RHO_GAS_FILL)
T_UNIFORM_EV = _env_float(PREFIX + "T0_EV", 1.0)
T_GAS_BALANCED_EV = _env_float(PREFIX + "T_GAS_BALANCED_EV", 1.0e3)
ABLATOR_ENABLED = _env_bool(PREFIX + "ABLATOR_ENABLED", True)

if min(RHO_UNIFORM, RHO_GAS_FILL, RHO_SHELL, RHO_ABLATOR, RHO_EXTERIOR) <= 0.0:
    raise ValueError("TENRYU_I1B_DISC densities must be > 0")
if T_UNIFORM_EV <= 0.0 or T_GAS_BALANCED_EV <= 0.0:
    raise ValueError("TENRYU_I1B_DISC temperatures must be > 0")


def _pressure_from_equal_t(rho_gcc, t_ev):
    return rho_gcc * EV_TO_ERG * (ZBAR + 1.0) * t_ev / (A_AMU * AMU_G)


def _equal_t_from_pressure(p_dyn_cm2, rho_gcc):
    return p_dyn_cm2 * A_AMU * AMU_G / (
        rho_gcc * EV_TO_ERG * (ZBAR + 1.0)
    )


P_BASE = _env_float(
    PREFIX + "P0_DYN_CM2",
    _pressure_from_equal_t(RHO_GAS_FILL, T_GAS_BALANCED_EV),
)
GASP_SCALE = _env_float(PREFIX + "GASP_SCALE", 0.1)
ETA = _env_float(PREFIX + "ETA", 0.5)
A_CONV = _env_float(PREFIX + "A_CONV", 8.0e8)
P_EXT = 0.0

if P_BASE <= 0.0:
    raise ValueError("TENRYU_I1B_DISC_P0_DYN_CM2 must be > 0")
if GASP_SCALE <= 0.0:
    raise ValueError("TENRYU_I1B_DISC_GASP_SCALE must be > 0")
if ETA <= 0.0:
    raise ValueError("TENRYU_I1B_DISC_ETA must be > 0")

smooth_width_default = min(
    2.0 * S_MAX / float(max(NZ, 1)),
    0.45 * (R_S - R_G),
    0.45 * (R_OUTER - R_S),
    0.25 * R_G,
)
SMOOTH_WIDTH = _env_float(PREFIX + "SMOOTH_WIDTH_CM", smooth_width_default)
if SMOOTH_WIDTH < 0.0:
    raise ValueError("TENRYU_I1B_DISC_SMOOTH_WIDTH_CM must be >= 0")

T_END = _env_float(PREFIX + "T_END_S", 5.0e-9)
DT_INITIAL = _env_float(PREFIX + "DT_INITIAL_S", 2.5e-11)
DT_MAX = _env_float(PREFIX + "DT_MAX_S", 5.0e-11)
CFL_HYDRO = _env_float(PREFIX + "CFL_HYDRO", 0.2)
DRIVER_RETRY = _env_bool(PREFIX + "RETRY", True)
DRIVER_RETRY_MAX_ATTEMPTS = _env_int(PREFIX + "RETRY_MAX_ATTEMPTS", 8)
MAX_STEPS = _env_int(PREFIX + "MAX_STEPS", 400)
PLOT_EVERY_STEPS = _env_int(PREFIX + "PLOT_EVERY_STEPS", 4)
HISTORY_EVERY_STEPS = 1

if not (T_END > 0.0 and DT_INITIAL > 0.0 and DT_MAX > 0.0):
    raise ValueError("TENRYU_I1B_DISC time controls must be > 0")
if MAX_STEPS <= 0 or PLOT_EVERY_STEPS < 0:
    raise ValueError("TENRYU_I1B_DISC step/cadence controls are invalid")

ALE_ENABLED = _env_bool(PREFIX + "ALE", CASE in ("C", "D"))
ALE_IDENTITY = _env_bool(PREFIX + "ALE_IDENTITY", False)
ALE_MOVER_DIAG = _env_bool(PREFIX + "ALE_MOVER_DIAG", False)
ALE_PRESERVE_VEL_CARRY = _env_bool(PREFIX + "ALE_PRESERVE_VEL_CARRY", False)
REFERENCE_BARRIER = _env_bool(PREFIX + "REFERENCE_BARRIER", False)
SWEPT_VOLUME_SIGN_FIXED = _env_bool(PREFIX + "SWEPT_VOLUME_SIGN_FIXED", False)
DIFFREF_ENABLED = _env_bool(PREFIX + "DIFFREF", CASE == "C")
CENTER_PATCH_REFERENCE_ENABLED = _env_bool(
    PREFIX + "MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED",
    CASE == "D",
)
SCALED_REFERENCE_ENABLED = _env_bool(
    PREFIX + "SCALED_REFERENCE",
    ALE_ENABLED and not DIFFREF_ENABLED and not CENTER_PATCH_REFERENCE_ENABLED,
)
DIFFREF_BAND_COUNT = (
    _env_int(PREFIX + "DIFFREF_BAND_COUNT", 0)
    if PREFIX + "DIFFREF_BAND_COUNT" in os.environ
    else None
)
DIFFREF_NU = (
    _env_float(PREFIX + "DIFFREF_NU", 0.0)
    if PREFIX + "DIFFREF_NU" in os.environ
    else None
)
ALE_DRIVER_ENABLED = ALE_ENABLED or ALE_IDENTITY or ALE_PRESERVE_VEL_CARRY
ALE_EVERY_N_STEPS = _env_int(PREFIX + "ALE_EVERY_N_STEPS", 1)
DEBUG_PER_REMAP_LOG = _env_bool("TENRYU_I1B_DEBUG_PER_REMAP_LOG", False)
CONSERVATIVE_REMAP_ENABLED = _env_bool(
    PREFIX + "CONSERVATIVE_REMAP_ENABLED", True
)
CONSERVATIVE_REMAP_ORDER = os.environ.get(
    PREFIX + "CONSERVATIVE_REMAP_ORDER", "second_order_van_leer"
)
CONSERVATIVE_REMAP_TARGET = os.environ.get(
    PREFIX + "CONSERVATIVE_REMAP_TARGET", "reference"
)
TOTAL_ENERGY_REMAP = _env_bool(PREFIX + "TOTAL_ENERGY_REMAP", CASE == "D")
HOTSPOT_GAS_TRACER = _env_bool(PREFIX + "HOTSPOT_GAS_TRACER", CASE == "D")
CORE_FREEZE = _env_bool(PREFIX + "CORE_FREEZE", False)
CORE_FREEZE_CUT = _env_float(PREFIX + "CORE_FREEZE_CUT", 0.5)
CORE_FREEZE_HALO = _env_int(PREFIX + "CORE_FREEZE_HALO", 1)
CENTRAL_PSEUDO_CORE_ENABLED = _env_bool(PREFIX + "CENTRAL_PSEUDO_CORE_ENABLED", CASE == "D")
CENTRAL_PSEUDO_CORE_S_C_CM = _env_float(
    PREFIX + "CENTRAL_PSEUDO_CORE_S_C_CM", 1.1 * R_HS if CASE == "D" else 0.0)
AXIS_REZONE = _env_bool(PREFIX + "AXIS_REZONE", CASE == "D")
RING7_QUOTIENT = _env_bool(
    "TENRYU_I1B_RING7_QUOTIENT",
    _env_bool(PREFIX + "RING7_QUOTIENT", _CASE_D),
)
if ALE_EVERY_N_STEPS <= 0:
    raise ValueError("TENRYU_I1B_DISC_ALE_EVERY_N_STEPS must be > 0")
if CENTER_PATCH_REFERENCE_ENABLED and not ALE_DRIVER_ENABLED:
    raise ValueError(
        "TENRYU_I1B_DISC_MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED "
        "requires ALE enabled"
    )
if CENTER_PATCH_REFERENCE_ENABLED and (DIFFREF_ENABLED or SCALED_REFERENCE_ENABLED):
    raise ValueError(
        "TENRYU_I1B_DISC center-patch reference conflicts with differential/scaled references"
    )
if REFERENCE_BARRIER and not ALE_DRIVER_ENABLED:
    raise ValueError("TENRYU_I1B_DISC_REFERENCE_BARRIER requires ALE enabled")
if not (0.0 <= CORE_FREEZE_CUT <= 1.0):
    raise ValueError("TENRYU_I1B_DISC_CORE_FREEZE_CUT must be in [0, 1]")
if CORE_FREEZE_HALO < 0:
    raise ValueError("TENRYU_I1B_DISC_CORE_FREEZE_HALO must be >= 0")
if CORE_FREEZE and not HOTSPOT_GAS_TRACER:
    raise ValueError(
        "TENRYU_I1B_DISC_CORE_FREEZE requires TENRYU_I1B_DISC_HOTSPOT_GAS_TRACER"
    )
if CENTRAL_PSEUDO_CORE_ENABLED and not (CENTRAL_PSEUDO_CORE_S_C_CM > 0.0):
    raise ValueError(
        "TENRYU_I1B_DISC_CENTRAL_PSEUDO_CORE_S_C_CM must be > 0 when "
        "TENRYU_I1B_DISC_CENTRAL_PSEUDO_CORE_ENABLED=1"
    )

CASE_NAME = (
    f"{CASE_NAME_BASE}_case{CASE}_nz{NZ}_eta{_safe_float_token(ETA)}"
    f"_seed{SEED}"
)
OUTDIR = os.environ.get(PREFIX + "OUTDIR", f"./build/output_verify_{CASE_NAME}")

V_G0 = (4.0 / 3.0) * math.pi * R_G**3
C_G = R_G / R_HS
DELTA_U_G = 1.5 * P_BASE * V_G0 * (C_G * C_G - 1.0)
V_SHELL = (4.0 / 3.0) * math.pi * (R_S**3 - R_G**3)
V_ABLATOR = (4.0 / 3.0) * math.pi * (R_OUTER**3 - R_S**3)
ACTIVE_SHELL_MASS = RHO_SHELL * V_SHELL
if ABLATOR_ENABLED:
    ACTIVE_SHELL_MASS += RHO_ABLATOR * V_ABLATOR
else:
    ACTIVE_SHELL_MASS += RHO_SHELL * V_ABLATOR
U0_SHELL = math.sqrt(2.0 * DELTA_U_G / (ETA * ACTIVE_SHELL_MASS))
AREAL_DENSITY = RHO_SHELL * (R_S - R_G)
AREAL_DENSITY += (RHO_ABLATOR if ABLATOR_ENABLED else RHO_SHELL) * (R_OUTER - R_S)
REQUESTED_IMPULSE_AREA = AREAL_DENSITY * U0_SHELL
IMPULSE_SURFACE_AREA = 4.0 * math.pi * R_OUTER * R_OUTER

D_T_R = _env_float_first(
    (PREFIX + "D_T_RAMP_UP_S", PREFIX + "T_RAMP_UP_S"),
    0.25e-9,
)
D_T_H = _env_float_first(
    (PREFIX + "D_T_HOLD_END_S", PREFIX + "T_HOLD_END_S"),
    1.0e-9,
)
D_T_D = _env_float_first(
    (PREFIX + "D_T_RAMP_DOWN_S", PREFIX + "T_RAMP_DOWN_S"),
    0.25e-9,
)
if not (D_T_R > 0.0 and D_T_H >= D_T_R and D_T_D > 0.0):
    raise ValueError(
        "TENRYU_I1B_DISC D ramp times must satisfy rise>0, hold_end>=rise, fall>0"
    )
D_IMPULSE_DURATION = 0.5 * D_T_R + (D_T_H - D_T_R) + 0.5 * D_T_D
D_P_PEAK_AUTO = REQUESTED_IMPULSE_AREA / max(D_IMPULSE_DURATION, 1.0e-300)
D_P_PEAK = _env_float_first(
    (PREFIX + "D_P_PEAK_DYN_CM2", PREFIX + "P_PEAK_DYN_CM2"),
    D_P_PEAK_AUTO,
)
if D_P_PEAK < 0.0:
    raise ValueError("TENRYU_I1B_DISC_D_P_PEAK_DYN_CM2 must be >= 0")
REALIZED_IMPULSE_AREA = D_P_PEAK * D_IMPULSE_DURATION

REGION_LABELS = (
    "uniform",
    "gas_fill",
    "shell",
    "ablator_density_layer",
    "exterior",
)
IC_PRESSURE_POLICY = (
    "balanced_equal_total_pressure"
    if CASE != "C"
    else "gas_fill_pressure_scaled_with_c2_transition_at_R_g"
)

print(
    "[deck:2d_rz_i1b_finite_cr_shell_hotspot] "
    f"case={CASE} topology_scheme=multiblock_half_butterfly_trifan_cap_5block "
    f"n_c={N_C} fan_layers={BRIDGE_LAYERS} nr_shell={NR_SHELL} nz={NZ} "
    f"seed={SEED} S_max_cm={S_MAX} r_c_cm={R_C} r_match_cm={R_MATCH} "
    f"multiblock_cap_p=6.0 passive_region_labels={','.join(REGION_LABELS)} "
    "region_thresholds_cm=uniform:case_A_all,"
    f"gas_fill:rr<=R_g({R_G}),shell:R_g<rr<=R_s({R_S}),"
    f"ablator_density_layer:R_s<rr<=R_outer({R_OUTER}),"
    "exterior:rr>R_outer "
    f"r_hs_cm={R_HS} smooth_width_cm={SMOOTH_WIDTH} "
    f"rho_uniform_gcc={RHO_UNIFORM} rho_gas_fill_gcc={RHO_GAS_FILL} "
    f"rho_shell_gcc={RHO_SHELL} rho_ablator_density_layer_gcc={RHO_ABLATOR} "
    f"rho_exterior_gcc={RHO_EXTERIOR} ablator_enabled={ABLATOR_ENABLED} "
    f"P_base_dyn_cm2={P_BASE} gas_pressure_scale={GASP_SCALE} "
    f"ic_pressure_policy={IC_PRESSURE_POLICY} "
    f"T_uniform_eV={T_UNIFORM_EV} T_gas_balanced_eV={_equal_t_from_pressure(P_BASE, RHO_GAS_FILL)} "
    f"gamma={GAMMA} A={A_AMU} Z={ZBAR} A_conv_s_inv={A_CONV} "
    f"eta={ETA} C_g={C_G} delta_U_g_erg={DELTA_U_G} "
    f"active_shell_mass_g={ACTIVE_SHELL_MASS} u0_shell_cm_s={U0_SHELL} "
    f"areal_density_g_cm2={AREAL_DENSITY} "
    f"d_p_peak_auto_dyn_cm2={D_P_PEAK_AUTO} d_p_peak_dyn_cm2={D_P_PEAK} "
    f"d_rise_s={D_T_R} d_hold_end_s={D_T_H} d_fall_s={D_T_D} "
    f"d_impulse_requested_dyn_s_cm2={REQUESTED_IMPULSE_AREA} "
    f"d_impulse_realized_dyn_s_cm2={REALIZED_IMPULSE_AREA} "
    f"d_impulse_surface_area_cm2={IMPULSE_SURFACE_AREA} "
    f"d_impulse_requested_total_g_cm_s={REQUESTED_IMPULSE_AREA * IMPULSE_SURFACE_AREA} "
    f"d_impulse_realized_total_g_cm_s={REALIZED_IMPULSE_AREA * IMPULSE_SURFACE_AREA} "
    f"P_ext_dyn_cm2={P_EXT} t_end_s={T_END} dt_initial_s={DT_INITIAL} "
    f"dt_max_s={DT_MAX} cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} "
    f"plot_every_steps={PLOT_EVERY_STEPS} history_every_steps={HISTORY_EVERY_STEPS} "
    f"driver_full_step_retry_enabled={DRIVER_RETRY} retry_max={DRIVER_RETRY_MAX_ATTEMPTS} "
    f"motion={'ale' if ALE_DRIVER_ENABLED else 'lagrangian'} "
    f"ale_enabled={ALE_DRIVER_ENABLED} ale_requested={ALE_ENABLED} "
    f"ale_identity_mode={ALE_IDENTITY} ale_mover_diag={ALE_MOVER_DIAG} "
    f"ale_preserve_lagrangian_velocity_carry={ALE_PRESERVE_VEL_CARRY} "
    f"swept_volume_sign_fixed={SWEPT_VOLUME_SIGN_FIXED} "
    f"multiblock_scaled_reference_enabled={SCALED_REFERENCE_ENABLED} "
    f"multiblock_differential_reference_enabled={DIFFREF_ENABLED} "
    "multiblock_lagrangian_bulk_center_patch_reference_enabled="
    f"{CENTER_PATCH_REFERENCE_ENABLED} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"debug_per_remap_log={DEBUG_PER_REMAP_LOG} "
    f"conservative_remap_enabled={CONSERVATIVE_REMAP_ENABLED} "
    f"conservative_remap_order={CONSERVATIVE_REMAP_ORDER} "
    f"conservative_remap_target={CONSERVATIVE_REMAP_TARGET} "
    f"total_energy_remap={TOTAL_ENERGY_REMAP} "
    f"hotspot_gas_tracer={HOTSPOT_GAS_TRACER} "
    f"core_freeze={CORE_FREEZE} core_freeze_cut={CORE_FREEZE_CUT} "
    f"core_freeze_halo={CORE_FREEZE_HALO} "
    f"central_pseudo_core_enabled={CENTRAL_PSEUDO_CORE_ENABLED} "
    f"central_pseudo_core_s_c_cm={CENTRAL_PSEUDO_CORE_S_C_CM} "
    f"av_model=csw_edge subzonal_pressure_enabled=True "
    f"av_C1={_env_float('TENRYU_I1B_DISC_AV_C1', 0.5 if CASE == 'D' else 0.1)} "
    f"av_C2={_env_float('TENRYU_I1B_DISC_AV_C2', 4.0 if CASE == 'D' else 1.5)} "
    f"trial_volume_cfl_enabled=True mesh_quality_dt_cfl_enabled=True "
    f"fatal_on_neg=False outdir={OUTDIR}"
)


def _blend(a, b, x):
    return a + (b - a) * c2_smoothstep(x)


def _transition(a, b, rr, r0):
    if SMOOTH_WIDTH <= 0.0:
        return a if rr <= r0 else b
    return _blend(a, b, (rr - (r0 - SMOOTH_WIDTH)) / (2.0 * SMOOTH_WIDTH))


def _rho_layered(rr):
    if SMOOTH_WIDTH <= 0.0:
        if rr <= R_G:
            return RHO_GAS_FILL
        if rr <= R_S or not ABLATOR_ENABLED:
            return RHO_SHELL
        if rr <= R_OUTER:
            return RHO_ABLATOR
        return RHO_EXTERIOR

    if rr < R_G - SMOOTH_WIDTH:
        return RHO_GAS_FILL
    if rr < R_G + SMOOTH_WIDTH:
        return _transition(RHO_GAS_FILL, RHO_SHELL, rr, R_G)
    if rr < R_S - SMOOTH_WIDTH:
        return RHO_SHELL
    if ABLATOR_ENABLED and rr < R_S + SMOOTH_WIDTH:
        return _transition(RHO_SHELL, RHO_ABLATOR, rr, R_S)
    if rr < R_OUTER - SMOOTH_WIDTH:
        return RHO_ABLATOR if ABLATOR_ENABLED else RHO_SHELL
    if rr < R_OUTER + SMOOTH_WIDTH:
        inner = RHO_ABLATOR if ABLATOR_ENABLED else RHO_SHELL
        return _transition(inner, RHO_EXTERIOR, rr, R_OUTER)
    return RHO_EXTERIOR


def _pressure_layered(rr):
    if CASE != "C":
        return P_BASE
    p_gas = P_BASE * GASP_SCALE
    if SMOOTH_WIDTH <= 0.0:
        return p_gas if rr <= R_G else P_BASE
    if rr < R_G - SMOOTH_WIDTH:
        return p_gas
    if rr < R_G + SMOOTH_WIDTH:
        return _transition(p_gas, P_BASE, rr, R_G)
    return P_BASE


# Optional two-step (foot + main) drive for CASE=D (verdict #6 route C):
# a weak foot sets the shell/cushion adiabat gently before the main pulse,
# lowering the wall shock that folds the polar mesh at shock transit.
# Equal-impulse verdict shape: foot rise 0->0.2 ns, hold to 0.7 ns, ramp to
# the main pressure by 1.05 ns, hold to 1.596 ns, fall over 0.25 ns.
# Env-gated (default off = the original single trapezoid, unchanged).
D_FOOT_P = _env_float(PREFIX + "D_FOOT_P_DYN_CM2", 0.0)


def _foot_main_pulse(t):
    tn = t * 1.0e9
    if tn < 0.20:
        return D_FOOT_P * tn / 0.20
    if tn < 0.70:
        return D_FOOT_P
    if tn < 1.05:
        return D_FOOT_P + (D_P_PEAK - D_FOOT_P) * (tn - 0.70) / 0.35
    if tn < 1.596:
        return D_P_PEAK
    if tn < 1.846:
        return D_P_PEAK * (1.0 - (tn - 1.596) / 0.25)
    return 0.0


def boundary_pressure(t):
    if CASE == "D":
        if D_FOOT_P > 0.0:
            return _foot_main_pulse(t)
        return ablation_pressure_ramp(t, D_P_PEAK, D_T_R, D_T_H, D_T_D)
    return P_EXT


def rho_init(r, z):
    if CASE == "A":
        return RHO_UNIFORM
    return _rho_layered(math.hypot(r, z))


def Te_init(r, z):
    if CASE == "A":
        del r, z
        return T_UNIFORM_EV
    rr = math.hypot(r, z)
    return _equal_t_from_pressure(_pressure_layered(rr), _rho_layered(rr))


def Ti_init(r, z):
    return Te_init(r, z)


def _shell_velocity_weight(rr):
    if rr <= R_G:
        return 0.0
    inner = c2_smoothstep((rr - R_G) / (R_S - R_G))
    if SMOOTH_WIDTH <= 0.0:
        outer = 1.0 if rr <= R_OUTER else 0.0
    else:
        outer = 1.0 - c2_smoothstep((rr - R_OUTER) / SMOOTH_WIDTH)
    return inner * outer


def velocity_init(r, z):
    if CASE == "A":
        return (-A_CONV * r, -A_CONV * z)
    if CASE == "D":
        return (0.0, 0.0)
    rr = math.hypot(r, z)
    if rr <= 0.0:
        return (0.0, 0.0)
    speed = -U0_SHELL * _shell_velocity_weight(rr)
    return (speed * r / rr, speed * z / rr)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=NR_SHELL,
    nz=NZ,
    grid="uniform",
    motion="ale" if ALE_DRIVER_ENABLED else "lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    topology_scheme="multiblock_half_butterfly_trifan_cap_5block",
    multiblock_cap_p=6.0,
    multiblock_cart_core_n_c=N_C,
    multiblock_cart_core_bridge_layers=BRIDGE_LAYERS,
    multiblock_cart_core_r_c=R_C,
    multiblock_cart_core_r_match=R_MATCH,
)

Materials(
    materials=[
        Material(
            name="gas",
            A=A_AMU,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(gas=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

ale_config = dict(enabled=ALE_DRIVER_ENABLED)
if ALE_DRIVER_ENABLED:
    ale_config = dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        multiblock_cross_seam_rezone_enabled=True,
        multiblock_scaled_reference_enabled=SCALED_REFERENCE_ENABLED,
        multiblock_differential_reference_enabled=DIFFREF_ENABLED,
        multiblock_lagrangian_bulk_center_patch_reference_enabled=(
            CENTER_PATCH_REFERENCE_ENABLED
        ),
        multiblock_path_admissibility_enabled=True,
        path_admissibility_floor=0.01,
        dt_rejection_factor=0.5,
        max_dt_rejections=8,
        debug_per_remap_log=DEBUG_PER_REMAP_LOG,
        conservative_remap_enabled=CONSERVATIVE_REMAP_ENABLED,
        conservative_remap_order=CONSERVATIVE_REMAP_ORDER,
        conservative_remap_target=CONSERVATIVE_REMAP_TARGET,
    )
    if DIFFREF_ENABLED and DIFFREF_BAND_COUNT is not None:
        ale_config["multiblock_differential_reference_band_count"] = DIFFREF_BAND_COUNT
    if DIFFREF_ENABLED and DIFFREF_NU is not None:
        ale_config["multiblock_differential_reference_nu"] = DIFFREF_NU
    if SWEPT_VOLUME_SIGN_FIXED:
        ale_config["swept_volume_sign_fixed"] = SWEPT_VOLUME_SIGN_FIXED
    if REFERENCE_BARRIER:
        ale_config["reference_barrier_enabled"] = True
        ale_config["driver_retry_reference_barrier_enabled"] = True
    if CORE_FREEZE:
        ale_config.update(
            core_freeze_enabled=True,
            core_freeze_source="gas_tracer",
            core_freeze_tracer_cut=CORE_FREEZE_CUT,
            core_freeze_halo_layers=CORE_FREEZE_HALO,
            core_freeze_apply_to_axis_rezone=True,
        )
if ALE_IDENTITY:
    ale_config["ale_identity_mode"] = True
if ALE_MOVER_DIAG:
    ale_config["ale_mover_diag"] = True
if ALE_PRESERVE_VEL_CARRY:
    ale_config["ale_preserve_lagrangian_velocity_carry"] = True
if CENTRAL_PSEUDO_CORE_ENABLED:
    ale_config.update(
        central_pseudo_core_enabled=True,
        central_pseudo_core_s_c=CENTRAL_PSEUDO_CORE_S_C_CM,
        central_pseudo_core_ring_absorption_enabled=(CASE == "D"),
    )
    # Namelist-path mode for the core1d sub-model (promotion package Wave A):
    # emit the promoted parameters through the namelist instead of relying on
    # the historical TENRYU_I1B_CORE_1D_* C++ envs. Default off; the
    # env-override convention keeps existing scripts working unchanged.
    if _env_float(PREFIX + "DECK_CORE1D_NAMELIST", 1.0 if CASE == "D" else 0.0) > 0.0:
        ale_config.update(
            central_pseudo_core_core1d_enabled=True,
            central_pseudo_core_core1d_build_shells=int(
                _env_float(PREFIX + "DECK_CORE1D_BUILD_SHELLS", 128 if CASE == "D" else 48)),
            central_pseudo_core_core1d_split_append=int(
                _env_float(PREFIX + "DECK_CORE1D_SPLIT_APPEND", 64 if CASE == "D" else 0)),
            central_pseudo_core_core1d_av_c1=_env_float(
                PREFIX + "DECK_CORE1D_AV_C1", 0.5),
            central_pseudo_core_core1d_av_c2=_env_float(
                PREFIX + "DECK_CORE1D_AV_C2", 4.0),
            central_pseudo_core_core1d_dist_append=_env_bool(
                PREFIX + "DECK_CORE1D_DIST_APPEND", False),
        )
    # Namelist-path mode for the absorption schedules, terminal endgame and
    # remap conservation gate (promotion Waves B/C/D verification).
    if _env_float(PREFIX + "DECK_PC_NAMELIST", 1.0 if CASE == "D" else 0.0) > 0.0:
        ale_config.update(
            central_pseudo_core_spherical_absorb_gasfront=True,
            central_pseudo_core_spherical_absorb_alpha=_env_float(
                PREFIX + "DECK_PC_ABSORB_ALPHA", 0.90),
            central_pseudo_core_spherical_absorb_pjump=_env_float(
                PREFIX + "DECK_PC_ABSORB_PJUMP", 2.0),
            central_pseudo_core_mixed_absorb_enabled=True,
            central_pseudo_core_absorb_watch_rows=int(
                _env_float(PREFIX + "DECK_PC_WATCH_ROWS", 2)),
            central_pseudo_core_terminal_absorb_enabled=True,
            remap_mass_closure_reject_tol=_env_float(
                PREFIX + "DECK_PC_CLOSURE_TOL", 1.0e-8),
        )
    # Namelist-path mode for the certified transport/robustness flags
    # (promotion Wave E verification).
    if _env_float(PREFIX + "DECK_FLAGS_NAMELIST", 1.0 if CASE == "D" else 0.0) > 0.0:
        ale_config.update(
            csr_optionb_coherent_enabled=True,
            csr_optionb_velocity_remap_enabled=True,
            pole_axis_bbsw_enabled=True,
            axis_contact_guard_enabled=True,
            mass_floor_absorb_enabled=True,
            interior_patch_remap_enabled=True,
        )
if AXIS_REZONE:
    ale_config["axis_rezone_enabled"] = True
if _env_bool(PREFIX + "CONV_REZONE", _CASE_D):
    ale_config["conv_rezone_enabled"] = True

hydro_config = dict(
    enabled=True,
    boundary_2d=dict(
        r_inner="axis",
        r_outer="pressure",
        z_bottom="reflect",
        z_top="reflect",
    ),
    boundary_pressure=boundary_pressure,
    av_model=os.environ.get(PREFIX + "AV_MODEL", "csw_edge"),
    subzonal_pressure_enabled=_env_bool(
        PREFIX + "SUBZONAL_PRESSURE", True),
    av_C1=_env_float("TENRYU_I1B_DISC_AV_C1", 0.5 if CASE == "D" else 0.1),
    av_C2=_env_float("TENRYU_I1B_DISC_AV_C2", 4.0 if CASE == "D" else 1.5),
    csw_limiter_enabled=_env_bool(PREFIX + "CSW_LIMITER_ENABLED", True),
    csw_zero_uniform_compression=_env_bool(
        PREFIX + "CSW_ZERO_UNIF_COMP", True),
    anti_hourglass_kappa=_env_float("TENRYU_I1B_DISC_ANTI_HOURGLASS_KAPPA", 0.05),
    trial_volume_cfl_enabled=True,
    mesh_quality_dt_cfl_enabled=True,
    ring7_quotient_enabled=RING7_QUOTIENT,
    driver_full_step_retry_enabled=DRIVER_RETRY,
    driver_full_step_retry_max_attempts=DRIVER_RETRY_MAX_ATTEMPTS,
    driver_retry_use_suggested_dt_enabled=True,
)
if TOTAL_ENERGY_REMAP:
    hydro_config["total_energy_remap_2d_rz"] = TOTAL_ENERGY_REMAP

diagnostics_config = dict(
    phase_resolved_energy=True,
    conservation=dict(enabled=True),
    mesh_quality_min=dict(enabled=True),
    production_audit=dict(
        enabled=True,
        tier="A",
        gcl=dict(enabled=True),
        positivity=dict(enabled=True, fatal_on_neg=False),
        audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
    ),
)
if HOTSPOT_GAS_TRACER:
    diagnostics_config["hotspot_gas"] = dict(enabled=True, R_g_cm=R_G)

Numerics(
    radiation_thermal_subcycle=False,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL_HYDRO,
        growth_factor=1.05,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=hydro_config,
    ale=ale_config,
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=diagnostics_config,
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
    safety=dict(energy_fatal=False, nan_fatal=True, energy_threshold=1.0),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "Pe", "Pi"],
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0))
