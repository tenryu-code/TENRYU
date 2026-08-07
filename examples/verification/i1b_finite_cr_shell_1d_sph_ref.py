"""1D spherical Lagrangian reference for the I1-B finite-CR shell hotspot.

Measurement-only diagnostic deck.  It mirrors the finite-CR layered IC and
velocity/pressure drive formulas from 2d_rz_i1b_finite_cr_shell_hotspot.py,
without ALE, laser, radiation, conduction, PLIC, or EOS tables.
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


PREFIX = "TENRYU_I1B_1D_"
CASE_NAME_BASE = "i1b_finite_cr_shell_1d_sph_ref"
DRIVE = os.environ.get(PREFIX + "DRIVE", "vel").strip().lower()
DRIVE_ALIASES = {
    "vel": "vel",
    "velocity": "vel",
    "velocity-ic": "vel",
    "press": "press",
    "pressure": "press",
    "sustained-pressure": "press",
}
if DRIVE not in DRIVE_ALIASES:
    raise ValueError("TENRYU_I1B_1D_DRIVE must be one of vel, velocity, press, pressure")
DRIVE = DRIVE_ALIASES[DRIVE]

NR = _env_int(PREFIX + "NR", 512)
SEED = _env_int(PREFIX + "SEED", 12345)
S_MAX = _env_float(PREFIX + "S_MAX_CM", 0.07)
R_G = _env_float(
    PREFIX + "R_G_CM",
    _env_float(PREFIX + "R_G_FRAC", 0.50) * S_MAX,
)
R_S = _env_float(
    PREFIX + "R_S_CM",
    _env_float(PREFIX + "R_S_FRAC", 0.65) * S_MAX,
)
R_OUTER = _env_float(
    PREFIX + "R_OUTER_CM",
    _env_float(PREFIX + "R_OUTER_FRAC", 0.70) * S_MAX,
)
R_HS = _env_float(
    PREFIX + "R_HS_CM",
    _env_float(PREFIX + "R_HS_FRAC", 50.0 / 700.0) * S_MAX,
)

if NR <= 0:
    raise ValueError("TENRYU_I1B_1D_NR must be positive")
if not (0.0 < R_HS < R_G < R_S < R_OUTER < S_MAX):
    raise ValueError(
        "TENRYU_I1B_1D radii must satisfy 0<R_HS<R_G<R_S<R_OUTER<S_MAX"
    )

A_AMU = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
EV_TO_ERG = 1.602176634e-12
AMU_G = 1.66053906660e-24

RHO_GAS_FILL = _env_float(PREFIX + "RHO_GAS_FILL_GCC", 1.0e-4)
RHO_SHELL = _env_float(PREFIX + "RHO_SHELL_GCC", 0.25)
RHO_ABLATOR = _env_float(PREFIX + "RHO_ABLATOR_GCC", 1.05)
RHO_EXTERIOR = _env_float(PREFIX + "RHO_EXTERIOR_GCC", RHO_GAS_FILL)
T_GAS_BALANCED_EV = _env_float(PREFIX + "T_GAS_BALANCED_EV", 1.0e3)
ABLATOR_ENABLED = _env_bool(PREFIX + "ABLATOR_ENABLED", True)

if min(RHO_GAS_FILL, RHO_SHELL, RHO_ABLATOR, RHO_EXTERIOR) <= 0.0:
    raise ValueError("TENRYU_I1B_1D densities must be > 0")
if T_GAS_BALANCED_EV <= 0.0:
    raise ValueError("TENRYU_I1B_1D_T_GAS_BALANCED_EV must be > 0")


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
P_EXT = 0.0

if P_BASE <= 0.0:
    raise ValueError("TENRYU_I1B_1D_P0_DYN_CM2 must be > 0")
if GASP_SCALE <= 0.0:
    raise ValueError("TENRYU_I1B_1D_GASP_SCALE must be > 0")
if ETA <= 0.0:
    raise ValueError("TENRYU_I1B_1D_ETA must be > 0")

smooth_width_default = min(
    2.0 * S_MAX / float(max(NR, 1)),
    0.45 * (R_S - R_G),
    0.45 * (R_OUTER - R_S),
    0.25 * R_G,
)
SMOOTH_WIDTH = _env_float(PREFIX + "SMOOTH_WIDTH_CM", smooth_width_default)
if SMOOTH_WIDTH < 0.0:
    raise ValueError("TENRYU_I1B_1D_SMOOTH_WIDTH_CM must be >= 0")

T_END = _env_float(PREFIX + "T_END_S", 5.0e-9)
DT_INITIAL = _env_float(PREFIX + "DT_INITIAL_S", 2.5e-11)
DT_MAX = _env_float(PREFIX + "DT_MAX_S", 5.0e-11)
CFL_HYDRO = _env_float(PREFIX + "CFL_HYDRO", 0.2)
DRIVER_RETRY = _env_bool(PREFIX + "RETRY", True)
DRIVER_RETRY_MAX_ATTEMPTS = _env_int(PREFIX + "RETRY_MAX_ATTEMPTS", 8)
MAX_STEPS = _env_int(PREFIX + "MAX_STEPS", 40000)
PLOT_EVERY_STEPS = _env_int(PREFIX + "PLOT_EVERY_STEPS", 4)
HISTORY_EVERY_STEPS = _env_int(PREFIX + "HISTORY_EVERY_STEPS", 1)

if not (T_END > 0.0 and DT_INITIAL > 0.0 and DT_MAX > 0.0):
    raise ValueError("TENRYU_I1B_1D time controls must be > 0")
if MAX_STEPS <= 0 or PLOT_EVERY_STEPS < 0 or HISTORY_EVERY_STEPS < 0:
    raise ValueError("TENRYU_I1B_1D step/cadence controls are invalid")

DR = S_MAX / float(NR)


def _last_initial_cell_index_le(radius_cm):
    idx = int(math.floor(radius_cm / DR - 0.5))
    return max(-1, min(NR - 1, idx))


GAS_I0 = 0
GAS_I1 = _last_initial_cell_index_le(R_G)
SHELL_I0 = GAS_I1 + 1
SHELL_I1 = _last_initial_cell_index_le(R_S)
ABLATOR_I0 = SHELL_I1 + 1
ABLATOR_I1 = _last_initial_cell_index_le(R_OUTER)
if GAS_I1 < GAS_I0 or SHELL_I1 < SHELL_I0 or ABLATOR_I1 < ABLATOR_I0:
    raise ValueError("TENRYU_I1B_1D_NR is too small to resolve gas/shell/ablator ranges")

CASE_NAME = (
    f"{CASE_NAME_BASE}_{DRIVE}_nr{NR}_eta{_safe_float_token(ETA)}"
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
AREAL_DENSITY += (RHO_ABLATOR if ABLATOR_ENABLED else RHO_SHELL) * (
    R_OUTER - R_S
)
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
        "TENRYU_I1B_1D D ramp times must satisfy rise>0, hold_end>=rise, fall>0"
    )
D_IMPULSE_DURATION = 0.5 * D_T_R + (D_T_H - D_T_R) + 0.5 * D_T_D
D_P_PEAK_AUTO = REQUESTED_IMPULSE_AREA / max(D_IMPULSE_DURATION, 1.0e-300)
D_P_PEAK = _env_float_first(
    (PREFIX + "D_P_PEAK_DYN_CM2", PREFIX + "P_PEAK_DYN_CM2"),
    D_P_PEAK_AUTO,
)
if D_P_PEAK < 0.0:
    raise ValueError("TENRYU_I1B_1D_D_P_PEAK_DYN_CM2 must be >= 0")
REALIZED_IMPULSE_AREA = D_P_PEAK * D_IMPULSE_DURATION

REGION_LABELS = (
    "gas_fill",
    "shell",
    "ablator_density_layer",
    "exterior",
)
IC_PRESSURE_POLICY = "gas_fill_pressure_scaled_with_c2_transition_at_R_g"

print(
    "[deck:i1b_finite_cr_shell_1d_sph_ref] "
    f"env_prefix={PREFIX} drive={DRIVE} dimension=1D_SPH motion=lagrangian "
    f"nr={NR} seed={SEED} S_max_cm={S_MAX} r_max_cm={S_MAX} "
    f"passive_region_labels={','.join(REGION_LABELS)} "
    f"region_thresholds_cm=gas_fill:rr<=R_g({R_G}),"
    f"shell:R_g<rr<=R_s({R_S}),"
    f"ablator_density_layer:R_s<rr<=R_outer({R_OUTER}),"
    "exterior:rr>R_outer "
    f"r_hs_cm={R_HS} target_CR_hs={C_G} smooth_width_cm={SMOOTH_WIDTH} "
    f"rho_gas_fill_gcc={RHO_GAS_FILL} rho_shell_gcc={RHO_SHELL} "
    f"rho_ablator_density_layer_gcc={RHO_ABLATOR} "
    f"rho_exterior_gcc={RHO_EXTERIOR} ablator_enabled={ABLATOR_ENABLED} "
    f"P_base_dyn_cm2={P_BASE} gas_pressure_scale={GASP_SCALE} "
    f"ic_pressure_policy={IC_PRESSURE_POLICY} "
    f"T_gas_balanced_eV={_equal_t_from_pressure(P_BASE, RHO_GAS_FILL)} "
    f"gamma={GAMMA} A={A_AMU} Z={ZBAR} "
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
    f"gas_cell_i0={GAS_I0} gas_cell_i1={GAS_I1} "
    f"shell_cell_i0={SHELL_I0} shell_cell_i1={SHELL_I1} "
    f"ablator_cell_i0={ABLATOR_I0} ablator_cell_i1={ABLATOR_I1} "
    f"gas_interface_node_index={GAS_I1 + 1} "
    f"shell_interface_node_index={SHELL_I1 + 1} "
    "cell_range_rule=uniform_grid_initial_cell_center_rr<=threshold "
    "laser=False radiation=False conduction=False eos_table=False ale=False "
    f"outdir={OUTDIR}"
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
    p_gas = P_BASE * GASP_SCALE
    if SMOOTH_WIDTH <= 0.0:
        return p_gas if rr <= R_G else P_BASE
    if rr < R_G - SMOOTH_WIDTH:
        return p_gas
    if rr < R_G + SMOOTH_WIDTH:
        return _transition(p_gas, P_BASE, rr, R_G)
    return P_BASE


def boundary_pressure(t):
    if DRIVE == "press":
        return ablation_pressure_ramp(t, D_P_PEAK, D_T_R, D_T_H, D_T_D)
    return P_EXT


def rho_init(r):
    return _rho_layered(r)


def Te_init(r):
    return _equal_t_from_pressure(_pressure_layered(r), _rho_layered(r))


def Ti_init(r):
    return Te_init(r)


def _shell_velocity_weight(rr):
    if rr <= R_G:
        return 0.0
    inner = c2_smoothstep((rr - R_G) / (R_S - R_G))
    if SMOOTH_WIDTH <= 0.0:
        outer = 1.0 if rr <= R_OUTER else 0.0
    else:
        outer = 1.0 - c2_smoothstep((rr - R_OUTER) / SMOOTH_WIDTH)
    return inner * outer


def velocity_init(r):
    if DRIVE == "press":
        return 0.0
    return -U0_SHELL * _shell_velocity_weight(r)


Main(
    name=CASE_NAME,
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    nr=NR,
    grid="uniform",
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-4, Ti_floor_eV=1.0e-4),
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
    volfrac=dict(gas=lambda r: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

hydro_config = dict(
    enabled=True,
    boundary_1d="pressure" if DRIVE == "press" else "free",
    boundary_pressure=boundary_pressure,
    av_C1=0.1,
    av_C2=1.5,
    driver_full_step_retry_enabled=DRIVER_RETRY,
    driver_full_step_retry_max_attempts=DRIVER_RETRY_MAX_ATTEMPTS,
    driver_retry_use_suggested_dt_enabled=True,
)

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
    ale=dict(enabled=False),
    conduction=dict(enabled=False),
    plic=dict(enabled=False),
    diagnostics=dict(
        phase_resolved_energy=True,
        conservation=dict(enabled=True),
        icf=dict(enabled=True),
        hotspot_gas=dict(enabled=True, R_g_cm=R_G),
    ),
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
