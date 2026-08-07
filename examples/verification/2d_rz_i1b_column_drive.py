"""Isolated smeared-column pressure-drive I1-B discriminator deck."""

import os

from tenryu_namelist import *

from i1b_polar_common import c2_smoothstep


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


PREFIX = "TENRYU_I1B_COL_"
CASE_NAME_BASE = "2d_rz_i1b_column_drive"

NR = _env_int(PREFIX + "NR", 20)
NZ = _env_int(PREFIX + "NZ", 8)
SEED = _env_int(PREFIX + "SEED", 12345)
R0 = _env_float(PREFIX + "R0_CM", 10.0)
L = 0.052
LZ = _env_float(PREFIX + "LZ_CM", 0.013)
R1 = R0 + L

A_AMU = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
EV_TO_ERG = 1.602176634e-12
AMU_G = 1.66053906660e-24

RHO_EXT = 1.0e-4
RHO_ABL = 1.05
RHO_SHELL = 0.25
RHO_GAS = 1.0e-4
T_BALANCED_EV = 200.0

D_BUFFER = 0.003
D_ABLATOR = 0.0065
D_SHELL = 0.017
SMOOTH_WIDTH = _env_float(PREFIX + "SMOOTH_WIDTH_CM", 1.575e-3)

P_PEAK = _env_float(PREFIX + "P_PEAK_DYN_CM2", 6.0e13)
T_RISE = 0.2e-9
T_HOLD_END = 1.2e-9
T_FALL_END = 1.35e-9

T_END = _env_float(PREFIX + "T_END_S", 1.45e-9)
DT_INITIAL = 2.5e-13
DT_MAX = 5.0e-12
CFL_HYDRO = 0.2
MAX_STEPS = _env_int(PREFIX + "MAX_STEPS", 2000000)
PLOT_EVERY_STEPS = _env_int(PREFIX + "PLOT_EVERY_STEPS", 1000)
DRIVER_RETRY = _env_bool(PREFIX + "RETRY", True)
DRIVER_RETRY_MAX_ATTEMPTS = _env_int(PREFIX + "RETRY_MAX_ATTEMPTS", 8)
HISTORY_EVERY_STEPS = 1
OUTDIR = os.environ.get(PREFIX + "OUTDIR", "./build/output_verify_2d_rz_i1b_column_drive")

if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_I1B_COL_NR and TENRYU_I1B_COL_NZ must be > 0")
if not (R0 > 0.0 and L > 0.0 and LZ > 0.0):
    raise ValueError("TENRYU_I1B_COL geometry extents must be > 0")
if min(RHO_EXT, RHO_ABL, RHO_SHELL, RHO_GAS) <= 0.0:
    raise ValueError("TENRYU_I1B_COL densities must be > 0")
if SMOOTH_WIDTH < 0.0:
    raise ValueError("TENRYU_I1B_COL_SMOOTH_WIDTH_CM must be >= 0")
if P_PEAK < 0.0:
    raise ValueError("TENRYU_I1B_COL_P_PEAK_DYN_CM2 must be >= 0")
if not (T_END > 0.0 and DT_INITIAL > 0.0 and DT_MAX > 0.0):
    raise ValueError("TENRYU_I1B_COL time controls must be > 0")
if MAX_STEPS <= 0 or PLOT_EVERY_STEPS < 0:
    raise ValueError("TENRYU_I1B_COL step/cadence controls are invalid")


def _pressure_from_equal_t(rho_gcc, t_ev):
    return rho_gcc * EV_TO_ERG * (ZBAR + 1.0) * t_ev / (A_AMU * AMU_G)


def _equal_t_from_pressure(p_dyn_cm2, rho_gcc):
    return p_dyn_cm2 * A_AMU * AMU_G / (
        rho_gcc * EV_TO_ERG * (ZBAR + 1.0)
    )


P_BASE = _pressure_from_equal_t(RHO_EXT, T_BALANCED_EV)

CASE_NAME = (
    f"{CASE_NAME_BASE}_nr{NR}_nz{NZ}_r0{_safe_float_token(R0)}"
    f"_seed{SEED}"
)

print(
    "[deck:2d_rz_i1b_column_drive] "
    "topology_scheme=single_block logical_mesh_2d=rectangular_rz "
    f"nr={NR} nz={NZ} seed={SEED} r0_cm={R0} r1_cm={R1} "
    f"L_cm={L} LZ_cm={LZ} depth_interfaces_cm="
    f"buffer:{D_BUFFER},ablator:{D_ABLATOR},shell:{D_SHELL} "
    f"smooth_width_cm={SMOOTH_WIDTH} rho_ext_gcc={RHO_EXT} "
    f"rho_ablator_gcc={RHO_ABL} rho_shell_gcc={RHO_SHELL} "
    f"rho_gas_gcc={RHO_GAS} P_base_dyn_cm2={P_BASE} "
    f"T_balanced_ref_eV={T_BALANCED_EV} gamma={GAMMA} A={A_AMU} Z={ZBAR} "
    f"P_peak_dyn_cm2={P_PEAK} rise_s={T_RISE} hold_end_s={T_HOLD_END} "
    f"fall_end_s={T_FALL_END} t_end_s={T_END} dt_initial_s={DT_INITIAL} "
    f"dt_max_s={DT_MAX} cfl_hydro={CFL_HYDRO} max_steps={MAX_STEPS} "
    f"plot_every_steps={PLOT_EVERY_STEPS} history_every_steps={HISTORY_EVERY_STEPS} "
    "motion=lagrangian ale_enabled=False radiation_enabled=False "
    "conduction_enabled=False laser_enabled=False "
    f"av_model={os.environ.get(PREFIX + 'AV_MODEL', 'csw_edge')} "
    f"subzonal_pressure_enabled={_env_bool(PREFIX + 'SUBZONAL', True)} "
    f"csw_limiter_enabled={_env_bool(PREFIX + 'CSW_LIMITER', True)} "
    f"av_C1={_env_float(PREFIX + 'AV_C1', 0.5)} "
    f"av_C2={_env_float(PREFIX + 'AV_C2', 4.0)} "
    f"trial_volume_cfl_enabled=True mesh_quality_dt_cfl_enabled=True "
    f"driver_full_step_retry_enabled={DRIVER_RETRY} "
    f"driver_full_step_retry_max_attempts={DRIVER_RETRY_MAX_ATTEMPTS} "
    f"outdir={OUTDIR}"
)


def _blend(a, b, x):
    return a + (b - a) * c2_smoothstep(x)


def _transition(a, b, dd, d0):
    if SMOOTH_WIDTH <= 0.0:
        return a if dd <= d0 else b
    return _blend(a, b, (dd - (d0 - SMOOTH_WIDTH)) / (2.0 * SMOOTH_WIDTH))


def _rho_layered_depth(dd):
    if SMOOTH_WIDTH <= 0.0:
        if dd < D_BUFFER:
            return RHO_EXT
        if dd < D_ABLATOR:
            return RHO_ABL
        if dd < D_SHELL:
            return RHO_SHELL
        return RHO_GAS

    if dd < D_BUFFER - SMOOTH_WIDTH:
        return RHO_EXT
    if dd < D_BUFFER + SMOOTH_WIDTH:
        return _transition(RHO_EXT, RHO_ABL, dd, D_BUFFER)
    if dd < D_ABLATOR - SMOOTH_WIDTH:
        return RHO_ABL
    if dd < D_ABLATOR + SMOOTH_WIDTH:
        return _transition(RHO_ABL, RHO_SHELL, dd, D_ABLATOR)
    if dd < D_SHELL - SMOOTH_WIDTH:
        return RHO_SHELL
    if dd < D_SHELL + SMOOTH_WIDTH:
        return _transition(RHO_SHELL, RHO_GAS, dd, D_SHELL)
    return RHO_GAS


def boundary_pressure(t):
    if t <= 0.0:
        return 0.0
    if t <= T_RISE:
        return P_PEAK * t / T_RISE
    if t <= T_HOLD_END:
        return P_PEAK
    if t <= T_FALL_END:
        return P_PEAK * (1.0 - (t - T_HOLD_END) / (T_FALL_END - T_HOLD_END))
    return 0.0


def rho_init(r, z):
    del z
    return _rho_layered_depth(R1 - r)


def Te_init(r, z):
    rho = rho_init(r, z)
    return _equal_t_from_pressure(P_BASE, rho)


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


main_config = dict(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
)

mesh_config = dict(
    r_min=R0,
    r_max=R1,
    z_min=0.0,
    z_max=LZ,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="rectangular_rz",
    topology_scheme="single_block",
)

materials_config = dict(
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

hydro_config = dict(
    enabled=True,
    boundary_2d=dict(
        r_inner="reflect",
        r_outer="pressure",
        z_bottom="reflect",
        z_top="reflect",
    ),
    boundary_pressure=boundary_pressure,
    av_model=os.environ.get(PREFIX + "AV_MODEL", "csw_edge"),
    av_C1=_env_float(PREFIX + "AV_C1", 0.5),
    av_C2=_env_float(PREFIX + "AV_C2", 4.0),
    subzonal_pressure_enabled=_env_bool(PREFIX + "SUBZONAL", True),
    csw_limiter_enabled=_env_bool(PREFIX + "CSW_LIMITER", True),
    trial_volume_cfl_enabled=True,
    mesh_quality_dt_cfl_enabled=True,
    driver_full_step_retry_enabled=DRIVER_RETRY,
    driver_full_step_retry_max_attempts=DRIVER_RETRY_MAX_ATTEMPTS,
    driver_retry_use_suggested_dt_enabled=True,
)

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

Main(**main_config)
Mesh(**mesh_config)
Materials(**materials_config)

Geometry(
    volfrac=dict(gas=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
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
