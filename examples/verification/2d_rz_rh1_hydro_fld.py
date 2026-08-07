import json
import math
import os
from pathlib import Path

from tenryu_namelist import *


# TENRYU 2D_RZ RH1 hydro+FLD deck.
# Manual-use modes are selected with TENRYU_RH1_MODE.

RH1_SUBSHOCK_REFERENCE_TABLE_DEFAULT = Path(
    "tests/verification/data/rh1_subshock_reference/rh1_subshock_M3.json"
)
LE_PROXY_TABLE_KEYS = (
    "x_cm",
    "rho_g_per_cc",
    "u_cm_per_s",
    "T_mat_eV",
    "T_rad_eV",
    "E_r_erg_per_cc",
    "F_r_erg_per_cm2_s",
)
LE_PROXY_TABLE_NESTED_KEYS = {
    "x_cm": "x_cm",
    "rho_g_per_cc": "rho_g_per_cc",
    "u_cm_per_s": "u_cm_per_s",
    "T_mat_eV": "T_eV",
    "T_rad_eV": "T_rad_eV",
    "E_r_erg_per_cc": "E_rad_erg_per_cm3",
    "F_r_erg_per_cm2_s": "F_rad_erg_per_cm2_s",
}


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_RH1_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


def _env_float(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


def _env_int(name, default):
    if name in os.environ:
        return int(os.environ[name])
    return int(default)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _reference_table_values(payload, key):
    values = payload.get(key)
    if values is None:
        table = payload.get("table", {})
        values = (
            table.get(LE_PROXY_TABLE_NESTED_KEYS[key])
            if isinstance(table, dict)
            else None
        )
    if values is None:
        raise RuntimeError(f"RH1 subshock reference table missing required key: {key}")
    return [float(value) for value in values]


def _load_rh1_subshock_reference_table(path):
    if not path.exists():
        raise RuntimeError(
            "lowrie_edwards_table_init_shock_frame requires RH1 subshock reference table at "
            f"{path}; set TENRYU_RH1_SUBSHOCK_REFERENCE_TABLE to override"
        )
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return {key: _reference_table_values(payload, key) for key in LE_PROXY_TABLE_KEYS}


MODE = os.environ.get("TENRYU_RH1_MODE", "shock_tube_grey").lower().replace("-", "_")
VALID_MODES = (
    "shock_tube_grey",
    "planar_radiative_shock",
    "cylindrical_blast_with_rad",
    "optically_thin_precursor",
    "multigroup_tmat",
    "lowrie_edwards_energy_proxy",
    "lowrie_edwards_table_init_shock_frame",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_RH1_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_RH1_GRID", ""))
NR = _env_int("TENRYU_RH1_NR", grid_env[0] if grid_env else 64)
NZ = _env_int("TENRYU_RH1_NZ", grid_env[1] if grid_env else 128)
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_RH1_NR and TENRYU_RH1_NZ must be positive")

SEED = _env_int("TENRYU_RH1_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_RH1_OUTDIR", "./build/output_verify_2d_rz_rh1")
PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_RH1_PRODUCTION_AUDIT_TIER", "none")
if PRODUCTION_AUDIT_TIER not in ("none", "A", "B"):
    raise ValueError('TENRYU_RH1_PRODUCTION_AUDIT_TIER must be one of: "none", "A", "B"')
MAX_STEPS = _env_int("TENRYU_RH1_MAX_STEPS", 1000000)
RH1_ALE = _env_bool("TENRYU_RH1_ALE", False)
RH1_RETRY = _env_bool("TENRYU_RH1_RETRY", False)
RH1_RETRY_MAX_ATTEMPTS = _env_int("TENRYU_RH1_RETRY_MAX_ATTEMPTS", 3)
RH1_RETRY_ACTIVE_REPAIR = _env_bool("TENRYU_RH1_RETRY_ACTIVE_REPAIR", False)
RH1_CORNER_BALANCE_THRESHOLD = _env_float("TENRYU_RH1_CORNER_BALANCE_THRESHOLD", 0.01)
RH1_LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_RH1_LOCAL_BOUNDARY_REPAIR", False)
RH1_MULTI_NODE_BOUNDARY_REPAIR = _env_bool("TENRYU_RH1_MULTI_NODE_BOUNDARY_REPAIR", False)
RH1_EMERGENCY_CELL_DEACTIVATION = _env_bool("TENRYU_RH1_EMERGENCY_CELL_DEACTIVATION", False)
ALE_ENABLED = RH1_ALE or RH1_RETRY_ACTIVE_REPAIR
ALE_EVERY_N_STEPS = _env_int("TENRYU_RH1_ALE_EVERY_N_STEPS", 5)
AXIS_REPAIR_MODE = os.environ.get("TENRYU_RH1_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_SCHEME = os.environ.get("TENRYU_RH1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_RH1_REMAP_MS2_LIMITER", "van_leer")

MODE_T_END_DEFAULTS = dict(
    shock_tube_grey=1.0e-9,
    planar_radiative_shock=1.0e-9,
    cylindrical_blast_with_rad=1.0e-9,
    optically_thin_precursor=1.0e-9,
    multigroup_tmat=5.0e-10,
    lowrie_edwards_energy_proxy=4.0e-10,
    lowrie_edwards_table_init_shock_frame=4.0e-10,
)
MODE_RHO_DEFAULTS = dict(
    shock_tube_grey=1.0,
    planar_radiative_shock=1.0,
    cylindrical_blast_with_rad=1.0e-3,
    optically_thin_precursor=1.0e-5,
    multigroup_tmat=1.0,
    lowrie_edwards_energy_proxy=1.0,
    lowrie_edwards_table_init_shock_frame=1.0,
)
MODE_KAPPA_DEFAULTS = dict(
    shock_tube_grey=10.0,
    planar_radiative_shock=100.0,
    cylindrical_blast_with_rad=10.0,
    optically_thin_precursor=0.01,
    multigroup_tmat=0.0,
    lowrie_edwards_energy_proxy=1.0,
    lowrie_edwards_table_init_shock_frame=1.0,
)
MODE_BLAST_E_DEFAULTS = dict(
    cylindrical_blast_with_rad=1.0e8,
    optically_thin_precursor=1.0e6,
)

T_END = _env_float("TENRYU_RH1_T_END_S", MODE_T_END_DEFAULTS[MODE])
DT_INITIAL = _env_float("TENRYU_RH1_DT_INITIAL_S", min(1.0e-13, T_END))
DT_MAX = _env_float("TENRYU_RH1_DT_MAX_S", max(T_END / 20.0, DT_INITIAL))
if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_RH1 time controls must be positive")

RHO_GCC = _env_float("TENRYU_RH1_RHO_GCC", MODE_RHO_DEFAULTS[MODE])
KAPPA_A = _env_float("TENRYU_RH1_KAPPA_A_CM2_G", MODE_KAPPA_DEFAULTS[MODE])
AV_C2 = _env_float("TENRYU_RH1_AV_C2", 1.5)
CFL = _env_float("TENRYU_RH1_CFL", 0.3)
BLAST_CELLS = _env_int("TENRYU_RH1_BLAST_CELLS", 4)
TMAT_FILE = os.environ.get("TENRYU_RH1_TMAT_FILE", "TMAT-H5/CD_gray6_bounded.tmat.h5")
TMAT_GROUP_BOUNDS_EV = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0] if MODE == "multigroup_tmat" else [0.0, 1.0e6]
N_GROUPS = len(TMAT_GROUP_BOUNDS_EV) - 1 if MODE == "multigroup_tmat" else 1
PRODUCTION_AUDIT = dict(enabled=False, tier="none")
if PRODUCTION_AUDIT_TIER in ("A", "B"):
    PRODUCTION_AUDIT = dict(
        enabled=True,
        tier=PRODUCTION_AUDIT_TIER,
        gcl=dict(enabled=False),
        positivity=dict(enabled=False),
        audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
    )

R_MIN = 0.0
R_MAX = _env_float("TENRYU_RH1_R_MAX_CM", 1.0)
Z_MIN_ENV = os.environ.get("TENRYU_RH1_Z_MIN_CM")
Z_MAX_ENV = os.environ.get("TENRYU_RH1_Z_MAX_CM")
Z_MIN = float(Z_MIN_ENV) if Z_MIN_ENV is not None else -1.0
Z_MAX = float(Z_MAX_ENV) if Z_MAX_ENV is not None else 1.0

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
CV_I = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
CV_E = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
R_GAS = EV_TO_ERG / (A * M_P)
LE_PROXY_MACH = _env_float("TENRYU_RH1_LE_PROXY_MACH", 3.0)
PISTON_VZ_DEFAULT = LE_PROXY_MACH * math.sqrt(GAMMA * R_GAS * 1.0) if MODE == "lowrie_edwards_energy_proxy" else 1.0e6
PISTON_VZ = _env_float("TENRYU_RH1_PISTON_VZ_CM_S", PISTON_VZ_DEFAULT)

LE_PROXY_TABLE = None
LE_PROXY_X = None
if MODE == "lowrie_edwards_table_init_shock_frame":
    le_proxy_table_path = Path(
        os.environ.get(
            "TENRYU_RH1_SUBSHOCK_REFERENCE_TABLE",
            str(RH1_SUBSHOCK_REFERENCE_TABLE_DEFAULT),
        )
    )
    LE_PROXY_TABLE = _load_rh1_subshock_reference_table(le_proxy_table_path)
    LE_PROXY_X = LE_PROXY_TABLE["x_cm"]
    if len(LE_PROXY_X) < 2:
        raise RuntimeError("RH1 subshock reference table must contain at least two x_cm points")
    if Z_MIN_ENV is None and Z_MAX_ENV is None:
        Z_MIN = LE_PROXY_X[0]
        Z_MAX = LE_PROXY_X[-1]
    elif not (
        math.isclose(Z_MIN, LE_PROXY_X[0], rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(Z_MAX, LE_PROXY_X[-1], rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise RuntimeError(
            "lowrie_edwards_table_init_shock_frame requires mesh z_min/z_max to match "
            "the RH1 subshock reference x range: "
            f"z=({Z_MIN}, {Z_MAX}) table=({LE_PROXY_X[0]}, {LE_PROXY_X[-1]})"
        )

if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("TENRYU_RH1 geometry bounds must satisfy r_max>r_min and z_max>z_min")

DR = (R_MAX - R_MIN) / NR
R_BLAST = BLAST_CELLS * DR
Z_LEN = Z_MAX - Z_MIN
V_PATCH = math.pi * R_BLAST * R_BLAST * Z_LEN
RHO_BLAST = RHO_GCC
if MODE == "shock_tube_grey":
    RHO_BLAST = 1.0
DEFAULT_BLAST_E_ERG = MODE_BLAST_E_DEFAULTS.get(MODE, 1.0e13)
BLAST_E_ERG = _env_float("TENRYU_RH1_BLAST_E_ERG", DEFAULT_BLAST_E_ERG)
E_INT_BLAST = BLAST_E_ERG / max(RHO_BLAST * V_PATCH, 1.0e-300)
T_BLAST_EV = E_INT_BLAST / (CV_I + CV_E)

CASE_NAME = (
    f"rh1_{MODE}_nr{NR}_nz{NZ}_rho{_safe_float_token(RHO_GCC)}"
    f"_ka{_safe_float_token(KAPPA_A)}_g{N_GROUPS}_seed{SEED}"
    + ("_ale" if ALE_ENABLED else "")
)

print(
    "[deck:2d_rz_rh1_hydro_fld] "
    f"mode={MODE} nr={NR} nz={NZ} groups={N_GROUPS} seed={SEED} outdir={OUTDIR} "
    f"rho_gcc={RHO_GCC} kappa_a_cm2_g={KAPPA_A} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} av_C2={AV_C2} "
    f"piston_vz_cm_s={PISTON_VZ} blast_E_erg={BLAST_E_ERG} "
    f"blast_cells={BLAST_CELLS} R_blast_cm={R_BLAST} tmat_file={TMAT_FILE} "
    f"ale_enabled={ALE_ENABLED}"
)


def interpolate_le_proxy(z, key):
    xs = LE_PROXY_X
    values = LE_PROXY_TABLE[key]
    if z <= xs[0]:
        return values[0]
    if z >= xs[-1]:
        return values[-1]
    lo = 0
    hi = len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= z:
            lo = mid
        else:
            hi = mid
    weight = (z - xs[lo]) / (xs[hi] - xs[lo])
    return values[lo] + weight * (values[hi] - values[lo])


def rho_init(r, z):
    del r
    if MODE == "lowrie_edwards_table_init_shock_frame":
        return interpolate_le_proxy(z, "rho_g_per_cc")
    if MODE == "shock_tube_grey":
        return 1.0 if z < 0.0 else 0.125
    if MODE == "multigroup_tmat":
        del z
        return RHO_GCC
    return RHO_GCC


def Te_init(r, z):
    if MODE == "lowrie_edwards_table_init_shock_frame":
        del r
        return interpolate_le_proxy(z, "T_mat_eV")
    if MODE == "shock_tube_grey":
        return 10.0 if z < 0.0 else 2.0
    if MODE == "multigroup_tmat":
        hot = math.exp(-((r / 0.28) ** 2 + ((z - 0.15) / 0.42) ** 2))
        axial = 0.5 * (1.0 + math.tanh((z + 0.25) / 0.28))
        return 25.0 + 170.0 * hot + 55.0 * axial
    if MODE in ("planar_radiative_shock", "lowrie_edwards_energy_proxy"):
        del r, z
        return 1.0
    if r < R_BLAST:
        return T_BLAST_EV
    return 1.0


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r
    if MODE == "lowrie_edwards_table_init_shock_frame":
        return (0.0, interpolate_le_proxy(z, "u_cm_per_s"))
    if MODE in ("planar_radiative_shock", "lowrie_edwards_energy_proxy"):
        return (0.0, PISTON_VZ)
    del z
    return (0.0, 0.0)


def h_volfrac(r, z):
    del r, z
    return 1.0


HYDRO_BOUNDARY = dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect")
if MODE == "planar_radiative_shock":
    HYDRO_BOUNDARY = dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="fixed")
elif MODE == "lowrie_edwards_energy_proxy":
    HYDRO_BOUNDARY = dict(r_inner="axis", r_outer="reflect", z_bottom="free", z_top="fixed")
elif MODE == "lowrie_edwards_table_init_shock_frame":
    z_table = LE_PROXY_TABLE["x_cm"]
    rho_table = LE_PROXY_TABLE["rho_g_per_cc"]
    u_table = LE_PROXY_TABLE["u_cm_per_s"]
    T_table = LE_PROXY_TABLE["T_mat_eV"]
    z_min_idx = 0
    z_max_idx = len(z_table) - 1
    HYDRO_BOUNDARY = dict(
        r_inner="axis",
        r_outer="reflect",
        z_bottom=dict(
            type="state_supply",
            rho_g_per_cc=rho_table[z_min_idx],
            u_z_cm_per_s=u_table[z_min_idx],
            T_eV=T_table[z_min_idx],
        ),
        z_top=dict(
            type="state_supply",
            rho_g_per_cc=rho_table[z_max_idx],
            u_z_cm_per_s=u_table[z_max_idx],
            T_eV=T_table[z_max_idx],
        ),
    )
elif MODE in ("cylindrical_blast_with_rad", "optically_thin_precursor", "multigroup_tmat"):
    HYDRO_BOUNDARY = dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect")

FLD_BOUNDARY = dict(inner_r="reflect", outer_r="vacuum", z="reflect")
if MODE == "lowrie_edwards_table_init_shock_frame":
    FLD_BOUNDARY = dict(
        inner_r="reflect",
        outer_r="vacuum",
        z="state_supply",
        z_bottom="state_supply",
        z_top="state_supply",
    )

plot_fields = [
    "rho",
    "Te",
    "Ti",
    "ee",
    "ei",
    "Pe",
    "Pi",
    "rad_E",
]

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_RH1_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale" if ALE_ENABLED else "lagrangian",
)

Materials(
    low_density_extrapolation=MODE == "multigroup_tmat",
    materials=[
        Material(
            name="CD" if MODE == "multigroup_tmat" else "H",
            A=7.0 if MODE == "multigroup_tmat" else A,
            Z=3.5 if MODE == "multigroup_tmat" else ZBAR,
            eos=dict(model="tmat", file=TMAT_FILE)
            if MODE == "multigroup_tmat"
            else dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(
                model="tmat",
                file=TMAT_FILE,
                lambda_method="finite_difference",
                lambda_fd_delta_rel=1.0e-4,
                lambda_fd_abs_min=1.0e-6,
                f_min=1.0e-4,
            )
            if MODE == "multigroup_tmat"
            else dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="tabular") if MODE == "multigroup_tmat" else dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac={("CD" if MODE == "multigroup_tmat" else "H"): h_volfrac},
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

ALE_CONFIG = dict(enabled=False)
if ALE_ENABLED:
    ALE_CONFIG = dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=True,
        remap_damage_dmax=0.05,
        remap_damage_axis_eta=0.02,
        remap_damage_axis_budget_enabled=True,
        remap_damage_axis_budget_factor=2.0,
        local_boundary_repair_enabled=RH1_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=RH1_MULTI_NODE_BOUNDARY_REPAIR,
        emergency_cell_deactivation_enabled=RH1_EMERGENCY_CELL_DEACTIVATION,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        ke_conservation_closure=True,
        ke_conservation_closure_audit=False,
        ke_closure_redistribute_floor=True,
    )

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=HYDRO_BOUNDARY,
        av_C1=0.1,
        av_C2=AV_C2,
        volume_rate_cfl_enabled=RH1_ALE,
        volume_rate_cfl_threshold=0.5,
        trial_volume_cfl_enabled=RH1_ALE,
        trial_volume_cfl_floor_fraction=0.05,
        trial_volume_cfl_shrink_fraction=0.5,
        corner_jacobian_ale_trigger_enabled=RH1_ALE,
        corner_jacobian_floor_eps=1.0e-6,
        corner_jacobian_ale_trigger_scale=0.5,
        driver_full_step_retry_enabled=RH1_RETRY,
        driver_full_step_retry_max_attempts=RH1_RETRY_MAX_ATTEMPTS,
        driver_retry_active_mesh_repair_enabled=RH1_RETRY_ACTIVE_REPAIR,
        driver_retry_corner_balance_threshold=RH1_CORNER_BALANCE_THRESHOLD,
    ),
    conduction=dict(enabled=False),
    ale=ALE_CONFIG,
    diagnostics=dict(phase_resolved_energy=True, production_audit=PRODUCTION_AUDIT),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0e-3,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

radiation_kwargs = dict(
    enabled=True,
    mode="multigroup_diffusion",
    groups=N_GROUPS,
    group_bounds_eV=TMAT_GROUP_BOUNDS_EV,
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="none"
        if MODE in ("lowrie_edwards_energy_proxy", "lowrie_edwards_table_init_shock_frame")
        else "levermore_pomraning",
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        linear_solver_2d="cusparse_cg_jacobi",
        boundary=FLD_BOUNDARY,
    ),
)
if MODE == "multigroup_tmat":
    radiation_kwargs["compute_T_range_eV"] = [10.0, 500.0]

Radiation(**radiation_kwargs)

Laser(enabled=False)

Output(
    directory=OUTDIR,
    plot_every=MAX_STEPS,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=T_END,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=plot_fields,
)

Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
