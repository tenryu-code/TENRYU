import math
import os

from tenryu_namelist import *


# TENRYU 2D_RZ E1 TMAT dry-run deck.
# Manual-use modes are selected with TENRYU_E1_MODE.


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_E1_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


def _env_float(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


MODE = os.environ.get("TENRYU_E1_MODE", "static_cd_uniform").lower().replace("-", "_")
VALID_MODES = (
    "static_cd_target",
    "static_cd_uniform",
    "isentropic",
    "noh_weak",
    "ale_remap",
    "cold_corner_sentinel",
    "cold_corner_guard_negative",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_E1_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_E1_GRID", ""))
NR = int(os.environ.get("TENRYU_E1_NR", str(grid_env[0] if grid_env else 64)))
NZ = int(os.environ.get("TENRYU_E1_NZ", str(grid_env[1] if grid_env else 128)))
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_E1_NR and TENRYU_E1_NZ must be positive")

SEED = int(os.environ.get("TENRYU_E1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_E1_OUTDIR", "./build/output_verify_2d_rz_e1_tmat")
MAX_STEPS = int(os.environ.get("TENRYU_E1_MAX_STEPS", "1000000"))
RHO_DENSE_GCC = float(os.environ.get("TENRYU_E1_RHO_GCC", "1.05"))

A_CD = 7.0
Z_CD = 3.5
RHO_FLOOR = 1.0e-9
TE_FLOOR = 0.1
TI_FLOOR = 0.1
CS_WARM_CD = 1.0e6
V_ISENTROPIC = 1.0e-5 * CS_WARM_CD
V_NOH_WEAK = 1.0e-3 * CS_WARM_CD
V_EXPANSION = 1.0e-5 * CS_WARM_CD

R_MIN = 0.0
R_MAX = 1.0
Z_MIN = -1.0
Z_MAX = 1.0
TARGET_RADIUS = 0.5
NOH_R_INNER = 0.05 * R_MAX
NOH_LR = R_MAX - NOH_R_INNER

MODE_T_END_DEFAULTS = dict(
    static_cd_target=1.0e-9,
    static_cd_uniform=1.0e-9,
    isentropic=1.0e-9,
    noh_weak=1.0e-9,
    ale_remap=5.0e-9,
    cold_corner_sentinel=1.0e-9,
    cold_corner_guard_negative=1.0e-9,
)
T_END = float(os.environ.get("TENRYU_E1_T_END_S", str(MODE_T_END_DEFAULTS[MODE])))
DT_INITIAL = _env_float("TENRYU_E1_DT_INITIAL_S", min(1.0e-12, T_END))
DT_MAX = _env_float("TENRYU_E1_DT_MAX_S", max(T_END / 20.0, DT_INITIAL))
if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_E1 time controls must be positive")

HYDRO_ENABLED = MODE not in ("static_cd_target", "static_cd_uniform", "cold_corner_sentinel")
ALE_ENABLED = MODE == "ale_remap"
LOW_DENSITY_EXTRAPOLATION = MODE in ("static_cd_target", "cold_corner_sentinel")
MESH_MOTION = "ale" if ALE_ENABLED else "lagrangian"
RHO_LABEL_GCC = 1.0e-5 if MODE == "cold_corner_sentinel" else RHO_DENSE_GCC
if MODE == "cold_corner_guard_negative" and "TENRYU_E1_RHO_GCC" not in os.environ:
    RHO_LABEL_GCC = 1.0

CASE_NAME = (
    f"e1_{MODE}_nr{NR}_nz{NZ}_rho{_safe_float_token(RHO_LABEL_GCC)}"
    f"_seed{SEED}"
)

print(
    "[deck:2d_rz_e1_tmat_dry] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"rho_label_gcc={RHO_LABEL_GCC} rho_dense_gcc={RHO_DENSE_GCC} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} hydro={HYDRO_ENABLED} "
    f"ale={ALE_ENABLED} low_density_extrapolation={LOW_DENSITY_EXTRAPOLATION}"
)


def _inside_cd_sphere(r, z):
    return math.hypot(r, z) <= TARGET_RADIUS


def rho_init(r, z):
    if MODE == "static_cd_target":
        return RHO_DENSE_GCC if _inside_cd_sphere(r, z) else RHO_FLOOR
    if MODE == "cold_corner_sentinel":
        del r, z
        return 1.0e-5
    if MODE == "cold_corner_guard_negative":
        del r, z
        return float(os.environ.get("TENRYU_E1_RHO_GCC", "1.0"))
    del r, z
    return RHO_DENSE_GCC


def Te_init(r, z):
    del r, z
    if MODE == "static_cd_target":
        return 1.0
    if MODE == "cold_corner_sentinel":
        return 0.15
    return 10.0


def Ti_init(r, z):
    return Te_init(r, z)


def _inward_linear_velocity(r):
    if R_MAX <= 0.0:
        return 0.0
    return -V_ISENTROPIC * (r / R_MAX)


def _noh_weak_velocity(r):
    ramp = max(0.0, (r - NOH_R_INNER) / NOH_LR)
    return -V_NOH_WEAK * ramp


def _expansion_velocity(r):
    if R_MAX <= 0.0:
        return 0.0
    return V_EXPANSION * (r / R_MAX)


def velocity_init(r, z):
    del z
    if MODE in ("isentropic", "ale_remap"):
        return (_inward_linear_velocity(r), 0.0)
    if MODE == "noh_weak":
        return (_noh_weak_velocity(r), 0.0)
    if MODE == "cold_corner_guard_negative":
        return (_expansion_velocity(r), 0.0)
    return (0.0, 0.0)


def cd_volfrac(r, z):
    del r, z
    return 1.0


volfrac = dict(CD=cd_volfrac)

plot_fields = ["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar"]
if MODE == "static_cd_target":
    plot_fields.append("volfrac")

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
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion=MESH_MOTION,
)

Materials(
    low_density_extrapolation=LOW_DENSITY_EXTRAPOLATION,
    materials=[
        Material(
            name="CD",
            A=A_CD,
            Z=Z_CD,
            eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="tabular"),
)

Geometry(
    volfrac=volfrac,
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

ale_config = dict(enabled=False)
if ALE_ENABLED:
    ale_config = dict(
        enabled=True,
        every_n_steps=1,
        warmup_steps=0,
        quality_threshold=float(os.environ.get("TENRYU_E1_ALE_QUALITY_THRESHOLD", "1.0")),
        max_iterations=int(os.environ.get("TENRYU_E1_ALE_MAX_ITERATIONS", "100")),
        convergence_tol=float(os.environ.get("TENRYU_E1_ALE_CONVERGENCE_TOL", "1.0e-9")),
        max_displacement_fraction=float(
            os.environ.get("TENRYU_E1_ALE_MAX_DISPLACEMENT_FRACTION", "0.5")
        ),
        preventive_axis_guard_fraction=float(
            os.environ.get("TENRYU_E1_PREVENTIVE_AXIS_GUARD_FRACTION", "0.1")
        ),
        axis_z_motion=os.environ.get("TENRYU_E1_AXIS_Z_MOTION", "fixed"),
        axis_repair_mode=os.environ.get("TENRYU_E1_AXIS_REPAIR_MODE", "axis_spine_only"),
        remap_scheme=os.environ.get("TENRYU_E1_REMAP_SCHEME", "ms2_moments"),
        remap_ms2_limiter=os.environ.get("TENRYU_E1_REMAP_MS2_LIMITER", "van_leer"),
        ke_conservation_closure=os.environ.get("TENRYU_E1_KE_CLOSURE", "1") == "1",
        ke_conservation_closure_audit=os.environ.get("TENRYU_E1_KE_CLOSURE_AUDIT", "0") == "1",
        ke_closure_redistribute_floor=os.environ.get(
            "TENRYU_E1_KE_CLOSURE_REDISTRIBUTE_FLOOR", "1"
        )
        == "1",
        debug_per_remap_log=os.environ.get("TENRYU_E1_DEBUG_PER_REMAP_LOG", "1") == "1",
    )

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=0.1,
        cfl_cond=0.1,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=HYDRO_ENABLED,
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    ale=ale_config,
    diagnostics=dict(phase_resolved_energy=True),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=RHO_FLOOR, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0e-3,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

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

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=5,
        outer_tol=1.0e-8,
        linear_solver_2d="auto",
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="reflect"),
    ),
)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
