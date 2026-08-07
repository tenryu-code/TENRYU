import math
import os

from tenryu_namelist import *


# TENRYU 2D_RZ R1 FLD-only deck.
# Manual-use modes are selected with TENRYU_R1_MODE.


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_R1_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


def _env_float(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


def _env_int(name, default):
    if name in os.environ:
        return int(os.environ[name])
    return int(default)


MODE = os.environ.get("TENRYU_R1_MODE", "radial_decoupled").lower().replace("-", "_")
VALID_MODES = (
    "radial_decoupled",
    "equilibrium_with_leakage",
    "thin_corona",
    "equilibrium_with_leakage_5g",
    "slab_marshak",
    "closed_box_equilibrium",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_R1_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_R1_GRID", ""))
NR = _env_int("TENRYU_R1_NR", grid_env[0] if grid_env else 64)
NZ = _env_int("TENRYU_R1_NZ", grid_env[1] if grid_env else 128)
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_R1_NR and TENRYU_R1_NZ must be positive")

MODE_T_END_DEFAULTS = dict(
    radial_decoupled=1.0e-9,
    equilibrium_with_leakage=1.0e-8,
    thin_corona=1.0e-10,
    equilibrium_with_leakage_5g=1.0e-8,
    slab_marshak=3.33e-10,
    closed_box_equilibrium=1.0e-10,
)
MODE_RHO_DEFAULTS = dict(
    radial_decoupled=1.0e-4,
    equilibrium_with_leakage=1.0e-3,
    thin_corona=1.0e-7,
    equilibrium_with_leakage_5g=1.0e-3,
    slab_marshak=1.0,
    closed_box_equilibrium=1.0e-3,
)

SEED = _env_int("TENRYU_R1_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_R1_OUTDIR", "./build/output_verify_2d_rz_r1_fld")
R1FLD_LINEAR_SOLVER = os.environ.get(
    "TENRYU_R1FLD_LINEAR_SOLVER_2D", "auto"
)
MAX_STEPS = _env_int("TENRYU_R1_MAX_STEPS", 1000000)
T_END = _env_float("TENRYU_R1_T_END_S", MODE_T_END_DEFAULTS[MODE])
DT_INITIAL = _env_float("TENRYU_R1_DT_INITIAL_S", min(1.0e-12, T_END))
DT_MAX = _env_float("TENRYU_R1_DT_MAX_S", max(T_END / 20.0, DT_INITIAL))
if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_R1 time controls must be positive")

RHO_GCC = _env_float("TENRYU_R1_RHO_GCC", MODE_RHO_DEFAULTS[MODE])
MODE_KAPPA_DEFAULTS = dict(
    thin_corona=1.0e-3,
    slab_marshak=1.0,
    closed_box_equilibrium=1.0e5,
)
KAPPA_A = _env_float("TENRYU_R1_KAPPA_A_CM2_G", MODE_KAPPA_DEFAULTS.get(MODE, 1.0))
CV_E_OVERRIDE = (
    1.0e15
    if MODE in ("radial_decoupled", "thin_corona")
    else (5.488e2 if MODE == "slab_marshak" else None)
)
RADIATION_FIELD = (
    "zero"
    if MODE in (
        "equilibrium_with_leakage",
        "equilibrium_with_leakage_5g",
        "slab_marshak",
        "closed_box_equilibrium",
    )
    else "equilibrium"
)
GROUP_BOUNDS = (
    [0.0, 1.0, 10.0, 100.0, 1000.0, 1.0e5]
    if MODE == "equilibrium_with_leakage_5g"
    else [0.0, 1.0e6]
)
GROUPS = len(GROUP_BOUNDS) - 1

R_MIN = 0.0
R_MAX = 1.0
Z_MIN = -1.0
Z_MAX = 1.0

T_FLOOR = 1.0
T_PEAK = 100.0
R_CENTER = 0.5
SIGMA_R = 0.1
TE_LEAKAGE = 10.0
TE_INNER = 100.0
TE_OUTER = 1.0
CORONA_EDGE = 0.5
MARSHAK_FLUX = _env_float("TENRYU_R1_MARSHAK_FLUX_ERG_CM2_S", 1.0e10)

CASE_NAME = (
    f"r1_{MODE}_nr{NR}_nz{NZ}_rho{_safe_float_token(RHO_GCC)}"
    f"_ka{_safe_float_token(KAPPA_A)}_g{GROUPS}_seed{SEED}"
)

print(
    "[deck:2d_rz_r1_fld_only] "
    f"mode={MODE} nr={NR} nz={NZ} groups={GROUPS} seed={SEED} "
    f"outdir={OUTDIR} rho_gcc={RHO_GCC} kappa_a_cm2_g={KAPPA_A} "
    f"radiation_field={RADIATION_FIELD} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} cv_e_override={CV_E_OVERRIDE}"
)


def rho_init(r, z):
    del r, z
    return RHO_GCC


def Te_init(r, z):
    del z
    if MODE == "radial_decoupled":
        profile = math.exp(-((r - R_CENTER) ** 2) / (SIGMA_R * SIGMA_R))
        return T_FLOOR + (T_PEAK - T_FLOOR) * profile
    if MODE == "thin_corona":
        return TE_INNER if r < CORONA_EDGE else TE_OUTER
    if MODE == "slab_marshak":
        return 0.1
    return TE_LEAKAGE


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def h_volfrac(r, z):
    del r, z
    return 1.0


eos = dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0))
if CV_E_OVERRIDE is not None:
    eos["cv_e_override"] = CV_E_OVERRIDE

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
    temperature_model="1T" if MODE in ("slab_marshak", "closed_box_equilibrium") else "2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_R1_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="H",
            A=1.0,
            Z=1.0,
            eos=eos,
            opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(H=h_volfrac),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field=RADIATION_FIELD,
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=0.3,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=False,
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect"),
    ),
    conduction=dict(enabled=False),
    ale=dict(enabled=False),
    diagnostics=dict(phase_resolved_energy=True),
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

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=GROUPS,
    group_bounds_eV=GROUP_BOUNDS,
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="none" if MODE == "slab_marshak" else "levermore_pomraning",
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        linear_solver_2d=R1FLD_LINEAR_SOLVER,
        boundary=(
            dict(inner_r="reflect", outer_r="reflect", z_bottom="reflect", z_top="marshak")
            if MODE == "slab_marshak"
            else (
                dict(inner_r="reflect", outer_r="reflect", z="reflect")
                if MODE == "closed_box_equilibrium"
                else dict(inner_r="reflect", outer_r="vacuum", z="reflect")
            )
        ),
        marshak=dict(flux_erg_per_cm2_s=MARSHAK_FLUX),
    ),
)

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
