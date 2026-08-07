import math
import os

from tenryu_namelist import *


# TENRYU 2D_RZ RH2 hydro+S_N deck.
# Manual-use modes are selected with TENRYU_RH2_MODE.


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_RH2_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


def _env_float(name, default):
    if name in os.environ:
        return float(os.environ[name])
    return float(default)


def _env_int(name, default):
    if name in os.environ:
        return int(os.environ[name])
    return int(default)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


MODE = os.environ.get("TENRYU_RH2_MODE", "shock_tube_grey_sn").lower().replace("-", "_")
VALID_MODES = (
    "shock_tube_grey_sn",
    "cylindrical_blast_sn",
    "sn_vs_fld_thick",
    "optically_thin_precursor_sn",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_RH2_MODE must be one of: " + ", ".join(VALID_MODES))

RADIATION_MODE = os.environ.get("TENRYU_RH2_RADIATION_MODE_OVERRIDE", "sn_transport").lower()
if RADIATION_MODE not in ("sn_transport", "multigroup_diffusion"):
    raise ValueError("TENRYU_RH2_RADIATION_MODE_OVERRIDE must be sn_transport or multigroup_diffusion")

grid_env = _parse_grid(os.environ.get("TENRYU_RH2_GRID", ""))
NR = _env_int("TENRYU_RH2_NR", grid_env[0] if grid_env else 32)
NZ = _env_int("TENRYU_RH2_NZ", grid_env[1] if grid_env else 64)
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_RH2_NR and TENRYU_RH2_NZ must be positive")

SEED = _env_int("TENRYU_RH2_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_RH2_OUTDIR", "./build/output_verify_2d_rz_rh2")
MAX_STEPS = _env_int("TENRYU_RH2_MAX_STEPS", 5 if MODE == "sn_vs_fld_thick" else 3)
RH2_ALE = _env_bool("TENRYU_RH2_ALE", False)
RH2_ALE_EVERY_N_STEPS = _env_int("TENRYU_RH2_ALE_EVERY_N_STEPS", 5)

MODE_T_END_DEFAULTS = dict(
    shock_tube_grey_sn=1.0e-12,
    cylindrical_blast_sn=1.0e-12,
    sn_vs_fld_thick=5.0e-13,
    optically_thin_precursor_sn=1.0e-12,
)
MODE_RHO_DEFAULTS = dict(
    shock_tube_grey_sn=1.0,
    cylindrical_blast_sn=1.0e-3,
    sn_vs_fld_thick=1.0,
    optically_thin_precursor_sn=1.0e-5,
)
MODE_KAPPA_DEFAULTS = dict(
    shock_tube_grey_sn=10.0,
    cylindrical_blast_sn=10.0,
    sn_vs_fld_thick=1.0e4,
    optically_thin_precursor_sn=0.01,
)
MODE_BLAST_E_DEFAULTS = dict(
    cylindrical_blast_sn=1.0e8,
    optically_thin_precursor_sn=1.0e6,
)

T_END = _env_float("TENRYU_RH2_T_END_S", MODE_T_END_DEFAULTS[MODE])
DT_INITIAL = _env_float("TENRYU_RH2_DT_INITIAL_S", min(1.0e-13, T_END))
DT_MAX = _env_float("TENRYU_RH2_DT_MAX_S", max(T_END / 20.0, DT_INITIAL))
if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_RH2 time controls must be positive")

RHO_GCC = _env_float("TENRYU_RH2_RHO_GCC", MODE_RHO_DEFAULTS[MODE])
KAPPA_A = _env_float("TENRYU_RH2_KAPPA_A_CM2_G", MODE_KAPPA_DEFAULTS[MODE])
AV_C2 = _env_float("TENRYU_RH2_AV_C2", 1.5)
CFL = _env_float("TENRYU_RH2_CFL", 0.3)
BLAST_CELLS = _env_int("TENRYU_RH2_BLAST_CELLS", 4)
N_ANGLES = _env_int("TENRYU_RH2_N_ANGLES", 16)
if RADIATION_MODE == "sn_transport" and N_ANGLES not in (8, 16, 32):
    raise ValueError("TENRYU_RH2_N_ANGLES must be one of: 8, 16, 32")

R_MIN = 0.0
R_MAX = _env_float("TENRYU_RH2_R_MAX_CM", 1.0)
Z_MIN = _env_float("TENRYU_RH2_Z_MIN_CM", -1.0)
Z_MAX = _env_float("TENRYU_RH2_Z_MAX_CM", 1.0)
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("TENRYU_RH2 geometry bounds must satisfy r_max>r_min and z_max>z_min")

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
CV_I = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
CV_E = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))

DR = (R_MAX - R_MIN) / NR
R_BLAST = BLAST_CELLS * DR
Z_LEN = Z_MAX - Z_MIN
V_PATCH = math.pi * R_BLAST * R_BLAST * Z_LEN
DEFAULT_BLAST_E_ERG = MODE_BLAST_E_DEFAULTS.get(MODE, 1.0e13)
BLAST_E_ERG = _env_float("TENRYU_RH2_BLAST_E_ERG", DEFAULT_BLAST_E_ERG)
E_INT_BLAST = BLAST_E_ERG / max(RHO_GCC * V_PATCH, 1.0e-300)
T_BLAST_EV = E_INT_BLAST / (CV_I + CV_E)

CASE_NAME = (
    f"rh2_{MODE}_{RADIATION_MODE}_nr{NR}_nz{NZ}_rho{_safe_float_token(RHO_GCC)}"
    f"_ka{_safe_float_token(KAPPA_A)}_g1_seed{SEED}"
)

print(
    "[deck:2d_rz_rh2_hydro_sn] "
    f"mode={MODE} radiation_mode={RADIATION_MODE} nr={NR} nz={NZ} groups=1 "
    f"n_angles={N_ANGLES} seed={SEED} outdir={OUTDIR} rho_gcc={RHO_GCC} "
    f"kappa_a_cm2_g={KAPPA_A} t_end_s={T_END} dt_initial_s={DT_INITIAL} "
    f"dt_max_s={DT_MAX} av_C2={AV_C2} blast_E_erg={BLAST_E_ERG} "
    f"blast_cells={BLAST_CELLS} R_blast_cm={R_BLAST} ale={RH2_ALE}"
)


def rho_init(r, z):
    del r
    if MODE == "shock_tube_grey_sn":
        return 1.0 if z < 0.0 else 0.125
    return RHO_GCC


def Te_init(r, z):
    if MODE == "shock_tube_grey_sn":
        return 10.0 if z < 0.0 else 2.0
    if MODE == "sn_vs_fld_thick":
        del r
        return 10.0 + 2.0 * math.exp(-25.0 * z * z)
    if r < R_BLAST:
        return T_BLAST_EV
    return 1.0


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def h_volfrac(r, z):
    del r, z
    return 1.0


HYDRO_BOUNDARY = dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect")

ale_config = dict(enabled=False)
if RH2_ALE:
    ale_config = dict(
        enabled=True,
        every_n_steps=RH2_ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
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
    verbosity=os.environ.get("TENRYU_RH2_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale" if RH2_ALE else "lagrangian",
)

Materials(
    materials=[
        Material(
            name="H",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(H=h_volfrac),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
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
        volume_rate_cfl_enabled=True if RH2_ALE else False,
        volume_rate_cfl_threshold=0.5,
    ),
    conduction=dict(enabled=False),
    ale=ale_config,
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

if RADIATION_MODE == "sn_transport":
    Radiation(
        enabled=True,
        mode="sn_transport",
        groups=1,
        group_bounds_eV=[0.0, 1.0e6],
        imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
        ddmc=dict(enabled=False),
        holo=dict(enabled=False),
        sn_transport=dict(
            spatial_scheme="linear_characteristic",
            n_angles=N_ANGLES,
            angular_quadrature=f"level_symmetric_{N_ANGLES}",
            dsa_enabled=True,
            max_outer_iterations=50,
            outer_tol=1.0e-8,
            boundary=dict(inner_r="reflect_parity", outer_r="vacuum", z="reflect"),
        ),
    )
else:
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
            max_outer_iterations=50,
            outer_tol=1.0e-8,
            linear_solver_2d="cusparse_cg_jacobi",
            boundary=dict(inner_r="reflect", outer_r="vacuum", z="reflect"),
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
