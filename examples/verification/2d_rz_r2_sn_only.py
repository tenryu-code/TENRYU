import math
import os

from tenryu_namelist import *


# TENRYU 2D_RZ R2 S_N-only deck.
# Manual-use modes are selected with TENRYU_R2_MODE.


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_R2_GRID must be formatted as NRxNZ or NR,NZ")
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
    value = os.environ.get(name, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


MODE = os.environ.get("TENRYU_R2_MODE", "axisymmetric_convergence").lower().replace("-", "_")
VALID_MODES = (
    "axisymmetric_convergence",
    "diffusion_limit_vs_fld",
    "thin_corona_sn",
    "multigroup_5g",
    "n_angles_sweep",
    "slab_marshak_sn",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_R2_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_R2_GRID", ""))
NR = _env_int("TENRYU_R2_NR", grid_env[0] if grid_env else 64)
NZ = _env_int("TENRYU_R2_NZ", grid_env[1] if grid_env else 128)
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_R2_NR and TENRYU_R2_NZ must be positive")

SEED = _env_int("TENRYU_R2_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_R2_OUTDIR", "./build/output_verify_2d_rz_r2_sn")
MAX_STEPS = _env_int("TENRYU_R2_MAX_STEPS", 1000000)
T_END = _env_float("TENRYU_R2_T_END_S", 1.0e-9)
DT_INITIAL = _env_float("TENRYU_R2_DT_INITIAL_S", min(1.0e-12, T_END))
DT_MAX = _env_float("TENRYU_R2_DT_MAX_S", max(T_END / 20.0, DT_INITIAL))
if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_R2 time controls must be positive")

N_ANGLES = _env_int("TENRYU_R2_N_ANGLES", 16)
if N_ANGLES not in (8, 16, 32):
    raise ValueError("TENRYU_R2_N_ANGLES must be one of: 8, 16, 32")

RHO_DEFAULTS = dict(
    axisymmetric_convergence=1.0e-3,
    diffusion_limit_vs_fld=1.0e-3,
    thin_corona_sn=1.0e-7,
    multigroup_5g=1.0e-3,
    n_angles_sweep=1.0e-3,
    slab_marshak_sn=1.0,
)
KAPPA_DEFAULTS = dict(
    axisymmetric_convergence=1.0,
    diffusion_limit_vs_fld=100.0,
    thin_corona_sn=1.0e-3,
    multigroup_5g=1.0,
    n_angles_sweep=1.0,
    slab_marshak_sn=1.0,
)

RHO_GCC = _env_float("TENRYU_R2_RHO_GCC", RHO_DEFAULTS[MODE])
KAPPA_A = _env_float("TENRYU_R2_KAPPA_A_CM2_G", KAPPA_DEFAULTS[MODE])
CV_E_OVERRIDE = _env_float(
    "TENRYU_R2_CV_E_OVERRIDE", 5.488e2 if MODE == "slab_marshak_sn" else 1.0e15
)
DSA_ENABLED = _env_bool("TENRYU_R2_DSA_ENABLED", "1")
SN_TIMING_ENABLED = _env_bool("TENRYU_R2_SN_TIMING_ENABLED", "1")
GROUP_BOUNDS = (
    [0.0, 1.0, 10.0, 100.0, 1000.0, 1.0e5]
    if MODE == "multigroup_5g"
    else [0.0, 1.0e6]
)
GROUPS = len(GROUP_BOUNDS) - 1
RADIATION_FIELD = "equilibrium"

R_MIN = 0.0
R_MAX = 1.0
Z_MIN = -1.0
Z_MAX = 1.0

TE_EQUILIBRIUM = 10.0
TE_INNER = 100.0
TE_OUTER = 1.0
CORONA_EDGE = 0.5
MARSHAK_FLUX = _env_float("TENRYU_R2_MARSHAK_FLUX_ERG_CM2_S", 1.0e10)

CASE_NAME = (
    f"r2_{MODE}_nr{NR}_nz{NZ}_rho{_safe_float_token(RHO_GCC)}"
    f"_ka{_safe_float_token(KAPPA_A)}_g{GROUPS}_s{N_ANGLES}_seed{SEED}"
    f"{'_dsaoff' if not DSA_ENABLED else ''}"
)

print(
    "[deck:2d_rz_r2_sn_only] "
    f"mode={MODE} nr={NR} nz={NZ} groups={GROUPS} n_angles={N_ANGLES} seed={SEED} "
    f"outdir={OUTDIR} rho_gcc={RHO_GCC} kappa_a_cm2_g={KAPPA_A} "
    f"radiation_field={RADIATION_FIELD} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} cv_e_override={CV_E_OVERRIDE} "
    f"dsa_enabled={DSA_ENABLED}"
)


def rho_init(r, z):
    del r, z
    return RHO_GCC


def Te_init(r, z):
    del z
    if MODE == "thin_corona_sn":
        return TE_INNER if r < CORONA_EDGE else TE_OUTER
    if MODE == "slab_marshak_sn":
        return 0.1
    return TE_EQUILIBRIUM


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def h_volfrac(r, z):
    del r, z
    return 1.0


eos = dict(
    model="ideal_gas",
    cv_e_override=CV_E_OVERRIDE,
    ideal_gas=dict(gamma=5.0 / 3.0),
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
    temperature_model="1T" if MODE == "slab_marshak_sn" else "2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_R2_VERBOSITY", "quiet"),
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
    mode="sn_transport",
    groups=GROUPS,
    group_bounds_eV=GROUP_BOUNDS,
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    sn_transport=dict(
        spatial_scheme="linear_characteristic",
        n_angles=N_ANGLES,
        angular_quadrature=f"level_symmetric_{N_ANGLES}",
        dsa_enabled=DSA_ENABLED,
        timing_enabled=SN_TIMING_ENABLED,
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        diffusion_fallback_mode="none",
        boundary=(
            dict(inner_r="reflect_parity", outer_r="reflect", z_bottom="reflect", z_top="marshak")
            if MODE == "slab_marshak_sn"
            else dict(inner_r="reflect_parity", outer_r="vacuum", z="reflect")
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
