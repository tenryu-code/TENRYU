import math
import os

from tenryu_namelist import *


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_C1_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


MODE = os.environ.get("TENRYU_C1_MODE", "slab").lower().replace("-", "_")
VALID_MODES = (
    "slab",
    "radial",
    "axis_pulse",
    "distorted",
    "limiter_spitzer",
    "boundary_flux",
    "zeldovich_raizer",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_C1_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_C1_GRID", ""))
NR = int(os.environ.get("TENRYU_C1_NR", str(grid_env[0] if grid_env else 64)))
NZ = int(os.environ.get("TENRYU_C1_NZ", str(grid_env[1] if grid_env else 128)))
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_C1_NR and TENRYU_C1_NZ must be positive")

SEED = int(os.environ.get("TENRYU_C1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_C1_OUTDIR", "./build/output_verify_2d_rz_c1_conduction")
CFL_COND = float(os.environ.get("TENRYU_C1_CFL_COND", "0.1"))
F_LIM = float(os.environ.get("TENRYU_C1_F_LIM", "0.06"))
SOLVER = os.environ.get("TENRYU_C1_SOLVER", "sts")
T_INIT_EV = float(os.environ.get("TENRYU_C1_T_INIT_eV", "10.0"))
T_AMP_EV = float(os.environ.get("TENRYU_C1_T_AMP_eV", "1.0"))
T_HOT_EV = float(os.environ.get("TENRYU_C1_T_HOT_eV", "2000.0"))
T_COLD_EV = float(os.environ.get("TENRYU_C1_T_COLD_eV", "50.0"))
T_BOUNDARY_EV = float(os.environ.get("TENRYU_C1_T_BOUNDARY_eV", "20.0"))
T_FLOOR_EV = float(os.environ.get("TENRYU_C1_T_FLOOR_eV", "1.0"))
RHO_GCC = float(os.environ.get("TENRYU_C1_RHO_GCC", "1.0e-3" if MODE == "limiter_spitzer" else "1.0"))
TEST_KAPPA = float(os.environ.get("TENRYU_C1_TEST_KAPPA", "2.0e14"))
T_END_DEFAULT = "1.0e-12" if MODE == "limiter_spitzer" else "1.0e-4"
T_END = float(os.environ.get("TENRYU_C1_T_END_S", T_END_DEFAULT))
MAX_STEPS_DEFAULT = "20" if MODE == "limiter_spitzer" else "200000"
MAX_STEPS = int(os.environ.get("TENRYU_C1_MAX_STEPS", MAX_STEPS_DEFAULT))
DT_INITIAL = float(os.environ.get("TENRYU_C1_DT_INITIAL_S", str(min(1.0e-8, T_END))))
DT_MAX = float(os.environ.get("TENRYU_C1_DT_MAX_S", str(max(T_END / 4.0, DT_INITIAL))))

A = float(os.environ.get("TENRYU_C1_A", "1.0"))
ZBAR = float(os.environ.get("TENRYU_C1_ZBAR", "1.0"))
GAMMA = float(os.environ.get("TENRYU_C1_GAMMA", str(5.0 / 3.0)))

R_MIN = 0.0
R_MAX = float(os.environ.get("TENRYU_C1_R_MAX_CM", "1.0"))
Z_MIN = float(os.environ.get("TENRYU_C1_Z_MIN_CM", "-1.0"))
Z_MAX = float(os.environ.get("TENRYU_C1_Z_MAX_CM", "1.0"))
Z_MID = 0.5 * (Z_MIN + Z_MAX)
LZ = Z_MAX - Z_MIN
SIGMA = float(os.environ.get("TENRYU_C1_SIGMA_CM", "0.12"))
ZR_SIGMA_CM = float(os.environ.get("TENRYU_C1_ZR_SIGMA_CM", "0.04"))

CASE_NAME = (
    f"c1_{MODE}_nr{NR}_nz{NZ}_cfl{_safe_float_token(CFL_COND)}"
    f"_flim{_safe_float_token(F_LIM)}_seed{SEED}"
)

print(
    "[deck:2d_rz_c1_conduction] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"cfl_cond={CFL_COND} f_lim={F_LIM} t_end_s={T_END} "
    f"solver={SOLVER} rho_gcc={RHO_GCC} "
    f"test_kappa={'disabled' if MODE == 'limiter_spitzer' else TEST_KAPPA}"
)


def rho_init(r, z):
    del r, z
    return RHO_GCC


def _slab_temperature(r, z):
    del r
    theta = 2.0 * math.pi * (z - Z_MID) / LZ
    return T_INIT_EV + T_AMP_EV * math.cos(theta)


def _radial_temperature(r, z):
    del z
    theta = math.pi * r / R_MAX
    return T_INIT_EV + T_AMP_EV * math.cos(theta)


def _axis_pulse_temperature(r, z):
    rr = r / SIGMA
    zz = (z - Z_MID) / SIGMA
    return T_INIT_EV + T_AMP_EV * math.exp(-0.5 * (rr * rr + zz * zz))


def _distorted_temperature(r, z):
    radial = math.cos(math.pi * r / R_MAX)
    axial = math.cos(2.0 * math.pi * (z - Z_MID) / LZ)
    return T_INIT_EV + 0.5 * T_AMP_EV * (radial + axial)


def _limiter_temperature(r, z):
    del z
    width = max(0.01 * R_MAX, 1.0e-12)
    front = 0.5 * R_MAX
    blend = 0.5 * (1.0 - math.tanh((r - front) / width))
    return T_COLD_EV + (T_HOT_EV - T_COLD_EV) * blend


def _zeldovich_raizer_temperature(r, z):
    del r
    x = (z - Z_MID) / max(ZR_SIGMA_CM, 1.0e-30)
    return T_FLOOR_EV + T_AMP_EV * math.exp(-0.5 * x * x)


if MODE == "slab":
    Te_init = _slab_temperature
elif MODE == "radial":
    Te_init = _radial_temperature
elif MODE == "axis_pulse":
    Te_init = _axis_pulse_temperature
elif MODE == "distorted":
    Te_init = _distorted_temperature
elif MODE == "limiter_spitzer":
    Te_init = _limiter_temperature
elif MODE == "zeldovich_raizer":
    Te_init = _zeldovich_raizer_temperature
else:
    Te_init = lambda r, z: T_INIT_EV


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


conduction_config = dict(
    enabled=True,
    solver=SOLVER,
    f_lim=F_LIM,
    sts_damping=0.01,
    sts_max_stages=40,
)
if MODE != "limiter_spitzer":
    conduction_config["test_kappa"] = TEST_KAPPA

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
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=0.3,
        cfl_cond=CFL_COND,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=False,
        boundary_2d=(
            dict(
                r_inner="axis",
                r_outer="reflect",
                z_bottom="reflect",
                z_top=dict(
                    type="state_supply",
                    rho_g_per_cc=RHO_GCC,
                    u_z_cm_per_s=0.0,
                    T_eV=T_BOUNDARY_EV,
                ),
            )
            if MODE == "boundary_flux"
            else dict(r_inner="axis", r_outer="reflect", z_bottom="reflect", z_top="reflect")
        ),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=conduction_config,
    ale=dict(enabled=False),
    diagnostics=dict(phase_resolved_energy=True),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
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
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
