import math
import os

from tenryu_namelist import *


# TENRYU 2D_RZ C2 Qei deck.
# Modes:
#   near_eq_analytic: small Te-Ti split, analytic constant-tau reference limit.
#   hot_e_rk4 / hot_i_rk4: ideal-gas Spitzer Qei, RK4 continuous-ODE reference.
#   tmat_hot_e: CD.tmat.h5 table-EOS Qei; monotonicity/table-domain/cv diagnostics.
#   dt_sweep: hot-electron ideal case with forced dt, finite-step map reference.
#   qei_plus_cond_uniform: Qei + no-op constant-kappa conduction, hot_e comparison.
#   qei_plus_cond_gradient: Qei + conduction, energy/positivity/Te-spread gates.


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _parse_grid(value):
    if not value:
        return None
    sep = "x" if "x" in value.lower() else ","
    parts = [part.strip() for part in value.lower().split(sep)]
    if len(parts) != 2:
        raise ValueError("TENRYU_C2_GRID must be formatted as NRxNZ or NR,NZ")
    return int(parts[0]), int(parts[1])


def _env_float(name, default, alias=None):
    if name in os.environ:
        return float(os.environ[name])
    if alias and alias in os.environ:
        return float(os.environ[alias])
    return float(default)


def _spitzer_tau_eq_s(rho_gcc, A, Zbar, Te_eV):
    m_p = 1.6726219e-24  # must match core::constants::proton_mass
    n_i = rho_gcc / (A * m_p)
    n_e = Zbar * n_i
    te = max(Te_eV, 1.0e-300)
    z = max(Zbar, 1.0e-300)
    if te >= 10.0 * z * z:
        ln_lambda_raw = 24.0 - 0.5 * math.log(n_e) + math.log(te)
    else:
        ln_lambda_raw = 23.0 - 0.5 * math.log(n_e) - math.log(z) + 1.5 * math.log(te)
    ln_lambda = max(2.0, ln_lambda_raw)
    return 3.16e8 * A * (te ** 1.5) / (z * z * n_i * ln_lambda)


MODE = os.environ.get("TENRYU_C2_MODE", "hot_e_rk4").lower().replace("-", "_")
VALID_MODES = (
    "near_eq_analytic",
    "hot_e_rk4",
    "hot_i_rk4",
    "tmat_hot_e",
    "dt_sweep",
    "qei_plus_cond_uniform",
    "qei_plus_cond_gradient",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_C2_MODE must be one of: " + ", ".join(VALID_MODES))

grid_env = _parse_grid(os.environ.get("TENRYU_C2_GRID", ""))
NR = int(os.environ.get("TENRYU_C2_NR", str(grid_env[0] if grid_env else 32)))
NZ = int(os.environ.get("TENRYU_C2_NZ", str(grid_env[1] if grid_env else 64)))
if NR <= 0 or NZ <= 0:
    raise ValueError("TENRYU_C2_NR and TENRYU_C2_NZ must be positive")

SEED = int(os.environ.get("TENRYU_C2_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_C2_OUTDIR", "./build/output_verify_2d_rz_c2_qei")
RHO_GCC = float(os.environ.get("TENRYU_C2_RHO_GCC", "1.0e-5"))
A = float(os.environ.get("TENRYU_C2_A", "1.0"))
ZBAR = float(os.environ.get("TENRYU_C2_ZBAR", "1.0"))
GAMMA = float(os.environ.get("TENRYU_C2_GAMMA", str(5.0 / 3.0)))

R_MIN = 0.0
R_MAX = float(os.environ.get("TENRYU_C2_R_MAX_CM", "1.0"))
Z_MIN = float(os.environ.get("TENRYU_C2_Z_MIN_CM", "-1.0"))
Z_MAX = float(os.environ.get("TENRYU_C2_Z_MAX_CM", "1.0"))
Z_MID = 0.5 * (Z_MIN + Z_MAX)
LZ = Z_MAX - Z_MIN
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("TENRYU_C2 geometry bounds must satisfy r_max>r_min and z_max>z_min")

F_LIM = float(os.environ.get("TENRYU_C2_F_LIM", "0.06"))
SOLVER = os.environ.get("TENRYU_C2_SOLVER", "sts")
TEST_KAPPA = float(os.environ.get("TENRYU_C2_TEST_KAPPA", "2.0e14"))
CFL_COND = float(os.environ.get("TENRYU_C2_CFL_COND", "0.1"))
MAX_STEPS = int(os.environ.get("TENRYU_C2_MAX_STEPS", "1000000"))
TE_FLOOR = float(os.environ.get("TENRYU_C2_TE_FLOOR_EV", "1.0e-6"))
TI_FLOOR = float(os.environ.get("TENRYU_C2_TI_FLOOR_EV", "1.0e-6"))

TAU_EQ_TE100 = _spitzer_tau_eq_s(RHO_GCC, A, ZBAR, 100.0)
TAU_EI_TE100 = 0.5 * TAU_EQ_TE100
MODE_T_END_DEFAULTS = dict(
    near_eq_analytic=2.8e-8,
    hot_e_rk4=2.8e-8,
    hot_i_rk4=2.8e-11,
    tmat_hot_e=2.8e-8,
    dt_sweep=5.6e-8,
    qei_plus_cond_uniform=2.8e-8,
    qei_plus_cond_gradient=2.8e-8,
)
T_END = float(os.environ.get("TENRYU_C2_T_END_S", str(MODE_T_END_DEFAULTS[MODE])))

TE0_EV = 100.0
TI0_EV = 1.0
if MODE == "near_eq_analytic":
    TE0_EV = 50.25
    TI0_EV = 49.75
elif MODE == "hot_i_rk4":
    TE0_EV = 1.0
    TI0_EV = 100.0

tau_initial = 0.5 * _spitzer_tau_eq_s(RHO_GCC, A, ZBAR, TE0_EV)
dt_default = max(min(0.01 * tau_initial, T_END), 1.0e-22)
DT_INITIAL = _env_float("TENRYU_C2_DT_INITIAL_S", dt_default)
DT_MAX = _env_float("TENRYU_C2_DT_MAX_S", dt_default)
DT_FORCE = _env_float("TENRYU_C2_DT_FORCE_S", 5.6e-9, alias="TENRYU_C2_DT_FORCE")
if MODE == "dt_sweep":
    DT_INITIAL = DT_FORCE
    DT_MAX = DT_FORCE

if T_END <= 0.0 or DT_INITIAL <= 0.0 or DT_MAX <= 0.0:
    raise ValueError("TENRYU_C2 time controls must be positive")

CONDUCTION_ENABLED = MODE in ("qei_plus_cond_uniform", "qei_plus_cond_gradient")
USE_TMAT = MODE == "tmat_hot_e"

CASE_NAME = (
    f"c2_{MODE}_nr{NR}_nz{NZ}_dt{_safe_float_token(DT_INITIAL)}"
    f"_rho{_safe_float_token(RHO_GCC)}_seed{SEED}"
)

print(
    "[deck:2d_rz_c2_qei] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"rho_gcc={RHO_GCC} Te0_eV={TE0_EV} Ti0_eV={TI0_EV} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"tau_ei_te100_s={TAU_EI_TE100} conduction={CONDUCTION_ENABLED} "
    f"test_kappa={'disabled' if not CONDUCTION_ENABLED else TEST_KAPPA}"
)


def rho_init(r, z):
    del r, z
    return RHO_GCC


def Te_init(r, z):
    del r
    if MODE == "qei_plus_cond_gradient":
        theta = math.pi * (z - Z_MID) / LZ
        return 100.0 * (1.0 + 0.1 * math.sin(theta))
    return TE0_EV


def Ti_init(r, z):
    del r, z
    return TI0_EV


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


if USE_TMAT:
    material = Material(
        name="CD",
        A=7.0,
        Z=3.5,
        eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
        opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
    )
    volfrac = dict(CD=lambda r, z: 1.0)
    zbar_config = dict(model="tabular")
else:
    material = Material(
        name="fuel",
        A=A,
        Z=ZBAR,
        eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
        opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
    )
    volfrac = dict(fuel=lambda r, z: 1.0)
    zbar_config = dict(model="fixed", fixed_value=ZBAR)

conduction_config = dict(
    enabled=CONDUCTION_ENABLED,
    solver=SOLVER,
    f_lim=F_LIM,
    sts_damping=0.01,
    sts_max_stages=40,
)
if CONDUCTION_ENABLED:
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
    materials=[material],
    zbar=zbar_config,
)

Geometry(
    volfrac=volfrac,
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=0.3,
        cfl_cond=CFL_COND,
        growth_factor=1.0 if MODE == "dt_sweep" else 1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=False,
        boundary_2d=dict(r_inner="axis", r_outer="reflect", z_bottom="reflect", z_top="reflect"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=conduction_config,
    ale=dict(enabled=False),
    diagnostics=dict(phase_resolved_energy=True),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
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
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar", "cv_e", "cv_i"],
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
