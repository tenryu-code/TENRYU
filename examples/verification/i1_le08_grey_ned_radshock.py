"""LE08 nED grey radiative-shock verification deck.

The defaults use a large 1D_SPH radius as the quasi-planar workaround and a
radiation-relaxation-time run so the two-state IC reaches the steady shock
structure.
"""

import bisect
import json
import os
from pathlib import Path

from tenryu_namelist import *


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _repo_root():
    return Path(__file__).resolve().parents[2]


M0 = _env_float("TENRYU_I1_LE08_M0", 2.0)
NR = _env_int("TENRYU_I1_LE08_NR", 64)
SEED = _env_int("TENRYU_I1_LE08_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_I1_LE08_OUTDIR", "./build/output_verify_i1_le08_nED")
INIT_MODE = os.environ.get("TENRYU_I1_LE08_INIT_MODE", "two_state").lower().replace("-", "_")
if INIT_MODE not in ("reference_table", "two_state"):
    raise ValueError("TENRYU_I1_LE08_INIT_MODE must be reference_table or two_state")

default_ref = (
    _repo_root()
    / "tests"
    / "verification"
    / "data"
    / "le08_ned_reference"
    / f"le08_nED_M{_safe_float_token(M0)}.json"
)
REFERENCE_PATH = Path(os.environ.get("TENRYU_I1_LE08_REFERENCE", str(default_ref)))
with REFERENCE_PATH.open(encoding="utf-8") as stream:
    REF = json.load(stream)

PARAMS = REF["parameters"]
TABLE = REF["table"]
UPSTREAM = REF["states"]["upstream"]
DOWNSTREAM = REF["states"]["downstream"]
GAMMA = float(PARAMS["gamma"])
A_AMU = float(PARAMS["A_amu"])
ZBAR = float(PARAMS["zbar"])
KAPPA_R = float(PARAMS["kappa_R_cm2_per_g"])
C_LIGHT_CM_PER_S = float(PARAMS["c_cm_per_s"])
A_EV_ERG_CM3_EV4 = float(PARAMS["a_eV_erg_cm3_eV4"])
EV_TO_ERG = float(PARAMS["eV_to_erg"])
PROTON_MASS_G = float(PARAMS["proton_mass_g"])
X_TABLE = [float(x) for x in TABLE["x_cm"]]
RHO_TABLE = [float(x) for x in TABLE["rho_g_per_cc"]]
T_TABLE = [float(x) for x in TABLE["T_eV"]]
U_TABLE = [float(x) for x in TABLE["u_cm_per_s"]]

DOMAIN_SPAN = max(X_TABLE[-1] - X_TABLE[0], 1.0e-30)
R_MIN = _env_float("TENRYU_I1_LE08_R_MIN_CM", 1.0e6)
R_MAX = _env_float("TENRYU_I1_LE08_R_MAX_CM", R_MIN + DOMAIN_SPAN)
if not R_MAX > R_MIN:
    raise ValueError("TENRYU_I1_LE08_R_MAX_CM must be greater than TENRYU_I1_LE08_R_MIN_CM")
PROFILE_SCALE = DOMAIN_SPAN / (R_MAX - R_MIN)
PROFILE_SHOCK_R_CM = R_MIN - X_TABLE[0] / PROFILE_SCALE
TWO_STATE_SHOCK_R_CM = 0.5 * (R_MIN + R_MAX)
SHOCK_R_CM = TWO_STATE_SHOCK_R_CM if INIT_MODE == "two_state" else PROFILE_SHOCK_R_CM


def _compute_tau_rel(rho_gcc, kappa_R_cm2_g, T_eV):
    cv_e_per_gram = ZBAR * EV_TO_ERG / (A_AMU * PROTON_MASS_G * (GAMMA - 1.0))
    sigma_a = kappa_R_cm2_g * rho_gcc
    return rho_gcc * cv_e_per_gram / (
        4.0 * C_LIGHT_CM_PER_S * sigma_a * A_EV_ERG_CM3_EV4 * T_eV**3
    )


TAU_REL = _compute_tau_rel(float(UPSTREAM["rho"]), KAPPA_R, float(PARAMS["T0_eV"]))
HYDRO_CROSSING_T_END = 3.0 * DOMAIN_SPAN / max(abs(float(UPSTREAM["u"])), 1.0e-300)
T_END = _env_float(
    "TENRYU_I1_LE08_T_END_S",
    max(5.0 * TAU_REL, HYDRO_CROSSING_T_END),
)
DT_INITIAL = _env_float("TENRYU_I1_LE08_DT_INITIAL_S", min(1.0e-12, T_END / 1000.0))
DT_MAX = _env_float("TENRYU_I1_LE08_DT_MAX_S", max(DT_INITIAL, T_END / 200.0))
MAX_STEPS = _env_int("TENRYU_I1_LE08_MAX_STEPS", 1000000)
PLOT_EVERY_STEPS = _env_int("TENRYU_I1_LE08_PLOT_EVERY_STEPS", 0)
PLOT_EVERY_S = _env_float("TENRYU_I1_LE08_PLOT_EVERY_S", T_END / 20.0)
HISTORY_EVERY_STEPS = _env_int("TENRYU_I1_LE08_HISTORY_EVERY_STEPS", 1)
CONDUCTION_KAPPA = _env_float("TENRYU_I1_LE08_CONDUCTION_KAPPA", 1.0e-20)

CASE_NAME = (
    f"i1_le08_grey_ned_radshock_M{_safe_float_token(M0)}"
    f"_nr{NR}_seed{SEED}_{INIT_MODE}"
)

print(
    "[deck:i1_le08_grey_ned_radshock] "
    f"M0={M0} nr={NR} seed={SEED} outdir={OUTDIR} "
    f"reference={REFERENCE_PATH} init_mode={INIT_MODE} "
    f"r_min_cm={R_MIN} r_max_cm={R_MAX} shock_r_cm={SHOCK_R_CM} "
    f"profile_scale={PROFILE_SCALE} tau_rel_s={TAU_REL} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"rho0_gcc={PARAMS['rho0_g_per_cc']} T0_eV={PARAMS['T0_eV']} "
    f"kappa_R_cm2_g={KAPPA_R} gamma={GAMMA} zbar={ZBAR} "
    "hydro=True grey_fld=True conduction=True qei=True laser=False dimension=1D_SPH"
)


def _interp(xs, values, x):
    if x <= xs[0]:
        return values[0]
    if x >= xs[-1]:
        return values[-1]
    hi = bisect.bisect_right(xs, x)
    lo = hi - 1
    w = (x - xs[lo]) / (xs[hi] - xs[lo])
    return values[lo] + w * (values[hi] - values[lo])


def _reference_coord(r):
    x = X_TABLE[0] + (r - R_MIN) * PROFILE_SCALE
    if x <= X_TABLE[0]:
        return X_TABLE[0]
    if x >= X_TABLE[-1]:
        return X_TABLE[-1]
    return x


def _two_state_selector(r):
    return UPSTREAM if r < SHOCK_R_CM else DOWNSTREAM


def rho_init(r):
    if INIT_MODE == "two_state":
        return float(_two_state_selector(r)["rho"])
    return _interp(X_TABLE, RHO_TABLE, _reference_coord(r))


def Te_init(r):
    if INIT_MODE == "two_state":
        return float(_two_state_selector(r)["T"])
    return _interp(X_TABLE, T_TABLE, _reference_coord(r))


def Ti_init(r):
    return Te_init(r)


def velocity_init(r):
    if INIT_MODE == "two_state":
        return float(_two_state_selector(r)["u"])
    return _interp(X_TABLE, U_TABLE, _reference_coord(r))


def volfrac_h(r):
    del r
    return 1.0


Main(
    name=CASE_NAME,
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_I1_LE08_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    nr=NR,
    grid="uniform",
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[
        Material(
            name="H",
            A=A_AMU,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=KAPPA_R, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(H=volfrac_h),
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
        cfl_hydro=0.2,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_1d="free",
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(
        enabled=True,
        solver=os.environ.get("TENRYU_I1_LE08_COND_SOLVER", "sts"),
        f_lim=0.06,
        test_kappa=CONDUCTION_KAPPA,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
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
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="none",
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        boundary=dict(inner_r="reflect", outer_r="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

Laser(enabled=False)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "rad_E", "zbar"],
)

Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
