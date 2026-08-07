"""2D RZ z-slab grey S_N radiative-shock deck."""

import bisect
import json
import os
from pathlib import Path

from tenryu_namelist import *


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_optional_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return float(value)


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _fld_multigroup_diffusion_overrides():
    overrides = {}
    if "TENRYU_I3_SN_RS_FLD_STATE_SUPPLY_POLICY" in os.environ:
        overrides["state_supply_boundary_policy"] = os.environ.get(
            "TENRYU_I3_SN_RS_FLD_STATE_SUPPLY_POLICY", "local_D_current"
        )
    if "TENRYU_I3_SN_RS_FLD_DIAG_AUDIT" in os.environ:
        overrides["diagnostic_radial_fourier_substage_enabled"] = _env_bool(
            "TENRYU_I3_SN_RS_FLD_DIAG_AUDIT", False
        )
    if "TENRYU_I3_SN_RS_FLD_CG_INNER_TOL" in os.environ:
        overrides["cg_inner_tol"] = _env_float(
            "TENRYU_I3_SN_RS_FLD_CG_INNER_TOL", 1.0e-10
        )
    return overrides


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _repo_root():
    return Path(__file__).resolve().parents[2]


MACH = _env_float("TENRYU_I3_SN_RS_MACH", 2.0)
NR = _env_int("TENRYU_I3_SN_RS_NR", 8)
NZ = _env_int("TENRYU_I3_SN_RS_NZ", 128)
N_ANGLES = _env_int("TENRYU_I3_SN_RS_N_ANGLES", 16)
if N_ANGLES not in (4, 8, 16, 32):
    raise ValueError("TENRYU_I3_SN_RS_N_ANGLES must be one of 4, 8, 16, 32")
KAPPA_SCALE = _env_int("TENRYU_I3_SN_RS_KAPPA_SCALE", 1)
if KAPPA_SCALE not in (1, 20, 200):
    raise ValueError("TENRYU_I3_SN_RS_KAPPA_SCALE must be one of 1, 20, 200")
GROUPS = _env_int("TENRYU_I3_SN_RS_GROUPS", 1)
if GROUPS < 1:
    raise ValueError("TENRYU_I3_SN_RS_GROUPS must be >= 1")
_bounds_env = os.environ.get("TENRYU_I3_SN_RS_GROUP_BOUNDS_EV", "")
if _bounds_env:
    GROUP_BOUNDS_EV = [float(tok) for tok in _bounds_env.split(",")]
    if len(GROUP_BOUNDS_EV) != GROUPS + 1:
        raise ValueError("TENRYU_I3_SN_RS_GROUP_BOUNDS_EV must have GROUPS+1 entries")
elif GROUPS == 1:
    GROUP_BOUNDS_EV = [0.0, 1.0e6]
else:
    GROUP_BOUNDS_EV = [
        1.0 * (1500.0 / 1.0) ** (k / GROUPS) for k in range(GROUPS + 1)
    ]
RADIATION_MODE = os.environ.get("TENRYU_I3_SN_RS_RADIATION_MODE", "sn_transport")
if RADIATION_MODE not in ("sn_transport", "multigroup_diffusion"):
    raise ValueError(
        "TENRYU_I3_SN_RS_RADIATION_MODE must be sn_transport or multigroup_diffusion"
    )
SEED = _env_int("TENRYU_I3_SN_RS_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_I3_SN_RS_OUTDIR", "./build/output_verify_i3_sn_radshock_2d_rz_slab")
INIT_MODE = os.environ.get("TENRYU_I3_SN_RS_INIT_MODE", "reference_table").lower().replace("-", "_")
if INIT_MODE not in ("reference_table", "two_state"):
    raise ValueError("TENRYU_I3_SN_RS_INIT_MODE must be reference_table or two_state")

# === ALE config (production-aligned, matches 2d_rz_l2_laser_hydro.py) ===
# PR 4.1: Lagrangian + Riemann shock IC tangles cells (mesh.cu:35 assert).
# Production decks (L2/A1) all use ALE rezone with Winslow + axis_spine_only.
# Escape-valve flags are kept OFF so the strict "no escape valves" gate
# (emergency_cell_deactivation = thermal_subcycle_floor_hit =
#  axis_spike_floor = newton_invalid = 0) remains satisfiable.
I3_SN_RS_ALE = _env_bool("TENRYU_I3_SN_RS_ALE_ENABLED", True)
# PR 4.3 fallback (per high-AI consultation 2026-05-15): annular+conduction-off
# was insufficient; apply stronger AV (av_C1 0.1->0.5, av_C2 1.5->2.0), tighter
# hydro CFL (0.2->0.1), and ALE every step (5->1).
I3_SN_RS_ALE_EVERY_N_STEPS = _env_int("TENRYU_I3_SN_RS_ALE_EVERY_N_STEPS", 1)
I3_SN_RS_AV_C1 = _env_float("TENRYU_I3_SN_RS_AV_C1", 0.5)
I3_SN_RS_AV_C2 = _env_float("TENRYU_I3_SN_RS_AV_C2", 2.0)
I3_SN_RS_CFL_HYDRO = _env_float("TENRYU_I3_SN_RS_CFL_HYDRO", 0.1)
I3_SN_RS_AXIS_REPAIR_MODE = os.environ.get(
    "TENRYU_I3_SN_RS_AXIS_REPAIR_MODE", "axis_spine_only"
)
I3_SN_RS_LOCAL_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I3_SN_RS_LOCAL_BOUNDARY_REPAIR", False
)
I3_SN_RS_MULTI_NODE_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I3_SN_RS_MULTI_NODE_BOUNDARY_REPAIR", False
)
I3_SN_RS_MULTI_NODE_INTERIOR_REPAIR = _env_bool(
    "TENRYU_I3_SN_RS_MULTI_NODE_INTERIOR_REPAIR", False
)
I3_SN_RS_EMERGENCY_CELL_DEACTIVATION = _env_bool(
    "TENRYU_I3_SN_RS_EMERGENCY_CELL_DEACTIVATION", False
)
I3_SN_RS_CONSERVATIVE_REMAP = _env_bool(
    "TENRYU_I3_SN_RS_CONSERVATIVE_REMAP", True
)
I3_SN_RS_HYDRO_ONLY = _env_bool("TENRYU_I3_SN_RS_HYDRO_ONLY", False)
TOTAL_ENERGY_REMAP = _env_bool("TENRYU_I3_SN_RS_TOTAL_ENERGY_REMAP", True)
WORK_SPLIT_AUDIT = _env_bool("TENRYU_I3_SN_RS_WORK_SPLIT_AUDIT", False)
HLLC_Z_FLUX = _env_bool("TENRYU_I3_SN_RS_HLLC_Z_FLUX", True)
HLLC_Z_AUDIT = _env_bool("TENRYU_I3_SN_RS_HLLC_Z_AUDIT", False)
HLLC_Z_STRICT = _env_bool("TENRYU_I3_SN_RS_HLLC_Z_STRICT", False)
VERBOSE_DIAG = _env_bool("TENRYU_I3_SN_RS_VERBOSE_DIAG", False)
RADIAL_FOURIER_AUDIT = _env_bool(
    "TENRYU_I3_SN_RS_RADIAL_FOURIER_AUDIT", VERBOSE_DIAG
)
RADIAL_FOURIER_T_START = _env_float(
    "TENRYU_I3_SN_RS_RADIAL_FOURIER_T_START_S", 1.35e-5
)
RADIAL_FOURIER_T_END = _env_float(
    "TENRYU_I3_SN_RS_RADIAL_FOURIER_T_END_S", 1.70e-5
)
RADIAL_FOURIER_MAX_MODE = _env_int(
    "TENRYU_I3_SN_RS_RADIAL_FOURIER_MAX_MODE", -1
)
RADIAL_FOURIER_COMPLEX_AUDIT = os.environ.get("TENRYU_I3_SN_RS_RFA_V2", "0") == "1"
RADIAL_FOURIER_COMPLEX_M_TARGETS = [14, 15, 16]
RADIAL_FOURIER_COMPLEX_J_TARGETS = [507, 508, 509, 510, 511]
RADIAL_FOURIER_COMPLEX_FIELDS = [
    "rho", "M", "V", "M_over_V", "P_r", "P_z", "u_r", "u_z",
    "E_e", "E_i", "E_rad", "T_e", "T_i", "x_r", "x_z",
    "A_r", "A_z", "dV_swept", "Q_visc", "lambda_FLD", "R_FLD",
    "kappa_eff", "f_Fleck",
]

MESH_MOTION = os.environ.get(
    "TENRYU_I3_SN_RS_MESH_MOTION",
    "ale" if I3_SN_RS_ALE else "lagrangian",
).lower()
if MESH_MOTION not in ("ale", "lagrangian"):
    raise ValueError(
        "TENRYU_I3_SN_RS_MESH_MOTION must be 'ale' or 'lagrangian'"
    )

ale_config = dict(enabled=False)
if I3_SN_RS_ALE:
    ale_config = dict(
        enabled=True,
        every_n_steps=I3_SN_RS_ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
        axis_repair_mode=I3_SN_RS_AXIS_REPAIR_MODE,
        conservative_remap_enabled=I3_SN_RS_CONSERVATIVE_REMAP,
        conservative_remap_target="reference",
        conservative_remap_radiation_enabled=not I3_SN_RS_HYDRO_ONLY,
        conservative_remap_order="second_order_van_leer",
        local_boundary_repair_enabled=I3_SN_RS_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=I3_SN_RS_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=I3_SN_RS_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=I3_SN_RS_EMERGENCY_CELL_DEACTIVATION,
    )

BOUNDARY_MODE_RAW = os.environ.get("TENRYU_I3_SN_RS_BOUNDARY_MODE", "state_supply").lower()
BOUNDARY_MODE = "reflecting" if BOUNDARY_MODE_RAW == "finite_shock_tube" else BOUNDARY_MODE_RAW
if BOUNDARY_MODE not in ("reflecting", "state_supply"):
    raise ValueError("TENRYU_I3_SN_RS_BOUNDARY_MODE must be reflecting or state_supply")
PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_I3_SN_RS_PRODUCTION_AUDIT_TIER", "none")
if PRODUCTION_AUDIT_TIER not in ("none", "A", "B"):
    raise ValueError('TENRYU_I3_SN_RS_PRODUCTION_AUDIT_TIER must be one of: "none", "A", "B"')

SN_REFERENCE_DIR = _repo_root() / "tests" / "verification" / "data" / "sn_radshock_reference"
if RADIATION_MODE == "sn_transport":
    default_ref = SN_REFERENCE_DIR / (
        f"sn_radshock_M{_safe_float_token(MACH)}_N{N_ANGLES}_K{KAPPA_SCALE}.json"
    )
elif KAPPA_SCALE == 1:
    default_ref = (
        _repo_root()
        / "tests"
        / "verification"
        / "data"
        / "fld_const_eddington_reference"
        / f"fld_ced_M{_safe_float_token(MACH)}.json"
    )
else:
    default_ref = SN_REFERENCE_DIR / f"fld_ced_M{_safe_float_token(MACH)}_K{KAPPA_SCALE}.json"
REFERENCE_PATH = Path(os.environ.get("TENRYU_I3_SN_RS_REFERENCE", str(default_ref)))
with REFERENCE_PATH.open(encoding="utf-8") as stream:
    REF = json.load(stream)

PARAMS = REF["parameters"]
TABLE = REF["table"]
UPSTREAM = REF["states"]["upstream"]
DOWNSTREAM = REF["states"]["downstream"]
GAMMA = float(PARAMS["gamma"])
A_AMU = float(PARAMS["A_amu"])
ZBAR = float(PARAMS["zbar"])
KAPPA_R = float(PARAMS.get("kappa_R_cm2_per_g", PARAMS.get("kappa_a_cm2_per_g", 0.5 * KAPPA_SCALE)))
C_LIGHT_CM_PER_S = float(PARAMS["c_cm_per_s"])
A_EV_ERG_CM3_EV4 = float(PARAMS["a_eV_erg_cm3_eV4"])
CV_ERG_PER_G_EV = float(PARAMS.get("Cv_erg_per_g_eV", 0.0))
if CV_ERG_PER_G_EV <= 0.0:
    EV_TO_ERG = float(PARAMS["eV_to_erg"])
    PROTON_MASS_G = float(PARAMS["proton_mass_g"])
    CV_ERG_PER_G_EV = ZBAR * EV_TO_ERG / (A_AMU * PROTON_MASS_G * (GAMMA - 1.0))
X_TABLE = [float(x) for x in TABLE["x_cm"]]
RHO_TABLE = [float(x) for x in TABLE["rho_g_per_cc"]]
T_TABLE = [float(x) for x in TABLE["T_eV"]]
U_TABLE = [float(x) for x in TABLE["u_cm_per_s"]]

R_MIN = _env_float("TENRYU_I3_SN_RS_R_MIN_CM", 0.5)  # PR 4.2: annular slab, axis row removed
R_MAX = _env_float("TENRYU_I3_SN_RS_R_MAX_CM", 10.0)
I3_SN_RS_ANNULAR_SLAB = R_MIN > 0.0
# PR 4.6 enabled r_inner="reflect" for 2D_RZ when r_min > 0 (annular).
# r_inner="axis" with r_min > 0 was a legacy ambiguity (semantically reflective).
R_INNER_BC = "reflect" if R_MIN > 0.0 else "axis"
Z_MIN = _env_float("TENRYU_I3_SN_RS_Z_MIN_CM", X_TABLE[0])
Z_MAX = _env_float("TENRYU_I3_SN_RS_Z_MAX_CM", X_TABLE[-1])
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("2D RZ slab bounds require R_MAX>0 and Z_MAX>Z_MIN")

DOMAIN_SPAN = max(X_TABLE[-1] - X_TABLE[0], 1.0e-30)
PROFILE_SCALE = DOMAIN_SPAN / (Z_MAX - Z_MIN)
PROFILE_SHOCK_Z_CM = Z_MIN - X_TABLE[0] / PROFILE_SCALE
TWO_STATE_SHOCK_Z_CM = 0.5 * (Z_MIN + Z_MAX)
SHOCK_Z_CM = TWO_STATE_SHOCK_Z_CM if INIT_MODE == "two_state" else PROFILE_SHOCK_Z_CM


def _compute_tau_rel(rho_gcc, kappa_R_cm2_g, T_eV):
    sigma_a = kappa_R_cm2_g * rho_gcc
    return rho_gcc * CV_ERG_PER_G_EV / (
        4.0 * C_LIGHT_CM_PER_S * sigma_a * A_EV_ERG_CM3_EV4 * T_eV**3
    )


TAU_REL = _compute_tau_rel(float(UPSTREAM["rho"]), KAPPA_R, float(PARAMS["T0_eV"]))
HYDRO_CROSSING_T_END = 3.0 * DOMAIN_SPAN / max(abs(float(UPSTREAM["u"])), 1.0e-300)
# Material-radiation relaxation time:
# tau_rel = rho Cv / (4 c sigma_a a_eV T^3), sigma_a = rho * (0.5*K).
DEFAULT_T_END = 5.0 * TAU_REL if KAPPA_SCALE > 1 else max(5.0 * TAU_REL, HYDRO_CROSSING_T_END)
T_END = _env_float(
    "TENRYU_I3_SN_RS_T_END_S",
    DEFAULT_T_END,
)
DT_INITIAL = _env_float("TENRYU_I3_SN_RS_DT_INITIAL_S", min(1.0e-12, T_END / 1000.0))
DEFAULT_DT_MAX = {1: None, 20: 7.2e-10, 200: 7.2e-11}[KAPPA_SCALE]
DT_MAX = _env_optional_float("TENRYU_I3_SN_RS_DT_MAX_S", DEFAULT_DT_MAX)
MAX_STEPS = _env_int("TENRYU_I3_SN_RS_MAX_STEPS", 1000000)
PLOT_EVERY_STEPS = _env_int("TENRYU_I3_SN_RS_PLOT_EVERY_STEPS", 0)
PLOT_EVERY_S = _env_float("TENRYU_I3_SN_RS_PLOT_EVERY_S", T_END / 20.0)
CHECKPOINT_EVERY = _env_int("TENRYU_I3_2D_RZ_CKPT_EVERY", 0)
CHECKPOINT_EVERY_S = _env_float("TENRYU_I3_2D_RZ_CKPT_EVERY_S", -1.0)
HISTORY_EVERY_STEPS = _env_int(
    "TENRYU_I3_SN_RS_HISTORY_EVERY_STEPS",
    1 if VERBOSE_DIAG else 0,
)
CONDUCTION_KAPPA = _env_float("TENRYU_I3_SN_RS_CONDUCTION_KAPPA", 1.0e-20)
I3_SN_RS_CONDUCTION_ENABLED = _env_bool("TENRYU_I3_SN_RS_CONDUCTION_ENABLED", False)
LINEAR_SOLVER_2D = os.environ.get("TENRYU_I3_SN_RS_LINEAR_SOLVER", "cusparse_cg_jacobi")
PRODUCTION_AUDIT = dict(enabled=False, tier="none")
if PRODUCTION_AUDIT_TIER in ("A", "B"):
    PRODUCTION_AUDIT = dict(
        enabled=True,
        tier=PRODUCTION_AUDIT_TIER,
        gcl=dict(enabled=False),
        positivity=dict(enabled=False),
        audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
    )

CASE_NAME = (
    f"i3_sn_radshock_2d_rz_slab_M{_safe_float_token(MACH)}"
    f"_nr{NR}_nz{NZ}_seed{SEED}_{INIT_MODE}"
)

print(
    "[deck:i3_sn_radshock_2d_rz_slab] "
    f"M0={MACH} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"reference={REFERENCE_PATH} init_mode={INIT_MODE} "
    f"radiation_mode={RADIATION_MODE} n_angles={N_ANGLES} kappa_scale={KAPPA_SCALE} "
    f"r_min_cm={R_MIN} r_max_cm={R_MAX} z_min_cm={Z_MIN} z_max_cm={Z_MAX} "
    f"annular_slab={I3_SN_RS_ANNULAR_SLAB} "
    f"shock_z_cm={SHOCK_Z_CM} profile_scale={PROFILE_SCALE} "
    f"tau_rel_s={TAU_REL} t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"checkpoint_every={CHECKPOINT_EVERY} checkpoint_every_s={CHECKPOINT_EVERY_S} "
    f"rho0_gcc={PARAMS['rho0_g_per_cc']} T0_eV={PARAMS['T0_eV']} "
    f"kappa_R_cm2_g={KAPPA_R} gamma={GAMMA} zbar={ZBAR} "
    f"hydro=True radiation={RADIATION_MODE} conduction={I3_SN_RS_CONDUCTION_ENABLED} qei=True "
    f"laser=False dimension=2D_RZ boundary_mode={BOUNDARY_MODE} "
    f"linear_solver_2d={LINEAR_SOLVER_2D} "
    f"av_C1={I3_SN_RS_AV_C1} av_C2={I3_SN_RS_AV_C2} cfl_hydro={I3_SN_RS_CFL_HYDRO} "
    f"ale_enabled={I3_SN_RS_ALE} ale_every_n={I3_SN_RS_ALE_EVERY_N_STEPS} "
    f"axis_repair={I3_SN_RS_AXIS_REPAIR_MODE} "
    f"production_audit_tier={PRODUCTION_AUDIT_TIER}"
)
if I3_SN_RS_HYDRO_ONLY:
    print(
        "[deck:i3_sn_radshock_2d_rz_slab:hydro_only] "
        "enabled=True radiation_enabled=False radiation_field=zero "
        "radiation_thermal_subcycle=False conservative_remap_radiation_enabled=False"
    )
if TOTAL_ENERGY_REMAP:
    print("[deck:i3_sn_radshock_2d_rz_slab:total_energy_remap] enabled=True")
if WORK_SPLIT_AUDIT:
    print("[deck:i3_sn_radshock_2d_rz_slab:work_split_audit] enabled=True")
if HLLC_Z_FLUX:
    print(
        "[deck:i3_sn_radshock_2d_rz_slab:hllc_z_flux] "
        f"enabled=True audit={HLLC_Z_AUDIT} strict_quasi_1d={HLLC_Z_STRICT}"
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


def _reference_coord(z):
    x = X_TABLE[0] + (z - Z_MIN) * PROFILE_SCALE
    if x <= X_TABLE[0]:
        return X_TABLE[0]
    if x >= X_TABLE[-1]:
        return X_TABLE[-1]
    return x


def _two_state_selector(z):
    return UPSTREAM if z < SHOCK_Z_CM else DOWNSTREAM


def rho_init(r, z):
    del r
    if INIT_MODE == "two_state":
        return float(_two_state_selector(z)["rho"])
    return _interp(X_TABLE, RHO_TABLE, _reference_coord(z))


def Te_init(r, z):
    del r
    if INIT_MODE == "two_state":
        return float(_two_state_selector(z)["T"])
    return _interp(X_TABLE, T_TABLE, _reference_coord(z))


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    del r
    if INIT_MODE == "two_state":
        return (0.0, float(_two_state_selector(z)["u"]))
    return (0.0, _interp(X_TABLE, U_TABLE, _reference_coord(z)))


def volfrac_h(r, z):
    del r, z
    return 1.0


def _state_supply(state):
    return dict(
        type="state_supply",
        rho_g_per_cc=float(state["rho"]),
        u_z_cm_per_s=float(state["u"]),
        T_eV=float(state["T"]),
    )


# The strict gate defaults to reflecting z boundaries with a finite shock-tube
# IC, matching the PR 3 mode that exercised the 2D RZ FLD path successfully.
# Keep state_supply selectable for focused PR 4 hardening/repros.
if BOUNDARY_MODE == "state_supply":
    HYDRO_BOUNDARY = dict(
        r_inner=R_INNER_BC,
        r_outer="reflect",
        z_bottom=_state_supply(UPSTREAM),
        z_top=_state_supply(DOWNSTREAM),
        mesh_tangential_target="reference",
        state_supply_donor_mode=os.environ.get(
            "TENRYU_I3_SN_RS_STATE_SUPPLY_DONOR_MODE", "interior_radial_average"
        ),
    )
else:
    HYDRO_BOUNDARY = dict(r_inner=R_INNER_BC, r_outer="reflect", z_bottom="reflect", z_top="reflect")

FLD_BOUNDARY = dict(
    inner_r="reflect",
    outer_r="reflect",
    z="reflect",
    z_bottom="reflect",
    z_top="reflect",
)
SN_BOUNDARY = dict(
    inner_r="reflect_parity",
    outer_r="reflect",
    z="reflect",
    z_bottom="reflect",
    z_top="reflect",
)

DT_CONFIG = dict(
    initial_s=DT_INITIAL,
    cfl_hydro=I3_SN_RS_CFL_HYDRO,
    cfl_cond=0.3,
    growth_factor=1.1,
    min_s=1.0e-22,
)
if DT_MAX is not None:
    DT_CONFIG["max_s"] = DT_MAX

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get(
        "TENRYU_I3_SN_RS_VERBOSITY",
        "verbose" if VERBOSE_DIAG else "quiet",
    ),
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
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=0.1, Ti_floor_eV=0.1),
)

Materials(
    materials=[
        Material(
            name="H",
            A=A_AMU,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(
                model="constant",
                kappa_a=0.5 * KAPPA_SCALE,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
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
    radiation_field="zero" if I3_SN_RS_HYDRO_ONLY else "equilibrium",
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=not I3_SN_RS_HYDRO_ONLY,
    dt=DT_CONFIG,
    hydro=dict(
        enabled=True,
        boundary_2d=HYDRO_BOUNDARY,
        # PR 4.3 fallback (per high-AI consultation 2026-05-15): AV 0.1/1.5->0.5/2.0.
        av_C1=I3_SN_RS_AV_C1,
        av_C2=I3_SN_RS_AV_C2,
        total_energy_remap_2d_rz=TOTAL_ENERGY_REMAP,
        work_split_audit_2d_rz=WORK_SPLIT_AUDIT,
        hllc_z_flux_2d_rz=HLLC_Z_FLUX,
        hllc_z_flux_audit_2d_rz=HLLC_Z_AUDIT,
        hllc_z_flux_strict_quasi_1d=HLLC_Z_STRICT,
        work_split_audit_cell_every_n_steps=_env_int(
            "TENRYU_I3_SN_RS_WORK_SPLIT_AUDIT_CELL_EVERY_N_STEPS", 0
        ),
        work_split_audit_all_rows=_env_bool(
            "TENRYU_I3_SN_RS_WORK_SPLIT_AUDIT_ALL_ROWS", False
        ),
        rz_geometric_cfl_enabled=True,
        rz_geometric_cfl_etaV=0.5,
        rz_geometric_cfl_r_floor=1.0e-10,
        rz_geometric_cfl_cumulative_protection_enabled=True,
        rz_geometric_cfl_v_initial_floor=0.1,
        geometric_retry_stagnation=dict(force_diagnostic_dump=VERBOSE_DIAG),
    ),
    conduction=dict(
        enabled=I3_SN_RS_CONDUCTION_ENABLED,
        solver=os.environ.get("TENRYU_I3_SN_RS_COND_SOLVER", "sts"),
        f_lim=0.06,
        test_kappa=CONDUCTION_KAPPA,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    ale=ale_config,
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

if RADIATION_MODE == "sn_transport":
    Radiation(
        enabled=not I3_SN_RS_HYDRO_ONLY,
        mode="sn_transport",
        groups=GROUPS,
        group_bounds_eV=GROUP_BOUNDS_EV,
        imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
        ddmc=dict(enabled=False),
        holo=dict(enabled=False),
        sn_transport=dict(
            spatial_scheme="linear_characteristic",
            angular_quadrature="product",
            n_angles=N_ANGLES,
            dsa_enabled=True,
            max_outer_iterations=50,
            inner_tol=_env_float("TENRYU_I3_SN_RS_SN_INNER_TOL", 1.0e-6),
            outer_tol=1.0e-8,
            boundary=SN_BOUNDARY,
        ),
    )
else:
    Radiation(
        enabled=not I3_SN_RS_HYDRO_ONLY,
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
            linear_solver_2d=LINEAR_SOLVER_2D,
            boundary=FLD_BOUNDARY,
            **_fld_multigroup_diffusion_overrides(),
        ),
    )

Laser(enabled=False)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
    checkpoint_every=CHECKPOINT_EVERY,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=CHECKPOINT_EVERY_S,
    plot_fields=(
        ["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "rad_E", "zbar", "volfrac"]
        + (["sn_F_z", "sn_P_zz", "sn_chi_z"] if RADIATION_MODE == "sn_transport" else [])
    ),
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    per_operator_radial_fourier_enabled=RADIAL_FOURIER_AUDIT,
    radial_fourier_window_t_start_s=RADIAL_FOURIER_T_START,
    radial_fourier_window_t_end_s=RADIAL_FOURIER_T_END,
    radial_fourier_max_mode=RADIAL_FOURIER_MAX_MODE,
    per_operator_radial_fourier_complex_enabled=RADIAL_FOURIER_COMPLEX_AUDIT,
    per_operator_radial_fourier_complex_m_targets=RADIAL_FOURIER_COMPLEX_M_TARGETS,
    per_operator_radial_fourier_complex_j_targets=RADIAL_FOURIER_COMPLEX_J_TARGETS,
    per_operator_radial_fourier_complex_fields=RADIAL_FOURIER_COMPLEX_FIELDS,
)
