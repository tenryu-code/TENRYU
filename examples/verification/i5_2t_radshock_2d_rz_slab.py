"""2D RZ z-slab 2T finite-Qei FLD-CED radiative-shock deck (I5).

The deck follows the I2 z-slab pattern: env-parameterized, annular slab.  The default is
reference_table init with z state-supply from the reference far states (spec
i5_2t_refgen_spec.md :302 canonical); the reflecting finite-shock-tube scaffold remains
selectable via TENRYU_I5_2D_RZ_BOUNDARY_MODE=reflecting and TENRYU_I5_2D_RZ_INIT_MODE=two_state.
"""

import bisect
import json
import os
import sys
from pathlib import Path


def _print_help_and_exit():
    print(
        """I5 2T finite-Qei FLD-CED 2D RZ z-slab deck.

Required reference tables are selected by:
  TENRYU_I5_2D_RZ_REGIME={R0p1,R1,R10}       default: R1

Primary grid/output knobs:
  TENRYU_I5_2D_RZ_NZ                         default: regime production rung
  TENRYU_I5_2D_RZ_NR                         default: 8
  TENRYU_I5_2D_RZ_R_MIN_CM/R_MAX_CM           default: centered at 0.5 cm with dr=dz
  TENRYU_I5_2D_RZ_PRODUCTION_AUDIT_TIER      default: A
  TENRYU_I5_2D_RZ_OUTDIR                     default: ./build/output_verify_i5_2t_radshock_2d_rz_slab
  TENRYU_I5_2D_RZ_SEED                       default: 12345

Dry parse:
  python3 examples/verification/i5_2t_radshock_2d_rz_slab.py --help

Normal parsing is performed by TENRYU with tenryu_namelist available; this
help path exits before importing tenryu_namelist and requires no GPU.
"""
    )
    raise SystemExit(0)


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    _print_help_and_exit()

from tenryu_namelist import *


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _repo_root():
    return Path(__file__).resolve().parents[2]


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def _fld_multigroup_diffusion_overrides():
    overrides = {}
    if "TENRYU_I5_2D_RZ_FLD_DIAG_AUDIT" in os.environ:
        overrides["diagnostic_radial_fourier_substage_enabled"] = _env_bool(
            "TENRYU_I5_2D_RZ_FLD_DIAG_AUDIT", False
        )
    if "TENRYU_I5_2D_RZ_FLD_CG_INNER_TOL" in os.environ:
        overrides["cg_inner_tol"] = _env_float(
            "TENRYU_I5_2D_RZ_FLD_CG_INNER_TOL", 1.0e-10
        )
    if "TENRYU_I5_2D_RZ_FLD_CG_MAX_ITER" in os.environ:
        overrides["cg_max_iter"] = _env_int(
            "TENRYU_I5_2D_RZ_FLD_CG_MAX_ITER", 500
        )
    return overrides


REGIME = os.environ.get("TENRYU_I5_2D_RZ_REGIME", "R1")
REFERENCE_BY_REGIME = {
    "R0p1": "fld_ced_2t_M3_APRIME_SCALED_R0p1.json",
    "R1": "fld_ced_2t_M3_APRIME_SCALED_R1.json",
    "R10": "fld_ced_2t_M3_APRIME_SCALED_R10.json",
}
RECOMMENDED_NZ_BY_REGIME = {
    "R0p1": 1024,
    "R1": 512,
    "R10": 2048,
}
if REGIME not in REFERENCE_BY_REGIME:
    raise ValueError("TENRYU_I5_2D_RZ_REGIME must be one of: R0p1, R1, R10")

NR = _env_int("TENRYU_I5_2D_RZ_NR", 8)
NZ = _env_int("TENRYU_I5_2D_RZ_NZ", RECOMMENDED_NZ_BY_REGIME[REGIME])
SEED = _env_int("TENRYU_I5_2D_RZ_SEED", 12345)
OUTDIR = os.environ.get(
    "TENRYU_I5_2D_RZ_OUTDIR",
    "./build/output_verify_i5_2t_radshock_2d_rz_slab",
)
INIT_MODE = os.environ.get("TENRYU_I5_2D_RZ_INIT_MODE", "reference_table").lower().replace("-", "_")
if INIT_MODE not in ("two_state", "reference_table"):
    raise ValueError("TENRYU_I5_2D_RZ_INIT_MODE must be two_state or reference_table")
BOUNDARY_MODE_RAW = os.environ.get("TENRYU_I5_2D_RZ_BOUNDARY_MODE", "state_supply").lower()
BOUNDARY_MODE = "reflecting" if BOUNDARY_MODE_RAW == "finite_shock_tube" else BOUNDARY_MODE_RAW
if BOUNDARY_MODE not in ("reflecting", "state_supply"):
    raise ValueError("TENRYU_I5_2D_RZ_BOUNDARY_MODE must be reflecting or state_supply")

default_ref = (
    _repo_root()
    / "tests"
    / "verification"
    / "data"
    / "fld_ced_2t_reference"
    / REFERENCE_BY_REGIME[REGIME]
)
REFERENCE_PATH = Path(os.environ.get("TENRYU_I5_2D_RZ_REFERENCE", str(default_ref)))
with REFERENCE_PATH.open(encoding="utf-8") as stream:
    REF = json.load(stream)

PARAMS = REF["parameters"]
TABLE = REF["table"]
INVARIANTS = REF["invariants"]
UPSTREAM = REF["states"]["upstream"]
DOWNSTREAM = REF["states"]["downstream"]


def _state_supply(name, state):
    te = float(state["T_e_eV"])
    ti = float(state["T_i_eV"])
    if abs(te - ti) > 1.0e-12 * max(te, 1.0):
        raise ValueError(
            f"I5 far state '{name}' is not Te=Ti equilibrated (Te={te} eV, Ti={ti} eV); "
            "state_supply imposes a single T"
        )
    return dict(
        type="state_supply",
        rho_g_per_cc=float(state["rho"]),
        u_z_cm_per_s=float(state["u"]),
        T_eV=te,
    )


GAMMA = float(PARAMS["gamma"])
A_AMU = float(PARAMS["A_amu"])
ZBAR = float(PARAMS["zbar"])
KAPPA_R = float(PARAMS["kappa_R_cm2_per_g"])
C_LIGHT_CM_PER_S = float(PARAMS["c_cm_per_s"])
A_EV_ERG_CM3_EV4 = float(PARAMS["a_eV_erg_cm3_eV4"])
EV_TO_ERG = float(PARAMS["eV_to_erg"])
PROTON_MASS_G = float(PARAMS["proton_mass_g"])
MACH = float(PARAMS["M0"])
QEI_MULTIPLIER = float(PARAMS["qei_multiplier"])
TABLE_REGIME_TAG = str(PARAMS["regime_tag"])
TAU_EI = float(INVARIANTS["tau_ei_s"])
TAU_RAD = float(INVARIANTS["tau_rad_s"])
L_EI = float(INVARIANTS["l_ei_cm"])
L_RAD = float(INVARIANTS["l_rad_cm"])

X_TABLE = [float(x) for x in TABLE["x_cm"]]
RHO_TABLE = [float(x) for x in TABLE["rho_g_per_cc"]]
TE_TABLE = [float(x) for x in TABLE["Te_eV"]]
TI_TABLE = [float(x) for x in TABLE["Ti_eV"]]
U_TABLE = [float(x) for x in TABLE["u_cm_per_s"]]

Z_MIN = _env_float("TENRYU_I5_2D_RZ_Z_MIN_CM", X_TABLE[0])
Z_MAX = _env_float("TENRYU_I5_2D_RZ_Z_MAX_CM", X_TABLE[-1])
if not (NR > 0 and NZ > 0):
    raise ValueError("2D RZ slab grid requires NR>0 and NZ>0")
if not (Z_MAX > Z_MIN):
    raise ValueError("2D RZ slab z bounds require Z_MAX>Z_MIN")
DOMAIN_SPAN = max(Z_MAX - Z_MIN, 1.0e-300)
DZ = DOMAIN_SPAN / float(NZ)
R_CENTER_CM = 0.5
DEFAULT_R_MIN = R_CENTER_CM - 0.5 * float(NR) * DZ
DEFAULT_R_MAX = R_CENTER_CM + 0.5 * float(NR) * DZ
R_MIN_OVERRIDDEN = "TENRYU_I5_2D_RZ_R_MIN_CM" in os.environ
R_MAX_OVERRIDDEN = "TENRYU_I5_2D_RZ_R_MAX_CM" in os.environ
RADIAL_BOUNDS_OVERRIDDEN = R_MIN_OVERRIDDEN or R_MAX_OVERRIDDEN
if RADIAL_BOUNDS_OVERRIDDEN:
    R_MIN = _env_float("TENRYU_I5_2D_RZ_R_MIN_CM", 0.5)
    R_MAX = _env_float("TENRYU_I5_2D_RZ_R_MAX_CM", 10.0)
else:
    R_MIN = DEFAULT_R_MIN
    R_MAX = DEFAULT_R_MAX
if not (R_MAX > R_MIN):
    raise ValueError("2D RZ slab radial bounds require R_MAX>R_MIN")
DR = (R_MAX - R_MIN) / float(NR) if RADIAL_BOUNDS_OVERRIDDEN else DZ
RADIAL_CELL_ASPECT = DR / DZ
RADIAL_WIDTH_OVER_CENTER = float(NR) * DR / R_CENTER_CM
I5_2D_RZ_ANNULAR_SLAB = R_MIN > 0.0
R_INNER_BC = "reflect" if R_MIN > 0.0 else "axis"
REFERENCE_SPAN = max(X_TABLE[-1] - X_TABLE[0], 1.0e-300)
PROFILE_SCALE = REFERENCE_SPAN / DOMAIN_SPAN
PROFILE_SHOCK_Z_CM = Z_MIN - X_TABLE[0] / PROFILE_SCALE
TWO_STATE_SHOCK_Z_CM = 0.5 * (Z_MIN + Z_MAX)
SHOCK_Z_CM = TWO_STATE_SHOCK_Z_CM if INIT_MODE == "two_state" else PROFILE_SHOCK_Z_CM

# Similarity-rescale transform (verbatim):
# under x -> lambda x with lambda = 1e6: kappa_R -> kappa_R/lambda (optical depth
# per structure invariant), qei_multiplier -> 1/lambda (tau_ei -> lambda
# tau_ei so l_ei scales with the domain), t -> lambda t. Then eps = l_ei/
# l_rad is INVARIANT, the (Ti, Te, E)(x/L) profiles in eV are IDENTICAL,
# and domains/t_end land at cm / 3e-7 s scale — inside the kernel's proven
# envelope.
#
# A-prime scaled deck numbers are derived from the committed table span L and the
# cost-reviewed spec-4.4 ladder:
#   cells/lam_R0 = {R0p1:60, R1:20, R10:20}, nz rounded up to power-of-two
#   production nz = {R0p1:1024, R1:512, R10:2048}
#   smoke nz = production/4, richardson nz = 2*production
#   t_cross = 3 L / |u0|
#   t_end = max(5 tau_rad, t_cross)
#   dz = L / nz
#   c_s0 = |u0| / M0
#   dt_max = min(t_end/200, CFL_hydro * dz / (|u0| + c_s0))
# Production rungs freeze:
#   R0p1: nz=1024,  t_end=3.2014174731668974e-06 s,
#         dt_max=7.8159606278488716e-11 s, cells/lam_R0=6.3275192722383409e+01,
#         l_ei/dz=8.1268134953444626e+00, dr=3.0598619184020421e-02 cm,
#         curvature=4.8957790694432674e-01, estimated_steps=40960.
#   R1:   nz=512,   t_end=3.0439407583493335e-07 s,
#         dt_max=1.4862991984127605e-11 s, cells/lam_R0=3.3274351191593304e+01,
#         l_ei/dz=4.2736250128719604e+01, dr=5.8186965532685438e-03 cm,
#         curvature=9.3099144852296700e-02, estimated_steps=20480.
#   R10:  nz=2048,  t_end=1.4379122073105449e-07 s,
#         dt_max=1.7552639249396299e-12 s, cells/lam_R0=2.8175615530565167e+01,
#         l_ei/dz=3.6187637315948274e+02, dr=6.8716636334257707e-04 cm,
#         curvature=1.0994661813481233e-02, estimated_steps=81920.
U0_ABS = abs(float(UPSTREAM["u"]))
CS0 = U0_ABS / MACH
I5_2D_RZ_CFL_HYDRO = _env_float("TENRYU_I5_2D_RZ_CFL_HYDRO", 0.1)
HYDRO_CROSSING_T_END = 3.0 * DOMAIN_SPAN / max(U0_ABS, 1.0e-300)
RELAXATION_T_END = 5.0 * TAU_RAD
DEFAULT_T_END = max(RELAXATION_T_END, HYDRO_CROSSING_T_END)
CFL_DT_CAP = I5_2D_RZ_CFL_HYDRO * DZ / max(U0_ABS + CS0, 1.0e-300)
DEFAULT_DT_MAX = min(DEFAULT_T_END / 200.0, CFL_DT_CAP)
T_END = _env_float("TENRYU_I5_2D_RZ_T_END_S", DEFAULT_T_END)
DT_MAX = _env_float("TENRYU_I5_2D_RZ_DT_MAX_S", DEFAULT_DT_MAX)
DT_INITIAL = _env_float("TENRYU_I5_2D_RZ_DT_INITIAL_S", min(DT_MAX, T_END / 1000.0))
MAX_STEPS = _env_int("TENRYU_I5_2D_RZ_MAX_STEPS", 1000000)
PLOT_EVERY_STEPS = _env_int("TENRYU_I5_2D_RZ_PLOT_EVERY_STEPS", 0)
PLOT_EVERY_S = _env_float("TENRYU_I5_2D_RZ_PLOT_EVERY_S", T_END / 20.0)
HISTORY_EVERY_STEPS = _env_int("TENRYU_I5_2D_RZ_HISTORY_EVERY_STEPS", 0)
CONDUCTION_KAPPA = _env_float("TENRYU_I5_2D_RZ_CONDUCTION_KAPPA", 1.0e-20)
I5_2D_RZ_CONDUCTION_ENABLED = _env_bool("TENRYU_I5_2D_RZ_CONDUCTION_ENABLED", False)
LINEAR_SOLVER_2D = os.environ.get("TENRYU_I5_2D_RZ_LINEAR_SOLVER", "cusparse_cg_jacobi")

# Production-aligned ALE and hydro knobs follow the I1/I2 z-slab decks.
I5_2D_RZ_ALE = _env_bool("TENRYU_I5_2D_RZ_ALE_ENABLED", True)
I5_2D_RZ_ALE_EVERY_N_STEPS = _env_int("TENRYU_I5_2D_RZ_ALE_EVERY_N_STEPS", 1)
I5_2D_RZ_AV_C1 = _env_float("TENRYU_I5_2D_RZ_AV_C1", 0.5)
I5_2D_RZ_AV_C2 = _env_float("TENRYU_I5_2D_RZ_AV_C2", 2.0)
I5_2D_RZ_AXIS_REPAIR_MODE = os.environ.get(
    "TENRYU_I5_2D_RZ_AXIS_REPAIR_MODE", "axis_spine_only"
)
I5_2D_RZ_LOCAL_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I5_2D_RZ_LOCAL_BOUNDARY_REPAIR", False
)
I5_2D_RZ_MULTI_NODE_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I5_2D_RZ_MULTI_NODE_BOUNDARY_REPAIR", False
)
I5_2D_RZ_MULTI_NODE_INTERIOR_REPAIR = _env_bool(
    "TENRYU_I5_2D_RZ_MULTI_NODE_INTERIOR_REPAIR", False
)
I5_2D_RZ_EMERGENCY_CELL_DEACTIVATION = _env_bool(
    "TENRYU_I5_2D_RZ_EMERGENCY_CELL_DEACTIVATION", False
)
I5_2D_RZ_CONSERVATIVE_REMAP = _env_bool(
    "TENRYU_I5_2D_RZ_CONSERVATIVE_REMAP", True
)
I5_2D_RZ_HYDRO_ONLY = _env_bool("TENRYU_I5_2D_RZ_HYDRO_ONLY", False)
TOTAL_ENERGY_REMAP = _env_bool("TENRYU_I5_2D_RZ_TOTAL_ENERGY_REMAP", True)
WORK_SPLIT_AUDIT = _env_bool("TENRYU_I5_2D_RZ_WORK_SPLIT_AUDIT", False)
HLLC_Z_FLUX = _env_bool("TENRYU_I5_2D_RZ_HLLC_Z_FLUX", True)
HLLC_Z_AUDIT = _env_bool("TENRYU_I5_2D_RZ_HLLC_Z_AUDIT", False)
HLLC_Z_STRICT = _env_bool("TENRYU_I5_2D_RZ_HLLC_Z_STRICT", False)
VERBOSE_DIAG = _env_bool("TENRYU_I5_2D_RZ_VERBOSE_DIAG", False)
RADIAL_FOURIER_AUDIT = _env_bool(
    "TENRYU_I5_2D_RZ_RADIAL_FOURIER_AUDIT", VERBOSE_DIAG
)
RADIAL_FOURIER_T_START = _env_float(
    "TENRYU_I5_2D_RZ_RADIAL_FOURIER_T_START_S", 0.0
)
RADIAL_FOURIER_T_END = _env_float(
    "TENRYU_I5_2D_RZ_RADIAL_FOURIER_T_END_S", T_END
)
RADIAL_FOURIER_MAX_MODE = _env_int("TENRYU_I5_2D_RZ_RADIAL_FOURIER_MAX_MODE", -1)
RADIAL_FOURIER_COMPLEX_AUDIT = os.environ.get("I1_RFA_V2", "0") == "1"
RADIAL_FOURIER_COMPLEX_M_TARGETS = [14, 15, 16]
RADIAL_FOURIER_COMPLEX_J_TARGETS = [507, 508, 509, 510, 511]
RADIAL_FOURIER_COMPLEX_FIELDS = [
    "rho", "M", "V", "M_over_V", "P_r", "P_z", "u_r", "u_z",
    "E_e", "E_i", "E_rad", "T_e", "T_i", "x_r", "x_z",
    "A_r", "A_z", "dV_swept", "Q_visc", "lambda_FLD", "R_FLD",
    "kappa_eff", "f_Fleck",
]

MESH_MOTION = os.environ.get(
    "TENRYU_I5_2D_RZ_MESH_MOTION",
    "ale" if I5_2D_RZ_ALE else "lagrangian",
).lower()
if MESH_MOTION not in ("ale", "lagrangian"):
    raise ValueError("TENRYU_I5_2D_RZ_MESH_MOTION must be ale or lagrangian")

ale_config = dict(enabled=False)
if I5_2D_RZ_ALE:
    ale_config = dict(
        enabled=True,
        every_n_steps=I5_2D_RZ_ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
        axis_repair_mode=I5_2D_RZ_AXIS_REPAIR_MODE,
        conservative_remap_enabled=I5_2D_RZ_CONSERVATIVE_REMAP,
        conservative_remap_target="reference",
        conservative_remap_radiation_enabled=not I5_2D_RZ_HYDRO_ONLY,
        conservative_remap_order="second_order_van_leer",
        local_boundary_repair_enabled=I5_2D_RZ_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=I5_2D_RZ_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=I5_2D_RZ_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=I5_2D_RZ_EMERGENCY_CELL_DEACTIVATION,
    )

PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_I5_2D_RZ_PRODUCTION_AUDIT_TIER", "A")
if PRODUCTION_AUDIT_TIER not in ("A", "B"):
    raise ValueError('TENRYU_I5_2D_RZ_PRODUCTION_AUDIT_TIER must be "A" or "B"')
PRODUCTION_AUDIT = dict(
    enabled=True,
    tier=PRODUCTION_AUDIT_TIER,
    gcl=dict(enabled=False),
    positivity=dict(enabled=False),
    audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
)

CELLS_PER_L_EI = L_EI / DZ
CELLS_PER_L_RAD = L_RAD / DZ
CASE_NAME = (
    f"i5_2t_radshock_2d_rz_slab_{REGIME}"
    f"_M{_safe_float_token(MACH)}_nr{NR}_nz{NZ}_seed{SEED}_{INIT_MODE}_{BOUNDARY_MODE}"
)

print(
    "[deck:i5_2t_radshock_2d_rz_slab] "
    f"regime={REGIME} table_regime_tag={TABLE_REGIME_TAG} M0={MACH} "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"reference={REFERENCE_PATH} init_mode={INIT_MODE} "
    f"r_center_cm={R_CENTER_CM} r_min_cm={R_MIN} r_max_cm={R_MAX} "
    f"dr_cm={DR} aspect={RADIAL_CELL_ASPECT} "
    f"curvature_dr_total_over_r_center={RADIAL_WIDTH_OVER_CENTER} "
    f"z_min_cm={Z_MIN} z_max_cm={Z_MAX} "
    f"annular_slab={I5_2D_RZ_ANNULAR_SLAB} boundary={BOUNDARY_MODE} "
    f"shock_z_cm={SHOCK_Z_CM} profile_scale={PROFILE_SCALE} "
    f"domain_span_cm={DOMAIN_SPAN} dz_cm={DZ} "
    f"tau_ei_s={TAU_EI} tau_rad_s={TAU_RAD} l_ei_cm={L_EI} l_rad_cm={L_RAD} "
    f"cells_per_l_ei={CELLS_PER_L_EI} cells_per_l_rad={CELLS_PER_L_RAD} "
    f"t_cross_3L_u0_s={HYDRO_CROSSING_T_END} relax_5tau_rad_s={RELAXATION_T_END} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_dt_cap_s={CFL_DT_CAP} max_steps={MAX_STEPS} "
    f"rho0_gcc={PARAMS['rho0_g_per_cc']} T0_eV={PARAMS['T0_eV']} "
    f"kappa_R_cm2_g={KAPPA_R} gamma={GAMMA} A_amu={A_AMU} zbar={ZBAR} "
    f"qei_multiplier={QEI_MULTIPLIER} qei=table_parameter "
    f"hydro=True grey_fld=True flux_limiter=none conduction={I5_2D_RZ_CONDUCTION_ENABLED} "
    f"laser=False dimension=2D_RZ linear_solver_2d={LINEAR_SOLVER_2D} "
    f"av_C1={I5_2D_RZ_AV_C1} av_C2={I5_2D_RZ_AV_C2} cfl_hydro={I5_2D_RZ_CFL_HYDRO} "
    f"ale_enabled={I5_2D_RZ_ALE} ale_every_n={I5_2D_RZ_ALE_EVERY_N_STEPS} "
    f"axis_repair={I5_2D_RZ_AXIS_REPAIR_MODE} mesh_motion={MESH_MOTION} "
    f"production_audit_tier={PRODUCTION_AUDIT_TIER} diagnostics_enabled=True "
    f"save_namelist_copy=True save_frozen_config=True"
)
if I5_2D_RZ_HYDRO_ONLY:
    print(
        "[deck:i5_2t_radshock_2d_rz_slab:hydro_only] "
        "enabled=True radiation_enabled=False radiation_field=zero "
        "radiation_thermal_subcycle=False conservative_remap_radiation_enabled=False"
    )
if TOTAL_ENERGY_REMAP:
    print("[deck:i5_2t_radshock_2d_rz_slab:total_energy_remap] enabled=True")
if WORK_SPLIT_AUDIT:
    print("[deck:i5_2t_radshock_2d_rz_slab:work_split_audit] enabled=True")
if HLLC_Z_FLUX:
    print(
        "[deck:i5_2t_radshock_2d_rz_slab:hllc_z_flux] "
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
        return float(_two_state_selector(z)["T_e_eV"])
    return _interp(X_TABLE, TE_TABLE, _reference_coord(z))


def Ti_init(r, z):
    del r
    if INIT_MODE == "two_state":
        return float(_two_state_selector(z)["T_i_eV"])
    return _interp(X_TABLE, TI_TABLE, _reference_coord(z))


def velocity_init(r, z):
    del r
    if INIT_MODE == "two_state":
        return (0.0, float(_two_state_selector(z)["u"]))
    return (0.0, _interp(X_TABLE, U_TABLE, _reference_coord(z)))


def volfrac_h(r, z):
    del r, z
    return 1.0


if BOUNDARY_MODE == "state_supply":
    HYDRO_BOUNDARY = dict(
        r_inner=R_INNER_BC,
        r_outer="reflect",
        z_bottom=_state_supply("upstream", UPSTREAM),
        z_top=_state_supply("downstream", DOWNSTREAM),
        mesh_tangential_target="reference",
        state_supply_donor_mode=os.environ.get(
            "TENRYU_I5_2D_RZ_STATE_SUPPLY_DONOR_MODE", "interior_radial_average"
        ),
    )
    FLD_BOUNDARY = dict(
        inner_r="reflect",
        outer_r="reflect",
        z="state_supply",
        z_bottom="state_supply",
        z_top="state_supply",
    )
else:
    HYDRO_BOUNDARY = dict(
        r_inner=R_INNER_BC,
        r_outer="reflect",
        z_bottom="reflect",
        z_top="reflect",
    )
    FLD_BOUNDARY = dict(inner_r="reflect", outer_r="reflect", z="reflect")

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get(
        "TENRYU_I5_2D_RZ_VERBOSITY",
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
    radiation_field="zero" if I5_2D_RZ_HYDRO_ONLY else "equilibrium",
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=not I5_2D_RZ_HYDRO_ONLY,
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=I5_2D_RZ_CFL_HYDRO,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=HYDRO_BOUNDARY,
        av_C1=I5_2D_RZ_AV_C1,
        av_C2=I5_2D_RZ_AV_C2,
        qei_multiplier=QEI_MULTIPLIER,
        total_energy_remap_2d_rz=TOTAL_ENERGY_REMAP,
        work_split_audit_2d_rz=WORK_SPLIT_AUDIT,
        hllc_z_flux_2d_rz=HLLC_Z_FLUX,
        hllc_z_flux_audit_2d_rz=HLLC_Z_AUDIT,
        hllc_z_flux_strict_quasi_1d=HLLC_Z_STRICT,
        work_split_audit_cell_every_n_steps=_env_int(
            "TENRYU_I5_2D_RZ_WORK_SPLIT_AUDIT_CELL_EVERY_N_STEPS", 0
        ),
        work_split_audit_all_rows=_env_bool(
            "TENRYU_I5_2D_RZ_WORK_SPLIT_AUDIT_ALL_ROWS", False
        ),
        rz_geometric_cfl_enabled=True,
        rz_geometric_cfl_etaV=0.5,
        rz_geometric_cfl_r_floor=1.0e-10,
        rz_geometric_cfl_cumulative_protection_enabled=True,
        rz_geometric_cfl_v_initial_floor=0.1,
        geometric_retry_stagnation=dict(force_diagnostic_dump=VERBOSE_DIAG),
    ),
    conduction=dict(
        enabled=I5_2D_RZ_CONDUCTION_ENABLED,
        solver=os.environ.get("TENRYU_I5_2D_RZ_COND_SOLVER", "sts"),
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

Radiation(
    enabled=not I5_2D_RZ_HYDRO_ONLY,
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
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "rad_E", "zbar", "volfrac"],
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
