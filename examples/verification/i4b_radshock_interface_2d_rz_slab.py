"""2D RZ z-slab LE08 grey radiative-shock material-interface deck (I4b).

The deck follows the I5 z-slab scaffold: env-parameterized, annular slab,
reflecting boundaries, two-state shock launch, production audit enabled by
default, and a frozen-config dump. It uses the committed LE08 nED grey
reference tables at native scale.
"""

import bisect
import json
import os
import sys
from pathlib import Path


def _print_help_and_exit():
    print(
        """I4b grey LE08 radiative shock / material-interface 2D RZ z-slab deck.

Required reference tables are selected by:
  TENRYU_I4B_M0                              default: 2.0

Primary grid/output knobs:
  TENRYU_I4B_LEG                             default: A
  TENRYU_I4B_KAPPA_RATIO                     default: 10.0
  TENRYU_I4B_NZ                              default: 512
  TENRYU_I4B_NR                              default: 8
  TENRYU_I4B_R_CENTER_CM                     default: 50 * NR * DZ
  TENRYU_I4B_R_MIN_CM/R_MAX_CM               default: centered at R_CENTER_CM with dr=dz
  TENRYU_I4B_PRODUCTION_AUDIT_TIER           default: A
  TENRYU_I4B_OUTDIR                          default: ./build/output_verify_i4b_radshock_interface_2d_rz_slab
  TENRYU_I4B_SEED                            default: 12345

Dry parse:
  python3 examples/verification/i4b_radshock_interface_2d_rz_slab.py --help

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
    if "TENRYU_I4B_FLD_CG_INNER_TOL" in os.environ:
        overrides["cg_inner_tol"] = _env_float(
            "TENRYU_I4B_FLD_CG_INNER_TOL", 1.0e-10
        )
    return overrides


M0 = _env_float("TENRYU_I4B_M0", 2.0)
LEG = os.environ.get("TENRYU_I4B_LEG", "A").upper()
if LEG not in ("A", "B"):
    raise ValueError("TENRYU_I4B_LEG must be A or B")
KAPPA_RATIO = _env_float("TENRYU_I4B_KAPPA_RATIO", 10.0)
if not (KAPPA_RATIO > 0.0):
    raise ValueError("TENRYU_I4B_KAPPA_RATIO must be > 0")
NR = _env_int("TENRYU_I4B_NR", 8)
NZ = _env_int("TENRYU_I4B_NZ", 512)
SEED = _env_int("TENRYU_I4B_SEED", 12345)
OUTDIR = os.environ.get(
    "TENRYU_I4B_OUTDIR",
    "./build/output_verify_i4b_radshock_interface_2d_rz_slab",
)
D_INT_FRAC = _env_float("TENRYU_I4B_D_INT_FRAC", 0.5)

default_ref = (
    _repo_root()
    / "tests"
    / "verification"
    / "data"
    / "le08_ned_reference"
    / f"le08_nED_M{_safe_float_token(M0)}.json"
)
REFERENCE_PATH = Path(os.environ.get("TENRYU_I4B_REFERENCE", str(default_ref)))
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

if not (NR > 0 and NZ > 0):
    raise ValueError("2D RZ slab grid requires NR>0 and NZ>0")
L_REF = X_TABLE[-1] - X_TABLE[0]
if not (L_REF > 0.0):
    raise ValueError("LE08 reference table requires positive x_cm span")
Z_MIN = 0.0
Z_MAX = 3.0 * L_REF
DOMAIN_SPAN = Z_MAX - Z_MIN
DZ = DOMAIN_SPAN / float(NZ)
SHOCK_Z_CM = 1.5 * L_REF
UPSTREAM_U = float(UPSTREAM["u"])
if UPSTREAM_U > 0.0:
    ORIENTATION = "upstream_low_z"
    Z_INT_CM = SHOCK_Z_CM - D_INT_FRAC * L_REF
    if not (Z_MIN < Z_INT_CM < SHOCK_Z_CM < Z_MAX):
        raise ValueError("I4b z geometry requires Z_MIN < Z_INT_CM < SHOCK_Z_CM < Z_MAX")
elif UPSTREAM_U < 0.0:
    ORIENTATION = "upstream_high_z"
    Z_INT_CM = SHOCK_Z_CM + D_INT_FRAC * L_REF
    if not (Z_MIN < SHOCK_Z_CM < Z_INT_CM < Z_MAX):
        raise ValueError("I4b mirrored z geometry requires Z_MIN < SHOCK_Z_CM < Z_INT_CM < Z_MAX")
else:
    raise ValueError("I4b deck requires nonzero upstream velocity for orientation selection")

R_CENTER_CM = _env_float("TENRYU_I4B_R_CENTER_CM", 50.0 * float(NR) * DZ)
DEFAULT_R_MIN = R_CENTER_CM - 0.5 * float(NR) * DZ
DEFAULT_R_MAX = R_CENTER_CM + 0.5 * float(NR) * DZ
R_MIN_OVERRIDDEN = "TENRYU_I4B_R_MIN_CM" in os.environ
R_MAX_OVERRIDDEN = "TENRYU_I4B_R_MAX_CM" in os.environ
if R_MIN_OVERRIDDEN != R_MAX_OVERRIDDEN:
    raise ValueError("TENRYU_I4B_R_MIN_CM and TENRYU_I4B_R_MAX_CM must be provided together")
RADIAL_BOUNDS_OVERRIDDEN = R_MIN_OVERRIDDEN and R_MAX_OVERRIDDEN
if RADIAL_BOUNDS_OVERRIDDEN:
    R_MIN = _env_float("TENRYU_I4B_R_MIN_CM", DEFAULT_R_MIN)
    R_MAX = _env_float("TENRYU_I4B_R_MAX_CM", DEFAULT_R_MAX)
else:
    R_MIN = DEFAULT_R_MIN
    R_MAX = DEFAULT_R_MAX
if R_MIN <= 0.0:
    raise ValueError("I4b annular slab requires R_MIN>0; raise TENRYU_I4B_R_CENTER_CM")
if not (R_MAX > R_MIN):
    raise ValueError("2D RZ slab radial bounds require R_MAX>R_MIN")
DR = (R_MAX - R_MIN) / float(NR) if RADIAL_BOUNDS_OVERRIDDEN else DZ
RADIAL_CELL_ASPECT = DR / DZ
RADIAL_WIDTH_OVER_CENTER = float(NR) * DR / R_CENTER_CM
I4B_ANNULAR_SLAB = R_MIN > 0.0
R_INNER_BC = "reflect"


def _compute_tau_rel(rho_gcc, kappa_R_cm2_g, T_eV):
    cv_e_per_gram = ZBAR * EV_TO_ERG / (A_AMU * PROTON_MASS_G * (GAMMA - 1.0))
    sigma_a = kappa_R_cm2_g * rho_gcc
    return rho_gcc * cv_e_per_gram / (
        4.0 * C_LIGHT_CM_PER_S * sigma_a * A_EV_ERG_CM3_EV4 * T_eV**3
    )


TAU_REL = _compute_tau_rel(float(UPSTREAM["rho"]), KAPPA_R, float(PARAMS["T0_eV"]))
TAU_REL_M2 = _compute_tau_rel(
    float(UPSTREAM["rho"]), KAPPA_R / KAPPA_RATIO, float(PARAMS["T0_eV"])
)
U0_ABS = abs(UPSTREAM_U)
CS0 = U0_ABS / M0
I4B_CFL_HYDRO = _env_float("TENRYU_I4B_CFL_HYDRO", 0.1)
T_CROSS_INT = abs(SHOCK_Z_CM - Z_INT_CM) / max(U0_ABS, 1.0e-300)
DEFAULT_T_END = T_CROSS_INT + min(
    3.0 * TAU_REL_M2,
    L_REF / max(U0_ABS, 1.0e-300),
)
CFL_DT_CAP = I4B_CFL_HYDRO * DZ / max(U0_ABS + CS0, 1.0e-300)
DEFAULT_DT_MAX = min(DEFAULT_T_END / 200.0, CFL_DT_CAP)
T_END = _env_float("TENRYU_I4B_T_END_S", DEFAULT_T_END)
DT_MAX = _env_float("TENRYU_I4B_DT_MAX_S", DEFAULT_DT_MAX)
# Cold-start ramp: the IC carries a discontinuous T^4 jump; starting at the full
# CFL dt makes the first radiation solves ill-conditioned (c*dt/dz^2). growth 1.1
# reaches DT_MAX in ~70 steps.
DT_INITIAL = _env_float("TENRYU_I4B_DT_INITIAL_S", min(DT_MAX, 1.0e-12))
MAX_STEPS = _env_int("TENRYU_I4B_MAX_STEPS", 1000000)
PLOT_EVERY_STEPS = 0
PLOT_EVERY_S = _env_float("TENRYU_I4B_PLOT_EVERY_S", T_END / 20.0)
HISTORY_EVERY_STEPS = 0
CONDUCTION_KAPPA = _env_float("TENRYU_I4B_CONDUCTION_KAPPA", 1.0e-20)
# Jacobi-CG stalls at I4b's native scale (77 cm domain, c*dt/dz^2 conditioning);
# the quasi-1D z-slab prefers z-line preconditioning. Override via env if needed
# (cusparse_cg_jacobi | cusparse_cg_zline | cusparse_cg_rgmg | amgx_cg).
LINEAR_SOLVER_2D = os.environ.get("TENRYU_I4B_LINEAR_SOLVER", "cusparse_cg_zline")

# Production-aligned ALE and hydro knobs follow the I5 z-slab deck.
I4B_ALE = _env_bool("TENRYU_I4B_ALE_ENABLED", True)
I4B_ALE_EVERY_N_STEPS = _env_int("TENRYU_I4B_ALE_EVERY_N_STEPS", 1)
I4B_AV_C1 = _env_float("TENRYU_I4B_AV_C1", 0.5)
I4B_AV_C2 = _env_float("TENRYU_I4B_AV_C2", 2.0)
I4B_AXIS_REPAIR_MODE = os.environ.get("TENRYU_I4B_AXIS_REPAIR_MODE", "axis_spine_only")
I4B_LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_I4B_LOCAL_BOUNDARY_REPAIR", False)
I4B_MULTI_NODE_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I4B_MULTI_NODE_BOUNDARY_REPAIR", False
)
I4B_MULTI_NODE_INTERIOR_REPAIR = _env_bool(
    "TENRYU_I4B_MULTI_NODE_INTERIOR_REPAIR", False
)
I4B_EMERGENCY_CELL_DEACTIVATION = _env_bool(
    "TENRYU_I4B_EMERGENCY_CELL_DEACTIVATION", False
)
# The conservative reference-target remap asserts against per-material
# conservation; per-material conserved slabs remap via the PLIC unified pass
# (plic.enabled + rho_material_aware_donor below).
I4B_CONSERVATIVE_REMAP = _env_bool("TENRYU_I4B_CONSERVATIVE_REMAP", False)
# Per-material hydro owns its conserved-slab remap; total_energy_remap_2d_rz is
# single-material-only (parse-enforced mutual exclusion).
TOTAL_ENERGY_REMAP = _env_bool("TENRYU_I4B_TOTAL_ENERGY_REMAP", False)
WORK_SPLIT_AUDIT = _env_bool("TENRYU_I4B_WORK_SPLIT_AUDIT", False)
# HLLC z-flux requires total_energy_remap_2d_rz, which excludes per-material
# hydro (parse-enforced chain) — per-material decks run the legacy z-flux path.
HLLC_Z_FLUX = _env_bool("TENRYU_I4B_HLLC_Z_FLUX", False)
HLLC_Z_AUDIT = _env_bool("TENRYU_I4B_HLLC_Z_AUDIT", False)
HLLC_Z_STRICT = _env_bool("TENRYU_I4B_HLLC_Z_STRICT", False)
VERBOSE_DIAG = _env_bool("TENRYU_I4B_VERBOSE_DIAG", False)
RADIAL_FOURIER_AUDIT = _env_bool("TENRYU_I4B_RADIAL_FOURIER_AUDIT", VERBOSE_DIAG)
RADIAL_FOURIER_T_START = _env_float("TENRYU_I4B_RADIAL_FOURIER_T_START_S", 0.0)
RADIAL_FOURIER_T_END = _env_float("TENRYU_I4B_RADIAL_FOURIER_T_END_S", T_END)
RADIAL_FOURIER_MAX_MODE = _env_int("TENRYU_I4B_RADIAL_FOURIER_MAX_MODE", -1)
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
    "TENRYU_I4B_MESH_MOTION",
    "ale" if I4B_ALE else "lagrangian",
).lower()
if MESH_MOTION not in ("ale", "lagrangian"):
    raise ValueError("TENRYU_I4B_MESH_MOTION must be ale or lagrangian")

ale_config = dict(enabled=False)
if I4B_ALE:
    ale_config = dict(
        enabled=True,
        every_n_steps=I4B_ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
        axis_repair_mode=I4B_AXIS_REPAIR_MODE,
        conservative_remap_enabled=I4B_CONSERVATIVE_REMAP,
        conservative_remap_target="reference",
        conservative_remap_radiation_enabled=True,
        conservative_remap_order="second_order_van_leer",
        local_boundary_repair_enabled=I4B_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=I4B_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=I4B_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=I4B_EMERGENCY_CELL_DEACTIVATION,
    )

PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_I4B_PRODUCTION_AUDIT_TIER", "A")
if PRODUCTION_AUDIT_TIER not in ("A", "B"):
    raise ValueError('TENRYU_I4B_PRODUCTION_AUDIT_TIER must be "A" or "B"')
PRODUCTION_AUDIT = dict(
    enabled=True,
    tier=PRODUCTION_AUDIT_TIER,
    gcl=dict(enabled=False),
    positivity=dict(enabled=False),
    audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
)

CASE_NAME = (
    f"i4b_radshock_interface_2d_rz_slab_leg{LEG}"
    f"_M{_safe_float_token(M0)}_k{_safe_float_token(KAPPA_RATIO)}"
    f"_nr{NR}_nz{NZ}_seed{SEED}"
)

print(
    "[deck:i4b_radshock_interface_2d_rz_slab] "
    f"leg={LEG} M0={M0} kappa_ratio={KAPPA_RATIO} "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"reference={REFERENCE_PATH} orientation={ORIENTATION} "
    f"r_center_cm={R_CENTER_CM} r_min_cm={R_MIN} r_max_cm={R_MAX} "
    f"dr_cm={DR} aspect={RADIAL_CELL_ASPECT} "
    f"curvature_dr_total_over_r_center={RADIAL_WIDTH_OVER_CENTER} "
    f"z_min_cm={Z_MIN} z_max_cm={Z_MAX} shock_z_cm={SHOCK_Z_CM} "
    f"z_int_cm={Z_INT_CM} d_int_frac={D_INT_FRAC} "
    f"L_ref_cm={L_REF} dz_cm={DZ} "
    f"t_cross_int_s={T_CROSS_INT} tau_rel_m1_s={TAU_REL} "
    f"tau_rel_m2_s={TAU_REL_M2} t_end_s={T_END} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} "
    f"cfl_dt_cap_s={CFL_DT_CAP} max_steps={MAX_STEPS} "
    f"u0_cm_per_s={UPSTREAM_U} cs0_cm_per_s={CS0} "
    f"mat1_kappa_cm2_g={KAPPA_R} mat2_kappa_cm2_g={KAPPA_R / KAPPA_RATIO} "
    f"per_material_conservation=True "
    f"rho0_gcc={PARAMS['rho0_g_per_cc']} T0_eV={PARAMS['T0_eV']} "
    f"gamma={GAMMA} A_amu={A_AMU} zbar={ZBAR} "
    f"annular_slab={I4B_ANNULAR_SLAB} boundary=reflecting "
    f"hydro=True grey_fld=True conduction=True dimension=2D_RZ "
    f"linear_solver_2d={LINEAR_SOLVER_2D} "
    f"av_C1={I4B_AV_C1} av_C2={I4B_AV_C2} cfl_hydro={I4B_CFL_HYDRO} "
    f"ale_enabled={I4B_ALE} ale_every_n={I4B_ALE_EVERY_N_STEPS} "
    f"axis_repair={I4B_AXIS_REPAIR_MODE} mesh_motion={MESH_MOTION} "
    f"production_audit_tier={PRODUCTION_AUDIT_TIER} diagnostics_enabled=True "
    f"save_namelist_copy=True save_frozen_config=True"
)
if TOTAL_ENERGY_REMAP:
    print("[deck:i4b_radshock_interface_2d_rz_slab:total_energy_remap] enabled=True")
if WORK_SPLIT_AUDIT:
    print("[deck:i4b_radshock_interface_2d_rz_slab:work_split_audit] enabled=True")
if HLLC_Z_FLUX:
    print(
        "[deck:i4b_radshock_interface_2d_rz_slab:hllc_z_flux] "
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
    x = X_TABLE[0] + (z - Z_MIN) * (L_REF / DOMAIN_SPAN)
    if x <= X_TABLE[0]:
        return X_TABLE[0]
    if x >= X_TABLE[-1]:
        return X_TABLE[-1]
    return x


def _two_state_selector(z):
    if ORIENTATION == "upstream_low_z":
        return UPSTREAM if z < SHOCK_Z_CM else DOWNSTREAM
    return DOWNSTREAM if z < SHOCK_Z_CM else UPSTREAM


def _is_mat2(z):
    if ORIENTATION == "upstream_low_z":
        return z < Z_INT_CM
    return z > Z_INT_CM


def _leg_b_mat2_override(field):
    if field == "rho":
        return 0.5 * float(UPSTREAM["rho"])
    if field == "T":
        return 2.0 * float(UPSTREAM["T"])
    if field == "u":
        return UPSTREAM_U
    raise ValueError("unknown I4b Leg B override field")


def rho_init(r, z):
    del r
    if LEG == "B" and _is_mat2(z):
        return _leg_b_mat2_override("rho")
    return float(_two_state_selector(z)["rho"])


def Te_init(r, z):
    del r
    if LEG == "B" and _is_mat2(z):
        return _leg_b_mat2_override("T")
    return float(_two_state_selector(z)["T"])


def Ti_init(r, z):
    del r
    if LEG == "B" and _is_mat2(z):
        return _leg_b_mat2_override("T")
    return float(_two_state_selector(z)["T"])


def velocity_init(r, z):
    del r
    if LEG == "B" and _is_mat2(z):
        return (0.0, _leg_b_mat2_override("u"))
    return (0.0, float(_two_state_selector(z)["u"]))


def volfrac_mat1(r, z):
    del r
    return 0.0 if _is_mat2(z) else 1.0


def volfrac_mat2(r, z):
    del r
    return 1.0 if _is_mat2(z) else 0.0


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
    verbosity="verbose" if VERBOSE_DIAG else "quiet",
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
            name="MAT1",
            A=A_AMU,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=KAPPA_R, kappa_s=0.0, units="cm2_per_g"),
        ),
        Material(
            name="MAT2",
            A=A_AMU,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(
                model="constant",
                kappa_a=KAPPA_R / KAPPA_RATIO,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        ),
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(MAT1=volfrac_mat1, MAT2=volfrac_mat2),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

Numerics(
    radiation_thermal_subcycle=True,
        materials=dict(per_material_conservation_enabled=True),
        plic=dict(enabled=True, rho_material_aware_donor=True),
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=I4B_CFL_HYDRO,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=HYDRO_BOUNDARY,
        av_C1=I4B_AV_C1,
        av_C2=I4B_AV_C2,
        total_energy_remap_2d_rz=TOTAL_ENERGY_REMAP,
        work_split_audit_2d_rz=WORK_SPLIT_AUDIT,
        hllc_z_flux_2d_rz=HLLC_Z_FLUX,
        hllc_z_flux_audit_2d_rz=HLLC_Z_AUDIT,
        hllc_z_flux_strict_quasi_1d=HLLC_Z_STRICT,
        work_split_audit_cell_every_n_steps=_env_int(
            "TENRYU_I4B_WORK_SPLIT_AUDIT_CELL_EVERY_N_STEPS", 0
        ),
        work_split_audit_all_rows=_env_bool(
            "TENRYU_I4B_WORK_SPLIT_AUDIT_ALL_ROWS", False
        ),
        rz_geometric_cfl_enabled=True,
        rz_geometric_cfl_etaV=0.5,
        rz_geometric_cfl_r_floor=1.0e-10,
        rz_geometric_cfl_cumulative_protection_enabled=True,
        rz_geometric_cfl_v_initial_floor=0.1,
        geometric_retry_stagnation=dict(force_diagnostic_dump=VERBOSE_DIAG),
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
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
        cg_max_iter=_env_int("TENRYU_I4B_FLD_CG_MAX_ITER", 4096),
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
