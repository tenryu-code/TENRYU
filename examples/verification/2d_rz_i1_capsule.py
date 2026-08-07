"""Stage 27 pre-PLIC characterization deck, with Stage 30 Wave E CF modes.

This manual-use I1 capsule deck represents a simplified direct-drive,
cryogenic-target-style 2D_RZ baseline for public Stage 27 characterization.
It is intended to generate early IFAR/CR and Legendre mode trajectories before
high-convergence PLIC comparisons. It is not production-comparable.
Stage 30 Wave E can enable PLIC and CF variants through TENRYU_I1CAPSULE_PLIC_*
environment variables; defaults preserve the Stage 27 characterization mode.
Round 4 ALE fixes can be activated via TENRYU_I1CAPSULE_REZONE_SOLVER,
TENRYU_I1CAPSULE_REJECT_ZERO_GAUSS_J, TENRYU_I1CAPSULE_STRATEGY_FIRST_RETRY,
TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_GUARD,
TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_GUARD, and
TENRYU_I1CAPSULE_HOURGLASS_ENABLED env vars; defaults preserve current behavior
bit-exactly.

The initial shell perturbation follows the form
R(theta) = R0 * (1 + sum_l a_l P_l(cos(theta))) from McKenty et al.,
Phys. Plasmas 8, 2315 (2001), Eq. (1). The target/drive context also follows
the direct-drive cryogenic implosion discussion in Radha et al.,
Phys. Plasmas 12, 056307 (2005). The laser pulse here is a simplified flat-top
25 kJ, 351 nm, 10 ns pulse, not a DRACO-shaped direct-drive pulse.
"""

import math
import os
import warnings

from tenryu_namelist import *


um = 1.0e-4
ns = 1.0e-9


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


MODE = os.environ.get("TENRYU_I1CAPSULE_MODE", "public_baseline").lower().replace("-", "_")
VALID_MODES = ("public_baseline", "extended_ale")
if MODE not in VALID_MODES:
    raise ValueError(
        "TENRYU_I1CAPSULE_MODE must be one of: " + ", ".join(VALID_MODES)
    )

NR = _env_int("TENRYU_I1CAPSULE_NR", 128)
NZ = _env_int("TENRYU_I1CAPSULE_NZ", 256)
SEED = _env_int("TENRYU_I1CAPSULE_SEED", 12345)
OUTDIR = os.environ.get(
    "TENRYU_I1CAPSULE_OUTDIR",
    f"./build/output_verify_2d_rz_i1_capsule_{MODE}",
)
RAYS_PER_BEAM = _env_int("TENRYU_I1CAPSULE_RAYS_PER_BEAM", 256)
MAX_STEPS = _env_int("TENRYU_I1CAPSULE_MAX_STEPS", 100000)
I1B_GEO_MODE = _env_bool("TENRYU_I1CAPSULE_I1B_GEO_MODE", False)
IMPLOSION_V_PK_CMS = _env_float("TENRYU_I1CAPSULE_IMPLOSION_V_PK_CMS", 0.0)
GEO_C_STAR = _env_float("TENRYU_I1CAPSULE_GEO_C_STAR", 8.0)

R_MIN = 0.0
R_MAX = _env_float("TENRYU_I1CAPSULE_R_MAX_CM", 900.0 * um)
Z_MIN = _env_float("TENRYU_I1CAPSULE_Z_MIN_CM", -900.0 * um)
Z_MAX = _env_float("TENRYU_I1CAPSULE_Z_MAX_CM", 900.0 * um)
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("TENRYU_I1CAPSULE geometry bounds must satisfy r_max>r_min and z_max>z_min")

R_DT_INNER = _env_float("TENRYU_I1CAPSULE_R_DT_INNER_CM", 500.0 * um)
R_ABLATOR_INNER = _env_float("TENRYU_I1CAPSULE_R_ABLATOR_INNER_CM", 650.0 * um)
R_OUTER = _env_float("TENRYU_I1CAPSULE_R_OUTER_CM", 700.0 * um)
R_CR_TARGET = _env_float("TENRYU_I1CAPSULE_R_CR_TARGET_CM", 50.0 * um)
if not (0.0 < R_DT_INNER < R_ABLATOR_INNER < R_OUTER < min(R_MAX, max(abs(Z_MIN), abs(Z_MAX)))):
    raise ValueError(
        "TENRYU_I1CAPSULE radii must satisfy 0<R_DT_INNER<R_ABLATOR_INNER<R_OUTER inside the domain"
    )

LEG_L_MIN = _env_int("TENRYU_I1CAPSULE_LEG_L_MIN", 2)
LEG_L_MAX = _env_int("TENRYU_I1CAPSULE_LEG_L_MAX", 10)
LEG_AMPLITUDE = _env_float("TENRYU_I1CAPSULE_LEG_AMPLITUDE", 1.0e-3)
if LEG_L_MIN < 0 or LEG_L_MAX < LEG_L_MIN:
    raise ValueError("TENRYU_I1CAPSULE_LEG_L_MIN/MAX must satisfy 0 <= L_MIN <= L_MAX")
if abs(LEG_AMPLITUDE) * max(LEG_L_MAX - LEG_L_MIN + 1, 1) >= 0.5:
    raise ValueError("TENRYU_I1CAPSULE_LEG_AMPLITUDE is too large for this baseline deck")
LEG_MODES = list(range(LEG_L_MIN, LEG_L_MAX + 1))

LASER_INPUT_ERG = _env_float("TENRYU_I1CAPSULE_LASER_INPUT_ERG", 25.0e3 * 1.0e7)
PULSE_DURATION = _env_float("TENRYU_I1CAPSULE_PULSE_DURATION_S", 10.0 * ns)
N_BEAMS = 2
P_PEAK_TOTAL_W = _env_float(
    "TENRYU_I1CAPSULE_LASER_PEAK_W",
    LASER_INPUT_ERG / 1.0e7 / max(PULSE_DURATION, 1.0e-300),
)
P_PEAK_PER_BEAM_W = P_PEAK_TOTAL_W / float(N_BEAMS)
SPOT_W0_UM = _env_float("TENRYU_I1CAPSULE_LASER_W0_UM", 850.0)
F_NUMBER = _env_float("TENRYU_I1CAPSULE_F_NUMBER", 4.0)
FOCUS_Z = _env_float("TENRYU_I1CAPSULE_FOCUS_Z_CM", 0.0)
LASERMESH_R_MAX_FACTOR = _env_float("TENRYU_I1CAPSULE_LASERMESH_R_MAX_FACTOR", 1.05)
LASERMESH_Z_SPAN_FACTOR = _env_float("TENRYU_I1CAPSULE_LASERMESH_Z_SPAN_FACTOR", 1.05)

T_END = _env_float("TENRYU_I1CAPSULE_T_END_S", 2.0 * ns)
if I1B_GEO_MODE:
    assert IMPLOSION_V_PK_CMS > 0.0, (
        "TENRYU_I1CAPSULE_IMPLOSION_V_PK_CMS must be > 0 when "
        "TENRYU_I1CAPSULE_I1B_GEO_MODE=true"
    )
    assert GEO_C_STAR > 1.0, "TENRYU_I1CAPSULE_GEO_C_STAR must be > 1 in I1B-GEO mode"
    if "TENRYU_I1CAPSULE_T_END_S" not in os.environ:
        T_END = (R_OUTER / IMPLOSION_V_PK_CMS) * (1.0 - 1.0 / GEO_C_STAR)
DT_INITIAL = _env_float("TENRYU_I1CAPSULE_DT_INITIAL_S", 2.0e-13)
DT_MAX = _env_float("TENRYU_I1CAPSULE_DT_MAX_S", 5.0e-13)
DT_GROWTH_FACTOR = _env_float("TENRYU_I1CAPSULE_DT_GROWTH_FACTOR", 1.02)
PLOT_EVERY_STEPS = _env_int("TENRYU_I1CAPSULE_PLOT_EVERY_STEPS", 50)
HISTORY_EVERY_STEPS = _env_int("TENRYU_I1CAPSULE_HISTORY_EVERY_STEPS", 10)

CFL_HYDRO = _env_float("TENRYU_I1CAPSULE_CFL_HYDRO", 0.15)
CFL_COND = _env_float("TENRYU_I1CAPSULE_CFL_COND", 0.10)
F_LIM = _env_float("TENRYU_I1CAPSULE_F_LIM", 0.06)
CONDUCTION_SOLVER = os.environ.get("TENRYU_I1CAPSULE_COND_SOLVER", "sts")
I1CAPSULE_LINEAR_SOLVER = os.environ.get(
    "TENRYU_I1CAPSULE_LINEAR_SOLVER", "auto"
)
KAPPA_A = _env_float("TENRYU_I1CAPSULE_KAPPA_A_CM2_G", 25.0)

GAMMA = _env_float("TENRYU_I1CAPSULE_GAMMA", 5.0 / 3.0)
RHO_CD = _env_float("TENRYU_I1CAPSULE_RHO_CD_GCC", 1.05)
RHO_DT_ICE = _env_float("TENRYU_I1CAPSULE_RHO_DT_ICE_GCC", 0.25)
RHO_DT_GAS = _env_float("TENRYU_I1CAPSULE_RHO_DT_GAS_GCC", 1.0e-4)
RHO_CORONA = _env_float("TENRYU_I1CAPSULE_RHO_CORONA_GCC", 1.0e-5)
TE_CD_EV = _env_float("TENRYU_I1CAPSULE_TE_CD_EV", 1.0)
TE_DT_ICE_EV = _env_float("TENRYU_I1CAPSULE_TE_DT_ICE_EV", 1.0)
TE_DT_GAS_EV = _env_float("TENRYU_I1CAPSULE_TE_DT_GAS_EV", 1.0)
TE_CORONA_EV = _env_float("TENRYU_I1CAPSULE_TE_CORONA_EV", 1.0)
TI_CD_EV = _env_float("TENRYU_I1CAPSULE_TI_CD_EV", TE_CD_EV)
TI_DT_ICE_EV = _env_float("TENRYU_I1CAPSULE_TI_DT_ICE_EV", TE_DT_ICE_EV)
TI_DT_GAS_EV = _env_float("TENRYU_I1CAPSULE_TI_DT_GAS_EV", TE_DT_GAS_EV)
TI_CORONA_EV = _env_float("TENRYU_I1CAPSULE_TI_CORONA_EV", TE_CORONA_EV)
RHO_FLOOR = _env_float("TENRYU_I1CAPSULE_RHO_FLOOR_GCC", 1.0e-12)
TE_FLOOR = _env_float("TENRYU_I1CAPSULE_TE_FLOOR_EV", 0.1)
TI_FLOOR = _env_float("TENRYU_I1CAPSULE_TI_FLOOR_EV", 0.1)

PROFILE_ENABLED = _env_bool("TENRYU_I1CAPSULE_PROFILE_ENABLED", True)
PROFILE_ENFORCE = _env_bool("TENRYU_I1CAPSULE_PROFILE_ENFORCE", True)
ESCAPE_ALLOW_NONSTANDARD_MESH_RESCUE = _env_bool("TENRYU_I1CAPSULE_ESCAPE_ALLOW_NONSTANDARD", False)
ALE_ENABLED = _env_bool("TENRYU_I1CAPSULE_ALE", True)
ALE_EVERY_N_STEPS = _env_int("TENRYU_I1CAPSULE_ALE_EVERY_N_STEPS", 5)
AXIS_REPAIR_MODE = os.environ.get("TENRYU_I1CAPSULE_AXIS_REPAIR_MODE", "full_winslow")
REMAP_SCHEME = os.environ.get("TENRYU_I1CAPSULE_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_I1CAPSULE_REMAP_MS2_LIMITER", "van_leer")
DRIVER_RETRY = _env_bool("TENRYU_I1CAPSULE_RETRY", True)
DRIVER_RETRY_MAX_ATTEMPTS = _env_int("TENRYU_I1CAPSULE_RETRY_MAX_ATTEMPTS", 4)

# Round 4 ALE fixes (default-off opt-in flags)
REZONE_SOLVER = os.environ.get("TENRYU_I1CAPSULE_REZONE_SOLVER", "legacy_winslow")
REZONE_LOCAL_LINESEARCH = _env_bool("TENRYU_I1CAPSULE_REZONE_LOCAL_LINESEARCH", False)
REZONE_LOCAL_J_FLOOR_REL = _env_float("TENRYU_I1CAPSULE_REZONE_LOCAL_J_FLOOR_REL", 1e-8)
REZONE_LOCAL_LINESEARCH_MAX_HALVES = _env_int("TENRYU_I1CAPSULE_REZONE_LOCAL_LINESEARCH_MAX_HALVES", 8)
REJECT_ZERO_GAUSS_J = _env_bool("TENRYU_I1CAPSULE_REJECT_ZERO_GAUSS_J", False)
ZERO_GAUSS_J_FLOOR_REL = _env_float("TENRYU_I1CAPSULE_ZERO_GAUSS_J_FLOOR_REL", 1e-8)
STRATEGY_FIRST_RETRY = _env_bool("TENRYU_I1CAPSULE_STRATEGY_FIRST_RETRY", False)
STRATEGY_FIRST_MAX_SAME_DT_ATTEMPTS = _env_int("TENRYU_I1CAPSULE_STRATEGY_FIRST_MAX_SAME_DT_ATTEMPTS", 2)
IN_HYDRO_GAUSS_J_GUARD = _env_bool("TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_GUARD", False)
IN_HYDRO_RZ_VOLUME_GUARD = _env_bool("TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_GUARD", False)
IN_HYDRO_GAUSS_J_FLOOR_REL = _env_float("TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_FLOOR_REL", 1e-8)
IN_HYDRO_RZ_VOLUME_FLOOR_REL = _env_float("TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_FLOOR_REL", 1e-8)
# PR 1+4 mesh-quality dt CFL (HYDRA/DRACO-compliant root-cause fix, pre-commit admissibility envelope)
MESH_QUALITY_DT_CFL_ENABLED = _env_bool("TENRYU_I1CAPSULE_MESH_QUALITY_DT_CFL_ENABLED", False)
MESH_QUALITY_DT_SAFETY_ALPHA = _env_float("TENRYU_I1CAPSULE_MESH_QUALITY_DT_SAFETY_ALPHA", 0.5)
MESH_DEGENERACY_FORENSICS_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_MESH_DEGENERACY_FORENSICS_ENABLED", False
)
GEOMETRIC_RETRY_STAGNATION_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_GEOMETRIC_RETRY_STAGNATION_ENABLED", False
)
CORNER_J_SOURCE_BUDGET_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_CORNER_J_SOURCE_BUDGET_ENABLED", False
)
VELOCITY_HISTORY_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_VELOCITY_HISTORY_ENABLED", False
)
VELOCITY_HISTORY_TARGET_CELL_C = _env_int(
    "TENRYU_I1CAPSULE_VELOCITY_HISTORY_TARGET_CELL_C", 16178
)
VELOCITY_HISTORY_SAMPLE_EVERY_N = _env_int(
    "TENRYU_I1CAPSULE_VELOCITY_HISTORY_SAMPLE_EVERY_N", 5
)
AXIS_MARGIN_ADDITIVE_IN_ACTION8 = _env_bool("TENRYU_I1CAPSULE_AXIS_MARGIN_ADDITIVE_IN_ACTION8", False)
HOURGLASS_ENABLED = _env_bool("TENRYU_I1CAPSULE_HOURGLASS_ENABLED", False)
HOURGLASS_SCALE = _env_float("TENRYU_I1CAPSULE_HOURGLASS_SCALE", 0.05)
HOURGLASS_MAX_FORCE_FRAC = _env_float("TENRYU_I1CAPSULE_HOURGLASS_MAX_FORCE_FRAC", 0.2)
HOURGLASS_ACTIVATION_CJ_THRESHOLD = _env_float("TENRYU_I1CAPSULE_HOURGLASS_ACTIVATION_CJ_THRESHOLD", 0.5)
HOURGLASS_ACTIVATION_AMPLITUDE_THRESHOLD = _env_float("TENRYU_I1CAPSULE_HOURGLASS_ACTIVATION_AMPLITUDE_THRESHOLD", 0.01)
if REZONE_SOLVER not in ("legacy_winslow", "rz_full_metric_winslow"):
    raise ValueError(
        "TENRYU_I1CAPSULE_REZONE_SOLVER must be in "
        f"(legacy_winslow, rz_full_metric_winslow), got: {REZONE_SOLVER}"
    )

PLIC_ENABLED = _env_bool("TENRYU_I1CAPSULE_PLIC_ENABLED", False)
PLIC_NORMAL_ESTIMATOR = os.environ.get(
    "TENRYU_I1CAPSULE_PLIC_NORMAL_ESTIMATOR", "youngs_seeded_LVIRA")
PLIC_T0_METHOD = os.environ.get(
    "TENRYU_I1CAPSULE_PLIC_T0_METHOD", "adaptive_subdivision_2x2")
PLIC_IN_RUN_DISABLED = _env_bool("TENRYU_I1CAPSULE_PLIC_IN_RUN_DISABLED", False)
PLIC_PER_CELL_STATE = os.environ.get(
    "TENRYU_I1CAPSULE_PLIC_PER_CELL_STATE", "off")
PLIC_RHO_MATERIAL_AWARE_DONOR = _env_bool(
    "TENRYU_I1CAPSULE_PLIC_RHO_MATERIAL_AWARE_DONOR", False)
PLIC_T0_MAX_DEPTH = _env_int("TENRYU_I1CAPSULE_PLIC_T0_MAX_DEPTH", 6)
PER_MATERIAL_CONSERVATION_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_PER_MATERIAL_CONSERVATION_ENABLED", False)
HDF5_EMIT_DERIVED_PER_MATERIAL = _env_bool(
    "TENRYU_I1CAPSULE_HDF5_EMIT_DERIVED_PER_MATERIAL", False)
if PLIC_T0_MAX_DEPTH < 4 or PLIC_T0_MAX_DEPTH > 16:
    raise ValueError(
        f"TENRYU_I1CAPSULE_PLIC_T0_MAX_DEPTH={PLIC_T0_MAX_DEPTH} out of range [4, 16]"
    )
if PLIC_T0_METHOD == "centroid_only_legacy" and PLIC_IN_RUN_DISABLED:
    warnings.warn(
        "TENRYU_I1CAPSULE_PLIC_T0_METHOD=centroid_only_legacy with "
        "TENRYU_I1CAPSULE_PLIC_IN_RUN_DISABLED=true disables both Hybrid_AB "
        "components; this run has no PLIC effect.",
        RuntimeWarning,
    )
if PLIC_RHO_MATERIAL_AWARE_DONOR and not PLIC_ENABLED:
    raise ValueError(
        "TENRYU_I1CAPSULE_PLIC_RHO_MATERIAL_AWARE_DONOR requires "
        "TENRYU_I1CAPSULE_PLIC_ENABLED=true"
    )

DISPATCHER_STATE_SENSITIVE_BYPASS = _env_bool(
    "TENRYU_I1CAPSULE_DISPATCHER_STATE_SENSITIVE_BYPASS", False
)
LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_I1CAPSULE_LOCAL_BOUNDARY_REPAIR", False)
MULTI_NODE_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I1CAPSULE_MULTI_NODE_BOUNDARY_REPAIR", False
)
MULTI_NODE_INTERIOR_REPAIR = _env_bool(
    "TENRYU_I1CAPSULE_MULTI_NODE_INTERIOR_REPAIR", False
)
AXIS_VARIATIONAL_PROJECTION = _env_bool(
    "TENRYU_I1CAPSULE_AXIS_VARIATIONAL_PROJECTION", False
)
EMERGENCY_CELL_DEACTIVATION = _env_bool(
    "TENRYU_I1CAPSULE_EMERGENCY_CELL_DEACTIVATION", False
)
DRIVER_RETRY_ACTIVE_REPAIR = _env_bool(
    "TENRYU_I1CAPSULE_RETRY_ACTIVE_REPAIR", False
)
CORNER_BALANCE_THRESHOLD = _env_float("TENRYU_I1CAPSULE_CORNER_BALANCE_THRESHOLD", 0.01)

ICF_DIAGNOSTICS_ENABLED = _env_bool("TENRYU_I1CAPSULE_ICF_DIAGNOSTICS", PROFILE_ENABLED)
CONSERVATION_DIAGNOSTICS_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_CONSERVATION_DIAGNOSTICS", PROFILE_ENABLED
)
ALE_PROVENANCE_ENABLED = _env_bool(
    "TENRYU_I1CAPSULE_ALE_PROVENANCE_DIAGNOSTICS", PROFILE_ENABLED
)
MESH_QUALITY_MIN_DIAG = I1B_GEO_MODE and _env_bool(
    "TENRYU_I1CAPSULE_MESH_QUALITY_MIN_DIAG", True
)

CASE_NAME = (
    f"i1_capsule_{MODE}_nr{NR}_nz{NZ}_rays{RAYS_PER_BEAM}"
    f"_e{_safe_float_token(LASER_INPUT_ERG)}_l{LEG_L_MIN}_{LEG_L_MAX}"
    f"_a{_safe_float_token(LEG_AMPLITUDE)}_seed{SEED}"
)

I1B_GEO_HEADER = ""
if I1B_GEO_MODE:
    I1B_GEO_HEADER = (
        f" i1b_geo_mode=True i1b_geo_v_pk_cms={IMPLOSION_V_PK_CMS}"
        f" i1b_geo_c_star={GEO_C_STAR} i1b_geo_a0={LEG_AMPLITUDE}"
        f" i1b_geo_t_end_s={T_END}"
        f" i1b_geo_operator_semantics=laser_off_fld_radiation_on_conduction_on"
    )

print(
    "[deck:2d_rz_i1_capsule] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"stage27_pre_plic_characterization=True production_gate=False "
    f"r_inner_cm={R_DT_INNER} r_ablator_inner_cm={R_ABLATOR_INNER} r_outer_cm={R_OUTER} "
    f"r_cr_target_cm={R_CR_TARGET} cr_target={R_OUTER / max(R_CR_TARGET, 1.0e-300)} "
    f"leg_l_min={LEG_L_MIN} leg_l_max={LEG_L_MAX} leg_amplitude={LEG_AMPLITUDE} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} "
    f"laser_input_erg={LASER_INPUT_ERG} laser_peak_total_W={P_PEAK_TOTAL_W} "
    f"pulse_duration_s={PULSE_DURATION} rays_per_beam={RAYS_PER_BEAM} "
    f"profile_enabled={PROFILE_ENABLED} profile_enforce={PROFILE_ENFORCE} "
    f"escape_allow_nonstandard={ESCAPE_ALLOW_NONSTANDARD_MESH_RESCUE} "
    f"claim_level=pre_plic_smoke ale={ALE_ENABLED} axis_repair_mode={AXIS_REPAIR_MODE} "
    f"retry={DRIVER_RETRY} retry_active_repair={DRIVER_RETRY_ACTIVE_REPAIR} "
    f"plic_enabled={PLIC_ENABLED} plic_normal_estimator={PLIC_NORMAL_ESTIMATOR} "
    f"plic_t0_method={PLIC_T0_METHOD} plic_in_run_disabled={PLIC_IN_RUN_DISABLED} "
    f"plic_per_cell_state={PLIC_PER_CELL_STATE} "
    f"plic_rho_material_aware_donor={PLIC_RHO_MATERIAL_AWARE_DONOR} "
    f"plic_t0_max_depth={PLIC_T0_MAX_DEPTH}"
    f" per_material_conservation_enabled={PER_MATERIAL_CONSERVATION_ENABLED}"
    f" hdf5_emit_derived_per_material={HDF5_EMIT_DERIVED_PER_MATERIAL}"
    f" hourglass_enabled={HOURGLASS_ENABLED}"
    f"{I1B_GEO_HEADER}"
)


def radius(r, z):
    return math.hypot(r, z)


def legendre_p(ell, x):
    if ell == 0:
        return 1.0
    if ell == 1:
        return x
    p_nm2 = 1.0
    p_nm1 = x
    for n in range(2, ell + 1):
        p_n = ((2 * n - 1) * x * p_nm1 - (n - 1) * p_nm2) / float(n)
        p_nm2 = p_nm1
        p_nm1 = p_n
    return p_nm1


def perturbation_shape(mu):
    return sum(legendre_p(ell, mu) for ell in LEG_MODES)


def geo_perturbation_phi(mu):
    return perturbation_shape(mu) / float(len(LEG_MODES))


def perturbation_factor(r, z):
    rr = radius(r, z)
    mu = 1.0 if rr == 0.0 else z / rr
    total = perturbation_shape(mu)
    if I1B_GEO_MODE:
        total = geo_perturbation_phi(mu)
    factor = 1.0 + LEG_AMPLITUDE * total
    if factor <= 0.0:
        raise ValueError("Legendre perturbation produced a nonpositive shell radius")
    return factor


def shell_radii(r, z):
    factor = perturbation_factor(r, z)
    return R_DT_INNER * factor, R_ABLATOR_INNER * factor, R_OUTER * factor


def region(r, z):
    rr = radius(r, z)
    r_inner, r_ab_inner, r_outer = shell_radii(r, z)
    if rr <= r_inner:
        return "DT_GAS"
    if rr <= r_ab_inner:
        return "DT_ICE"
    if rr <= r_outer:
        return "CD"
    return "H_CORONA"


def vf_dt_gas(r, z):
    return 1.0 if region(r, z) == "DT_GAS" else 0.0


def vf_dt_ice(r, z):
    return 1.0 if region(r, z) == "DT_ICE" else 0.0


def vf_cd(r, z):
    return 1.0 if region(r, z) == "CD" else 0.0


def vf_corona(r, z):
    return 1.0 if region(r, z) == "H_CORONA" else 0.0


def rho_profile(r, z):
    mat = region(r, z)
    if mat == "DT_GAS":
        return RHO_DT_GAS
    if mat == "DT_ICE":
        return RHO_DT_ICE
    if mat == "CD":
        return RHO_CD
    return RHO_CORONA


def Te_profile(r, z):
    mat = region(r, z)
    if mat == "DT_GAS":
        return TE_DT_GAS_EV
    if mat == "DT_ICE":
        return TE_DT_ICE_EV
    if mat == "CD":
        return TE_CD_EV
    return TE_CORONA_EV


def Ti_profile(r, z):
    mat = region(r, z)
    if mat == "DT_GAS":
        return TI_DT_GAS_EV
    if mat == "DT_ICE":
        return TI_DT_ICE_EV
    if mat == "CD":
        return TI_CD_EV
    return TI_CORONA_EV


def velocity_profile(r, z):
    del r, z
    return (0.0, 0.0)


def geo_velocity_profile(r, z):
    s = math.hypot(r, z)
    if s == 0.0:
        return (0.0, 0.0)
    phi = geo_perturbation_phi(z / s)
    denom = 1.0 + LEG_AMPLITUDE * phi
    if denom <= 0.0:
        raise ValueError("I1B-GEO velocity perturbation produced a nonpositive radius factor")
    s_bar = s / denom
    delta_taper = 2.5 * R_MAX / float(max(NR, 1))
    w = 0.5 * (1.0 - math.tanh((s - R_OUTER) / delta_taper))
    speed = -IMPLOSION_V_PK_CMS * (s_bar / R_OUTER) * w
    return (speed * r / s, speed * z / s)


geometry_velocity = velocity_profile
if I1B_GEO_MODE:
    geometry_velocity = geo_velocity_profile


def laser_power_per_beam(t_s):
    if 0.0 <= t_s <= PULSE_DURATION:
        return P_PEAK_PER_BEAM_W
    return 0.0


mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
    opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
)

mat_dt_ice = Material(
    name="DT_ICE",
    A=2.5,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
    opacity=dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g"),
)

mat_dt_gas = Material(
    name="DT_GAS",
    A=2.5,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
    opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
)

mat_corona = Material(
    name="H_CORONA",
    A=1.0,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
    opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
)

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_I1CAPSULE_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale" if ALE_ENABLED else "lagrangian",
)

Materials(
    materials=[mat_cd, mat_dt_ice, mat_dt_gas, mat_corona],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed"),
)

Geometry(
    volfrac=dict(CD=vf_cd, DT_ICE=vf_dt_ice, DT_GAS=vf_dt_gas, H_CORONA=vf_corona),
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=geometry_velocity,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

_mgd_kwargs = dict(
    flux_limiter="levermore_pomraning",
    max_outer_iterations=50,
    outer_tol=1.0e-8,
    linear_solver_2d=I1CAPSULE_LINEAR_SOLVER,
    boundary=dict(inner_r="reflect", outer_r="vacuum", z="vacuum"),
)
if "TENRYU_I1CAPSULE_FLD_CG_INNER_TOL" in os.environ:
    _mgd_kwargs["cg_inner_tol"] = _env_float(
        "TENRYU_I1CAPSULE_FLD_CG_INNER_TOL", 1.0e-10
    )

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=_mgd_kwargs,
    boundary=dict(inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"),
)

Laser(
    enabled=not I1B_GEO_MODE,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=RAYS_PER_BEAM,
    ray_output_count=0,
    ray_output_trajectory=False,
    absorption=dict(
        eps_n=1.0e-4,
        coulomb_log_floor=2.0,
        debug_dump_lasermesh=_env_bool("TENRYU_I1CAPSULE_DEBUG_DUMP_LASERMESH", False),
    ),
    lasermesh=dict(
        nr=128,
        nz=256,
        r_max_factor=LASERMESH_R_MAX_FACTOR,
        z_span_factor=LASERMESH_Z_SPAN_FACTOR,
        critical_margin=0.9999,
    ),
    raytrace=dict(
        cfl_ray=0.8,
        eps_crit=1.0e-4,
        intensity_cutoff=1.0e-8,
        max_steps=100000,
        debug_one_ray=_env_bool("TENRYU_I1CAPSULE_DEBUG_ONE_RAY", False),
    ),
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="axial_top",
            direction=(0.0, 0.0, -1.0),
            focus=(0.0, 0.0, FOCUS_Z),
            f_number=F_NUMBER,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=8),
            power=laser_power_per_beam,
        ),
        LaserBeam(
            name="axial_bottom",
            direction=(0.0, 0.0, 1.0),
            focus=(0.0, 0.0, -FOCUS_Z),
            f_number=F_NUMBER,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=8),
            power=laser_power_per_beam,
        ),
    ],
)

Numerics(
    radiation_thermal_subcycle=True,
    dt=dict(
        initial_s=DT_INITIAL,
        max_s=DT_MAX,
        min_s=1.0e-22,
        growth_factor=DT_GROWTH_FACTOR,
        cfl_hydro=CFL_HYDRO,
        cfl_cond=CFL_COND,
        f_min_fleck=0.01,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
        dispatcher_state_sensitive_bypass_enabled=DISPATCHER_STATE_SENSITIVE_BYPASS,
        driver_full_step_retry_enabled=DRIVER_RETRY,
        driver_full_step_retry_max_attempts=DRIVER_RETRY_MAX_ATTEMPTS,
        driver_retry_active_mesh_repair_enabled=DRIVER_RETRY_ACTIVE_REPAIR,
        driver_retry_corner_balance_threshold=CORNER_BALANCE_THRESHOLD,
        strategy_first_retry_enabled=STRATEGY_FIRST_RETRY,
        strategy_first_max_same_dt_attempts=STRATEGY_FIRST_MAX_SAME_DT_ATTEMPTS,
        geometric_retry_stagnation=dict(
            enabled=GEOMETRIC_RETRY_STAGNATION_ENABLED,
        ),
        in_hydro_gauss_j_guard_enabled=IN_HYDRO_GAUSS_J_GUARD,
        in_hydro_rz_volume_guard_enabled=IN_HYDRO_RZ_VOLUME_GUARD,
        in_hydro_gauss_j_floor_rel=IN_HYDRO_GAUSS_J_FLOOR_REL,
        in_hydro_rz_volume_floor_rel=IN_HYDRO_RZ_VOLUME_FLOOR_REL,
        mesh_quality_dt_cfl_enabled=MESH_QUALITY_DT_CFL_ENABLED,
        mesh_quality_dt_safety_alpha=MESH_QUALITY_DT_SAFETY_ALPHA,
        axis_margin_additive_in_action8_enabled=AXIS_MARGIN_ADDITIVE_IN_ACTION8,
        volume_rate_cfl_enabled=False,
        trial_volume_cfl_enabled=False,
        corner_jacobian_ale_trigger_enabled=False,
        mesh_geometry_soft_fail_enabled=False,
        hourglass=dict(
            enabled=HOURGLASS_ENABLED,
            scale=HOURGLASS_SCALE,
            max_force_per_node_fraction=HOURGLASS_MAX_FORCE_FRAC,
            activation_corner_j_ratio_threshold=HOURGLASS_ACTIVATION_CJ_THRESHOLD,
            activation_hourglass_amplitude_threshold=HOURGLASS_ACTIVATION_AMPLITUDE_THRESHOLD,
        ),
    ),
    conduction=dict(
        enabled=True,
        solver=CONDUCTION_SOLVER,
        f_lim=F_LIM,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    ale=dict(
        enabled=ALE_ENABLED,
        every_n_steps=ALE_EVERY_N_STEPS,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        rezone_solver=REZONE_SOLVER,
        rezone_local_admissibility_linesearch=REZONE_LOCAL_LINESEARCH,
        rezone_local_j_floor_rel=REZONE_LOCAL_J_FLOOR_REL,
        rezone_local_linesearch_max_halves=REZONE_LOCAL_LINESEARCH_MAX_HALVES,
        reject_zero_gauss_j=REJECT_ZERO_GAUSS_J,
        zero_gauss_j_floor_rel=ZERO_GAUSS_J_FLOOR_REL,
        remap_damage_gate_enabled=True,
        remap_damage_dmax=0.05,
        remap_damage_axis_eta=0.02,
        remap_damage_axis_budget_enabled=True,
        remap_damage_axis_budget_factor=2.0,
        local_boundary_repair_enabled=LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=MULTI_NODE_INTERIOR_REPAIR,
        axis_variational_projection_enabled=AXIS_VARIATIONAL_PROJECTION,
        emergency_cell_deactivation_enabled=EMERGENCY_CELL_DEACTIVATION,
        ke_conservation_closure=True,
        ke_conservation_closure_audit=False,
        ke_closure_redistribute_floor=True,
    ),
    plic=dict(
        enabled=PLIC_ENABLED,
        normal_estimator=PLIC_NORMAL_ESTIMATOR,
        t0_volume_cut_method=PLIC_T0_METHOD,
        t0_volume_cut_max_depth=PLIC_T0_MAX_DEPTH,
        in_run_disabled=PLIC_IN_RUN_DISABLED,
        material_interface_per_cell_state=PLIC_PER_CELL_STATE,
        rho_material_aware_donor=PLIC_RHO_MATERIAL_AWARE_DONOR,
    ),
    diagnostics=dict(
        phase_resolved_energy=True,
        icf=dict(enabled=ICF_DIAGNOSTICS_ENABLED),
        conservation=dict(enabled=CONSERVATION_DIAGNOSTICS_ENABLED),
        ale_provenance_emission=dict(enabled=ALE_PROVENANCE_ENABLED),
        mesh_quality_min=dict(enabled=MESH_QUALITY_MIN_DIAG),
        mesh_attribution=dict(
            enabled=False,
            record_node_displacements=False,
            dump_on_failure_only=True,
            enable_leave_one_out_replay=False,
        ),
        mesh_degeneracy_forensics=dict(
            enabled=MESH_DEGENERACY_FORENSICS_ENABLED,
            corner_j_source_budget_enabled=CORNER_J_SOURCE_BUDGET_ENABLED,
            velocity_history_enabled=VELOCITY_HISTORY_ENABLED,
            velocity_history_target_cell_c=VELOCITY_HISTORY_TARGET_CELL_C,
            velocity_history_sample_every_n_steps=VELOCITY_HISTORY_SAMPLE_EVERY_N,
        ),
    ),
    profile=dict(
        icf_standard_ale=dict(
            enabled=PROFILE_ENABLED,
            enforce=PROFILE_ENFORCE,
            claim_level="pre_plic_smoke",
            allowed_when_enabled=dict(
                ale_enabled_required_value=True,
                ale_axis_repair_mode_required_value="full_winslow",
                ale_remap_scheme_allowed_values=["legacy_split", "ms2_moments"],
                hydro_driver_full_step_retry_enabled_required_value=True,
            ),
            forbidden_when_enabled=dict(
                hydro_dispatcher_state_sensitive_bypass_enabled_forbidden_value=True,
                ale_local_boundary_repair_enabled_forbidden_value=True,
                ale_multi_node_boundary_repair_enabled_forbidden_value=True,
                ale_multi_node_interior_repair_enabled_forbidden_value=True,
                ale_axis_variational_projection_enabled_forbidden_value=True,
                ale_emergency_cell_deactivation_enabled_forbidden_value=True,
                hydro_driver_retry_active_mesh_repair_enabled_forbidden_value=True,
            ),
            escape_valves=dict(
                allow_nonstandard_mesh_rescue=ESCAPE_ALLOW_NONSTANDARD_MESH_RESCUE,
                require_deck_reason=True,
                mark_run_nonstandard=True,
            ),
        )
    ),
    diagnostics_every=1,
    materials=dict(
        per_material_conservation_enabled=PER_MATERIAL_CONSERVATION_ENABLED,
        hdf5_emit_derived_per_material=HDF5_EMIT_DERIVED_PER_MATERIAL,
    ),
    floors=dict(rho_floor_gcc=RHO_FLOOR, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
    positivity=dict(clamp=True),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=PLOT_EVERY_STEPS,
    history_every=HISTORY_EVERY_STEPS,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "zbar", "volfrac", "rad_E"],
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0),
    areal_density=dict(enabled=True, angles_deg=[0.0, 30.0, 60.0, 90.0]),
    sphericity=dict(enabled=True, rho_threshold=0.1, modes=LEG_MODES),
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=True),
)
