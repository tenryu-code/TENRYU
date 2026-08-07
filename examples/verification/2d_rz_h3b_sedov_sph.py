import math
import os

from tenryu_namelist import *


NR = int(os.environ.get("TENRYU_H3B_NR", "128"))
NZ = int(os.environ.get("TENRYU_H3B_NZ", "256"))
SEED = int(os.environ.get("TENRYU_H3B_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_H3B_OUTDIR", "./build/output_verify_2d_rz_h3b")
AV_C2 = float(os.environ.get("TENRYU_H3B_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_H3B_CFL", "0.1"))
DT_GROWTH_FACTOR = float(os.environ.get("TENRYU_H3B_DT_GROWTH_FACTOR", "1.1"))
DT_FLOOR_STALL_MAX_CONSECUTIVE_STEPS = int(
    os.environ.get("TENRYU_H3B_DT_FLOOR_STALL_MAX_CONSECUTIVE_STEPS", "32")
)
T_END = float(os.environ.get("TENRYU_H3B_T_END_S", "6.0e-9"))
E_TOTAL = float(os.environ.get("TENRYU_H3B_E_TOTAL", "1.0e13"))
BLAST_CELLS = int(os.environ.get("TENRYU_H3B_BLAST_CELLS", "4"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_H3B_ALE_EVERY_N_STEPS", "0"))
AXIS_Z_MOTION = os.environ.get("TENRYU_H3B_AXIS_Z_MOTION", "fixed")
# Phase 9-11 opt-in env hooks (default = legacy behavior)
AXIS_REPAIR_MODE = os.environ.get("TENRYU_H3B_AXIS_REPAIR_MODE", "full_winslow")
CORNER_JACOBIAN_ALE_TRIGGER = (
    os.environ.get("TENRYU_H3B_CORNER_JACOBIAN_ALE_TRIGGER_ENABLED", "0") == "1"
)
REMAP_DAMAGE_GATE = os.environ.get("TENRYU_H3B_REMAP_DAMAGE_GATE", "0") == "1"
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_H3B_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_H3B_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = os.environ.get("TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET", "0") == "1"
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(os.environ.get("TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0"))
REMAP_SCHEME = os.environ.get("TENRYU_H3B_REMAP_SCHEME", "legacy_split")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_H3B_REMAP_MS2_LIMITER", "van_leer")
AXIS_MOTION_FLOOR_FRACTION = float(os.environ.get("TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION", "0.0"))
PREVENTIVE_AXIS_GUARD_FRACTION = float(os.environ.get("TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION", "0.1"))
PHASE_RESOLVED_ENERGY = os.environ.get("TENRYU_H3B_PHASE_RESOLVED_ENERGY", "0") == "1"
KE_CONSERVATION_CLOSURE = os.environ.get("TENRYU_H3B_KE_CLOSURE", "0") == "1"
KE_CONSERVATION_CLOSURE_AUDIT = os.environ.get("TENRYU_H3B_KE_CLOSURE_AUDIT", "0") == "1"
KE_CLOSURE_REDISTRIBUTE_FLOOR = (
    os.environ.get("TENRYU_H3B_KE_CLOSURE_REDISTRIBUTE_FLOOR", "0") == "1"
)
AXIS_MARGIN_DT_FLOOR_FRACTION = float(os.environ.get("TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION", "0.0"))
H3B_RETRY = os.environ.get("TENRYU_H3B_RETRY", "0") == "1"
H3B_RETRY_MAX_ATTEMPTS = int(os.environ.get("TENRYU_H3B_RETRY_MAX_ATTEMPTS", "3"))
H3B_RETRY_ACTIVE_REPAIR = os.environ.get("TENRYU_H3B_RETRY_ACTIVE_REPAIR", "0") == "1"
H3B_CORNER_BALANCE_THRESHOLD = float(os.environ.get("TENRYU_H3B_CORNER_BALANCE_THRESHOLD", "0.01"))
H3B_LOCAL_BOUNDARY_REPAIR = os.environ.get("TENRYU_H3B_LOCAL_BOUNDARY_REPAIR", "0") == "1"
H3B_MULTI_NODE_BOUNDARY_REPAIR = os.environ.get("TENRYU_H3B_MULTI_NODE_BOUNDARY_REPAIR", "0") == "1"
H3B_EMERGENCY_CELL_DEACTIVATION = os.environ.get("TENRYU_H3B_EMERGENCY_CELL_DEACTIVATION", "0") == "1"
# Stage 19 Phase 2 mesh-stability flags (Stage 20 empirical validation)
PHASE2_MESH_GEOMETRY_SOFT_FAIL = os.environ.get("TENRYU_H3B_MESH_GEOMETRY_SOFT_FAIL", "0") == "1"
PHASE2_IN_HYDRO_CORNER_J_GUARD = os.environ.get("TENRYU_H3B_IN_HYDRO_CORNER_J_GUARD", "0") == "1"
PHASE2_REGIME_AWARE_CORNER_J_GUARD = os.environ.get("TENRYU_H3B_REGIME_AWARE_CORNER_J_GUARD", "0") == "1"
PHASE2_AXIS_MARGIN_GUARD = os.environ.get("TENRYU_H3B_AXIS_MARGIN_GUARD", "0") == "1"
PHASE2_AXIS_GUARD_BAND_CELLS = int(os.environ.get("TENRYU_H3B_AXIS_GUARD_BAND_CELLS", "2"))
PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT = (
    os.environ.get("TENRYU_H3B_DRIVER_RETRY_USE_SUGGESTED_DT", "0") == "1"
)
PHASE2_MULTI_NODE_INTERIOR_REPAIR = os.environ.get("TENRYU_H3B_MULTI_NODE_INTERIOR_REPAIR", "0") == "1"
PHASE2_DISPATCHER_STATE_SENSITIVE_BYPASS = (
    os.environ.get("TENRYU_H3B_DISPATCHER_STATE_SENSITIVE_BYPASS", "0") == "1"
)
PHASE2_DISPATCHER_STATE_SENSITIVE_REPAIR_CAP = int(
    os.environ.get("TENRYU_H3B_DISPATCHER_STATE_SENSITIVE_REPAIR_CAP", "3")
)
AXIS_VARIATIONAL_PROJECTION = os.environ.get("TENRYU_H3B_AXIS_VARIATIONAL_PROJECTION", "0") == "1"
PHASE2_AXIS_BAND_MANAGED_REMAP = (
    os.environ.get("TENRYU_H3B_AXIS_BAND_MANAGED_REMAP", "0") == "1"
)
PHASE2_AXIS_BAND_MANAGED_REMAP_WIDTH = int(
    os.environ.get("TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_WIDTH", "3")
)
PHASE2_AXIS_BAND_MANAGED_REMAP_MAX_WIDTH = int(
    os.environ.get("TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_MAX_WIDTH", "6")
)
PHASE2_AXIS_BAND_MANAGED_REMAP_EVERY_HYDRO_HALF_STEP = (
    os.environ.get(
        "TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_EVERY_HYDRO_HALF_STEP", "1"
    )
    == "1"
)
PHASE2_AXIS_BAND_MANAGED_REMAP_MARGIN_TRIGGER = float(
    os.environ.get(
        "TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_MARGIN_TRIGGER", "1.0e-4"
    )
)
PHASE2_AXIS_BAND_MANAGED_REMAP_EQUAL_VOLUME = (
    os.environ.get("TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_EQUAL_VOLUME", "1")
    == "1"
)
PHASE2_AXIS_BAND_MANAGED_REMAP_INCLUDE_RADIATION_GROUPS = (
    os.environ.get(
        "TENRYU_H3B_AXIS_BAND_MANAGED_REMAP_INCLUDE_RADIATION_GROUPS",
        "1",
    )
    == "1"
)
_MESH_ATTRIBUTION_ENABLED_ENV = os.environ.get("TENRYU_H3B_MESH_ATTRIBUTION_ENABLED", None)
if _MESH_ATTRIBUTION_ENABLED_ENV is None:
    PHASE2_MESH_ATTRIBUTION_ENABLED = AXIS_VARIATIONAL_PROJECTION
    if AXIS_VARIATIONAL_PROJECTION:
        print(
            "[deck] AxisVariationalProjection enabled; auto-enabling mesh_attribution.enabled "
            "for telemetry capture (override with TENRYU_H3B_MESH_ATTRIBUTION_ENABLED=0)"
        )
else:
    PHASE2_MESH_ATTRIBUTION_ENABLED = _MESH_ATTRIBUTION_ENABLED_ENV == "1"
    if AXIS_VARIATIONAL_PROJECTION and not PHASE2_MESH_ATTRIBUTION_ENABLED:
        print(
            "[deck] AxisVariationalProjection enabled but mesh_attribution.enabled=false "
            "explicit; AV local/global telemetry will be silent. Set "
            "TENRYU_H3B_MESH_ATTRIBUTION_ENABLED=1 for full telemetry."
        )
PHASE2_MESH_ATTRIBUTION_RECORD_DISP = os.environ.get("TENRYU_H3B_MESH_ATTRIBUTION_RECORD_DISP", "0") == "1"
PHASE2_MESH_ATTRIBUTION_DUMP_FAIL_ONLY = (
    os.environ.get("TENRYU_H3B_MESH_ATTRIBUTION_DUMP_FAIL_ONLY", "1") == "1"
)
PHASE2_MESH_ATTRIBUTION_LEAVE_ONE_OUT = (
    os.environ.get("TENRYU_H3B_MESH_ATTRIBUTION_LEAVE_ONE_OUT", "0") == "1"
)

if ALE_EVERY_N_STEPS <= 0 and H3B_RETRY_ACTIVE_REPAIR:
    ALE_EVERY_N_STEPS = 5


def _safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


CASE_NAME = (
    f"2d_rz_h3b_sedov_sph_nr{NR}_nz{NZ}_seed{SEED}"
    f"_C2{_safe_float_token(AV_C2)}_cfl{_safe_float_token(CFL)}"
    f"_ale{ALE_EVERY_N_STEPS}"
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO_0_GCC = 1.0

R_MIN, R_MAX = 0.0, 1.2
Z_MIN, Z_MAX = -1.2, 1.2
DR = (R_MAX - R_MIN) / NR
DZ = (Z_MAX - Z_MIN) / NZ
R_BLAST = BLAST_CELLS * DR

P_AMBIENT = 1.0e6
cv_i = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
cv_e = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
T_AMBIENT_eV = P_AMBIENT / ((GAMMA - 1.0) * RHO_0_GCC * (cv_i + cv_e))

V_SPHERE = (4.0 / 3.0) * math.pi * R_BLAST**3


def _selected_blast_volume_cm3() -> tuple[float, int]:
    volume = 0.0
    count = 0
    for i in range(NR):
        r_inner = R_MIN + i * DR
        r_outer = R_MIN + (i + 1) * DR
        r_cell = 0.5 * (r_inner + r_outer)
        for j in range(NZ):
            z_bottom = Z_MIN + j * DZ
            z_top = Z_MIN + (j + 1) * DZ
            z_cell = 0.5 * (z_bottom + z_top)
            if math.hypot(r_cell, z_cell) < R_BLAST:
                volume += math.pi * (r_outer * r_outer - r_inner * r_inner) * (z_top - z_bottom)
                count += 1
    return volume, count


V_PATCH, N_BLAST_CELLS = _selected_blast_volume_cm3()
if V_PATCH <= 0.0:
    raise ValueError("spherical Sedov blast patch selected no cells")

E_INT_BLAST = E_TOTAL / (RHO_0_GCC * V_PATCH)
T_BLAST_eV = E_INT_BLAST / (cv_i + cv_e)
P_BLAST = (GAMMA - 1.0) * RHO_0_GCC * E_INT_BLAST

print(
    "[deck:2d_rz_h3b] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"AV_C2={AV_C2} cfl={CFL} t_end_s={T_END} "
    f"dt_growth_factor={DT_GROWTH_FACTOR} "
    f"dt_floor_stall_max={DT_FLOOR_STALL_MAX_CONSECUTIVE_STEPS} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_z_motion={AXIS_Z_MOTION} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"corner_jacobian_ale_trigger={CORNER_JACOBIAN_ALE_TRIGGER} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} "
    f"retry={H3B_RETRY} retry_active_repair={H3B_RETRY_ACTIVE_REPAIR} "
    f"local_boundary_repair={H3B_LOCAL_BOUNDARY_REPAIR} "
    f"multi_node_boundary_repair={H3B_MULTI_NODE_BOUNDARY_REPAIR} "
    f"emergency_deactivation={H3B_EMERGENCY_CELL_DEACTIVATION} "
    f"remap_scheme={REMAP_SCHEME}"
)
print(
    "[deck:2d_rz_h3b] phase2_mesh_flags "
    f"soft_fail={PHASE2_MESH_GEOMETRY_SOFT_FAIL} "
    f"in_hydro_guard={PHASE2_IN_HYDRO_CORNER_J_GUARD} "
    f"regime_aware={PHASE2_REGIME_AWARE_CORNER_J_GUARD} "
    f"axis_margin={PHASE2_AXIS_MARGIN_GUARD} "
    f"axis_band={PHASE2_AXIS_GUARD_BAND_CELLS} "
    f"suggested_dt={PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT} "
    f"multi_node_interior={PHASE2_MULTI_NODE_INTERIOR_REPAIR} "
    f"axis_variational_projection={AXIS_VARIATIONAL_PROJECTION} "
    f"axis_band_managed_remap={PHASE2_AXIS_BAND_MANAGED_REMAP} "
    f"axis_band_K={PHASE2_AXIS_BAND_MANAGED_REMAP_WIDTH}-{PHASE2_AXIS_BAND_MANAGED_REMAP_MAX_WIDTH} "
    f"dispatcher_state_sensitive_bypass={PHASE2_DISPATCHER_STATE_SENSITIVE_BYPASS} "
    f"dispatcher_state_sensitive_repair_cap={PHASE2_DISPATCHER_STATE_SENSITIVE_REPAIR_CAP} "
    f"attribution={PHASE2_MESH_ATTRIBUTION_ENABLED}"
)
print(
    f"  R_blast={R_BLAST:.4f}cm V_sphere={V_SPHERE:.4e}cm^3 "
    f"V_patch={V_PATCH:.4e}cm^3 n_blast_cells={N_BLAST_CELLS} "
    f"P_blast={P_BLAST:.4e}erg/cm^3"
)
print(f"  T_ambient={T_AMBIENT_eV:.4e}eV T_blast={T_BLAST_eV:.4e}eV")
print(f"  E_total={E_TOTAL:.4e}erg E_int_blast={E_INT_BLAST:.4e}erg/g")


def _inside_blast(r, z):
    return math.hypot(r, z) < R_BLAST


def rho_init(r, z):
    return RHO_0_GCC


def Te_init(r, z):
    return T_BLAST_eV if _inside_blast(r, z) else T_AMBIENT_eV


def Ti_init(r, z):
    return T_BLAST_eV if _inside_blast(r, z) else T_AMBIENT_eV


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=20000,
    seed=SEED,
    verbosity="quiet",
    temperature_model="2T",
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
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

numerics_kwargs = {
    "dt": dict(
        initial_s=1.0e-13,
        cfl_hydro=CFL,
        growth_factor=DT_GROWTH_FACTOR,
        max_s=1.0e-10,
        min_s=1.0e-22,
        floor_stall_max_consecutive_steps=DT_FLOOR_STALL_MAX_CONSECUTIVE_STEPS,
    ),
    "hydro": dict(
        boundary_2d=dict(
            r_inner="axis",
            r_outer="free",
            z_bottom="free",
            z_top="free",
        ),
        av_C1=0.1,
        av_C2=AV_C2,
        axis_motion_floor_fraction=AXIS_MOTION_FLOOR_FRACTION,
        axis_margin_dt_floor_fraction=AXIS_MARGIN_DT_FLOOR_FRACTION,
        driver_full_step_retry_enabled=H3B_RETRY,
        driver_full_step_retry_max_attempts=H3B_RETRY_MAX_ATTEMPTS,
        driver_retry_active_mesh_repair_enabled=H3B_RETRY_ACTIVE_REPAIR,
        driver_retry_corner_balance_threshold=H3B_CORNER_BALANCE_THRESHOLD,
        in_hydro_corner_j_guard_enabled=PHASE2_IN_HYDRO_CORNER_J_GUARD,
        regime_aware_corner_j_guard_enabled=PHASE2_REGIME_AWARE_CORNER_J_GUARD,
        axis_margin_guard_enabled=PHASE2_AXIS_MARGIN_GUARD,
        axis_guard_band_cells=PHASE2_AXIS_GUARD_BAND_CELLS,
        driver_retry_use_suggested_dt_enabled=PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT,
        dispatcher_state_sensitive_bypass_enabled=PHASE2_DISPATCHER_STATE_SENSITIVE_BYPASS,
        dispatcher_state_sensitive_repair_cap_per_step=PHASE2_DISPATCHER_STATE_SENSITIVE_REPAIR_CAP,
        mesh_geometry_soft_fail_enabled=PHASE2_MESH_GEOMETRY_SOFT_FAIL,
    ),
    "conduction": dict(enabled=False),
    "floors": dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
    "diagnostics": dict(
        phase_resolved_energy=PHASE_RESOLVED_ENERGY,
        mesh_attribution=dict(
            enabled=PHASE2_MESH_ATTRIBUTION_ENABLED,
            record_node_displacements=PHASE2_MESH_ATTRIBUTION_RECORD_DISP,
            dump_on_failure_only=PHASE2_MESH_ATTRIBUTION_DUMP_FAIL_ONLY,
            enable_leave_one_out_replay=PHASE2_MESH_ATTRIBUTION_LEAVE_ONE_OUT,
        ),
    ),
}

if PHASE2_AXIS_BAND_MANAGED_REMAP or ALE_EVERY_N_STEPS > 0:
    ale_kwargs = dict()
    if ALE_EVERY_N_STEPS > 0:
        ale_kwargs.update(
            enabled=True,
            every_n_steps=ALE_EVERY_N_STEPS,
            max_iterations=100,
            preventive_axis_guard_fraction=PREVENTIVE_AXIS_GUARD_FRACTION,
            axis_z_motion=AXIS_Z_MOTION,
            axis_repair_mode=AXIS_REPAIR_MODE,
            remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
            remap_damage_dmax=REMAP_DAMAGE_DMAX,
            remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
            remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
            remap_damage_axis_budget_factor=REMAP_DAMAGE_AXIS_BUDGET_FACTOR,
            local_boundary_repair_enabled=H3B_LOCAL_BOUNDARY_REPAIR,
            multi_node_boundary_repair_enabled=H3B_MULTI_NODE_BOUNDARY_REPAIR,
            multi_node_interior_repair_enabled=PHASE2_MULTI_NODE_INTERIOR_REPAIR,
            axis_variational_projection_enabled=AXIS_VARIATIONAL_PROJECTION,
            emergency_cell_deactivation_enabled=H3B_EMERGENCY_CELL_DEACTIVATION,
            remap_scheme=REMAP_SCHEME,
            remap_ms2_limiter=REMAP_MS2_LIMITER,
            ke_conservation_closure=KE_CONSERVATION_CLOSURE,
            ke_conservation_closure_audit=KE_CONSERVATION_CLOSURE_AUDIT,
            ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
        )
    if PHASE2_AXIS_BAND_MANAGED_REMAP:
        ale_kwargs.update(
            axis_band_managed_remap_enabled=True,
            axis_band_managed_remap_width=PHASE2_AXIS_BAND_MANAGED_REMAP_WIDTH,
            axis_band_managed_remap_max_width=PHASE2_AXIS_BAND_MANAGED_REMAP_MAX_WIDTH,
            axis_band_managed_remap_every_hydro_half_step=PHASE2_AXIS_BAND_MANAGED_REMAP_EVERY_HYDRO_HALF_STEP,
            axis_band_managed_remap_margin_trigger=PHASE2_AXIS_BAND_MANAGED_REMAP_MARGIN_TRIGGER,
            axis_band_managed_remap_equal_volume=PHASE2_AXIS_BAND_MANAGED_REMAP_EQUAL_VOLUME,
            axis_band_managed_remap_include_radiation_groups=PHASE2_AXIS_BAND_MANAGED_REMAP_INCLUDE_RADIATION_GROUPS,
        )
    numerics_kwargs["ale"] = ale_kwargs

if CORNER_JACOBIAN_ALE_TRIGGER:
    numerics_kwargs["hydro"].update(
        corner_jacobian_ale_trigger_enabled=True,
        corner_jacobian_ale_trigger_scale=0.5,
    )

Numerics(**numerics_kwargs)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=T_END / 4.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
