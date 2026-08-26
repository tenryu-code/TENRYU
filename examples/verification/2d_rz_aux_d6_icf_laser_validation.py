import math
import os

from tenryu_namelist import *


def _env_bool(name, default="1"):
    return os.environ.get(name, default) == "1"


um = 1.0e-4
ns = 1.0e-9

NR = int(os.environ.get("TENRYU_D6_NR", "128"))
NZ = int(os.environ.get("TENRYU_D6_NZ", "256"))
SEED = int(os.environ.get("TENRYU_D6_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_D6_OUTDIR", "./build/output_verify_2d_rz_aux_d6_icf_laser")
D6ICF_LINEAR_SOLVER = os.environ.get(
    "TENRYU_D6ICF_LINEAR_SOLVER_2D", "auto"
)
T_END = float(os.environ.get("TENRYU_D6_T_END_S", str(6.0 * ns)))
MAX_STEPS = int(os.environ.get("TENRYU_D6_MAX_STEPS", "200000"))
CFL = float(os.environ.get("TENRYU_D6_CFL", "0.2"))
DT_INITIAL = float(os.environ.get("TENRYU_D6_DT_INITIAL_S", "2.0e-13"))
DT_MAX = float(os.environ.get("TENRYU_D6_DT_MAX_S", "2.0e-12"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_D6_ALE_EVERY_N_STEPS", "5"))
MAX_ALE_ITERATIONS = int(os.environ.get("TENRYU_D6_ALE_MAX_ITERATIONS", "100"))
AXIS_MOTION_FLOOR_FRACTION = float(
    os.environ.get("TENRYU_D6_AXIS_MOTION_FLOOR_FRACTION", "0.0")
)
AXIS_MARGIN_DT_FLOOR_FRACTION = float(
    os.environ.get("TENRYU_D6_AXIS_MARGIN_DT_FLOOR_FRACTION", "0.0")
)

AXIS_REPAIR_MODE = os.environ.get("TENRYU_D6_AXIS_REPAIR_MODE", "axis_spine_only")
PREVENTIVE_AXIS_GUARD_FRACTION = float(
    os.environ.get("TENRYU_D6_PREVENTIVE_AXIS_GUARD_FRACTION", "0.1")
)
REMAP_DAMAGE_GATE = _env_bool("TENRYU_D6_REMAP_DAMAGE_GATE")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_D6_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_D6_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_D6_REMAP_DAMAGE_AXIS_BUDGET")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_D6_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_D6_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_D6_REMAP_MS2_LIMITER", "van_leer")
SWEPT_VOLUME_SIGN_FIXED = _env_bool(
    "TENRYU_D6_SWEPT_VOLUME_SIGN_FIXED",
    os.environ.get("TENRYU_D6_DONOR_SIGN_FIXED", "1"),
)
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_D6_KE_CLOSURE")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_D6_KE_CLOSURE_AUDIT", "0")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_D6_KE_CLOSURE_REDISTRIBUTE_FLOOR")
PHASE_RESOLVED_ENERGY = _env_bool("TENRYU_D6_PHASE_RESOLVED_ENERGY")
D6_RETRY_ACTIVE_REPAIR = _env_bool("TENRYU_D6_RETRY_ACTIVE_REPAIR", "0")
D6_CORNER_BALANCE_THRESHOLD = float(os.environ.get("TENRYU_D6_CORNER_BALANCE_THRESHOLD", "0.01"))
D6_LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_D6_LOCAL_BOUNDARY_REPAIR", "0")
D6_MULTI_NODE_BOUNDARY_REPAIR = _env_bool("TENRYU_D6_MULTI_NODE_BOUNDARY_REPAIR", "0")
D6_EMERGENCY_CELL_DEACTIVATION = _env_bool("TENRYU_D6_EMERGENCY_CELL_DEACTIVATION", "0")
D6_RETRY = _env_bool("TENRYU_D6_RETRY", "0")
D6_RETRY_MAX_ATTEMPTS = int(os.environ.get("TENRYU_D6_RETRY_MAX_ATTEMPTS", "8"))
D6_VOLUME_RATE_CFL = _env_bool("TENRYU_D6_VOLUME_RATE_CFL", "0")
D6_TRIAL_VOLUME_CFL = _env_bool("TENRYU_D6_TRIAL_VOLUME_CFL", "0")
D6_CORNER_JACOBIAN_ALE_TRIGGER = _env_bool("TENRYU_D6_CORNER_JACOBIAN_ALE_TRIGGER", "0")
ICF_PROFILE_ENABLED = _env_bool("TENRYU_D6_ICF_PROFILE_ENABLED", "0")
ICF_PROFILE_ENFORCE = _env_bool("TENRYU_D6_ICF_PROFILE_ENFORCE", "0")
# Mesh-stability flags (empirically validated)
PHASE2_MESH_GEOMETRY_SOFT_FAIL = _env_bool("TENRYU_D6_MESH_GEOMETRY_SOFT_FAIL", "0")
PHASE2_IN_HYDRO_CORNER_J_GUARD = _env_bool("TENRYU_D6_IN_HYDRO_CORNER_J_GUARD", "0")
PHASE2_REGIME_AWARE_CORNER_J_GUARD = _env_bool("TENRYU_D6_REGIME_AWARE_CORNER_J_GUARD", "0")
PHASE2_AXIS_MARGIN_GUARD = _env_bool("TENRYU_D6_AXIS_MARGIN_GUARD", "0")
PHASE2_AXIS_GUARD_BAND_CELLS = int(os.environ.get("TENRYU_D6_AXIS_GUARD_BAND_CELLS", "2"))
PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT = _env_bool("TENRYU_D6_DRIVER_RETRY_USE_SUGGESTED_DT", "0")
PHASE2_MULTI_NODE_INTERIOR_REPAIR = _env_bool("TENRYU_D6_MULTI_NODE_INTERIOR_REPAIR", "0")
PHASE2_MESH_ATTRIBUTION_ENABLED = _env_bool("TENRYU_D6_MESH_ATTRIBUTION_ENABLED", "0")
PHASE2_MESH_ATTRIBUTION_RECORD_DISP = _env_bool("TENRYU_D6_MESH_ATTRIBUTION_RECORD_DISP", "0")
PHASE2_MESH_ATTRIBUTION_DUMP_FAIL_ONLY = _env_bool("TENRYU_D6_MESH_ATTRIBUTION_DUMP_FAIL_ONLY", "1")
PHASE2_MESH_ATTRIBUTION_LEAVE_ONE_OUT = _env_bool("TENRYU_D6_MESH_ATTRIBUTION_LEAVE_ONE_OUT", "0")

R_MIN, R_MAX = 0.0, float(os.environ.get("TENRYU_D6_R_MAX_CM", str(500.0 * um)))
Z_MIN = float(os.environ.get("TENRYU_D6_Z_MIN_CM", str(-1000.0 * um)))
Z_MAX = float(os.environ.get("TENRYU_D6_Z_MAX_CM", str(1000.0 * um)))

R_FUEL = float(os.environ.get("TENRYU_D6_R_FUEL_CM", str(120.0 * um)))
R_ABLATOR_OUT = float(os.environ.get("TENRYU_D6_R_ABLATOR_OUT_CM", str(220.0 * um)))
RHO_FUEL = float(os.environ.get("TENRYU_D6_RHO_FUEL_GCC", "2.0e-2"))
RHO_ABLATOR = float(os.environ.get("TENRYU_D6_RHO_ABLATOR_GCC", "2.0"))
RHO_CORONA = float(os.environ.get("TENRYU_D6_RHO_CORONA_GCC", "1.0e-4"))
TE_FUEL_EV = float(os.environ.get("TENRYU_D6_TE_FUEL_EV", "20.0"))
TE_ABLATOR_EV = float(os.environ.get("TENRYU_D6_TE_ABLATOR_EV", "5.0"))
TE_CORONA_EV = float(os.environ.get("TENRYU_D6_TE_CORONA_EV", "20.0"))
TI_FUEL_EV = float(os.environ.get("TENRYU_D6_TI_FUEL_EV", str(TE_FUEL_EV)))
TI_ABLATOR_EV = float(os.environ.get("TENRYU_D6_TI_ABLATOR_EV", str(TE_ABLATOR_EV)))
TI_CORONA_EV = float(os.environ.get("TENRYU_D6_TI_CORONA_EV", str(TE_CORONA_EV)))

P_PEAK_W = float(os.environ.get("TENRYU_D6_LASER_PEAK_W", "2.0e13"))
PULSE_RISE = float(os.environ.get("TENRYU_D6_PULSE_RISE_S", str(1.0 * ns)))
PULSE_PLATEAU = float(os.environ.get("TENRYU_D6_PULSE_PLATEAU_S", str(4.0 * ns)))
PULSE_FALL = float(os.environ.get("TENRYU_D6_PULSE_FALL_S", str(1.0 * ns)))
SPOT_W0_UM = float(os.environ.get("TENRYU_D6_LASER_W0_UM", "200.0"))
RAYS_PER_BEAM = int(os.environ.get("TENRYU_D6_RAYS_PER_BEAM", "4096"))
LASER_TEST_KAPPA = float(os.environ.get("TENRYU_D6_LASER_TEST_KAPPA", "10.0"))

GROUP_BOUNDS_EV = [0.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0]


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p")


CASE_NAME = (
    f"2d_rz_aux_d6_icf_laser_validation_nr{NR}_nz{NZ}_seed{SEED}"
    f"_cfl{_safe_float_token(CFL)}_ale{ALE_EVERY_N_STEPS}_p9_11_t31"
)

print(
    "[deck:2d_rz_aux_d6_icf_laser_validation] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} cfl={CFL} "
    f"r_fuel_cm={R_FUEL} r_ablator_out_cm={R_ABLATOR_OUT} "
    f"rho_fuel_gcc={RHO_FUEL} rho_ablator_gcc={RHO_ABLATOR} "
    f"rho_corona_gcc={RHO_CORONA} p_peak_W={P_PEAK_W} "
    f"test_kappa={LASER_TEST_KAPPA} rays_per_beam={RAYS_PER_BEAM} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_motion_floor_fraction={AXIS_MOTION_FLOOR_FRACTION} "
    f"axis_margin_dt_floor_fraction={AXIS_MARGIN_DT_FLOOR_FRACTION} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"preventive_axis_guard_fraction={PREVENTIVE_AXIS_GUARD_FRACTION} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} ke_closure={KE_CONSERVATION_CLOSURE} "
    f"retry_active_repair={D6_RETRY_ACTIVE_REPAIR} corner_balance_threshold={D6_CORNER_BALANCE_THRESHOLD} "
    f"retry={D6_RETRY} retry_max={D6_RETRY_MAX_ATTEMPTS} "
    f"volume_rate_cfl={D6_VOLUME_RATE_CFL} trial_volume_cfl={D6_TRIAL_VOLUME_CFL} "
    f"corner_jacobian_ale_trigger={D6_CORNER_JACOBIAN_ALE_TRIGGER} "
    f"icf_profile_enabled={ICF_PROFILE_ENABLED} icf_profile_enforce={ICF_PROFILE_ENFORCE} "
    f"local_boundary_repair={D6_LOCAL_BOUNDARY_REPAIR} multi_node_boundary_repair={D6_MULTI_NODE_BOUNDARY_REPAIR} "
    f"emergency_cell_deactivation={D6_EMERGENCY_CELL_DEACTIVATION}"
)
print(
    "[deck:2d_rz_aux_d6_icf_laser_validation] phase2_mesh_flags "
    f"soft_fail={PHASE2_MESH_GEOMETRY_SOFT_FAIL} "
    f"in_hydro_guard={PHASE2_IN_HYDRO_CORNER_J_GUARD} "
    f"regime_aware={PHASE2_REGIME_AWARE_CORNER_J_GUARD} "
    f"axis_margin={PHASE2_AXIS_MARGIN_GUARD} "
    f"axis_band={PHASE2_AXIS_GUARD_BAND_CELLS} "
    f"suggested_dt={PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT} "
    f"multi_node_interior={PHASE2_MULTI_NODE_INTERIOR_REPAIR} "
    f"attribution={PHASE2_MESH_ATTRIBUTION_ENABLED}"
)


def radius(r, z):
    return math.hypot(r, z)


def in_fuel(r, z):
    return radius(r, z) <= R_FUEL


def in_ablator(r, z):
    rr = radius(r, z)
    return R_FUEL < rr <= R_ABLATOR_OUT


def vf_dt(r, z):
    return 1.0 if in_fuel(r, z) else 0.0


def vf_cd(r, z):
    return 1.0 if in_ablator(r, z) else 0.0


def vf_corona(r, z):
    return 0.0 if radius(r, z) <= R_ABLATOR_OUT else 1.0


def rho_profile(r, z):
    if in_fuel(r, z):
        return RHO_FUEL
    if in_ablator(r, z):
        return RHO_ABLATOR
    return RHO_CORONA


def Te_profile(r, z):
    if in_fuel(r, z):
        return TE_FUEL_EV
    if in_ablator(r, z):
        return TE_ABLATOR_EV
    return TE_CORONA_EV


def Ti_profile(r, z):
    if in_fuel(r, z):
        return TI_FUEL_EV
    if in_ablator(r, z):
        return TI_ABLATOR_EV
    return TI_CORONA_EV


def velocity_init(r, z):
    return (0.0, 0.0)


def laser_power(t_s):
    pulse_end = PULSE_RISE + PULSE_PLATEAU + PULSE_FALL
    if t_s < 0.0 or t_s > pulse_end:
        return 0.0
    if t_s < PULSE_RISE:
        x = t_s / max(PULSE_RISE, 1.0e-300)
        return P_PEAK_W * math.sin(0.5 * math.pi * x) ** 2
    if t_s <= PULSE_RISE + PULSE_PLATEAU:
        return P_PEAK_W
    x = (pulse_end - t_s) / max(PULSE_FALL, 1.0e-300)
    return P_PEAK_W * math.sin(0.5 * math.pi * x) ** 2


mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
)

mat_dt = Material(
    name="DT",
    A=2.5,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=25.0, kappa_s=0.0, units="cm2_per_g"),
)

mat_corona = Material(
    name="H_CORONA",
    A=1.0,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
)

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_D6_VERBOSITY", "quiet"),
)

Mesh(
    r_min=R_MIN,
    r_max=R_MAX,
    z_min=Z_MIN,
    z_max=Z_MAX,
    nr=NR,
    nz=NZ,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[mat_cd, mat_dt, mat_corona],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed"),
)

Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=velocity_init,
    volfrac=dict(CD=vf_cd, DT=vf_dt, H_CORONA=vf_corona),
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=len(GROUP_BOUNDS_EV) - 1,
    group_bounds_eV=GROUP_BOUNDS_EV,
    compute_T_range_eV=[0.1, 3.0e4],
    group_repack_hard_xray=True,
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=10,
        outer_tol=1.0e-5,
        linear_solver_2d=D6ICF_LINEAR_SOLVER,
        boundary=dict(inner_r="reflect", outer_r="vacuum", z="vacuum"),
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"),
)

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=RAYS_PER_BEAM,
    ray_output_count=0,
    ray_output_trajectory=False,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(nr=128, nz=256, r_max_factor=1.0, z_span_factor=1.0, critical_margin=0.9999),
    raytrace=dict(
        test_kappa=LASER_TEST_KAPPA,
        cfl_ray=0.8,
        eps_crit=1.0e-4,
        intensity_cutoff=1.0e-6,
        max_steps=100000,
    ),
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="axial_top",
            direction=(0.0, 0.0, -1.0),
            focus=(0.0, 0.0, 0.0),
            f_number=4.0,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=4),
            power=laser_power,
        )
    ],
)

Numerics(
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL,
        cfl_cond=0.25,
        f_min_fleck=0.01,
        growth_factor=1.05,
        max_s=DT_MAX,
        min_s=1.0e-22,
    ),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
        axis_motion_floor_fraction=AXIS_MOTION_FLOOR_FRACTION,
        axis_margin_dt_floor_fraction=AXIS_MARGIN_DT_FLOOR_FRACTION,
        driver_retry_active_mesh_repair_enabled=D6_RETRY_ACTIVE_REPAIR,
        driver_retry_corner_balance_threshold=D6_CORNER_BALANCE_THRESHOLD,
        driver_full_step_retry_enabled=D6_RETRY,
        driver_full_step_retry_max_attempts=D6_RETRY_MAX_ATTEMPTS,
        in_hydro_corner_j_guard_enabled=PHASE2_IN_HYDRO_CORNER_J_GUARD,
        regime_aware_corner_j_guard_enabled=PHASE2_REGIME_AWARE_CORNER_J_GUARD,
        axis_margin_guard_enabled=PHASE2_AXIS_MARGIN_GUARD,
        axis_guard_band_cells=PHASE2_AXIS_GUARD_BAND_CELLS,
        driver_retry_use_suggested_dt_enabled=PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT,
        mesh_geometry_soft_fail_enabled=PHASE2_MESH_GEOMETRY_SOFT_FAIL,
        volume_rate_cfl_enabled=D6_VOLUME_RATE_CFL,
        trial_volume_cfl_enabled=D6_TRIAL_VOLUME_CFL,
        corner_jacobian_ale_trigger_enabled=D6_CORNER_JACOBIAN_ALE_TRIGGER,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        max_iterations=MAX_ALE_ITERATIONS,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        preventive_axis_guard_fraction=PREVENTIVE_AXIS_GUARD_FRACTION,
        axis_z_motion="fixed",
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_dmax=REMAP_DAMAGE_DMAX,
        remap_damage_axis_eta=REMAP_DAMAGE_AXIS_ETA,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_axis_budget_factor=REMAP_DAMAGE_AXIS_BUDGET_FACTOR,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        swept_volume_sign_fixed=SWEPT_VOLUME_SIGN_FIXED,
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_conservation_closure_audit=KE_CONSERVATION_CLOSURE_AUDIT,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
        local_boundary_repair_enabled=D6_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=D6_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=PHASE2_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=D6_EMERGENCY_CELL_DEACTIVATION,
    ),
    diagnostics=dict(
        phase_resolved_energy=PHASE_RESOLVED_ENERGY,
        mesh_attribution=dict(
            enabled=PHASE2_MESH_ATTRIBUTION_ENABLED,
            record_node_displacements=PHASE2_MESH_ATTRIBUTION_RECORD_DISP,
            dump_on_failure_only=PHASE2_MESH_ATTRIBUTION_DUMP_FAIL_ONLY,
            enable_leave_one_out_replay=PHASE2_MESH_ATTRIBUTION_LEAVE_ONE_OUT,
        ),
    ),
    profile=dict(
        icf_standard_ale=dict(
            enabled=ICF_PROFILE_ENABLED,
            enforce=ICF_PROFILE_ENFORCE,
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
                allow_nonstandard_mesh_rescue=False,
                require_deck_reason=True,
                mark_run_nonstandard=True,
            ),
        )
    ),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    positivity=dict(clamp=True),
    safety=dict(energy_threshold=1.0, nan_fatal=True),
    radiation_thermal_subcycle=True,
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=10,
    checkpoint_every=0,
    plot_every_s=1.0e-9,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, rho_threshold=0.1, modes=[0, 2, 4]),
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=True),
)
