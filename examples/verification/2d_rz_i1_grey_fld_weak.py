import math
import os

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


MODE = os.environ.get("TENRYU_I1_MODE", "i1a_grey_fld_1pct").lower().replace("-", "_")
VALID_MODES = (
    "i1a_grey_fld_1pct",
    "i1a_ale_on_probe",
    "i1_tmat_opacity_coupling",
    "i1b_grey_fld_5pct_powerstress",
    "i1c_grey_fld_10pct_powerstress",
    "i1d_grey_fld_25pct_powerstress",
)
if MODE not in VALID_MODES:
    raise ValueError("TENRYU_I1_MODE must be one of: " + ", ".join(VALID_MODES))

NR = _env_int("TENRYU_I1_NR", 128)
NZ = _env_int("TENRYU_I1_NZ", 256)
SEED = _env_int("TENRYU_I1_SEED", 12345)
OUTDIR = os.environ.get("TENRYU_I1_OUTDIR", "./build/output_verify_2d_rz_i1_grey_fld_weak")
I1GREYWEAK_LINEAR_SOLVER = os.environ.get(
    "TENRYU_I1GREYWEAK_LINEAR_SOLVER_2D", "auto"
)
RAYS_PER_BEAM = _env_int("TENRYU_I1_RAYS_PER_BEAM", 72)
MAX_STEPS = _env_int("TENRYU_I1_MAX_STEPS", 1000000)

R_MIN = 0.0
R_MAX = _env_float("TENRYU_I1_R_MAX_CM", 200.0 * um)
Z_MIN = _env_float("TENRYU_I1_Z_MIN_CM", -200.0 * um)
Z_MAX = _env_float("TENRYU_I1_Z_MAX_CM", 200.0 * um)
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("TENRYU_I1 geometry bounds must satisfy r_max>r_min and z_max>z_min")

GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
LASER_INPUT_ERG = _env_float("TENRYU_I1_LASER_INPUT_ERG", GXII_REFERENCE_E_ERG * 0.01)
PULSE_DURATION = _env_float("TENRYU_I1_PULSE_DURATION_S", 1.0 * ns)
P_PEAK_W = _env_float(
    "TENRYU_I1_LASER_PEAK_W", LASER_INPUT_ERG / 1.0e7 / max(PULSE_DURATION, 1.0e-300)
)
T_END_DEFAULT = 5.0e-11
if MODE == "i1_tmat_opacity_coupling":
    # Capability probe: short t_end (~10 steps) demonstrates TMAT opacity
    # coupling in FLD before the unrelated mesh-inversion issue at step ~20.
    T_END_DEFAULT = 2.0e-12
elif MODE in (
    "i1b_grey_fld_5pct_powerstress",
    "i1c_grey_fld_10pct_powerstress",
    "i1d_grey_fld_25pct_powerstress",
):
    # Ladder rows stay at 5e-11 s for Wave 1; first-shock convergence is Wave 2.
    T_END_DEFAULT = 5.0e-11
T_END = _env_float("TENRYU_I1_T_END_S", T_END_DEFAULT)
DT_INITIAL = _env_float("TENRYU_I1_DT_INITIAL_S", 2.0e-13)
DT_MAX = _env_float("TENRYU_I1_DT_MAX_S", 2.0e-13)
DT_GROWTH_FACTOR = _env_float("TENRYU_I1_DT_GROWTH_FACTOR", 1.0)
PLOT_EVERY_STEPS = _env_int("TENRYU_I1_PLOT_EVERY_STEPS", 50)

CFL_HYDRO = _env_float("TENRYU_I1_CFL_HYDRO", 0.15)
CFL_COND = _env_float("TENRYU_I1_CFL_COND", 0.1)
F_LIM = _env_float("TENRYU_I1_F_LIM", 0.06)
CONDUCTION_SOLVER = os.environ.get("TENRYU_I1_COND_SOLVER", "sts")
TEST_KAPPA = _env_float("TENRYU_I1_TEST_KAPPA", 10.0)

SPHERE_RADIUS = _env_float("TENRYU_I1_SPHERE_RADIUS_CM", 100.0 * um)
SPOT_W0_UM = _env_float("TENRYU_I1_LASER_W0_UM", 200.0)
F_NUMBER = _env_float("TENRYU_I1_F_NUMBER", 4.0)
FOCUS_Z = _env_float("TENRYU_I1_FOCUS_Z_CM", 0.0)
LASERMESH_R_MAX_FACTOR = _env_float("TENRYU_I1_LASERMESH_R_MAX_FACTOR", 1.1)
LASERMESH_Z_SPAN_FACTOR = _env_float("TENRYU_I1_LASERMESH_Z_SPAN_FACTOR", 1.25)

A_CD = 7.0
Z_CD = 3.5
GAMMA = _env_float("TENRYU_I1_GAMMA", 5.0 / 3.0)
KAPPA_A = _env_float("TENRYU_I1_KAPPA_A_CM2_G", 10.0)
TMAT_OPACITY_FILE = "TMAT-H5/CD_1grp.tmat.h5"
RHO_CD = _env_float("TENRYU_I1_RHO_CD_GCC", 1.05)
RHO_VAC = _env_float("TENRYU_I1_RHO_VAC_GCC", 1.0e-2)
TE_CD_EV = _env_float("TENRYU_I1_TE_CD_EV", 1.0)
TI_CD_EV = _env_float("TENRYU_I1_TI_CD_EV", 1.0)
TE_VAC_EV = _env_float("TENRYU_I1_TE_VAC_EV", 1.0)
TI_VAC_EV = _env_float("TENRYU_I1_TI_VAC_EV", 1.0)
RHO_FLOOR = _env_float("TENRYU_I1_RHO_FLOOR_GCC", 1.0e-12)
TE_FLOOR = _env_float("TENRYU_I1_TE_FLOOR_EV", 0.1)
TI_FLOOR = _env_float("TENRYU_I1_TI_FLOOR_EV", 0.1)

I1_ALE = _env_bool("TENRYU_I1_ALE", False)
I1_CORNER_J_ALE_TRIGGER = _env_bool("TENRYU_I1_CORNER_J_ALE_TRIGGER", I1_ALE)
I1_RETRY = _env_bool("TENRYU_I1_RETRY", False)
I1_RETRY_MAX_ATTEMPTS = _env_int("TENRYU_I1_RETRY_MAX_ATTEMPTS", 3)
I1_RETRY_ACTIVE_REPAIR = _env_bool("TENRYU_I1_RETRY_ACTIVE_REPAIR", False)
ICF_PROFILE_ENABLED = _env_bool("TENRYU_I1_ICF_PROFILE_ENABLED", False)
ICF_PROFILE_ENFORCE = _env_bool("TENRYU_I1_ICF_PROFILE_ENFORCE", False)
I1_CORNER_BALANCE_THRESHOLD = _env_float("TENRYU_I1_CORNER_BALANCE_THRESHOLD", 0.01)
I1_LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_I1_LOCAL_BOUNDARY_REPAIR", False)
I1_MULTI_NODE_BOUNDARY_REPAIR = _env_bool("TENRYU_I1_MULTI_NODE_BOUNDARY_REPAIR", False)
I1_EMERGENCY_CELL_DEACTIVATION = _env_bool("TENRYU_I1_EMERGENCY_CELL_DEACTIVATION", False)
I1_CASCADE_ON_HYDRO_RETRY = _env_bool("TENRYU_I1_CASCADE_ON_HYDRO_RETRY", False)
I1_ALE_STRICT_J_FLOOR = _env_bool("TENRYU_I1_ALE_STRICT_J_FLOOR", False)
ALE_ENABLED = MODE == "i1a_ale_on_probe" or I1_ALE or I1_RETRY_ACTIVE_REPAIR
ALE_EVERY_N_STEPS = _env_int("TENRYU_I1_ALE_EVERY_N_STEPS", 5)
AXIS_REPAIR_MODE = os.environ.get("TENRYU_I1_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_SCHEME = os.environ.get("TENRYU_I1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_I1_REMAP_MS2_LIMITER", "van_leer")
# Stage 19 Phase 2 mesh-stability flags (Stage 20 empirical validation)
PHASE2_MESH_GEOMETRY_SOFT_FAIL = _env_bool("TENRYU_I1_MESH_GEOMETRY_SOFT_FAIL", False)
PHASE2_IN_HYDRO_CORNER_J_GUARD = _env_bool("TENRYU_I1_IN_HYDRO_CORNER_J_GUARD", False)
PHASE2_REGIME_AWARE_CORNER_J_GUARD = _env_bool("TENRYU_I1_REGIME_AWARE_CORNER_J_GUARD", False)
PHASE2_AXIS_MARGIN_GUARD = _env_bool("TENRYU_I1_AXIS_MARGIN_GUARD", False)
PHASE2_AXIS_GUARD_BAND_CELLS = _env_int("TENRYU_I1_AXIS_GUARD_BAND_CELLS", 2)
PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT = _env_bool("TENRYU_I1_DRIVER_RETRY_USE_SUGGESTED_DT", False)
PHASE2_MULTI_NODE_INTERIOR_REPAIR = _env_bool("TENRYU_I1_MULTI_NODE_INTERIOR_REPAIR", False)
PHASE2_MESH_ATTRIBUTION_ENABLED = _env_bool("TENRYU_I1_MESH_ATTRIBUTION_ENABLED", False)
PHASE2_MESH_ATTRIBUTION_RECORD_DISP = _env_bool("TENRYU_I1_MESH_ATTRIBUTION_RECORD_DISP", False)
PHASE2_MESH_ATTRIBUTION_DUMP_FAIL_ONLY = _env_bool("TENRYU_I1_MESH_ATTRIBUTION_DUMP_FAIL_ONLY", True)
PHASE2_MESH_ATTRIBUTION_LEAVE_ONE_OUT = _env_bool("TENRYU_I1_MESH_ATTRIBUTION_LEAVE_ONE_OUT", False)

CASE_NAME = (
    f"i1_{MODE}_nr{NR}_nz{NZ}_rays{RAYS_PER_BEAM}"
    f"_e{_safe_float_token(LASER_INPUT_ERG)}_ka{_safe_float_token(KAPPA_A)}_seed{SEED}"
)

if MODE == "i1_tmat_opacity_coupling":
    opacity_block = dict(model="tmat", file=TMAT_OPACITY_FILE)
else:
    opacity_block = dict(model="constant", kappa_a=KAPPA_A, kappa_s=0.0, units="cm2_per_g")

print(
    "[deck:2d_rz_i1_grey_fld_weak] "
    f"mode={MODE} nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"r_max_cm={R_MAX} z_min_cm={Z_MIN} z_max_cm={Z_MAX} "
    f"t_end_s={T_END} dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} dt_growth_factor={DT_GROWTH_FACTOR} "
    f"laser_input_erg={LASER_INPUT_ERG} laser_peak_W={P_PEAK_W} "
    f"pulse_duration_s={PULSE_DURATION} rays_per_beam={RAYS_PER_BEAM} "
    f"kappa_a_cm2_g={KAPPA_A} cfl_hydro={CFL_HYDRO} cfl_cond={CFL_COND} "
    f"test_kappa={TEST_KAPPA} hydro=True ale={ALE_ENABLED} i1_ale={I1_ALE} "
    f"i1_retry={I1_RETRY} retry_max={I1_RETRY_MAX_ATTEMPTS} "
    f"ale_strict_j_floor={I1_ALE_STRICT_J_FLOOR} "
    f"icf_profile_enabled={ICF_PROFILE_ENABLED} icf_profile_enforce={ICF_PROFILE_ENFORCE} "
    f"conduction=True qei=True grey_fld=True eos=ideal_gas opacity_model={opacity_block['model']} "
    f"tmat_opacity_capability_probe={MODE == 'i1_tmat_opacity_coupling'} "
    f"power_stress_ladder={'True' if MODE in ('i1b_grey_fld_5pct_powerstress','i1c_grey_fld_10pct_powerstress','i1d_grey_fld_25pct_powerstress') else 'False'} "
)
print(
    "[deck:2d_rz_i1_grey_fld_weak] phase2_mesh_flags "
    f"soft_fail={PHASE2_MESH_GEOMETRY_SOFT_FAIL} "
    f"in_hydro_guard={PHASE2_IN_HYDRO_CORNER_J_GUARD} "
    f"regime_aware={PHASE2_REGIME_AWARE_CORNER_J_GUARD} "
    f"axis_margin={PHASE2_AXIS_MARGIN_GUARD} "
    f"axis_band={PHASE2_AXIS_GUARD_BAND_CELLS} "
    f"suggested_dt={PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT} "
    f"multi_node_interior={PHASE2_MULTI_NODE_INTERIOR_REPAIR} "
    f"attribution={PHASE2_MESH_ATTRIBUTION_ENABLED}"
)


def _radius(r, z):
    return math.hypot(r, z)


def _in_cd(r, z):
    return _radius(r, z) <= SPHERE_RADIUS


def rho_profile(r, z):
    return RHO_CD if _in_cd(r, z) else RHO_VAC


def Te_profile(r, z):
    return TE_CD_EV if _in_cd(r, z) else TE_VAC_EV


def Ti_profile(r, z):
    return TI_CD_EV if _in_cd(r, z) else TI_VAC_EV


def velocity_profile(r, z):
    del r, z
    return (0.0, 0.0)


def vf_cd(r, z):
    del r, z
    return 1.0


def laser_power(t_s):
    if 0.0 <= t_s <= PULSE_DURATION:
        return P_PEAK_W
    return 0.0


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get("TENRYU_I1_VERBOSITY", "quiet"),
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
    low_density_extrapolation=True,
    materials=[
        Material(
            name="CD",
            A=A_CD,
            Z=Z_CD,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=opacity_block,
        )
    ],
    zbar=dict(model="fixed", fixed_value=Z_CD),
)

Geometry(
    volfrac=dict(CD=vf_cd),
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    velocity=velocity_profile,
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
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
        max_outer_iterations=50,
        outer_tol=1.0e-8,
        linear_solver_2d=I1GREYWEAK_LINEAR_SOLVER,
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
    absorption=dict(
        eps_n=1.0e-4,
        coulomb_log_floor=2.0,
        debug_dump_lasermesh=_env_bool("TENRYU_I1_DEBUG_DUMP_LASERMESH", False),
    ),
    lasermesh=dict(
        nr=64,
        nz=128,
        r_max_factor=LASERMESH_R_MAX_FACTOR,
        z_span_factor=LASERMESH_Z_SPAN_FACTOR,
        critical_margin=0.9999,
    ),
    raytrace=dict(
        cfl_ray=0.8,
        eps_crit=1.0e-4,
        intensity_cutoff=1.0e-8,
        max_steps=50000,
        debug_one_ray=_env_bool("TENRYU_I1_DEBUG_ONE_RAY", False),
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
            power=laser_power,
        )
    ],
)

ALE_CONFIG = dict(enabled=False)
if ALE_ENABLED:
    ALE_CONFIG = dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=True,
        remap_damage_dmax=0.05,
        remap_damage_axis_eta=0.02,
        remap_damage_axis_budget_enabled=True,
        remap_damage_axis_budget_factor=2.0,
        local_boundary_repair_enabled=I1_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=I1_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=PHASE2_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=I1_EMERGENCY_CELL_DEACTIVATION,
        corner_post_tangle_strict_floor_enabled=I1_ALE_STRICT_J_FLOOR,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        ke_conservation_closure=True,
        ke_conservation_closure_audit=False,
        ke_closure_redistribute_floor=True,
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
        volume_rate_cfl_enabled=I1_ALE,
        volume_rate_cfl_threshold=0.5,
        trial_volume_cfl_enabled=I1_ALE,
        trial_volume_cfl_floor_fraction=0.05,
        trial_volume_cfl_shrink_fraction=0.5,
        corner_jacobian_ale_trigger_enabled=I1_CORNER_J_ALE_TRIGGER,
        corner_jacobian_floor_eps=1.0e-6,
        corner_jacobian_ale_trigger_scale=0.5,
        driver_full_step_retry_enabled=I1_RETRY,
        driver_full_step_retry_max_attempts=I1_RETRY_MAX_ATTEMPTS,
        driver_retry_active_mesh_repair_enabled=I1_RETRY_ACTIVE_REPAIR,
        driver_retry_corner_balance_threshold=I1_CORNER_BALANCE_THRESHOLD,
        cascade_on_hydro_retry_enabled=I1_CASCADE_ON_HYDRO_RETRY,
        in_hydro_corner_j_guard_enabled=PHASE2_IN_HYDRO_CORNER_J_GUARD,
        regime_aware_corner_j_guard_enabled=PHASE2_REGIME_AWARE_CORNER_J_GUARD,
        axis_margin_guard_enabled=PHASE2_AXIS_MARGIN_GUARD,
        axis_guard_band_cells=PHASE2_AXIS_GUARD_BAND_CELLS,
        driver_retry_use_suggested_dt_enabled=PHASE2_DRIVER_RETRY_USE_SUGGESTED_DT,
        mesh_geometry_soft_fail_enabled=PHASE2_MESH_GEOMETRY_SOFT_FAIL,
    ),
    conduction=dict(
        enabled=True,
        solver=CONDUCTION_SOLVER,
        f_lim=F_LIM,
        test_kappa=TEST_KAPPA,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    ale=ALE_CONFIG,
    diagnostics=dict(
        phase_resolved_energy=True,
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
    history_every=1,
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
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=False),
)
