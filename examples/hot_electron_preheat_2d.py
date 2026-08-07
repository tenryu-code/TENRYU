"""2D RZ hot-electron preheat example (i7a-derived axis-disk ablation deck).

Hot-electron preheat: NUMERICS §5.11 (model) + its 2D subsection (transport); spec docs/design/2d_hote_port_spec.md.
"""

import math
import os
import sys


def _print_help_and_exit():
    print(
        """I7a coupled-stack planar laser-ablation 2D RZ axis-disk deck.

Primary knobs:
  TENRYU_I7A_NR                         default: 16
  TENRYU_I7A_NZ                         default: 512
  TENRYU_I7A_SEED                       default: 12345
  TENRYU_I7A_OUTDIR                     default: ./build/output_verify_i7a_coupled_stack_2d_rz_slab
  TENRYU_I7A_T_END_NS                   default: 1.0
  TENRYU_I7A_RAMP_NS                    default: 0.2
  TENRYU_I7A_P_PEAK_W                   default: 2.0e10
  TENRYU_I7A_SPOT_W0_UM                 default: 220.0
  TENRYU_I7A_RAYS_PER_BEAM              default: 32
  TENRYU_I7A_SLAB_THICK_UM              default: 50.0
  TENRYU_I7A_RHO_SLAB                   default: 1.0
  TENRYU_I7A_RHO_CORONA                 default: 1.0e-4
  TENRYU_I7A_RHO_CORONA_MIN             default: 1.0e-4
  TENRYU_I7A_CORONA_SCALE_UM            default: 10.0
  TENRYU_I7A_T0_EV                      default: 0.1
  TENRYU_I7A_KAPPA_CM2_G                default: 10.0
  TENRYU_I7A_COND_SOLVER                default: sts
  TENRYU_I7A_CFL_HYDRO                  default: 0.2
  TENRYU_I7A_DT_INITIAL_S               default: 1.0e-13
  TENRYU_I7A_DT_MAX_S                   default: min(T_END/200, CFL_HYDRO*DZ/3.0e7)
  TENRYU_I7A_MAX_STEPS                  default: 2000000
  TENRYU_I7A_PLOT_EVERY_S               default: T_END/20

Dry parse:
  python3 examples/verification/i7a_coupled_stack_2d_rz_slab.py --help

Normal parsing is performed by TENRYU with tenryu_namelist available; this
help path exits before importing tenryu_namelist and requires no GPU.
"""
    )
    raise SystemExit(0)


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    _print_help_and_exit()

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


NR = _env_int("TENRYU_I7A_NR", 16)
NZ = _env_int("TENRYU_I7A_NZ", 512)
SEED = _env_int("TENRYU_I7A_SEED", 12345)
OUTDIR = os.environ.get(
    "TENRYU_I7A_OUTDIR",
    "./build/output_verify_i7a_coupled_stack_2d_rz_slab",
)
T_END_NS = _env_float("TENRYU_I7A_T_END_NS", 1.0)
RAMP_NS = _env_float("TENRYU_I7A_RAMP_NS", 0.2)
T_END = T_END_NS * ns
RAMP_S = RAMP_NS * ns
P_PEAK_W = _env_float("TENRYU_I7A_P_PEAK_W", 2.0e10)
SPOT_W0_UM = _env_float("TENRYU_I7A_SPOT_W0_UM", 220.0)
RAYS_PER_BEAM = _env_int("TENRYU_I7A_RAYS_PER_BEAM", 32)
# Geometric optics: rays converge to a POINT at the focus (no diffraction
# waist). Defocusing behind the slab gives the cone radius
# DEFOCUS/(2*f_number) ~ 200 um at the surface (f/4, 1.6 cm).
# Measured: defocus <= 0.10 cm absorbs fully (7-cell footprint); >= 0.15 cm the beam
# deposits NOTHING (mechanism unmapped — open laser finding, see i7a spec).
DEFOCUS_CM = _env_float("TENRYU_I7A_DEFOCUS_CM", 0.10)
SLAB_THICK_UM = _env_float("TENRYU_I7A_SLAB_THICK_UM", 50.0)
RHO_SLAB = _env_float("TENRYU_I7A_RHO_SLAB", 1.0)
# Corona base ~2x the 351 nm critical density for CH (rho_crit ~ 2.9e-2 g/cc):
# the critical surface must sit INSIDE the resolved exponential ramp, else the
# laser deposits into underdense cells whose flux-limited conduction cannot
# drain the intensity (measured Te runaway to ~1e5 eV at 1e-4 g/cc).
RHO_CORONA = _env_float("TENRYU_I7A_RHO_CORONA", 6.0e-2)
RHO_CORONA_MIN = _env_float("TENRYU_I7A_RHO_CORONA_MIN", 1.0e-4)
CORONA_SCALE_UM = _env_float("TENRYU_I7A_CORONA_SCALE_UM", 10.0)
T0_EV = _env_float("TENRYU_I7A_T0_EV", 0.1)
KAPPA_CM2_G = _env_float("TENRYU_I7A_KAPPA_CM2_G", 10.0)
CFL_HYDRO = _env_float("TENRYU_I7A_CFL_HYDRO", 0.2)
DT_INITIAL = _env_float("TENRYU_I7A_DT_INITIAL_S", 1.0e-13)
MAX_STEPS = _env_int("TENRYU_I7A_MAX_STEPS", 2000000)

R_MIN = 0.0
R_MAX = 2.0 * SPOT_W0_UM * um
Z_MIN = 0.0
Z_MAX = 40.0 * SLAB_THICK_UM * um
Z_SLAB_TOP_CM = 10.0 * SLAB_THICK_UM * um
SLAB_THICK_CM = SLAB_THICK_UM * um
SLAB_BOTTOM_CM = Z_SLAB_TOP_CM - SLAB_THICK_CM
CORONA_SCALE_CM = CORONA_SCALE_UM * um
DR = (R_MAX - R_MIN) / float(NR)
DZ = (Z_MAX - Z_MIN) / float(NZ)
W0_CM = SPOT_W0_UM * um
LASER_INTENSITY_W_CM2 = P_PEAK_W / (math.pi * W0_CM * W0_CM)
# 3.0e7 cm/s is the c_s scale for a 100 eV CH-like slab dt cap.
CFL_DT_CAP = CFL_HYDRO * DZ / 3.0e7
DEFAULT_DT_MAX = min(T_END / 200.0, CFL_DT_CAP)
DT_MAX = _env_float("TENRYU_I7A_DT_MAX_S", DEFAULT_DT_MAX)
PLOT_EVERY_S = _env_float("TENRYU_I7A_PLOT_EVERY_S", T_END / 20.0)

if not (NR > 0 and NZ > 0):
    raise ValueError("I7a 2D RZ slab grid requires NR>0 and NZ>0")
if not (R_MAX > R_MIN and Z_MAX > Z_MIN):
    raise ValueError("I7a geometry bounds require r_max>r_min and z_max>z_min")
if not (SLAB_THICK_CM > 0.0 and CORONA_SCALE_CM > 0.0 and RAMP_S > 0.0):
    raise ValueError("I7a slab thickness, corona scale, and ramp duration must be positive")

A_CH = 6.5
ZBAR = 3.5
GAMMA = 5.0 / 3.0
RHO_FLOOR = 1.0e-8
TE_FLOOR = 1.0e-3
TI_FLOOR = 1.0e-3

LINEAR_SOLVER_2D = os.environ.get("TENRYU_I7A_LINEAR_SOLVER", "cusparse_cg_zline")
PRODUCTION_AUDIT_TIER = os.environ.get("TENRYU_I7A_PRODUCTION_AUDIT_TIER", "A")
if PRODUCTION_AUDIT_TIER not in ("A", "B"):
    raise ValueError('TENRYU_I7A_PRODUCTION_AUDIT_TIER must be "A" or "B"')
VERBOSE_DIAG = _env_bool("TENRYU_I7A_VERBOSE_DIAG", False)

I7A_ALE = _env_bool("TENRYU_I7A_ALE_ENABLED", True)
I7A_ALE_EVERY_N_STEPS = _env_int("TENRYU_I7A_ALE_EVERY_N_STEPS", 1)
I7A_AXIS_REPAIR_MODE = os.environ.get("TENRYU_I7A_AXIS_REPAIR_MODE", "axis_spine_only")
I7A_LOCAL_BOUNDARY_REPAIR = _env_bool("TENRYU_I7A_LOCAL_BOUNDARY_REPAIR", False)
I7A_MULTI_NODE_BOUNDARY_REPAIR = _env_bool(
    "TENRYU_I7A_MULTI_NODE_BOUNDARY_REPAIR", False
)
I7A_MULTI_NODE_INTERIOR_REPAIR = _env_bool(
    "TENRYU_I7A_MULTI_NODE_INTERIOR_REPAIR", False
)
I7A_EMERGENCY_CELL_DEACTIVATION = _env_bool(
    "TENRYU_I7A_EMERGENCY_CELL_DEACTIVATION", False
)
I7A_CONSERVATIVE_REMAP = _env_bool("TENRYU_I7A_CONSERVATIVE_REMAP", True)
TOTAL_ENERGY_REMAP = _env_bool("TENRYU_I7A_TOTAL_ENERGY_REMAP", True)
WORK_SPLIT_AUDIT = _env_bool("TENRYU_I7A_WORK_SPLIT_AUDIT", False)
HLLC_Z_FLUX = _env_bool("TENRYU_I7A_HLLC_Z_FLUX", True)
HLLC_Z_AUDIT = _env_bool("TENRYU_I7A_HLLC_Z_AUDIT", False)
HLLC_Z_STRICT = _env_bool("TENRYU_I7A_HLLC_Z_STRICT", False)
I7A_CONDUCTION_ENABLED = _env_bool("TENRYU_I7A_CONDUCTION_ENABLED", True)
I7A_RADIATION_ENABLED = _env_bool("TENRYU_I7A_RADIATION_ENABLED", True)
I7A_QEI_PHYSICAL = _env_bool("TENRYU_I7A_QEI_PHYSICAL", True)

MESH_MOTION = os.environ.get(
    "TENRYU_I7A_MESH_MOTION",
    "ale" if I7A_ALE else "lagrangian",
).lower()
if MESH_MOTION not in ("ale", "lagrangian"):
    raise ValueError("TENRYU_I7A_MESH_MOTION must be ale or lagrangian")

ale_config = dict(enabled=False)
if I7A_ALE:
    ale_config = dict(
        enabled=True,
        every_n_steps=I7A_ALE_EVERY_N_STEPS,
        max_iterations=100,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
        predictive_acceptance_enabled=True,
        predictive_acceptance_axis_floor_fraction=0.10,
        predictive_acceptance_cell_vol_floor_fraction=0.05,
        axis_repair_mode=I7A_AXIS_REPAIR_MODE,
        conservative_remap_enabled=I7A_CONSERVATIVE_REMAP,
        conservative_remap_target="reference",
        conservative_remap_radiation_enabled=True,
        conservative_remap_order="second_order_van_leer",
        local_boundary_repair_enabled=I7A_LOCAL_BOUNDARY_REPAIR,
        multi_node_boundary_repair_enabled=I7A_MULTI_NODE_BOUNDARY_REPAIR,
        multi_node_interior_repair_enabled=I7A_MULTI_NODE_INTERIOR_REPAIR,
        emergency_cell_deactivation_enabled=I7A_EMERGENCY_CELL_DEACTIVATION,
    )

PRODUCTION_AUDIT = dict(
    enabled=True,
    tier=PRODUCTION_AUDIT_TIER,
    gcl=dict(enabled=False),
    positivity=dict(enabled=False),
    audit_json_path=os.path.join(OUTDIR, "audit_summary.json"),
)

CASE_NAME = f"i7a_coupled_stack_2d_rz_slab_nr{NR}_nz{NZ}_seed{SEED}"

print(
    "[deck:i7a_coupled_stack_2d_rz_slab] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"r_min_cm={R_MIN} r_max_cm={R_MAX} dr_cm={DR} "
    f"z_min_cm={Z_MIN} z_max_cm={Z_MAX} dz_cm={DZ} "
    f"slab_thick_cm={SLAB_THICK_CM} z_slab_bottom_cm={SLAB_BOTTOM_CM} "
    f"z_slab_top_cm={Z_SLAB_TOP_CM} rho_slab_gcc={RHO_SLAB} "
    f"rho_corona_gcc={RHO_CORONA} rho_corona_min={RHO_CORONA_MIN} corona_scale_cm={CORONA_SCALE_CM} "
    f"rho_floor_gcc={RHO_FLOOR} T0_eV={T0_EV} "
    f"A_amu={A_CH} zbar={ZBAR} gamma={GAMMA} kappa_a_cm2_g={KAPPA_CM2_G} "
    f"laser_peak_W={P_PEAK_W} ramp_s={RAMP_S} spot_w0_um={SPOT_W0_UM} "
    f"defocus_cm={DEFOCUS_CM} spot_at_surface_um={DEFOCUS_CM / (2.0 * 4.0) * 1.0e4} "
    f"rays_per_beam={RAYS_PER_BEAM} intensity_W_cm2={LASER_INTENSITY_W_CM2} "
    f"dt_initial_s={DT_INITIAL} dt_max_s={DT_MAX} cfl_dt_cap_s={CFL_DT_CAP} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} plot_every_s={PLOT_EVERY_S} "
    f"hydro=True ale={I7A_ALE} laser=True conduction={I7A_CONDUCTION_ENABLED} "
    f"radiation={I7A_RADIATION_ENABLED} qei_physical={I7A_QEI_PHYSICAL} qei=True grey_fld=True "
    f"flux_limiter=levermore_pomraning linear_solver_2d={LINEAR_SOLVER_2D} "
    f"production_audit_tier={PRODUCTION_AUDIT_TIER} "
    f"hllc={HLLC_Z_FLUX} total_energy_remap={TOTAL_ENERGY_REMAP} single_material=True"
)


def rho_init(r, z):
    del r
    # Laser IB into near-vacuum cells (cv ~ 0) is a solver pathology, not physics;
    # the certified laser lineage keeps the whole domain at corona density.
    if SLAB_BOTTOM_CM <= z <= Z_SLAB_TOP_CM:
        return max(RHO_SLAB, RHO_CORONA_MIN)
    if z > Z_SLAB_TOP_CM:
        return max(RHO_CORONA * math.exp(-(z - Z_SLAB_TOP_CM) / CORONA_SCALE_CM), RHO_CORONA_MIN)
    return RHO_CORONA_MIN


def Te_init(r, z):
    del r, z
    return T0_EV


def Ti_init(r, z):
    del r, z
    return T0_EV


def velocity_init(r, z):
    del r, z
    return (0.0, 0.0)


def vf_ch(r, z):
    del r, z
    return 1.0


def laser_power(t_s):
    if t_s < 0.0 or t_s > T_END:
        return 0.0
    return min(t_s / RAMP_S, 1.0) * P_PEAK_W


HYDRO_BOUNDARY = dict(
    r_inner="axis",
    r_outer="free",
    z_bottom="reflect",
    z_top="free",
)
FLD_BOUNDARY = dict(inner_r="reflect", outer_r="vacuum", z="vacuum")

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity=os.environ.get(
        "TENRYU_I7A_VERBOSITY",
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
    floors=dict(rho_floor_gcc=RHO_FLOOR, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
)

Materials(
    materials=[
        Material(
            name="CH",
            A=A_CH,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=KAPPA_CM2_G, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(CH=vf_ch),
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
        cfl_hydro=CFL_HYDRO,
        cfl_cond=0.3,
        growth_factor=1.1,
        max_s=DT_MAX,
        min_s=1.0e-22,
        f_min_fleck=0.01,
    ),
    hydro=dict(
        enabled=True,
        boundary_2d=HYDRO_BOUNDARY,
        av_C1=0.5,
        av_C2=2.0,
        qei_multiplier=1.0 if I7A_QEI_PHYSICAL else 0.0,
        total_energy_remap_2d_rz=TOTAL_ENERGY_REMAP,
        work_split_audit_2d_rz=WORK_SPLIT_AUDIT,
        hllc_z_flux_2d_rz=HLLC_Z_FLUX,
        hllc_z_flux_audit_2d_rz=HLLC_Z_AUDIT,
        hllc_z_flux_strict_quasi_1d=HLLC_Z_STRICT,
        work_split_audit_cell_every_n_steps=_env_int(
            "TENRYU_I7A_WORK_SPLIT_AUDIT_CELL_EVERY_N_STEPS", 0
        ),
        work_split_audit_all_rows=_env_bool(
            "TENRYU_I7A_WORK_SPLIT_AUDIT_ALL_ROWS", False
        ),
        # rz_geometric_cfl_* omitted: annular z-slab-family knobs — the tri-fan center
        # CFL is degenerate at the r=0 axis (measured dt_hydro=0 at step 0).
    ),
    conduction=dict(
        enabled=I7A_CONDUCTION_ENABLED,
        solver=os.environ.get("TENRYU_I7A_COND_SOLVER", "sts"),
        f_lim=0.06,
        mfp_limiter_C=0.1,
        test_kappa=_env_float("TENRYU_I7A_COND_TEST_KAPPA", -1.0),
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    ale=ale_config,
    diagnostics=dict(phase_resolved_energy=True, production_audit=PRODUCTION_AUDIT),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=RHO_FLOOR, Te_floor_eV=TE_FLOOR, Ti_floor_eV=TI_FLOOR),
    safety=dict(
        energy_fatal=False,
        nan_fatal=True,
        energy_threshold=1.0e-3,
        clamp_warn_threshold=0,
        clamp_fatal_threshold=1000000000,
    ),
)

Radiation(
    enabled=I7A_RADIATION_ENABLED,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(enabled=False, two_stage=False, difference=dict(enabled=False)),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=50,
        # outer_tol 1e-4 (user-approved 2026-07-10): 1e-8 is unreachable at rho~0.87/outer; see 2d-compute-speed census.
        outer_tol=_env_float("TENRYU_I7A_OUTER_TOL", 1.0e-4),
        cg_max_iter=_env_int("TENRYU_I7A_FLD_CG_MAX_ITER", 4096),
        linear_solver_2d=LINEAR_SOLVER_2D,
        boundary=FLD_BOUNDARY,
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum", bottom_z="vacuum", top_z="vacuum"),
)

Laser(
    hot_electron=dict(enable=True, eta_hot=0.02, T_hot_eV=5.0e4),  # prescribed LPI preheat, 2% of ray power at n_c/4
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=RAYS_PER_BEAM,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    lasermesh=dict(nr=64, nz=128, r_max_factor=1.2, z_span_factor=1.2, critical_margin=0.9999),
    raytrace=dict(cfl_ray=0.8, eps_crit=1.0e-4, intensity_cutoff=1.0e-6, max_steps=50000),
    raytrace_skip_config=dict(enabled=False, threshold=0.01, max_consecutive=10),
    deposit=dict(deposit_smooth_passes=1, deposit_smooth_alpha=0.25),
    beams=[
        LaserBeam(
            name="axial_top",
            direction=[0.0, 0.0, -1.0],
            focus=[0.0, 0.0, Z_SLAB_TOP_CM - DEFOCUS_CM],
            f_number=4.0,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=4),
            power=laser_power,
        )
    ],
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=PLOT_EVERY_S,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "Pe", "Pi", "Qvisc", "rad_E", "zbar"],
)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=False),
)
