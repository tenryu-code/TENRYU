import math
import os

from tenryu_namelist import *


def _env_bool(name, default="1"):
    return os.environ.get(name, default) == "1"


NR = int(os.environ.get("TENRYU_D1_NR", "32"))
NZ = int(os.environ.get("TENRYU_D1_NZ", "64"))
SEED = int(os.environ.get("TENRYU_D1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_D1_OUTDIR", "./build/output_verify_2d_rz_aux_d1_laser_smoke")
T_END = float(os.environ.get("TENRYU_D1_T_END_S", "8.0e-12"))
MAX_STEPS = int(os.environ.get("TENRYU_D1_MAX_STEPS", "80"))
DT_S = float(os.environ.get("TENRYU_D1_DT_S", "1.0e-13"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_D1_ALE_EVERY_N_STEPS", "5"))
MAX_ALE_ITERATIONS = int(os.environ.get("TENRYU_D1_ALE_MAX_ITERATIONS", "60"))
AXIS_REPAIR_MODE = os.environ.get("TENRYU_D1_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_DAMAGE_GATE = _env_bool("TENRYU_D1_REMAP_DAMAGE_GATE")
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_D1_REMAP_DAMAGE_AXIS_BUDGET")
REMAP_SCHEME = os.environ.get("TENRYU_D1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_D1_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_D1_KE_CLOSURE")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_D1_KE_CLOSURE_REDISTRIBUTE_FLOOR")

R_MIN, R_MAX = 0.0, 8.0e-2
Z_MIN, Z_MAX = -8.0e-2, 8.0e-2

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0

RHO_ABLATOR = float(os.environ.get("TENRYU_D1_RHO_ABLATOR_GCC", "2.0e-2"))
RHO_CORONA = float(os.environ.get("TENRYU_D1_RHO_CORONA_GCC", "2.0e-3"))
T_ABLATOR_EV = float(os.environ.get("TENRYU_D1_T_ABLATOR_EV", "1.5"))
TARGET_RADIUS = float(os.environ.get("TENRYU_D1_TARGET_RADIUS_CM", "3.5e-2"))
TRANSITION_WIDTH = float(os.environ.get("TENRYU_D1_TRANSITION_WIDTH_CM", "6.0e-3"))
P_PEAK_W = float(os.environ.get("TENRYU_D1_LASER_PEAK_W", "2.0e10"))
SPOT_W0_UM = float(os.environ.get("TENRYU_D1_LASER_W0_UM", "220.0"))
RAYS_PER_BEAM = int(os.environ.get("TENRYU_D1_RAYS_PER_BEAM", "32"))

cv_i = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
cv_e = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
P0 = (GAMMA - 1.0) * RHO_ABLATOR * (cv_i + cv_e) * T_ABLATOR_EV

CASE_NAME = f"2d_rz_aux_d1_laser_smoke_nr{NR}_nz{NZ}_seed{SEED}"

print(
    "[deck:2d_rz_aux_d1_laser_smoke] "
    f"nr={NR} nz={NZ} max_steps={MAX_STEPS} dt_s={DT_S} outdir={OUTDIR} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} axis_repair_mode={AXIS_REPAIR_MODE} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} ke_closure={KE_CONSERVATION_CLOSURE} "
    f"laser_mode=raytrace_3d rays_per_beam={RAYS_PER_BEAM}"
)


def rho_init(r, z):
    radius = math.hypot(r, z)
    blend = 0.5 * (1.0 - math.tanh((radius - TARGET_RADIUS) / TRANSITION_WIDTH))
    return RHO_CORONA + (RHO_ABLATOR - RHO_CORONA) * blend


def temperature_init(r, z):
    rho = max(rho_init(r, z), 1.0e-30)
    return P0 / ((GAMMA - 1.0) * rho * (cv_i + cv_e))


def velocity_init(r, z):
    return (0.0, 0.0)


def laser_power(t_s):
    if t_s < 0.0 or t_s > T_END:
        return 0.0
    x = t_s / max(T_END, 1.0e-300)
    return P_PEAK_W * math.sin(math.pi * x) ** 2


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=MAX_STEPS,
    seed=SEED,
    verbosity="quiet",
    # This deck intentionally starts with Te == Ti while forcing 2T mode.
    # A step-1 "2T model collapse" warning is expected before laser heating
    # creates electron/ion separation in this short smoke run.
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
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="hydrogen",
            A=A,
            Z=ZBAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=ZBAR),
)

Geometry(
    volfrac=dict(hydrogen=lambda r, z: 1.0),
    rho=rho_init,
    Te=temperature_init,
    Ti=temperature_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=DT_S,
        max_s=DT_S,
        min_s=1.0e-20,
        growth_factor=1.0,
        cfl_hydro=0.25,
    ),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        quality_threshold=0.2,
        max_iterations=MAX_ALE_ITERATIONS,
        convergence_tol=1.0e-8,
        max_displacement_fraction=0.5,
        preventive_axis_guard_fraction=0.1,
        axis_z_motion="fixed",
        axis_repair_mode=AXIS_REPAIR_MODE,
        remap_damage_gate_enabled=REMAP_DAMAGE_GATE,
        remap_damage_dmax=0.05,
        remap_damage_axis_eta=0.02,
        remap_damage_axis_budget_enabled=REMAP_DAMAGE_AXIS_BUDGET,
        remap_damage_axis_budget_factor=2.0,
        remap_scheme=REMAP_SCHEME,
        remap_ms2_limiter=REMAP_MS2_LIMITER,
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_conservation_closure_audit=False,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
    ),
    diagnostics=dict(phase_resolved_energy=True),
    floors=dict(rho_floor_gcc=1.0e-8, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)

Laser(
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
            focus=[0.0, 0.0, 0.0],
            f_number=4.0,
            profile=dict(model="super_gaussian", w0_um=SPOT_W0_UM, m=4),
            power=laser_power,
        )
    ],
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=5,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(
    enabled=True,
    every=5,
    laser_pattern=dict(enabled=True, absorbed_power_profile=True, critical_surface=True, per_beam=False),
)
