import math
import os

from tenryu_namelist import *


def _env_bool(name, default="1"):
    return os.environ.get(name, default) == "1"


def _safe_float_token(value):
    return f"{value:.12g}".replace(".", "p")


um = 1.0e-4
ns = 1.0e-9

NR = int(os.environ.get("TENRYU_A1_NR", "128"))
NZ = int(os.environ.get("TENRYU_A1_NZ", "256"))
SEED = int(os.environ.get("TENRYU_A1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_A1_OUTDIR", "./build/output_verify_2d_rz_a1_icf_shell_hydro")
T_END = float(os.environ.get("TENRYU_A1_T_END_S", str(0.2 * ns)))
MAX_STEPS = int(os.environ.get("TENRYU_A1_MAX_STEPS", "5000"))
CFL = float(os.environ.get("TENRYU_A1_CFL", "0.1"))
DT_INITIAL = float(os.environ.get("TENRYU_A1_DT_INITIAL_S", "2.0e-13"))
DT_MAX = float(os.environ.get("TENRYU_A1_DT_MAX_S", "2.0e-12"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_A1_ALE_EVERY_N_STEPS", "5"))
MAX_ALE_ITERATIONS = int(os.environ.get("TENRYU_A1_ALE_MAX_ITERATIONS", "100"))
AXIS_MOTION_FLOOR_FRACTION = float(
    os.environ.get("TENRYU_A1_AXIS_MOTION_FLOOR_FRACTION", "0.20")
)
AXIS_MARGIN_DT_FLOOR_FRACTION = float(
    os.environ.get("TENRYU_A1_AXIS_MARGIN_DT_FLOOR_FRACTION", "0.05")
)

AXIS_REPAIR_MODE = os.environ.get("TENRYU_A1_AXIS_REPAIR_MODE", "axis_spine_only")
PREVENTIVE_AXIS_GUARD_FRACTION = float(
    os.environ.get("TENRYU_A1_PREVENTIVE_AXIS_GUARD_FRACTION", "0.5")
)
REMAP_DAMAGE_GATE = _env_bool("TENRYU_A1_REMAP_DAMAGE_GATE")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_A1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_A1_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_A1_KE_CLOSURE")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_A1_KE_CLOSURE_AUDIT", "0")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_A1_KE_CLOSURE_REDISTRIBUTE_FLOOR")
PHASE_RESOLVED_ENERGY = _env_bool("TENRYU_A1_PHASE_RESOLVED_ENERGY")
DEBUG_PER_REMAP_LOG = _env_bool("TENRYU_A1_DEBUG_PER_REMAP_LOG")

R_MIN, R_MAX = 0.0, float(os.environ.get("TENRYU_A1_R_MAX_CM", str(500.0 * um)))
Z_MIN = float(os.environ.get("TENRYU_A1_Z_MIN_CM", str(-1000.0 * um)))
Z_MAX = float(os.environ.get("TENRYU_A1_Z_MAX_CM", str(1000.0 * um)))

R_FUEL = float(os.environ.get("TENRYU_A1_R_FUEL_CM", str(120.0 * um)))
R_ABLATOR_OUT = float(os.environ.get("TENRYU_A1_R_ABLATOR_OUT_CM", str(220.0 * um)))
RHO_FUEL = float(os.environ.get("TENRYU_A1_RHO_FUEL_GCC", "2.0e-2"))
RHO_ABLATOR = float(os.environ.get("TENRYU_A1_RHO_ABLATOR_GCC", "2.0"))
RHO_CORONA = float(os.environ.get("TENRYU_A1_RHO_CORONA_GCC", "1.0e-4"))
TE_FUEL_EV = float(os.environ.get("TENRYU_A1_TE_FUEL_EV", "20.0"))
TE_ABLATOR_EV = float(os.environ.get("TENRYU_A1_TE_ABLATOR_EV", "5.0"))
TE_CORONA_EV = float(os.environ.get("TENRYU_A1_TE_CORONA_EV", "20.0"))
TI_FUEL_EV = float(os.environ.get("TENRYU_A1_TI_FUEL_EV", str(TE_FUEL_EV)))
TI_ABLATOR_EV = float(os.environ.get("TENRYU_A1_TI_ABLATOR_EV", str(TE_ABLATOR_EV)))
TI_CORONA_EV = float(os.environ.get("TENRYU_A1_TI_CORONA_EV", str(TE_CORONA_EV)))
IMPL_VELOCITY = float(os.environ.get("TENRYU_A1_IMPL_VELOCITY_CM_S", "1.0e7"))
IMPL_RADIUS = float(os.environ.get("TENRYU_A1_IMPL_RADIUS_CM", str(R_ABLATOR_OUT)))

CASE_NAME = (
    f"2d_rz_a1_icf_shell_hydro_nr{NR}_nz{NZ}_seed{SEED}"
    f"_cfl{_safe_float_token(CFL)}_ale{ALE_EVERY_N_STEPS}"
    f"_axisdt{_safe_float_token(AXIS_MARGIN_DT_FLOOR_FRACTION)}"
    f"_axismotion{_safe_float_token(AXIS_MOTION_FLOOR_FRACTION)}_p9_11_t31"
)

print(
    "[deck:2d_rz_a1_icf_shell_hydro] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} cfl={CFL} "
    f"r_fuel_cm={R_FUEL} r_ablator_out_cm={R_ABLATOR_OUT} "
    f"rho_fuel_gcc={RHO_FUEL} rho_ablator_gcc={RHO_ABLATOR} "
    f"rho_corona_gcc={RHO_CORONA} impl_velocity_cm_s={IMPL_VELOCITY} "
    f"impl_radius_cm={IMPL_RADIUS} ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_motion_floor_fraction={AXIS_MOTION_FLOOR_FRACTION} "
    f"axis_margin_dt_floor_fraction={AXIS_MARGIN_DT_FLOOR_FRACTION} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"preventive_axis_guard_fraction={PREVENTIVE_AXIS_GUARD_FRACTION} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} ke_closure={KE_CONSERVATION_CLOSURE}"
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
    rr = radius(r, z)
    if rr <= 1.0e-300 or rr > IMPL_RADIUS:
        return (0.0, 0.0)
    return (-IMPL_VELOCITY * r / rr, -IMPL_VELOCITY * z / rr)


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
    verbosity="quiet",
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
    radiation_field="zero",
    enforce_sum_to_one=True,
)

Numerics(
    dt=dict(
        initial_s=DT_INITIAL,
        cfl_hydro=CFL,
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
    ),
    conduction=dict(enabled=False),
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
        ke_conservation_closure=KE_CONSERVATION_CLOSURE,
        ke_conservation_closure_audit=KE_CONSERVATION_CLOSURE_AUDIT,
        ke_closure_redistribute_floor=KE_CLOSURE_REDISTRIBUTE_FLOOR,
        debug_per_remap_log=DEBUG_PER_REMAP_LOG,
    ),
    diagnostics=dict(phase_resolved_energy=PHASE_RESOLVED_ENERGY),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    positivity=dict(clamp=True),
    safety=dict(energy_threshold=1.0, nan_fatal=True),
)

Output(
    directory=OUTDIR,
    format="hdf5",
    plot_every=0,
    history_every=10,
    checkpoint_every=0,
    plot_every_s=T_END / 4.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    save_namelist_copy=True,
    save_frozen_config=True,
)

Radiation(enabled=False)
Laser(enabled=False)

Diagnostics(
    enabled=True,
    every=1,
    energy_budget=dict(enabled=True, warn_threshold=1.0e-3),
    areal_density=dict(enabled=True, angles_deg=[0.0]),
    sphericity=dict(enabled=True, rho_threshold=0.1, modes=[0, 2, 4]),
)
