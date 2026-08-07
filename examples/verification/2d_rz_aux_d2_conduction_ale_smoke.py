import math
import os

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


NR = int(os.environ.get("TENRYU_D2_NR", "32"))
NZ = int(os.environ.get("TENRYU_D2_NZ", "32"))
SEED = int(os.environ.get("TENRYU_D2_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_D2_OUTDIR", "./build/output_verify_2d_rz_aux_d2_conduction_ale")
T_END = float(os.environ.get("TENRYU_D2_T_END_S", "2.0e-10"))
MAX_STEPS = int(os.environ.get("TENRYU_D2_MAX_STEPS", "80"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_D2_ALE_EVERY_N_STEPS", "5"))
CFL = float(os.environ.get("TENRYU_D2_CFL", "0.2"))
CONDUCTION_TEST_KAPPA = float(os.environ.get("TENRYU_D2_TEST_KAPPA", "1.0e14"))

AXIS_REPAIR_MODE = os.environ.get("TENRYU_D2_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_DAMAGE_GATE = _env_bool("TENRYU_D2_REMAP_DAMAGE_GATE", "1")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_D2_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_D2_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_D2_REMAP_DAMAGE_AXIS_BUDGET", "1")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_D2_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_D2_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_D2_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_D2_KE_CLOSURE", "1")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_D2_KE_CLOSURE_AUDIT")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool("TENRYU_D2_KE_CLOSURE_REDISTRIBUTE_FLOOR", "1")
PHASE_RESOLVED_ENERGY = _env_bool("TENRYU_D2_PHASE_RESOLVED_ENERGY", "1")

R_MIN, R_MAX = 0.0, 5.0e-3
Z_MIN, Z_MAX = -2.5e-3, 2.5e-3
RHO_CD_GCC = 1.0
TE_AMBIENT_EV = 20.0
TE_HOT_EV = 100.0
TI_INIT_EV = 20.0
HOT_R0 = 1.5e-3
HOT_Z0 = 0.0
HOT_SIGMA = 7.5e-4


def _safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


CASE_NAME = (
    f"2d_rz_aux_d2_conduction_ale_smoke_nr{NR}_nz{NZ}_seed{SEED}"
    f"_cfl{_safe_float_token(CFL)}_ale{ALE_EVERY_N_STEPS}_p9_11_t31"
)

print(
    "[deck:2d_rz_aux_d2_conduction_ale_smoke] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"t_end_s={T_END} max_steps={MAX_STEPS} cfl={CFL} "
    f"Te_ambient_eV={TE_AMBIENT_EV} Te_hot_eV={TE_HOT_EV} "
    f"test_kappa={CONDUCTION_TEST_KAPPA:.6e} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} "
    f"remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} "
    f"ke_closure={KE_CONSERVATION_CLOSURE} "
    f"ke_redistribute_floor={KE_CLOSURE_REDISTRIBUTE_FLOOR}"
)


def rho_init(r, z):
    return RHO_CD_GCC


def _hot_weight(r, z):
    dr = (r - HOT_R0) / HOT_SIGMA
    dz = (z - HOT_Z0) / HOT_SIGMA
    return math.exp(-(dr * dr + dz * dz))


def Te_init(r, z):
    return TE_AMBIENT_EV + (TE_HOT_EV - TE_AMBIENT_EV) * _hot_weight(r, z)


def Ti_init(r, z):
    return TI_INIT_EV


def velocity_init(r, z):
    return (0.0, 0.0)


mat_cd = Material(
    name="CD",
    A=7.0,
    Z=3.5,
    eos=dict(model="tmat", file="TMAT-H5/CD.tmat.h5"),
    opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
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
    materials=[mat_cd],
    zbar=dict(model="tabular"),
)

Geometry(
    volfrac=dict(CD=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-12,
        cfl_hydro=CFL,
        cfl_cond=0.25,
        growth_factor=1.1,
        max_s=1.0e-11,
        min_s=1.0e-22,
    ),
    hydro=dict(
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
        test_kappa=CONDUCTION_TEST_KAPPA,
    ),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
        max_iterations=80,
        convergence_tol=1.0e-9,
        max_displacement_fraction=0.5,
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
    ),
    diagnostics=dict(phase_resolved_energy=PHASE_RESOLVED_ENERGY),
    diagnostics_every=1,
    floors=dict(rho_floor_gcc=1.0e-8, Te_floor_eV=0.1, Ti_floor_eV=0.1),
    safety=dict(energy_threshold=1.0, nan_fatal=True),
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
