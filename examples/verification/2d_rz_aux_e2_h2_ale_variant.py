import os

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


NR = int(os.environ.get("TENRYU_E2_H2_NR", "64"))
NZ = int(os.environ.get("TENRYU_E2_H2_NZ", "8"))
SEED = int(os.environ.get("TENRYU_E2_H2_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_E2_H2_OUTDIR", "./build/output_verify_2d_rz_aux_e2_h2_ale")
AV_C2 = float(os.environ.get("TENRYU_E2_H2_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_E2_H2_CFL", "0.1"))
T_END = float(os.environ.get("TENRYU_E2_H2_T_END_S", "6.0e-7"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_E2_H2_ALE_EVERY_N_STEPS", "5"))

AXIS_REPAIR_MODE = os.environ.get("TENRYU_E2_H2_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_DAMAGE_GATE = _env_bool("TENRYU_E2_H2_REMAP_DAMAGE_GATE", "1")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_E2_H2_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_E2_H2_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_E2_H2_REMAP_DAMAGE_AXIS_BUDGET", "1")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_E2_H2_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_E2_H2_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_E2_H2_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_E2_H2_KE_CLOSURE", "1")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_E2_H2_KE_CLOSURE_AUDIT")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool(
    "TENRYU_E2_H2_KE_CLOSURE_REDISTRIBUTE_FLOOR", "1"
)
PHASE_RESOLVED_ENERGY = _env_bool("TENRYU_E2_H2_PHASE_RESOLVED_ENERGY", "1")

RHO_0_GCC = 1.0
T_INIT_eV = 1.0e-3
U0_CM_S = 1.0e6
GAMMA = 5.0 / 3.0


def _safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


CASE_NAME = (
    f"2d_rz_aux_e2_h2_ale_variant_nr{NR}_nz{NZ}_seed{SEED}"
    f"_C2{_safe_float_token(AV_C2)}_cfl{_safe_float_token(CFL)}"
    f"_ale{ALE_EVERY_N_STEPS}_p9_11_t31"
)

print(
    "[deck:2d_rz_aux_e2_h2_ale_variant] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"AV_C2={AV_C2} cfl={CFL} t_end_s={T_END} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} "
    f"remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} "
    f"ke_closure={KE_CONSERVATION_CLOSURE} "
    f"ke_redistribute_floor={KE_CLOSURE_REDISTRIBUTE_FLOOR}"
)


def rho_init(r, z):
    return RHO_0_GCC


def Te_init(r, z):
    return T_INIT_eV


def Ti_init(r, z):
    return T_INIT_eV


def velocity_init(r, z):
    return (-U0_CM_S, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=20000,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    z_min=-0.02,
    z_max=0.02,
    nr=NR,
    nz=NZ,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=1.0,
            Z=0.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-11,
        cfl_hydro=CFL,
        growth_factor=1.1,
        max_s=1.0e-9,
        min_s=1.0e-20,
    ),
    hydro=dict(
        boundary_2d=dict(
            r_inner="axis",
            r_outer="free",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_C1=0.1,
        av_C2=AV_C2,
    ),
    conduction=dict(enabled=False),
    ale=dict(
        enabled=True,
        every_n_steps=ALE_EVERY_N_STEPS,
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
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

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
