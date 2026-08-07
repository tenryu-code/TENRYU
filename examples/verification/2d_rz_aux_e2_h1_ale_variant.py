import os

from tenryu_namelist import *


def _env_bool(name, default="0"):
    return os.environ.get(name, default) == "1"


NR = int(os.environ.get("TENRYU_E2_H1_NR", "16"))
NZ = int(os.environ.get("TENRYU_E2_H1_NZ", "64"))
SEED = int(os.environ.get("TENRYU_E2_H1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_E2_H1_OUTDIR", "./build/output_verify_2d_rz_aux_e2_h1_ale")
AV_C2 = float(os.environ.get("TENRYU_E2_H1_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_E2_H1_CFL", "0.1"))
T_END = float(os.environ.get("TENRYU_E2_H1_T_END_S", "1.0e-7"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_E2_H1_ALE_EVERY_N_STEPS", "5"))

AXIS_REPAIR_MODE = os.environ.get("TENRYU_E2_H1_AXIS_REPAIR_MODE", "axis_spine_only")
REMAP_DAMAGE_GATE = _env_bool("TENRYU_E2_H1_REMAP_DAMAGE_GATE", "1")
REMAP_DAMAGE_DMAX = float(os.environ.get("TENRYU_E2_H1_REMAP_DAMAGE_DMAX", "0.05"))
REMAP_DAMAGE_AXIS_ETA = float(os.environ.get("TENRYU_E2_H1_REMAP_DAMAGE_AXIS_ETA", "0.02"))
REMAP_DAMAGE_AXIS_BUDGET = _env_bool("TENRYU_E2_H1_REMAP_DAMAGE_AXIS_BUDGET", "1")
REMAP_DAMAGE_AXIS_BUDGET_FACTOR = float(
    os.environ.get("TENRYU_E2_H1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR", "2.0")
)
REMAP_SCHEME = os.environ.get("TENRYU_E2_H1_REMAP_SCHEME", "ms2_moments")
REMAP_MS2_LIMITER = os.environ.get("TENRYU_E2_H1_REMAP_MS2_LIMITER", "van_leer")
KE_CONSERVATION_CLOSURE = _env_bool("TENRYU_E2_H1_KE_CLOSURE", "1")
KE_CONSERVATION_CLOSURE_AUDIT = _env_bool("TENRYU_E2_H1_KE_CLOSURE_AUDIT")
KE_CLOSURE_REDISTRIBUTE_FLOOR = _env_bool(
    "TENRYU_E2_H1_KE_CLOSURE_REDISTRIBUTE_FLOOR", "1"
)
PHASE_RESOLVED_ENERGY = _env_bool("TENRYU_E2_H1_PHASE_RESOLVED_ENERGY", "1")

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
A = 1.0
GAMMA = 5.0 / 3.0
P_REF_ERG_PER_CC = 1.0e12
RHO_L_GCC = 1.0
RHO_R_GCC = 0.125
P_L_ERG_PER_CC = P_REF_ERG_PER_CC
P_R_ERG_PER_CC = 0.1 * P_REF_ERG_PER_CC


def _safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def temp_eV_from_pressure(p_erg_per_cc, rho_gcc):
    cv_eV = 1.5 * EV_TO_ERG / (A * M_P)
    return p_erg_per_cc / (rho_gcc * (GAMMA - 1.0) * cv_eV)


T_L_eV = temp_eV_from_pressure(P_L_ERG_PER_CC, RHO_L_GCC)
T_R_eV = temp_eV_from_pressure(P_R_ERG_PER_CC, RHO_R_GCC)

CASE_NAME = (
    f"2d_rz_aux_e2_h1_ale_variant_nr{NR}_nz{NZ}_seed{SEED}"
    f"_C2{_safe_float_token(AV_C2)}_cfl{_safe_float_token(CFL)}"
    f"_ale{ALE_EVERY_N_STEPS}_p9_11_t31"
)

print(
    "[deck:2d_rz_aux_e2_h1_ale_variant] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"AV_C2={AV_C2} cfl={CFL} t_end_s={T_END} "
    f"Te_L_eV={T_L_eV:.6g} Te_R_eV={T_R_eV:.6g} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS} "
    f"axis_repair_mode={AXIS_REPAIR_MODE} "
    f"remap_damage_gate={REMAP_DAMAGE_GATE} "
    f"remap_damage_axis_budget={REMAP_DAMAGE_AXIS_BUDGET} "
    f"remap_scheme={REMAP_SCHEME} "
    f"ke_closure={KE_CONSERVATION_CLOSURE} "
    f"ke_redistribute_floor={KE_CLOSURE_REDISTRIBUTE_FLOOR}"
)


def rho_init(r, z):
    return RHO_L_GCC if z < 0.0 else RHO_R_GCC


def Te_init(r, z):
    return T_L_eV if z < 0.0 else T_R_eV


def Ti_init(r, z):
    return T_L_eV if z < 0.0 else T_R_eV


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=10000,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    z_min=-0.5,
    z_max=0.5,
    nr=NR,
    nz=NZ,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=A,
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
            r_outer="reflect",
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
