import os

from tenryu_namelist import *


NR = int(os.environ.get("TENRYU_H1_NR", "64"))
NZ = int(os.environ.get("TENRYU_H1_NZ", "256"))
SEED = int(os.environ.get("TENRYU_H1_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_H1_OUTDIR", "./build/output_verify_2d_rz_h1")
AV_C2 = float(os.environ.get("TENRYU_H1_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_H1_CFL", "0.1"))
T_END = float(os.environ.get("TENRYU_H1_T_END_S", "1.0e-7"))

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
    f"2d_rz_h1_sod_planar_nr{NR}_nz{NZ}_seed{SEED}"
    f"_C2{_safe_float_token(AV_C2)}_cfl{_safe_float_token(CFL)}"
)

print(
    "[deck:2d_rz_h1_sod_planar] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"AV_C2={AV_C2} cfl={CFL} t_end_s={T_END} "
    f"Te_L_eV={T_L_eV:.6g} Te_R_eV={T_R_eV:.6g}"
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
