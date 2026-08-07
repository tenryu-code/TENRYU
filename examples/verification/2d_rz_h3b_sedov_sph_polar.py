import os

from tenryu_namelist import *


GAMMA = 5.0 / 3.0
A = 1.0
ZBAR = 1.0
RHO0 = 1.0
E_TOTAL = 1.0e13
P_AMBIENT = 1.0e6

NS = int(os.environ.get("TENRYU_H3BP_NS", "128"))
NTHETA = int(os.environ.get("TENRYU_H3BP_NTHETA", "256"))
S_MAX = float(os.environ.get("TENRYU_H3BP_S_MAX", "1.2"))
KAPPA = float(os.environ.get("TENRYU_H3BP_KAPPA", "4.0"))

DELTA_S = S_MAX / (NS + KAPPA)
S_MIN = KAPPA * DELTA_S
R_BLAST = S_MIN + 4.0 * DELTA_S
T_END = float(os.environ.get("TENRYU_H3BP_T_END_S", "6.0e-9"))
OUTDIR = os.environ.get("TENRYU_H3BP_OUTDIR", "./build/output_verify_2d_rz_h3bp")
CFL = float(os.environ.get("TENRYU_H3BP_CFL", "0.1"))
AV_C2 = float(os.environ.get("TENRYU_H3BP_AV_C2", "1.5"))


Main(
    name="h3bp_sedov_polar",
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=300000,
    seed=12345,
    verbosity="quiet",
    temperature_model="2T",
)

Mesh(
    r_min=0.0,
    r_max=S_MAX,
    z_min=-S_MAX,
    z_max=S_MAX,
    nr=NS,
    nz=NTHETA,
    grid="uniform",
    motion="lagrangian",
    logical_mesh_2d="spherical_polar_halfplane",
    spherical_polar_s_max=S_MAX,
    spherical_polar_kappa=KAPPA,
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


def temp_ev(p, rho):
    ev_to_erg = 1.6022e-12
    m_p = 1.6726219e-24
    cv_i = ev_to_erg / (A * m_p * (GAMMA - 1.0))
    cv_e = ZBAR * ev_to_erg / (A * m_p * (GAMMA - 1.0))
    return p / (rho * (GAMMA - 1.0) * (cv_i + cv_e))


def rho_init(r, z):
    return RHO0


def pressure_init(r, z):
    s = (r * r + z * z) ** 0.5
    if s < R_BLAST:
        v_dep = (4.0 / 3.0) * 3.141592653589793 * R_BLAST**3
        return P_AMBIENT + (GAMMA - 1.0) * E_TOTAL / v_dep
    return P_AMBIENT


def Te_init(r, z):
    return temp_ev(pressure_init(r, z), RHO0)


def Ti_init(r, z):
    return Te_init(r, z)


def velocity_init(r, z):
    return (0.0, 0.0)


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
        initial_s=1.0e-15,
        cfl_hydro=CFL,
        growth_factor=1.1,
        max_s=5.0e-9,
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
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    plot_every_s=T_END / 32.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
