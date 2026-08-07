import math
import os

from tenryu_namelist import *


NR = int(os.environ.get("TENRYU_H3A_NR", "256"))
NZ = int(os.environ.get("TENRYU_H3A_NZ", "32"))
SEED = int(os.environ.get("TENRYU_H3A_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_H3A_OUTDIR", "./build/output_verify_2d_rz_h3a")
AV_C2 = float(os.environ.get("TENRYU_H3A_AV_C2", "1.5"))
CFL = float(os.environ.get("TENRYU_H3A_CFL", "0.1"))
T_END = float(os.environ.get("TENRYU_H3A_T_END_S", "8.0e-9"))
E_PER_CM = float(os.environ.get("TENRYU_H3A_E_PER_CM", "1.0e15"))
BLAST_CELLS = int(os.environ.get("TENRYU_H3A_BLAST_CELLS", "8"))
ALE_EVERY_N_STEPS = int(os.environ.get("TENRYU_H3A_ALE_EVERY_N_STEPS", "0"))


def _safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


CASE_NAME = (
    f"2d_rz_h3a_sedov_cyl_nr{NR}_nz{NZ}_seed{SEED}"
    f"_C2{_safe_float_token(AV_C2)}_cfl{_safe_float_token(CFL)}"
    f"_ale{ALE_EVERY_N_STEPS}"
)

EV_TO_ERG = 1.6022e-12
M_P = 1.6726219e-24  # must match core::constants::proton_mass
A = 1.0
ZBAR = 1.0
GAMMA = 5.0 / 3.0
RHO_0_GCC = 1.0

R_MIN, R_MAX = 0.0, 1.0
Z_MIN, Z_MAX = -0.02, 0.02
LZ = Z_MAX - Z_MIN
DR = (R_MAX - R_MIN) / NR
R_BLAST = BLAST_CELLS * DR

P_AMBIENT = 1.0e6
cv_i = EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
cv_e = ZBAR * EV_TO_ERG / (A * M_P * (GAMMA - 1.0))
T_AMBIENT_eV = P_AMBIENT / ((GAMMA - 1.0) * RHO_0_GCC * (cv_i + cv_e))

V_PATCH = math.pi * R_BLAST**2 * LZ
E_TOTAL = E_PER_CM * LZ
E_INT_BLAST = E_TOTAL / (RHO_0_GCC * V_PATCH)
T_BLAST_eV = E_INT_BLAST / (cv_i + cv_e)
P_BLAST = (GAMMA - 1.0) * RHO_0_GCC * E_INT_BLAST

print(
    "[deck:2d_rz_h3a] "
    f"nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR} "
    f"AV_C2={AV_C2} cfl={CFL} t_end_s={T_END} "
    f"ale_every_n_steps={ALE_EVERY_N_STEPS}"
)
print(f"  R_blast={R_BLAST:.4f}cm V_patch={V_PATCH:.4e}cm^3 P_blast={P_BLAST:.4e}erg/cm^3")
print(f"  T_ambient={T_AMBIENT_eV:.4e}eV T_blast={T_BLAST_eV:.4e}eV")
print(f"  E_total={E_TOTAL:.4e}erg E_int_blast={E_INT_BLAST:.4e}erg/g")


def rho_init(r, z):
    return RHO_0_GCC


def Te_init(r, z):
    return T_BLAST_eV if r < R_BLAST else T_AMBIENT_eV


def Ti_init(r, z):
    return T_BLAST_eV if r < R_BLAST else T_AMBIENT_eV


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=T_END,
    max_steps=20000,
    seed=SEED,
    verbosity="quiet",
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

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=rho_init,
    Te=Te_init,
    Ti=Ti_init,
    velocity=velocity_init,
    radiation_field="zero",
)

numerics_kwargs = {
    "dt": dict(
        initial_s=1.0e-13,
        cfl_hydro=CFL,
        growth_factor=1.1,
        max_s=1.0e-10,
        min_s=1.0e-22,
    ),
    "hydro": dict(
        boundary_2d=dict(
            r_inner="axis",
            r_outer="free",
            z_bottom="reflect",
            z_top="reflect",
        ),
        av_C1=0.1,
        av_C2=AV_C2,
    ),
    "conduction": dict(enabled=False),
    "floors": dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
}

if ALE_EVERY_N_STEPS > 0:
    numerics_kwargs["ale"] = dict(enabled=True, every_n_steps=ALE_EVERY_N_STEPS)

Numerics(**numerics_kwargs)

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
