import os

from tenryu_namelist import *


NR = int(os.environ.get("TENRYU_H0_NR", "64"))
NZ = int(os.environ.get("TENRYU_H0_NZ", "128"))
SEED = int(os.environ.get("TENRYU_H0_SEED", "12345"))
OUTDIR = os.environ.get("TENRYU_H0_OUTDIR", "./build/output_verify_2d_rz_h0")

CASE_NAME = f"2d_rz_h0_geom_dry_nr{NR}_nz{NZ}_seed{SEED}"

print(f"[deck:2d_rz_h0_geom_dry] nr={NR} nz={NZ} seed={SEED} outdir={OUTDIR}")

Main(
    name=CASE_NAME,
    dimension="2D_RZ",
    t_end=2.0e-7,
    max_steps=1000,
    seed=SEED,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    z_min=-1.0,
    z_max=1.0,
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
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 1.0e-3,
    Ti=lambda r, z: 1.0e-3,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-9,
        cfl_hydro=0.1,
        growth_factor=1.05,
        max_s=1.0e-7,
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
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory=OUTDIR,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=1.0e-7,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
