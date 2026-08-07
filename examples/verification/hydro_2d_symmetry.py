from tenryu_namelist import *

Main(
    name="hydro_2d_symmetry",
    dimension="2D_RZ",
    t_end=0.005,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=2.0,
    z_min=-2.0,
    z_max=2.0,
    nr=100,
    nz=100,
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
        initial_s=1.0e-7,
        cfl_hydro=0.01,
        growth_factor=1.2,
        max_s=1.0e-1,
        min_s=1.0e-20,
    ),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="free", z_top="free"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_hydro_2d_symmetry",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
