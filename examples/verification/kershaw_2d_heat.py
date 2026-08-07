from tenryu_namelist import *
import math

Main(
    name="kershaw_2d_heat",
    dimension="2D_RZ",
    t_end=1.0e-4,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=1.0,
    r_max=2.0,
    z_min=0.0,
    z_max=1.0,
    nr=50,
    nz=50,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(fuel=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 10.0 + math.cos(2.0 * math.pi * z),
    Ti=lambda r, z: 10.0 + math.cos(2.0 * math.pi * z),
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    T_start_eV=1.0e9,
    dt=dict(
        initial_s=1.0e-8,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        growth_factor=1.2,
        max_s=1.0,
        min_s=1.0e-20,
    ),
    hydro=dict(
        boundary_2d=dict(r_inner="axis", r_outer="free", z_bottom="reflect", z_top="reflect"),
        av_C1=0.1,
        av_C2=1.5,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
        test_kappa=2.0e14,
    ),
    ale=dict(
        enabled=False,
    ),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_kershaw_2d_heat",
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
