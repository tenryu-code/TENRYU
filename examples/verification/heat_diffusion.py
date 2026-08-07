from tenryu_namelist import *
import math

L = 1.0
T0 = 10.0
AMP = 1.0
KAPPA_TEST = 1.0e10

Main(
    name="heat_diffusion",
    dimension="1D_SPH",
    t_end=2.0e-2,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=L,
    nr=50,
    grid="uniform",
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
    volfrac=dict(fuel=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: T0 + AMP * math.cos(2.0 * math.pi * r / L),
    Ti=lambda r: T0 + AMP * math.cos(2.0 * math.pi * r / L),
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    T_start_eV=1.0e9,
    dt=dict(
        initial_s=1.0e-7,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        growth_factor=1.2,
        max_s=1.0,
        min_s=1.0e-20,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
        test_kappa=KAPPA_TEST,
        test_planar=True,
    ),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_heat_diffusion",
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
