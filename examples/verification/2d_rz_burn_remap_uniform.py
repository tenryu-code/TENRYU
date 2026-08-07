"""G2D-REMAP-A: uniform-Y species transport through the CSR conservative
remap (spec section 10 item 5a) — burn network inert (cold), remap active."""

import math

from tenryu_namelist import *


Main(
    name="burn_remap_2d_uniform",
    dimension="2D_RZ",
    t_end=4.0e-11,
    max_steps=80,
    seed=1,
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    z_min=-0.05,
    z_max=0.05,
    nr=8,
    nz=8,
    grid="uniform",
    motion="ale",
)

Materials(
    materials=[
        Material(
            name="DT",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(DT=lambda r, z: 1.0),
    rho=lambda r, z: 10.0 * (1.0 + 0.3 * math.sin(2.0 * math.pi * z / 0.1)),
    Te=lambda r, z: 5.0e1,
    Ti=lambda r, z: 5.0e1,
    velocity=lambda r, z: (0.0, 1.0e5 * math.sin(2.0 * math.pi * z / 0.1)),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-12,
        cfl_hydro=0.1,
        growth_factor=1.1,
        max_s=1.0e-12,
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
    ale=dict(enabled=True, conservative_remap_enabled=True, every_n_steps=1),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)

Output(
    directory="build/output_burn_remap_2d_uniform",
    plot_every=20,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Burn(enabled=True, scheme="local", fuels=["DT"], fuel_materials=["DT"])
