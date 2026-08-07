from tenryu_namelist import *

import math

Main(name="geometry_eval", dimension="1D_SPH", t_end=1e-15, max_steps=1)

Mesh(r_min=0.0, r_max=0.01, nr=20, grid="uniform")

Materials(
    materials=[
        Material(
            name="fuel",
            A=2.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(
                model="constant",
                kappa_a=0.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        )
    ]
)


def rho_func(r):
    L = 0.01
    return 1.0 + 0.5 * math.sin(2.0 * math.pi * r / L)


Geometry(
    volfrac=dict(fuel=lambda r: 1.0),
    rho=rho_func,
    Te=lambda r: 10.0,
    Ti=lambda r: 5.0,
    velocity=None,
    radiation_field="zero",
)

Output(directory="./output_geom_test")

Radiation(enabled=False)
Laser(enabled=False)
