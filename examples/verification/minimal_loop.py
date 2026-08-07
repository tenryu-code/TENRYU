from tenryu_namelist import *

Main(
    name="minimal_loop",
    dimension="1D_SPH",
    t_end=1e-6,
    max_steps=100,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=0.05,
    nr=10,
    grid="uniform",
)

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
        ),
    ],
)


def rho_init(r_cm):
    return 1.0


Geometry(
    volfrac=dict(fuel=lambda r_cm: 1.0),
    rho=rho_init,
    Te=lambda r_cm: 1.0,
    Ti=lambda r_cm: 1.0,
    velocity=None,
    radiation_field="zero",
)

Output(
    directory="./output_minimal",
    plot_every=10,
)

Radiation(enabled=False)
Laser(enabled=False)
