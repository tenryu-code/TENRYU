"""G2D-NET0: uniform DT box, burn scheme=local plumbing gate (spec docs/design/2d_burn_port_spec.md section 10 item 3)."""

from tenryu_namelist import *


def velocity_init(r, z):
    return (0.0, 0.0)


Main(
    name="burn_net0_2d",
    dimension="2D_RZ",
    t_end=2.0e-11,
    max_steps=40,
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
    rho=lambda r, z: 10.0,
    Te=lambda r, z: 1.0e4,
    Ti=lambda r, z: 1.0e4,
    velocity=velocity_init,
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
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)

Output(
    directory="build/output_burn_net0_2d",
    plot_every=20,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Burn(enabled=True, scheme="local", fuels=["DT"], fuel_materials=["DT"])
