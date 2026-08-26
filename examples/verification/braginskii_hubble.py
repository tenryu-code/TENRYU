from tenryu_namelist import *

# Braginskii homologous-expansion null gate deck: cold spherical expansion u = H r.
# For the trace-free Braginskii stress this flow is EXACTLY stress-free in
# spherical geometry, so a constant-eta viscous run must reproduce the
# inviscid trajectory to roundoff when both runs share the same dt path.

Main(
    name="braginskii_hubble",
    dimension="1D_SPH",
    t_end=2.5e-6,
    max_steps=100000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=64,
    grid="uniform",
    geometry_1d="spherical",
)

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=0.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.6666666666666667)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 2.0e5 * r,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-8,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-7,
        min_s=1.0e-20,
    ),
    hydro=dict(boundary_1d="free", av_C1=0.0, av_C2=0.0),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_braginskii_hubble",
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
