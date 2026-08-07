from tenryu_namelist import *

Main(
    name="beer_lambert",
    dimension="1D_SPH",
    t_end=1.0e-10,
    max_steps=1,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.5,
    nr=100,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="absorber",
            A=12.0,
            Z=6.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=10.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(absorber=lambda r: 1.0),
    rho=lambda r: 2.0,
    Te=lambda r: 1.0,
    Ti=lambda r: 1.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-10,
        max_s=1.0e-10,
        min_s=1.0e-20,
        growth_factor=1.0,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    groups=1,
    imc=dict(
        alpha=1.0,
        f_max=1.0,
        particles_per_cell_group=100,
        implicit_capture=True,
        cutoff_fraction=0.0,
        inelastic_scatter=True,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
    ),
    ddmc=dict(enabled=False),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_beer_lambert",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
