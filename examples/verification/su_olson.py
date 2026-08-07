from tenryu_namelist import *

Main(
    name="su_olson",
    dimension="1D_SPH",
    t_end=3.33e-10,
    max_steps=100,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=1.0e4,
    r_max=1.002e4,
    nr=800,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="hohlraum",
            A=12.0,
            Z=6.0,
            eos=dict(
                model="ideal_gas",
                cv_e_override=5.488e2,
                ideal_gas=dict(gamma=5.0 / 3.0),
            ),
            opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(hohlraum=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=3.33e-12,
        max_s=3.33e-12,
        min_s=1.0e-20,
        growth_factor=1.0,
        f_min_fleck=0.01,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    volume_source_rate=3.2928e12,
    volume_source_x_max=1.00005e4,
    imc=dict(
        alpha=1.0,
        f_max=1.0,
        linearized_planck=True,
        particles_per_cell_group=250,
        implicit_capture=True,
        cutoff_fraction=1.0e-4,
        inelastic_scatter=True,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
    ),
    ddmc=dict(enabled=False),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_su_olson",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
