from tenryu_namelist import *

Main(
    name="ddmc_diffusion",
    dimension="1D_SPH",
    t_end=4.0e-10,
    max_steps=20,
    seed=12345,
    verbosity="normal",
)

Mesh(
    r_min=0.0,
    r_max=2.0,
    nr=20,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="ddmc_mat",
            A=12.0,
            Z=6.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0), cv_e_override=5.488e11),
            opacity=dict(model="constant", kappa_a=50.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(ddmc_mat=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 10.0,
    Ti=lambda r: 10.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=2.0e-11, max_s=2.0e-11, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(enabled=False),
    conduction=dict(enabled=False),
)

Radiation(
    enabled=True,
    groups=1,
    group_bounds_eV=[0.0, 1.0e6],
    imc=dict(
        particles_per_cell_group=500,
        implicit_capture=True,
        cutoff_fraction=0.0,
        inelastic_scatter=True,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
        linearized_planck=True,
    ),
    ddmc=dict(enabled=True, tau_ddmc=3.0, omega_ddmc=0.9, leak_stencil="4"),
    boundary=dict(inner_r="vacuum", outer_r="marshak", marshak_Tr_eV=1000.0, marshak_particles=5000),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_ddmc_diffusion",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
