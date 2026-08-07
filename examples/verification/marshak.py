from tenryu_namelist import *

Main(
    name="marshak",
    dimension="1D_SPH",
    t_end=5.0e-10,
    seed=42,
    max_steps=100,
    verbosity="quiet",
)

Mesh(
    r_min=100.0,
    r_max=104.0,
    nr=200,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fc_marshak",
            A=12.0,
            Z=6.0,
            eos=dict(
                model="ideal_gas",
                cv_e_override=5.488e11,
                eos_T_ref_eV=1000.0,
                ideal_gas=dict(gamma=5.0 / 3.0),
            ),
            opacity=dict(model="freq_dep_marshak", kappa_a=0.0, kappa_s=0.0),
        )
    ],
    zbar=dict(model="fixed", fixed_value=6.0),
)

Geometry(
    volfrac=dict(fc_marshak=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 10.0,
    Ti=lambda r: 10.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=8.33e-12, max_s=8.33e-12, f_min_fleck=1.0e-6),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0, Ti_floor_eV=1.0),
)

Radiation(
    enabled=True,
    groups=1,
    group_bounds_eV="log_uniform",
    compute_T_range_eV=[1.0, 30000.0],
    imc=dict(
        alpha=1.0,
        f_max=1.0,
        particles_per_cell_group=100,
        implicit_capture=True,
        cutoff_fraction=1.0e-3,
        inelastic_scatter=False,
        weight_cutoff=1.0e-10,
        roulette_survival=0.1,
        linearized_planck=False,
    ),
    ddmc=dict(enabled=False),
    boundary=dict(
        inner_r="vacuum",
        outer_r="marshak",
        marshak_Tr_eV=1000.0,
        marshak_particles=50000,
    ),
)

Laser(enabled=False)

Output(
    directory="./build/output_verify_marshak",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
