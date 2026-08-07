from tenryu_namelist import *

Main(
    name="negative_temp_guard",
    dimension="1D_SPH",
    t_end=1.0e-8,
    max_steps=1,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=256,
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
    Te=lambda r: 100.0 if r < 0.5 else 1.0e-6,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    T_start_eV=1.0e9,
    dt=dict(
        initial_s=1.0e-8,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        growth_factor=1.0,
        max_s=1.0e-8,
        min_s=1.0e-20,
    ),
    hydro=dict(boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
        sts_damping=0.01,
        sts_max_stages=40,
    ),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_negative_temp_guard",
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
