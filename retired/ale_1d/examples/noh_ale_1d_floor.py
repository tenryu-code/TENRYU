"""A deck that runs the retired 1D ALE (Numerics.ale1d) at 6d62bf929, the last commit that builds it.

noh_ale_1d.py with the min-width floor at 24 um, so that the floor trigger fires.

A current build refuses Numerics.ale1d.enabled=True; see ../README.md.
"""

from tenryu_namelist import *

Main(
    name="noh",
    dimension="1D_SPH",
    t_end=0.6,
    max_steps=600,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=400,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="fuel",
            A=1.0,
            Z=0.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ]
)

Geometry(
    volfrac=dict(fuel=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 1.0e-3,
    Ti=lambda r: 1.0e-3,
    velocity=lambda r: -1.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-4,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-1,
        min_s=1.0e-20,
    ),
    # The conservative (compatible) energy update, as in the verification deck.
    hydro=dict(boundary_1d="free", av_C1=0.5, av_C2=1.5, compatible_energy=True),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
    ale1d=dict(
        enabled=True,
        every_n_steps=20,
        emergency_enabled=True,
        emergency_max_dr_ratio=1.2,
        enable_benefit_gate=False,
        min_width_floor=dict(enabled=True, floor_cm=2.4e-3),
    ),
)

Output(
    directory="./output_noh_ale_1d_floor",
    format="hdf5",
    plot_every=100,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Radiation(enabled=False)
Laser(enabled=False)
Diagnostics(enabled=True)
