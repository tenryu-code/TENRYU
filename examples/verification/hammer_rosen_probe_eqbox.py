# Hammer-Rosen supersonic nonlinear Marshak wave deck.
# Paper case: J.H. Hammer and M.D. Rosen, Phys. Plasmas 10, 1829 (2003),
# Eq. 31 material fit with the Fig. 2a radiation-temperature drive; this is
# the paper's Fig. 2 HYDRA comparison configuration.
# Design: docs/design/hammer_rosen_supersonic_gate_20260710.md.
#
# Ion/electron isolation is by parameter choice, not a qei switch: A=1.0e5 and
# fixed zbar=1.0 suppress cv_i and the qei rate while leaving the power-law
# opacity/EOS, which do not use A or Z, untouched.
from tenryu_namelist import *


Main(
    name="hammer_rosen_probe_eqbox",
    dimension="1D_SPH",
    t_end=2.0e-10,
    seed=314159,
    max_steps=500000,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.06,
    nr=400,
    grid="uniform",
    geometry_1d="planar",
)

Materials(
    materials=[
        Material(
            name="au_fit",
            A=1.0e5,
            Z=1.0,
            eos=dict(
                model="power_law_te",
                f_erg_g=2.14526e10,
                beta=1.6,
                mu_rho=0.14,
            ),
            opacity=dict(model="constant", kappa_a=2840.0, kappa_s=0.0),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(au_fit=lambda r: 1.0),
    rho=lambda r: 0.2,
    Te=lambda r: 200.0,
    Ti=lambda r: 200.0,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
)

Numerics(
    dt=dict(
        initial_s=1.0e-14,
        max_s=2.0e-12,
        min_s=1.0e-22,
        growth_factor=1.2,
    ),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0, Ti_floor_eV=1.0),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=1,
    group_bounds_eV=[0.1, 1.0e5],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e8,
        fleck_cv_source="table",
        boundary=dict(inner_r="reflect", outer_r="reflect"),
    ),
)

Laser(enabled=False)

Output(
    directory="outputs/hammer_rosen_probe_eqbox",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=1.0e-11,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
