from math import exp

from tenryu_namelist import *

Main(
    name="laser_skip_consistency",
    dimension="1D_SPH",
    t_end=2.0e-12,
    max_steps=20,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    nr=100,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="plasma",
            A=2.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(plasma=lambda r: 1.0),
    # VERIFICATION §6.11 corona profile:
    # rho(r)=rho0*exp(-(r-R_crit)/H), rho0=1.0 g/cc, R_crit=0.025 cm, H=50 um.
    rho=lambda r: exp(-(r - 0.025) / (50.0e-4)),
    Te=lambda r: 100.0,
    Ti=lambda r: 100.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-13, max_s=1.0e-13, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=128,
    absorption=dict(eps_n=1.0e-4, coulomb_log_floor=2.0),
    raytrace=dict(intensity_cutoff=1.0e-6, max_steps=100000),
    lasermesh=dict(nr=32, nz=64, critical_margin=0.9999),
    raytrace_skip_config=dict(
        enabled=True,
        threshold=0.05,
        max_consecutive=10,
        norm="max_relative",
        crit_guard=0.01,
    ),
    beams=[
        dict(
            name="b0",
            direction=[0.0, 0.0, -1.0],
            f_number=3.0,
            defocus_DR=0.0,
            profile=dict(model="gaussian", w0_um=100.0, m=2),
            power=lambda t: 1.0e5,
        )
    ],
)

Output(
    directory="./build/output_verify_laser_skip_consistency",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
