from tenryu_namelist import *
import math

# Braginskii standing-wave null gate deck: spherical standing sound wave (first rigid-wall
# radial acoustic eigenmode u ~ j1(k r), kR = x1: tan x = x); otherwise a
# verbatim mirror of braginskii_wave.py. See wh_braginskii_viscosity_design.md N1.

Main(
    name="braginskii_wave_sph",
    dimension="1D_SPH",
    t_end=8.0*math.pi/(4.4934094579090642*1.0e5),
    max_steps=2000000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=96,
    grid="uniform",
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


def j1_spherical(x):
    if abs(x) < 1.0e-6:
        return x / 3.0 - x**3 / 30.0
    return math.sin(x) / (x * x) - math.cos(x) / x


Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 6.2637e-3,
    Ti=lambda r: 6.2637e-3,
    velocity=lambda r: 100.0 * j1_spherical(4.4934094579090642 * r),
    radiation_field="zero",
)

Numerics(
    dt=dict(
        initial_s=1.0e-9,
        cfl_hydro=0.3,
        growth_factor=1.2,
        max_s=1.0e-8,
        min_s=1.0e-20,
    ),
    # av_C1=0: the LINEAR AV term damps linear sound waves and would bias
    # the decay-rate fit. av_C2=1.5 (default-scale quadratic): inert for
    # smooth waves (q/p ~ 1e-9 at Mach 1e-3) but suppresses the odd-even
    # sawtooth instability of the AV-less staggered scheme; its residual
    # damping cancels in the viscous-minus-inviscid differential.
    hydro=dict(boundary_1d="reflect", av_C1=0.0, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Output(
    directory="./build/output_verify_braginskii_wave_sph",
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
