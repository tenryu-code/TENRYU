from tenryu_namelist import *
import math

# Braginskii standing-wave null gate deck: cylindrical standing sound wave (first rigid-wall
# radial acoustic eigenmode u ~ J1(k r), kR = j_{1,1}); otherwise a
# verbatim mirror of braginskii_wave.py. See wh_braginskii_viscosity_design.md N1.

Main(
    name="braginskii_wave_cyl",
    dimension="1D_SPH",
    t_end=8.0*math.pi/(3.8317059702075123*1.0e5),
    max_steps=2000000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=1.0,
    nr=96,
    grid="uniform",
    geometry_1d="cylindrical",
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


def j1_bessel(x):
    # J1(x) by power series; converges to double precision for |x| <= 5
    term = x / 2.0
    s = term
    for m in range(1, 41):
        term *= -(x * x / 4.0) / (m * (m + 1.0))
        s += term
        if abs(term) < 1.0e-18 * max(abs(s), 1.0e-30):
            break
    return s


Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: 6.2637e-3,
    Ti=lambda r: 6.2637e-3,
    velocity=lambda r: 100.0 * j1_bessel(3.8317059702075123 * r),
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
    directory="./build/output_verify_braginskii_wave_cyl",
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
