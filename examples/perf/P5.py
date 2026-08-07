from tenryu_namelist import *

from math import log10

um = 1.0e-4
ns = 1.0e-9

R_OUT = 250.0 * um
SHELL_THICKNESS = 20.0 * um
R_IN = R_OUT - SHELL_THICKNESS

P_TOTAL_W = 7.85e10

Main(
    name="P5_icf_integrated",
    dimension="1D_SPH",
    t_end=2.0 * ns,
    max_steps=1000,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.05,
    nr=200,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="CH",
            A=6.5,
            Z=3.5,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        ),
        Material(
            name="DT",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        ),
    ],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed"),
)


def vf_dt(r):
    return 1.0 if r <= R_IN else 0.0


def vf_ch(r):
    return 1.0 if (R_IN < r <= R_OUT) else 0.0


def rho_profile(r):
    if r <= R_IN:
        return 0.010
    if r <= R_OUT:
        return 1.05
    return 1.0e-10


Geometry(
    volfrac=dict(CH=vf_ch, DT=vf_dt),
    rho=rho_profile,
    Te=lambda r: 0.025,
    Ti=lambda r: 0.025,
    velocity=lambda r: 0.0,
    radiation_field="equilibrium",
)

group_bounds = [10.0 ** (-2.0 + 4.0 * i / 16.0) for i in range(17)]

Radiation(
    enabled=True,
    groups=16,
    group_bounds_eV=group_bounds,
    imc=dict(alpha=1.0, f_max=1.0, particles_per_cell_group=31),
    ddmc=dict(enabled=True, leak_stencil="9_kershaw"),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
)

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=1000,
    beams=[
        LaserBeam(
            name="beam_00",
            direction=(0.0, 0.0, -1.0),
            power=lambda t_s: P_TOTAL_W if 0.0 <= t_s <= 1.0 * ns else 0.0,
            f_number=3.0,
            profile=dict(model="super_gaussian", w0_um=250.0, m=2),
        )
    ],
)

Numerics(
    dt=dict(initial_s=1.0e-15, max_s=1.0e-11, min_s=1.0e-20, growth_factor=1.2),
    hydro=dict(enabled=True, boundary_1d="free", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=True, solver="sts", f_lim=0.06),
)

Output(
    directory="./output_perf_P5",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
)

Diagnostics(enabled=False)
