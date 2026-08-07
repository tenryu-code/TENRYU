from tenryu_namelist import *

from math import sqrt

um = 1.0e-4

R_OUT = 500.0 * um
SHELL_THICK = 20.0 * um
R_IN = R_OUT - SHELL_THICK

Main(
    name="P4_laser_raytrace",
    dimension="2D_RZ",
    t_end=1.0e-10,
    max_steps=100,
    seed=12345,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.1,
    z_min=-0.1,
    z_max=0.1,
    nr=500,
    nz=250,
    grid="uniform",
)

Materials(
    materials=[
        Material(
            name="target",
            A=2.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=10.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)


def rho_profile(r, z):
    rr = sqrt(r * r + z * z)
    if rr < R_IN:
        return 1.0e-4
    if rr <= R_OUT:
        frac = (R_OUT - rr) / max(SHELL_THICK, 1.0e-30)
        return max(1.0e-4, frac)
    return 1.0e-6


Geometry(
    volfrac=dict(target=lambda r, z: 1.0),
    rho=rho_profile,
    Te=lambda r, z: 1.0,
    Ti=lambda r, z: 1.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)


def _icosahedron_dirs():
    phi = (1.0 + sqrt(5.0)) / 2.0
    dirs = []
    for s1 in (-1.0, 1.0):
        for s2 in (-1.0, 1.0):
            dirs.append((0.0, s1, s2 * phi))
            dirs.append((s1, s2 * phi, 0.0))
            dirs.append((s1 * phi, 0.0, s2))
    out = []
    for x, y, z in dirs:
        n = sqrt(x * x + y * y + z * z)
        out.append((x / n, y / n, z / n))
    return out


P_TOTAL_W = 1.0e12
beams = []
for i, vec in enumerate(_icosahedron_dirs()):
    beams.append(
        LaserBeam(
            name=f"beam_{i:02d}",
            direction=(-vec[0], -vec[1], -vec[2]),
            power=lambda t_s: P_TOTAL_W / 12.0,
            f_number=3.0,
            profile=dict(model="super_gaussian", w0_um=250.0, m=2),
        )
    )

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_3d",
    rays_per_beam=5000,
    lasermesh=dict(nr=256, nz=128, r_max_factor=1.0, z_span_factor=1.0),
    raytrace=dict(test_kappa=10.0, cfl_ray=0.8, intensity_cutoff=1.0e-6),
    beams=beams,
)

Radiation(enabled=False)

Numerics(
    dt=dict(initial_s=1.0e-12, max_s=1.0e-12, min_s=1.0e-20, growth_factor=1.0),
    hydro=dict(enabled=False),
    conduction=dict(enabled=False),
)

Output(
    directory="./output_perf_P4",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
)

Diagnostics(enabled=False)
