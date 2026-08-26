# FLD (multigroup_diffusion) rebuild of the GXII 1D benchmark (SPECIFICATION §8).
# Replaces the retired IMC/DDMC deck.
from tenryu_namelist import *

from math import log10, sqrt

try:
    import numpy as np
except ImportError:
    class _NumpyFallback:
        @staticmethod
        def sqrt(x):
            return sqrt(x)

        @staticmethod
        def where(cond, a, b):
            return a if cond else b

        @staticmethod
        def logspace(start, stop, num):
            if num <= 1:
                return [10.0 ** start]
            step = (stop - start) / (num - 1)
            return [10.0 ** (start + i * step) for i in range(num)]

    np = _NumpyFallback()

um = 1.0e-4
ns = 1.0e-9

R_OUT = 250.0 * um
SHELL_THICKNESS = 20.0 * um
R_IN = R_OUT - SHELL_THICKNESS
DOMAIN_R_MAX = 500.0 * um

# Pre-plasma seed corona (SPECIFICATION §8.1 note): an exponential blowoff ramp
# rho = RHO_CORONA0 * exp(-(r-R_OUT)/L_CORONA), floored at RHO_CORONA_MIN, out
# to R_CORONA_END. Without it the cold solid edge is a single-cell density
# step: the 351 nm rays reflect at critical with ~zero inverse-bremsstrahlung
# path (measured absorbed fraction 4e-4 over the full 1 ns pulse) and no
# corona can bootstrap. The tail floor 1e-4 g/cc keeps the laser-loaded
# outermost seed cells from being effectively massless, and the ramp stays
# strictly monotone with NO uniform floor plateau: a 3-cell rho=1e-4 plateau
# next to void formed a resonant cavity where trace IB deposition drove an
# exponentially growing antisymmetric node pair (laser-off runs are stable;
# tracked as a tail artifact). (A 1e-7 tail let the
# first heated cells accelerate their nodes into crossing within ~1 ps).
# The seed adds <1% of shell mass and places
# the critical surface (0.028 g/cc for CH at 351 nm) inside a resolved ramp.
L_CORONA = 2.0 * um
R_CORONA_END = R_OUT + 10.0 * um
RHO_CORONA0 = 0.05
RHO_CORONA_MIN = 3.0e-4

PULSE_ON = 0.0
PULSE_RAMP = 10.0e-12
PULSE_OFF = 1.0 * ns
# SPECIFICATION §8.2: E_total = 10 kJ over a 1 ns square pulse -> P_total = 1e13 W.
# (The retired legacy deck carried a 1e6 W placeholder that the 2 ps CI cap
# never exposed; the FLD rebuild restores the benchmark drive.)
P_TOTAL_W = 1.0e13


def pulse_power(t_s):
    if t_s < PULSE_ON or t_s > PULSE_OFF:
        return 0.0
    if t_s < PULSE_ON + PULSE_RAMP:
        return P_TOTAL_W * (t_s - PULSE_ON) / PULSE_RAMP
    if t_s > PULSE_OFF - PULSE_RAMP:
        return P_TOTAL_W * (PULSE_OFF - t_s) / PULSE_RAMP
    return P_TOTAL_W


mat_ch = Material(
    name="CH",
    A=6.5,
    Z=3.5,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
)
mat_dt = Material(
    name="DT",
    A=2.5,
    Z=1.0,
    eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
    opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
)
# SPECIFICATION §8.1 exterior vacuum: floor-density fill MASKED as void so the
# Spitzer conduction D_eff ~ T^{5/2}/rho cannot diverge in laser-heated floor
# cells (kappa_eff = 0 in void cells; the legacy rho-fill approach stalled the
# full-t_end run with STS n_sub ~ 7.7e4).
mat_void = Material(name="VOID", A=1.0, Z=1.0, is_void=True)

Main(
    name="gxii_1d_fld_regression",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0 * ns,
    seed=12345,
    max_steps=10000000,
    verbosity="quiet",
)

Mesh(
    r_min=0.0,
    r_max=0.05,
    nr=200,
    grid="uniform",
    motion="lagrangian",
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Materials(
    materials=[mat_ch, mat_dt, mat_void],
    opacity_mix_rule="linear_mass",
    zbar=dict(model="fixed"),
    void_config=dict(rho=1.0e-10, Te=1.0e-3, Ti=1.0e-3),
)


def _radius(r_cm, z_cm=0.0):
    return np.sqrt(r_cm * r_cm + z_cm * z_cm)


def vf_dt(r_cm, z_cm=0.0):
    return np.where(_radius(r_cm, z_cm) <= R_IN, 1.0, 0.0)


def vf_ch(r_cm, z_cm=0.0):
    rr = _radius(r_cm, z_cm)
    return np.where((rr > R_IN) & (rr <= R_CORONA_END), 1.0, 0.0)


def vf_void(r_cm, z_cm=0.0):
    return np.where(_radius(r_cm, z_cm) > R_CORONA_END, 1.0, 0.0)


def rho_profile(r_cm, z_cm=0.0):
    rr = _radius(r_cm, z_cm)
    if rr <= R_IN:
        return 0.010
    if rr <= R_OUT:
        return 1.05
    if rr <= R_CORONA_END:
        from math import exp
        return max(RHO_CORONA0 * exp(-(rr - R_OUT) / L_CORONA), RHO_CORONA_MIN)
    return 1.0e-10


def Te_profile(r_cm, z_cm=0.0):
    return 0.025


def Ti_profile(r_cm, z_cm=0.0):
    return 0.025


Geometry(
    rho=rho_profile,
    Te=Te_profile,
    Ti=Ti_profile,
    volfrac=dict(CH=vf_ch, DT=vf_dt, VOID=vf_void),
    radiation_field="equilibrium",
    enforce_sum_to_one=True,
)

GROUPS = 20
group_bounds = list(np.logspace(log10(0.01), log10(100.0), GROUPS + 1))

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=GROUPS,
    group_bounds_eV=group_bounds,
    compute_T_range_eV=[0.01, 10000.0],
    multigroup_diffusion=dict(
        flux_limiter="levermore_pomraning",
        max_outer_iterations=20,
        outer_tol=1.0e-5,
    ),
    boundary=dict(inner_r="reflect", outer_r="vacuum"),
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
        norm = sqrt(x * x + y * y + z * z)
        out.append((x / norm, y / norm, z / norm))
    return out


beams = []
for i, vec in enumerate(_icosahedron_dirs()):
    beams.append(
        LaserBeam(
            name=f"beam_{i:02d}",
            direction=(-vec[0], -vec[1], -vec[2]),
            power=lambda t_s, p=pulse_power: p(t_s) / 12.0,
            f_number=3.0,
            profile=dict(model="super_gaussian", w0_um=200.0, m=4),
        )
    )

Laser(
    enabled=True,
    wavelength_nm=351.0,
    mode="raytrace_2d",
    rays_per_beam=1000,
    beams=beams,
)

Numerics(
    dt=dict(
        initial_s=1.0e-15,
        growth_factor=1.2,
        max_s=1.0e-11,
        min_s=1.0e-20,
        cfl_hydro=0.3,
        cfl_cond=0.25,
        f_min_fleck=0.01,
    ),
    # driver_full_step_retry: the rebound shock re-compresses the gas/shell
    # interface (cell ~91) hard enough to cross nodes within one step at the
    # post-bang dt; the retry restores the snapshot and re-runs at dt/2
    # (1D soft-fail plumbing added with the W-B FLD conservation fixes).
    hydro=dict(
        boundary_1d="free",
        av_C1=0.1,
        av_C2=1.5,
        driver_full_step_retry_enabled=True,
    ),
    conduction=dict(
        enabled=True,
        solver="sts",
        f_lim=0.06,
    ),
)

Output(
    directory="./build/output_verify_gxii_1d_regression",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=-1.0,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(
    enabled=True,
    every=1,
    areal_density=dict(enabled=True, angles_deg=[0.0], r_range="shell"),
    sphericity=dict(enabled=True, rho_threshold=10.0, modes=[0, 2, 4]),
)
