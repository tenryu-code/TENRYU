# XC-1 — Sod shock tube (TENRYU side), hydro operator + AV comparison vs MULTI-IFE.
# Design: docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md (XC-1)
# MULTI pair: ops/xc/cases/xc1_sod/multi.input (identical states, gamma=1.4 tables).
# Differences from the standing sod_planar_2t gate deck: Z=1 (electrons live, e-i
# exchange stiff-locked Te=Ti — matching MULTI, which cannot disable exchange) and
# N via env for the ladder. Reference: Toro exact Riemann (gate machinery).
import os

from tenryu_namelist import *

NR = int(os.environ.get("TENRYU_XC1_NR", "200"))

Main(
    name=f"xc1_sod_planar_nr{NR}",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0e-7,
    max_steps=500000,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=1.0, nr=NR, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="gas",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=1.4)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(gas=lambda r: 1.0),
    rho=lambda r: 1.0 if r < 0.5 else 0.125,
    Te=lambda r: 1.0 if r < 0.5 else 0.8,
    Ti=lambda r: 1.0 if r < 0.5 else 0.8,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-10, cfl_hydro=0.3, growth_factor=1.2, max_s=1.0e-7,
            min_s=1.0e-20),
    # reflect walls: matched to MULTI ileft=iright=1 rigid (free would drive a
    # vacuum rarefaction from the outer wall — sod_planar_2t deck note).
    hydro=dict(boundary_1d="reflect", av_C1=0.5, av_C2=1.5),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(enabled=False)
Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory=f"./tmp/xc/xc1_sod/tenryu_nr{NR}",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=2.0e-8,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
