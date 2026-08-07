# xc2b_streaming — XC-2b closure envelope: optically thin gap + thick wall (TENRYU side).
# MULTI pair: ops/xc/cases/xc2b_streaming/multi.input.
# Design: docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md §XC-2b,
# promoted after the Addendum-5 finding (MULTI two-stream D = c/4kappa*rho).
#
# Constant Tr = 200 eV Marshak drive at the OUTER (right) boundary, transparent
# gap (tau ~ 4e-7) in front of a thick wall (tau ~ 40, mfp ~ 2.5 cells): the gap
# measures the STREAMING closure delta, the wall Marshak front the DIFFUSION one.
# 1T on purpose: MULTI's live kei stiff-locks Te=Ti in the wall (tau_ei ~ 0.065 ns
# << 3 ns), while TENRYU's exchange is hydro-off-dead (Addendum 2 finding 3) —
# 1T's cv_total matches the locked limit and removes the asymmetry.
# Mode switch: TENRYU_XC2B_MODE = "fld" (default) | "sn" (S16 referee).
import os

from tenryu_namelist import *

MODE = os.environ.get("TENRYU_XC2B_MODE", "fld")
outdir = os.environ.get("TENRYU_XC2B_OUTDIR", f"tmp/xc/xc2b_streaming/tenryu_{MODE}")

WALL_X = 0.2    # cm: wall occupies [0, 0.2], gap [0.2, 0.6], drive at r=0.6
RHO_WALL = 2.0  # g/cc (A=27: heated depth ~0.12 cm at 3 ns, inside the wall)
RHO_GAP = 1.0e-6

Main(
    name=f"xc2b_streaming_{MODE}",
    dimension="1D_SPH",
    temperature_model="1T",
    t_end=3.0e-9,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=0.6, nr=200, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="wall",
            A=27.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0,
                         units="cm2_per_g"),
        ),
        Material(
            name="gap",
            A=1.0,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=1.0, kappa_s=0.0,
                         units="cm2_per_g"),
        ),
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(
        wall=lambda r: 1.0 if r < WALL_X else 0.0,
        gap=lambda r: 0.0 if r < WALL_X else 1.0,
    ),
    rho=lambda r: RHO_WALL if r < WALL_X else RHO_GAP,
    Te=lambda r: 1.0,
    Ti=lambda r: 1.0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-14, max_s=5.0e-13, growth_factor=1.2, min_s=1.0e-20),
    hydro=dict(enabled=False, boundary_1d="reflect"),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

if MODE == "sn":
    Radiation(
        enabled=True,
        mode="sn_transport",
        groups=1,
        group_bounds_eV=[0.1, 1.0e5],
        compute_T_range_eV=[0.1, 1.0e5],
        sn_transport=dict(
            boundary=dict(inner_r="reflect_parity", outer_r="marshak"),
        ),
        boundary=dict(
            marshak_Tr=lambda t: 200.0,
        ),
    )
else:
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
            boundary=dict(inner_r="reflect", outer_r="marshak"),
        ),
        boundary=dict(
            marshak_Tr=lambda t: 200.0,
        ),
    )

Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory=outdir,
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=1.0e-10,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
