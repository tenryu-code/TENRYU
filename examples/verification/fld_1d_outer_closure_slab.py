"""1D planar slab for the grid convergence of the FLD outer boundary closure (NUMERICS §6.7 1D BC, the internal verification record §23).

The 1D counterpart of examples/indirect_drive_slab_2d.py: CH-like ideal gas (A 6.5, Z 3.5), rho 1 g/cc, constant
kappa_a 100 cm^2/g (sigma 100 /cm), 8 Planck groups, 2T, Lagrangian hydro, the slab on r in [0, 0.5] cm with a
reflecting r = 0. The outer cell's optical depth is 100 * 0.5 / NR (0.78 at 64 cells).
FLD_CLOSURE_MODE=marshak (default): the outer face is a Marshak face driven by Tr(t) (0 -> 150 eV over 0.05 ns, then
  held); the slab starts at 1 eV with no radiation.
FLD_CLOSURE_MODE=vacuum: the outer face is a vacuum face; the slab starts at 150 eV with the radiation in equilibrium.
FLD_CLOSURE_NR sets the number of cells, FLD_CLOSURE_OUTDIR the output directory. The time step is capped at 2e-13 s so
every grid takes the same 1012 steps to 0.2 ns. tools/validation/fld_1d_outer_closure_ladder.py runs the ladder.
"""
import os

from tenryu_namelist import *

MODE = os.environ.get("FLD_CLOSURE_MODE", "marshak")
NR = int(os.environ.get("FLD_CLOSURE_NR", "64"))
OUT = os.environ.get("FLD_CLOSURE_OUTDIR", f"./build/output_fld_1d_outer_closure/{MODE}_nr{NR}")
T_HOT = 150.0

if MODE not in ("marshak", "vacuum"):
    raise ValueError(f"FLD_CLOSURE_MODE must be marshak or vacuum, got {MODE!r}")


def tr_drive_eV(t_s):
    t_ramp = 5.0e-11
    if t_s <= 0.0:
        return 0.0
    if t_s < t_ramp:
        return T_HOT * (t_s / t_ramp)
    return T_HOT


T0 = 1.0 if MODE == "marshak" else T_HOT

Main(
    name=f"fld_1d_outer_closure_{MODE}_nr{NR}",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=2.0e-10,
    max_steps=200000,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=0.5, nr=NR, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="slab",
            A=6.5,
            Z=3.5,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=100.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=3.5),
)

Geometry(
    volfrac=dict(slab=lambda r: 1.0),
    rho=lambda r: 1.0,
    Te=lambda r: T0,
    Ti=lambda r: T0,
    velocity=lambda r: 0.0,
    radiation_field="zero" if MODE == "marshak" else "equilibrium",
)

Numerics(
    dt=dict(initial_s=1.0e-14, max_s=2.0e-13, min_s=1.0e-20, growth_factor=1.2),
    hydro=dict(enabled=True, boundary_1d="reflect"),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=8,
    group_bounds_eV=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    multigroup_diffusion=dict(
        outer_tol=1.0e-8,
        max_outer_iterations=60,
        boundary=dict(inner_r="reflect", outer_r=MODE),
    ),
    boundary=dict(marshak_Tr=tr_drive_eV) if MODE == "marshak" else dict(),
)

Laser(enabled=False)
Burn(enabled=False)

Output(
    directory=OUT,
    format="hdf5",
    plot_every=0,
    plot_every_s=1.0e-10,
    history_every=1,
    checkpoint_every=0,
)

Diagnostics(enabled=True)
