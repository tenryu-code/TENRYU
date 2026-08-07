# TMAT multigroup-opacity probe: uniform DT box, 20-group FLD wired to the exact
# MULTI group bounds [1,5000] eV carried by the converted tables. Gate: per-group
# chi = kappa*rho in the snapshot matches the MULTI mirror opacity evaluation.
# (tmp/xc/tmat/DT_multi.h5 from ops/xc/convert_multi_tables.py).
# Gate: TENRYU loads the file, runs a few steps, and the snapshot P_e/P_i/Te at the
# initial state match the MULTI mirror evaluation at (rho0, T0) — the end-to-end
# same-EOS chain check (design doc same-tables directive, 2026-07-10).
from tenryu_namelist import *

RHO0 = 0.25     # g/cc (DT ice-ish; inside table range)
TE0 = 100.0     # eV
TI0 = 100.0

Main(
    name="xc_tmat_probe_dt_mg",
    dimension="1D_SPH",
    temperature_model="2T",
    t_end=1.0e-13,
    max_steps=20,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=1.0, nr=16, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="DT",
            A=2.5,
            Z=1.0,
            eos=dict(model="tmat", file="tmp/xc/tmat/DT_multi.h5"),
            opacity=dict(model="tmat", file="tmp/xc/tmat/DT_multi.h5"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)

Geometry(
    volfrac=dict(DT=lambda r: 1.0),
    rho=lambda r: RHO0,
    Te=lambda r: TE0,
    Ti=lambda r: TI0,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=5.0e-15, max_s=5.0e-15, growth_factor=1.2, min_s=1.0e-22),
    hydro=dict(enabled=True, boundary_1d="reflect"),
    conduction=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=20,
    group_bounds_eV=[1.0, 10.0, 50.0, 100.0, 150.0, 250.0, 400.0, 550.0, 700.0, 900.0,
                     1200.0, 1600.0, 2000.0, 2200.0, 2400.0, 2600.0, 2800.0, 3100.0,
                     3500.0, 4000.0, 5000.0],
    compute_T_range_eV=[0.1, 1.0e5],
    multigroup_diffusion=dict(
        opacity_floor=0.0,
        opacity_cap=1.0e30,
        boundary=dict(inner_r="reflect", outer_r="reflect"),
    ),
)
Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory="./tmp/xc/tmat_probe_dt_mg",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=5.0e-14,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
