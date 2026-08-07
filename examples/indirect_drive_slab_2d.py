"""Indirect-drive slab (2D_RZ): time-dependent hohlraum-style Tr(t) marshak drive.

A CH-like slab is driven from the z_bottom face by a blackbody radiation
temperature ramp (0 -> 150 eV over 0.05 ns, then hold). Multigroup FLD; the
Tr(t) route supplies per-group Planck weights (grey constant-flux drive would
be groups=1 only). See docs/design/2d_tr_drive_port_spec.md and NUMERICS §6.7.

Run:  ./build/tenryu run examples/indirect_drive_slab_2d.py
"""
from tenryu_namelist import *


def tr_drive_eV(t_s):
    """Ramp to 150 eV over 0.05 ns, then hold (hohlraum-style waveform)."""
    t_ramp = 5.0e-11
    t_hold = 150.0
    if t_s <= 0.0:
        return 0.0
    if t_s < t_ramp:
        return t_hold * (t_s / t_ramp)
    return t_hold


Main(
    name="indirect_drive_slab_2d",
    dimension="2D_RZ",
    temperature_model="2T",
    t_end=2.0e-10,
    max_steps=200000,
    seed=12345,
)

Mesh(
    r_min=0.0,
    r_max=0.125,
    z_min=0.0,
    z_max=0.5,
    nr=16,
    nz=64,
    grid="uniform",
    motion="lagrangian",
)

Materials(
    materials=[
        Material(
            name="slab",
            A=6.5,
            Z=3.5,
            eos=dict(
                model="ideal_gas",
                ideal_gas=dict(gamma=5.0 / 3.0),
            ),
            opacity=dict(
                model="constant",
                kappa_a=100.0,
                kappa_s=0.0,
                units="cm2_per_g",
            ),
        )
    ],
)

Geometry(
    volfrac=dict(slab=lambda r, z: 1.0),
    rho=lambda r, z: 1.0,
    Te=lambda r, z: 1.0,
    Ti=lambda r, z: 1.0,
    velocity=lambda r, z: (0.0, 0.0),
    radiation_field="zero",
)

Numerics(
    dt=dict(initial_s=1.0e-14, max_s=2.0e-13, min_s=1.0e-20, growth_factor=1.2),
    hydro=dict(
        enabled=True,
        boundary_2d=dict(
            r_inner="axis",
            r_outer="reflect",
            z_bottom="reflect",
            z_top="reflect",
        ),
    ),
    conduction=dict(enabled=False),
    ale=dict(enabled=False),
    floors=dict(rho_floor_gcc=1.0e-12, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

Radiation(
    enabled=True,
    mode="multigroup_diffusion",
    groups=8,
    group_bounds_eV=[1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0],
    imc=dict(enabled=False),
    ddmc=dict(enabled=False),
    holo=dict(enabled=False),
    multigroup_diffusion=dict(
        linear_solver_2d="cusparse_cg_jacobi",
        outer_tol=1.0e-8,
        max_outer_iterations=60,
        boundary=dict(
            inner_r="reflect",
            outer_r="reflect",
            z_bottom="marshak",
            z_top="reflect",
        ),
    ),
    # Time-dependent blackbody drive: frozen to a lookup table at init
    # (no runtime Python). Per-face tables via marshak_Tr_map are also
    # supported; the constant marshak_Tr_eV wins over this callable if both
    # are set (IMC-2D precedence).
    boundary=dict(marshak_Tr=tr_drive_eV),
)

Laser(enabled=False)

Output(
    directory="./build/indirect_drive_slab_2d",
    format="hdf5",
    plot_every=0,
    history_every=1,
    checkpoint_every=0,
    plot_every_s=5.0e-11,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
    plot_fields=["rho", "Te", "Ti", "ee", "ei", "rad_E"],
)

Diagnostics(enabled=True, every=1, energy_budget=dict(enabled=True, warn_threshold=1.0e-3))
