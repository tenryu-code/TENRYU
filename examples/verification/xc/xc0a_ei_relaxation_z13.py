# XC-0a leg z1 — electron-ion relaxation coefficient probe (TENRYU side).
# Cross-comparison vs MULTI-IFE per docs/design/xc_tenryu_multiife_1d_comparison_design_20260710.md
# MULTI pair: ops/xc/cases/xc0a_z13/ (identical rho/Te/Ti/A/Z, ihydro=0, iradia=0, iheation=0).
# Uniform state => conduction and hydro inert by construction (hydro must stay ENABLED:
# TENRYU applies Q_ei inside the hydro step — see Numerics comment). Deliverable:
# nu_ei_eff(t) from d(Te-Ti)/dt, ratio vs MULTI.
from tenryu_namelist import *

A = 27.0
Z_BAR = 13.0
RHO0 = 0.1     # g/cc
TE0 = 1000.0   # eV
TI0 = 100.0    # eV
GAMMA = 5.0 / 3.0

# NRL relaxation time (TENRYU-side convention, eos_device.cuh:88-108) for scaling only.
import math
m_p = 1.6726219e-24
n_i = RHO0 / (A * m_p)
n_e = Z_BAR * n_i
if TE0 >= 10.0 * Z_BAR * Z_BAR:
    ln_lambda = max(2.0, 24.0 - 0.5 * math.log(n_e) + math.log(TE0))
else:
    ln_lambda = max(2.0, 23.0 - 0.5 * math.log(n_e) - math.log(Z_BAR) + 1.5 * math.log(TE0))
TAU_EQ = 3.16e8 * A * (TE0 ** 1.5) / (Z_BAR * Z_BAR * n_i * ln_lambda)

Main(
    name="xc0a_ei_relaxation_z13",
    dimension="1D_SPH",
    temperature_model="2T",          # explicit (XC rule: never rely on defaults)
    # (1+Z)=14x faster decay than tau: short window, fine cadence
    t_end=0.5 * TAU_EQ,
    max_steps=1_000_000,
    seed=12345,
    verbosity="quiet",
)

Mesh(r_min=0.0, r_max=1.0, nr=20, grid="uniform", geometry_1d="planar")

Materials(
    materials=[
        Material(
            name="alu",
            A=A,
            Z=Z_BAR,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=GAMMA)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=Z_BAR),
)

Geometry(
    volfrac=dict(alu=lambda r: 1.0),
    rho=lambda r: RHO0,
    Te=lambda r: TE0,
    Ti=lambda r: TI0,
    velocity=lambda r: 0.0,
    radiation_field="zero",          # match MULTI zero-field start (Addendum 1.C)
)

Numerics(
    # growth_factor MUST be > 1 here: with time-based outputs enabled, a one-off
    # FP output-boundary clip (dt_output -> ~2e-20) is otherwise unrecoverable and
    # the run Zeno-grinds at min_s (observed 2026-07-10, design doc Addendum 2).
    dt=dict(initial_s=0.01 * TAU_EQ, max_s=0.01 * TAU_EQ, growth_factor=1.2, min_s=1.0e-20),
    # hydro ON: TENRYU's Q_ei transfer executes INSIDE the hydro step
    # (hydro_1d.cu apply_qei_transfer_2t_kernel) — hydro.enabled=False silently
    # disables the exchange (observed: delta frozen at 1800 eV for the full run,
    # design doc Addendum 2). The uniform state keeps hydro inert physically
    # (zero gradients), which is the same isolation MULTI gets from ihydro=0.
    hydro=dict(enabled=True, boundary_1d="reflect"),
    conduction=dict(enabled=False),   # uniform state: inert anyway; keep OFF for parity
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0e-3, Ti_floor_eV=1.0e-3),
)

# XC explicit-OFF block (design doc S1.1): assert TENRYU-only physics disabled.
Radiation(enabled=False)
Laser(enabled=False, cbet=dict(enable=False), hot_electron=dict(enable=False))
Burn(enabled=False)

Output(
    directory="./tmp/xc/xc0a_z13/tenryu",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=0.0025 * TAU_EQ,    # ~200 snapshots across the (1+Z)-fast decay
    history_every_s=0.0025 * TAU_EQ,
    checkpoint_every_s=-1.0,
)

Diagnostics(enabled=True)
