"""Electron conduction with the hydrodynamics off: a Spitzer heat wave in 1D.

A hot region (2 keV) next to cold DT (20 eV) at 0.1 g/cm^3, Spitzer conductivity with the default Kirchhoff face
closure and the STS solver, flux limiter 0.06, Numerics.hydro.enabled=False, no radiation or laser. Used for the
time-step ladder of the STS face conductivity in the internal verification record §4.z5: dt.max_s from 1e-13 s to 1e-10 s against a run
at 1e-14 s, below the explicit limit. tests/hydro/conduction_hydro_off_energy_check.py runs it in 1T and 2T and
checks that the total energy stays at its initial value.

Environment:
  TENRYU_COND_HOFF_TEMPERATURE_MODEL  1T (default) or 2T
  TENRYU_COND_HOFF_GEOMETRY           planar (default) or spherical
  TENRYU_COND_HOFF_MAX_S              Numerics.dt.max_s [s] (default 1e-11)
  TENRYU_COND_HOFF_T_END_S            Main.t_end [s] (default 1e-9)
  TENRYU_COND_HOFF_NR                 cells (default 200)
"""
import math
import os

from tenryu_namelist import *

TEMPERATURE_MODEL = os.environ.get("TENRYU_COND_HOFF_TEMPERATURE_MODEL", "1T")
GEOMETRY = os.environ.get("TENRYU_COND_HOFF_GEOMETRY", "planar")
MAX_S = float(os.environ.get("TENRYU_COND_HOFF_MAX_S", "1.0e-11"))
T_END = float(os.environ.get("TENRYU_COND_HOFF_T_END_S", "1.0e-9"))
NR = int(os.environ.get("TENRYU_COND_HOFF_NR", "200"))

R_MAX = 0.05
R_HOT = 0.01
WIDTH = 0.001
T_HOT = 2000.0
T_COLD = 20.0


def te_init(r):
    return T_COLD + (T_HOT - T_COLD) * 0.5 * (1.0 - math.tanh((r - R_HOT) / WIDTH))


Main(
    name="conduction_hydro_off_1d",
    dimension="1D_SPH",
    temperature_model=TEMPERATURE_MODEL,
    t_end=T_END,
    max_steps=2000000,
    seed=1,
    verbosity="quiet",
)
Mesh(r_min=0.0, r_max=R_MAX, nr=NR, grid="uniform", geometry_1d=GEOMETRY)
Materials(
    materials=[
        Material(
            name="DT",
            A=2.5,
            Z=1.0,
            eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0 / 3.0)),
            opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g"),
        )
    ],
    zbar=dict(model="fixed", fixed_value=1.0),
)
Geometry(
    volfrac=dict(DT=lambda r: 1.0),
    rho=lambda r: 0.1,
    Te=te_init,
    Ti=te_init,
    velocity=lambda r: 0.0,
    radiation_field="zero",
)
Numerics(
    dt=dict(
        initial_s=min(1.0e-14, MAX_S),
        max_s=MAX_S,
        min_s=1.0e-22,
        growth_factor=1.2,
        cfl_hydro=0.3,
        cfl_cond=0.25,
    ),
    hydro=dict(enabled=False, boundary_1d="reflect", av_C1=0.1, av_C2=1.5),
    conduction=dict(enabled=True, solver="sts", f_lim=0.06),
    floors=dict(rho_floor_gcc=1.0e-10, Te_floor_eV=1.0, Ti_floor_eV=1.0),
)
Radiation(enabled=False)
Laser(enabled=False)
Output(
    directory="./build/output_conduction_hydro_off_1d",
    plot_every=0,
    history_every=0,
    checkpoint_every=0,
    plot_every_s=T_END,
    history_every_s=-1.0,
    checkpoint_every_s=-1.0,
)
Diagnostics(enabled=False)
