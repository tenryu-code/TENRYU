"""Uniform ideal gas and radiation on an expanding closed radiation domain.

Infrastructure verification only. No material tables, external drive, or burn.
The gas boundary is free; both radiation boundaries reflect. Changing MODE
compares passive cell-energy transport against documented historical behavior.
"""
import os
from tenryu_namelist import *

MODE = os.environ.get('TENRYU_MESH_MOTION_MODE', 'conservative_advection')
OUT = os.environ.get('TENRYU_MESH_MOTION_OUTPUT', 'build_infra/mesh_motion_example')
Main(name='radiation_mesh_advection', dimension='1D_SPH', temperature_model='1T',
     t_end=1.005e-9, max_steps=1000, seed=12345)
Mesh(r_min=0., r_max=1., nr=32, motion='lagrangian', geometry_1d='spherical')
Materials(materials=[Material(name='gas', A=1., Z=1., eos=dict(model='ideal_gas'),
    opacity=dict(model='constant', kappa_a=1., kappa_s=0., units='cm2_per_g'))],
    zbar=dict(model='fixed', fixed_value=1.))
Geometry(rho=lambda r: 1., Te=lambda r: 10., Ti=lambda r: 10.,
         velocity=lambda r: 1.e8*r, volfrac=dict(gas=lambda r: 1.),
         radiation_field='equilibrium')
Radiation(enabled=True, mode='multigroup_diffusion', groups=1,
          multigroup_diffusion=dict(hydro_coupling=MODE,
              boundary=dict(inner_r='reflect', outer_r='reflect')))
Laser(enabled=False)
Numerics(dt=dict(initial_s=1.e-11, max_s=1.e-11, min_s=1.e-20),
         hydro=dict(boundary_1d='free', compatible_energy=True),
         conduction=dict(enabled=False),
         diagnostics=dict(phase_resolved_energy=True, conservation=dict(enabled=True)))
Output(directory=OUT, history_every=1, plot_every=10, checkpoint_every=0,
       write_final_snapshot=True)
Diagnostics(enabled=True, energy_budget=dict(enabled=True))
