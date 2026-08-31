# SPEC: short 1D probe of a GXII-class solid CD sphere (FLD)

Adapt the starting template into a PROBE deck with these exact requirements:

1. Keep the physics setup of the template (solid CD sphere R=100 um, rho0=1.05 g/cc,
   Te=Ti=1 eV initial, TMAT tables, tabular zbar, multigroup FLD radiation with
   hard-x-ray repack and hydro_coupling="none", conduction sts with f_lim=0.06,
   527 nm single-beam square-pulse laser 120 J / 6 ns, raytrace_2d).
2. Probe duration: Main.t_end = 1.0e-9 s (the pulse is still on at probe end; keep the
   square pulse definition unchanged).
3. Mesh: graded, single segment from 0 to the sphere radius with nr = 300 and
   grading edge_ratio = 0.25 (finer zones at the outer surface).
4. Main.name = "assist_demo_probe".
5. Output block: directory "outputs/assist_demo_probe", format hdf5,
   plot_every_s = 1.0e-10, history_every_s = 1.0e-9, checkpoint_every = 5000,
   checkpoint_keep_last = 2, save_namelist_copy = True, save_frozen_config = True.
6. Diagnostics: enabled, every = 1, energy_budget enabled with warn_threshold 1.0e-3.
   Drop any mc_stats / sphericity / areal_density blocks if present.
7. Numerics dt block: initial_s = 1.0e-12, max_s = 1.0e-10, cfl_hydro = 0.3,
   cfl_cond = 0.25 (keep other Numerics settings from the template).
8. Do not add any keys the template does not use unless required above.
