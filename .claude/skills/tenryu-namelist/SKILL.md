---
name: tenryu-namelist
description: Write or revise a complete TENRYU input deck (the Python namelist with the Main/Mesh/Materials/Geometry/Radiation/Laser/Burn/Numerics/Output/Diagnostics blocks) for a 1D_SPH laser-target or radiation-driven simulation from an experimental specification, validate it with the solver, and hand the mesh block to tenryu-mesh-1d. Use for "make a deck", "write a namelist", "set up a simulation of ...", or for repairing deck validation errors that are not mesh certificates (Claude Code variant).
metadata:
  short-description: Write a complete TENRYU 1D deck from an experimental specification
---

# TENRYU namelist (deck) authoring (work-item skill, Claude Code variant)

Scope: produce or revise a **complete, validating TENRYU deck** for one experiment
(1D_SPH: planar foil, spherical shell or solid sphere, cylindrical liner; laser-driven or
radiation-driven). Physics choices come from the specification; the solver's validator is
the authority on keys and values. The mesh block is a separate work item: design it with the
`tenryu-mesh-1d` skill (`recommend-mesh`) and paste its block. Do not edit solver source.

Claude Code notes: run `tenryu validate`, `lint-deck` and probe runs where a build of the solver
and the material tables are available (the machine that builds TENRYU may not be the one you edit
on), following the compute rules of `CLAUDE.md`. Commit decks only when asked; example decks
live under `examples/`.

## 1. What a deck is

- One Python file. First line `from tenryu_namelist import *`. Each block function is called
  once (a second call overwrites with a warning). `Material(...)` is a helper used only inside
  `Materials(materials=[...])`; the list order fixes the material indices and `materials[0]`
  must not be the void material.
- Units are cgs + eV throughout: cm, s, g/cm^3, eV, erg. Laser waveforms are power in W
  (see §3 for the intensity-to-power convention), wavelengths in nm, beam spot sizes in µm
  (`w0_um`, `radius_um`), group boundaries in eV.
- Types: `True`/`False` only (never 1/0 or "yes"), quoted strings, Python numbers, lists,
  dicts, callables (`def` or `lambda`). Unknown keys are rejected with a did-you-mean hint;
  a wrong type or range is a `ConfigError`/`ValueError` naming the key and the constraint.
- Relative file paths (EOS/opacity tables, `Main.restart_from`, `Output.directory`) resolve
  from the **working directory where `tenryu` is launched**, not from the deck's directory:
  the examples point at `TMAT-H5/...` and are run from the repository root. `~` inside a deck
  path is not expanded (it becomes a directory named `~`); write
  `os.path.expanduser("~/...")` or an absolute path instead. Environment variables are not
  expanded either.
- Initial profiles are functions of the cell-centre coordinate `r_cm` (planar decks use the
  same argument for position; the laser-side void padding lies beyond the target). The
  solver calls them once per cell centre with a scalar `r_cm` at initialisation, so they
  must accept a float and be finite everywhere on the mesh (numpy functions work if they
  accept scalars). Multi-material cells are declared with `volfrac={NAME: f(r)}` whose
  values sum to 1.
- User constants and helper functions are welcome; keep them above the blocks that use them.

## 2. Block order and the keys a typical 1D deck needs

Recommended order (the solver accepts any order): material definitions → `Main` → `Mesh` →
`Materials` → `Geometry` → `Radiation` → `Laser` → `Burn` → `Numerics` → `Output` →
`Diagnostics`. Defaults and full key lists: `docs/SPECIFICATION.md` §6.4.1–§6.4.11 (the
document map from `assist.py docmap --keys-out keys.json` lists every accepted key).

| Block | Keys you normally set | Notes |
|---|---|---|
| `Main` | `name`, `dimension="1D_SPH"`, `temperature_model="2T"`, `t_end` [s], `seed=12345`, `max_steps=10_000_000`, `verbosity="normal"` | `name` uses `[A-Za-z0-9_-]`; `t_end > 0`; `units` stays `"cgs_eV"` |
| `Mesh` | `r_min`, `r_max` [cm], `geometry_1d` (`"planar"`/`"spherical"`/`"cylindrical"`), one mesh form, `motion="lagrangian"`, `floors=dict(rho_floor_gcc, Te_floor_eV, Ti_floor_eV)` | Forms: `nr` (uniform), `grid=dict(type="graded", segments=[...], grading=...)`, `auto_regions`+`auto_zone`, `zoning_intent` (declarative; use with `resolution_requirement=dict(apply="enforce")` for laser decks). Design with `tenryu-mesh-1d` |
| `Materials` | `materials=[Material(...), ...]`, `opacity_mix_rule="linear_mass"`, `zbar=dict(model="fixed")` (each cell takes the volume-weighted `Z` of its materials, i.e. full ionization per material) or `dict(model="thomas_fermi")`, `void_config=dict(rho=1e-9, Te=1.0, Ti=1.0)` | `zbar=dict(model="fixed", fixed_value=X)` sets **every** cell to `X`, including a fill of another material (a D2 fill would get the ablator's 3.5): use it only for one-material decks. `Material(name, A, Z, eos=dict(model=...), opacity=dict(model=...), is_void)`. Table EOS/opacity: `eos=dict(model="tmat", file="TMAT-H5/<mat>.tmat.h5")`, `opacity=dict(model="tmat", file=..., lambda_method="finite_difference", lambda_fd_delta_rel=1e-4, lambda_fd_abs_min=1e-6, f_min=1e-4)`. Table-free smoke: `eos=dict(model="ideal_gas", ideal_gas=dict(gamma=5.0/3.0))`, `opacity=dict(model="constant", kappa_a=0.0, kappa_s=0.0, units="cm2_per_g")`. The void material is `Material(name="VOID", A=1.0, Z=1.0, is_void=True)` |
| `Geometry` | `rho`, `Te`, `Ti` callables of `r_cm` (values in g/cm^3 and eV), `volfrac=dict(NAME=fn, ...)`, `enforce_sum_to_one=True` | Void padding uses the `void_config` density (1e-9 g/cm^3) |
| `Radiation` | `enabled=False`, or `enabled=True, mode="multigroup_diffusion", multigroup_diffusion=dict(flux_limiter="levermore_pomraning", boundary=dict(inner_r="reflect", outer_r="vacuum")), boundary=dict(inner_r="reflect", outer_r="vacuum")`, plus `groups=dict(bounds_eV=[...])` (or `groups=N, group_bounds_eV=[...]`) only with constant/gray opacities | 1D_SPH accepts `multigroup_diffusion` (FLD) or `sn_transport`. With table opacities (`opacity=dict(model="tmat", ...)`) the solver takes the group structure from the table and silently replaces the deck's groups (one INFO line): `CD.tmat.h5`, `CD_lte`, `D2`, `DT`, `DT_nlte` and `KR_lte` have 80 groups (0.1 eV–300 keV), `CD_gray6`, `CD_gray6_bounded` and `CD_dense4x` 6, `CD_1grp` 1, so an N-group run with table opacities needs an N-group table. `ex05_kr_radiative_shock.py` keeps 24 groups with gray constant opacities; `ex07_solid_sphere.py` runs the CD table's 80 groups. Do not copy ex07's `hydro_coupling="none"` (its energy-budget closure is not physical; SPEC §6.4) — keep the default |
| `Laser` | `enabled=True`, `wavelength_nm`, `mode="radial_absorption_1d"`, `rays_per_beam=8000`, `absorption=dict(model="inverse_bremsstrahlung")`, `lasermesh=dict(..., ghost_corona=dict(enabled=True, ...))` (§3), `deposit=dict(...)`, `beams=[LaserBeam(...)]`, `cbet=dict(enable=False)`, `hot_electron=dict(enable=False)` | A target that starts as bare solid needs the ghost corona (§3): without it the laser deposits nothing. `LaserBeam(name, direction, power=<callable t_s -> W>, f_number, focus, profile=dict(model="super_gaussian", w0_um, m))`; `power`, `direction`, `f_number` are required; `energy_J` rescales the waveform to a total energy. See §3 for the power convention. `Laser(enabled=False)` when the drive is radiative |
| `Burn` | `Burn(enabled=False)` unless the specification asks for fusion burn | |
| `Numerics` | `dt=dict(initial_s=1e-13, max_s=1e-10, cfl_hydro=0.3, cfl_cond=0.25)`, `hydro=dict(boundary_1d="free", driver_full_step_retry_enabled=True)`, `conduction=dict(enabled=True, solver="implicit", f_lim=0.06)`, `positivity=dict(clamp=True)`, `safety=dict(nan_fatal=True)`, `diagnostics_every=100` | These are the minimum for a deck written from scratch. When you start from an example (§4), keep that example's whole `Numerics` block (some add artificial-viscosity and odd-even damping terms, more retry attempts or another `f_lim`) and change only what the specification requires. `boundary_1d` sets the **outer** node only (§3). The implicit conduction solver and the full-step retry are the suite conventions (low-density buffers and fine ablator cells). Never set `nan_fatal=False` |
| `Output` | `directory="outputs/<case>"`, `format="hdf5"`, `plot_every_s`, `history_every_s` [s], `checkpoint_every`, `checkpoint_keep_last=2`, `save_namelist_copy=True`, `save_frozen_config=True`; `write_final_snapshot=True` when the final state is needed; `write_final_checkpoint=True` when the run may be continued past its end | A time cadence alone (`plot_every_s` without `plot_every`, likewise history and checkpoint) replaces the step cadence; give both to write at either. `*_every_s=0.0` is an error (use -1.0 to disable); a unique directory per case |
| `Diagnostics` | `enabled=True`, `every=1`, `energy_budget=dict(enabled=True, warn_threshold=1e-3)` | Keep the energy budget on for every production deck |

## 3. Conventions that decks get wrong

- **Laser power from intensity.** Planar decks model a 1 cm^2 column: `P[W] = I[W/cm^2]`.
  Spherical targets: `P[W] = I[W/cm^2] × 4π R_target^2` with the beam aimed at the centre
  (`direction=(0.0, 0.0, -1.0)`, `focus` on the axis). In `radial_absorption_1d` the power
  enters as one inward radial flux and the beam direction, spot, focus, f-number and ray
  count do not change the absorption (NUMERICS §5.4a); with `raytrace_2d` choose a spot at
  least as wide as the target. Cylindrical liners: `P = I × 2π R × 1 cm`. State the
  convention in a deck comment.
- **Required keys an experiment never states** are taken from the example-suite conventions
  and written into the deck header instead of being asked: `f_number=3.0`,
  `rays_per_beam=8000`, `focus=(0.0, 0.0, 0.0)` for planar decks (target-centre focus on
  the axis for spheres), `profile=dict(model="super_gaussian", w0_um=250.0, m=4)` (no
  effect in `radial_absorption_1d`), the `lasermesh`/`raytrace`/`deposit` sub-blocks of
  `ex01_cd_foil_breakout.py` (below), `seed=12345`, `max_steps=10_000_000`, `verbosity="normal"`, the
  `dt`/`conduction`/`positivity`/`safety` values of §2, `checkpoint_every=5000`,
  `checkpoint_keep_last=2`. Ask only when a missing item changes the physics or the
  observables (material, density, thickness, wavelength, pulse, radiation, run duration,
  output cadence).
- **Waveform callables** take `t_s` in seconds and return W; return 0.0 outside the pulse.
  The solver freezes each callable into a linear table on absolute times `k × 2^-40 s`,
  bisecting up to 7 times where the curve bends (SPEC §6.4.6); non-finite or negative values
  are a `ConfigError`.
- **The laser needs a corona to absorb in.** `radial_absorption_1d` integrates inward from
  `r_max` and books the remaining power as unabsorbed at the first cell at or above the
  critical density (NUMERICS §5.4a). A bare solid surface is far above critical at t=0, so
  without a ghost corona (a laser-only synthetic under-dense profile outside the surface,
  NUMERICS §5.7.5; default off) nothing is ever deposited — `validate` does not detect this
  (a verification deck started from bare solid recorded exactly zero deposition). Every
  example carries it; copy the `lasermesh` block with its `ghost_corona` (and the
  `raytrace` and `deposit` blocks) of `ex01_cd_foil_breakout.py`:
  `lasermesh=dict(mesh_factor=0.1, rmax_n_hat_threshold=0.001, ghost_corona=dict(enabled=True,
  n_out=12, ne_min_frac=0.03, ne_max_frac=0.99, Te_min_eV=50.0, zbar_min=1.0, zbar_max=4.0,
  handoff_cells=6, handoff_decay=2.0, transition_enabled=True, transition_resolved_nhat=0.9,
  transition_resolved_cells=3, transition_density_exponent=1.0))`,
  `raytrace=dict(ds_adapt_g_target=0.05, ds_adapt_tau_target=0.05, ds_adapt_max_factor=2.0)`,
  `deposit=dict(deposit_smooth_passes=3, deposit_smooth_alpha=0.25)`. In the first run,
  check that the energy budget shows laser deposition from the start of the pulse.
- **Planar geometry and boundaries.** The inner node at `r_min` is held fixed (u = 0) in
  every geometry (NUMERICS §3.1.11); `hydro.boundary_1d="free"` makes only the **outer**
  (laser-side) face free. A planar target whose rear face sits at `r_min=0` therefore rests
  on a rigid wall: it behaves as the mid-plane of a target twice as thick driven from both
  sides, so the shock arrival time at the rear face is right, but the shock reflects there
  (the suite's ex01 reports a reflected-at-wall state) instead of breaking out and releasing
  into vacuum. When rear-surface release matters (free-surface velocity, rear expansion),
  put a VOID region between `r_min` and the rear face. Since main 125a94572 (2026-09-29) the
  solver keeps every run of void cells evenly spaced between the faces that bound it; before
  that only the exterior padding followed the surface, and a void inside the target collapsed
  its first cell and aborted the run within 20–140 ps. A void interval that closes (the rear
  face reaching `r_min`, two layers meeting) has no contact model: the run stops with a message
  naming the closed void cells. Make the void wider than the rear face travels during the run:
  a 25 µm CD foil at 1e14 W/cm² (351 nm) with 50 µm of VOID behind it moved its rear face
  38–41 µm in 1 ns and reached 1.3e7 cm/s (measured 2026-09-29). Do not put a low-density
  material there instead: 1e-3 g/cc CD behind the same foil slowed the rear face by 19 % at
  1 ns. With the suite's 1 eV initial temperature the CD table gives 0.36 Mbar at solid density
  (30 kbar at its lowest temperature, 0.1 eV), so a free rear face expands at 11–14 µm/ns from
  t = 0 and had moved about 8 µm before the shock arrived (~0.7 ns). Read breakout times and
  free-surface velocities with that in mind (a zero-pressure initial state was not tested).
  Put 50–100 µm of void padding on the laser side. The suite's spherical shells hold a gas
  fill, then the shell, then void up to `r_max`; a vacuum core in a shell is an interior void
  like the one above (not run). Material interfaces must coincide with mesh nodes (mesh forms
  with segments/pins do this).
- **Temperatures**: initial `Te`/`Ti` of 1 eV for cold targets is the suite convention with
  table EOS; do not start colder than the table's range.
- **Tables.** The repository checkout tracks `TMAT-H5/` (CD, D2, DT, Kr and variants); the
  beta distribution does not ship tables. `TMAT-H5/*.tmat.h5` (or SESAME files) must exist
  relative to the directory the run is launched from (or be given as absolute paths); say so
  in the deck header. Table EOS values are clamped at the table's density edge, not
  extrapolated: above the table's highest density the pressure stops rising, so material
  compressed beyond it cannot resist, cells collapse and the time step falls toward zero
  without a warning. A 10 µm CD shell driven at 1e15 W/cm² stalled at dt 8e-19 s after the
  shell passed 117 g/cc: the pressure of its cells at 300–400 g/cc no longer rose with
  density. In the same run the D2 fill's central ion temperature passed the D2 table's
  20 keV (69 keV behind the converging shock) and its ion pressure stayed ideal
  (n_i k T_i within 1.5 %). Ranges (mass density; temperature): CD 1.2e-7–117 g/cc;
  0.1 eV–5 keV (`CD_gray6`/`CD_1grp`/`CD_dense4x` to 30 keV). `D2` 3.4e-8–33.6 g/cc;
  0.1 eV–20 keV. `DT`/`DT_nlte` 4.2e-8–42.2 g/cc; 0.1 eV–20 keV. `D2_propaceos_wide`
  3.4e-7–1010 g/cc; 1e-4 eV–10 keV. `KR_lte` 1.4e-6–1400 g/cc; 0.1 eV–20 keV. Compare the
  expected peak density (shell and fuel at stagnation) with the table before running an
  implosion, and say in the deck header when it may exceed it. A missing table is not a
  clean error: `validate` and `run` abort in the table
  reader (`TMAT_E001: Failed to open TMAT file: <resolved path>`, exit 134). For a
  mesh/validation smoke without tables use the ideal-gas material form.
- **Radiation coverage note.** The mesh recommender's evidence was measured with radiation
  off; a radiation-on deck is flagged `extrapolation` and needs the two-level convergence
  pair described in `tenryu-mesh-1d`.

## 4. Procedure

1. **Intake.** From the specification collect: target layers (material, A, Z, density,
   thickness, order from the rear/centre outward), geometry and outer radius, fill gas;
   laser wavelength, pulse shape and timing, peak intensity or energy + spot; physics
   (1T/2T, radiation on/off and groups, conduction, burn); run duration; outputs
   (snapshot/history cadence, final snapshot, directory); restart. Missing items that
   change the deck are a question, not a guess (`UNCERTAIN:` in the harness).
2. **Start from the nearest example** and keep its numerics conventions:

   | Experiment | Start from |
   |---|---|
   | planar foil shock / breakout | `examples/laser_plasma_1d/ex01_cd_foil_breakout.py` |
   | double pulse, shock timing | `ex02_two_pulse_timing.py` |
   | layered target, impedance match | `ex03_impedance_match.py` |
   | thin exploding foil, burn-through | `ex04_burnthrough_tag.py` |
   | radiative shock in gas | `ex05_kr_radiative_shock.py` (24-group FLD) |
   | cylindrical liner | `ex06_cyl_liner.py` |
   | solid sphere | `ex07_solid_sphere.py` (16-group FLD) |
   | shell implosion with gas fill | `ex08_d2_shell_implosion.py`; GXII shells: `examples/implosion/gxii_shell_1D.py` |
   | corona physics (CBET, hot electrons) | `ex09_cbet_hote.py` |
   | radiation (Marshak) drive, burn | `ex10_marshak_dt.py`; `examples/templates/template_1d_indirect_tr.py` |
   | minimal complete planar deck | `tools/assist/examples/decks/planar_cd_foil_minimal.py` (§7) |

3. **Write the deck**: constants → materials → blocks in the order of §2; a header comment
   with the scientific target, the expected observable window, and the conventions used.
4. **Mesh**: run `python tools/assist/assist.py recommend-mesh --conditions <json>` (or
   `--deck <deck> --deck-out <deck>` with a placeholder mesh that has a node at every material
   interface) and paste the block; keep
   `resolution_requirement=dict(apply="enforce", ...)` and the retry companion. A pasted
   `--conditions` block carries no `motion` or `floors`: add the deck's own (the `--deck`
   route keeps them). With a binary, finish with `recommend-mesh --deck <deck> --deck-out
   <deck> --tenryu build/tenryu`: it folds in the solver's own bands and makes the block
   verifiable by `lint-deck`. Follow the `tenryu-mesh-1d` skill for flags and errors.
5. **Validate**: `./build/tenryu validate <deck> --mesh-preview`. Fix every error by class
   (§6). Then `python tools/assist/assist.py lint-deck <deck> --tenryu build/tenryu`
   (hard lints must pass) and `freeze-baseline <deck>` for the frozen configuration.
6. **Probe when the run is authorised**: a short run (`tenryu run <deck> --output-dir ...`),
   then `assist.py zoning-report <output_dir>` and the energy budget in the log; adjust the
   mesh or cadence and re-validate.
7. **Deliver** the deck plus a short note: conventions, tables required, what was validated
   (validate/lint/freeze only, or a run), and open items.

## 5. Hard contracts

- Never invent keys, values or blocks: if the validator rejects a key, look it up in
  `docs/SPECIFICATION.md` §6.4 (or the key index) and use the documented one; do not retry
  spellings blindly.
- Never change physics the specification fixed (materials, densities, pulse, wavelength,
  radiation on/off, burn) to make validation pass; ask instead. Numerics conventions may be
  taken from the examples.
- Keep every block once, the void material last and never first, all callables finite,
  `safety.nan_fatal=True`, `save_frozen_config=True`, a unique `Output.directory`.
- Mesh errors (`[mesh-zoning-intent] MESH_*`, `[mesh-requirement] MESH_RESOLUTION_REQUIREMENT_*`)
  belong to `tenryu-mesh-1d`: never weaken `resolution_requirement`, never delete injected
  bands, never edit an `empirical` dictionary by hand.
- 1D decks must not carry 2D keys (`nz`, `rectangular_rz`, `topology_scheme`, ...).
- A restart runs the original deck file, unchanged, with `tenryu run <deck.py> --restart <prefix>`.
  The checkpoint's frozen configuration includes the deck's sha256 and every key but
  `Main.restart_from`, so any edit of the deck (adding `restart_from`, a longer `t_end`, another
  output cadence, even a comment) is refused (`checkpoint frozen_config JSON mismatch`, which
  lists the differing keys). To keep the prefix in the deck, read it from the environment, e.g.
  `restart_from=os.environ.get("RESTART_FROM", "")`, in the deck of the original run too.
- To continue a run to a later end time or step limit, keep the deck unchanged and pass the new
  value: `tenryu run <deck.py> --restart <checkpoint> --t-end <s>` (or `--max-steps <N>`;
  restarts only). `Output.write_final_checkpoint=True` in the original deck keeps a checkpoint of
  the final step to continue from. `LaserBeam.energy_J` keeps normalizing over the deck's
  `t_end`, so a pulse that goes on after it adds energy in the extended interval (reported in a
  WARNING); do not edit `t_end` or `energy_J` in the deck for a continuation.

## 6. Validate errors and the fix

| Error text (abridged) | Fix |
|---|---|
| `Unknown parameter '<key>' in <Block>. Did you mean '<candidate>'?` | Use the candidate if it is the intended key; otherwise remove the key |
| `<Block>.<key> must be <type/range>` (ValueError/ConfigError) | Correct the type (bool literal, quoted string) or the value range |
| `Duplicate material name` / void material first | Rename; move the void material to the end of `materials` |
| `beams` empty / `f_number` or `power` or `direction` missing | Every `LaserBeam` needs `direction`, `power`, `f_number` |
| `Callable <name> returned non-finite` | Guard the waveform/profile for all `t` in `[0, t_end]` and all `r` on the mesh |
| `plot_every_s`/`history_every_s` `= 0.0` | Use a positive interval or -1.0 |
| `TMAT_E001: Failed to open TMAT file: <path>` (abort, exit 134) | The printed path is the resolved one: launch from the directory that holds `TMAT-H5/`, or give an absolute path (`os.path.expanduser` for `~`), or switch to the ideal-gas smoke form |
| `checkpoint files not found: <prefix>_r*.h5` | Fix `restart_from` (prefix without the rank suffix) |
| `checkpoint frozen_config JSON mismatch between checkpoint and namelist` | Restart with the original deck file unchanged and pass the prefix with `--restart`; for a later end time or step limit pass `--t-end` / `--max-steps` instead of editing the deck |
| `the checkpoint is at t=... at or after the end time ...` / `... at or beyond the step limit ...` | Pass `--t-end <later time>` / `--max-steps <larger limit>` with the restart |
| `Mesh.resolution_requirement is 1D only`, `MESH_*` certificates | Mesh work item (`tenryu-mesh-1d`) |
| `radiation.mode` rejected for 1D_SPH | Use `multigroup_diffusion` or `sn_transport` |

## 7. Minimal complete example (validated 2026-09-11, run 2026-09-28)

`tools/assist/examples/decks/planar_cd_foil_minimal.py`: a 25 µm CD foil, 351 nm square
pulse of 1e14 W/cm² for 1.2 ns, planar 1 cm² column, radiation off, the ex01 laser mesh with
its ghost corona, a uniform 0.1 µm grid (1250 cells, interface on a node), implicit
conduction, full-step retry, 10 ps snapshots.
Read it as the reference shape of a deck; its plain grid is coarser than the physics
requirement (validate reports the report-mode warning), so replace the `Mesh(...)` block
with the recommender's block for a production run. Run to 0.4 ns on an RTX 4090, it
deposits 73 % of the incident laser energy; the same deck without the ghost corona
deposited exactly nothing (all of it escaped). It needs `TMAT-H5/CD.tmat.h5` under the
directory it is launched from (the repository root: `./build/tenryu validate
tools/assist/examples/decks/planar_cd_foil_minimal.py`).
