# TENRYU Post-Processing Manual — `tenryu-plot` (1D)

The standard visualization package for TENRYU 1D (`Main.dimension="1D_SPH"`)
run outputs. One CLI, nine subcommands, all reading the frozen HDF5 output
schema. Design and as-built record:
`docs/design/plot1d_standard_package_20260709.md`.

```
tools/tenryu-plot <subcommand> <run-dir> [options]          # from the repo root
PYTHONPATH=tools python3 -m tenryu_plot <subcommand> ...    # equivalent
```

- **Requirements**: python3 with `numpy`, `h5py`, `matplotlib` (no scipy, no
  GPU, no display — figures render headless via Agg). Missing packages produce
  a one-line install hint, and the ctest `tenryu_plot_selftest` SKIPs.
- **`<run-dir>`** is a run's output directory (`Output.directory`, possibly
  auto-versioned `..._001`) or its `results/` subdirectory — both work.
- **Exit codes**: 0 = success, 2 = usable input not found / bad option (one
  line on stderr, no traceback).
- Default figure output is `<run-dir>/plots/<auto-name>.png`; override with
  `-o`. `--dpi` (default 140) applies everywhere.

## 1. Producing plottable data

A run yields two independent products under `<outdir>/results/`:

| product | contains | enabled by (namelist `Output`) |
|---|---|---|
| `<case>_NNNN.h5` snapshots | full profiles (mesh, hydro, radiation, laser) at output times | `plot_every` (steps) or `plot_every_s` (seconds); **many verification decks ship with plot output OFF** |
| `<case>_history.h5` | scalar time series every history step: energy budget, conservation residuals, laser/CBET, implosion metrics, dt breakdown | `history_every` (default 1 → ON) |

To visualize a deck that has plot output disabled, **copy the deck** and set
`plot_every_s` in the copy (e.g. `t_end/20`). Do NOT add output cadence to a
certified/golden deck in place: the output timer participates in the dt
controller (`dt_output`), so changing cadence changes the step sequence and
therefore the certified trajectory. Gate-deck copies used for this manual live
in `tmp/plot1d_gate/`.

Unit conventions everywhere: cgs + eV; radii in cm (or `--xunit um`);
times displayed with an adaptive unit (s/ms/us/ns/ps) picked from the run's
scale; `summary.txt` reports canonical seconds (`*_s` keys).

## 2. Subcommands

### 2.1 `profile` — radial profiles at one time

```
tools/tenryu-plot profile RUN [RUN2 ...] [--at 1.2ns | --index 42 | --last]
    [--fields rho,Te,Ti,Tr,u,P] [--logy FIELD,...] [--window r0,r1]
    [--overlay {noh,sod,rmtv}] [--overlay-params k=v,...]
```

- One panel per field. Cell fields plot at cell centers; `u` plots node
  velocity at node coordinates. `P` shows Pe+Pi with Qvisc dashed. `Tr` is
  derived from `radiation/energy_density` (panel is dropped with a title note
  when the run has no radiation data).
- Available fields: `rho, Te, Ti, Tr, u, P, Pe, Pi, Qvisc, zbar, ee, ei,
  laser_dep`. Per-field log-y defaults exist (`rho`, pressures, `ee`/`ei`);
  `--logy` forces more.
- Multiple runs overlay on shared axes (legend = case names).
- Time selection: `--at` accepts `1.2ns`, `250ps`, `3e-9` (bare = seconds) and
  picks the nearest snapshot; `--index` is the 4-digit file index (NOT the
  step number); default `--last`.
- Vertical dotted lines mark material interfaces (`cell_material_id` changes).

### 2.2 `history` — scalar time-series dashboard

```
tools/tenryu-plot history RUN [--panels energy,conservation,laser,implosion,dt]
```

- Default = auto-detect among those five panels (absent ledger groups are
  skipped): `energy` (budget components + `conservation_error` on a log twin
  axis), `conservation` (|mass/momentum/GCL residuals|), `laser`
  (absorbed/commanded/unabsorbed power + critical-surface radius),
  `implosion` (rho_peak, shell radii, center temperature, bang-time marker),
  `dt` (chosen dt + per-physics candidate dts).

### 2.3 `spacetime` — r–t field map

```
tools/tenryu-plot spacetime RUN [--field rho] [--log] [--mesh N] [--shock]
```

- Reads every snapshot; handles the moving Lagrangian mesh exactly (per-row
  node coordinates). `--log` uses LogNorm on positive data. `--mesh N` overlays
  every Nth node trajectory (thin gray). `--shock` overlays the windowed-Qvisc
  shock-radius estimate per snapshot (black dots).

### 2.4 `summary` — one-shot standard bundle

```
tools/tenryu-plot summary RUN [-o DIR]
```

- Emits `profile_final.png`, `spacetime_rho.png`, `spacetime_Te.png`,
  `history_energy.png` (+ `history_laser.png` / `history_implosion.png` when
  present) and `summary.txt` (also printed to stdout): case, geometry,
  sizes, termination reason, final t/step, final/max conservation error,
  laser absorbed total & fraction, bang time, rho_peak max, shock-radius
  estimate. Unavailable items print `n/a` — never fatal. This is the
  recommended first look at any finished run and a building block for
  run-wrapper self-summaries (CLAUDE.md bundled-diagnostics directive).

### 2.5 `compare` — two-run A/B

```
tools/tenryu-plot compare RUN_A RUN_B [--at T | --last] [--fields ...]
```

- Per field: overlaid profiles (A solid, B dashed) at the nearest common time
  (default: min of the two final times) + a relative-difference subpanel
  ((B−A)/max|A|, B linearly interpolated onto A's grid — beware this adds its
  own O(dx) on mismatched grids; same-grid pairs are exact). When both runs
  have history files, an extra column overlays `conservation_error` and laser
  absorbed energy.
- Legend labels fall back to output-directory basenames when the two runs
  share a case name (typical for auto-versioned `_001` reruns).

### 2.6 `convergence` — resolution ladder

```
tools/tenryu-plot convergence RUN1 RUN2 [...] (--ref {noh,sod,rmtv} | --ref-run RUN)
    [--field rho] [--at T | --last]
```

- ≥2 sample runs. Error = L1/L2 norms of (field − reference) with cell-width
  weights; log-log plot vs `n_cells` with fitted order p (= −slope, positive
  for convergence) in the legend, plus a stdout table. With `--ref-run`, the
  reference run must have strictly more cells than every sample and is
  interpolated onto each sample grid.

### 2.7 `spectrum` — multigroup radiation spectrum

```
tools/tenryu-plot spectrum RUN [--at T | --index N | --last]
    [--cells i,j,... | --at-r r1,r2,...]
```

- Per-group `radiation/energy_density` vs group-center energy (geometric mean
  of `metadata/group_bounds_eV`) as log-log stairs, one line per selected cell
  (default: the max-Te cell). Exits 2 on grey runs (n_groups == 1).

### 2.8 `laser` — drive diagnostics

```
tools/tenryu-plot laser RUN [--times 0.5ns,1.0ns | --all-snapshots]
```

- Panel 1: `laser/deposited_power` vs r at selected times (default: three
  spanning snapshots). Panel 2 (when history has `laser/*`): absorbed &
  commanded power with critical-surface / absorption-weighted radii on a twin
  axis. Exits 2 when the run has no laser data at all.

## 3. Exact-reference overlays (`--overlay`, `--ref`)

| ref | defines | parameters (`--overlay-params`) | valid for |
|---|---|---|---|
| `noh` | rho, u, P | `rho0,v0,gamma` (defaults 1, 1, 5/3) | Noh decks, any 1D geometry (planar/cyl/sph read from the run) |
| `sod` | rho, u, P | `rho_l,p_l,rho_r,p_r,gamma,x0` (classic Sod defaults) | two-state at-rest shock tubes |
| `rmtv` | rho, Te only | none (frozen instance) | the `rmtv_1d` verification instance |

- Provenance: `noh` generalizes `src/verification/noh_analytic.cpp`; `sod`
  transcribes the Toro exact Riemann solver from `cmd_verify.cpp`; `rmtv`
  parses the frozen ODE table `src/verification/rmtv_reference_table.hpp` at
  runtime (single source of truth) and reproduces the C++ verify
  dimensionalization to machine precision.
- **RMtV caveat**: the E0 center-injection initial condition exists only in
  the verify driver, so a bare `tenryu run` of the rmtv deck stays cold — the
  overlay is meaningful on verify-class data. Certified similarity band:
  ξ ∈ [0.34, 2.0].
- Marshak / Su-Olson overlays are deliberately absent: the repo has no dense
  live reference for them (see the design doc §9; the live SN gates are
  scalar-probe gates).

## 4. Blessed derived quantities

Encoded once in `tools/tenryu_plot/derived.py` (project-canonical estimators):

- `Tr = (Σ_g energy_density / a_eV)^(1/4)` with `a_eV = 1.3720e2` mirroring
  `src/core/constants.hpp`.
- Shock radius = location of the (optionally windowed) `Qvisc` maximum —
  density-jump locators are known to false-lock onto advected entropy waves.
- Bang time = argmax of `implosion/rho_peak` (fallback: center temperature).
- ρR = Σ rho·dr.

## 5. Recipes

```bash
# First look at any finished run
tools/tenryu-plot summary outputs/my_run

# ON/OFF physics A/B (fresh output dirs per run, then):
tools/tenryu-plot compare outputs/case_on outputs/case_off --at 1.5ns

# Verification overlay (deck copy with plot output, then):
tools/tenryu-plot profile build/plot1d_gate_noh --last --fields rho,u,P --overlay noh

# Resolution study
tools/tenryu-plot convergence run_nr100 run_nr200 run_nr400 --ref noh --field rho

# Where is the drive going?
tools/tenryu-plot laser outputs/gxii_run
tools/tenryu-plot spectrum outputs/gxii_run --at-r 0.01,0.02
```

## 6. Troubleshooting

| symptom | cause / fix |
|---|---|
| `no snapshot files matched` | the deck ran with plot output disabled — copy the deck, set `plot_every_s` (§1) |
| history panels missing | `history_every=0` in the deck, or that ledger group is not produced by the enabled physics |
| `Tr n/a` in the title | run has no `radiation/energy_density` (radiation disabled) |
| geometry reported as `spherical (default)` | older snapshot without `geometry_1d` in the frozen config — spherical (the code default) is assumed |
| spectrum exits 2 | grey run (`n_groups==1`) — nothing to plot |
| 2D_RZ run | unsupported by design in v1 (clear error; 2D extension is future work) |
| figures look truncated in panels | increase `--dpi` or widen via fewer `--fields` |

## 7. Extending the package

- New field: add it to `FIELD_META` in `style.py` (label/unit/log default) —
  profile/compare/spacetime pick it up.
- New subcommand: `cmd/<name>.py` exposing `add_parser(subparsers)` +
  `run(args) -> int`, register in `cli.py`.
- New exact reference: `refs/<name>.py` + overlay wiring in `cmd/profile.py`
  (+ `cmd/convergence.py`) + anchors in the selftest.
- Keep `tests/tools/test_tenryu_plot_selftest.py` green — it is registered in
  ctest as `tenryu_plot_selftest` and is the schema-drift tripwire (synthetic
  HDF5, no GPU).

## 8. Legacy standalone scripts (predate the package, still functional)

| script | purpose |
|---|---|
| `tools/plot_density_rt.py` | r–t density colormap (superseded by `spacetime --field rho`) |
| `tools/plot_density_animation.py` | ρ(r,t) animation (mp4/gif) — not covered by the package |
| `tools/plot_laser_energy.py` | deposited laser energy/power vs time |
| `tools/plot_laser_rays.py` | 2D ray trajectories over n_e maps (2D runs) |
| `tools/postproc/stage28_cr_ladder_plot.py` | ICF convergence-ratio ladder from history diagnostics |
| `tools/postproc/phase2_mesh_stability_physics_critique.py` | mesh-stability critique bundle (2D) |
