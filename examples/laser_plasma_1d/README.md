# 1D laser-plasma example suite (EX-01 .. EX-10)

Ten self-contained 1D_SPH decks modeling typical laser-plasma experimental
setups: foil shocks, impedance matching, burn-through, a radiative shock,
cylindrical and spherical compressions, a shell implosion, CBET/hot-electron
corona physics, and a Marshak-driven DT exploding pusher. The physical
designs were produced by an external design pass and the decks were
generated/tuned through the `tools/assist` deck-generation loop, then
commissioned run-by-run; the metrics below are measured from those runs.

Run any deck with:

    ./build/tenryu run examples/laser_plasma_1d/ex01_cd_foil_breakout.py

Outputs land in `outputs/<deck name>/`.

## Suite conventions

- Buffers/exteriors are VOID material regions (`is_void=True`, uniform cell
  widths); planar targets put the rear solid face exactly at x=0 (the
  `boundary_1d="free"` face is the release surface — no rear-side void).
- Conduction uses the implicit solver in every deck (low-density buffers
  make explicit/STS stage counts explode).
- Every deck carries `driver_full_step_retry_enabled=True`,
  `driver_full_step_retry_max_attempts=8`, and the artificial-viscosity /
  odd-even damping set (`av_heat_C=0.5`, `av_heat_to="ion"`,
  `odd_even_damping_C=1.0`, `ee_odd_even_C=0.15`).
- Multi-material radiative decks use per-material CONSTANT opacities
  (volume-fraction-mixed per cell). Table opacity (`tmat`) in a
  multi-material deck is rejected by namelist validation (Phase-1
  table-NLTE runtime is single-material); the per-material table-opacity
  upgrade is designed in
  `docs/design/multimaterial_table_opacity_20260829.md`. Table EOS
  (`eos.model="tmat"`) is fine in any deck.

## Examples and measured commissioning metrics

| deck | setup | status / key measured metrics |
|---|---|---|
| ex01_cd_foil_breakout | 50 um CD foil, 351 nm, 1e14 W/cm², shock breakout | complete (3 ns): shock 46.7 um/ns, breakout 1.13 ns, ~83 Mbar reflected-at-wall state |
| ex02_two_pulse_timing | double pulse, shock merger timing | complete: merger and breakout inside design windows |
| ex03_impedance_match | layered target, transit through witness | complete: witness transit 1.95 ns (in window) |
| ex04_burnthrough_tag | thin exploding foil, burn-through tag | complete: tag time 0.76 ns (in window) |
| ex05_kr_radiative_shock | CD piston into 0.04 g/cc Kr, 800 um column, 24-group FLD, 18 ns | complete: material shock 57 um/ns (design window 70–120 — the gray CD opacity radiates drive energy away; see caveats), witness arrival 13.4–13.5 ns, peak witness compression 1.59 g/cc, reflected re-compression after arrival; radiative precursor exists but is short (~10–50 um mid-run) under the gray Kr constant |
| ex06_cyl_liner | cylindrical CD liner onto D2 column | deck validated; requires a user-provided SESAME ASCII library at `SESAME/xsesame_short` (material 5263 = D2) — not shipped with the repository |
| ex07_solid_sphere | solid CD sphere, two-step pulse, 16-group FLD | complete (5 ns) at 600/900/1350 cells: focus 3.62/3.56/3.50 ns, peak rhoR 0.115/0.117/0.118 g/cm² (converged); central-focus point values (pressure, Ti) grow with resolution and are NOT converged observables — quote rhoR and focus time |
| ex08_d2_shell_implosion | 430 um CD shell, D2 fill, 3-picket + main pulse | complete (5 ns): stagnation 4.22 ns (window 3.6–4.5), CR 5.9, peak shell density 2.9 g/cc; CR and rhoR (0.008 g/cm²) sit below the design windows — the pulse was retuned (pickets x0.25) after the opacity fix and the remaining gap is the gray-opacity ceiling (see caveats) |
| ex09_cbet_hote | spherical corona, 3-port CBET + TPD/SRS hot electrons | complete (6 ns): absorption 79%, nc/4 density scale length 177 um (window 100–300), CBET port exchange active (peak exchanged power 2.2e20 erg/s, conservation ledger residual <2e-16), hot-electron conversion 0.04% of incident (design window 0.5–3% — threshold-limited at this intensity). Runs at 2000 rays/beam: CBET per-step cost is linear in ray count and insensitive to section counts |
| ex10_marshak_dt | Marshak Tr(t) drive, thin CD pusher, DT fill, burn ON | complete (7 ns): DT neutron yield 7.5e9 (design window 1e8–1e11), bang 2.71 ns (window 4.0–5.5 ns — early; the gray pusher opacity shifts drive coupling, see caveats) |

## Table-opacity variants (2026-08-30)

Since per-material table opacities landed, a multi-material 1D FLD deck may give each material its
own LTE tmat opacity table instead of the gray constants: set
`opacity=dict(model="tmat", file="TMAT-H5/<mat>.tmat.h5", ...)` per
material (tables must be LTE — `--kirchhoff-pe`; the runtime asserts it).
Measured variant behavior on this suite:

- EX-05 with the Kr+CD tables: material shock 63.6 um/ns, witness arrival
  12.8 ns (in window), and a SHARP radiative front (mid-run precursor
  collapses to ~0: cold Kr's real Planck opacity ~1e5 cm2/g absorbs the
  precursor within microns — physically expected for opaque cold Kr,
  unlike the gray constant's 45-70 um precursor).
- EX-08 with the D2+CD tables: clearly better implosion physics than the
  gray constants — stagnation 3.54 ns, CR 7.2 (gray: 5.9), peak D2 density
  1.1 g/cc (gray: 0.51), peak central pressure 3.5 Gbar, rhoR 0.017 g/cm2
  (gray: 0.008). No burn is involved, so the LTE caveat below does not
  bite; table opacities are the better choice here.
- EX-10 with LTE (Kirchhoff) DT+CD tables: bang moves toward the design
  window (2.71 -> 3.04 ns) but the neutron yield collapses to ~9e5 — the
  forced emission = absorption grossly over-radiates hot low-density DT
  and radiation-cools the burn.
- EX-10 with NLTE tables (TMAT-H5/DT_nlte.tmat.h5 + the NLTE CD.tmat.h5):
  the yield recovers to 8.5e9 — inside the 1e8-1e11 design window —
  with bang at 2.24 ns. NLTE tables are the correct choice for burn
  physics; the remaining bang-time gap is drive-tuning, not opacity.

The shipped decks stay on documented gray constants for reproducibility;
the table variants are one-line-per-material edits. For burn problems use
the NLTE tables; LTE tables are fine for pure radiative-hydro
(EX-05/EX-08 class).

## Caveats

- **Gray constant opacities**: EX-05/08/10 approximate real spectral
  opacities with per-material gray constants (Kr: 3.5e3 cm²/g — the
  geometric mean of the PrOpacEOS table Planck/Rosseland means at 18.7 eV,
  0.044 g/cc; CD: 5.0e3 — the CD-table Planck mean at 1.05 g/cc, 100–150 eV,
  i.e. the surface-absorption regime; D2/DT: 1). Precursor lengths, drive coupling, and
  burn timing shift accordingly; the table-opacity upgrade
  (docs/design/multimaterial_table_opacity_20260829.md) is the fix path.
- **History note**: runs performed before commit 48bce971b silently applied
  the FIRST non-void material's opacity to the whole mesh in 1D; any archived
  results from before that commit are superseded by reruns.
- EX-05's witness pressure also rises early (~1 ns) from radiative preheat;
  the shock-arrival signal is the late (>10 ns) rise.
- EX-07's central-focus values depend on resolution by construction
  (converging spherical shock); a 600/900/1350-cell study is the shipped
  convergence evidence.
- Tables `TMAT-H5/{D2,DT,DT_nlte,KR_lte,CD_lte}.tmat.h5` were produced
  with PrOpacEOS (QEOS on) and converted with
  `tools/tmat/propaceos_to_tmat.py` (`--no-ionization --kirchhoff-pe` for
  the LTE variants; drop `--kirchhoff-pe` for NLTE). NOTE: `.tmat.h5`
  tables are NOT included in the beta source snapshot (they derive from
  licensed PrOpacEOS data). Every deck in this suite uses a tmat EOS table
  for its solid materials, so running the suite requires generating the
  tables first with your own PrOpacEOS/SESAME workflow (see
  `tools/tmat/propaceos_to_tmat.py`); ex06 additionally needs a SESAME
  ASCII library.
