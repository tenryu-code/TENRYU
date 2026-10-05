# 1D laser-plasma example suite (EX-01 .. EX-10)

Ten self-contained 1D_SPH decks modeling typical laser-plasma experimental
setups: foil shocks, impedance matching, burn-through, a radiative shock,
cylindrical and spherical compressions, a shell implosion, CBET/hot-electron
corona physics, and a Marshak-driven DT exploding pusher. The physical
designs were produced by an external design pass and the decks were
generated/tuned through the `tools/assist` deck-generation loop, then
commissioned run-by-run; the metrics below are measured from those runs
(ex08 and ex10 re-measured on 2026-09-29 after their Zbar correction).

Run any deck with:

    ./build/tenryu run examples/laser_plasma_1d/ex01_cd_foil_breakout.py

Outputs land in `outputs/<deck name>/`.

## Suite conventions

- Buffers/exteriors are VOID material regions (`is_void=True`, uniform cell
  widths); planar targets put the rear solid face exactly at x=0, on the inner
  node, which is held fixed in every geometry (NUMERICS §3.1.11):
  `boundary_1d="free"` frees only the outer, laser-side face. The shock
  therefore reaches the rear face at the right time and reflects there
  (ex01's reflected-at-wall state); modelling a free rear surface needs a VOID
  region behind the target.
- Conduction uses the implicit solver in every deck (low-density buffers
  make explicit/STS stage counts explode).
- Every deck carries `driver_full_step_retry_enabled=True`; ex05, ex07, ex08,
  ex09 and ex10 also set `driver_full_step_retry_max_attempts=8`, and ex05,
  ex07, ex08 and ex10 add the artificial-viscosity / odd-even damping set
  (`av_heat_C=0.5`, `av_heat_to="ion"`, `odd_even_damping_C=1.0`,
  `ee_odd_even_C=0.15`).
- Multi-material radiative decks may give each material its own opacity:
  per-material constants, or a per-material tmat table (LTE or NLTE) — see
  the table-opacity variants section below. The shipped decks stay on gray
  constants for reproducibility. Mixed cells evaluate each contributing
  material's table at its partial density and combine with mass-fraction
  weights (the internal design note multimaterial_table_opacity_20260829.md). Table
  EOS (`eos.model="tmat"`) is fine in any deck.

## Examples and measured commissioning metrics

| deck | setup | status / key measured metrics |
|---|---|---|
| ex01_cd_foil_breakout | 50 um CD foil, 351 nm, 1e14 W/cm², shock breakout | complete (3 ns): shock 46.7 um/ns, breakout 1.13 ns, ~83 Mbar reflected-at-wall state |
| ex02_two_pulse_timing | double pulse, shock merger timing | complete: merger and breakout inside design windows |
| ex03_impedance_match | layered target, transit through witness | complete: witness transit 1.95 ns (in window) |
| ex04_burnthrough_tag | thin exploding foil, burn-through tag | complete: tag time 0.76 ns (in window) |
| ex05_kr_radiative_shock | CD piston into 0.04 g/cc Kr, 800 um column, 24-group FLD, 18 ns | complete: material shock 57 um/ns (design window 70–120 — the gray CD opacity radiates drive energy away; see caveats), witness arrival 13.4–13.5 ns, peak witness compression 1.59 g/cc, reflected re-compression after arrival; radiative precursor exists but is short (~10–50 um mid-run) under the gray Kr constant |
| ex06_cyl_liner | cylindrical CD liner onto D2 column | deck validated; requires a user-provided SESAME ASCII library at `SESAME/xsesame_short` (material 5263 = D2) — not shipped with the repository |
| ex07_solid_sphere | solid CD sphere, two-step pulse, FLD with the CD table's 80 groups (repacked toward 2–5 keV) | complete (5 ns) at 600/900/1350 cells: focus 3.62/3.56/3.50 ns, peak rhoR 0.115/0.117/0.118 g/cm² (converged); central-focus point values (pressure, Ti) grow with resolution and are NOT converged observables — quote rhoR and focus time |
| ex08_d2_shell_implosion | 430 um CD shell, D2 fill, 3-picket + main pulse | complete (6.5 ns; measured 2026-09-29): stagnation 5.03 ns (window 3.6–4.5 — late), CR 6.8, peak density 17.7 g/cc, peak rhoR 0.040 g/cm² at 5.09 ns; CR and rhoR sit below the design windows (10–20, 0.08–0.20). Each cell takes its material's Zbar (D2 1, CD 3.5): the former `fixed_value=3.5` (3.5 electrons per ion in the fill) changed CR and rhoR by under 1 % on the same binary and raised the peak central temperature from 0.92 to 1.01 keV. The deck ran to 5 ns before, 26 ps short of stagnation; the values listed before (stagnation 4.22 ns, CR 5.9, 2.9 g/cc, rhoR 0.008) came from an older binary and are superseded |
| ex09_cbet_hote | spherical corona, 3-port CBET + TPD/SRS hot electrons | complete (6 ns): absorption 79%, nc/4 density scale length 177 um (window 100–300), CBET port exchange active (peak exchanged power 2.2e20 erg/s, conservation ledger residual <2e-16), hot-electron conversion 0.04% of incident (design window 0.5–3% — threshold-limited at this intensity). Runs at 2000 rays/beam: CBET per-step cost is linear in ray count and insensitive to section counts |
| ex10_marshak_dt | Marshak Tr(t) drive, thin CD pusher, DT fill, burn ON | complete (7 ns; measured 2026-09-29): DT neutron yield 7.8e10 (design window 1e8–1e11), bang 2.79 ns (window 4.0–5.5 ns — early; the gray pusher opacity shifts drive coupling, see caveats), stagnation 3.07 ns, CR 3.9. Each cell takes its material's Zbar (DT 1, CD 3.5): the former `fixed_value=3.5` gave the DT fuel 3.5 electrons per ion and a yield of 3.6e9 on the same binary (bang 2.79 ns); the 7.5e9 listed before came from an older binary |

## Table-opacity variants

Since per-material table opacities landed, a multi-material 1D FLD deck may give each material its
own tmat opacity table instead of the gray constants: set
`opacity=dict(model="tmat", file="TMAT-H5/<mat>.tmat.h5", ...)` per
material (the run then uses the table's group structure: 80 groups for these tables). Both LTE
tables (`--kirchhoff-pe`, emission = absorption) and NLTE tables are accepted; NLTE tables carry
emissivity != absorptivity (in hot, thin DT the `DT_nlte` emission falls to 2 % of the absorption in
some groups). Measured variant behavior on this suite:

- EX-05 with the Kr+CD tables (measured 2026-08-30 on an older binary): material shock 63.6 um/ns,
  witness arrival 12.8 ns (in window), and a SHARP radiative front (mid-run precursor
  collapses to ~0: cold Kr's real Planck opacity ~1e5 cm2/g absorbs the
  precursor within microns — physically expected for opaque cold Kr,
  unlike the gray constant's 45-70 um precursor).
- EX-08 with the LTE D2 and CD_lte tables (measured 2026-09-29, 6.5 ns): stagnation 3.51 ns,
  CR 9.95 (gray: 6.8), peak D2 density 16.4 g/cc (gray: 2.3), peak central pressure 21.9 Gbar
  (gray: 1.1), peak rhoR 0.154 g/cm2 (gray: 0.040). Stagnation and CR fall just short of their
  design windows (3.6–4.5 ns, 10–20) and the peak rhoR is inside its window (0.08–0.20); the gray
  constants miss all three by far. No burn is involved; table opacities are the better choice here.
- EX-10 with the LTE DT and CD_lte tables (measured 2026-09-29): DT yield 2.4e10, bang 2.38 ns.
  With the NLTE DT_nlte and CD tables: yield 2.6e10, bang 2.37 ns. The gray constants give 7.8e10
  and 2.79 ns. With the former Zbar of 3.5 in every cell the same binary gives 2.8e9 (LTE) and
  3.1e9 (NLTE), so the two table sets differ by 10 % or less either way. The notes of 2026-08-30
  (LTE yield collapsing to ~9e5, NLTE 8.5e9 at bang 2.24 ns) are not reproduced by the current
  binary with the same one-line edits and either Zbar setting; the solver change responsible was
  not identified.

The shipped decks stay on documented gray constants for reproducibility;
the table variants are one-line-per-material edits. For burning fuel the NLTE
tables remain the physically appropriate choice (emission below absorption in
hot, thin plasma), although in EX-10 the two table sets give yields within 10 %;
LTE tables are fine for pure radiative-hydro (EX-05/EX-08 class).

## Caveats

- **Gray constant opacities**: EX-05/08/10 approximate real spectral
  opacities with per-material gray constants (Kr: 3.5e3 cm²/g — the
  geometric mean of the PrOpacEOS table Planck/Rosseland means at 18.7 eV,
  0.044 g/cc; CD: 5.0e3 — the CD-table Planck mean at 1.05 g/cc, 100–150 eV,
  i.e. the surface-absorption regime; D2/DT: 1). Precursor lengths, drive coupling, and
  burn timing shift accordingly; the table-opacity upgrade
  (the internal design note multimaterial_table_opacity_20260829.md) is the fix path.
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
