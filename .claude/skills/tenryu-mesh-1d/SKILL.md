---
name: tenryu-mesh-1d
description: Generate or revise a TENRYU initial 1D mesh from experimental conditions using the measured convergence campaign and deterministic recommender; repair mesh validation or zoning-report errors. Mesh work item only, including the required fine-mesh retry companion.
metadata:
  short-description: Generate an initial 1D mesh from experimental conditions
---

# TENRYU 1D mesh design (work-item skill, Claude Code variant)

Scope: produce or revise the **initial-mesh specification of a 1D_SPH deck** — the `Mesh(...)` block and its supporting constants — to meet a stated resolution objective under the solver's hard contracts. Out of scope: physics blocks (Materials/Radiation/Laser/Numerics) except where the task explicitly includes them; solver source code (never edit `src/` under this skill).

## Generate a mesh from experimental conditions

1. Gather wavelength, pulse shape/peak or energy-duration-spot (or a power table), layers
   in inner-to-outer order with material/A/Z/density/thickness, geometry and radii, run duration,
   and any pinned mesh budget or boundaries. The conditions schema and four examples are in
   `tools/assist/README.md` and `tools/assist/examples/mesh_conditions/`. If physics is omitted,
   the tool assumes the campaign's CD tmat EOS, radiation off, 2T implicit conduction.
   Every layer needs explicit A/Z: CD uses 7/3.5; Al uses 26.98/13. Missing A/Z is an error.
   Vacuum inside the target (behind a planar foil whose rear surface must release, inside a
   hollow shell, between two layers) is a layer `{"material": "vacuum", "void": true,
   "thickness_cm": ...}` without A/Z; the outermost layer must be material. From a deck, a
   VOID material inside the target is recognized automatically. The campaign has no such
   target, so these conditions are `extrapolation` (a-priori ceiling, convergence pair). The
   solver re-spaces the void cells evenly every step (main 125a94572), so the zoning of a void
   layer only sets its cell count; it has no contact model, so a void that closes during the
   run stops it (flag `interior_void`: make the void wider than the travel of the faces around
   it; see tenryu-namelist).
2. Run the deterministic recommender:
   ```bash
   python tools/assist/assist.py recommend-mesh --conditions conditions.json \
     -o recommendation.json --mesh-out mesh.py
   # With an authorized binary, also pass --tenryu PATH.
   # Existing deck: --deck DECK --deck-out recommended.py (uniform placeholder allowed).
   ```
   A new binary extracts actual conditions. Without one, --deck only accepts a literal
   `MESH_EXPERIMENTAL_CONDITIONS` dictionary; otherwise use --conditions. In a headless
   invocation without command access, use the supplied MESH RECOMMENDATION section. If
   neither conditions nor a recommendation is available, return UNCERTAIN with the missing
   information. A missing cell budget alone is not ambiguous: the tool estimates one.
   If later lint cannot recompute the recommendation after a failed validate, the generation
   harness retains the previous section and journals its fallback source.
3. In `--deck` mode use `--deck-out FILE` for the assembled deck; `mesh_block` and
   `--mesh-out` already carry preserved Mesh keys, so do not apply `replace_mesh` again.
   In `--conditions` mode paste the returned `mesh_block` including its empirical dictionary
   into your own deck. Without `--deck`, `--deck-out` emits only a synthetic mesh-check
   candidate and requires binary validation to have run; otherwise it fails with exit 2.
   A candidate from failed validation is not approved for production. Do not edit the
   ceiling, reference baseline, SHA-256 or case IDs to obtain a coarser grid. Merge the
   Numerics retry companion into the existing Numerics block. Match density regions to the
   physical Geometry and keep all harness pins. A pinned budget that cannot meet the
   requirement requires an explicit task decision; never weaken the requirement.
4. Validate with `lint-deck DECK --tenryu PATH`. `empirical-mesh-integrity` is a hard failure:
   rerun the recommender against the actual deck conditions. The binary checks provenance
   structure and the physical cap; the independent lint checks the learned value. A digest
   alone does not establish integrity. A block computed offline carries the tool's replica
   of the a-priori ceiling; the lint recomputes on that baseline when it is within the
   solver's own 3 % provenance tolerance, and every other number must match exactly. `learned-surface-resolution` warns if achieved
   surface cells are coarser than the recommendation. If validation is available, the tool
   rebuilds on band/box/chain/segment certificates at margins 1.25, 1.45, 1.70 and then
   grows counts by 1.5; width-capacity/admissibility conflicts lower dr_min, up to eight
   attempts. Inspect validation.attempts[].certificate for the parsed numbers. Validation
   first folds the solver's injected bands, shock ceiling and ablated depth from the preview
   into the construction (`fold_in_solver_requirement`; the solver's band radii carry mesh
   round-off, so a requirement equal to 1e-9 of R0 does not refold). An ablation-rule
   MESH_RESOLUTION_REQUIREMENT_VIOLATED extends the ablation zone down to the worst cell's
   depth and rebuilds (`extend_ablation_zone`). A candidate is `validated` only with exit
   code 0, no ablation- or shock-rule violation (the solver only reports shock violations
   of zoning_intent decks; the margin ladder refines such a mesh) and an achieved surface
   cell measure at or below the recommendation. You may apply the repaired block after failure without
   changing the empirical dictionary by hand. Other errors need the specific corrections in
   the contracts below.
5. Act on the flags and report the evidence. `unvalidated` needs actual-binary validation;
   a conditions-only synthetic mesh check does not qualify the production EOS deck.
   `legacy_binary` keeps calibrated enforce and explicit finer bands, so the mesh may cost
   more. `reference_table_mismatch` removes empirical and retries with calibrated enforce
   and explicit finer bands; the warning names the tool digest and the learned efficiency
   benefit is lost until deployment agrees. Other empirical errors appear verbatim in
   `validation.attempts[].error`. The recommender, shipped table and binary must come from
   the same checkout state (the table digest is compiled into the binary);
   `reference_table_mismatch` indicates a stale copy.
   `unconverged_reference` means use the finest measured level or finer and explicitly
   say the mesh is not proven converged. `extrapolation` takes precedence outside coverage
   and uses the material-aware calibrated a-priori ceiling at factor one (no extra
   multiplier or ladder cap; use the solver ceiling when available): compare
   two otherwise identical decks with the surface cell mass halved before production.
   `sparse_evidence` means nearby measurements limited relaxation; inspect the reported
   evidence distance and allowed factor, and retain the returned ceiling.
   Use `tools/validation/mesh_convergence_campaign.py` and
   `docs/design/mesh_convergence_campaign_20260903.md` for ladder/verdict procedures; for an
   existing campaign case, `gen --root DIR --cases C28 --levels 4,5` generates a pair.
   Do not launch a run or allocate a compute venue unless the task authorizes it.
6. After an authorized probe, use `zoning-report OUTPUT_DIR` and the campaign observables.
   State remaining limitations in deck comments when the output contract only allows a deck.

## Evidence and coverage

The tool reads the shipped table, omits sanity rows from fitting, and uses a log-linear
ridge trend plus five nearby measured residuals. Exact converged matches use 95% of the
measured a_conv; all 22 are within a factor of two. Interpolation subtracts the LOO 90th
percentile margin **0.2840422014231186 dex**; **20/22 (90.91%)** are safe before the evidence
gate, **22/22 (100%)** after it, and **0/22** exceed both a-priori and measured ceilings.
The gate permits relaxation only inside the median training nearest-neighbor spacing
(0.36537242214822635), linearly reducing the weighted measured/a-priori excess to zero at
that distance. Exact matches retain 0.95*a_conv, and so do conditions within a normalized
feature distance 0.025 of a converged case with the same geometry, layer count and waveform
class (the measured a_conv is resolved only to its factor-2 ladder step). This is
calibration coverage, not an
independent convergence guarantee. The JSON supplies raw/adjusted/final values, IDs,
distances, weights, `evidence_distance`, `allowed_apriori_factor`, and the full LOO table.

Coverage is CD tmat, radiation off, 2T implicit conduction; 351/527/1053 nm; 3e13–1e15 W/cm2;
planar one/two-layer targets 10–520 micrometers, densities 0.05–2.5 g/cc; and only the two
GXII shells (250 micrometer outer radius, 7 micrometer CD shell at 1.05 g/cc with 0.02 g/cc
fill). The tool freezes every pulse as the solver does (absolute times k*2^-40 s, bisected
where the curve bends) and reads all pulse quantities from that table, so an offline block
and the lint's recomputation from the solver's exported pulse agree even when the pulse
jumps on a feature sample time. A pulse belongs to a campaign waveform class when its
peak-normalized shape on its own energy window (0.5 %–99.5 % of the pulse energy) lies
within an L1 distance 0.10 of that waveform, wherever a low tail is cut; its effective
duration (the integral of I over [0, t_end] divided by the peak) must lie within
0.54–3.94 ns (the campaign's 0.676–3.15 ns widened by 1.25). A run that continues after the
pulse stays in coverage. These are marginal ranges with sparse combinations. Every 1053 nm class, 527 nm at 1e15 and the planar 0.05 g/cc foam remain
unconverged. Other materials, geometries, shell dimensions, out-of-range drives/durations,
or unmatched pulse shapes flag extrapolation. The quiet interior target is 2e-5 g/cm2,
from the <=1% C02-S1 sensitivity study; shock limits and layer minima still apply. The
planar low-density unconverged class (C15, an ablation front in 0.05 g/cc foam) applies only
when the predicted ablated depth reaches such a layer (not to a gas behind a pusher).
The Al fallback example (`tools/assist/examples/mesh_conditions/aluminium_foil.json`)
now yields 9.131233090608189e-7 g/cm2, 2381 estimated cells, a 3.382 nm surface width
ceiling, and `extrapolation`/`unvalidated`; it still needs the convergence pair.
See `docs/design/mesh_recommendation_from_campaign_20260908.md` for the model and limitations.

The block uses planar areal_mass, cylindrical cylindrical_line_mass and spherical
spherical_cell_mass and splits the target into three regions. The ablation zone runs from
the surface down to the deeper of the predicted ablated depth (mu_abl_total_g_cm2: the
solver preview when available, otherwise the Python formation integral) and, for learned
recommendations, the campaign's laser-side half (planar: all layers; spheres and cylinders:
the outer layer). Its cells are capped at 0.95*area(R0)*surface ceiling, the ablation rule's
own reference-area measure, across layers and fill. A spherical or cylindrical innermost
layer lighter than 0.1 of the densest layer (gas fill) outside that zone gets no band of the
recommender, only the 40-cell segment minimum (the solver's own payload band may still bound
it). The campaign measured ablation-side observables only, so fill resolution for
stagnation physics is not covered: check an imploding fill with a separate convergence
pair. A void layer inside the target belongs to none of the regions: it is its own 40-cell
segment (uniform cells in a planar void, equal-mass cells in a curved one), and the payload
starts at the innermost material layer. The remaining unablated payload keeps the local rule
rho*dr <= 2e-5 g/cm2 (C02-S1; the solver's shock ceiling when finer) through a core and
pieces of radius ratio 1.2, the same construction the solver uses for its injected payload
band. The planar C02-S1 result does not justify coarse cells in a burned-through shell.

With a probe preview the solver's injected caps join the count. A segment whose cell
requirement exceeds its share of the zoning monitor has its preferred measures scaled down
(`profile_scale`) so the solver's allocation meets it without inflating the total. Padding
inherits the terminal profile weight with no band and a 40-cell segment minimum. Inspect
budget.segments (required/allocated cells, profile_scale), budget.band_regions,
budget.solver_band_regions, budget.ablation_r_lo_cm, budget.fill_region_cm and
recommendation.mu_abl_source.

## Worked example: GXII shell, Gaussian 527 nm

Run the recommender on `tools/assist/examples/mesh_conditions/gxii_gaussian.json`:
3e14 W/cm2 peak, 1 ns FWHM centered at 1.2 ns, t_end=2.4 ns, 250 micrometer CD shell
with the fill specified above. Actual offline output selects C28 (distance 0, weight 1):
`raw_g_cm2=6.694052038291298e-6`, surface `6.359349436376733e-6`, factor
`8.759239849793895` relative to calibrated a-priori `7.260161321563045e-7`.
The estimated budget is 222 cells, `dr_min=2.39641877934272e-07 cm`; status is `unvalidated`.
Predicted mu_abl=0.001273622204335836 g/cm2 reaches the centre, so the whole target is one
ablation zone. Paste this actual returned block, then validate:

```python
Mesh(
    r_min=0.0,
    r_max=0.04,
    geometry_1d='spherical',
    zoning_intent=dict(
        n_cells=222,
        measure='spherical_cell_mass',
        density_regions=[{'r_end': 0.024300000000000002, 'rho': 0.02}, {'r_end': 0.025, 'rho': 1.05}, {'r_end': 0.04, 'rho': 1e-09}],
        pins=[{'r': 0.024300000000000002, 'ratio_jump_allowed': True}, {'r': 0.025, 'ratio_jump_allowed': True}],
        profile=[{'r': 0.0, 'w': 4.74489029934624e-08}, {'r': 0.024299975700000004, 'w': 4.74489029934624e-08}, {'r': 0.024300000000000002, 'w': 2.7923135083265143e-08}, {'r': 0.025, 'w': 2.7923135083265143e-08}],
        bands=[{'measure_frac_begin': 0.0, 'measure_frac_end': 0.999999970265094, 'cell_measure_max': 4.74489029934624e-08}],
        dr_min=2.39641877934272e-07,
        preferred_ratio=1.3,
        ratio_hard_max=1.3,
        min_cells_per_segment=40,
    ),
    resolution_requirement=dict(
        apply='enforce',
        empirical={'reference_sha256': '9877d1a8c61ead63c6b8a3822ccfc069d149bd849838da79eb5965edd62fd724', 'case_ids': ['C28'], 'surface_ceiling_g_cm2': 6.359349436376733e-06, 'reference_apriori_g_cm2': 7.260161321563045e-07},
    ),
)
```

Merge `Numerics(hydro=dict(driver_full_step_retry_enabled=True))` before a run. Keep the
conditions and evidence with the deck; the 9.220252473467259 relaxation cap comes from C28's
measured/calibrated ratio, not a tunable safety factor.

## Mesh vocabulary (complete, current — do not invent keys)

Exactly four mesh forms exist today, followed by one companion key.

1. **uniform**: `Mesh(nr=N, grid="uniform", r_min=..., r_max=...)`.
2. **graded** (per-segment counts + in-segment mass grading):
   ```python
   Mesh(r_min=0.0, r_max=R,
        grid=dict(type="graded",
                  segments=[{"r_start": 0.0, "r_end": R, "nr": 450}, ...],  # contiguous, monotone
                  grading=dict(edge_ratio=0.25,   # (0,1); smaller = finer segment edges
                               sg_order=4,        # even, >= 2
                               sg_sigma=0.7)))    # (0,1)
   ```
3. **auto (equal-mass regions + geometric interface bridges)** — top-level keys, NOT inside grid:
   ```python
   Mesh(r_min=0.0, r_max=R_MAX,
        auto_regions=[{"r_end": R1, "nz": 30, "rho_ref": 0.02},
                      {"r_end": R2, "nz": 40, "rho_ref": 1.05, "material_group": "shell"},
                      {"r_end": R_MAX, "nz": 40, "rho_ref": 1e-10, "is_void": True}],
        auto_zone=dict(mass_ratio_max=1.3, dr_min=1.0e-6))
   ```
   Do NOT pass `nr` together with `auto_regions` (the count is derived as sum of `nz`). Region keys are exactly `r_end, nz, rho_ref, is_void, material_group`; same `material_group` on adjacent regions enables interface mass matching; `is_void` regions get equal Δr and no bridges.
4. **zoning_intent (declarative constrained zoning)** — top-level key, NOT inside grid; mutually exclusive with `auto_regions` and `grid` segments; 1D only; Experimental:
   ```python
   Mesh(r_min=0.0, r_max=0.06, geometry_1d="spherical",
        zoning_intent=dict(
          n_cells=400,                       # required, total cells (do NOT also pass nr)
          measure="spherical_cell_mass",     # width|areal_mass|cylindrical_line_mass|spherical_cell_mass
          density_regions=[{"r_end": 0.0425, "rho": 1.05},   # zoning density model;
                           {"r_end": 0.06,   "rho": 0.01}],  # REQUIRED for mass measures; last r_end == r_max
          pins=[{"r": 0.0425, "ratio_jump_allowed": True}],  # exact node at each pin
          profile=[{"r": 0.0, "w": 4.0}, {"r": 0.06, "w": 1.0}],  # preferred cell-measure shape (log-linear)
          anchors=[{"r": 0.0425, "half_width": 0.004, "log_amplitude": -1.0}],  # local refine (<0) / coarsen (>0)
          bands=[{"measure_frac_begin": 0.9, "measure_frac_end": 1.0,  # fractions of TOTAL measure
                  "cell_measure_max": 2.6e-6}],               # hard per-cell bound inside the band
          dr_min=1.0e-6, ratio_hard_max=2.0, min_cells_per_segment=1))
   ```
   Semantics you must know: the solver equidistributes the chosen MEASURE weighted by profile×anchors, then projects onto the hard constraints and independently verifies (fail-closed). `measure` must match geometry (`spherical_cell_mass`→spherical, `cylindrical_line_mass`→cylindrical; `width`/`areal_mass` always allowed — areal mass ρ·Δr is the ablation-resolution measure, so an ablator band limit is naturally a `bands` entry with `measure="areal_mass"` or a band ceiling in the run's mass measure). `bands` select by fractions of the cumulative measure (Lagrangian-invariant: the outer 10% of mass is `0.9..1.0` regardless of radii). Per-anchor `|log_amplitude| <= ln(1e4)`; overlapping anchors' summed magnitude `<= ln(1e6)`. `ratio_hard_max` is capped at 2.0 by solver policy — never try to raise it.
5. **resolution_requirement (physics-derived resolution requirement; companion of any form, 1D only)** — TOP-LEVEL key `resolution_requirement=dict(apply="report"|"enforce", zones_per_scale_length=9, intensity_exponent=0.4, intensity_reference_W_cm2=1e14, scale_length_factor=0.12, ablation_mass_safety=1.5, formation_ablated_fraction=0.1, absorbed_fraction=1.0, shock_cells_per_separation=8, shock_event_min_separation_frac=0.05, min_cells_per_layer=10, zbar_override=0.0, n_bands=6)`. The solver computes, from the deck's laser waveform/wavelength/material layers/geometry, the areal-mass ceiling profile of the ablated band, a shock-separation ceiling for the payload and a layer minimum, and (with `apply="enforce"` + `zoning_intent`) injects them as hard bands. Laser decks: ALWAYS set `apply="enforce"` with a mass measure. Never set `enabled=False`, never loosen the coefficients to pass. The preview/lint summary also carries `dr_min_admissible_cm`, the largest `dr_min` compatible with the injected ceilings.

## Hard contracts (validator- and physics-enforced; never violate, never silence)

- Adjacent-cell mass ratio: target <= 1.3, hard <= 2.0 (`auto_zone.mass_ratio_max` / `zoning_intent.ratio_hard_max` in (1.0, 2.0]; 2.0 is an immutable solver policy ceiling).
- `dr_min` is a hard width floor; a `dr_min binding` warning means the floor is active near an interface. With `resolution_requirement.apply="enforce"`, `dr_min` must not exceed the summary's `dr_min_admissible_cm` (≈ formation ceiling / surface density, e.g. 9.4e-7/1.05 ≈ 9e-7 cm for a CD shell at 3.3e14 W/cm²); a larger floor is refused before solving as `[mesh-requirement] MESH_RESOLUTION_REQUIREMENT_DR_MIN_CONFLICT` (the message names the admissible value) — lower or omit `dr_min`, never the requirement.
- Nodes strictly monotone; segments contiguous (`|r_end[k]-r_start[k+1]| < 1e-14*r_max`).
- Unknown keys are rejected (with did-you-mean hints) — treat every such error as authoritative.
- User-pinned values (intent pins) are inviolable: if a pin conflicts with the objective, stop and ask — never change a pinned value to make an error go away.
- `[mesh-requirement] MESH_RESOLUTION_REQUIREMENT_VIOLATED` (validate/run refusal for non-intent forms) and injected-band infeasibility (`MESH_*` certificates prefixed with `[mesh-requirement] injected bands active`): the mesh is under-resolved for its own laser — refine the outer ablator / grow `n_cells`; when the budget is pinned answer `UNCERTAIN:`; never touch the requirement coefficients.
- `zoning_intent` failures carry stable machine codes in three classes; react by class, never by retrying blindly:
  - invalid input (`MESH_PIN_OUT_OF_DOMAIN`, `MESH_PROFILE_*`, `MESH_ANCHOR_*`, `MESH_BAND_*_INVALID`, ...): fix exactly the named element.
  - infeasible WITH certificate (`MESH_DR_MIN_COUNT_INFEASIBLE`, `MESH_SEGMENT_MIN_COUNT_INFEASIBLE`, `MESH_CELL_MEASURE_BOX_INFEASIBLE`, `MESH_CHAIN_SUM_INFEASIBLE` — the message contains the violated numbers, e.g. the feasible total-measure window): the INTENT over-constrains; relax the named constraint or grow the budget. Never respond to an infeasibility by weakening a safety bound the task pinned.
  - numerical failure (`MESH_PROJECTION_STAGNATED`, `MESH_QUADRATURE_NOT_CONVERGED`, `MESH_POSTCHECK_*`): do NOT change physics intent in response, except `MESH_POSTCHECK_WIDTH` reporting a minimum width below `dr_min` while injected bands are active — that is the floor/ceiling conflict of the dr_min contract above: lower `dr_min`; `MESH_QUADRATURE_NOT_CONVERGED` usually means an undeclared density discontinuity — declare it via `pins`/`extra_events`; a `MESH_POSTCHECK_RATIO_CROSS_PIN` names the pin whose two sides disagree — set `ratio_jump_allowed` on it or rebalance per-side cells.

## Runtime companion of fine meshes (verified 2026-08-28)

A zoning_intent mesh with very fine ablator cells (dr_min below ~1e-6 cm, or any
probe-promoted band) makes transient cell inversions at the ablation front likely;
without mitigation the run hard-asserts ("non-positive cell volume"). Pair such a
mesh with `Numerics(hydro=dict(driver_full_step_retry_enabled=True))` — the
documented snapshot + dt/2 soft-retry path for exactly this case (FLD/SN modes
only). This is a Numerics key you MAY add when the mesh you designed needs it;
state it in your summary when you do.

## Physics rules for laser-ablation problems (calibrated)

- The binding criterion is the **mass-form ablation lint**: every zone that will be ablated (its density falls toward the critical density during the run) must satisfy areal mass `rho0*dr <= kappa * rho_c * L_c`. What matters is the zone's mass (its expanded size at the critical surface), not the adjacent ratio.
- Therefore: finest zones at the **outer (ablator) surface**; the quiet payload interior may be coarse; corona/void regions equal-Δr; keep ratios smooth (feathering) but remember ratio smoothness alone does not satisfy the mass criterion.
- Before a probe, `rho_c`/`L_c` come from the solver's deterministic requirement model (a scaling-law estimate with stated coefficients); a probe plus `zoning-report` remains the arbiter and reports the observed/predicted ratios.
- Use the deterministic recommender below to select a data-backed ceiling. Never manually loosen a-priori bands from a superficially similar row; the empirical dictionary and independent integrity lint are required.

## Guardrails

- Prefer omitting keys so solver defaults apply; never add physics keys to fix a mesh error.
- Change only what the feedback names; keep diffs minimal between iterations.
- If the objective is ambiguous (e.g. conflicting pins or unknown material layering), ask one concise question instead of guessing.
