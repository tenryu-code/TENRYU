---
name: tenryu-mesh-1d
description: TENRYU の 1D_SPH 初期メッシュ（Mesh ブロック/ゾーニング）を設計・修正する作業単位スキル（codex headless 変種）。アシスタント層やユーザーから 1D メッシュの新規設計・改訂・validator エラー修正・zoning-report 違反への応答を依頼されたときに使う。出力はマーカー間のデッキ本文のみ。Use when asked (typically headless via codex exec, often by the TENRYU assistant harness) to design or revise the initial 1D mesh of a TENRYU deck, fix mesh-related validator errors, or respond to zoning-report verdicts. Mesh work item only; output only between markers.
metadata:
  short-description: 1D 初期メッシュ設計（codex headless 変種）
---

# TENRYU 1D mesh design (work-item skill, codex headless variant)

You are invoked non-interactively (often by an automated harness in a scratch working directory, possibly with a read-only sandbox). Your entire product is TEXT following the output contract below. Do not write files, do not run commands unless the invoking prompt explicitly provides tools for it, and do not produce prose outside the contract.

## Output contract (exactly one of these)

1. A complete TENRYU deck between two marker lines — a line `BEGIN_DECK` and a line `END_DECK` — and nothing else outside them. The deck starts with `from tenryu_namelist import *`. (When the invoking prompt supplies a starting deck/template, modify it minimally: change only the mesh-related parts unless instructed otherwise.)
2. A single line starting with `UNCERTAIN: ` followed by one concise question — when the task is ambiguous (no budget, conflicting pins, unknown layering) or when feedback conflicts with a pinned value. Never guess; never change a pinned value to silence an error.

## Mesh vocabulary (complete, current — do not invent keys)

Exactly four forms exist today:

1. **uniform**: `Mesh(nr=N, grid="uniform", r_min=..., r_max=...)`.
2. **graded**: `grid=dict(type="graded", segments=[{"r_start":..., "r_end":..., "nr":...}, ...], grading=dict(edge_ratio=..., sg_order=..., sg_sigma=...))` — segments contiguous and monotone; `edge_ratio` in (0,1) (smaller = finer segment edges), `sg_order` even >= 2, `sg_sigma` in (0,1).
3. **auto (equal-mass regions + interface bridges)** — TOP-LEVEL keys, not inside `grid`:
   `auto_regions=[{"r_end":..., "nz":..., "rho_ref":..., "is_void":..., "material_group":...}, ...]` plus `auto_zone=dict(mass_ratio_max=1.3, dr_min=...)`. Do NOT pass `nr` together with `auto_regions` (the count is the sum of `nz`). Region keys are exactly those five; same `material_group` on adjacent regions enables interface mass matching; `is_void` regions get equal Δr and no bridges.
4. **zoning_intent (declarative constrained zoning)** — TOP-LEVEL key; mutually exclusive with `auto_regions` and `grid` segments; 1D only; do NOT also pass `nr`:
   `zoning_intent=dict(n_cells=N [required], measure="width"|"areal_mass"|"cylindrical_line_mass"|"spherical_cell_mass", density_regions=[{"r_end":..., "rho":...}, ...] [REQUIRED for mass measures; last r_end == r_max], pins=[{"r":..., "ratio_jump_allowed": bool}, ...], profile=[{"r":..., "w":...}, ...] [preferred cell-measure shape, log-linear, w>0], anchors=[{"r":..., "half_width":..., "log_amplitude":...}, ...] [local refine <0 / coarsen >0; per-anchor |A|<=ln(1e4), overlapping sum <=ln(1e6)], bands=[{"measure_frac_begin":..., "measure_frac_end":..., "cell_measure_min":..., "cell_measure_max":...}, ...] [fractions of TOTAL cumulative measure — Lagrangian-invariant selectors], extra_events=[...], dr_min=..., cell_measure_min=..., cell_measure_max=..., preferred_ratio=1.3, ratio_hard_max=2.0, min_cells_per_segment=1)`.
   `measure` must match geometry: `spherical_cell_mass`→`geometry_1d="spherical"`, `cylindrical_line_mass`→`"cylindrical"`; `width`/`areal_mass` always allowed (areal mass ρ·Δr is the ablation-resolution measure — an ablator resolution objective is naturally a `bands` ceiling). The solver equidistributes the measure weighted by profile×anchors, projects onto the hard constraints, and independently verifies (fail-closed). Prefer this form whenever the task states resolution OBJECTIVES (band mass limits, local refinement, size profile) rather than zone counts.

## Hard contracts (never violate)

- Adjacent-cell mass ratio: target <= 1.3, hard <= 2.0 (`auto_zone.mass_ratio_max` / `zoning_intent.ratio_hard_max` in (1.0, 2.0]; 2.0 is an immutable solver policy ceiling — never raise it).
- `dr_min` is a hard width floor (a `dr_min binding` warning means it is active).
- Segments/regions contiguous and strictly monotone in r.
- Unknown namelist keys are rejected by the validator — its error text (with did-you-mean hints) is authoritative; fix exactly the named key at the named location.
- Intent pins supplied by the harness are inviolable.
- `zoning_intent` errors are `[mesh-zoning-intent] MESH_*` codes in three classes; react by class:
  - invalid input (`MESH_PIN_*`, `MESH_PROFILE_*`, `MESH_ANCHOR_*`, `MESH_BAND_*_INVALID`): fix exactly the named element.
  - infeasible with certificate (`MESH_DR_MIN_COUNT_INFEASIBLE`, `MESH_SEGMENT_MIN_COUNT_INFEASIBLE`, `MESH_CELL_MEASURE_BOX_INFEASIBLE`, `MESH_CHAIN_SUM_INFEASIBLE`): the intent over-constrains — relax the constraint the message names or grow the budget; if the conflicting bound is harness-pinned, answer `UNCERTAIN:` instead of weakening it.
  - numerical (`MESH_PROJECTION_STAGNATED`, `MESH_QUADRATURE_NOT_CONVERGED`, `MESH_POSTCHECK_*`): do NOT change physics intent; `MESH_QUADRATURE_NOT_CONVERGED` = declare the density discontinuity via `pins`/`extra_events`; `MESH_POSTCHECK_RATIO_CROSS_PIN` = set `ratio_jump_allowed` on the named pin or rebalance per-side cells.

## Runtime companion of fine meshes (verified 2026-08-28)

A zoning_intent mesh with very fine ablator cells (dr_min below ~1e-6 cm, or any
probe-promoted band) makes transient cell inversions at the ablation front likely;
without mitigation the run hard-asserts ("non-positive cell volume"). Pair such a
mesh with `Numerics(hydro=dict(driver_full_step_retry_enabled=True))` — the
documented snapshot + dt/2 soft-retry path for exactly this case (FLD/SN modes
only). This is the ONE Numerics key you may add for a mesh task; mention it in
the deck comment above the Mesh block when you add it.

## Physics rules for laser-ablation problems

- Binding criterion: every zone that will be ablated must satisfy areal mass `rho0*dr <= kappa*rho_c*L_c` (mass matters, not the neighbor ratio). Finest zones at the OUTER (ablator) surface; quiet payload interior may be coarse; corona/void equal-Δr; keep ratios smooth as well.
- A probe run plus the harness's zoning report is the arbiter of adequacy; before probe feedback exists, follow the task's stated budget and the reference conventions of TENRYU's GXII examples.

## Feedback interpretation (the harness feeds these back verbatim)

- `validate.stderr_tail`: fix exactly the named key/structure; keep every other line of the deck unchanged.
- `lints[]` entries `adjacent-dr-ratio` / `autozone-mass-ratio`: adjust the `nz` distribution, `edge_ratio`, or `mass_ratio_max` (within its legal range) to meet the stated bound.
- Zoning verdict (`lint_a` violations, usually outer ablator cells): reduce areal mass there — refine the outermost region/segment or lower `edge_ratio`; keep the total cell budget unless the task allows growth.
- `ITERATION-FEEDBACK` sections list what failed on the PREVIOUS attempt; change only what the feedback names, in minimal diffs.
- A feedback payload whose top level contains `"error"` is an infrastructure failure, not a deck defect: respond with `UNCERTAIN: tool error reported — <quote it briefly>` instead of editing the deck.
