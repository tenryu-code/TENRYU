# EXPERIMENTAL: TENRYU Assistant Infrastructure

This directory provides deterministic assistant infrastructure around TENRYU. LLM-driven verbs land in later milestones, and this layer never runs inside the solver.

The assistant defaults to OFF with `enabled = false`. `TENRYU_ASSIST_DISABLE` is the kill switch and beats every other setting when its value is `1`, `true`, `TRUE`, or `yes`. Deterministic verbs work regardless of this setting; only LLM invocations are gated.

Configuration is selected in this order:

`--config-or-defaults PATH` (what TENRYU Studio passes): PATH when it exists, otherwise the built-in defaults; the list below is then skipped.

1. `--config PATH`
2. `TENRYU_ASSIST_CONFIG`
3. `./assistant.toml`
4. `~/.tenryu/assistant.toml`
5. `assistant.toml` in the Studio app configuration folder (macOS `~/Library/Application Support/jp.osaka-u.ile.tenryu-studio/`, Linux `~/.config/jp.osaka-u.ile.tenryu-studio/`, Windows `%APPDATA%\jp.osaka-u.ile.tenryu-studio\`) — the file TENRYU Studio's LLM settings form writes
6. Built-in defaults

| Key | Meaning |
| --- | --- |
| `enabled` | Boolean opt-in for LLM invocation |
| `[providers.NAME]` | Non-empty `command` template and `model` |
| `[roles]` | Known role to provider name or `dry_run`; `question_answering` is used by `ask` |
| `[budget]` | Positive intervention and token limits |

Usage:

```sh
tools/assist/assist.py status
```

Python 3.9+ is supported through a bundled strict TOML-subset parser. Configuration files must stay within that subset.

## Verbs

`import-deck` records a Python namelist without a CUDA binary and supports
Studio form import and numerical equivalence checks. It runs the deck in a
timed child process in the deck's directory; this executes ordinary local
Python, including its imports and side effects. No assistant/provider config
or server profile is needed. Errors include the traceback. A mesh-planner deck
can use `--repo-root` to locate a checkout or mirror.
Usage: `tools/assist/assist.py import-deck --deck DECK [--repo-root DIR] [--timeout SECONDS]`
The alternative `--request FILE` takes the JSON record/verify/compare protocol
documented in [`docs/gui/DECK_IMPORT.md`](../../docs/gui/DECK_IMPORT.md).

`status` reports the resolved assistant configuration.
Usage: `tools/assist/assist.py status [--config FILE]`

`digest` summarizes run metadata, frozen configuration, and 1D history diagnostics.
Usage: `tools/assist/assist.py digest OUTPUT_DIR [-o FILE] [--series-points N]`

`zoning-report` reports the mass-form ablation-zoning lint from 1D probe outputs. When the run directory holds `mesh_requirement.json`, the report adds `prediction_vs_observation` (ablated mass, ρ_c and scale-length ratios, suggested coefficients; report-only).
Usage: `tools/assist/assist.py zoning-report OUTPUT_DIR [--kappa F] [--ablate-margin F] [--strict] [-o FILE]`

`promote-zoning` compiles observed ablated cells into a suggested hard spherical-measure band.
Usage: `tools/assist/assist.py promote-zoning OUTPUT_DIR [--kappa F] [--margin-cells N]`

`lint-deck` runs the solver's validators and reports mesh, intent-lock, and baseline checks. For 1D laser decks the payload carries the solver's physics-derived resolution requirement (`mesh_requirement`, compact `resolution_requirement_summary`) and the lints `ablation-band-resolution` (hard), `resolution-requirement-disabled` (hard), `shock-separation-resolution` (warn), `layer-min-cells` (info).
Usage: `tools/assist/assist.py lint-deck DECK [--tenryu PATH] [--baseline FROZEN.json] [--intent INTENT.json] [-o FILE] [--keep-tmp]`

`generate-deck` generates a deck from a specification and retries with validator feedback. When a template is given, or from the second iteration on, the prompt carries a `RESOLUTION REQUIREMENTS` section with that summary and the campaign mesh recommendation. `--conditions JSON` supplies a recommendation from the first prompt.
Usage: `tools/assist/assist.py generate-deck SPEC --out-deck FILE [--tenryu PATH] [--template FILE] [--conditions JSON] [--intent INTENT.json] [--baseline FROZEN.json] [--max-iters N] [--workdir DIR] [--config FILE]`

`freeze-baseline` freezes a deck for later baseline comparison.
Usage: `tools/assist/assist.py freeze-baseline DECK [--tenryu PATH] [-o FILE]`

`docmap` writes a deterministic document map (reader pages with headings, canonical
documents with heading line ranges, example decks) and, with `--keys-out`, the namelist
key index extracted from the `enforce_known_keys` lists in `src/core/namelist/builder.cpp`.
Usage: `tools/assist/assist.py docmap [--repo-root DIR] [-o FILE] [--keys-out FILE]`

`ask` answers a question about TENRYU from the checkout's documents and source. It
regenerates the document map and key index into the working directory, assembles the prompt
(the `tenryu-docs-qa` skill body, the map, the previous turns of the same working directory,
the question), runs the provider configured for the `question_answering` role with the
repository root as its working directory, and then checks the answer: every cited path
(`path:line`, `path:line-line`, or `path`) must exist in the checkout, and every namelist key
mentioned in code (`key=` or `Block.sub.key`) must exist in the key index. Checks only warn;
the answer is printed unchanged, followed by one check line.
Usage: `tools/assist/assist.py ask "QUESTION" | --question-file FILE [--workdir DIR] [--repo-root DIR] [--max-history N] [--timeout-s S] [--json] [--print-prompt] [--config FILE]`
Exit codes: 0 answered, 2 disabled / configuration error / provider failure. `--workdir` defaults
to `~/.tenryu/ask/<UTC stamp>/`; pass the same directory again to ask a follow-up (the previous
turns are included). `--print-prompt` prints the assembled prompt without calling any provider
and works while the assistant is disabled.

`digest` requires `h5py` for history reductions and gracefully degrades when it is unavailable. `lint-deck` and `freeze-baseline` require a built `tenryu` binary, selected with `--tenryu`, `TENRYU_BIN`, or found at `./build/tenryu`.

`generate-deck` requires `enabled = true` and a configured `deck_design` role. Its exit codes are: 0 accepted, 2 failure/disabled/exhausted, and 3 when the model asks a clarifying question (`UNCERTAIN`). Every model call and accepted deck is journaled. Every prompt starts with the deck-authoring skill body (`tools/assist/skills/codex/tenryu-namelist/SKILL.md`, journaled as `deck_skill_context`) before the SPEC section.

Every LLM invocation is journaled as JSONL with model identity and prompt/response hashes. Replay never re-queries an LLM.

## Question answering (per-user setup)

Each user runs the question answering with their own model access; nothing is shared.
1. Have a checkout that contains `docs/site/`, `docs/SPECIFICATION.md`, and `src/`.
2. Install Claude Code (`claude`) or Codex CLI (`codex`) and log in with your own subscription or API key.
3. Copy `tools/assist/assistant.example.toml` to `~/.tenryu/assistant.toml`, set `enabled = true`,
   and point `question_answering` at a read-only provider (`claude_readonly` or `codex_readonly`). (TENRYU Studio writes this file into its app configuration folder from アシスタント → 設定 → LLM 設定 and reads only that file, via `--config-or-defaults`; the CLI finds it automatically when no `./assistant.toml` or `~/.tenryu/assistant.toml` exists.)
The Claude Code template passes --add-dir with the working directory so the model can read the generated document map and key index.
4. `python3 tools/assist/assist.py ask "質問"` (or use the Studio assistant view).
Answers end with a `根拠 / Evidence` list of `path:line` citations; the check line after the
answer reports which citations and namelist keys could be verified against this checkout.
The skill text used as the prompt head is `tools/assist/skills/codex/tenryu-docs-qa/SKILL.md`
(Claude Code users can also load `.claude/skills/tenryu-docs-qa/`).

`tools/assist/examples/qa_eval.jsonl` holds 22 evaluation questions with the sources and
namelist keys a correct answer is expected to cite (`expected_sources`, `expected_keys`); run
them with the `ask` verb and compare the answers' evidence lists against those fields.

## Remote execution wrapper

`tenryu_remote.sh` lets a CUDA-less workstation drive a `tenryu` binary on a remote GPU host while presenting the local CLI subset used by the assistant verbs.

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `TENRYU_REMOTE_HOST` | SSH host | Required |
| `TENRYU_REMOTE_REPO` | Remote repository and working directory | Required |
| `TENRYU_REMOTE_BIN` | Remote `tenryu` binary | `$TENRYU_REMOTE_REPO/build/tenryu` |
| `TENRYU_REMOTE_TMPDIR` | Remote temporary directory | `/tmp` |
| `TENRYU_REMOTE_SSH` | SSH command | `ssh` |
| `TENRYU_REMOTE_SSH_OPTS` | Extra options inserted after the ssh command (word-split; no spaces inside values) | empty |
| `TENRYU_REMOTE_SCP_OPTS` | Extra options inserted after the scp command (word-split; no spaces inside values) | empty |
| `TENRYU_REMOTE_SCP` | SCP command | `scp` |
| `TENRYU_REMOTE_RSYNC` | rsync command | `rsync` |

```sh
TENRYU_REMOTE_HOST=parma TENRYU_REMOTE_REPO=... tools/assist/assist.py lint-deck deck.py --tenryu tools/assist/tenryu_remote.sh
```

Interpolated paths and arguments must contain only letters, digits, `_./+=:@-`; spaces and quotes are not supported. rsync callers can use the standard `RSYNC_RSH` environment variable for equivalent per-connection options.

## Work-item skills (CC / codex pairs)

Each LLM-performed work item gets a dedicated skill, in two variants for the two
providers actually used: a Claude Code skill under `.claude/skills/<name>/`
(auto-available to repo sessions) and a codex variant kept canonically under
`tools/assist/skills/codex/<name>/` and installed with:

```sh
cp tools/assist/skills/codex/<name>/SKILL.md ~/.codex/skills/<name>/SKILL.md
```

Invocation convention: for codex providers prefix the prompt with `$<name>`;
for Claude providers instruct "Invoke the <name> skill". Current items:

- `tenryu-mesh-1d` — 1D initial-mesh design/revision (mesh work item only).
- `tenryu-namelist` — complete 1D deck authoring from an experimental specification (block order, key essentials, laser power convention, validate/lint/freeze procedure, error dictionary); its codex variant is the prompt head of `generate-deck`.

Planned items follow the same pattern (zoning repair,
run forensics, plain-language run reports).

## TENRYU Studio integration

TENRYU Studio (gui/) exposes this harness in its アシスタント view: deterministic
verbs run on the selected server profile (`python3 tools/assist/assist.py …` inside
the server-side checkout), while `generate-deck` runs on the local machine with
`--tenryu tools/assist/tenryu_remote.sh` driving the server binary. Per-profile ssh
options are forwarded via `TENRYU_REMOTE_SSH_OPTS` / `TENRYU_REMOTE_SCP_OPTS` and
`RSYNC_RSH`. TENRYU Studio bundles this directory as an app resource and copies it onto
the server mirror, so the server checkout needs only docs/ and src/. Studio workdirs
live under `generate/<stamp>/` in the app configuration folder. Design:
docs/design/gui_assistant_integration_20260902.md.

## Mesh recommendations from experimental conditions

`recommend-mesh` reads the shipped convergence table and emits deterministic JSON containing
`recommendation`, reference IDs/distances/weights, raw and adjusted ceilings, `loo` statistics,
`confidence`, `flags`, `warnings`, `budget`, validation evidence and a ready-to-paste
`mesh_block` string. `--mesh-out` also writes that string as a Python block. It needs only the
standard library (Python 3.9+, retaining the assistant floor); it makes no LLM call.

```bash
python tools/assist/assist.py recommend-mesh \
  --conditions tools/assist/examples/mesh_conditions/gxii_gaussian.json \
  -o recommendation.json --mesh-out mesh.py
python tools/assist/assist.py recommend-mesh --deck case.py --deck-out recommended.py --tenryu build/tenryu
python tools/assist/assist.py generate-deck spec.md --conditions conditions.json \
  --out-deck case.py --tenryu build/tenryu
```

The conditions JSON fields are (each layer requires explicit `A` and `Z`; missing values
are errors, never guessed from names; campaign CD uses A=7, Z=3.5, Al uses A=26.98, Z=13):

| Field | Meaning |
|---|---|
| `wavelength_nm` | Positive laser wavelength |
| `geometry` | `planar` (default), `spherical`, or `cylindrical` |
| `layers` | Inner-to-outer list of `{material, A, Z, rho_gcc, thickness_cm}`; numeric fields finite and positive. A void layer `{material, "void": true, thickness_cm}` (optional `rho_gcc`, default 1e-9) is vacuum inside the target — behind a planar foil with a free rear surface, inside a hollow shell, between layers; it needs no `A`/`Z`, and the outermost layer must be material |
| `r_min_cm`, `r_max_cm` | Inner radius (default 0); domain outer edge (default target outer edge) |
| `t_end_s` | Run duration |
| `pulse` | One of the forms below |
| `physics` | Optional `{eos, radiation_enabled, temperature_model, conduction_solver}`; defaults to `tmat`, false, `2T`, `implicit` |

`pulse` accepts `shape` (`square`, `gaussian`, `foot_main`, `picket`, `ramp`, `long_low`),
`peak_intensity_W_cm2`, `duration_s` (default t_end); or replace peak with `energy_J` and
`spot_radius_cm`/`area_cm2`. Gaussian accepts `fwhm_s`/`center_s`; square and long_low accept
`rise_s` (default 1e-10). Other named shapes retain the campaign stage ratios. For arbitrary
levels/timing supply `intensity_table: [[time_s, W_cm2], ...]`, or
`power_table: [[time_s, W], ...]` with optional `area_cm2`. Times must strictly increase;
values must be finite and nonnegative. Power defaults to the solver's reference area
(planar 1 cm2, spherical 4*pi*R_target^2, cylindrical 2*pi*R_target per unit length).
The spot convention is an equivalent uniform disk. Other spatial profiles need an explicit
appropriate area. Unknown pulse keys are rejected.

Examples include [GXII Gaussian](examples/mesh_conditions/gxii_gaussian.json),
[foam on solid](examples/mesh_conditions/foam_on_solid.json),
[1053 nm foil](examples/mesh_conditions/infrared_foil.json), and
[Al foil](examples/mesh_conditions/aluminium_foil.json). The learned method, input coverage,
LOO calibration and exact solver/lint integrity boundary are in
[the design](../../docs/design/mesh_recommendation_from_campaign_20260908.md).

`--tenryu`, then `TENRYU_BIN`, then `build/tenryu` select the optional binary, as for
`lint-deck`. With a new binary, --deck uses solver-observed conditions from
`validate --mesh-preview`; a placeholder mesh is allowed if every material interface is one of its
nodes (the preview samples the layers at the placeholder's cell centres, so an interface between two
nodes moves to a node: a uniform 2 µm mesh read a 243 µm interface as 244 µm). Offline/old-binary deck
extraction requires a literal `MESH_EXPERIMENTAL_CONDITIONS = {...}` assignment, otherwise
use --conditions. Both flags may be combined to validate material definitions in a real
deck. Conditions-only validation uses a clearly identified ideal-gas mesh-check deck for
the supplied material atoms; validate the production EOS deck separately. No binary means `unvalidated`.
A deck's VOID material inside the target (between `r_min` and the first material, or between
materials) becomes a void layer; before 2026-09-29 such decks were refused with "material Z
must be finite and positive" (the solver exports void materials with Z = 0). The 1D solver keeps
the cells of every void run evenly spaced between the faces around it (main 125a94572; before it
only the exterior padding followed the surface and an interior void aborted the run) but has no
contact model: a void that closes stops the run. The recommendation carries the flag
`interior_void` and a warning to make the void wider than the travel of the faces around it
(decks and measurements in `tests/data/decks/interior_void_1d/`).

The block uses planar areal mass, cylindrical line mass, or spherical cell mass and splits
the target into three regions. The ablation zone reaches the deeper of the predicted ablated
depth (mu_abl_total_g_cm2: solver preview, otherwise the Python formation integral) and, for
learned recommendations, the campaign's laser-side half; its cells are capped at
0.95*area(R0)*surface ceiling, the ablation rule's reference-area measure, across layers and
fill. A spherical or cylindrical gas fill (innermost layer below 0.1 of the densest density)
outside that zone gets no band of the recommender and the 40-cell segment minimum; the
campaign did not measure stagnation, so an imploding fill needs its own convergence check.
A void layer inside the target is its own 40-cell segment in no region (uniform cells in a
planar void, equal-mass cells in a curved one); the payload starts at the innermost material
layer, depths and features skip the void, and the case is `extrapolation`.
The unablated payload keeps the local rule rho*dr <= 2e-5 g/cm2 (or the solver's finer shock
ceiling) through a core and pieces of radius ratio 1.2, the construction the solver uses
for its injected payload band. With a probe preview the solver's injected caps join the
count. Interval edges within 1e-6 of the target thickness merge (layer edges win). Each
segment needs max(40, capped cells + ratio-limited transitions + 2); a segment whose
requirement exceeds its share of the zoning monitor has its preferred measures scaled down
(budget.segments[].profile_scale) so the solver's largest-deficit allocation meets it
without inflating the total. The monitor integrals replicate the solver's quadrature.
Padding has a constant terminal profile weight and no band. Explicit caps include 5%
headroom. Inspect recommendation.mu_abl_total_g_cm2, mu_abl_source, budget.segments,
budget.band_regions and budget.solver_band_regions.
A capable binary validates the assembled mesh (at most eight attempts). It first folds the
solver's injected bands, shock ceiling and ablated depth from the preview into the
construction; band/box/chain and segment-budget certificates rebuild it with margins 1.25,
1.45, 1.70, then grow counts by 1.5; width conflicts lower dr_min. An ablation-rule
requirement violation extends the ablation zone to the worst cell's depth. `validated`
needs exit code 0, no ablation- or shock-rule violation (the solver only reports shock
violations of zoning_intent decks; the margin ladder refines such a mesh) and an achieved
surface cell measure at or below the recommendation.
Attempts include parsed certificate numbers.
Achieved surface mass is the integrated cell measure divided by the outer reference area. Unknown empirical
keys trigger `legacy_binary`: keep calibrated enforce plus explicit bands, at the cost of
possible over-resolution. `unconverged_reference` means even the finest run was not proven
converged. `extrapolation` means a-priori fallback; run a surface-mass-halving convergence
pair before production. Without a binary this uses the material-aware calibrated
formation integral itself (factor 1), with no extra multiplier or ladder cap; the solver's
own formation ceiling is used when available. The Al example emits 9.131233090608189e-7
g/cm2, 2381 estimated cells and a 3.382 nm surface width ceiling (`unvalidated`).
The campaign tool and its design describe the observable tests.
`sparse_evidence` means nearby measurements reduced the permitted relaxation of the
a-priori ceiling. Inspect `recommendation.evidence_distance`, `evidence_radius`, and
`allowed_apriori_factor`: the median training spacing is 0.36537242214822635, with linear
shrinkage to factor one at that distance. LOO final safety is 22/22 (100%), with zero
cases above both the a-priori and measured ceilings; the margin is
0.2840422014231186 dex (90% quantile, 20/22 safe before the gate). Exact matches remain
0.95*a_conv, as do conditions within a normalized feature distance 0.025 of a converged case
with the same geometry, layer count and waveform class. Every pulse is first frozen as the
solver freezes a beam power callable (absolute times k*2^-40 s, bisected up to 7 times where
the curve bends), and the features, class and duration are read from that table, so a table
exported by the solver and the conditions it came from give identical numbers even where
the pulse jumps. A pulse belongs to a waveform class when its peak-normalized shape on its
own energy window (0.5 %–99.5 % of the pulse energy) is within an L1 distance 0.10, whatever
the tail cut; coverage requires an effective duration (integral of I over [0, t_end]
divided by the peak) of 0.54–3.94 ns, so a run continuing after the pulse stays covered.
The planar low-density unconverged class (C15) applies only when the predicted ablated
depth reaches a layer of 0.05 g/cc or less.
LOO is a calibration statistic, not a convergence guarantee.
These flags are not errors: exit **0** means a recommendation was emitted (inspect validation
status); exit **2** means bad input, missing explicitly requested binary, or failed validation.

In `--deck` mode, use `--deck-out FILE` to write the fully assembled deck. The JSON
`mesh_block` and `--mesh-out` already contain preserved Mesh keys such as `motion`;
do not feed that block through `replace_mesh` again. The assembled output preserves each
key exactly once. In `--conditions` mode, paste the Mesh block into your own production
deck and merge the indicated Numerics retry companion. Without `--deck`, `--deck-out`
writes the synthetic mesh-check candidate only if binary validation ran; otherwise it is
an input error (exit 2). A failed validation still returns exit 2; any written candidate
is not a validated production deck.

The recommender, shipped table and binary must come from the same checkout state because
the table digest is compiled into the binary; `reference_table_mismatch` indicates a stale copy.
For this flag, the validate loop removes `empirical` and retries with calibrated enforce
and explicit finer bands. The warning names the tool's digest; the learned efficiency
benefit is lost until the table and binary agree. Other empirical errors retain the first
matching log line verbatim in `validation.attempts[].error` (including any diagnostic log
prefix). The independent `empirical-mesh-integrity` lint explicitly reports the table's
digest and the differing deck digest; its hard-failure contract is unchanged.
`lint-deck` independently recomputes empirical provenance/values and rejects
`empirical-mesh-integrity` violations; `learned-surface-resolution` warns on coarser surface
cells. The only deck claim the recomputation uses is the a-priori baseline, and only within
the solver's own 3 % provenance tolerance, so a block computed offline (the tool's replica
of the ceiling) verifies exactly on the numbers it was computed from. This integrity lint is
required before production use of empirical relaxation.
`generate-deck` includes the same recommendation from --conditions, template lint or the
previous iteration beside its resolution section, with a deterministic context hash in
the journal. If a failed validation leaves a later lint without a recommendation, retain
the previous section and record its source in mesh_recommendation_context_fallback. No additional provider call is made.
