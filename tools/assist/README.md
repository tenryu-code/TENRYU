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

`generate-deck` generates a deck from a specification and retries with validator feedback. When a template is given, or from the second iteration on, the prompt carries a `RESOLUTION REQUIREMENTS` section with that summary.
Usage: `tools/assist/assist.py generate-deck SPEC --out-deck FILE [--tenryu PATH] [--template FILE] [--intent INTENT.json] [--baseline FROZEN.json] [--max-iters N] [--workdir DIR] [--config FILE]`

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

`digest` requires `h5py` for history reductions and gracefully degrades when it is unavailable. `lint-deck` and `freeze-baseline` require a built `tenryu` binary, selected with `--tenryu` or found at `./build/tenryu`.

`generate-deck` requires `enabled = true` and a configured `deck_design` role. Its exit codes are: 0 accepted, 2 failure/disabled/exhausted, and 3 when the model asks a clarifying question (`UNCERTAIN`). Every model call and accepted deck is journaled.

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

Planned items follow the same pattern (full-deck generation, zoning repair,
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
