# EXPERIMENTAL: TENRYU Assistant Infrastructure

This directory provides deterministic assistant infrastructure around TENRYU. LLM-driven verbs land in later milestones, and this layer never runs inside the solver.

The assistant defaults to OFF with `enabled = false`. `TENRYU_ASSIST_DISABLE` is the kill switch and beats every other setting when its value is `1`, `true`, `TRUE`, or `yes`. Deterministic verbs work regardless of this setting; only LLM invocations are gated.

Configuration is selected in this order:

1. `--config PATH`
2. `TENRYU_ASSIST_CONFIG`
3. `./assistant.toml`
4. `~/.tenryu/assistant.toml`
5. Built-in defaults

| Key | Meaning |
| --- | --- |
| `enabled` | Boolean opt-in for LLM invocation |
| `[providers.NAME]` | Non-empty `command` template and `model` |
| `[roles]` | Known role to provider name or `dry_run` |
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

`zoning-report` reports the mass-form ablation-zoning lint from 1D probe outputs.
Usage: `tools/assist/assist.py zoning-report OUTPUT_DIR [--kappa F] [--ablate-margin F] [--strict] [-o FILE]`

`promote-zoning` compiles observed ablated cells into a suggested hard spherical-measure band.
Usage: `tools/assist/assist.py promote-zoning OUTPUT_DIR [--kappa F] [--margin-cells N]`

`lint-deck` runs the solver's validators and reports mesh, intent-lock, and baseline checks.
Usage: `tools/assist/assist.py lint-deck DECK [--tenryu PATH] [--baseline FROZEN.json] [--intent INTENT.json] [-o FILE] [--keep-tmp]`

`generate-deck` generates a deck from a specification and retries with validator feedback.
Usage: `tools/assist/assist.py generate-deck SPEC --out-deck FILE [--tenryu PATH] [--template FILE] [--intent INTENT.json] [--baseline FROZEN.json] [--max-iters N] [--workdir DIR] [--config FILE]`

`freeze-baseline` freezes a deck for later baseline comparison.
Usage: `tools/assist/assist.py freeze-baseline DECK [--tenryu PATH] [-o FILE]`

`digest` requires `h5py` for history reductions and gracefully degrades when it is unavailable. `lint-deck` and `freeze-baseline` require a built `tenryu` binary, selected with `--tenryu` or found at `./build/tenryu`.

`generate-deck` requires `enabled = true` and a configured `deck_design` role. Its exit codes are: 0 accepted, 2 failure/disabled/exhausted, and 3 when the model asks a clarifying question (`UNCERTAIN`). Every model call and accepted deck is journaled.

Every LLM invocation is journaled as JSONL with model identity and prompt/response hashes. Replay never re-queries an LLM.

## Remote execution wrapper

`tenryu_remote.sh` lets a CUDA-less workstation drive a `tenryu` binary on a remote GPU host while presenting the local CLI subset used by the assistant verbs.

| Environment variable | Meaning | Default |
| --- | --- | --- |
| `TENRYU_REMOTE_HOST` | SSH host | Required |
| `TENRYU_REMOTE_REPO` | Remote repository and working directory | Required |
| `TENRYU_REMOTE_BIN` | Remote `tenryu` binary | `$TENRYU_REMOTE_REPO/build/tenryu` |
| `TENRYU_REMOTE_TMPDIR` | Remote temporary directory | `/tmp` |
| `TENRYU_REMOTE_SSH` | SSH command | `ssh` |
| `TENRYU_REMOTE_SCP` | SCP command | `scp` |
| `TENRYU_REMOTE_RSYNC` | rsync command | `rsync` |

```sh
TENRYU_REMOTE_HOST=parma TENRYU_REMOTE_REPO=... tools/assist/assist.py lint-deck deck.py --tenryu tools/assist/tenryu_remote.sh
```

Interpolated paths and arguments must contain only letters, digits, `_./+=:@-`; spaces and quotes are not supported.

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
