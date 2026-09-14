---
name: tenryu-docs-qa
description: TENRYU に関する質問（ビルド・実行・namelist キーと既定値・物理と数値法・出力・GUI・ソースの挙動）に、ローカルチェックアウトの文書とソースだけを根拠に答えるスキル（codex headless 変種）。tools/assist の質問動詞（assist.py ask）が組み立てるプロンプト本文としても、codex exec で直接呼ぶ入口としても使う。Use when asked any question about TENRYU that must be answered from the local checkout's documents and source with citations; read-only, no deck generation, no code edits.
metadata:
  short-description: TENRYU 質問応答（文書とソースを根拠に回答、codex headless 変種）
---

# TENRYU question answering (work-item skill)

Scope: answer one question about TENRYU — how to build and run it, what a namelist key means and what its default is, the physics and numerics of a kernel, the output files, the Studio GUI, or what the source code actually does — using ONLY the documents and source in the local checkout you are running in. Out of scope: editing files, running the solver or a build, generating decks (that is the deck-generation verb), and questions unrelated to TENRYU.

## Where to look (in priority order)

| Question about | Read first | Then, if needed |
|---|---|---|
| Building, running, restart, command line | `docs/TUTORIAL_ja.md`; `docs/site/ja/use/requirements.html`, `building.html`, `running.html` | `src/main.cpp` |
| A namelist key: meaning, type, allowed values | `docs/SPECIFICATION.md` §6.4.<block> (line ranges are in the document map); `docs/site/ja/use/namelist.html` | `src/core/namelist/builder.cpp` — each `enforce_known_keys(..., "<Block[.sub]>", {...})` list is the complete set of keys the solver accepts in that block |
| A default value | `docs/SPECIFICATION.md` §9.1 and the key's own §6.4 entry | `src/core/namelist/builder.cpp`, `src/core/config.hpp` |
| Physics or numerics of a kernel | `docs/site/ja/physics/**` (curated and verified against the canon) | `docs/NUMERICS.md` sections listed in the map; `docs/sections/*.md` |
| Time-step control | `docs/site/ja/overview/timestep.html` | `docs/NUMERICS.md` §2.2 |
| Output files, HDF5 layout, plotting | `docs/site/ja/use/outputs.html`; `docs/OUTPUT_SCHEMA.md`; `docs/POSTPROCESSING.md` | `src/io/` |
| Studio GUI | `gui/manual/` | `docs/site/ja/use/gui.html` |
| Verification, golden references | `docs/site/ja/verification/`; `docs/VERIFICATION.md` | `examples/verification/`, `tests/` |
| Example decks | `examples/**/*.py` (listed in the map) | — |
| What the code does in a given situation | `src/` via grep; cite `file:line` | design notes under `docs/design/` |
| Architecture, module map | `docs/site/ja/overview/architecture.html`; `docs/ARCHITECTURE.md` | — |

## Procedure

1. If the prompt does not already contain a section headed `== DOCUMENT MAP ==`, generate the map first: run `python3 tools/assist/assist.py docmap --keys-out /tmp/tenryu_namelist_keys.txt` from the checkout root. It prints the map (every reader page with its headings; heading-to-line-range tables for the canonical documents; the example decks) and writes the namelist key index (`<Context>\t<key>` per line). When the map is already in the prompt, do not regenerate it.
2. Classify the question with the table above and read the listed sources. Read only the line ranges the map gives you (`sed -n 'A,Bp' FILE`, or a read tool with an offset); `docs/NUMERICS.md` and `docs/SPECIFICATION.md` are far too large to read whole.
3. For every namelist key you mention, confirm that it exists: it must appear in the key index or in an `enforce_known_keys` list in `src/core/namelist/builder.cpp`. Write keys with their full block path (for example `Numerics.dt.initial_s`). If a key is not there, say that this checkout does not accept it.
4. For behavior questions, grep `src/` and read the code path before answering. Distinguish "documented" (cite the document) from "read from the source" (cite `file:line`).
5. Compose the answer.

## Rules

- Never invent a key, a default, a unit, a file name, or a behavior. If the documents and the source do not settle the question, say so explicitly — 「この checkout の文書には記載なし」/ "not documented in this checkout" — and list where you looked.
- TENRYU uses cgs units plus eV for temperatures throughout. State the unit whenever you give a value.
- The canon for behavior and namelist is `docs/site/ja/**` together with `docs/SPECIFICATION.md`; `docs/NUMERICS.md` for equations and discretization; `src/` when the documents are silent or disagree. If two sources disagree, report both with citations. Do not pick one silently.
- Use standard terminology. If a document uses a project-internal label, explain in plain words what it refers to (which file, which operator, which quantity).
- Do not modify any file, do not run the solver, do not build. Reading, grep, and glob are the only actions.
- Answer in the language of the question (a Japanese question gets a Japanese answer). Keep code, paths, and keys verbatim.
- Lead with the answer, then the details. No speculation; no padding.

## Output contract

1. The answer.
2. A section titled `根拠 / Evidence`: one citation per line, repository-relative, as `path:line` or `path:line-line` (for an HTML page, `path` followed by the heading text). Every factual claim must trace to a line here.
3. An optional section titled `未確認 / Not verified`: what you could not confirm, and why.
