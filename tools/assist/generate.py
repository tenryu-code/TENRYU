"""LLM-driven TENRYU deck generation with validator feedback."""

import json
import os
import shutil
import sys
import tempfile

from tools.assist import providers
from tools.assist.config import AssistConfigError, load_config
from tools.assist.deck_lint import (
    _load_lint_inputs,
    _missing_binary_error,
    _resolve_tenryu,
    run_deck_lint,
)
from tools.assist.journal import JournalWriter, sha256_hex
from tools.assist.providers import AssistDisabledError
from tools.assist.tomlmini import TomlSubsetError


PROMPT_HEADER = """You are generating a TENRYU input deck (a Python namelist file).
Rules:
- Output the COMPLETE deck file between two marker lines: a line 'BEGIN_DECK' and a line 'END_DECK'. No prose, no code fences, nothing else outside the markers.
- The deck must start with 'from tenryu_namelist import *'.
- Satisfy every requirement in the SPEC section. Do not invent physics settings the spec does not imply; prefer omitting a key so solver defaults apply.
- If a requirement is ambiguous or missing information you need, output instead a single line starting with 'UNCERTAIN: ' followed by one concise question. Do not guess.
"""
FEEDBACK_HEADER = (
    "ITERATION-FEEDBACK (previous attempt failed validation; fix ONLY what is needed):"
)


def extract_deck(response_text: str):
    """Return ('deck', text) | ('uncertain', question) | ('missing', None)."""
    lines = response_text.splitlines(keepends=True)
    begin_index = None
    for index, line in enumerate(lines):
        if line.strip() == "BEGIN_DECK":
            begin_index = index
            break
    if begin_index is not None:
        for index in range(begin_index + 1, len(lines)):
            if lines[index].strip() == "END_DECK":
                return "deck", "".join(lines[begin_index + 1 : index])

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("UNCERTAIN: "):
            return "uncertain", stripped[len("UNCERTAIN: ") :]
    return "missing", None


def generate_deck(
    cfg,
    spec_text,
    out_deck,
    tenryu,
    workdir,
    journal,
    pins=None,
    baseline=None,
    template_text=None,
    max_iters=10,
    lint_fn=None,
    invoke_fn=None,
) -> dict:
    """Generate and validate a TENRYU deck, feeding lint failures back."""
    if lint_fn is None:
        lint_fn = lambda deck_path: run_deck_lint(  # noqa: E731
            deck_path, tenryu, pins=pins, baseline=baseline
        )
    if invoke_fn is None:
        invoke_fn = providers.invoke

    workdir = os.fspath(workdir)
    out_deck = os.fspath(out_deck)
    os.makedirs(workdir, exist_ok=True)
    feedback = ""
    previous_deck = ""
    last_lint = None

    for i in range(max_iters):
        prompt = PROMPT_HEADER + "\n== SPEC ==\n" + spec_text
        if template_text and i == 0:
            prompt += (
                "\n== STARTING TEMPLATE (modify as needed) ==\n" + template_text
            )
        if i > 0:
            prompt += (
                "\n== "
                + FEEDBACK_HEADER
                + " ==\n"
                + feedback
                + "\n== PREVIOUS DECK ==\n"
                + previous_deck
            )

        response = invoke_fn(cfg, "deck_design", prompt, workdir, journal)
        response_kind, response_value = extract_deck(response)
        if response_kind == "uncertain":
            journal.append(
                "generation_uncertain",
                {"iteration": i, "question": response_value},
            )
            return {
                "status": "uncertain",
                "question": response_value,
                "iterations": i + 1,
            }
        if response_kind == "missing":
            feedback = "response did not contain BEGIN_DECK/END_DECK markers"
            previous_deck = ""
            continue

        deck_text = response_value
        deck_path = os.path.join(workdir, "deck_iter{0}.py".format(i))
        with open(deck_path, "w", encoding="utf-8") as stream:
            stream.write(deck_text)
        deck_sha256 = sha256_hex(deck_text)
        journal.append(
            "deck_iteration",
            {"iteration": i, "deck_sha256": deck_sha256},
        )

        payload, code = lint_fn(deck_path)
        if isinstance(payload, dict) and "error" in payload:
            journal.append(
                "tool_error",
                {"iteration": i, "error": payload["error"]},
            )
            return {
                "status": "tool_error",
                "iterations": i + 1,
                "error": payload["error"],
                "lint": payload,
            }
        last_lint = payload
        lint_record = {"iteration": i, "exit_code": code}
        lints = payload.get("lints") if isinstance(payload, dict) else None
        if isinstance(lints, list):
            lint_record["hard_failed_lints"] = sum(
                1
                for lint in lints
                if isinstance(lint, dict)
                and lint.get("severity") == "hard"
                and not lint.get("ok", False)
            )
        intent_lock = payload.get("intent_lock") if isinstance(payload, dict) else None
        if isinstance(intent_lock, list):
            lint_record["not_ok_intent_pins"] = sum(
                1
                for entry in intent_lock
                if isinstance(entry, dict) and not entry.get("ok", False)
            )
        journal.append("lint_result", lint_record)

        if code == 0:
            os.makedirs(os.path.dirname(os.path.abspath(out_deck)), exist_ok=True)
            shutil.copyfile(deck_path, out_deck)
            journal.append(
                "deck_accepted",
                {"deck_sha256": deck_sha256, "out_path": out_deck},
            )
            return {
                "status": "accepted",
                "deck_path": out_deck,
                "iterations": i + 1,
                "lint": payload,
            }

        feedback = json.dumps(payload, sort_keys=True)[:6000]
        previous_deck = deck_text

    journal.append("generation_failed", {"iterations": max_iters})
    return {
        "status": "exhausted",
        "iterations": max_iters,
        "last_lint": last_lint,
    }


def _read_text(path, label):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return stream.read()
    except OSError as error:
        print(
            "assist generate-deck: cannot read {0}: {1}".format(label, error),
            file=sys.stderr,
        )
        return None


def main_generate_deck(args, lint_fn=None) -> int:
    """CLI entry point for generate-deck."""
    try:
        cfg = load_config(cli_path=args.config)
    except (AssistConfigError, TomlSubsetError) as error:
        print("assist: config error: {0}".format(error), file=sys.stderr)
        return 2

    if not cfg.enabled:
        print(
            "assist generate-deck: assistant is disabled (enable it in "
            "assistant.toml; TENRYU_ASSIST_DISABLE must be unset)",
            file=sys.stderr,
        )
        return 2

    tenryu = _resolve_tenryu(args.tenryu)
    if tenryu is None:
        _missing_binary_error(args.tenryu)
        return 2

    spec_text = _read_text(args.spec, "spec")
    if spec_text is None:
        return 2
    template_text = None
    if args.template is not None:
        template_text = _read_text(args.template, "template")
        if template_text is None:
            return 2

    try:
        pins, baseline = _load_lint_inputs(args.intent, args.baseline)
    except ValueError as error:
        print("assist generate-deck: {0}".format(error), file=sys.stderr)
        return 2

    workdir = args.workdir or tempfile.mkdtemp(prefix="assist_deck_")
    os.makedirs(workdir, exist_ok=True)
    journal = JournalWriter(os.path.join(workdir, "journal.jsonl"))
    try:
        result = generate_deck(
            cfg,
            spec_text,
            args.out_deck,
            tenryu,
            workdir,
            journal,
            pins=pins,
            baseline=baseline,
            template_text=template_text,
            max_iters=args.max_iters,
            lint_fn=lint_fn,
        )
    except (RuntimeError, AssistDisabledError) as error:
        print("assist generate-deck: {0}".format(error), file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "tool_error":
        print(
            "assist generate-deck: tool error (not a deck defect): {0}".format(
                result["error"]
            ),
            file=sys.stderr,
        )
        return 2
    if result["status"] == "accepted":
        return 0
    if result["status"] == "uncertain":
        return 3
    return 2
