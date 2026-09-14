"""Question answering about TENRYU with deterministic evidence checks."""

import builtins
import json
import keyword
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional

from tools.assist import providers
from tools.assist.config import AssistConfigError, load_config
from tools.assist.docmap import build_docmap
from tools.assist.journal import JournalWriter, sha256_hex
from tools.assist.providers import AssistDisabledError
from tools.assist.tomlmini import TomlSubsetError


ROLE = "question_answering"
SKILL_RELPATH = "tools/assist/skills/codex/tenryu-docs-qa/SKILL.md"
DEFAULT_WORKDIR_PARENT = "~/.tenryu/ask"
MAP_SECTION = "== DOCUMENT MAP =="
HISTORY_SECTION = "== CONVERSATION SO FAR =="
QUESTION_SECTION = "== QUESTION =="
HISTORY_ANSWER_CHAR_CAP = 6000
CITATION_RE = re.compile(
    r"(?<![A-Za-z0-9_./\-])"
    r"((?:docs|src|examples|tools|gui|gui-common|tests|ops)/[A-Za-z0-9_./\-]*?"
    r"\.(?:md|html|py|cpp|hpp|cu|cuh|toml|sh|json|txt|js|ts|tsx|cmake|ya?ml))"
    r"(?::(\d+)(?:-(\d+))?)?(?![A-Za-z0-9_])"
)


def load_skill_body(repo_root: str) -> str:
    """Load the question-answering skill body without YAML frontmatter."""
    path = os.path.join(repo_root, SKILL_RELPATH)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as stream:
        text = stream.read()

    lines = text.splitlines(keepends=True)
    if lines and lines[0].rstrip("\r\n") == "---":
        for index in range(1, len(lines)):
            if lines[index].rstrip("\r\n") == "---":
                lines = lines[index + 1 :]
                break
    while lines and not lines[0].strip():
        lines.pop(0)
    return "".join(lines)


def load_history(path: str, max_turns: int) -> List[dict]:
    """Load the last conversation turns and their saved answers."""
    if not os.path.exists(path):
        return []

    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
            except (TypeError, ValueError):
                continue
    records = records[-max_turns:] if max_turns > 0 else []

    history_dir = os.path.dirname(path)
    for record in records:
        answer_path = os.path.join(history_dir, record["answer_path"])
        if not os.path.isfile(answer_path):
            record["answer"] = "(answer file missing)"
            continue
        with open(answer_path, "r", encoding="utf-8") as stream:
            answer = stream.read()
        if len(answer) > HISTORY_ANSWER_CHAR_CAP:
            answer = answer[:HISTORY_ANSWER_CHAR_CAP] + "\n... (truncated)"
        record["answer"] = answer
    return records


def build_prompt(
    skill_body: str,
    docmap_text: str,
    history: List[dict],
    question: str,
) -> str:
    """Assemble the skill, map, optional history, and current question."""
    blocks = [
        skill_body.rstrip(),
        MAP_SECTION + "\n" + docmap_text.rstrip(),
    ]
    if history:
        turns = []
        for record in history:
            turns.append(
                "Q{0}: {1}\nA{0}: {2}".format(
                    record["turn"], record["question"], record["answer"]
                ).rstrip()
            )
        blocks.append(HISTORY_SECTION + "\n" + "\n\n".join(turns))
    blocks.append(QUESTION_SECTION + "\n" + question.strip())
    return "\n\n".join(blocks) + "\n"


def extract_citations(answer: str) -> List[dict]:
    """Extract unique repository-relative citations in encounter order."""
    citations = []
    seen = set()
    for match in CITATION_RE.finditer(answer):
        citation = match.group(0)
        if citation in seen:
            continue
        seen.add(citation)
        citations.append(
            {
                "citation": citation,
                "path": match.group(1),
                "start": int(match.group(2)) if match.group(2) else None,
                "end": int(match.group(3)) if match.group(3) else None,
            }
        )
    return citations


def check_citations(answer: str, repo_root: str) -> dict:
    """Verify that cited files and optional line ranges exist."""
    root = os.path.abspath(repo_root)
    verified = []
    unverified = []
    for entry in extract_citations(answer):
        citation = entry["citation"]
        resolved = os.path.normpath(os.path.join(root, entry["path"]))
        try:
            inside = os.path.commonpath([root, resolved]) == root
        except ValueError:
            inside = False
        if not inside:
            unverified.append(
                {"citation": citation, "reason": "path escapes the checkout"}
            )
            continue
        if not os.path.isfile(resolved):
            unverified.append(
                {"citation": citation, "reason": "file not found"}
            )
            continue
        if entry["start"] is not None:
            start = entry["start"]
            end = entry["end"] if entry["end"] is not None else start
            if start > end:
                unverified.append(
                    {"citation": citation, "reason": "start > end"}
                )
                continue
            with open(resolved, "r", encoding="utf-8", errors="replace") as stream:
                line_count = len(stream.read().splitlines())
            if start < 1 or end > line_count:
                unverified.append(
                    {
                        "citation": citation,
                        "reason": "line out of range (file has {0} lines)".format(
                            line_count
                        ),
                    }
                )
                continue
        verified.append(citation)
    return {"verified": verified, "unverified": unverified}


def code_regions(answer: str) -> List[str]:
    """Return fenced code followed by inline code outside fences."""
    fence_re = re.compile(r"```[^\n]*\n(.*?)```", re.S)
    fenced = [match.group(1) for match in fence_re.finditer(answer)]
    without_fences = fence_re.sub("", answer)
    inline = [
        match.group(1)
        for match in re.finditer(r"`([^`\n]*)`", without_fences)
    ]
    return fenced + inline


def check_keys(answer: str, key_index: Dict[str, List[str]]) -> dict:
    """Check unambiguous namelist-key mentions found in code regions."""
    top_level_blocks = {
        re.sub(r"\[.*\]$", "", context.split(".", 1)[0])
        for context in key_index
    }
    all_keys = {key for keys in key_index.values() for key in keys}
    ignored_assignments = set(keyword.kwlist) | set(dir(builtins)) | {"lambda"}
    known = set()
    unknown = set()

    dotted_re = re.compile(r"\b([A-Z][A-Za-z]*(?:\.[a-z_][A-Za-z0-9_]*)+)\b")
    # Deck assignments use key=value; exclude spaced assignments, flags, and qualified names.
    assignment_re = re.compile(r"(?<![-\w.])([a-z_][a-z0-9_]*)=(?!=)")
    for region in code_regions(answer):
        for match in dotted_re.finditer(region):
            dotted = match.group(1)
            context, key = dotted.rsplit(".", 1)
            if context.split(".", 1)[0] not in top_level_blocks:
                continue
            if key in key_index.get(context, []) or key in key_index.get(
                context + "[k]", []
            ):
                known.add(dotted)
            else:
                unknown.add(dotted)

        for match in assignment_re.finditer(region):
            name = match.group(1)
            if name in ignored_assignments:
                continue
            if name in all_keys:
                known.add(name)
            else:
                unknown.add(name)

    return {"known": sorted(known), "unknown": sorted(unknown)}


def parse_key_index_text(text: str) -> Dict[str, List[str]]:
    """Parse the tab-separated key-index format produced by docmap.py."""
    result = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        context, key = line.split("\t", 1)
        result.setdefault(context, []).append(key)
    return result


def format_check_line(checks: dict) -> str:
    """Format the deterministic evidence summary shown after an answer."""
    citation_checks = checks["citations"]
    verified = citation_checks["verified"]
    unverified = citation_checks["unverified"]
    if not verified and not unverified:
        citations_text = "citations: none found"
    else:
        citations_text = "citations {0} verified, {1} unverified".format(
            len(verified), len(unverified)
        )
        if unverified:
            details = "; ".join(
                "{0}: {1}".format(entry["citation"], entry["reason"])
                for entry in unverified[:8]
            )
            if len(unverified) > 8:
                details += ", ..."
            citations_text += " ({0})".format(details)

    key_checks = checks["keys"]
    known = key_checks["known"]
    unknown = key_checks["unknown"]
    if not known and not unknown:
        keys_text = "namelist keys: none found"
    else:
        keys_text = "namelist keys {0} known, {1} unknown".format(
            len(known), len(unknown)
        )
        if unknown:
            details = ", ".join(unknown[:8])
            if len(unknown) > 8:
                details += ", ..."
            keys_text += " ({0})".format(details)

    return "根拠の検査 / evidence check: {0}; {1}".format(
        citations_text, keys_text
    )


def _last_history_turn(path: str) -> int:
    last_turn = 0
    if not os.path.exists(path):
        return last_turn
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except (TypeError, ValueError):
                continue
            last_turn = record["turn"]
    return last_turn


def ask(
    cfg,
    question,
    repo_root,
    workdir,
    journal,
    max_history=6,
    invoke_fn=None,
    timeout_s=900,
) -> dict:
    """Answer a question, verify its evidence, and persist the turn."""
    os.makedirs(workdir, exist_ok=True)
    keys_path = os.path.join(workdir, "namelist_keys.txt")
    docmap_path = os.path.join(workdir, "docmap.md")
    docmap = build_docmap(repo_root, keys_path)
    with open(docmap_path, "w", encoding="utf-8") as stream:
        stream.write(docmap.text)
    with open(keys_path, "w", encoding="utf-8") as stream:
        stream.write(docmap.key_index_text)
    journal.append(
        "docmap_generated",
        {
            "page_count": docmap.page_count,
            "key_count": docmap.key_count,
            "context_count": docmap.context_count,
            "unparsed_calls": docmap.unparsed_calls,
            "sources_missing": docmap.sources_missing,
        },
    )

    history_path = os.path.join(workdir, "history.jsonl")
    history = load_history(history_path, max_history)
    turn = _last_history_turn(history_path) + 1
    prompt = build_prompt(
        load_skill_body(repo_root), docmap.text, history, question
    )

    if invoke_fn is None:
        invoke_fn = providers.invoke
    answer = invoke_fn(
        cfg,
        ROLE,
        prompt,
        workdir,
        journal,
        timeout_s=timeout_s,
        cwd=repo_root,
    )
    if not answer.strip():
        raise RuntimeError("provider returned an empty answer")

    checks = {
        "citations": check_citations(answer, repo_root),
        "keys": check_keys(
            answer, parse_key_index_text(docmap.key_index_text)
        ),
    }
    answer_path = os.path.join(workdir, "answer_{0}.md".format(turn))
    with open(answer_path, "w", encoding="utf-8") as stream:
        stream.write(answer)
    answer_sha256 = sha256_hex(answer)
    history_record = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "turn": turn,
        "question": question,
        "answer_path": "answer_{0}.md".format(turn),
        "answer_sha256": answer_sha256,
        "checks": {
            "citations_verified": len(checks["citations"]["verified"]),
            "citations_unverified": len(checks["citations"]["unverified"]),
            "keys_known": len(checks["keys"]["known"]),
            "keys_unknown": len(checks["keys"]["unknown"]),
        },
    }
    with open(history_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(history_record, ensure_ascii=False) + "\n")
    journal.append(
        "answer_checked",
        {
            "turn": turn,
            "answer_sha256": answer_sha256,
            "citations_verified": len(checks["citations"]["verified"]),
            "citations_unverified": checks["citations"]["unverified"],
            "keys_known": len(checks["keys"]["known"]),
            "keys_unknown": checks["keys"]["unknown"],
        },
    )
    return {
        "status": "answered",
        "answer": answer,
        "checks": checks,
        "turn": turn,
        "workdir": workdir,
        "answer_path": answer_path,
        "history_path": history_path,
        "docmap_path": docmap_path,
        "keys_path": keys_path,
    }


def main_ask(args, invoke_fn=None) -> int:
    """CLI entry point for the ask verb."""
    repo_root = os.path.abspath(
        args.repo_root or Path(__file__).resolve().parents[2]
    )
    if args.workdir is not None:
        workdir = os.path.abspath(os.path.expanduser(args.workdir))
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        workdir = os.path.join(os.path.expanduser(DEFAULT_WORKDIR_PARENT), stamp)

    try:
        if args.question_file is not None:
            with open(args.question_file, "r", encoding="utf-8") as stream:
                question = stream.read()
        else:
            question = args.question
    except OSError as error:
        if args.json:
            print(
                json.dumps(
                    {"status": "error", "error": str(error), "workdir": workdir},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print("assist: ask: {0}".format(error), file=sys.stderr)
        return 2
    if question is None or not question.strip():
        print("assist: ask: a question is required", file=sys.stderr)
        return 2

    try:
        cfg = load_config(
            cli_path=args.config,
            optional_cli_path=getattr(args, "config_or_defaults", None),
        )
    except (AssistConfigError, TomlSubsetError) as error:
        print("assist: config error: {0}".format(error), file=sys.stderr)
        return 2

    if args.print_prompt:
        keys_path = os.path.join(workdir, "namelist_keys.txt")
        docmap = build_docmap(repo_root, keys_path)
        history = load_history(
            os.path.join(workdir, "history.jsonl"), args.max_history
        )
        print(
            build_prompt(
                load_skill_body(repo_root), docmap.text, history, question
            ),
            end="",
        )
        return 0

    journal = JournalWriter(os.path.join(workdir, "journal.jsonl"))
    try:
        result = ask(
            cfg,
            question,
            repo_root,
            workdir,
            journal,
            max_history=args.max_history,
            invoke_fn=invoke_fn,
            timeout_s=args.timeout_s,
        )
    except (
        AssistDisabledError,
        AssistConfigError,
        RuntimeError,
        subprocess.TimeoutExpired,
        FileNotFoundError,
        OSError,
    ) as error:
        if args.json:
            print(
                json.dumps(
                    {"status": "error", "error": str(error), "workdir": workdir},
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        else:
            print("assist: ask: {0}".format(error), file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        sys.stdout.write(result["answer"])
        sys.stdout.write("\n---\n")
        print(format_check_line(result["checks"]))
        print("workdir: {0}".format(workdir))
    return 0
