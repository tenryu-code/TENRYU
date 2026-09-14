"""Deterministic document map and namelist key index for the assistant."""

from dataclasses import dataclass, field
import html
import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Tuple


SITE_DIR = "docs/site/ja"
KEY_SOURCE = "src/core/namelist/builder.cpp"
EXAMPLES_DIR = "examples"

# Canonical documents and their heading policy, in this order.
# policy "spec": keep every level-2 heading, plus level-3/4 headings whose text
#   (after stripping the leading '#'s and spaces) matches ^(6\.4|9\.1)(\.|\s|$)
# policy "all2": keep level-2 headings only.
# policy "all23": keep level-2 and level-3 headings.
CANON_FILES: List[Tuple[str, str]] = [
    ("docs/SPECIFICATION.md", "spec"),
    ("docs/TUTORIAL_ja.md", "all23"),
    ("docs/OUTPUT_SCHEMA.md", "all23"),
    ("docs/POSTPROCESSING.md", "all23"),
    ("docs/NUMERICS.md", "all2"),
    ("docs/ARCHITECTURE.md", "all2"),
    ("docs/VERIFICATION.md", "all2"),
]


@dataclass
class DocMap:
    text: str
    key_index_text: str
    sources_present: List[str] = field(default_factory=list)
    sources_missing: List[str] = field(default_factory=list)
    page_count: int = 0
    key_count: int = 0
    context_count: int = 0
    unparsed_calls: int = 0


def _clean_html_text(fragment: str) -> str:
    text = re.sub(r"<[^>]+>", " ", fragment)
    return " ".join(html.unescape(text).split())


def extract_html_headings(html: str) -> Tuple[str, List[str]]:
    """Return (h1_text, [h2_texts]) from an HTML page."""
    h1_match = re.search(r"<h1\b[^>]*>(.*?)</h1\s*>", html, re.S | re.I)
    h1_text = _clean_html_text(h1_match.group(1)) if h1_match else ""
    if h1_text.endswith(" reference"):
        h1_text = h1_text[: -len(" reference")]
    h2_texts = [
        _clean_html_text(match.group(1))
        for match in re.finditer(r"<h2\b[^>]*>(.*?)</h2\s*>", html, re.S | re.I)
    ]
    return h1_text, h2_texts


def markdown_heading_ranges(
    text: str, keep: Callable[[int, str], bool]
) -> List[Tuple[int, int, int, str]]:
    """Return kept headings as (level, start_line, end_line, heading_text), 1-based inclusive."""
    headings = []
    in_fence = False
    lines = text.splitlines()
    for line_no, line in enumerate(lines, 1):
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if match:
            headings.append((len(match.group(1)), line_no, match.group(2)))

    ranges = []
    last_line = len(lines)
    for index, (level, start_line, heading_text) in enumerate(headings):
        if not keep(level, heading_text):
            continue
        end_line = last_line
        for next_level, next_line, _ in headings[index + 1 :]:
            if next_level <= level:
                end_line = next_line - 1
                break
        ranges.append((level, start_line, end_line, heading_text))
    return ranges


def spec_keep(level: int, text: str) -> bool:
    if level == 2:
        return True
    return level in (3, 4) and re.match(r"^(6\.4|9\.1)(\.|\s|$)", text) is not None


def _scan_call_arguments(cpp_text: str, open_paren: int) -> Optional[List[str]]:
    arguments = []
    current = []
    paren_depth = 0
    bracket_depth = 0
    brace_depth = 0
    index = open_paren + 1
    in_string = False

    while index < len(cpp_text):
        char = cpp_text[index]
        next_char = cpp_text[index + 1] if index + 1 < len(cpp_text) else ""

        if in_string:
            current.append(char)
            if char == "\\" and next_char:
                current.append(next_char)
                index += 2
                continue
            if char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            current.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            current.append(" ")
            index += 2
            while index < len(cpp_text) and cpp_text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            current.append(" ")
            index += 2
            while index + 1 < len(cpp_text):
                if cpp_text[index] == "*" and cpp_text[index + 1] == "/":
                    index += 2
                    break
                index += 1
            else:
                return None
            continue

        if char == "(":
            paren_depth += 1
        elif char == "[":
            bracket_depth += 1
        elif char == "{":
            brace_depth += 1
        elif char == ")":
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                arguments.append("".join(current))
                return arguments
            if paren_depth == 0:
                return None
            paren_depth -= 1
        elif char == "]":
            if bracket_depth == 0:
                return None
            bracket_depth -= 1
        elif char == "}":
            if brace_depth == 0:
                return None
            brace_depth -= 1
        elif (
            char == ","
            and paren_depth == 0
            and bracket_depth == 0
            and brace_depth == 0
        ):
            arguments.append("".join(current))
            current = []
            index += 1
            continue

        current.append(char)
        index += 1
    return None


def parse_known_keys(cpp_text: str) -> Tuple[Dict[str, List[str]], int]:
    """Return ({context: sorted unique keys}, unparsed_call_count) from builder.cpp text."""
    token = "enforce_known_keys("
    merged = {}
    unparsed = 0
    search_from = 0

    while True:
        occurrence = cpp_text.find(token, search_from)
        if occurrence < 0:
            break
        search_from = occurrence + len(token)
        line_start = cpp_text.rfind("\n", 0, occurrence) + 1
        line_prefix = cpp_text[line_start:occurrence].lstrip()
        if line_prefix.startswith("void ") or line_prefix.startswith("static void "):
            continue

        arguments = _scan_call_arguments(
            cpp_text, occurrence + len("enforce_known_keys")
        )
        if arguments is None or len(arguments) < 3:
            unparsed += 1
            continue

        context_match = re.match(
            r'^"((?:\\.|[^"\\])*)"$', arguments[1].strip(), re.S
        )
        key_list = arguments[2].strip()
        if context_match is None or not key_list.startswith("{") or not key_list.endswith("}"):
            unparsed += 1
            continue

        context = context_match.group(1)
        if context not in merged:
            merged[context] = set()
        for key_match in re.finditer(r'"((?:\\.|[^"\\])*)"', key_list, re.S):
            merged[context].add(key_match.group(1))

    result = {
        context: sorted(merged[context])
        for context in sorted(merged)
    }
    return result, unparsed


def build_key_index_text(
    keys: Dict[str, List[str]], unparsed: int, source_relpath: str
) -> str:
    key_count = sum(len(values) for values in keys.values())
    lines = [
        "# TENRYU namelist key index (generated by tools/assist/docmap.py; do not edit)",
        "# source: {0} (enforce_known_keys lists)".format(source_relpath),
        "# contexts: {0}  keys: {1}  calls_not_parsed: {2}".format(
            len(keys), key_count, unparsed
        ),
        "# format: <Context><TAB><key>  — Context is the block path (Mesh, Numerics.dt, Laser.beams, ...)",
    ]
    for context in sorted(keys):
        for key in sorted(keys[context]):
            lines.append("{0}\t{1}".format(context, key))
    return "\n".join(lines) + "\n"


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _title_text(page_text: str) -> str:
    match = re.search(r"<title\b[^>]*>(.*?)</title\s*>", page_text, re.S | re.I)
    return _clean_html_text(match.group(1)) if match else ""


def _canon_keep(policy: str) -> Callable[[int, str], bool]:
    if policy == "spec":
        return spec_keep
    if policy == "all2":
        return lambda level, text: level == 2
    return lambda level, text: level in (2, 3)


def _example_comment(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        if stripped.startswith("#"):
            return " ".join(stripped.lstrip("# ").split())[:160]
    return "(no leading comment)"


def build_docmap(repo_root: str, keys_path: Optional[str] = None) -> DocMap:
    root_text = os.path.abspath(repo_root)
    root = Path(root_text)
    sources_present = []
    sources_missing = []

    reader_lines = []
    site_path = root / SITE_DIR
    if site_path.is_dir():
        sources_present.append(SITE_DIR)
        try:
            html_paths = sorted(
                site_path.rglob("*.html"),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        except OSError:
            html_paths = []
        for page_path in html_paths:
            page_text = _read_text(page_path)
            if page_text is None:
                continue
            h1_text, h2_texts = extract_html_headings(page_text)
            if not h1_text:
                h1_text = _title_text(page_text)
            if not h1_text:
                h1_text = "(untitled)"
            relpath = page_path.relative_to(root).as_posix()
            line = "- {0} — {1}".format(relpath, h1_text)
            if h2_texts:
                line += " :: " + " / ".join(h2_texts)
            reader_lines.append(line)
    else:
        sources_missing.append(SITE_DIR)
        reader_lines.append("- (docs/site/ja not present in this checkout)")

    canon_lines = []
    for relpath, policy in CANON_FILES:
        canon_lines.append("### {0}".format(relpath))
        source_text = _read_text(root / relpath)
        if source_text is None:
            sources_missing.append(relpath)
            canon_lines.append("- (not present in this checkout)")
        else:
            sources_present.append(relpath)
            for level, start_line, end_line, heading_text in markdown_heading_ranges(
                source_text, _canon_keep(policy)
            ):
                canon_lines.append(
                    "- L{0}-L{1} {2} {3}".format(
                        start_line, end_line, "#" * level, heading_text
                    )
                )

    example_lines = []
    examples_path = root / EXAMPLES_DIR
    if examples_path.is_dir():
        sources_present.append(EXAMPLES_DIR)
        try:
            example_paths = sorted(
                (
                    path
                    for path in examples_path.rglob("*.py")
                    if "__pycache__" not in path.relative_to(examples_path).parts
                ),
                key=lambda path: path.relative_to(root).as_posix(),
            )
        except OSError:
            example_paths = []
        for example_path in example_paths:
            example_text = _read_text(example_path)
            if example_text is None:
                continue
            example_lines.append(
                "- {0} — {1}".format(
                    example_path.relative_to(root).as_posix(),
                    _example_comment(example_text),
                )
            )
    else:
        sources_missing.append(EXAMPLES_DIR)

    key_source_text = _read_text(root / KEY_SOURCE)
    if key_source_text is None:
        sources_missing.append(KEY_SOURCE)
        keys = {}
        unparsed = 0
    else:
        sources_present.append(KEY_SOURCE)
        keys, unparsed = parse_known_keys(key_source_text)
    key_count = sum(len(values) for values in keys.values())
    key_index_text = build_key_index_text(keys, unparsed, KEY_SOURCE)

    lines = [
        "# TENRYU document map",
        "repo_root: {0}".format(root_text),
        "generated_by: tools/assist/assist.py docmap (deterministic; regenerate after document changes)",
        "sources_present: {0}".format(", ".join(sources_present)),
        "sources_missing: {0}".format(
            ", ".join(sources_missing) if sources_missing else "(none)"
        ),
        "",
        "## 1. Reader documentation (docs/site/ja)",
    ]
    lines.extend(reader_lines)
    lines.extend(
        [
            "",
            "## 2. Canonical documents (heading → line range)",
        ]
    )
    lines.extend(canon_lines)
    lines.extend(
        [
            "",
            "## 3. Example decks (examples/**/*.py)",
        ]
    )
    lines.extend(example_lines)
    lines.extend(
        [
            "",
            "## 4. Namelist key index",
            "- file: {0}".format(
                keys_path
                if keys_path is not None
                else "(not written; run docmap with --keys-out FILE)"
            ),
        ]
    )
    if key_source_text is None:
        lines.append(
            "- source: src/core/namelist/builder.cpp not present in this checkout"
        )
    else:
        top_level = sorted(
            context for context in keys if "." not in context and "[" not in context
        )
        lines.extend(
            [
                "- source: src/core/namelist/builder.cpp (enforce_known_keys lists): {0} keys in {1} contexts; {2} calls not parsed".format(
                    key_count, len(keys), unparsed
                ),
                "- top-level blocks: {0}".format(", ".join(top_level)),
                '- how to check a key: grep the index for "<Context><TAB><key>", or grep builder.cpp for the key inside the enforce_known_keys list of its block',
            ]
        )

    return DocMap(
        text="\n".join(lines) + "\n",
        key_index_text=key_index_text,
        sources_present=sources_present,
        sources_missing=sources_missing,
        page_count=len(reader_lines) if site_path.is_dir() else 0,
        key_count=key_count,
        context_count=len(keys),
        unparsed_calls=unparsed,
    )


def main_docmap(args) -> int:
    """CLI entry: args has repo_root, output, keys_out."""
    repo_root = (
        args.repo_root
        if args.repo_root is not None
        else str(Path(__file__).resolve().parents[2])
    )
    keys_path = os.path.abspath(args.keys_out) if args.keys_out is not None else None
    docmap = build_docmap(repo_root, keys_path=keys_path)

    if args.output is None:
        print(docmap.text, end="")
    else:
        with open(args.output, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(docmap.text)

    if keys_path is not None:
        Path(keys_path).parent.mkdir(parents=True, exist_ok=True)
        with open(keys_path, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(docmap.key_index_text)
    return 0
