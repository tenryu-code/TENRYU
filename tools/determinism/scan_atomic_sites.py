#!/usr/bin/env python3
"""Maintain the repository ledger of raw CUDA atomic call sites."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ATOMIC_RE = re.compile(r"\batomic(Add|Sub|Min|Max|Exch|CAS|Or|And)\s*\(")
WHITESPACE_RE = re.compile(r"\s+")
SOURCE_SUFFIXES = {".cu", ".cuh", ".hpp"}
CLASSES = ("CONTROL", "STATE", "LEDGER", "DIAGNOSTIC", "UNCLASSIFIED")
HEADER = "id\top\tclass\tdest\tnote"
POLICY = (
    "New raw floating-point/control atomics are prohibited (design doc 9.9 "
    "Phase 0): migrate to a deterministic reduction, or deliberately classify "
    "the site via --generate and a curated class."
)


@dataclass(frozen=True)
class Site:
    id: str
    op: str
    path: str
    line: int
    code: str


@dataclass(frozen=True)
class AllowlistEntry:
    op: str
    site_class: str
    dest: str
    note: str


def normalize_line(line: str) -> str:
    return WHITESPACE_RE.sub(" ", line.strip())


def remove_block_comments(line: str, in_block_comment: bool) -> tuple[str, bool]:
    """Remove block-comment spans while retaining code on the same line."""
    parts: list[str] = []
    offset = 0
    while offset < len(line):
        if in_block_comment:
            end = line.find("*/", offset)
            if end < 0:
                return "".join(parts), True
            in_block_comment = False
            offset = end + 2
            continue

        start = line.find("/*", offset)
        if start < 0:
            parts.append(line[offset:])
            break
        parts.append(line[offset:start])
        in_block_comment = True
        offset = start + 2

    return "".join(parts), in_block_comment


def scan(repo_root: Path) -> list[Site]:
    sites: list[Site] = []
    source_root = repo_root / "src"
    paths = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix in SOURCE_SUFFIXES
    )

    for path in paths:
        relative_path = path.relative_to(repo_root).as_posix()
        duplicate_counts: defaultdict[str, int] = defaultdict(int)
        in_block_comment = False
        with path.open("r", encoding="utf-8") as source:
            for line_number, source_line in enumerate(source, start=1):
                if not in_block_comment and source_line.lstrip().startswith("//"):
                    continue
                code_for_matching, in_block_comment = remove_block_comments(
                    source_line, in_block_comment
                )
                if code_for_matching.lstrip().startswith("//"):
                    continue

                normalized = normalize_line(source_line)
                line_hash = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]
                for match in ATOMIC_RE.finditer(code_for_matching):
                    duplicate_index = duplicate_counts[line_hash]
                    duplicate_counts[line_hash] += 1
                    site_id = f"{relative_path}#{line_hash}#{duplicate_index}"
                    sites.append(
                        Site(
                            id=site_id,
                            op=f"atomic{match.group(1)}",
                            path=relative_path,
                            line=line_number,
                            code=normalized[:160],
                        )
                    )

    return sites


def read_allowlist(path: Path) -> dict[str, AllowlistEntry]:
    if not path.exists():
        return {}

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != HEADER:
        raise ValueError(f"{path}: expected header {HEADER!r}")

    entries: dict[str, AllowlistEntry] = {}
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split("\t", 4)
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 tab-separated fields")
        site_id, op, site_class, dest, note = fields
        if site_class not in CLASSES:
            raise ValueError(
                f"{path}:{line_number}: invalid class {site_class!r}; "
                f"expected one of {', '.join(CLASSES)}"
            )
        if site_id in entries:
            raise ValueError(f"{path}:{line_number}: duplicate id {site_id}")
        entries[site_id] = AllowlistEntry(op, site_class, dest, note)

    return entries


def generate(sites: list[Site], allowlist_path: Path) -> int:
    existing = read_allowlist(allowlist_path)
    current_ids = {site.id for site in sites}
    new_count = sum(site.id not in existing for site in sites)
    removed_count = sum(site_id not in current_ids for site_id in existing)

    output = [HEADER]
    for site in sites:
        previous = existing.get(site.id)
        if previous is None:
            entry = AllowlistEntry(site.op, "UNCLASSIFIED", "", "")
        else:
            entry = AllowlistEntry(
                site.op, previous.site_class, previous.dest, previous.note
            )
        output.append(
            "\t".join(
                (site.id, entry.op, entry.site_class, entry.dest, entry.note)
            )
        )

    allowlist_path.parent.mkdir(parents=True, exist_ok=True)
    allowlist_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"Atomic site ledger generated: total={len(sites)} new={new_count} removed={removed_count}")
    return 0


def check(sites: list[Site], allowlist_path: Path) -> int:
    existing = read_allowlist(allowlist_path)
    site_by_id = {site.id: site for site in sites}
    missing = [site for site in sites if site.id not in existing]
    stale = sorted(site_id for site_id in existing if site_id not in site_by_id)

    if missing or stale:
        for site in missing:
            print(
                f"MISSING\t{site.id}\t{site.op}\t{site.path}:{site.line}\t{site.code}",
                file=sys.stderr,
            )
        for site_id in stale:
            print(f"STALE\t{site_id}", file=sys.stderr)
        print(POLICY, file=sys.stderr)
        return 1

    counts = Counter(existing[site.id].site_class for site in sites)
    print(
        "Atomic site ledger check: "
        + " ".join(f"{site_class}={counts[site_class]}" for site_class in CLASSES)
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate or check the raw CUDA atomic-site ledger."
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--generate", action="store_true", help="update the ledger")
    modes.add_argument("--check", action="store_true", help="check the ledger (default)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    allowlist_path = repo_root / "docs/determinism/atomic_site_allowlist.tsv"

    try:
        sites = scan(repo_root)
        if args.generate:
            return generate(sites, allowlist_path)
        return check(sites, allowlist_path)
    except (OSError, UnicodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
