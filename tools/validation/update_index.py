#!/usr/bin/env python3
"""Regenerate the top-level 2D RZ validation index.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: index.json regenerated
1: filesystem or JSON error
2: invalid arguments
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROADMAP_ORDER = [
    "H0",
    "H1",
    "H2",
    "H3",
    "A1",
    "C1",
    "C2",
    "E1",
    "R1",
    "R2",
    "RH1",
    "RH2",
    "L1",
    "L2",
    "I1",
    "I2",
    "I3",
]


@dataclass(frozen=True)
class IndexEntry:
    stage_id: str
    title: str
    status: str
    last_attempt_status: str | None
    attempted_runs: int
    passing_runs: int
    canonical_run_id: str | None
    archive_doi: str | None
    paper_figure_count: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_sort_key(stage_id: str) -> tuple[int, str]:
    try:
        return (ROADMAP_ORDER.index(stage_id), stage_id)
    except ValueError:
        return (len(ROADMAP_ORDER), stage_id)


def paper_figure_count(stage_dir: Path) -> int:
    figures_dir = stage_dir / "figures"
    if not figures_dir.is_dir():
        return 0
    return sum(1 for path in figures_dir.iterdir() if path.is_file() and path.name.startswith("paper_"))


def archive_doi(stage_dir: Path) -> str | None:
    archive_path = stage_dir / "archive.md"
    if not archive_path.exists():
        return None
    doi: str | None = None
    for line in archive_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| archive_name ") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 3:
            doi = cells[2]
    return doi


def build_entry(stage_dir: Path) -> IndexEntry:
    stage = read_json(stage_dir / "stage.json")
    return IndexEntry(
        stage_id=str(stage["stage_id"]),
        title=str(stage["title"]),
        status=str(stage["status"]),
        last_attempt_status=stage["last_attempt_status"] if stage["last_attempt_status"] is None else str(stage["last_attempt_status"]),
        attempted_runs=int(stage["attempted_runs"]),
        passing_runs=int(stage["passing_runs"]),
        canonical_run_id=stage["canonical_run_id"] if stage["canonical_run_id"] is None else str(stage["canonical_run_id"]),
        archive_doi=archive_doi(stage_dir),
        paper_figure_count=paper_figure_count(stage_dir),
    )


def regenerate(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    entries: list[IndexEntry] = []
    for stage_json in sorted(root.glob("*/stage.json")):
        entries.append(build_entry(stage_json.parent))
    entries.sort(key=lambda entry: stage_sort_key(entry.stage_id))
    payload = {
        "schema_version": 1,
        "generated_iso": now_iso(),
        "stages": [entry.__dict__ for entry in entries],
    }
    write_json(root / "index.json", payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return regenerate(Path(args.root))
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"failed to update validation index: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
