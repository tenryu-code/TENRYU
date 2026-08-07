#!/usr/bin/env python3
"""Attach a pre-existing archive DOI to a 2D RZ validation stage.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: archive.md and index.json updated
1: missing stage, filesystem error, or JSON error
2: invalid arguments
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ArchiveVersion:
    archive_name: str
    doi: str
    doi_url: str
    archive_version: str
    attached_iso: str


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_archive_md(path: Path, stage_id: str, version: ArchiveVersion) -> None:
    rows: list[list[str]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("| ") or line.startswith("| archive_name ") or line.startswith("|---"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) == 5:
                rows.append(cells)

    rows = [
        cells
        for cells in rows
        if not (cells[0] == version.archive_name and cells[1] == version.archive_version)
    ]
    rows.append(
        [
            version.archive_name,
            version.archive_version,
            version.doi,
            version.doi_url,
            version.attached_iso,
        ]
    )

    lines = [
        f"# {stage_id} Archive",
        "",
        "Status: archived versions listed below.",
        "",
        "| archive_name | archive_version | doi | doi_url | attached_iso |",
        "|---|---|---|---|---|",
    ]
    lines.extend(f"| {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} | {cells[4]} |" for cells in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def paper_figure_count(stage_dir: Path) -> int:
    figures_dir = stage_dir / "figures"
    if not figures_dir.is_dir():
        return 0
    return sum(1 for path in figures_dir.iterdir() if path.is_file() and path.name.startswith("paper_"))


def stage_index_entry(stage_dir: Path, archive_doi: str) -> dict[str, object]:
    stage = read_json(stage_dir / "stage.json")
    return {
        "stage_id": stage["stage_id"],
        "title": stage["title"],
        "status": stage["status"],
        "last_attempt_status": stage["last_attempt_status"],
        "attempted_runs": stage["attempted_runs"],
        "passing_runs": stage["passing_runs"],
        "canonical_run_id": stage["canonical_run_id"],
        "archive_doi": archive_doi,
        "paper_figure_count": paper_figure_count(stage_dir),
    }


def update_index(root: Path, stage_id: str, archive_doi: str) -> None:
    index_path = root / "index.json"
    if index_path.exists():
        index = read_json(index_path)
    else:
        index = {"schema_version": 1, "generated_iso": now_iso(), "stages": []}

    stages = index.setdefault("stages", [])
    if not isinstance(stages, list):
        raise ValueError(f"{index_path} has non-list stages")

    entry = stage_index_entry(root / stage_id, archive_doi)
    replaced = False
    for idx, current in enumerate(stages):
        if isinstance(current, dict) and current.get("stage_id") == stage_id:
            current.update(entry)
            stages[idx] = current
            replaced = True
            break
    if not replaced:
        stages.append(entry)
    index["generated_iso"] = now_iso()
    write_json(index_path, index)


def attach(root: Path, stage_id: str, version: ArchiveVersion) -> int:
    stage_dir = root / stage_id
    if not (stage_dir / "stage.json").exists():
        print(f"stage directory is not initialized: {stage_dir}")
        return 1
    update_archive_md(stage_dir / "archive.md", stage_id, version)
    update_index(root, stage_id, version.doi)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--doi", required=True)
    parser.add_argument("--doi-url", required=True)
    parser.add_argument("--archive-version", required=True)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    version = ArchiveVersion(
        archive_name=args.archive_name,
        doi=args.doi,
        doi_url=args.doi_url,
        archive_version=args.archive_version,
        attached_iso=now_iso(),
    )
    try:
        return attach(Path(args.root), args.stage_id, version)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to attach archive for {args.stage_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
