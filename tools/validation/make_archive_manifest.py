#!/usr/bin/env python3
"""Generate deterministic offline archive manifests and SHA-256 checksums.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: archive_manifest.jsonl and checksums.sha256 updated
1: missing stage/source directory, filesystem error, or JSON error
2: invalid arguments
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ManifestRow:
    archive_name: str
    file_path_rel: str
    size_bytes: int
    sha256: str
    mtime_iso: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def build_rows(source_dir: Path, archive_name: str) -> list[ManifestRow]:
    files = sorted(path for path in source_dir.rglob("*") if path.is_file())
    rows: list[ManifestRow] = []
    for path in files:
        rel = path.relative_to(source_dir).as_posix()
        rows.append(
            ManifestRow(
                archive_name=archive_name,
                file_path_rel=rel,
                size_bytes=path.stat().st_size,
                sha256=sha256_file(path),
                mtime_iso=mtime_iso(path),
            )
        )
    return rows


def load_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def write_checksums(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [f"{row['sha256']}  {row['file_path_rel']}\n" for row in rows]
    path.write_text("".join(lines), encoding="utf-8")


def make_manifest(root: Path, stage_id: str, source_dir: Path, archive_name: str) -> int:
    stage_dir = root / stage_id
    manifest_path = stage_dir / "archive_manifest.jsonl"
    checksums_path = stage_dir / "checksums.sha256"
    if not manifest_path.exists() or not checksums_path.exists():
        print(f"stage directory is not initialized: {stage_dir}")
        return 1
    if not source_dir.is_dir():
        print(f"source directory does not exist: {source_dir}")
        return 1

    new_rows = [asdict(row) for row in build_rows(source_dir, archive_name)]
    existing = [row for row in load_existing(manifest_path) if row.get("archive_name") != archive_name]
    merged = existing + new_rows
    write_manifest(manifest_path, merged)
    write_checksums(checksums_path, merged)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--archive-name", required=True)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return make_manifest(Path(args.root), args.stage_id, Path(args.source_dir), args.archive_name)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to make archive manifest for {args.stage_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
