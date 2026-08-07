#!/usr/bin/env python3
"""Bootstrap a 2D RZ validation stage evidence tree.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: stage tree created or refreshed with --force
1: filesystem or JSON write error
2: invalid arguments or existing stage without --force
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


ROADMAP_SECTIONS = {
    "H0": "response §1",
    "H1": "response §2",
    "H2": "response §3",
    "H3": "response §4",
    "A1": "response §5",
    "C1": "response §6",
    "C2": "response §7",
    "E1": "response §8",
    "R1": "response §9",
    "R2": "response §10",
    "RH1": "response §11",
    "RH2": "response §12",
    "L1": "response §13",
    "L2": "response §14",
    "I1": "response §15",
    "I2": "response §16",
    "I3": "response §17",
}


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    title: str
    physics_enabled: list[str]
    purpose: str
    root: Path


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def stage_json(spec: StageSpec) -> dict[str, object]:
    return {
        "stage_id": spec.stage_id,
        "title": spec.title,
        "physics_enabled": spec.physics_enabled,
        "purpose": spec.purpose,
        "gates": [],
        "status": "NOT_STARTED",
        "last_attempt_status": None,
        "attempted_runs": 0,
        "passing_runs": 0,
        "canonical_run_id": None,
        "blocking_gate_ids": [],
        "references": {
            "verbatim_response": "docs/design/2d_rz_production_roadmap_response_verbatim.md",
            "section": ROADMAP_SECTIONS[spec.stage_id],
        },
    }


def results_template(spec: StageSpec) -> str:
    physics = ", ".join(spec.physics_enabled)
    return f"""# {spec.stage_id} Validation Results

Stage: {spec.title}

Physics enabled: {physics}

## Status

NOT_STARTED

## Purpose

{spec.purpose}

## Run Matrix

No runs recorded yet.

## Gates

No gates recorded yet.

## Resolution Trend

No resolution sweep recorded yet.

## Statistical Reproducibility

No seed sweep recorded yet.

## Figures

No figures recorded yet.

## Reproduction

No reproduction command recorded yet.

## Deviations

No deviations recorded yet.
"""


def archive_template(stage_id: str) -> str:
    return f"""# {stage_id} Archive

Status: not yet archived. Will be Zenodo at paper submission.
"""


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_stage(spec: StageSpec, force: bool) -> int:
    stage_dir = spec.root / spec.stage_id
    existed = stage_dir.exists()
    if existed and not force:
        print(f"{stage_dir} already exists; rerun with --force to refresh stage.json and results.md")
        return 2

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "figures").mkdir(exist_ok=True)
    (stage_dir / "metrics").mkdir(exist_ok=True)

    write_json(stage_dir / "stage.json", stage_json(spec))
    (stage_dir / "results.md").write_text(results_template(spec), encoding="utf-8")

    for name in ("runs.jsonl", "archive_manifest.jsonl", "checksums.sha256"):
        (stage_dir / name).touch(exist_ok=True)
    stats_path = stage_dir / "stats.json"
    if not existed or not stats_path.exists():
        write_json(stats_path, {"sweeps": []})
    (stage_dir / "figures" / ".gitkeep").touch(exist_ok=True)
    (stage_dir / "metrics" / ".gitkeep").touch(exist_ok=True)
    archive_path = stage_dir / "archive.md"
    if not existed or not archive_path.exists():
        archive_path.write_text(archive_template(spec.stage_id), encoding="utf-8")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True, choices=ROADMAP_SECTIONS.keys())
    parser.add_argument("--title", required=True)
    parser.add_argument("--physics-enabled", required=True, help="Comma-separated physics labels")
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--force", action="store_true", help="Refresh stage.json and results.md for an existing stage")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    physics_enabled = parse_csv(args.physics_enabled)
    if not physics_enabled:
        parser.error("--physics-enabled must contain at least one value")

    spec = StageSpec(
        stage_id=args.stage_id,
        title=args.title,
        physics_enabled=physics_enabled,
        purpose=args.purpose,
        root=Path(args.root),
    )
    try:
        return create_stage(spec, args.force)
    except OSError as exc:
        print(f"failed to initialize {spec.stage_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
