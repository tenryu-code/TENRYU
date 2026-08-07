#!/usr/bin/env python3
"""Append one 2D RZ validation run row and update stage counters.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: run row appended and stage counters updated
1: missing stage, filesystem error, JSON error, or failed git auto-detect
2: invalid arguments
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


PROFILE_CLASSES = ("ci_short", "paper_full", "exploratory")
EVIDENCE_ROLES = ("ci_gate", "paper_figure", "exploratory")
RESULTS = ("PASS", "FAIL", "CRASHED")


@dataclass(frozen=True)
class RunRow:
    run_id: str
    stage_id: str
    ladder_row: str
    seed: int
    deck_path: str
    commit_sha: str
    gpu_model: str
    cuda_version: str
    run_profile: str
    profile_class: str
    evidence_role: str
    result: str
    failure_class: str | None
    failed_gates: list[str]
    exit_code: int
    metrics_ref: str | None
    archive_ref: str | None
    wall_time_s: float
    output_bytes: int


def parse_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def detect_commit_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_stage(stage_path: Path, result: str) -> None:
    stage = read_json(stage_path)
    stage["attempted_runs"] = int(stage.get("attempted_runs", 0)) + 1
    if result == "PASS":
        stage["passing_runs"] = int(stage.get("passing_runs", 0)) + 1
    stage["last_attempt_status"] = result
    if stage.get("status") == "NOT_STARTED":
        stage["status"] = "IN_PROGRESS"
    write_json(stage_path, stage)


def append_run(root: Path, row: RunRow) -> int:
    stage_dir = root / row.stage_id
    stage_path = stage_dir / "stage.json"
    runs_path = stage_dir / "runs.jsonl"
    if not stage_path.exists() or not runs_path.exists():
        print(f"stage directory is not initialized: {stage_dir}")
        return 1

    with runs_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(row), sort_keys=False) + "\n")
    update_stage(stage_path, row.result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ladder-row", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deck-path", required=True)
    parser.add_argument("--commit-sha")
    parser.add_argument("--gpu-model", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--run-profile", required=True)
    parser.add_argument("--profile-class", required=True, choices=PROFILE_CLASSES)
    parser.add_argument("--evidence-role", required=True, choices=EVIDENCE_ROLES)
    parser.add_argument("--result", required=True, choices=RESULTS)
    parser.add_argument("--failure-class")
    parser.add_argument("--failed-gates")
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--metrics-ref")
    parser.add_argument("--archive-ref")
    parser.add_argument("--wall-time-s", type=float, required=True)
    parser.add_argument("--output-bytes", type=int, required=True)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        commit_sha = args.commit_sha or detect_commit_sha()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"failed to auto-detect commit SHA; pass --commit-sha explicitly: {exc}")
        return 1

    row = RunRow(
        run_id=args.run_id,
        stage_id=args.stage_id,
        ladder_row=args.ladder_row,
        seed=args.seed,
        deck_path=args.deck_path,
        commit_sha=commit_sha,
        gpu_model=args.gpu_model,
        cuda_version=args.cuda_version,
        run_profile=args.run_profile,
        profile_class=args.profile_class,
        evidence_role=args.evidence_role,
        result=args.result,
        failure_class=args.failure_class,
        failed_gates=parse_csv(args.failed_gates),
        exit_code=args.exit_code,
        metrics_ref=args.metrics_ref,
        archive_ref=args.archive_ref,
        wall_time_s=args.wall_time_s,
        output_bytes=args.output_bytes,
    )
    try:
        return append_run(Path(args.root), row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to append run {args.run_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
