#!/usr/bin/env python3
"""Aggregate seed-sweep statistics from 2D RZ validation metrics files.

Agreed schema reference:
tmp/discussions/20260430-203923-2d-rz-paper-validation-storage/log.md

Exit codes:
0: stats.json updated; CV PASS/FAIL is recorded as data
1: missing stage, missing metrics, filesystem error, or JSON error
2: invalid arguments
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


PROFILE_CLASSES = ("ci_short", "paper_full", "exploratory")


@dataclass(frozen=True)
class MetricValue:
    run_id: str
    ladder_row: str
    value: float


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_runs(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row: {exc}") from exc
    return rows


def resolve_metrics_path(stage_dir: Path, metrics_ref: object) -> Path:
    if not isinstance(metrics_ref, str) or not metrics_ref:
        raise ValueError("run row is missing metrics_ref")
    path = Path(metrics_ref)
    if path.exists():
        return path
    candidate = stage_dir / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"metrics_ref not found: {metrics_ref}")


def extract_metric(metrics_path: Path, metric: str) -> float:
    payload = read_json(metrics_path)
    value = payload.get(metric)
    if value is None and isinstance(payload.get("metrics"), dict):
        value = payload["metrics"].get(metric)  # type: ignore[index]
    if not isinstance(value, (int, float)):
        raise ValueError(f"{metrics_path} does not contain numeric metric {metric}")
    return float(value)


def sanitize(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def sample_sigma(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def build_sweeps(
    stage_id: str,
    rows: list[dict[str, object]],
    stage_dir: Path,
    metric: str,
    profile_class: str,
    threshold_cv: float,
) -> list[dict[str, object]]:
    grouped: dict[str, list[MetricValue]] = {}
    for row in rows:
        if row.get("profile_class") != profile_class:
            continue
        ladder_row = row.get("ladder_row")
        run_id = row.get("run_id")
        if not isinstance(ladder_row, str) or not isinstance(run_id, str):
            raise ValueError("run row is missing ladder_row or run_id")
        metrics_path = resolve_metrics_path(stage_dir, row.get("metrics_ref"))
        value = extract_metric(metrics_path, metric)
        grouped.setdefault(ladder_row, []).append(MetricValue(run_id, ladder_row, value))

    sweeps: list[dict[str, object]] = []
    for ladder_row in sorted(grouped):
        values = [item.value for item in grouped[ladder_row]]
        mean = sum(values) / len(values)
        sigma = sample_sigma(values, mean)
        cv = 0.0 if mean == 0.0 and sigma == 0.0 else (sigma / abs(mean) if mean != 0.0 else None)
        result = "PASS" if cv is not None and cv <= threshold_cv else "FAIL"
        sweeps.append(
            {
                "sweep_id": f"{stage_id}_{sanitize(profile_class)}_{sanitize(ladder_row)}_{sanitize(metric)}",
                "metric": metric,
                "profile_class": profile_class,
                "ladder_row": ladder_row,
                "mean": mean,
                "sigma": sigma,
                "cv": cv,
                "threshold": threshold_cv,
                "result": result,
                "n_seeds": len(values),
                "run_ids": [item.run_id for item in grouped[ladder_row]],
            }
        )
    return sweeps


def merge_sweeps(
    existing: list[dict[str, object]],
    computed: list[dict[str, object]],
    metric: str,
    profile_class: str,
) -> list[dict[str, object]]:
    computed_ladders = {sweep["ladder_row"] for sweep in computed}
    retained = [
        sweep
        for sweep in existing
        if not (
            sweep.get("metric") == metric
            and sweep.get("profile_class") == profile_class
            and sweep.get("ladder_row") in computed_ladders
        )
    ]
    return retained + computed


def aggregate(root: Path, stage_id: str, metric: str, profile_class: str, threshold_cv: float) -> int:
    stage_dir = root / stage_id
    runs_path = stage_dir / "runs.jsonl"
    stats_path = stage_dir / "stats.json"
    if not runs_path.exists() or not stats_path.exists():
        print(f"stage directory is not initialized: {stage_dir}")
        return 1

    rows = load_runs(runs_path)
    computed = build_sweeps(stage_id, rows, stage_dir, metric, profile_class, threshold_cv)
    stats = read_json(stats_path)
    existing = stats.get("sweeps", [])
    if not isinstance(existing, list):
        raise ValueError(f"{stats_path} has non-list sweeps")
    stats["sweeps"] = merge_sweeps(existing, computed, metric, profile_class)
    write_json(stats_path, stats)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", required=True)
    parser.add_argument("--metric", required=True)
    parser.add_argument("--profile-class", required=True, choices=PROFILE_CLASSES)
    parser.add_argument("--threshold-cv", type=float, default=0.001)
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return aggregate(Path(args.root), args.stage_id, args.metric, args.profile_class, args.threshold_cv)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"failed to aggregate stats for {args.stage_id}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
