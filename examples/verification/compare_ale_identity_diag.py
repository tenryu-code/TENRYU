#!/usr/bin/env python3
"""Compare TENRYU ALE identity/mover diagnostic JSONL streams."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, 1):
            line = raw.strip()
            if not line:
                continue
            start = line.find("{")
            if start < 0:
                continue
            try:
                obj = json.loads(line[start:])
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if isinstance(obj, dict):
                obj["_source_line"] = lineno
                records.append(obj)
    return records


def record_kind(record: dict[str, Any]) -> str:
    return str(record.get("record") or record.get("tag") or "")


def record_key(record: dict[str, Any]) -> tuple[Any, ...]:
    kind = record_kind(record)
    step = record.get("step")
    rank = record.get("rank", 0)
    stage = record.get("stage")
    if kind == "ale_identity_diag":
        return (kind, rank, step, stage, record.get("field"))
    if kind == "ale_mover_diag":
        return (kind, rank, step, stage, record.get("band"))
    return (kind, rank, step, stage, record.get("field"), record.get("band"))


def index_records(records: list[dict[str, Any]], kind_filter: str | None) -> dict[tuple[Any, ...], dict[str, Any]]:
    out: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        kind = record_kind(record)
        if kind_filter is not None and kind != kind_filter:
            continue
        key = record_key(record)
        if key in out:
            raise SystemExit(f"duplicate diagnostic key {key}")
        out[key] = record
    return out


def number_delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        fa = float(a)
        fb = float(b)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fa) or not math.isfinite(fb):
        return None
    return abs(fb - fa)


def compare_records(ref: dict[str, Any], cand: dict[str, Any]) -> tuple[bool, str]:
    kind = record_kind(ref)
    if kind == "ale_identity_diag":
        if ref.get("hash") != cand.get("hash"):
            delta = number_delta(ref.get("linf"), cand.get("linf"))
            return (
                False,
                "hash mismatch "
                f"ref={ref.get('hash')} cand={cand.get('hash')} "
                f"linf_ref={ref.get('linf')} linf_cand={cand.get('linf')} "
                f"linf_delta={delta}",
            )
        for field in ("count", "nonfinite_count"):
            if ref.get(field) != cand.get(field):
                return False, f"{field} mismatch ref={ref.get(field)} cand={cand.get(field)}"
        return True, ""

    if kind == "ale_mover_diag":
        for field in ("count",):
            if ref.get(field) != cand.get(field):
                return False, f"{field} mismatch ref={ref.get(field)} cand={cand.get(field)}"
        deltas = {
            name: number_delta(ref.get(name), cand.get(name))
            for name in (
                "mass_sum",
                "C_stored",
                "C_move",
                "delta_C",
                "du_linf",
                "du_mass_l2",
            )
        }
        changed = {
            name: delta
            for name, delta in deltas.items()
            if delta is not None and delta != 0.0
        }
        if changed:
            return False, "numeric mismatch " + json.dumps(changed, sort_keys=True)
        return True, ""

    return (ref == cand), "record mismatch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--kind",
        choices=("ale_identity_diag", "ale_mover_diag"),
        default="ale_identity_diag",
        help="record type to compare",
    )
    args = parser.parse_args()

    ref = index_records(load_jsonl(args.reference), args.kind)
    cand = index_records(load_jsonl(args.candidate), args.kind)
    if not ref:
        print(f"no {args.kind} records found in {args.reference}", file=sys.stderr)
        return 2
    if not cand:
        print(f"no {args.kind} records found in {args.candidate}", file=sys.stderr)
        return 2

    ref_keys = set(ref)
    cand_keys = set(cand)
    missing = sorted(ref_keys - cand_keys)
    extra = sorted(cand_keys - ref_keys)
    if missing:
        print(f"first missing candidate key: {missing[0]}", file=sys.stderr)
        return 1
    if extra:
        print(f"first extra candidate key: {extra[0]}", file=sys.stderr)
        return 1

    for key in sorted(ref_keys):
        ok, reason = compare_records(ref[key], cand[key])
        if not ok:
            print(f"first mismatch key={key}: {reason}", file=sys.stderr)
            print(
                "reference line="
                f"{ref[key].get('_source_line')} candidate line={cand[key].get('_source_line')}",
                file=sys.stderr,
            )
            return 1

    print(f"PASS: {len(ref_keys)} {args.kind} records match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
