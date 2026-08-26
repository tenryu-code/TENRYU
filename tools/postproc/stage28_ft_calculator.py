#!/usr/bin/env python3
"""Compute F_t from baseline and profile-enabled history HDF5 files.

Usage: python tools/postproc/stage28_ft_calculator.py --baseline a.h5 --enabled b.h5 --label D6
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROVENANCE_PATH = "/diagnostics/ale_provenance/v1"


def decode_attr(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    arr = np.asarray(value)
    if arr.shape == () and arr.dtype.kind in {"S", "O", "U"}:
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").rstrip("\x00")
        return str(item)
    return str(value)


def last_finite(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return math.nan
    return float(finite[-1])


def find_dataset_named(group: h5py.Group, name: str) -> h5py.Dataset | None:
    if name in group and isinstance(group[name], h5py.Dataset):
        return group[name]
    found: h5py.Dataset | None = None

    def visitor(_: str, obj: h5py.Dataset | h5py.Group) -> bool | None:
        nonlocal found
        if isinstance(obj, h5py.Dataset) and Path(obj.name).name == name:
            found = obj
            return True
        return None

    group.visititems(visitor)
    return found


def read_history_value(handle: h5py.File, name: str) -> float:
    path = f"/history/{name}"
    if path in handle and isinstance(handle[path], h5py.Dataset):
        return last_finite(handle[path][()])
    if "/history" in handle and isinstance(handle["/history"], h5py.Group):
        dataset = find_dataset_named(handle["/history"], name)
        if dataset is not None:
            return last_finite(dataset[()])
    return math.nan


def read_provenance(handle: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {
        "final_provenance": "n/a",
        "final_claim_level": "n/a",
        "reached_t_end": "n/a",
        "profile_enabled": 0,
    }
    if PROVENANCE_PATH not in handle:
        return result
    group = handle[PROVENANCE_PATH]
    for key in ("final_provenance", "final_claim_level"):
        if key in group.attrs:
            result[key] = decode_attr(group.attrs[key])
    for key in ("reached_t_end", "profile_enabled"):
        if key in group.attrs:
            result[key] = int(np.asarray(group.attrs[key]).item())
    return result


def read_run(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as handle:
            result = {
                "path": str(path),
                "t_reached_s": read_history_value(handle, "time"),
                "step_reached": read_history_value(handle, "step"),
            }
            result.update(read_provenance(handle))
            return result
    except (OSError, KeyError, ValueError) as exc:
        print(f"error: failed to read HDF5 file {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def format_float(value: Any) -> str:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(fvalue):
        return "nan"
    return f"{fvalue:.6e}"


def format_step(value: Any) -> str:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(fvalue):
        return "nan"
    return f"{int(fvalue):d}"


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(denominator) or denominator <= 0.0:
        return math.nan
    return numerator / denominator


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--enabled", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    baseline = read_run(args.baseline)
    enabled = read_run(args.enabled)
    ft = ratio(float(enabled["t_reached_s"]), float(baseline["t_reached_s"]))
    fstep = ratio(float(enabled["step_reached"]), float(baseline["step_reached"]))

    print(f"## F_t for {args.label}")
    print()
    print("| Field | Run A (baseline / disabled) | Run B (profile enabled) |")
    print("|-------|------------------------------|--------------------------|")
    print(f"| t_reached_s | {format_float(baseline['t_reached_s'])} | {format_float(enabled['t_reached_s'])} |")
    print(f"| step_reached | {format_step(baseline['step_reached'])} | {format_step(enabled['step_reached'])} |")
    print(f"| reached_t_end | {baseline['reached_t_end']} | {enabled['reached_t_end']} |")
    print(f"| final_provenance | {baseline['final_provenance']} | {enabled['final_provenance']} |")
    print(f"| final_claim_level | {baseline['final_claim_level']} | {enabled['final_claim_level']} |")
    print()
    print(f"F_t = {ft:.4f}  (>= 0.95 ⇒ PUBLIC_BASELINE empirically confirmed; < 0.95 ⇒ PUBLIC_BASELINE_FAILED_NO_ESCAPE)")
    print(f"F_step = {fstep:.4f}")

    if args.json_out is not None:
        write_json(
            args.json_out,
            {
                "label": args.label,
                "baseline": baseline,
                "enabled": enabled,
                "F_t": ft,
                "F_step": fstep,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
