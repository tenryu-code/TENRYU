#!/usr/bin/env python3
"""Summarize Stage 28 ALE provenance diagnostics from a history HDF5 file.

Usage: python tools/postproc/stage28_provenance_reader.py --input history.h5
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


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(item, np.generic):
            return item.item()
        return item
    if arr.dtype.kind == "S":
        return [item.decode("utf-8", errors="replace").rstrip("\x00") for item in arr.reshape(-1)]
    return arr.tolist()


def scalar_to_json(value: Any) -> Any:
    value = decode_value(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def format_value(value: Any) -> str:
    value = scalar_to_json(value)
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6e}"
    if isinstance(value, int):
        return f"{value:d}"
    return str(value)


def last_nonzero_index(values: np.ndarray) -> int | None:
    arr = np.asarray(values).reshape(-1)
    idx = np.flatnonzero(arr != 0)
    if idx.size == 0:
        return None
    return int(idx[-1])


def counter_summary(dataset: h5py.Dataset, steps: np.ndarray | None) -> dict[str, Any] | None:
    arr = np.asarray(dataset[()])
    if arr.ndim != 1:
        return None
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    if arr.size == 0:
        return {"last_value": None, "max_over_history": None, "last_nonzero_step": None}
    finite = arr[np.isfinite(arr)] if arr.dtype.kind == "f" else arr
    max_value = None if finite.size == 0 else scalar_to_json(np.max(finite))
    last_idx = last_nonzero_index(arr)
    last_step = last_idx
    if steps is not None and last_idx is not None and last_idx < steps.size:
        last_step = scalar_to_json(steps[last_idx])
    return {
        "last_value": scalar_to_json(arr[-1]),
        "max_over_history": max_value,
        "last_nonzero_step": last_step,
    }


def read_event(name: str, obj: h5py.Dataset | h5py.Group) -> dict[str, Any]:
    attrs = {key: scalar_to_json(value) for key, value in obj.attrs.items()}
    step = attrs.get("step", "n/a")
    values: dict[str, Any] = {}
    if isinstance(obj, h5py.Dataset):
        arr = np.asarray(obj[()])
        if arr.ndim == 0:
            values["value"] = scalar_to_json(arr)
        elif arr.ndim == 1 and arr.size > 0:
            values["last_value"] = scalar_to_json(arr[-1])
            if name == "step":
                step = scalar_to_json(arr[-1])
    else:
        for child_name, child in obj.items():
            if isinstance(child, h5py.Dataset):
                arr = np.asarray(child[()])
                if arr.ndim == 0:
                    values[child_name] = scalar_to_json(arr)
                elif arr.ndim == 1 and arr.size > 0:
                    values[child_name] = scalar_to_json(arr[-1])
                    if child_name == "step":
                        step = scalar_to_json(arr[-1])
    attrs.update(values)
    return {"event_name": name, "step": step, "attributes": attrs}


def summarize(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as handle:
            if PROVENANCE_PATH not in handle:
                return {
                    "input": str(path),
                    "final_provenance": "n/a",
                    "final_claim_level": "n/a",
                    "reached_t_end": "n/a",
                    "profile_enabled": "n/a",
                    "counters": {},
                    "escape_valve_events": [],
                }
            group = handle[PROVENANCE_PATH]
            result: dict[str, Any] = {
                "input": str(path),
                "final_provenance": "n/a",
                "final_claim_level": "n/a",
                "reached_t_end": "n/a",
                "profile_enabled": "n/a",
                "counters": {},
                "escape_valve_events": [],
            }
            for key in ("final_provenance", "final_claim_level", "reached_t_end", "profile_enabled"):
                if key in group.attrs:
                    result[key] = scalar_to_json(group.attrs[key])
            steps = None
            if "step" in group and isinstance(group["step"], h5py.Dataset):
                steps_arr = np.asarray(group["step"][()])
                if steps_arr.ndim == 1:
                    steps = steps_arr
            for name, obj in group.items():
                if name in {"escape_valve_events", "step", "time_s"} or not isinstance(obj, h5py.Dataset):
                    continue
                summary = counter_summary(obj, steps)
                if summary is not None:
                    result["counters"][name] = summary
            if "escape_valve_events" in group and isinstance(group["escape_valve_events"], h5py.Group):
                events = group["escape_valve_events"]
                for name in sorted(events.keys()):
                    child = events[name]
                    if isinstance(child, (h5py.Dataset, h5py.Group)):
                        result["escape_valve_events"].append(read_event(name, child))
            return result
    except OSError as exc:
        print(f"error: failed to read HDF5 file {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def format_attributes(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "n/a"
    return ", ".join(f"{key}={format_value(value)}" for key, value in sorted(attrs.items()))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    summary = summarize(args.input)
    print(f"## ale_provenance summary — {args.input.name}")
    print()
    print(f"**Final provenance**: {format_value(summary['final_provenance'])}")
    print(f"**Final claim level**: {format_value(summary['final_claim_level'])}")
    print(f"**Reached t_end**: {format_value(summary['reached_t_end'])}")
    print(f"**Profile enabled**: {format_value(summary['profile_enabled'])}")
    print()
    print("### Per-step counters (last value)")
    print()
    print("| Counter | Last value | Max over history | Last step at which it was nonzero |")
    print("|---------|------------|-------------------|-----------------------------------|")
    if summary["counters"]:
        for name in sorted(summary["counters"]):
            counter = summary["counters"][name]
            last_step = counter["last_nonzero_step"]
            last_step_text = "n/a" if last_step is None else format_value(last_step)
            print(
                f"| {name} | {format_value(counter['last_value'])} | "
                f"{format_value(counter['max_over_history'])} | {last_step_text} |"
            )
    else:
        print("| (none) | n/a | n/a | n/a |")
    print()
    print("### Escape valve events")
    print()
    print("| Event name | Step | Attributes |")
    print("|------------|------|------------|")
    if summary["escape_valve_events"]:
        for event in summary["escape_valve_events"]:
            print(
                f"| {event['event_name']} | {format_value(event['step'])} | "
                f"{format_attributes(event['attributes'])} |"
            )
    else:
        print("| (none) | n/a | n/a |")

    if args.json_out is not None:
        write_json(args.json_out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
