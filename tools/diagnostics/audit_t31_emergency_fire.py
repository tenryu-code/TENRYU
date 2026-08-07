#!/usr/bin/env python3
"""Audit retained H3-B outputs for T31 redistribution emergency fires.

The durable production signature of the T31 D > C branch is nonzero
energy/redistribution_unresolved in the history HDF5. Current source does not
emit per-cell D/C values to the warning log, so logs are scanned as supporting
evidence only.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

try:
    import h5py
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment failure path
    print(f"missing Python dependency: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_RUN_GLOBS = [
    "build/output_h3b_phase9_11_*",
    "build/output_h3b_*T31*",
    "build/output_h3b_no_ale_ref",
    "build/output_h3b_baseline_128",
    "build/h3b_runs/h3b_nr128_*",
    "build/h3b_runs_256*/h3b_*",
]

HISTORY_SIGNATURE = "energy/redistribution_unresolved"
HISTORY_SIGNATURE_COMPAT = "energy/E_redistribution_unresolved"
AUDIT_BASE = "energy/ale_closure_audit/"

LOG_PATTERNS = {
    "expected_t31_tags": re.compile(r"\[(?:t31_redistribute|t31_emergency)\]"),
    "energy_budget_debug": re.compile(
        r"\[diag:energy\].*E_redistribution_unresolved="
    ),
    "redistribution_warning": re.compile(
        r"Energy redistribution unresolved contribution is large"
    ),
    "ale_closure_audit": re.compile(r"\[ale-closure-audit\]"),
}

KEY_VALUE_RE = re.compile(
    r"([A-Za-z0-9_./:-]+)="
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def json_float(value: Any) -> float | None:
    out = finite_float(value)
    return out


def dataset_array(handle: h5py.File, name: str) -> np.ndarray | None:
    if name not in handle:
        return None
    return np.asarray(handle[name][...])


def dataset_list(handle: h5py.File, name: str, n: int) -> np.ndarray:
    arr = dataset_array(handle, name)
    if arr is None:
        return np.arange(n, dtype=np.int64)
    if arr.shape[0] != n:
        return np.arange(n, dtype=np.int64)
    return arr


def read_json_file(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read {path}: {exc}", file=sys.stderr)
        return None


def extract_run_config(run_dir: Path) -> dict[str, Any]:
    config_files = sorted((run_dir / "config").glob("*_frozen.json"))
    payload = read_json_file(config_files[0]) if config_files else None
    if payload is None:
        return {
            "frozen_config": None,
            "schema_version": None,
            "t31_enabled": None,
            "ke_conservation_closure": None,
            "ke_closure_redistribute_floor": None,
            "ke_conservation_closure_audit": None,
        }

    numerics = payload.get("numerics", {})
    if not isinstance(numerics, dict):
        numerics = {}
    ale = numerics.get("ale", {})
    if not isinstance(ale, dict):
        ale = {}
    main = payload.get("main", {})
    if not isinstance(main, dict):
        main = {}
    mesh = payload.get("mesh", {})
    if not isinstance(mesh, dict):
        mesh = {}

    ke_closure = ale.get("ke_conservation_closure")
    redistribute = ale.get("ke_closure_redistribute_floor")
    t31_enabled = None
    if isinstance(ke_closure, bool) or isinstance(redistribute, bool):
        t31_enabled = bool(ke_closure) and bool(redistribute)

    return {
        "frozen_config": str(config_files[0]),
        "schema_version": payload.get("_schema_version"),
        "name": main.get("name"),
        "seed": main.get("seed"),
        "t_end_s": main.get("t_end"),
        "mesh": {
            "nr": mesh.get("nr"),
            "nz": mesh.get("nz"),
            "motion": mesh.get("motion"),
        },
        "ale": {
            "enabled": ale.get("enabled"),
            "every_n_steps": ale.get("every_n_steps"),
            "axis_repair_mode": ale.get("axis_repair_mode"),
            "remap_damage_gate_enabled": ale.get("remap_damage_gate_enabled"),
            "remap_scheme": ale.get("remap_scheme"),
            "ke_conservation_closure": ke_closure,
            "ke_conservation_closure_audit": ale.get("ke_conservation_closure_audit"),
            "ke_closure_redistribute_floor": redistribute,
        },
        "t31_enabled": t31_enabled,
        "ke_conservation_closure": ke_closure,
        "ke_closure_redistribute_floor": redistribute,
        "ke_conservation_closure_audit": ale.get("ke_conservation_closure_audit"),
    }


def parse_log_values(line: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, raw in KEY_VALUE_RE.findall(line):
        value = finite_float(raw)
        if value is not None:
            values[key] = value
    return values


def scan_log_file(path: Path, max_hits: int) -> dict[str, Any]:
    counts = {name: 0 for name in LOG_PATTERNS}
    hits: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                matched = [name for name, pattern in LOG_PATTERNS.items() if pattern.search(line)]
                if not matched:
                    continue
                for name in matched:
                    counts[name] += 1
                if len(hits) < max_hits:
                    hits.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "patterns": matched,
                            "values": parse_log_values(line),
                            "text": line.rstrip("\n"),
                        }
                    )
    except OSError as exc:
        return {"file": str(path), "error": str(exc), "pattern_counts": counts, "hits": hits}
    return {"file": str(path), "pattern_counts": counts, "hits": hits}


def scan_logs(run_dir: Path, max_hits: int) -> dict[str, Any]:
    log_dir = run_dir / "log"
    files = sorted(path for path in log_dir.glob("*") if path.is_file())
    total_counts = {name: 0 for name in LOG_PATTERNS}
    hits: list[dict[str, Any]] = []
    per_file = []
    for path in files:
        result = scan_log_file(path, max_hits=max_hits)
        per_file.append(result)
        for name, count in result.get("pattern_counts", {}).items():
            total_counts[name] = total_counts.get(name, 0) + int(count)
        for hit in result.get("hits", []):
            if len(hits) < max_hits:
                hits.append(hit)
    return {
        "files": [str(path) for path in files],
        "pattern_counts": total_counts,
        "hits": hits,
    }


def fire_record(
    index: int,
    cycles: np.ndarray,
    times: np.ndarray,
    values: np.ndarray,
    denom: np.ndarray | None,
) -> dict[str, Any]:
    unresolved = float(values[index])
    record: dict[str, Any] = {
        "history_index": int(index),
        "cycle": int(cycles[index]) if np.isfinite(cycles[index]) else None,
        "t_s": json_float(times[index]) if index < len(times) else None,
        "E_redistribution_unresolved_erg": unresolved,
    }
    if denom is not None and index < len(denom) and denom[index] != 0.0:
        record["E_unresolved_over_E_denom"] = float(unresolved / denom[index])
    return record


def audit_summary(handle: h5py.File) -> dict[str, Any] | None:
    required = [
        AUDIT_BASE + "I_raw",
        AUDIT_BASE + "K_remap_total",
        AUDIT_BASE + "K_node_postBC",
    ]
    if not all(name in handle for name in required):
        return None

    i_raw = np.asarray(handle[AUDIT_BASE + "I_raw"][...], dtype=float)
    k_remap = np.asarray(handle[AUDIT_BASE + "K_remap_total"][...], dtype=float)
    k_node = np.asarray(handle[AUDIT_BASE + "K_node_postBC"][...], dtype=float)
    n = len(i_raw)
    margin = i_raw + k_remap - k_node
    finite = np.isfinite(margin)
    out: dict[str, Any] = {"rows": int(n)}
    if n > 0 and np.any(finite):
        idx = int(np.nanargmin(np.where(finite, margin, np.nan)))
        cycles = dataset_list(handle, AUDIT_BASE + "cycle", n)
        times = dataset_list(handle, AUDIT_BASE + "t", n)
        out["min_global_tentative_internal_margin_erg"] = float(margin[idx])
        out["min_margin_cycle"] = int(cycles[idx])
        out["min_margin_t_s"] = json_float(times[idx])
        out["min_margin_components_erg"] = {
            "I_raw": float(i_raw[idx]),
            "K_remap_total": float(k_remap[idx]),
            "K_node_postBC": float(k_node[idx]),
        }

    optional_max_abs = {
        "F_floor": AUDIT_BASE + "F_floor",
    }
    for label, name in optional_max_abs.items():
        arr = dataset_array(handle, name)
        if arr is not None and arr.size:
            out[f"max_abs_{label}"] = float(np.nanmax(np.abs(arr.astype(float))))

    optional_min = {
        "min_tentative_e_e_erg_per_g": AUDIT_BASE + "min_tentative_e_e",
        "min_tentative_e_i_erg_per_g": AUDIT_BASE + "min_tentative_e_i",
    }
    for label, name in optional_min.items():
        arr = dataset_array(handle, name)
        if arr is not None and arr.size:
            out[label] = float(np.nanmin(arr.astype(float)))

    optional_max = {
        "max_n_cells_negative_dI": AUDIT_BASE + "n_cells_negative_dI",
        "max_n_cells_floor_e": AUDIT_BASE + "n_cells_floor_e",
        "max_n_cells_floor_i": AUDIT_BASE + "n_cells_floor_i",
    }
    for label, name in optional_max.items():
        arr = dataset_array(handle, name)
        if arr is not None and arr.size:
            out[label] = int(np.nanmax(arr.astype(float)))
    return out


def snapshot_summary_for_cycle(run_dir: Path, cycle: int | None) -> dict[str, Any] | None:
    if cycle is None:
        return None
    snapshots = []
    for path in sorted((run_dir / "results").glob("*.h5")):
        if path.name.endswith("_history.h5"):
            continue
        try:
            with h5py.File(path, "r") as handle:
                step = int(handle.attrs.get("step", handle.attrs.get("cycle", -1)))
                snapshots.append((abs(step - cycle), step, path))
        except OSError:
            continue
    if not snapshots:
        return None
    _, step, path = min(snapshots, key=lambda item: (item[0], item[1]))
    try:
        with h5py.File(path, "r") as handle:
            out: dict[str, Any] = {
                "file": str(path),
                "step": step,
                "t_s": json_float(handle.attrs.get("t")),
                "exact_step_match": step == cycle,
            }
            for label, name in (
                ("rho_g_cm3", "hydro/rho"),
                ("Te_eV", "hydro/Te"),
                ("Ti_eV", "hydro/Ti"),
            ):
                arr = dataset_array(handle, name)
                if arr is not None and arr.size:
                    arr = arr.astype(float)
                    out[label] = {
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "mean": float(np.nanmean(arr)),
                    }
            vr = dataset_array(handle, "mesh/v_r")
            vz = dataset_array(handle, "mesh/v_z")
            if vr is not None and vz is not None and vr.shape == vz.shape and vr.size:
                speed = np.sqrt(vr.astype(float) * vr.astype(float) + vz.astype(float) * vz.astype(float))
                out["node_speed_cm_s"] = {
                    "max": float(np.nanmax(speed)),
                    "mean": float(np.nanmean(speed)),
                }
            return out
    except OSError:
        return None


def scan_history_file(
    path: Path,
    run_dir: Path,
    threshold: float,
    max_fire_details: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(path),
        "signature_dataset_present": False,
        "fire_count": 0,
        "max_abs_unresolved_erg": None,
        "fire_details": [],
    }
    try:
        with h5py.File(path, "r") as handle:
            dataset_name = None
            if HISTORY_SIGNATURE in handle:
                dataset_name = HISTORY_SIGNATURE
            elif HISTORY_SIGNATURE_COMPAT in handle:
                dataset_name = HISTORY_SIGNATURE_COMPAT
            result["audit"] = audit_summary(handle)
            if dataset_name is None:
                return result

            values = np.asarray(handle[dataset_name][...], dtype=float)
            n = len(values)
            cycles = dataset_list(handle, "cycle", n)
            times = dataset_list(handle, "t", n)
            denom = dataset_array(handle, "energy/E_denom")
            if denom is not None and len(denom) != n:
                denom = None

            finite = np.isfinite(values)
            fire_mask = finite & (np.abs(values) > threshold)
            indices = np.flatnonzero(fire_mask)
            result["signature_dataset_present"] = True
            result["signature_dataset"] = dataset_name
            result["history_rows"] = int(n)
            result["fire_count"] = int(indices.size)
            result["max_abs_unresolved_erg"] = (
                float(np.nanmax(np.abs(values))) if values.size else 0.0
            )
            result["sum_abs_unresolved_erg"] = float(np.nansum(np.abs(values[finite])))
            if denom is not None and values.size:
                valid_ratio = finite & np.isfinite(denom) & (denom != 0.0)
                if np.any(valid_ratio):
                    result["max_abs_unresolved_over_E_denom"] = float(
                        np.nanmax(np.abs(values[valid_ratio] / denom[valid_ratio]))
                    )
            details = []
            for index in indices[:max_fire_details]:
                record = fire_record(int(index), cycles, times, values, denom)
                record["nearest_snapshot"] = snapshot_summary_for_cycle(
                    run_dir, record.get("cycle")
                )
                record["per_cell_D_C_available"] = False
                record["per_cell_D_C_note"] = (
                    "Current production source does not persist per-cell T31 D/C values."
                )
                details.append(record)
            result["fire_details"] = details
            if indices.size:
                max_index = int(indices[np.nanargmax(np.abs(values[indices]))])
                result["max_fire"] = fire_record(max_index, cycles, times, values, denom)
                result["first_fire"] = fire_record(int(indices[0]), cycles, times, values, denom)
            return result
    except OSError as exc:
        result["error"] = str(exc)
        return result


def scan_histories(
    run_dir: Path,
    threshold: float,
    max_fire_details: int,
) -> dict[str, Any]:
    files = sorted((run_dir / "results").glob("*history.h5"))
    histories = [
        scan_history_file(path, run_dir, threshold, max_fire_details)
        for path in files
    ]
    fire_count = sum(int(item.get("fire_count", 0)) for item in histories)
    signature_present = any(bool(item.get("signature_dataset_present")) for item in histories)
    max_abs_values = [
        item.get("max_abs_unresolved_erg")
        for item in histories
        if item.get("max_abs_unresolved_erg") is not None
    ]
    return {
        "files": [str(path) for path in files],
        "signature_dataset_present": signature_present,
        "fire_count": fire_count,
        "max_abs_unresolved_erg": max(max_abs_values) if max_abs_values else None,
        "histories": histories,
    }


def evidence_status(config: dict[str, Any], histories: dict[str, Any]) -> str:
    t31_enabled = config.get("t31_enabled")
    signature_present = bool(histories.get("signature_dataset_present"))
    fire_count = int(histories.get("fire_count", 0))
    if signature_present and fire_count > 0:
        return "instrumented_fire"
    if signature_present:
        return "instrumented_no_fire"
    if t31_enabled is True:
        return "t31_configured_but_history_signature_missing"
    if t31_enabled is False:
        return "t31_disabled"
    return "not_t31_instrumented_or_unknown"


def expand_run_globs(patterns: list[str]) -> list[Path]:
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for raw in glob.glob(pattern):
            path = Path(raw)
            if path.is_file():
                path = path.parent
            if path.is_dir():
                paths[str(path)] = path
    return [paths[key] for key in sorted(paths)]


def scan_run(
    run_dir: Path,
    threshold: float,
    max_log_hits: int,
    max_fire_details: int,
) -> dict[str, Any]:
    config = extract_run_config(run_dir)
    logs = scan_logs(run_dir, max_hits=max_log_hits)
    histories = scan_histories(run_dir, threshold, max_fire_details)
    status = evidence_status(config, histories)
    notes = []
    if status == "not_t31_instrumented_or_unknown":
        notes.append(
            "No T31 config flags and no history redistribution signature were found; "
            "log scan can only rule out explicit warning/debug tags retained in the log."
        )
    elif status == "t31_configured_but_history_signature_missing":
        notes.append(
            "Frozen config enables T31 but history lacks energy/redistribution_unresolved; "
            "this should be investigated before treating the run as no-fire evidence."
        )
    if logs["pattern_counts"].get("expected_t31_tags", 0) == 0:
        notes.append("No [t31_redistribute] or [t31_emergency] log tags found.")
    return {
        "run_dir": str(run_dir),
        "config": config,
        "logs": logs,
        "history": histories,
        "evidence_status": status,
        "observed_fire_count": int(histories.get("fire_count", 0)),
        "notes": notes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-glob",
        action="append",
        default=[],
        help="Run directory glob to scan. May be passed more than once.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Write summary JSON to this path. Defaults to stdout.",
    )
    parser.add_argument(
        "--threshold-erg",
        type=float,
        default=0.0,
        help="Absolute unresolved-energy threshold counted as a fire.",
    )
    parser.add_argument(
        "--max-log-hits",
        type=int,
        default=20,
        help="Maximum matching log lines retained in JSON per run.",
    )
    parser.add_argument(
        "--max-fire-details",
        type=int,
        default=20,
        help="Maximum per-fire detail rows retained per history file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_globs = args.run_glob or DEFAULT_RUN_GLOBS
    run_dirs = expand_run_globs(run_globs)
    print(f"Scanning {len(run_dirs)} run directories", file=sys.stderr)
    runs = [
        scan_run(
            run_dir,
            threshold=args.threshold_erg,
            max_log_hits=max(args.max_log_hits, 0),
            max_fire_details=max(args.max_fire_details, 0),
        )
        for run_dir in run_dirs
    ]
    total_fire_count = sum(int(run.get("observed_fire_count", 0)) for run in runs)
    payload = {
        "tool": "tools/diagnostics/audit_t31_emergency_fire.py",
        "run_globs": run_globs,
        "threshold_erg": args.threshold_erg,
        "source_observability": {
            "emergency_branch_source": "src/hydro/ale_driver.cu",
            "history_signature": HISTORY_SIGNATURE,
            "history_signature_compat": HISTORY_SIGNATURE_COMPAT,
            "log_patterns_scanned": sorted(LOG_PATTERNS.keys()),
            "note": (
                "Current production source records D > C through "
                "E_redistribution_unresolved; it does not persist per-cell D/C."
            ),
        },
        "run_count": len(runs),
        "total_observed_fire_count": total_fire_count,
        "status": "fires_observed" if total_fire_count else "no_fires_observed",
        "runs": runs,
    }

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.write_text(text, encoding="utf-8")
        print(
            f"Wrote {out_path} with total_observed_fire_count={total_fire_count}",
            file=sys.stderr,
        )
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
