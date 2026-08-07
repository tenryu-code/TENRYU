#!/usr/bin/env python3
"""Characterize the I1B-LITE 2D RZ admissible drive fraction.

This harness is diagnostic-first: it only evaluates geometry admissibility for
the existing 2D RZ I1 capsule deck.  It does not freeze a CI gate.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "I1B-LITE"
MODE = "i1b_lite_2d_rz_characterize"
ERG_PER_KJ = 1.0e10
CAPSULE_NOMINAL_ERG = 25.0e3 * 1.0e7
ADMISSIBILITY_FLOOR_REL = 1.0e-8
DEFAULT_COARSE_GRID = (0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_BISECTION_STEPS = 4
DEFAULT_NR = 64
DEFAULT_NZ = 128
DEFAULT_RAYS_PER_BEAM = 72
DEFAULT_T_END_S = 5.0e-11
DEFAULT_DT_S = 2.0e-13
DEFAULT_MAX_STEPS = 1_000_000
NUMBER_RE = r"[-+0-9.eE]+"
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_i1_capsule\].*?"
    r"mode=(?P<mode>\S+).*?nr=(?P<nr>\d+).*?nz=(?P<nz>\d+).*?"
    r"seed=(?P<seed>\d+).*?outdir=(?P<outdir>\S+).*?"
    r"t_end_s=(?P<t_end>{n}).*?max_steps=(?P<max_steps>\d+).*?"
    r"laser_input_erg=(?P<laser>{n}).*?"
    r"pulse_duration_s=(?P<pulse>{n}).*?rays_per_beam=(?P<rays>\d+)".format(n=NUMBER_RE)
)
NEWTON_RE = re.compile(
    r"\[fld_2d_rz_diagnostics_h3\].*?"
    r"newton_converged=(?P<converged>[0-9]+).*?"
    r"newton_cap_hit=(?P<cap_hit>[0-9]+).*?"
    r"newton_invalid=(?P<invalid>[0-9]+).*?"
    r"newton_resid_abs_max=(?P<resid_abs>[-+0-9.eE]+).*?"
    r"newton_resid_rel_max=(?P<resid_rel>[-+0-9.eE]+).*?"
    r"newton_reject=(?P<reject>[0-9]+).*?"
    r"newton_reject_resid_rel_max=(?P<reject_resid_rel>[-+0-9.eE]+)"
)
PLOT_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>[0-9]{4,})\.h5$")


@dataclass(frozen=True)
class MeshSnapshot:
    path: Path
    t: float
    step: int
    r: Any
    z: Any


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_float_list(value: str) -> list[float]:
    out = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    if any(not math.isfinite(item) or item <= 0.0 for item in out):
        raise argparse.ArgumentTypeError("drive fractions must be finite and positive")
    return sorted(dict.fromkeys(out))


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = read_text(log_path)
    out: dict[str, Any] = {}
    match = DECK_LINE_RE.search(text)
    if match:
        out.update(
            {
                "deck_mode": match.group("mode"),
                "deck_nr": int(match.group("nr")),
                "deck_nz": int(match.group("nz")),
                "deck_seed": int(match.group("seed")),
                "deck_outdir": match.group("outdir"),
                "deck_t_end_s": float(match.group("t_end")),
                "deck_max_steps": int(match.group("max_steps")),
                "deck_laser_input_erg": float(match.group("laser")),
                "deck_pulse_duration_s": float(match.group("pulse")),
                "deck_rays_per_beam": int(match.group("rays")),
            }
        )
    return out


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def allocate_run_dir(out_root: Path, stem: str) -> tuple[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    for idx in range(1000):
        run_id = stem if idx == 0 else f"{stem}_{idx:03d}"
        path = out_root / run_id
        if not path.exists():
            return run_id, path
    raise RuntimeError(f"could not allocate unique run directory under {out_root}")


def discover_results_dir(outdir: Path) -> Path:
    candidates = [outdir / "results"]
    if outdir.parent.exists():
        candidates.extend(path / "results" for path in outdir.parent.glob(f"{outdir.name}_*"))
    valid = [path for path in candidates if path.exists() and any(PLOT_RE.match(item.name) for item in path.glob("*.h5"))]
    if not valid:
        raise FileNotFoundError(f"no plot HDF5 files found under {outdir} or sibling suffixed outdirs")
    return max(valid, key=lambda path: path.stat().st_mtime)


def locate_plot_files(results_dir: Path) -> list[Path]:
    def plot_index(path: Path) -> int:
        match = PLOT_RE.match(path.name)
        return int(match.group("index")) if match else -1

    return sorted(
        (path for path in results_dir.glob("*.h5") if PLOT_RE.match(path.name) and not path.name.endswith("_history.h5")),
        key=plot_index,
    )


def locate_history_file(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.glob("*_history.h5"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def locate_run_info(outdir: Path, results_dir: Path | None = None) -> Path | None:
    candidates = [outdir / "run_info.json"]
    if results_dir is not None:
        candidates.append(results_dir.parent / "run_info.json")
    for path in candidates:
        if path.exists():
            return path
    return None


def locate_frozen_config(outdir: Path, results_dir: Path | None = None) -> Path | None:
    candidates: list[Path] = []
    for base in [outdir, results_dir.parent if results_dir is not None else None]:
        if base is None:
            continue
        config_dir = base / "config"
        if config_dir.exists():
            candidates.extend(config_dir.glob("*_frozen.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalar_dataset(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else default


def reshape_mesh_nodes(value: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(value, dtype=float)
    expected = (nr + 1, nz + 1)
    if arr.shape == expected:
        return arr
    if arr.size == expected[0] * expected[1]:
        return arr.reshape(expected)
    raise ValueError(f"{name} has shape {arr.shape}, expected {expected}")


def read_mesh_snapshot(path: Path, nr: int, nz: int) -> MeshSnapshot:
    import h5py

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = scalar_dataset(handle, "time_state/t", t_value)
        step = int(scalar_dataset(handle, "time_state/step", 0.0))
        return MeshSnapshot(
            path=path,
            t=t_value,
            step=step,
            r=reshape_mesh_nodes(handle["mesh/x_r"][()], nr, nz, "mesh/x_r"),
            z=reshape_mesh_nodes(handle["mesh/x_z"][()], nr, nz, "mesh/x_z"),
        )


def rz_quad_volume_exact(r: Any, z: Any) -> Any:
    import math as _math

    rs = [r[:-1, :-1], r[1:, :-1], r[1:, 1:], r[:-1, 1:]]
    zs = [z[:-1, :-1], z[1:, :-1], z[1:, 1:], z[:-1, 1:]]
    total = 0.0
    for k in range(4):
        kp = (k + 1) & 3
        total = total + (rs[k] * zs[kp] - rs[kp] * zs[k]) * (rs[k] + rs[kp])
    return _math.pi / 3.0 * total


def corner_jacobians(r: Any, z: Any) -> list[Any]:
    r0, r1, r2, r3 = r[:-1, :-1], r[1:, :-1], r[1:, 1:], r[:-1, 1:]
    z0, z1, z2, z3 = z[:-1, :-1], z[1:, :-1], z[1:, 1:], z[:-1, 1:]

    def cross(ar: Any, az: Any, br: Any, bz: Any) -> Any:
        return ar * bz - az * br

    return [
        cross(r1 - r0, z1 - z0, r3 - r0, z3 - z0),
        cross(r1 - r0, z1 - z0, r2 - r1, z2 - z1),
        cross(r2 - r3, z2 - z3, r2 - r1, z2 - z1),
        cross(r2 - r3, z2 - z3, r3 - r0, z3 - z0),
    ]


def signed_ratio_min(new_value: Any, old_value: Any) -> float:
    import numpy as np

    ratio = np.where(old_value != 0.0, new_value / old_value, -1.0e300)
    finite = ratio[np.isfinite(ratio)]
    if finite.size == 0:
        return -math.inf
    return float(np.min(finite))


def evaluate_mesh_pair(prev: MeshSnapshot, cur: MeshSnapshot) -> dict[str, Any]:
    import numpy as np

    old_v = rz_quad_volume_exact(prev.r, prev.z)
    new_v = rz_quad_volume_exact(cur.r, cur.z)
    old_j = corner_jacobians(prev.r, prev.z)
    new_j = corner_jacobians(cur.r, cur.z)
    old_gauss = 0.25 * sum(old_j)
    new_gauss = 0.25 * sum(new_j)
    corner_min = min(signed_ratio_min(new, old) for old, new in zip(old_j, new_j))
    old_v_finite = old_v[np.isfinite(old_v)]
    new_v_finite = new_v[np.isfinite(new_v)]
    return {
        "from_step": prev.step,
        "to_step": cur.step,
        "from_t_s": prev.t,
        "to_t_s": cur.t,
        "from_path": str(prev.path),
        "to_path": str(cur.path),
        "min_rz_volume_rel": signed_ratio_min(new_v, old_v),
        "min_corner_j_rel": corner_min,
        "min_gauss_j_rel": signed_ratio_min(new_gauss, old_gauss),
        "min_rz_volume_cm3": float(np.min(new_v_finite)) if new_v_finite.size else -math.inf,
        "negative_rz_volume_count": int(np.count_nonzero(~np.isfinite(new_v) | (new_v <= 0.0))),
        "old_negative_rz_volume_count": int(np.count_nonzero(~np.isfinite(old_v) | (old_v <= 0.0))),
    }


def evaluate_geometry_from_snapshots(
    results_dir: Path,
    nr: int,
    nz: int,
    plot_every_steps: int,
) -> dict[str, Any]:
    plot_files = locate_plot_files(results_dir)
    if len(plot_files) < 2:
        return {
            "available": False,
            "source": "plot_snapshot_mesh",
            "source_path": str(results_dir),
            "missing": ["plot_snapshot_mesh_pair"],
            "plot_file_count": len(plot_files),
        }
    snapshots = [read_mesh_snapshot(path, nr, nz) for path in plot_files]
    pairs = [evaluate_mesh_pair(prev, cur) for prev, cur in zip(snapshots[:-1], snapshots[1:])]
    min_rz = min(float(pair["min_rz_volume_rel"]) for pair in pairs)
    min_corner = min(float(pair["min_corner_j_rel"]) for pair in pairs)
    min_gauss = min(float(pair["min_gauss_j_rel"]) for pair in pairs)
    negative_snapshot_count = int(pairs[0]["old_negative_rz_volume_count"]) + sum(
        int(pair["negative_rz_volume_count"]) for pair in pairs
    )
    missing: list[str] = []
    coverage = "all_accepted_steps" if plot_every_steps == 1 else "sampled_plot_steps"
    if plot_every_steps != 1:
        missing.append("plot_every_steps_is_1_for_all_run_geometry")
    return {
        "available": not missing,
        "source": "plot_snapshot_mesh_candidate_mesh_admissibility_formula",
        "source_path": str(results_dir),
        "coverage": coverage,
        "plot_file_count": len(plot_files),
        "plot_every_steps": plot_every_steps,
        "first_step": snapshots[0].step,
        "last_step": snapshots[-1].step,
        "first_t_s": snapshots[0].t,
        "last_t_s": snapshots[-1].t,
        "min_rz_volume_rel": min_rz,
        "min_corner_j_rel": min_corner,
        "min_gauss_j_rel": min_gauss,
        "negative_rz_volume_count": negative_snapshot_count,
        "worst_pairs": sorted(
            pairs,
            key=lambda pair: min(
                float(pair["min_rz_volume_rel"]),
                float(pair["min_corner_j_rel"]),
                float(pair["min_gauss_j_rel"]),
            ),
        )[:5],
        "missing": missing,
    }


def hdf5_series_max(path: Path | None, dataset_path: str) -> int | None:
    if path is None or not path.exists():
        return None
    import h5py
    import numpy as np

    key = dataset_path[1:] if dataset_path.startswith("/") else dataset_path
    with h5py.File(path, "r") as handle:
        if key not in handle:
            return None
        arr = np.asarray(handle[key][()], dtype=float).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return None
        return int(np.max(finite))


def count_jsonl_events(outdir: Path | None) -> int | None:
    if outdir is None:
        return None
    path = outdir / "escape_valve_events.jsonl"
    if not path.exists():
        return None
    return sum(1 for line in read_text(path).splitlines() if line.strip())


def parse_newton_diagnostic(log_path: Path) -> dict[str, Any]:
    records = []
    for line in read_text(log_path).splitlines():
        match = NEWTON_RE.search(line)
        if match:
            records.append(
                {
                    "newton_invalid_count": int(match.group("invalid")),
                    "newton_cap_hit_count": int(match.group("cap_hit")),
                    "newton_reject_count": int(match.group("reject")),
                    "newton_resid_abs_max": float(match.group("resid_abs")),
                    "newton_resid_rel_max": float(match.group("resid_rel")),
                }
            )
    if not records:
        return {"records": 0, "newton_invalid_count": None}
    return {
        "records": len(records),
        "newton_invalid_count": max(int(row["newton_invalid_count"]) for row in records),
        "newton_cap_hit_count": max(int(row["newton_cap_hit_count"]) for row in records),
        "newton_reject_count": max(int(row["newton_reject_count"]) for row in records),
        "newton_resid_abs_max": max(float(row["newton_resid_abs_max"]) for row in records),
        "newton_resid_rel_max": max(float(row["newton_resid_rel_max"]) for row in records),
    }


def run_audit_summary(run_info: Path | None, history: Path | None, frozen_config: Path | None, outdir: Path) -> dict[str, Any]:
    output = outdir / "audit_summary.json"
    if run_info is None or history is None:
        return {
            "path": str(output),
            "created": False,
            "missing_inputs": [
                name
                for name, value in (("run_info", run_info), ("history_hdf5", history))
                if value is None
            ],
        }
    cmd = [
        os.environ.get("PYTHON", "python3"),
        "tools/validation/audit_summary.py",
        "--run-info",
        str(run_info),
        "--history-hdf5",
        str(history),
        "--output",
        str(output),
    ]
    if frozen_config is not None:
        cmd.extend(["--frozen-config", str(frozen_config)])
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {
            "path": str(output),
            "created": False,
            "returncode": proc.returncode,
            "stderr": proc.stderr[-4000:],
        }
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(output), "created": False, "error": str(exc)}
    return {"path": str(output), "created": True, "payload": payload}


def count_empty_repair_polytope(log_text: str) -> int:
    explicit = log_text.count("empty_repair_polytope")
    local_empty = 0
    for line in log_text.splitlines():
        if "local_boundary_repositioning" not in line:
            continue
        if "applied_count=0" in line and "ring_size=0" in line and "dof_count=0" in line and "constraint_count=0" in line:
            local_empty += 1
    return explicit + local_empty


def no_escape_valves_from_sources(
    log_path: Path,
    history: Path | None,
    actual_outdir: Path | None,
    audit: dict[str, Any],
) -> dict[str, Any]:
    log_text = read_text(log_path)
    newton = parse_newton_diagnostic(log_path)
    jsonl_events = count_jsonl_events(actual_outdir)
    emergency = hdf5_series_max(
        history, "/diagnostics/escape_valve_audit/v1/emergency_cell_deactivation_count"
    )
    if emergency is None:
        emergency = hdf5_series_max(
            history, "/diagnostics/ale_provenance/v1/emergency_cell_deactivation_fired"
        )
    if emergency is None and jsonl_events is not None:
        emergency = jsonl_events
    escape_activations = hdf5_series_max(
        history, "/diagnostics/ale_provenance/v1/escape_valve_activations"
    )
    class_c = hdf5_series_max(history, "/diagnostics/ale_provenance/v1/class_c_runtime_fires")
    audit_payload = audit.get("payload") if isinstance(audit.get("payload"), dict) else {}
    audit_firings = audit_payload.get("escape_valve_firings") if isinstance(audit_payload, dict) else None
    audit_firings_total = None
    if isinstance(audit_firings, dict) and audit_firings:
        audit_firings_total = sum(1 for value in audit_firings.values() if bool(value))

    values: dict[str, int | None] = {
        "emergency_cell_deactivation_fired_count": emergency,
        "thermal_subcycle_floor_hit_count": log_text.count("[thermal-subcycle] floor hit"),
        "axis_spike_floor_activation_count": log_text.lower().count("axis_spike_floor"),
        "newton_invalid_count": newton["newton_invalid_count"],
        "ale_provenance_escape_valve_activations_count": escape_activations,
        "ale_provenance_class_c_runtime_fires_count": class_c,
        "audit_escape_valve_firings_total": audit_firings_total,
        "escape_valve_events_jsonl_count": jsonl_events if jsonl_events is not None else 0,
    }
    required_keys = (
        "emergency_cell_deactivation_fired_count",
        "thermal_subcycle_floor_hit_count",
        "axis_spike_floor_activation_count",
        "newton_invalid_count",
        "ale_provenance_escape_valve_activations_count",
        "ale_provenance_class_c_runtime_fires_count",
    )
    missing = [key for key in required_keys if values.get(key) is None]
    nonzero = [key for key, value in values.items() if value is not None and int(value) != 0]
    return {
        **{key: int(value) if value is not None else None for key, value in values.items()},
        "newton_diagnostic": newton,
        "missing_counters": missing,
        "nonzero_counters": nonzero,
        "all_zero": not missing and not nonzero,
        "sources": {
            "run_log": str(log_path),
            "history_hdf5": str(history) if history is not None else None,
            "actual_outdir": str(actual_outdir) if actual_outdir is not None else None,
            "audit_summary_json": audit.get("path"),
            "escape_valve_events_jsonl": str(actual_outdir / "escape_valve_events.jsonl")
            if actual_outdir is not None
            else None,
        },
    }


def signal(value: Any, passed: bool, source: str, required: bool = True) -> dict[str, Any]:
    return {"value": value, "passed": bool(passed), "source": source, "required": required}


def evaluate_predicate(
    *,
    exit_code: int,
    timed_out: bool,
    outdir: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    metadata = parse_log_metadata(log_path)
    results_dir: Path | None = None
    history: Path | None = None
    run_info_path: Path | None = None
    frozen_config: Path | None = None
    geometry: dict[str, Any] = {
        "available": False,
        "missing": ["results_dir"],
        "source": "plot_snapshot_mesh_candidate_mesh_admissibility_formula",
    }
    run_info: dict[str, Any] = {}
    analysis_errors: list[str] = []
    try:
        results_dir = discover_results_dir(outdir)
        history = locate_history_file(results_dir)
        run_info_path = locate_run_info(outdir, results_dir)
        frozen_config = locate_frozen_config(outdir, results_dir)
        nr = int(metadata.get("deck_nr", args.nr))
        nz = int(metadata.get("deck_nz", args.nz))
        geometry = evaluate_geometry_from_snapshots(results_dir, nr, nz, args.plot_every_steps)
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        analysis_errors.append(str(exc))
        run_info_path = locate_run_info(outdir, results_dir)
    if run_info_path is not None:
        try:
            run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            analysis_errors.append(f"run_info parse failed: {exc}")

    actual_outdir = results_dir.parent if results_dir is not None else outdir
    audit = run_audit_summary(run_info_path, history, frozen_config, outdir)
    no_escape = no_escape_valves_from_sources(log_path, history, actual_outdir, audit)
    log_text = read_text(log_path)
    empty_repair_polytope_count = count_empty_repair_polytope(log_text)

    t_end = float(metadata.get("deck_t_end_s", args.t_end_s))
    final_t = finite_float(run_info.get("t"))
    termination_reason = str(run_info.get("termination_reason", "missing"))
    reached_t_end = (
        exit_code == 0
        and not timed_out
        and termination_reason == "t_end_reached"
        and final_t is not None
        and final_t >= 0.999 * t_end
    )
    geometry_missing = list(geometry.get("missing", []))
    min_rz = geometry.get("min_rz_volume_rel")
    min_corner = geometry.get("min_corner_j_rel")
    min_gauss = geometry.get("min_gauss_j_rel")
    neg_vol = int(geometry.get("negative_rz_volume_count", 0)) if geometry.get("negative_rz_volume_count") is not None else None

    signals = {
        "reached_t_end": signal(
            reached_t_end,
            reached_t_end,
            str(run_info_path) if run_info_path is not None else "missing run_info.json",
        ),
        "min_corner_j_rel": signal(
            min_corner,
            isinstance(min_corner, (int, float)) and math.isfinite(float(min_corner)) and float(min_corner) > ADMISSIBILITY_FLOOR_REL,
            str(geometry.get("source_path", "missing geometry source")),
        ),
        "min_gauss_j_rel": signal(
            min_gauss,
            isinstance(min_gauss, (int, float)) and math.isfinite(float(min_gauss)) and float(min_gauss) > ADMISSIBILITY_FLOOR_REL,
            str(geometry.get("source_path", "missing geometry source")),
        ),
        "min_rz_volume_rel": signal(
            min_rz,
            isinstance(min_rz, (int, float)) and math.isfinite(float(min_rz)) and float(min_rz) > 0.0,
            str(geometry.get("source_path", "missing geometry source")),
        ),
        "negative_rz_volume_count": signal(
            neg_vol,
            neg_vol == 0,
            str(geometry.get("source_path", "missing geometry source")),
        ),
        "emergency_cell_deactivation_fired_count": signal(
            no_escape.get("emergency_cell_deactivation_fired_count"),
            no_escape.get("emergency_cell_deactivation_fired_count") == 0,
            str(no_escape["sources"].get("history_hdf5") or no_escape["sources"].get("escape_valve_events_jsonl")),
        ),
        "empty_repair_polytope_count": signal(
            empty_repair_polytope_count,
            empty_repair_polytope_count == 0,
            str(log_path),
        ),
        "escape_valve_counters_all_zero": signal(
            no_escape.get("nonzero_counters"),
            bool(no_escape.get("all_zero")),
            str(no_escape["sources"].get("history_hdf5") or log_path),
        ),
    }
    missing_signals = []
    if not run_info:
        missing_signals.append("run_info")
    missing_signals.extend(geometry_missing)
    missing_signals.extend(no_escape.get("missing_counters", []))
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing_signals.append(name)
    failed_signals = [name for name, item in signals.items() if item["required"] and not item["passed"]]
    missing_signals = sorted(set(missing_signals))
    predicate_passed = not timed_out and exit_code == 0 and not analysis_errors and not missing_signals and not failed_signals
    return {
        "passed": predicate_passed,
        "missing_signals": missing_signals,
        "failed_signals": failed_signals,
        "signals": signals,
        "thresholds": {
            "corner_j_rel_floor": ADMISSIBILITY_FLOOR_REL,
            "gauss_j_rel_floor": ADMISSIBILITY_FLOOR_REL,
            "rz_volume_rel_floor": 0.0,
        },
        "geometry": geometry,
        "no_escape_valves": no_escape,
        "audit_summary": audit,
        "run_info": run_info,
        "metadata": metadata,
        "analysis_errors": analysis_errors,
        "paths": {
            "outdir": str(outdir),
            "actual_outdir": str(actual_outdir) if actual_outdir is not None else None,
            "log_path": str(log_path),
            "results_dir": str(results_dir) if results_dir is not None else None,
            "history_hdf5": str(history) if history is not None else None,
            "run_info": str(run_info_path) if run_info_path is not None else None,
            "frozen_config": str(frozen_config) if frozen_config is not None else None,
        },
    }


def active_env(args: argparse.Namespace, outdir: Path, x: float, laser_input_erg: float) -> dict[str, str]:
    return {
        "TENRYU_I1CAPSULE_OUTDIR": str(outdir),
        "TENRYU_I1CAPSULE_LASER_INPUT_ERG": f"{laser_input_erg:.17g}",
        "TENRYU_I1CAPSULE_NR": str(args.nr),
        "TENRYU_I1CAPSULE_NZ": str(args.nz),
        "TENRYU_I1CAPSULE_RAYS_PER_BEAM": str(args.rays_per_beam),
        "TENRYU_I1CAPSULE_SEED": str(args.seed),
        "TENRYU_I1CAPSULE_T_END_S": f"{args.t_end_s:.17g}",
        "TENRYU_I1CAPSULE_DT_INITIAL_S": f"{args.dt_initial_s:.17g}",
        "TENRYU_I1CAPSULE_DT_MAX_S": f"{args.dt_max_s:.17g}",
        "TENRYU_I1CAPSULE_PLOT_EVERY_STEPS": str(args.plot_every_steps),
        "TENRYU_I1CAPSULE_HISTORY_EVERY_STEPS": str(args.history_every_steps),
        "TENRYU_I1CAPSULE_MAX_STEPS": str(args.max_steps),
        "TENRYU_I1CAPSULE_VERBOSITY": args.verbosity,
        "TENRYU_I1CAPSULE_PROFILE_ENABLED": "true",
        "TENRYU_I1CAPSULE_PROFILE_ENFORCE": "true",
        "TENRYU_I1CAPSULE_ALE_PROVENANCE_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_ICF_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_CONSERVATION_DIAGNOSTICS": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_GUARD": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_GUARD": "true",
        "TENRYU_I1CAPSULE_IN_HYDRO_GAUSS_J_FLOOR_REL": f"{ADMISSIBILITY_FLOOR_REL:.17g}",
        "TENRYU_I1CAPSULE_IN_HYDRO_RZ_VOLUME_FLOOR_REL": f"{ADMISSIBILITY_FLOOR_REL:.17g}",
        "TENRYU_I1CAPSULE_REJECT_ZERO_GAUSS_J": "true",
        "TENRYU_I1CAPSULE_ZERO_GAUSS_J_FLOOR_REL": f"{ADMISSIBILITY_FLOOR_REL:.17g}",
        "TENRYU_I1B_LITE_DRIVE_FRACTION_X": f"{x:.17g}",
    }


def laser_input_erg_from_x(x: float) -> float:
    return x * CAPSULE_NOMINAL_ERG


def laser_input_kj_from_erg(laser_input_erg: float) -> float:
    return laser_input_erg / ERG_PER_KJ


def run_one(args: argparse.Namespace, x: float, label: str) -> dict[str, Any]:
    laser_input_erg = laser_input_erg_from_x(x)
    stem = f"{label}_x{safe_float_token(x)}_nr{args.nr}_nz{args.nz}_seed{args.seed}"
    run_id, outdir = allocate_run_dir(Path(args.out_root), stem)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    env_overrides = active_env(args, outdir, x, laser_input_erg)
    env = os.environ.copy()
    env.update(env_overrides)
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    wall_time_s = time.monotonic() - start
    predicate = evaluate_predicate(
        exit_code=proc.returncode,
        timed_out=timed_out,
        outdir=outdir,
        log_path=log_path,
        args=args,
    )
    result = "ADMISSIBLE" if predicate["passed"] else "INADMISSIBLE"
    if timed_out:
        result = "WALL_TIME_TIMEOUT"
    elif proc.returncode != 0:
        result = "INADMISSIBLE"
    payload = {
        "schema_version": "tenryu.i1b_lite_2d_rz_run_predicate.v1",
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "mode": MODE,
        "x_laser_fraction": x,
        "x_reference": "capsule_nominal_laser_input",
        "capsule_nominal_erg": CAPSULE_NOMINAL_ERG,
        "capsule_nominal_kj": laser_input_kj_from_erg(CAPSULE_NOMINAL_ERG),
        "laser_input_erg": laser_input_erg,
        "laser_input_kj": laser_input_kj_from_erg(laser_input_erg),
        "deck_path": args.deck,
        "result": result,
        "geometry_admissible": bool(predicate["passed"]),
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
        "predicate": predicate,
        "env_overrides": env_overrides,
    }
    write_json(outdir / "predicate_summary.json", payload)
    return row_from_payload(payload, outdir / "predicate_summary.json")


def row_from_payload(payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    predicate = payload["predicate"]
    signals = predicate.get("signals", {})
    return {
        "run_id": payload["run_id"],
        "x_laser_fraction": payload["x_laser_fraction"],
        "laser_input_erg": payload["laser_input_erg"],
        "laser_input_kj": payload["laser_input_kj"],
        "result": payload["result"],
        "geometry_admissible": payload["geometry_admissible"],
        "exit_code": payload["exit_code"],
        "timed_out": payload["timed_out"],
        "wall_time_s": payload["wall_time_s"],
        "predicate_summary_json": str(summary_path),
        "missing_signals": predicate.get("missing_signals", []),
        "failed_signals": predicate.get("failed_signals", []),
        "final_t_s": predicate.get("run_info", {}).get("t"),
        "termination_reason": predicate.get("run_info", {}).get("termination_reason"),
        "min_rz_volume_rel": signals.get("min_rz_volume_rel", {}).get("value"),
        "min_corner_j_rel": signals.get("min_corner_j_rel", {}).get("value"),
        "min_gauss_j_rel": signals.get("min_gauss_j_rel", {}).get("value"),
        "negative_rz_volume_count": signals.get("negative_rz_volume_count", {}).get("value"),
        "emergency_cell_deactivation_fired_count": signals.get(
            "emergency_cell_deactivation_fired_count", {}
        ).get("value"),
        "empty_repair_polytope_count": signals.get("empty_repair_polytope_count", {}).get("value"),
        "escape_valve_counters_all_zero": signals.get("escape_valve_counters_all_zero", {}).get("passed"),
    }


def find_initial_bracket(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    sorted_rows = sorted(rows, key=lambda row: float(row["x_laser_fraction"]))
    last_admissible: dict[str, Any] | None = None
    for row in sorted_rows:
        if bool(row["geometry_admissible"]):
            last_admissible = row
            continue
        if last_admissible is not None:
            return last_admissible, row
    return last_admissible, None


def characterize(args: argparse.Namespace) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for x in args.coarse_grid:
        rows.append(run_one(args, x, "coarse"))

    lower_row, upper_row = find_initial_bracket(rows)
    bisection_rows: list[dict[str, Any]] = []
    status = "bracketed"
    x_geom_lower = None
    x_geom_upper = None
    x_geom_selected = None
    x_gate = None
    if lower_row is None:
        status = "no_admissible_point"
    elif upper_row is None:
        max_coarse = max(args.coarse_grid)
        if any(bool(row["geometry_admissible"]) and abs(float(row["x_laser_fraction"]) - max_coarse) < 1.0e-15 for row in rows):
            status = "nominal_admissible_premise_change" if abs(max_coarse - 1.0) < 1.0e-15 else "upper_grid_admissible"
        else:
            status = "unbracketed"
    else:
        lower = float(lower_row["x_laser_fraction"])
        upper = float(upper_row["x_laser_fraction"])
        for step in range(args.bisection_steps):
            mid = 0.5 * (lower + upper)
            row = run_one(args, mid, f"bisect{step + 1:02d}")
            bisection_rows.append(row)
            rows.append(row)
            if bool(row["geometry_admissible"]):
                lower = mid
                lower_row = row
            else:
                upper = mid
                upper_row = row
        x_geom_lower = lower
        x_geom_upper = upper
        x_geom_selected = lower
        x_gate = 0.8 * x_geom_selected

    summary = {
        "schema_version": "tenryu.i1b_lite_2d_rz_characterization.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "x_mapping": {
            "definition": "X = LASER_INPUT_ERG / CAPSULE_NOMINAL_ERG",
            "capsule_nominal_erg": CAPSULE_NOMINAL_ERG,
            "capsule_nominal_kj": laser_input_kj_from_erg(CAPSULE_NOMINAL_ERG),
            "laser_input_erg": "X * CAPSULE_NOMINAL_ERG",
            "source": "examples/verification/2d_rz_i1_capsule.py LASER_INPUT_ERG default",
        },
        "nominal_cross_check": {
            "x": 1.0,
            "laser_input_erg": laser_input_erg_from_x(1.0),
            "laser_input_kj": laser_input_kj_from_erg(laser_input_erg_from_x(1.0)),
            "deck_default_laser_input_erg": 2.5e11,
            "deck_default_laser_input_kj": 25.0,
            "matches_capsule_deck_default": abs(laser_input_erg_from_x(1.0) - 2.5e11) == 0.0,
        },
        "separate_1d_radial_drive_note": {
            "source": "tools/validation/run_i1_radial.py",
            "gxii_reference_e_erg": 3600.0 * 1.0e7,
            "x_0p05_laser_input_erg": 0.05 * 3600.0 * 1.0e7,
            "x_0p05_laser_input_kj": laser_input_kj_from_erg(0.05 * 3600.0 * 1.0e7),
            "note": "This 1D radial drive is separate from the 25 kJ capsule nominal and is not the I1B-LITE X reference.",
        },
        "t4_anchor_drive_mismatch_note": (
            "T4 physics-activation thresholds cannot simply scale the 180 J 1D-radial metrics; "
            "they must use physical lower bounds at the capsule X_gate."
        ),
        "predicate_definition": {
            "reached_t_end": True,
            "min_corner_j_rel": f"> {ADMISSIBILITY_FLOOR_REL:.1e}",
            "min_gauss_j_rel": f"> {ADMISSIBILITY_FLOOR_REL:.1e}",
            "min_rz_volume_rel": "> 0.0",
            "negative_rz_volume_count": 0,
            "emergency_cell_deactivation_fired_count": 0,
            "empty_repair_polytope_count": 0,
            "escape_valve_counters_all_zero": True,
            "missing_required_signal_policy": "fail_closed",
            "geometry_only": True,
        },
        "budget": {
            "nr": args.nr,
            "nz": args.nz,
            "rays_per_beam": args.rays_per_beam,
            "t_end_s": args.t_end_s,
            "dt_initial_s": args.dt_initial_s,
            "dt_max_s": args.dt_max_s,
            "plot_every_steps": args.plot_every_steps,
            "history_every_steps": args.history_every_steps,
            "max_steps": args.max_steps,
            "choice": (
                "64x128 cells, 72 rays/beam, t_end=5.0e-11 s by default: this is a cheap "
                "early-drive/ALE geometry-stress window, matching the prior I1B failure timescale "
                "without attempting a production-length implosion."
            ),
        },
        "coarse_grid": list(args.coarse_grid),
        "bisection_steps": args.bisection_steps,
        "characterization_status": status,
        "x_geom_lower": x_geom_lower,
        "x_geom_upper": x_geom_upper,
        "x_geom_selected": x_geom_selected,
        "x_gate_proposed": x_gate,
        "bracket": {
            "lower_run_id": lower_row.get("run_id") if lower_row is not None else None,
            "lower_x": lower_row.get("x_laser_fraction") if lower_row is not None else None,
            "upper_run_id": upper_row.get("run_id") if upper_row is not None else None,
            "upper_x": upper_row.get("x_laser_fraction") if upper_row is not None else None,
        },
        "coarse_runs": [row for row in rows if str(row["run_id"]).startswith("coarse_")],
        "bisection_runs": bisection_rows,
        "runs": sorted(rows, key=lambda row: (float(row["x_laser_fraction"]), str(row["run_id"]))),
    }
    write_json(Path(args.summary_json), summary)
    return summary


def dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for x in args.coarse_grid:
        laser_input_erg = laser_input_erg_from_x(x)
        rows.append(
            {
                "x_laser_fraction": x,
                "laser_input_erg": laser_input_erg,
                "laser_input_kj": laser_input_kj_from_erg(laser_input_erg),
                "predicate_result": "not_evaluated_dry_run",
                "planned": True,
            }
        )
    return {
        "schema_version": "tenryu.i1b_lite_2d_rz_dry_run.v1",
        "mode": args.mode,
        "x_mapping": {
            "definition": "X = LASER_INPUT_ERG / CAPSULE_NOMINAL_ERG",
            "capsule_nominal_erg": CAPSULE_NOMINAL_ERG,
            "capsule_nominal_kj": laser_input_kj_from_erg(CAPSULE_NOMINAL_ERG),
            "laser_input_erg": "X * CAPSULE_NOMINAL_ERG",
        },
        "nominal_cross_check": {
            "x": 1.0,
            "laser_input_erg": laser_input_erg_from_x(1.0),
            "laser_input_kj": laser_input_kj_from_erg(laser_input_erg_from_x(1.0)),
            "deck_default_laser_input_erg": 2.5e11,
            "deck_default_laser_input_kj": 25.0,
            "matches_capsule_deck_default": abs(laser_input_erg_from_x(1.0) - 2.5e11) == 0.0,
        },
        "coarse_grid": rows,
        "bisection_steps": args.bisection_steps,
        "budget": {
            "nr": args.nr,
            "nz": args.nz,
            "rays_per_beam": args.rays_per_beam,
            "t_end_s": args.t_end_s,
            "plot_every_steps": args.plot_every_steps,
            "history_every_steps": args.history_every_steps,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("characterize",), default="characterize")
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_i1_capsule.py")
    parser.add_argument("--out-root", default="./build/i1b_lite_2d_rz_runs")
    parser.add_argument("--summary-json", default="./build/i1b_lite_2d_rz_characterization_summary.json")
    parser.add_argument("--coarse-grid", type=parse_float_list, default=list(DEFAULT_COARSE_GRID))
    parser.add_argument("--bisection-steps", type=int, default=DEFAULT_BISECTION_STEPS)
    parser.add_argument("--nr", type=int, default=DEFAULT_NR)
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    parser.add_argument("--rays-per-beam", type=int, default=DEFAULT_RAYS_PER_BEAM)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--t-end-s", type=float, default=DEFAULT_T_END_S)
    parser.add_argument("--dt-initial-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--dt-max-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--plot-every-steps", type=int, default=1)
    parser.add_argument("--history-every-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without invoking TENRYU")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.bisection_steps < 0:
        raise ValueError("--bisection-steps must be non-negative")
    if args.nr <= 0 or args.nz <= 0:
        raise ValueError("--nr and --nz must be positive")
    if args.rays_per_beam <= 0:
        raise ValueError("--rays-per-beam must be positive")
    if args.t_end_s <= 0.0 or args.dt_initial_s <= 0.0 or args.dt_max_s <= 0.0:
        raise ValueError("time-step arguments must be positive")
    if args.plot_every_steps <= 0 or args.history_every_steps <= 0:
        raise ValueError("--plot-every-steps and --history-every-steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.dry_run:
        print(json.dumps(dry_run_summary(args), indent=2, sort_keys=True))
        return 0
    summary = characterize(args)
    print(json.dumps(json_sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary.get("characterization_status") == "bracketed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
