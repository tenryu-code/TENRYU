"""Deterministic one-dimensional run digest generation."""

import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Optional

from tools.assist.journal import sha256_hex


_REDUCTIONS = {
    "energy/E_total": "energy_total",
    "energy/conservation_error": "conservation_error",
    "energy/laser_deposited": "final",
    "energy/laser_incident": "final",
    "energy/numerical_loss": "final",
    "energy/safety_injected": "final",
    "energy/floor_injected": "final",
    "energy/internal_electron": "final",
    "energy/internal_ion": "final",
    "energy/kinetic": "final",
    "energy/radiation_field": "final",
    "laser/absorbed_total": "final",
    "laser/commanded_power_total": "final",
    "safety/clamp_count": "final",
    "implosion/rho_peak": "maximum",
    "implosion/rho_R": "maximum",
    "implosion/center_temperature": "maximum",
    "implosion/shell_radius_min": "minimum",
    "laser/absorption_efficiency_total": "final_max",
    "dt": "dt",
}

_TRACK_DATASETS = (
    "energy/conservation_error",
    "implosion/rho_peak",
    "implosion/rho_R",
    "implosion/shell_radius_min",
    "laser/absorption_efficiency_total",
    "dt",
)


def _read_json_dict(path: Path, notes: list, label: str) -> Optional[dict]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        notes.append("{0} could not be read: {1}".format(label, error))
        return None
    if not isinstance(value, dict):
        notes.append("{0} is not a JSON object".format(label))
        return None
    return value


def _nested_value(values: dict, section: str, key: str):
    nested = values.get(section)
    if not isinstance(nested, dict):
        return None
    return nested.get(key)


def _number(value, conversion):
    if value is None:
        return None
    try:
        return conversion(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _plain_float(value) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _sample_indices(length: int, limit: int) -> list:
    if length <= 0 or limit <= 0:
        return []
    if length <= limit:
        return list(range(length))
    if limit == 1:
        return [0]
    stride = int(math.ceil(float(length - 1) / float(limit - 1)))
    indices = list(range(0, length, stride))
    if indices[-1] != length - 1:
        indices.append(length - 1)
    return indices


def _values(dataset, name: str, notes: list, numpy_module):
    raw = dataset[...]
    shape = tuple(raw.shape)
    ndim = len(shape)
    if ndim == 2:
        if shape[1] == 0:
            note = "{0}: unsupported dataset shape {1} — skipped".format(
                name, shape
            )
            if note not in notes:
                notes.append(note)
            return None
        note = "{0}: multi-column dataset (shape {1}) — using column 0".format(
            name, shape
        )
        if note not in notes:
            notes.append(note)
        if numpy_module is not None:
            return numpy_module.asarray(raw[:, 0], dtype=float)
        return [float(row[0]) for row in raw]
    if ndim != 1:
        note = "{0}: unsupported dataset shape {1} — skipped".format(
            name, shape
        )
        if note not in notes:
            notes.append(note)
        return None
    if numpy_module is not None:
        return numpy_module.asarray(raw, dtype=float).reshape(-1)
    return [float(value) for value in list(raw)]


def _length(values) -> int:
    return int(len(values))


def _value(values, index: int) -> Optional[float]:
    return _plain_float(values[index])


def _argmax(values, numpy_module) -> int:
    if numpy_module is not None:
        return int(numpy_module.argmax(values))
    return max(range(len(values)), key=lambda index: values[index])


def _argmin(values, numpy_module) -> int:
    if numpy_module is not None:
        return int(numpy_module.argmin(values))
    return min(range(len(values)), key=lambda index: values[index])


def _reduce(values, kind: str, time_values, numpy_module) -> dict:
    if _length(values) == 0:
        if kind == "energy_total":
            return {"first": None, "final": None, "drift_rel": None}
        if kind == "conservation_error":
            return {"max_abs": None, "final": None}
        if kind == "final":
            return {"final": None}
        if kind == "maximum":
            return {"max": None, "t_at_max_s": None}
        if kind == "minimum":
            return {"min": None, "t_at_min_s": None}
        if kind == "final_max":
            return {"final": None, "max": None}
        return {"min": None, "median": None, "final": None}

    first = _value(values, 0)
    final = _value(values, -1)
    if kind == "energy_total":
        drift = None
        if first is not None and final is not None:
            drift = (final - first) / max(abs(first), 1.0e-300)
        return {"first": first, "final": final, "drift_rel": drift}
    if kind == "conservation_error":
        if numpy_module is not None:
            maximum = _plain_float(numpy_module.max(numpy_module.abs(values)))
        else:
            maximum = _plain_float(max(abs(value) for value in values))
        return {"max_abs": maximum, "final": final}
    if kind == "final":
        return {"final": final}
    if kind in ("maximum", "minimum"):
        if kind == "maximum":
            index = _argmax(values, numpy_module)
            result = {"max": _value(values, index), "t_at_max_s": None}
            time_key = "t_at_max_s"
        else:
            index = _argmin(values, numpy_module)
            result = {"min": _value(values, index), "t_at_min_s": None}
            time_key = "t_at_min_s"
        if time_values is not None and index < _length(time_values):
            result[time_key] = _value(time_values, index)
        return result
    if kind == "final_max":
        if numpy_module is not None:
            maximum = _plain_float(numpy_module.max(values))
        else:
            maximum = _plain_float(max(values))
        return {"final": final, "max": maximum}

    if numpy_module is not None:
        minimum = _plain_float(numpy_module.min(values))
        median = _plain_float(numpy_module.median(values))
    else:
        minimum = _plain_float(min(values))
        median = _plain_float(statistics.median(values))
    return {"min": minimum, "median": median, "final": final}


def build_digest(output_dir: str, series_points: int = 33) -> dict:
    """Build a JSON-safe digest from an output directory."""
    root = Path(output_dir).resolve()
    notes = []

    run_info_path = root / "run_info.json"
    if run_info_path.is_file():
        run_info = _read_json_dict(run_info_path, notes, "run_info.json")
    else:
        run_info = None
        notes.append("run_info.json not found")

    if run_info is None:
        run = {
            "terminated_normally": None,
            "steps": None,
            "t_final_s": None,
            "dt_final_s": None,
        }
    else:
        termination = run_info.get("termination_reason")
        run = {
            "terminated_normally": (
                termination == "t_end_reached" if termination is not None else None
            ),
            "steps": _number(run_info.get("step"), int),
            "t_final_s": _number(run_info.get("t"), float),
            "dt_final_s": _number(run_info.get("dt"), float),
        }

    frozen_paths = sorted((root / "config").glob("*_frozen.json"))
    frozen_config = None
    if not frozen_paths:
        notes.append("frozen config not found")
    else:
        if len(frozen_paths) > 1:
            notes.append(
                "multiple frozen configs found; using lexicographically first: {0}".format(
                    frozen_paths[0]
                )
            )
        frozen_values = _read_json_dict(
            frozen_paths[0], notes, "frozen config"
        )
        if frozen_values is not None:
            try:
                frozen_hash = sha256_hex(frozen_paths[0].read_bytes())
            except OSError as error:
                notes.append("frozen config hash unavailable: {0}".format(error))
                frozen_hash = None
            frozen_config = {
                "path": str(frozen_paths[0].resolve()),
                "sha256": frozen_hash,
                "schema_version": frozen_values.get("_schema_version"),
                "namelist_source_hash": frozen_values.get(
                    "_namelist_source_hash"
                ),
                "name": _nested_value(frozen_values, "main", "name"),
                "dimension": _nested_value(
                    frozen_values, "main", "dimension"
                ),
                "t_end_s": _nested_value(frozen_values, "main", "t_end"),
                "nr": _nested_value(frozen_values, "mesh", "nr"),
                "r_max_cm": _nested_value(frozen_values, "mesh", "r_max"),
            }

    history_paths = sorted((root / "results").glob("*_history.h5"))
    history = {
        "path": None,
        "n_samples": None,
        "series": {},
        "tracks": {},
        "missing": [],
    }
    derived = {
        "bang_time_proxy_s": None,
        "laser_deposition_centroid_inside_critical_fraction": None,
        "dt_collapse_ratio": None,
    }
    if not history_paths:
        notes.append("history file not found")
        return {
            "schema": "tenryu.assist.digest.v0",
            "output_dir": str(root),
            "run_info": run_info,
            "run": run,
            "frozen_config": frozen_config,
            "history": history,
            "derived": derived,
            "notes": notes,
        }

    history_path = history_paths[0]
    history["path"] = str(history_path.resolve())
    if len(history_paths) > 1:
        notes.append(
            "multiple history files found; using lexicographically first: {0}".format(
                history_path
            )
        )

    try:
        import h5py
    except ImportError:
        notes.append("h5py unavailable — history reductions skipped")
        return {
            "schema": "tenryu.assist.digest.v0",
            "output_dir": str(root),
            "run_info": run_info,
            "run": run,
            "frozen_config": frozen_config,
            "history": history,
            "derived": derived,
            "notes": notes,
        }

    try:
        import numpy as np
    except ImportError:
        np = None

    try:
        with h5py.File(str(history_path), "r") as handle:
            time_values = (
                _values(handle["t"], "t", notes, np) if "t" in handle else None
            )
            if time_values is None:
                if "t" in handle:
                    history["missing"].append("t")
                notes.append("history dataset 't' missing — tracks use sample indices")
            else:
                history["n_samples"] = _length(time_values)

            cached = {}
            for name, kind in _REDUCTIONS.items():
                if name not in handle:
                    history["missing"].append(name)
                    continue
                values = _values(handle[name], name, notes, np)
                if values is None:
                    history["missing"].append(name)
                    continue
                cached[name] = values
                history["series"][name] = _reduce(
                    values, kind, time_values, np
                )
                if name == "dt" and _length(values) >= 3:
                    clipped_stats = _reduce(
                        values[:-1], kind, time_values, np
                    )
                    history["series"][name]["min"] = clipped_stats["min"]
                    history["series"][name]["median"] = clipped_stats["median"]
                    notes.append(
                        "dt min/median exclude the final sample (t_end step clip)"
                    )
                if history["n_samples"] is None:
                    history["n_samples"] = _length(values)

            for name in _TRACK_DATASETS:
                if name not in cached:
                    continue
                values = cached[name]
                indices = _sample_indices(_length(values), series_points)
                if time_values is None:
                    track_times = indices
                else:
                    track_times = [_value(time_values, index) for index in indices]
                history["tracks"][name] = {
                    "t_s": track_times,
                    "value": [_value(values, index) for index in indices],
                }

            shell_stats = history["series"].get(
                "implosion/shell_radius_min"
            )
            if shell_stats is not None:
                derived["bang_time_proxy_s"] = shell_stats["t_at_min_s"]

            # This fraction indicates deposition centroids radially inside critical surfaces.
            weighted_name = "laser/absorption_weighted_r"
            critical_name = "laser/critical_surface_r"
            if weighted_name in handle and critical_name in handle:
                weighted = _values(handle[weighted_name], weighted_name, notes, np)
                critical = _values(handle[critical_name], critical_name, notes, np)
                if weighted is None:
                    history["missing"].append(weighted_name)
                if critical is None:
                    history["missing"].append(critical_name)
                if weighted is not None and critical is not None:
                    valid = []
                    for weighted_value, critical_value in zip(weighted, critical):
                        w_value = float(weighted_value)
                        c_value = float(critical_value)
                        if (
                            math.isfinite(w_value)
                            and math.isfinite(c_value)
                            and w_value > 0.0
                            and c_value > 0.0
                        ):
                            valid.append(w_value < c_value)
                    if valid:
                        derived[
                            "laser_deposition_centroid_inside_critical_fraction"
                        ] = sum(valid) / float(len(valid))

            dt_stats = history["series"].get("dt")
            if (
                dt_stats is not None
                and dt_stats["min"] is not None
                and dt_stats["median"] not in (None, 0.0)
            ):
                derived["dt_collapse_ratio"] = (
                    dt_stats["min"] / dt_stats["median"]
                )
    except (OSError, ValueError, TypeError) as error:
        history["n_samples"] = None
        history["series"] = {}
        history["tracks"] = {}
        history["missing"] = []
        derived = {
            "bang_time_proxy_s": None,
            "laser_deposition_centroid_inside_critical_fraction": None,
            "dt_collapse_ratio": None,
        }
        notes.append("history could not be read: {0}".format(error))

    return {
        "schema": "tenryu.assist.digest.v0",
        "output_dir": str(root),
        "run_info": run_info,
        "run": run,
        "frozen_config": frozen_config,
        "history": history,
        "derived": derived,
        "notes": notes,
    }


def main_digest(args) -> int:
    """CLI entry point for the digest verb."""
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(
            "assist digest: output directory does not exist or is not a directory: "
            "{0}".format(output_dir),
            file=sys.stderr,
        )
        return 2

    digest = build_digest(output_dir, series_points=args.series_points)
    rendered = json.dumps(digest, indent=2, sort_keys=True) + "\n"
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as stream:
                stream.write(rendered)
        except OSError as error:
            print("assist digest: cannot write output: {0}".format(error), file=sys.stderr)
            return 2
    else:
        print(rendered, end="")
    return 0
