#!/usr/bin/env python3
"""Run the finite-CR shell-on-hotspot I1-B discriminator ladder.

This harness is measurement-only.  It runs Case A through the 1D spherical
reference deck and Cases B/C/D through the controlled 2D deck, collects
CR50(t), eta_contact(t), center-patch sigma, termination, mesh-quality, and
energy diagnostics, then writes consolidated JSON and text artifacts with a
routing verdict.  The verdict selects a next implementation branch; it is not
a physics pass/fail assertion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_i1b_cap_compression_c72 as cap
import run_i1b_lite_2d_rz as lite


CASE_NAME = "2d_rz_i1b_finite_cr_shell_hotspot"
DEFAULT_DECK = Path("examples/verification/2d_rz_i1b_finite_cr_shell_hotspot.py")
DEFAULT_1D_DECK = Path("examples/verification/i1b_finite_cr_shell_1d_sph_ref.py")
DEFAULT_OUTROOT = Path("build/i1b_finite_cr_shell_discriminator")
ENV_PREFIX = "TENRYU_I1B_DISC_"
ENV_PREFIX_1D = "TENRYU_I1B_1D_"
DIFFREF_DIAG_ENV = "TENRYU_I1B_DIFFREF_DIAG"
VALID_CASES = ("A", "B", "C", "D")
SMOKE_CASES = ("A", "B")
SMOKE_NZ = (64, 128)
FULL_CASES = VALID_CASES
FULL_NZ = (64, 128, 256)
SMOKE_MAX_STEPS = 400
FULL_MAX_STEPS = 20000
NON_POSITIVE_VOLUME_TOKEN = "non-positive 2D RZ cell"
AFFINE_SELF_TEST_MAX = 1.0e-10
HEALTHY_MIN_CORNER_J_REL = 1.0e-8
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
DECK_PREFIX = f"[deck:{CASE_NAME}]"
DECK_PREFIX_1D = "[deck:i1b_finite_cr_shell_1d_sph_ref]"
LOG_KV_NUMBER_RE = re.compile(rf"([A-Za-z0-9_./:-]+)=({NUMBER_RE})")
NON_POSITIVE_RE = re.compile(
    rf"Encountered non-positive 2D RZ cell (?P<kind>area|volume) "
    rf"at cell=(?P<cell>-?\d+), (?P<value_name>area|vol)=(?P<value>{NUMBER_RE})"
)
OUTPUT_DIR_RE = re.compile(r"^\[TENRYU\] Output directory:\s*(\S+)\s*$", re.MULTILINE)
STEP_RE = re.compile(rf"(?:^|[,\s])(?:step|cycle)=(?P<step>-?\d+)(?:$|[,\s])")
TIME_RE = re.compile(rf"(?:^|[,\s])t=(?P<t>{NUMBER_RE})(?:$|[,\s])")
Q_KEYS = {
    "min_corner_j",
    "mesh_quality_min",
    "mesh_quality_min_corner_j",
    "achieved_min_corner_j_rel",
    "core_quality_min_corner_j",
}
CASE_LABELS = {
    "A": "1D spherical pure-Lagrangian reference",
    "B": "2D pure-Lagrangian",
    "C": "2D full ALE differential reference",
    "D": "2D Lagrangian-bulk center-patch ALE",
}
CASE_A_COMMON_ENV_KEYS = (
    "SEED",
    "S_MAX_CM",
    "R_G_CM",
    "R_G_FRAC",
    "R_S_CM",
    "R_S_FRAC",
    "R_OUTER_CM",
    "R_OUTER_FRAC",
    "R_HS_CM",
    "R_HS_FRAC",
    "RHO_GAS_FILL_GCC",
    "RHO_SHELL_GCC",
    "RHO_ABLATOR_GCC",
    "RHO_EXTERIOR_GCC",
    "T_GAS_BALANCED_EV",
    "P0_DYN_CM2",
    "GASP_SCALE",
    "ETA",
    "ABLATOR_ENABLED",
    "SMOOTH_WIDTH_CM",
    "T_END_S",
    "DT_INITIAL_S",
    "DT_MAX_S",
    "CFL_HYDRO",
    "RETRY",
    "RETRY_MAX_ATTEMPTS",
    "PLOT_EVERY_STEPS",
    "HISTORY_EVERY_STEPS",
    "D_T_RAMP_UP_S",
    "T_RAMP_UP_S",
    "D_T_HOLD_END_S",
    "T_HOLD_END_S",
    "D_T_RAMP_DOWN_S",
    "T_RAMP_DOWN_S",
    "D_P_PEAK_DYN_CM2",
    "P_PEAK_DYN_CM2",
)
HOTSPOT_GROUP = "/diagnostics/hotspot_gas/v1/"
CR50_TARGET = 2.6
CR50_STALL_BASELINE = 1.14
CR50_IMPROVEMENT_ABS = 0.05
CR50_IMPROVEMENT_REL = 0.03
CR50_CLOSE_REL = 0.10
ETA_CONTACT_ZERO_ABS = 1.0e-12
CENTER_PATCH_SIGMA_COLLAPSE = 0.5


@dataclass(frozen=True)
class CellGeometry:
    cell: int
    cell_id_stable: int | None
    centroid_r_cm: float | None
    centroid_z_cm: float | None
    radius_cm: float | None
    initial_centroid_r_cm: float | None
    initial_centroid_z_cm: float | None
    initial_radius_cm: float | None
    passive_region: str | None


def json_sanitize(value: Any) -> Any:
    """JSON sanitizer with numpy support, following the cap harness pattern."""
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return json_sanitize(value.tolist())
        if isinstance(value, np.generic):
            return json_sanitize(value.item())
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def parse_csv_cases(value: str | None, default: tuple[str, ...]) -> list[str]:
    if value is None:
        return list(default)
    cases = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not cases:
        raise argparse.ArgumentTypeError("expected at least one case")
    bad = [case for case in cases if case not in VALID_CASES]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid case(s): {','.join(bad)}")
    return list(dict.fromkeys(cases))


def parse_csv_ints(value: str | None, default: tuple[int, ...]) -> list[int]:
    if value is None:
        return list(default)
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            item = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {part!r}") from exc
        if item <= 0:
            raise argparse.ArgumentTypeError("nz values must be positive")
        if item % 4 != 0:
            raise argparse.ArgumentTypeError("nz values must satisfy NZ % 4 == 0")
        out.append(item)
    if not out:
        raise argparse.ArgumentTypeError("expected at least one nz value")
    return sorted(dict.fromkeys(out))


def parse_set_env(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--set-env expects NAME=VALUE, got {item!r}")
        name, value = item.split("=", 1)
        name = name.strip()
        if not name:
            raise argparse.ArgumentTypeError("--set-env name must be non-empty")
        if not name.startswith("TENRYU_"):
            name = ENV_PREFIX + name
        out[name] = value
    return out


def parse_deck_kv(log_text: str) -> dict[str, str]:
    for line in log_text.splitlines():
        if not (line.startswith(DECK_PREFIX) or line.startswith(DECK_PREFIX_1D)):
            continue
        return {
            match.group(1): match.group(2)
            for match in re.finditer(r"([A-Za-z0-9_]+)=([^ \t]+)", line)
        }
    return {}


def parse_actual_outdir(repo_root: Path, log_text: str, requested_outdir: Path) -> Path:
    matches = OUTPUT_DIR_RE.findall(log_text)
    if matches:
        return cap.resolve_under_repo(repo_root, matches[-1])
    kv = parse_deck_kv(log_text)
    if "outdir" in kv:
        return cap.resolve_under_repo(repo_root, kv["outdir"])
    return requested_outdir


def allocate_run_dir(outroot: Path, stem: str) -> tuple[str, Path]:
    outroot.mkdir(parents=True, exist_ok=True)
    for idx in range(1000):
        run_id = stem if idx == 0 else f"{stem}_{idx:03d}"
        path = outroot / run_id
        if not path.exists():
            return run_id, path
    raise RuntimeError(f"could not allocate unique run directory under {outroot}")


def h5_path_exists(handle: Any, path: str) -> bool:
    key = path[1:] if path.startswith("/") else path
    return key in handle


def h5_array_first_present(
    handle: Any,
    candidates: tuple[str, ...],
    *,
    dtype: Any = float,
) -> tuple[str | None, Any | None]:
    import numpy as np

    for path in candidates:
        key = path[1:] if path.startswith("/") else path
        if key in handle:
            return path, np.asarray(handle[key][()], dtype=dtype)
    return None, None


def h5_scalar(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    key = path[1:] if path.startswith("/") else path
    if key not in handle:
        return default
    arr = np.asarray(handle[key][()]).reshape(-1)
    return float(arr[-1]) if arr.size else default


def h5_hotspot_candidates(name: str) -> tuple[str, ...]:
    return (
        HOTSPOT_GROUP + name,
        "diagnostics/hotspot_gas/v1/" + name,
        name,
    )


def series_basic_summary(
    times: list[float],
    values: list[float | None],
) -> dict[str, Any]:
    usable = [
        (t, v)
        for t, v in zip(times, values)
        if finite_float(t) is not None and finite_float(v) is not None
    ]
    if not usable:
        return {
            "available": False,
            "row_count": len(values),
            "missing": ["finite_series_values"],
        }
    final_t, final_value = usable[-1]
    peak_t, peak_value = max(usable, key=lambda item: float(item[1]))
    return {
        "available": True,
        "row_count": len(usable),
        "initial_t_s": usable[0][0],
        "initial_value": usable[0][1],
        "final_t_s": final_t,
        "final_value": final_value,
        "peak_t_s": peak_t,
        "peak_value": peak_value,
        "missing": [],
    }


def read_hotspot_gas_history(history_path: Path | None) -> dict[str, Any]:
    if history_path is None:
        return {
            "available": False,
            "source": "history_hdf5",
            "source_path": None,
            "missing": ["history_hdf5"],
        }
    import h5py
    import numpy as np

    try:
        with h5py.File(history_path, "r") as handle:
            t_path, t_s = h5_array_first_present(
                handle,
                h5_hotspot_candidates("time_s") + h5_hotspot_candidates("t_s"),
                dtype=float,
            )
            step_path, step = h5_array_first_present(
                handle,
                h5_hotspot_candidates("step"),
                dtype=float,
            )
            cr50_path, cr50 = h5_array_first_present(
                handle,
                h5_hotspot_candidates("CR50")
                + h5_hotspot_candidates("compression_ratio_50")
                + h5_hotspot_candidates("cr50"),
                dtype=float,
            )
            rho50_path, rho50 = h5_array_first_present(
                handle,
                h5_hotspot_candidates("rho50_gcc"),
                dtype=float,
            )
            p50_path, p50 = h5_array_first_present(
                handle,
                h5_hotspot_candidates("p50_dyn_cm2"),
                dtype=float,
            )
            mass_drift_path, mass_drift = h5_array_first_present(
                handle,
                h5_hotspot_candidates("gas_mass_rel_drift"),
                dtype=float,
            )
    except OSError as exc:
        return {
            "available": False,
            "source": "history_hdf5",
            "source_path": str(history_path),
            "missing": [],
            "analysis_errors": [str(exc)],
        }

    missing = []
    if t_path is None:
        missing.append("diagnostics/hotspot_gas/v1/time_s")
    if step_path is None:
        missing.append("diagnostics/hotspot_gas/v1/step")
    if cr50_path is None:
        missing.append("diagnostics/hotspot_gas/v1/CR50")
    if missing:
        return {
            "available": False,
            "source": "history_hdf5",
            "source_path": str(history_path),
            "missing": missing,
            "paths": {
                "time_s": t_path,
                "step": step_path,
                "CR50": cr50_path,
                "rho50_gcc": rho50_path,
                "p50_dyn_cm2": p50_path,
                "gas_mass_rel_drift": mass_drift_path,
            },
        }

    t_arr = np.asarray(t_s, dtype=float).reshape(-1)
    step_arr = np.asarray(step, dtype=float).reshape(-1)
    cr50_arr = np.asarray(cr50, dtype=float).reshape(-1)
    n = min(t_arr.size, step_arr.size, cr50_arr.size)
    if n == 0:
        return {
            "available": False,
            "source": "history_hdf5",
            "source_path": str(history_path),
            "missing": ["empty_hotspot_gas_series"],
        }
    t_arr = t_arr[:n]
    step_arr = step_arr[:n]
    cr50_arr = cr50_arr[:n]

    def optional_series(raw: Any | None) -> list[float | None] | None:
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=float).reshape(-1)[:n]
        return [float(item) if math.isfinite(float(item)) else None for item in arr]

    rho50_series = optional_series(rho50)
    p50_series = optional_series(p50)
    mass_drift_series = optional_series(mass_drift)

    def ratio_series(series: list[float | None] | None) -> list[float | None] | None:
        if not series:
            return None
        base = finite_float(series[0])
        if base is None or abs(base) <= 0.0:
            return [None for _ in series]
        return [
            (float(item) / base if finite_float(item) is not None else None)
            for item in series
        ]

    times = [float(item) for item in t_arr]
    cr50_values = [
        float(item) if math.isfinite(float(item)) else None for item in cr50_arr
    ]
    rho50_ratio = ratio_series(rho50_series)
    p50_ratio = ratio_series(p50_series)
    return {
        "available": True,
        "source": "history_hdf5",
        "source_path": str(history_path),
        "paths": {
            "time_s": t_path,
            "step": step_path,
            "CR50": cr50_path,
            "rho50_gcc": rho50_path,
            "p50_dyn_cm2": p50_path,
            "gas_mass_rel_drift": mass_drift_path,
        },
        "row_count": int(n),
        "series": {
            "step": [int(item) for item in step_arr],
            "t_s": times,
            "CR50": cr50_values,
            "rho50_gcc": rho50_series,
            "rho50_ratio": rho50_ratio,
            "p50_dyn_cm2": p50_series,
            "p50_ratio": p50_ratio,
            "gas_mass_rel_drift": mass_drift_series,
        },
        "CR50": series_basic_summary(times, cr50_values),
        "rho50_ratio": series_basic_summary(times, rho50_ratio or []),
        "p50_ratio": series_basic_summary(times, p50_ratio or []),
        "missing": [],
        "note": "CR50 is the hotspot-gas mass-radius P50 compression from history_writer; rho50_ratio is the gas density-P50 ratio alias.",
    }


def parse_numeric_pairs(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, raw in LOG_KV_NUMBER_RE.findall(line):
        value = finite_float(raw)
        if value is not None:
            out[key] = value
    return out


def first_present_numeric(values: dict[str, float], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None


def parse_eta_contact_log(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    eta_contact_cumulative = 0.0
    for line in log_text.splitlines():
        if "eta_contact" not in line:
            continue
        values = parse_numeric_pairs(line)
        eta_contact_step = first_present_numeric(
            values, ("eta_contact_step", "eta_contact_increment")
        )
        eta_contact_logged_cumulative = first_present_numeric(
            values,
            (
                "eta_contact_cumulative",
                "eta_contact_cum",
                "eta_contact_total",
                "eta_contact",
            ),
        )
        if eta_contact_step is not None:
            eta_contact_cumulative += eta_contact_step
        eta_value = eta_contact_logged_cumulative
        if eta_value is None and eta_contact_step is not None:
            eta_value = eta_contact_cumulative
        if eta_value is None:
            continue
        eta_contact_cumulative = eta_value
        step, t_s = parse_step_t_from_line(line)
        records.append(
            {
                "step": step,
                "t_s": t_s,
                "eta_contact": eta_value,
                "eta_contact_cumulative": eta_value,
                "eta_contact_step": eta_contact_step,
                "contact_swept_volume_cm3": first_present_numeric(
                    values, ("contact_swept_volume_cm3", "contact_swept_volume")
                ),
                "hotspot_volume_cm3": first_present_numeric(
                    values, ("hotspot_volume_cm3", "hotspot_volume")
                ),
                "outside_p50": first_present_numeric(
                    values, ("outside_p50", "eta_contact_outside_p50")
                ),
                "outside_p95": first_present_numeric(
                    values, ("outside_p95", "eta_contact_outside_p95")
                ),
                "outside_max": first_present_numeric(
                    values, ("outside_max", "eta_contact_outside_max")
                ),
                "raw_values": values,
                "line": line,
            }
        )
    if not records:
        return {
            "available": False,
            "source": "log",
            "row_count": 0,
            "records": [],
            "summary": None,
            "missing": ["eta_contact_log_records"],
        }
    times = [
        float(item["t_s"]) if item["t_s"] is not None else float(idx)
        for idx, item in enumerate(records)
    ]
    values = [finite_float(item.get("eta_contact")) for item in records]
    eta_contact_step_values = [
        finite_float(item.get("eta_contact_step")) for item in records
    ]
    eta_contact_step_finite = [
        float(item) for item in eta_contact_step_values if item is not None
    ]
    contact_swept_volume_values = [
        finite_float(item.get("contact_swept_volume_cm3")) for item in records
    ]
    hotspot_volume_values = [
        finite_float(item.get("hotspot_volume_cm3")) for item in records
    ]
    summary = series_basic_summary(times, values)
    cumulative_sum = (
        sum(eta_contact_step_finite)
        if eta_contact_step_finite
        else final_summary_value(summary)
    )
    return {
        "available": True,
        "source": "log",
        "row_count": len(records),
        "records": records,
        "summary": summary,
        "eta_contact_cumulative_sum": cumulative_sum,
        "eta_contact_step_summary": series_basic_summary(
            times, eta_contact_step_values
        ),
        "contact_swept_volume_cm3": series_basic_summary(
            times, contact_swept_volume_values
        ),
        "hotspot_volume_cm3": series_basic_summary(times, hotspot_volume_values),
        "missing": [],
    }


def parse_center_patch_sigma_log(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        lowered = line.lower()
        if (
            "center-patch" not in lowered
            and "center_patch" not in lowered
            and "patch_min_corner_j" not in lowered
        ):
            continue
        values = parse_numeric_pairs(line)
        sigma = first_present_numeric(
            values,
            ("center_patch_sigma", "center_patch_sigma_accepted", "sigma_accepted"),
        )
        patch_min_corner_j = first_present_numeric(
            values,
            ("patch_min_corner_j", "center_patch_min_corner_j"),
        )
        if sigma is None and patch_min_corner_j is None:
            continue
        step, t_s = parse_step_t_from_line(line)
        records.append(
            {
                "step": step,
                "t_s": t_s,
                "center_patch_sigma": sigma,
                "patch_min_corner_j": patch_min_corner_j,
                "raw_values": values,
                "line": line,
            }
        )
    sigma_records = [
        item for item in records if finite_float(item.get("center_patch_sigma")) is not None
    ]
    if not sigma_records:
        return {
            "available": False,
            "source": "log",
            "row_count": len(records),
            "records": records,
            "summary": None,
            "missing": ["center_patch_sigma_records"],
        }
    sigma_values = [float(item["center_patch_sigma"]) for item in sigma_records]
    collapse_runs: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in sigma_records:
        if float(item["center_patch_sigma"]) < CENTER_PATCH_SIGMA_COLLAPSE:
            current.append(item)
        elif current:
            collapse_runs.append(current)
            current = []
    if current:
        collapse_runs.append(current)
    longest_collapse = max((len(item) for item in collapse_runs), default=0)
    return {
        "available": True,
        "source": "log",
        "row_count": len(sigma_records),
        "records": records,
        "threshold": CENTER_PATCH_SIGMA_COLLAPSE,
        "min_sigma": min(sigma_values),
        "final_sigma": sigma_values[-1],
        "consecutive_below_threshold_max": longest_collapse,
        "collapse_trigger_observed": longest_collapse >= 2,
        "missing": [],
    }


def parse_diff_ref_diag_log(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if "[diff_ref_diag]" not in line:
            continue
        values = parse_numeric_pairs(line)
        c_rho_bulk = first_present_numeric(values, ("C_rho_bulk", "rho50_ratio"))
        if c_rho_bulk is None:
            continue
        step, t_s = parse_step_t_from_line(line)
        records.append(
            {
                "step": step,
                "t_s": t_s,
                "C_rho_bulk": c_rho_bulk,
                "C_mesh_gas": first_present_numeric(values, ("C_mesh_gas",)),
                "raw_values": values,
                "line": line,
            }
        )
    if not records:
        return {
            "available": False,
            "source": "log",
            "row_count": 0,
            "records": [],
            "summary": None,
            "missing": ["diff_ref_diag_C_rho_bulk_records"],
        }
    times = [
        float(item["t_s"]) if item["t_s"] is not None else float(idx)
        for idx, item in enumerate(records)
    ]
    values = [finite_float(item.get("C_rho_bulk")) for item in records]
    return {
        "available": True,
        "source": "log",
        "row_count": len(records),
        "records": records,
        "summary": series_basic_summary(times, values),
        "missing": [],
    }


def read_all_run_logs(harness_log: Path, actual_outdir: Path) -> tuple[str, list[str]]:
    paths: list[Path] = [harness_log]
    out_log_dir = actual_outdir / "log"
    if out_log_dir.exists():
        paths.extend(sorted(out_log_dir.glob("*.log")))
    text_parts: list[str] = []
    used: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        used.append(str(path))
        text_parts.append(cap.read_text(path))
    return "\n".join(text_parts), used


def parse_step_t_from_line(line: str) -> tuple[int | None, float | None]:
    step_match = STEP_RE.search(line)
    time_match = TIME_RE.search(line)
    step = int(step_match.group("step")) if step_match else None
    t_s = finite_float(time_match.group("t")) if time_match else None
    return step, t_s


def parse_non_positive(log_text: str) -> dict[str, Any]:
    present = NON_POSITIVE_VOLUME_TOKEN.lower() in log_text.lower()
    events: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        if NON_POSITIVE_VOLUME_TOKEN.lower() not in line.lower():
            continue
        step, t_s = parse_step_t_from_line(line)
        match = NON_POSITIVE_RE.search(line)
        event: dict[str, Any] = {
            "line": line,
            "step": step,
            "t_s": t_s,
            "cell": None,
            "kind": None,
            "value": None,
            "value_name": None,
            "coords_from_log": None,
        }
        if match:
            event.update(
                {
                    "cell": int(match.group("cell")),
                    "kind": match.group("kind"),
                    "value_name": match.group("value_name"),
                    "value": finite_float(match.group("value")),
                }
            )
        coords: dict[str, float] = {}
        for key in ("centroid_r", "centroid_z", "r", "z", "x_r", "x_z"):
            coord_match = re.search(rf"(?:^|[,\s]){key}=({NUMBER_RE})(?:$|[,\s])", line)
            if coord_match:
                coords[key] = float(coord_match.group(1))
        if coords:
            event["coords_from_log"] = coords
        events.append(event)
    last = events[-1] if events else None
    return {
        "present": present,
        "token": NON_POSITIVE_VOLUME_TOKEN,
        "event_count": len(events),
        "events": events,
        "last_event": last,
        "failing_cell": last.get("cell") if last else None,
    }


def parse_quality_log(log_text: str) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    kv_re = re.compile(rf"([A-Za-z0-9_]+)=({NUMBER_RE})")
    for line in log_text.splitlines():
        if "[ale_provenance]" in line and "last_min_corner_j" in line:
            continue
        pairs = kv_re.findall(line)
        if not pairs:
            continue
        values: dict[str, float] = {}
        cells: dict[str, int] = {}
        for key, raw in pairs:
            if key.endswith("_cell"):
                try:
                    cells[key] = int(float(raw))
                except ValueError:
                    pass
                continue
            if key not in Q_KEYS:
                continue
            value = finite_float(raw)
            if value is None:
                continue
            values[key] = value
        for key, value in values.items():
            step, t_s = parse_step_t_from_line(line)
            cell = cells.get(f"{key}_cell")
            if cell is None and "cell" in cells:
                cell = cells["cell"]
            records.append(
                {
                    "key": key,
                    "value": value,
                    "step": step,
                    "t_s": t_s,
                    "cell": cell,
                    "line": line,
                }
            )
    usable = [item for item in records if abs(float(item["value"])) < 1.0e250]
    if not usable:
        return {
            "available": False,
            "source": "log",
            "row_count": 0,
            "records": [],
            "min_record": None,
            "missing": ["log_min_corner_j_or_mesh_quality_min"],
        }
    min_record = min(usable, key=lambda item: float(item["value"]))
    return {
        "available": True,
        "source": "log",
        "row_count": len(usable),
        "records": usable,
        "min_value": min_record["value"],
        "min_record": min_record,
        "missing": [],
    }


def read_mesh_quality_history_fallback(history_path: Path | None) -> dict[str, Any]:
    if history_path is None:
        return {
            "available": False,
            "source": "history_fallback",
            "source_path": None,
            "missing": ["history_hdf5"],
        }
    import h5py
    import numpy as np

    group = "diagnostics/mesh_quality_min/v1"
    try:
        with h5py.File(history_path, "r") as handle:
            required = {
                "step": f"{group}/step",
                "time_s": f"{group}/time_s",
                "achieved_min_corner_j_rel": f"{group}/achieved_min_corner_j_rel",
            }
            missing = [path for path in required.values() if not h5_path_exists(handle, path)]
            if missing:
                return {
                    "available": False,
                    "source": "history_fallback",
                    "source_path": str(history_path),
                    "missing": missing,
                }
            step = np.asarray(handle[required["step"]][()], dtype=int).reshape(-1)
            time_s = np.asarray(handle[required["time_s"]][()], dtype=float).reshape(-1)
            corner = np.asarray(
                handle[required["achieved_min_corner_j_rel"]][()], dtype=float
            ).reshape(-1)
    except OSError as exc:
        return {
            "available": False,
            "source": "history_fallback",
            "source_path": str(history_path),
            "missing": [],
            "analysis_errors": [str(exc)],
        }
    n = min(step.size, time_s.size, corner.size)
    if n == 0:
        return {
            "available": False,
            "source": "history_fallback",
            "source_path": str(history_path),
            "missing": ["empty_mesh_quality_series"],
        }
    step = step[:n]
    time_s = time_s[:n]
    corner = corner[:n]
    mask = np.isfinite(corner) & (np.abs(corner) < 1.0e250)
    if not np.any(mask):
        return {
            "available": False,
            "source": "history_fallback",
            "source_path": str(history_path),
            "row_count": int(n),
            "missing": ["finite_non_sentinel_achieved_min_corner_j_rel"],
        }
    finite_indices = np.nonzero(mask)[0]
    min_idx = int(finite_indices[np.argmin(corner[mask])])
    records = [
        {
            "key": "achieved_min_corner_j_rel",
            "value": float(corner[idx]),
            "step": int(step[idx]),
            "t_s": float(time_s[idx]),
            "cell": None,
        }
        for idx in finite_indices
    ]
    return {
        "available": True,
        "source": "history_fallback",
        "source_path": str(history_path),
        "row_count": len(records),
        "records": records,
        "min_value": float(corner[min_idx]),
        "min_record": records[int(np.argmin(corner[mask]))],
        "missing": [],
        "note": "Used only because no log min_corner_j/mesh_quality_min records were found.",
    }


def read_history_energy(history_path: Path | None) -> dict[str, Any]:
    if history_path is None:
        return {
            "available": False,
            "source_path": None,
            "missing": ["history_hdf5"],
        }
    import h5py
    import numpy as np

    try:
        with h5py.File(history_path, "r") as handle:
            e_path, e_total = h5_array_first_present(
                handle,
                ("time_state/E_total", "energy/E_total"),
                dtype=float,
            )
            t_path, t_s = h5_array_first_present(handle, ("time_state/t", "t"), dtype=float)
            step_path, step = h5_array_first_present(
                handle,
                ("time_state/step", "cycle", "diagnostics/conservation/v1/step"),
                dtype=float,
            )
            dt_path, dt = h5_array_first_present(handle, ("time_state/dt", "dt"), dtype=float)
    except OSError as exc:
        return {
            "available": False,
            "source_path": str(history_path),
            "missing": [],
            "analysis_errors": [str(exc)],
        }
    missing = []
    if e_path is None:
        missing.append("E_total")
    if t_path is None:
        missing.append("t")
    if step_path is None:
        missing.append("step")
    if missing:
        return {
            "available": False,
            "source_path": str(history_path),
            "missing": missing,
            "paths": {"E_total": e_path, "t": t_path, "step": step_path, "dt": dt_path},
        }
    e_arr = np.asarray(e_total, dtype=float).reshape(-1)
    t_arr = np.asarray(t_s, dtype=float).reshape(-1)
    step_arr = np.asarray(step, dtype=float).reshape(-1)
    n = min(e_arr.size, t_arr.size, step_arr.size)
    if n == 0:
        return {
            "available": False,
            "source_path": str(history_path),
            "missing": ["empty_energy_series"],
        }
    e_arr = e_arr[:n]
    t_arr = t_arr[:n]
    step_arr = step_arr[:n]
    e0 = float(e_arr[0])
    if not math.isfinite(e0) or abs(e0) <= 0.0:
        return {
            "available": False,
            "source_path": str(history_path),
            "missing": ["finite_nonzero_initial_E_total"],
            "E0_erg": e0 if math.isfinite(e0) else None,
        }
    drift = np.abs(e_arr - e0) / abs(e0)
    finite_mask = np.isfinite(drift)
    max_idx = int(np.argmax(np.where(finite_mask, drift, -np.inf)))
    dt_arr = np.asarray(dt, dtype=float).reshape(-1)[:n] if dt_path is not None else None
    return {
        "available": True,
        "source_path": str(history_path),
        "paths": {"E_total": e_path, "t": t_path, "step": step_path, "dt": dt_path},
        "row_count": int(n),
        "E0_erg": e0,
        "max_rel_drift": float(drift[max_idx]) if finite_mask[max_idx] else None,
        "max_rel_drift_step": int(step_arr[max_idx]),
        "max_rel_drift_t_s": float(t_arr[max_idx]),
        "final_rel_drift": float(drift[-1]) if math.isfinite(float(drift[-1])) else None,
        "series": {
            "step": [int(item) for item in step_arr],
            "t_s": [float(item) for item in t_arr],
            "E_total_erg": [float(item) for item in e_arr],
            "rel_drift": [float(item) if math.isfinite(float(item)) else None for item in drift],
            "dt_s": [float(item) for item in dt_arr] if dt_arr is not None else None,
        },
        "missing": [],
    }


def audit_summary_payload(actual_outdir: Path, energy: dict[str, Any]) -> dict[str, Any]:
    audit_path = actual_outdir / "audit_summary.json"
    if audit_path.exists():
        payload = cap.read_json(audit_path)
        return {
            "available": True,
            "path": str(audit_path),
            "audit_status": payload.get("audit_status"),
            "termination_reason": payload.get("termination_reason"),
            "conservation_residual_max": payload.get("conservation_residual_max"),
            "gcl_residual": payload.get("gcl_residual"),
            "positivity_min": payload.get("positivity_min"),
            "raw": payload,
        }
    return {
        "available": False,
        "path": str(audit_path),
        "absent": True,
        "fallback": "history_h5_E_total_drift",
        "fallback_available": bool(energy.get("available")),
        "note": "audit_summary.json was absent; no audit values were invented.",
    }


def unique_nodes(indices: Any) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for raw in indices:
        node = int(raw)
        if node in seen:
            continue
        seen.add(node)
        out.append(node)
    return out


def polygon_centroid(points: Any) -> tuple[float | None, float | None]:
    import numpy as np

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] != 2:
        return None, None
    finite = np.all(np.isfinite(pts), axis=1)
    pts = pts[finite]
    if pts.shape[0] == 0:
        return None, None
    if pts.shape[0] < 3:
        return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
    x = pts[:, 0]
    y = pts[:, 1]
    xp = np.roll(x, -1)
    yp = np.roll(y, -1)
    cross = x * yp - xp * y
    area2 = float(np.sum(cross))
    if not math.isfinite(area2) or abs(area2) < 1.0e-300:
        return float(np.mean(x)), float(np.mean(y))
    cx = float(np.sum((x + xp) * cross) / (3.0 * area2))
    cy = float(np.sum((y + yp) * cross) / (3.0 * area2))
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return float(np.mean(x)), float(np.mean(y))
    return cx, cy


def cell_nodes_for_cell(offsets: Any, indices: Any, cell: int) -> list[int]:
    if cell < 0 or cell + 1 >= len(offsets):
        return []
    begin = int(offsets[cell])
    end = int(offsets[cell + 1])
    return unique_nodes(indices[begin:end])


def passive_region_for_initial_radius(
    case: str | None,
    radius: float | None,
    thresholds: dict[str, float | None],
) -> str | None:
    if radius is None or not math.isfinite(radius):
        return None
    if case == "A":
        return "uniform"
    r_g = thresholds.get("R_g_cm")
    r_s = thresholds.get("R_s_cm")
    r_outer = thresholds.get("R_outer_cm")
    if r_g is None or r_s is None or r_outer is None:
        return None
    if radius <= r_g:
        return "gas_fill"
    if radius <= r_s:
        return "shell"
    if radius <= r_outer:
        return "ablator_density_layer"
    return "exterior"


def parse_region_thresholds(deck_kv: dict[str, str]) -> dict[str, float | None]:
    thresholds = {
        "R_hs_cm": finite_float(deck_kv.get("r_hs_cm")),
        "R_g_cm": None,
        "R_s_cm": None,
        "R_outer_cm": None,
    }
    text = deck_kv.get("region_thresholds_cm", "")
    patterns = {
        "R_g_cm": r"R_g\(([^)]+)\)",
        "R_s_cm": r"R_s\(([^)]+)\)",
        "R_outer_cm": r"R_outer\(([^)]+)\)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            thresholds[key] = finite_float(match.group(1))
    return thresholds


def load_snapshot_geometry(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        required = (
            "mesh/x_r",
            "mesh/x_z",
            "mesh/v_r",
            "mesh/v_z",
            "mesh/topology/v3/cell_node_csr_offsets",
            "mesh/topology/v3/cell_node_csr_indices",
        )
        missing = [item for item in required if not h5_path_exists(handle, item)]
        if missing:
            raise KeyError(f"{path}: missing {', '.join(missing)}")
        node_mass_path, node_mass = h5_array_first_present(
            handle,
            ("hydro/node_mass", "mesh/node_mass", "node_mass"),
            dtype=float,
        )
        block_id_path, block_id = h5_array_first_present(
            handle, ("mesh/topology/v3/cell_block_id",), dtype=int
        )
        block_role_path, block_role = h5_array_first_present(
            handle, ("mesh/topology/v3/block_role",), dtype=int
        )
        stable_path, stable = h5_array_first_present(
            handle, ("mesh/topology/v3/cell_id_stable",), dtype=int
        )
        orient_path, orient = h5_array_first_present(
            handle, ("mesh/topology/v3/cell_orientation_sign",), dtype=int
        )
        return {
            "path": str(path),
            "step": int(h5_scalar(handle, "time_state/step", 0.0)),
            "t_s": float(h5_scalar(handle, "time_state/t", 0.0)),
            "x_r": np.asarray(handle["mesh/x_r"][()], dtype=float).reshape(-1),
            "x_z": np.asarray(handle["mesh/x_z"][()], dtype=float).reshape(-1),
            "v_r": np.asarray(handle["mesh/v_r"][()], dtype=float).reshape(-1),
            "v_z": np.asarray(handle["mesh/v_z"][()], dtype=float).reshape(-1),
            "offsets": np.asarray(
                handle["mesh/topology/v3/cell_node_csr_offsets"][()], dtype=int
            ).reshape(-1),
            "indices": np.asarray(
                handle["mesh/topology/v3/cell_node_csr_indices"][()], dtype=int
            ).reshape(-1),
            "node_mass_path": node_mass_path,
            "node_mass": np.asarray(node_mass, dtype=float).reshape(-1)
            if node_mass is not None
            else None,
            "cell_block_id": np.asarray(block_id, dtype=int).reshape(-1)
            if block_id is not None
            else None,
            "block_role": np.asarray(block_role, dtype=int).reshape(-1)
            if block_role is not None
            else None,
            "cell_id_stable": np.asarray(stable, dtype=int).reshape(-1)
            if stable is not None
            else None,
            "cell_orientation_sign": np.asarray(orient, dtype=int).reshape(-1)
            if orient is not None
            else None,
            "paths": {
                "node_mass": node_mass_path,
                "cell_block_id": block_id_path,
                "block_role": block_role_path,
                "cell_id_stable": stable_path,
                "cell_orientation_sign": orient_path,
            },
        }


def stable_cell_id(snapshot: dict[str, Any], cell: int) -> int | None:
    stable = snapshot.get("cell_id_stable")
    if stable is None or cell < 0 or cell >= len(stable):
        return None
    return int(stable[cell])


def cell_geometry_record(
    snapshot: dict[str, Any],
    initial_snapshot: dict[str, Any],
    cell: int,
    case: str | None,
    thresholds: dict[str, float | None],
) -> CellGeometry:
    nodes_now = cell_nodes_for_cell(snapshot["offsets"], snapshot["indices"], cell)
    nodes_initial = cell_nodes_for_cell(
        initial_snapshot["offsets"], initial_snapshot["indices"], cell
    )
    centroid_r = centroid_z = radius = None
    initial_r = initial_z = initial_radius = None
    if nodes_now:
        points = [
            (float(snapshot["x_r"][node]), float(snapshot["x_z"][node]))
            for node in nodes_now
            if 0 <= node < len(snapshot["x_r"])
        ]
        centroid_r, centroid_z = polygon_centroid(points)
        if centroid_r is not None and centroid_z is not None:
            radius = math.hypot(centroid_r, centroid_z)
    if nodes_initial:
        points0 = [
            (float(initial_snapshot["x_r"][node]), float(initial_snapshot["x_z"][node]))
            for node in nodes_initial
            if 0 <= node < len(initial_snapshot["x_r"])
        ]
        initial_r, initial_z = polygon_centroid(points0)
        if initial_r is not None and initial_z is not None:
            initial_radius = math.hypot(initial_r, initial_z)
    return CellGeometry(
        cell=cell,
        cell_id_stable=stable_cell_id(snapshot, cell),
        centroid_r_cm=centroid_r,
        centroid_z_cm=centroid_z,
        radius_cm=radius,
        initial_centroid_r_cm=initial_r,
        initial_centroid_z_cm=initial_z,
        initial_radius_cm=initial_radius,
        passive_region=passive_region_for_initial_radius(case, initial_radius, thresholds),
    )


def cell_geometry_to_json(geom: CellGeometry, r_hs_cm: float | None) -> dict[str, Any]:
    ratio = None
    if geom.radius_cm is not None and r_hs_cm is not None and r_hs_cm > 0.0:
        ratio = geom.radius_cm / r_hs_cm
    return {
        "cell": geom.cell,
        "cell_id_stable": geom.cell_id_stable,
        "centroid_r_cm": geom.centroid_r_cm,
        "centroid_z_cm": geom.centroid_z_cm,
        "radius_cm": geom.radius_cm,
        "R_over_R_hs_target": ratio,
        "initial_centroid_r_cm": geom.initial_centroid_r_cm,
        "initial_centroid_z_cm": geom.initial_centroid_z_cm,
        "initial_radius_cm": geom.initial_radius_cm,
        "passive_region_from_initial_radius": geom.passive_region,
    }


def cell_corner_metric(snapshot: dict[str, Any], cell: int) -> float | None:
    import numpy as np

    nodes = cell_nodes_for_cell(snapshot["offsets"], snapshot["indices"], cell)
    if len(nodes) < 3:
        return None
    pts = np.asarray(
        [(snapshot["x_r"][node], snapshot["x_z"][node]) for node in nodes],
        dtype=float,
    )
    if pts.shape[0] < 3 or not np.all(np.isfinite(pts)):
        return None
    orient = 1
    signs = snapshot.get("cell_orientation_sign")
    if signs is not None and 0 <= cell < len(signs):
        orient = 1 if int(signs[cell]) >= 0 else -1
    values: list[float] = []
    n = pts.shape[0]
    for idx in range(n):
        prev_pt = pts[(idx - 1) % n]
        cur = pts[idx]
        next_pt = pts[(idx + 1) % n]
        a = next_pt - cur
        b = prev_pt - cur
        cross = float(a[0] * b[1] - a[1] * b[0])
        values.append(orient * cross)
    return min(values) if values else None


def find_worst_cell(snapshot: dict[str, Any]) -> dict[str, Any]:
    values: list[tuple[float, int]] = []
    n_cells = len(snapshot["offsets"]) - 1
    tri_count = 0
    for cell in range(n_cells):
        begin = int(snapshot["offsets"][cell])
        end = int(snapshot["offsets"][cell + 1])
        raw_nodes = snapshot["indices"][begin:end]
        if len(set(int(item) for item in raw_nodes)) < len(raw_nodes):
            tri_count += 1
        metric = cell_corner_metric(snapshot, cell)
        if metric is None or not math.isfinite(metric):
            continue
        values.append((metric, cell))
    if not values:
        return {
            "available": False,
            "missing": ["finite_cell_corner_metric"],
            "tri_cell_count": tri_count,
        }
    metric, cell = min(values, key=lambda item: item[0])
    return {
        "available": True,
        "cell": cell,
        "cell_id_stable": stable_cell_id(snapshot, cell),
        "corner_metric": metric,
        "tri_cell_count": tri_count,
        "evaluated_cell_count": len(values),
    }


def choose_snapshot_by_step(plot_files: list[Path], step: int | None) -> Path | None:
    if not plot_files:
        return None
    if step is None:
        return plot_files[-1]
    best_path = None
    best_delta = None
    for path in plot_files:
        try:
            snap = load_snapshot_geometry(path)
        except Exception:
            continue
        delta = abs(int(snap["step"]) - step)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_path = path
    return best_path or plot_files[-1]


def affine_residual_fractions(
    snapshot: dict[str, Any],
    *,
    velocity_override: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    x = np.column_stack((snapshot["x_r"], snapshot["x_z"]))
    if velocity_override is None:
        u = np.column_stack((snapshot["v_r"], snapshot["v_z"]))
    else:
        u = np.asarray(velocity_override, dtype=float)
    node_mass = snapshot.get("node_mass")
    offsets = snapshot["offsets"]
    indices = snapshot["indices"]
    n_cells = len(offsets) - 1
    fractions: list[float] = []
    cells: list[int] = []
    tri_count = 0
    invalid_weight_cell_count = 0
    singular_fit_count = 0
    for cell in range(n_cells):
        begin = int(offsets[cell])
        end = int(offsets[cell + 1])
        raw_nodes = [int(item) for item in indices[begin:end]]
        nodes = unique_nodes(raw_nodes)
        if len(nodes) < 4:
            tri_count += 1
            continue
        if len(nodes) != len(raw_nodes):
            tri_count += 1
            continue
        if any(node < 0 or node >= x.shape[0] for node in nodes):
            continue
        pts = x[nodes, :]
        vel = u[nodes, :]
        if not (np.all(np.isfinite(pts)) and np.all(np.isfinite(vel))):
            continue
        if node_mass is not None and len(node_mass) == x.shape[0]:
            w = np.asarray(node_mass[nodes], dtype=float)
            if not np.all(np.isfinite(w)) or float(np.sum(w)) <= 0.0:
                invalid_weight_cell_count += 1
                w = np.ones(len(nodes), dtype=float)
        else:
            w = np.ones(len(nodes), dtype=float)
        w_sum = float(np.sum(w))
        x_c = np.sum(pts * w[:, None], axis=0) / w_sum
        u_c = np.sum(vel * w[:, None], axis=0) / w_sum
        x_rel = pts - x_c
        u_rel = vel - u_c
        sqrt_w = np.sqrt(w)
        a = x_rel * sqrt_w[:, None]
        b = u_rel * sqrt_w[:, None]
        try:
            coef, _, rank, _ = np.linalg.lstsq(a, b, rcond=None)
        except np.linalg.LinAlgError:
            singular_fit_count += 1
            coef = np.linalg.pinv(a) @ b
            rank = -1
        if rank < 2:
            singular_fit_count += 1
        resid = u_rel - x_rel @ coef
        numerator = float(np.sum(w * np.sum(resid * resid, axis=1)))
        denominator = float(np.sum(w * np.sum(u_rel * u_rel, axis=1)))
        eps = max(1.0e-300, 1.0e-30 * denominator)
        frac = numerator / (denominator + eps)
        if math.isfinite(frac):
            fractions.append(frac)
            cells.append(cell)
    arr = np.asarray(fractions, dtype=float)
    if arr.size == 0:
        return {
            "available": False,
            "fractions": [],
            "cells": [],
            "tri_cell_count": tri_count,
            "nontri_cell_count": 0,
            "invalid_weight_cell_count": invalid_weight_cell_count,
            "singular_fit_count": singular_fit_count,
            "missing": ["finite_nontri_affine_residual_cells"],
        }
    max_idx = int(np.argmax(arr))
    return {
        "available": True,
        "fractions": [float(item) for item in arr],
        "cells": cells,
        "tri_cell_count": tri_count,
        "nontri_cell_count": int(arr.size),
        "invalid_weight_cell_count": invalid_weight_cell_count,
        "singular_fit_count": singular_fit_count,
        "max_fraction": float(arr[max_idx]),
        "mean_fraction": float(np.mean(arr)),
        "max_cell": int(cells[max_idx]),
        "max_cell_id_stable": stable_cell_id(snapshot, int(cells[max_idx])),
        "missing": [],
    }


def run_affine_self_test(snapshot: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    x = np.column_stack((snapshot["x_r"], snapshot["x_z"]))
    a = np.array([1.25e7, -3.5e7], dtype=float)
    scale = 8.0e8
    matrix = np.array([[-scale, 0.125 * scale], [-0.0625 * scale, -0.5 * scale]])
    u = a + x @ matrix.T
    result = affine_residual_fractions(snapshot, velocity_override=u)
    if not result.get("available"):
        return {
            "passed": False,
            "trustworthy": False,
            "threshold": AFFINE_SELF_TEST_MAX,
            "max_residual_fraction": None,
            "missing": result.get("missing", []),
            "detail": result,
        }
    max_fraction = float(result["max_fraction"])
    passed = max_fraction < AFFINE_SELF_TEST_MAX
    return {
        "passed": passed,
        "trustworthy": passed,
        "threshold": AFFINE_SELF_TEST_MAX,
        "max_residual_fraction": max_fraction,
        "nontri_cell_count": result.get("nontri_cell_count"),
        "tri_cell_count": result.get("tri_cell_count"),
        "max_cell": result.get("max_cell"),
        "mass_weighting": "hydro/node_mass" if snapshot.get("node_mass") is not None else "uniform",
    }


def analyze_affine_residual_trend(plot_files: list[Path]) -> dict[str, Any]:
    if not plot_files:
        return {
            "available": False,
            "missing": ["plot_snapshots"],
            "self_test": {"passed": False, "trustworthy": False, "missing": ["plot_snapshots"]},
        }
    snapshots: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in plot_files:
        try:
            snapshots.append(load_snapshot_geometry(path))
        except Exception as exc:  # noqa: BLE001 - recorded in diagnostic JSON.
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    if not snapshots:
        return {
            "available": False,
            "missing": ["loadable_plot_snapshots"],
            "analysis_errors": errors,
            "self_test": {"passed": False, "trustworthy": False, "missing": ["loadable_plot_snapshots"]},
        }
    self_test = run_affine_self_test(snapshots[0])
    records: list[dict[str, Any]] = []
    for snapshot in snapshots:
        result = affine_residual_fractions(snapshot)
        if not result.get("available"):
            records.append(
                {
                    "path": snapshot["path"],
                    "step": snapshot["step"],
                    "t_s": snapshot["t_s"],
                    "available": False,
                    "missing": result.get("missing", []),
                }
            )
            continue
        records.append(
            {
                "path": snapshot["path"],
                "step": snapshot["step"],
                "t_s": snapshot["t_s"],
                "available": True,
                "max_fraction": result["max_fraction"],
                "mean_fraction": result["mean_fraction"],
                "max_cell": result["max_cell"],
                "max_cell_id_stable": result["max_cell_id_stable"],
                "nontri_cell_count": result["nontri_cell_count"],
                "tri_cell_count": result["tri_cell_count"],
                "invalid_weight_cell_count": result["invalid_weight_cell_count"],
                "singular_fit_count": result["singular_fit_count"],
            }
        )
    usable = [record for record in records if record.get("available")]
    if not usable:
        return {
            "available": False,
            "records": records,
            "analysis_errors": errors,
            "self_test": self_test,
            "missing": ["finite_affine_residual_snapshots"],
        }
    first = usable[0]
    peak = max(usable, key=lambda item: float(item["max_fraction"]))
    last = usable[-1]
    first_value = float(first["max_fraction"])
    peak_value = float(peak["max_fraction"])
    last_value = float(last["max_fraction"])
    blind_intervals = [
        {
            "from_step": int(a["step"]),
            "to_step": int(b["step"]),
            "dt_s": float(b["t_s"]) - float(a["t_s"]),
            "dstep": int(b["step"]) - int(a["step"]),
        }
        for a, b in zip(usable[:-1], usable[1:])
    ]
    node_mass_path = snapshots[0].get("node_mass_path")
    mass_weighting = "node_mass"
    approximate = False
    if node_mass_path is None:
        mass_weighting = "uniform"
        approximate = True
    return {
        "available": True,
        "trustworthy": bool(self_test.get("passed")),
        "metric_status": "TRUSTWORTHY" if self_test.get("passed") else "UNTRUSTWORTHY",
        "self_test": self_test,
        "mass_weighting": mass_weighting,
        "mass_weighting_source": node_mass_path,
        "mass_weighting_approximate": approximate,
        "records": records,
        "trend": {
            "first_max_fraction": first_value,
            "last_max_fraction": last_value,
            "peak_max_fraction": peak_value,
            "peak_step": peak["step"],
            "peak_t_s": peak["t_s"],
            "peak_cell": peak["max_cell"],
            "growth_peak_over_first": peak_value / first_value
            if first_value > 0.0
            else (math.inf if peak_value > 0.0 else 1.0),
            "growth_last_over_first": last_value / first_value
            if first_value > 0.0
            else (math.inf if last_value > 0.0 else 1.0),
            "peak_gt_first": peak_value > first_value,
            "last_gt_first": last_value > first_value,
        },
        "blind_interval": {
            "source": "plot_snapshot_cadence",
            "intervals": blind_intervals,
            "max_dt_s": max((item["dt_s"] for item in blind_intervals), default=None),
            "max_dstep": max((item["dstep"] for item in blind_intervals), default=None),
        },
        "analysis_errors": errors,
        "missing": [],
    }


def q_record_at_or_before(q_records: list[dict[str, Any]], step: int) -> dict[str, Any] | None:
    with_steps = [
        record for record in q_records if isinstance(record.get("step"), int) and record["step"] <= step
    ]
    if not with_steps:
        return None
    return min(with_steps, key=lambda record: float(record["value"]))


def cross_check_velocity_first(
    velocity: dict[str, Any],
    quality: dict[str, Any],
) -> dict[str, Any]:
    if not velocity.get("available"):
        return {
            "available": False,
            "missing": ["velocity_residual_trend"],
        }
    if not quality.get("available") or quality.get("source") != "log":
        return {
            "available": False,
            "missing": ["log_min_corner_j_full_cadence"],
            "quality_source": quality.get("source"),
            "note": "Cross-check requires log-derived min_corner_j records.",
        }
    records = [item for item in velocity.get("records", []) if item.get("available")]
    if len(records) < 2:
        return {
            "available": False,
            "missing": ["at_least_two_velocity_snapshots"],
        }
    q_records = quality.get("records", [])
    first = float(records[0]["max_fraction"])
    candidates: list[dict[str, Any]] = []
    for record in records[1:]:
        value = float(record["max_fraction"])
        if value <= first:
            continue
        q_record = q_record_at_or_before(q_records, int(record["step"]))
        healthy = (
            q_record is not None
            and finite_float(q_record.get("value")) is not None
            and float(q_record["value"]) > HEALTHY_MIN_CORNER_J_REL
        )
        candidates.append(
            {
                "velocity_step": record["step"],
                "velocity_t_s": record["t_s"],
                "velocity_max_fraction": value,
                "q_min_record": q_record,
                "q_min_healthy_by_threshold": healthy,
            }
        )
    true_candidates = [item for item in candidates if item["q_min_healthy_by_threshold"]]
    return {
        "available": True,
        "healthy_min_corner_j_rel_threshold": HEALTHY_MIN_CORNER_J_REL,
        "residual_grew_while_min_corner_j_healthy": bool(true_candidates),
        "residual_growth_candidates": candidates,
        "matching_records": true_candidates,
        "note": "Descriptive heuristic only; not a physics pass/fail assertion.",
    }


def angle_span(theta: Any) -> float | None:
    import numpy as np

    arr = np.asarray(theta, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return None
    return float(np.max(arr) - np.min(arr))


def near_axis_aspect(snapshot: dict[str, Any], pole_sign: int) -> dict[str, Any]:
    import numpy as np

    r = snapshot["x_r"]
    z = snapshot["x_z"]
    max_r = float(np.max(np.abs(r))) if len(r) else 0.0
    axis_tol = max(1.0e-14, 1.0e-12 * max(1.0, max_r))
    candidates: list[dict[str, Any]] = []
    n_cells = len(snapshot["offsets"]) - 1
    for cell in range(n_cells):
        nodes = cell_nodes_for_cell(snapshot["offsets"], snapshot["indices"], cell)
        if len(nodes) < 3:
            continue
        rr = np.asarray([r[node] for node in nodes], dtype=float)
        zz = np.asarray([z[node] for node in nodes], dtype=float)
        if not (np.all(np.isfinite(rr)) and np.all(np.isfinite(zz))):
            continue
        if not np.any(np.abs(rr) <= axis_tol):
            continue
        centroid_r, centroid_z = polygon_centroid(np.column_stack((rr, zz)))
        if centroid_r is None or centroid_z is None:
            continue
        if pole_sign > 0 and centroid_z < 0.0:
            continue
        if pole_sign < 0 and centroid_z > 0.0:
            continue
        radius_nodes = np.sqrt(rr * rr + zz * zz)
        d_radius = float(np.max(radius_nodes) - np.min(radius_nodes))
        theta = np.arctan2(rr, zz)
        dtheta = angle_span(theta)
        radius_centroid = math.hypot(centroid_r, centroid_z)
        if dtheta is None or d_radius <= 0.0:
            continue
        aspect = radius_centroid * dtheta / d_radius
        if not math.isfinite(aspect):
            continue
        candidates.append(
            {
                "cell": cell,
                "cell_id_stable": stable_cell_id(snapshot, cell),
                "centroid_r_cm": centroid_r,
                "centroid_z_cm": centroid_z,
                "centroid_radius_cm": radius_centroid,
                "dtheta_rad": dtheta,
                "dR_cm": d_radius,
                "aspect_r_dtheta_over_dR": aspect,
            }
        )
    if not candidates:
        return {
            "available": False,
            "pole": "+z" if pole_sign > 0 else "-z",
            "missing": ["axis_adjacent_inner_ring_cells"],
        }
    candidates.sort(key=lambda item: float(item["centroid_radius_cm"]))
    ring = candidates[: min(8, len(candidates))]
    aspects = [float(item["aspect_r_dtheta_over_dR"]) for item in ring]
    return {
        "available": True,
        "pole": "+z" if pole_sign > 0 else "-z",
        "definition": "axis-adjacent cells sorted by centroid spherical radius; first up to 8 cells form the inner-ring sample",
        "sampled_cell_count": len(ring),
        "min": min(aspects),
        "mean": sum(aspects) / len(aspects),
        "max": max(aspects),
        "cells": ring,
        "missing": [],
    }


def near_axis_aspect_pair(initial: dict[str, Any], worst: dict[str, Any]) -> dict[str, Any]:
    return {
        "available": True,
        "definition": "spherical_radius_centroid * dtheta / dR from mesh coordinates and CSR topology",
        "t0": {
            "step": initial["step"],
            "t_s": initial["t_s"],
            "positive_z_pole": near_axis_aspect(initial, +1),
            "negative_z_pole": near_axis_aspect(initial, -1),
        },
        "worst_snapshot": {
            "path": worst["path"],
            "step": worst["step"],
            "t_s": worst["t_s"],
            "positive_z_pole": near_axis_aspect(worst, +1),
            "negative_z_pole": near_axis_aspect(worst, -1),
        },
    }


def frozen_config_summary(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {
            "available": False,
            "path": None,
            "missing": ["frozen_config"],
        }
    payload = cap.read_json(path)
    if not payload:
        return {
            "available": False,
            "path": str(path),
            "missing": ["parseable_frozen_config_json"],
        }
    numerics = payload.get("numerics", {}) if isinstance(payload.get("numerics"), dict) else {}
    return {
        "available": True,
        "path": str(path),
        "schema_version": payload.get("_schema_version"),
        "frozen_at": payload.get("_frozen_at"),
        "main": payload.get("main"),
        "mesh": payload.get("mesh"),
        "output": payload.get("output"),
        "hydro": numerics.get("hydro") if isinstance(numerics, dict) else None,
        "ale": numerics.get("ale") if isinstance(numerics, dict) else None,
        "diagnostics": numerics.get("diagnostics") if isinstance(numerics, dict) else None,
    }


def env_keys_for_log(env: dict[str, str]) -> dict[str, str]:
    return {
        key: env[key]
        for key in sorted(env)
        if key.startswith(ENV_PREFIX)
        or key.startswith(ENV_PREFIX_1D)
        or key == DIFFREF_DIAG_ENV
    }


def mirror_disc_env_to_1d(env: dict[str, str]) -> None:
    for key in CASE_A_COMMON_ENV_KEYS:
        src = ENV_PREFIX + key
        dst = ENV_PREFIX_1D + key
        if src in env and dst not in env:
            env[dst] = env[src]


def apply_case_environment(
    env: dict[str, str],
    *,
    case: str,
    nz: int,
    max_steps: int,
    outdir: Path,
) -> None:
    env.setdefault(ENV_PREFIX + "ETA", "0.5")
    env.setdefault(ENV_PREFIX + "GASP_SCALE", "0.1")
    env[ENV_PREFIX + "CASE"] = case
    env[ENV_PREFIX + "NZ"] = str(nz)
    env[ENV_PREFIX + "MAX_STEPS"] = str(max_steps)
    env[ENV_PREFIX + "OUTDIR"] = str(outdir)
    env[ENV_PREFIX + "HOTSPOT_GAS_TRACER"] = "1"
    env.setdefault(DIFFREF_DIAG_ENV, "1")

    if case == "A":
        env[ENV_PREFIX + "ALE"] = "0"
        env.setdefault(ENV_PREFIX + "DIFFREF", "0")
        env.setdefault(
            ENV_PREFIX + "MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED",
            "0",
        )
        mirror_disc_env_to_1d(env)
        env.setdefault(ENV_PREFIX_1D + "DRIVE", "vel")
        env.setdefault(ENV_PREFIX_1D + "NR", str(max(512, 4 * nz)))
        env[ENV_PREFIX_1D + "MAX_STEPS"] = str(max_steps)
        env[ENV_PREFIX_1D + "OUTDIR"] = str(outdir)
        return

    if case == "B":
        env[ENV_PREFIX + "ALE"] = "0"
        env[ENV_PREFIX + "DIFFREF"] = "0"
        env[ENV_PREFIX + "SCALED_REFERENCE"] = "0"
        env[ENV_PREFIX + "MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED"] = "0"
        return

    env[ENV_PREFIX + "ALE"] = "1"
    env[ENV_PREFIX + "CONSERVATIVE_REMAP_ENABLED"] = "1"
    env[ENV_PREFIX + "CONSERVATIVE_REMAP_TARGET"] = "reference"
    if case == "C":
        env[ENV_PREFIX + "DIFFREF"] = "1"
        env[ENV_PREFIX + "SCALED_REFERENCE"] = "0"
        env[ENV_PREFIX + "MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED"] = "0"
    elif case == "D":
        env[ENV_PREFIX + "DIFFREF"] = "0"
        env[ENV_PREFIX + "SCALED_REFERENCE"] = "0"
        env[ENV_PREFIX + "MULTIBLOCK_LAGRANGIAN_BULK_CENTER_PATCH_REFERENCE_ENABLED"] = "1"


def run_deck(
    *,
    repo_root: Path,
    tenryu_bin: str,
    deck_path: Path,
    outdir: Path,
    log_path: Path,
    case: str,
    nz: int,
    max_steps: int,
    wall_time_s: float,
    env_extra: dict[str, str],
) -> dict[str, Any]:
    cmd = [str(tenryu_bin), "run", str(deck_path)]
    env = os.environ.copy()
    env.update(env_extra)
    apply_case_environment(
        env,
        case=case,
        nz=nz,
        max_steps=max_steps,
        outdir=outdir,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    timed_out = False
    invocation_error = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[harness:i1b_finite_cr_shell_discriminator] command=" + " ".join(cmd) + "\n")
        log.write(
            "[harness:i1b_finite_cr_shell_discriminator] env="
            + json.dumps(
                env_keys_for_log(env),
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log,
                stderr=log,
                timeout=wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=cmd, returncode=-1)
        except OSError as exc:
            invocation_error = f"{type(exc).__name__}: {exc}"
            log.write(f"[harness:i1b_finite_cr_shell_discriminator] invocation_error={invocation_error}\n")
            proc = subprocess.CompletedProcess(args=cmd, returncode=-127)
    return {
        "command": cmd,
        "env": env_keys_for_log(env),
        "log_path": str(log_path),
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "invocation_error": invocation_error,
        "wall_time_s": time.monotonic() - start,
        "requested_outdir": str(outdir),
    }


def analyze_run(
    *,
    repo_root: Path,
    deck_path: Path,
    run_id: str,
    case: str,
    nz: int,
    mode: str,
    max_steps: int,
    requested_outdir: Path,
    harness_log: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    initial_log_text = cap.read_text(harness_log)
    actual_outdir = parse_actual_outdir(repo_root, initial_log_text, requested_outdir)
    log_text, log_paths = read_all_run_logs(harness_log, actual_outdir)
    actual_outdir = parse_actual_outdir(repo_root, log_text, requested_outdir)
    if str(actual_outdir / "log") not in "\n".join(log_paths):
        log_text, log_paths = read_all_run_logs(harness_log, actual_outdir)
    deck_kv = parse_deck_kv(log_text)
    thresholds = parse_region_thresholds(deck_kv)
    run_info_path = actual_outdir / "run_info.json"
    run_info = cap.read_json(run_info_path)
    frozen_path = cap.locate_frozen_config(actual_outdir)
    history_path: Path | None = None
    plot_files: list[Path] = []
    results_dir: Path | None = None
    discovery_errors: list[str] = []
    try:
        results_dir = lite.discover_results_dir(actual_outdir)
        plot_files = lite.locate_plot_files(results_dir)
        history_path = lite.locate_history_file(results_dir)
    except Exception as exc:  # noqa: BLE001 - preserve in JSON.
        discovery_errors.append(f"{type(exc).__name__}: {exc}")
    non_positive = parse_non_positive(log_text)
    q_log = parse_quality_log(log_text)
    q_metric = q_log if q_log.get("available") else read_mesh_quality_history_fallback(history_path)
    q_metric["log_parse"] = q_log
    hotspot_gas = read_hotspot_gas_history(history_path)
    eta_contact = parse_eta_contact_log(log_text)
    center_patch_sigma = parse_center_patch_sigma_log(log_text)
    diff_ref_diag = parse_diff_ref_diag_log(log_text)
    energy = read_history_energy(history_path)
    audit = audit_summary_payload(actual_outdir, energy)
    velocity = analyze_affine_residual_trend(plot_files)
    cross_check = cross_check_velocity_first(velocity, q_metric)

    geometry_metric: dict[str, Any] = {
        "available": False,
        "missing": ["plot_snapshots"],
    }
    aspect_metric: dict[str, Any] = {
        "available": False,
        "missing": ["plot_snapshots"],
    }
    if plot_files:
        try:
            initial_snapshot = load_snapshot_geometry(plot_files[0])
            failure_step = None
            if non_positive.get("last_event"):
                failure_step = non_positive["last_event"].get("step")
            if failure_step is None and isinstance(run_info.get("step"), int):
                failure_step = int(run_info["step"])
            q_min_record = q_metric.get("min_record") if q_metric.get("available") else None
            if q_min_record and isinstance(q_min_record.get("step"), int):
                failure_step = int(q_min_record["step"])
            worst_path = choose_snapshot_by_step(plot_files, failure_step)
            worst_snapshot = load_snapshot_geometry(worst_path) if worst_path else initial_snapshot
            selected_cell = non_positive.get("failing_cell")
            selected_source = "non_positive_log"
            if selected_cell is None and q_min_record and q_min_record.get("cell") is not None:
                selected_cell = int(q_min_record["cell"])
                selected_source = "log_q_min_cell"
            worst_cell = find_worst_cell(worst_snapshot)
            if selected_cell is None and worst_cell.get("available"):
                selected_cell = int(worst_cell["cell"])
                selected_source = "snapshot_worst_corner_metric"
            if selected_cell is not None:
                geom = cell_geometry_record(
                    worst_snapshot,
                    initial_snapshot,
                    int(selected_cell),
                    deck_kv.get("case", case),
                    thresholds,
                )
                geometry_metric = {
                    "available": True,
                    "selected_cell_source": selected_source,
                    "snapshot_path": worst_snapshot["path"],
                    "snapshot_step": worst_snapshot["step"],
                    "snapshot_t_s": worst_snapshot["t_s"],
                    "region_thresholds_cm": thresholds,
                    "failing_or_worst_cell": cell_geometry_to_json(
                        geom, thresholds.get("R_hs_cm")
                    ),
                    "snapshot_worst_cell": worst_cell,
                    "missing": [],
                }
            else:
                geometry_metric = {
                    "available": False,
                    "snapshot_path": worst_snapshot["path"],
                    "snapshot_step": worst_snapshot["step"],
                    "missing": ["failing_or_worst_cell_id"],
                    "snapshot_worst_cell": worst_cell,
                }
            aspect_metric = near_axis_aspect_pair(initial_snapshot, worst_snapshot)
        except Exception as exc:  # noqa: BLE001 - preserve in JSON.
            geometry_metric = {
                "available": False,
                "missing": [],
                "analysis_errors": [f"{type(exc).__name__}: {exc}"],
            }
            aspect_metric = {
                "available": False,
                "missing": [],
                "analysis_errors": [f"{type(exc).__name__}: {exc}"],
            }

    case_d_velocity_readout = None
    if case == "D":
        if velocity.get("trustworthy"):
            case_d_velocity_readout = {
                "emitted": True,
                "descriptive_only": True,
                "velocity_residual_grows": velocity.get("trend", {}).get("peak_gt_first"),
                "peak_max_fraction": velocity.get("trend", {}).get("peak_max_fraction"),
                "growth_peak_over_first": velocity.get("trend", {}).get(
                    "growth_peak_over_first"
                ),
            }
        else:
            case_d_velocity_readout = {
                "emitted": False,
                "reason": "affine_residual_self_test_failed_or_unavailable",
            }

    return {
        "schema_version": "tenryu.i1b_finite_cr_shell_discriminator.run.v1",
        "run_id": run_id,
        "mode": mode,
        "case": case,
        "nz": nz,
        "N_C": nz // 4,
        "max_steps_cap": max_steps,
        "deck_path": str(deck_path),
        "requested_outdir": str(requested_outdir),
        "actual_outdir": str(actual_outdir),
        "results_dir": str(results_dir) if results_dir is not None else None,
        "plot_file_count": len(plot_files),
        "history_path": str(history_path) if history_path is not None else None,
        "run": run,
        "log_paths": log_paths,
        "deck_banner": deck_kv,
        "run_info": {
            "path": str(run_info_path),
            "available": bool(run_info),
            "payload": run_info,
            "termination_reason": run_info.get("termination_reason", "unknown"),
            "final_step": run_info.get("step"),
            "final_t_s": run_info.get("t"),
            "final_dt_s": run_info.get("dt"),
        },
        "frozen_config": frozen_config_summary(frozen_path),
        "non_positive_2d_rz_cell": non_positive,
        "q_min_trend": q_metric,
        "failure_or_worst_cell_geometry": geometry_metric,
        "near_axis_aspect_ratio": aspect_metric,
        "energy_conservation": energy,
        "audit_summary": audit,
        "hotspot_gas": hotspot_gas,
        "eta_contact": eta_contact,
        "center_patch_sigma": center_patch_sigma,
        "diff_ref_diag": diff_ref_diag,
        "patchwise_affine_residual_velocity_ke": velocity,
        "velocity_first_cross_check": cross_check,
        "case_d_velocity_readout": case_d_velocity_readout,
        "discovery_errors": discovery_errors,
    }


def finer_earlier_for_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: int(item["nz"]))
    table: list[dict[str, Any]] = []
    for record in ordered:
        run_info = record.get("run_info", {})
        velocity = record.get("patchwise_affine_residual_velocity_ke", {})
        table.append(
            {
                "nz": record.get("nz"),
                "termination_reason": run_info.get("termination_reason"),
                "final_step": run_info.get("final_step"),
                "final_t_s": run_info.get("final_t_s"),
                "non_positive_token": record.get("non_positive_2d_rz_cell", {}).get(
                    "present"
                ),
                "velocity_residual_peak_gt_first": velocity.get("trend", {}).get(
                    "peak_gt_first"
                ),
                "velocity_residual_peak_max_fraction": velocity.get("trend", {}).get(
                    "peak_max_fraction"
                ),
            }
        )
    adjacent: list[dict[str, Any]] = []
    for prev, cur in zip(table[:-1], table[1:]):
        prev_step = prev.get("final_step")
        cur_step = cur.get("final_step")
        prev_t = finite_float(prev.get("final_t_s"))
        cur_t = finite_float(cur.get("final_t_s"))
        adjacent.append(
            {
                "coarser_nz": prev["nz"],
                "finer_nz": cur["nz"],
                "finer_terminated_earlier_by_step": (
                    cur_step < prev_step
                    if isinstance(prev_step, int) and isinstance(cur_step, int)
                    else None
                ),
                "finer_terminated_earlier_by_time": (
                    cur_t < prev_t if prev_t is not None and cur_t is not None else None
                ),
            }
        )
    return {
        "termination_table": table,
        "adjacent_finer_earlier": adjacent,
        "all_adjacent_finer_earlier_by_step": all(
            item["finer_terminated_earlier_by_step"] is True for item in adjacent
        )
        if adjacent
        else None,
        "all_adjacent_finer_earlier_by_time": all(
            item["finer_terminated_earlier_by_time"] is True for item in adjacent
        )
        if adjacent
        else None,
    }


def final_summary_value(summary: dict[str, Any] | None, key: str = "final_value") -> float | None:
    if not isinstance(summary, dict) or not summary.get("available"):
        return None
    return finite_float(summary.get(key))


def comparison_row(record: dict[str, Any]) -> dict[str, Any]:
    hotspot = record.get("hotspot_gas", {})
    eta = record.get("eta_contact", {})
    sigma = record.get("center_patch_sigma", {})
    diff_ref = record.get("diff_ref_diag", {})
    run_info = record.get("run_info", {})
    return {
        "case": record.get("case"),
        "case_label": CASE_LABELS.get(str(record.get("case")), ""),
        "nz": record.get("nz"),
        "deck_path": record.get("deck_path"),
        "termination_reason": run_info.get("termination_reason"),
        "final_step": run_info.get("final_step"),
        "final_t_s": run_info.get("final_t_s"),
        "CR50_final": final_summary_value(hotspot.get("CR50")),
        "CR50_peak": final_summary_value(hotspot.get("CR50"), "peak_value"),
        "rho50_ratio_final": final_summary_value(hotspot.get("rho50_ratio")),
        "p50_ratio_final": final_summary_value(hotspot.get("p50_ratio")),
        "C_rho_bulk_final": final_summary_value(
            diff_ref.get("summary") if isinstance(diff_ref, dict) else None
        ),
        "eta_contact_available": bool(eta.get("available")),
        "eta_contact_final": final_summary_value(
            eta.get("summary") if isinstance(eta, dict) else None
        ),
        "eta_contact_peak": final_summary_value(
            eta.get("summary") if isinstance(eta, dict) else None, "peak_value"
        ),
        "eta_contact_cumulative_sum": finite_float(
            eta.get("eta_contact_cumulative_sum")
        )
        if isinstance(eta, dict)
        else None,
        "eta_contact_step_peak": final_summary_value(
            eta.get("eta_contact_step_summary") if isinstance(eta, dict) else None,
            "peak_value",
        ),
        "contact_swept_volume_cm3_final": final_summary_value(
            eta.get("contact_swept_volume_cm3") if isinstance(eta, dict) else None
        ),
        "hotspot_volume_cm3_final": final_summary_value(
            eta.get("hotspot_volume_cm3") if isinstance(eta, dict) else None
        ),
        "center_patch_sigma_available": bool(sigma.get("available")),
        "center_patch_sigma_min": finite_float(sigma.get("min_sigma"))
        if isinstance(sigma, dict)
        else None,
        "center_patch_sigma_final": finite_float(sigma.get("final_sigma"))
        if isinstance(sigma, dict)
        else None,
        "center_patch_sigma_collapse_trigger_observed": sigma.get(
            "collapse_trigger_observed"
        )
        if isinstance(sigma, dict)
        else None,
        "history_path": record.get("history_path"),
    }


def finite_xy_from_hotspot(record: dict[str, Any], field: str) -> tuple[list[float], list[float]]:
    series = record.get("hotspot_gas", {}).get("series", {})
    times = series.get("t_s") if isinstance(series, dict) else None
    values = series.get(field) if isinstance(series, dict) else None
    if not isinstance(times, list) or not isinstance(values, list):
        return [], []
    out_t: list[float] = []
    out_v: list[float] = []
    for raw_t, raw_v in zip(times, values):
        t = finite_float(raw_t)
        v = finite_float(raw_v)
        if t is None or v is None:
            continue
        out_t.append(t)
        out_v.append(v)
    return out_t, out_v


def finite_xy_from_eta(record: dict[str, Any]) -> tuple[list[float], list[float]]:
    eta = record.get("eta_contact", {})
    records = eta.get("records") if isinstance(eta, dict) else None
    if not isinstance(records, list):
        return [], []
    out_t: list[float] = []
    out_v: list[float] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        t = finite_float(item.get("t_s"))
        v = finite_float(item.get("eta_contact"))
        if t is None or v is None:
            continue
        out_t.append(t)
        out_v.append(v)
    return out_t, out_v


def interpolate_to(times: list[float], values: list[float], targets: list[float]) -> list[float | None]:
    if len(times) < 2 or len(values) < 2 or not targets:
        return [None for _ in targets]
    import numpy as np

    order = np.argsort(np.asarray(times, dtype=float))
    t = np.asarray(times, dtype=float)[order]
    v = np.asarray(values, dtype=float)[order]
    unique_t, unique_indices = np.unique(t, return_index=True)
    unique_v = v[unique_indices]
    if unique_t.size < 2:
        return [None for _ in targets]
    out: list[float | None] = []
    lo = float(unique_t[0])
    hi = float(unique_t[-1])
    for target in targets:
        if target < lo or target > hi:
            out.append(None)
        else:
            out.append(float(np.interp(target, unique_t, unique_v)))
    return out


def build_aligned_curves(records_by_case: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cr_series = {
        case: finite_xy_from_hotspot(record, "CR50")
        for case, record in records_by_case.items()
    }
    usable_cr = {
        case: item for case, item in cr_series.items() if len(item[0]) >= 2
    }
    missing_cr = [case for case in VALID_CASES if case not in usable_cr]
    if missing_cr:
        return {
            "available": False,
            "missing": [f"CR50_series_case_{case}" for case in missing_cr],
        }
    common_min = max(times[0] for times, _ in usable_cr.values())
    common_max = min(times[-1] for times, _ in usable_cr.values())
    if not (common_max >= common_min):
        return {
            "available": False,
            "missing": ["overlapping_time_axis"],
            "time_bounds_by_case": {
                case: [times[0], times[-1]] for case, (times, _) in usable_cr.items()
            },
        }
    base_times = cr_series.get("D", next(iter(usable_cr.values())))[0]
    target_times = [t for t in base_times if common_min <= t <= common_max]
    if len(target_times) < 2:
        target_times = [common_min, common_max]
    aligned_cr = {
        case: interpolate_to(times, values, target_times)
        for case, (times, values) in usable_cr.items()
    }
    eta_series = {
        case: finite_xy_from_eta(record) for case, record in records_by_case.items()
    }
    aligned_eta = {
        case: interpolate_to(times, values, target_times)
        for case, (times, values) in eta_series.items()
        if len(times) >= 2
    }
    return {
        "available": True,
        "time_s": target_times,
        "CR50": aligned_cr,
        "eta_contact": aligned_eta,
        "missing_eta_contact_cases": [
            case for case in VALID_CASES if case not in aligned_eta
        ],
    }


def build_four_curve_comparison(run_records: list[dict[str, Any]]) -> dict[str, Any]:
    table = [comparison_row(record) for record in run_records]
    by_nz: dict[int, dict[str, dict[str, Any]]] = {}
    for record in run_records:
        nz_raw = record.get("nz")
        case = str(record.get("case"))
        if finite_float(nz_raw) is None:
            continue
        by_nz.setdefault(int(nz_raw), {})[case] = record
    aligned_by_nz = {
        str(nz): build_aligned_curves(records_by_case)
        for nz, records_by_case in sorted(by_nz.items())
    }
    return {
        "schema_version": "tenryu.i1b_finite_cr_shell_discriminator.comparison.v1",
        "description": "Four-curve CR50/eta_contact comparison; Case A is the 1D spherical reference, B pure-Lagrangian 2D, C full-ALE differential reference, D center-patch ALE.",
        "table": table,
        "aligned_by_nz": aligned_by_nz,
    }


def classify_discriminating_verdict(
    comparison: dict[str, Any],
    requested_cases: list[str],
    requested_nz: list[int],
) -> dict[str, Any]:
    table = comparison.get("table", [])
    rows_by_nz_case: dict[int, dict[str, dict[str, Any]]] = {}
    for row in table:
        if not isinstance(row, dict):
            continue
        nz = finite_float(row.get("nz"))
        case = row.get("case")
        if nz is None or case is None:
            continue
        rows_by_nz_case.setdefault(int(nz), {})[str(case)] = row
    complete_nz = [
        nz
        for nz in sorted(rows_by_nz_case)
        if all(case in rows_by_nz_case[nz] for case in VALID_CASES)
    ]
    if not complete_nz:
        return {
            "available": False,
            "branch_id": "AMBIGUOUS_INCOMPLETE_CASE_SET",
            "route": "run_all_four_cases",
            "missing": [
                f"case_{case}"
                for case in VALID_CASES
                if case not in requested_cases
            ],
            "requested_cases": requested_cases,
            "requested_nz": requested_nz,
        }
    nz = complete_nz[-1]
    rows = rows_by_nz_case[nz]
    cr = {case: finite_float(rows[case].get("CR50_final")) for case in VALID_CASES}
    missing_cr = [case for case, value in cr.items() if value is None]
    if missing_cr:
        return {
            "available": False,
            "branch_id": "AMBIGUOUS_MISSING_CR50",
            "route": "rerun_with_hotspot_gas_tracer",
            "nz": nz,
            "missing": [f"CR50_case_{case}" for case in missing_cr],
            "case_rows": rows,
        }
    eta_d = finite_float(rows["D"].get("eta_contact_final"))
    eta_d_peak = finite_float(rows["D"].get("eta_contact_peak"))
    eta_d_cumulative_sum = finite_float(rows["D"].get("eta_contact_cumulative_sum"))
    if eta_d is None:
        return {
            "available": False,
            "branch_id": "AMBIGUOUS_MISSING_ETA_CONTACT",
            "route": "rerun_after_eta_contact_diag_or_check_TENRYU_I1B_DIFFREF_DIAG",
            "nz": nz,
            "thresholds": {
                "eta_contact_zero_abs": ETA_CONTACT_ZERO_ABS,
                "cr50_target": CR50_TARGET,
            },
            "case_rows": rows,
        }

    a = float(cr["A"])
    b = float(cr["B"])
    c = float(cr["C"])
    d = float(cr["D"])
    lag_anchor = max(a, b)
    d_gain_over_c = d - c
    improvement_threshold = max(CR50_IMPROVEMENT_ABS, CR50_IMPROVEMENT_REL * abs(c))
    improved = d_gain_over_c > improvement_threshold
    eta_quiet = abs(eta_d) <= ETA_CONTACT_ZERO_ABS
    eta_active = eta_d > ETA_CONTACT_ZERO_ABS
    jumped_toward_lag = (
        d >= (1.0 - CR50_CLOSE_REL) * lag_anchor
        or (
            lag_anchor > c
            and d_gain_over_c >= 0.5 * (lag_anchor - c)
        )
    )
    stalled = (
        not improved
        or abs(d - CR50_STALL_BASELINE) <= CR50_IMPROVEMENT_ABS
    )
    saturated_below_target = d < (1.0 - CR50_CLOSE_REL) * CR50_TARGET
    sigma = rows["D"].get("center_patch_sigma_collapse_trigger_observed")

    branch_id = "AMBIGUOUS_STRADDLE"
    branch_label = "signals do not cleanly select a route"
    route = "reconfirm_eta_contact_sanity_and_reassess"
    if d >= (1.0 - CR50_CLOSE_REL) * CR50_TARGET and eta_quiet:
        branch_id = "TERMINATE_SUCCESS"
        branch_label = "center-patch closure reaches the finite-CR target with quiet contact sweep"
        route = "terminate_success"
    elif jumped_toward_lag:
        branch_id = "1"
        branch_label = "D jumps toward pure-Lagrangian/1D reference; center-patch bulk closure is sufficient"
        route = "proceed_to_MMALE_only_if_residual_interface_gap_remains"
    elif stalled and eta_quiet:
        branch_id = "2"
        branch_label = "D stalls near the ALE baseline while eta_contact is quiet; remap is not the cause"
        route = (
            "stage4_center_patch_phi_barrier"
            if sigma is True
            else "stop_and_reassess_setup_center_diagnostic_EOS"
        )
    elif improved and saturated_below_target and eta_active:
        branch_id = "3"
        branch_label = "D improves but saturates with active contact sweep; contact smearing needs MMALE"
        route = "proceed_to_MMALE_PLIC"

    return {
        "available": branch_id not in ("AMBIGUOUS_STRADDLE",),
        "branch_id": branch_id,
        "branch_label": branch_label,
        "route": route,
        "nz": nz,
        "metrics": {
            "CR50_A_final": a,
            "CR50_B_final": b,
            "CR50_C_final": c,
            "CR50_D_final": d,
            "CR50_D_gain_over_C": d_gain_over_c,
            "lag_anchor_max_A_B": lag_anchor,
            "eta_contact_D_final": eta_d,
            "eta_contact_D_peak": eta_d_peak,
            "eta_contact_D_cumulative_sum": eta_d_cumulative_sum,
            "center_patch_sigma_collapse_trigger_observed": sigma,
        },
        "predicates": {
            "improved_over_C": improved,
            "jumped_toward_lag": jumped_toward_lag,
            "stalled": stalled,
            "saturated_below_target": saturated_below_target,
            "eta_quiet": eta_quiet,
            "eta_active": eta_active,
        },
        "thresholds": {
            "cr50_target": CR50_TARGET,
            "cr50_stall_baseline": CR50_STALL_BASELINE,
            "cr50_improvement_abs": CR50_IMPROVEMENT_ABS,
            "cr50_improvement_rel": CR50_IMPROVEMENT_REL,
            "cr50_close_rel": CR50_CLOSE_REL,
            "eta_contact_zero_abs": ETA_CONTACT_ZERO_ABS,
            "center_patch_sigma_collapse": CENTER_PATCH_SIGMA_COLLAPSE,
        },
        "case_rows": rows,
    }


def format_optional_float(value: Any, precision: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    return f"{number:.{precision}g}"


def build_text_summary(summary: dict[str, Any]) -> str:
    comparison = summary.get("four_curve_comparison", {})
    verdict = summary.get("verdict", {})
    lines = [
        "TENRYU I1-B finite-CR shell discriminator",
        f"generated_iso: {summary.get('generated_iso')}",
        f"mode: {summary.get('mode')} run_count: {summary.get('run_count')}",
        "",
        "case nz final_t_ns CR50_final CR50_peak rho50_ratio eta_contact center_patch_sigma_min termination",
    ]
    table = comparison.get("table", []) if isinstance(comparison, dict) else []
    for row in table:
        if not isinstance(row, dict):
            continue
        final_t = finite_float(row.get("final_t_s"))
        final_t_ns = final_t * 1.0e9 if final_t is not None else None
        lines.append(
            " ".join(
                [
                    str(row.get("case")),
                    str(row.get("nz")),
                    format_optional_float(final_t_ns),
                    format_optional_float(row.get("CR50_final")),
                    format_optional_float(row.get("CR50_peak")),
                    format_optional_float(row.get("rho50_ratio_final")),
                    format_optional_float(row.get("eta_contact_final")),
                    format_optional_float(row.get("center_patch_sigma_min")),
                    str(row.get("termination_reason")),
                ]
            )
        )
    lines.extend(
        [
            "",
            "VERDICT",
            f"available: {verdict.get('available')}",
            f"branch_id: {verdict.get('branch_id')}",
            f"branch_label: {verdict.get('branch_label')}",
            f"route: {verdict.get('route')}",
        ]
    )
    metrics = verdict.get("metrics")
    if isinstance(metrics, dict):
        lines.append(
            "metrics: "
            + " ".join(
                f"{key}={format_optional_float(value)}"
                for key, value in sorted(metrics.items())
            )
        )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_overall_readout(
    run_records: list[dict[str, Any]],
    requested_cases: list[str],
    requested_nz: list[int],
    mode: str,
) -> dict[str, Any]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for record in run_records:
        by_case.setdefault(str(record["case"]), []).append(record)
    full_default = {(case, nz) for case in FULL_CASES for nz in FULL_NZ}
    requested = {(case, nz) for case in requested_cases for nz in requested_nz}
    skipped = sorted(
        (
            {"case": case, "nz": nz, "reason": "not_in_smoke_ladder"}
            for case, nz in full_default - requested
        ),
        key=lambda item: (item["case"], item["nz"]),
    )
    case_readouts = {
        case: finer_earlier_for_case(records) for case, records in sorted(by_case.items())
    }
    return {
        "descriptive_only": True,
        "no_physics_pass_fail_gate": True,
        "mode": mode,
        "requested_cases": requested_cases,
        "requested_nz": requested_nz,
        "skipped_by_mode": skipped if mode == "smoke" else [],
        "case_readouts": case_readouts,
        "artifact_vs_defect_readout": {
            "description": "Descriptive signals for whether behavior is resolution-convergent, finer-earlier, and accompanied by a velocity nullspace residual.",
            "case_B": case_readouts.get("B"),
            "notes": [
                "finer-earlier is reported, not asserted",
                "velocity-residual growth is reported only after the synthetic affine self-test",
                "C>=7.2 is never asserted by this harness",
            ],
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--cases", default=None, help="Comma-separated cases, e.g. A,B,C,D")
    parser.add_argument("--nz", default=None, help="Comma-separated nz values, e.g. 64,128,256")
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default=str(DEFAULT_DECK), help="2D B/C/D deck")
    parser.add_argument("--deck-1d", default=str(DEFAULT_1D_DECK), help="1D Case A deck")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--outroot", default=str(DEFAULT_OUTROOT))
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-text", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke-max-steps", type=int, default=SMOKE_MAX_STEPS)
    parser.add_argument("--full-max-steps", type=int, default=FULL_MAX_STEPS)
    parser.add_argument("--wall-time-s", type=float, default=7200.0)
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help=f"Extra deck env override as NAME=VALUE; {ENV_PREFIX} is added if omitted",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    default_cases = SMOKE_CASES if args.mode == "smoke" else FULL_CASES
    default_nz = SMOKE_NZ if args.mode == "smoke" else FULL_NZ
    cases = parse_csv_cases(args.cases, default_cases)
    nz_values = parse_csv_ints(args.nz, default_nz)
    max_steps = args.max_steps
    if max_steps is None:
        max_steps = args.smoke_max_steps if args.mode == "smoke" else args.full_max_steps
    if max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    repo_root = Path(args.repo_root).resolve()
    deck_path = cap.resolve_under_repo(repo_root, args.deck)
    deck_1d_path = cap.resolve_under_repo(repo_root, args.deck_1d)
    outroot = cap.resolve_under_repo(repo_root, args.outroot)
    summary_path = (
        cap.resolve_under_repo(repo_root, args.summary_json)
        if args.summary_json
        else outroot / "discriminator_summary.json"
    )
    summary_text_path = (
        cap.resolve_under_repo(repo_root, args.summary_text)
        if args.summary_text
        else outroot / "discriminator_summary.txt"
    )
    env_extra = parse_set_env(args.set_env)

    run_records: list[dict[str, Any]] = []
    for case in cases:
        for nz in nz_values:
            run_id, run_outdir = allocate_run_dir(outroot / "runs", f"case{case}_nz{nz}")
            log_path = outroot / "logs" / f"{run_id}.log"
            case_deck_path = deck_1d_path if case == "A" else deck_path
            run = run_deck(
                repo_root=repo_root,
                tenryu_bin=args.tenryu_bin,
                deck_path=case_deck_path,
                outdir=run_outdir,
                log_path=log_path,
                case=case,
                nz=nz,
                max_steps=max_steps,
                wall_time_s=args.wall_time_s,
                env_extra=env_extra,
            )
            run_records.append(
                analyze_run(
                    repo_root=repo_root,
                    deck_path=case_deck_path,
                    run_id=run_id,
                    case=case,
                    nz=nz,
                    mode=args.mode,
                    max_steps=max_steps,
                    requested_outdir=run_outdir,
                    harness_log=log_path,
                    run=run,
                )
            )

    comparison = build_four_curve_comparison(run_records)
    verdict = classify_discriminating_verdict(comparison, cases, nz_values)
    summary = {
        "schema_version": "tenryu.i1b_finite_cr_shell_discriminator.summary.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "deck_path": str(deck_path),
        "deck_1d_path": str(deck_1d_path),
        "tenryu_bin": args.tenryu_bin,
        "outroot": str(outroot),
        "max_steps_cap": max_steps,
        "run_count": len(run_records),
        "runs": run_records,
        "four_curve_comparison": comparison,
        "verdict": verdict,
        "overall_readout": build_overall_readout(run_records, cases, nz_values, args.mode),
    }
    write_json(summary_path, summary)
    write_text(summary_text_path, build_text_summary(summary))
    print(
        json.dumps(
            {
                "summary_json": str(summary_path),
                "summary_text": str(summary_text_path),
                "mode": args.mode,
                "run_count": len(run_records),
                "verdict_branch": verdict.get("branch_id"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
