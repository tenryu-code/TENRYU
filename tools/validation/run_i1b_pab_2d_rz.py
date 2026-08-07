#!/usr/bin/env python3
"""Run the I1B-PAB-POLAR 2D RZ pressure-drive validation harness.

This harness drives the a0=0 spherical-polar pressure-boundary deck and reuses
the I1B-GEO all-run mesh-quality and fail-closed completion/escape predicates.
The pressure-drive validity signal is separate from the GEO velocity-IC signal:
the run must compress the polar outer boundary to the requested C* fraction and
must show nontrivial pressure-drive work or kinetic energy.
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

import run_i1b_geo_2d_rz as geo
import run_i1b_lite_2d_rz as lite


STAGE_ID = "I1B-PAB-POLAR"
MODE = "i1b_pab_polar_2d_rz_pressure_gate"
DECK_PREFIX = "TENRYU_I1B_PAB_POLAR_"
MESH_QUALITY_GROUP = geo.MESH_QUALITY_GROUP
R_OUTER_DEFAULT_CM = 0.07
DEFAULT_RHO0_GCC = 0.25
DEFAULT_P0_DYN_CM2 = 1.0e14
DEFAULT_C_STAR = 8.0
DEFAULT_T_END_S = 4.4e-9
DEFAULT_DT_S = 2.0e-13
DEFAULT_NR = 64
DEFAULT_NZ = 128
DEFAULT_ALE_EVERY_N_STEPS = 5
DEFAULT_PLOT_EVERY_STEPS = 50
DEFAULT_HISTORY_EVERY_STEPS = 1
DEFAULT_MAX_STEPS = 1_000_000
DEFAULT_LADDER_P0_GRID = (2.5e13, 5.0e13, 1.0e14, 1.5e14, 2.0e14)
DEFAULT_T_RAMP_UP_S = 0.5e-9
DEFAULT_T_HOLD_END_S = 2.5e-9
DEFAULT_T_RAMP_DOWN_S = 0.5e-9
DRIVE_C_FRACTION_FLOOR = 0.9
DRIVE_ENERGY_FLOOR_ERG = 1.0e11


@dataclass(frozen=True)
class PabCase:
    p0_dyn_cm2: float
    c_star: float
    t_end_s: float


def parse_float_list(value: str) -> list[float]:
    return geo.parse_float_list(value)


def json_sanitize(value: Any) -> Any:
    return lite.json_sanitize(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    lite.write_json(path, payload)


def signal(value: Any, passed: bool, source: str, required: bool = True) -> dict[str, Any]:
    return lite.signal(value, passed, source, required)


def safe_float_token(value: float) -> str:
    return lite.safe_float_token(value)


def mbar_from_dyn_cm2(value: float) -> float:
    return value / 1.0e12


def compact_case_label(case: PabCase) -> str:
    return (
        f"P{safe_float_token(mbar_from_dyn_cm2(case.p0_dyn_cm2))}Mbar"
        f"_C{safe_float_token(case.c_star)}_T{safe_float_token(case.t_end_s)}"
    )


def c2_smoothstep(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x * x * x * (10.0 + x * (-15.0 + 6.0 * x))


def ablation_pressure_ramp(t: float, p0: float, t_r: float, t_h: float, t_d: float) -> float:
    if t <= 0.0:
        return 0.0
    if t <= t_r:
        return p0 * c2_smoothstep(t / t_r)
    if t <= t_h:
        return p0
    if t <= t_h + t_d:
        return p0 * (1.0 - c2_smoothstep((t - t_h) / t_d))
    return 0.0


def planned_cases(args: argparse.Namespace) -> list[PabCase]:
    if args.mode == "calibrate":
        t_grid = args.calibrate_t_end_grid if args.calibrate_t_end_grid is not None else [args.t_end_s]
        return [PabCase(args.ci_p0_dyn_cm2, args.ci_c_star, t_end_s) for t_end_s in t_grid]
    if args.mode == "ci":
        return [PabCase(args.ci_p0_dyn_cm2, args.ci_c_star, args.t_end_s)]
    return [PabCase(p0, args.ci_c_star, args.t_end_s) for p0 in args.p0_grid]


def active_env(args: argparse.Namespace, outdir: Path, case: PabCase) -> dict[str, str]:
    return {
        DECK_PREFIX + "OUTDIR": str(outdir),
        DECK_PREFIX + "NR": str(args.nr),
        DECK_PREFIX + "NZ": str(args.nz),
        DECK_PREFIX + "SEED": str(args.seed),
        DECK_PREFIX + "R_OUTER_CM": f"{args.r_outer_cm:.17g}",
        DECK_PREFIX + "RHO0_GCC": f"{args.rho0_gcc:.17g}",
        DECK_PREFIX + "P0_DYN_CM2": f"{case.p0_dyn_cm2:.17g}",
        DECK_PREFIX + "C_STAR": f"{case.c_star:.17g}",
        DECK_PREFIX + "T_END_S": f"{case.t_end_s:.17g}",
        DECK_PREFIX + "DT_INITIAL_S": f"{args.dt_initial_s:.17g}",
        DECK_PREFIX + "DT_MAX_S": f"{args.dt_max_s:.17g}",
        DECK_PREFIX + "ALE_EVERY_N_STEPS": str(args.ale_every_n_steps),
        DECK_PREFIX + "PLOT_EVERY_STEPS": str(args.plot_every_steps),
        DECK_PREFIX + "HISTORY_EVERY_STEPS": str(args.history_every_steps),
        DECK_PREFIX + "MAX_STEPS": str(args.max_steps),
        DECK_PREFIX + "MESH_QUALITY_MIN_DIAG": "true",
    }


def parse_deck_kv(log_path: Path) -> dict[str, str]:
    text = lite.read_text(log_path)
    for line in text.splitlines():
        if not line.startswith("[deck:2d_rz_i1b_pab_polar]"):
            continue
        return {match.group(1): match.group(2) for match in re.finditer(r"([A-Za-z0-9_]+)=([^ \t]+)", line)}
    return {}


def parse_pab_log_metadata(log_path: Path) -> dict[str, Any]:
    kv = parse_deck_kv(log_path)
    out: dict[str, Any] = {}
    float_keys = {
        "P0": "deck_p0_dyn_cm2",
        "t_r_s": "deck_t_ramp_up_s",
        "t_h_s": "deck_t_hold_end_s",
        "t_d_s": "deck_t_ramp_down_s",
        "C_star": "deck_c_star",
        "S_max": "deck_r_outer_cm",
        "gamma": "deck_gamma",
        "rho0_gcc": "deck_rho0_gcc",
        "T0_eV": "deck_T0_eV",
        "t_end_s": "deck_t_end_s",
    }
    int_keys = {"nr": "deck_nr", "nz": "deck_nz", "seed": "deck_seed"}
    bool_keys = {
        "i1b_pab_polar_mode": "i1b_pab_polar_mode",
        "mesh_quality_min_diag": "deck_mesh_quality_min_diag",
        "conservative_remap_enabled": "deck_conservative_remap_enabled",
    }
    for key, out_key in float_keys.items():
        value = geo.maybe_float(kv.get(key))
        if value is not None:
            out[out_key] = value
    for key, out_key in int_keys.items():
        value = geo.maybe_int(kv.get(key))
        if value is not None:
            out[out_key] = value
    for key, out_key in bool_keys.items():
        value = geo.bool_token(kv.get(key))
        if value is not None:
            out[out_key] = value
    if "outdir" in kv:
        out["deck_outdir"] = kv["outdir"]
    if "operator_semantics" in kv:
        out["deck_operator_semantics"] = kv["operator_semantics"]
    if "P0_units" in kv:
        out["deck_p0_units"] = kv["P0_units"]
    if "S_max_units" in kv:
        out["deck_r_outer_units"] = kv["S_max_units"]
    return out


def read_optional_history_metric(history: Path | None, dataset: str) -> dict[str, Any]:
    if history is None:
        return {"available": False, "source_path": None, "missing": ["history_hdf5"]}
    import h5py
    import numpy as np

    key = dataset[1:] if dataset.startswith("/") else dataset
    try:
        with h5py.File(history, "r") as handle:
            if key not in handle:
                return {"available": False, "source_path": str(history), "missing": [dataset]}
            arr = np.asarray(handle[key][()], dtype=float).reshape(-1)
    except OSError as exc:
        return {
            "available": False,
            "source_path": str(history),
            "missing": [],
            "analysis_errors": [str(exc)],
        }
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {
            "available": False,
            "source_path": str(history),
            "missing": [f"finite_{dataset}"],
            "analysis_errors": [],
        }
    return {
        "available": True,
        "source_path": str(history),
        "row_count": int(arr.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "final": float(arr[-1]),
        "missing": [],
        "analysis_errors": [],
    }


def attach_conditioning_observability(mesh_quality: dict[str, Any], history: Path | None) -> dict[str, Any]:
    metric_paths = {
        "min_edge_length_rel": f"{MESH_QUALITY_GROUP}/achieved_min_edge_length_rel",
        "min_altitude_rel": f"{MESH_QUALITY_GROUP}/achieved_min_altitude_rel",
        "max_condition_number": f"{MESH_QUALITY_GROUP}/achieved_max_condition_number",
    }
    metrics = {name: read_optional_history_metric(history, path) for name, path in metric_paths.items()}
    out = dict(mesh_quality)
    out["conditioning_observability"] = {
        "achieved_min_edge_length_rel": metrics["min_edge_length_rel"].get("min"),
        "achieved_min_altitude_rel": metrics["min_altitude_rel"].get("min"),
        "achieved_max_condition_number": metrics["max_condition_number"].get("max"),
        "metrics": metrics,
        "note": "reported only; not part of the I1B-PAB-POLAR pass/fail predicate",
    }
    return out


def outer_boundary_mean_radius(snapshot: geo.DriveSnapshot) -> float | None:
    import numpy as np

    radius = np.sqrt(snapshot.r_nodes[-1, :] * snapshot.r_nodes[-1, :] + snapshot.z_nodes[-1, :] * snapshot.z_nodes[-1, :])
    finite = radius[np.isfinite(radius) & (radius > 0.0)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def total_domain_volume(snapshot: geo.DriveSnapshot) -> float | None:
    import numpy as np

    vol = np.asarray(snapshot.vol, dtype=float)
    finite = vol[np.isfinite(vol)]
    if finite.size == 0:
        return None
    return float(np.sum(finite))


def pressure_params_from_metadata(metadata: dict[str, Any], case: PabCase) -> dict[str, float]:
    return {
        "p0_dyn_cm2": float(metadata.get("deck_p0_dyn_cm2", case.p0_dyn_cm2)),
        "t_ramp_up_s": float(metadata.get("deck_t_ramp_up_s", DEFAULT_T_RAMP_UP_S)),
        "t_hold_end_s": float(metadata.get("deck_t_hold_end_s", DEFAULT_T_HOLD_END_S)),
        "t_ramp_down_s": float(metadata.get("deck_t_ramp_down_s", DEFAULT_T_RAMP_DOWN_S)),
    }


def pressure_at(t: float, pressure: dict[str, float]) -> float:
    return ablation_pressure_ramp(
        t,
        pressure["p0_dyn_cm2"],
        pressure["t_ramp_up_s"],
        pressure["t_hold_end_s"],
        pressure["t_ramp_down_s"],
    )


def integrate_drive_work(samples: list[dict[str, Any]], pressure: dict[str, float]) -> dict[str, Any]:
    increments = []
    total = 0.0
    positive_only = 0.0
    for prev, cur in zip(samples[:-1], samples[1:]):
        v0 = prev.get("domain_volume_cm3")
        v1 = cur.get("domain_volume_cm3")
        t0 = prev.get("time_s")
        t1 = cur.get("time_s")
        if not all(isinstance(item, (int, float)) and math.isfinite(float(item)) for item in (v0, v1, t0, t1)):
            continue
        p0 = pressure_at(float(t0), pressure)
        p1 = pressure_at(float(t1), pressure)
        p_avg = 0.5 * (p0 + p1)
        dV_inward = float(v0) - float(v1)
        dW = p_avg * dV_inward
        total += dW
        if dW > 0.0:
            positive_only += dW
        increments.append(
            {
                "from_step": prev.get("step"),
                "to_step": cur.get("step"),
                "from_time_s": float(t0),
                "to_time_s": float(t1),
                "pressure_avg_dyn_cm2": p_avg,
                "minus_dV_domain_cm3": dV_inward,
                "dW_drive_erg": dW,
            }
        )
    return {
        "W_drive_erg": total if increments else None,
        "W_drive_positive_increments_erg": positive_only if increments else None,
        "increment_count": len(increments),
        "increments": increments[:20],
        "increments_truncated": max(0, len(increments) - 20),
        "definition": "sum over plot snapshots of 0.5*(P_i+P_{i+1})*(V_i - V_{i+1})",
    }


def evaluate_drive_validity(
    results_dir: Path | None,
    *,
    nr: int,
    nz: int,
    c_star: float,
    r_outer_cm: float,
    pressure: dict[str, float],
) -> dict[str, Any]:
    missing: list[str] = []
    failed: list[str] = []
    analysis_errors: list[str] = []
    if results_dir is None:
        missing.append("results_dir")
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "signals": {},
        }

    plot_files = lite.locate_plot_files(results_dir)
    if len(plot_files) < 2:
        missing.append("at_least_two_plot_snapshots")
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "plot_file_count": len(plot_files),
            "signals": {},
        }

    try:
        snapshots = [geo.read_drive_snapshot(path, nr, nz) for path in plot_files]
        samples = []
        kinetic_values = []
        for snap in snapshots:
            r_mean = outer_boundary_mean_radius(snap)
            domain_volume = total_domain_volume(snap)
            kinetic = geo.kinetic_energy_erg(snap)
            if math.isfinite(kinetic):
                kinetic_values.append(kinetic)
            samples.append(
                {
                    "path": str(snap.path),
                    "step": snap.step,
                    "time_s": snap.t,
                    "R_mean_outer_cm": r_mean,
                    "C_meas": r_outer_cm / r_mean if r_mean is not None and r_mean > 0.0 else None,
                    "domain_volume_cm3": domain_volume,
                    "boundary_pressure_dyn_cm2": pressure_at(snap.t, pressure),
                    "kinetic_energy_erg": kinetic if math.isfinite(kinetic) else None,
                }
            )
    except (OSError, KeyError, ValueError, RuntimeError, FloatingPointError) as exc:
        analysis_errors.append(str(exc))
        return {
            "available": False,
            "passed": False,
            "missing": missing,
            "failed": failed,
            "analysis_errors": analysis_errors,
            "source_path": str(results_dir),
            "signals": {},
        }

    finite_c = [
        float(item["C_meas"])
        for item in samples
        if isinstance(item.get("C_meas"), (int, float)) and math.isfinite(float(item["C_meas"]))
    ]
    finite_volume = [
        float(item["domain_volume_cm3"])
        for item in samples
        if isinstance(item.get("domain_volume_cm3"), (int, float))
        and math.isfinite(float(item["domain_volume_cm3"]))
    ]
    if not finite_c:
        missing.append("finite_outer_boundary_compression_series")
    if len(finite_volume) < 2:
        missing.append("finite_domain_volume_pair")
    if not kinetic_values:
        missing.append("finite_kinetic_energy_series")

    final_c = samples[-1].get("C_meas") if samples else None
    peak_ke = max(kinetic_values) if kinetic_values else None
    work = integrate_drive_work(samples, pressure)
    w_drive = work.get("W_drive_erg")
    if w_drive is None:
        missing.append("delivered_drive_work")

    c_required = DRIVE_C_FRACTION_FLOOR * c_star
    drive_energy_value = {"peak_kinetic_energy_erg": peak_ke, "W_drive_erg": w_drive}
    drive_energy_reported = (
        isinstance(peak_ke, (int, float))
        and math.isfinite(float(peak_ke))
        and isinstance(w_drive, (int, float))
        and math.isfinite(float(w_drive))
    )
    signals = {
        "C_meas_final": signal(
            final_c,
            isinstance(final_c, (int, float)) and math.isfinite(float(final_c)) and float(final_c) >= c_required,
            str(plot_files[-1]),
        ),
        "peak_kinetic_energy_erg_reported": signal(
            peak_ke,
            isinstance(peak_ke, (int, float)) and math.isfinite(float(peak_ke)),
            str(results_dir),
        ),
        "delivered_drive_work_erg_reported": signal(
            w_drive,
            isinstance(w_drive, (int, float)) and math.isfinite(float(w_drive)),
            str(results_dir),
        ),
        "drive_energy_nontrivial": signal(
            drive_energy_value if drive_energy_reported else None,
            drive_energy_reported
            and (float(peak_ke) > DRIVE_ENERGY_FLOOR_ERG or float(w_drive) > DRIVE_ENERGY_FLOOR_ERG),
            str(results_dir),
        ),
    }
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing.append(name)
        if item["required"] and not item["passed"]:
            failed.append(name)
    missing = sorted(set(missing))
    failed = sorted(set(failed))
    return {
        "available": not missing and not analysis_errors,
        "passed": not missing and not failed and not analysis_errors,
        "missing": missing,
        "failed": failed,
        "analysis_errors": analysis_errors,
        "source_path": str(results_dir),
        "plot_file_count": len(plot_files),
        "first_snapshot": {
            "path": str(snapshots[0].path),
            "step": snapshots[0].step,
            "time_s": snapshots[0].t,
        },
        "last_snapshot": {
            "path": str(snapshots[-1].path),
            "step": snapshots[-1].step,
            "time_s": snapshots[-1].t,
        },
        "compression_measurement": {
            "definition": "C_meas(t)=R0/R_mean_outer(t); R_mean_outer is mean sqrt(r^2+z^2) over i=nr polar outer-boundary nodes",
            "R0_outer_cm": r_outer_cm,
            "C_star": c_star,
            "fraction": DRIVE_C_FRACTION_FLOOR,
            "C_required": c_required,
            "C_meas_final": final_c,
            "C_meas_peak": max(finite_c) if finite_c else None,
        },
        "compression_series": samples,
        "pressure_ramp": pressure,
        "drive_work": work,
        "peak_kinetic_energy_erg": peak_ke,
        "drive_energy_floor_erg": DRIVE_ENERGY_FLOOR_ERG,
        "signals": signals,
    }


def deck_contract_signals(metadata: dict[str, Any], log_path: Path) -> dict[str, Any]:
    required_params = (
        "deck_p0_dyn_cm2",
        "deck_t_ramp_up_s",
        "deck_t_hold_end_s",
        "deck_t_ramp_down_s",
        "deck_c_star",
        "deck_r_outer_cm",
        "deck_t_end_s",
        "deck_nr",
        "deck_nz",
        "deck_seed",
        "deck_outdir",
    )
    missing_params = [name for name in required_params if name not in metadata]
    expected_operator = "laser_off_radiation_off_conduction_off_pressure_drive"
    return {
        "deck_header_required_params_present": signal(
            missing_params,
            not missing_params,
            str(log_path),
        ),
        "deck_i1b_pab_polar_mode": signal(
            metadata.get("i1b_pab_polar_mode"),
            metadata.get("i1b_pab_polar_mode") is True,
            str(log_path),
        ),
        "deck_p0_units_dyn_per_cm2": signal(
            metadata.get("deck_p0_units"),
            metadata.get("deck_p0_units") == "dyn_per_cm2",
            str(log_path),
        ),
        "deck_r_outer_units_cm": signal(
            metadata.get("deck_r_outer_units"),
            metadata.get("deck_r_outer_units") == "cm",
            str(log_path),
        ),
        "deck_mesh_quality_min_diag_enabled": signal(
            metadata.get("deck_mesh_quality_min_diag"),
            metadata.get("deck_mesh_quality_min_diag") is True,
            str(log_path),
        ),
        "deck_conservative_remap_disabled": signal(
            metadata.get("deck_conservative_remap_enabled"),
            metadata.get("deck_conservative_remap_enabled") is False,
            str(log_path),
        ),
        "deck_operator_semantics": signal(
            metadata.get("deck_operator_semantics"),
            metadata.get("deck_operator_semantics") == expected_operator,
            str(log_path),
        ),
    }


def evaluate_predicate(
    *,
    exit_code: int,
    timed_out: bool,
    outdir: Path,
    log_path: Path,
    args: argparse.Namespace,
    case: PabCase,
) -> dict[str, Any]:
    metadata = parse_pab_log_metadata(log_path)
    analysis_errors: list[str] = []
    results_dir: Path | None = None
    history: Path | None = None
    run_info_path: Path | None = None
    frozen_config: Path | None = None
    try:
        results_dir = lite.discover_results_dir(outdir)
        history = lite.locate_history_file(results_dir)
        run_info_path = lite.locate_run_info(outdir, results_dir)
        frozen_config = lite.locate_frozen_config(outdir, results_dir)
    except (OSError, RuntimeError, FileNotFoundError) as exc:
        analysis_errors.append(str(exc))
        run_info_path = lite.locate_run_info(outdir, results_dir)

    nr = int(metadata.get("deck_nr", args.nr))
    nz = int(metadata.get("deck_nz", args.nz))
    r_outer_cm = float(metadata.get("deck_r_outer_cm", args.r_outer_cm))
    t_end_s = float(metadata.get("deck_t_end_s", case.t_end_s))
    mesh_quality = attach_conditioning_observability(geo.read_mesh_quality_min_history(history), history)
    completion = geo.evaluate_completion_escape_and_fail_closed(
        exit_code=exit_code,
        timed_out=timed_out,
        outdir=outdir,
        log_path=log_path,
        results_dir=results_dir,
        history=history,
        frozen_config=frozen_config,
        run_info_path=run_info_path,
        t_end_s=t_end_s,
    )
    drive = evaluate_drive_validity(
        results_dir,
        nr=nr,
        nz=nz,
        c_star=float(metadata.get("deck_c_star", case.c_star)),
        r_outer_cm=r_outer_cm,
        pressure=pressure_params_from_metadata(metadata, case),
    )

    geometry_signals = geo.mesh_quality_geometry_signals(mesh_quality)
    deck_signals = deck_contract_signals(metadata, log_path)
    signals = {
        **deck_signals,
        **completion.get("signals", {}),
        **geometry_signals,
        **{f"drive_{name}": value for name, value in drive.get("signals", {}).items()},
    }
    missing_signals: list[str] = []
    missing_signals.extend(mesh_quality.get("missing", []))
    missing_signals.extend(completion.get("missing_signals", []))
    missing_signals.extend(f"drive:{name}" for name in drive.get("missing", []))
    for name, item in signals.items():
        if item["required"] and item["value"] is None:
            missing_signals.append(name)
    failed_signals = [name for name, item in signals.items() if item["required"] and not item["passed"]]
    missing_signals = sorted(set(missing_signals))
    all_analysis_errors = (
        analysis_errors
        + mesh_quality.get("analysis_errors", [])
        + completion.get("analysis_errors", [])
        + drive.get("analysis_errors", [])
    )
    predicate_passed = (
        not timed_out
        and exit_code == 0
        and not all_analysis_errors
        and not missing_signals
        and not failed_signals
    )
    deck_contract_valid = all(item["passed"] for item in deck_signals.values())
    geometry_admissible = all(item["passed"] for item in geometry_signals.values()) and mesh_quality.get("available")
    return {
        "passed": predicate_passed,
        "deck_contract_valid": bool(deck_contract_valid),
        "geometry_admissible": bool(geometry_admissible),
        "completion_escape_fail_closed_passed": bool(completion.get("passed")),
        "drive_validity_passed": bool(drive.get("passed")),
        "missing_signals": missing_signals,
        "failed_signals": failed_signals,
        "signals": signals,
        "thresholds": {
            "corner_j_rel_floor": lite.ADMISSIBILITY_FLOOR_REL,
            "gauss_j_rel_floor": lite.ADMISSIBILITY_FLOOR_REL,
            "rz_volume_rel_floor": 0.0,
            "negative_rz_volume_count_total": 0,
            "C_meas_final_floor": DRIVE_C_FRACTION_FLOOR * case.c_star,
            "drive_energy_floor_erg": DRIVE_ENERGY_FLOOR_ERG,
        },
        "mesh_quality_min": mesh_quality,
        "completion_escape_fail_closed": completion,
        "drive_validity": drive,
        "metadata": metadata,
        "analysis_errors": all_analysis_errors,
        "paths": {
            "outdir": str(outdir),
            "actual_outdir": completion.get("actual_outdir"),
            "log_path": str(log_path),
            "results_dir": str(results_dir) if results_dir is not None else None,
            "history_hdf5": str(history) if history is not None else None,
            "run_info": str(run_info_path) if run_info_path is not None else None,
            "frozen_config": str(frozen_config) if frozen_config is not None else None,
        },
    }


def first_compression_crossing(compression_series: list[dict[str, Any]], c_star: float) -> dict[str, Any] | None:
    prev: dict[str, Any] | None = None
    for item in compression_series:
        c_value = item.get("C_meas")
        t_value = item.get("time_s")
        if not (
            isinstance(c_value, (int, float))
            and math.isfinite(float(c_value))
            and isinstance(t_value, (int, float))
            and math.isfinite(float(t_value))
        ):
            prev = item
            continue
        if float(c_value) >= c_star:
            t_cross = float(t_value)
            interpolation = "snapshot"
            if prev is not None:
                c0 = prev.get("C_meas")
                t0 = prev.get("time_s")
                if (
                    isinstance(c0, (int, float))
                    and math.isfinite(float(c0))
                    and isinstance(t0, (int, float))
                    and math.isfinite(float(t0))
                    and float(c_value) != float(c0)
                ):
                    frac = (c_star - float(c0)) / (float(c_value) - float(c0))
                    if math.isfinite(frac):
                        frac = min(1.0, max(0.0, frac))
                        t_cross = float(t0) + frac * (float(t_value) - float(t0))
                        interpolation = "linear_C_between_plot_snapshots"
            return {
                "crossed": True,
                "C_star": c_star,
                "t_cross_s": t_cross,
                "interpolation": interpolation,
                "snapshot_step": item.get("step"),
                "snapshot_time_s": float(t_value),
                "snapshot_C_meas": float(c_value),
                "snapshot_path": item.get("path"),
            }
        prev = item
    return None


def row_from_payload(payload: dict[str, Any], summary_path: Path) -> dict[str, Any]:
    predicate = payload["predicate"]
    signals = predicate.get("signals", {})
    drive = predicate.get("drive_validity", {})
    compression = drive.get("compression_series", [])
    final_compression = compression[-1] if compression else {}
    mesh_quality = predicate.get("mesh_quality_min", {})
    conditioning = mesh_quality.get("conditioning_observability", {})
    missing_signals = predicate.get("missing_signals", [])
    analysis_errors = predicate.get("analysis_errors", [])
    harness_clean = (
        payload["exit_code"] == 0
        and not bool(payload["timed_out"])
        and not missing_signals
        and not analysis_errors
    )
    crossing = first_compression_crossing(compression, float(payload["c_star"]))
    return {
        "run_id": payload["run_id"],
        "case_label": payload["case_label"],
        "p0_dyn_cm2": payload["p0_dyn_cm2"],
        "p0_mbar": payload["p0_mbar"],
        "c_star": payload["c_star"],
        "t_end_s": payload["t_end_s"],
        "result": payload["result"],
        "admissible": predicate["passed"],
        "deck_contract_valid": predicate.get("deck_contract_valid"),
        "geometry_admissible": predicate.get("geometry_admissible"),
        "drive_validity_passed": predicate.get("drive_validity_passed"),
        "completion_escape_fail_closed_passed": predicate.get("completion_escape_fail_closed_passed"),
        "exit_code": payload["exit_code"],
        "timed_out": payload["timed_out"],
        "wall_time_s": payload["wall_time_s"],
        "harness_clean": harness_clean,
        "predicate_summary_json": str(summary_path),
        "missing_signals": missing_signals,
        "failed_signals": predicate.get("failed_signals", []),
        "analysis_errors": analysis_errors,
        "final_t_s": predicate.get("completion_escape_fail_closed", {}).get("final_t_s"),
        "termination_reason": predicate.get("completion_escape_fail_closed", {}).get("termination_reason"),
        "min_corner_j_rel": signals.get("min_corner_j_rel", {}).get("value"),
        "min_gauss_j_rel": signals.get("min_gauss_j_rel", {}).get("value"),
        "min_rz_volume_rel": signals.get("min_rz_volume_rel", {}).get("value"),
        "negative_rz_volume_count_total": signals.get("negative_rz_volume_count_total", {}).get("value"),
        "achieved_min_edge_length_rel": conditioning.get("achieved_min_edge_length_rel"),
        "achieved_min_altitude_rel": conditioning.get("achieved_min_altitude_rel"),
        "achieved_max_condition_number": conditioning.get("achieved_max_condition_number"),
        "mesh_geometry_failures_observed_count": signals.get("mesh_geometry_failures_observed_count", {}).get("value"),
        "escape_valve_counters_all_zero": signals.get("escape_valve_counters_all_zero", {}).get("passed"),
        "C_meas_final": final_compression.get("C_meas"),
        "C_meas_peak": drive.get("compression_measurement", {}).get("C_meas_peak"),
        "C_required": drive.get("compression_measurement", {}).get("C_required"),
        "peak_kinetic_energy_erg": drive.get("peak_kinetic_energy_erg"),
        "W_drive_erg": drive.get("drive_work", {}).get("W_drive_erg"),
        "W_drive_positive_increments_erg": drive.get("drive_work", {}).get("W_drive_positive_increments_erg"),
        "calibration_crossing": crossing,
    }


def run_one(args: argparse.Namespace, case: PabCase, label: str) -> dict[str, Any]:
    stem = f"{label}_{compact_case_label(case)}_nr{args.nr}_nz{args.nz}_seed{args.seed}"
    run_id, outdir = lite.allocate_run_dir(Path(args.out_root), stem)
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    env_overrides = active_env(args, outdir, case)
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
        case=case,
    )
    result = "ADMISSIBLE" if predicate["passed"] else "INVALID"
    if timed_out:
        result = "WALL_TIME_TIMEOUT"
    payload = {
        "schema_version": "tenryu.i1b_pab_polar_2d_rz_run_predicate.v1",
        "run_id": run_id,
        "case_label": compact_case_label(case),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "p0_dyn_cm2": case.p0_dyn_cm2,
        "p0_mbar": mbar_from_dyn_cm2(case.p0_dyn_cm2),
        "c_star": case.c_star,
        "t_end_s": case.t_end_s,
        "deck_path": args.deck,
        "result": result,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "output_bytes": lite.output_bytes(outdir),
        "predicate": predicate,
        "env_overrides": env_overrides,
    }
    summary_path = outdir / "predicate_summary.json"
    write_json(summary_path, payload)
    return row_from_payload(payload, summary_path)


def calibration_recommendation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed = [
        row
        for row in rows
        if isinstance(row.get("calibration_crossing"), dict)
        and row["calibration_crossing"].get("crossed")
    ]
    clean = [
        row
        for row in observed
        if bool(row.get("harness_clean")) and bool(row.get("completion_escape_fail_closed_passed"))
    ]
    selected = clean[0] if clean else None
    first_observed = observed[0] if observed else None
    return {
        "status": "recommended" if selected is not None else "not_reached_cleanly",
        "recommended_t_end_s": selected["calibration_crossing"]["t_cross_s"] if selected is not None else None,
        "recommended_run_id": selected.get("run_id") if selected is not None else None,
        "recommended_predicate_summary_json": selected.get("predicate_summary_json") if selected is not None else None,
        "first_observed_crossing": first_observed,
        "selection_rule": (
            "first planned calibration run whose plot compression series reaches C_star "
            "and whose completion/escape predicate is clean"
        ),
    }


def calibrate(args: argparse.Namespace) -> dict[str, Any]:
    rows = [run_one(args, case, "calibrate") for case in planned_cases(args)]
    recommendation = calibration_recommendation(rows)
    clean_result = (
        recommendation.get("recommended_t_end_s") is not None
        and all(bool(row.get("harness_clean")) for row in rows)
    )
    summary = {
        "schema_version": "tenryu.i1b_pab_polar_2d_rz_calibration.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "calibration_status": recommendation["status"],
        "clean_result": clean_result,
        "recommendation": recommendation,
        "predicate_definition": predicate_definition(args.ci_c_star),
        "budget": budget_summary(args),
        "runs": rows,
    }
    write_json(Path(args.summary_json), summary)
    return summary


def ci(args: argparse.Namespace) -> dict[str, Any]:
    case = planned_cases(args)[0]
    row = run_one(args, case, "ci")
    summary = {
        "schema_version": "tenryu.i1b_pab_polar_2d_rz_ci.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "ci_status": "passed" if bool(row.get("admissible")) else "failed",
        "clean_result": bool(row.get("harness_clean")) and bool(row.get("admissible")),
        "gate_point": {
            "name": "I1B-PAB-POLAR-C8-P100Mbar",
            "p0_dyn_cm2": case.p0_dyn_cm2,
            "p0_mbar": mbar_from_dyn_cm2(case.p0_dyn_cm2),
            "c_star": case.c_star,
            "t_end_s": case.t_end_s,
        },
        "predicate_definition": predicate_definition(case.c_star),
        "budget": budget_summary(args),
        "run": row,
    }
    write_json(Path(args.summary_json), summary)
    return summary


def ladder(args: argparse.Namespace) -> dict[str, Any]:
    rows = [run_one(args, case, "ladder") for case in planned_cases(args)]
    admissible = [row for row in rows if bool(row.get("admissible"))]
    max_admissible = max(admissible, key=lambda row: float(row["p0_dyn_cm2"])) if admissible else None
    clean_result = all(bool(row.get("harness_clean")) for row in rows)
    summary = {
        "schema_version": "tenryu.i1b_pab_polar_2d_rz_ladder.v1",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "mode": MODE,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "ladder_status": "admissible_found" if max_admissible is not None else "none_admissible",
        "clean_result": clean_result,
        "max_admissible_p0": max_admissible,
        "predicate_definition": predicate_definition(args.ci_c_star),
        "budget": budget_summary(args),
        "p0_grid_dyn_cm2": list(args.p0_grid),
        "p0_grid_mbar": [mbar_from_dyn_cm2(item) for item in args.p0_grid],
        "runs": sorted(rows, key=lambda row: float(row["p0_dyn_cm2"])),
    }
    write_json(Path(args.summary_json), summary)
    return summary


def predicate_definition(c_star: float) -> dict[str, Any]:
    return {
        "deck_contract": {
            "header": "[deck:2d_rz_i1b_pab_polar]",
            "i1b_pab_polar_mode": True,
            "P0_units": "dyn_per_cm2",
            "S_max_units": "cm",
            "mesh_quality_min_diag": True,
            "conservative_remap_enabled": False,
            "operator_semantics": "laser_off_radiation_off_conduction_off_pressure_drive",
        },
        "geometry_source": MESH_QUALITY_GROUP,
        "min_corner_j_rel": f"> {lite.ADMISSIBILITY_FLOOR_REL:.1e}",
        "min_gauss_j_rel": f"> {lite.ADMISSIBILITY_FLOOR_REL:.1e}",
        "min_rz_volume_rel": "> 0.0",
        "negative_rz_volume_count_total": 0,
        "reported_not_gated": [
            "achieved_min_edge_length_rel",
            "achieved_min_altitude_rel",
            "achieved_max_condition_number",
        ],
        "reached_t_end": True,
        "termination_reason": "t_end_reached",
        "final_t_margin": "final_t >= 0.999 * t_end",
        "dt_floor_abort": "fail: termination_reason must be t_end_reached",
        "escape_valve_counters_all_zero": True,
        "mesh_geometry_failures_observed_count": 0,
        "C_meas_final": f">= {DRIVE_C_FRACTION_FLOOR:.1f} * C_star = {DRIVE_C_FRACTION_FLOOR * c_star:.6g}",
        "drive_energy_nontrivial": (
            f"peak_kinetic_energy_erg > {DRIVE_ENERGY_FLOOR_ERG:.1e} "
            f"OR W_drive_erg > {DRIVE_ENERGY_FLOOR_ERG:.1e}"
        ),
        "missing_required_signal_policy": "fail_closed_INVALID",
    }


def budget_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "nr": args.nr,
        "nz": args.nz,
        "r_outer_cm": args.r_outer_cm,
        "rho0_gcc": args.rho0_gcc,
        "dt_initial_s": args.dt_initial_s,
        "dt_max_s": args.dt_max_s,
        "ale_every_n_steps": args.ale_every_n_steps,
        "plot_every_steps": args.plot_every_steps,
        "history_every_steps": args.history_every_steps,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }


def dry_run_summary(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    for case in planned_cases(args):
        planned_outdir = Path(args.out_root) / (
            f"planned_{compact_case_label(case)}_nr{args.nr}_nz{args.nz}_seed{args.seed}"
        )
        rows.append(
            {
                "case_label": compact_case_label(case),
                "p0_dyn_cm2": case.p0_dyn_cm2,
                "p0_mbar": mbar_from_dyn_cm2(case.p0_dyn_cm2),
                "c_star": case.c_star,
                "t_end_s": case.t_end_s,
                "planned_outdir": str(planned_outdir),
                "env_overrides": active_env(args, planned_outdir, case),
                "predicate_result": "not_evaluated_dry_run",
                "planned": True,
            }
        )
    summary = {
        "schema_version": "tenryu.i1b_pab_polar_2d_rz_dry_run.v1",
        "mode": args.mode,
        "stage_id": STAGE_ID,
        "deck_path": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "predicate_definition": predicate_definition(args.ci_c_star),
        "budget": budget_summary(args),
        "planned_runs": rows,
    }
    write_json(Path(args.summary_json), summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("calibrate", "ci", "ladder"), default="ci")
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_i1b_pab_polar.py")
    parser.add_argument("--out-root", default="./build/i1b_pab_polar_2d_rz_runs")
    parser.add_argument("--summary-json", default="./build/i1b_pab_polar_2d_rz_summary.json")
    parser.add_argument("--ci-p0-dyn-cm2", type=float, default=DEFAULT_P0_DYN_CM2)
    parser.add_argument("--ci-c-star", type=float, default=DEFAULT_C_STAR)
    parser.add_argument("--p0-grid", type=parse_float_list, default=list(DEFAULT_LADDER_P0_GRID))
    parser.add_argument("--calibrate-t-end-grid", type=parse_float_list, default=None)
    parser.add_argument("--nr", type=int, default=DEFAULT_NR)
    parser.add_argument("--nz", type=int, default=DEFAULT_NZ)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--r-outer-cm", type=float, default=R_OUTER_DEFAULT_CM)
    parser.add_argument("--rho0-gcc", type=float, default=DEFAULT_RHO0_GCC)
    parser.add_argument("--t-end-s", type=float, default=DEFAULT_T_END_S)
    parser.add_argument("--dt-initial-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--dt-max-s", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--ale-every-n-steps", type=int, default=DEFAULT_ALE_EVERY_N_STEPS)
    parser.add_argument("--plot-every-steps", type=int, default=DEFAULT_PLOT_EVERY_STEPS)
    parser.add_argument("--history-every-steps", type=int, default=DEFAULT_HISTORY_EVERY_STEPS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without invoking TENRYU")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.nr < 4 or args.nz < 4:
        raise ValueError("--nr and --nz must both be >= 4")
    if args.r_outer_cm <= 0.0 or not math.isfinite(args.r_outer_cm):
        raise ValueError("--r-outer-cm must be finite and positive")
    if args.rho0_gcc <= 0.0 or not math.isfinite(args.rho0_gcc):
        raise ValueError("--rho0-gcc must be finite and positive")
    if args.ci_p0_dyn_cm2 <= 0.0 or not math.isfinite(args.ci_p0_dyn_cm2):
        raise ValueError("--ci-p0-dyn-cm2 must be finite and positive")
    if args.ci_c_star <= 1.0 or not math.isfinite(args.ci_c_star):
        raise ValueError("--ci-c-star must be finite and greater than 1")
    if args.t_end_s <= 0.0 or not math.isfinite(args.t_end_s):
        raise ValueError("--t-end-s must be finite and positive")
    if args.calibrate_t_end_grid is not None and any(
        value <= 0.0 or not math.isfinite(value) for value in args.calibrate_t_end_grid
    ):
        raise ValueError("--calibrate-t-end-grid values must be finite and positive")
    if any(value <= 0.0 or not math.isfinite(value) for value in args.p0_grid):
        raise ValueError("--p0-grid values must be finite and positive")
    if args.dt_initial_s <= 0.0 or args.dt_max_s <= 0.0:
        raise ValueError("time-step arguments must be positive")
    if args.ale_every_n_steps <= 0:
        raise ValueError("--ale-every-n-steps must be positive")
    if args.plot_every_steps <= 0 or args.history_every_steps <= 0:
        raise ValueError("--plot-every-steps and --history-every-steps must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.wall_time_s is not None and args.wall_time_s <= 0.0:
        raise ValueError("--wall-time-s must be positive when set")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if args.dry_run:
        summary = dry_run_summary(args)
        print(json.dumps(json_sanitize(summary), indent=2, sort_keys=True))
        return 0
    if args.mode == "calibrate":
        summary = calibrate(args)
    elif args.mode == "ci":
        summary = ci(args)
    else:
        summary = ladder(args)
    print(json.dumps(json_sanitize(summary), indent=2, sort_keys=True))
    return 0 if summary.get("clean_result") else 2


if __name__ == "__main__":
    raise SystemExit(main())
