#!/usr/bin/env python3
"""Run and analyze the I1 FLD constant-Eddington nED grey shock gate."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_i1_le08 as base


STAGE_ID = "I1"
LADDER_ROW = "i1_fld_ced_grey_radshock"
NED_TEMPERATURE_SEPARATION_LINF_GATE = 0.2
SHOCK_WINDOWED_L2_GATE = 0.10
CONVOLVED_PEAK_GAP_GATE = 0.15
RICHARDSON_DINF_LOW = 0.65
RICHARDSON_DINF_HIGH = 0.87
PEAK_GAP_CONTRACTION_RATIO_GATE = 0.5
PEAK_GAP_CONTRACTION_ABS_FLOOR = 0.02
PEAK_GAP_CONTRACTION_COARSE_NR = 512
PEAK_GAP_CONTRACTION_FINE_NR = 1024
REFERENCE_DTR_PEAK = 0.7614
HYDRO_KERNEL_HALF_WIDTH_CM = 1.0
SHOCK_WINDOW_MFP_COUNT = 10


def case_name(mach: float, nr: int, nz: int, seed: int, init_mode: str, dimension: str) -> str:
    if dimension == "1D_SPH":
        return (
            f"i1_fld_ced_grey_radshock_M{base.safe_float_token(mach)}"
            f"_nr{nr}_seed{seed}_{init_mode}"
        )
    return (
        f"i1_fld_ced_grey_radshock_M{base.safe_float_token(mach)}"
        f"_nr{nr}_nz{nz}_seed{seed}_{init_mode}"
    )


def active_env(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    outdir: Path,
    *,
    t_end_s: float | None = None,
    max_steps: int | None = None,
    dt_initial_s: float | None = None,
    dt_max_s: float | None = None,
) -> dict[str, str]:
    env = {
        "TENRYU_I1_FLD_CED_M0": str(args.mach),
        "TENRYU_I1_FLD_CED_NR": str(nr),
        "TENRYU_I1_FLD_CED_SEED": str(args.seed),
        "TENRYU_I1_FLD_CED_OUTDIR": str(outdir),
        "TENRYU_I1_FLD_CED_REFERENCE": str(args.reference),
        "TENRYU_I1_FLD_CED_INIT_MODE": args.init_mode,
        "TENRYU_I1_FLD_CED_MAX_STEPS": str(max_steps if max_steps is not None else args.max_steps),
        "TENRYU_I1_FLD_CED_HISTORY_EVERY_STEPS": "1",
        "TENRYU_I1_FLD_CED_VERBOSITY": args.verbosity,
    }
    if dimension == "2D_RZ":
        env["TENRYU_I1_FLD_CED_NZ"] = str(nz)
        env["TENRYU_I1_FLD_CED_MESH_MOTION"] = args.mesh_motion
    effective_t_end_s = t_end_s if t_end_s is not None else args.t_end_s
    effective_dt_initial_s = dt_initial_s if dt_initial_s is not None else args.dt_initial_s
    effective_dt_max_s = dt_max_s if dt_max_s is not None else args.dt_max_s
    if effective_t_end_s is not None:
        env["TENRYU_I1_FLD_CED_T_END_S"] = str(effective_t_end_s)
    if effective_dt_initial_s is not None:
        env["TENRYU_I1_FLD_CED_DT_INITIAL_S"] = str(effective_dt_initial_s)
    if effective_dt_max_s is not None:
        env["TENRYU_I1_FLD_CED_DT_MAX_S"] = str(effective_dt_max_s)
    return env


def trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def detect_shock_front_index(Te: np.ndarray, x: np.ndarray) -> int:
    """Locate the cell with maximum |dT_e/dx| using centered finite differences."""
    Te_arr = np.asarray(Te, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    mask = np.isfinite(Te_arr) & np.isfinite(x_arr)
    if np.count_nonzero(mask) < 3:
        finite = np.nonzero(mask)[0]
        return int(finite[0]) if finite.size else 0
    local_indices = np.nonzero(mask)[0]
    grad = np.gradient(Te_arr[mask], x_arr[mask])
    return int(local_indices[int(np.nanargmax(np.abs(grad)))])


def shock_front_position(Te: np.ndarray, x: np.ndarray) -> float:
    """Return x at maximum |dT_e/dx|."""
    idx = detect_shock_front_index(Te, x)
    return float(np.asarray(x, dtype=float)[idx])


def sorted_unique_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if x_arr.size == 0:
        return x_arr, y_arr
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    unique = np.concatenate(([True], np.diff(x_arr) > 0.0))
    return x_arr[unique], y_arr[unique]


def normalize_kernel(x_local: np.ndarray, kernel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_arr, k_arr = sorted_unique_xy(x_local, np.maximum(np.asarray(kernel, dtype=float), 0.0))
    if x_arr.size < 2:
        return x_arr, k_arr
    integral = trapezoid(k_arr, x_arr)
    if not (math.isfinite(integral) and integral > 0.0):
        k_arr = np.ones_like(x_arr)
        integral = trapezoid(k_arr, x_arr)
    return x_arr, k_arr / max(integral, base.E_FLOOR)


def measure_hydro_kernel(
    Te: np.ndarray,
    x: np.ndarray,
    x_shock: float,
    half_width_cm: float = HYDRO_KERNEL_HALF_WIDTH_CM,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (x_relative, normalized_kernel) for the measured TENRYU shock-front T_e profile."""
    x_arr = np.asarray(x, dtype=float)
    Te_arr = np.asarray(Te, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(Te_arr)
    if np.count_nonzero(finite) < 3:
        return np.asarray([0.0], dtype=float), np.asarray([1.0], dtype=float)

    mask = finite & (np.abs(x_arr - float(x_shock)) < float(half_width_cm))
    if np.count_nonzero(mask) < 3:
        candidates = np.nonzero(finite)[0]
        nearest = candidates[np.argsort(np.abs(x_arr[candidates] - float(x_shock)))[:3]]
        mask = np.zeros_like(finite, dtype=bool)
        mask[np.sort(nearest)] = True

    x_local = x_arr[mask] - float(x_shock)
    Te_local = Te_arr[mask]
    x_local, Te_local = sorted_unique_xy(x_local, Te_local)
    if x_local.size < 3:
        return normalize_kernel(x_local, np.ones_like(x_local))
    kernel = np.abs(np.gradient(Te_local, x_local))
    return normalize_kernel(x_local, kernel)


def cell_edges_from_centers(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    if x_arr.size < 2:
        return np.asarray([x_arr[0] - 0.5, x_arr[0] + 0.5], dtype=float)
    edges = np.empty(x_arr.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_arr[:-1] + x_arr[1:])
    edges[0] = x_arr[0] - 0.5 * (x_arr[1] - x_arr[0])
    edges[-1] = x_arr[-1] + 0.5 * (x_arr[-1] - x_arr[-2])
    return edges


def convolved_reference_at(
    ref_x: np.ndarray,
    ref_dTr: np.ndarray,
    kernel_x: np.ndarray,
    kernel: np.ndarray,
    query_x: np.ndarray,
) -> np.ndarray:
    query = np.asarray(query_x, dtype=float)
    out = np.full(query.shape, np.nan, dtype=float)
    ref_x_arr, ref_dtr_arr = sorted_unique_xy(ref_x, ref_dTr)
    kernel_x_arr, kernel_arr = normalize_kernel(kernel_x, kernel)
    if ref_x_arr.size < 2 or kernel_x_arr.size < 2:
        return np.interp(query, ref_x_arr, ref_dtr_arr, left=np.nan, right=np.nan)

    for start in range(0, query.size, 128):
        stop = min(start + 128, query.size)
        q = query[start:stop]
        offsets = q[:, None] - ref_x_arr[None, :]
        weights = np.interp(offsets, kernel_x_arr, kernel_arr, left=0.0, right=0.0)
        out[start:stop] = np.trapz(ref_dtr_arr[None, :] * weights, ref_x_arr, axis=1)
    return out


def cell_averaged_convolved_reference(
    ref_x: np.ndarray,
    ref_Tr: np.ndarray,
    ref_Te: np.ndarray,
    kernel_x: np.ndarray,
    kernel: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    """Convolve reference T_rad/T_e profile with the measured hydro kernel, then cell-average."""
    target = np.asarray(target_x, dtype=float)
    ref_dTr = np.asarray(ref_Tr, dtype=float) / np.maximum(np.asarray(ref_Te, dtype=float), 1.0e-30) - 1.0
    if target.size < 2:
        return convolved_reference_at(ref_x, ref_dTr, kernel_x, kernel, target)
    edges = cell_edges_from_centers(target)
    left = convolved_reference_at(ref_x, ref_dTr, kernel_x, kernel, edges[:-1])
    center = convolved_reference_at(ref_x, ref_dTr, kernel_x, kernel, target)
    right = convolved_reference_at(ref_x, ref_dTr, kernel_x, kernel, edges[1:])
    return (left + 4.0 * center + right) / 6.0


def shock_windowed_l2(
    code_dTr: np.ndarray,
    ref_dTr_conv: np.ndarray,
    x: np.ndarray,
    x_shock: float,
    mfp_post_cm: float,
    N: int = SHOCK_WINDOW_MFP_COUNT,
) -> dict[str, Any]:
    """Compute L2 of (code_dTr - ref_dTr_conv) within +/- N*mfp_post of the shock front."""
    x_arr = np.asarray(x, dtype=float)
    code = np.asarray(code_dTr, dtype=float)
    ref_conv = np.asarray(ref_dTr_conv, dtype=float)
    window_half = float(N) * float(mfp_post_cm)
    mask = (
        np.isfinite(x_arr)
        & np.isfinite(code)
        & np.isfinite(ref_conv)
        & (np.abs(x_arr - float(x_shock)) < window_half)
    )
    if np.count_nonzero(mask) < 3:
        return {
            "L2_abs": math.inf,
            "L2_norm": math.inf,
            "max_dTr_in_window": 0.0,
            "window_half_cm": window_half,
            "n_cells_in_window": int(np.count_nonzero(mask)),
        }

    diff_sq = (code[mask] - ref_conv[mask]) ** 2
    L2_abs = float(np.sqrt(np.mean(diff_sq)))
    return {
        "L2_abs": L2_abs,
        "L2_norm": L2_abs / REFERENCE_DTR_PEAK,
        "max_dTr_in_window": float(np.max(np.abs(code[mask]))),
        "window_half_cm": window_half,
        "n_cells_in_window": int(np.count_nonzero(mask)),
    }


def convolved_peak_gap(code_dTr_max: float, ref_dTr_conv_max: float) -> float:
    """Return relative peak gap |code - conv_ref| / conv_ref."""
    if ref_dTr_conv_max <= 1.0e-30:
        return math.inf
    return abs(float(code_dTr_max) - float(ref_dTr_conv_max)) / float(ref_dTr_conv_max)


def kernel_width_cm(kernel_x: np.ndarray, kernel: np.ndarray) -> float:
    x_arr, k_arr = normalize_kernel(kernel_x, kernel)
    if x_arr.size < 2:
        return 0.0
    second = trapezoid((x_arr**2) * k_arr, x_arr)
    return math.sqrt(max(second, 0.0)) if math.isfinite(second) else math.inf


def measured_postshock_mfp_cm(
    rho: np.ndarray,
    x: np.ndarray,
    x_shock: float,
    kappa_R_cm2_per_g: float,
) -> float:
    rho_arr = np.asarray(rho, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    dx = base.median_cell_spacing(x_arr)
    post_width = max(5.0 * dx, base.E_FLOOR)
    mask = (
        np.isfinite(rho_arr)
        & np.isfinite(x_arr)
        & (rho_arr > 0.0)
        & (x_arr > float(x_shock))
        & (x_arr < float(x_shock) + post_width)
    )
    if np.count_nonzero(mask) == 0:
        idx = int(np.nanargmin(np.abs(x_arr - float(x_shock))))
        rho_post = float(rho_arr[idx]) if math.isfinite(float(rho_arr[idx])) else math.nan
    else:
        rho_post = float(np.mean(rho_arr[mask]))
    if not (math.isfinite(rho_post) and rho_post > 0.0):
        return math.inf
    return 1.0 / max(float(kappa_R_cm2_per_g) * rho_post, base.E_FLOOR)


def wave12_snapshot_metrics(
    snapshot: base.Snapshot,
    ref: dict[str, Any],
    snapshot_index: int,
) -> dict[str, Any]:
    coord = np.asarray(base.profile_centers(snapshot), dtype=float)
    raw_x = np.asarray(base.reference_frame_coord(snapshot, ref), dtype=float)
    Te = np.asarray(base.profile_average(snapshot, snapshot.Te), dtype=float)
    rho = np.asarray(base.profile_average(snapshot, snapshot.rho), dtype=float)
    Trad = np.power(np.maximum(base.profile_average(snapshot, snapshot.rad_E) / base.A_EV, 0.0), 0.25)

    shock_coord = shock_front_position(Te, coord)
    shock_ref_frame = float(np.interp(shock_coord, coord, raw_x))
    ref_shock = base.reference_shock_position(ref)
    aligned_x = raw_x - (shock_ref_frame - ref_shock)
    code_dTr = np.asarray(Trad, dtype=float) / np.maximum(Te, base.E_FLOOR) - 1.0

    kernel_x, kernel = measure_hydro_kernel(Te, aligned_x, ref_shock)
    ref_conv = cell_averaged_convolved_reference(
        np.asarray(ref["table"]["x_cm"], dtype=float),
        np.asarray(ref["table"]["T_rad_eV"], dtype=float),
        np.asarray(ref["table"]["T_eV"], dtype=float),
        kernel_x,
        kernel,
        aligned_x,
    )
    mfp_post_cm = measured_postshock_mfp_cm(
        rho,
        aligned_x,
        ref_shock,
        float(ref["parameters"]["kappa_R_cm2_per_g"]),
    )
    l2 = shock_windowed_l2(code_dTr, ref_conv, aligned_x, ref_shock, mfp_post_cm)
    window_half = float(l2["window_half_cm"])
    window = (
        np.isfinite(aligned_x)
        & np.isfinite(ref_conv)
        & (np.abs(aligned_x - ref_shock) < window_half)
    )
    ref_peak = float(np.max(np.abs(ref_conv[window]))) if np.count_nonzero(window) else 0.0
    code_peak = float(l2["max_dTr_in_window"])

    return {
        "snapshot_index": int(snapshot_index),
        "time_s": float(snapshot.t),
        "step": int(snapshot.step),
        "shock_windowed_l2_abs": float(l2["L2_abs"]),
        "shock_windowed_l2_norm": float(l2["L2_norm"]),
        "max_dTr_in_window": code_peak,
        "convolved_ref_dTr_max": ref_peak,
        "convolved_peak_gap": convolved_peak_gap(code_peak, ref_peak),
        "shock_front_position_cm": shock_coord,
        "shock_front_reference_frame_cm": shock_ref_frame,
        "mfp_post_cm": float(mfp_post_cm),
        "window_half_cm": window_half,
        "n_cells_in_window": int(l2["n_cells_in_window"]),
        "measured_hydro_kernel_width_cm": kernel_width_cm(kernel_x, kernel),
        "measured_hydro_kernel_cell_count": int(np.asarray(kernel_x).size),
    }


def wave12_profile_metrics(snapshots: list[base.Snapshot], ref: dict[str, Any]) -> dict[str, Any]:
    if not snapshots:
        return {}
    steady = base.locate_steady_start_index(snapshots)
    start_index = int(steady.get("steady_start_index", len(snapshots) - 1))
    start_index = max(0, min(start_index, len(snapshots) - 1))
    rows = [
        wave12_snapshot_metrics(snapshot, ref, index)
        for index, snapshot in enumerate(snapshots[start_index:], start=start_index)
    ]
    finite_rows = [
        row
        for row in rows
        if math.isfinite(float(row.get("shock_windowed_l2_abs", math.inf)))
        and math.isfinite(float(row.get("convolved_peak_gap", math.inf)))
    ]
    if not finite_rows:
        return {"wave12_snapshot_metrics": rows, "wave12_steady_start_index": start_index}

    l2_row = max(finite_rows, key=lambda row: float(row["shock_windowed_l2_abs"]))
    peak_row = max(finite_rows, key=lambda row: float(row["max_dTr_in_window"]))
    return {
        "wave12_snapshot_metrics": rows,
        "wave12_steady_start_index": start_index,
        "wave12_steady_start_method": steady.get("steady_start_method"),
        "wave12_snapshot_count": len(rows),
        "wave12_l2_snapshot_index": int(l2_row["snapshot_index"]),
        "wave12_peak_snapshot_index": int(peak_row["snapshot_index"]),
        "wave12_shock_windowed_l2_abs": float(l2_row["shock_windowed_l2_abs"]),
        "wave12_shock_windowed_l2_norm": float(l2_row["shock_windowed_l2_norm"]),
        "wave12_max_dTr_in_window": float(peak_row["max_dTr_in_window"]),
        "wave12_convolved_ref_dTr_max": float(peak_row["convolved_ref_dTr_max"]),
        "wave12_convolved_peak_gap": float(peak_row["convolved_peak_gap"]),
        "wave12_shock_front_position_cm": float(peak_row["shock_front_position_cm"]),
        "wave12_shock_front_reference_frame_cm": float(peak_row["shock_front_reference_frame_cm"]),
        "wave12_mfp_post_cm": float(peak_row["mfp_post_cm"]),
        "wave12_measured_hydro_kernel_width_cm": float(peak_row["measured_hydro_kernel_width_cm"]),
        "wave12_measured_hydro_kernel_cell_count": int(peak_row["measured_hydro_kernel_cell_count"]),
    }


def run_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    *,
    run_label: str = "production",
    t_end_s: float | None = None,
    max_steps: int | None = None,
    dt_initial_s: float | None = None,
    dt_max_s: float | None = None,
) -> dict[str, Any]:
    metrics = base.run_one(
        args,
        nr,
        nz,
        dimension,
        ref,
        run_label=run_label,
        t_end_s=t_end_s,
        max_steps=max_steps,
        dt_initial_s=dt_initial_s,
        dt_max_s=dt_max_s,
    )
    if metrics.get("exit_code") != 0 or metrics.get("timed_out") or metrics.get("run_status") == "analysis_failure":
        return metrics
    try:
        outdir = Path(args.out_root) / str(metrics["run_id"])
        results_dir = base.discover_results_dir(outdir)
        snapshots = base.read_snapshots(results_dir, str(metrics["case_name"]), nr, nz, dimension)
        metrics.update(wave12_profile_metrics(snapshots, ref))
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["wave12_analysis_error"] = str(exc)
    return metrics


def richardson_d_values(rows: list[dict[str, Any]]) -> dict[int, float]:
    out: dict[int, float] = {}
    for row in rows:
        try:
            nr = int(row.get("nr"))
        except (TypeError, ValueError):
            continue
        closure = row.get("closure_defects", {})
        value = closure.get("max_abs_delta_Tr_interior", row.get("max_abs_delta_Tr_interior", math.inf))
        try:
            out[nr] = float(value)
        except (TypeError, ValueError):
            pass
    return out


def richardson_d_infinity_fit(rows: list[dict[str, Any]], ref: dict[str, Any]) -> float:
    d_by_nr = richardson_d_values(rows)
    required = [256, 512, 1024]
    if not all(nr in d_by_nr and math.isfinite(d_by_nr[nr]) for nr in required):
        return math.inf
    ref_x = np.asarray(ref["table"]["x_cm"], dtype=float)
    span = float(ref_x[-1] - ref_x[0])
    x_coarse = span / float(required[0])
    x_fine = span / float(required[-1])
    d_coarse = d_by_nr[required[0]]
    d_fine = d_by_nr[required[-1]]
    denom = d_coarse * x_coarse - d_fine * x_fine
    if abs(denom) <= base.E_FLOOR:
        return math.inf
    C = (d_fine - d_coarse) / denom
    D_inf = d_fine * (1.0 + C * x_fine)
    return float(D_inf) if math.isfinite(D_inf) else math.inf


def aggregate(rows: list[dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    out = base.aggregate(rows, ref)
    gates = dict(out.get("gates", {}))
    fine = rows[-1] if rows else {}
    fine_l2_abs = float(fine.get("wave12_shock_windowed_l2_abs", math.inf))
    fine_l2_norm = float(fine.get("wave12_shock_windowed_l2_norm", math.inf))
    fine_conv_ref_peak = float(fine.get("wave12_convolved_ref_dTr_max", math.inf))
    fine_peak_gap = float(fine.get("wave12_convolved_peak_gap", math.inf))
    richardson_d_inf = richardson_d_infinity_fit(rows, ref)
    gates["gate6_shock_windowed_l2_abs"] = fine_l2_abs <= SHOCK_WINDOWED_L2_GATE
    gates["gate7_convolved_peak_gap"] = fine_peak_gap <= CONVOLVED_PEAK_GAP_GATE
    gates["gate8_richardson_d_infinity_fit"] = (
        richardson_d_inf >= RICHARDSON_DINF_LOW and richardson_d_inf <= RICHARDSON_DINF_HIGH
    )
    peak_gap_by_nr: dict[int, float] = {}
    for row in rows:
        try:
            nr = int(row.get("nr"))
        except (TypeError, ValueError):
            continue
        try:
            peak_gap_by_nr[nr] = float(row.get("wave12_convolved_peak_gap", math.inf))
        except (TypeError, ValueError):
            pass
    pg_coarse = peak_gap_by_nr.get(PEAK_GAP_CONTRACTION_COARSE_NR, math.inf)
    pg_fine = peak_gap_by_nr.get(PEAK_GAP_CONTRACTION_FINE_NR, math.inf)
    gates["gate8b_convolved_peak_gap_contraction"] = (
        math.isfinite(pg_coarse)
        and math.isfinite(pg_fine)
        and pg_fine
        <= max(PEAK_GAP_CONTRACTION_RATIO_GATE * pg_coarse, PEAK_GAP_CONTRACTION_ABS_FLOOR)
    )
    out["gates"] = gates
    out["nED_temperature_separation_l_inf_gate"] = NED_TEMPERATURE_SEPARATION_LINF_GATE
    out["nED_temperature_separation_l_inf_status"] = "diagnostic_only_wave12"
    out["fine_shock_windowed_l2_abs"] = fine_l2_abs
    out["fine_shock_windowed_l2_norm"] = fine_l2_norm
    out["fine_convolved_ref_dTr_max"] = fine_conv_ref_peak
    out["fine_convolved_peak_gap"] = fine_peak_gap
    out["richardson_d_infinity_fit"] = richardson_d_inf
    out["richardson_d_infinity_fit_status"] = "diagnostic_only_20260715"
    out["peak_gap_contraction_coarse"] = pg_coarse
    out["peak_gap_contraction_fine"] = pg_fine
    out["fine_shock_front_position_cm"] = float(fine.get("wave12_shock_front_position_cm", math.inf))
    out["fine_measured_hydro_kernel_width_cm"] = float(
        fine.get("wave12_measured_hydro_kernel_width_cm", math.inf)
    )
    out["all_checks_passed"] = all(
        bool(gates.get(name))
        for name in (
            "termination_t_end_reached",
            "gate1_low_prad_only",
            "gate2_reference_conservation",
            "gate6_shock_windowed_l2_abs",
            "gate7_convolved_peak_gap",
            "gate8b_convolved_peak_gap_contraction",
        )
    )
    return out


def reference_identity_branch_wise(ref: dict[str, Any]) -> dict[str, Any]:
    primary = base.le08_identity_stats(
        ref,
        "F_rad_erg_per_cm2_s",
        branch_oriented_flux=False,
    )
    energy = base.le08_identity_stats(
        ref,
        "F_rad_energy_invariant_erg_per_cm2_s",
        branch_oriented_flux=False,
    )
    passed = all(
        primary[branch]["median_rel"] < base.REFERENCE_IDENTITY_MEDIAN_GATE
        and primary[branch]["p95_rel"] < base.REFERENCE_IDENTITY_P95_GATE
        for branch in ("upstream", "downstream")
    )
    return {
        "schema_version": ref.get("schema_version"),
        "schema_v3": base.le08_schema_is_v3(ref),
        "problem": ref.get("problem"),
        "reference_model": ref.get("reference_model"),
        "thresholds": {
            "median_rel_lt": base.REFERENCE_IDENTITY_MEDIAN_GATE,
            "p95_rel_lt": base.REFERENCE_IDENTITY_P95_GATE,
        },
        "passed": passed,
        "upstream": primary["upstream"],
        "downstream": primary["downstream"],
        "total": primary["total"],
        "upstream_median_rel": primary["upstream"]["median_rel"],
        "upstream_p95_rel": primary["upstream"]["p95_rel"],
        "downstream_median_rel": primary["downstream"]["median_rel"],
        "downstream_p95_rel": primary["downstream"]["p95_rel"],
        "total_median_rel": primary["total"]["median_rel"],
        "total_p95_rel": primary["total"]["p95_rel"],
        "energy_invariant_check": energy,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/i1_fld_ced_grey_radshock.py")
    parser.add_argument(
        "--reference",
        default="tests/verification/data/fld_const_eddington_reference/fld_ced_M2.json",
    )
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--grids", type=base.parse_grid_list, default=base.parse_grid_list("256;512;1024"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-mode", choices=("reference_table", "two_state"), default="two_state")
    parser.add_argument("--mesh-motion", choices=("lagrangian", "ale"), default="lagrangian")
    parser.add_argument("--out-root", default="./build/i1_fld_ced_runs")
    parser.add_argument("--summary-json", default="./build/i1_fld_ced_runs/summary.json")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="quiet")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--dump-profiles", action="store_true", help="write per-snapshot profile CSV dumps")
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="deprecated Wave 10/11 compatibility flag; Wave 12 gates remain strict",
    )
    return parser


def install_base_hooks() -> None:
    base.LADDER_ROW = LADDER_ROW
    base.case_name = case_name
    base.active_env = active_env


def main(argv: list[str] | None = None) -> int:
    install_base_hooks()
    parser = build_parser()
    args = parser.parse_args(argv)
    dimension = base.deck_dimension(Path(args.deck))
    try:
        base.validate_grids_for_dimension(args.grids, dimension)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if dimension == "1D_SPH" and args.mesh_motion != "lagrangian":
        parser.error("1D_SPH FLD-CED requires --mesh-motion lagrangian")

    ref = base.read_json(Path(args.reference))
    ref_identity = reference_identity_branch_wise(ref)
    t0_rows = [base.run_t0_audit_one(args, nr, nz, dimension, ref) for nr, nz in args.grids]
    t0_aggregate = base.aggregate_t0_audits(t0_rows)
    rows: list[dict[str, Any]] = []
    quarter_diag: dict[str, Any] | None = None
    if t0_aggregate["passed"]:
        rows = [run_one(args, nr, nz, dimension, ref) for nr, nz in args.grids]
        agg = aggregate(rows, ref)
        fine_nr, fine_nz = args.grids[-1]
        production_t_end = args.t_end_s if args.t_end_s is not None else base.reference_t_end_s(ref, 3.0)
        quarter_row = run_one(
            args,
            fine_nr,
            fine_nz,
            dimension,
            ref,
            run_label="quarter_t_end",
            t_end_s=0.25 * production_t_end,
        )
        quarter_diag = base.quarter_t_end_summary(quarter_row, agg)
    else:
        agg = aggregate(rows, ref)

    payload = {
        "schema_version": "tenryu.i1_fld_ced_grey_radshock.summary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "ladder_row": LADDER_ROW,
        "dimension": dimension,
        "deck_path": args.deck,
        "reference_path": args.reference,
        "reference_parameters": ref["parameters"],
        "reference_identity_branch_wise": ref_identity,
        "t0_admissibility": {
            "rows": t0_rows,
            "aggregate": t0_aggregate,
        },
        "matrix": rows,
        "aggregate": agg,
        "quarter_t_end_diagnostic": quarter_diag,
    }
    base.write_json(Path(args.summary_json), payload)
    print(
        json.dumps(
            {
                "ladder_row": LADDER_ROW,
                "reference_identity_branch_wise": ref_identity,
                "t0_admissibility": t0_aggregate,
                "aggregate": agg,
                "quarter_t_end_diagnostic": quarter_diag,
            },
            indent=2,
            sort_keys=True,
        )
    )
    strict_pass = bool(t0_aggregate["passed"]) and bool(agg["all_checks_passed"])
    partial_pass = bool(args.allow_partial_checkpoint) and strict_pass
    return 0 if strict_pass or partial_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
