#!/usr/bin/env python3
"""Physics-validity critique for TENRYU HDF5 dumps."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DECK_CONFIG: dict[str, dict[str, Any]] = {
    "d4_h3b_sedov_sph": {
        "expected_compression_ratio": 4.0,
        "compression_tolerance": 0.10,
        "qvisc_spike_count": 1,
        "monotonic_rho_required": False,
        "energy_balance_tolerance": 0.005,
    },
    "d6_icf_laser": {
        "expected_compression_ratio": None,
        "qvisc_spike_count": "1+",
        "monotonic_rho_required": False,
        "energy_balance_tolerance": 0.005,
    },
    "l2_laser_hydro": {
        "expected_compression_ratio": None,
        "qvisc_spike_count": "0+",
        "monotonic_rho_required": False,
        "energy_balance_tolerance": 0.005,
    },
    "i1c_grey_fld_powerstress": {
        "expected_compression_ratio": None,
        "qvisc_spike_count": "0+",
        "monotonic_rho_required": False,
        "energy_balance_tolerance": 0.005,
    },
}

RHO_PATHS = ("/state/rho", "/rho", "/hydro/rho")
TE_PATHS = ("/state/Te", "/state/te", "/Te", "/temperature/Te")
UR_PATHS = ("/state/u_r", "/state/ur", "/u_r", "/velocity/u_r")
QVISIC_PATHS = (
    "/diagnostics/qvisc",
    "/diagnostics/Qvisc",
    "/state/qvisc",
    "/hydro/qvisc",
)
XR_PATHS = ("/mesh/x_r", "/mesh/r", "/x_r", "/r")
XZ_PATHS = ("/mesh/x_z", "/mesh/z", "/x_z", "/z")
ENERGY_GROUP_PATHS = ("/global/energy_balance", "/energy_balance", "/global/energy")
TIME_PATHS = ("/time", "/global/time", "/metadata/time")
STEP_PATHS = ("/step", "/global/step", "/metadata/step")


@dataclass
class Snapshot:
    path: Path
    step: int | None
    time: float | None


def warn(message: str) -> None:
    warnings.warn(message, RuntimeWarning, stacklevel=2)


def scalar_or_none(value: Any) -> float | None:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        return float(arr.reshape(-1)[-1])
    except (TypeError, ValueError):
        return None


def read_optional(handle: h5py.File, paths: tuple[str, ...]) -> np.ndarray | None:
    for path in paths:
        if path in handle:
            return np.asarray(handle[path][()])
    return None


def read_scalar_path(handle: h5py.File, paths: tuple[str, ...]) -> float | None:
    data = read_optional(handle, paths)
    if data is None:
        return None
    return scalar_or_none(data)


def infer_step_from_name(path: Path) -> int | None:
    for pattern in (r"_t(\d+)", r"_snapshot_(\d+)", r"_(\d+)$"):
        match = re.search(pattern, path.stem)
        if match:
            return int(match.group(1))
    return None


def discover_hdf5_files(input_dir: Path) -> list[Path]:
    patterns = ("*_t*.h5", "*_snapshot_*.h5", "*.h5")
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted(input_dir.glob(pattern)):
            if path.is_file() and path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files, key=lambda p: (infer_step_from_name(p) is None, infer_step_from_name(p) or 0, p.name))


def load_snapshots(files: list[Path]) -> list[Snapshot]:
    snapshots: list[Snapshot] = []
    for path in files:
        step = infer_step_from_name(path)
        time = None
        try:
            with h5py.File(path, "r") as handle:
                time = read_scalar_path(handle, TIME_PATHS)
                file_step = read_scalar_path(handle, STEP_PATHS)
                if file_step is not None:
                    step = int(file_step)
        except OSError as exc:
            warn(f"Skipping unreadable HDF5 file {path}: {exc}")
            continue
        snapshots.append(Snapshot(path=path, step=step, time=time))
    return sorted(snapshots, key=lambda s: (s.time is None, s.time if s.time is not None else math.inf, s.step or 0, s.path.name))


def mid_z_index(shape: tuple[int, ...]) -> int:
    if len(shape) < 2:
        return 0
    return shape[1] // 2


def as_2d_field(data: np.ndarray, name: str) -> np.ndarray | None:
    arr = np.asarray(data, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    warn(f"Skipping {name}: expected 1D or 2D field, got shape {arr.shape}")
    return None


def radial_profile(field: np.ndarray, name: str) -> np.ndarray | None:
    arr = as_2d_field(field, name)
    if arr is None:
        return None
    return np.asarray(arr[:, mid_z_index(arr.shape)], dtype=float)


def radial_coordinates(handle: h5py.File, n_cells: int, rho_shape: tuple[int, ...]) -> np.ndarray | None:
    x_r = read_optional(handle, XR_PATHS)
    if x_r is None:
        warn("Skipping radial-coordinate dependent analysis: /mesh/x_r not found")
        return None
    arr = np.asarray(x_r, dtype=float).squeeze()
    if arr.ndim == 1:
        if arr.size == n_cells + 1:
            return 0.5 * (arr[:-1] + arr[1:])
        if arr.size == n_cells:
            return arr
        if len(rho_shape) >= 2:
            nr, nz = rho_shape[0], rho_shape[1]
            if arr.size == (nr + 1) * (nz + 1):
                shaped = arr.reshape((nr + 1, nz + 1))
                j = min(mid_z_index(rho_shape), nz)
                return 0.5 * (shaped[:-1, j] + shaped[1:, j])
            if arr.size == nr * nz:
                shaped = arr.reshape((nr, nz))
                return shaped[:, min(mid_z_index(rho_shape), nz - 1)]
    if arr.ndim == 2:
        if arr.shape[0] == n_cells + 1:
            j = min(mid_z_index(rho_shape), arr.shape[1] - 1)
            return 0.5 * (arr[:-1, j] + arr[1:, j])
        if arr.shape[0] == n_cells:
            return arr[:, min(mid_z_index(rho_shape), arr.shape[1] - 1)]
    warn(f"Skipping radial-coordinate dependent analysis: unsupported /mesh/x_r shape {arr.shape}")
    return None


def gradient(r: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.size < 2 or r.size != values.size:
        return np.asarray([], dtype=float)
    return np.gradient(values, r, edge_order=1)


def count_runs(mask: np.ndarray, min_len: int) -> int:
    count = 0
    run = 0
    for item in mask:
        if bool(item):
            run += 1
        else:
            if run >= min_len:
                count += 1
            run = 0
    if run >= min_len:
        count += 1
    return count


def profile_metrics(r: np.ndarray, rho: np.ndarray, monotonic_required: bool) -> dict[str, Any]:
    drho = gradient(r, rho)
    if drho.size == 0:
        return {
            "available": False,
            "reason": "insufficient radial cells",
            "check": "SKIP",
        }
    scale = max(float(np.nanmax(np.abs(drho))), 1.0e-300)
    plateau_mask = np.abs(drho) <= max(1.0e-12 * scale, 1.0e-12 * max(float(np.nanmax(np.abs(rho))), 1.0))
    signs = np.sign(drho)
    signs[np.abs(drho) <= 1.0e-10 * scale] = 0.0
    nonzero = signs[signs != 0.0]
    sign_changes = int(np.count_nonzero(nonzero[1:] * nonzero[:-1] < 0.0)) if nonzero.size > 1 else 0
    plateau_zones = count_runs(plateau_mask, 5)
    passed = not monotonic_required or (sign_changes == 0 and plateau_zones == 0)
    return {
        "available": True,
        "rho_max_dr": float(np.nanmax(np.abs(drho))),
        "plateau_zones": plateau_zones,
        "sign_change_zones": sign_changes,
        "check": "PASS" if passed else "FAIL",
    }


def count_qvisc_spikes(q_profile: np.ndarray) -> int:
    q = np.asarray(q_profile, dtype=float)
    finite = np.isfinite(q)
    if q.size == 0 or not np.any(finite):
        return 0
    q = np.where(finite, q, 0.0)
    qmax = float(np.max(np.abs(q)))
    if qmax <= 0.0:
        return 0
    threshold = max(0.25 * qmax, float(np.mean(np.abs(q)) + 3.0 * np.std(np.abs(q))))
    mask = np.abs(q) >= threshold
    return count_runs(mask, 1)


def shock_metrics(
    handle: h5py.File,
    r: np.ndarray,
    rho: np.ndarray,
    rho_shape: tuple[int, ...],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    drho = gradient(r, rho)
    if drho.size < 3:
        return {"available": False, "reason": "insufficient radial cells", "check": "SKIP"}
    front_idx = int(np.nanargmax(np.abs(drho)))
    pre_start = min(front_idx + 2, rho.size - 1)
    pre_stop = min(pre_start + 5, rho.size)
    post_stop = max(front_idx - 1, 1)
    post_start = max(post_stop - 5, 0)
    rho_0 = float(np.nanmedian(rho[pre_start:pre_stop])) if pre_start < pre_stop else float(rho[-1])
    rho_1 = float(np.nanmedian(rho[post_start:post_stop])) if post_start < post_stop else float(rho[0])
    compression = rho_1 / rho_0 if rho_0 != 0.0 else math.nan

    q_spikes = None
    qvisc = read_optional(handle, QVISIC_PATHS)
    if qvisc is not None:
        q_profile = radial_profile(qvisc, "qvisc")
        if q_profile is not None:
            q_spikes = count_qvisc_spikes(q_profile)
    else:
        warn(f"Skipping Qvisc spike check for {handle.filename}: diagnostics/qvisc not found")

    post_drho = np.abs(gradient(r[post_start:post_stop], rho[post_start:post_stop]))
    full_scale = max(float(np.nanmax(np.abs(drho))), 1.0e-300)
    post_plateau = bool(post_drho.size >= 3 and float(np.nanmax(post_drho)) <= 0.15 * full_scale)

    expected = cfg["expected_compression_ratio"]
    compression_check = "SKIP"
    if expected is not None:
        tolerance = float(cfg["compression_tolerance"])
        compression_check = "PASS" if math.isfinite(compression) and abs(compression - expected) <= tolerance * expected else "FAIL"

    q_check = "SKIP"
    expected_q = cfg["qvisc_spike_count"]
    if q_spikes is not None:
        if isinstance(expected_q, int):
            q_check = "PASS" if q_spikes == expected_q else "FAIL"
        elif expected_q == "1+":
            q_check = "PASS" if q_spikes >= 1 else "FAIL"
        elif expected_q == "0+":
            q_check = "PASS"

    checks = [item for item in (compression_check, q_check) if item != "SKIP"]
    check = "PASS" if checks and all(item == "PASS" for item in checks) else ("SKIP" if not checks else "FAIL")
    return {
        "available": True,
        "position_cm": float(r[front_idx]),
        "front_index": front_idx,
        "rho_0": rho_0,
        "rho_1": rho_1,
        "compression": float(compression),
        "compression_check": compression_check,
        "qvisc_spikes": q_spikes,
        "qvisc_check": q_check,
        "post_shock_plateau": post_plateau,
        "check": check,
    }


def energy_terms(handle: h5py.File) -> dict[str, float] | None:
    group = None
    for path in ENERGY_GROUP_PATHS:
        if path in handle and isinstance(handle[path], h5py.Group):
            group = handle[path]
            break
    if group is None:
        warn(f"Skipping energy balance for {handle.filename}: energy_balance group not found")
        return None

    terms: dict[str, float] = {}
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            value = scalar_or_none(item[()])
            if value is not None and math.isfinite(value):
                terms[key] = value
    if not terms:
        warn(f"Skipping energy balance for {handle.filename}: no scalar energy terms found")
        return None
    return terms


def classify_energy_terms(terms: dict[str, float]) -> tuple[float, float]:
    total = 0.0
    input_energy = 0.0
    total_keys = ("total", "E_total")
    for key, value in terms.items():
        lowered = key.lower()
        if key in total_keys or lowered in ("total", "e_total"):
            total = value
            break
    else:
        for key, value in terms.items():
            lowered = key.lower()
            if "input" in lowered or "source" in lowered:
                input_energy += value
            elif "diss" in lowered or "loss" in lowered or "escape" in lowered:
                input_energy -= value
            elif lowered.startswith("e_"):
                total += value
    if input_energy == 0.0:
        for key, value in terms.items():
            lowered = key.lower()
            if "laser" in lowered or "input" in lowered or "source" in lowered:
                input_energy += value
    return total, input_energy


def energy_metrics(snapshots: list[Snapshot], cfg: dict[str, Any]) -> tuple[dict[Path, dict[str, Any]], str]:
    per_path: dict[Path, dict[str, Any]] = {}
    initial_total = None
    final_check = "SKIP"
    tolerance = float(cfg["energy_balance_tolerance"])
    for snapshot in snapshots:
        try:
            with h5py.File(snapshot.path, "r") as handle:
                terms = energy_terms(handle)
        except OSError as exc:
            warn(f"Skipping energy balance for {snapshot.path}: {exc}")
            terms = None
        if terms is None:
            per_path[snapshot.path] = {"available": False, "check": "SKIP"}
            continue
        total, input_energy = classify_energy_terms(terms)
        if initial_total is None:
            initial_total = total
        residual = total - initial_total - input_energy
        denom = max(abs(input_energy), 1.0e-300)
        rel = abs(residual) / denom
        check = "PASS" if rel < tolerance else "FAIL"
        per_path[snapshot.path] = {
            "available": True,
            "terms": terms,
            "E_total": total,
            "E_input": input_energy,
            "residual": residual,
            "relative_residual": rel,
            "check": check,
        }
        final_check = check
    return per_path, final_check


def select_profile_snapshots(snapshots: list[Snapshot]) -> list[tuple[str, Snapshot]]:
    if len(snapshots) < 2:
        return []
    times = np.asarray(
        [snapshot.time if snapshot.time is not None else float(i) for i, snapshot in enumerate(snapshots)],
        dtype=float,
    )
    t_end = float(times[-1])
    targets = [
        ("t_end_4", 0.25 * t_end),
        ("t_end_2", 0.50 * t_end),
        ("3t_end_4", 0.75 * t_end),
        ("t_end", t_end),
    ]
    selected: list[tuple[str, Snapshot]] = []
    for label, target in targets:
        idx = int(np.argmin(np.abs(times - target)))
        snap = snapshots[idx]
        selected.append((label, snap))
    return selected


def write_profile_plots(deck: str, snapshots: list[Snapshot], output_root: Path) -> list[dict[str, Any]]:
    selected = select_profile_snapshots(snapshots)
    if not selected:
        return []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local environment
        warn(f"Skipping multi-time profile PNGs: matplotlib unavailable ({exc})")
        return []

    output_root.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, Any]] = []
    for label, snapshot in selected:
        try:
            with h5py.File(snapshot.path, "r") as handle:
                rho = read_optional(handle, RHO_PATHS)
                te = read_optional(handle, TE_PATHS)
                ur = read_optional(handle, UR_PATHS)
                if rho is None:
                    warn(f"Skipping profile PNG for {snapshot.path}: rho not found")
                    continue
                rho_profile = radial_profile(rho, "rho")
                if rho_profile is None:
                    continue
                r = radial_coordinates(handle, rho_profile.size, as_2d_field(rho, "rho").shape)
                if r is None:
                    continue
                te_profile = radial_profile(te, "Te") if te is not None else None
                ur_profile = radial_profile(ur, "u_r") if ur is not None else None
        except OSError as exc:
            warn(f"Skipping profile PNG for {snapshot.path}: {exc}")
            continue

        fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True)
        axes[0].plot(r, rho_profile)
        axes[0].set_ylabel("rho [g/cc]")
        axes[1].plot(r, te_profile if te_profile is not None else np.full_like(r, np.nan))
        axes[1].set_ylabel("Te [eV]")
        axes[2].plot(r, ur_profile if ur_profile is not None else np.full_like(r, np.nan))
        axes[2].set_ylabel("u_r [cm/s]")
        axes[2].set_xlabel("r [cm]")
        fig.suptitle(f"{deck} {label}")
        fig.tight_layout()
        out_path = output_root / f"{deck}_{label}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        written.append({"label": label, "snapshot": str(snapshot.path), "path": str(out_path)})
    return written


def verdict_from_details(details: dict[str, str]) -> str:
    checks = [value for value in details.values() if value != "SKIP"]
    if not checks:
        return "PARTIAL"
    if all(value == "PASS" for value in checks):
        return "PASS"
    if any(value == "PASS" for value in checks):
        return "PARTIAL"
    return "FAIL"


def analyze_snapshot(snapshot: Snapshot, cfg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "file": str(snapshot.path),
        "step": snapshot.step,
        "t": snapshot.time,
    }
    try:
        with h5py.File(snapshot.path, "r") as handle:
            rho_data = read_optional(handle, RHO_PATHS)
            if rho_data is None:
                warn(f"Skipping rho and shock analysis for {snapshot.path}: rho not found")
                result["rho_profile"] = {"available": False, "check": "SKIP"}
                result["shock"] = {"available": False, "check": "SKIP"}
                return result
            rho_2d = as_2d_field(rho_data, "rho")
            if rho_2d is None:
                result["rho_profile"] = {"available": False, "check": "SKIP"}
                result["shock"] = {"available": False, "check": "SKIP"}
                return result
            rho = radial_profile(rho_2d, "rho")
            if rho is None:
                result["rho_profile"] = {"available": False, "check": "SKIP"}
                result["shock"] = {"available": False, "check": "SKIP"}
                return result
            r = radial_coordinates(handle, rho.size, rho_2d.shape)
            if r is None:
                result["rho_profile"] = {"available": False, "check": "SKIP"}
                result["shock"] = {"available": False, "check": "SKIP"}
                return result
            monotonic = bool(cfg["monotonic_rho_required"])
            result["rho_profile"] = profile_metrics(r, rho, monotonic)
            result["shock"] = shock_metrics(handle, r, rho, rho_2d.shape, cfg)
    except OSError as exc:
        warn(f"Skipping unreadable HDF5 file {snapshot.path}: {exc}")
        result["rho_profile"] = {"available": False, "check": "SKIP"}
        result["shock"] = {"available": False, "check": "SKIP"}
    return result


def compare_baseline(current: list[dict[str, Any]], baseline_dir: Path) -> dict[str, Any]:
    files = discover_hdf5_files(baseline_dir)
    if not files:
        warn(f"Baseline directory {baseline_dir} contains no HDF5 files")
        return {"available": False, "reason": "no baseline files"}
    return {
        "available": True,
        "baseline_dir": str(baseline_dir),
        "baseline_files": len(files),
        "current_files": len(current),
    }


def print_snapshot_report(deck: str, snapshot: dict[str, Any], energy: dict[str, Any] | None) -> None:
    print(f"## Deck: {deck} (snapshot step={snapshot.get('step')}, t={snapshot.get('t')})")
    rho = snapshot.get("rho_profile", {})
    if rho.get("available"):
        print(
            "- rho profile: max(|drho/dr|)="
            f"{rho['rho_max_dr']:.6e} (plateau zones={rho['plateau_zones']}, "
            f"sign-change={rho['sign_change_zones']})"
        )
    else:
        print("- rho profile: SKIP")

    shock = snapshot.get("shock", {})
    if shock.get("available"):
        qspikes = shock["qvisc_spikes"] if shock["qvisc_spikes"] is not None else "SKIP"
        print(
            "- Shock: position="
            f"{shock['position_cm']:.6e} cm, compression=rho1/rho0={shock['compression']:.6e}, "
            f"Qvisc spikes={qspikes}, post-shock plateau={shock['post_shock_plateau']}"
        )
    else:
        print("- Shock: SKIP")

    if energy and energy.get("available"):
        print(
            "- Energy balance: E_total residual = "
            f"{energy['residual']:.6e} (relative={energy['relative_residual']:.6e})"
        )
    else:
        print("- Energy balance: SKIP")
    print()


def analyze(input_dir: Path, deck: str, output_json: Path | None, baseline: Path | None) -> dict[str, Any]:
    cfg = DECK_CONFIG[deck]
    files = discover_hdf5_files(input_dir)
    if not files:
        print(f"WARNING: no HDF5 files found under {input_dir}")
        summary = {
            "deck": deck,
            "input_dir": str(input_dir),
            "snapshots": [],
            "verdict": "PARTIAL",
            "details": {
                "compression_ratio_check": "SKIP",
                "energy_balance_check": "SKIP",
                "rho_monotonicity_check": "SKIP",
            },
        }
        if output_json is not None:
            output_json.parent.mkdir(parents=True, exist_ok=True)
            output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary

    snapshots = load_snapshots(files)
    energy_by_path, energy_check = energy_metrics(snapshots, cfg)
    snapshot_results = [analyze_snapshot(snapshot, cfg) for snapshot in snapshots]
    profile_dir = output_json.parent if output_json is not None else input_dir
    pngs = write_profile_plots(deck, snapshots, profile_dir)

    for item in snapshot_results:
        print_snapshot_report(deck, item, energy_by_path.get(Path(item["file"])))
    if pngs:
        print("- Multi-time snapshot: t_end/4, t_end/2, 3*t_end/4, t_end -> PNG written")
        for png in pngs:
            print(f"  {png['label']}: {png['path']}")
    elif len(snapshots) < 2:
        print("- Multi-time snapshot: SKIP (single snapshot)")
    else:
        print("- Multi-time snapshot: SKIP")

    compression_checks = [item["shock"].get("compression_check", "SKIP") for item in snapshot_results]
    compression_check = next((item for item in reversed(compression_checks) if item != "SKIP"), "SKIP")
    rho_checks = [item["rho_profile"].get("check", "SKIP") for item in snapshot_results]
    if cfg["monotonic_rho_required"]:
        rho_check = "FAIL" if "FAIL" in rho_checks else ("PASS" if "PASS" in rho_checks else "SKIP")
    else:
        rho_check = "SKIP"
    details = {
        "compression_ratio_check": compression_check,
        "energy_balance_check": energy_check,
        "rho_monotonicity_check": rho_check,
    }
    verdict = verdict_from_details(details)

    summary_snapshots: list[dict[str, Any]] = []
    for item in snapshot_results:
        energy = energy_by_path.get(Path(item["file"]), {})
        rho = item.get("rho_profile", {})
        shock = item.get("shock", {})
        summary_snapshots.append(
            {
                "file": item["file"],
                "step": item["step"],
                "t": item["t"],
                "rho_max_dr": rho.get("rho_max_dr"),
                "plateau_zones": rho.get("plateau_zones"),
                "sign_change_zones": rho.get("sign_change_zones"),
                "compression": shock.get("compression"),
                "qvisc_spikes": shock.get("qvisc_spikes"),
                "energy_residual": energy.get("residual"),
                "energy_relative_residual": energy.get("relative_residual"),
            }
        )

    summary: dict[str, Any] = {
        "deck": deck,
        "input_dir": str(input_dir),
        "snapshots": summary_snapshots,
        "profile_pngs": pngs,
        "verdict": verdict,
        "details": details,
    }
    if baseline is not None:
        summary["baseline"] = compare_baseline(snapshot_results, baseline)
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"## Verdict (deck {deck}): {verdict}")
    expected = cfg["expected_compression_ratio"]
    if expected is not None:
        tol = cfg["compression_tolerance"] * expected
        print(f"- Compression ratio: {details['compression_ratio_check']} (vs analytical {expected}+-{tol})")
    else:
        print(f"- Compression ratio: {details['compression_ratio_check']}")
    print(f"- Energy balance: {details['energy_balance_check']} (vs 0.5%)")
    print(f"- rho monotonicity: {details['rho_monotonicity_check']}")
    return summary


def write_smoke_file(path: Path) -> None:
    nr = 64
    nz = 8
    r_edges = np.linspace(0.0, 1.0, nr + 1)
    x_r = np.repeat(r_edges[:, None], nz + 1, axis=1)
    x_z = np.repeat(np.linspace(-0.1, 0.1, nz + 1)[None, :], nr + 1, axis=0)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    rho_1d = np.where(r_centers < 0.55, 4.0, 1.0)
    rho = np.repeat(rho_1d[:, None], nz, axis=1)
    te = np.repeat((10.0 + 2.0 * rho_1d)[:, None], nz, axis=1)
    ur = np.repeat(np.linspace(1.0e5, 0.0, nr)[:, None], nz, axis=1)
    qvisc = np.zeros((nr, nz), dtype=float)
    qvisc[34, :] = 1.0
    with h5py.File(path, "w") as handle:
        mesh = handle.create_group("mesh")
        mesh.create_dataset("x_r", data=x_r)
        mesh.create_dataset("x_z", data=x_z)
        state = handle.create_group("state")
        state.create_dataset("rho", data=rho)
        state.create_dataset("Te", data=te)
        state.create_dataset("u_r", data=ur)
        diag = handle.create_group("diagnostics")
        diag.create_dataset("qvisc", data=qvisc)
        energy = handle.create_group("global/energy_balance")
        energy.create_dataset("E_int", data=1.0e10)
        energy.create_dataset("E_kin", data=2.0e9)
        energy.create_dataset("E_laser_input", data=0.0)
        handle.create_dataset("time", data=0.0)
        handle.create_dataset("step", data=0)


def run_smoke(deck: str) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        write_smoke_file(root / "smoke_t0000.h5")
        summary = analyze(root, deck, None, None)
    return 0 if summary["snapshots"] else 1


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Directory containing TENRYU HDF5 dumps")
    parser.add_argument("--deck", choices=sorted(DECK_CONFIG), required=True, help="Deck identifier")
    parser.add_argument("--output", type=Path, help="Optional JSON summary path")
    parser.add_argument("--baseline", type=Path, help="Optional baseline HDF5 directory")
    parser.add_argument("--smoke", action="store_true", help="Run against a minimal synthetic HDF5 file")
    args = parser.parse_args(argv)
    if not args.smoke and args.input is None:
        parser.error("--input is required unless --smoke is set")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    warnings.simplefilter("always", RuntimeWarning)
    if args.smoke:
        return run_smoke(args.deck)
    assert args.input is not None
    analyze(args.input, args.deck, args.output, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
