#!/usr/bin/env python3
"""Run Stage L1 2D RZ laser-deposition validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "L1"
MODES = (
    "vacuum_transparent",
    "uniform_absorbing_cylinder",
    "cd_planar_slab",
    "cd_sphere_weak_pulse",
    "l1_conduction_submode",
    "l1_qei_submode",
    "l1_multibeam",
)
FULL_RAYS_BY_MODE: dict[str, tuple[int, ...]] = {
    "vacuum_transparent": (72,),
    "uniform_absorbing_cylinder": (72,),
    "cd_planar_slab": (16, 36, 72, 101),
    "cd_sphere_weak_pulse": (72,),
    "l1_conduction_submode": (72,),
    "l1_qei_submode": (72,),
    "l1_multibeam": (72,),
}
SMOKE_RAYS_BY_MODE: dict[str, tuple[int, ...]] = {mode: (rays[0],) for mode, rays in FULL_RAYS_BY_MODE.items()}

NR = 64
NZ = 128
P_DEFAULT_W = 1.0e10
P_WEAK_W = 3.0e9
T_END_DEFAULT_S = 1.0e-10
E_FLOOR = 1.0e-300
L1_CONDUCTION_PULSE_DURATION_S = 1.0e-10
L1_CONDUCTION_TE0_EV = 1.0
L1_QEI_TE0_EV = 10.0
L1_QEI_TI0_EV = 1.0
L1_QEI_RHO_CD_GCC = 1.0e-5


@dataclass(frozen=True)
class Snapshot:
    path: Path
    deposited_power: Any
    vol: Any
    e_laser_deposited: float
    e_laser_escaped: float
    sim_time: float
    rho: Any | None = None
    Te: Any | None = None
    Ti: Any | None = None
    a_q_r_shells: Any | None = None
    a_q_z_shells: Any | None = None


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    out = [part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip()]
    bad = [part for part in out if part not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid {label}: {', '.join(bad)}")
    if not out:
        raise argparse.ArgumentTypeError(f"{label} list is empty")
    return out


def parse_ray_list(value: str) -> list[int]:
    out = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not out:
        raise argparse.ArgumentTypeError("ray list is empty")
    if any(n < 2 for n in out):
        raise argparse.ArgumentTypeError("2D_RZ rays per axis must be >= 2")
    return out


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def case_name(mode: str, rays_per_axis: int, seed: int, power_w: float) -> str:
    return f"l1_{mode}_nr{NR}_nz{NZ}_rays{rays_per_axis}_p{safe_float_token(power_w)}_seed{seed}"


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=False) + "\n")


def discover_results_dir(outdir: Path) -> Path:
    def has_numbered_plot(path: Path) -> bool:
        return path.exists() and any(
            len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_"
            for p in path.glob("*.h5")
        )

    direct = outdir / "results"
    if has_numbered_plot(direct):
        return direct
    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if has_numbered_plot(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no plot files found under {outdir} or {outdir}_*/results")


def locate_plot_files(results_dir: Path, run_id: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    files = sorted(
        (p for p in results_dir.glob(f"{run_id}_*.h5") if p.stem.removeprefix(f"{run_id}_").isdigit()),
        key=plot_index,
    )
    if files:
        return files
    return sorted((p for p in results_dir.glob("*_*.h5") if p.stem.rsplit("_", 1)[-1].isdigit()), key=plot_index)


def reshape_cells(values: Any, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == NR * NZ:
        return arr.reshape((NR, NZ))
    if arr.shape == (NR, NZ):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(NR, NZ)} or flat {NR * NZ}")


def scalar_dataset(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else default


def read_snapshot(path: Path) -> Snapshot:
    import h5py

    with h5py.File(path, "r") as handle:
        return Snapshot(
            path=path,
            deposited_power=reshape_cells(handle["laser/deposited_power"][()], "laser/deposited_power"),
            vol=reshape_cells(handle["hydro/vol"][()], "hydro/vol"),
            e_laser_deposited=scalar_dataset(handle, "time_state/E_laser_deposited"),
            e_laser_escaped=scalar_dataset(handle, "time_state/E_laser_escaped"),
            sim_time=scalar_dataset(handle, "time_state/t"),
            rho=reshape_cells(handle["hydro/rho"][()], "hydro/rho") if "hydro/rho" in handle else None,
            Te=reshape_cells(handle["hydro/Te"][()], "hydro/Te") if "hydro/Te" in handle else None,
            Ti=reshape_cells(handle["hydro/Ti"][()], "hydro/Ti") if "hydro/Ti" in handle else None,
            a_q_r_shells=handle["laser/A_Q_r_shells"][()] if "laser/A_Q_r_shells" in handle else None,
            a_q_z_shells=handle["laser/A_Q_z_shells"][()] if "laser/A_Q_z_shells" in handle else None,
        )


def read_final_snapshot(results_dir: Path, run_id: str) -> Snapshot:
    files = locate_plot_files(results_dir, run_id)
    if not files:
        raise RuntimeError("expected at least one plot snapshot")
    return read_snapshot(files[-1])


def read_two_snapshots(results_dir: Path, run_id: str, target_time_s: float) -> tuple[Snapshot, Snapshot]:
    files = locate_plot_files(results_dir, run_id)
    if not files:
        raise RuntimeError("expected at least one plot snapshot")
    snapshots = [read_snapshot(path) for path in files]
    target = min(snapshots, key=lambda snapshot: abs(snapshot.sim_time - target_time_s))
    return target, snapshots[-1]


def read_history_series(results_dir: Path, run_id: str, path: str) -> list[float]:
    import h5py
    import numpy as np

    candidates = list(results_dir.glob(f"{run_id}_history.h5")) + list(results_dir.glob("*_history.h5"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        with h5py.File(candidate, "r") as handle:
            if path in handle:
                arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
                if arr.size:
                    return [float(x) for x in arr]
    return []


def radial_axis_spike_fraction(field: Any, vol: Any) -> float:
    import numpy as np

    dep = np.maximum(np.asarray(field, dtype=float), 0.0) * np.asarray(vol, dtype=float)
    total = float(np.sum(dep))
    if total <= 0.0:
        return 0.0
    return float(np.sum(dep[0:1, :]) / total)


def active_aq_max(aq_shells: Any | None, shell_q_max: Any, eps_active: float) -> float | None:
    import numpy as np

    if aq_shells is None:
        return None
    aq = np.asarray(aq_shells, dtype=float).reshape(-1)
    aq = np.where(np.isfinite(aq) & (aq > 0.0), aq, 0.0)
    qmax = np.asarray(shell_q_max, dtype=float).reshape(-1)
    if aq.size != qmax.size:
        raise ValueError(f"A_Q shape mismatch: aq={aq.size}, qmax={qmax.size}")
    if aq.size == 0:
        return 0.0
    global_q_max = float(np.max(qmax)) if qmax.size else 0.0
    if not math.isfinite(global_q_max) or global_q_max <= 0.0:
        return 0.0
    active = qmax >= max(eps_active, 0.0) * global_q_max
    if not np.any(active):
        return 0.0
    return float(np.max(aq[active]))


def a_q_metrics(snapshot: Snapshot, eps_active: float) -> dict[str, Any]:
    import numpy as np

    dep = np.asarray(snapshot.deposited_power, dtype=float) * np.asarray(snapshot.vol, dtype=float)
    dep = np.where(np.isfinite(dep) & (dep > 0.0), dep, 0.0)
    qmax_r = np.max(dep, axis=1)
    qmax_z = np.max(dep, axis=0)
    max_r = active_aq_max(snapshot.a_q_r_shells, qmax_r, eps_active)
    max_z = active_aq_max(snapshot.a_q_z_shells, qmax_z, eps_active)
    return {
        "A_Q_native_present": snapshot.a_q_r_shells is not None and snapshot.a_q_z_shells is not None,
        "A_Q_active_eps": eps_active,
        "A_Q_r_shells_max_active": max_r,
        "A_Q_z_shells_max_active": max_z,
        "A_Q_r_shells": None if snapshot.a_q_r_shells is None else [float(x) for x in np.asarray(snapshot.a_q_r_shells).reshape(-1)],
        "A_Q_z_shells": None if snapshot.a_q_z_shells is None else [float(x) for x in np.asarray(snapshot.a_q_z_shells).reshape(-1)],
        "A_Q_gate_policy": "informational_calibration",
    }


def z_profile(field: Any, vol: Any) -> Any:
    import numpy as np

    dep = np.maximum(np.asarray(field, dtype=float), 0.0) * np.asarray(vol, dtype=float)
    prof = np.sum(dep, axis=0)
    total = float(np.sum(prof))
    return prof / max(total, E_FLOOR)


def profile_l2(a: Any, b: Any) -> float:
    import numpy as np

    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((aa - bb) ** 2)) / max(np.sqrt(np.mean(bb**2)), E_FLOOR))


def axis_te_rise_fraction(te_field: Any, te0_ev: float, vol: Any) -> float | None:
    import numpy as np

    rise = np.maximum(np.asarray(te_field, dtype=float) - te0_ev, 0.0)
    weighted = rise * np.asarray(vol, dtype=float)
    total = float(np.sum(weighted))
    if total <= 0.0:
        return None
    return float(np.sum(weighted[0:1, :]) / total)


def cd_mask(rho_field: Any, rho_cd_threshold: float) -> Any:
    import numpy as np

    return np.asarray(rho_field, dtype=float) > 0.5 * rho_cd_threshold


def mass_weighted_mean(field: Any, rho: Any, vol: Any, mask: Any) -> float | None:
    import numpy as np

    f = np.asarray(field, dtype=float)
    m = (np.asarray(rho, dtype=float) * np.asarray(vol, dtype=float))[mask]
    if float(np.sum(m)) <= 0.0:
        return None
    return float(np.sum(f[mask] * m) / np.sum(m))


def z_mirror_asymmetry(deposit_power: Any, vol: Any) -> float | None:
    import numpy as np

    dep = np.maximum(np.asarray(deposit_power, dtype=float), 0.0) * np.asarray(vol, dtype=float)
    nz = dep.shape[1]
    half = nz // 2
    lower = float(np.sum(dep[:, :half]))
    upper = float(np.sum(dep[:, half:]))
    total = lower + upper
    if total <= 0.0:
        return None
    return abs(lower - upper) / total


def parse_deck_log(log_path: Path) -> dict[str, bool]:
    out = {"conduction_enabled": False, "radiation_enabled": False}
    if not log_path.exists():
        return out
    text = log_path.read_text(encoding="utf-8", errors="replace")
    out["conduction_enabled"] = "conduction=True" in text
    out["radiation_enabled"] = "radiation=True" in text
    return out


def effective_laser_energies(snapshot: Snapshot, results_dir: Path, run_id: str) -> tuple[float, float, float]:
    incident_series = read_history_series(results_dir, run_id, "energy/laser_incident")
    escaped_series = read_history_series(results_dir, run_id, "energy/laser_escaped")
    deposited_series = read_history_series(results_dir, run_id, "energy/laser_deposited")

    incident = sum(max(x, 0.0) for x in incident_series)
    escaped = sum(max(x, 0.0) for x in escaped_series)
    deposited = deposited_series[-1] if deposited_series else snapshot.e_laser_deposited
    if incident > 0.0:
        return incident, deposited, escaped

    total = snapshot.e_laser_deposited + snapshot.e_laser_escaped
    if total > 0.0:
        return total, snapshot.e_laser_deposited, snapshot.e_laser_escaped
    return P_DEFAULT_W * T_END_DEFAULT_S, snapshot.e_laser_deposited, snapshot.e_laser_escaped


def compute_metrics(snapshot: Snapshot, results_dir: Path, run_id: str, wall_time_s: float, bytes_out: int, aq_active_eps: float) -> dict[str, Any]:
    import numpy as np

    e_input, e_dep, e_esc = effective_laser_energies(snapshot, results_dir, run_id)
    dep_energy = np.maximum(np.asarray(snapshot.deposited_power, dtype=float), 0.0) * np.asarray(snapshot.vol, dtype=float)
    dep_total_from_field_rate = float(np.sum(dep_energy))
    metrics = {
        "E_laser_input": e_input,
        "E_laser_deposited": e_dep,
        "E_laser_escaped": e_esc,
        "absorbed_fraction": e_dep / max(e_input, E_FLOOR),
        "escaped_fraction": e_esc / max(e_input, E_FLOOR),
        "deposited_power_volume_integral": dep_total_from_field_rate,
        "axis_spike_fraction": radial_axis_spike_fraction(snapshot.deposited_power, snapshot.vol),
        "deposited_power_nan_count": int(np.count_nonzero(~np.isfinite(snapshot.deposited_power))),
        "deposited_power_negative_count": int(np.count_nonzero(np.asarray(snapshot.deposited_power) < 0.0)),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(a_q_metrics(snapshot, aq_active_eps))
    return metrics


def compute_conduction_metrics(target: Snapshot, final: Snapshot) -> dict[str, Any]:
    if target.Te is None or final.Te is None:
        raise RuntimeError("l1_conduction_submode requires hydro/Te in plot snapshots")
    pulse_axis_fraction = axis_te_rise_fraction(target.Te, L1_CONDUCTION_TE0_EV, target.vol)
    final_axis_fraction = axis_te_rise_fraction(final.Te, L1_CONDUCTION_TE0_EV, final.vol)
    ratio = None
    if pulse_axis_fraction is not None and pulse_axis_fraction > 1.0e-12 and final_axis_fraction is not None:
        ratio = final_axis_fraction / pulse_axis_fraction
    return {
        "te_rise_axis_fraction_end_of_pulse": pulse_axis_fraction,
        "te_rise_axis_fraction_final": final_axis_fraction,
        "te_rise_axis_fraction_ratio": ratio,
        "end_of_pulse_snapshot_time_s": target.sim_time,
        "final_snapshot_time_s": final.sim_time,
    }


def compute_qei_metrics(final: Snapshot) -> dict[str, Any]:
    import numpy as np

    if final.rho is None or final.Te is None or final.Ti is None:
        raise RuntimeError("l1_qei_submode requires hydro/rho, hydro/Te, and hydro/Ti in final snapshot")
    mask = cd_mask(final.rho, L1_QEI_RHO_CD_GCC)
    gap = np.abs(np.asarray(final.Te, dtype=float) - np.asarray(final.Ti, dtype=float))
    gap_final = mass_weighted_mean(gap, final.rho, final.vol, mask)
    ti_mean = mass_weighted_mean(final.Ti, final.rho, final.vol, mask)
    initial_gap = L1_QEI_TE0_EV - L1_QEI_TI0_EV
    return {
        "te_ti_gap_final_ev": gap_final,
        "te_ti_gap_initial_ev": initial_gap,
        "te_ti_gap_final_over_initial": None if gap_final is None else gap_final / initial_gap,
        "ti_mean_final_ev": ti_mean,
        "ti_mean_initial_ev": L1_QEI_TI0_EV,
        "ti_mean_increased": ti_mean is not None and ti_mean > L1_QEI_TI0_EV,
        "cd_mask_cell_count": int(np.count_nonzero(mask)),
    }


def compute_multibeam_metrics(snapshot: Snapshot) -> dict[str, Any]:
    return {"z_mirror_asymmetry": z_mirror_asymmetry(snapshot.deposited_power, snapshot.vol)}


def evaluate_conduction_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("conduction_enabled") is not True:
        failed.append("l1_conduction_enabled")
    ratio = metrics.get("te_rise_axis_fraction_ratio")
    if ratio is None:
        failed.append("l1_conduction_no_signal")
    elif ratio > 0.95:
        failed.append("l1_conduction_axis_smoothing")
    return failed


def evaluate_qei_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("qei_enabled") is not True:
        failed.append("l1_qei_enabled")
    gap_ratio = metrics.get("te_ti_gap_final_over_initial")
    if gap_ratio is None or gap_ratio > 0.50:
        failed.append("l1_qei_te_ti_gap_reduction")
    if metrics.get("ti_mean_increased") is not True:
        failed.append("l1_qei_ti_mean_increased")
    return failed


def evaluate_multibeam_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    asymmetry = metrics.get("z_mirror_asymmetry")
    if asymmetry is None or asymmetry > 0.05:
        failed.append("l1_multibeam_z_mirror_asymmetry")
    return failed


def evaluate_gates(mode: str, metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    warnings: list[str] = []
    if metrics.get("deposited_power_nan_count", 0) != 0:
        failed.append("l1_deposited_power_finite")
    if not metrics.get("A_Q_native_present", False):
        warnings.append("l1_native_a_q_missing")
    if metrics.get("E_laser_input", 0.0) <= 0.0:
        failed.append("l1_positive_laser_input")
    absorbed = metrics.get("absorbed_fraction", math.inf)
    if mode == "vacuum_transparent" and absorbed >= 1.0e-3:
        failed.append("l1_vacuum_transparent_absorption")
    if mode == "uniform_absorbing_cylinder":
        if not (1.0e-4 <= absorbed <= 1.0):
            failed.append("l1_uniform_cylinder_absorbed_fraction_range")
        if metrics.get("axis_spike_fraction", math.inf) >= 0.70:
            failed.append("l1_uniform_cylinder_axis_spike_proxy")
        metrics["A_Q_gate_metric"] = "A_Q_r_shells_max_active"
        metrics["A_Q_gate_value"] = metrics.get("A_Q_r_shells_max_active")
    if mode == "cd_planar_slab":
        if not (1.0e-5 <= absorbed <= 1.0):
            failed.append("l1_planar_slab_absorbed_fraction_range")
        metrics["A_Q_gate_metric"] = "A_Q_z_shells_max_active"
        metrics["A_Q_gate_value"] = metrics.get("A_Q_z_shells_max_active")
    if mode == "cd_sphere_weak_pulse":
        if not (1.0e-6 <= absorbed <= 1.0):
            failed.append("l1_sphere_absorbed_fraction_range")
        if metrics.get("axis_spike_fraction", math.inf) >= 0.95:
            failed.append("l1_sphere_axis_spike")
        metrics["A_Q_gate_metric"] = "A_Q_r_shells_max_active,A_Q_z_shells_max_active"
        metrics["A_Q_gate_value"] = [
            metrics.get("A_Q_r_shells_max_active"),
            metrics.get("A_Q_z_shells_max_active"),
        ]
    if mode == "l1_conduction_submode":
        failed.extend(evaluate_conduction_gates(metrics))
    if mode == "l1_qei_submode":
        failed.extend(evaluate_qei_gates(metrics))
    if mode == "l1_multibeam":
        failed.extend(evaluate_multibeam_gates(metrics))
    metrics["informational_gates"] = warnings
    return failed


def active_env(args: argparse.Namespace, mode: str, rays: int, seed: int, outdir: Path, power_w: float) -> dict[str, str]:
    env = {
        "TENRYU_L1_MODE": mode,
        "TENRYU_L1_NR": str(NR),
        "TENRYU_L1_NZ": str(NZ),
        "TENRYU_L1_SEED": str(seed),
        "TENRYU_L1_OUTDIR": str(outdir),
        "TENRYU_L1_N_RAYS_PER_AXIS": str(rays),
        "TENRYU_L1_MAX_STEPS": str(args.max_steps),
        "TENRYU_L1_LASER_PEAK_W": str(power_w),
        "TENRYU_L1_VERBOSITY": args.verbosity,
    }
    if args.t_end_s is not None:
        env["TENRYU_L1_T_END_S"] = str(args.t_end_s)
    if args.dt_s is not None:
        env["TENRYU_L1_DT_S"] = str(args.dt_s)
    return env


def power_for_mode(mode: str) -> float:
    return P_WEAK_W if mode == "cd_sphere_weak_pulse" else P_DEFAULT_W


def run_one(args: argparse.Namespace, mode: str, rays: int, seed: int) -> dict[str, Any]:
    power_w = power_for_mode(mode)
    run_id = case_name(mode, rays, seed, power_w)
    ladder_row = f"{mode}_rays{rays}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, rays, seed, outdir, power_w)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    metrics: dict[str, Any] = {"wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    result = "PASS"
    failed_gates: list[str] = []
    failure_class: str | None = None
    if proc.returncode != 0:
        result = "CRASHED"
        failed_gates = ["run_failure"]
        failure_class = "run_failure"
    else:
        try:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            snapshot = read_final_snapshot(results_dir, run_id)
            metrics = compute_metrics(snapshot, results_dir, run_id, wall_time_s, bytes_out, args.aq_active_eps)
            metrics.update(parse_deck_log(log_path))
            metrics["qei_enabled"] = (
                mode == "l1_qei_submode"
                and metrics.get("conduction_enabled") is True
                and metrics.get("radiation_enabled") is True
            )
            metrics["z_profile"] = [float(x) for x in z_profile(snapshot.deposited_power, snapshot.vol)]
            if mode == "l1_conduction_submode":
                end_of_pulse, final_snapshot = read_two_snapshots(
                    results_dir, run_id, L1_CONDUCTION_PULSE_DURATION_S
                )
                metrics.update(compute_conduction_metrics(end_of_pulse, final_snapshot))
            if mode == "l1_qei_submode":
                metrics.update(compute_qei_metrics(snapshot))
            if mode == "l1_multibeam":
                metrics.update(compute_multibeam_metrics(snapshot))
            failed_gates = evaluate_gates(mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_l1] analysis failed: {exc}\n")

    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": NR,
        "nz": NZ,
        "rays_per_axis": rays,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
        "env_overrides": env_overrides,
    }
    write_json(Path(metrics_ref), payload)
    return row_from_payload(payload, metrics_ref)


def row_from_payload(payload: dict[str, Any], metrics_ref: str) -> dict[str, Any]:
    metrics = payload["metrics"]
    return {
        "run_id": payload["run_id"],
        "stage_id": payload["stage_id"],
        "ladder_row": payload["ladder_row"],
        "seed": payload["seed"],
        "deck_path": payload["deck_path"],
        "mode": payload["mode"],
        "nr": payload["nr"],
        "nz": payload["nz"],
        "rays_per_axis": payload["rays_per_axis"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "absorbed_fraction": metrics.get("absorbed_fraction"),
        "escaped_fraction": metrics.get("escaped_fraction"),
        "axis_spike_fraction": metrics.get("axis_spike_fraction"),
        "A_Q_r_shells_max_active": metrics.get("A_Q_r_shells_max_active"),
        "A_Q_z_shells_max_active": metrics.get("A_Q_z_shells_max_active"),
        "planar_l2_to_101": metrics.get("planar_l2_to_101"),
        "multibeam_absorption_ratio": metrics.get("multibeam_absorption_ratio"),
        "z_mirror_asymmetry": metrics.get("z_mirror_asymmetry"),
    }


def load_payload(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def update_row(row: dict[str, Any], payload: dict[str, Any]) -> None:
    write_json(Path(row["metrics_ref"]), payload)
    updated = row_from_payload(payload, row["metrics_ref"])
    row.clear()
    row.update(updated)


def apply_planar_convergence_gate(rows: list[dict[str, Any]]) -> None:
    by_ray = {row["rays_per_axis"]: row for row in rows if row["mode"] == "cd_planar_slab" and row["result"] != "CRASHED"}
    if not {16, 36, 72, 101}.issubset(by_ray):
        return
    payloads = {ray: load_payload(row["metrics_ref"]) for ray, row in by_ray.items()}
    profiles = {ray: payloads[ray]["metrics"].get("z_profile") for ray in (16, 36, 72, 101)}
    if any(profile is None for profile in profiles.values()):
        return

    l2 = {ray: profile_l2(profiles[ray], profiles[101]) for ray in (16, 36, 72)}
    e72 = float(payloads[72]["metrics"].get("E_laser_deposited", 0.0))
    e101 = float(payloads[101]["metrics"].get("E_laser_deposited", 0.0))
    rel_e_72_101 = abs(e72 - e101) / max(abs(e101), E_FLOOR)
    l2_trend_ok = l2[72] <= l2[36] * 1.25 and l2[36] <= l2[16] * 1.25
    absorbed_values = [
        float(payloads[ray]["metrics"].get("absorbed_fraction", math.nan))
        for ray in (16, 36, 72, 101)
    ]
    absorbed_mean = sum(absorbed_values) / len(absorbed_values)
    absorbed_rel_spread = max(abs(value - absorbed_mean) for value in absorbed_values) / max(
        abs(absorbed_mean), E_FLOOR
    )
    absorbed_consistency_ok = (
        all(math.isfinite(value) for value in absorbed_values)
        and absorbed_rel_spread < 0.02
    )
    convergence_ok = (rel_e_72_101 < 0.02 and l2_trend_ok) or absorbed_consistency_ok

    for ray, payload in payloads.items():
        payload["metrics"]["planar_l2_to_101"] = None if ray == 101 else l2.get(ray)
        payload["metrics"]["planar_E_rel_72_101"] = rel_e_72_101
        payload["metrics"]["planar_l2_trend_ok"] = l2_trend_ok
        payload["metrics"]["planar_absorbed_consistency_ok"] = absorbed_consistency_ok
        payload["metrics"]["planar_absorbed_rel_spread"] = absorbed_rel_spread
        payload["metrics"]["planar_convergence_gate"] = "pass" if convergence_ok else "fail"
        if not convergence_ok and "l1_planar_convergence" not in payload["failed_gates"]:
            payload["failed_gates"].append("l1_planar_convergence")
            payload["result"] = "FAIL"
            payload["failure_class"] = "gate_failure"
        update_row(by_ray[ray], payload)


def apply_multibeam_gate(rows: list[dict[str, Any]]) -> None:
    references = {
        row["rays_per_axis"]: row
        for row in rows
        if row["mode"] == "cd_planar_slab" and row["result"] != "CRASHED"
    }
    for row in [row for row in rows if row["mode"] == "l1_multibeam" and row["result"] != "CRASHED"]:
        payload = load_payload(row["metrics_ref"])
        ref_row = references.get(row["rays_per_axis"])
        if ref_row is None:
            payload["metrics"]["multibeam_absorption_ratio"] = None
            if "l1_multibeam_no_reference" not in payload["failed_gates"]:
                payload["failed_gates"].append("l1_multibeam_no_reference")
            payload["result"] = "FAIL"
            payload["failure_class"] = "gate_failure"
            print(f"[run_l1] warning: no cd_planar_slab reference for l1_multibeam rays={row['rays_per_axis']}")
            update_row(row, payload)
            continue

        ref_payload = load_payload(ref_row["metrics_ref"])
        # Use E_laser_deposited (extensive). absorbed_fraction is intensive
        # and would always give 0.5 regardless of multibeam correctness.
        multibeam_e_dep = float(payload["metrics"].get("E_laser_deposited", math.nan))
        reference_e_dep = float(ref_payload["metrics"].get("E_laser_deposited", math.nan))
        ratio = multibeam_e_dep / max(2.0 * reference_e_dep, E_FLOOR)
        payload["metrics"]["multibeam_absorption_ratio"] = ratio
        if not math.isfinite(ratio) or not (0.75 <= ratio <= 1.25):
            if "l1_multibeam_absorption_ratio" not in payload["failed_gates"]:
                payload["failed_gates"].append("l1_multibeam_absorption_ratio")
            payload["result"] = "FAIL"
            payload["failure_class"] = "gate_failure"
        update_row(row, payload)


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int]]:
    rows = []
    for mode in args.mode_list:
        rays_list = tuple(args.ray_list) if args.ray_list is not None else (
            SMOKE_RAYS_BY_MODE[mode] if args.smoke else FULL_RAYS_BY_MODE[mode]
        )
        for rays in rays_list:
            for seed in args.seed_list:
                rows.append((mode, rays, seed))
    return rows


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    aggregate = {
        "total_runs": len(rows),
        "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
        "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
        "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
    }
    for key in (
        "wall_time_s",
        "absorbed_fraction",
        "axis_spike_fraction",
        "A_Q_r_shells_max_active",
        "A_Q_z_shells_max_active",
        "planar_l2_to_101",
    ):
        values = finite_values(rows, key)
        aggregate[f"{key}_max"] = max(values) if values else None
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": aggregate,
    }
    write_json(Path(args.summary_json), payload)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Stage L1 laser-deposition sweep summary")
    print(
        "ladder_row seed result absorbed_fraction escaped_fraction axis_spike_fraction "
        "A_Q_r_shells_max_active A_Q_z_shells_max_active planar_l2_to_101 failed_gates"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {absorbed_fraction} {escaped_fraction} "
            "{axis_spike_fraction} {A_Q_r_shells_max_active} "
            "{A_Q_z_shells_max_active} {planar_l2_to_101} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_l1_laser_dep.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/l1_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--ray-list", type=parse_ray_list, default=None)
    parser.add_argument("--seed-list", type=parse_ray_list, default=[12345])
    parser.add_argument("--smoke", action="store_true", help="use each mode's smallest ray count only")
    parser.add_argument("--t-end-s", type=float, default=None, help="Override deck t_end. Default: deck MODE_DEFAULTS.")
    parser.add_argument("--dt-s", type=float, default=None, help="Override deck dt. Default: deck MODE_DEFAULTS.")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/L1/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/L1/runs.jsonl")
    parser.add_argument(
        "--aq-active-eps",
        type=float,
        default=float(os.environ.get("TENRYU_L1_AQ_ACTIVE_EPS", "1.0e-3")),
        help="Active-shell threshold for native A_Q aggregation as a fraction of global shell q_max.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [run_one(args, mode, rays, seed) for mode, rays, seed in matrix_rows(args)]
    apply_planar_convergence_gate(rows)
    apply_multibeam_gate(rows)
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
