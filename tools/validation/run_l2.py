#!/usr/bin/env python3
"""Run Stage L2 2D RZ laser+hydro validation sweep."""

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


STAGE_ID = "L2"
MODES = (
    "planar_slab_pulse",
    "spherical_cd_weak_1pct",
    "spherical_cd_weak_5pct",
    "spherical_cd_weak_10pct",
)
NR = 128
NZ = 256
RAYS_PER_BEAM = 72
T_END_DEFAULT_S = 5.0e-11
SMOKE_T_END_DEFAULT_S = 5.0e-11
DT_DEFAULT_S = 2.0e-13
GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
MODE_ENERGY_FRACTION = {
    "planar_slab_pulse": 0.01,
    "spherical_cd_weak_1pct": 0.01,
    "spherical_cd_weak_5pct": 0.05,
    "spherical_cd_weak_10pct": 0.10,
}
E_FLOOR = 1.0e-300


@dataclass(frozen=True)
class Snapshot:
    path: Path
    t: float
    rho: Any
    Te: Any
    Ti: Any
    Pe: Any
    Pi: Any
    vol: Any
    r_nodes: Any
    z_nodes: Any
    e_laser_deposited: float
    e_laser_escaped: float


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    out = [part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip()]
    bad = [part for part in out if part not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid {label}: {', '.join(bad)}")
    if not out:
        raise argparse.ArgumentTypeError(f"{label} list is empty")
    return out


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def laser_input_erg(mode: str) -> float:
    return GXII_REFERENCE_E_ERG * MODE_ENERGY_FRACTION[mode]


def case_name(mode: str, seed: int, e_input: float) -> str:
    return f"l2_{mode}_nr{NR}_nz{NZ}_rays{RAYS_PER_BEAM}_e{safe_float_token(e_input)}_seed{seed}"


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


def reshape_nodes(values: Any, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == (NR + 1) * (NZ + 1):
        return arr.reshape((NR + 1, NZ + 1))
    if arr.shape == (NR + 1, NZ + 1):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(NR + 1, NZ + 1)}")


def scalar_dataset(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else default


def read_snapshot(path: Path) -> Snapshot:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        return Snapshot(
            path=path,
            t=t_value,
            rho=reshape_cells(handle["hydro/rho"][()], "hydro/rho"),
            Te=reshape_cells(handle["hydro/Te"][()], "hydro/Te"),
            Ti=reshape_cells(handle["hydro/Ti"][()], "hydro/Ti"),
            Pe=reshape_cells(handle["hydro/Pe"][()], "hydro/Pe"),
            Pi=reshape_cells(handle["hydro/Pi"][()], "hydro/Pi"),
            vol=reshape_cells(handle["hydro/vol"][()], "hydro/vol"),
            r_nodes=reshape_nodes(handle[r_node_path][()], r_node_path),
            z_nodes=reshape_nodes(handle[z_node_path][()], z_node_path),
            e_laser_deposited=scalar_dataset(handle, "time_state/E_laser_deposited"),
            e_laser_escaped=scalar_dataset(handle, "time_state/E_laser_escaped"),
        )


def read_snapshots(results_dir: Path, run_id: str) -> list[Snapshot]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least two plot snapshots for shock metrics")
    return [read_snapshot(path) for path in files]


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
                return [float(x) for x in arr]
    return []


def effective_laser_energies(snapshots: list[Snapshot], results_dir: Path, run_id: str, expected_input: float) -> tuple[float, float, float]:
    incident_series = read_history_series(results_dir, run_id, "energy/laser_incident")
    escaped_series = read_history_series(results_dir, run_id, "energy/laser_escaped")
    deposited_series = read_history_series(results_dir, run_id, "energy/laser_deposited")
    incident = sum(max(x, 0.0) for x in incident_series)
    escaped = sum(max(x, 0.0) for x in escaped_series)
    deposited = deposited_series[-1] if deposited_series else snapshots[-1].e_laser_deposited
    if incident > 0.0:
        return incident, deposited, escaped
    total_state = snapshots[-1].e_laser_deposited + snapshots[-1].e_laser_escaped
    if total_state > 0.0:
        return total_state, snapshots[-1].e_laser_deposited, snapshots[-1].e_laser_escaped
    return expected_input, snapshots[-1].e_laser_deposited, snapshots[-1].e_laser_escaped


def cell_centers(snapshot: Snapshot) -> tuple[Any, Any]:
    import numpy as np

    rn = np.asarray(snapshot.r_nodes, dtype=float)
    zn = np.asarray(snapshot.z_nodes, dtype=float)
    rc = 0.25 * (rn[:-1, :-1] + rn[1:, :-1] + rn[:-1, 1:] + rn[1:, 1:])
    zc = 0.25 * (zn[:-1, :-1] + zn[1:, :-1] + zn[:-1, 1:] + zn[1:, 1:])
    return rc, zc


def z_profile(snapshot: Snapshot, field: Any) -> tuple[Any, Any]:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    weights = np.asarray(snapshot.vol, dtype=float)
    zc = cell_centers(snapshot)[1]
    denom = np.maximum(np.sum(weights, axis=0), E_FLOOR)
    profile = np.sum(arr * weights, axis=0) / denom
    centers = np.mean(zc, axis=0)
    return centers, profile


def estimate_shock_z(snapshot: Snapshot, mode: str) -> float | None:
    import numpy as np

    zc, rho_profile = z_profile(snapshot, snapshot.rho)
    if rho_profile.size < 3 or not np.all(np.isfinite(rho_profile)):
        return None
    if mode == "planar_slab_pulse":
        candidate = np.where(zc > 0.0)[0]
    else:
        candidate = np.where((zc > -0.012) & (zc < 0.012))[0]
    if candidate.size < 3:
        candidate = np.arange(rho_profile.size)
    jumps = np.abs(np.diff(rho_profile[candidate]))
    if jumps.size == 0 or float(np.nanmax(jumps)) <= 0.0:
        return None
    local = int(np.nanargmax(jumps))
    idx0 = int(candidate[local])
    idx1 = int(candidate[min(local + 1, candidate.size - 1)])
    return float(0.5 * (zc[idx0] + zc[idx1]))


def shock_metrics(snapshots: list[Snapshot], mode: str) -> dict[str, Any]:
    positions = []
    for snapshot in snapshots:
        z = estimate_shock_z(snapshot, mode)
        if z is not None and math.isfinite(z):
            positions.append((snapshot.t, z))
    if len(positions) < 2:
        return {
            "shock_launch_time_s": None,
            "shock_speed_cm_s": None,
            "shock_position_count": len(positions),
        }
    z0 = positions[0][1]
    launch = None
    motion_threshold = 0.5 * abs(positions[-1][1] - positions[0][1])
    motion_threshold = max(motion_threshold, 1.0e-5)
    for t, z in positions[1:]:
        if abs(z - z0) >= motion_threshold:
            launch = t
            break
    if launch is None:
        launch = positions[1][0]
    t_a, z_a = positions[0]
    t_b, z_b = positions[-1]
    speed = (z_b - z_a) / max(t_b - t_a, E_FLOOR)
    return {
        "shock_launch_time_s": float(launch),
        "shock_speed_cm_s": float(speed),
        "shock_position_count": len(positions),
        "shock_z_initial_cm": float(z_a),
        "shock_z_final_cm": float(z_b),
    }


def ablation_pressure_avg(snapshot: Snapshot, mode: str) -> float | None:
    import numpy as np

    rc, zc = cell_centers(snapshot)
    pressure = np.asarray(snapshot.Pe, dtype=float) + np.asarray(snapshot.Pi, dtype=float)
    rho = np.asarray(snapshot.rho, dtype=float)
    if mode == "planar_slab_pulse":
        mask = (np.abs(zc - 0.002) <= 8.0e-4) & (rho > 1.0e-6)
    else:
        radius = np.sqrt(rc * rc + zc * zc)
        mask = (np.abs(radius - 0.01) <= 8.0e-4) & (zc > 0.0) & (rho > 1.0e-6)
    values = pressure[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.mean(values))


def compute_metrics(mode: str, snapshots: list[Snapshot], results_dir: Path, run_id: str, wall_time_s: float, bytes_out: int, expected_input: float) -> dict[str, Any]:
    import numpy as np

    e_input, e_dep, e_esc = effective_laser_energies(snapshots, results_dir, run_id, expected_input)
    rho_all = np.concatenate([np.asarray(s.rho, dtype=float).reshape(-1) for s in snapshots])
    te_all = np.concatenate([np.asarray(s.Te, dtype=float).reshape(-1) for s in snapshots])
    ti_all = np.concatenate([np.asarray(s.Ti, dtype=float).reshape(-1) for s in snapshots])
    finite_nan_count = int(
        np.count_nonzero(~np.isfinite(rho_all))
        + np.count_nonzero(~np.isfinite(te_all))
        + np.count_nonzero(~np.isfinite(ti_all))
    )
    metrics: dict[str, Any] = {
        "E_laser_input": e_input,
        "E_laser_input_expected": expected_input,
        "E_laser_deposited": e_dep,
        "E_laser_escaped": e_esc,
        "absorbed_fraction": e_dep / max(e_input, E_FLOOR),
        "escaped_fraction": e_esc / max(e_input, E_FLOOR),
        "absorbed_E_budget_rel": abs((e_dep + e_esc) - e_input) / max(e_input, E_FLOOR),
        "nan_count_rho_T": finite_nan_count,
        "rho_min": float(np.nanmin(rho_all)),
        "rho_negative_count": int(np.count_nonzero(rho_all < 0.0)),
        "Te_min_eV": float(np.nanmin(te_all)),
        "Ti_min_eV": float(np.nanmin(ti_all)),
        "Te_negative_count": int(np.count_nonzero(te_all < 0.0)),
        "Ti_negative_count": int(np.count_nonzero(ti_all < 0.0)),
        "ablation_pressure_avg": ablation_pressure_avg(snapshots[-1], mode),
        "snapshot_count": len(snapshots),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(shock_metrics(snapshots, mode))
    return metrics


def evaluate_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("absorbed_fraction", 0.0) <= 0.1:
        failed.append("l2_laser_absorbed_fraction")
    if metrics.get("absorbed_E_budget_rel", math.inf) >= 0.02:
        failed.append("l2_absorbed_E_budget")
    if metrics.get("nan_count_rho_T", 1) != 0:
        failed.append("l2_no_nan_rho_T")
    if metrics.get("rho_min", 0.0) <= 0.0 or metrics.get("rho_negative_count", 1) != 0:
        failed.append("l2_positive_density")
    if metrics.get("Te_min_eV", 0.0) < 0.0 or metrics.get("Ti_min_eV", 0.0) < 0.0:
        failed.append("l2_nonnegative_temperature")
    if metrics.get("shock_launch_time_s") is None:
        failed.append("l2_shock_launch_time_reported")
    if metrics.get("shock_speed_cm_s") is None:
        failed.append("l2_shock_speed_reported")
    if metrics.get("ablation_pressure_avg") is None:
        failed.append("l2_ablation_pressure_reported")
    return failed


def active_env(args: argparse.Namespace, mode: str, seed: int, outdir: Path, e_input: float) -> dict[str, str]:
    return {
        "TENRYU_L2_MODE": mode,
        "TENRYU_L2_NR": str(NR),
        "TENRYU_L2_NZ": str(NZ),
        "TENRYU_L2_SEED": str(seed),
        "TENRYU_L2_OUTDIR": str(outdir),
        "TENRYU_L2_RAYS_PER_BEAM": str(RAYS_PER_BEAM),
        "TENRYU_L2_LASER_INPUT_ERG": str(e_input),
        "TENRYU_L2_T_END_S": str(args.t_end_s),
        "TENRYU_L2_DT_INITIAL_S": str(args.dt_s),
        "TENRYU_L2_DT_MAX_S": str(args.dt_s),
        "TENRYU_L2_PLOT_EVERY_STEPS": str(args.plot_every_steps),
        "TENRYU_L2_MAX_STEPS": str(args.max_steps),
        "TENRYU_L2_VERBOSITY": args.verbosity,
    }


def run_one(args: argparse.Namespace, mode: str, seed: int) -> dict[str, Any]:
    e_input = laser_input_erg(mode)
    run_id = case_name(mode, seed, e_input)
    ladder_row = f"{mode}_nr{NR}_nz{NZ}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, seed, outdir, e_input)
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
            snapshots = read_snapshots(results_dir, run_id)
            metrics = compute_metrics(mode, snapshots, results_dir, run_id, wall_time_s, bytes_out, e_input)
            failed_gates = evaluate_gates(metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_l2] analysis failed: {exc}\n")

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
        "rays_per_beam": RAYS_PER_BEAM,
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
        "rays_per_beam": payload["rays_per_beam"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "absorbed_E_budget_rel": metrics.get("absorbed_E_budget_rel"),
        "absorbed_fraction": metrics.get("absorbed_fraction"),
        "rho_min": metrics.get("rho_min"),
        "Te_min_eV": metrics.get("Te_min_eV"),
        "Ti_min_eV": metrics.get("Ti_min_eV"),
        "shock_launch_time_s": metrics.get("shock_launch_time_s"),
        "shock_speed_cm_s": metrics.get("shock_speed_cm_s"),
        "ablation_pressure_avg": metrics.get("ablation_pressure_avg"),
    }


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(mode, seed) for mode in args.mode_list for seed in args.seed_list]


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
    for key in ("wall_time_s", "absorbed_E_budget_rel", "absorbed_fraction", "shock_speed_cm_s", "ablation_pressure_avg"):
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
    print("Stage L2 laser+hydro sweep summary")
    print(
        "ladder_row seed result absorbed_E_budget_rel absorbed_fraction rho_min "
        "Te_min_eV Ti_min_eV shock_launch_time_s shock_speed_cm_s ablation_pressure_avg failed_gates"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {absorbed_E_budget_rel} {absorbed_fraction} "
            "{rho_min} {Te_min_eV} {Ti_min_eV} {shock_launch_time_s} "
            "{shock_speed_cm_s} {ablation_pressure_avg} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_l2_laser_hydro.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/l2_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--seed-list", type=parse_seed_list, default=[12345])
    parser.add_argument("--smoke", action="store_true", help="kept for parity; selected modes still run at 128x256")
    parser.add_argument("--t-end-s", type=float, default=T_END_DEFAULT_S)
    parser.add_argument("--dt-s", type=float, default=DT_DEFAULT_S)
    parser.add_argument("--plot-every-steps", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/L2/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/L2/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.smoke and args.t_end_s == T_END_DEFAULT_S:
        args.t_end_s = SMOKE_T_END_DEFAULT_S
    rows = [run_one(args, mode, seed) for mode, seed in matrix_rows(args)]
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
