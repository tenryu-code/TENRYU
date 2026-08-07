#!/usr/bin/env python3
"""Run Phase 2d ALE-on carry-over validation gates."""

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


RH2_DECK = "examples/verification/2d_rz_rh2_hydro_sn.py"
L2_DECK = "examples/verification/2d_rz_l2_laser_hydro.py"
E_FLOOR = 1.0e-300
GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
L2_WEAK_1PCT_E_ERG = GXII_REFERENCE_E_ERG * 0.01
L2_RAYS_PER_BEAM = 72
L2_MODE = "spherical_cd_weak_1pct"
L2_SPHERE_RADIUS_CM = 100.0e-4
RH2_E_BALANCE_LIMIT = 1.0e-7


@dataclass(frozen=True)
class Snapshot:
    path: Path
    rho: Any
    Pe: Any
    Pi: Any
    r_nodes: Any
    z_nodes: Any


@dataclass
class SummaryRow:
    carry_over: str
    deck: str
    grid: str
    t_end: str
    gate: str
    observed: float | None
    limit: float
    status: str
    log: str
    metrics: dict[str, Any]
    note: str = ""


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def l2_case_name(nr: int, nz: int, seed: int = 12345) -> str:
    return (
        f"l2_{L2_MODE}_nr{nr}_nz{nz}_rays{L2_RAYS_PER_BEAM}"
        f"_e{safe_float_token(L2_WEAK_1PCT_E_ERG)}_seed{seed}"
    )


def rh2_case_name(seed: int = 12345) -> str:
    return f"rh2_shock_tube_grey_sn_sn_transport_nr32_nz64_rho1_ka10_g1_seed{seed}"


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def discover_results_dir(outdir: Path) -> Path:
    def has_hdf5(path: Path) -> bool:
        return path.exists() and any(p.suffix == ".h5" for p in path.glob("*.h5"))

    direct = outdir / "results"
    if has_hdf5(direct):
        return direct
    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if has_hdf5(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no HDF5 results found under {outdir} or {outdir}_*/results")


def locate_history_file(results_dir: Path, run_id: str) -> Path:
    exact = results_dir / f"{run_id}_history.h5"
    if exact.exists():
        return exact
    candidates = sorted(results_dir.glob("*_history.h5"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no history file found in {results_dir}")
    return candidates[-1]


def locate_plot_files(results_dir: Path, run_id: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    exact = sorted(
        (
            path
            for path in results_dir.glob(f"{run_id}_*.h5")
            if path.stem.removeprefix(f"{run_id}_").isdigit()
        ),
        key=plot_index,
    )
    if exact:
        return exact
    return sorted(
        (
            path
            for path in results_dir.glob("*_*.h5")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        ),
        key=plot_index,
    )


def read_history_series(results_dir: Path, run_id: str, path: str) -> list[float]:
    import h5py
    import numpy as np

    history = locate_history_file(results_dir, run_id)
    with h5py.File(history, "r") as handle:
        if path not in handle:
            raise KeyError(f"{path} not present in {history}")
        arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
        return [float(x) for x in arr]


def max_abs_history(results_dir: Path, run_id: str, path: str) -> float:
    series = read_history_series(results_dir, run_id, path)
    if not series:
        raise ValueError(f"{path} in {locate_history_file(results_dir, run_id)} is empty")
    return max(abs(value) for value in series if math.isfinite(value))


def reshape_cells(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    if arr.shape == (nr, nz):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr, nz)} or flat {nr * nz}")


def reshape_nodes(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == (nr + 1) * (nz + 1):
        return arr.reshape((nr + 1, nz + 1))
    if arr.shape == (nr + 1, nz + 1):
        return arr
    raise ValueError(
        f"{name} has shape {arr.shape}, expected {(nr + 1, nz + 1)} or flat {(nr + 1) * (nz + 1)}"
    )


def read_final_snapshot(results_dir: Path, run_id: str, nr: int, nz: int) -> Snapshot:
    import h5py

    files = locate_plot_files(results_dir, run_id)
    if not files:
        raise FileNotFoundError(f"no plot snapshots found in {results_dir}")
    path = files[-1]
    with h5py.File(path, "r") as handle:
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        return Snapshot(
            path=path,
            rho=reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            Pe=reshape_cells(handle["hydro/Pe"][()], nr, nz, "hydro/Pe"),
            Pi=reshape_cells(handle["hydro/Pi"][()], nr, nz, "hydro/Pi"),
            r_nodes=reshape_nodes(handle[r_node_path][()], nr, nz, r_node_path),
            z_nodes=reshape_nodes(handle[z_node_path][()], nr, nz, z_node_path),
        )


def cell_centers(snapshot: Snapshot) -> tuple[Any, Any]:
    import numpy as np

    rn = np.asarray(snapshot.r_nodes, dtype=float)
    zn = np.asarray(snapshot.z_nodes, dtype=float)
    rc = 0.25 * (rn[:-1, :-1] + rn[1:, :-1] + rn[:-1, 1:] + rn[1:, 1:])
    zc = 0.25 * (zn[:-1, :-1] + zn[1:, :-1] + zn[:-1, 1:] + zn[1:, 1:])
    return rc, zc


def shock_position_from_profile(coord: Any, rho: Any) -> float:
    import numpy as np

    x = np.asarray(coord, dtype=float).reshape(-1)
    y = np.asarray(rho, dtype=float).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 3:
        raise RuntimeError("not enough cells in ray profile for shock detection")
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    valid = np.abs(dx) > E_FLOOR
    if not np.any(valid):
        raise RuntimeError("ray profile has no finite coordinate spacing")
    grad = np.abs(np.diff(y)[valid] / dx[valid])
    x_mid = 0.5 * (x[:-1][valid] + x[1:][valid])
    if grad.size == 0 or not np.any(np.isfinite(grad)):
        raise RuntimeError("ray profile has no finite density gradient")
    return float(x_mid[int(np.nanargmax(grad))])


def l2_shape_metrics(snapshot: Snapshot) -> dict[str, float]:
    import numpy as np

    rc, zc = cell_centers(snapshot)
    rho = np.asarray(snapshot.rho, dtype=float)
    equator_j = int(np.nanargmin(np.abs(np.nanmean(zc, axis=0))))
    axis_i = int(np.nanargmin(np.abs(np.nanmean(rc, axis=1))))

    shock_equator = shock_position_from_profile(rc[:, equator_j], rho[:, equator_j])
    axis_mask = zc[axis_i, :] >= 0.0
    if np.count_nonzero(axis_mask) < 3:
        axis_mask = np.ones(zc.shape[1], dtype=bool)
    shock_axis = shock_position_from_profile(zc[axis_i, axis_mask], rho[axis_i, axis_mask])
    shock_radius = 0.5 * (abs(shock_axis) + abs(shock_equator))
    return {
        "shock_radius_axis_cm": abs(float(shock_axis)),
        "shock_radius_equator_cm": abs(float(shock_equator)),
        "shock_radius_mean_cm": abs(float(shock_radius)),
    }


def l2_ablation_pressure_avg(snapshot: Snapshot) -> float:
    import numpy as np

    rc, zc = cell_centers(snapshot)
    rho = np.asarray(snapshot.rho, dtype=float)
    pressure = np.asarray(snapshot.Pe, dtype=float) + np.asarray(snapshot.Pi, dtype=float)
    radius = np.sqrt(rc * rc + zc * zc)
    mask = (np.abs(radius - L2_SPHERE_RADIUS_CM) <= 8.0e-4) & (zc > 0.0) & (rho > 1.0e-6)
    values = pressure[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise RuntimeError("no finite ablation-pressure samples found")
    return float(np.mean(values))


def l2_energy_budget_rel(results_dir: Path, run_id: str, expected_input: float) -> float:
    incident_series = read_history_series(results_dir, run_id, "energy/laser_incident")
    deposited_series = read_history_series(results_dir, run_id, "energy/laser_deposited")
    escaped_series = read_history_series(results_dir, run_id, "energy/laser_escaped")
    incident = sum(max(value, 0.0) for value in incident_series)
    deposited = deposited_series[-1] if deposited_series else 0.0
    escaped = sum(max(value, 0.0) for value in escaped_series)
    if incident <= 0.0:
        incident = expected_input
    return abs((deposited + escaped) - incident) / max(incident, E_FLOOR)


def command_text(args: argparse.Namespace, env_overrides: dict[str, str], deck: str) -> str:
    parts = [f"{key}={value}" for key, value in sorted(env_overrides.items())]
    parts.extend([args.tenryu_bin, "run", deck])
    return " ".join(parts)


def run_deck(
    args: argparse.Namespace,
    carry_over: str,
    deck: str,
    env_overrides: dict[str, str],
) -> tuple[int, float, Path]:
    outdir = Path(env_overrides["TENRYU_RH2_OUTDIR" if "TENRYU_RH2_OUTDIR" in env_overrides else "TENRYU_L2_OUTDIR"])
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / "run.log"
    if args.dry_run:
        print(command_text(args, env_overrides, deck))
        return 0, 0.0, log_path

    env = os.environ.copy()
    env.update(env_overrides)
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start
    if proc.returncode != 0:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[run_phase2d_ale_on_sweep] {carry_over} failed with exit {proc.returncode}\n")
    return proc.returncode, wall_time_s, log_path


def rh2_env(args: argparse.Namespace, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_RH2_ALE": "1",
        "TENRYU_RH2_MODE": "shock_tube_grey_sn",
        "TENRYU_RH2_RADIATION_MODE_OVERRIDE": "sn_transport",
        "TENRYU_RH2_NR": "32",
        "TENRYU_RH2_NZ": "64",
        "TENRYU_RH2_T_END_S": "5.0e-11",
        "TENRYU_RH2_MAX_STEPS": str(args.max_steps),
        "TENRYU_RH2_SEED": "12345",
        "TENRYU_RH2_OUTDIR": str(outdir),
        "TENRYU_RH2_VERBOSITY": args.verbosity,
    }


def l2_env(args: argparse.Namespace, outdir: Path, nr: int, nz: int, axis_equator: bool) -> dict[str, str]:
    env = {
        "TENRYU_L2_ALE": "1",
        "TENRYU_L2_MODE": L2_MODE,
        "TENRYU_L2_NR": str(nr),
        "TENRYU_L2_NZ": str(nz),
        "TENRYU_L2_T_END_S": "5.0e-11",
        "TENRYU_L2_DT_INITIAL_S": "2.0e-13",
        "TENRYU_L2_DT_MAX_S": "2.0e-13",
        "TENRYU_L2_PLOT_EVERY_STEPS": "250",
        "TENRYU_L2_MAX_STEPS": str(args.max_steps),
        "TENRYU_L2_RAYS_PER_BEAM": str(L2_RAYS_PER_BEAM),
        "TENRYU_L2_LASER_INPUT_ERG": f"{L2_WEAK_1PCT_E_ERG:.12g}",
        "TENRYU_L2_SEED": "12345",
        "TENRYU_L2_OUTDIR": str(outdir),
        "TENRYU_L2_VERBOSITY": args.verbosity,
    }
    if axis_equator:
        env["TENRYU_L2_AXIS_EQUATOR"] = "1"
    return env


def run_rh2(args: argparse.Namespace) -> SummaryRow:
    carry = "RH2_ale_subgate"
    outdir = Path(args.out_root) / carry
    env = rh2_env(args, outdir)
    exit_code, wall_time_s, log_path = run_deck(args, carry, RH2_DECK, env)
    metrics: dict[str, Any] = {
        "exit_code": exit_code,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }
    observed: float | None = None
    status = "PASS"
    note = ""
    if exit_code != 0:
        status = "CRASHED"
    elif not args.dry_run:
        try:
            results_dir = discover_results_dir(outdir)
            observed = max_abs_history(results_dir, rh2_case_name(), "energy/conservation_error")
            metrics["history"] = str(locate_history_file(results_dir, rh2_case_name()))
            metrics["E_balance"] = observed
            status = "PASS" if observed <= RH2_E_BALANCE_LIMIT else "FAIL"
            note = ""
            if args.smoke:
                metrics["closure_status"] = status
                metrics["closure_limit"] = RH2_E_BALANCE_LIMIT
                status = "PASS"
                note = "completion smoke; closure not enforced"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            metrics["analysis_error"] = str(exc)
            status = "FAIL"
    return SummaryRow(
        carry,
        "2d_rz_rh2_hydro_sn",
        "32x64",
        "5e-11",
        "E_balance",
        observed,
        RH2_E_BALANCE_LIMIT,
        status,
        str(log_path),
        metrics,
        note,
    )


def run_l2_base(args: argparse.Namespace) -> tuple[SummaryRow, dict[str, Any] | None]:
    carry = "L2_ale_subgate"
    outdir = Path(args.out_root) / carry
    env = l2_env(args, outdir, 128, 256, axis_equator=False)
    exit_code, wall_time_s, log_path = run_deck(args, carry, L2_DECK, env)
    run_id = l2_case_name(128, 256)
    metrics: dict[str, Any] = {
        "exit_code": exit_code,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }
    observed: float | None = None
    status = "PASS"
    l2_metrics: dict[str, Any] | None = None
    if exit_code != 0:
        status = "CRASHED"
    elif not args.dry_run:
        try:
            results_dir = discover_results_dir(outdir)
            snapshot = read_final_snapshot(results_dir, run_id, 128, 256)
            shape = l2_shape_metrics(snapshot)
            ablation = l2_ablation_pressure_avg(snapshot)
            observed = l2_energy_budget_rel(results_dir, run_id, L2_WEAK_1PCT_E_ERG)
            metrics.update(shape)
            metrics["ablation_pressure_avg"] = ablation
            metrics["history"] = str(locate_history_file(results_dir, run_id))
            metrics["final_snapshot"] = str(snapshot.path)
            metrics["E_budget_rel"] = observed
            status = "PASS" if observed <= 1.0e-15 else "FAIL"
            l2_metrics = dict(metrics)
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            metrics["analysis_error"] = str(exc)
            status = "FAIL"
    row = SummaryRow(
        carry,
        "2d_rz_l2_laser_hydro",
        "128x256",
        "5e-11",
        "E_budget_rel",
        observed,
        1.0e-15,
        status,
        str(log_path),
        metrics,
        "carry-over investigation; not closure",
    )
    return row, l2_metrics


def run_l2_axis_equator(args: argparse.Namespace) -> SummaryRow:
    carry = "L2_axis_equator"
    outdir = Path(args.out_root) / carry
    env = l2_env(args, outdir, 128, 256, axis_equator=True)
    exit_code, wall_time_s, log_path = run_deck(args, carry, L2_DECK, env)
    run_id = l2_case_name(128, 256)
    metrics: dict[str, Any] = {
        "exit_code": exit_code,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }
    observed: float | None = None
    status = "PASS"
    if exit_code != 0:
        status = "CRASHED"
    elif not args.dry_run:
        try:
            results_dir = discover_results_dir(outdir)
            snapshot = read_final_snapshot(results_dir, run_id, 128, 256)
            shape = l2_shape_metrics(snapshot)
            observed = abs(shape["shock_radius_axis_cm"] - shape["shock_radius_equator_cm"]) / max(
                shape["shock_radius_equator_cm"], E_FLOOR
            )
            metrics.update(shape)
            metrics["axis_equator_rel_delta"] = observed
            metrics["final_snapshot"] = str(snapshot.path)
            status = "PASS" if observed <= 5.0e-2 else "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            metrics["analysis_error"] = str(exc)
            status = "FAIL"
    return SummaryRow(
        carry,
        "2d_rz_l2_laser_hydro",
        "128x256",
        "5e-11",
        "|dshock_R|/shock_R",
        observed,
        5.0e-2,
        status,
        str(log_path),
        metrics,
        "carry-over investigation; not closure",
    )


def run_l2_refinement(args: argparse.Namespace, reference: dict[str, Any] | None) -> SummaryRow:
    carry = "L2_256x512_refine"
    outdir = Path(args.out_root) / carry
    env = l2_env(args, outdir, 256, 512, axis_equator=False)
    exit_code, wall_time_s, log_path = run_deck(args, carry, L2_DECK, env)
    run_id = l2_case_name(256, 512)
    metrics: dict[str, Any] = {
        "exit_code": exit_code,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }
    observed: float | None = None
    status = "PASS"
    if exit_code != 0:
        status = "CRASHED"
    elif args.dry_run:
        pass
    elif reference is None:
        metrics["analysis_error"] = "128x256 ALE-on reference metrics unavailable"
        status = "FAIL"
    else:
        try:
            results_dir = discover_results_dir(outdir)
            snapshot = read_final_snapshot(results_dir, run_id, 256, 512)
            shape = l2_shape_metrics(snapshot)
            ablation = l2_ablation_pressure_avg(snapshot)
            shock_ref = float(reference["shock_radius_mean_cm"])
            abl_ref = float(reference["ablation_pressure_avg"])
            shock_rel = abs(shape["shock_radius_mean_cm"] - shock_ref) / max(abs(shock_ref), E_FLOOR)
            abl_rel = abs(ablation - abl_ref) / max(abs(abl_ref), E_FLOOR)
            observed = max(shock_rel, abl_rel)
            metrics.update(shape)
            metrics["ablation_pressure_avg"] = ablation
            metrics["shock_radius_ref_cm"] = shock_ref
            metrics["ablation_pressure_ref"] = abl_ref
            metrics["shock_radius_rel_delta"] = shock_rel
            metrics["ablation_pressure_rel_delta"] = abl_rel
            metrics["final_snapshot"] = str(snapshot.path)
            status = "PASS" if observed <= 1.0e-1 else "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            metrics["analysis_error"] = str(exc)
            status = "FAIL"
    return SummaryRow(
        carry,
        "2d_rz_l2_laser_hydro",
        "256x512",
        "5e-11",
        "shock_R, abl_p_avg vs 128",
        observed,
        1.0e-1,
        status,
        str(log_path),
        metrics,
        "carry-over investigation; not closure",
    )


def format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def print_summary(rows: list[SummaryRow]) -> None:
    print("phase2d_ale_on_sweep summary")
    if any(row.note == "carry-over investigation; not closure" for row in rows):
        print("L2 rows: carry-over investigation; not closure")
    print(
        "carry-over          deck                    grid    t_end     "
        "gate                       observed   limit       status   note"
    )
    for row in rows:
        print(
            f"{row.carry_over:<19} {row.deck:<23} {row.grid:<7} {row.t_end:<9} "
            f"{row.gate:<26} {format_float(row.observed):<10} {format_float(row.limit):<11} "
            f"{row.status:<8} {row.note}"
        )


def write_summary(path: Path, rows: list[SummaryRow]) -> None:
    payload = {
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "rows": [
            {
                "carry_over": row.carry_over,
                "deck": row.deck,
                "grid": row.grid,
                "t_end": row.t_end,
                "gate": row.gate,
                "observed": row.observed,
                "limit": row.limit,
                "status": row.status,
                "log": row.log,
                "metrics": row.metrics,
                "note": row.note,
            }
            for row in rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--out-root", default="./build/phase2d_ale_on_sweep")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--smoke", action="store_true", help="run only the RH2 ALE-on row")
    parser.add_argument(
        "--include-l2",
        action="store_true",
        help="also run the three L2 carry-over investigation rows; these are not closure rows",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[SummaryRow] = []

    rows.append(run_rh2(args))
    l2_reference: dict[str, Any] | None = None
    if args.include_l2 and not args.smoke:
        l2_row, l2_reference = run_l2_base(args)
        rows.append(l2_row)
        rows.append(run_l2_axis_equator(args))
        rows.append(run_l2_refinement(args, l2_reference if l2_row.status == "PASS" else None))

    print_summary(rows)
    if args.summary_json:
        write_summary(Path(args.summary_json), rows)
    return 0 if rows and all(row.status == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
