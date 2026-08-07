#!/usr/bin/env python3
"""Run the I1 radial 1D 5% GXII grey-FLD production gate."""

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


STAGE_ID = "I1"
MODE = "i1_radial_5pct_grey_fld"
NR = 256
RAYS_PER_BEAM = 128
SEED_DEFAULT = 12345
T_END_S = 5.0e-11
DT_S = 5.0e-14
GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
LASER_INPUT_ERG = GXII_REFERENCE_E_ERG * 0.05
KAPPA_A_CM2_G = 10.0
ENERGY_RESIDUAL_MAX = 3.0e-3
E_FLOOR = 1.0e-300
NUMBER_RE = r"[-+0-9.eE]+"
DECK_LINE_RE = re.compile(
    r"\[deck:i1_radial_1d_5pct_grey_fld\].*?"
    r"t_end_s=(?P<t_end>{n}).*?laser_input_erg=(?P<laser>{n}).*?"
    r"kappa_a_cm2_g=(?P<kappa>{n})".format(n=NUMBER_RE)
)
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")


@dataclass(frozen=True)
class Snapshot:
    path: str
    t: float
    step: int
    rho: Any
    Te: Any
    Ti: Any
    ee: Any
    ei: Any
    vol: Any
    mass: Any
    v_r: Any
    rad_E: Any
    E_laser_deposited_state: float
    E_laser_escaped_state: float
    E_rad_escaped_state: float


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def case_name(seed: int = SEED_DEFAULT) -> str:
    return (
        f"i1_radial_1d_5pct_grey_fld_nr{NR}_rays{RAYS_PER_BEAM}"
        f"_e{safe_float_token(LASER_INPUT_ERG)}_ka{safe_float_token(KAPPA_A_CM2_G)}_seed{seed}"
    )


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
    if outdir.parent.exists():
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
            t_value = scalar_dataset(handle, "time_state/t", t_value)
        rad = np.asarray(handle["radiation/energy_density"][()], dtype=float)
        if rad.ndim == 2:
            rad = np.sum(rad, axis=1)
        return Snapshot(
            path=str(path),
            t=t_value,
            step=int(scalar_dataset(handle, "time_state/step", 0.0)),
            rho=np.asarray(handle["hydro/rho"][()], dtype=float),
            Te=np.asarray(handle["hydro/Te"][()], dtype=float),
            Ti=np.asarray(handle["hydro/Ti"][()], dtype=float),
            ee=np.asarray(handle["hydro/ee"][()], dtype=float),
            ei=np.asarray(handle["hydro/ei"][()], dtype=float),
            vol=np.asarray(handle["hydro/vol"][()], dtype=float),
            mass=np.asarray(handle["hydro/mass"][()], dtype=float),
            v_r=np.asarray(handle["mesh/v_r"][()], dtype=float),
            rad_E=rad,
            E_laser_deposited_state=scalar_dataset(handle, "time_state/E_laser_deposited", 0.0),
            E_laser_escaped_state=scalar_dataset(handle, "time_state/E_laser_escaped", 0.0),
            E_rad_escaped_state=scalar_dataset(handle, "time_state/E_rad_escaped", 0.0),
        )


def read_snapshots(results_dir: Path, run_id: str) -> list[Snapshot]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
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
                return [float(x) for x in np.asarray(handle[path][()], dtype=float).reshape(-1)]
    return []


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out: dict[str, Any] = {"n_clamp": sum(int(match.group("count")) for match in CLAMP_RE.finditer(text))}
    deck_match = DECK_LINE_RE.search(text)
    if deck_match:
        out["deck_t_end_s"] = float(deck_match.group("t_end"))
        out["deck_laser_input_erg"] = float(deck_match.group("laser"))
        out["deck_kappa_a_cm2_g"] = float(deck_match.group("kappa"))
    return out


def compute_U_internal_total(snapshot: Snapshot) -> float:
    import numpy as np

    return float(np.sum(snapshot.rho * (snapshot.ee + snapshot.ei) * snapshot.vol))


def compute_K_cell_centered_1d(snapshot: Snapshot) -> float:
    import numpy as np

    v_r = np.asarray(snapshot.v_r, dtype=float)
    mass = np.asarray(snapshot.mass, dtype=float)
    if v_r.size != mass.size + 1:
        return 0.0
    u = 0.5 * (v_r[:-1] + v_r[1:])
    return float(0.5 * np.sum(mass * u * u))


def compute_E_rad_total(snapshot: Snapshot) -> float:
    import numpy as np

    return float(np.sum(np.asarray(snapshot.rad_E, dtype=float) * snapshot.vol))


def effective_laser_energies(results_dir: Path, run_id: str, final: Snapshot) -> tuple[float, float, float]:
    incident_series = read_history_series(results_dir, run_id, "energy/laser_incident")
    escaped_series = read_history_series(results_dir, run_id, "energy/laser_escaped")
    deposited_series = read_history_series(results_dir, run_id, "energy/laser_deposited")
    incident = sum(max(x, 0.0) for x in incident_series)
    escaped = sum(max(x, 0.0) for x in escaped_series)
    deposited = deposited_series[-1] if deposited_series else final.E_laser_deposited_state
    if incident > 0.0:
        return incident, deposited, escaped
    state_total = final.E_laser_deposited_state + final.E_laser_escaped_state
    if state_total > 0.0:
        return state_total, final.E_laser_deposited_state, final.E_laser_escaped_state
    return LASER_INPUT_ERG, final.E_laser_deposited_state, final.E_laser_escaped_state


def final_status(exit_code: int, final_t: float | None, t_end_s: float) -> str:
    if exit_code != 0:
        return "crashed"
    if final_t is None or final_t < 0.999 * t_end_s:
        return "dt_collapse"
    return "completed"


def compute_metrics(
    snapshots: list[Snapshot],
    results_dir: Path,
    run_id: str,
    log_metrics: dict[str, Any],
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    first = snapshots[0]
    final = snapshots[-1]
    u0 = compute_U_internal_total(first)
    u1 = compute_U_internal_total(final)
    k0 = compute_K_cell_centered_1d(first)
    k1 = compute_K_cell_centered_1d(final)
    rad0 = compute_E_rad_total(first)
    rad1 = compute_E_rad_total(final)
    laser_in, laser_dep, laser_esc = effective_laser_energies(results_dir, run_id, final)
    rad_escaped_series = read_history_series(results_dir, run_id, "energy/radiation_escaped")
    rad_escaped = final.E_rad_escaped_state if final.E_rad_escaped_state != 0.0 else sum(
        max(x, 0.0) for x in rad_escaped_series
    )
    initial_total = u0 + k0 + rad0
    final_total_with_escape = u1 + k1 + rad1 + rad_escaped + laser_esc
    energy_residual = abs(final_total_with_escape - (initial_total + laser_in)) / max(
        abs(initial_total + laser_in), E_FLOOR
    )

    rho_all = np.concatenate([np.asarray(s.rho, dtype=float).reshape(-1) for s in snapshots])
    te_all = np.concatenate([np.asarray(s.Te, dtype=float).reshape(-1) for s in snapshots])
    ti_all = np.concatenate([np.asarray(s.Ti, dtype=float).reshape(-1) for s in snapshots])
    rad_final = np.asarray(final.rad_E, dtype=float)
    history_sample_count = max(
        len(read_history_series(results_dir, run_id, "t")),
        len(read_history_series(results_dir, run_id, "cycle")),
        len(read_history_series(results_dir, run_id, "dt")),
        len(read_history_series(results_dir, run_id, "energy/laser_incident")),
        len(read_history_series(results_dir, run_id, "energy/radiation_escaped")),
    )
    metrics: dict[str, Any] = {
        "final_t_s": final.t,
        "final_step": final.step,
        "completed_fraction": final.t / max(float(log_metrics.get("deck_t_end_s", T_END_S)), E_FLOOR),
        "E_initial_matter_rad": initial_total,
        "E_final_matter_rad_escaped": final_total_with_escape,
        "E_laser_incident": laser_in,
        "E_laser_deposited": laser_dep,
        "E_laser_escaped": laser_esc,
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "E_rad_escaped": rad_escaped,
        "energy_residual": energy_residual,
        "rho_min": float(np.nanmin(rho_all)),
        "rho_max": float(np.nanmax(rho_all)),
        "Te_min_eV": float(np.nanmin(te_all)),
        "Ti_min_eV": float(np.nanmin(ti_all)),
        "nan_count_rho_T": int(
            np.count_nonzero(~np.isfinite(rho_all))
            + np.count_nonzero(~np.isfinite(te_all))
            + np.count_nonzero(~np.isfinite(ti_all))
        ),
        "final_rad_E_bad_count": int(np.count_nonzero(~np.isfinite(rad_final)) + np.count_nonzero(rad_final < 0.0)),
        "snapshot_count": len(snapshots),
        "history_sample_count": history_sample_count,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(log_metrics)
    return metrics


def evaluate_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if metrics.get("run_status") != "completed":
        failed.append("i1_radial_completion")
    if metrics.get("energy_residual", math.inf) > ENERGY_RESIDUAL_MAX:
        failed.append("i1_radial_energy_residual")
    if metrics.get("nan_count_rho_T", 1) != 0:
        failed.append("i1_radial_no_nan_rho_T")
    if metrics.get("final_rad_E_bad_count", 1) != 0:
        failed.append("i1_radial_rad_E_finite_nonnegative")
    if metrics.get("rho_min", 0.0) <= 0.0:
        failed.append("i1_radial_positive_density")
    return failed


def active_env(args: argparse.Namespace, seed: int, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_I1_RADIAL_NR": str(NR),
        "TENRYU_I1_RADIAL_SEED": str(seed),
        "TENRYU_I1_RADIAL_OUTDIR": str(outdir),
        "TENRYU_I1_RADIAL_RAYS_PER_BEAM": str(RAYS_PER_BEAM),
        "TENRYU_I1_RADIAL_LASER_INPUT_ERG": str(LASER_INPUT_ERG),
        "TENRYU_I1_RADIAL_PULSE_DURATION_S": "1.0e-9",
        "TENRYU_I1_RADIAL_KAPPA_A_CM2_G": str(KAPPA_A_CM2_G),
        "TENRYU_I1_RADIAL_RHO_BACKGROUND_GCC": "1.0e-3",
        "TENRYU_I1_RADIAL_T_END_S": str(args.t_end_s),
        "TENRYU_I1_RADIAL_DT_INITIAL_S": str(args.dt_s),
        "TENRYU_I1_RADIAL_DT_MAX_S": str(args.dt_s),
        "TENRYU_I1_RADIAL_PLOT_EVERY_STEPS": str(args.plot_every_steps),
        "TENRYU_I1_RADIAL_HISTORY_EVERY_STEPS": "1",
        "TENRYU_I1_RADIAL_MAX_STEPS": str(args.max_steps),
        "TENRYU_I1_RADIAL_VERBOSITY": args.verbosity,
    }


def run_one(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    run_id = case_name(seed)
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    wall_time_s = time.monotonic() - start

    log_metrics = parse_log_metadata(log_path)
    metrics: dict[str, Any] = {**log_metrics, "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    failed_gates: list[str] = []
    failure_class: str | None = None
    result = "PASS"

    try:
        if timed_out:
            metrics["run_status"] = "wall_time_timeout"
            result = "FAIL"
            failed_gates = ["run_failure"]
            failure_class = "run_failure"
        elif proc.returncode != 0:
            metrics["run_status"] = final_status(proc.returncode, None, args.t_end_s)
            result = "CRASHED"
            failed_gates = ["run_failure"]
            failure_class = "run_failure"
        else:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            snapshots = read_snapshots(results_dir, run_id)
            metrics = compute_metrics(snapshots, results_dir, run_id, log_metrics, wall_time_s, bytes_out)
            metrics["run_status"] = final_status(proc.returncode, metrics.get("final_t_s"), args.t_end_s)
            failed_gates = evaluate_gates(metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["analysis_error"] = str(exc)
        metrics["run_status"] = "dt_collapse" if proc.returncode == 0 else "crashed"
        result = "FAIL"
        failed_gates = ["analysis_failure"]
        failure_class = "analysis_failure"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[run_i1_radial] analysis failed: {exc}\n")

    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": MODE,
        "seed": seed,
        "deck_path": args.deck,
        "mode": MODE,
        "nr": NR,
        "nz": 1,
        "rays_per_beam": RAYS_PER_BEAM,
        "groups": 1,
        "kappa_a_cm2_g": KAPPA_A_CM2_G,
        "mode_energy_fraction": 0.05,
        "mode_pulse_duration_s": 1.0e-9,
        "informational": False,
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
        "informational": payload["informational"],
        "nr": payload["nr"],
        "nz": payload["nz"],
        "rays_per_beam": payload["rays_per_beam"],
        "groups": payload["groups"],
        "kappa_a_cm2_g": payload["kappa_a_cm2_g"],
        "mode_energy_fraction": payload["mode_energy_fraction"],
        "mode_pulse_duration_s": payload["mode_pulse_duration_s"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "run_status": metrics.get("run_status"),
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "final_t_s": metrics.get("final_t_s"),
        "final_step": metrics.get("final_step"),
        "energy_residual": metrics.get("energy_residual"),
        "E_laser_incident": metrics.get("E_laser_incident"),
        "E_laser_deposited": metrics.get("E_laser_deposited"),
        "E_laser_escaped": metrics.get("E_laser_escaped"),
        "E_rad_escaped": metrics.get("E_rad_escaped"),
        "n_clamp": metrics.get("n_clamp"),
        "rho_min": metrics.get("rho_min"),
        "rho_max": metrics.get("rho_max"),
        "Te_min_eV": metrics.get("Te_min_eV"),
        "Ti_min_eV": metrics.get("Ti_min_eV"),
    }


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    values = [
        float(row["energy_residual"])
        for row in rows
        if isinstance(row.get("energy_residual"), (int, float)) and math.isfinite(float(row["energy_residual"]))
    ]
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": {
            "total_runs": len(rows),
            "gating_runs": len(rows),
            "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
            "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
            "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
            "energy_residual_max": max(values) if values else None,
            "energy_residual_gate": ENERGY_RESIDUAL_MAX,
        },
    }
    write_json(Path(args.summary_json), payload)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Stage I1 radial 1D 5% GXII grey-FLD summary")
    columns = [
        "ladder_row",
        "seed",
        "result",
        "run_status",
        "final_t_s",
        "final_step",
        "energy_residual",
        "E_laser_incident",
        "E_laser_deposited",
        "E_laser_escaped",
        "E_rad_escaped",
        "failed_gates",
    ]
    print(" ".join(columns))
    for row in rows:
        print(" ".join(str(row.get(column)) for column in columns))


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/i1_radial_1d_5pct_grey_fld.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/i1_radial_runs")
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--t-end-s", type=float, default=T_END_S)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--plot-every-steps", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/I1/radial_stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/I1/radial_runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [run_one(args, seed) for seed in args.seed_list]
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    print_summary(rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
