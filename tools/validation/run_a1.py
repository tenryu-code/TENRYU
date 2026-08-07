#!/usr/bin/env python3
"""Run Stage A1 ALE remap observability sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "A1"
CADENCES = ("quality_only", "every20", "every5", "every1")
ICS = ("sod", "noh", "sedov")
SMOKE_MESHES = ((64, 128), (128, 256))
FULL_MESHES = ((256, 512), (512, 1024))
NOH_SMOKE_MESHES = ((256, 16), (512, 32))
NOH_FULL_MESHES = ((1024, 64), (2048, 128))

NUMBER_RE = r"[-+0-9.eE]+"
REMAP_DELTA_RE = re.compile(
    rf"\[ale-stats\] remap_delta .*?n_applied=(?P<n>\d+) "
    rf".*?dM=(?P<dM>{NUMBER_RE}) .*?dM_rel=(?P<dM_rel>{NUMBER_RE}) "
    rf".*?dE=(?P<dE>{NUMBER_RE}) .*?dE_rel=(?P<dE_rel>{NUMBER_RE})"
)
AXIS_UR_RE = re.compile(rf"\[ale-stats\] axis_uR_max=(?P<value>{NUMBER_RE})")


def parse_mesh_list(value: str) -> list[tuple[int, int]]:
    out = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        sep = "x" if "x" in item.lower() else ","
        parts = [part.strip() for part in item.lower().split(sep)]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid mesh entry: {item!r}")
        nr = int(parts[0])
        nz = int(parts[1])
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("mesh dimensions must be positive")
        out.append((nr, nz))
    if not out:
        raise argparse.ArgumentTypeError("mesh list is empty")
    return out


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


def case_name(ic: str, cadence: str, nr: int, nz: int, seed: int) -> str:
    return f"a1_{ic}_{cadence}_nr{nr}_nz{nz}_seed{seed}"


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def discover_results_dir(outdir: Path) -> Path:
    def has_numbered_plot(path: Path) -> bool:
        return any(
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


def locate_plot_files(results_dir: Path, name: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        return int(path.stem.removeprefix(f"{name}_"))

    return sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=plot_index,
    )


def latest_history_file(results_dir: Path) -> Path | None:
    histories = sorted(results_dir.glob("*_history.h5"), key=lambda path: path.stat().st_mtime)
    return histories[-1] if histories else None


def parse_ale_stats_log(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    remap_rows = []
    for match in REMAP_DELTA_RE.finditer(text):
        remap_rows.append(
            {
                "n": int(match.group("n")),
                "dM": float(match.group("dM")),
                "dM_rel": float(match.group("dM_rel")),
                "dE": float(match.group("dE")),
                "dE_rel": float(match.group("dE_rel")),
            }
        )
    axis_values = [float(match.group("value")) for match in AXIS_UR_RE.finditer(text)]
    if remap_rows:
        n_remaps = max(row["n"] for row in remap_rows)
        mass_delta_abs_max = max(abs(row["dM"]) for row in remap_rows)
        mass_delta_max = max(abs(row["dM_rel"]) for row in remap_rows)
        energy_delta_abs_max = max(abs(row["dE"]) for row in remap_rows)
        energy_delta_max_rel = max(abs(row["dE_rel"]) for row in remap_rows)
    else:
        n_remaps = 0
        mass_delta_abs_max = None
        mass_delta_max = None
        energy_delta_abs_max = None
        energy_delta_max_rel = None
    return {
        "n_remaps_applied": n_remaps,
        "mass_delta_max": mass_delta_max,
        "mass_delta_abs_max": mass_delta_abs_max,
        "energy_delta_abs_max": energy_delta_abs_max,
        "energy_delta_max_rel": energy_delta_max_rel,
        "axis_uR_max": max(axis_values) if axis_values else 0.0,
    }


def mesh_quality_min_from_plot(path: Path, nr: int, nz: int) -> float:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        x_r = np.asarray(handle["mesh/x_r"][()]).reshape((nr + 1, nz + 1)).astype(float)
        x_z = np.asarray(handle["mesh/x_z"][()]).reshape((nr + 1, nz + 1)).astype(float)

    r00 = x_r[:-1, :-1]
    z00 = x_z[:-1, :-1]
    r10 = x_r[1:, :-1]
    z10 = x_z[1:, :-1]
    r11 = x_r[1:, 1:]
    z11 = x_z[1:, 1:]
    r01 = x_r[:-1, 1:]
    z01 = x_z[:-1, 1:]
    area = 0.5 * np.abs(
        r00 * z10 - r10 * z00
        + r10 * z11 - r11 * z10
        + r11 * z01 - r01 * z11
        + r01 * z00 - r00 * z01
    )
    l0 = (r10 - r00) ** 2 + (z10 - z00) ** 2
    l1 = (r11 - r10) ** 2 + (z11 - z10) ** 2
    l2 = (r01 - r11) ** 2 + (z01 - z11) ** 2
    l3 = (r00 - r01) ** 2 + (z00 - z01) ** 2
    denom = np.maximum(l0 + l1 + l2 + l3, 1.0e-300)
    quality = 4.0 * area / denom
    finite = quality[np.isfinite(quality)]
    if finite.size == 0:
        return math.nan
    return float(np.min(finite))


def quality_metrics(plot_files: list[Path], nr: int, nz: int) -> dict[str, Any]:
    values = []
    for path in plot_files:
        try:
            value = mesh_quality_min_from_plot(path, nr, nz)
        except (OSError, KeyError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        return {
            "quality_min_final": None,
            "quality_min_series": [],
            "quality_oscillation_metric": None,
        }
    if len(values) < 2:
        oscillation = 0.0
    else:
        oscillation = max(abs(values[i] - values[i - 1]) for i in range(1, len(values)))
    return {
        "quality_min_final": values[-1],
        "quality_min_series": values,
        "quality_oscillation_metric": oscillation,
    }


def history_metrics(results_dir: Path) -> dict[str, Any]:
    history = latest_history_file(results_dir)
    if history is None:
        return {"ale_invocation_count": None}
    import h5py
    import numpy as np

    with h5py.File(history, "r") as handle:
        if "mesh/ale_rezone_invocations" not in handle:
            return {"ale_invocation_count": None}
        values = np.asarray(handle["mesh/ale_rezone_invocations"][()])
        return {"ale_invocation_count": int(values.reshape(-1)[-1]) if values.size else 0}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_run(args: argparse.Namespace, run: dict[str, Any]) -> int:
    cmd = [
        sys.executable,
        "tools/validation/append_run.py",
        "--stage-id",
        STAGE_ID,
        "--run-id",
        str(run["run_id"]),
        "--ladder-row",
        str(run["ladder_row"]),
        "--seed",
        str(run["seed"]),
        "--deck-path",
        args.deck,
        "--gpu-model",
        args.gpu_model,
        "--cuda-version",
        args.cuda_version,
        "--run-profile",
        args.run_profile,
        "--profile-class",
        args.profile_class,
        "--evidence-role",
        args.evidence_role,
        "--result",
        str(run["result"]),
        "--exit-code",
        str(run["exit_code"]),
        "--wall-time-s",
        f"{run['wall_time_s']:.9g}",
        "--output-bytes",
        str(run["output_bytes"]),
        "--root",
        args.root,
    ]
    failed_gates = run.get("failed_gates") or []
    if failed_gates:
        cmd.extend(["--failed-gates", ",".join(str(gate) for gate in failed_gates)])
    if run.get("metrics_ref"):
        cmd.extend(["--metrics-ref", str(run["metrics_ref"])])
    return subprocess.run(cmd).returncode


def active_env(args: argparse.Namespace, ic: str, cadence: str, nr: int, nz: int, seed: int, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_A1_IC": ic,
        "TENRYU_A1_CADENCE": cadence,
        "TENRYU_A1_NR": str(nr),
        "TENRYU_A1_NZ": str(nz),
        "TENRYU_A1_SEED": str(seed),
        "TENRYU_A1_OUTDIR": str(outdir),
        "TENRYU_A1_DEBUG_PER_REMAP_LOG": "1",
        "TENRYU_A1_MAX_STEPS": str(args.max_steps),
    }


def empty_metrics(wall_time_s: float, bytes_out: int) -> dict[str, Any]:
    return {
        "n_remaps_applied": 0,
        "ale_invocation_count": None,
        "mass_delta_max": None,
        "mass_delta_abs_max": None,
        "energy_delta_abs_max": None,
        "energy_delta_max_rel": None,
        "axis_uR_max": 0.0,
        "quality_min_final": None,
        "quality_min_series": [],
        "quality_oscillation_metric": None,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }


def run_one(args: argparse.Namespace, ic: str, cadence: str, nr: int, nz: int, seed: int) -> dict[str, Any]:
    run_id = case_name(ic, cadence, nr, nz, seed)
    ladder_row = f"{ic}_{cadence}_nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(active_env(args, ic, cadence, nr, nz, seed, outdir))

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics = empty_metrics(wall_time_s, output_bytes(outdir))
    metrics.update(parse_ale_stats_log(log_path))

    if proc.returncode != 0:
        result = "CRASHED"
        failed_gates = ["run_failure"]
    else:
        try:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            metrics["output_bytes"] = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                metrics["output_bytes"] += output_bytes(outdir)
            plot_files = locate_plot_files(results_dir, run_id)
            if not plot_files:
                raise RuntimeError("expected at least one plot file")
            metrics.update(quality_metrics(plot_files, nr, nz))
            metrics.update(history_metrics(results_dir))
            metrics.update(parse_ale_stats_log(log_path))
            metrics["wall_time_s"] = wall_time_s
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_a1] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "stage_id": STAGE_ID,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "ic": ic,
        "cadence": cadence,
        "nr": nr,
        "nz": nz,
        "metrics": metrics,
        "env_overrides": active_env(args, ic, cadence, nr, nz, seed, outdir),
        "result": result,
        "failed_gates": failed_gates,
    }
    write_json(Path(rel_metrics_ref), payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "ic": ic,
        "cadence": cadence,
        "nr": nr,
        "nz": nz,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics["output_bytes"],
        "n_remaps_applied": metrics["n_remaps_applied"],
        "mass_delta_max": metrics["mass_delta_max"],
        "energy_delta_max_rel": metrics["energy_delta_max_rel"],
        "axis_uR_max": metrics["axis_uR_max"],
        "quality_oscillation_metric": metrics["quality_oscillation_metric"],
    }
    append_code = append_run(args, run)
    if append_code != 0 and result == "PASS":
        run["result"] = "FAIL"
        run["failed_gates"] = ["append_run_failure"]
    return run


def meshes_for_ic(args: argparse.Namespace, ic: str) -> list[tuple[int, int]]:
    if args.mesh_list:
        return list(args.mesh_list)

    meshes = list(NOH_SMOKE_MESHES if ic == "noh" else SMOKE_MESHES)
    if args.matrix_full:
        meshes.extend(NOH_FULL_MESHES if ic == "noh" else FULL_MESHES)
    return meshes


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
    }
    for key in (
        "n_remaps_applied",
        "mass_delta_max",
        "energy_delta_max_rel",
        "axis_uR_max",
        "quality_oscillation_metric",
        "wall_time_s",
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
    print("Stage A1 ALE sweep summary")
    print("ladder_row seed result n_remaps mass_delta_max energy_delta_max_rel axis_uR_max quality_osc wall_time_s")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {n_remaps_applied} {mass_delta_max} "
            "{energy_delta_max_rel} {axis_uR_max} {quality_oscillation_metric} {wall_time_s:.3f}".format(
                **row
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_a1_ale_sweep.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/a1_runs")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--run-profile", default="A1 ALE remap smoke sweep")
    parser.add_argument("--ic-list", type=lambda v: parse_csv(v, ICS, "IC"), default=list(ICS))
    parser.add_argument("--cadence-list", type=lambda v: parse_csv(v, CADENCES, "cadence"), default=list(CADENCES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--matrix-full", action="store_true", help="include full-resolution meshes")
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/A1/metrics/a1_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    matrix = [
        (ic, cadence, nr, nz, seed)
        for ic in args.ic_list
        for cadence in args.cadence_list
        for nr, nz in meshes_for_ic(args, ic)
        for seed in args.seed_list
    ]

    if args.dry_run:
        print("Planned Stage A1 runs:")
        for ic, cadence, nr, nz, seed in matrix:
            run_id = case_name(ic, cadence, nr, nz, seed)
            print(f"{run_id} ic={ic} cadence={cadence} nr={nr} nz={nz} seed={seed}")
        print(f"total_runs={len(matrix)}")
        print(f"summary_json={args.summary_json}")
        return 0

    rows = [run_one(args, ic, cadence, nr, nz, seed) for ic, cadence, nr, nz, seed in matrix]
    print_summary(rows)
    write_summary(args, rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
