#!/usr/bin/env python3
"""Run Stage H0 2D RZ geometry dry validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


STAGE_ID = "H0"
RHO_INIT_GCC = 1.0
R_MAX_CM = 1.0
Z_MIN_CM = -1.0
Z_MAX_CM = 1.0
MASS_DRIFT_MAX = 1.0e-10
AXIS_UR_OVER_CS_MAX = 1.0e-10

# e = 1.5 * k_B_eV * T / (A * m_p), cs = sqrt(gamma * (gamma - 1) * e)
# for gamma=5/3, A=1, T=1.0e-3 eV, k_B_eV=1.6022e-12 erg/eV.
C_S_REF_CM_S = 39953.04184022754


def parse_mesh_list(value: str) -> list[tuple[int, int]]:
    pairs = []
    for part in value.split(";"):
        fields = [item.strip() for item in part.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1]:
            raise argparse.ArgumentTypeError(f"invalid mesh entry: {part!r}")
        nr = int(fields[0])
        nz = int(fields[1])
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("mesh dimensions must be positive")
        pairs.append((nr, nz))
    if not pairs:
        raise argparse.ArgumentTypeError("mesh list is empty")
    return pairs


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def case_name(nr: int, nz: int, seed: int) -> str:
    return f"2d_rz_h0_geom_dry_nr{nr}_nz{nz}_seed{seed}"


def total_box_volume() -> float:
    return np.pi * R_MAX_CM * R_MAX_CM * (Z_MAX_CM - Z_MIN_CM)


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name][()]))


def read_int_scalar(handle: h5py.File, name: str) -> int:
    return int(np.asarray(handle[name][()]))


def discover_results_dir(outdir: Path) -> Path:
    def _has_numbered_plot(d: Path) -> bool:
        return any(len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_" for p in d.glob("*.h5"))

    results_dir = outdir / "results"
    if _has_numbered_plot(results_dir):
        return results_dir

    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if _has_numbered_plot(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no plot files found under {outdir} or {outdir}_*/results")


def locate_plot_files(results_dir: Path, name: str) -> list[Path]:
    return sorted(p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit())


def initial_mass(plot_files: list[Path]) -> float:
    first = plot_files[0] if plot_files else None
    if first is not None:
        with h5py.File(first, "r") as handle:
            t0 = read_scalar(handle, "time_state/t")
            if abs(t0) <= 1.0e-15:
                return float(np.sum(np.asarray(handle["hydro/mass"])))
    return RHO_INIT_GCC * total_box_volume()


def axis_velocity_over_sound_speed(handle: h5py.File, nr: int, nz: int) -> tuple[float, float]:
    x_r = np.asarray(handle["mesh/x_r"])
    v_r = np.asarray(handle["mesh/v_r"])

    if x_r.shape == (nr + 1, nz + 1):
        xr_nodes = x_r
        vr_nodes = v_r.reshape((nr + 1, nz + 1)) if v_r.size == (nr + 1) * (nz + 1) else v_r
        if vr_nodes.shape == xr_nodes.shape:
            column_index = int(np.argmin(np.min(np.abs(xr_nodes), axis=1)))
            max_axis_ur = float(np.max(np.abs(vr_nodes[column_index, :])))
        else:
            max_axis_ur = float(np.max(np.abs(v_r)))
    elif x_r.shape == (nr, nz):
        vr_cells = v_r.reshape((nr, nz)) if v_r.size == nr * nz else v_r
        if vr_cells.shape == x_r.shape:
            column_index = int(np.argmin(np.min(np.abs(x_r), axis=1)))
            max_axis_ur = float(np.max(np.abs(vr_cells[column_index, :])))
        else:
            max_axis_ur = float(np.max(np.abs(v_r)))
    else:
        x_flat = np.ravel(x_r)
        v_flat = np.ravel(v_r)
        if x_flat.size == v_flat.size:
            r_min = float(np.min(np.abs(x_flat)))
            max_axis_ur = float(np.max(np.abs(v_flat[np.isclose(np.abs(x_flat), r_min)])))
        else:
            max_axis_ur = float(np.max(np.abs(v_flat)))

    return max_axis_ur, max_axis_ur / C_S_REF_CM_S


def write_metrics(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_run(args: argparse.Namespace, run: dict[str, object]) -> int:
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
    metrics_ref = run.get("metrics_ref")
    if metrics_ref:
        cmd.extend(["--metrics-ref", str(metrics_ref)])
    return subprocess.run(cmd).returncode


def run_one(args: argparse.Namespace, nr: int, nz: int, seed: int) -> dict[str, object]:
    run_id = f"h0_nr{nr}_nz{nz}_seed{seed}"
    ladder_row = f"nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H0_NR": str(nr),
            "TENRYU_H0_NZ": str(nz),
            "TENRYU_H0_SEED": str(seed),
            "TENRYU_H0_OUTDIR": str(outdir),
        }
    )

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics: dict[str, object] = {
        "mass_initial_g": None,
        "mass_final_g": None,
        "mass_drift": None,
        "max_axis_uR_cm_s": None,
        "c_s_ref_cm_s": C_S_REF_CM_S,
        "axis_uR_over_cs": None,
        "t_final_s": None,
        "step_final": None,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
    }

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

            plot_files = locate_plot_files(results_dir, case_name(nr, nz, seed))
            if len(plot_files) < 2:
                raise RuntimeError(f"expected at least 2 plot files, found {len(plot_files)}")

            m0 = initial_mass(plot_files)
            with h5py.File(plot_files[-1], "r") as handle:
                m1 = float(np.sum(np.asarray(handle["hydro/mass"])))
                mass_drift = abs(m1 - m0) / max(m0, 1.0e-300)
                max_axis_ur, axis_ur_over_cs = axis_velocity_over_sound_speed(handle, nr, nz)
                metrics.update(
                    {
                        "mass_initial_g": m0,
                        "mass_final_g": m1,
                        "mass_drift": mass_drift,
                        "max_axis_uR_cm_s": max_axis_ur,
                        "axis_uR_over_cs": axis_ur_over_cs,
                        "t_final_s": read_scalar(handle, "time_state/t"),
                        "step_final": read_int_scalar(handle, "time_state/step"),
                    }
                )

            if mass_drift > MASS_DRIFT_MAX:
                failed_gates.append("mass_drift")
            if axis_ur_over_cs > AXIS_UR_OVER_CS_MAX:
                failed_gates.append("axis_uR_over_cs")
            if failed_gates:
                result = "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_h0] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/H0/metrics/{run_id}.json"
    metrics_path = Path(rel_metrics_ref)
    payload = {
        "stage_id": STAGE_ID,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "metrics": metrics,
        "thresholds": {
            "mass_drift_max": MASS_DRIFT_MAX,
            "axis_uR_over_cs_max": AXIS_UR_OVER_CS_MAX,
        },
        "result": result,
    }
    write_metrics(metrics_path, payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics["output_bytes"],
        "mass_drift": metrics["mass_drift"],
        "axis_uR_over_cs": metrics["axis_uR_over_cs"],
    }
    append_code = append_run(args, run)
    if append_code != 0 and result == "PASS":
        run["result"] = "FAIL"
        run["failed_gates"] = ["append_run_failure"]
    return run


def print_summary(rows: list[dict[str, object]]) -> None:
    print("Stage H0 thresholds: mass_drift <= 1e-10, axis_uR_over_cs <= 1e-10")
    print("ladder_row seed result mass_drift axis_uR_over_cs wall_time_s")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {mass_drift} {axis_uR_over_cs} {wall_time_s:.3f}".format(
                ladder_row=row["ladder_row"],
                seed=row["seed"],
                result=row["result"],
                mass_drift=row["mass_drift"],
                axis_uR_over_cs=row["axis_uR_over_cs"],
                wall_time_s=float(row["wall_time_s"]),
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_h0_geom_dry.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=parse_mesh_list("32,64;64,128;128,256"))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345,12346,12347,12348,12349"))
    parser.add_argument("--out-root", default="./build/h0_runs")
    parser.add_argument("--run-profile", default="H0 mesh ladder seed sweep")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matrix = [(nr, nz, seed) for nr, nz in args.mesh_list for seed in args.seed_list]

    if args.dry_run:
        print("Planned Stage H0 runs:")
        for nr, nz, seed in matrix:
            print(f"h0_nr{nr}_nz{nz}_seed{seed} nr{nr}_nz{nz} seed={seed}")
        print(f"total_runs={len(matrix)}")
        return 0

    rows = [run_one(args, nr, nz, seed) for nr, nz, seed in matrix]
    print_summary(rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
