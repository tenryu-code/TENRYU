#!/usr/bin/env python3
"""Run the G1 graded-polar theta-invariance calibration matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


TINY = 1.0e-300
GATE_RATIO_MAX = 2.0
GATE_METRIC_MAX = 5.0e-2
CASE_ENV = {
    "SU": ("uniform", "uniform"),
    "SG": ("uniform", "graded"),
    "GU": ("auto", "uniform"),
    "GG": ("auto", "graded"),
}
FIELDS = {
    "rho": "hydro/rho",
    "Te": "hydro/Te",
    "Ti": "hydro/Ti",
}


def parse_cases(value: str) -> tuple[str, ...]:
    cases = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    if not cases:
        raise argparse.ArgumentTypeError("--cases must select at least one case")
    invalid = [case for case in cases if case not in CASE_ENV]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unknown case(s) {','.join(invalid)}; choose from {','.join(CASE_ENV)}"
        )
    if len(set(cases)) != len(cases):
        raise argparse.ArgumentTypeError("--cases must not contain duplicates")
    return cases


def discover_actual_outdir(outdir: Path) -> Path:
    if (outdir / "run_info.json").exists():
        return outdir
    candidates: list[Path] = []
    if outdir.parent.exists():
        for candidate in outdir.parent.glob(f"{outdir.name}_*"):
            if (candidate / "run_info.json").exists():
                candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda candidate: candidate.stat().st_mtime)
    return outdir


def scalar_dataset(handle: Any, name: str, default: float = 0.0) -> float:
    if name not in handle:
        return default
    values = np.asarray(handle[name][()], dtype=float).reshape(-1)
    return float(values[-1]) if values.size else default


def snapshot_time_step(snapshot_path: Path) -> tuple[float, int]:
    import h5py

    with h5py.File(snapshot_path, "r") as handle:
        for dataset in ("mesh/x_r", "mesh/x_z", *FIELDS.values()):
            if dataset not in handle:
                raise KeyError(f"{snapshot_path}: missing dataset {dataset}")
        time_s = float(handle.attrs.get("t", 0.0))
        time_s = scalar_dataset(handle, "time_state/t", time_s)
        step = int(round(scalar_dataset(handle, "time_state/step", 0.0)))
    return time_s, step


def final_snapshot_path(outdir: Path) -> Path:
    actual = discover_actual_outdir(outdir)
    candidates = sorted((actual / "results").glob("*.h5"))
    candidates.extend(sorted(actual.glob("*.h5")))
    if not candidates:
        raise FileNotFoundError(f"no HDF5 snapshots under {actual}")

    readable: list[tuple[float, int, float, Path]] = []
    errors: list[str] = []
    for candidate in candidates:
        try:
            time_s, step = snapshot_time_step(candidate)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            continue
        readable.append((time_s, step, candidate.stat().st_mtime, candidate))
    if not readable:
        detail = "; ".join(errors)
        raise RuntimeError(f"no readable G1 snapshots under {actual}: {detail}")
    return max(readable, key=lambda item: item[:3])[3]


def load_snapshot(snapshot_path: Path) -> dict[str, Any]:
    import h5py

    with h5py.File(snapshot_path, "r") as handle:
        x_r = np.asarray(handle["mesh/x_r"][()], dtype=float)
        x_z = np.asarray(handle["mesh/x_z"][()], dtype=float)
        if x_r.ndim != 2 or x_z.shape != x_r.shape:
            raise ValueError(
                f"{snapshot_path}: mesh coordinate shapes {x_r.shape}, {x_z.shape} "
                "do not identify a single-block 2D polar mesh"
            )
        nr = x_r.shape[0] - 1
        ntheta = x_r.shape[1] - 1
        if nr <= 0 or ntheta <= 0:
            raise ValueError(f"{snapshot_path}: invalid mesh coordinate shape {x_r.shape}")

        # The writer stores single-block nodes as (nr+1, nz+1).  On this polar
        # mesh, hypot(x_r, x_z) advances along axis 0 while atan2(x_r, x_z)
        # spans [0, pi] along axis 1.  Thus flat cell data are (nr, ntheta),
        # with theta as the second axis.
        outer_theta = np.arctan2(x_r[-1, :], x_z[-1, :])
        if float(np.ptp(outer_theta)) <= 0.5 * math.pi:
            raise ValueError(
                f"{snapshot_path}: mesh coordinates do not show theta on axis 1"
            )

        fields: dict[str, np.ndarray] = {}
        for field, dataset in FIELDS.items():
            values = np.asarray(handle[dataset][()], dtype=float)
            if values.shape == (nr, ntheta):
                fields[field] = values
            elif values.size == nr * ntheta:
                fields[field] = values.reshape((nr, ntheta))
            else:
                raise ValueError(
                    f"{snapshot_path}: {dataset} has shape {values.shape}, "
                    f"expected {(nr, ntheta)}"
                )
        step = int(round(scalar_dataset(handle, "time_state/step", 0.0)))

    return {
        "fields": fields,
        "nr": nr,
        "ntheta": ntheta,
        "step": step,
    }


def radial_invariance_max_rel(field2d: np.ndarray, exclude_center: bool) -> float:
    arr = np.asarray(field2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"theta-invariance field must be 2D, got shape {arr.shape}")
    rings = arr[1:, :] if exclude_center else arr
    if rings.shape[0] == 0:
        raise ValueError("theta-invariance metric has no radial rings to measure")
    ring_mean = np.mean(rings, axis=1)
    scale = np.maximum(np.abs(ring_mean), TINY)
    return float(
        np.max(np.abs(rings - ring_mean[:, None]) / scale[:, None])
    )


def run_case(args: argparse.Namespace, case: str) -> dict[str, float | int]:
    radial, theta = CASE_ENV[case]
    outdir = args.out_root / case
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "TENRYU_G1_RADIAL": radial,
            "TENRYU_G1_THETA": theta,
            "TENRYU_G1_CENTER": args.center,
            "TENRYU_G1_OUTDIR": str(outdir),
        }
    )
    log_path = outdir / "run.log"
    cmd = [str(args.tenryu_bin), "run", str(args.deck)]
    print(
        f"[run_g1_polar_graded_invariance] case={case} "
        f"radial={radial} theta={theta} center={args.center}"
    )
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.run(
            cmd,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    wall_s = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"case {case} failed with return code {proc.returncode}; see {log_path}"
        )

    snapshot_path = final_snapshot_path(outdir)
    snapshot = load_snapshot(snapshot_path)
    exclude_center = args.center == "tri_fan"
    result: dict[str, float | int] = {
        field: radial_invariance_max_rel(values, exclude_center)
        for field, values in snapshot["fields"].items()
    }
    result["n_rings_used"] = int(snapshot["nr"] - int(exclude_center))
    result["steps"] = int(snapshot["step"])
    result["wall_s"] = float(wall_s)
    return result


def print_table(summary: dict[str, dict[str, float | int]]) -> None:
    print()
    print(
        f"{'case':<6} {'rho max_rel':>14} {'Te max_rel':>14} "
        f"{'Ti max_rel':>14} {'rings':>7} {'steps':>9} {'wall_s':>10}"
    )
    for case, result in summary.items():
        print(
            f"{case:<6} {float(result['rho']):14.6e} "
            f"{float(result['Te']):14.6e} {float(result['Ti']):14.6e} "
            f"{int(result['n_rings_used']):7d} {int(result['steps']):9d} "
            f"{float(result['wall_s']):10.3f}"
        )


def print_gate_verdict(summary: dict[str, dict[str, float | int]]) -> bool:
    passed = True
    print()
    print(
        f"{'case':<6} {'field':<6} {'metric':>14} "
        f"{'ratio':>14} {'PASS/FAIL':>9}"
    )
    for case in CASE_ENV:
        for field in FIELDS:
            metric = float(summary[case][field])
            control = float(summary["SU"][field])
            if control > 0.0:
                ratio = metric / control
            else:
                ratio = 0.0 if metric == 0.0 else math.inf
            row_passed = metric <= GATE_METRIC_MAX and (
                case == "SU" or ratio <= GATE_RATIO_MAX
            )
            passed = passed and row_passed
            verdict = "PASS" if row_passed else "FAIL"
            print(
                f"{case:<6} {field:<6} {metric:14.6e} "
                f"{ratio:14.6e} {verdict:>9}"
            )
    print("GATE PASS" if passed else "GATE FAIL")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", type=Path, required=True)
    parser.add_argument(
        "--deck",
        type=Path,
        default=Path("examples/verification/2d_rz_polar_graded_uniform_flow.py"),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("build/g1_polar_graded"),
    )
    parser.add_argument(
        "--center",
        choices=("tri_fan", "button"),
        default="tri_fan",
    )
    parser.add_argument(
        "--cases",
        type=parse_cases,
        default="SU,SG,GU,GG",
        help="comma-separated subset of SU,SG,GU,GG",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    if args.gate:
        missing_cases = [case for case in CASE_ENV if case not in args.cases]
        if missing_cases:
            parser.error(
                "--gate requires all cases SU,SG,GU,GG; missing "
                + ",".join(missing_cases)
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | int]] = {}
    for case in args.cases:
        summary[case] = run_case(args, case)

    summary_path = args.summary_json or (args.out_root / "summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print_table(summary)
    if args.center == "tri_fan":
        print("tri_fan center ring i=0 excluded from all invariance metrics")
    print(f"summary_json={summary_path}")
    if args.gate:
        return 0 if print_gate_verdict(summary) else 1
    print("CALIBRATION MODE: no gate thresholds pinned yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
