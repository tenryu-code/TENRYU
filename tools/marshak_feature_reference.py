#!/usr/bin/env python3
"""One-dimensional Marshak feature-crossing reference checker."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


CASE_NAME = "marshak_feature_1d"
THRESHOLD_TE_EV = 1240.0
PROBES_CM = (0.005, 0.010)
BASE_DX_CM = 0.04 / 512.0
REQUIRED_ROLES = (
    "flk_dt0",
    "flk_dt1",
    "flk_dt2",
    "flk_dt3",
    "qxp_dt0",
    "qxp_dt1",
    "qxp_dt2",
    "qxp_dt3",
    "ref",
)


class StopCheck(RuntimeError):
    pass


def stop(message: str) -> None:
    raise StopCheck(message)


def as_scalar(value: Any, name: str) -> Any:
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    if array.size == 1:
        return array.reshape(-1)[0].item()
    stop(f"{name} must be scalar, got shape {array.shape}")


def require_dataset(handle: h5py.File, name: str, source: Path) -> h5py.Dataset:
    obj = handle.get(name)
    if obj is None:
        stop(f"{source}: missing required dataset '{name}'")
    if not isinstance(obj, h5py.Dataset):
        stop(f"{source}: expected dataset at '{name}'")
    return obj


def snapshot_index(path: Path) -> int:
    suffix = path.name.rsplit("_", 1)[-1][:-3]
    if not suffix.isdigit():
        stop(f"{path}: snapshot suffix is not all digits")
    return int(suffix)


def snapshot_paths(run_dir: Path) -> list[Path]:
    results_dir = run_dir / "results"
    if not results_dir.is_dir():
        stop(f"{run_dir}: results directory not found: {results_dir}")
    paths = []
    for path in results_dir.glob(f"{CASE_NAME}_[0-9]*.h5"):
        if not path.is_file():
            continue
        suffix = path.name.rsplit("_", 1)[-1][:-3]
        if suffix.isdigit():
            paths.append(path)
    paths.sort(key=snapshot_index)
    if not paths:
        stop(
            f"{run_dir}: no numeric-suffix snapshots matched "
            f"'{CASE_NAME}_[0-9]*.h5' in {results_dir}"
        )
    return paths


def read_time(handle: h5py.File, source: Path) -> float:
    if "t" in handle.attrs:
        value = float(as_scalar(handle.attrs["t"], "root attribute 't'"))
    elif "time_state/t" in handle:
        dataset = require_dataset(handle, "time_state/t", source)
        value = float(as_scalar(dataset[()], "dataset 'time_state/t'"))
    else:
        stop(f"{source}: time attribute 't' and fallback dataset 'time_state/t' not found")
    if not math.isfinite(value):
        stop(f"{source}: snapshot time is not finite")
    return value


@dataclass
class LegData:
    role: str
    run_dir: Path
    times: np.ndarray
    fronts: np.ndarray
    arrivals: tuple[float, float]
    endpoint: float
    max_te: float
    dx: float


def arrival_time(times: np.ndarray, fronts: np.ndarray, probe: float) -> float:
    crossings = np.flatnonzero(fronts >= probe)
    if crossings.size == 0 or crossings[0] == 0:
        return math.nan
    upper = int(crossings[0])
    lower = upper - 1
    s0 = float(fronts[lower])
    s1 = float(fronts[upper])
    if s1 <= s0:
        return math.nan
    fraction = (probe - s0) / (s1 - s0)
    return float(times[lower] + fraction * (times[upper] - times[lower]))


def read_leg(role: str, run_dir: Path) -> LegData:
    times: list[float] = []
    fronts: list[float] = []
    max_te = -math.inf
    dx = math.nan
    expected_cells: int | None = None

    for path in snapshot_paths(run_dir):
        with h5py.File(path, "r") as handle:
            time = read_time(handle, path)
            x_r = np.asarray(
                require_dataset(handle, "mesh/x_r", path)[...], dtype=np.float64
            )
            te = np.asarray(
                require_dataset(handle, "hydro/Te", path)[...], dtype=np.float64
            )
        if x_r.ndim != 1:
            stop(f"{path}: mesh/x_r must be rank 1")
        if te.ndim != 1:
            stop(f"{path}: hydro/Te must be rank 1")
        if x_r.size != te.size + 1:
            stop(
                f"{path}: mesh/x_r has {x_r.size} nodes but hydro/Te has "
                f"{te.size} cells"
            )
        if x_r.size < 2 or np.any(~np.isfinite(x_r)) or np.any(np.diff(x_r) <= 0.0):
            stop(f"{path}: mesh/x_r is empty, non-finite, or not strictly increasing")
        if np.any(~np.isfinite(te)):
            stop(f"{path}: hydro/Te contains non-finite values")
        if expected_cells is None:
            expected_cells = te.size
            dx = float(np.max(np.diff(x_r)))
        elif te.size != expected_cells:
            stop(f"{path}: cell count changed within leg {role}")

        centers = 0.5 * (x_r[:-1] + x_r[1:])
        distance = float(x_r[-1]) - centers
        hot = distance[te >= THRESHOLD_TE_EV]
        front = float(np.max(hot)) if hot.size else 0.0
        times.append(time)
        fronts.append(front)
        max_te = max(max_te, float(np.max(te)))

    time_array = np.asarray(times, dtype=float)
    front_array = np.asarray(fronts, dtype=float)
    if np.any(np.diff(time_array) <= 0.0):
        stop(f"{role}: snapshot times are not strictly increasing in numeric suffix order")
    arrivals = tuple(
        arrival_time(time_array, front_array, probe) for probe in PROBES_CM
    )
    return LegData(
        role=role,
        run_dir=run_dir,
        times=time_array,
        fronts=front_array,
        arrivals=arrivals,
        endpoint=float(front_array[-1]),
        max_te=max_te,
        dx=dx,
    )


def parse_legs(values: list[str]) -> dict[str, Path]:
    legs: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            stop(f"--leg expects role=path, got {raw!r}")
        role, raw_path = raw.split("=", 1)
        if role not in REQUIRED_ROLES:
            stop(
                f"unknown --leg role {role!r}; expected one of "
                f"{', '.join(REQUIRED_ROLES)}"
            )
        if role in legs:
            stop(f"duplicate --leg role {role!r}")
        if not raw_path:
            stop(f"--leg {role}: empty path")
        legs[role] = Path(raw_path)
    missing = [role for role in REQUIRED_ROLES if role not in legs]
    if missing:
        stop(f"missing required --leg roles: {', '.join(missing)}")
    return legs


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def run_check(args: argparse.Namespace) -> int:
    leg_paths = parse_legs(args.leg)
    runs = {role: read_leg(role, leg_paths[role]) for role in REQUIRED_ROLES}
    ref = runs["ref"]
    errors = {
        role: abs(run.endpoint - ref.endpoint) for role, run in runs.items()
    }

    print("1-D Marshak feature-crossing reference")
    print(
        "role       t_c(0.005 cm) [s]  t_c(0.010 cm) [s]  "
        "s_c(t_end) [cm]  err vs ref [cm]"
    )
    for role in REQUIRED_ROLES:
        run = runs[role]
        print(
            f"{role:9s}  {run.arrivals[0]:18.8e}  {run.arrivals[1]:18.8e}  "
            f"{run.endpoint:16.8e}  {errors[role]:16.8e}"
        )

    print()
    print("qxp/flk arrival-time-error ratios per rung (measurement -- no band)")
    for rung in range(4):
        flk = runs[f"flk_dt{rung}"]
        qxp = runs[f"qxp_dt{rung}"]
        values = []
        for probe_index in range(len(PROBES_CM)):
            flk_error = abs(flk.arrivals[probe_index] - ref.arrivals[probe_index])
            qxp_error = abs(qxp.arrivals[probe_index] - ref.arrivals[probe_index])
            values.append(ratio(qxp_error, flk_error))
        print(
            f"  dt{rung}: probe 0.005 cm={values[0]:.8e}, "
            f"probe 0.010 cm={values[1]:.8e}"
        )

    split = abs(runs["qxp_dt3"].endpoint - runs["flk_dt3"].endpoint)
    split_ok = split <= 2.0 * BASE_DX_CM
    fine_fleck_ok = errors["flk_dt3"] <= 4.0 * BASE_DX_CM
    fine_qxp_ok = errors["qxp_dt3"] <= 4.0 * BASE_DX_CM
    gate_a = split_ok and fine_fleck_ok and fine_qxp_ok
    print()
    print("(A) SPLITTING CONSISTENCY [binding]")
    print(
        f"  |s_qxp_dt3-s_flk_dt3|={split:.8e} <= "
        f"2*dx={2.0 * BASE_DX_CM:.8e}: {status(split_ok)}"
    )
    print(
        f"  flk_dt3 err={errors['flk_dt3']:.8e}, "
        f"qxp_dt3 err={errors['qxp_dt3']:.8e} <= "
        f"4*dx={4.0 * BASE_DX_CM:.8e}: "
        f"{status(fine_fleck_ok and fine_qxp_ok)}"
    )
    print(f"  gate (A): {status(gate_a)}")

    gate_b = True
    print()
    print("(B) NON-DEGRADATION [binding]")
    for rung in range(4):
        flk_role = f"flk_dt{rung}"
        qxp_role = f"qxp_dt{rung}"
        limit = max(errors[flk_role] * 1.10, 2.0 * BASE_DX_CM)
        rung_ok = errors[qxp_role] <= limit
        gate_b = gate_b and rung_ok
        print(
            f"  dt{rung}: err(qxp)={errors[qxp_role]:.8e} <= "
            f"max(1.10*err(flk),2*dx)={limit:.8e}: {status(rung_ok)}"
        )
    print(f"  gate (B): {status(gate_b)}")

    gate_c = True
    print()
    print("(C) SANITY")
    for role in REQUIRED_ROLES:
        run = runs[role]
        monotone_ok = bool(np.all(np.diff(run.fronts) >= -run.dx))
        max_te_ok = run.max_te <= 1600.0
        arrival_ok = math.isfinite(run.arrivals[0])
        leg_ok = monotone_ok and max_te_ok and arrival_ok
        gate_c = gate_c and leg_ok
        print(
            f"  {role}: monotone within dx={run.dx:.8e}: {status(monotone_ok)}; "
            f"max Te={run.max_te:.8e} <= 1600 eV: {status(max_te_ok)}; "
            f"t_c(0.005) finite: {status(arrival_ok)}"
        )
    print(f"  gate (C): {status(gate_c)}")

    all_pass = gate_a and gate_b and gate_c
    print()
    print(f"Overall: {status(all_pass)}")
    return 0 if all_pass else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--leg",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="run root tagged with one of the required roles (repeatable)",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_check(args)
    except StopCheck as exc:
        print(f"STOP: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
