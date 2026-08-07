#!/usr/bin/env python3
"""Run the A1 hydro-only ICF-shell axis-floor prep sweep."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any


AXIS_FLOORS_DEFAULT = (0.05, 0.10, 0.20, 0.50)
DECK = "examples/verification/2d_rz_a1_icf_shell_hydro.py"


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("float list is empty")
    for value in values:
        if value < 0.0 or value > 1.0:
            raise argparse.ArgumentTypeError("axis-floor values must be in [0, 1]")
    return values


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def run_id(args: argparse.Namespace, axis_floor: float) -> str:
    return (
        f"a1_icf_shell_hydro_nr{args.nr}_nz{args.nz}_seed{args.seed}"
        f"_axisdt{safe_float_token(axis_floor)}"
    )


def env_for(args: argparse.Namespace, axis_floor: float, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_A1_NR": str(args.nr),
        "TENRYU_A1_NZ": str(args.nz),
        "TENRYU_A1_SEED": str(args.seed),
        "TENRYU_A1_OUTDIR": str(outdir),
        "TENRYU_A1_T_END_S": f"{args.t_end_s:.12g}",
        "TENRYU_A1_MAX_STEPS": str(args.max_steps),
        "TENRYU_A1_CFL": f"{args.cfl:.12g}",
        "TENRYU_A1_IMPL_VELOCITY_CM_S": f"{args.impl_velocity_cm_s:.12g}",
        "TENRYU_A1_ALE_EVERY_N_STEPS": "5",
        "TENRYU_A1_AXIS_MOTION_FLOOR_FRACTION": f"{args.axis_motion_floor:.12g}",
        "TENRYU_A1_AXIS_MARGIN_DT_FLOOR_FRACTION": f"{axis_floor:.12g}",
        "TENRYU_A1_PREVENTIVE_AXIS_GUARD_FRACTION": f"{args.preventive_axis_guard:.12g}",
        "TENRYU_A1_AXIS_REPAIR_MODE": "axis_spine_only",
        "TENRYU_A1_REMAP_DAMAGE_GATE": "1",
        "TENRYU_A1_REMAP_DAMAGE_DMAX": "0.05",
        "TENRYU_A1_REMAP_DAMAGE_AXIS_ETA": "0.02",
        "TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET": "1",
        "TENRYU_A1_REMAP_DAMAGE_AXIS_BUDGET_FACTOR": "2.0",
        "TENRYU_A1_REMAP_SCHEME": "ms2_moments",
        "TENRYU_A1_REMAP_MS2_LIMITER": "van_leer",
        "TENRYU_A1_KE_CLOSURE": "1",
        "TENRYU_A1_KE_CLOSURE_REDISTRIBUTE_FLOOR": "1",
        "TENRYU_A1_KE_CLOSURE_AUDIT": "0",
        "TENRYU_A1_PHASE_RESOLVED_ENERGY": "1",
        "TENRYU_A1_DEBUG_PER_REMAP_LOG": "1",
    }


def format_command(args: argparse.Namespace, env: dict[str, str]) -> str:
    env_part = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    cmd_part = " ".join(shlex.quote(part) for part in (args.tenryu_bin, "run", args.deck))
    return f"{env_part} {cmd_part}"


def run_one(args: argparse.Namespace, axis_floor: float) -> dict[str, Any]:
    outdir = Path(args.out_root) / run_id(args, axis_floor)
    outdir.mkdir(parents=True, exist_ok=True)
    active_env = os.environ.copy()
    overrides = env_for(args, axis_floor, outdir)
    active_env.update(overrides)

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=active_env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start
    return {
        "axis_floor": axis_floor,
        "run_id": run_id(args, axis_floor),
        "result": "PASS" if proc.returncode == 0 else "CRASHED",
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "log": str(log_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default=DECK)
    parser.add_argument("--out-root", default="./build/a1_icf_prep")
    parser.add_argument("--axis-floor-list", type=parse_float_list, default=list(AXIS_FLOORS_DEFAULT))
    parser.add_argument("--nr", type=int, default=128)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--t-end-s", type=float, default=2.0e-10)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--cfl", type=float, default=0.1)
    parser.add_argument("--impl-velocity-cm-s", type=float, default=1.0e7)
    parser.add_argument("--axis-motion-floor", type=float, default=0.20)
    parser.add_argument("--preventive-axis-guard", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.nr <= 0 or args.nz <= 0:
        raise SystemExit("--nr and --nz must be positive")

    axis_floors = list(args.axis_floor_list)
    if args.dry_run:
        print("Planned A1 ICF-shell hydro prep runs:")
        for axis_floor in axis_floors:
            outdir = Path(args.out_root) / run_id(args, axis_floor)
            print(format_command(args, env_for(args, axis_floor, outdir)))
        print(f"total_runs={len(axis_floors)}")
        return 0

    rows = [run_one(args, axis_floor) for axis_floor in axis_floors]
    print("A1 ICF-shell hydro prep summary")
    print("run_id axis_floor result exit_code wall_time_s log")
    for row in rows:
        print(
            "{run_id} {axis_floor:.12g} {result} {exit_code} {wall_time_s:.3f} {log}".format(
                **row
            )
        )
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
