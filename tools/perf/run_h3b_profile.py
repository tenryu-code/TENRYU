#!/usr/bin/env python3
"""Profile H3-B baseline and Phase 9-11+T31 modes reproducibly.

The wrapper only invokes the existing H3-B verification deck with environment
overrides. It does not edit or generate namelist decks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DECK_DEFAULT = "examples/verification/2d_rz_h3b_sedov_sph.py"
TENRYU_DEFAULT = "./build/tenryu"

PHASE_9_11_T31_ENV = {
    "TENRYU_H3B_AXIS_REPAIR_MODE": "axis_spine_only",
    "TENRYU_H3B_REMAP_DAMAGE_GATE": "1",
    "TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET": "1",
    "TENRYU_H3B_REMAP_SCHEME": "ms2_moments",
    "TENRYU_H3B_REMAP_MS2_LIMITER": "van_leer",
    "TENRYU_H3B_KE_CLOSURE": "1",
    "TENRYU_H3B_KE_CLOSURE_REDISTRIBUTE_FLOOR": "1",
}

PHASE_OFF_ENV = {
    "TENRYU_H3B_AXIS_REPAIR_MODE": "full_winslow",
    "TENRYU_H3B_REMAP_DAMAGE_GATE": "0",
    "TENRYU_H3B_REMAP_DAMAGE_AXIS_BUDGET": "0",
    "TENRYU_H3B_REMAP_SCHEME": "legacy_split",
    "TENRYU_H3B_REMAP_MS2_LIMITER": "van_leer",
    "TENRYU_H3B_KE_CLOSURE": "0",
    "TENRYU_H3B_KE_CLOSURE_AUDIT": "0",
    "TENRYU_H3B_KE_CLOSURE_REDISTRIBUTE_FLOOR": "0",
    "TENRYU_H3B_PHASE_RESOLVED_ENERGY": "0",
}


@dataclass(frozen=True)
class ModeConfig:
    name: str
    nr: int
    nz: int
    cfl: float
    t_end_s: float
    ale_every: int
    env: dict[str, str]
    description: str

    @property
    def mesh(self) -> str:
        return f"{self.nr}x{self.nz}"


def mode_configs() -> dict[str, ModeConfig]:
    production_256_env = {
        **PHASE_9_11_T31_ENV,
        "TENRYU_H3B_KE_CLOSURE_AUDIT": "0",
        "TENRYU_H3B_PHASE_RESOLVED_ENERGY": "0",
        "TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION": "0.0",
        "TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION": "0.0",
        "TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION": "0.1",
    }
    fast_512_env = {
        **PHASE_OFF_ENV,
        "TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION": "0.20",
        "TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION": "0.03",
        "TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION": "0.5",
    }
    full_512_env = {
        **PHASE_9_11_T31_ENV,
        "TENRYU_H3B_KE_CLOSURE_AUDIT": "0",
        "TENRYU_H3B_PHASE_RESOLVED_ENERGY": "0",
        "TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION": "0.20",
        "TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION": "0.03",
        "TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION": "0.5",
    }
    return {
        "baseline-256": ModeConfig(
            name="baseline-256",
            nr=256,
            nz=512,
            cfl=0.1,
            t_end_s=6.0e-9,
            ale_every=0,
            env={
                **PHASE_OFF_ENV,
                "TENRYU_H3B_AXIS_MOTION_FLOOR_FRACTION": "0.0",
                "TENRYU_H3B_AXIS_MARGIN_DT_FLOOR_FRACTION": "0.0",
                "TENRYU_H3B_PREVENTIVE_AXIS_GUARD_FRACTION": "0.1",
            },
            description="256x512 ALE off, Phase 9-11/T31 off ratio baseline.",
        ),
        "production-256": ModeConfig(
            name="production-256",
            nr=256,
            nz=512,
            cfl=0.1,
            t_end_s=6.0e-9,
            ale_every=5,
            env=production_256_env,
            description="256x512 Phase 9-11+T31 production configuration.",
        ),
        "fast-512-full": ModeConfig(
            name="fast-512-full",
            nr=512,
            nz=1024,
            cfl=0.1,
            t_end_s=6.0e-9,
            ale_every=5,
            env=fast_512_env,
            description="512x1024 T21+T26 fast full-target mode, Phase 9-11 off.",
        ),
        "full-stack-512-progress": ModeConfig(
            name="full-stack-512-progress",
            nr=512,
            nz=1024,
            cfl=0.05,
            t_end_s=3.0e-9,
            ale_every=5,
            env=full_512_env,
            description="512x1024 Phase 9-11+T31 progress mode to 50% target time.",
        ),
    }


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def format_float(value: float) -> str:
    return f"{value:.12g}"


def discover_output_dir(requested: Path) -> Path | None:
    if (requested / "run_info.json").exists():
        return requested
    parent = requested.parent
    if not parent.exists():
        return None
    candidates = [
        path
        for path in parent.glob(f"{requested.name}_*")
        if path.is_dir() and (path / "run_info.json").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_run_info(output_dir: Path | None) -> dict[str, Any]:
    if output_dir is None:
        return {}
    run_info = output_dir / "run_info.json"
    if not run_info.exists():
        return {}
    try:
        return json.loads(run_info.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def latest_history_file(output_dir: Path | None) -> Path | None:
    if output_dir is None:
        return None
    results_dir = output_dir / "results"
    if not results_dir.exists():
        return None
    histories = sorted(results_dir.glob("*_history.h5"), key=lambda p: p.stat().st_mtime)
    return histories[-1] if histories else None


def read_ale_events_from_history(output_dir: Path | None) -> int | None:
    history = latest_history_file(output_dir)
    if history is None:
        return None
    try:
        import h5py  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        with h5py.File(history, "r") as handle:
            if "mesh/ale_rezone_invocations" not in handle:
                return None
            data = handle["mesh/ale_rezone_invocations"]
            if data.shape[0] == 0:
                return 0
            return int(data[-1])
    except (OSError, KeyError, TypeError, ValueError):
        return None


def parse_ale_events_from_log(log_path: Path) -> int | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    accepted = len(re.findall(r"\[ale-stats\] backtrack_lambda_accepted=", text))
    rejected = len(re.findall(r"all lambda trials rejected", text))
    triggered_without_backtrack = len(re.findall(r"preventive axis guard triggered ALE", text))
    count = max(accepted + rejected, triggered_without_backtrack)
    return count if count > 0 else None


def parse_io_runtime_from_phase_timing(log_path: Path) -> float | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values_ms = [float(match) for match in re.findall(r"\bdiag=([0-9]+(?:\.[0-9]+)?)\b", text)]
    if not values_ms:
        return None
    return sum(values_ms) / 1000.0


def command_for(args: argparse.Namespace, run_dir: Path, iteration: int) -> list[str]:
    base = [args.tenryu_bin, "run", args.deck]
    if args.profiler == "none":
        return base
    profile_base = run_dir / f"{args.profiler}_run_{iteration:02d}"
    if args.profiler == "nsys":
        return [
            "nsys",
            "profile",
            "-t",
            "cuda,nvtx,osrt",
            "--force-overwrite=true",
            "-o",
            str(profile_base),
            *base,
        ]
    if args.profiler == "ncu":
        return [
            "ncu",
            "--set",
            "full",
            "--force-overwrite",
            "-o",
            str(profile_base),
            *base,
        ]
    raise ValueError(f"unknown profiler: {args.profiler}")


def query_gpu(args: argparse.Namespace) -> dict[str, Any] | None:
    nvidia_smi = shutil.which(args.nvidia_smi)
    if nvidia_smi is None:
        return None
    query = "name,index,clocks.current.sm,clocks.max.sm,temperature.gpu,pstate"
    try:
        proc = subprocess.run(
            [nvidia_smi, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return {
        "command": f"{nvidia_smi} --query-gpu={query} --format=csv,noheader,nounits",
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
        "clock_policy": args.gpu_clock_policy,
    }


def build_env(args: argparse.Namespace, mode: ModeConfig, requested_outdir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H3B_NR": str(mode.nr),
            "TENRYU_H3B_NZ": str(mode.nz),
            "TENRYU_H3B_SEED": str(args.seed),
            "TENRYU_H3B_OUTDIR": str(requested_outdir),
            "TENRYU_H3B_CFL": format_float(mode.cfl),
            "TENRYU_H3B_T_END_S": format_float(mode.t_end_s),
            "TENRYU_H3B_ALE_EVERY_N_STEPS": str(mode.ale_every),
        }
    )
    env.update(mode.env)
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return env


def run_one(
    args: argparse.Namespace,
    mode: ModeConfig,
    run_index: int,
    warmup: bool,
) -> dict[str, Any]:
    phase = "warmup" if warmup else "measured"
    run_dir = args.out_root / mode.name / f"{phase}_{run_index:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    requested_outdir = run_dir / "tenryu_output"
    env = build_env(args, mode, requested_outdir)
    log_path = run_dir / "run.log"
    command = command_for(args, run_dir, run_index)

    start = time.monotonic()
    timeout = None if args.timeout_s <= 0 else args.timeout_s
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# command: " + " ".join(command) + "\n")
        log.write("# mode: " + mode.name + "\n")
        log.write("# warmup: " + str(warmup).lower() + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=log,
                check=False,
                timeout=timeout,
            )
            returncode: int | str = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = "timeout"
            log.write(f"\n[run_h3b_profile] timeout after {timeout} s\n")
    wall_time_s = time.monotonic() - start

    output_dir = discover_output_dir(requested_outdir)
    run_info = read_run_info(output_dir)
    steps = int(run_info.get("step", 0) or 0)
    ale_events = read_ale_events_from_history(output_dir)
    if ale_events is None:
        ale_events = parse_ale_events_from_log(log_path)
    if ale_events is None:
        ale_events = 0 if mode.ale_every == 0 else None
    io_runtime_s = parse_io_runtime_from_phase_timing(log_path)
    output_bytes = 0
    if output_dir is not None:
        output_bytes = sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())

    return {
        "mode": mode.name,
        "description": mode.description,
        "seed": args.seed,
        "mesh": mode.mesh,
        "nr": mode.nr,
        "nz": mode.nz,
        "cfl": mode.cfl,
        "t_end_s": mode.t_end_s,
        "ale_every": mode.ale_every,
        "warmup": warmup,
        "run_index": run_index,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "wall_per_step": (wall_time_s / steps) if steps > 0 else None,
        "steps": steps,
        "t_final_s": run_info.get("t"),
        "termination_reason": run_info.get("termination_reason"),
        "ale_events": ale_events,
        "io_runtime_s": io_runtime_s,
        "io_runtime_method": "phase_timing_diag_ms" if io_runtime_s is not None else "unavailable",
        "output_dir": str(output_dir) if output_dir is not None else None,
        "output_bytes": output_bytes,
        "log_path": str(log_path),
    }


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize(mode: ModeConfig, args: argparse.Namespace, runs: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [run for run in runs if not run["warmup"]]
    walls = [float(run["wall_time_s"]) for run in measured]
    wall_per_step_values = [
        float(run["wall_per_step"]) for run in measured if run["wall_per_step"] is not None
    ]
    io_values = [
        float(run["io_runtime_s"]) for run in measured if run["io_runtime_s"] is not None
    ]
    ale_values = [
        int(run["ale_events"]) for run in measured if isinstance(run["ale_events"], int)
    ]
    mean_s = statistics.mean(walls) if walls else None
    sigma_s = statistics.pstdev(walls) if len(walls) > 1 else 0.0 if walls else None
    cv = (sigma_s / abs(mean_s)) if mean_s not in (None, 0.0) else None
    return {
        "seed": args.seed,
        "mesh": mode.mesh,
        "mode": mode.name,
        "wall_per_step": median_or_none(wall_per_step_values),
        "ale_events": int(statistics.median(ale_values)) if ale_values else None,
        "io_runtime_s": median_or_none(io_values),
        "median_s": median_or_none(walls),
        "min_s": min(walls) if walls else None,
        "max_s": max(walls) if walls else None,
        "CV": cv,
        "measured_runs": len(measured),
        "warmup_runs": len([run for run in runs if run["warmup"]]),
        "cv_accept_threshold": args.cv_threshold,
        "cv_accepted": bool(cv is not None and cv <= args.cv_threshold),
        "io_runtime_method": "phase_timing_diag_ms" if io_values else "unavailable",
        "description": mode.description,
    }


def build_ratios(summaries: list[dict[str, Any]], cv_threshold: float) -> list[dict[str, Any]]:
    by_mode = {summary["mode"]: summary for summary in summaries}
    out: list[dict[str, Any]] = []
    baseline = by_mode.get("baseline-256")
    production = by_mode.get("production-256")
    if baseline is not None and production is not None:
        b = baseline.get("median_s")
        p = production.get("median_s")
        b_cv = baseline.get("CV")
        p_cv = production.get("CV")
        ratio = (p / b) if isinstance(b, (int, float)) and b > 0 and isinstance(p, (int, float)) else None
        accepted = (
            ratio is not None
            and isinstance(b_cv, (int, float))
            and isinstance(p_cv, (int, float))
            and b_cv <= cv_threshold
            and p_cv <= cv_threshold
        )
        out.append(
            {
                "name": "production-256_vs_baseline-256",
                "numerator_mode": "production-256",
                "denominator_mode": "baseline-256",
                "mesh": "256x512",
                "ratio": ratio,
                "accepted": accepted,
                "acceptance_rule": f"both CV <= {cv_threshold:.6g}",
            }
        )
    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    modes = mode_configs()
    parser = argparse.ArgumentParser(
        description="Run reproducible H3-B performance profiles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tenryu-bin", default=TENRYU_DEFAULT)
    parser.add_argument("--deck", default=DECK_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=Path("build/h3b_profile"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--mode",
        choices=sorted(modes),
        action="append",
        help="Mode to run. Repeat to run multiple modes.",
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--measured-runs", type=int, default=5)
    parser.add_argument("--cv-threshold", type=float, default=0.05)
    parser.add_argument("--timeout-s", type=float, default=0.0, help="Per-run timeout; 0 disables it.")
    parser.add_argument("--profiler", choices=("none", "nsys", "ncu"), default="none")
    parser.add_argument("--nvidia-smi", default="nvidia-smi")
    parser.add_argument("--gpu-clock-policy", default="inherit")
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-run-error", action="store_true")
    args = parser.parse_args(argv)

    if args.warmup_runs < 0:
        parser.error("--warmup-runs must be >= 0")
    if args.measured_runs <= 0:
        parser.error("--measured-runs must be > 0")
    if args.cv_threshold <= 0.0 or not math.isfinite(args.cv_threshold):
        parser.error("--cv-threshold must be a positive finite number")
    if args.timeout_s < 0.0:
        parser.error("--timeout-s must be >= 0")
    if args.mode is None:
        args.mode = ["baseline-256", "production-256"]
    args.out_root = args.out_root.resolve()
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    modes = mode_configs()
    selected_modes = [modes[name] for name in args.mode]
    plan = {
        "deck": args.deck,
        "tenryu_bin": args.tenryu_bin,
        "out_root": str(args.out_root),
        "seed": args.seed,
        "warmup_runs": args.warmup_runs,
        "measured_runs": args.measured_runs,
        "profiler": args.profiler,
        "modes": [
            {
                "mode": mode.name,
                "mesh": mode.mesh,
                "cfl": mode.cfl,
                "t_end_s": mode.t_end_s,
                "ale_every": mode.ale_every,
                "env": mode.env,
                "description": mode.description,
            }
            for mode in selected_modes
        ],
    }
    if args.dry_run:
        print(json.dumps({"dry_run": True, "plan": plan}, indent=2, sort_keys=True))
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)
    gpu_before = query_gpu(args)
    all_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    had_run_error = False

    for mode in selected_modes:
        mode_runs: list[dict[str, Any]] = []
        for i in range(args.warmup_runs):
            run = run_one(args, mode, i, warmup=True)
            mode_runs.append(run)
            all_runs.append(run)
            if run["returncode"] not in (0,):
                had_run_error = True
        for i in range(args.measured_runs):
            run = run_one(args, mode, i, warmup=False)
            mode_runs.append(run)
            all_runs.append(run)
            if run["returncode"] not in (0,):
                had_run_error = True
        summaries.append(summarize(mode, args, mode_runs))

    gpu_after = query_gpu(args)
    payload = {
        "schema": "tenryu.h3b_profile.v1",
        "created_unix_s": time.time(),
        "plan": plan,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "summaries": summaries,
        "ratios": build_ratios(summaries, args.cv_threshold),
        "runs": all_runs,
    }
    summary_path = args.out_root / "h3b_profile_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")
    print(str(summary_path))
    if args.fail_on_run_error and had_run_error:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
