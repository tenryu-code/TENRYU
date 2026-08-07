#!/usr/bin/env python3
"""Run Stage C1 conduction-only diffusion geometry sweep."""

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


STAGE_ID = "C1"
MODES = ("slab", "radial", "axis_pulse", "distorted", "limiter_spitzer")
DEFAULT_MESHES = ((64, 128), (128, 256), (256, 512))
DEFAULT_CFLS = (0.1, 0.25)
DEFAULT_TEST_KAPPA_CFL_MAX = 0.1
NUMBER_RE = r"[-+0-9.eE]+"
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")

EV_TO_ERG = 1.6022e-12
M_P = 1.6726e-24
M_E = 9.1093837015e-28
KAPPA0 = 3.06e9


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


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


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("float list is empty")
    return values


def case_name(mode: str, nr: int, nz: int, cfl_cond: float, f_lim: float, seed: int) -> str:
    return (
        f"c1_{mode}_nr{nr}_nz{nz}_cfl{safe_float_token(cfl_cond)}"
        f"_flim{safe_float_token(f_lim)}_seed{seed}"
    )


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def read_optional(handle: Any, path: str) -> Any | None:
    return handle[path][()] if path in handle else None


def reshape_cells(values: Any, nr: int, nz: int) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    return arr


def read_snapshot(path: Path, nr: int, nz: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        data = {
            "t": float(handle.attrs.get("t", 0.0)),
            "cycle": int(handle.attrs.get("cycle", 0)),
            "x_r": np.asarray(handle["mesh/x_r"][()], dtype=float).reshape((nr + 1, nz + 1)),
            "x_z": np.asarray(handle["mesh/x_z"][()], dtype=float).reshape((nr + 1, nz + 1)),
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz),
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz),
            "mass": reshape_cells(handle["hydro/mass"][()], nr, nz),
        }
        vol = read_optional(handle, "hydro/vol")
        if vol is not None:
            data["vol"] = reshape_cells(vol, nr, nz)
        return data


def cell_centers(snapshot: dict[str, Any]) -> tuple[Any, Any]:
    x_r = snapshot["x_r"]
    x_z = snapshot["x_z"]
    r = 0.25 * (x_r[:-1, :-1] + x_r[1:, :-1] + x_r[:-1, 1:] + x_r[1:, 1:])
    z = 0.25 * (x_z[:-1, :-1] + x_z[1:, :-1] + x_z[:-1, 1:] + x_z[1:, 1:])
    return r, z


def electron_energy(snapshot: dict[str, Any]) -> float:
    import numpy as np

    return float(np.sum(snapshot["mass"] * snapshot["ee"]))


def history_metrics(results_dir: Path) -> dict[str, Any]:
    history = latest_history_file(results_dir)
    if history is None:
        return {}
    import h5py
    import numpy as np

    out: dict[str, Any] = {}
    with h5py.File(history, "r") as handle:
        internal_e = read_optional(handle, "energy/internal_electron")
        if internal_e is not None:
            arr = np.asarray(internal_e, dtype=float).reshape(-1)
            if arr.size >= 2:
                out["history_E_e_initial"] = float(arr[0])
                out["history_E_e_final"] = float(arr[-1])
        floor = read_optional(handle, "energy/floor_injected")
        if floor is not None:
            arr = np.asarray(floor, dtype=float).reshape(-1)
            out["E_floor_injected"] = float(arr[-1]) if arr.size else 0.0
        conservation = read_optional(handle, "energy/conservation_error")
        if conservation is not None:
            arr = np.asarray(conservation, dtype=float).reshape(-1)
            out["energy_conservation_error_max"] = float(np.nanmax(arr)) if arr.size else None
    return out


def parse_clamp_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return sum(int(match.group("count")) for match in CLAMP_RE.finditer(text))


def cv_e(args: argparse.Namespace) -> float:
    return args.zbar * EV_TO_ERG / (args.A * M_P * (args.gamma - 1.0))


def test_chi(args: argparse.Namespace) -> float:
    return args.test_kappa / (args.rho_gcc * cv_e(args))


def slab_residual(snapshot: dict[str, Any], args: argparse.Namespace) -> float:
    import numpy as np

    r, z = cell_centers(snapshot)
    del r
    z_min = float(snapshot["x_z"][0, 0])
    z_max = float(snapshot["x_z"][0, -1])
    z_mid = 0.5 * (z_min + z_max)
    length = z_max - z_min
    k = 2.0 * math.pi / length
    expected = args.t_init_ev + args.t_amp_ev * math.exp(-test_chi(args) * k * k * snapshot["t"]) * np.cos(
        k * (z - z_mid)
    )
    return float(np.nanmax(np.abs(snapshot["Te"] - expected)))


def radial_residual(first: dict[str, Any], final: dict[str, Any], args: argparse.Namespace) -> float:
    import numpy as np

    r, _ = cell_centers(first)
    r_max = float(first["x_r"][-1, 0])
    k = math.pi / r_max
    safe_r = np.maximum(r, 1.0e-300)
    lap = -args.t_amp_ev * ((k * np.sin(k * safe_r) / safe_r) + (k * k * np.cos(k * safe_r)))
    expected = first["Te"] + test_chi(args) * final["t"] * lap
    mask = (r > 0.1 * r_max) & (r < 0.9 * r_max)
    if not np.any(mask):
        mask = np.isfinite(expected)
    return float(np.nanmax(np.abs(final["Te"][mask] - expected[mask])))


def axis_unphysical_count(snapshot: dict[str, Any]) -> int:
    import numpy as np

    axis = np.asarray(snapshot["Te"])[0, :]
    return int(np.count_nonzero((~np.isfinite(axis)) | (axis <= 0.0)))


def monotonicity_violations(first: dict[str, Any], final: dict[str, Any]) -> int:
    import numpy as np

    lo = float(np.nanmin(first["Te"])) - 1.0e-8
    hi = float(np.nanmax(first["Te"])) + 1.0e-8
    te = np.asarray(final["Te"])
    return int(np.count_nonzero((te < lo) | (te > hi) | (~np.isfinite(te))))


def coulomb_log(n_e: Any, te: Any, zbar: float) -> Any:
    import numpy as np

    n_safe = np.maximum(n_e, 1.0e-300)
    t_safe = np.maximum(te, 1.0e-300)
    high = t_safe >= 10.0 * zbar * zbar
    raw = np.where(
        high,
        24.0 - 0.5 * np.log(n_safe) + np.log(t_safe),
        23.0 - 0.5 * np.log(n_safe) - math.log(zbar) + 1.5 * np.log(t_safe),
    )
    return np.maximum(2.0, raw)


def flux_limiter_active_fraction(snapshot: dict[str, Any], args: argparse.Namespace) -> float:
    import numpy as np

    te = np.asarray(snapshot["Te"], dtype=float)
    rho = np.asarray(snapshot["rho"], dtype=float)
    r, z = cell_centers(snapshot)
    del z
    dr = np.maximum(np.gradient(r, axis=0), 1.0e-300)
    dz = np.maximum(np.gradient(cell_centers(snapshot)[1], axis=1), 1.0e-300)
    dte_dr = np.gradient(te, axis=0) / dr
    dte_dz = np.gradient(te, axis=1) / dz
    grad = np.sqrt(dte_dr * dte_dr + dte_dz * dte_dz)
    n_e = np.maximum(rho * args.zbar / (args.A * M_P), 1.0e-300)
    ln_lambda = coulomb_log(n_e, te, args.zbar)
    kappa = KAPPA0 * np.power(np.maximum(te, 1.0e-300), 2.5) / (args.zbar * ln_lambda)
    q_sh = kappa * grad
    vth = np.sqrt(EV_TO_ERG * np.maximum(te, 1.0e-300) / M_E)
    q_max = args.f_lim * n_e * EV_TO_ERG * te * vth
    active = q_sh > q_max
    finite = np.isfinite(q_sh) & np.isfinite(q_max) & (q_max > 0.0)
    if not np.any(finite):
        return 0.0
    return float(np.count_nonzero(active & finite) / np.count_nonzero(finite))


def distorted_mesh_quality(snapshot: dict[str, Any], amplitude_fraction: float) -> dict[str, Any]:
    import numpy as np

    x_r = np.array(snapshot["x_r"], copy=True, dtype=float)
    x_z = np.array(snapshot["x_z"], copy=True, dtype=float)
    nr = x_r.shape[0] - 1
    nz = x_r.shape[1] - 1
    dr = (float(np.nanmax(x_r)) - float(np.nanmin(x_r))) / max(nr, 1)
    dz = (float(np.nanmax(x_z)) - float(np.nanmin(x_z))) / max(nz, 1)
    amp = amplitude_fraction * min(abs(dr), abs(dz))
    for i in range(1, nr):
        for j in range(1, nz):
            s_i = math.sin(math.pi * i / nr)
            s_j = math.sin(math.pi * j / nz)
            x_r[i, j] += amp * s_i * math.sin(2.0 * math.pi * j / nz)
            x_z[i, j] += amp * math.sin(2.0 * math.pi * i / nr) * s_j
    x_r[0, :] = 0.0
    quality = mesh_quality_min(x_r, x_z)
    return {
        "distortion_applied_in_harness": True,
        "distortion_amplitude_fraction": amplitude_fraction,
        "distorted_quality_min": quality,
    }


def mesh_quality_min(x_r: Any, x_z: Any) -> float:
    import numpy as np

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
    values = 4.0 * area / denom
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else math.nan


def collect_metrics(
    args: argparse.Namespace,
    mode: str,
    nr: int,
    nz: int,
    results_dir: Path,
    run_id: str,
    log_path: Path,
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    plot_files = locate_plot_files(results_dir, run_id)
    if len(plot_files) < 2:
        raise RuntimeError("expected initial and final plot files")
    first = read_snapshot(plot_files[0], nr, nz)
    final = read_snapshot(plot_files[-1], nr, nz)
    e0 = electron_energy(first)
    e1 = electron_energy(final)
    hist = history_metrics(results_dir)
    if "history_E_e_initial" in hist and "history_E_e_final" in hist:
        e0 = float(hist["history_E_e_initial"])
        e1 = float(hist["history_E_e_final"])
    delta = e1 - e0
    denom = max(abs(e0), 1.0e-300)
    metrics: dict[str, Any] = {
        "delta_E_e": delta,
        "delta_E_e_rel": abs(delta) / denom,
        "dE_e": delta,
        "dE_e_rel": abs(delta) / denom,
        "n_clamp": parse_clamp_count(log_path),
        "E_floor_injected": hist.get("E_floor_injected", 0.0),
        "max_T_residual": None,
        "flux_limiter_active_fraction": 0.0,
        "axis_T_unphysical_count": axis_unphysical_count(final),
        "monotonicity_violation_count": monotonicity_violations(first, final),
        "Te_min_eV": float(np.nanmin(final["Te"])),
        "Te_max_eV": float(np.nanmax(final["Te"])),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(hist)
    if mode == "slab":
        metrics["max_T_residual"] = slab_residual(final, args)
    elif mode == "radial":
        metrics["max_T_residual"] = radial_residual(first, final, args)
        metrics["radial_gate_scope"] = "manufactured_short_time_residual"
    elif mode == "limiter_spitzer":
        metrics["flux_limiter_active_fraction"] = flux_limiter_active_fraction(first, args)
    elif mode == "distorted":
        metrics.update(distorted_mesh_quality(first, args.distortion_amplitude_fraction))
        metrics["distorted_solver_mesh_source"] = "uniform_run_harness_in_memory_metric_distortion"
    return metrics


def evaluate_gates(args: argparse.Namespace, mode: str, metrics: dict[str, Any]) -> list[str]:
    failed = []
    if metrics["delta_E_e_rel"] > args.energy_rel_threshold:
        failed.append("c1_energy_conservation")
    if metrics["n_clamp"] != 0:
        failed.append("c1_no_floor_clamps")
    if abs(float(metrics.get("E_floor_injected") or 0.0)) > args.floor_energy_threshold:
        failed.append("c1_no_floor_energy")
    if metrics["axis_T_unphysical_count"] != 0:
        failed.append("c1_axis_temperature_physical")
    if mode == "limiter_spitzer":
        active_fraction = float(metrics.get("flux_limiter_active_fraction") or 0.0)
        if active_fraction <= 0.0 or active_fraction > args.flux_limiter_fraction_max:
            failed.append("c1_flux_limiter_localized")
    if mode == "distorted" and metrics.get("monotonicity_violation_count") != 0:
        failed.append("c1_distorted_monotonicity")
    return failed


def active_env(
    args: argparse.Namespace,
    mode: str,
    nr: int,
    nz: int,
    cfl_cond: float,
    seed: int,
    outdir: Path,
) -> dict[str, str]:
    t_end = args.t_end_s if args.t_end_s is not None else (1.0e-12 if mode == "limiter_spitzer" else 1.0e-4)
    max_steps = args.max_steps if args.max_steps is not None else (20 if mode == "limiter_spitzer" else 200000)
    dt_initial = args.dt_initial_s if args.dt_initial_s is not None else min(1.0e-8, t_end)
    dt_max = args.dt_max_s if args.dt_max_s is not None else max(t_end / 4.0, dt_initial)
    env = {
        "TENRYU_C1_MODE": mode,
        "TENRYU_C1_NR": str(nr),
        "TENRYU_C1_NZ": str(nz),
        "TENRYU_C1_SEED": str(seed),
        "TENRYU_C1_OUTDIR": str(outdir),
        "TENRYU_C1_CFL_COND": str(cfl_cond),
        "TENRYU_C1_F_LIM": str(args.f_lim),
        "TENRYU_C1_SOLVER": args.solver,
        "TENRYU_C1_T_INIT_eV": str(args.t_init_ev),
        "TENRYU_C1_T_AMP_eV": str(args.t_amp_ev),
        "TENRYU_C1_T_HOT_eV": str(args.t_hot_ev),
        "TENRYU_C1_T_COLD_eV": str(args.t_cold_ev),
        "TENRYU_C1_RHO_GCC": str(args.rho_gcc if mode != "limiter_spitzer" else args.limiter_rho_gcc),
        "TENRYU_C1_TEST_KAPPA": str(args.test_kappa),
        "TENRYU_C1_T_END_S": str(t_end),
        "TENRYU_C1_MAX_STEPS": str(max_steps),
        "TENRYU_C1_DT_INITIAL_S": str(dt_initial),
        "TENRYU_C1_DT_MAX_S": str(dt_max),
        "TENRYU_C1_R_MAX_CM": str(args.r_max_cm),
        "TENRYU_C1_Z_MIN_CM": str(args.z_min_cm),
        "TENRYU_C1_Z_MAX_CM": str(args.z_max_cm),
        "TENRYU_C1_SIGMA_CM": str(args.sigma_cm),
        "TENRYU_C1_A": str(args.A),
        "TENRYU_C1_ZBAR": str(args.zbar),
        "TENRYU_C1_GAMMA": str(args.gamma),
    }
    return env


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


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, cfl_cond: float, seed: int) -> dict[str, Any]:
    run_id = case_name(mode, nr, nz, cfl_cond, args.f_lim, seed)
    ladder_row = f"{mode}_nr{nr}_nz{nz}_cfl{safe_float_token(cfl_cond)}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env_overrides = active_env(args, mode, nr, nz, cfl_cond, seed, outdir)
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics: dict[str, Any] = {
        "delta_E_e": None,
        "delta_E_e_rel": None,
        "dE_e": None,
        "dE_e_rel": None,
        "n_clamp": parse_clamp_count(log_path),
        "max_T_residual": None,
        "flux_limiter_active_fraction": None,
        "axis_T_unphysical_count": None,
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
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            metrics = collect_metrics(args, mode, nr, nz, results_dir, run_id, log_path, wall_time_s, bytes_out)
            failed_gates = evaluate_gates(args, mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_c1] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "stage_id": STAGE_ID,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "cfl_cond": cfl_cond,
        "metrics": metrics,
        "env_overrides": env_overrides,
        "result": result,
        "failed_gates": failed_gates,
    }
    write_json(Path(rel_metrics_ref), payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "cfl_cond": cfl_cond,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics.get("output_bytes", output_bytes(outdir)),
        "delta_E_e_rel": metrics.get("delta_E_e_rel"),
        "n_clamp": metrics.get("n_clamp"),
        "max_T_residual": metrics.get("max_T_residual"),
        "flux_limiter_active_fraction": metrics.get("flux_limiter_active_fraction"),
        "axis_T_unphysical_count": metrics.get("axis_T_unphysical_count"),
    }
    if not args.no_append:
        append_code = append_run(args, run)
        if append_code != 0 and result == "PASS":
            run["result"] = "FAIL"
            run["failed_gates"] = ["append_run_failure"]
    return run


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
        "delta_E_e_rel",
        "n_clamp",
        "max_T_residual",
        "flux_limiter_active_fraction",
        "axis_T_unphysical_count",
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
    print("Stage C1 conduction sweep summary")
    print("ladder_row seed result dE_e_rel n_clamp max_T_residual limiter_active axis_bad wall_time_s")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {delta_E_e_rel} {n_clamp} "
            "{max_T_residual} {flux_limiter_active_fraction} "
            "{axis_T_unphysical_count} {wall_time_s:.3f}".format(**row)
        )


def include_matrix_case(args: argparse.Namespace, mode: str, cfl_cond: float) -> bool:
    if mode == "limiter_spitzer":
        return True
    if args.include_test_kappa_high_cfl:
        return True
    return cfl_cond <= args.test_kappa_cfl_max + 1.0e-15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_c1_conduction.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/c1_runs")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--run-profile", default="C1 conduction-only diffusion smoke sweep")
    parser.add_argument("--mode-list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--cfl-list", type=parse_float_list, default=list(DEFAULT_CFLS))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--f-lim", type=float, default=0.06)
    parser.add_argument("--solver", default="sts")
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--t-init-ev", type=float, default=10.0)
    parser.add_argument("--t-amp-ev", type=float, default=1.0)
    parser.add_argument("--t-hot-ev", type=float, default=2000.0)
    parser.add_argument("--t-cold-ev", type=float, default=50.0)
    parser.add_argument("--rho-gcc", type=float, default=1.0)
    parser.add_argument("--limiter-rho-gcc", type=float, default=1.0e-3)
    parser.add_argument("--test-kappa", type=float, default=2.0e14)
    parser.add_argument(
        "--test-kappa-cfl-max",
        type=float,
        default=DEFAULT_TEST_KAPPA_CFL_MAX,
        help="Maximum cfl_cond included by default for fixed test_kappa manufactured modes.",
    )
    parser.add_argument(
        "--include-test-kappa-high-cfl",
        action="store_true",
        help="Include fixed test_kappa cases above --test-kappa-cfl-max for exploratory runs.",
    )
    parser.add_argument("--r-max-cm", type=float, default=1.0)
    parser.add_argument("--z-min-cm", type=float, default=-1.0)
    parser.add_argument("--z-max-cm", type=float, default=1.0)
    parser.add_argument("--sigma-cm", type=float, default=0.12)
    parser.add_argument("--A", type=float, default=1.0)
    parser.add_argument("--zbar", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=5.0 / 3.0)
    parser.add_argument("--energy-rel-threshold", type=float, default=1.0e-6)
    parser.add_argument("--floor-energy-threshold", type=float, default=0.0)
    parser.add_argument("--flux-limiter-fraction-max", type=float, default=0.25)
    parser.add_argument("--distortion-amplitude-fraction", type=float, default=0.10)
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/C1/metrics/c1_summary.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-append", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meshes = list(args.mesh_list or DEFAULT_MESHES)
    matrix = [
        (mode, nr, nz, cfl_cond, seed)
        for mode in args.mode_list
        for nr, nz in meshes
        for cfl_cond in args.cfl_list
        for seed in args.seed_list
        if include_matrix_case(args, mode, cfl_cond)
    ]
    if not matrix:
        print(
            "No Stage C1 runs selected after cfl_cond/test_kappa scope filtering; "
            "use --include-test-kappa-high-cfl for exploratory fixed-kappa high-CFL rows.",
            file=sys.stderr,
        )
        return 2

    if args.dry_run:
        print("Planned Stage C1 runs:")
        for mode, nr, nz, cfl_cond, seed in matrix:
            run_id = case_name(mode, nr, nz, cfl_cond, args.f_lim, seed)
            print(f"{run_id} mode={mode} nr={nr} nz={nz} cfl_cond={cfl_cond} seed={seed}")
        print(f"total_runs={len(matrix)}")
        print(f"summary_json={args.summary_json}")
        return 0

    rows = [run_one(args, mode, nr, nz, cfl_cond, seed) for mode, nr, nz, cfl_cond, seed in matrix]
    print_summary(rows)
    write_summary(args, rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
