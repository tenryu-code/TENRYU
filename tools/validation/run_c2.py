#!/usr/bin/env python3
"""Run Stage C2 Qei and Qei+conduction validation sweep."""

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


STAGE_ID = "C2"
MODES = (
    "near_eq_analytic",
    "hot_e_rk4",
    "hot_i_rk4",
    "tmat_hot_e",
    "dt_sweep",
    "qei_plus_cond_uniform",
    "qei_plus_cond_gradient",
)
NON_DT_MODES = tuple(mode for mode in MODES if mode != "dt_sweep")
SMOKE_MESHES = ((32, 64),)
FULL_MESHES = ((32, 64), (128, 256))
DT_FACTORS = (0.01, 0.1, 1.0, 10.0)
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")

EV_TO_ERG = 1.6022e-12
M_P = 1.6726e-24
RHO_GCC_DEFAULT = 1.0e-5
A_DEFAULT = 1.0
Z_DEFAULT = 1.0
GAMMA_DEFAULT = 5.0 / 3.0
TMAT_RHO_BOUNDS = (1.0e-7, 100.0)
TMAT_T_BOUNDS = (0.1, 1.0e6)


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


def cv_e_mass(args: argparse.Namespace) -> float:
    return args.zbar * EV_TO_ERG / (args.A * M_P * (args.gamma - 1.0))


def cv_i_mass(args: argparse.Namespace) -> float:
    return EV_TO_ERG / (args.A * M_P * (args.gamma - 1.0))


def tau_eq(te_ev: float, rho_gcc: float, A: float, Z: float) -> float:
    n_i = rho_gcc / (A * M_P)
    te = max(te_ev, 1.0e-300)
    z = max(Z, 1.0e-300)
    if te >= 10.0 * z * z:
        ll = 24.0 - 0.5 * math.log(z * n_i) + math.log(te)
    else:
        ll = 23.0 - 0.5 * math.log(z * n_i) - math.log(z) + 1.5 * math.log(te)
    ll = max(2.0, ll)
    return 3.16e8 * A * te**1.5 / (z * z * n_i * ll)


def tau_eff(te_ev: float, args: argparse.Namespace) -> float:
    cve = cv_e_mass(args)
    cvi = cv_i_mass(args)
    return tau_eq(te_ev, args.rho_gcc, args.A, args.zbar) * cvi / (cve + cvi)


def mode_initial_temperatures(mode: str) -> tuple[float, float]:
    if mode == "near_eq_analytic":
        return 50.25, 49.75
    if mode == "hot_i_rk4":
        return 1.0, 100.0
    return 100.0, 1.0


def rk4_reference(te0: float, ti0: float, t_end: float, args: argparse.Namespace) -> tuple[float, float]:
    cve = cv_e_mass(args)
    cvi = cv_i_mass(args)
    cvsum = cve + cvi
    dt_ref = max(tau_eff(te0, args) / 200.0, 1.0e-30)
    te = float(te0)
    ti = float(ti0)
    t = 0.0

    def rhs(te_v: float, ti_v: float) -> tuple[float, float]:
        theta = te_v - ti_v
        tau_loc = tau_eff(te_v, args)
        return -cvi / cvsum * theta / tau_loc, cve / cvsum * theta / tau_loc

    while t < t_end - 1.0e-30:
        dt = min(dt_ref, t_end - t)
        k1e, k1i = rhs(te, ti)
        k2e, k2i = rhs(te + 0.5 * dt * k1e, ti + 0.5 * dt * k1i)
        k3e, k3i = rhs(te + 0.5 * dt * k2e, ti + 0.5 * dt * k2i)
        k4e, k4i = rhs(te + dt * k3e, ti + dt * k3i)
        te += dt * (k1e + 2.0 * k2e + 2.0 * k3e + k4e) / 6.0
        ti += dt * (k1i + 2.0 * k2i + 2.0 * k3i + k4i) / 6.0
        t += dt
    return te, ti


def stepwise_map_step(
    te_n: float,
    ti_n: float,
    dt_n: float,
    cv_e_value: float,
    cv_i_value: float,
    A: float,
    Z: float,
    rho_gcc: float,
) -> tuple[float, float]:
    n_i = rho_gcc / (A * M_P)
    if te_n >= 10.0 * Z * Z:
        ll = 24.0 - 0.5 * math.log(Z * n_i) + math.log(te_n)
    else:
        ll = 23.0 - 0.5 * math.log(Z * n_i) - math.log(Z) + 1.5 * math.log(te_n)
    ll = max(2.0, ll)
    tau_eq_n = 3.16e8 * A * te_n**1.5 / (Z * Z * n_i * ll)
    tau_eff_n = tau_eq_n * cv_i_value / (cv_e_value + cv_i_value)
    theta_n = te_n - ti_n
    t_eq = (cv_e_value * te_n + cv_i_value * ti_n) / (cv_e_value + cv_i_value)
    theta_np1 = theta_n * math.exp(-dt_n / tau_eff_n)
    te_np1 = t_eq + (cv_i_value / (cv_e_value + cv_i_value)) * theta_np1
    ti_np1 = t_eq - (cv_e_value / (cv_e_value + cv_i_value)) * theta_np1
    return te_np1, ti_np1


def stepwise_map_reference(te0: float, ti0: float, t_end: float, dt_force: float, args: argparse.Namespace) -> tuple[float, float]:
    te = te0
    ti = ti0
    t = 0.0
    cve = cv_e_mass(args)
    cvi = cv_i_mass(args)
    while t < t_end - 1.0e-30:
        dt = min(dt_force, t_end - t)
        te, ti = stepwise_map_step(te, ti, dt, cve, cvi, args.A, args.zbar, args.rho_gcc)
        t += dt
    return te, ti


def rel_err(value: float, ref: float) -> float:
    return abs(value / ref - 1.0) if ref != 0.0 else abs(value - ref)


def case_name(mode: str, nr: int, nz: int, dt_label: str, seed: int) -> str:
    return f"c2_{mode}_nr{nr}_nz{nz}_{dt_label}_seed{seed}"


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
        return int(path.stem.rsplit("_", 1)[1])

    files = sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=plot_index,
    )
    if files:
        return files
    return sorted((p for p in results_dir.glob("*_*.h5") if p.stem.rsplit("_", 1)[-1].isdigit()), key=plot_index)


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
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        data = {
            "t": t_value,
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz),
            "Ti": reshape_cells(handle["hydro/Ti"][()], nr, nz),
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz),
            "ei": reshape_cells(handle["hydro/ei"][()], nr, nz),
            "mass": reshape_cells(handle["hydro/mass"][()], nr, nz),
        }
        for field in ("cv_e", "cv_i"):
            raw = read_optional(handle, f"hydro/{field}")
            if raw is not None:
                data[field] = reshape_cells(raw, nr, nz)
        return data


def use_ideal_energy_temperatures(snapshot: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np

    out = dict(snapshot)
    out["Te_raw"] = np.array(snapshot["Te"], copy=True)
    out["Ti_raw"] = np.array(snapshot["Ti"], copy=True)
    out["Te"] = np.asarray(snapshot["ee"], dtype=float) / cv_e_mass(args)
    out["Ti"] = np.asarray(snapshot["ei"], dtype=float) / cv_i_mass(args)
    out["raw_Te_max_abs_delta"] = float(np.nanmax(np.abs(out["Te_raw"] - out["Te"])))
    out["raw_Ti_max_abs_delta"] = float(np.nanmax(np.abs(out["Ti_raw"] - out["Ti"])))
    return out


def parse_clamp_count(log_path: Path) -> int:
    if not log_path.exists():
        return 0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return sum(int(match.group("count")) for match in CLAMP_RE.finditer(text))


def history_metrics(results_dir: Path) -> dict[str, Any]:
    history = latest_history_file(results_dir)
    if history is None:
        return {}
    import h5py
    import numpy as np

    out: dict[str, Any] = {}
    with h5py.File(history, "r") as handle:
        for name, path in (
            ("history_E_e", "energy/internal_electron"),
            ("history_E_i", "energy/internal_ion"),
            ("history_clamp_count", "safety/clamp_count"),
        ):
            if path in handle:
                arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
                if arr.size:
                    out[f"{name}_initial"] = float(arr[0])
                    out[f"{name}_final"] = float(arr[-1])
                    out[f"{name}_max"] = float(np.nanmax(arr))
    return out


def total_internal_energy(snapshot: dict[str, Any]) -> float:
    import numpy as np

    return float(np.sum(snapshot["mass"] * (snapshot["ee"] + snapshot["ei"])))


def mean_temperature(snapshot: dict[str, Any], field: str) -> float:
    import numpy as np

    return float(np.nanmean(snapshot[field]))


def common_metrics(args: argparse.Namespace, first: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    te0 = mean_temperature(first, "Te")
    ti0 = mean_temperature(first, "Ti")
    te1 = mean_temperature(final, "Te")
    ti1 = mean_temperature(final, "Ti")
    cve = cv_e_mass(args)
    cvi = cv_i_mass(args)
    initial_partition = cve * te0 + cvi * ti0
    final_partition = cve * te1 + cvi * ti1
    partition_drift = abs(final_partition - initial_partition) / max(abs(initial_partition), 1.0e-300)
    all_te = np.asarray([np.nanmin(first["Te"]), np.nanmin(final["Te"])])
    all_ti = np.asarray([np.nanmin(first["Ti"]), np.nanmin(final["Ti"])])
    e0 = total_internal_energy(first)
    e1 = total_internal_energy(final)
    return {
        "tau_ei_initial_s": tau_eff(te0, args),
        "T_eq_eV": (cve * te0 + cvi * ti0) / (cve + cvi),
        "Te_initial_eV": te0,
        "Ti_initial_eV": ti0,
        "theta_initial_eV": te0 - ti0,
        "Te_final_eV": te1,
        "Ti_final_eV": ti1,
        "theta_final_eV": te1 - ti1,
        "t_final_s": float(final["t"]),
        "partition_drift": partition_drift,
        "total_E_initial": e0,
        "total_E_final": e1,
        "total_E_rel_drift": abs(e1 - e0) / max(abs(e0), 1.0e-300),
        "positivity_ok": bool(float(np.nanmin(all_te)) > args.te_floor_ev and float(np.nanmin(all_ti)) > args.ti_floor_ev),
        "no_nan": bool(np.all(np.isfinite(first["Te"])) and np.all(np.isfinite(final["Te"]))
                       and np.all(np.isfinite(first["Ti"])) and np.all(np.isfinite(final["Ti"]))),
        "Te_min_eV": float(np.nanmin(final["Te"])),
        "Ti_min_eV": float(np.nanmin(final["Ti"])),
        "Te_max_eV": float(np.nanmax(final["Te"])),
        "Ti_max_eV": float(np.nanmax(final["Ti"])),
        "hdf5_sample_count": 2,
        "sampling_note": "deck currently writes initial/final plot snapshots only",
    }


def monotonic_abs_theta(plot_files: list[Path], nr: int, nz: int) -> tuple[bool, list[float]]:
    import numpy as np

    values = []
    for path in plot_files:
        snap = read_snapshot(path, nr, nz)
        values.append(float(np.nanmax(np.abs(snap["Te"] - snap["Ti"]))))
    ok = all(values[i + 1] <= values[i] + 1.0e-12 for i in range(len(values) - 1))
    return ok, values


def te_spread_series(plot_files: list[Path], nr: int, nz: int, args: argparse.Namespace) -> tuple[bool, list[float]]:
    import numpy as np

    values = []
    for path in plot_files:
        snap = use_ideal_energy_temperatures(read_snapshot(path, nr, nz), args)
        values.append(float(np.nanmax(snap["Te"]) - np.nanmin(snap["Te"])))
    # Tolerate FMA-roundoff fluctuations; physical conduction smoothing gives O(1) eV spread reduction.
    spread_eps = 1.0e-2
    ok = all(values[i + 1] <= values[i] + spread_eps for i in range(len(values) - 1))
    return ok, values


def collect_metrics(
    args: argparse.Namespace,
    mode: str,
    nr: int,
    nz: int,
    dt_factor: float | None,
    results_dir: Path,
    run_id: str,
    log_path: Path,
    wall_time_s: float,
    bytes_out: int,
    baselines: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    plot_files = locate_plot_files(results_dir, run_id)
    if len(plot_files) < 2:
        raise RuntimeError("expected initial and final plot files")
    first = read_snapshot(plot_files[0], nr, nz)
    final = read_snapshot(plot_files[-1], nr, nz)
    if mode != "tmat_hot_e":
        first = use_ideal_energy_temperatures(first, args)
        final = use_ideal_energy_temperatures(final, args)
    metrics = common_metrics(args, first, final)
    metrics.update(history_metrics(results_dir))
    metrics.update({"n_clamp": parse_clamp_count(log_path), "wall_time_s": wall_time_s, "output_bytes": bytes_out})
    if mode != "tmat_hot_e":
        metrics.update({
            "temperature_source": "ideal_energy_reclosure_ee_over_cv",
            "raw_Te_max_abs_delta": final.get("raw_Te_max_abs_delta"),
            "raw_Ti_max_abs_delta": final.get("raw_Ti_max_abs_delta"),
        })

    te0 = metrics["Te_initial_eV"]
    ti0 = metrics["Ti_initial_eV"]
    t_final = metrics["t_final_s"]
    if mode == "near_eq_analytic":
        tau_const = 0.5 * tau_eq(50.0, args.rho_gcc, args.A, args.zbar)
        theta_ref = (te0 - ti0) * math.exp(-t_final / tau_const)
        metrics.update({
            "tau_ei_const_s": tau_const,
            "theta_ref_final_eV": theta_ref,
            "rel_err_theta": rel_err(metrics["theta_final_eV"], theta_ref),
        })
    if mode in ("hot_e_rk4", "hot_i_rk4", "dt_sweep", "qei_plus_cond_uniform"):
        te_rk4, ti_rk4 = rk4_reference(te0, ti0, t_final, args)
        metrics.update({
            "Te_RK4_final_eV": te_rk4,
            "Ti_RK4_final_eV": ti_rk4,
            "theta_RK4_final_eV": te_rk4 - ti_rk4,
            "rel_err_Te": rel_err(metrics["Te_final_eV"], te_rk4),
            "rel_err_Ti": rel_err(metrics["Ti_final_eV"], ti_rk4),
            "rel_err_theta": rel_err(metrics["theta_final_eV"], te_rk4 - ti_rk4),
        })
    if mode == "dt_sweep":
        dt_force = (dt_factor or 1.0) * tau_eff(100.0, args)
        te_map, ti_map = stepwise_map_reference(te0, ti0, t_final, dt_force, args)
        eps = max(1.0e-12, 1.0e-10 * metrics["T_eq_eV"])
        metrics.update({
            "dt_factor": dt_factor,
            "dt_force_s": dt_force,
            "Te_map_final_eV": te_map,
            "Ti_map_final_eV": ti_map,
            "theta_map_final_eV": te_map - ti_map,
            "rel_err_Te_map": rel_err(metrics["Te_final_eV"], te_map),
            "rel_err_Ti_map": rel_err(metrics["Ti_final_eV"], ti_map),
            "Te_map_Teq_norm_err": abs(metrics["Te_final_eV"] - te_map) / max(abs(metrics["T_eq_eV"]), 1.0e-300),
            "dt_sweep_bounds_ok": bool(metrics["Te_final_eV"] >= metrics["T_eq_eV"] - eps
                                       and metrics["Ti_final_eV"] <= metrics["T_eq_eV"] + eps),
            "theta_sign_unchanged": bool(metrics["theta_initial_eV"] * metrics["theta_final_eV"] >= 0.0),
            "dt_sweep_epsilon_eV": eps,
        })
    if mode == "tmat_hot_e":
        mono, theta_series = monotonic_abs_theta(plot_files, nr, nz)
        rho = np.asarray(final["rho"], dtype=float)
        te = np.asarray(final["Te"], dtype=float)
        ti = np.asarray(final["Ti"], dtype=float)
        cv_e_arr = final.get("cv_e")
        metrics.update({
            "monotonic_theta": mono,
            "theta_abs_max_series": theta_series,
            "table_bounds_source": "hardcoded_CD_tmat_plan_bounds",
            "rho_table_min": TMAT_RHO_BOUNDS[0],
            "rho_table_max": TMAT_RHO_BOUNDS[1],
            "T_table_min_eV": TMAT_T_BOUNDS[0],
            "T_table_max_eV": TMAT_T_BOUNDS[1],
            "table_domain_margin_ok": bool(
                np.nanmin(rho) > 1.1 * TMAT_RHO_BOUNDS[0]
                and np.nanmax(rho) < 0.9 * TMAT_RHO_BOUNDS[1]
                and np.nanmin(te) > 1.1 * TMAT_T_BOUNDS[0]
                and np.nanmin(ti) > 1.1 * TMAT_T_BOUNDS[0]
                and np.nanmax(te) < 0.9 * TMAT_T_BOUNDS[1]
                and np.nanmax(ti) < 0.9 * TMAT_T_BOUNDS[1]
            ),
            "cv_e_diagnostic": "present" if cv_e_arr is not None else "missing_warn",
            "cv_e_positive": bool(cv_e_arr is not None and np.all(np.asarray(cv_e_arr) > 0.0)),
        })
    if mode == "qei_plus_cond_gradient":
        spread_ok, spread = te_spread_series(plot_files, nr, nz, args)
        metrics.update({"Te_spread_decreases": spread_ok, "Te_spread_series": spread})
    if mode == "qei_plus_cond_uniform":
        baseline = baselines.get((nr, nz, int(args.seed_current)))
        if baseline is None:
            metrics.update({"hot_e_baseline_available": False, "max_abs_Te_vs_hot_e": None})
        else:
            metrics.update({
                "hot_e_baseline_available": True,
                "max_abs_Te_vs_hot_e": float(np.nanmax(np.abs(final["Te"] - baseline["Te"]))),
            })
    if mode == "hot_e_rk4":
        baselines[(nr, nz, int(args.seed_current))] = {"Te": np.array(final["Te"], copy=True)}
    return metrics


def evaluate_gates(args: argparse.Namespace, mode: str, metrics: dict[str, Any]) -> list[str]:
    failed = []
    if not metrics.get("no_nan", False):
        failed.append("c2_no_nan")
    if mode in ("tmat_hot_e", "qei_plus_cond_gradient") and not metrics.get("positivity_ok", False):
        failed.append("c2_positivity")
    if mode == "near_eq_analytic":
        # Constant-tau analytic reference is approximate because Spitzer tau varies with Te.
        if metrics.get("rel_err_theta", math.inf) > 0.02:
            failed.append("c2_near_eq_theta_ref")
        if metrics.get("partition_drift", math.inf) > 1.0e-6:
            failed.append("c2_partition_drift")
    elif mode in ("hot_e_rk4", "hot_i_rk4"):
        if metrics.get("rel_err_Te", math.inf) > 0.01:
            failed.append("c2_rk4_Te")
        if metrics.get("rel_err_Ti", math.inf) > 0.01:
            failed.append("c2_rk4_Ti")
        if metrics.get("rel_err_theta", math.inf) > 0.01:
            failed.append("c2_rk4_theta")
        if metrics.get("partition_drift", math.inf) > 1.0e-6:
            failed.append("c2_partition_drift")
    elif mode == "tmat_hot_e":
        if not metrics.get("monotonic_theta", False):
            failed.append("c2_tmat_monotonic_theta")
        if metrics.get("cv_e_diagnostic") == "present" and not metrics.get("cv_e_positive", False):
            failed.append("c2_tmat_cv_e_positive")
        if metrics.get("n_clamp", 0) != 0:
            failed.append("c2_tmat_no_clamps")
        if not metrics.get("table_domain_margin_ok", False):
            failed.append("c2_tmat_table_domain_margin")
    elif mode == "dt_sweep":
        factor = float(metrics.get("dt_factor") or 1.0)
        if factor <= 0.1 and metrics.get("rel_err_Te", math.inf) > (0.005 if factor <= 0.01 else 0.01):
            failed.append("c2_dt_sweep_rk4_Te")
        if abs(factor - 1.0) < 1.0e-12:
            if metrics.get("rel_err_Te", math.inf) > 0.02:
                failed.append("c2_dt_sweep_tau_rk4_Te")
            if metrics.get("rel_err_Te_map", math.inf) > 0.005:
                failed.append("c2_dt_sweep_tau_map_Te")
        if factor >= 10.0 and metrics.get("Te_map_Teq_norm_err", math.inf) > 0.001:
            failed.append("c2_dt_sweep_10tau_map_Te")
        if factor >= 10.0 and metrics.get("partition_drift", math.inf) > 1.0e-6:
            failed.append("c2_partition_drift")
        if not metrics.get("dt_sweep_bounds_ok", False):
            failed.append("c2_dt_sweep_bounds")
        if not metrics.get("theta_sign_unchanged", False):
            failed.append("c2_dt_sweep_theta_sign")
    elif mode == "qei_plus_cond_uniform":
        if not metrics.get("hot_e_baseline_available", False):
            failed.append("c2_uniform_hot_e_baseline_missing")
        elif metrics.get("max_abs_Te_vs_hot_e", math.inf) > args.uniform_te_abs_tol:
            failed.append("c2_uniform_matches_hot_e")
    elif mode == "qei_plus_cond_gradient":
        if metrics.get("total_E_rel_drift", math.inf) > 1.0e-6:
            failed.append("c2_gradient_energy_conservation")
        if not metrics.get("Te_spread_decreases", False):
            failed.append("c2_gradient_te_spread_decreases")
    return failed


def active_env(
    args: argparse.Namespace,
    mode: str,
    nr: int,
    nz: int,
    dt_factor: float | None,
    seed: int,
    outdir: Path,
) -> dict[str, str]:
    te0, _ = mode_initial_temperatures(mode)
    tau0 = tau_eff(te0, args)
    dt_force = (dt_factor if dt_factor is not None else 1.0) * tau_eff(100.0, args)
    t_end_default = args.t_end_s
    if t_end_default is None:
        t_end_default = 5.0 * tau0
        if mode == "hot_i_rk4":
            t_end_default = min(t_end_default, 2.8e-11)
        if mode == "dt_sweep":
            t_end_default = 5.0 * tau_eff(100.0, args)
    dt_initial = args.dt_initial_s if args.dt_initial_s is not None else min(0.001 * tau0, t_end_default)
    dt_max = args.dt_max_s if args.dt_max_s is not None else dt_initial
    env = {
        "TENRYU_C2_MODE": mode,
        "TENRYU_C2_NR": str(nr),
        "TENRYU_C2_NZ": str(nz),
        "TENRYU_C2_SEED": str(seed),
        "TENRYU_C2_OUTDIR": str(outdir),
        "TENRYU_C2_RHO_GCC": str(args.rho_gcc),
        "TENRYU_C2_A": str(args.A),
        "TENRYU_C2_ZBAR": str(args.zbar),
        "TENRYU_C2_GAMMA": str(args.gamma),
        "TENRYU_C2_T_END_S": str(t_end_default),
        "TENRYU_C2_DT_INITIAL_S": str(dt_initial),
        "TENRYU_C2_DT_MAX_S": str(dt_max),
        "TENRYU_C2_DT_FORCE_S": str(dt_force),
        "TENRYU_C2_MAX_STEPS": str(args.max_steps),
        "TENRYU_C2_TE_FLOOR_EV": str(args.te_floor_ev),
        "TENRYU_C2_TI_FLOOR_EV": str(args.ti_floor_ev),
        "TENRYU_C2_SOLVER": args.solver,
        "TENRYU_C2_F_LIM": str(args.f_lim),
        "TENRYU_C2_TEST_KAPPA": str(args.test_kappa),
        "TENRYU_C2_CFL_COND": str(args.cfl_cond),
    }
    if mode == "dt_sweep":
        env["TENRYU_C2_DT_INITIAL_S"] = str(dt_force)
        env["TENRYU_C2_DT_MAX_S"] = str(dt_force)
    return env


def append_run(args: argparse.Namespace, run: dict[str, Any]) -> int:
    cmd = [
        sys.executable,
        "tools/validation/append_run.py",
        "--stage-id", STAGE_ID,
        "--run-id", str(run["run_id"]),
        "--ladder-row", str(run["ladder_row"]),
        "--seed", str(run["seed"]),
        "--deck-path", args.deck,
        "--gpu-model", args.gpu_model,
        "--cuda-version", args.cuda_version,
        "--run-profile", args.run_profile,
        "--profile-class", args.profile_class,
        "--evidence-role", args.evidence_role,
        "--result", str(run["result"]),
        "--exit-code", str(run["exit_code"]),
        "--wall-time-s", f"{run['wall_time_s']:.9g}",
        "--output-bytes", str(run["output_bytes"]),
        "--root", args.root,
    ]
    if run.get("failure_class"):
        cmd.extend(["--failure-class", str(run["failure_class"])])
    if run.get("failed_gates"):
        cmd.extend(["--failed-gates", ",".join(str(gate) for gate in run["failed_gates"])])
    if run.get("metrics_ref"):
        cmd.extend(["--metrics-ref", str(run["metrics_ref"])])
    return subprocess.run(cmd).returncode


def run_one(
    args: argparse.Namespace,
    mode: str,
    nr: int,
    nz: int,
    dt_label: str,
    dt_factor: float | None,
    seed: int,
    baselines: dict[tuple[int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    args.seed_current = seed
    run_id = case_name(mode, nr, nz, dt_label, seed)
    ladder_row = f"{mode}_nr{nr}_nz{nz}_{dt_label}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, nr, nz, dt_factor, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    metrics: dict[str, Any] = {"n_clamp": parse_clamp_count(log_path), "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
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
            metrics = collect_metrics(
                args, mode, nr, nz, dt_factor, results_dir, run_id, log_path, wall_time_s, bytes_out, baselines
            )
            failed_gates = evaluate_gates(args, mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_c2] analysis failed: {exc}\n")

    rel_metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "dt_label": dt_label,
        "dt_factor": dt_factor,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
        "env_overrides": env_overrides,
    }
    write_json(Path(rel_metrics_ref), payload)

    run = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "dt_label": dt_label,
        "dt_factor": dt_factor,
        "result": result,
        "failure_class": failure_class,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics.get("output_bytes", output_bytes(outdir)),
        "tau_ei_initial_s": metrics.get("tau_ei_initial_s"),
        "Te_final_eV": metrics.get("Te_final_eV"),
        "Ti_final_eV": metrics.get("Ti_final_eV"),
        "rel_err_Te": metrics.get("rel_err_Te"),
        "partition_drift": metrics.get("partition_drift"),
        "n_clamp": metrics.get("n_clamp"),
    }
    if not args.no_append:
        append_code = append_run(args, run)
        if append_code != 0 and result == "PASS":
            run["result"] = "FAIL"
            run["failed_gates"] = ["append_run_failure"]
            run["failure_class"] = "append_run_failure"
    return run


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int, str, float | None, int]]:
    meshes = list(args.mesh_list or (SMOKE_MESHES if args.smoke else FULL_MESHES))
    selected = list(args.mode_list)
    rows: list[tuple[str, int, int, str, float | None, int]] = []
    for mode in selected:
        if mode == "dt_sweep":
            for nr, nz in meshes:
                for seed in args.seed_list:
                    rows.append((mode, nr, nz, "dt_nominal", 1.0, seed))
                    for factor in DT_FACTORS:
                        rows.append((mode, nr, nz, f"dt{safe_float_token(factor)}tau", factor, seed))
        else:
            for nr, nz in meshes:
                for seed in args.seed_list:
                    rows.append((mode, nr, nz, "default", None, seed))
    return rows


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
    for key in ("wall_time_s", "rel_err_Te", "partition_drift", "n_clamp"):
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
    print("Stage C2 Qei sweep summary")
    print("ladder_row seed result rel_err_Te partition_drift n_clamp wall_time_s failed_gates")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {rel_err_Te} {partition_drift} {n_clamp} "
            "{wall_time_s:.3f} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_c2_qei.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/c2_runs")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--run-profile", default="C2 Qei and Qei+conduction validation sweep")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    matrix = parser.add_mutually_exclusive_group()
    matrix.add_argument("--smoke", action="store_true", help="use 32x64 grid only")
    matrix.add_argument("--full", "--matrix-full", dest="full", action="store_true", help="use 32x64 and 128x256 grids")
    parser.add_argument("--rho-gcc", type=float, default=RHO_GCC_DEFAULT)
    parser.add_argument("--A", type=float, default=A_DEFAULT)
    parser.add_argument("--zbar", type=float, default=Z_DEFAULT)
    parser.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--te-floor-ev", type=float, default=1.0e-6)
    parser.add_argument("--ti-floor-ev", type=float, default=1.0e-6)
    parser.add_argument("--solver", default="sts")
    parser.add_argument("--f-lim", type=float, default=0.06)
    parser.add_argument("--test-kappa", type=float, default=2.0e14)
    parser.add_argument("--cfl-cond", type=float, default=0.1)
    parser.add_argument("--uniform-te-abs-tol", type=float, default=1.0e-10)
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/C2/stats.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-append", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows_spec = matrix_rows(args)
    if args.dry_run:
        print("Planned Stage C2 runs:")
        for mode, nr, nz, dt_label, dt_factor, seed in rows_spec:
            run_id = case_name(mode, nr, nz, dt_label, seed)
            print(f"{run_id} mode={mode} nr={nr} nz={nz} dt_label={dt_label} dt_factor={dt_factor} seed={seed}")
        print(f"total_runs={len(rows_spec)}")
        print(f"summary_json={args.summary_json}")
        return 0

    baselines: dict[tuple[int, int, int], dict[str, Any]] = {}
    rows = [
        run_one(args, mode, nr, nz, dt_label, dt_factor, seed, baselines)
        for mode, nr, nz, dt_label, dt_factor, seed in rows_spec
    ]
    print_summary(rows)
    write_summary(args, rows)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
