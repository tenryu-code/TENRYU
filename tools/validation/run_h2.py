#!/usr/bin/env python3
"""Run Stage H2 2D RZ cylindrical Noh validation."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np


STAGE_ID = "H2"
RHO_0_GCC = 1.0
V_SCALE_CM_S = 1.0e6
R_MIN_CM = 0.0
R_MAX_CM = 1.0
PRESHOCK_SAMPLE_R_CM = 0.3

SHOCK_RADIUS_ERR_CELLS_MAX = 1.5
# Mesh-ladder rho_post gate: 128x16 informational only, 256+ blocking
RHO_POST_ERR_PCT_MAX_INFO_NR = 128       # nr <= this → informational only (no failed_gate)
RHO_POST_ERR_PCT_MAX_FINE = 5.0          # spec §3 (gate at nr >= 256)
RHO_POST_ERR_PCT_MAX_INFORMATIONAL = 8.0 # tracked but not blocking at coarse mesh
P_POST_ERR_PCT_MAX = 8.0                 # spec §3: 5–8%
RHO_PRESHOCK_ERR_PCT_MAX = 5.0
TE_AXIS_MAX_EV_MAX = 100.0

ALE_DELTA_SHOCK_RADIUS_CELLS_MAX = 1.0
ALE_DELTA_RHO_POST_PCT_MAX = 3.0


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


def parse_float_list(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("float list is empty")
    return values


def parse_ale_every_n_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("ALE every_n list is empty")
    if any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("ALE every_n entries must be non-negative")
    return values


def float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p")


def case_name(nr: int, nz: int, seed: int, av_c2: float, cfl: float, ale_every: int) -> str:
    return (
        f"2d_rz_h2_noh_cyl_nr{nr}_nz{nz}_seed{seed}"
        f"_C2{float_token(av_c2)}_cfl{float_token(cfl)}_ale{ale_every}"
    )


def run_id_for(nr: int, nz: int, seed: int, av_c2: float, cfl: float, ale_every: int) -> str:
    return (
        f"h2_nr{nr}_nz{nz}_C2{float_token(av_c2)}"
        f"_cfl{float_token(cfl)}_ale{ale_every}_seed{seed}"
    )


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
    def plot_index(path: Path) -> int:
        suffix = path.stem.removeprefix(f"{name}_")
        return int(suffix)

    return sorted(
        (p for p in results_dir.glob(f"{name}_*.h5") if p.stem.removeprefix(f"{name}_").isdigit()),
        key=plot_index,
    )


def noh_shock_radius_dim(t_s: float) -> float:
    return (V_SCALE_CM_S / 3.0) * t_s


def noh_preshock_rho(R_cm: float, t_s: float) -> float:
    if R_cm <= 0.0:
        raise ValueError("R_cm must be positive for the Noh preshock density")
    return RHO_0_GCC * (1.0 + V_SCALE_CM_S * t_s / R_cm)


def noh_postshock_rho() -> float:
    return RHO_0_GCC * 16.0


def noh_postshock_pressure_erg_per_cc() -> float:
    return RHO_0_GCC * V_SCALE_CM_S * V_SCALE_CM_S * 16.0 / 3.0


def as_cell_field(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr, nz):
        return data.astype(float, copy=False)
    if data.size == nr * nz:
        return data.reshape((nr, nz)).astype(float, copy=False)
    raise ValueError(f"{name} has shape {data.shape}, expected {(nr, nz)} or flat {nr * nz}")


def as_node_field(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr + 1, nz + 1):
        return data.astype(float, copy=False)
    if data.size == (nr + 1) * (nz + 1):
        return data.reshape((nr + 1, nz + 1)).astype(float, copy=False)
    raise ValueError(f"{name} has shape {data.shape}, expected {(nr + 1, nz + 1)}")


def field_to_cells(handle: h5py.File, name: str, nr: int, nz: int) -> np.ndarray:
    data = np.asarray(handle[name][()])
    if data.shape == (nr, nz) or data.size == nr * nz:
        return as_cell_field(handle, name, nr, nz)
    if data.shape == (nr + 1, nz + 1) or data.size == (nr + 1) * (nz + 1):
        nodes = as_node_field(handle, name, nr, nz)
        return 0.25 * (nodes[:-1, :-1] + nodes[1:, :-1] + nodes[:-1, 1:] + nodes[1:, 1:])
    raise ValueError(f"{name} has shape {data.shape}, cannot map to cells")


def read_pressure(handle: h5py.File, nr: int, nz: int) -> np.ndarray:
    if "hydro/P" in handle:
        return as_cell_field(handle, "hydro/P", nr, nz)
    if "hydro/Pi+Pe" in handle:
        return as_cell_field(handle, "hydro/Pi+Pe", nr, nz)
    if "hydro/Pi" in handle and "hydro/Pe" in handle:
        return as_cell_field(handle, "hydro/Pi", nr, nz) + as_cell_field(handle, "hydro/Pe", nr, nz)
    raise KeyError("no pressure field found; expected hydro/P, hydro/Pi+Pe, or hydro/Pi plus hydro/Pe")


def cell_centers(handle: h5py.File, nr: int, nz: int) -> tuple[np.ndarray, np.ndarray]:
    x_r = as_node_field(handle, "mesh/x_r", nr, nz)
    x_z = as_node_field(handle, "mesh/x_z", nr, nz)
    r_cell = 0.25 * (x_r[:-1, :-1] + x_r[1:, :-1] + x_r[:-1, 1:] + x_r[1:, 1:])
    z_cell = 0.25 * (x_z[:-1, :-1] + x_z[1:, :-1] + x_z[:-1, 1:] + x_z[1:, 1:])
    return r_cell, z_cell


def shock_radius(r_axis: np.ndarray, rho_radial: np.ndarray) -> float:
    grad = np.abs(np.gradient(rho_radial, r_axis))
    mask = (r_axis >= 0.05) & (r_axis <= 0.5)
    if not np.any(mask):
        mask = np.ones_like(r_axis, dtype=bool)
    indices = np.nonzero(mask)[0]
    return float(r_axis[indices[int(np.argmax(grad[indices]))]])


def postshock_plateau_mean(r_axis: np.ndarray, values: np.ndarray, shock_radius_sim_cm: float, nr: int) -> float:
    dR = (R_MAX_CM - R_MIN_CM) / nr
    lo = max(0.05, 3.0 * dR)
    hi = shock_radius_sim_cm - 2.0 * dR
    mask = (r_axis >= lo) & (r_axis <= hi)
    if not np.any(mask):
        mask = (r_axis >= 3.0 * dR) & (r_axis < shock_radius_sim_cm - dR)
    if not np.any(mask):
        raise RuntimeError("could not identify post-shock plateau cells")
    return float(np.mean(values[mask]))


def percent_error(sim: float, analytic: float) -> float:
    return abs(sim - analytic) / max(abs(analytic), 1.0e-300) * 100.0


def analyze_plot(path: Path, nr: int, nz: int, ale_every: int) -> dict[str, float | int | None]:
    with h5py.File(path, "r") as handle:
        r_cell, _ = cell_centers(handle, nr, nz)
        rho_cell = as_cell_field(handle, "hydro/rho", nr, nz)
        pressure_cell = read_pressure(handle, nr, nz)
        Te_cell = as_cell_field(handle, "hydro/Te", nr, nz)
        v_r_cell = field_to_cells(handle, "mesh/v_r", nr, nz)
        t_final_s = read_scalar(handle, "time_state/t")
        step_final = read_int_scalar(handle, "time_state/step")

    r_axis = r_cell.mean(axis=1)
    rho_radial = rho_cell.mean(axis=1)
    pressure_radial = pressure_cell.mean(axis=1)
    v_r_radial = v_r_cell.mean(axis=1)

    shock_radius_analytic = noh_shock_radius_dim(t_final_s)
    shock_radius_sim = shock_radius(r_axis, rho_radial)
    dR = (R_MAX_CM - R_MIN_CM) / nr
    shock_radius_err_cells = abs(shock_radius_sim - shock_radius_analytic) / max(dR, 1.0e-300)

    rho_post_sim = postshock_plateau_mean(r_axis, rho_radial, shock_radius_sim, nr)
    pressure_post_sim = postshock_plateau_mean(r_axis, pressure_radial, shock_radius_sim, nr)
    u_r_post_sim = postshock_plateau_mean(r_axis, v_r_radial, shock_radius_sim, nr)

    preshock_idx = int(np.argmin(np.abs(r_axis - PRESHOCK_SAMPLE_R_CM)))
    rho_preshock_sim = float(rho_radial[preshock_idx])
    rho_preshock_analytic = noh_preshock_rho(PRESHOCK_SAMPLE_R_CM, t_final_s)

    return {
        "shock_radius_sim_cm": shock_radius_sim,
        "shock_radius_analytic_cm": shock_radius_analytic,
        "shock_radius_err_cells": float(shock_radius_err_cells),
        "rho_post_sim_gcc": rho_post_sim,
        "rho_post_analytic_gcc": noh_postshock_rho(),
        "rho_post_err_pct": percent_error(rho_post_sim, noh_postshock_rho()),
        "P_post_sim_erg_per_cc": pressure_post_sim,
        "P_post_analytic_erg_per_cc": noh_postshock_pressure_erg_per_cc(),
        "P_post_err_pct": percent_error(pressure_post_sim, noh_postshock_pressure_erg_per_cc()),
        "rho_preshock_at_0p3_sim_gcc": rho_preshock_sim,
        "rho_preshock_at_0p3_analytic_gcc": rho_preshock_analytic,
        "rho_preshock_err_pct": percent_error(rho_preshock_sim, rho_preshock_analytic),
        "Te_axis_max_eV": float(np.max(Te_cell[: min(2, nr), :])),
        "u_r_post_sim_cm_s": u_r_post_sim,
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "t_final_s": t_final_s,
        "step_final": step_final,
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
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


def empty_metrics(wall_time_s: float, bytes_out: int, ale_every: int) -> dict[str, object]:
    return {
        "shock_radius_sim_cm": None,
        "shock_radius_analytic_cm": None,
        "shock_radius_err_cells": None,
        "rho_post_sim_gcc": None,
        "rho_post_analytic_gcc": noh_postshock_rho(),
        "rho_post_err_pct": None,
        "P_post_sim_erg_per_cc": None,
        "P_post_analytic_erg_per_cc": noh_postshock_pressure_erg_per_cc(),
        "P_post_err_pct": None,
        "rho_preshock_at_0p3_sim_gcc": None,
        "rho_preshock_at_0p3_analytic_gcc": None,
        "rho_preshock_err_pct": None,
        "Te_axis_max_eV": None,
        "u_r_post_sim_cm_s": None,
        "ale_every_n_steps": ale_every,
        "ale_event_count": None,
        "t_final_s": None,
        "step_final": None,
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }


def thresholds_payload() -> dict[str, float]:
    return {
        "shock_radius_err_cells_max": SHOCK_RADIUS_ERR_CELLS_MAX,
        "rho_post_err_pct_max_fine": RHO_POST_ERR_PCT_MAX_FINE,
        "rho_post_err_pct_max_coarse_informational": RHO_POST_ERR_PCT_MAX_INFORMATIONAL,
        "rho_post_err_pct_max_info_nr": RHO_POST_ERR_PCT_MAX_INFO_NR,
        "P_post_err_pct_max": P_POST_ERR_PCT_MAX,
        "rho_preshock_err_pct_max": RHO_PRESHOCK_ERR_PCT_MAX,
        "Te_axis_max_eV_max": TE_AXIS_MAX_EV_MAX,
    }


def run_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    seed: int,
    av_c2: float,
    cfl: float,
    ale_every: int,
) -> dict[str, object]:
    run_id = run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
    ladder_row = f"nr{nr}_nz{nz}_ale{ale_every}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TENRYU_H2_NR": str(nr),
            "TENRYU_H2_NZ": str(nz),
            "TENRYU_H2_SEED": str(seed),
            "TENRYU_H2_OUTDIR": str(outdir),
            "TENRYU_H2_AV_C2": f"{av_c2:.12g}",
            "TENRYU_H2_CFL": f"{cfl:.12g}",
            "TENRYU_H2_T_END_S": f"{args.t_end_s:.12g}",
            "TENRYU_H2_ALE_EVERY_N_STEPS": str(ale_every),
        }
    )

    log_path = outdir / "run.log"
    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    result = "PASS"
    failed_gates: list[str] = []
    metrics = empty_metrics(wall_time_s, output_bytes(outdir), ale_every)

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

            plot_files = locate_plot_files(results_dir, case_name(nr, nz, seed, av_c2, cfl, ale_every))
            if not plot_files:
                raise RuntimeError("expected at least 1 plot file, found 0")

            metrics.update(analyze_plot(plot_files[-1], nr, nz, ale_every))
            metrics["wall_time_s"] = wall_time_s
            metrics["output_bytes"] = metrics["output_bytes"]

            if float(metrics["shock_radius_err_cells"]) > SHOCK_RADIUS_ERR_CELLS_MAX:
                failed_gates.append("shock_radius_err_cells")
            # rho_post: mesh-ladder gate. Coarse (nr <= 128) is informational only.
            if nr > RHO_POST_ERR_PCT_MAX_INFO_NR and float(metrics["rho_post_err_pct"]) > RHO_POST_ERR_PCT_MAX_FINE:
                failed_gates.append("rho_post_err_pct")
            elif nr <= RHO_POST_ERR_PCT_MAX_INFO_NR and float(metrics["rho_post_err_pct"]) > RHO_POST_ERR_PCT_MAX_INFORMATIONAL:
                # informational tier upper bound (catastrophic if exceeded, e.g. > 8%)
                failed_gates.append("rho_post_err_pct_coarse_informational")
            if float(metrics["P_post_err_pct"]) > P_POST_ERR_PCT_MAX:
                failed_gates.append("P_post_err_pct")
            if float(metrics["rho_preshock_err_pct"]) > RHO_PRESHOCK_ERR_PCT_MAX:
                failed_gates.append("rho_preshock_err_pct")
            if float(metrics["Te_axis_max_eV"]) > TE_AXIS_MAX_EV_MAX:
                failed_gates.append("Te_axis_max_eV")
            if failed_gates:
                result = "FAIL"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_h2] analysis failed: {exc}\n")

    rel_metrics_ref = f"docs/validation/2d_rz/H2/metrics/{run_id}.json"
    metrics_path = Path(rel_metrics_ref)
    payload = {
        "stage_id": STAGE_ID,
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "metrics": metrics,
        "thresholds": thresholds_payload(),
        "result": result,
    }
    write_json(metrics_path, payload)

    run = {
        "run_id": run_id,
        "ladder_row": ladder_row,
        "seed": seed,
        "av_c2": av_c2,
        "cfl": cfl,
        "ale_every": ale_every,
        "nr": nr,
        "nz": nz,
        "result": result,
        "exit_code": proc.returncode,
        "failed_gates": failed_gates,
        "metrics_ref": rel_metrics_ref,
        "wall_time_s": wall_time_s,
        "output_bytes": metrics["output_bytes"],
        "shock_radius_sim_cm": metrics["shock_radius_sim_cm"],
        "shock_radius_err_cells": metrics["shock_radius_err_cells"],
        "rho_post_sim_gcc": metrics["rho_post_sim_gcc"],
        "rho_post_err_pct": metrics["rho_post_err_pct"],
        "P_post_err_pct": metrics["P_post_err_pct"],
        "rho_preshock_err_pct": metrics["rho_preshock_err_pct"],
        "Te_axis_max_eV": metrics["Te_axis_max_eV"],
    }
    append_code = append_run(args, run)
    if append_code != 0 and result == "PASS":
        run["result"] = "FAIL"
        run["failed_gates"] = ["append_run_failure"]
    return run


def ale_variant_comparisons(rows: list[dict[str, object]]) -> dict[str, object]:
    by_key: dict[tuple[int, int, int, float, float], dict[int, dict[str, object]]] = {}
    for row in rows:
        key = (int(row["nr"]), int(row["nz"]), int(row["seed"]), float(row["av_c2"]), float(row["cfl"]))
        by_key.setdefault(key, {})[int(row["ale_every"])] = row

    comparisons = []
    for key, variants in sorted(by_key.items()):
        base = variants.get(0)
        if base is None:
            continue
        nr, nz, seed, av_c2, cfl = key
        dR = (R_MAX_CM - R_MIN_CM) / nr
        for ale_every in sorted(item for item in variants if item != 0):
            row = variants[ale_every]
            shock_base = base.get("shock_radius_sim_cm")
            shock_variant = row.get("shock_radius_sim_cm")
            rho_base = base.get("rho_post_sim_gcc")
            rho_variant = row.get("rho_post_sim_gcc")
            if shock_base is None or shock_variant is None or rho_base is None or rho_variant is None:
                comparison = {
                    "nr": nr,
                    "nz": nz,
                    "seed": seed,
                    "av_c2": av_c2,
                    "cfl": cfl,
                    "ale_every_n_steps": ale_every,
                    "baseline_ale_every_n_steps": 0,
                    "delta_shock_radius_cells": None,
                    "delta_rho_post_pct": None,
                    "within_threshold": False,
                }
            else:
                delta_shock = abs(float(shock_variant) - float(shock_base)) / max(dR, 1.0e-300)
                delta_rho = abs(float(rho_variant) - float(rho_base)) / max(abs(float(rho_base)), 1.0e-300) * 100.0
                comparison = {
                    "nr": nr,
                    "nz": nz,
                    "seed": seed,
                    "av_c2": av_c2,
                    "cfl": cfl,
                    "ale_every_n_steps": ale_every,
                    "baseline_ale_every_n_steps": 0,
                    "delta_shock_radius_cells": float(delta_shock),
                    "delta_rho_post_pct": float(delta_rho),
                    "within_threshold": (
                        delta_shock < ALE_DELTA_SHOCK_RADIUS_CELLS_MAX
                        and delta_rho < ALE_DELTA_RHO_POST_PCT_MAX
                    ),
                }
            comparisons.append(comparison)

    return {
        "stage_id": STAGE_ID,
        "thresholds": {
            "delta_shock_radius_cells_lt": ALE_DELTA_SHOCK_RADIUS_CELLS_MAX,
            "delta_rho_post_pct_lt": ALE_DELTA_RHO_POST_PCT_MAX,
        },
        "comparisons": comparisons,
    }


def write_ale_variant_check(args: argparse.Namespace, rows: list[dict[str, object]]) -> dict[str, object]:
    payload = ale_variant_comparisons(rows)
    write_json(Path(args.root) / STAGE_ID / "ale_variant_check.json", payload)
    return payload


def print_ale_variant_check(payload: dict[str, object]) -> None:
    comparisons = payload.get("comparisons", [])
    if not comparisons:
        print("ale_variant_check no comparable ALE-on/off pairs")
        return
    print("ale_variant_check ale baseline delta_shock_cells delta_rho_post_pct status")
    for item in comparisons:
        delta_shock = item["delta_shock_radius_cells"]
        delta_rho = item["delta_rho_post_pct"]
        status = "PASS" if item["within_threshold"] else "CHECK"
        print(
            "nr{nr}_nz{nz} seed={seed} C2={C2} cfl={cfl} ale={ale} baseline=0 "
            "shock_delta={shock} rho_delta={rho} {status}".format(
                nr=item["nr"],
                nz=item["nz"],
                seed=item["seed"],
                C2=float_token(float(item["av_c2"])),
                cfl=float_token(float(item["cfl"])),
                ale=item["ale_every_n_steps"],
                shock=delta_shock,
                rho=delta_rho,
                status=status,
            )
        )


def print_summary(rows: list[dict[str, object]], ale_payload: dict[str, object]) -> None:
    print(
        "Stage H2 thresholds: shock_radius_err_cells <= 1.5, rho_post_err_pct <= 5 (fine, nr>128) / <=8 (coarse informational), "
        "P_post_err_pct <= 8, rho_preshock_err_pct <= 5, Te_axis_max_eV <= 100"
    )
    print(
        "ladder_row seed C2 cfl result shock_cells rho_post_pct P_post_pct "
        "rho_preshock_pct Te_axis_max_eV wall_time_s"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {av_c2} {cfl} {result} {shock_radius_err_cells} "
            "{rho_post_err_pct} {P_post_err_pct} {rho_preshock_err_pct} "
            "{Te_axis_max_eV} {wall_time_s:.3f}".format(
                ladder_row=row["ladder_row"],
                seed=row["seed"],
                av_c2=float_token(float(row["av_c2"])),
                cfl=float_token(float(row["cfl"])),
                result=row["result"],
                shock_radius_err_cells=row["shock_radius_err_cells"],
                rho_post_err_pct=row["rho_post_err_pct"],
                P_post_err_pct=row["P_post_err_pct"],
                rho_preshock_err_pct=row["rho_preshock_err_pct"],
                Te_axis_max_eV=row["Te_axis_max_eV"],
                wall_time_s=float(row["wall_time_s"]),
            )
        )
    print_ale_variant_check(ale_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_h2_noh_cyl.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--gpu-model", default="NVIDIA GeForce RTX 4090")
    parser.add_argument("--cuda-version", default="12.0")
    parser.add_argument("--profile-class", default="ci_short", choices=("ci_short", "paper_full", "exploratory"))
    parser.add_argument("--evidence-role", default="ci_gate", choices=("ci_gate", "paper_figure", "exploratory"))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=parse_mesh_list("128,16;256,16;512,16"))
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345,12346,12347,12348,12349"))
    parser.add_argument("--av-c2-list", type=parse_float_list, default=parse_float_list("1.5"))
    parser.add_argument("--cfl-list", type=parse_float_list, default=parse_float_list("0.1"))
    parser.add_argument("--ale-every-n-list", type=parse_ale_every_n_list, default=parse_ale_every_n_list("0,2,10"))
    parser.add_argument("--t-end-s", type=float, default=6.0e-7)
    parser.add_argument("--out-root", default="./build/h2_runs")
    parser.add_argument("--run-profile", default="H2 cylindrical Noh mesh ladder seed ALE sweep")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mesh_list = parse_mesh_list("128,16;256,16") if args.quick else list(args.mesh_list)
    ale_every_n_list = parse_ale_every_n_list("0") if args.quick else list(args.ale_every_n_list)

    matrix = [
        (nr, nz, seed, av_c2, cfl, ale_every)
        for nr, nz in mesh_list
        for seed in args.seed_list
        for av_c2 in args.av_c2_list
        for cfl in args.cfl_list
        for ale_every in ale_every_n_list
    ]

    if args.dry_run:
        label = "quick " if args.quick else ""
        print(f"Planned Stage H2 {label}runs:")
        for nr, nz, seed, av_c2, cfl, ale_every in matrix:
            run_id = run_id_for(nr, nz, seed, av_c2, cfl, ale_every)
            print(
                f"{run_id} nr{nr}_nz{nz}_ale{ale_every} seed={seed} "
                f"C2={float_token(av_c2)} cfl={float_token(cfl)}"
            )
        print(f"total_runs={len(matrix)}")
        return 0

    rows = [run_one(args, nr, nz, seed, av_c2, cfl, ale_every) for nr, nz, seed, av_c2, cfl, ale_every in matrix]
    ale_payload = write_ale_variant_check(args, rows)
    print_summary(rows, ale_payload)
    return 0 if all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
