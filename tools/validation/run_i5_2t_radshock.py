#!/usr/bin/env python3
"""Run and analyze the I5 2T FLD-CED 2D RZ rad-shock battery."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import radiative_shock_metrics as shock_metrics
import run_i1_2d_rz_fld_ced as i1_rz
import run_i1_le08 as i1_base


STAGE_ID = "I5"
LADDER_ROW = "i5_2t_radshock_2d_rz_slab"
SUMMARY_SCHEMA = "tenryu.i5_2t_radshock_2d_rz_slab.summary.v1"
DIMENSION = "2D_RZ"

REGIME_ORDER = ("R1", "R0p1", "R10")
REFERENCE_BY_REGIME = {
    "R0p1": "fld_ced_2t_M3_APRIME_SCALED_R0p1.json",
    "R1": "fld_ced_2t_M3_APRIME_SCALED_R1.json",
    "R10": "fld_ced_2t_M3_APRIME_SCALED_R10.json",
}
NZ_BY_REGIME_RUNG = {
    "R0p1": {"SMOKE": 256, "PRODUCTION": 1024, "RICHARDSON": 2048},
    "R1": {"SMOKE": 128, "PRODUCTION": 512, "RICHARDSON": 1024},
    "R10": {"SMOKE": 512, "PRODUCTION": 2048, "RICHARDSON": 4096},
}

PRODUCTION_CG_MAX_ITER = 8192
PRODUCTION_CG_INNER_TOL = 1.0e-8

LOW_PRAD_GATE = i1_base.LOW_PRAD_GATE
CONSERVATION_GATE = i1_base.CONSERVATION_GATE
REFERENCE_IDENTITY_MEDIAN_GATE = i1_base.REFERENCE_IDENTITY_MEDIAN_GATE
REFERENCE_IDENTITY_P95_GATE = i1_base.REFERENCE_IDENTITY_P95_GATE
L2_REL_GATE = 0.05
SEP_REL_GATE = 0.10
RADIAL_INVARIANCE_GATE = i1_rz.RADIAL_INVARIANCE_GATE
RADIAL_INVARIANCE_GATE_FIELDS = i1_rz.RADIAL_INVARIANCE_GATE_FIELDS
SHOCK_WINDOW_MFP_COUNT = 10


@dataclass(frozen=True)
class LegSpec:
    regime: str
    rung: str
    nr: int
    nz: int
    seed: int
    init_mode: str
    boundary_mode: str
    reference: Path

    @property
    def run_id(self) -> str:
        return (
            f"{LADDER_ROW}_{self.regime}_{self.rung.lower()}"
            f"_nr{self.nr}_nz{self.nz}_seed{self.seed}_{self.init_mode}_{self.boundary_mode}"
        )


def truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def optional_finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def output_bytes(path: Path) -> int:
    return i1_base.output_bytes(path)


def discover_actual_outdir(outdir: Path) -> Path:
    return i1_base.discover_actual_outdir(outdir)


def discover_results_dir(outdir: Path) -> Path:
    return i1_base.discover_results_dir(outdir)


def run_info(outdir: Path) -> dict[str, Any]:
    path = discover_actual_outdir(outdir) / "run_info.json"
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def termination_reason(outdir: Path) -> str | None:
    info = run_info(outdir)
    value = info.get("termination_reason")
    return str(value) if value is not None else None


def reference_path_for_regime(reference_root: Path, regime: str) -> Path:
    return reference_root / REFERENCE_BY_REGIME[regime]


def deck_case_name(spec: LegSpec, ref: dict[str, Any]) -> str:
    mach = float(ref["parameters"]["M0"])
    return (
        f"{LADDER_ROW}_{spec.regime}_M{i1_base.safe_float_token(mach)}"
        f"_nr{spec.nr}_nz{spec.nz}_seed{spec.seed}_{spec.init_mode}_{spec.boundary_mode}"
    )


def leg_env(args: argparse.Namespace, spec: LegSpec, outdir: Path) -> dict[str, str]:
    env = {
        "TENRYU_I5_2D_RZ_REGIME": spec.regime,
        "TENRYU_I5_2D_RZ_NR": str(spec.nr),
        "TENRYU_I5_2D_RZ_NZ": str(spec.nz),
        "TENRYU_I5_2D_RZ_SEED": str(spec.seed),
        "TENRYU_I5_2D_RZ_OUTDIR": str(outdir),
        "TENRYU_I5_2D_RZ_REFERENCE": str(spec.reference),
        "TENRYU_I5_2D_RZ_INIT_MODE": spec.init_mode,
        "TENRYU_I5_2D_RZ_BOUNDARY_MODE": spec.boundary_mode,
        "TENRYU_I5_2D_RZ_PRODUCTION_AUDIT_TIER": args.production_audit_tier,
        "TENRYU_I5_2D_RZ_VERBOSITY": args.verbosity,
    }
    if spec.rung in ("PRODUCTION", "RICHARDSON"):
        env["TENRYU_I5_2D_RZ_FLD_CG_MAX_ITER"] = str(PRODUCTION_CG_MAX_ITER)
        env["TENRYU_I5_2D_RZ_FLD_CG_INNER_TOL"] = f"{PRODUCTION_CG_INNER_TOL:.12g}"
    return env


def run_deck(args: argparse.Namespace, spec: LegSpec, ref: dict[str, Any]) -> dict[str, Any]:
    del ref
    outdir = Path(args.out_root) / spec.run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = leg_env(args, spec, outdir)
    log_path = outdir / "run.log"
    resume_enabled = bool(args.resume) or truthy_env("TENRYU_I5_BATTERY_RESUME")
    actual = discover_actual_outdir(outdir)
    existing_reason = termination_reason(outdir)
    if resume_enabled and existing_reason == "t_end_reached":
        return {
            "run_id": spec.run_id,
            "run_label": spec.rung.lower(),
            "regime": spec.regime,
            "rung": spec.rung,
            "dimension": DIMENSION,
            "nr": spec.nr,
            "nz": spec.nz,
            "exit_code": 0,
            "timed_out": False,
            "wall_time_s": 0.0,
            "output_bytes": output_bytes(actual),
            "termination_reason": existing_reason,
            "run_status": "resumed_completed",
            "resume_enabled": True,
            "actual_outdir": str(actual),
            "log_path": str(log_path),
            "env": env_overrides,
            "analysis_error": None,
        }

    env = os.environ.copy()
    env.update(env_overrides)
    start = time.monotonic()
    timed_out = False
    proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
    wall_time_s = time.monotonic() - start
    actual = discover_actual_outdir(outdir)
    reason = termination_reason(outdir)
    return {
        "run_id": spec.run_id,
        "run_label": spec.rung.lower(),
        "regime": spec.regime,
        "rung": spec.rung,
        "dimension": DIMENSION,
        "nr": spec.nr,
        "nz": spec.nz,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(actual),
        "termination_reason": reason,
        "run_status": (
            "wall_time_timeout"
            if timed_out
            else "completed"
            if proc.returncode == 0 and reason == "t_end_reached"
            else "crashed"
            if proc.returncode != 0
            else "not_completed"
        ),
        "resume_enabled": resume_enabled,
        "actual_outdir": str(actual),
        "log_path": str(log_path),
        "env": env_overrides,
        "analysis_error": None,
    }


def read_leg_snapshots(args: argparse.Namespace, spec: LegSpec, ref: dict[str, Any]) -> list[Any]:
    outdir = Path(args.out_root) / spec.run_id
    results_dir = discover_results_dir(outdir)
    return i1_base.read_snapshots(
        results_dir,
        deck_case_name(spec, ref),
        spec.nr,
        spec.nz,
        DIMENSION,
        min_count=1,
    )


def sorted_profile(z: Any, *fields: Any) -> tuple[np.ndarray, ...]:
    z_arr = np.asarray(z, dtype=float)
    arrays = [np.asarray(field, dtype=float) for field in fields]
    mask = np.isfinite(z_arr)
    for arr in arrays:
        mask &= np.isfinite(arr)
    if np.count_nonzero(mask) == 0:
        return (np.asarray([], dtype=float),) + tuple(np.asarray([], dtype=float) for _ in arrays)
    order = np.argsort(z_arr[mask])
    out = [z_arr[mask][order]]
    out.extend(arr[mask][order] for arr in arrays)
    return tuple(out)


def locate_shock_by_max_grad_te(z: np.ndarray, Te: np.ndarray) -> dict[str, Any]:
    z_sorted, te_sorted = sorted_profile(z, Te)
    if z_sorted.size < 3:
        return {
            "shock_front_locator": "max_abs_grad_Te",
            "shock_front_position_cm": math.nan,
            "shock_front_index": -1,
            "shock_front_peak_abs_grad_Te_eV_per_cm": math.nan,
        }
    grad = np.abs(np.gradient(te_sorted, z_sorted))
    grad[~np.isfinite(grad)] = -math.inf
    idx = int(np.nanargmax(grad))
    return {
        "shock_front_locator": "max_abs_grad_Te",
        "shock_front_position_cm": float(z_sorted[idx]),
        "shock_front_index": idx,
        "shock_front_peak_abs_grad_Te_eV_per_cm": float(grad[idx]),
    }


def reference_postshock_mfp_cm(ref: dict[str, Any]) -> float:
    rho_post = float(ref["states"]["downstream"]["rho"])
    kappa = float(ref["parameters"]["kappa_R_cm2_per_g"])
    return 1.0 / max(kappa * rho_post, i1_base.E_FLOOR)


def rel_l2_window(
    code: np.ndarray,
    ref_conv: np.ndarray,
    x_aligned: np.ndarray,
    mfp_post_cm: float,
) -> dict[str, Any]:
    return shock_metrics.shock_windowed_l2(
        np.asarray(code, dtype=float),
        np.asarray(ref_conv, dtype=float),
        np.asarray(x_aligned, dtype=float),
        0.0,
        float(mfp_post_cm),
        N=SHOCK_WINDOW_MFP_COUNT,
    )


def interpolate_reference(ref: dict[str, Any], table_key: str, x: np.ndarray) -> np.ndarray:
    ref_x, ref_y = sorted_profile(ref["table"]["x_cm"], ref["table"][table_key])
    if ref_x.size < 2:
        return np.full(np.asarray(x, dtype=float).shape, np.nan, dtype=float)
    return np.interp(np.asarray(x, dtype=float), ref_x, ref_y, left=np.nan, right=np.nan)


def convolved_reference(
    ref: dict[str, Any],
    table_key: str,
    kernel: dict[str, Any],
    x_aligned: np.ndarray,
) -> np.ndarray:
    return shock_metrics.cell_averaged_convolved_reference(
        np.asarray(ref["table"]["x_cm"], dtype=float),
        np.asarray(ref["table"][table_key], dtype=float),
        np.asarray(kernel["kernel_z"], dtype=float),
        np.asarray(kernel["kernel"], dtype=float),
        np.asarray(x_aligned, dtype=float),
    )


def code_low_prad_max(snapshot: Any) -> float:
    p_rad = np.asarray(snapshot.rad_E, dtype=float) / 3.0
    p_mat = np.asarray(snapshot.Pe, dtype=float) + np.asarray(snapshot.Pi, dtype=float)
    ratio = p_rad / np.maximum(p_mat, i1_base.E_FLOOR)
    finite = ratio[np.isfinite(ratio)]
    return float(np.max(finite)) if finite.size else math.inf


def reference_identity_2t(ref: dict[str, Any]) -> dict[str, Any]:
    diag = ref.get("diagnostics", {})
    sv3 = diag.get("sv3_invariants", {})
    conservation = max(
        [
            abs(float(sv3.get("mass_flux_residual_rel_max", math.inf))),
            abs(float(sv3.get("momentum_flux_residual_rel_max", math.inf))),
            abs(float(sv3.get("energy_flux_residual_rel_max", math.inf))),
        ]
    )
    sv4 = diag.get("sv4_moment_identities", {})
    branch_rows: dict[str, Any] = {}
    medians: list[float] = []
    p95s: list[float] = []
    for branch in ("upstream", "downstream"):
        branch_rows[branch] = {}
        for identity in ("flux_law", "moment"):
            row = sv4.get(branch, {}).get(identity, {})
            median = float(row.get("median_rel", math.inf))
            p95 = float(row.get("p95_rel", math.inf))
            branch_rows[branch][identity] = {
                "median_rel": median,
                "p95_rel": p95,
                "max_rel": float(row.get("max_rel", math.inf)),
            }
            medians.append(median)
            p95s.append(p95)
    identity_passed = bool(medians) and max(medians) <= REFERENCE_IDENTITY_MEDIAN_GATE and max(p95s) <= REFERENCE_IDENTITY_P95_GATE
    return {
        "schema_version": ref.get("schema_version"),
        "problem": ref.get("problem"),
        "reference_model": ref.get("reference_model"),
        "reference_conservation_residual_rel_max": conservation,
        "reference_conservation_passed": conservation <= CONSERVATION_GATE,
        "reference_identity_median_gate": REFERENCE_IDENTITY_MEDIAN_GATE,
        "reference_identity_p95_gate": REFERENCE_IDENTITY_P95_GATE,
        "reference_identity_max_median_rel": max(medians) if medians else math.inf,
        "reference_identity_max_p95_rel": max(p95s) if p95s else math.inf,
        "reference_identity_passed": identity_passed,
        "sv4_moment_identities": branch_rows,
    }


def parse_newton_multisolve(log_path: Path, n_cells: int, snapshot: Any | None) -> dict[str, Any]:
    diag = i1_rz.parse_newton_diagnostic(log_path, n_cells)
    records: list[dict[str, Any]] = []
    for line in read_text(log_path).splitlines():
        match = i1_rz._NEWTON_RE.search(line)
        if not match:
            continue
        records.append(
            {
                "newton_converged_count": int(match.group("converged")),
                "newton_cap_hit_count": int(match.group("cap_hit")),
                "newton_invalid_count": int(match.group("invalid")),
            }
        )
    if snapshot is not None:
        i1_rz.repair_newton_residual_scale(diag, snapshot)
    else:
        i1_rz.repair_newton_residual_scale(diag, None)
    converged_counts = [row["newton_converged_count"] for row in records]
    count_rule = bool(converged_counts) and all(
        count >= n_cells and count % max(n_cells, 1) == 0 for count in converged_counts
    )
    invalid_zero = bool(records) and all(row["newton_invalid_count"] == 0 for row in records)
    cap_zero = bool(records) and all(row["newton_cap_hit_count"] == 0 for row in records)
    diag.update(
        {
            "multisolve_tolerant_rule": "converged_count % n_cells == 0 and converged_count >= n_cells and invalid == 0 and cap_hit == 0",
            "multisolve_tolerant_count_pass": count_rule,
            "all_invalid_zero_multisolve": invalid_zero,
            "all_cap_hit_zero_multisolve": cap_zero,
            "newton_multisolve_pass": count_rule and invalid_zero and cap_zero,
        }
    )
    return diag


def no_escape_for_leg(
    args: argparse.Namespace,
    spec: LegSpec,
    ref: dict[str, Any],
    newton: dict[str, Any],
) -> dict[str, Any]:
    outdir = Path(args.out_root) / spec.run_id
    try:
        results_dir = discover_results_dir(outdir)
    except (OSError, RuntimeError, FileNotFoundError):
        results_dir = None
    return i1_rz.no_escape_valves_from_sources(
        outdir / "run.log",
        results_dir,
        deck_case_name(spec, ref),
        newton,
    )


def analyze_profile_leg(
    snapshot: Any,
    ref: dict[str, Any],
) -> dict[str, Any]:
    bundle = i1_rz.profile_bundle(snapshot)
    z = np.asarray(bundle["z_cm"], dtype=float)
    profiles = bundle["profiles"]
    Te = np.asarray(profiles["Te"], dtype=float)
    Ti = np.asarray(profiles["Ti"], dtype=float)
    shock = locate_shock_by_max_grad_te(z, Te)
    z_shock = float(shock["shock_front_position_cm"])
    x_aligned = z - z_shock if math.isfinite(z_shock) else np.full_like(z, np.nan)
    mfp_post = reference_postshock_mfp_cm(ref)
    dx = i1_base.median_cell_spacing(x_aligned)
    half_width = max(3.0 * dx, min(1.0, SHOCK_WINDOW_MFP_COUNT * mfp_post))
    kernel = shock_metrics.measure_hydro_kernel(Te, x_aligned, 0.0, half_width_cm=half_width)

    ref_Te_conv = convolved_reference(ref, "Te_eV", kernel, x_aligned)
    ref_Ti_conv = convolved_reference(ref, "Ti_eV", kernel, x_aligned)
    delta_code = Ti / np.maximum(Te, i1_base.E_FLOOR) - 1.0
    ref_delta_conv = convolved_reference(ref, "delta_ei", kernel, x_aligned)

    l2_Te = rel_l2_window(Te, ref_Te_conv, x_aligned, mfp_post)
    l2_Ti = rel_l2_window(Ti, ref_Ti_conv, x_aligned, mfp_post)
    sep = rel_l2_window(delta_code, ref_delta_conv, x_aligned, mfp_post)
    sep_gap = shock_metrics.convolved_peak_gap(
        float(sep["max_dTr_in_window"]),
        float(sep["ref_dTr_max_in_window"]),
    )

    ref_Te_direct = interpolate_reference(ref, "Te_eV", x_aligned)
    ref_Ti_direct = interpolate_reference(ref, "Ti_eV", x_aligned)
    legacy_l2_Te = rel_l2_window(Te, ref_Te_direct, x_aligned, mfp_post)
    legacy_l2_Ti = rel_l2_window(Ti, ref_Ti_direct, x_aligned, mfp_post)
    radial = bundle["radial_invariance"]
    radial_gate = max(float(radial[name]) for name in RADIAL_INVARIANCE_GATE_FIELDS)

    return {
        **shock,
        "shock_front_reference_frame_cm": 0.0,
        "shock_frame_alignment": "run_z_minus_max_abs_grad_Te_position; reference_x0_is_shock",
        "mfp_post_cm": mfp_post,
        "window_half_cm": float(l2_Te["window_half_cm"]),
        "window_min_cm": float(l2_Te["window_min_cm"]),
        "window_max_cm": float(l2_Te["window_max_cm"]),
        "n_cells_in_window": int(l2_Te["n_cells_in_window"]),
        "measured_hydro_kernel_half_width_cm": half_width,
        "measured_hydro_kernel_cell_count": int(kernel["n_cells"]),
        "l2_Te_rel": float(l2_Te["L2_rel_rms"]),
        "l2_Te_norm_by_ref_peak": float(l2_Te["L2_norm"]),
        "l2_Te_abs_eV": float(l2_Te["L2_abs"]),
        "l2_Ti_rel": float(l2_Ti["L2_rel_rms"]),
        "l2_Ti_norm_by_ref_peak": float(l2_Ti["L2_norm"]),
        "l2_Ti_abs_eV": float(l2_Ti["L2_abs"]),
        "legacy_unconvolved_l2_Te_rel": float(legacy_l2_Te["L2_rel_rms"]),
        "legacy_unconvolved_l2_Ti_rel": float(legacy_l2_Ti["L2_rel_rms"]),
        "delta_ei_code_peak_in_window": float(sep["max_dTr_in_window"]),
        "delta_ei_reference_convolved_peak_in_window": float(sep["ref_dTr_max_in_window"]),
        "delta_ei_peak_gap_rel": float(sep_gap),
        "delta_ei_l2_rel_diagnostic": float(sep["L2_rel_rms"]),
        "max_P_rad_over_P_mat": code_low_prad_max(snapshot),
        "radial_invariance": radial,
        "radial_invariance_max_rel": radial_gate,
        "radial_invariance_gate_fields": list(RADIAL_INVARIANCE_GATE_FIELDS),
        "_richardson_profile": {
            "x_aligned_cm": x_aligned,
            "Te_eV": Te,
            "mfp_post_cm": mfp_post,
        },
    }


def analyze_leg(
    args: argparse.Namespace,
    spec: LegSpec,
    ref: dict[str, Any],
    row: dict[str, Any],
) -> dict[str, Any]:
    out = dict(row)
    out["case_name"] = deck_case_name(spec, ref)
    out["boundary_mode"] = spec.boundary_mode
    out["reference_path"] = str(spec.reference)
    out["reference_parameters"] = ref.get("parameters", {})
    reference_identity = reference_identity_2t(ref)
    out["reference_identity_2t"] = reference_identity
    out["reference_low_prad_max"] = float(
        ref.get("diagnostics", {}).get("sv7_low_prad", {}).get("max_P_rad_over_P_mat", math.inf)
    )

    snapshot = None
    if row.get("exit_code") == 0 and not row.get("timed_out"):
        try:
            snapshots = read_leg_snapshots(args, spec, ref)
            snapshot = snapshots[-1]
            out["snapshot_count"] = len(snapshots)
            out["final_snapshot_path"] = str(snapshot.path)
            out["final_t_s"] = float(snapshot.t)
            out["final_step"] = int(snapshot.step)
            out.update(analyze_profile_leg(snapshot, ref))
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            out["analysis_error"] = str(exc)
            out["run_status"] = "analysis_failure"

    n_cells = spec.nr * spec.nz
    newton = parse_newton_multisolve(Path(args.out_root) / spec.run_id / "run.log", n_cells, snapshot)
    no_escape = no_escape_for_leg(args, spec, ref, newton)
    out["newton_diagnostic"] = newton
    out["no_escape_valves"] = no_escape
    completed = row.get("exit_code") == 0 and not row.get("timed_out") and row.get("termination_reason") == "t_end_reached"

    if spec.rung in ("SMOKE", "RICHARDSON"):
        gates = {"leg_completed": completed}
    else:
        max_prad = max(float(out.get("max_P_rad_over_P_mat", math.inf)), float(out["reference_low_prad_max"]))
        gates = {
            "leg_completed": completed,
            "gate_l2_Te": float(out.get("l2_Te_rel", math.inf)) <= L2_REL_GATE,
            "gate_l2_Ti": float(out.get("l2_Ti_rel", math.inf)) <= L2_REL_GATE,
            "gate_sep_amplitude": float(out.get("delta_ei_peak_gap_rel", math.inf)) <= SEP_REL_GATE,
            "gate_low_prad_only": max_prad <= LOW_PRAD_GATE,
            "gate_reference_conservation": bool(reference_identity["reference_conservation_passed"]),
            "gate_reference_identity_branch_wise": bool(reference_identity["reference_identity_passed"]),
            "radial_invariance_pass": float(out.get("radial_invariance_max_rel", math.inf)) <= RADIAL_INVARIANCE_GATE,
            "gate9_no_escape_valves": bool(no_escape.get("all_zero")),
            "gate10_newton_multisolve": bool(newton.get("newton_multisolve_pass")),
        }
    out["gates"] = gates
    out["all_checks_passed"] = all(gates.values())
    return out


def max_float(rows: list[dict[str, Any]], key: str) -> float:
    return max([float(row.get(key, math.inf)) for row in rows] or [math.inf])


def metric_by_regime(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        str(row["regime"]): float(row.get(key, math.inf))
        for row in rows
        if row.get("rung") == "PRODUCTION"
    }


def te_profile_distance(production: dict[str, Any] | None, other: dict[str, Any] | None) -> dict[str, Any]:
    base = {
        "l2_Te_distance_abs_eV": math.inf,
        "l2_Te_distance_rel_to_production_rms": math.inf,
        "n_cells_in_window": 0,
    }
    if production is None or other is None:
        return base
    prod_profile = production.get("_richardson_profile", {})
    other_profile = other.get("_richardson_profile", {})
    prod_x = np.asarray(prod_profile.get("x_aligned_cm", []), dtype=float)
    prod_te = np.asarray(prod_profile.get("Te_eV", []), dtype=float)
    mfp_post = float(prod_profile.get("mfp_post_cm", math.inf))
    if prod_x.size != prod_te.size:
        return base
    window_half = SHOCK_WINDOW_MFP_COUNT * mfp_post
    mask = np.isfinite(prod_x) & np.isfinite(prod_te) & (np.abs(prod_x) <= window_half)
    if np.count_nonzero(mask) < 3:
        return base
    other_te = i1_base.interp_sorted(
        other_profile.get("x_aligned_cm", []),
        other_profile.get("Te_eV", []),
        prod_x[mask],
    )
    finite = np.isfinite(other_te)
    if np.count_nonzero(finite) < 3:
        return base
    prod_window = prod_te[mask][finite]
    diff = prod_window - other_te[finite]
    l2_abs = float(np.sqrt(np.mean(diff * diff)))
    prod_rms = float(np.sqrt(np.mean(prod_window * prod_window)))
    return {
        "l2_Te_distance_abs_eV": l2_abs,
        "l2_Te_distance_rel_to_production_rms": l2_abs / max(prod_rms, i1_base.E_FLOOR),
        "n_cells_in_window": int(np.count_nonzero(finite)),
    }


def richardson_monotone_by_regime(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_rung = {(str(row.get("regime")), str(row.get("rung"))): row for row in rows}
    regimes = [regime for regime in REGIME_ORDER if any(row.get("regime") == regime for row in rows)]
    out: dict[str, dict[str, Any]] = {}
    for regime in regimes:
        production = by_rung.get((regime, "PRODUCTION"))
        smoke = by_rung.get((regime, "SMOKE"))
        richardson = by_rung.get((regime, "RICHARDSON"))
        prod_smoke = te_profile_distance(production, smoke)
        prod_richardson = te_profile_distance(production, richardson)
        prod_smoke_l2 = float(prod_smoke["l2_Te_distance_abs_eV"])
        prod_richardson_l2 = float(prod_richardson["l2_Te_distance_abs_eV"])
        gate = math.isfinite(prod_smoke_l2) and math.isfinite(prod_richardson_l2) and prod_richardson_l2 < prod_smoke_l2
        out[regime] = {
            "provisional_monotone_only": True,
            "metric": "shock_windowed_Te_L2_abs_eV_on_production_grid",
            "production_vs_smoke_l2_Te_distance_abs_eV": prod_smoke_l2,
            "production_vs_richardson_l2_Te_distance_abs_eV": prod_richardson_l2,
            "production_vs_smoke_l2_Te_distance_rel_to_production_rms": float(
                prod_smoke["l2_Te_distance_rel_to_production_rms"]
            ),
            "production_vs_richardson_l2_Te_distance_rel_to_production_rms": float(
                prod_richardson["l2_Te_distance_rel_to_production_rms"]
            ),
            "production_vs_smoke_n_cells_in_window": int(prod_smoke["n_cells_in_window"]),
            "production_vs_richardson_n_cells_in_window": int(prod_richardson["n_cells_in_window"]),
            "gate_richardson_monotone": gate,
        }
    return out


def public_leg_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def aggregate(rows: list[dict[str, Any]], expected_regime_count: int) -> dict[str, Any]:
    production = [row for row in rows if row.get("rung") == "PRODUCTION"]
    smoke = [row for row in rows if row.get("rung") == "SMOKE"]
    richardson = [row for row in rows if row.get("rung") == "RICHARDSON"]
    richardson_monotone = richardson_monotone_by_regime(rows)
    gate_richardson_monotone = len(richardson_monotone) == expected_regime_count and all(
        bool(row.get("gate_richardson_monotone")) for row in richardson_monotone.values()
    )
    gate_names = sorted({name for row in production for name in row.get("gates", {})})
    production_gate_summary = {
        name: bool(production) and all(bool(row.get("gates", {}).get(name)) for row in production)
        for name in gate_names
    }
    gates = {
        "smoke_legs_completed": len(smoke) == expected_regime_count
        and all(bool(row.get("gates", {}).get("leg_completed")) for row in smoke),
        "production_legs_completed": len(production) == expected_regime_count
        and all(bool(row.get("gates", {}).get("leg_completed")) for row in production),
        "gate_richardson_monotone": gate_richardson_monotone,
        **production_gate_summary,
    }
    out = {
        "leg_count": len(rows),
        "smoke_leg_count": len(smoke),
        "production_leg_count": len(production),
        "richardson_leg_count": len(richardson),
        "l2_rel_gate": L2_REL_GATE,
        "sep_rel_gate": SEP_REL_GATE,
        "low_prad_gate": LOW_PRAD_GATE,
        "reference_conservation_gate": CONSERVATION_GATE,
        "radial_invariance_gate": RADIAL_INVARIANCE_GATE,
        "shock_window_mfp_count": SHOCK_WINDOW_MFP_COUNT,
        "production_cg_max_iter": PRODUCTION_CG_MAX_ITER,
        "production_cg_inner_tol": PRODUCTION_CG_INNER_TOL,
        "max_l2_Te_rel": max_float(production, "l2_Te_rel"),
        "max_l2_Ti_rel": max_float(production, "l2_Ti_rel"),
        "max_delta_ei_peak_gap_rel": max_float(production, "delta_ei_peak_gap_rel"),
        "max_radial_invariance": max_float(production, "radial_invariance_max_rel"),
        "max_P_rad_over_P_mat": max_float(production, "max_P_rad_over_P_mat"),
        "max_reference_low_prad": max_float(production, "reference_low_prad_max"),
        "l2_Te_rel_by_regime": metric_by_regime(production, "l2_Te_rel"),
        "l2_Ti_rel_by_regime": metric_by_regime(production, "l2_Ti_rel"),
        "delta_ei_peak_gap_rel_by_regime": metric_by_regime(production, "delta_ei_peak_gap_rel"),
        "radial_invariance_by_regime": metric_by_regime(production, "radial_invariance_max_rel"),
        "richardson_monotone_by_regime": richardson_monotone,
        "gates": gates,
        "all_checks_passed": all(gates.values()),
    }
    return out


def parse_regimes(value: str) -> list[str]:
    regimes = [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    invalid = [item for item in regimes if item not in REGIME_ORDER]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid regime(s): {', '.join(invalid)}")
    if not regimes:
        raise argparse.ArgumentTypeError("at least one regime is required")
    return regimes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/i5_2t_radshock_2d_rz_slab.py")
    parser.add_argument(
        "--reference-root",
        default="tests/verification/data/fld_ced_2t_reference",
    )
    parser.add_argument("--regimes", type=parse_regimes, default=list(REGIME_ORDER))
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--init-mode", choices=("two_state", "reference_table"), default="reference_table"
    )
    parser.add_argument(
        "--boundary-mode",
        default="state_supply",
        choices=("state_supply", "reflecting", "finite_shock_tube"),
        help="z-face boundary mode exported to the deck (finite_shock_tube aliases to reflecting)",
    )
    parser.add_argument("--production-audit-tier", choices=("A", "B"), default="A")
    parser.add_argument("--out-root", default="./build/i5_2t_radshock_2d_rz_slab")
    parser.add_argument("--summary-json", default="./build/i5_2t_radshock_2d_rz_slab/summary.json")
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="quiet")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed leg output; also enabled by TENRYU_I5_BATTERY_RESUME",
    )
    return parser


def leg_specs(args: argparse.Namespace, boundary_mode: str) -> list[LegSpec]:
    reference_root = Path(args.reference_root)
    specs: list[LegSpec] = []
    for regime in args.regimes:
        reference = reference_path_for_regime(reference_root, regime)
        for rung in ("SMOKE", "PRODUCTION", "RICHARDSON"):
            specs.append(
                LegSpec(
                    regime=regime,
                    rung=rung,
                    nr=args.nr,
                    nz=NZ_BY_REGIME_RUNG[regime][rung],
                    seed=args.seed,
                    init_mode=args.init_mode,
                    boundary_mode=boundary_mode,
                    reference=reference,
                )
            )
    return specs


def main(argv: list[str] | None = None) -> int:
    i1_rz.install_base_hooks()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.nr <= 0:
        parser.error("--nr must be positive")
    boundary_mode = (
        "reflecting"
        if args.boundary_mode.lower() == "finite_shock_tube"
        else args.boundary_mode.lower()
    )

    rows: list[dict[str, Any]] = []
    refs: dict[str, dict[str, Any]] = {}
    for spec in leg_specs(args, boundary_mode):
        ref = refs.setdefault(spec.regime, read_json(spec.reference))
        row = run_deck(args, spec, ref)
        rows.append(analyze_leg(args, spec, ref, row))

    agg = aggregate(rows, len(args.regimes))
    public_rows = [public_leg_row(row) for row in rows]
    payload = {
        "schema_version": SUMMARY_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "ladder_row": LADDER_ROW,
        "dimension": DIMENSION,
        "deck_path": args.deck,
        "reference_root": args.reference_root,
        "regimes": args.regimes,
        "init_mode": args.init_mode,
        "boundary_mode": boundary_mode,
        "production_audit_tier": args.production_audit_tier,
        "resume_enabled": bool(args.resume) or truthy_env("TENRYU_I5_BATTERY_RESUME"),
        "strict_scope": "I5 W4 APRIME_SCALED production profile gates plus smoke completion and Richardson monotone convergence",
        "legs": public_rows,
        "matrix": public_rows,
        "aggregate": agg,
    }
    write_json(Path(args.summary_json), payload)
    print(
        json.dumps(
            {
                "ladder_row": LADDER_ROW,
                "production_audit_tier": args.production_audit_tier,
                "aggregate": agg,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if bool(agg["all_checks_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
