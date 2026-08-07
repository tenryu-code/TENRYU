#!/usr/bin/env python3
"""Run and analyze the I3 grey S_N 2D RZ radiative-shock gate."""

from __future__ import annotations

import argparse
import copy
from fnmatch import fnmatch
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import radiative_shock_metrics as shock_metrics
import run_i1_2d_rz_fld_ced as i1
import run_i1_le08 as base
import rz_profile_average as rz_average


_ORIGINAL_BASE_RUN_ONE = base.run_one
STAGE_ID = "I3"
LADDER_ROW = "i3_sn_radshock_2d_rz_slab"
ENV_PREFIX = "TENRYU_I3_SN_RS"
REFERENCE_SELF_IDENTITY_GATE = 1.0e-8
ANGULAR_MOMENT_SELF_IDENTITY_GATE = 1.0e-10
G_B_DTR_L2_GATE = 0.10
G_B_FZ_L2_GATE = 0.08
G_B_CHI_LINF_GATE = 0.05
G_B_RICHARDSON_LOW = 0.6215
G_B_RICHARDSON_HIGH = 0.8408
RADIAL_INVARIANCE_GATE = 1.0e-5
G_D_K2_E_L2_GATE = 0.03
G_D_K2_TRAD_L2_GATE = 0.02
G_D_K2_CHI_INTERIOR_GATE = 0.02
G_D_K2_E_FLOOR = 1.20e-3
G_D_K2_TRAD_FLOOR = 7.28e-4
G_D_SECOND_DIFFERENCE_GATE = 0.05
G_D_INTERIOR_COLLAR_K2_CM = 0.05
G_E_SN_OUTER_LIMIT = 50
G_E_SN_INNER_LIMIT = 100
SMOKE_MAX_STEPS = 200
SMOKE_GRID = (16, 256)
SMOKE_N_ANGLES = 8
SMOKE_KAPPA_SCALE = 1
FIRST_STEP_AUDIT_T_END_S = i1.FIRST_STEP_AUDIT_T_END_S
SHOCK_WINDOW_MFP_COUNT = i1.SHOCK_WINDOW_MFP_COUNT
SHOCK_FRONT_BOUNDARY_EXCLUSION_CELLS = i1.SHOCK_FRONT_BOUNDARY_EXCLUSION_CELLS
SHOCK_FRONT_COLOCATION_CELLS = i1.SHOCK_FRONT_COLOCATION_CELLS
A_EV = base.A_EV
E_FLOOR = base.E_FLOOR

# Addendum W0 §4/§5 frozen constants echoed into every summary.
FROZEN_PREDICTION_REFS = {
    "source": "docs/design/i3_sn_radshock_spec_addendum_w0.md",
    "K1": 20,
    "K2": 200,
    "dt_caps_s": {"20": 7.2e-10, "200": 7.2e-11},
    "thick_limit_floors": {
        "K2_E_windowed_L2": G_D_K2_E_FLOOR,
        "K2_T_rad_windowed_L2": G_D_K2_TRAD_FLOOR,
    },
    "richardson_D_inf_window": [G_B_RICHARDSON_LOW, G_B_RICHARDSON_HIGH],
    "K2_interior_collar_cm": G_D_INTERIOR_COLLAR_K2_CM,
}

_SN_TIMING_RE = re.compile(
    r"\[sn_2d_rz_timing\].*?"
    r"outer_iters=(?P<outer>[0-9]+).*?"
    r"inner_iters=(?P<inner>[0-9]+).*?"
    r"converged=(?P<converged>[0-9]+).*?"
    r"outer_residual=(?P<outer_resid>[-+0-9.eE]+).*?"
    r"inner_residual=(?P<inner_resid>[-+0-9.eE]+)"
)

_CASE_CONTEXT: dict[str, Any] = {
    "n_angles": 16,
    "kappa_scale": 1,
    "radiation_mode": "sn_transport",
    "reference": "tests/verification/data/sn_radshock_reference/sn_radshock_M2_N16_K1.json",
}


@dataclass(frozen=True)
class RZSnapshot:
    path: Path
    t: float
    step: int
    rho: Any
    Te: Any
    Ti: Any
    ee: Any
    ei: Any
    Pe: Any
    Pi: Any
    vol: Any
    rad_E: Any
    rad_F: Any | None
    sn_F_z: Any
    sn_P_zz: Any
    sn_chi_z: Any
    sn_rad_E_pre_newton: Any
    sn_stream_theta: Any
    sn_ap_alpha: Any
    sn_radial_fixup_count: Any
    sn_radial_fixup_artificial_abs: Any
    sn_angular_fixup_count: Any
    sn_angular_fixup_artificial_abs: Any
    sn_ap_alpha_max: float
    sn_ap_alpha_active_faces: float
    dt: float
    r_nodes: Any
    v_r_nodes: Any
    z_nodes: Any
    v_z_nodes: Any
    dimension: str


def parse_grid_list(value: str) -> list[tuple[int, int]]:
    return i1.parse_grid_list(value)


def parse_int_csv(value: str, allowed: set[int], label: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        val = int(item)
        if val not in allowed:
            raise argparse.ArgumentTypeError(f"{label} must be in {sorted(allowed)}, got {val}")
        out.append(val)
    if not out:
        raise argparse.ArgumentTypeError(f"{label} list must not be empty")
    return out


def safe_float_token(value: float) -> str:
    return base.safe_float_token(value)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sn_reference_path(mach: float, n_angles: int, kappa_scale: int) -> Path:
    return (
        repo_root()
        / "tests"
        / "verification"
        / "data"
        / "sn_radshock_reference"
        / f"sn_radshock_M{safe_float_token(mach)}_N{n_angles}_K{kappa_scale}.json"
    )


def fld_reference_path(mach: float, kappa_scale: int) -> Path:
    if kappa_scale == 1:
        return (
            repo_root()
            / "tests"
            / "verification"
            / "data"
            / "fld_const_eddington_reference"
            / f"fld_ced_M{safe_float_token(mach)}.json"
        )
    return (
        repo_root()
        / "tests"
        / "verification"
        / "data"
        / "sn_radshock_reference"
        / f"fld_ced_M{safe_float_token(mach)}_K{kappa_scale}.json"
    )


def reference_path(mach: float, n_angles: int, kappa_scale: int, radiation_mode: str) -> Path:
    if radiation_mode == "sn_transport":
        return sn_reference_path(mach, n_angles, kappa_scale)
    return fld_reference_path(mach, kappa_scale)


def read_json(path: Path) -> dict[str, Any]:
    return base.read_json(path)


def _max_numeric_leaf(value: Any) -> float:
    if isinstance(value, dict):
        if not value:
            return math.inf
        return max(_max_numeric_leaf(child) for child in value.values())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected numeric diagnostic leaf, got {type(value).__name__}")
    return float(value)


def reference_for_i1(ref: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(ref)
    params = out.setdefault("parameters", {})
    if "kappa_R_cm2_per_g" not in params and "kappa_a_cm2_per_g" in params:
        params["kappa_R_cm2_per_g"] = params["kappa_a_cm2_per_g"]
    diagnostics = out.setdefault("diagnostics", {})
    if "conservation_residual_rel_max" not in diagnostics:
        diagnostics["conservation_residual_rel_max"] = _max_numeric_leaf(
            diagnostics.get("rh_residual_rel", 0.0)
        )
    if "max_P_rad_over_P_mat" not in diagnostics:
        ratios = []
        for state_name in ("upstream", "downstream"):
            state = out.get("states", {}).get(state_name, {})
            try:
                ratios.append(abs(float(state["P_rad"])) / max(abs(float(state["P_mat"])), E_FLOOR))
            except (KeyError, TypeError, ValueError):
                pass
        if not ratios and "table" in out:
            p_rad = np.asarray(out["table"].get("P_rad_dyne_per_cm2", []), dtype=float)
            p_mat = np.asarray(out["table"].get("P_mat_dyne_per_cm2", []), dtype=float)
            if p_rad.size and p_rad.shape == p_mat.shape:
                ratios.append(float(np.nanmax(np.abs(p_rad) / np.maximum(np.abs(p_mat), E_FLOOR))))
        diagnostics["max_P_rad_over_P_mat"] = max(ratios) if ratios else 0.0
    diagnostics.setdefault("max_abs_grad_P_rad_over_abs_grad_P_mat", 0.0)
    return out


def case_name(mach: float, nr: int, nz: int, seed: int, init_mode: str, dimension: str) -> str:
    del dimension
    mode = str(_CASE_CONTEXT["radiation_mode"])
    mode_token = "sn" if mode == "sn_transport" else "fld"
    return (
        f"{LADDER_ROW}_M{safe_float_token(mach)}"
        f"_N{int(_CASE_CONTEXT['n_angles'])}"
        f"_K{int(_CASE_CONTEXT['kappa_scale'])}"
        f"_{mode_token}_nr{nr}_nz{nz}_seed{seed}_{init_mode}"
    )


def active_env(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    outdir: Path,
    *,
    t_end_s: float | None = None,
    max_steps: int | None = None,
    dt_initial_s: float | None = None,
    dt_max_s: float | None = None,
) -> dict[str, str]:
    del dimension
    verbose_diag = os.environ.get(f"{ENV_PREFIX}_VERBOSE_DIAG", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    env = {
        f"{ENV_PREFIX}_MACH": str(args.mach),
        f"{ENV_PREFIX}_NR": str(nr),
        f"{ENV_PREFIX}_NZ": str(nz),
        f"{ENV_PREFIX}_SEED": str(args.seed),
        f"{ENV_PREFIX}_OUTDIR": str(outdir),
        f"{ENV_PREFIX}_REFERENCE": str(_CASE_CONTEXT["reference"]),
        f"{ENV_PREFIX}_INIT_MODE": args.init_mode,
        f"{ENV_PREFIX}_BOUNDARY_MODE": args.boundary_mode,
        f"{ENV_PREFIX}_PRODUCTION_AUDIT_TIER": args.production_audit_tier,
        f"{ENV_PREFIX}_N_ANGLES": str(int(_CASE_CONTEXT["n_angles"])),
        f"{ENV_PREFIX}_KAPPA_SCALE": str(int(_CASE_CONTEXT["kappa_scale"])),
        f"{ENV_PREFIX}_RADIATION_MODE": str(_CASE_CONTEXT["radiation_mode"]),
        f"{ENV_PREFIX}_MAX_STEPS": str(max_steps if max_steps is not None else args.max_steps),
        f"{ENV_PREFIX}_HISTORY_EVERY_STEPS": os.environ.get(
            f"{ENV_PREFIX}_HISTORY_EVERY_STEPS",
            "1" if verbose_diag else "0",
        ),
        f"{ENV_PREFIX}_VERBOSITY": "verbose" if verbose_diag else args.verbosity,
    }
    effective_t_end_s = t_end_s if t_end_s is not None else args.t_end_s
    effective_dt_initial_s = dt_initial_s if dt_initial_s is not None else args.dt_initial_s
    effective_dt_max_s = dt_max_s if dt_max_s is not None else args.dt_max_s
    if effective_t_end_s is not None:
        env[f"{ENV_PREFIX}_T_END_S"] = str(effective_t_end_s)
    if effective_dt_initial_s is not None:
        env[f"{ENV_PREFIX}_DT_INITIAL_S"] = str(effective_dt_initial_s)
    if effective_dt_max_s is not None:
        env[f"{ENV_PREFIX}_DT_MAX_S"] = str(effective_dt_max_s)
    return env


def _reshape_rad_or_zero(handle: Any, path: str, nr: int, nz: int, dimension: str) -> Any:
    if path in handle:
        return base.reshape_rad(handle[path][()], nr, nz, dimension)
    return np.zeros((nr, nz), dtype=float)


def _reshape_cell_or_zero(handle: Any, path: str, nr: int, nz: int, dimension: str) -> Any:
    if path in handle:
        return base.reshape_cells(handle[path][()], nr, nz, path, dimension)
    return np.zeros((nr, nz), dtype=float)


def read_snapshot(path: Path, nr: int, nz: int, dimension: str) -> RZSnapshot:
    if dimension != "2D_RZ":
        raise ValueError("run_i3_sn_radshock.py supports only dimension=2D_RZ")
    import h5py

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = base.scalar_dataset(handle, "time_state/t", t_value)
        r_nodes = base.reshape_nodes(handle["mesh/x_r"][()], nr, nz, "mesh/x_r", dimension)
        z_nodes = base.reshape_nodes(handle["mesh/x_z"][()], nr, nz, "mesh/x_z", dimension)
        v_r_nodes = base.reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r", dimension)
        v_z_nodes = (
            base.reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z", dimension)
            if "mesh/v_z" in handle
            else np.zeros_like(v_r_nodes)
        )
        rad_f = (
            base.reshape_rad(handle["radiation/diag_F_first_moment"][()], nr, nz, dimension)
            if "radiation/diag_F_first_moment" in handle
            else None
        )
        return RZSnapshot(
            path=path,
            t=t_value,
            step=int(round(base.scalar_dataset(handle, "time_state/step", 0.0))),
            rho=base.reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho", dimension),
            Te=base.reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te", dimension),
            Ti=base.reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti", dimension),
            ee=base.reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee", dimension),
            ei=base.reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei", dimension),
            Pe=base.reshape_cells(handle["hydro/Pe"][()], nr, nz, "hydro/Pe", dimension),
            Pi=base.reshape_cells(handle["hydro/Pi"][()], nr, nz, "hydro/Pi", dimension),
            vol=base.reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol", dimension),
            rad_E=base.reshape_rad(handle["radiation/energy_density"][()], nr, nz, dimension),
            rad_F=rad_f,
            sn_F_z=_reshape_rad_or_zero(handle, "radiation/sn_F_z", nr, nz, dimension),
            sn_P_zz=_reshape_rad_or_zero(handle, "radiation/sn_P_zz", nr, nz, dimension),
            sn_chi_z=_reshape_rad_or_zero(handle, "radiation/sn_chi_z", nr, nz, dimension),
            sn_rad_E_pre_newton=_reshape_rad_or_zero(handle, "radiation/sn_rad_E_pre_newton", nr, nz, dimension),
            sn_stream_theta=_reshape_rad_or_zero(handle, "radiation/diag_stream_theta", nr, nz, dimension),
            sn_ap_alpha=_reshape_cell_or_zero(handle, "radiation/sn_ap_alpha", nr, nz, dimension),
            sn_radial_fixup_count=_reshape_rad_or_zero(handle, "radiation/sn_radial_fixup_count", nr, nz, dimension),
            sn_radial_fixup_artificial_abs=_reshape_rad_or_zero(
                handle, "radiation/sn_radial_fixup_artificial_abs", nr, nz, dimension
            ),
            sn_angular_fixup_count=_reshape_rad_or_zero(handle, "radiation/sn_angular_fixup_count", nr, nz, dimension),
            sn_angular_fixup_artificial_abs=_reshape_rad_or_zero(
                handle, "radiation/sn_angular_fixup_artificial_abs", nr, nz, dimension
            ),
            sn_ap_alpha_max=base.scalar_dataset(handle, "time_state/sn_ap_alpha_max", 0.0),
            sn_ap_alpha_active_faces=base.scalar_dataset(handle, "time_state/sn_ap_alpha_active_faces", 0.0),
            dt=base.scalar_dataset(handle, "time_state/dt", 0.0),
            r_nodes=r_nodes,
            v_r_nodes=v_r_nodes,
            z_nodes=z_nodes,
            v_z_nodes=v_z_nodes,
            dimension=dimension,
        )


def cell_center_from_nodes(nodes: Any) -> np.ndarray:
    return i1.cell_center_from_nodes(nodes)


def profile_velocity(snapshot: RZSnapshot) -> np.ndarray:
    return base.profile_average(snapshot, cell_center_from_nodes(snapshot.v_z_nodes))


def radial_reduce(snapshot: RZSnapshot, field: Any) -> tuple[np.ndarray, float]:
    arr = np.asarray(field, dtype=float)
    vol = rz_average.cell_volumes_from_hdf5(snapshot)
    profile = rz_average.radial_average(arr, vol, axis_r=0)
    eps = rz_average.radial_invariance_metric(arr, profile, axis_r=0)
    return np.asarray(profile, dtype=float), float(eps)


def profile_bundle(snapshot: RZSnapshot) -> dict[str, Any]:
    fields = {
        "rho": snapshot.rho,
        "Te": snapshot.Te,
        "Ti": snapshot.Ti,
        "E_rad": snapshot.rad_E,
        "Pe": snapshot.Pe,
        "Pi": snapshot.Pi,
        "u_r": cell_center_from_nodes(snapshot.v_r_nodes),
        "u_z": cell_center_from_nodes(snapshot.v_z_nodes),
        "F_z": snapshot.sn_F_z,
        "P_zz": snapshot.sn_P_zz,
        "chi_z": snapshot.sn_chi_z,
    }
    profiles: dict[str, np.ndarray] = {}
    radial_invariance: dict[str, float] = {}
    for name, field in fields.items():
        profiles[name], radial_invariance[name] = radial_reduce(snapshot, field)
    profiles["P_mat"] = profiles["Pe"] + profiles["Pi"]
    profiles["P_rad"] = profiles["E_rad"] / 3.0
    profiles["T_rad"] = np.power(np.maximum(profiles["E_rad"] / A_EV, 0.0), 0.25)
    profiles["dTr"] = profiles["T_rad"] / np.maximum(profiles["Te"], E_FLOOR) - 1.0
    return {
        "z_cm": np.asarray(base.profile_centers(snapshot), dtype=float),
        "profiles": profiles,
        "radial_invariance": radial_invariance,
    }


def install_base_hooks() -> None:
    base.LADDER_ROW = LADDER_ROW
    base.case_name = case_name
    base.active_env = active_env
    base.read_snapshot = read_snapshot
    base.profile_velocity = profile_velocity
    i1.LADDER_ROW = LADDER_ROW
    i1.case_name = case_name
    i1.active_env = active_env
    i1.read_snapshot = read_snapshot
    i1.profile_velocity = profile_velocity
    i1.profile_bundle = profile_bundle
    i1.radial_reduce = radial_reduce


def env_resume_enabled() -> bool:
    return os.environ.get("TENRYU_I3_BATTERY_RESUME", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def legs_filter_active(args: argparse.Namespace) -> bool:
    return getattr(args, "legs", None) is not None


def legs_filter_patterns(args: argparse.Namespace) -> list[str]:
    value = getattr(args, "legs", None)
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def leg_selected(args: argparse.Namespace, label: str) -> bool:
    if not legs_filter_active(args):
        return True
    return any(fnmatch(label, pattern) for pattern in legs_filter_patterns(args))


def skipped_leg_row(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    *,
    run_label: str,
    base_run_id: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_name": base_run_id,
        "run_label": run_label,
        "dimension": dimension,
        "nr": nr,
        "nz": nz if dimension == "2D_RZ" else None,
        "exit_code": None,
        "timed_out": False,
        "wall_time_s": 0.0,
        "output_bytes": 0,
        "termination_reason": None,
        "analysis_error": None,
        "run_status": "skipped_by_legs_filter",
        "legs_filter": args.legs,
        "resumed": False,
    }


def analyze_existing_base_run(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    *,
    run_label: str,
    base_run_id: str,
    run_id: str,
    outdir: Path,
    log_path: Path,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "run_id": run_id,
        "case_name": base_run_id,
        "run_label": run_label,
        "dimension": dimension,
        "nr": nr,
        "nz": nz if dimension == "2D_RZ" else None,
        "exit_code": 0,
        "timed_out": False,
        "wall_time_s": 0.0,
        "output_bytes": base.output_bytes(outdir),
        "termination_reason": base.run_info_termination(outdir),
        "analysis_error": None,
        "resumed": True,
    }
    try:
        results_dir = base.discover_results_dir(outdir)
        actual_outdir = results_dir.parent
        metrics["output_bytes"] = base.output_bytes(actual_outdir)
        snapshots = base.read_snapshots(results_dir, base_run_id, nr, nz, dimension)
        final = snapshots[-1]
        shock_velocity, shock_positions = base.fit_shock_velocity(snapshots)
        final_profile = base.profile_metrics(final, ref)
        window_profile = base.profile_window_metrics(snapshots, ref)
        steady_start_index = int(window_profile.get("steady_start_index", len(snapshots) - 1))
        steady_start_index = max(0, min(steady_start_index, len(snapshots) - 1))
        closure_rows = [
            dict({"snapshot_index": index}, **base.closure_defect_summary(snapshot, ref, shock_velocity))
            for index, snapshot in enumerate(snapshots)
        ]
        rh_rows = [
            dict({"snapshot_index": index}, **base.rh_residual_summary(snapshot, shock_velocity))
            for index, snapshot in enumerate(snapshots)
        ]
        metrics.update(final_profile)
        metrics.update(
            {
                "final_postshock_Te_l2": final_profile.get("postshock_Te_l2"),
                "final_precursor_Trad_l2": final_profile.get("precursor_Trad_l2"),
                "final_shock_position_cm": final_profile.get("shock_position_cm"),
                "final_profile_l2_three_coords": final_profile.get("profile_l2_three_coords"),
            }
        )
        metrics.update(window_profile)
        metrics["postshock_Te_l2"] = window_profile.get("steady_window_postshock_Te_l2", math.inf)
        metrics["precursor_Trad_l2"] = window_profile.get("steady_window_precursor_Trad_l2", math.inf)
        metrics["closure_defect_snapshots"] = closure_rows
        metrics["closure_defects"] = base.aggregate_closure_defects(closure_rows[steady_start_index:])
        metrics["rh_residual_snapshots"] = rh_rows
        metrics["rh_residuals"] = base.aggregate_rh_residuals(rh_rows[steady_start_index:])
        if getattr(args, "dump_profiles", False):
            metrics["profile_dumps"] = base.dump_profiles_for_run(
                snapshots,
                ref,
                actual_outdir,
                run_id,
                shock_velocity,
            )
        metrics.update(
            {
                "run_status": "completed"
                if metrics.get("termination_reason") == "t_end_reached"
                else "not_completed",
                "final_t_s": final.t,
                "final_step": final.step,
                "snapshot_count": len(snapshots),
                "shock_velocity_cm_per_s": shock_velocity,
                "shock_positions_cm": shock_positions,
                "dz_nominal_cm": base.nominal_cell_width(final, ref, nr if dimension == "1D_SPH" else nz),
            }
        )
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["run_status"] = "analysis_failure"
        metrics["analysis_error"] = str(exc)
        metrics["log_tail"] = base.read_text(log_path)[-4000:]
    return metrics


def run_one_with_resume(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    *,
    run_label: str = "production",
    t_end_s: float | None = None,
    max_steps: int | None = None,
    dt_initial_s: float | None = None,
    dt_max_s: float | None = None,
) -> dict[str, Any]:
    base_run_id = base.case_name(args.mach, nr, nz, args.seed, args.init_mode, dimension)
    run_id = base_run_id if run_label == "production" else f"{base_run_id}_{run_label}"
    outdir = Path(args.out_root) / run_id
    log_path = outdir / "run.log"
    resumed = False
    if getattr(args, "resume", False) and log_path.exists():
        try:
            prior = i1.run_info(outdir)
        except Exception:
            prior = {}
        if str(prior.get("termination_reason", "")) == "t_end_reached":
            resumed = True
            print(f"[run_i3_sn_radshock] resume: skipping {run_id}")
    if resumed:
        return analyze_existing_base_run(
            args,
            nr,
            nz,
            dimension,
            ref,
            run_label=run_label,
            base_run_id=base_run_id,
            run_id=run_id,
            outdir=outdir,
            log_path=log_path,
        )
    if not leg_selected(args, run_id):
        print(f"[run_i3_sn_radshock] legs: skipping {run_id} (filter={args.legs})")
        return skipped_leg_row(
            args,
            nr,
            nz,
            dimension,
            run_label=run_label,
            base_run_id=base_run_id,
            run_id=run_id,
        )
    row = _ORIGINAL_BASE_RUN_ONE(
        args,
        nr,
        nz,
        dimension,
        ref,
        run_label=run_label,
        t_end_s=t_end_s,
        max_steps=max_steps,
        dt_initial_s=dt_initial_s,
        dt_max_s=dt_max_s,
    )
    row["resumed"] = False
    return row


def call_with_resume_runner(func: Any, *args: Any, **kwargs: Any) -> Any:
    previous_run_one = base.run_one
    base.run_one = run_one_with_resume
    try:
        return func(*args, **kwargs)
    finally:
        base.run_one = previous_run_one


def finite_rel_linf(values: Any, expected: Any) -> float:
    arr = np.asarray(values, dtype=float)
    exp = np.asarray(expected, dtype=float)
    finite = np.isfinite(arr) & np.isfinite(exp)
    if np.count_nonzero(finite) == 0:
        return math.inf
    denom = np.maximum(np.maximum(np.abs(arr[finite]), np.abs(exp[finite])), E_FLOOR)
    return float(np.nanmax(np.abs(arr[finite] - exp[finite]) / denom))


def reference_identity_branch_wise(ref: dict[str, Any]) -> dict[str, Any]:
    diagnostics = ref.get("diagnostics", {})
    transport_rel = float(diagnostics.get("transport_selfidentity_max_rel", math.inf))
    rh_rel = _max_numeric_leaf(diagnostics.get("rh_residual_rel", math.inf))
    angular = ref.get("angular", {})
    try:
        mu = np.asarray(angular["mu"], dtype=float)
        w = np.asarray(angular["w"], dtype=float)
        psi = np.asarray(angular["psi_table"], dtype=float)
        if psi.shape[0] != mu.size and psi.shape[-1] == mu.size:
            psi = np.swapaxes(psi, 0, -1)
        c_light = float(ref["parameters"]["c_cm_per_s"])
        e_calc = np.sum(w[:, None] * psi, axis=0) / c_light
        f_calc = np.sum((w * mu)[:, None] * psi, axis=0)
        p_calc = np.sum((w * mu * mu)[:, None] * psi, axis=0) / c_light
        chi_calc = p_calc / np.maximum(e_calc, E_FLOOR)
        angular_rel = max(
            finite_rel_linf(e_calc, angular["E_rad_erg_per_cm3"]),
            finite_rel_linf(f_calc, angular["F_rad_erg_per_cm2_s"]),
            finite_rel_linf(p_calc, angular["P_rad_dyne_per_cm2"]),
            finite_rel_linf(chi_calc, angular["chi_z"]),
        )
        angular_error = None
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        angular_rel = math.inf
        angular_error = str(exc)
    passed = (
        transport_rel <= REFERENCE_SELF_IDENTITY_GATE
        and rh_rel <= REFERENCE_SELF_IDENTITY_GATE
        and angular_rel <= ANGULAR_MOMENT_SELF_IDENTITY_GATE
    )
    return {
        "schema_version": ref.get("schema_version"),
        "problem": ref.get("problem"),
        "reference_model": ref.get("reference_model"),
        "thresholds": {
            "transport_selfidentity_max_rel_le": REFERENCE_SELF_IDENTITY_GATE,
            "rh_residual_rel_le": REFERENCE_SELF_IDENTITY_GATE,
            "angular_moment_recompute_max_rel_le": ANGULAR_MOMENT_SELF_IDENTITY_GATE,
        },
        "passed": bool(passed),
        "transport_selfidentity_max_rel": transport_rel,
        "rh_residual_rel": rh_rel,
        "angular_moment_recompute_max_rel": angular_rel,
        "angular_moment_recompute_error": angular_error,
        "upstream_median_rel": transport_rel,
        "upstream_p95_rel": transport_rel,
        "downstream_median_rel": rh_rel,
        "downstream_p95_rel": rh_rel,
        "total_median_rel": max(transport_rel, rh_rel),
        "total_p95_rel": max(transport_rel, rh_rel, angular_rel),
    }


def reference_shock_and_alignment(snapshot: RZSnapshot, ref: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    bundle = profile_bundle(snapshot)
    p = bundle["profiles"]
    raw_x = np.asarray(base.reference_frame_coord(snapshot, ref), dtype=float)
    ref_shock = base.reference_shock_position(ref)
    density_shock = i1.density_half_jump_shock_position(raw_x, p["rho"], ref)
    if math.isfinite(float(density_shock)):
        aligned_x = raw_x - (float(density_shock) - ref_shock)
        return aligned_x, ref_shock, float(density_shock)
    front = shock_metrics.detect_shock_front(
        p["Te"],
        np.asarray(bundle["z_cm"], dtype=float),
        rho_profile=p["rho"],
        uz_profile=p["u_z"],
        boundary_exclusion_cells=SHOCK_FRONT_BOUNDARY_EXCLUSION_CELLS,
        co_location_cells=SHOCK_FRONT_COLOCATION_CELLS,
    )
    if math.isfinite(float(front.z_shock)):
        shock_ref_frame = float(np.interp(float(front.z_shock), bundle["z_cm"], raw_x))
        aligned_x = raw_x - (shock_ref_frame - ref_shock)
        return aligned_x, ref_shock, shock_ref_frame
    return raw_x, ref_shock, math.nan


def windowed_l2(
    x: Any,
    code: Any,
    ref_x: Any,
    ref_values: Any,
    center: float,
    half_width: float,
    *,
    normalize_by_reference: bool = True,
) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=float)
    code_arr = np.asarray(code, dtype=float)
    ref_interp = np.interp(x_arr, np.asarray(ref_x, dtype=float), np.asarray(ref_values, dtype=float))
    mask = (
        np.isfinite(x_arr)
        & np.isfinite(code_arr)
        & np.isfinite(ref_interp)
        & (np.abs(x_arr - float(center)) <= float(half_width))
    )
    if np.count_nonzero(mask) == 0:
        return {"L2": math.inf, "norm": math.inf, "n": 0}
    diff = code_arr[mask] - ref_interp[mask]
    l2 = math.sqrt(float(np.mean(diff * diff)))
    norm = math.sqrt(float(np.mean(ref_interp[mask] * ref_interp[mask]))) if normalize_by_reference else 1.0
    return {"L2": l2 / max(norm, E_FLOOR), "norm": norm, "n": int(np.count_nonzero(mask))}


def chi_linf_outside_collar(
    x: Any,
    chi: Any,
    ref: dict[str, Any],
    center: float,
    half_width: float,
    collar_cells: int = 3,
) -> dict[str, Any]:
    x_arr = np.asarray(x, dtype=float)
    chi_arr = np.asarray(chi, dtype=float)
    ref_interp = np.interp(x_arr, np.asarray(ref["table"]["x_cm"], dtype=float), np.asarray(ref["table"]["chi_z"], dtype=float))
    dx = base.median_cell_spacing(x_arr)
    collar = collar_cells * dx if math.isfinite(dx) and dx > 0.0 else 0.0
    mask = (
        np.isfinite(x_arr)
        & np.isfinite(chi_arr)
        & np.isfinite(ref_interp)
        & (np.abs(x_arr - float(center)) <= float(half_width))
        & (np.abs(x_arr - float(center)) > collar)
    )
    if np.count_nonzero(mask) == 0:
        return {"linf": math.inf, "n": 0, "collar_cm": float(collar)}
    return {
        "linf": float(np.nanmax(np.abs(chi_arr[mask] - ref_interp[mask]))),
        "n": int(np.count_nonzero(mask)),
        "collar_cm": float(collar),
    }


def parse_sn_iteration_diagnostic(log_path: Path) -> dict[str, Any]:
    records = []
    for line in i1.read_text(log_path).splitlines():
        match = _SN_TIMING_RE.search(line)
        if not match:
            continue
        records.append(
            {
                "outer_iters": int(match.group("outer")),
                "inner_iters": int(match.group("inner")),
                "converged": int(match.group("converged")) != 0,
                "outer_residual": float(match.group("outer_resid")),
                "inner_residual": float(match.group("inner_resid")),
            }
        )
    if not records:
        return {
            "records": 0,
            "outer_iters_max": 0,
            "inner_iters_max": 0,
            "all_converged": False,
            "outer_residual_max": math.inf,
            "inner_residual_max": math.inf,
        }
    return {
        "records": len(records),
        "outer_iters_max": max(row["outer_iters"] for row in records),
        "inner_iters_max": max(row["inner_iters"] for row in records),
        "all_converged": all(row["converged"] for row in records),
        "outer_residual_max": max(float(row["outer_residual"]) for row in records),
        "inner_residual_max": max(float(row["inner_residual"]) for row in records),
    }


def final_snapshot_for_row(args: argparse.Namespace, row: dict[str, Any], min_count: int = 1) -> RZSnapshot | None:
    try:
        outdir = Path(args.out_root) / str(row["run_id"])
        results_dir = base.discover_results_dir(outdir)
        snapshots = base.read_snapshots(
            results_dir,
            str(row["case_name"]),
            int(row["nr"]),
            int(row["nz"]),
            "2D_RZ",
            min_count=min_count,
        )
        return snapshots[-1]
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError):
        return None


def i3_snapshot_metrics(snapshot: RZSnapshot, ref: dict[str, Any]) -> dict[str, Any]:
    bundle = profile_bundle(snapshot)
    p = bundle["profiles"]
    aligned_x, ref_shock, shock_ref_frame = reference_shock_and_alignment(snapshot, ref)
    mfp_post = i1.measured_postshock_mfp_cm(
        p["rho"],
        aligned_x,
        ref_shock,
        float(reference_for_i1(ref)["parameters"]["kappa_R_cm2_per_g"]),
    )
    half_width = SHOCK_WINDOW_MFP_COUNT * mfp_post
    fz_l2 = windowed_l2(
        aligned_x,
        p["F_z"],
        ref["table"]["x_cm"],
        ref["table"]["F_rad_erg_per_cm2_s"],
        ref_shock,
        half_width,
    )
    chi = chi_linf_outside_collar(aligned_x, p["chi_z"], ref, ref_shock, half_width)
    radial = bundle["radial_invariance"]
    theta = np.asarray(snapshot.sn_stream_theta, dtype=float)
    radial_fixup_count = np.asarray(snapshot.sn_radial_fixup_count, dtype=float)
    angular_fixup_count = np.asarray(snapshot.sn_angular_fixup_count, dtype=float)
    radial_fixup_abs = np.asarray(snapshot.sn_radial_fixup_artificial_abs, dtype=float)
    angular_fixup_abs = np.asarray(snapshot.sn_angular_fixup_artificial_abs, dtype=float)
    e_pre = np.asarray(snapshot.sn_rad_E_pre_newton, dtype=float)
    e_post = np.asarray(snapshot.rad_E, dtype=float)
    e_sweep_rel = float(
        np.nanmax(np.abs(e_pre - e_post) / np.maximum(np.maximum(np.abs(e_pre), np.abs(e_post)), E_FLOOR))
    )
    radial_gate_fields = ("rho", "Te", "Ti", "E_rad", "u_z", "F_z", "chi_z")
    return {
        "i3_shock_ref_frame_cm": float(shock_ref_frame),
        "i3_window_half_cm": float(half_width),
        "i3_F_z_windowed_L2": float(fz_l2["L2"]),
        "i3_F_z_windowed_cells": int(fz_l2["n"]),
        "i3_chi_z_linf_outside_front_collar": float(chi["linf"]),
        "i3_chi_z_collar_cm": float(chi["collar_cm"]),
        "i3_radial_invariance_max_rel": max(float(radial.get(name, math.inf)) for name in radial_gate_fields),
        "i3_radial_invariance_max_rel_by_field": {name: float(radial.get(name, math.inf)) for name in radial_gate_fields},
        "i3_sn_stream_theta_activity_count": int(np.count_nonzero(np.isfinite(theta) & (theta < 1.0 - 1.0e-14))),
        "i3_sn_stream_theta_min": float(np.nanmin(theta)) if theta.size else 1.0,
        "i3_sn_radial_fixup_count_max": float(np.nanmax(radial_fixup_count)) if radial_fixup_count.size else 0.0,
        "i3_sn_angular_fixup_count_max": float(np.nanmax(angular_fixup_count)) if angular_fixup_count.size else 0.0,
        "i3_sn_radial_fixup_artificial_abs_max": float(np.nanmax(radial_fixup_abs)) if radial_fixup_abs.size else 0.0,
        "i3_sn_angular_fixup_artificial_abs_max": float(np.nanmax(angular_fixup_abs)) if angular_fixup_abs.size else 0.0,
        "i3_sn_ap_alpha_max": float(snapshot.sn_ap_alpha_max),
        "i3_sn_ap_alpha_active_faces": float(snapshot.sn_ap_alpha_active_faces),
        "i3_E_sweep_minus_E_plus_rel_linf": e_sweep_rel,
        "i3_final_dt_s": float(snapshot.dt),
    }


def augment_i3_run_metrics(args: argparse.Namespace, row: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    log_path = Path(args.out_root) / str(row.get("run_id", "")) / "run.log"
    out["sn_iteration_diagnostic"] = parse_sn_iteration_diagnostic(log_path)
    if row.get("exit_code") != 0 or row.get("timed_out") or row.get("run_status") == "analysis_failure":
        return out
    snapshot = final_snapshot_for_row(args, row)
    if snapshot is None:
        out["i3_analysis_error"] = "missing final snapshot"
        return out
    try:
        out.update(i3_snapshot_metrics(snapshot, ref))
    except (KeyError, ValueError, RuntimeError, FloatingPointError) as exc:
        out["i3_analysis_error"] = str(exc)
    return out


def set_case_context(n_angles: int, kappa_scale: int, radiation_mode: str, ref_path: Path) -> None:
    _CASE_CONTEXT.update(
        {
            "n_angles": int(n_angles),
            "kappa_scale": int(kappa_scale),
            "radiation_mode": str(radiation_mode),
            "reference": str(ref_path),
        }
    )


def run_t0_audit_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    ref_path: Path,
    *,
    n_angles: int,
    kappa_scale: int,
    radiation_mode: str,
) -> dict[str, Any]:
    set_case_context(n_angles, kappa_scale, radiation_mode, ref_path)
    return call_with_resume_runner(
        i1.run_t0_audit_one_dispatch,
        args,
        nr,
        nz,
        dimension,
        reference_for_i1(ref),
    )


def run_first_step_audit_one(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    ref_path: Path,
    *,
    n_angles: int,
    kappa_scale: int,
    radiation_mode: str,
) -> dict[str, Any]:
    set_case_context(n_angles, kappa_scale, radiation_mode, ref_path)
    return call_with_resume_runner(
        i1.run_first_step_audit_one,
        args,
        nr,
        nz,
        dimension,
        reference_for_i1(ref),
    )


def run_one_case(
    args: argparse.Namespace,
    nr: int,
    nz: int,
    dimension: str,
    ref: dict[str, Any],
    ref_path: Path,
    *,
    n_angles: int,
    kappa_scale: int,
    radiation_mode: str,
    run_label: str = "production",
    t_end_s: float | None = None,
    dt_initial_s: float | None = None,
    dt_max_s: float | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    set_case_context(n_angles, kappa_scale, radiation_mode, ref_path)
    row = call_with_resume_runner(
        i1.run_one,
        args,
        nr,
        nz,
        dimension,
        reference_for_i1(ref),
        t_end_s=t_end_s,
        dt_initial_s=dt_initial_s,
        dt_max_s=dt_max_s,
        max_steps=max_steps,
    )
    if run_label != "production":
        row["i3_run_label"] = run_label
    row["n_angles"] = int(n_angles)
    row["kappa_scale"] = int(kappa_scale)
    row["radiation_mode"] = radiation_mode
    row["reference_path"] = str(ref_path)
    return augment_i3_run_metrics(args, row, ref)


def finite_max(rows: list[dict[str, Any]], key: str, default: float = math.inf) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, math.inf))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return max(values) if values else default


def final_profile(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any] | None:
    snapshot = final_snapshot_for_row(args, row)
    if snapshot is None:
        return None
    bundle = profile_bundle(snapshot)
    return {
        "snapshot": snapshot,
        "z": np.asarray(bundle["z_cm"], dtype=float),
        "profiles": bundle["profiles"],
    }


def compare_profiles(
    left: dict[str, Any],
    right: dict[str, Any],
    field: str,
    ref: dict[str, Any],
    *,
    half_width_cm: float | None = None,
) -> float:
    x_left = np.asarray(left["z"], dtype=float)
    y_left = np.asarray(left["profiles"][field], dtype=float)
    x_right = np.asarray(right["z"], dtype=float)
    y_right = np.asarray(right["profiles"][field], dtype=float)
    center = base.reference_shock_position(ref)
    if half_width_cm is None:
        half_width_cm = 0.5 * (
            float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
        )
    return float(
        windowed_l2(
            x_left,
            y_left,
            x_right,
            y_right,
            center,
            half_width_cm,
        )["L2"]
    )


def aggregate_quadrature_ladder(
    args: argparse.Namespace,
    rung_rows: dict[int, dict[str, Any]],
    refs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    residuals: dict[str, dict[str, float]] = {field: {} for field in ("E", "T_rad", "F_z", "chi_z")}
    profiles: dict[int, dict[str, Any]] = {}
    for n, row in rung_rows.items():
        prof = final_profile(args, row)
        if prof is not None:
            profiles[n] = prof
    for n, prof in profiles.items():
        ref = refs[n]
        ref_x = ref["table"]["x_cm"]
        ref_shock = base.reference_shock_position(ref)
        span = float(ref_x[-1]) - float(ref_x[0])
        half_width = 0.5 * span
        mapping = {
            "E": ("E_rad", "E_rad_erg_per_cm3"),
            "T_rad": ("T_rad", "T_rad_eV"),
            "F_z": ("F_z", "F_rad_erg_per_cm2_s"),
            "chi_z": ("chi_z", "chi_z"),
        }
        for metric, (code_field, ref_field) in mapping.items():
            residuals[metric][str(n)] = float(
                windowed_l2(
                    prof["z"],
                    prof["profiles"][code_field],
                    ref_x,
                    ref["table"][ref_field],
                    ref_shock,
                    half_width,
                )["L2"]
            )
    monotone = {}
    for field, values in residuals.items():
        ordered = [values.get(str(n), math.inf) for n in (4, 8, 16)]
        monotone[field] = ordered[2] <= ordered[1] <= ordered[0]
    cauchy = {}
    comparability = {}
    for field in ("E_rad", "T_rad", "F_z", "chi_z"):
        if all(n in profiles for n in (4, 8, 16)):
            ref = refs[16]
            span = float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
            d48 = compare_profiles(profiles[4], profiles[8], field, ref, half_width_cm=0.5 * span)
            d816 = compare_profiles(profiles[8], profiles[16], field, ref, half_width_cm=0.5 * span)
            cauchy[field] = {"S4_S8": d48, "S8_S16": d816, "passed": d48 > d816}
        else:
            cauchy[field] = {"S4_S8": math.inf, "S8_S16": math.inf, "passed": False}
    for field, values in residuals.items():
        vals = [float(v) for v in values.values() if math.isfinite(float(v))]
        comparability[field] = (max(vals) / max(min(vals), E_FLOOR)) if vals else math.inf
    return {
        "residuals": residuals,
        "monotone_by_field": monotone,
        "cauchy": cauchy,
        "rung_residual_comparability": comparability,
        "G_C_monotone_matched_reference": all(monotone.values()) if monotone else False,
        "G_C_cauchy_decreasing": all(row["passed"] for row in cauchy.values()) if cauchy else False,
        "G_C_rung_residual_comparability": all(value <= 2.0 for value in comparability.values()) if comparability else False,
    }


def fleck_min_from_ref(ref: dict[str, Any], dt_s: float | None) -> float:
    if dt_s is None or not math.isfinite(float(dt_s)) or float(dt_s) <= 0.0:
        return 1.0
    params = ref["parameters"]
    cv = float(params.get("Cv_erg_per_g_eV", 0.0))
    if cv <= 0.0:
        ev_to_erg = float(params["eV_to_erg"])
        proton_mass = float(params["proton_mass_g"])
        cv = float(params["zbar"]) * ev_to_erg / (
            float(params["A_amu"]) * proton_mass * (float(params["gamma"]) - 1.0)
        )
    sigma_candidates = []
    kappa = float(params.get("kappa_R_cm2_per_g", params.get("kappa_a_cm2_per_g", 0.0)))
    for state in ref.get("states", {}).values():
        try:
            rho = float(state["rho"])
            T = float(state["T"])
        except (KeyError, TypeError, ValueError):
            continue
        beta = 4.0 * float(params["a_eV_erg_cm3_eV4"]) * T**3 / max(rho * cv, E_FLOOR)
        sigma_candidates.append(beta * float(params["c_cm_per_s"]) * kappa * rho * float(dt_s))
    z = max(sigma_candidates) if sigma_candidates else 0.0
    return 1.0 / (1.0 + max(z, 0.0))


def aggregate_thick_limit(
    args: argparse.Namespace,
    pair_rows: dict[int, dict[str, dict[str, Any]]],
    sn_refs: dict[int, dict[str, Any]],
    fld_refs: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    by_k: dict[str, Any] = {}
    for kappa_scale, rows in pair_rows.items():
        sn_row = rows.get("sn_transport")
        fld_row = rows.get("multigroup_diffusion")
        if sn_row is None or fld_row is None:
            continue
        sn_prof = final_profile(args, sn_row)
        fld_prof = final_profile(args, fld_row)
        if sn_prof is None or fld_prof is None:
            continue
        ref = sn_refs[kappa_scale]
        span = float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
        half_width = min(5.8, 0.5 * span)
        E_l2 = compare_profiles(sn_prof, fld_prof, "E_rad", ref, half_width_cm=half_width)
        T_l2 = compare_profiles(sn_prof, fld_prof, "T_rad", ref, half_width_cm=half_width)
        x = np.asarray(sn_prof["z"], dtype=float)
        chi = np.asarray(sn_prof["profiles"]["chi_z"], dtype=float)
        center = base.reference_shock_position(ref)
        collar = 10.0 / max(0.5 * float(kappa_scale) * float(ref["states"]["upstream"]["rho"]), E_FLOOR)
        mask = np.isfinite(x) & np.isfinite(chi) & (np.abs(x - center) <= half_width) & (np.abs(x - center) > collar)
        chi_linf = float(np.nanmax(np.abs(chi[mask] - 1.0 / 3.0))) if np.count_nonzero(mask) else math.inf
        fld_dt = optional_float(fld_row.get("i3_final_dt_s"))
        by_k[str(kappa_scale)] = {
            "SN_minus_FLD_E_windowed_L2": E_l2,
            "SN_minus_FLD_T_rad_windowed_L2": T_l2,
            "SN_chi_z_minus_one_third_linf_interior": chi_linf,
            "interior_collar_cm": float(collar),
            "FLD_fleck_min_analytic": fleck_min_from_ref(fld_refs[kappa_scale], fld_dt),
        }
    e_vals = [by_k.get(str(k), {}).get("SN_minus_FLD_E_windowed_L2", math.inf) for k in (1, 20, 200)]
    t_vals = [by_k.get(str(k), {}).get("SN_minus_FLD_T_rad_windowed_L2", math.inf) for k in (1, 20, 200)]
    k2 = by_k.get("200", {})
    k2_E = float(k2.get("SN_minus_FLD_E_windowed_L2", math.inf))
    k2_T = float(k2.get("SN_minus_FLD_T_rad_windowed_L2", math.inf))
    k2_eng_E = math.inf
    k2_eng_T = math.inf
    k2_eng_rows = pair_rows.get(200, {})
    sn_eng_row = k2_eng_rows.get("sn_transport_engineering")
    fld_eng_row = k2_eng_rows.get("multigroup_diffusion_engineering")
    if sn_eng_row is not None and fld_eng_row is not None:
        sn_eng_prof = final_profile(args, sn_eng_row)
        fld_eng_prof = final_profile(args, fld_eng_row)
        if sn_eng_prof is not None and fld_eng_prof is not None:
            ref = sn_refs[200]
            span = float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
            half_width = min(5.8, 0.5 * span)
            k2_eng_E = compare_profiles(sn_eng_prof, fld_eng_prof, "E_rad", ref, half_width_cm=half_width)
            k2_eng_T = compare_profiles(sn_eng_prof, fld_eng_prof, "T_rad", ref, half_width_cm=half_width)
    e_code_term = max(0.0, k2_eng_E - G_D_K2_E_FLOOR) if math.isfinite(k2_eng_E) else math.inf
    t_code_term = max(0.0, k2_eng_T - G_D_K2_TRAD_FLOOR) if math.isfinite(k2_eng_T) else math.inf
    e_landing_reason = None if math.isfinite(k2_eng_E) else "engineering_rung_measurement_missing"
    t_landing_reason = None if math.isfinite(k2_eng_T) else "engineering_rung_measurement_missing"
    second_difference = second_difference_metric(args, pair_rows, sn_refs, fld_refs)
    metrics.update(
        {
            "by_kappa_scale": by_k,
            "G_D_SN_FLD_E_monotone_decreasing_with_K": e_vals[2] <= e_vals[1] <= e_vals[0],
            "G_D_SN_FLD_T_rad_monotone_decreasing_with_K": t_vals[2] <= t_vals[1] <= t_vals[0],
            "G_D_K2_E_registry_ceiling": k2_E <= G_D_K2_E_L2_GATE,
            "G_D_K2_T_rad_registry_ceiling": k2_T <= G_D_K2_TRAD_L2_GATE,
            "G_D_K2_chi_z_interior_registry_ceiling": float(
                k2.get("SN_chi_z_minus_one_third_linf_interior", math.inf)
            )
            <= G_D_K2_CHI_INTERIOR_GATE,
            "G_D_K2_E_engineering_rung_L2": k2_eng_E,
            "G_D_K2_T_rad_engineering_rung_L2": k2_eng_T,
            "G_D_K2_E_prediction_landing": e_landing_reason is None
            and k2_E <= 1.5 * (G_D_K2_E_FLOOR + e_code_term),
            "G_D_K2_T_rad_prediction_landing": t_landing_reason is None
            and k2_T <= 1.5 * (G_D_K2_TRAD_FLOOR + t_code_term),
            "G_D_K2_E_prediction_landing_reason": e_landing_reason,
            "G_D_K2_T_rad_prediction_landing_reason": t_landing_reason,
            "G_D_K1_second_difference_E_windowed_L2": second_difference,
            "G_D_K1_second_difference": second_difference <= G_D_SECOND_DIFFERENCE_GATE,
            "G_D_FLD_fleck_min": min(
                [float(row.get("FLD_fleck_min_analytic", 1.0)) for row in by_k.values()] or [1.0]
            ),
            "G_D_FLD_fleck_min_pass": all(
                float(row.get("FLD_fleck_min_analytic", 1.0)) >= 0.99 for row in by_k.values()
            ),
        }
    )
    return metrics


def optional_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def second_difference_metric(
    args: argparse.Namespace,
    pair_rows: dict[int, dict[str, dict[str, Any]]],
    sn_refs: dict[int, dict[str, Any]],
    fld_refs: dict[int, dict[str, Any]],
) -> float:
    rows = pair_rows.get(1, {})
    sn_prof = final_profile(args, rows.get("sn_transport", {}))
    fld_prof = final_profile(args, rows.get("multigroup_diffusion", {}))
    if sn_prof is None or fld_prof is None:
        return math.inf
    sn_ref = sn_refs[1]
    fld_ref = fld_refs[1]
    x = np.asarray(sn_prof["z"], dtype=float)
    code_delta = np.asarray(sn_prof["profiles"]["E_rad"], dtype=float) - np.interp(
        x, np.asarray(fld_prof["z"], dtype=float), np.asarray(fld_prof["profiles"]["E_rad"], dtype=float)
    )
    ref_delta = np.interp(x, sn_ref["table"]["x_cm"], sn_ref["table"]["E_rad_erg_per_cm3"]) - np.interp(
        x, fld_ref["table"]["x_cm"], fld_ref["table"]["E_rad_erg_per_cm3"]
    )
    center = base.reference_shock_position(sn_ref)
    half_width = 0.5 * (float(sn_ref["table"]["x_cm"][-1]) - float(sn_ref["table"]["x_cm"][0]))
    return float(windowed_l2(x, code_delta, x, ref_delta, center, half_width)["L2"])


def aggregate_smoke(row: dict[str, Any], t0_aggregate: dict[str, Any], first_step_rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_step = i1.aggregate_first_step_audit(first_step_rows, reference_for_i1(read_json(Path(row["reference_path"])))) if first_step_rows else {}
    step = max(float(row.get("shock_gate_snapshot_step", row.get("step", SMOKE_MAX_STEPS))), 1.0)
    gates = {
        "G_A_t0_admissibility": bool(t0_aggregate.get("passed")),
        "G_A_first_step_rh_flux": bool(first_step.get("first_step_rh_flux_pass", False)),
        "G_F_theta_fixup_tallies_zero": (
            float(row.get("i3_sn_stream_theta_activity_count", math.inf)) == 0.0
            and float(row.get("i3_sn_radial_fixup_count_max", math.inf)) == 0.0
            and float(row.get("i3_sn_angular_fixup_count_max", math.inf)) == 0.0
        ),
        "G_F_sn_ap_alpha_zero": float(row.get("i3_sn_ap_alpha_max", math.inf)) == 0.0,
    }
    return {
        "mode": "smoke",
        "s_per_step_wall_clock": float(row.get("wall_time_s", math.inf)) / step,
        "gates": gates,
        "all_checks_passed": all(gates.values()),
    }


def all_leg_rows(
    t0_rows: list[dict[str, Any]],
    first_step_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    rung_rows: dict[int, dict[str, Any]],
    pair_rows: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(t0_rows)
    rows.extend(first_step_rows)
    rows.extend(strict_rows)
    rows.extend(rung_rows.values())
    for modes in pair_rows.values():
        rows.extend(modes.values())
    return rows


def aggregate_skipped_for_legs(
    args: argparse.Namespace,
    t0_rows: list[dict[str, Any]],
    first_step_rows: list[dict[str, Any]],
    strict_rows: list[dict[str, Any]],
    rung_rows: dict[int, dict[str, Any]],
    pair_rows: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    rows = all_leg_rows(t0_rows, first_step_rows, strict_rows, rung_rows, pair_rows)
    skipped = [row for row in rows if row.get("run_status") == "skipped_by_legs_filter"]
    selected = [row for row in rows if row.get("run_status") != "skipped_by_legs_filter"]
    failed = [
        str(row.get("run_id", ""))
        for row in selected
        if row.get("exit_code") != 0 or bool(row.get("timed_out"))
    ]
    return {
        "mode": args.mode,
        "legs_filter": args.legs,
        "aggregate_skipped": True,
        "selected_leg_count": len(selected),
        "skipped_leg_count": len(skipped),
        "failed_leg_ids": failed,
        "gates": {},
        "all_checks_passed": not failed,
    }


def aggregate_full(
    args: argparse.Namespace,
    strict_rows: list[dict[str, Any]],
    first_step_rows: list[dict[str, Any]],
    ref: dict[str, Any],
    t0_aggregate: dict[str, Any],
    ref_identity: dict[str, Any],
    rung_rows: dict[int, dict[str, Any]],
    rung_refs: dict[int, dict[str, Any]],
    pair_rows: dict[int, dict[str, dict[str, Any]]],
    sn_refs_by_k: dict[int, dict[str, Any]],
    fld_refs_by_k: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    base_agg = i1.aggregate(strict_rows, first_step_rows, reference_for_i1(ref), t0_aggregate, ref_identity, len(strict_rows))
    fine = strict_rows[-1] if strict_rows else {}
    base_gates = base_agg.get("gates", {})
    ladder = aggregate_quadrature_ladder(args, rung_rows, rung_refs)
    thick = aggregate_thick_limit(args, pair_rows, sn_refs_by_k, fld_refs_by_k)
    no_escape = base_agg.get("no_escape_valves", {})
    sn_iterations = [row.get("sn_iteration_diagnostic", {}) for row in strict_rows]
    sn_outer_max = max([int(row.get("outer_iters_max", 0)) for row in sn_iterations] or [0])
    sn_inner_max = max([int(row.get("inner_iters_max", 0)) for row in sn_iterations] or [0])
    fld_newton_passes = [
        bool(modes.get("multigroup_diffusion", {}).get("newton_convergence_pass"))
        for modes in pair_rows.values()
        if "multigroup_diffusion" in modes
    ]
    gates = {
        "G_A_t0_admissibility": bool(t0_aggregate.get("passed")),
        "G_A_first_step_rh_flux": bool(base_gates.get("gate11_first_step_rh_flux", False)),
        "G_A_reference_self_identity": bool(ref_identity.get("passed")),
        "G_A_radiation_equilibrium": bool(
            base_agg.get("reference_table_radiation_equilibrium_admissibility", False)
        ),
        "G_A_low_prad": bool(base_gates.get("gate1_low_prad_only", False)),
        "G_A_reference_conservation": bool(base_gates.get("gate2_reference_conservation", False)),
        "G_B_S16_dTr_windowed_L2": float(base_agg.get("fine_shock_windowed_l2_abs", math.inf)) <= G_B_DTR_L2_GATE,
        "G_B_S16_F_z_windowed_L2": float(fine.get("i3_F_z_windowed_L2", math.inf)) <= G_B_FZ_L2_GATE,
        "G_B_S16_chi_z_linf_outside_collar": float(fine.get("i3_chi_z_linf_outside_front_collar", math.inf))
        <= G_B_CHI_LINF_GATE,
        "G_B_fixed_precursor_triplet": bool(base_gates.get("gate7a_fixed_precursor_amplitude", False))
        and bool(base_gates.get("gate7b_fixed_precursor_integrated", False))
        and bool(base_gates.get("gate7c_fixed_precursor_shape_l2", False)),
        "G_B_matter_shock_width": bool(base_gates.get("gate7d_matter_shock_thickness", False)),
        "G_B_richardson_D_inf_window": G_B_RICHARDSON_LOW
        <= float(base_agg.get("richardson_d_infinity_fit", math.inf))
        <= G_B_RICHARDSON_HIGH,
        "G_B_radial_invariance": finite_max(strict_rows, "i3_radial_invariance_max_rel") <= RADIAL_INVARIANCE_GATE,
        "G_C_monotone_matched_reference": bool(ladder.get("G_C_monotone_matched_reference")),
        "G_C_cauchy_decreasing": bool(ladder.get("G_C_cauchy_decreasing")),
        "G_C_rung_residual_comparability": bool(ladder.get("G_C_rung_residual_comparability")),
        "G_D_SN_FLD_E_monotone_decreasing_with_K": bool(thick.get("G_D_SN_FLD_E_monotone_decreasing_with_K")),
        "G_D_SN_FLD_T_rad_monotone_decreasing_with_K": bool(thick.get("G_D_SN_FLD_T_rad_monotone_decreasing_with_K")),
        "G_D_K2_E_registry_ceiling": bool(thick.get("G_D_K2_E_registry_ceiling")),
        "G_D_K2_T_rad_registry_ceiling": bool(thick.get("G_D_K2_T_rad_registry_ceiling")),
        "G_D_K2_chi_z_interior_registry_ceiling": bool(thick.get("G_D_K2_chi_z_interior_registry_ceiling")),
        "G_D_K2_E_prediction_landing": bool(thick.get("G_D_K2_E_prediction_landing")),
        "G_D_K2_T_rad_prediction_landing": bool(thick.get("G_D_K2_T_rad_prediction_landing")),
        "G_D_K1_second_difference": bool(thick.get("G_D_K1_second_difference")),
        "G_E_sn_iteration_bounds": sn_outer_max <= G_E_SN_OUTER_LIMIT and sn_inner_max <= G_E_SN_INNER_LIMIT,
        "G_E_newton_gate": all(fld_newton_passes) if fld_newton_passes else bool(base_agg.get("newton_convergence_pass", True)),
        "G_E_escape_valves_zero": bool(no_escape.get("all_zero", False)),
        "G_E_conservation": bool(base_gates.get("gate2_reference_conservation", False)),
        "G_E_t_end_reached": bool(base_gates.get("termination_t_end_reached", False)),
        "G_F_theta_fixup_tallies_zero": finite_max(strict_rows, "i3_sn_stream_theta_activity_count") == 0.0
        and finite_max(strict_rows, "i3_sn_radial_fixup_count_max") == 0.0
        and finite_max(strict_rows, "i3_sn_angular_fixup_count_max") == 0.0,
        "G_F_sn_ap_alpha_zero": finite_max(strict_rows, "i3_sn_ap_alpha_max") == 0.0,
        "G_F_FLD_fleck_min": bool(thick.get("G_D_FLD_fleck_min_pass")),
    }
    base_agg.update(
        {
            "sn_outer_iters_max": sn_outer_max,
            "sn_inner_iters_max": sn_inner_max,
            "quadrature_ladder": ladder,
            "thick_limit": thick,
            "gates": gates,
            "all_checks_passed": all(gates.values()),
        }
    )
    return base_agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/i3_sn_radshock_2d_rz_slab.py")
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--rungs", type=lambda s: parse_int_csv(s, {4, 8, 16, 32}, "rungs"), default=parse_int_csv("4,8,16", {4, 8, 16, 32}, "rungs"))
    parser.add_argument("--kappa-scales", type=lambda s: parse_int_csv(s, {1, 20, 200}, "kappa scales"), default=parse_int_csv("1,20,200", {1, 20, 200}, "kappa scales"))
    parser.add_argument("--grids", type=parse_grid_list, default=parse_grid_list("16x256;32x512;64x1024"))
    parser.add_argument("--mode", choices=("full", "smoke", "t0-only", "rung", "thick"), default="full")
    parser.add_argument("--radiation-mode", choices=("sn_transport", "multigroup_diffusion"), default=None)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-mode", choices=("reference_table", "two_state"), default="reference_table")
    parser.add_argument("--mesh-motion", choices=("lagrangian",), default="lagrangian")
    parser.add_argument("--boundary-mode", choices=("reflecting", "finite_shock_tube", "state_supply"), default="state_supply")
    parser.add_argument("--production-audit-tier", choices=("none", "A", "B"), default="A")
    parser.add_argument("--out-root", default="./build/i3_sn_radshock_2d_rz_slab")
    parser.add_argument("--summary-json", default="./build/i3_sn_radshock_2d_rz_slab/summary.json")
    parser.add_argument("--legs", default=None, help="comma-separated fnmatch globs; run only legs whose label matches")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="quiet")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip a leg's subprocess when its run.log exists AND run_info "
            "reports termination_reason=t_end_reached; existing outputs are "
            "then re-read and gated normally"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    install_base_hooks()
    parser = build_parser()
    args = parser.parse_args(argv)
    if env_resume_enabled():
        args.resume = True
    filter_active = legs_filter_active(args)
    # base-engine namespace contract (one-pass audit vs run_i1_le08 argparse defaults)
    for name, default in {
        "reference": "tests/verification/data/le08_ned_reference/le08_nED_M2.json",
        "dump_profiles": False,
        "allow_partial_checkpoint": False,
    }.items():
        if not hasattr(args, name):
            setattr(args, name, default)
    dimension = base.deck_dimension(Path(args.deck))
    if dimension != "2D_RZ":
        parser.error("I3 SN radiative shock requires a deck with dimension=\"2D_RZ\"")

    s16 = 16 if 16 in args.rungs else args.rungs[-1]
    sn_ref_path = sn_reference_path(args.mach, s16, 1)
    sn_ref = read_json(sn_ref_path)
    ref_identity = reference_identity_branch_wise(sn_ref)
    active_mode = args.radiation_mode or "sn_transport"
    smoke_grid = [SMOKE_GRID]
    selected_grids = smoke_grid if args.mode == "smoke" else args.grids
    strict_run_grids = selected_grids
    full_anchor_grid: tuple[int, int] | None = None
    if args.mode == "full" and len(selected_grids) > 1:
        strict_run_grids = selected_grids[:-1]
        full_anchor_grid = selected_grids[-1]
    t0_rows: list[dict[str, Any]] = []
    t0_aggregate: dict[str, Any] = {"passed": True, "gates": {}}
    first_step_rows: list[dict[str, Any]] = []
    strict_rows: list[dict[str, Any]] = []
    rung_rows: dict[int, dict[str, Any]] = {}
    rung_refs: dict[int, dict[str, Any]] = {}
    pair_rows: dict[int, dict[str, dict[str, Any]]] = {}
    sn_refs_by_k: dict[int, dict[str, Any]] = {}
    fld_refs_by_k: dict[int, dict[str, Any]] = {}

    if args.mode in ("full", "smoke", "t0-only"):
        audit_n = SMOKE_N_ANGLES if args.mode == "smoke" else s16
        audit_ref_path = reference_path(args.mach, audit_n, 1, active_mode)
        audit_ref = read_json(audit_ref_path)
        t0_rows = [
            run_t0_audit_one(
                args,
                nr,
                nz,
                dimension,
                audit_ref,
                audit_ref_path,
                n_angles=audit_n,
                kappa_scale=1,
                radiation_mode=active_mode,
            )
            for nr, nz in selected_grids
        ]
        if filter_active:
            t0_aggregate = {"passed": None, "gates": {}, "aggregate_skipped": True, "legs_filter": args.legs}
        else:
            t0_aggregate = i1.aggregate_t0_audits_dispatch(args, t0_rows)
        if args.mode == "t0-only":
            if filter_active:
                aggregate = aggregate_skipped_for_legs(
                    args,
                    t0_rows,
                    first_step_rows,
                    strict_rows,
                    rung_rows,
                    pair_rows,
                )
            else:
                aggregate = {
                    "mode": "t0-only",
                    "gates": {
                        "G_A_t0_admissibility": bool(t0_aggregate.get("passed")),
                        "G_A_reference_self_identity": bool(ref_identity.get("passed")),
                    },
                    "all_checks_passed": bool(t0_aggregate.get("passed")) and bool(ref_identity.get("passed")),
                }
            payload = summary_payload(args, ref_identity, t0_rows, t0_aggregate, [], [], aggregate)
            base.write_json(Path(args.summary_json), payload)
            print_summary(ref_identity, t0_aggregate, [], aggregate)
            return 0 if bool(aggregate["all_checks_passed"]) else 1
        if filter_active or bool(t0_aggregate.get("passed")):
            first_step_rows = [
                run_first_step_audit_one(
                    args,
                    nr,
                    nz,
                    dimension,
                    audit_ref,
                    audit_ref_path,
                    n_angles=audit_n,
                    kappa_scale=1,
                    radiation_mode=active_mode,
                )
                for nr, nz in selected_grids
            ]
            strict_rows = [
                run_one_case(
                    args,
                    nr,
                    nz,
                    dimension,
                    audit_ref,
                    audit_ref_path,
                    n_angles=audit_n,
                    kappa_scale=1,
                    radiation_mode=active_mode,
                    max_steps=SMOKE_MAX_STEPS if args.mode == "smoke" else None,
                )
                for nr, nz in strict_run_grids
            ]

    if args.mode in ("full", "rung"):
        rung_grid = args.grids[1] if len(args.grids) > 1 else args.grids[0]
        for n_angles in args.rungs:
            ref_path = sn_reference_path(args.mach, n_angles, 1)
            ref = read_json(ref_path)
            rung_refs[n_angles] = ref
            if n_angles == 16:
                reused_row = next(
                    (
                        row
                        for row in strict_rows
                        if int(row.get("nr", -1)) == rung_grid[0]
                        and int(row.get("nz", -1)) == rung_grid[1]
                        and int(row.get("n_angles", -1)) == 16
                        and int(row.get("kappa_scale", -1)) == 1
                        and row.get("radiation_mode") == "sn_transport"
                    ),
                    None,
                )
                if reused_row is not None:
                    rung_rows[n_angles] = reused_row
                    continue
            rung_rows[n_angles] = run_one_case(
                args,
                rung_grid[0],
                rung_grid[1],
                dimension,
                ref,
                ref_path,
                n_angles=n_angles,
                kappa_scale=1,
                radiation_mode="sn_transport",
                run_label="production" if n_angles == 16 else f"rung_N{n_angles}",
            )

    if args.mode in ("full", "thick"):
        thick_grid = args.grids[1] if len(args.grids) > 1 else args.grids[0]
        thick_engineering_grid = (32, 512)
        thick_production_grid = (64, 1024)
        for kappa_scale in args.kappa_scales:
            pair_rows[kappa_scale] = {}
            sn_ref_k_path = sn_reference_path(args.mach, 16, kappa_scale)
            fld_ref_k_path = fld_reference_path(args.mach, kappa_scale)
            sn_refs_by_k[kappa_scale] = read_json(sn_ref_k_path)
            fld_refs_by_k[kappa_scale] = read_json(fld_ref_k_path)
            dt_cap = FROZEN_PREDICTION_REFS["dt_caps_s"].get(str(kappa_scale))
            for radiation_mode in ("sn_transport", "multigroup_diffusion"):
                ref = sn_refs_by_k[kappa_scale] if radiation_mode == "sn_transport" else fld_refs_by_k[kappa_scale]
                ref_path = sn_ref_k_path if radiation_mode == "sn_transport" else fld_ref_k_path
                run_grid = thick_production_grid if kappa_scale == 200 else thick_grid
                pair_rows[kappa_scale][radiation_mode] = run_one_case(
                    args,
                    run_grid[0],
                    run_grid[1],
                    dimension,
                    ref,
                    ref_path,
                    n_angles=16,
                    kappa_scale=kappa_scale,
                    radiation_mode=radiation_mode,
                    run_label=f"thick_K{kappa_scale}_{radiation_mode}",
                    dt_max_s=float(dt_cap) if dt_cap is not None else None,
                )
                if kappa_scale == 200:
                    pair_rows[kappa_scale][f"{radiation_mode}_engineering"] = run_one_case(
                        args,
                        thick_engineering_grid[0],
                        thick_engineering_grid[1],
                        dimension,
                        ref,
                        ref_path,
                        n_angles=16,
                        kappa_scale=kappa_scale,
                        radiation_mode=radiation_mode,
                        run_label=f"thick_K{kappa_scale}_{radiation_mode}_engineering",
                        dt_max_s=float(dt_cap) if dt_cap is not None else None,
                    )

    if args.mode == "full" and full_anchor_grid is not None and (filter_active or bool(t0_aggregate.get("passed"))):
        strict_rows.append(
            run_one_case(
                args,
                full_anchor_grid[0],
                full_anchor_grid[1],
                dimension,
                audit_ref,
                audit_ref_path,
                n_angles=audit_n,
                kappa_scale=1,
                radiation_mode=active_mode,
            )
        )

    if filter_active:
        aggregate = aggregate_skipped_for_legs(
            args,
            t0_rows,
            first_step_rows,
            strict_rows,
            rung_rows,
            pair_rows,
        )
    elif args.mode == "smoke":
        aggregate = aggregate_smoke(strict_rows[-1] if strict_rows else {}, t0_aggregate, first_step_rows)
    elif args.mode == "rung":
        ladder = aggregate_quadrature_ladder(args, rung_rows, rung_refs)
        gates = {
            key: bool(ladder.get(key))
            for key in (
                "G_C_monotone_matched_reference",
                "G_C_cauchy_decreasing",
                "G_C_rung_residual_comparability",
            )
        }
        aggregate = {"mode": "rung", "quadrature_ladder": ladder, "gates": gates, "all_checks_passed": all(gates.values())}
    elif args.mode == "thick":
        thick = aggregate_thick_limit(args, pair_rows, sn_refs_by_k, fld_refs_by_k)
        gate_keys = [key for key in thick if key.startswith("G_D_") and isinstance(thick[key], bool)]
        gates = {key: bool(thick[key]) for key in gate_keys}
        aggregate = {"mode": "thick", "thick_limit": thick, "gates": gates, "all_checks_passed": all(gates.values())}
    else:
        aggregate = aggregate_full(
            args,
            strict_rows,
            first_step_rows,
            sn_ref,
            t0_aggregate,
            ref_identity,
            rung_rows,
            rung_refs,
            pair_rows,
            sn_refs_by_k,
            fld_refs_by_k,
        )

    payload = summary_payload(
        args,
        ref_identity,
        t0_rows,
        t0_aggregate,
        first_step_rows,
        strict_rows,
        aggregate,
        rung_rows=rung_rows,
        pair_rows=pair_rows,
    )
    base.write_json(Path(args.summary_json), payload)
    print_summary(ref_identity, t0_aggregate, first_step_rows, aggregate)
    return 0 if bool(aggregate.get("all_checks_passed")) else 1


def summary_payload(
    args: argparse.Namespace,
    ref_identity: dict[str, Any],
    t0_rows: list[dict[str, Any]],
    t0_aggregate: dict[str, Any],
    first_step_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
    *,
    rung_rows: dict[int, dict[str, Any]] | None = None,
    pair_rows: dict[int, dict[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "tenryu.i3_sn_radshock_2d_rz_slab.summary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "deck_path": args.deck,
        "mode": args.mode,
        "grids": [{"nr": nr, "nz": nz} for nr, nz in args.grids],
        "rungs": list(args.rungs),
        "kappa_scales": list(args.kappa_scales),
        "init_mode": args.init_mode,
        "boundary_mode": args.boundary_mode,
        "frozen_prediction_refs": FROZEN_PREDICTION_REFS,
        "legs_filter": args.legs,
        "aggregate_skipped": legs_filter_active(args),
        "reference_identity_branch_wise": ref_identity,
        "t0_admissibility": {"rows": t0_rows, "aggregate": t0_aggregate},
        "first_step_audit": {"rows": first_step_rows},
        "matrix": rows,
        "rung_matrix": {str(key): value for key, value in (rung_rows or {}).items()},
        "thick_matrix": {
            str(k): {mode: row for mode, row in modes.items()}
            for k, modes in (pair_rows or {}).items()
        },
        "gates": aggregate.get("gates", {}),
        "aggregate": aggregate,
    }


def print_summary(
    ref_identity: dict[str, Any],
    t0_aggregate: dict[str, Any],
    first_step_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> None:
    print(
        json.dumps(
            {
                "ladder_row": LADDER_ROW,
                "frozen_prediction_refs": FROZEN_PREDICTION_REFS,
                "reference_identity_branch_wise": ref_identity,
                "t0_admissibility": t0_aggregate,
                "first_step_audit": first_step_rows,
                "aggregate": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
