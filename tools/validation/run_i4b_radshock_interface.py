#!/usr/bin/env python3
"""I4b 2D RZ radiative-shock material-interface validation harness.

Tiers:
  smoke (dev-time, default): one deck leg with kinematic/interface smoke gates,
    per-material mass leakage, operator-ledger closure, and CG health checks.
    The shock/interface estimators use REAL mesh coordinates from mesh/x_z
    because the mesh advects by tens of cm under Lagrangian-dominant motion.
  cert (campaign): legs A+B across a resolution pair, LE08 anchor, and
    shock-windowed L2 comparison against a fine reference; see
    docs/design/i4_mm_rad_interface_spec.md Addendum 6.
"""

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

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import run_i1_le08 as base  # noqa: E402


LADDER_ROW = "i4b_radshock_interface_2d_rz_slab"
DECK_ECHO_PREFIX = "[deck:i4b_radshock_interface_2d_rz_slab]"
DECK_ECHO_KEYS = (
    "reference",
    "orientation",
    "shock_z_cm",
    "z_int_cm",
    "t_cross_int_s",
    "t_end_s",
    "u0_cm_per_s",
    "L_ref_cm",
    "dz_cm",
    "mat1_kappa_cm2_g",
)
# Pre-crossing the two-state frame pins the shock (measured drift ~0.7 dz);
# post-crossing the front relaxes toward the MAT2 structure and legitimately
# displaces O(0.1 L_ref) — the tight matched-time position gate is cert-tier.
SMOKE_SHOCK_PRECROSS_TOL_DZ = 2.0
SMOKE_SHOCK_FINAL_TOL_FRAC = 0.15
SMOKE_IFACE_TOL_DZ = 2.0
SMOKE_IFACE_PRECROSS_TOL_DZ = 2.0
SMOKE_LEAKAGE_MAX = 1.0e-10
SMOKE_LEDGER_EPS_MAX = 1.0e-6
SMOKE_SHOCK_SEARCH_FRAC = 0.25
SMOKE_LEDGER_SKIP_T_FRAC = 0.1
SMOKE_CG_MAX_ITER_WARNINGS_MAX = 2
SMOKE_CG_STARTUP_LINE_MAX = 200
TINY = 1.0e-300
CERT_TIME_MATCH_T_END_FRAC = 0.02
CERT_SHOCK_WINDOW_DZ_COARSE = 10.0
CERT_SHOCK_L2_MAX = 0.12
CERT_LE08_ANCHOR_L2_MAX = 0.12
CERT_LE08_WINDOW_UPSTREAM_FRAC = 0.4
CERT_LE08_WINDOW_DOWNSTREAM_FRAC = 0.1


@dataclass(frozen=True)
class I4bSnapshot:
    path: Path
    t: float
    step: int
    rho_z: Any
    qvisc_z: Any
    pressure_z: Any
    Te_z: Any
    rad_E_z: Any
    vol_frac_z: Any
    vol_frac_min: float
    vol_frac_max: float
    z_edges: Any
    zc: Any
    mass_per_material: Any | None


def default_workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "tmp" / "i4b_radshock_interface" / stamp


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_info(outdir: Path) -> dict[str, Any]:
    path = base.discover_actual_outdir(outdir) / "run_info.json"
    if not path.exists():
        return {}
    try:
        return base.read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def shock_face_z(rho_z: Any, z_edges: Any, center_z: float | None = None,
                 half_width: float | None = None) -> dict[str, float]:
    """Interior face with the max |rho jump|. With center_z/half_width given, the
    argmax is restricted to faces within the window (the closed box grows wall
    pile-up fronts with bigger jumps than the primary shock); the unrestricted
    argmax is also returned as a diagnostic."""
    rho = np.asarray(rho_z, dtype=float)
    edges = np.asarray(z_edges, dtype=float)
    d = np.abs(np.diff(rho))
    face_z = edges[1:-1]
    j_global = int(np.argmax(d))
    result = {"global_z": float(face_z[j_global])}
    if center_z is not None and half_width is not None:
        mask = np.abs(face_z - center_z) <= half_width
        if not mask.any():
            raise RuntimeError("shock search window contains no faces")
        d_win = np.where(mask, d, -1.0)
        j = int(np.argmax(d_win))
        result["windowed_z"] = float(face_z[j])
    return result


def shock_qvisc_z(qvisc_z: Any, zc: Any, center_z: float, half_width: float) -> dict[str, float]:
    """Primary-shock locator: cell-center z of the max artificial viscosity within
    the window. Qvisc localizes at the compression front and is ~zero in advected
    entropy waves, at the material interface, and in the radiative precursor."""
    q = np.asarray(qvisc_z, dtype=float)
    z = np.asarray(zc, dtype=float)
    mask = np.abs(z - center_z) <= half_width
    if not mask.any():
        raise RuntimeError("shock search window contains no cells")
    q_win = np.where(mask, q, -np.inf)
    j = int(np.argmax(q_win))
    return {"z": float(z[j]), "qvisc_peak": float(q[j])}


def interface_z(vf_m2_z: Any, zc: Any) -> float | None:
    """Subcell-interpolated z where the MAT2 volume fraction crosses 0.5,
    scanning from the MAT2-dominant end."""
    vf_m2_z = np.asarray(vf_m2_z, dtype=float)
    zc = np.asarray(zc, dtype=float)
    above = vf_m2_z > 0.5
    if not above.any() or above.all():
        return None
    # MAT2 occupies a contiguous block starting at one end; find the last index of
    # that block and interpolate to the neighbor.
    if above[0]:
        j = int(np.max(np.nonzero(above)[0]))
        if j + 1 >= vf_m2_z.size:
            return float(zc[j])
        f0, f1 = vf_m2_z[j], vf_m2_z[j + 1]
    else:
        j = int(np.min(np.nonzero(above)[0]))
        if j - 1 < 0:
            return float(zc[j])
        f0, f1 = vf_m2_z[j], vf_m2_z[j - 1]
        return float(zc[j] + (zc[j - 1] - zc[j]) * (0.5 - f0) / (f1 - f0 + 1e-300))
    return float(zc[j] + (zc[j + 1] - zc[j]) * (0.5 - f0) / (f1 - f0 + 1e-300))


def predicted_interface_z(
    t: float,
    z_int0: float,
    shock_z: float,
    u0_abs: float,
    t_cross: float,
    u1_abs: float,
    orientation: str,
) -> float:
    sign = 1.0 if orientation == "upstream_low_z" else -1.0
    if t <= t_cross:
        return z_int0 + sign * u0_abs * t
    return shock_z + sign * u1_abs * (t - t_cross)


def _reshape_cells(values: Any, nr: int, nz: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (nr, nz):
        return arr
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    raise ValueError(f"{name}: expected shape ({nr}, {nz}) or size {nr * nz}, got {arr.shape}")


def _reshape_vol_frac(values: Any, nr: int, nz: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_cells = nr * nz
    if arr.size % n_cells != 0:
        raise ValueError(f"hydro/volFrac: size {arr.size} is not divisible by {n_cells}")
    n_mat = arr.size // n_cells
    if n_mat <= 0:
        raise ValueError("hydro/volFrac: material count must be positive")
    if arr.shape == (nr, nz, n_mat):
        return arr
    return arr.reshape((nr, nz, n_mat))


def _strictly_increasing(values: Any) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(arr.size >= 2 and np.all(np.diff(arr) > 0.0))


def _reshape_z_nodes(values: Any, nr: int, nz: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    expected = (nr + 1) * (nz + 1)
    if arr.size != expected:
        raise ValueError(f"mesh/x_z: expected size {expected}, got shape {arr.shape}")
    first = arr.reshape((nr + 1, nz + 1))
    if _strictly_increasing(first[0, :]):
        return first
    fallback = arr.reshape((nz + 1, nr + 1)).T
    if _strictly_increasing(fallback[0, :]):
        return fallback
    raise ValueError("mesh/x_z: row-0 z edges are not strictly increasing")


def _read_mass_per_material(
    handle: Any,
    nr: int,
    nz: int,
    n_mat: int,
) -> np.ndarray | None:
    path = "hydro/per_material/v1/mass"
    if path not in handle:
        return None
    n_cells = nr * nz
    arr = np.asarray(handle[path][()], dtype=float)
    if arr.shape == (n_cells, n_mat):
        return arr
    if arr.size == n_cells * n_mat:
        return arr.reshape((n_cells, n_mat))
    raise ValueError(f"{path}: expected shape ({n_cells}, {n_mat}), got {arr.shape}")


def read_snapshot(path: Path, nr: int, nz: int) -> I4bSnapshot:
    import h5py

    with h5py.File(path, "r") as handle:
        t_value = base.scalar_dataset(handle, "time_state/t", float(handle.attrs.get("t", 0.0)))
        step = int(round(base.scalar_dataset(handle, "time_state/step", 0.0)))
        rho = _reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho")
        qvisc = _reshape_cells(handle["hydro/Qvisc"][()], nr, nz, "hydro/Qvisc")
        Pe = _reshape_cells(handle["hydro/Pe"][()], nr, nz, "hydro/Pe")
        Pi = _reshape_cells(handle["hydro/Pi"][()], nr, nz, "hydro/Pi")
        Te = _reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te")
        rad_E = base.reshape_rad(handle["radiation/energy_density"][()], nr, nz, "2D_RZ")
        vol_frac = _reshape_vol_frac(handle["hydro/volFrac"][()], nr, nz)
        z_nodes = _reshape_z_nodes(handle["mesh/x_z"][()], nr, nz)
        z_edges = np.asarray(z_nodes[0, :], dtype=float)
        if not _strictly_increasing(z_edges):
            raise ValueError("mesh/x_z: row-0 z edges are not strictly increasing")
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])
        n_mat = int(vol_frac.shape[2])
        mass_per_material = _read_mass_per_material(handle, nr, nz, n_mat)
        return I4bSnapshot(
            path=path,
            t=float(t_value),
            step=step,
            rho_z=np.mean(rho, axis=0),
            qvisc_z=np.mean(qvisc, axis=0),
            pressure_z=np.mean(Pe + Pi, axis=0),
            Te_z=np.mean(Te, axis=0),
            rad_E_z=np.mean(rad_E, axis=0),
            vol_frac_z=np.mean(vol_frac, axis=0),
            vol_frac_min=float(np.min(vol_frac)),
            vol_frac_max=float(np.max(vol_frac)),
            z_edges=z_edges,
            zc=zc,
            mass_per_material=mass_per_material,
        )


def read_snapshots(outdir: Path, nr: int, nz: int) -> list[I4bSnapshot]:
    actual = base.discover_actual_outdir(outdir)
    results_dir = actual / "results"
    candidates = sorted(
        path for path in results_dir.glob("*.h5") if "history" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"no non-history HDF5 snapshots under {results_dir}")
    snapshots = [read_snapshot(path, nr, nz) for path in candidates]
    return sorted(snapshots, key=lambda snap: (snap.t, snap.step, str(snap.path)))


def parse_ledger_eps(log_path: Path) -> dict[str, Any]:
    """All [wj_step_audit] (t, eps) pairs; gate value = max eps over t >=
    SMOKE_LEDGER_SKIP_T_FRAC * t_final (skips the startup ramp AND survives the
    degenerate final dt-remainder step which reports eps=0)."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"eps_max_late": None, "eps_final": None, "n_lines": 0}
    ts: list[float] = []
    epss: list[float] = []
    for line in text.splitlines():
        if "[wj_step_audit]" not in line:
            continue
        t_val = eps_val = None
        for token in line.split():
            if token.startswith("t="):
                try: t_val = float(token[2:])
                except ValueError: pass
            elif token.startswith("eps="):
                try: eps_val = float(token[4:])
                except ValueError: pass
        if t_val is not None and eps_val is not None:
            ts.append(t_val); epss.append(eps_val)
    if not ts:
        return {"eps_max_late": None, "eps_final": None, "n_lines": 0}
    t_final = max(ts)
    late = [e for t, e in zip(ts, epss) if t >= SMOKE_LEDGER_SKIP_T_FRAC * t_final]
    return {
        "eps_max_late": max(late) if late else None,
        "eps_final": epss[-1],
        "n_lines": len(ts),
    }


def parse_deck_echo(log_path: Path) -> dict[str, Any]:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FileNotFoundError(f"could not read deck log {log_path}") from exc
    parsed: dict[str, str] | None = None
    for line in text.splitlines():
        if not line.startswith(DECK_ECHO_PREFIX):
            continue
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            fields[key] = value
        parsed = fields
        break
    if parsed is None:
        raise RuntimeError(f"missing {DECK_ECHO_PREFIX} echo line in {log_path}")
    missing = [key for key in DECK_ECHO_KEYS if key not in parsed]
    if missing:
        raise RuntimeError(f"deck echo missing keys: {', '.join(missing)}")
    orientation = parsed["orientation"]
    if orientation not in ("upstream_low_z", "upstream_high_z"):
        raise RuntimeError(f"deck echo has unsupported orientation={orientation!r}")
    return {
        "reference": parsed["reference"],
        "orientation": orientation,
        "shock_z_cm": float(parsed["shock_z_cm"]),
        "z_int_cm": float(parsed["z_int_cm"]),
        "t_cross_int_s": float(parsed["t_cross_int_s"]),
        "t_end_s": float(parsed["t_end_s"]),
        "u0_cm_per_s": float(parsed["u0_cm_per_s"]),
        "L_ref_cm": float(parsed["L_ref_cm"]),
        "dz_cm": float(parsed["dz_cm"]),
        "mat1_kappa_cm2_g": float(parsed["mat1_kappa_cm2_g"]),
    }


def cg_health(log_path: Path) -> dict[str, Any]:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    warning_lines = [
        idx for idx, line in enumerate(lines, start=1) if "reached max_iter" in line
    ]
    late_warning_lines = [
        idx for idx in warning_lines if idx > SMOKE_CG_STARTUP_LINE_MAX
    ]
    return {
        "max_iter_warning_count": len(warning_lines),
        "max_iter_warning_lines": warning_lines,
        "late_max_iter_warning_count": len(late_warning_lines),
        "late_max_iter_warning_lines": late_warning_lines,
        "warning_count_max": SMOKE_CG_MAX_ITER_WARNINGS_MAX,
        "startup_line_max": SMOKE_CG_STARTUP_LINE_MAX,
    }


def mat2_column_index(first: I4bSnapshot, orientation: str) -> int:
    vf = np.asarray(first.vol_frac_z, dtype=float)
    sample_count = min(5, vf.shape[0])
    sample = vf[:sample_count, :] if orientation == "upstream_low_z" else vf[-sample_count:, :]
    dominant = np.mean(sample, axis=0) > 0.5
    candidates = np.nonzero(dominant)[0]
    if candidates.size != 1:
        raise RuntimeError(
            f"expected exactly one MAT2 volFrac column at the {orientation} end, got {candidates.size}"
        )
    return int(candidates[0])


def mass_leakage(first: I4bSnapshot, final: I4bSnapshot) -> dict[str, Any]:
    if first.mass_per_material is None or final.mass_per_material is None:
        return {
            "status": "missing",
            "relative": None,
            "reason": "per-material mass dataset missing",
        }
    initial = np.sum(np.asarray(first.mass_per_material, dtype=float), axis=0)
    ending = np.sum(np.asarray(final.mass_per_material, dtype=float), axis=0)
    rel = np.abs(ending - initial) / np.maximum(initial, TINY)
    return {
        "status": "available",
        "relative": float(np.max(rel)) if rel.size else 0.0,
        "initial_g": [float(v) for v in initial],
        "final_g": [float(v) for v in ending],
    }


def smoke_metrics(
    deck: dict[str, Any],
    echo: dict[str, Any],
    reference: dict[str, Any],
    snapshots: list[I4bSnapshot],
    deck_log: Path,
    leg: str,
) -> dict[str, Any]:
    first = snapshots[0]
    final = snapshots[-1]
    u1_abs = abs(float(reference["states"]["downstream"]["u"]))
    u0_abs = abs(float(echo["u0_cm_per_s"]))
    mat2_idx = mat2_column_index(first, str(echo["orientation"]))
    precross_cutoff_s = 0.9 * float(echo["t_cross_int_s"])
    precross_candidates = [
        snap for snap in snapshots if float(snap.t) <= precross_cutoff_s
    ]
    precross_fallback = False
    if precross_candidates:
        precross = max(precross_candidates, key=lambda snap: float(snap.t))
    else:
        precross = min(
            (snap for snap in snapshots if float(snap.t) > 0.0),
            key=lambda snap: float(snap.t),
        )
        precross_fallback = True
    shock_rho_precross = shock_face_z(
        precross.rho_z,
        precross.z_edges,
        center_z=echo["shock_z_cm"],
        half_width=SMOKE_SHOCK_SEARCH_FRAC * echo["L_ref_cm"],
    )
    shock_rho_final = shock_face_z(
        final.rho_z,
        final.z_edges,
        center_z=echo["shock_z_cm"],
        half_width=SMOKE_SHOCK_SEARCH_FRAC * echo["L_ref_cm"],
    )
    shock_precross = shock_qvisc_z(
        precross.qvisc_z,
        precross.zc,
        echo["shock_z_cm"],
        SMOKE_SHOCK_SEARCH_FRAC * echo["L_ref_cm"],
    )
    shock = shock_qvisc_z(
        final.qvisc_z,
        final.zc,
        echo["shock_z_cm"],
        SMOKE_SHOCK_SEARCH_FRAC * echo["L_ref_cm"],
    )
    interface_precross = interface_z(np.asarray(precross.vol_frac_z)[:, mat2_idx], precross.zc)
    interface_precross_pred = predicted_interface_z(
        float(precross.t),
        float(echo["z_int_cm"]),
        float(echo["shock_z_cm"]),
        u0_abs,
        float(echo["t_cross_int_s"]),
        u1_abs,
        str(echo["orientation"]),
    )
    interface_precross_error = (
        None if interface_precross is None else abs(float(interface_precross) - interface_precross_pred)
    )
    interface_final = interface_z(np.asarray(final.vol_frac_z)[:, mat2_idx], final.zc)
    interface_pred = None
    interface_error = None
    interface_tolerance = None
    interface_crossed = None
    interface_excursion = None
    interface_bound = None
    if leg == "A":
        interface_pred = predicted_interface_z(
            float(final.t),
            float(echo["z_int_cm"]),
            float(echo["shock_z_cm"]),
            u0_abs,
            float(echo["t_cross_int_s"]),
            u1_abs,
            str(echo["orientation"]),
        )
        interface_error = (
            None if interface_final is None else abs(float(interface_final) - interface_pred)
        )
        interface_tolerance = SMOKE_IFACE_TOL_DZ * float(echo["dz_cm"])
    elif leg == "B":
        sign = 1.0 if str(echo["orientation"]) == "upstream_low_z" else -1.0
        if interface_final is not None:
            interface_crossed = bool(sign * (float(interface_final) - float(echo["shock_z_cm"])) >= 0.0)
            interface_excursion = abs(float(interface_final) - float(echo["shock_z_cm"]))
        else:
            interface_crossed = False
        interface_bound = u0_abs * max(0.0, float(final.t) - float(echo["t_cross_int_s"]))
    vof_min = float(min(snap.vol_frac_min for snap in snapshots))
    vof_max = float(max(snap.vol_frac_max for snap in snapshots))
    mass = mass_leakage(first, final)
    cg = cg_health(deck_log)
    ledger = parse_ledger_eps(deck_log)
    metrics = {
        "tier": "dev_smoke",
        "leg": leg,
        "run_completed": bool(deck["returncode"] == 0 and deck["t_end_reached"]),
        "shock_rho_face_precross_cm": shock_rho_precross["windowed_z"],
        "shock_rho_face_final_cm": shock_rho_final["windowed_z"],
        "shock_rho_face_global_cm": shock_rho_final["global_z"],
        "shock_reference_cm": float(echo["shock_z_cm"]),
        "shock_precross_t_s": float(precross.t),
        "shock_precross_err_cm": abs(shock_precross["z"] - float(echo["shock_z_cm"])),
        "shock_precross_tol_cm": SMOKE_SHOCK_PRECROSS_TOL_DZ * float(echo["dz_cm"]),
        "precross_fallback": precross_fallback,
        "shock_final_err_cm": abs(shock["z"] - float(echo["shock_z_cm"])),
        "shock_final_tol_cm": SMOKE_SHOCK_FINAL_TOL_FRAC * float(echo["L_ref_cm"]),
        "qvisc_peak_precross": shock_precross["qvisc_peak"],
        "qvisc_peak_final": shock["qvisc_peak"],
        "interface_precross_error_cm": interface_precross_error,
        "interface_precross_tolerance_cm": SMOKE_IFACE_PRECROSS_TOL_DZ * float(echo["dz_cm"]),
        "interface_final_cm": interface_final,
        "interface_predicted_cm": interface_pred,
        "interface_error_cm": interface_error,
        "interface_tolerance_cm": interface_tolerance,
        "mat2_column_index": mat2_idx,
        "vof_min": vof_min,
        "vof_max": vof_max,
        "vof_min_allowed": -1.0e-9,
        "vof_max_allowed": 1.0 + 1.0e-9,
        "mass_leakage": mass,
        "mass_leakage_max": SMOKE_LEAKAGE_MAX,
        "ledger_eps_max_late": ledger["eps_max_late"],
        "ledger_eps_final": ledger["eps_final"],
        "ledger_n_lines": ledger["n_lines"],
        "ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
        "cg_health": cg,
        "t_final_s": float(final.t),
        "final_step": int(final.step),
        "snapshot_count": len(snapshots),
        "u1_cm_per_s": float(reference["states"]["downstream"]["u"]),
    }
    if leg == "B":
        metrics.update(
            {
                "interface_crossed": interface_crossed,
                "interface_excursion_cm": interface_excursion,
                "interface_bound_cm": interface_bound,
            }
        )
    return metrics


def incomplete_run_metrics(deck: dict[str, Any]) -> dict[str, Any]:
    return {
        "tier": "dev_smoke",
        "run_completed": False,
        "deck_returncode": int(deck["returncode"]),
        "termination_reason": str(deck["termination_reason"]),
        "t_end_reached": bool(deck["t_end_reached"]),
    }


def failed_smoke_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if not metrics.get("run_completed", False):
        failed.append("run_completed")
        return failed
    if float(metrics["shock_precross_err_cm"]) > float(metrics["shock_precross_tol_cm"]):
        failed.append("shock_precross_stationary")
    if float(metrics["shock_final_err_cm"]) > float(metrics["shock_final_tol_cm"]):
        failed.append("shock_final_sanity")
    iface_precross_err = metrics["interface_precross_error_cm"]
    if iface_precross_err is None or float(iface_precross_err) > float(metrics["interface_precross_tolerance_cm"]):
        failed.append("interface_precross_tracks")
    iface_err = metrics["interface_error_cm"]
    if iface_err is not None and float(iface_err) > float(metrics["interface_tolerance_cm"]):
        failed.append("interface_tracks")
    if metrics.get("leg") == "B":
        iface_excursion = metrics.get("interface_excursion_cm")
        iface_bound = metrics.get("interface_bound_cm")
        if (
            not bool(metrics.get("interface_crossed", False))
            or iface_excursion is None
            or iface_bound is None
            or float(iface_excursion) > float(iface_bound)
        ):
            failed.append("interface_crossed_and_bounded")
    if float(metrics["vof_min"]) < float(metrics["vof_min_allowed"]) or float(metrics["vof_max"]) > float(metrics["vof_max_allowed"]):
        failed.append("vof_in_range")
    mass = metrics["mass_leakage"]
    if mass["status"] != "available":
        failed.append("mass_leakage")
    elif float(mass["relative"]) > SMOKE_LEAKAGE_MAX:
        failed.append("mass_leakage")
    ledger_eps = metrics["ledger_eps_max_late"]
    if ledger_eps is None or not math.isfinite(float(ledger_eps)) or float(ledger_eps) > SMOKE_LEDGER_EPS_MAX:
        failed.append("ledger_closure")
    cg = metrics["cg_health"]
    if int(cg["max_iter_warning_count"]) > SMOKE_CG_MAX_ITER_WARNINGS_MAX:
        failed.append("cg_health")
    elif int(cg["late_max_iter_warning_count"]) > 0:
        failed.append("cg_health")
    return failed


def format_float(value: Any) -> str:
    if value == "not_available":
        return "not_available"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{fvalue:.6e}" if math.isfinite(fvalue) else "missing"


def print_leg_line(leg: str, metrics: dict[str, Any], verdict: str) -> None:
    if not metrics.get("run_completed", False):
        print(
            "[i4b] "
            f"leg={leg} tier=smoke "
            f"run_completed=False "
            f"deck_rc={metrics.get('deck_returncode', 'missing')} "
            f"termination={metrics.get('termination_reason', 'missing')} "
            f"verdict={verdict}"
        )
        return
    mass = metrics["mass_leakage"]
    mass_value: Any = mass["relative"] if mass["status"] == "available" else "not_available"
    print(
        "[i4b] "
        f"leg={leg} tier=smoke "
        f"run_completed={metrics['run_completed']} "
        f"shock_pre_err_cm={metrics['shock_precross_err_cm']:.6e} "
        f"shock_final_err_cm={metrics['shock_final_err_cm']:.6e} "
        f"shock_rho_global_cm={format_float(metrics['shock_rho_face_global_cm'])} "
        f"iface_pre_err_cm={format_float(metrics['interface_precross_error_cm'])} "
        f"iface_err_cm={format_float(metrics['interface_error_cm'])} "
        f"iface_tol_cm={format_float(metrics['interface_tolerance_cm'])} "
        f"iface_excursion_cm={format_float(metrics.get('interface_excursion_cm'))} "
        f"vof_min={metrics['vof_min']:.6e} "
        f"vof_max={metrics['vof_max']:.6e} "
        f"mass_leakage={format_float(mass_value)} "
        f"ledger_eps_late={format_float(metrics['ledger_eps_max_late'])} "
        f"cg_max_iter_count={metrics['cg_health']['max_iter_warning_count']} "
        f"cg_late_count={metrics['cg_health']['late_max_iter_warning_count']} "
        f"verdict={verdict}"
    )


def cert_gate(
    name: str,
    formula: str,
    passed: bool,
    value: Any = None,
    limit: Any = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "formula": formula,
        "value": value,
        "limit": limit,
        "passed": bool(passed),
        "details": {} if details is None else details,
    }


def finite_leq(value: Any, limit: Any) -> bool:
    try:
        fvalue = float(value)
        flimit = float(limit)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fvalue) and math.isfinite(flimit) and fvalue <= flimit


def failed_cert_gates(gate_table: list[dict[str, Any]]) -> list[str]:
    return [str(gate["name"]) for gate in gate_table if not bool(gate.get("passed", False))]


def nearest_snapshot(snapshots: list[I4bSnapshot], target_t: float) -> I4bSnapshot:
    if not snapshots:
        raise RuntimeError("snapshot list is empty")
    return min(snapshots, key=lambda snap: abs(float(snap.t) - target_t))


def matched_last_common_snapshot(
    coarse: list[I4bSnapshot],
    fine: list[I4bSnapshot],
    t_end: float,
) -> dict[str, Any]:
    target_t = min(float(coarse[-1].t), float(fine[-1].t), float(t_end))
    coarse_snap = nearest_snapshot(coarse, target_t)
    fine_snap = nearest_snapshot(fine, target_t)
    dt_abs = abs(float(coarse_snap.t) - float(fine_snap.t))
    dt_limit = CERT_TIME_MATCH_T_END_FRAC * float(t_end)
    return {
        "target_t_s": target_t,
        "coarse": coarse_snap,
        "fine": fine_snap,
        "coarse_t_s": float(coarse_snap.t),
        "fine_t_s": float(fine_snap.t),
        "dt_abs_s": dt_abs,
        "dt_limit_s": dt_limit,
        "passed": dt_abs <= dt_limit,
    }


def cert_rho_shock_z(snapshot: I4bSnapshot, echo: dict[str, Any]) -> dict[str, float]:
    return shock_face_z(
        snapshot.rho_z,
        snapshot.z_edges,
        center_z=float(echo["shock_z_cm"]),
        half_width=SMOKE_SHOCK_SEARCH_FRAC * float(echo["L_ref_cm"]),
    )


def cert_pressure_shock_z(snapshot: I4bSnapshot, echo: dict[str, Any], leg: str) -> dict[str, float]:
    p = np.asarray(snapshot.pressure_z, dtype=float)
    edges = np.asarray(snapshot.z_edges, dtype=float)
    d = np.abs(np.diff(p))
    face_z = edges[1:-1]
    if d.size != face_z.size or d.size == 0:
        raise RuntimeError("pressure shock search has no interior faces")
    j_global = int(np.argmax(d))
    l_ref = float(echo["L_ref_cm"])
    if leg == "B":
        window_lo = float(echo["z_int_cm"]) - 0.1 * l_ref
        window_hi = float(echo["shock_z_cm"]) + SMOKE_SHOCK_SEARCH_FRAC * l_ref
    else:
        window_lo = float(echo["shock_z_cm"]) - SMOKE_SHOCK_SEARCH_FRAC * l_ref
        window_hi = float(echo["shock_z_cm"]) + SMOKE_SHOCK_SEARCH_FRAC * l_ref
    mask = (face_z >= window_lo) & (face_z <= window_hi)
    if not mask.any():
        raise RuntimeError(
            "pressure shock search window contains no faces "
            f"[{window_lo:.6e}, {window_hi:.6e}]"
        )
    d_win = np.where(mask, d, -1.0)
    j = int(np.argmax(d_win))
    return {
        "global_z": float(face_z[j_global]),
        "windowed_z": float(face_z[j]),
        "window_lo_cm": window_lo,
        "window_hi_cm": window_hi,
    }


def max_in_z_window(values: Any, zc: Any, center_z: float, half_width: float) -> float:
    z = np.asarray(zc, dtype=float)
    arr = np.asarray(values, dtype=float)
    mask = np.abs(z - center_z) <= half_width
    if not mask.any():
        return math.nan
    return float(np.max(arr[mask]))


def l2_rel(values: Any, reference: Any) -> float:
    val = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    denom = float(np.linalg.norm(ref))
    if not math.isfinite(denom) or denom <= TINY:
        return math.inf
    num = float(np.linalg.norm(val - ref))
    return num / denom


def shock_windowed_l2(
    coarse: I4bSnapshot,
    fine: I4bSnapshot,
    z_shock_fine: float,
    dz_coarse: float,
) -> dict[str, Any]:
    fields = {
        "rho": (coarse.rho_z, fine.rho_z),
        "Te": (coarse.Te_z, fine.Te_z),
        "rad_E": (coarse.rad_E_z, fine.rad_E_z),
    }
    initial_half_width = CERT_SHOCK_WINDOW_DZ_COARSE * float(dz_coarse)
    coarse_z_raw = np.asarray(coarse.zc, dtype=float)
    fine_z_raw = np.asarray(fine.zc, dtype=float)
    coarse_order = np.argsort(coarse_z_raw)
    fine_order = np.argsort(fine_z_raw)
    coarse_z = coarse_z_raw[coarse_order]
    fine_z = fine_z_raw[fine_order]

    def l2_infra(reason: str, half_width: float, widened_once: bool) -> dict[str, Any]:
        window_lo = float(z_shock_fine) - half_width
        window_hi = float(z_shock_fine) + half_width
        if coarse_z.size > 0 and fine_z.size > 0:
            clip_lo: Any = max(window_lo, float(np.min(coarse_z)), float(np.min(fine_z)))
            clip_hi: Any = min(window_hi, float(np.max(coarse_z)), float(np.max(fine_z)))
            mask = (coarse_z >= clip_lo) & (coarse_z <= clip_hi)
            coarse_cell_count = int(np.count_nonzero(mask))
        else:
            clip_lo = "not_available"
            clip_hi = "not_available"
            coarse_cell_count = 0
        return {
            "status": "INFRA",
            "reason": reason,
            "center_z_cm": float(z_shock_fine),
            "initial_half_width_cm": initial_half_width,
            "half_width_cm": half_width,
            "widened_once": widened_once,
            "window_lo_cm": window_lo,
            "window_hi_cm": window_hi,
            "clipped_lo_cm": clip_lo,
            "clipped_hi_cm": clip_hi,
            "coarse_cell_count": coarse_cell_count,
            "fields": {name: "INFRA" for name in fields},
            "max_L2_rel": "INFRA",
        }

    if coarse_z.size == 0 or fine_z.size < 2:
        return l2_infra("insufficient z samples for shock-windowed interpolation", initial_half_width, False)

    half_width = initial_half_width
    widened_once = False
    window_lo = float(z_shock_fine) - half_width
    window_hi = float(z_shock_fine) + half_width
    clip_lo = max(window_lo, float(np.min(coarse_z)), float(np.min(fine_z)))
    clip_hi = min(window_hi, float(np.max(coarse_z)), float(np.max(fine_z)))
    mask = (coarse_z >= clip_lo) & (coarse_z <= clip_hi)
    if int(np.count_nonzero(mask)) < 4:
        half_width = 2.0 * initial_half_width
        widened_once = True
        window_lo = float(z_shock_fine) - half_width
        window_hi = float(z_shock_fine) + half_width
        clip_lo = max(window_lo, float(np.min(coarse_z)), float(np.min(fine_z)))
        clip_hi = min(window_hi, float(np.max(coarse_z)), float(np.max(fine_z)))
        mask = (coarse_z >= clip_lo) & (coarse_z <= clip_hi)
    coarse_cell_count = int(np.count_nonzero(mask))
    if coarse_cell_count < 4:
        return l2_infra(
            f"coarse shock-window cell count {coarse_cell_count} < 4 after one x2 widening",
            half_width,
            widened_once,
        )

    l2_by_field: dict[str, Any] = {}
    coarse_centers = coarse_z[mask]
    for name, (coarse_values, fine_values) in fields.items():
        coarse_profile = np.asarray(coarse_values, dtype=float)[coarse_order][mask]
        fine_profile = np.interp(
            coarse_centers,
            fine_z,
            np.asarray(fine_values, dtype=float)[fine_order],
        )
        field_l2 = l2_rel(coarse_profile, fine_profile)
        if not math.isfinite(field_l2):
            return l2_infra(f"non-finite L2 for field {name}", half_width, widened_once)
        l2_by_field[name] = field_l2
    max_l2 = max(float(value) for value in l2_by_field.values())
    return {
        "status": "OK",
        "center_z_cm": float(z_shock_fine),
        "initial_half_width_cm": initial_half_width,
        "half_width_cm": half_width,
        "widened_once": widened_once,
        "window_lo_cm": window_lo,
        "window_hi_cm": window_hi,
        "clipped_lo_cm": clip_lo,
        "clipped_hi_cm": clip_hi,
        "coarse_cell_count": coarse_cell_count,
        "fields": l2_by_field,
        "max_L2_rel": max_l2,
    }


def le08_anchor_metrics(
    echo: dict[str, Any],
    reference: dict[str, Any],
    snapshots: list[I4bSnapshot],
) -> dict[str, Any]:
    target_t = 0.5 * float(echo["t_cross_int_s"])
    snapshot = nearest_snapshot(snapshots, target_t)
    shock = cert_pressure_shock_z(snapshot, echo, "A")
    z_pressure_face = float(shock["windowed_z"])
    table = reference["table"]
    x_ref = np.asarray(table["x_cm"], dtype=float)
    Te_ref = np.asarray(table["T_eV"], dtype=float)
    rho_ref = np.asarray(table["rho_g_per_cc"], dtype=float)
    order = np.argsort(x_ref)
    x_ref = x_ref[order]
    Te_ref = Te_ref[order]
    rho_ref = rho_ref[order]
    span = float(x_ref[-1] - x_ref[0])
    rho_grad = np.gradient(rho_ref, x_ref)
    rho_face_idx = int(np.argmax(np.abs(rho_grad)))
    x_rho_face = float(x_ref[rho_face_idx])
    offset = z_pressure_face - x_rho_face
    table_z_lo = float(x_ref[0]) + offset
    table_z_hi = float(x_ref[-1]) + offset
    window_lo = max(z_pressure_face - CERT_LE08_WINDOW_UPSTREAM_FRAC * span, table_z_lo)
    window_hi = min(z_pressure_face + CERT_LE08_WINDOW_DOWNSTREAM_FRAC * span, table_z_hi)
    z = np.asarray(snapshot.zc, dtype=float)
    mask = (z >= window_lo) & (z <= window_hi)
    if mask.any():
        table_Te = np.interp(z[mask] - offset, x_ref, Te_ref)
        l2 = l2_rel(np.asarray(snapshot.Te_z, dtype=float)[mask], table_Te)
    else:
        l2 = math.inf
    return {
        "target_t_s": target_t,
        "snapshot_t_s": float(snapshot.t),
        "snapshot_step": int(snapshot.step),
        "snapshot_path": str(snapshot.path),
        "pressure_face_z_cm": z_pressure_face,
        "pressure_face_global_z_cm": float(shock["global_z"]),
        "pressure_search_window_lo_cm": float(shock["window_lo_cm"]),
        "pressure_search_window_hi_cm": float(shock["window_hi_cm"]),
        "table_rho_face_x_cm": x_rho_face,
        "table_rho_face_index": rho_face_idx,
        "offset_cm": offset,
        "table_span_cm": span,
        "window_lo_cm": window_lo,
        "window_hi_cm": window_hi,
        "cell_count": int(np.count_nonzero(mask)),
        "L2_rel": l2,
        "limit": CERT_LE08_ANCHOR_L2_MAX,
    }


def cert_run_qois(
    deck: dict[str, Any],
    echo: dict[str, Any],
    snapshots: list[I4bSnapshot],
    deck_log: Path,
    leg: str,
    label: str,
) -> dict[str, Any]:
    first = snapshots[0]
    final = snapshots[-1]
    mat2_idx = mat2_column_index(first, str(echo["orientation"]))
    shock = cert_pressure_shock_z(final, echo, leg)
    rho_shock = cert_rho_shock_z(final, echo)
    qvisc = shock_qvisc_z(
        final.qvisc_z,
        final.zc,
        float(echo["shock_z_cm"]),
        SMOKE_SHOCK_SEARCH_FRAC * float(echo["L_ref_cm"]),
    )
    interface_final = interface_z(np.asarray(final.vol_frac_z)[:, mat2_idx], final.zc)
    mass = mass_leakage(first, final)
    ledger = parse_ledger_eps(deck_log)
    return {
        "tier": "campaign_cert",
        "leg": leg,
        "label": label,
        "run_completed": bool(deck["returncode"] == 0 and deck["t_end_reached"]),
        "nz": int(deck["env"]["TENRYU_I4B_NZ"]),
        "nr": int(deck["env"]["TENRYU_I4B_NR"]),
        "dz_cm": float(echo["dz_cm"]),
        "t_end_s": float(echo["t_end_s"]),
        "t_final_s": float(final.t),
        "final_step": int(final.step),
        "snapshot_count": len(snapshots),
        "first_snapshot_path": str(first.path),
        "final_snapshot_path": str(final.path),
        "shock_pressure_face_final_cm": float(shock["windowed_z"]),
        "shock_pressure_face_global_final_cm": float(shock["global_z"]),
        "shock_pressure_search_window_lo_cm": float(shock["window_lo_cm"]),
        "shock_pressure_search_window_hi_cm": float(shock["window_hi_cm"]),
        "shock_rho_face_final_cm": float(rho_shock["windowed_z"]),
        "shock_rho_face_global_final_cm": float(rho_shock["global_z"]),
        "shock_qvisc_final_cm": float(qvisc["z"]),
        "qvisc_peak_final": float(qvisc["qvisc_peak"]),
        "interface_final_cm": interface_final,
        "mat2_column_index": mat2_idx,
        "vof_min": float(min(snap.vol_frac_min for snap in snapshots)),
        "vof_max": float(max(snap.vol_frac_max for snap in snapshots)),
        "vof_min_allowed": -1.0e-9,
        "vof_max_allowed": 1.0 + 1.0e-9,
        "mass_leakage": mass,
        "mass_leakage_max": SMOKE_LEAKAGE_MAX,
        "ledger_eps_max_late": ledger["eps_max_late"],
        "ledger_eps_final": ledger["eps_final"],
        "ledger_n_lines": ledger["n_lines"],
        "ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
    }


def append_cert_run_gates(
    gate_table: list[dict[str, Any]],
    label: str,
    run: dict[str, Any],
) -> None:
    complete = bool(run.get("deck", {}).get("returncode") == 0 and run.get("deck", {}).get("t_end_reached"))
    gate_table.append(
        cert_gate(
            f"{label}_run_completed",
            "returncode == 0 and termination_reason == t_end_reached",
            complete,
            value=complete,
            limit=True,
            details={
                "returncode": run.get("deck", {}).get("returncode"),
                "termination_reason": run.get("deck", {}).get("termination_reason"),
            },
        )
    )
    if not complete or "qois" not in run:
        return
    qois = run["qois"]
    mass = qois["mass_leakage"]
    mass_value = mass["relative"] if mass["status"] == "available" else None
    gate_table.append(
        cert_gate(
            f"{label}_mass_leakage",
            "max_m |M_m(final)-M_m(first)|/max(M_m(first), tiny) <= 1e-10",
            mass["status"] == "available" and finite_leq(mass_value, SMOKE_LEAKAGE_MAX),
            value=mass_value,
            limit=SMOKE_LEAKAGE_MAX,
            details=mass,
        )
    )
    vof_passed = (
        float(qois["vof_min"]) >= float(qois["vof_min_allowed"])
        and float(qois["vof_max"]) <= float(qois["vof_max_allowed"])
    )
    gate_table.append(
        cert_gate(
            f"{label}_vof_in_range",
            "min(volFrac) >= -1e-9 and max(volFrac) <= 1+1e-9",
            vof_passed,
            value={"min": qois["vof_min"], "max": qois["vof_max"]},
            limit={"min": qois["vof_min_allowed"], "max": qois["vof_max_allowed"]},
        )
    )
    gate_table.append(
        cert_gate(
            f"{label}_ledger_closure",
            "eps_max_late <= 1e-6",
            finite_leq(qois["ledger_eps_max_late"], SMOKE_LEDGER_EPS_MAX),
            value=qois["ledger_eps_max_late"],
            limit=SMOKE_LEDGER_EPS_MAX,
            details={
                "eps_final": qois["ledger_eps_final"],
                "n_lines": qois["ledger_n_lines"],
            },
        )
    )


def print_cert_leg_line(leg: str, result: dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    verdict = str(result.get("verdict", "FAIL"))
    if "shock_pos_err_cm" not in metrics:
        print(
            "[i4b] "
            f"leg={leg} tier=cert "
            f"failed_gates={','.join(result.get('failed_gates', []))} "
            f"verdict={verdict}"
        )
        return
    print(
        "[i4b] "
        f"leg={leg} tier=cert "
        f"match_dt_s={format_float(metrics['matched_time_dt_abs_s'])} "
        f"shock_err_cm={format_float(metrics['shock_pos_err_cm'])} "
        f"shock_gate_cm={format_float(metrics['shock_pos_gate_cm'])} "
        f"iface_err_cm={format_float(metrics['iface_pos_err_cm'])} "
        f"iface_gate_cm={format_float(metrics['iface_pos_gate_cm'])} "
        f"shock_L2_max={format_float(metrics['shock_windowed_L2_max'])} "
        f"le08_L2={format_float(metrics.get('le08_anchor_L2_rel'))} "
        f"failed_gates={','.join(result.get('failed_gates', []))} "
        f"verdict={verdict}"
    )


def run_deck(args: argparse.Namespace, outdir: Path, leg: str, nz: int | None = None) -> dict[str, Any]:
    run_nz = args.nz if nz is None else nz
    env_extra = {
        "TENRYU_I4B_LEG": leg,
        "TENRYU_I4B_NZ": str(run_nz),
        "TENRYU_I4B_NR": str(args.nr),
        "TENRYU_I4B_OUTDIR": str(outdir),
        "TENRYU_ENERGY_OPERATOR_AUDIT": "1",
        "TENRYU_ENERGY_OPERATOR_AUDIT_TMAX": "0",
    }
    if args.kappa_ratio is not None:
        env_extra["TENRYU_I4B_KAPPA_RATIO"] = f"{args.kappa_ratio:.17g}"
    env = dict(os.environ)
    env.update(env_extra)
    log_path = outdir.parent / "run.log"
    cmd = [str(args.binary), "run", str(args.deck)]
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - t0
    info = run_info(outdir)
    return {
        "returncode": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "cmd": cmd,
        "env": env_extra,
        "outdir": str(outdir),
        "actual_outdir": str(base.discover_actual_outdir(outdir)),
        "log": str(log_path),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "t_end_reached": str(info.get("termination_reason", "unknown")) == "t_end_reached",
    }


def run_leg(args: argparse.Namespace, leg: str) -> dict[str, Any]:
    leg_dir = Path(args.workdir) / f"leg{leg}"
    outdir = leg_dir / "out"
    leg_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"leg": leg, "label": f"leg{leg}"}

    deck = run_deck(args, outdir, leg)
    result["deck"] = deck
    run_completed = deck["returncode"] == 0 and deck["t_end_reached"]
    if not run_completed:
        metrics = incomplete_run_metrics(deck)
        gates = failed_smoke_gates(metrics)
        leg_verdict = "PASS" if not gates else "FAIL"
        print_leg_line(leg, metrics, leg_verdict)
        result.update(
            {
                "tier": "dev_smoke",
                "metrics": metrics,
                "failed_gates": gates,
                "smoke_gate_passed": len(gates) == 0,
                "verdict": leg_verdict,
            }
        )
        return result

    echo = parse_deck_echo(Path(deck["log"]))
    reference = base.read_json(Path(str(echo["reference"])))
    snapshots = read_snapshots(outdir, args.nr, args.nz)
    first = snapshots[0]
    final = snapshots[-1]
    metrics = smoke_metrics(deck, echo, reference, snapshots, Path(deck["log"]), leg)
    gates = failed_smoke_gates(metrics)
    leg_verdict = "PASS" if not gates else "FAIL"
    print_leg_line(leg, metrics, leg_verdict)

    result.update(
        {
            "first_snapshot_path": str(first.path),
            "final_snapshot_path": str(final.path),
            "deck_echo": echo,
            "reference_path": str(echo["reference"]),
            "tier": "dev_smoke",
            "metrics": metrics,
            "failed_gates": gates,
            "smoke_gate_passed": len(gates) == 0,
            "verdict": leg_verdict,
        }
    )
    return result


def run_cert_leg(args: argparse.Namespace, leg: str) -> dict[str, Any]:
    leg_dir = Path(args.workdir) / f"leg{leg}"
    leg_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"leg": leg, "tier": "campaign_cert"}
    gate_table: list[dict[str, Any]] = []

    for label, nz in (("coarse", args.nz), ("fine", args.fine_nz)):
        run_dir = leg_dir / label
        outdir = run_dir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        deck = run_deck(args, outdir, leg, nz=nz)
        run: dict[str, Any] = {
            "label": label,
            "tier": "campaign_cert",
            "leg": leg,
            "nz": int(nz),
            "deck": deck,
        }
        if deck["returncode"] == 0 and deck["t_end_reached"]:
            echo = parse_deck_echo(Path(deck["log"]))
            reference = base.read_json(Path(str(echo["reference"])))
            snapshots = read_snapshots(outdir, args.nr, int(nz))
            run.update(
                {
                    "deck_echo": echo,
                    "reference_path": str(echo["reference"]),
                    "qois": cert_run_qois(
                        deck,
                        echo,
                        snapshots,
                        Path(deck["log"]),
                        leg,
                        label,
                    ),
                    "snapshots": snapshots,
                    "snapshot_paths": [str(snap.path) for snap in snapshots],
                    "reference": reference,
                }
            )
        result[label] = run

    append_cert_run_gates(gate_table, "coarse", result["coarse"])
    append_cert_run_gates(gate_table, "fine", result["fine"])

    both_complete = all(
        bool(result[label]["deck"]["returncode"] == 0 and result[label]["deck"]["t_end_reached"])
        for label in ("coarse", "fine")
    )
    metrics: dict[str, Any] = {}
    if both_complete:
        coarse = result["coarse"]
        fine = result["fine"]
        coarse_echo = coarse["deck_echo"]
        fine_echo = fine["deck_echo"]
        coarse_snapshots = coarse["snapshots"]
        fine_snapshots = fine["snapshots"]
        dz_high = float(fine_echo["dz_cm"])
        dz_coarse = float(coarse_echo["dz_cm"])
        t_end = float(fine_echo["t_end_s"])

        matched = matched_last_common_snapshot(coarse_snapshots, fine_snapshots, t_end)
        coarse_matched = matched["coarse"]
        fine_matched = matched["fine"]
        gate_table.append(
            cert_gate(
                "matched_time",
                "|t_c - t_f| <= 0.02 * t_end",
                bool(matched["passed"]),
                value=matched["dt_abs_s"],
                limit=matched["dt_limit_s"],
                details={
                    "target_t_s": matched["target_t_s"],
                    "coarse_t_s": matched["coarse_t_s"],
                    "fine_t_s": matched["fine_t_s"],
                },
            )
        )

        coarse_shock = cert_pressure_shock_z(coarse_matched, coarse_echo, leg)
        fine_shock = cert_pressure_shock_z(fine_matched, fine_echo, leg)
        coarse_rho_shock = cert_rho_shock_z(coarse_matched, coarse_echo)
        fine_rho_shock = cert_rho_shock_z(fine_matched, fine_echo)
        z_shock_coarse = float(coarse_shock["windowed_z"])
        z_shock_fine = float(fine_shock["windowed_z"])
        shock_window_half = CERT_SHOCK_WINDOW_DZ_COARSE * dz_coarse
        rho_post = max_in_z_window(
            fine_matched.rho_z,
            fine_matched.zc,
            z_shock_fine,
            shock_window_half,
        )
        kappa_post = float(fine_echo["mat1_kappa_cm2_g"])
        lambda_mfp_post = (
            1.0 / (kappa_post * rho_post)
            if math.isfinite(rho_post) and rho_post > 0.0 and kappa_post > 0.0
            else math.nan
        )
        shock_gate_2dz = 2.0 * dz_high
        shock_gate_mfp = 0.25 * lambda_mfp_post if math.isfinite(lambda_mfp_post) else math.nan
        shock_gate_value = (
            max(shock_gate_2dz, shock_gate_mfp)
            if math.isfinite(shock_gate_mfp)
            else math.nan
        )
        shock_pos_err = abs(z_shock_coarse - z_shock_fine)
        gate_table.append(
            cert_gate(
                "shock_pos_err",
                "|z_pressure_shock_coarse(t)-z_pressure_shock_fine(t)| <= max(2*dz_high, 0.25/(mat1_kappa*rho_post))",
                finite_leq(shock_pos_err, shock_gate_value),
                value=shock_pos_err,
                limit=shock_gate_value,
                details={
                    "z_pressure_shock_coarse_cm": z_shock_coarse,
                    "z_pressure_shock_fine_cm": z_shock_fine,
                    "dz_high_cm": dz_high,
                    "dz_coarse_cm": dz_coarse,
                    "candidate_2dz_high_cm": shock_gate_2dz,
                    "candidate_quarter_lambda_mfp_post_cm": shock_gate_mfp,
                    "rho_post_gcc": rho_post,
                    "lambda_mfp_post_cm": lambda_mfp_post,
                    "kappa_post_cm2_g": kappa_post,
                    "fine_pressure_shock_global_cm": fine_shock["global_z"],
                    "coarse_pressure_shock_global_cm": coarse_shock["global_z"],
                    "fine_pressure_search_window_lo_cm": fine_shock["window_lo_cm"],
                    "fine_pressure_search_window_hi_cm": fine_shock["window_hi_cm"],
                    "coarse_pressure_search_window_lo_cm": coarse_shock["window_lo_cm"],
                    "coarse_pressure_search_window_hi_cm": coarse_shock["window_hi_cm"],
                    "fine_rho_shock_windowed_cm": fine_rho_shock["windowed_z"],
                    "coarse_rho_shock_windowed_cm": coarse_rho_shock["windowed_z"],
                    "fine_rho_shock_global_cm": fine_rho_shock["global_z"],
                    "coarse_rho_shock_global_cm": coarse_rho_shock["global_z"],
                },
            )
        )

        coarse_mat2 = mat2_column_index(coarse_snapshots[0], str(coarse_echo["orientation"]))
        fine_mat2 = mat2_column_index(fine_snapshots[0], str(fine_echo["orientation"]))
        iface_coarse = interface_z(
            np.asarray(coarse_matched.vol_frac_z)[:, coarse_mat2],
            coarse_matched.zc,
        )
        iface_fine = interface_z(
            np.asarray(fine_matched.vol_frac_z)[:, fine_mat2],
            fine_matched.zc,
        )
        iface_err = (
            None
            if iface_coarse is None or iface_fine is None
            else abs(float(iface_coarse) - float(iface_fine))
        )
        gate_table.append(
            cert_gate(
                "iface_pos_err",
                "|z_iface_coarse(t)-z_iface_fine(t)| <= dz_coarse",
                iface_err is not None and finite_leq(iface_err, dz_coarse),
                value=iface_err,
                limit=dz_coarse,
                details={
                    "z_iface_coarse_cm": iface_coarse,
                    "z_iface_fine_cm": iface_fine,
                    "dz_coarse_cm": dz_coarse,
                    "dz_high_cm": dz_high,
                },
            )
        )

        l2_metrics = shock_windowed_l2(
            coarse_matched,
            fine_matched,
            z_shock_fine,
            dz_coarse,
        )
        gate_table.append(
            cert_gate(
                "shock_windowed_L2",
                "max_field ||coarse-field - interp(fine-field)||_2 / ||interp(fine-field)||_2 <= 0.12 for rho, Te, rad_E on z_pressure_shock_fine +/- 10*dz_coarse",
                finite_leq(l2_metrics["max_L2_rel"], CERT_SHOCK_L2_MAX),
                value=l2_metrics["max_L2_rel"],
                limit=CERT_SHOCK_L2_MAX,
                details=l2_metrics,
            )
        )

        metrics.update(
            {
                "matched_time_target_s": matched["target_t_s"],
                "matched_time_coarse_s": matched["coarse_t_s"],
                "matched_time_fine_s": matched["fine_t_s"],
                "matched_time_dt_abs_s": matched["dt_abs_s"],
                "matched_time_dt_limit_s": matched["dt_limit_s"],
                "dz_high_cm": dz_high,
                "dz_coarse_cm": dz_coarse,
                "shock_pressure_coarse_cm": z_shock_coarse,
                "shock_pressure_fine_cm": z_shock_fine,
                "shock_rho_coarse_cm": float(coarse_rho_shock["windowed_z"]),
                "shock_rho_fine_cm": float(fine_rho_shock["windowed_z"]),
                "shock_rho_global_coarse_cm": float(coarse_rho_shock["global_z"]),
                "shock_rho_global_fine_cm": float(fine_rho_shock["global_z"]),
                "shock_pos_err_cm": shock_pos_err,
                "shock_pos_gate_cm": shock_gate_value,
                "shock_pos_gate_2dz_high_cm": shock_gate_2dz,
                "shock_pos_gate_quarter_lambda_mfp_post_cm": shock_gate_mfp,
                "rho_post_gcc": rho_post,
                "lambda_mfp_post_cm": lambda_mfp_post,
                "kappa_post_cm2_g": kappa_post,
                "iface_coarse_cm": iface_coarse,
                "iface_fine_cm": iface_fine,
                "iface_pos_err_cm": iface_err,
                "iface_pos_gate_cm": dz_coarse,
                "shock_windowed_L2": l2_metrics,
                "shock_windowed_L2_max": l2_metrics["max_L2_rel"],
            }
        )

        if leg == "A":
            anchor = le08_anchor_metrics(fine_echo, fine["reference"], fine_snapshots)
            gate_table.append(
                cert_gate(
                    "le08_anchor",
                    "rho-face-aligned ||Te_fine(z)-interp(LE08_Te, z-offset)||_2 / ||interp(LE08_Te)||_2 <= 0.12 at nearest 0.5*t_cross_int",
                    finite_leq(anchor["L2_rel"], CERT_LE08_ANCHOR_L2_MAX),
                    value=anchor["L2_rel"],
                    limit=CERT_LE08_ANCHOR_L2_MAX,
                    details=anchor,
                )
            )
            metrics.update(
                {
                    "le08_anchor_L2_rel": anchor["L2_rel"],
                    "le08_anchor_limit": CERT_LE08_ANCHOR_L2_MAX,
                    "le08_anchor_offset_cm": anchor["offset_cm"],
                    "le08_anchor": anchor,
                }
            )

    failed = failed_cert_gates(gate_table)
    leg_verdict = "PASS" if not failed else "FAIL"
    result.update(
        {
            "metrics": metrics,
            "gate_table": gate_table,
            "failed_gates": failed,
            "cert_gate_passed": len(failed) == 0,
            "verdict": leg_verdict,
        }
    )
    for label in ("coarse", "fine"):
        if "snapshots" in result[label]:
            del result[label]["snapshots"]
        if "reference" in result[label]:
            del result[label]["reference"]
    print_cert_leg_line(leg, result)
    return result


def cert_gate_formulas() -> dict[str, str]:
    return {
        "matched_time": "|t_c - t_f| <= 0.02 * t_end",
        "shock_pos_err": "|z_pressure_shock_coarse(t)-z_pressure_shock_fine(t)| <= max(2*dz_high, 0.25/(mat1_kappa*rho_post))",
        "iface_pos_err": "|z_iface_coarse(t)-z_iface_fine(t)| <= dz_coarse",
        "shock_windowed_L2": "max_{rho,Te,rad_E} ||coarse - interp(fine)||_2 / ||interp(fine)||_2 <= 0.12 on z_pressure_shock_fine +/- 10*dz_coarse",
        "mass_leakage": "max_m |M_m(final)-M_m(first)|/max(M_m(first), tiny) <= 1e-10 for coarse and fine",
        "vof_in_range": "min(volFrac) >= -1e-9 and max(volFrac) <= 1+1e-9 for coarse and fine",
        "ledger_closure": "eps_max_late <= 1e-6 for coarse and fine",
        "le08_anchor": "leg A only: rho-face-aligned ||Te_fine - interp(LE08_Te)||_2 / ||interp(LE08_Te)||_2 <= 0.12 at nearest 0.5*t_cross_int",
    }


def run_cert_campaign(args: argparse.Namespace) -> int:
    legs = ["A", "B"]
    summary_path = Path(args.workdir) / "summary.json"
    runs: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": "cert",
        "tier": "campaign_cert",
        "binary": str(args.binary),
        "deck": str(args.deck),
        "workdir": str(args.workdir),
        "legs": legs,
        "nr": int(args.nr),
        "coarse_nz": int(args.nz),
        "fine_nz": int(args.fine_nz),
        "kappa_ratio": None if args.kappa_ratio is None else float(args.kappa_ratio),
        "deck_echo_keys": list(DECK_ECHO_KEYS),
        "fine_run_echo_keys_consumed": [
            "reference",
            "shock_z_cm",
            "z_int_cm",
            "t_cross_int_s",
            "t_end_s",
            "L_ref_cm",
            "dz_cm",
            "mat1_kappa_cm2_g",
            "orientation",
        ],
        "gate_formulas": cert_gate_formulas(),
        "cert_time_match_t_end_frac": CERT_TIME_MATCH_T_END_FRAC,
        "cert_shock_window_dz_coarse": CERT_SHOCK_WINDOW_DZ_COARSE,
        "cert_shock_L2_max": CERT_SHOCK_L2_MAX,
        "cert_le08_anchor_L2_max": CERT_LE08_ANCHOR_L2_MAX,
        "cert_mass_leakage_max": SMOKE_LEAKAGE_MAX,
        "cert_ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
        "runs": {},
        "errors": [],
    }
    exit_code = 1
    verdict = "FAIL"

    try:
        for leg in legs:
            runs[leg] = run_cert_leg(args, leg)
        cert_pass = all(bool(run.get("cert_gate_passed")) for run in runs.values())
        summary["cert_gate_passed"] = cert_pass
        verdict = "PASS" if cert_pass else "FAIL"
        exit_code = 0 if cert_pass else 1
    except Exception as exc:
        summary["errors"].append(str(exc))
        verdict = "FAIL"
        exit_code = 1
    finally:
        summary["verdict"] = verdict
        summary["all_checks_passed"] = exit_code == 0
        summary["runs"] = {label: sanitize(run) for label, run in runs.items()}
        write_summary(summary_path, summary)
        print(f"[i4b] tier=cert verdict={verdict} summary={summary_path}")

    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO_ROOT / "build" / "tenryu"))
    parser.add_argument(
        "--deck",
        default=str(REPO_ROOT / "examples" / "verification" / "i4b_radshock_interface_2d_rz_slab.py"),
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mode", choices=("smoke", "cert"), default="smoke")
    parser.add_argument("--nz", type=int, default=512)
    parser.add_argument("--fine-nz", type=int, default=1024)
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--leg", choices=("A", "B"), default="A")
    parser.add_argument("--kappa-ratio", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.nr <= 0 or args.nz < 5:
        parser.error("--nr must be positive and --nz must be at least 5")
    if args.mode == "cert" and args.fine_nz != 2 * args.nz:
        parser.error("--fine-nz must equal 2 * --nz for cert mode")

    args.workdir = str(default_workdir() if args.workdir is None else Path(args.workdir))
    if args.mode == "cert":
        return run_cert_campaign(args)

    legs = [args.leg]
    summary_path = Path(args.workdir) / "summary.json"
    runs: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": args.mode,
        "tier": "dev_smoke",
        "binary": str(args.binary),
        "deck": str(args.deck),
        "workdir": str(args.workdir),
        "legs": legs,
        "nr": int(args.nr),
        "nz": int(args.nz),
        "kappa_ratio": None if args.kappa_ratio is None else float(args.kappa_ratio),
        "deck_echo_keys": list(DECK_ECHO_KEYS),
        "smoke_shock_precross_tol_dz": SMOKE_SHOCK_PRECROSS_TOL_DZ,
        "smoke_shock_final_tol_frac": SMOKE_SHOCK_FINAL_TOL_FRAC,
        "smoke_iface_tol_dz": SMOKE_IFACE_TOL_DZ,
        "smoke_leakage_max": SMOKE_LEAKAGE_MAX,
        "smoke_ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
        "smoke_cg_max_iter_warnings_max": SMOKE_CG_MAX_ITER_WARNINGS_MAX,
        "smoke_cg_startup_line_max": SMOKE_CG_STARTUP_LINE_MAX,
        "runs": {},
        "errors": [],
    }
    exit_code = 1
    verdict = "FAIL"

    try:
        for leg in legs:
            run = run_leg(args, leg)
            runs[leg] = run

        smoke_pass = all(bool(run.get("smoke_gate_passed")) for run in runs.values())
        summary["smoke_gate_passed"] = smoke_pass
        verdict = "PASS" if smoke_pass else "FAIL"
        exit_code = 0 if smoke_pass else 2
    except Exception as exc:
        summary["errors"].append(str(exc))
        verdict = "FAIL"
        exit_code = 1
    finally:
        summary["verdict"] = verdict
        summary["all_checks_passed"] = exit_code == 0
        summary["runs"] = {label: sanitize(run) for label, run in runs.items()}
        write_summary(summary_path, summary)
        print(f"[i4b] verdict={verdict} summary={summary_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
