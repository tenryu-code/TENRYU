#!/usr/bin/env python3
"""I7a coupled-stack smoke and cert harness.

Smoke tier runs two identical replicas in fresh output directories.  The replica
pair is the dev-time noise-band instrument.  The certification runs evaluate one fresh or
one or more existing runs against a frozen high-resolution ensemble reference.
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


LADDER_ROW = "i7a_coupled_stack_2d_rz_slab"
DECK_ECHO_PREFIX = "[deck:i7a_coupled_stack_2d_rz_slab]"
DECK_ECHO_KEYS = (
    "z_slab_top_cm",
    "slab_thick_cm",
    "t_end_s",
    "laser_peak_W",
    "ramp_s",
    "nr",
    "nz",
    "r_max_cm",
    "z_max_cm",
    "dz_cm",
    "rho_slab_gcc",
)
DECK_ECHO_BINDINGS = {
    "p_peak_w": "laser_peak_W",
    "slab_thick": "slab_thick_cm",
    "ramp": "ramp_s",
    "rho_slab": "rho_slab_gcc",
}
SMOKE_LEDGER_SKIP_T_FRAC = 0.1
SMOKE_LEDGER_EPS_MAX = 1.0e-6
SMOKE_ABSORPTION_FRAC_MIN = 0.10
SMOKE_ABSORPTION_FRAC_MAX = 0.999
SMOKE_LASER_IN_EXPECTED_MIN_FRAC = 0.30
SMOKE_LASER_IN_EXPECTED_MAX_FRAC = 1.05
SMOKE_ABLATION_GRAD_MIN_EV_PER_CM = 100.0
SMOKE_SHOCK_WINDOW_DZ = 5.0
RHO_SLAB_DEFAULT_GCC = 1.0
SMOKE_SHOCK_RHO_JUMP_MIN_GCC = 0.05 * RHO_SLAB_DEFAULT_GCC
# Measured population: 0.045 clean .. 5.9 polluted .. 2160 runaway;
# docs/design/2d_campaign_plan_20260708.md Addendum 12 exec-record 6.
FLOOR_RUNAWAY_RATIO = 50.0
FLOOR_POLLUTED_RATIO = 0.5
REPLICA_BAND_TOL = {
    "z_shock_final_cm": 5.0e-2,
    "z_ablation_final_cm": 5.0e-2,
    "laser_in_cum": 5.0e-2,
    "mass_ablated": 5.0e-2,
}
# Chaotic energy-partition spread, measured 6-24% pair-to-pair - cert-tier
# reference must be band-aware/ensemble; smoke gates only the stable mechanism QoIs.
REPLICA_PARTITION_QOIS = ("E_rad", "E_e", "E_i", "E_kin")
MASS_ABLATED_RHO_FRACTION = 0.5
TINY = 1.0e-300
CERT_REFERENCE_NZ = 2048
CERT_REFERENCE_DT_MAX_S = 6.510416666666668e-13
CERT_DT_MAX_REL_TOL = 1.0e-9
CERT_REFERENCE_DEFAULT = (
    REPO_ROOT
    / "tests/verification/data/i7a_ensemble_reference/i7a_nz2048_donor_v5.json"
)
REPLICA_LABELS = ("A", "B")
REPLICA_QOI_FINAL_KEYS = {
    "z_shock_final_cm": "z_shock_cm",
    "z_ablation_final_cm": "z_ablation_cm",
    "E_e": "E_e_erg",
    "E_i": "E_i_erg",
    "E_kin": "E_kin_erg",
    "laser_in_cum": "laser_in_cum",
    "mass_ablated": "mass_ablated_g",
    "E_rad": "E_rad_erg",
}


@dataclass(frozen=True)
class I7aSnapshot:
    path: Path
    t: float
    step: int
    z_edges: Any
    zc: Any
    rho_z: Any
    qvisc_z: Any
    Te_z: Any
    Ti_z: Any
    z_shock: float | None
    qvisc_peak_slab: float
    z_ablation: float | None
    ablation_grad_abs_eV_per_cm: float | None
    E_rad: float
    E_e: float
    E_i: float
    E_kin: float
    E_total: float
    Te_max: float
    Ti_max: float
    mass_slab: float
    absorption_frac: float


def default_workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "tmp" / "i7a_coupled_stack" / stamp


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


def floor_injection_metrics(outdir: Path) -> dict[str, Any]:
    import h5py

    results_dir = base.discover_actual_outdir(outdir) / "results"
    candidates = sorted(
        path for path in results_dir.glob("*_history.h5") if not path.name.endswith(".tmp")
    )
    if not candidates:
        return {"floor_metrics_available": False}

    with h5py.File(candidates[0], "r") as handle:
        if "energy/floor_injected" not in handle or "energy/laser_incident" not in handle:
            return {"floor_metrics_available": False}
        floor = np.asarray(handle["energy/floor_injected"][()], dtype=float)
        laser = np.asarray(handle["energy/laser_incident"][()], dtype=float)

    floor_total = float(np.sum(floor))
    laser_total = float(np.sum(laser))
    if laser_total > 0.0:
        ratio = floor_total / laser_total
    else:
        ratio = math.inf if floor_total > 0.0 else 0.0
    return {
        "floor_metrics_available": True,
        "floor_injected_total_erg": floor_total,
        "laser_incident_total_erg": laser_total,
        "floor_to_laser_total_ratio": ratio,
        "floor_step_max_erg": float(np.max(floor)),
        "laser_step_max_erg": float(np.max(laser)),
        "floor_runaway": bool(ratio > FLOOR_RUNAWAY_RATIO),
        "floor_polluted": bool(ratio > FLOOR_POLLUTED_RATIO),
    }


def _strictly_increasing(values: Any) -> bool:
    arr = np.asarray(values, dtype=float)
    return bool(arr.size >= 2 and np.all(np.diff(arr) > 0.0))


def _reshape_cells(values: Any, nr: int, nz: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape == (nr, nz):
        return arr
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    raise ValueError(f"{name}: expected shape ({nr}, {nz}) or size {nr * nz}, got {arr.shape}")


def _reshape_rad_energy(values: Any, nr: int, nz: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n_cells = nr * nz
    if arr.shape == (nr, nz):
        return arr
    if arr.ndim == 3 and arr.shape[0] == nr and arr.shape[1] == nz:
        return np.sum(arr, axis=2)
    if arr.ndim == 2 and arr.shape[0] == n_cells:
        return np.sum(arr, axis=1).reshape((nr, nz))
    if arr.size == n_cells:
        return arr.reshape((nr, nz))
    if arr.size % n_cells == 0:
        return arr.reshape((n_cells, arr.size // n_cells)).sum(axis=1).reshape((nr, nz))
    raise ValueError(f"radiation/energy_density: cannot reshape {arr.shape} for ({nr}, {nz})")


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


def _required(handle: Any, path: str) -> Any:
    if path not in handle:
        raise KeyError(f"snapshot missing required dataset {path}")
    return handle[path][()]


def _core_average(field: np.ndarray, nr: int) -> np.ndarray:
    core_nr = nr // 2
    if core_nr < 1:
        raise ValueError("near-axis core requires nr//2 >= 1")
    return np.mean(field[:core_nr, :], axis=0)


def shock_rho_jump_face_z(
    rho_z: Any,
    qvisc_z: Any,
    z_edges: Any,
    zc: Any,
    z_slab_top_cm: float,
    dz_cm: float,
) -> dict[str, Any]:
    rho = np.asarray(rho_z, dtype=float)
    q = np.asarray(qvisc_z, dtype=float)
    edges = np.asarray(z_edges, dtype=float)
    z = np.asarray(zc, dtype=float)
    cell_mask = z < z_slab_top_cm + SMOKE_SHOCK_WINDOW_DZ * dz_cm
    if not cell_mask.any():
        raise RuntimeError("shock search window contains no cells")
    qvisc_peak_slab = float(np.max(np.where(cell_mask, q, -np.inf)))
    if rho.size < 2:
        return {"z": None, "qvisc_peak_slab": qvisc_peak_slab}
    rho_jump = np.abs(np.diff(rho))
    face_z = edges[1:-1]
    mask = face_z < z_slab_top_cm + SMOKE_SHOCK_WINDOW_DZ * dz_cm
    if not mask.any():
        raise RuntimeError("shock search window contains no faces")
    jump_win = np.where(mask, rho_jump, -np.inf)
    j = int(np.argmax(jump_win))
    peak = float(rho_jump[j])
    return {
        "z": None if peak < SMOKE_SHOCK_RHO_JUMP_MIN_GCC else float(face_z[j]),
        "qvisc_peak_slab": qvisc_peak_slab,
    }


def ablation_face_z(Te_z: Any, z_edges: Any, zc: Any, z_slab_top_cm: float, dz_cm: float) -> dict[str, Any]:
    Te = np.asarray(Te_z, dtype=float)
    centers = np.asarray(zc, dtype=float)
    edges = np.asarray(z_edges, dtype=float)
    if Te.size < 2:
        return {"z": None, "grad_abs_eV_per_cm": None}
    dz = np.diff(centers)
    if np.any(dz <= 0.0):
        raise RuntimeError("cell-center z coordinates are not strictly increasing")
    grad = np.abs(np.diff(Te) / dz)
    face_z = edges[1:-1]
    mask = face_z > z_slab_top_cm - SMOKE_SHOCK_WINDOW_DZ * dz_cm
    if not mask.any():
        return {"z": None, "grad_abs_eV_per_cm": None}
    grad_max = float(np.max(np.where(mask, grad, -np.inf)))
    candidates = mask & (grad >= 0.3 * grad_max)
    candidate_idx = np.flatnonzero(candidates)
    j = int(candidate_idx[np.argmin(face_z[candidate_idx])])
    value = float(grad[j])
    return {"z": float(face_z[j]), "grad_abs_eV_per_cm": value}


def slab_mass(rho: np.ndarray, mass: np.ndarray, zc: Any, echo: dict[str, Any]) -> float:
    z = np.asarray(zc, dtype=float)
    z_top = float(echo["z_slab_top_cm"])
    z_bottom = z_top - float(echo["slab_thick_cm"])
    rho_slab = float(echo.get("rho_slab_gcc", RHO_SLAB_DEFAULT_GCC))
    z_mask = (z >= z_bottom) & (z <= z_top)
    cell_mask = z_mask[None, :] & (rho >= MASS_ABLATED_RHO_FRACTION * rho_slab)
    return float(np.sum(np.where(cell_mask, mass, 0.0)))


def read_snapshot(path: Path, echo: dict[str, Any]) -> I7aSnapshot:
    import h5py

    nr = int(echo["nr"])
    nz = int(echo["nz"])
    with h5py.File(path, "r") as handle:
        t_value = base.scalar_dataset(handle, "time_state/t", float(handle.attrs.get("t", 0.0)))
        step = int(round(base.scalar_dataset(handle, "time_state/step", 0.0)))
        rho = _reshape_cells(_required(handle, "hydro/rho"), nr, nz, "hydro/rho")
        qvisc = _reshape_cells(_required(handle, "hydro/Qvisc"), nr, nz, "hydro/Qvisc")
        Te = _reshape_cells(_required(handle, "hydro/Te"), nr, nz, "hydro/Te")
        Ti = _reshape_cells(_required(handle, "hydro/Ti"), nr, nz, "hydro/Ti")
        ee = _reshape_cells(_required(handle, "hydro/ee"), nr, nz, "hydro/ee")
        ei = _reshape_cells(_required(handle, "hydro/ei"), nr, nz, "hydro/ei")
        vol = _reshape_cells(_required(handle, "hydro/vol"), nr, nz, "hydro/vol")
        mass = _reshape_cells(_required(handle, "hydro/mass"), nr, nz, "hydro/mass")
        rad_E = _reshape_rad_energy(_required(handle, "radiation/energy_density"), nr, nz)
        z_nodes = _reshape_z_nodes(_required(handle, "mesh/x_z"), nr, nz)
        z_edges = np.asarray(z_nodes[0, :], dtype=float)
        if not _strictly_increasing(z_edges):
            raise ValueError("mesh/x_z: row-0 z edges are not strictly increasing")
        zc = 0.5 * (z_edges[:-1] + z_edges[1:])

        rho_z = _core_average(rho, nr)
        qvisc_z = _core_average(qvisc, nr)
        Te_z = _core_average(Te, nr)
        Ti_z = _core_average(Ti, nr)
        shock = shock_rho_jump_face_z(
            rho_z,
            qvisc_z,
            z_edges,
            zc,
            float(echo["z_slab_top_cm"]),
            float(echo["dz_cm"]),
        )
        ablation = ablation_face_z(
            Te_z,
            z_edges,
            zc,
            float(echo["z_slab_top_cm"]),
            float(echo["dz_cm"]),
        )

        return I7aSnapshot(
            path=path,
            t=float(t_value),
            step=step,
            z_edges=z_edges,
            zc=zc,
            rho_z=rho_z,
            qvisc_z=qvisc_z,
            Te_z=Te_z,
            Ti_z=Ti_z,
            z_shock=shock["z"],
            qvisc_peak_slab=float(shock["qvisc_peak_slab"]),
            z_ablation=ablation["z"],
            ablation_grad_abs_eV_per_cm=ablation["grad_abs_eV_per_cm"],
            E_rad=float(np.sum(rad_E * vol)),
            E_e=float(np.sum(ee * mass)),
            E_i=float(np.sum(ei * mass)),
            E_kin=float(base.scalar_dataset(handle, "time_state/E_kinetic", 0.0)),
            E_total=float(base.scalar_dataset(handle, "time_state/E_total", 0.0)),
            Te_max=float(np.max(Te)),
            Ti_max=float(np.max(Ti)),
            mass_slab=slab_mass(rho, mass, zc, echo),
            absorption_frac=float(base.scalar_dataset(handle, "laser/absorption_fraction", 0.0)),
        )


def read_snapshots(outdir: Path, echo: dict[str, Any]) -> list[I7aSnapshot]:
    actual = base.discover_actual_outdir(outdir)
    results_dir = actual / "results"
    candidates = sorted(
        path for path in results_dir.glob("*.h5") if "history" not in path.name
    )
    if not candidates:
        raise FileNotFoundError(f"no non-history HDF5 snapshots under {results_dir}")
    snapshots = [read_snapshot(path, echo) for path in candidates]
    return sorted(snapshots, key=lambda snap: (snap.t, snap.step, str(snap.path)))


def parse_ledger_eps(log_path: Path) -> dict[str, Any]:
    """All [wj_step_audit] (t, eps) pairs; gate value is max late eps."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {
            "series": [],
            "eps_max_late": None,
            "eps_final": None,
            "n_lines": 0,
            "laser_in_cum": None,
        }
    ts: list[float] = []
    epss: list[float] = []
    laser_in_cum = 0.0
    laser_in_seen = False
    for line in text.splitlines():
        if "[wj_step_audit]" not in line:
            continue
        t_val = eps_val = None
        for token in line.split():
            if token.startswith("t="):
                try:
                    t_val = float(token[2:])
                except ValueError:
                    pass
            elif token.startswith("eps="):
                try:
                    eps_val = float(token[4:])
                except ValueError:
                    pass
            elif token.startswith("laser_in="):
                try:
                    laser_in_cum += float(token[len("laser_in="):])
                    laser_in_seen = True
                except ValueError:
                    pass
        if t_val is not None and eps_val is not None:
            ts.append(t_val)
            epss.append(eps_val)
    if not ts:
        return {
            "series": [],
            "eps_max_late": None,
            "eps_final": None,
            "n_lines": 0,
            "laser_in_cum": laser_in_cum if laser_in_seen else None,
        }
    t_final = max(ts)
    late = [e for t, e in zip(ts, epss) if t >= SMOKE_LEDGER_SKIP_T_FRAC * t_final]
    return {
        "series": [{"t": t, "eps": e} for t, e in zip(ts, epss)],
        "eps_max_late": max(late) if late else None,
        "eps_final": epss[-1],
        "n_lines": len(ts),
        "laser_in_cum": laser_in_cum if laser_in_seen else None,
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
    return {
        "z_slab_top_cm": float(parsed["z_slab_top_cm"]),
        "slab_thick_cm": float(parsed["slab_thick_cm"]),
        "slab_thick_um": float(parsed["slab_thick_cm"]) * 1.0e4,
        "t_end_s": float(parsed["t_end_s"]),
        "p_peak_w": float(parsed["laser_peak_W"]),
        "laser_peak_W": float(parsed["laser_peak_W"]),
        "ramp_s": float(parsed["ramp_s"]),
        "nr": int(parsed["nr"]),
        "nz": int(parsed["nz"]),
        "r_max_cm": float(parsed["r_max_cm"]),
        "z_max_cm": float(parsed["z_max_cm"]),
        "dz_cm": float(parsed["dz_cm"]),
        "rho_slab_gcc": float(parsed["rho_slab_gcc"]),
        "dt_max_s": float(parsed["dt_max_s"]) if "dt_max_s" in parsed else None,
    }


def _is_finite_nonnegative(value: Any) -> bool:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(fvalue) and fvalue >= 0.0


def energy_series_sane(snapshots: list[I7aSnapshot]) -> bool:
    for snap in snapshots:
        for value in (snap.E_rad, snap.E_e, snap.E_i, snap.E_kin):
            if not _is_finite_nonnegative(value):
                return False
    return True


def shock_launched(snapshots: list[I7aSnapshot], echo: dict[str, Any]) -> bool:
    if not snapshots:
        return False
    half = snapshots[len(snapshots) // 2:]
    if not half or any(snap.z_shock is None for snap in half):
        return False
    first_defined = next((snap.z_shock for snap in snapshots if snap.z_shock is not None), None)
    final = snapshots[-1].z_shock
    dz = float(echo["dz_cm"])
    return (
        first_defined is not None
        and final is not None
        and float(final) <= float(first_defined) + 0.5 * dz
        and float(final) < float(echo["z_slab_top_cm"]) - dz
    )


def expected_ramp_energy(echo: dict[str, Any]) -> float:
    # deck power is Watts; cgs conversion x1e7
    return 1.0e7 * float(echo["p_peak_w"]) * max(
        float(echo["t_end_s"]) - 0.5 * float(echo["ramp_s"]),
        0.0,
    )


def qoi_series(snapshots: list[I7aSnapshot], mass_slab_initial: float) -> dict[str, Any]:
    return {
        "t_s": [snap.t for snap in snapshots],
        "z_shock_cm": [snap.z_shock for snap in snapshots],
        "qvisc_peak_slab": [snap.qvisc_peak_slab for snap in snapshots],
        "z_ablation_cm": [snap.z_ablation for snap in snapshots],
        "ablation_grad_abs_eV_per_cm": [snap.ablation_grad_abs_eV_per_cm for snap in snapshots],
        "E_rad_erg": [snap.E_rad for snap in snapshots],
        "E_e_erg": [snap.E_e for snap in snapshots],
        "E_i_erg": [snap.E_i for snap in snapshots],
        "E_kin_erg": [snap.E_kin for snap in snapshots],
        "E_total_erg": [snap.E_total for snap in snapshots],
        "Te_max_eV": [snap.Te_max for snap in snapshots],
        "Ti_max_eV": [snap.Ti_max for snap in snapshots],
        "M_slab_g": [snap.mass_slab for snap in snapshots],
        "mass_ablated_g": [mass_slab_initial - snap.mass_slab for snap in snapshots],
    }


def smoke_metrics(
    deck: dict[str, Any],
    echo: dict[str, Any],
    snapshots: list[I7aSnapshot],
    deck_log: Path,
    replica: str,
) -> dict[str, Any]:
    first = snapshots[0]
    final = snapshots[-1]
    ledger = parse_ledger_eps(deck_log)
    mass_slab_initial = float(first.mass_slab)
    series = qoi_series(snapshots, mass_slab_initial)
    mass_ablated_final = float(mass_slab_initial - final.mass_slab)
    laser_in_cum = ledger["laser_in_cum"]
    e_expected = expected_ramp_energy(echo)
    ablation_grad_final = final.ablation_grad_abs_eV_per_cm

    laser_absorbing = (
        _is_finite_nonnegative(final.absorption_frac)
        and SMOKE_ABSORPTION_FRAC_MIN <= float(final.absorption_frac) <= SMOKE_ABSORPTION_FRAC_MAX
        and laser_in_cum is not None
        and math.isfinite(float(laser_in_cum))
        and SMOKE_LASER_IN_EXPECTED_MIN_FRAC * e_expected <= float(laser_in_cum)
        and float(laser_in_cum) <= SMOKE_LASER_IN_EXPECTED_MAX_FRAC * e_expected
    )
    ablation_active = (
        final.z_ablation is not None
        and ablation_grad_final is not None
        and math.isfinite(float(ablation_grad_final))
        and float(ablation_grad_final) >= SMOKE_ABLATION_GRAD_MIN_EV_PER_CM
    )
    ledger_eps = ledger["eps_max_late"]
    ledger_closure = (
        ledger_eps is not None
        and math.isfinite(float(ledger_eps))
        and float(ledger_eps) <= SMOKE_LEDGER_EPS_MAX
    )
    gates = {
        "run_completed": bool(deck["returncode"] == 0 and deck["t_end_reached"]),
        "laser_absorbing": bool(laser_absorbing),
        "ablation_active": bool(ablation_active),
        "shock_launched": bool(shock_launched(snapshots, echo)),
        "ledger_closure": bool(ledger_closure),
        "energies_sane": bool(energy_series_sane(snapshots)),
    }
    return {
        "tier": "dev_smoke",
        "replica": replica,
        "run_completed": gates["run_completed"],
        "t_final_s": float(final.t),
        "final_step": int(final.step),
        "snapshot_count": len(snapshots),
        "first_snapshot_path": str(first.path),
        "final_snapshot_path": str(final.path),
        "E_ramp_expected": float(e_expected),
        "laser_in_cum": None if laser_in_cum is None else float(laser_in_cum),
        "absorption_frac": float(final.absorption_frac),
        "ablation_grad_final_eV_per_cm": ablation_grad_final,
        "Te_max_run_eV": float(max(snap.Te_max for snap in snapshots)),
        "Ti_max_run_eV": float(max(snap.Ti_max for snap in snapshots)),
        "mass_slab_initial_g": mass_slab_initial,
        "mass_ablated_final_g": mass_ablated_final,
        "ledger_eps_max_late": ledger["eps_max_late"],
        "ledger_eps_final": ledger["eps_final"],
        "ledger_n_lines": ledger["n_lines"],
        "ledger_series": ledger["series"],
        "final_qoi": {
            "z_shock_cm": final.z_shock,
            "z_ablation_cm": final.z_ablation,
            "E_rad_erg": float(final.E_rad),
            "E_e_erg": float(final.E_e),
            "E_i_erg": float(final.E_i),
            "E_kin_erg": float(final.E_kin),
            "laser_in_cum": None if laser_in_cum is None else float(laser_in_cum),
            "mass_ablated_g": mass_ablated_final,
        },
        "series": series,
        "gates": gates,
    }


def incomplete_run_metrics(deck: dict[str, Any], replica: str) -> dict[str, Any]:
    return {
        "tier": "dev_smoke",
        "replica": replica,
        "run_completed": False,
        "deck_returncode": int(deck["returncode"]),
        "termination_reason": str(deck["termination_reason"]),
        "t_end_reached": bool(deck["t_end_reached"]),
        "gates": {
            "run_completed": False,
            "laser_absorbing": False,
            "ablation_active": False,
            "shock_launched": False,
            "ledger_closure": False,
            "energies_sane": False,
        },
    }


def failed_smoke_gates(metrics: dict[str, Any]) -> list[str]:
    gates = metrics.get("gates", {})
    failed = [str(name) for name, passed in gates.items() if not bool(passed)]
    if metrics.get("floor_runaway") is True:
        failed.append("floor_runaway")
    return failed


def format_float(value: Any) -> str:
    if value == "not_available":
        return "not_available"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{fvalue:.6e}" if math.isfinite(fvalue) else "missing"


def print_replica_line(replica: str, metrics: dict[str, Any], verdict: str) -> None:
    if not metrics.get("run_completed", False):
        print(
            "[i7a] "
            f"replica={replica} tier=smoke "
            f"run_completed=False "
            f"deck_rc={metrics.get('deck_returncode', 'missing')} "
            f"termination={metrics.get('termination_reason', 'missing')} "
            f"verdict={verdict}"
        )
        return
    final = metrics["final_qoi"]
    floor_ratio = (
        format_float(metrics.get("floor_to_laser_total_ratio"))
        if metrics.get("floor_metrics_available", False)
        else "n/a"
    )
    print(
        "[i7a] "
        f"replica={replica} tier=smoke "
        f"run_completed={metrics['run_completed']} "
        f"z_shock_final_cm={format_float(final['z_shock_cm'])} "
        f"z_ablation_final_cm={format_float(final['z_ablation_cm'])} "
        f"ablation_grad={format_float(metrics['ablation_grad_final_eV_per_cm'])} "
        f"E_rad={format_float(final['E_rad_erg'])} "
        f"E_e={format_float(final['E_e_erg'])} "
        f"E_i={format_float(final['E_i_erg'])} "
        f"E_kin={format_float(final['E_kin_erg'])} "
        f"laser_in_cum={format_float(final['laser_in_cum'])} "
        f"absorption_frac={format_float(metrics['absorption_frac'])} "
        f"mass_ablated={format_float(final['mass_ablated_g'])} "
        f"ledger_eps_late={format_float(metrics['ledger_eps_max_late'])} "
        f"floor_ratio={floor_ratio} "
        f"verdict={verdict}"
    )


def relative_difference(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    try:
        avalue = float(a)
        bvalue = float(b)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(avalue) or not math.isfinite(bvalue):
        return None
    return abs(avalue - bvalue) / max(abs(avalue), abs(bvalue), TINY)


def pair_metrics(runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    a = runs[REPLICA_LABELS[0]]
    b = runs[REPLICA_LABELS[1]]
    diffs: dict[str, Any] = {}
    for qoi, tol in REPLICA_BAND_TOL.items():
        key = REPLICA_QOI_FINAL_KEYS[qoi]
        aval = a.get("metrics", {}).get("final_qoi", {}).get(key)
        bval = b.get("metrics", {}).get("final_qoi", {}).get(key)
        diffs[qoi] = {
            "a": aval,
            "b": bval,
            "relative": relative_difference(aval, bval),
            "relative_tolerance": tol,
        }
    replica_partition_diffs: dict[str, Any] = {}
    for qoi in REPLICA_PARTITION_QOIS:
        key = REPLICA_QOI_FINAL_KEYS[qoi]
        aval = a.get("metrics", {}).get("final_qoi", {}).get(key)
        bval = b.get("metrics", {}).get("final_qoi", {}).get(key)
        replica_partition_diffs[qoi] = {
            "a": aval,
            "b": bval,
            "relative": relative_difference(aval, bval),
        }
    replica_band = all(
        item["relative"] is not None
        and float(item["relative"]) <= float(item["relative_tolerance"])
        for item in diffs.values()
    )
    finite_diffs = {
        qoi: item for qoi, item in diffs.items() if item["relative"] is not None
    }
    worst_qoi = max(
        finite_diffs,
        key=lambda qoi: float(finite_diffs[qoi]["relative"]),
        default=None,
    )
    worst = None
    if worst_qoi is not None:
        worst = {
            "qoi": worst_qoi,
            "relative": finite_diffs[worst_qoi]["relative"],
            "relative_tolerance": finite_diffs[worst_qoi]["relative_tolerance"],
        }
    return {
        "tier": "dev_smoke",
        "diffs": diffs,
        "replica_partition_diffs": replica_partition_diffs,
        "max_relative_diff": None if worst is None else worst["relative"],
        "relative_tolerances": dict(REPLICA_BAND_TOL),
        "worst": worst,
        "gates": {"replica_band": bool(replica_band)},
        "failed_gates": [] if replica_band else ["replica_band"],
        "verdict": "PASS" if replica_band else "FAIL",
    }


def print_pair_line(pair: dict[str, Any], verdict: str) -> None:
    partition_info = ",".join(
        f"{qoi}:{format_float(pair['replica_partition_diffs'][qoi]['relative'])}"
        for qoi in REPLICA_PARTITION_QOIS
    )
    print(
        "[i7a] "
        f"pair tier=smoke "
        f"replica_band={pair['gates']['replica_band']} "
        f"worst_qoi={None if pair['worst'] is None else pair['worst']['qoi']} "
        f"rel_diff={format_float(None if pair['worst'] is None else pair['worst']['relative'])} "
        f"tol={format_float(None if pair['worst'] is None else pair['worst']['relative_tolerance'])} "
        f"partition_diffs={partition_info} "
        f"verdict={verdict}"
    )


def run_deck(args: argparse.Namespace, outdir: Path, replica: str) -> dict[str, Any]:
    env_extra = {
        "TENRYU_I7A_OUTDIR": str(outdir),
        "TENRYU_ENERGY_OPERATOR_AUDIT": "1",
        "TENRYU_ENERGY_OPERATOR_AUDIT_TMAX": "0",
    }
    if args.t_end_ns is not None:
        env_extra["TENRYU_I7A_T_END_NS"] = f"{args.t_end_ns:.17g}"
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
        "replica": replica,
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


def run_replica(args: argparse.Namespace, replica: str) -> dict[str, Any]:
    replica_dir = Path(args.workdir) / f"replica{replica}"
    outdir = replica_dir / "out"
    replica_dir.mkdir(parents=True, exist_ok=True)
    if outdir.exists() and any(outdir.iterdir()):
        raise RuntimeError(f"replica {replica} outdir is not fresh: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"replica": replica, "label": f"replica{replica}"}

    deck = run_deck(args, outdir, replica)
    result["deck"] = deck
    run_completed = deck["returncode"] == 0 and deck["t_end_reached"]
    if not run_completed:
        metrics = incomplete_run_metrics(deck, replica)
        gates = failed_smoke_gates(metrics)
        replica_verdict = "PASS" if not gates else "FAIL"
        print_replica_line(replica, metrics, replica_verdict)
        result.update(
            {
                "tier": "dev_smoke",
                "metrics": metrics,
                "failed_gates": gates,
                "smoke_gate_passed": len(gates) == 0,
                "verdict": replica_verdict,
            }
        )
        return result

    echo = parse_deck_echo(Path(deck["log"]))
    snapshots = read_snapshots(outdir, echo)
    metrics = smoke_metrics(deck, echo, snapshots, Path(deck["log"]), replica)
    metrics.update(floor_injection_metrics(outdir))
    gates = failed_smoke_gates(metrics)
    replica_verdict = "PASS" if not gates else "FAIL"
    print_replica_line(replica, metrics, replica_verdict)

    result.update(
        {
            "deck_echo": echo,
            "tier": "dev_smoke",
            "metrics": metrics,
            "failed_gates": gates,
            "smoke_gate_passed": len(gates) == 0,
            "verdict": replica_verdict,
        }
    )
    return result


def cert_run_label(entry: Path) -> str:
    if (entry / "out").is_dir():
        return entry.name
    return entry.parent.name


def resolve_cert_run(entry: Path) -> dict[str, Path]:
    if (entry / "out").is_dir():
        run_dir = entry
        outdir = entry / "out"
    else:
        run_dir = entry.parent
        outdir = entry
    log_path = run_dir / "run.log"
    if not log_path.is_file():
        raise FileNotFoundError(
            f"deck echo log not found for {entry}: expected {log_path}"
        )
    return {
        "run_dir": run_dir,
        "outdir": outdir,
        "actual_outdir": base.discover_actual_outdir(outdir),
        "log": log_path,
    }


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def cert_analysis_failure(entry: Path, reason: str) -> dict[str, Any]:
    return {
        "name": cert_run_label(entry),
        "input": str(entry),
        "gate_table": {
            "analysis": {
                "passed": False,
                "reason": reason,
            }
        },
        "tier_S_counts": {"pass": 0, "fail": 0},
        "tier_B_counts": {"pass": 0, "fail": 0},
        "floor": {"available": False},
        "ledger": {"eps_max_late": None},
        "failed_gates": ["analysis"],
        "verdict": "FAIL",
    }


def evaluate_cert_run(entry: Path, reference: dict[str, Any]) -> dict[str, Any]:
    resolved = resolve_cert_run(entry)
    outdir = resolved["outdir"]
    log_path = resolved["log"]
    echo = parse_deck_echo(log_path)
    snapshots = read_snapshots(outdir, echo)
    series = qoi_series(snapshots, snapshots[0].mass_slab)
    reference_qoi = reference["qoi"]
    reference_gates = reference["gates"]

    nz_passed = int(echo["nz"]) == CERT_REFERENCE_NZ
    run_dt_max_raw = echo.get("dt_max_s")
    run_dt_max = None if run_dt_max_raw is None else float(run_dt_max_raw)
    dt_max_rel_diff = (
        None
        if run_dt_max is None
        else abs(run_dt_max - CERT_REFERENCE_DT_MAX_S) / CERT_REFERENCE_DT_MAX_S
    )
    dt_max_passed = (
        dt_max_rel_diff is not None and dt_max_rel_diff <= CERT_DT_MAX_REL_TOL
    )
    config_passed = nz_passed and dt_max_passed
    config_gate = {
        "passed": config_passed,
        "run_nz": int(echo["nz"]),
        "reference_nz": CERT_REFERENCE_NZ,
        "run_dt_max_s": run_dt_max,
        "reference_dt_max_s": CERT_REFERENCE_DT_MAX_S,
        "dt_max_rel_diff": dt_max_rel_diff,
        "dt_max_rel_tol": CERT_DT_MAX_REL_TOL,
    }
    failed_gates = [] if config_passed else ["config_mismatch"]

    tier_s: dict[str, Any] = {}
    tier_s_pass = 0
    for name, tolerance_value in reference_gates["tier_S"].items():
        tolerance = float(tolerance_value)
        run_final = finite_float(series[name][-1])
        ref_final = float(reference_qoi[name]["final"]["median"])
        relative = (
            None
            if run_final is None
            else abs(run_final - ref_final) / max(abs(ref_final), TINY)
        )
        passed = relative is not None and relative <= tolerance
        tier_s[name] = {
            "passed": passed,
            "run_final": run_final,
            "reference_final_median": ref_final,
            "relative_error": relative,
            "tolerance": tolerance,
        }
        if passed:
            tier_s_pass += 1
        else:
            failed_gates.append(f"tier_S:{name}")

    tier_b: dict[str, Any] = {}
    tier_b_pass = 0
    band_factor = float(reference_gates["tier_B_band_factor"])
    for name in reference_gates["tier_B"]:
        run_final = finite_float(series[name][-1])
        ref_final = reference_qoi[name]["final"]
        center = float(ref_final["median"])
        ref_min = float(ref_final["min"])
        ref_max = float(ref_final["max"])
        half = max(center - ref_min, ref_max - center) * band_factor
        half = max(half, TINY)
        absolute_error = None if run_final is None else abs(run_final - center)
        passed = absolute_error is not None and absolute_error <= half
        tier_b[name] = {
            "passed": passed,
            "run_final": run_final,
            "reference_final_median": center,
            "reference_final_min": ref_min,
            "reference_final_max": ref_max,
            "band_factor": band_factor,
            "half_width": half,
            "absolute_error": absolute_error,
        }
        if passed:
            tier_b_pass += 1
        else:
            failed_gates.append(f"tier_B:{name}")

    tier_d: dict[str, Any] = {}
    for name in reference_gates["tier_D"]:
        ref_final = reference_qoi[name]["final"]
        tier_d[name] = {
            "run_final": finite_float(series[name][-1]),
            "reference_final_median": float(ref_final["median"]),
            "reference_final_min": float(ref_final["min"]),
            "reference_final_max": float(ref_final["max"]),
        }

    floor_metrics = floor_injection_metrics(outdir)
    floor_available = bool(floor_metrics.get("floor_metrics_available", False))
    floor_ratio = (
        float(floor_metrics["floor_to_laser_total_ratio"])
        if floor_available
        else None
    )
    floor_runaway_threshold = float(reference_gates["floor_runaway_ratio"])
    floor_polluted_threshold = float(reference_gates["floor_polluted_ratio"])
    floor_gate: dict[str, Any] = {
        "available": floor_available,
        "passed": None,
        "ratio": floor_ratio,
        "threshold": floor_runaway_threshold,
    }
    floor_polluted = None
    if floor_available:
        floor_gate["passed"] = bool(floor_ratio <= floor_runaway_threshold)
        floor_polluted = bool(floor_ratio > floor_polluted_threshold)
        if not floor_gate["passed"]:
            failed_gates.append("floor_runaway")

    ledger = parse_ledger_eps(log_path)
    ledger_eps = finite_float(ledger["eps_max_late"])
    ledger_passed = ledger_eps is not None and ledger_eps <= SMOKE_LEDGER_EPS_MAX
    ledger_gate = {
        "passed": ledger_passed,
        "eps_max_late": ledger_eps,
        "threshold": SMOKE_LEDGER_EPS_MAX,
    }
    if not ledger_passed:
        failed_gates.append("ledger_closure")

    return {
        "name": cert_run_label(entry),
        "input": str(entry),
        "run_dir": str(resolved["run_dir"]),
        "outdir": str(outdir),
        "actual_outdir": str(resolved["actual_outdir"]),
        "log": str(log_path),
        "deck_echo": echo,
        "snapshot_count": len(snapshots),
        "t_final_s": float(snapshots[-1].t),
        "final_qoi": {name: values[-1] for name, values in series.items() if name != "t_s"},
        "gate_table": {
            "config_mismatch": config_gate,
            "tier_S": tier_s,
            "tier_B": tier_b,
            "floor_runaway": floor_gate,
            "ledger_closure": ledger_gate,
        },
        "tier_S_counts": {
            "pass": tier_s_pass,
            "fail": len(tier_s) - tier_s_pass,
        },
        "tier_B_counts": {
            "pass": tier_b_pass,
            "fail": len(tier_b) - tier_b_pass,
        },
        "tier_D": tier_d,
        "floor": {
            "available": floor_available,
            "metrics": floor_metrics,
            "runaway_threshold": floor_runaway_threshold,
            "polluted_threshold": floor_polluted_threshold,
            "floor_polluted": floor_polluted,
        },
        "ledger": ledger,
        "failed_gates": failed_gates,
        "verdict": "PASS" if not failed_gates else "FAIL",
    }


def print_cert_run(run: dict[str, Any]) -> None:
    tier_s = run["tier_S_counts"]
    tier_b = run["tier_B_counts"]
    floor_ratio = run.get("floor", {}).get("metrics", {}).get(
        "floor_to_laser_total_ratio"
    )
    ledger_eps = run.get("ledger", {}).get("eps_max_late")
    print(
        "[i7a-cert] "
        f"run={run['name']} "
        f"tier_S={tier_s['pass']}pass/{tier_s['fail']}fail "
        f"tier_B={tier_b['pass']}pass/{tier_b['fail']}fail "
        f"floor_ratio={format_float(floor_ratio) if floor_ratio is not None else 'n/a'} "
        f"ledger_eps={format_float(ledger_eps)} "
        f"verdict={run['verdict']}"
    )
    floor = run.get("floor", {})
    if floor.get("floor_polluted") is True:
        print(
            "[i7a-cert] "
            f"run={run['name']} WARNING floor_polluted "
            f"ratio={format_float(floor_ratio)} "
            f"threshold={format_float(floor['polluted_threshold'])}"
        )
    for gate in run["failed_gates"]:
        if gate == "analysis":
            detail = run["gate_table"]["analysis"]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate=analysis reason={detail['reason']} FAIL"
            )
        elif gate == "config_mismatch":
            detail = run["gate_table"]["config_mismatch"]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate=config_mismatch "
                f"run_nz={detail['run_nz']} reference_nz={detail['reference_nz']} "
                f"run_dt_max_s={detail['run_dt_max_s']} "
                f"reference_dt_max_s={detail['reference_dt_max_s']} FAIL"
            )
        elif gate.startswith("tier_S:"):
            name = gate.split(":", 1)[1]
            detail = run["gate_table"]["tier_S"][name]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate={gate} "
                f"run_final={format_float(detail['run_final'])} "
                f"reference_median={format_float(detail['reference_final_median'])} "
                f"rel={format_float(detail['relative_error'])} "
                f"tol={format_float(detail['tolerance'])} FAIL"
            )
        elif gate.startswith("tier_B:"):
            name = gate.split(":", 1)[1]
            detail = run["gate_table"]["tier_B"][name]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate={gate} "
                f"run_final={format_float(detail['run_final'])} "
                f"center={format_float(detail['reference_final_median'])} "
                f"abs_error={format_float(detail['absolute_error'])} "
                f"half={format_float(detail['half_width'])} FAIL"
            )
        elif gate == "floor_runaway":
            detail = run["gate_table"]["floor_runaway"]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate=floor_runaway "
                f"ratio={format_float(detail['ratio'])} "
                f"threshold={format_float(detail['threshold'])} FAIL"
            )
        elif gate == "ledger_closure":
            detail = run["gate_table"]["ledger_closure"]
            print(
                "[i7a-cert] "
                f"run={run['name']} gate=ledger_closure "
                f"eps_max={format_float(detail['eps_max_late'])} "
                f"threshold={format_float(detail['threshold'])} FAIL"
            )


def run_cert(args: argparse.Namespace) -> int:
    summary_path = Path(args.workdir) / "cert_summary.json"
    reference_path = Path(args.cert_reference)
    runs: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": "cert",
        "tier": "cert",
        "workdir": str(args.workdir),
        "reference": {"path": str(reference_path)},
        "runs": runs,
        "errors": [],
        "verdict": "FAIL",
        "all_checks_passed": False,
    }
    try:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        summary["reference"].update(
            {
                "schema_version": reference.get("schema_version"),
                "provenance": reference.get("provenance", {}),
            }
        )

        if args.cert_analyze:
            entries = [Path(value) for value in args.cert_analyze]
            fresh_error = None
        else:
            entries = [Path(args.workdir) / f"replica{REPLICA_LABELS[0]}" / "out"]
            try:
                run_replica(args, REPLICA_LABELS[0])
                fresh_error = None
            except Exception as exc:
                fresh_error = str(exc)

        for entry in entries:
            if fresh_error is not None:
                run = cert_analysis_failure(entry, fresh_error)
            else:
                try:
                    run = evaluate_cert_run(entry, reference)
                except Exception as exc:
                    run = cert_analysis_failure(entry, str(exc))
            runs.append(run)
            print_cert_run(run)

        passed = bool(runs) and all(run["verdict"] == "PASS" for run in runs)
        summary["verdict"] = "PASS" if passed else "FAIL"
        summary["all_checks_passed"] = passed
        return 0 if passed else 1
    except Exception as exc:
        summary["errors"].append(str(exc))
        print(f"[i7a-cert] error={exc}", file=sys.stderr)
        return 1
    finally:
        write_summary(summary_path, summary)
        print(f"[i7a-cert] verdict={summary['verdict']} summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO_ROOT / "build" / "tenryu"))
    parser.add_argument(
        "--deck",
        default=str(REPO_ROOT / "examples" / "verification" / "i7a_coupled_stack_2d_rz_slab.py"),
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mode", choices=("smoke", "cert"), default="smoke")
    parser.add_argument("--t-end-ns", type=float, default=None)
    parser.add_argument("--cert-reference", default=str(CERT_REFERENCE_DEFAULT))
    parser.add_argument("--cert-analyze", action="append", default=None)
    return parser


def gate_table() -> dict[str, Any]:
    return {
        "run_completed": "rc==0 and run_info termination_reason == t_end_reached",
        "laser_absorbing": {
            "absorption_frac_min": SMOKE_ABSORPTION_FRAC_MIN,
            "absorption_frac_max": SMOKE_ABSORPTION_FRAC_MAX,
            "laser_in_expected_min_frac": SMOKE_LASER_IN_EXPECTED_MIN_FRAC,
            "laser_in_expected_max_frac": SMOKE_LASER_IN_EXPECTED_MAX_FRAC,
            "E_ramp_expected": "1.0e7 * laser_peak_W * max(t_end_s - ramp_s/2, 0)",
        },
        "ablation_active": {
            "final_z_ablation_required": True,
            "final_abs_dTe_dz_min_eV_per_cm": SMOKE_ABLATION_GRAD_MIN_EV_PER_CM,
        },
        "shock_launched": {
            "rho_jump_min_gcc": SMOKE_SHOCK_RHO_JUMP_MIN_GCC,
            "window": "interior face z < z_slab_top_cm + 5*dz_cm",
            "last_half_defined": True,
            "plateau_tolerant_monotonicity": "z_final <= z_first_defined + 0.5*dz_cm",
            "final_inside_slab": "z_final < z_slab_top_cm - dz_cm",
        },
        "ledger_closure": {
            "eps_max_late_max": SMOKE_LEDGER_EPS_MAX,
            "skip_t_frac": SMOKE_LEDGER_SKIP_T_FRAC,
        },
        "energies_sane": "E_rad/E_e/E_i/E_kin finite and >= 0 at every snapshot",
        "replica_band": {
            "relative_tolerances": dict(REPLICA_BAND_TOL),
            "qoi_final_keys": {qoi: REPLICA_QOI_FINAL_KEYS[qoi] for qoi in REPLICA_BAND_TOL},
            "recorded_only_partition_diffs": {
                qoi: REPLICA_QOI_FINAL_KEYS[qoi] for qoi in REPLICA_PARTITION_QOIS
            },
        },
    }


def qoi_inventory() -> list[str]:
    return [
        "z_shock(t): near-axis r-average max |rho jump| interior face, z < z_slab_top_cm + 5*dz_cm",
        "qvisc_peak_slab(t): near-axis r-average Qvisc peak in the shock window, diagnostic only",
        "z_ablation(t): smallest-z face with |dTe/dz| >= 0.3*window max, z > z_slab_top_cm - 5*dz_cm",
        "E_rad(t): sum(radiation/energy_density * hydro/vol) over all cells",
        "E_e(t): sum(hydro/ee * hydro/mass) over all cells",
        "E_i(t): sum(hydro/ei * hydro/mass) over all cells",
        "E_kin(t): time_state/E_kinetic",
        "E_total(t): time_state/E_total",
        "Te_max_run/Ti_max_run: max field value over snapshots",
        "mass_ablated(t): initial fixed slab-window mass minus current thresholded mass",
        "laser_in_cum: sum of laser_in over all [wj_step_audit] lines",
        "absorption_frac: laser/absorption_fraction from final snapshot",
        "ledger eps(t): [wj_step_audit] (t, eps) series with max-late gate",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.t_end_ns is not None and args.t_end_ns <= 0.0:
        parser.error("--t-end-ns must be positive")

    args.workdir = str(default_workdir() if args.workdir is None else Path(args.workdir))
    if args.mode == "cert":
        return run_cert(args)

    summary_path = Path(args.workdir) / "summary.json"
    runs: dict[str, dict[str, Any]] = {}
    pair: dict[str, Any] = {}
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": args.mode,
        "tier": "dev_smoke",
        "binary": str(args.binary),
        "deck": str(args.deck),
        "workdir": str(args.workdir),
        "replicas": list(REPLICA_LABELS),
        "t_end_ns": None if args.t_end_ns is None else float(args.t_end_ns),
        "deck_echo_prefix": DECK_ECHO_PREFIX,
        "deck_echo_keys": list(DECK_ECHO_KEYS),
        "deck_echo_bindings": DECK_ECHO_BINDINGS,
        "qoi_inventory": qoi_inventory(),
        "gate_table": gate_table(),
        "runs": {},
        "pair": {},
        "errors": [],
    }
    exit_code = 1
    verdict = "FAIL"

    try:
        for replica in REPLICA_LABELS:
            run = run_replica(args, replica)
            runs[replica] = run

        pair = pair_metrics(runs)
        pair_verdict = "PASS" if not pair["failed_gates"] else "FAIL"
        print_pair_line(pair, pair_verdict)
        smoke_pass = (
            all(bool(run.get("smoke_gate_passed")) for run in runs.values())
            and bool(pair["gates"]["replica_band"])
        )
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
        summary["pair"] = sanitize(pair)
        write_summary(summary_path, summary)
        print(f"[i7a] verdict={verdict} summary={summary_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
