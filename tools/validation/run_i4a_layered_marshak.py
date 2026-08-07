#!/usr/bin/env python3
"""I4a 2D RZ layered-Marshak FLD validation harness.

Tiers:
  smoke (dev-time, default): SHORT-WINDOW MECHANISM CHECK. Runs the layered deck for
    t_end=1e-13 s (~3.5e3 dt-limited steps, tens of seconds) and gates on the BUG-14
    defect class: boundary-cell Te ignition, radiative front formed but still short of
    the interface, per-step energy-ledger closure (operator audit), and both layered
    materials materialized in state. No reference generator, no profile comparison.
  cert (campaign): full registry window (t_end=3e-10 s) with the reference-generator
    profile gates (eps_L2_E, interface flux continuity, mass conservation) over
    k in {3,10,30}. The deck is dt-limiter-bound (dt ~3e-17 s at the driven boundary),
    so cert is POST-CODE-COMPLETION CAMPAIGN tier — do not run it as a dev-time check.
    See docs/design/i4_mm_rad_interface_spec.md Addendum 5.
  cert --tier pre-crossing: pre-interface Tier 2 profile comparison over the
    boundary-trimmed front window.
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
import rz_profile_average as rz_average  # noqa: E402


LADDER_ROW = "i4a_layered_marshak_2d_rz_slab"
EPS_L2_E_GATE = 2.0e-2
FLUX_CONT_DIFF_GATE = 1.0e-3
MASS_CONSERVATION_GATE = 1.0e-12
PRE_CROSSING_EPS_2W_GATE = 0.12
PRE_CROSSING_EXCLUDED_BOUNDARY_CELLS = 2
PRE_CROSSING_FRONT_PAD_CELLS = 10
SMOKE_T_END_S = 1.0e-13
CERT_T_END_S = 3.0e-10
SMOKE_TE_BOUNDARY_MIN_EV = 50.0
SMOKE_FRONT_TE_EV = 1.0
SMOKE_LEDGER_EPS_MAX = 1.0e-10
TINY = 1.0e-300
C_LIGHT = 2.99792458e10


@dataclass(frozen=True)
class I4Snapshot:
    path: Path
    t: float
    step: int
    Te: Any
    vol: Any
    rad_E: Any
    r_nodes: Any
    z_nodes: Any
    mass_per_material: Any | None
    vol_frac: Any | None
    warnings: tuple[str, ...]


def default_workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "tmp" / "i4a_layered_marshak" / stamp


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


def _read_grouped_field(
    handle: Any,
    path: str,
    nr: int,
    nz: int,
    groups: int,
) -> np.ndarray:
    n_cells = nr * nz
    arr = np.asarray(handle[path][()], dtype=float)
    if arr.shape == (nr, nz, groups):
        return arr
    if groups == 1 and arr.shape == (nr, nz):
        return arr.reshape((nr, nz, 1))
    if arr.shape == (n_cells, groups):
        return arr.reshape((nr, nz, groups))
    if arr.size == n_cells * groups:
        return arr.reshape((nr, nz, groups))
    raise ValueError(
        f"{path}: expected shape ({nr}, {nz}, {groups}) or ({n_cells}, {groups}), got {arr.shape}"
    )


def _read_optional_mass_per_material(
    handle: Any,
    nr: int,
    nz: int,
    n_mat: int,
    warnings: list[str],
) -> np.ndarray | None:
    path = "hydro/per_material/v1/mass"
    if path not in handle:
        warnings.append("missing optional dataset hydro/per_material/v1/mass")
        return None
    n_cells = nr * nz
    arr = np.asarray(handle[path][()], dtype=float)
    if arr.shape == (n_cells, n_mat):
        return arr
    if arr.size == n_cells * n_mat:
        return arr.reshape((n_cells, n_mat))
    raise ValueError(f"{path}: expected shape ({n_cells}, {n_mat}), got {arr.shape}")


def _read_optional_vol_frac(
    handle: Any,
    nr: int,
    nz: int,
    warnings: list[str],
) -> np.ndarray | None:
    path = "hydro/volFrac"
    if path not in handle:
        warnings.append("missing optional dataset hydro/volFrac")
        return None
    arr = np.asarray(handle[path][()], dtype=float)
    n_cells = nr * nz
    if arr.ndim == 3 and arr.shape[:2] == (nr, nz):
        return arr
    if arr.ndim == 2 and arr.shape[0] == n_cells:
        return arr.reshape((nr, nz, arr.shape[1]))
    if arr.size % n_cells == 0:
        return arr.reshape((nr, nz, arr.size // n_cells))
    raise ValueError(f"{path}: cannot map shape {arr.shape} onto ({nr}, {nz}, n_mat)")


def read_snapshot(path: Path, nr: int, nz: int, groups: int = 1, n_mat: int = 2) -> I4Snapshot:
    import h5py

    warnings: list[str] = []
    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = base.scalar_dataset(handle, "time_state/t", t_value)
        rad_E_groups = _read_grouped_field(
            handle, "radiation/energy_density", nr, nz, groups
        )
        return I4Snapshot(
            path=path,
            t=t_value,
            step=int(round(base.scalar_dataset(handle, "time_state/step", 0.0))),
            Te=base.reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te", "2D_RZ"),
            vol=base.reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol", "2D_RZ"),
            rad_E=np.sum(rad_E_groups, axis=2),
            r_nodes=base.reshape_nodes(handle["mesh/x_r"][()], nr, nz, "mesh/x_r", "2D_RZ"),
            z_nodes=base.reshape_nodes(handle["mesh/x_z"][()], nr, nz, "mesh/x_z", "2D_RZ"),
            mass_per_material=_read_optional_mass_per_material(handle, nr, nz, n_mat, warnings),
            vol_frac=_read_optional_vol_frac(handle, nr, nz, warnings),
            warnings=tuple(warnings),
        )


def read_snapshots(outdir: Path, nr: int, nz: int) -> list[I4Snapshot]:
    actual = base.discover_actual_outdir(outdir)
    results_dir = actual / "results"
    candidates = sorted(results_dir.glob("*.h5")) + sorted(actual.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"no HDF5 snapshots under {results_dir} or {actual}")
    snapshots: list[I4Snapshot] = []
    errors: list[str] = []
    for path in candidates:
        try:
            snapshots.append(read_snapshot(path, nr, nz))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if not snapshots:
        tail = "; ".join(errors[-3:])
        raise RuntimeError(f"no readable snapshots under {actual}: {tail}")
    return sorted(snapshots, key=lambda snap: (snap.t, snap.step, str(snap.path)))


def profile(snapshot: I4Snapshot, field: Any) -> np.ndarray:
    reduced = rz_average.reduce_2d_rz_to_z_profile(np.asarray(field, dtype=float), snapshot)
    return np.asarray(reduced["profile_z"], dtype=float)


def rel_l2(candidate: Any, reference: Any) -> float:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = cand - ref
    return float(np.linalg.norm(diff) / max(np.linalg.norm(ref), TINY))


def te_front_index(te_profile: Any, threshold_eV: float = SMOKE_FRONT_TE_EV) -> int:
    heated = np.nonzero(np.asarray(te_profile, dtype=float) >= threshold_eV)[0]
    return int(heated.max()) if heated.size else -1


def interface_continuity(profile_z: Any, iface: int, dz: float, kappa1: float, kappa2: float) -> dict[str, float]:
    E = np.asarray(profile_z, dtype=float)
    if iface < 2 or iface + 1 >= E.size:
        raise ValueError(f"interface face index {iface} is not usable for one-sided two-point slopes")
    D1 = C_LIGHT / (3.0 * kappa1)
    D2 = C_LIGHT / (3.0 * kappa2)
    F_L = -D1 * (E[iface - 1] - E[iface - 2]) / dz
    F_R = -D2 * (E[iface + 1] - E[iface]) / dz
    continuity = abs(F_L - F_R) / max(abs(F_L), abs(F_R), TINY)
    return {
        "F_L": float(F_L),
        "F_R": float(F_R),
        "continuity": float(continuity),
    }


def mass_conservation(first: I4Snapshot, final: I4Snapshot) -> dict[str, Any]:
    if first.mass_per_material is None or final.mass_per_material is None:
        return {"status": "not_available", "relative": None}
    initial = np.sum(np.asarray(first.mass_per_material, dtype=float), axis=0)
    ending = np.sum(np.asarray(final.mass_per_material, dtype=float), axis=0)
    rel = np.abs(ending - initial) / np.maximum(np.abs(initial), TINY)
    return {
        "status": "available",
        "relative": float(np.max(rel)) if rel.size else 0.0,
        "initial_g": [float(v) for v in initial],
        "final_g": [float(v) for v in ending],
    }


def parse_ledger_eps(log_path: Path) -> float | None:
    """Return eps from the LAST [wj_step_audit] line of the deck log, or None."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    eps_value: float | None = None
    for line in text.splitlines():
        if "[wj_step_audit]" not in line:
            continue
        for token in line.split():
            if token.startswith("eps="):
                try:
                    eps_value = float(token[len("eps="):])
                except ValueError:
                    pass
    return eps_value


def deck_snap_times(t_end_s: float) -> list[float]:
    interval = t_end_s / 5.0
    return [0.0] + [float(interval * i) for i in range(1, 6)]


def generator_n_steps(t_end_s: float, dt_s: float) -> int:
    if t_end_s <= 0.0 or dt_s <= 0.0:
        raise ValueError("--t-end-s and --dt-s must be positive")
    # Reference runs at its own validated time resolution (>= 3000 steps; S4 dt-insensitivity 2.6e-5), NOT the deck dt.
    n_steps = max(3000, int(round(t_end_s / dt_s)))
    if n_steps <= 0:
        raise ValueError("computed generator n_steps must be positive")
    return n_steps


def run_generator(
    args: argparse.Namespace,
    leg_dir: Path,
    kappa1: float,
    kappa2: float,
    n_steps: int,
    snap_times: list[float],
) -> dict[str, Any]:
    ref_path = leg_dir / "ref.json"
    log_path = leg_dir / "refgen.log"
    cmd = [
        "python3",
        str(args.generator),
        "--L-cm",
        str(args.L_cm),
        "--nz",
        str(args.nz),
        "--interface-z-cm",
        str(0.5 * args.L_cm),
        "--kappa1",
        f"{kappa1:.17g}",
        "--kappa2",
        f"{kappa2:.17g}",
        "--rho",
        "1.0",
        "--cv-e-volume",
        "5.488e2",
        "--T-init-eV",
        "1.0e-3",
        "--t-end-s",
        f"{args.t_end_s:.17g}",
        "--n-steps",
        str(n_steps),
        "--snap-times-s",
        ",".join(f"{t:.17g}" for t in snap_times),
        "--output",
        str(ref_path),
    ]
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.monotonic() - t0
    return {
        "returncode": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "cmd": cmd,
        "log": str(log_path),
        "path": str(ref_path),
    }


def run_deck(
    args: argparse.Namespace,
    outdir: Path,
    kappa1: float,
    kappa2: float,
) -> dict[str, Any]:
    env_extra = {
        "TENRYU_I4A_NR": str(args.nr),
        "TENRYU_I4A_NZ": str(args.nz),
        "TENRYU_I4A_L_CM": str(args.L_cm),
        "TENRYU_I4A_KAPPA1": f"{kappa1:.17g}",
        "TENRYU_I4A_KAPPA2": f"{kappa2:.17g}",
        "TENRYU_I4A_T_END_S": f"{args.t_end_s:.17g}",
        "TENRYU_I4A_DT_S": f"{args.dt_s:.17g}",
        "TENRYU_I4A_OUTDIR": str(outdir),
        "TENRYU_ENERGY_OPERATOR_AUDIT": "1",
    }
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


def compute_metrics(
    args: argparse.Namespace,
    first: I4Snapshot,
    final: I4Snapshot,
    reference: dict[str, Any],
    kappa1: float,
    kappa2: float,
) -> dict[str, Any]:
    ref_final = reference["snaps"][-1]
    deck_E = profile(final, final.rad_E)
    deck_Te = profile(final, final.Te)
    ref_E = np.asarray(ref_final["E"], dtype=float)
    ref_Te = np.asarray(ref_final["Te"], dtype=float)
    if deck_E.shape != ref_E.shape:
        raise ValueError(f"E profile shape mismatch: deck {deck_E.shape}, reference {ref_E.shape}")
    if deck_Te.shape != ref_Te.shape:
        raise ValueError(f"Te profile shape mismatch: deck {deck_Te.shape}, reference {ref_Te.shape}")
    iface = int(reference.get("interface_face_index", args.nz // 2))
    dz = args.L_cm / float(args.nz)
    deck_flux = interface_continuity(deck_E, iface, dz, kappa1, kappa2)
    ref_flux = interface_continuity(ref_E, iface, dz, kappa1, kappa2)
    mass = mass_conservation(first, final)
    flux_cont_diff = abs(deck_flux["continuity"] - ref_flux["continuity"])
    return {
        "eps_L2_E": rel_l2(deck_E, ref_E),
        "eps_L2_Te": rel_l2(deck_Te, ref_Te),
        "interface_face_index": iface,
        "dz_cm": dz,
        "flux_deck": deck_flux,
        "flux_ref": ref_flux,
        "flux_cont_deck": deck_flux["continuity"],
        "flux_cont_ref": ref_flux["continuity"],
        "flux_cont_diff": float(flux_cont_diff),
        "mass_per_material": mass,
    }


def compute_pre_crossing_metrics(
    args: argparse.Namespace,
    first: I4Snapshot,
    final: I4Snapshot,
    reference: dict[str, Any],
    deck_log: Path,
) -> dict[str, Any]:
    ref_final = reference["snaps"][-1]
    deck_E = profile(final, final.rad_E)
    deck_Te = profile(final, final.Te)
    ref_E = np.asarray(ref_final["E"], dtype=float)
    ref_Te = np.asarray(ref_final["Te"], dtype=float)
    if deck_E.shape != ref_E.shape:
        raise ValueError(f"E profile shape mismatch: deck {deck_E.shape}, reference {ref_E.shape}")
    if deck_Te.shape != ref_Te.shape:
        raise ValueError(f"Te profile shape mismatch: deck {deck_Te.shape}, reference {ref_Te.shape}")

    iface = int(reference.get("interface_face_index", args.nz // 2))
    front = te_front_index(deck_Te)
    window_start = PRE_CROSSING_EXCLUDED_BOUNDARY_CELLS
    window_stop = min(deck_E.size, front + PRE_CROSSING_FRONT_PAD_CELLS + 1)
    if window_stop <= window_start:
        raise ValueError(
            f"pre-crossing L2 window is empty: start={window_start} stop={window_stop}"
        )
    excluded = list(range(PRE_CROSSING_EXCLUDED_BOUNDARY_CELLS))
    mass = mass_conservation(first, final)
    ledger_eps = parse_ledger_eps(deck_log)
    return {
        "tier": "pre_crossing",
        "epsilon_2W_E": rel_l2(deck_E[window_start:window_stop], ref_E[window_start:window_stop]),
        "epsilon_2W_Te": rel_l2(deck_Te[window_start:window_stop], ref_Te[window_start:window_stop]),
        "epsilon_2W_gate": PRE_CROSSING_EPS_2W_GATE,
        "window_start_index": int(window_start),
        "window_stop_index_exclusive": int(window_stop),
        "window_end_index": int(window_stop - 1),
        "front_pad_cells": PRE_CROSSING_FRONT_PAD_CELLS,
        "excluded_boundary_cell_count": PRE_CROSSING_EXCLUDED_BOUNDARY_CELLS,
        "excluded_boundary_cells": {
            "indices": excluded,
            "deck_E": [float(v) for v in deck_E[excluded]],
            "ref_E": [float(v) for v in ref_E[excluded]],
            "deck_Te": [float(v) for v in deck_Te[excluded]],
            "ref_Te": [float(v) for v in ref_Te[excluded]],
        },
        "front_index": int(front),
        "front_te_threshold_eV": SMOKE_FRONT_TE_EV,
        "interface_face_index": iface,
        "mass_per_material": mass,
        "ledger_eps": ledger_eps,
        "ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
    }


def failed_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if float(metrics["eps_L2_E"]) > EPS_L2_E_GATE:
        failed.append("eps_L2_E")
    if float(metrics["flux_cont_diff"]) > FLUX_CONT_DIFF_GATE:
        failed.append("interface_flux_continuity")
    mass = metrics["mass_per_material"]
    if mass["status"] == "available" and float(mass["relative"]) > MASS_CONSERVATION_GATE:
        failed.append("mass_per_material")
    return failed


def failed_pre_crossing_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if float(metrics["epsilon_2W_E"]) > PRE_CROSSING_EPS_2W_GATE:
        failed.append("epsilon_2W_E")
    if float(metrics["epsilon_2W_Te"]) > PRE_CROSSING_EPS_2W_GATE:
        failed.append("epsilon_2W_Te")
    mass = metrics["mass_per_material"]
    if mass["status"] == "available" and float(mass["relative"]) > MASS_CONSERVATION_GATE:
        failed.append("mass_per_material")
    eps = metrics["ledger_eps"]
    if eps is None or not math.isfinite(float(eps)) or float(eps) > SMOKE_LEDGER_EPS_MAX:
        failed.append("ledger_closure")
    if not (int(metrics["front_index"]) < int(metrics["interface_face_index"])):
        failed.append("front_before_interface")
    return failed


def smoke_mechanism_metrics(
    args: argparse.Namespace,
    first: I4Snapshot,
    final: I4Snapshot,
    deck_log: Path,
) -> dict[str, Any]:
    te_profile = profile(final, final.Te)
    iface = args.nz // 2
    te_boundary = float(te_profile[0])
    front_reach = te_front_index(te_profile)
    ledger_eps = parse_ledger_eps(deck_log)
    mass = mass_conservation(first, final)
    vf = final.vol_frac
    materials_layered = False
    material_share_region1: list[float] | None = None
    material_share_region2: list[float] | None = None
    dominant_region1 = -1
    dominant_region2 = -1
    if vf is not None:
        vf_arr = np.asarray(vf, dtype=float)
        vol3 = np.asarray(final.vol, dtype=float)[:, :, None]
        weighted = vf_arr * vol3
        share1 = np.sum(weighted[:, :iface, :], axis=(0, 1))
        share2 = np.sum(weighted[:, iface:, :], axis=(0, 1))
        material_share_region1 = [float(v) for v in share1]
        material_share_region2 = [float(v) for v in share2]
        if share1.size >= 2 and float(share1.sum()) > 0.0 and float(share2.sum()) > 0.0:
            dominant_region1 = int(np.argmax(share1))
            dominant_region2 = int(np.argmax(share2))
            materials_layered = bool(
                dominant_region1 != dominant_region2
                and share1[dominant_region1] > 0.0
                and share2[dominant_region2] > 0.0
            )
    return {
        "tier": "dev_smoke_mechanism",
        "interface_face_index": iface,
        "te_boundary_eV": te_boundary,
        "te_boundary_min_eV": SMOKE_TE_BOUNDARY_MIN_EV,
        "front_reach_index": front_reach,
        "front_te_threshold_eV": SMOKE_FRONT_TE_EV,
        "ledger_eps": ledger_eps,
        "ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
        "mass_per_material": mass,
        "materials_layered": materials_layered,
        "material_share_region1": material_share_region1,
        "material_share_region2": material_share_region2,
        "dominant_material_region1": dominant_region1,
        "dominant_material_region2": dominant_region2,
    }


def failed_smoke_gates(metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    if float(metrics["te_boundary_eV"]) < SMOKE_TE_BOUNDARY_MIN_EV:
        failed.append("te_boundary_ignition")
    front = int(metrics["front_reach_index"])
    if not (1 <= front < int(metrics["interface_face_index"])):
        failed.append("te_front_formed_before_interface")
    eps = metrics["ledger_eps"]
    if eps is None or not math.isfinite(float(eps)) or float(eps) > SMOKE_LEDGER_EPS_MAX:
        failed.append("ledger_closure")
    if not metrics["materials_layered"]:
        failed.append("materials_layered")
    return failed


def format_float(value: Any) -> str:
    if value == "not_available":
        return "not_available"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{fvalue:.6e}" if math.isfinite(fvalue) else "missing"


def format_failed_gates(gates: list[str]) -> str:
    return ",".join(gates) if gates else "none"


def selected_tier(args: argparse.Namespace) -> str:
    if args.mode == "smoke":
        return "dev_smoke_mechanism"
    return "pre_crossing" if args.tier == "pre-crossing" else "campaign_cert"


def run_leg(args: argparse.Namespace, k_value: int, n_steps: int, snap_times: list[float]) -> dict[str, Any]:
    label = f"k{k_value}"
    leg_dir = Path(args.workdir) / label
    outdir = leg_dir / "out"
    leg_dir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)
    kappa1 = 30.0
    kappa2 = 30.0 / float(k_value)
    result: dict[str, Any] = {
        "label": label,
        "k": int(k_value),
        "kappa1": kappa1,
        "kappa2": kappa2,
    }

    if args.mode == "cert":
        generator = run_generator(args, leg_dir, kappa1, kappa2, n_steps, snap_times)
        result["generator"] = generator
        if generator["returncode"] != 0:
            result["failed"] = True
            result["error"] = f"generator returned rc={generator['returncode']}"
            return result

    deck = run_deck(args, outdir, kappa1, kappa2)
    result["deck"] = deck
    if deck["returncode"] != 0:
        result["failed"] = True
        result["error"] = f"deck returned rc={deck['returncode']}"
        return result
    if not deck["t_end_reached"]:
        result["failed"] = True
        result["error"] = f"deck termination_reason={deck['termination_reason']}"
        return result

    if args.mode == "cert":
        reference = base.read_json(Path(generator["path"]))
        snapshots = read_snapshots(outdir, args.nr, args.nz)
        first = snapshots[0]
        final = snapshots[-1]
        if args.tier == "pre-crossing":
            metrics = compute_pre_crossing_metrics(args, first, final, reference, Path(deck["log"]))
            gates = failed_pre_crossing_gates(metrics)
            cert_gate_passed = len(gates) == 0
            leg_verdict = "PASS" if cert_gate_passed else "FAIL"

            mass_value: Any = (
                metrics["mass_per_material"]["relative"]
                if metrics["mass_per_material"]["status"] == "available"
                else "not_available"
            )
            if mass_value == "not_available":
                print(
                    f"[i4a] WARNING: leg={label} mass_per_material dataset not available; "
                    "mass conservation gate skipped",
                    file=sys.stderr,
                )
            print(
                "[i4a] "
                f"leg={label} tier=pre_crossing "
                f"epsilon_2W_E={metrics['epsilon_2W_E']:.6e} "
                f"epsilon_2W_Te={metrics['epsilon_2W_Te']:.6e} "
                f"front_index={metrics['front_index']} "
                f"iface={metrics['interface_face_index']} "
                f"ledger_eps={format_float(metrics['ledger_eps'])} "
                f"mass_cons={format_float(mass_value)} "
                f"failed_gates={format_failed_gates(gates)} "
                f"verdict={leg_verdict}"
            )

            result.update(
                {
                    "first_snapshot_path": str(first.path),
                    "final_snapshot_path": str(final.path),
                    "t_final": float(final.t),
                    "final_step": int(final.step),
                    "snapshot_count": len(snapshots),
                    "warnings": sorted(set(first.warnings + final.warnings)),
                    "tier": "pre_crossing",
                    "metrics": metrics,
                    "failed_gates": gates,
                    "cert_gate_passed": cert_gate_passed,
                    "verdict": leg_verdict,
                }
            )
            return result

        metrics = compute_metrics(args, first, final, reference, kappa1, kappa2)
        gates = failed_gates(metrics)
        cert_gate_passed = len(gates) == 0
        leg_verdict = "PASS" if cert_gate_passed else "FAIL"

        mass_value: Any = (
            metrics["mass_per_material"]["relative"]
            if metrics["mass_per_material"]["status"] == "available"
            else "not_available"
        )
        if mass_value == "not_available":
            print(
                f"[i4a] WARNING: leg={label} mass_per_material dataset not available; "
                "mass conservation gate skipped",
                file=sys.stderr,
            )
        print(
            "[i4a] "
            f"leg={label} "
            f"eps_L2_E={metrics['eps_L2_E']:.6e} "
            f"flux_cont_deck={metrics['flux_cont_deck']:.6e} "
            f"flux_cont_ref={metrics['flux_cont_ref']:.6e} "
            f"flux_cont_diff={metrics['flux_cont_diff']:.6e} "
            f"mass_cons={format_float(mass_value)} "
            f"failed_gates={format_failed_gates(gates)} "
            f"verdict={leg_verdict}"
        )

        result.update(
            {
                "first_snapshot_path": str(first.path),
                "final_snapshot_path": str(final.path),
                "t_final": float(final.t),
                "final_step": int(final.step),
                "snapshot_count": len(snapshots),
                "warnings": sorted(set(first.warnings + final.warnings)),
                "tier": "campaign_cert",
                "metrics": metrics,
                "failed_gates": gates,
                "cert_gate_passed": cert_gate_passed,
                "verdict": leg_verdict,
            }
        )
        return result

    snapshots = read_snapshots(outdir, args.nr, args.nz)
    first = snapshots[0]
    final = snapshots[-1]
    metrics = smoke_mechanism_metrics(args, first, final, Path(deck["log"]))
    gates = failed_smoke_gates(metrics)
    leg_verdict = "PASS" if not gates else "FAIL"
    print(
        "[i4a] "
        f"leg={label} tier=smoke "
        f"te_boundary_eV={metrics['te_boundary_eV']:.6e} "
        f"front_reach={metrics['front_reach_index']} "
        f"iface={metrics['interface_face_index']} "
        f"ledger_eps={format_float(metrics['ledger_eps'])} "
        f"materials_layered={metrics['materials_layered']} "
        f"verdict={leg_verdict}"
    )

    result.update(
        {
            "first_snapshot_path": str(first.path),
            "final_snapshot_path": str(final.path),
            "t_final": float(final.t),
            "final_step": int(final.step),
            "snapshot_count": len(snapshots),
            "warnings": sorted(set(first.warnings + final.warnings)),
            "tier": "dev_smoke_mechanism",
            "metrics": metrics,
            "failed_gates": gates,
            "smoke_gate_passed": len(gates) == 0,
            "verdict": leg_verdict,
        }
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO_ROOT / "build" / "tenryu"))
    parser.add_argument(
        "--deck",
        default=str(REPO_ROOT / "examples" / "verification" / "i4a_layered_marshak_2d_rz_slab.py"),
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mode", choices=("smoke", "cert"), default="smoke")
    parser.add_argument("--tier", choices=("pre-crossing",), default=None)
    parser.add_argument("--nr", type=int, default=8)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--L-cm", dest="L_cm", type=float, default=1.0)
    parser.add_argument(
        "--t-end-s",
        type=float,
        default=None,
        help="simulated end time; default: 1e-13 (smoke) / 3e-10 (cert)",
    )
    parser.add_argument("--dt-s", type=float, default=3.33e-12)
    parser.add_argument(
        "--generator",
        default=str(REPO_ROOT / "tools" / "validation" / "layered_marshak_reference_generator.py"),
    )
    parser.add_argument("--k-values", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tier is not None and args.mode != "cert":
        parser.error("--tier is only meaningful with --mode cert")
    if args.t_end_s is None:
        args.t_end_s = SMOKE_T_END_S if args.mode == "smoke" else CERT_T_END_S
    if args.nr <= 0 or args.nz <= 3:
        parser.error("--nr must be positive and --nz must be greater than 3")
    if args.t_end_s <= 0.0 or args.dt_s <= 0.0:
        parser.error("--t-end-s and --dt-s must be positive")

    args.workdir = str(default_workdir() if args.workdir is None else Path(args.workdir))
    if args.k_values is None:
        k_values = [3] if args.mode == "smoke" else [3, 10, 30]
    else:
        try:
            k_values = [int(k_value.strip()) for k_value in args.k_values.split(",")]
        except ValueError:
            parser.error("--k-values must be comma-separated integers greater than 1")
        if not k_values or any(k_value <= 1 for k_value in k_values):
            parser.error("--k-values must be comma-separated integers greater than 1")
    if args.mode == "cert":
        n_steps = generator_n_steps(args.t_end_s, args.dt_s)
        snap_times = deck_snap_times(args.t_end_s)
    else:
        n_steps = 0
        snap_times = []
    summary_path = Path(args.workdir) / "summary.json"
    runs: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": args.mode,
        "tier": selected_tier(args),
        "binary": str(args.binary),
        "deck": str(args.deck),
        "generator": str(args.generator),
        "workdir": str(args.workdir),
        "k_values": k_values,
        "nr": int(args.nr),
        "nz": int(args.nz),
        "params": {"L_cm": float(args.L_cm)},
        "t_end_s": float(args.t_end_s),
        "dt_s": float(args.dt_s),
        "generator_n_steps": None if args.mode == "smoke" else int(n_steps),
        "generator_dt_effective_s": None if args.mode == "smoke" else float(args.t_end_s / n_steps),
        "snap_times_s": snap_times,
        "eps_L2_E_gate": EPS_L2_E_GATE,
        "flux_cont_diff_gate": FLUX_CONT_DIFF_GATE,
        "pre_crossing_epsilon_2W_gate": PRE_CROSSING_EPS_2W_GATE,
        "pre_crossing_excluded_boundary_cells": PRE_CROSSING_EXCLUDED_BOUNDARY_CELLS,
        "pre_crossing_front_pad_cells": PRE_CROSSING_FRONT_PAD_CELLS,
        "mass_conservation_gate": MASS_CONSERVATION_GATE,
        "smoke_te_boundary_min_eV": SMOKE_TE_BOUNDARY_MIN_EV,
        "smoke_front_te_eV": SMOKE_FRONT_TE_EV,
        "smoke_ledger_eps_max": SMOKE_LEDGER_EPS_MAX,
        "runs": {},
        "errors": [],
    }
    exit_code = 1
    verdict = "FAIL"

    try:
        for k_value in k_values:
            run = run_leg(args, k_value, n_steps, snap_times)
            runs[f"k{k_value}"] = run
            if run.get("failed"):
                raise RuntimeError(str(run.get("error", f"k{k_value} failed")))

        if args.mode == "cert":
            cert_pass = all(bool(run.get("cert_gate_passed")) for run in runs.values())
            summary["failed_gates"] = sorted(
                {
                    gate
                    for run in runs.values()
                    for gate in run.get("failed_gates", [])
                }
            )
            summary["cert_gate_passed"] = cert_pass
            verdict = "PASS" if cert_pass else "FAIL"
            exit_code = 0 if cert_pass else 1
        else:
            smoke_pass = all(bool(run.get("smoke_gate_passed")) for run in runs.values())
            summary["smoke_gate_passed"] = smoke_pass
            verdict = "PASS" if smoke_pass else "FAIL"
            exit_code = 0 if smoke_pass else 2
    except Exception as exc:
        summary["errors"].append(str(exc))
        verdict = "FAIL"
        exit_code = 1
    finally:
        if args.mode == "cert" and "failed_gates" not in summary:
            summary["failed_gates"] = sorted(
                {
                    gate
                    for run in runs.values()
                    for gate in run.get("failed_gates", [])
                }
            )
        summary["verdict"] = verdict
        summary["all_checks_passed"] = exit_code == 0
        summary["runs"] = {label: sanitize(run) for label, run in runs.items()}
        write_summary(summary_path, summary)
        print(f"[i4a] verdict={verdict} summary={summary_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
