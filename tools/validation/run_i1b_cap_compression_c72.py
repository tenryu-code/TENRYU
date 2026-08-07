#!/usr/bin/env python3
"""Run the S3 cap C>=7.2 verification gate.

This gate certifies that the pinned TRI-fan cap topology carries an idealized
homologous Noh-class convergent implosion to ICF-class peak compression
C>=7.2 with the mesh geometrically admissible, exact mass conservation, and
<=10% total-energy drift through the C>=7.2 compression target (first crossing)
at N_C=8 on the compatible staggered-Lagrangian pure-Lagrangian path with
CSW-edge AV and subzonal pressure.

It does not certify a physical PAB/ablation drive, post-stagnation/rebound
energy conservation, or fine-grid apex convergence where N_C refinement also
refines apex angular resolution and tightens dt scaling.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_i1b_lite_2d_rz as lite


CASE_NAME = "2d_rz_i1b_cap_compression_c72"
DEFAULT_DECK = Path("examples/verification/2d_rz_i1b_cap_compression_c72.py")
DEFAULT_OUTDIR = Path("build/output_verify_2d_rz_i1b_cap_compression_c72")
NON_POSITIVE_VOLUME_TOKEN = "non-positive 2D RZ cell"

C_PEAK_MIN = 7.2
MASS_REL_MAX = 1.0e-6
ENERGY_C72_REL_MAX = 0.10


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_sanitize(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def resolve_under_repo(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def clean_deck_output(repo_root: Path) -> None:
    outdir = repo_root / DEFAULT_OUTDIR
    parent = outdir.parent
    if not parent.exists():
        return
    for path in [outdir, *parent.glob(f"{outdir.name}_*")]:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def parse_deck_kv(log_text: str) -> dict[str, str]:
    for line in log_text.splitlines():
        if not line.startswith(f"[deck:{CASE_NAME}]"):
            continue
        return {
            match.group(1): match.group(2)
            for match in re.finditer(r"([A-Za-z0-9_]+)=([^ \t]+)", line)
        }
    return {}


def parse_actual_outdir(repo_root: Path, log_text: str) -> Path:
    matches = re.findall(r"^\[TENRYU\] Output directory:\s*(\S+)\s*$", log_text, re.MULTILINE)
    if matches:
        return resolve_under_repo(repo_root, matches[-1])
    kv = parse_deck_kv(log_text)
    if "outdir" in kv:
        return resolve_under_repo(repo_root, kv["outdir"])
    return repo_root / DEFAULT_OUTDIR


def locate_frozen_config(actual_outdir: Path) -> Path | None:
    config_dir = actual_outdir / "config"
    if not config_dir.exists():
        return None
    candidates = sorted(config_dir.glob("*_frozen.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def run_audit_summary(repo_root: Path, actual_outdir: Path, log_path: Path) -> dict[str, Any]:
    run_info = actual_outdir / "run_info.json"
    history = lite.locate_history_file(actual_outdir / "results")
    frozen = locate_frozen_config(actual_outdir)
    audit_path = actual_outdir / "audit_summary.json"
    if not run_info.exists():
        return {"available": False, "audit_status": "RUN_INFO_MISSING", "termination_reason": "run_info_missing"}
    if history is None:
        return {
            "available": False,
            "audit_status": "HISTORY_MISSING",
            "termination_reason": read_json(run_info).get("termination_reason", "unknown"),
        }

    cmd = [
        sys.executable,
        "tools/validation/audit_summary.py",
        "--run-info",
        str(run_info),
        "--history-hdf5",
        str(history),
        "--tier",
        "A",
        "--production-audit-enabled",
        "--require-positivity",
        "--require-gcl",
        "--output",
        str(audit_path),
    ]
    if frozen is not None:
        cmd.extend(["--frozen-config", str(frozen)])
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {
            "available": False,
            "path": str(audit_path),
            "audit_status": "AUDIT_POSTPROCESS_FAILED",
            "audit_returncode": result.returncode,
            "termination_reason": read_json(run_info).get("termination_reason", "unknown"),
        }
    payload = read_json(audit_path)
    payload["available"] = True
    payload["path"] = str(audit_path)
    return payload


def scalar_dataset(handle: Any, path: str) -> float | None:
    import numpy as np

    key = path[1:] if path.startswith("/") else path
    if key not in handle:
        return None
    arr = np.asarray(handle[key][()], dtype=float).reshape(-1)
    if arr.size == 0:
        return None
    value = float(arr[-1])
    return value if math.isfinite(value) else None


def array_dataset(handle: Any, candidates: tuple[str, ...]) -> tuple[str, Any] | tuple[None, None]:
    import numpy as np

    for path in candidates:
        key = path[1:] if path.startswith("/") else path
        if key in handle:
            return path, np.asarray(handle[key][()], dtype=float)
    return None, None


def finite_sum(values: Any) -> float | None:
    import numpy as np

    arr = np.asarray(values, dtype=float).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.sum(finite, dtype=np.float64))


def fallback_total_energy(handle: Any, mass: Any) -> tuple[float | None, str]:
    import numpy as np

    e_internal = scalar_dataset(handle, "time_state/E_internal")
    e_kinetic = scalar_dataset(handle, "time_state/E_kinetic")
    if e_internal is not None and e_kinetic is not None:
        return e_internal + e_kinetic, "time_state/E_internal+time_state/E_kinetic"

    pieces: list[float] = []
    ee_path, ee = array_dataset(handle, ("hydro/ee",))
    ei_path, ei = array_dataset(handle, ("hydro/ei",))
    if ee_path is not None and ei_path is not None:
        mass_arr = np.asarray(mass, dtype=float).reshape(-1)
        ee_arr = np.asarray(ee, dtype=float).reshape(-1)
        ei_arr = np.asarray(ei, dtype=float).reshape(-1)
        if mass_arr.size == ee_arr.size == ei_arr.size:
            internal = mass_arr * (ee_arr + ei_arr)
            total = finite_sum(internal)
            if total is not None:
                pieces.append(total)

    node_mass_path, node_mass = array_dataset(handle, ("hydro/node_mass",))
    vr_path, v_r = array_dataset(handle, ("mesh/v_r",))
    vz_path, v_z = array_dataset(handle, ("mesh/v_z",))
    if node_mass_path is not None and vr_path is not None and vz_path is not None:
        node_mass_arr = np.asarray(node_mass, dtype=float).reshape(-1)
        vr_arr = np.asarray(v_r, dtype=float).reshape(-1)
        vz_arr = np.asarray(v_z, dtype=float).reshape(-1)
        if node_mass_arr.size == vr_arr.size == vz_arr.size:
            kinetic = 0.5 * node_mass_arr * (vr_arr * vr_arr + vz_arr * vz_arr)
            total = finite_sum(kinetic)
            if total is not None:
                pieces.append(total)

    rad_path, rad_e = array_dataset(handle, ("radiation/energy_density",))
    vol_path, vol = array_dataset(handle, ("hydro/vol",))
    if rad_path is not None and vol_path is not None:
        vol_arr = np.asarray(vol, dtype=float).reshape(-1)
        rad_arr = np.asarray(rad_e, dtype=float)
        if rad_arr.shape[0] == vol_arr.size:
            if rad_arr.ndim == 1:
                rad_total = finite_sum(rad_arr * vol_arr)
            else:
                rad_total = finite_sum(rad_arr * vol_arr.reshape((-1,) + (1,) * (rad_arr.ndim - 1)))
            if rad_total is not None:
                pieces.append(rad_total)

    if not pieces:
        return None, "missing_energy"
    return float(sum(pieces)), "computed_from_snapshot_fields"


def read_snapshot(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        r_path, r = array_dataset(handle, ("mesh/x_r",))
        z_path, z = array_dataset(handle, ("mesh/x_z",))
        if r_path is None or z_path is None:
            raise KeyError(f"{path}: missing mesh/x_r or mesh/x_z")
        r_arr = np.asarray(r, dtype=float).reshape(-1)
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        if r_arr.size != z_arr.size:
            raise ValueError(f"{path}: mesh/x_r and mesh/x_z sizes differ")
        radius = np.sqrt(r_arr * r_arr + z_arr * z_arr)
        finite_radius = radius[np.isfinite(radius)]
        if finite_radius.size == 0:
            raise ValueError(f"{path}: no finite nodal radii")
        max_radius = float(np.max(finite_radius))

        mass_path, mass = array_dataset(handle, ("hydro/cell_mass", "hydro/mass"))
        if mass_path is None:
            raise KeyError(f"{path}: missing hydro/cell_mass or hydro/mass")
        mass_total = finite_sum(mass)
        if mass_total is None:
            raise ValueError(f"{path}: no finite cell masses")

        energy = scalar_dataset(handle, "time_state/E_total")
        energy_source = "time_state/E_total"
        if energy is None:
            energy, energy_source = fallback_total_energy(handle, mass)
        if energy is None:
            raise ValueError(f"{path}: could not determine total energy")

        return {
            "path": str(path),
            "step": int(scalar_dataset(handle, "time_state/step") or 0),
            "time_s": float(scalar_dataset(handle, "time_state/t") or 0.0),
            "max_radius_cm": max_radius,
            "mass_g": mass_total,
            "mass_source": mass_path,
            "E_total_erg": energy,
            "energy_source": energy_source,
        }


def analyze_snapshots(results_dir: Path) -> dict[str, Any]:
    plot_files = lite.locate_plot_files(results_dir)
    if not plot_files:
        raise FileNotFoundError(f"no HDF5 plot snapshots found under {results_dir}")

    snapshots = [read_snapshot(path) for path in plot_files]
    r0_max = snapshots[0]["max_radius_cm"]
    if not (math.isfinite(r0_max) and r0_max > 0.0):
        raise ValueError(f"invalid initial maximum radius {r0_max!r}")
    m0 = snapshots[0]["mass_g"]
    if not (math.isfinite(m0) and m0 > 0.0):
        raise ValueError(f"invalid initial mass {m0!r}")
    e0 = snapshots[0]["E_total_erg"]
    if not (math.isfinite(e0) and abs(e0) > 0.0):
        raise ValueError(f"invalid initial total energy {e0!r}")

    for snapshot in snapshots:
        r_now = snapshot["max_radius_cm"]
        snapshot["C"] = r0_max / r_now if r_now > 0.0 else math.inf
        snapshot["mass_rel"] = abs(snapshot["mass_g"] - m0) / m0

    peak = max(snapshots, key=lambda item: item["C"])
    mass_rel = max(snapshot["mass_rel"] for snapshot in snapshots)
    energy_through_peak_rel = abs(peak["E_total_erg"] - e0) / abs(e0)
    c72_crossing = next(
        (snapshot for snapshot in sorted(snapshots, key=lambda item: (item["step"], item["path"])) if snapshot["C"] >= C_PEAK_MIN),
        None,
    )
    energy_at_c72_crossing_rel = (
        abs(c72_crossing["E_total_erg"] - e0) / abs(e0) if c72_crossing is not None else None
    )
    c72_crossing_snapshot = (
        {
            "path": c72_crossing["path"],
            "step": c72_crossing["step"],
            "time_s": c72_crossing["time_s"],
            "C": c72_crossing["C"],
            "E_over_E0": c72_crossing["E_total_erg"] / e0,
            "E_total_erg": c72_crossing["E_total_erg"],
            "energy_source": c72_crossing["energy_source"],
        }
        if c72_crossing is not None
        else None
    )

    return {
        "results_dir": str(results_dir),
        "snapshot_count": len(snapshots),
        "r0_max_cm": r0_max,
        "M0_g": m0,
        "E0_erg": e0,
        "C_peak": peak["C"],
        "mass_rel": mass_rel,
        "energy_at_c72_crossing_rel": energy_at_c72_crossing_rel,
        "energy_through_peak_rel": energy_through_peak_rel,
        "c72_crossing_snapshot": c72_crossing_snapshot,
        "peak_snapshot": peak,
        "mass_source": snapshots[0]["mass_source"],
        "energy_source": peak["energy_source"],
    }


def run_deck(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    log_dir = resolve_under_repo(repo_root, args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "run.log"
    deck_path = resolve_under_repo(repo_root, args.deck)
    cmd = [str(args.tenryu_bin), "run", str(deck_path)]
    env = os.environ.copy()
    clean_deck_output(repo_root)
    start = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        log.write("[harness:i1b_cap_compression_c72] command=" + " ".join(cmd) + "\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=cmd, returncode=-1)
    return {
        "log_path": log_path,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": time.monotonic() - start,
    }


def build_verdict(args: argparse.Namespace, run: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    log_text = read_text(run["log_path"])
    deck_kv = parse_deck_kv(log_text)
    actual_outdir = parse_actual_outdir(repo_root, log_text)
    deck_outdir = resolve_under_repo(repo_root, deck_kv.get("outdir", DEFAULT_OUTDIR))
    run_info = read_json(actual_outdir / "run_info.json")
    termination_reason = str(run_info.get("termination_reason", "unknown"))
    audit_json = run_audit_summary(repo_root, actual_outdir, run["log_path"])
    audit_status = str(audit_json.get("audit_status", "unknown"))
    admissible = (
        run["exit_code"] == 0
        and not run["timed_out"]
        and "[TENRYU] Run completed." in log_text
        and NON_POSITIVE_VOLUME_TOKEN.lower() not in log_text.lower()
        and termination_reason == "t_end_reached"
        and audit_status == "TIER_A_PASS"
    )

    metrics: dict[str, Any] = {}
    analysis_error = None
    try:
        results_dir = lite.discover_results_dir(actual_outdir)
        metrics = analyze_snapshots(results_dir)
    except Exception as exc:  # noqa: BLE001 - preserve failure reason in verdict JSON.
        analysis_error = f"{type(exc).__name__}: {exc}"

    c_peak = metrics.get("C_peak")
    mass_rel = metrics.get("mass_rel")
    energy_c72_rel = metrics.get("energy_at_c72_crossing_rel")
    energy_peak_rel = metrics.get("energy_through_peak_rel")
    checks = {
        "admissible": bool(admissible),
        "C_peak_ge_7p2": isinstance(c_peak, (int, float)) and c_peak >= C_PEAK_MIN,
        "mass_rel_le_1e_6": isinstance(mass_rel, (int, float)) and mass_rel <= MASS_REL_MAX,
        "energy_at_c72_crossing_rel_le_0p10": (
            isinstance(energy_c72_rel, (int, float))
            and energy_c72_rel <= ENERGY_C72_REL_MAX
        ),
    }
    all_checks_passed = all(checks.values())

    return {
        "schema_version": "tenryu.i1b_cap_compression_c72.verdict.v2",
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "case_name": CASE_NAME,
        "deck_path": str(resolve_under_repo(repo_root, args.deck)),
        "tenryu_bin": str(args.tenryu_bin),
        "exit_code": run["exit_code"],
        "timed_out": run["timed_out"],
        "wall_time_s": run["wall_time_s"],
        "log_path": str(run["log_path"]),
        "actual_outdir": str(actual_outdir),
        "deck_outdir": str(deck_outdir),
        "deck_banner": deck_kv,
        "termination_reason": termination_reason,
        "audit_json": audit_json,
        "audit_status": audit_status,
        "admissible": admissible,
        "C_peak": c_peak,
        "mass_rel": mass_rel,
        "energy_at_c72_crossing_rel": energy_c72_rel,
        "energy_through_peak_rel": energy_peak_rel,
        "all_checks_passed": all_checks_passed,
        "checks": checks,
        "thresholds": {
            "C_peak_min": C_PEAK_MIN,
            "mass_rel_max": MASS_REL_MAX,
            "energy_at_c72_crossing_rel_max": ENERGY_C72_REL_MAX,
            "energy_through_peak_rel_asserted": False,
        },
        "metric_definition": {
            "compression": "r0_max / max_node_hypot(x_r,x_z), matching compression_reached in the barrier-cycle test",
            "mass_rel": "max over snapshots |sum(cell_mass)-M0|/M0; accepts hydro/cell_mass or hydro/mass",
            "energy_at_c72_crossing_rel": "|E_total-E0|/|E0| at the FIRST snapshot reaching C>=7.2; measured at the compression target, before the run-sensitive near-stagnation/rebound region",
            "energy_through_peak_rel": "|E_total(t_peak)-E_total(t0)|/|E_total(t0)| at the deepest-compression snapshot with C_peak; diagnostic only, not asserted",
            "admissible": "exit code 0, Run completed, t_end_reached, TIER_A_PASS audit, and no non-positive 2D RZ cell line in the combined log",
        },
        "metrics": metrics,
        "analysis_error": analysis_error,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", required=True, help="Path to the tenryu executable")
    parser.add_argument("--deck", default=str(DEFAULT_DECK), help="Verification deck to run")
    parser.add_argument("--repo-root", default=".", help="Repository root used as the run working directory")
    parser.add_argument(
        "--log-dir",
        default="build/i1b_cap_compression_c72",
        help="Directory for harness logs",
    )
    parser.add_argument(
        "--verdict-json",
        default="build/i1b_cap_compression_c72/verdict.json",
        help="Path to write the machine-readable verdict JSON",
    )
    parser.add_argument("--wall-time-s", type=float, default=1800.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run = run_deck(args)
    verdict = build_verdict(args, run)
    verdict_path = resolve_under_repo(Path(args.repo_root).resolve(), args.verdict_json)
    write_json(verdict_path, verdict)
    print(json.dumps(json_sanitize(verdict), sort_keys=True))
    return 0 if verdict["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
