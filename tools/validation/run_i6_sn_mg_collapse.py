#!/usr/bin/env python3
"""I6 2D RZ multigroup S_N grey-collapse battery harness."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
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
import run_i1_2d_rz_fld_ced as i1  # noqa: E402
import rz_profile_average as rz_average  # noqa: E402


LADDER_ROW = "i6_sn_mg_collapse_2d_rz_slab"
CERT_EPS2_GATE = 1.0e-3
SMOKE_EPS2_TRIPWIRE = 1.0e-2
U_ABSOLUTE = 5.0e-3  # Provisional pending the qualification campaign.
TIME_MATCH_RTOL = 1.0e-9
REPLICA_TIME_MATCH_RTOL = 1.0e-14
VARIABILITY_CAP_MULTIPLIER = 3.0
VARIABILITY_CAPS_PATH = TOOLS_DIR / "data" / "i6_variability_caps.json"
BINDING_FIELDS = ("rad_E", "sn_F_z")
TINY = 1.0e-300


@dataclass(frozen=True)
class I6Snapshot:
    path: Path
    t: float
    step: int
    rho: Any
    Te: Any
    Ti: Any
    vol: Any
    rad_E: Any
    sn_F_z: Any
    u_z: Any
    r_nodes: Any
    z_nodes: Any
    n_groups_attr: int | None
    sn_radial_fixup_count: float | None
    sn_angular_fixup_count: float | None
    warnings: tuple[str, ...]


def default_workdir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "tmp" / "i6_sn_mg_collapse" / stamp


def parse_groups(value: str) -> list[int]:
    groups = [int(tok.strip()) for tok in value.split(",") if tok.strip()]
    if not groups:
        raise argparse.ArgumentTypeError("expected at least one group count")
    if any(g < 2 for g in groups):
        raise argparse.ArgumentTypeError("I6 multigroup legs require group counts >= 2")
    return groups


def eps_norms(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    cand = np.asarray(candidate, dtype=float)
    ref = np.asarray(reference, dtype=float)
    diff = cand - ref
    eps2 = float(np.linalg.norm(diff) / max(np.linalg.norm(ref), TINY))
    epsinf = float(np.max(np.abs(diff)) / max(float(np.max(np.abs(ref))), TINY))
    return {"eps2": eps2, "epsinf": epsinf}


def sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items() if k != "snapshot"}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(summary), indent=2) + "\n", encoding="utf-8")


def run_info(outdir: Path) -> dict[str, Any]:
    path = base.discover_actual_outdir(outdir) / "run_info.json"
    if not path.exists():
        return {}
    try:
        return base.read_json(path)
    except (OSError, json.JSONDecodeError):
        return {}


def _read_cell_group(
    handle: Any,
    path: str,
    nr: int,
    nz: int,
    groups: int,
) -> np.ndarray:
    n_cells = nr * nz
    arr = np.asarray(handle[path][()], dtype=float)
    if arr.ndim == 2 and arr.shape == (n_cells, groups):
        return arr
    if arr.size == n_cells * groups:
        arr = arr.reshape((n_cells, groups))
        if arr.shape[1] == groups:
            return arr
    raise ValueError(
        f"{path}: expected shape ({n_cells}, {groups}), got {arr.shape}"
    )


def _optional_fixup_sum(
    handle: Any,
    path: str,
    nr: int,
    nz: int,
    groups: int,
    warnings: list[str],
) -> float | None:
    if path not in handle:
        message = f"missing optional dataset {path}"
        warnings.append(message)
        print(f"[run_i6_sn_mg_collapse] WARNING: {message}", file=sys.stderr)
        return None
    return float(np.sum(_read_cell_group(handle, path, nr, nz, groups)))


def read_snapshot(path: Path, nr: int, nz: int, groups: int) -> I6Snapshot:
    import h5py

    warnings: list[str] = []
    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = base.scalar_dataset(handle, "time_state/t", t_value)
        r_nodes = base.reshape_nodes(handle["mesh/x_r"][()], nr, nz, "mesh/x_r", "2D_RZ")
        z_nodes = base.reshape_nodes(handle["mesh/x_z"][()], nr, nz, "mesh/x_z", "2D_RZ")
        v_r_nodes = base.reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r", "2D_RZ")
        v_z_nodes = (
            base.reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z", "2D_RZ")
            if "mesh/v_z" in handle
            else np.zeros_like(v_r_nodes)
        )
        rad_E_groups = _read_cell_group(
            handle, "radiation/energy_density", nr, nz, groups
        )
        sn_F_z_groups = _read_cell_group(handle, "radiation/sn_F_z", nr, nz, groups)
        n_groups_attr = (
            int(handle.attrs["n_groups"]) if "n_groups" in handle.attrs else None
        )
        return I6Snapshot(
            path=path,
            t=t_value,
            step=int(round(base.scalar_dataset(handle, "time_state/step", 0.0))),
            rho=base.reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho", "2D_RZ"),
            Te=base.reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te", "2D_RZ"),
            Ti=base.reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti", "2D_RZ"),
            vol=base.reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol", "2D_RZ"),
            rad_E=np.sum(rad_E_groups, axis=1).reshape((nr, nz)),
            sn_F_z=np.sum(sn_F_z_groups, axis=1).reshape((nr, nz)),
            u_z=i1.cell_center_from_nodes(v_z_nodes),
            r_nodes=r_nodes,
            z_nodes=z_nodes,
            n_groups_attr=n_groups_attr,
            sn_radial_fixup_count=_optional_fixup_sum(
                handle, "radiation/sn_radial_fixup_count", nr, nz, groups, warnings
            ),
            sn_angular_fixup_count=_optional_fixup_sum(
                handle, "radiation/sn_angular_fixup_count", nr, nz, groups, warnings
            ),
            warnings=tuple(warnings),
        )


def latest_snapshot(outdir: Path, nr: int, nz: int, groups: int) -> I6Snapshot:
    actual = base.discover_actual_outdir(outdir)
    results_dir = actual / "results"
    candidates = sorted(results_dir.glob("*.h5")) + sorted(actual.glob("*.h5"))
    if not candidates:
        raise FileNotFoundError(f"no HDF5 snapshots under {results_dir} or {actual}")
    best: I6Snapshot | None = None
    errors: list[str] = []
    for path in candidates:
        try:
            snap = read_snapshot(path, nr, nz, groups)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
            continue
        if best is None or snap.t > best.t:
            best = snap
    if best is None:
        tail = "; ".join(errors[-3:])
        raise RuntimeError(f"no readable snapshots under {actual}: {tail}")
    return best


def profile(snapshot: I6Snapshot, field: Any) -> np.ndarray:
    reduced = rz_average.reduce_2d_rz_to_z_profile(np.asarray(field, dtype=float), snapshot)
    return np.asarray(reduced["profile_z"], dtype=float)


def profiles(snapshot: I6Snapshot) -> dict[str, np.ndarray]:
    return {
        "rad_E": profile(snapshot, snapshot.rad_E),
        "sn_F_z": profile(snapshot, snapshot.sn_F_z),
        "rho": profile(snapshot, snapshot.rho),
        "Te": profile(snapshot, snapshot.Te),
        "Ti": profile(snapshot, snapshot.Ti),
        "u_z": profile(snapshot, snapshot.u_z),
    }


def compare_profiles(candidate: I6Snapshot, reference: I6Snapshot) -> dict[str, dict[str, float]]:
    cand = profiles(candidate)
    ref = profiles(reference)
    for name in cand:
        if cand[name].shape != ref[name].shape:
            raise ValueError(
                f"profile {name}: candidate shape {cand[name].shape} does not match "
                f"reference shape {ref[name].shape}"
            )
    return {name: eps_norms(cand[name], ref[name]) for name in cand}


def leg_env(args: argparse.Namespace, outdir: Path, groups: int) -> dict[str, str]:
    env = {
        "TENRYU_I3_SN_RS_RADIATION_MODE": "sn_transport",
        "TENRYU_I3_SN_RS_NR": str(args.nr),
        "TENRYU_I3_SN_RS_NZ": str(args.nz),
        "TENRYU_I3_SN_RS_N_ANGLES": str(args.n_angles),
        "TENRYU_I3_SN_RS_KAPPA_SCALE": str(args.kappa_scale),
        "TENRYU_I3_SN_RS_GROUPS": str(groups),
        "TENRYU_I3_SN_RS_OUTDIR": str(outdir),
    }
    if args.t_end_s is not None:
        env["TENRYU_I3_SN_RS_T_END_S"] = f"{args.t_end_s:.17g}"
    return env


def run_leg(
    args: argparse.Namespace,
    label: str,
    groups: int,
    replica_index: int | None = None,
) -> dict[str, Any]:
    run_label = label if replica_index is None else f"{label}_rep{replica_index}"
    outdir = Path(args.workdir) / run_label
    if replica_index is not None and outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env_extra = leg_env(args, outdir, groups)
    env = dict(os.environ)
    env.update(env_extra)
    log_path = outdir / "run.log"
    cmd = [str(args.binary), "run", str(args.deck)]
    t0 = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.monotonic() - t0
    info = run_info(outdir)
    result: dict[str, Any] = {
        "label": label,
        "groups": groups,
        "returncode": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "outdir": str(outdir),
        "actual_outdir": str(base.discover_actual_outdir(outdir)),
        "log": str(log_path),
        "cmd": cmd,
        "env": env_extra,
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "t_end_reached": str(info.get("termination_reason", "unknown")) == "t_end_reached",
    }
    if replica_index is not None:
        result["replica_index"] = replica_index
    if proc.returncode != 0:
        result["failed"] = True
        return result
    snap = latest_snapshot(outdir, args.nr, args.nz, groups)
    fixups = None
    if snap.sn_radial_fixup_count is not None and snap.sn_angular_fixup_count is not None:
        fixups = snap.sn_radial_fixup_count + snap.sn_angular_fixup_count
    result.update(
        {
            "t_final": float(snap.t),
            "final_step": int(snap.step),
            "snapshot_path": str(snap.path),
            "n_groups_attr": snap.n_groups_attr,
            "sn_radial_fixup_count": snap.sn_radial_fixup_count,
            "sn_angular_fixup_count": snap.sn_angular_fixup_count,
            "fixup_count": fixups,
            "warnings": list(snap.warnings),
            "snapshot": snap,
        }
    )
    return result


def run_leg_replicas(
    args: argparse.Namespace,
    label: str,
    groups: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, np.ndarray] | None,
    dict[str, dict[str, float]] | None,
]:
    replica_runs: list[dict[str, Any]] = []
    aggregate: dict[str, Any] = {
        "label": label,
        "groups": groups,
        "replica_count": int(args.replicas),
        "replicas": replica_runs,
    }
    for replica_index in range(args.replicas):
        run = run_leg(args, label, groups, replica_index)
        replica_runs.append(run)
        if run.get("returncode") != 0:
            aggregate["failed_reason"] = (
                f"{label} replica {replica_index} returned rc={run.get('returncode')}"
            )
            return aggregate, None, None, None
        if not run.get("t_end_reached"):
            aggregate["failed_reason"] = (
                f"{label} replica {replica_index} "
                f"termination_reason={run.get('termination_reason')}"
            )
            return aggregate, None, None, None

    replica_profiles = [profiles(run["snapshot"]) for run in replica_runs]
    medoid_sums = [
        sum(
            eps_norms(
                replica_profiles[other_index]["rad_E"],
                replica_profiles[candidate_index]["rad_E"],
            )["eps2"]
            for other_index in range(args.replicas)
            if other_index != candidate_index
        )
        for candidate_index in range(args.replicas)
    ]
    medoid_index = min(range(args.replicas), key=lambda index: medoid_sums[index])
    medoid_run = replica_runs[medoid_index]
    medoid_profiles = replica_profiles[medoid_index]
    radii = {
        field: {
            norm: max(
                eps_norms(rep_profiles[field], medoid_profiles[field])[norm]
                for rep_profiles in replica_profiles
            )
            for norm in ("eps2", "epsinf")
        }
        for field in BINDING_FIELDS
    }
    aggregate.update(
        {
            "medoid_replica": medoid_index,
            "medoid_selection_eps2_sums": medoid_sums,
        }
    )
    return aggregate, medoid_run, medoid_profiles, radii


def replica_config_determinism(
    label: str,
    replica_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    step_counts = [int(run["final_step"]) for run in replica_runs]
    final_times = [float(run["t_final"]) for run in replica_runs]
    identical_step_counts = all(step == step_counts[0] for step in step_counts)
    matching_final_times = all(
        abs(value - final_times[0]) / max(abs(final_times[0]), TINY)
        <= REPLICA_TIME_MATCH_RTOL
        for value in final_times
    )
    return {
        "leg": label,
        "step_counts": step_counts,
        "t_final": final_times,
        "step_counts_identical": identical_step_counts,
        "t_final_match_rtol": REPLICA_TIME_MATCH_RTOL,
        "t_final_matched": matching_final_times,
        "passed": identical_step_counts and matching_final_times,
    }


def load_variability_caps(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "provisional": True,
            "frozen_by": "PASS-AND-FREEZE bootstrap pending qualification campaign",
            "caps": {},
        }
    data = base.read_json(path)
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("provisional"), bool)
        or not isinstance(data.get("frozen_by"), str)
        or not isinstance(data.get("caps"), dict)
    ):
        raise ValueError(f"invalid variability caps schema in {path}")
    return data


def apply_variability_caps(
    caps_path: Path,
    caps_data: dict[str, Any],
    leg_radii: dict[str, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    caps = caps_data["caps"]
    freshly_frozen: list[str] = []
    per_leg: dict[str, Any] = {}
    layer_passed = True
    for leg, field_radii in leg_radii.items():
        if leg not in caps:
            caps[leg] = {}
        if not isinstance(caps[leg], dict):
            raise ValueError(f"invalid variability caps entry for leg {leg}")
        per_leg[leg] = {}
        for field in BINDING_FIELDS:
            radius = field_radii[field]
            if field not in caps[leg]:
                caps[leg][field] = {
                    norm: float(radius[norm]) * VARIABILITY_CAP_MULTIPLIER
                    for norm in ("eps2", "epsinf")
                }
                item = f"{leg}.{field}"
                freshly_frozen.append(item)
                print(
                    "[run_i6_sn_mg_collapse] WARNING: freshly froze provisional "
                    f"variability cap {item}: "
                    f"eps2={caps[leg][field]['eps2']:.6e} "
                    f"epsinf={caps[leg][field]['epsinf']:.6e}",
                    file=sys.stderr,
                )
            cap = caps[leg][field]
            if not isinstance(cap, dict) or any(
                norm not in cap for norm in ("eps2", "epsinf")
            ):
                raise ValueError(f"invalid variability caps entry for {leg}.{field}")
            cap_values: dict[str, float] = {}
            norm_passed: dict[str, bool] = {}
            for norm in ("eps2", "epsinf"):
                value = float(cap[norm])
                if not math.isfinite(value) or value < 0.0:
                    raise ValueError(
                        f"invalid variability cap {leg}.{field}.{norm}={cap[norm]}"
                    )
                cap_values[norm] = value
                norm_passed[norm] = float(radius[norm]) <= value
            field_passed = all(norm_passed.values())
            layer_passed = layer_passed and field_passed
            per_leg[leg][field] = {
                "radius": radius,
                "cap": cap_values,
                "norm_passed": norm_passed,
                "passed": field_passed,
            }

    if freshly_frozen:
        caps_data["provisional"] = True
        caps_data["frozen_by"] = (
            "PASS-AND-FREEZE bootstrap pending qualification campaign"
        )
        write_summary(caps_path, caps_data)

    return {
        "passed": layer_passed,
        "per_leg": per_leg,
        "freshly_frozen": freshly_frozen,
        "caps_path": str(caps_path),
        "cap_multiplier": VARIABILITY_CAP_MULTIPLIER,
        "provisional": bool(caps_data.get("provisional", False)),
        "frozen_by": caps_data.get("frozen_by"),
    }


def final_time_match(candidate: float, reference: float) -> bool:
    return abs(candidate - reference) / max(abs(reference), TINY) <= TIME_MATCH_RTOL


def format_float(value: Any) -> str:
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return "missing"
    return f"{fvalue:.6e}" if math.isfinite(fvalue) else "missing"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", default=str(REPO_ROOT / "build" / "tenryu"))
    parser.add_argument(
        "--deck",
        default=str(REPO_ROOT / "examples" / "verification" / "i3_sn_radshock_2d_rz_slab.py"),
    )
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--mode", choices=("smoke", "cert"), default="smoke")
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--nr", type=int, default=16)
    parser.add_argument("--nz", type=int, default=256)
    # Default 16: the I3 deck's reference table sn_radshock_M2_N{n_angles}_K20.json only exists for N16 (S_8 cert legs need a generated N8_K20 reference — campaign-time task).
    parser.add_argument("--n-angles", type=int, default=16)
    parser.add_argument("--kappa-scale", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "smoke" and args.t_end_s is None:
        parser.error("--t-end-s is required when --mode smoke")
    if args.replicas is None:
        args.replicas = 3 if args.mode == "cert" else 1
    if args.replicas < 1:
        parser.error("--replicas must be >= 1")
    args.workdir = str(default_workdir() if args.workdir is None else Path(args.workdir))
    default_groups = "2,4" if args.mode == "smoke" else "2,4,8"
    try:
        groups_list = parse_groups(args.groups or default_groups)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    summary_path = Path(args.workdir) / "summary.json"
    runs: dict[str, dict[str, Any]] = {}
    summary: dict[str, Any] = {
        "ladder_row": LADDER_ROW,
        "dimension": "2D_RZ",
        "mode": args.mode,
        "binary": str(args.binary),
        "deck": str(args.deck),
        "workdir": str(args.workdir),
        "groups_list": groups_list,
        "nr": int(args.nr),
        "nz": int(args.nz),
        "n_angles": int(args.n_angles),
        "kappa_scale": int(args.kappa_scale),
        "t_end_s": args.t_end_s,
        "replicas": int(args.replicas),
        "cert_eps2_gate": CERT_EPS2_GATE,
        "u_absolute": U_ABSOLUTE,
        "smoke_eps2_tripwire": SMOKE_EPS2_TRIPWIRE,
        "time_match_rtol": TIME_MATCH_RTOL,
        "runs": {},
        "layers": {
            layer: {
                "skipped": args.replicas == 1,
                "reason": "legacy single-run path" if args.replicas == 1 else None,
            }
            for layer in ("A", "B", "C")
        },
        "errors": [],
    }
    exit_code = 1
    verdict = "FAIL"

    try:
        if args.replicas == 1:
            r0 = run_leg(args, "R0_grey", 1)
            runs["R0"] = r0
            if r0.get("returncode") != 0:
                raise RuntimeError(f"R0_grey returned rc={r0.get('returncode')}")
            if not r0.get("t_end_reached"):
                raise RuntimeError(
                    f"R0_grey termination_reason={r0.get('termination_reason')}"
                )

            for groups in groups_list:
                label = f"G{groups}"
                run = run_leg(args, label, groups)
                runs[label] = run
                if run.get("returncode") != 0:
                    raise RuntimeError(f"{label} returned rc={run.get('returncode')}")
                if not run.get("t_end_reached"):
                    raise RuntimeError(
                        f"{label} termination_reason={run.get('termination_reason')}"
                    )
                if not final_time_match(float(run["t_final"]), float(r0["t_final"])):
                    raise RuntimeError(
                        f"{label} final time {run['t_final']:.17g} does not match "
                        f"R0 {r0['t_final']:.17g} within {TIME_MATCH_RTOL:.1e}"
                    )
                metrics = compare_profiles(run["snapshot"], r0["snapshot"])
                run["metrics_vs_R0"] = metrics
                print(
                    "[i6_g2] "
                    f"leg=G{groups} "
                    f"eps2_E={metrics['rad_E']['eps2']:.6e} "
                    f"epsinf_E={metrics['rad_E']['epsinf']:.6e} "
                    f"eps2_F={metrics['sn_F_z']['eps2']:.6e} "
                    f"epsinf_F={metrics['sn_F_z']['epsinf']:.6e} "
                    f"fixups={format_float(run.get('fixup_count'))} "
                    f"t_final={float(run['t_final']):.17e}"
                )

            binding_eps2 = [
                max(
                    float(runs[f"G{groups}"]["metrics_vs_R0"]["rad_E"]["eps2"]),
                    float(
                        runs[f"G{groups}"]["metrics_vs_R0"]["sn_F_z"]["eps2"]
                    ),
                )
                for groups in groups_list
            ]
            cert_pass = all(value <= CERT_EPS2_GATE for value in binding_eps2)
            smoke_tripwire_pass = all(
                value <= SMOKE_EPS2_TRIPWIRE for value in binding_eps2
            )
            summary["cert_gate_passed"] = cert_pass
            summary["smoke_tripwire_passed"] = smoke_tripwire_pass
            if args.mode == "cert":
                verdict = "PASS" if cert_pass else "FAIL"
                exit_code = 0 if cert_pass else 1
            else:
                verdict = "REPORT" if smoke_tripwire_pass else "FAIL"
                exit_code = 0 if smoke_tripwire_pass else 2
        else:
            medoid_runs: dict[str, dict[str, Any]] = {}
            medoid_profiles: dict[str, dict[str, np.ndarray]] = {}
            leg_radii: dict[str, dict[str, dict[str, float]]] = {}
            config_determinism: dict[str, dict[str, Any]] = {}

            r0_ensemble, r0_medoid, r0_profiles, r0_radii = run_leg_replicas(
                args, "R0_grey", 1
            )
            runs["R0_grey"] = r0_ensemble
            if r0_medoid is None or r0_profiles is None or r0_radii is None:
                raise RuntimeError(
                    str(r0_ensemble.get("failed_reason", "R0_grey failed"))
                )
            medoid_runs["R0_grey"] = r0_medoid
            medoid_profiles["R0_grey"] = r0_profiles
            leg_radii["R0_grey"] = r0_radii
            config_determinism["R0_grey"] = replica_config_determinism(
                "R0_grey", r0_ensemble["replicas"]
            )

            medoid_time_matches: dict[str, bool] = {}
            for groups in groups_list:
                label = f"G{groups}"
                ensemble, medoid_run, medoid_profile, radii = run_leg_replicas(
                    args, label, groups
                )
                runs[label] = ensemble
                if medoid_run is None or medoid_profile is None or radii is None:
                    raise RuntimeError(
                        str(ensemble.get("failed_reason", f"{label} failed"))
                    )
                medoid_runs[label] = medoid_run
                medoid_profiles[label] = medoid_profile
                leg_radii[label] = radii
                config_determinism[label] = replica_config_determinism(
                    label, ensemble["replicas"]
                )
                medoid_time_matches[label] = final_time_match(
                    float(medoid_run["t_final"]), float(r0_medoid["t_final"])
                )
                metrics = compare_profiles(
                    medoid_run["snapshot"], r0_medoid["snapshot"]
                )
                medoid_run["metrics_vs_R0"] = metrics
                ensemble["metrics_vs_R0"] = metrics
                print(
                    "[i6_g2] "
                    f"leg=G{groups} "
                    f"eps2_E={metrics['rad_E']['eps2']:.6e} "
                    f"epsinf_E={metrics['rad_E']['epsinf']:.6e} "
                    f"eps2_F={metrics['sn_F_z']['eps2']:.6e} "
                    f"epsinf_F={metrics['sn_F_z']['epsinf']:.6e} "
                    f"fixups={format_float(medoid_run.get('fixup_count'))} "
                    f"t_final={float(medoid_run['t_final']):.17e} "
                    f"medoid_rep={ensemble['medoid_replica']}"
                )

            binding_eps2 = [
                max(
                    float(runs[f"G{groups}"]["metrics_vs_R0"]["rad_E"]["eps2"]),
                    float(
                        runs[f"G{groups}"]["metrics_vs_R0"]["sn_F_z"]["eps2"]
                    ),
                )
                for groups in groups_list
            ]
            smoke_tripwire_pass = all(
                value <= SMOKE_EPS2_TRIPWIRE for value in binding_eps2
            )
            layer_a_pass = (
                all(item["passed"] for item in config_determinism.values())
                and all(run["t_end_reached"] for run in medoid_runs.values())
                and all(medoid_time_matches.values())
                and smoke_tripwire_pass
            )
            summary["layers"]["A"] = {
                "skipped": False,
                "passed": layer_a_pass,
                "config_determinism": config_determinism,
                "medoid_t_end_reached": {
                    leg: bool(run["t_end_reached"])
                    for leg, run in medoid_runs.items()
                },
                "medoid_time_match_vs_R0": medoid_time_matches,
                "time_match_rtol": TIME_MATCH_RTOL,
                "profile_shape_asserts_passed": True,
                "smoke_tripwire_passed": smoke_tripwire_pass,
                "smoke_eps2_tripwire": SMOKE_EPS2_TRIPWIRE,
            }

            caps_data = load_variability_caps(VARIABILITY_CAPS_PATH)
            layer_b = apply_variability_caps(
                VARIABILITY_CAPS_PATH, caps_data, leg_radii
            )
            layer_b["skipped"] = False
            summary["layers"]["B"] = layer_b

            layer_c_per_leg: dict[str, Any] = {}
            layer_c_pass = True
            for groups in groups_list:
                label = f"G{groups}"
                layer_c_per_leg[label] = {}
                for field in BINDING_FIELDS:
                    distance = eps_norms(
                        medoid_profiles[label][field],
                        medoid_profiles["R0_grey"][field],
                    )["eps2"]
                    allowance = (
                        CERT_EPS2_GATE
                        + leg_radii[label][field]["eps2"]
                        + leg_radii["R0_grey"][field]["eps2"]
                    )
                    within_allowance = distance <= allowance
                    within_absolute_cap = distance <= U_ABSOLUTE
                    field_pass = within_allowance and within_absolute_cap
                    layer_c_pass = layer_c_pass and field_pass
                    layer_c_per_leg[label][field] = {
                        "d_eps2": distance,
                        "allowance": allowance,
                        "t_design": CERT_EPS2_GATE,
                        "b_leg_eps2": leg_radii[label][field]["eps2"],
                        "b_r0_eps2": leg_radii["R0_grey"][field]["eps2"],
                        "u_absolute": U_ABSOLUTE,
                        "within_allowance": within_allowance,
                        "within_absolute_cap": within_absolute_cap,
                        "passed": field_pass,
                    }
            summary["layers"]["C"] = {
                "skipped": False,
                "passed": layer_c_pass,
                "per_leg": layer_c_per_leg,
                "t_design": CERT_EPS2_GATE,
                "u_absolute": U_ABSOLUTE,
            }

            overall_pass = layer_a_pass and layer_b["passed"] and layer_c_pass
            summary["cert_gate_passed"] = overall_pass
            summary["smoke_tripwire_passed"] = smoke_tripwire_pass
            verdict = "PASS" if overall_pass else "FAIL"
            exit_code = 0 if overall_pass else 1
    except Exception as exc:
        summary["errors"].append(str(exc))
        if args.replicas > 1:
            summary["cert_gate_passed"] = False
            for layer in ("A", "B", "C"):
                if "passed" not in summary["layers"][layer]:
                    summary["layers"][layer] = {
                        "skipped": False,
                        "passed": False,
                        "reason": "not evaluated due to harness error",
                    }
        verdict = "FAIL"
        exit_code = 1
    finally:
        summary["verdict"] = verdict
        summary["all_checks_passed"] = exit_code == 0
        summary["runs"] = {label: sanitize(run) for label, run in runs.items()}
        write_summary(summary_path, summary)
        print(f"[i6_g2] verdict={verdict} summary={summary_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
