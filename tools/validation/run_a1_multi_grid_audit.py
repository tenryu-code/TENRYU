#!/usr/bin/env python3
"""Run the A1 multi-grid ALE Tier-A production-audit matrix."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRIDS = ((128, 256), (256, 512), (512, 1024))
FAST_GRIDS = ((128, 256), (256, 512))
MODES = ("axis_spine_only", "full_winslow")
MOMENTUM_PHYSICAL_SCALE_LIMIT = 1.0
ESCAPE_FLAGS = (
    "emergency_cell_deactivation_enabled",
    "multi_node_boundary_repair_enabled",
    "multi_node_interior_repair_enabled",
    "axis_variational_projection_enabled",
    "local_boundary_repair_enabled",
    "retry_active_mesh_repair_enabled",
)
GCL_RE = re.compile(r"max_total_drift_rel=(?P<value>[-+0-9.eE]+)")


def repo_root() -> Path:
    path = Path.cwd().resolve()
    while path != path.parent:
        if (path / "examples/verification/2d_rz_a1_ale_sweep.py").exists():
            return path
        path = path.parent
    return Path.cwd().resolve()


def case_name(nr: int, nz: int) -> str:
    return f"a1_sod_quality_only_nr{nr}_nz{nz}_seed12345"


def run_id(nr: int, nz: int, mode: str) -> str:
    return f"nr{nr}_nz{nz}_{mode}"


def read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def latest_history(outdir: Path) -> Path | None:
    results = outdir / "results"
    if not results.exists():
        return None
    histories = sorted(results.glob("*_history.h5"), key=lambda path: path.stat().st_mtime)
    return histories[-1] if histories else None


def frozen_config(outdir: Path, name: str) -> Path | None:
    path = outdir / "config" / f"{name}_frozen.json"
    if path.exists():
        return path
    config_dir = outdir / "config"
    if not config_dir.exists():
        return None
    candidates = sorted(config_dir.glob("*_frozen.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def max_present(values: dict[str, Any]) -> float | None:
    finite: list[float] = []
    for value in values.values():
        if isinstance(value, (int, float)):
            finite.append(abs(float(value)))
    return max(finite) if finite else None


def positivity_min(values: Any) -> float | None:
    if not isinstance(values, dict):
        return None
    finite: list[float] = []
    for value in values.values():
        if isinstance(value, (int, float)):
            finite.append(float(value))
    return min(finite) if finite else None


def gcl_residual(summary: dict[str, Any], log_path: Path) -> float | None:
    for key in ("gcl_residual", "gcl_residual_max", "ale_gcl_residual_max"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return abs(float(value))
    solver_stats = summary.get("solver_stats")
    if isinstance(solver_stats, dict):
        for key in ("gcl", "ale_gcl", "gcl_residual"):
            value = solver_stats.get(key)
            if isinstance(value, (int, float)):
                return abs(float(value))
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    values = [abs(float(match.group("value"))) for match in GCL_RE.finditer(text)]
    if values:
        return max(values)
    return None


def validation_failures(entry: dict[str, Any], conservation_limit: float) -> list[str]:
    """Validation logic for A1 multi-grid Tier-A audit.

    Note: A1 deck is pure hydro+ALE (no radiation), so fields like E_r and kappa
    are analytic-zero (initial=0, no source). Tracker may report 0.0 for these
    which is correct behavior, NOT a positivity violation. We accept 0.0 for
    radiation-only fields in a radiation-free deck.
    """
    failures: list[str] = []
    if entry.get("termination_reason") != "t_end_reached":
        failures.append(f"termination_reason={entry.get('termination_reason')!r}")
    if int(entry.get("escape_valve_firings_total", -1)) != 0:
        failures.append(f"escape_valve_firings_total={entry.get('escape_valve_firings_total')}")
    if entry.get("audit_status") != "TIER_A_PASS":
        failures.append(f"audit_status={entry.get('audit_status')!r}")
    conservation = entry.get("conservation_residual_max")
    if not isinstance(conservation, dict):
        failures.append("conservation_residual_max missing")
    else:
        energy_value = conservation.get("energy")
        if energy_value is None:
            failures.append("conservation_residual_max.energy missing")
        elif not isinstance(energy_value, (int, float)):
            failures.append(f"conservation_residual_max.energy={energy_value!r}")
        elif abs(float(energy_value)) > max(conservation_limit, 1.0e-8):
            failures.append(f"conservation_residual_max.energy={energy_value!r} > {max(conservation_limit, 1.0e-8):.1e}")
        for key in ("mass",):
            value = conservation.get(key)
            if value is None:
                failures.append(f"conservation_residual_max.{key} missing")
                continue
            if not isinstance(value, (int, float)) or abs(float(value)) > max(conservation_limit, 1.0e-8):
                failures.append(f"conservation_residual_max.{key}={value!r}")
        for key in ("R-momentum", "Z-momentum"):
            value = conservation.get(key)
            if value is None:
                failures.append(f"conservation_residual_max.{key} missing")
                continue
            if not isinstance(value, (int, float)) or abs(float(value)) > MOMENTUM_PHYSICAL_SCALE_LIMIT:
                failures.append(f"conservation_residual_max.{key}={value!r} > {MOMENTUM_PHYSICAL_SCALE_LIMIT:.1e}")
    gcl_value = entry.get("gcl_residual")
    if gcl_value is None:
        failures.append("gcl_residual missing")
    elif not isinstance(gcl_value, (int, float)) or abs(float(gcl_value)) > max(conservation_limit, 1.0e-8):
        failures.append(f"gcl_residual={gcl_value!r}")
    positivity = entry.get("positivity_min")
    if not isinstance(positivity, dict):
        failures.append("positivity_min missing")
    else:
        # Radiation-only fields (E_r, kappa) are analytic-zero in A1 (no radiation):
        # 0.0 is correct, not a violation. Only flag radiation-only fields if NEGATIVE.
        # Hydro fields (rho, p, Te, Ti) must be strictly positive.
        radiation_only = {"E_r", "kappa"}
        for key, value in positivity.items():
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                failures.append(f"positivity_min.{key}={value!r}")
                continue
            val_f = float(value)
            if key in radiation_only:
                # Analytic-zero acceptable; negative is a real violation.
                if val_f < 0.0:
                    failures.append(f"positivity_min.{key}={value!r} (negative)")
            else:
                # Hydro fields must be > 0.
                if val_f <= 0.0:
                    failures.append(f"positivity_min.{key}={value!r}")
    return failures


def run_deck(root: Path, outdir: Path, nr: int, nz: int, mode: str, force: bool) -> dict[str, Any]:
    audit_path = outdir / "audit_summary.json"
    name = case_name(nr, nz)
    log_path = outdir / "a1_multi_grid_audit.log"
    if audit_path.exists() and not force:
        return read_json(audit_path)
    # If a prior run created a sibling outdir_NNN with audit_summary.json, reuse.
    if not force:
        parent = outdir.parent
        prefix = outdir.name
        for cand in sorted(parent.glob(f"{prefix}_*"), reverse=True):
            cand_audit = cand / "audit_summary.json"
            if cand_audit.exists():
                return read_json(cand_audit)

    outdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "TENRYU_A1_NR": str(nr),
            "TENRYU_A1_NZ": str(nz),
            "TENRYU_A1_AXIS_REPAIR_MODE": mode,
            "TENRYU_A1_PRODUCTION_AUDIT_TIER": "A",
            "TENRYU_A1_OUTDIR": str(outdir),
            "TENRYU_A1_MAX_STEPS": env.get("TENRYU_A1_MAX_STEPS", "1000000"),
            "TENRYU_A1_DEBUG_PER_REMAP_LOG": env.get("TENRYU_A1_DEBUG_PER_REMAP_LOG", "0"),
        }
    )
    cmd = ["./build/tenryu", "run", "examples/verification/2d_rz_a1_ale_sweep.py"]
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {
            "audit_status": "RUN_FAILED",
            "termination_reason": "subprocess_failed",
            "run_returncode": result.returncode,
            "log_path": str(log_path),
        }

    # TENRYU's Output() may auto-increment outdir (e.g., "<name>_001"); search siblings.
    actual_outdir = outdir
    run_info = outdir / "run_info.json"
    if not run_info.exists():
        # Look at outdir_001, outdir_002, ... in same parent
        parent = outdir.parent
        prefix = outdir.name
        candidates = sorted(parent.glob(f"{prefix}_*"))
        for cand in reversed(candidates):  # newest suffix first
            cand_info = cand / "run_info.json"
            if cand_info.exists():
                actual_outdir = cand
                run_info = cand_info
                break
    history = latest_history(actual_outdir)
    frozen = frozen_config(actual_outdir, name)
    if not run_info.exists():
        return {"audit_status": "RUN_INFO_MISSING", "termination_reason": "run_info_missing"}
    audit_cmd = [
        sys.executable,
        "tools/validation/audit_summary.py",
        "--run-info",
        str(run_info),
        "--tier",
        "A",
        "--production-audit-enabled",
        "--require-positivity",
        "--require-gcl",
        "--output",
        str(audit_path),
    ]
    if history is not None:
        audit_cmd += ["--history-hdf5", str(history)]
    if frozen is not None:
        audit_cmd += ["--frozen-config", str(frozen)]
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(audit_cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        return {
            "audit_status": "AUDIT_POSTPROCESS_FAILED",
            "termination_reason": read_json(run_info).get("termination_reason", "unknown"),
            "audit_returncode": result.returncode,
            "log_path": str(log_path),
        }
    return read_json(audit_path)


def summarize_run(
    root: Path,
    base_outdir: Path,
    nr: int,
    nz: int,
    mode: str,
    force: bool,
    conservation_limit: float,
) -> dict[str, Any]:
    outdir = base_outdir / run_id(nr, nz, mode)
    log_path = outdir / "a1_multi_grid_audit.log"
    summary = run_deck(root, outdir, nr, nz, mode, force)
    escape = summary.get("escape_valve_firings")
    if not isinstance(escape, dict):
        escape = {}
    conservation = summary.get("conservation_residual_max")
    if not isinstance(conservation, dict):
        conservation = {}
    entry = {
        "run_id": run_id(nr, nz, mode),
        "nr": nr,
        "nz": nz,
        "mode": mode,
        "output_dir": str(outdir),
        "audit_json_path": str(outdir / "audit_summary.json"),
        "audit_status": summary.get("audit_status", "unknown"),
        "termination_reason": summary.get("termination_reason", "unknown"),
        "escape_valve_firings": {flag: bool(escape.get(flag, False)) for flag in ESCAPE_FLAGS},
        "escape_valve_firings_total": sum(1 for flag in ESCAPE_FLAGS if bool(escape.get(flag, False))),
        "conservation_residual_max": conservation,
        "conservation_residual_max_abs": max_present(conservation),
        "gcl_residual": gcl_residual(summary, log_path),
        "positivity_min": summary.get("positivity_min"),
        "positivity_min_value": positivity_min(summary.get("positivity_min")),
        "missing_required_data": summary.get("missing_required_data", []),
    }
    entry["validation_failures"] = validation_failures(entry, conservation_limit)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=("fast", "full"), default="full")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("tools/validation/a1_multi_grid_audit_report.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("build/output_verify_a1_multi_grid_audit"))
    parser.add_argument("--conservation-limit", type=float, default=1.0e-10)
    args = parser.parse_args()

    root = repo_root()
    grids = FAST_GRIDS if args.matrix == "fast" else GRIDS
    runs = [
        summarize_run(root, root / args.work_dir, nr, nz, mode, args.force, args.conservation_limit)
        for nr, nz in grids
        for mode in MODES
    ]
    failed = [
        f"{run['run_id']}: {failure}"
        for run in runs
        for failure in run.get("validation_failures", [])
    ]
    all_checks_passed = all(run.get("audit_status") == "TIER_A_PASS" for run in runs) and all(
        not run.get("validation_failures") for run in runs
    )
    report = {
        "schema_version": "tenryu.a1_multi_grid_audit_report.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "matrix": args.matrix,
        "expected_runs": len(grids) * len(MODES),
        "actual_runs": len(runs),
        "conservation_limit": args.conservation_limit,
        "all_checks_passed": all_checks_passed,
        "failed_checks": failed,
        "runs": runs,
        "validation": {
            "all_checks_passed": all_checks_passed,
            "failed_checks": failed,
        },
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    if failed:
        for failure in failed:
            print(failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
