#!/usr/bin/env python3
"""Emit a TENRYU production-audit JSON summary from run outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ESCAPE_FLAGS = (
    "emergency_cell_deactivation_enabled",
    "multi_node_boundary_repair_enabled",
    "multi_node_interior_repair_enabled",
    "axis_variational_projection_enabled",
    "local_boundary_repair_enabled",
    "retry_active_mesh_repair_enabled",
)


def decode_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    arr = np.asarray(value)
    if arr.shape == ():
        item = arr.item()
        if isinstance(item, bytes):
            return item.decode("utf-8", errors="replace").rstrip("\x00")
        if isinstance(item, np.generic):
            return item.item()
        return item
    if arr.dtype.kind == "S":
        return [item.decode("utf-8", errors="replace").rstrip("\x00") for item in arr.reshape(-1)]
    return arr.tolist()


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def numeric_array(dataset: h5py.Dataset) -> np.ndarray | None:
    arr = np.asarray(dataset[()])
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        return None
    return arr.astype(float, copy=False).reshape(-1)


def max_abs_dataset(dataset: h5py.Dataset) -> float | None:
    arr = numeric_array(dataset)
    if arr is None or arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.max(np.abs(finite)))


def min_dataset(dataset: h5py.Dataset) -> float | None:
    arr = numeric_array(dataset)
    if arr is None or arr.size == 0:
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def last_dataset(dataset: h5py.Dataset) -> Any:
    value = decode_value(dataset[()])
    if isinstance(value, list):
        return value[-1] if value else None
    return value


def h5_get(handle: h5py.File, path: str) -> h5py.Dataset | h5py.Group | None:
    key = path[1:] if path.startswith("/") else path
    return handle[key] if key in handle else None


def max_abs_first_present(handle: h5py.File, paths: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for path in paths:
        obj = h5_get(handle, path)
        if isinstance(obj, h5py.Dataset):
            value = max_abs_dataset(obj)
            if value is not None:
                values.append(value)
    return max(values) if values else None


def read_conservation_residual_max(handle: h5py.File | None) -> dict[str, float | None]:
    if handle is None:
        return {"mass": None, "R-momentum": None, "Z-momentum": None, "energy": None}
    return {
        "mass": max_abs_first_present(
            handle,
            (
                "/diagnostics/conservation/v1/mass_residual",
                "/diagnostics/conservation/v1/mass_max_residual",
                "/diagnostics/conservation/v1/per_material_mass_max_rel_residual",
                "mass/conservation_error",
            ),
        ),
        "R-momentum": max_abs_first_present(
            handle,
            (
                "/diagnostics/conservation/v1/R_momentum_residual",
                "/diagnostics/conservation/v1/R-momentum",
                "momentum/R_conservation_error",
            ),
        ),
        "Z-momentum": max_abs_first_present(
            handle,
            (
                "/diagnostics/conservation/v1/Z_momentum_residual",
                "/diagnostics/conservation/v1/Z-momentum",
                "momentum/Z_conservation_error",
            ),
        ),
        "energy": max_abs_first_present(
            handle,
            (
                "energy/conservation_error",
                "/diagnostics/conservation/v1/energy_residual",
                "/diagnostics/conservation/v1/energy_max_residual",
            ),
        ),
    }


def read_gcl_residual(handle: h5py.File | None) -> float | None:
    if handle is None:
        return None
    return max_abs_first_present(
        handle,
        (
            "/diagnostics/conservation/v1/gcl_residual",
            "/diagnostics/conservation/v1/gcl_residual_max",
            "/diagnostics/conservation/v1/ale_gcl_residual",
            "/diagnostics/conservation/v1/ale_gcl_residual_max",
        ),
    )


def read_positivity_min(handle: h5py.File | None) -> dict[str, float | None] | None:
    if handle is None:
        return None
    group = h5_get(handle, "/diagnostics/positivity/v1")
    if not isinstance(group, h5py.Group):
        return None
    out: dict[str, float | None] = {}
    candidates = {
        "rho": ("rho", "rho_min", "min_rho"),
        "p": ("p", "p_min", "pressure_min", "min_p"),
        "T_e": ("T_e", "Te", "T_e_min", "Te_min", "min_T_e", "min_Te"),
        "T_i": ("T_i", "Ti", "T_i_min", "Ti_min", "min_T_i", "min_Ti"),
        "E_r": ("E_r", "Er", "E_r_min", "Er_min", "min_E_r", "min_Er"),
        "kappa": ("kappa", "kappa_min", "min_kappa"),
    }
    for field, names in candidates.items():
        value = None
        for name in names:
            if name in group and isinstance(group[name], h5py.Dataset):
                value = min_dataset(group[name])
                break
        out[field] = value
    return out


def read_solver_stats(handle: h5py.File | None) -> dict[str, float]:
    if handle is None:
        return {}
    out: dict[str, float] = {}
    energy_solver = h5_get(handle, "energy/solver_residual")
    if isinstance(energy_solver, h5py.Dataset):
        value = max_abs_dataset(energy_solver)
        if value is not None:
            out["energy"] = value
    conservation = h5_get(handle, "/diagnostics/conservation/v1")
    if isinstance(conservation, h5py.Group):
        for name, obj in conservation.items():
            if name.startswith("eps_E_") and isinstance(obj, h5py.Dataset):
                value = max_abs_dataset(obj)
                if value is not None:
                    out[name.removeprefix("eps_E_")] = value
    solver_group = h5_get(handle, "/diagnostics/solver_stats/v1")
    if isinstance(solver_group, h5py.Group):
        for name, obj in solver_group.items():
            if isinstance(obj, h5py.Dataset):
                value = max_abs_dataset(obj)
                if value is not None:
                    out[name] = value
    return out


def string_values(dataset: h5py.Dataset) -> list[str]:
    value = decode_value(dataset[()])
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def read_escape_valve_firings(handle: h5py.File | None) -> dict[str, bool] | None:
    if handle is None:
        return None
    audit = h5_get(handle, "/diagnostics/escape_valve_audit/v1")
    if isinstance(audit, h5py.Group):
        out = {flag: False for flag in ESCAPE_FLAGS}
        for flag in ESCAPE_FLAGS:
            aliases = (flag, flag.replace("_enabled", "_fired"), flag.replace("_enabled", ""))
            for alias in aliases:
                if alias in audit and isinstance(audit[alias], h5py.Dataset):
                    value = max_abs_dataset(audit[alias])
                    out[flag] = bool(value is not None and value > 0.0)
                    break
        return out
    provenance = h5_get(handle, "/diagnostics/ale_provenance/v1")
    if not isinstance(provenance, h5py.Group):
        return None
    out = {flag: False for flag in ESCAPE_FLAGS}
    emergency = provenance.get("emergency_cell_deactivation_fired")
    if isinstance(emergency, h5py.Dataset):
        value = max_abs_dataset(emergency)
        out["emergency_cell_deactivation_enabled"] = bool(value is not None and value > 0.0)
    events = provenance.get("escape_valve_events")
    if isinstance(events, h5py.Group):
        operators: list[str] = []
        operator_dataset = events.get("operator_inserted")
        if isinstance(operator_dataset, h5py.Dataset):
            operators.extend(string_values(operator_dataset))
        for operator in operators:
            compact = "".join(ch for ch in operator.lower() if ch.isalnum())
            if "multinodeboundary" in compact:
                out["multi_node_boundary_repair_enabled"] = True
            if "multinodeinterior" in compact:
                out["multi_node_interior_repair_enabled"] = True
            if "axisvariational" in compact:
                out["axis_variational_projection_enabled"] = True
            if "localboundary" in compact:
                out["local_boundary_repair_enabled"] = True
            if "retryactivemesh" in compact or "hydrodriverretry" in compact:
                out["retry_active_mesh_repair_enabled"] = True
            if "emergencycell" in compact:
                out["emergency_cell_deactivation_enabled"] = True
    return out


def read_symmetry_residual(handle: h5py.File | None) -> dict[str, float]:
    if handle is None:
        return {}
    dataset = h5_get(handle, "modes/P_ell")
    if not isinstance(dataset, h5py.Dataset):
        return {}
    arr = np.asarray(dataset[()])
    if arr.size == 0 or arr.dtype.kind not in {"b", "i", "u", "f"}:
        return {}
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return {}
    ell_values = dataset.attrs.get("ell_values")
    decoded_ell = decode_value(ell_values) if ell_values is not None else list(range(arr.shape[1]))
    if not isinstance(decoded_ell, list):
        decoded_ell = [decoded_ell]
    out: dict[str, float] = {}
    n_modes = min(arr.shape[1], len(decoded_ell))
    for idx in range(n_modes):
        values = arr[:, idx].astype(float, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size:
            out[f"ell_{decoded_ell[idx]}"] = float(np.max(np.abs(finite)))
    return out


def load_reference(paths: list[Path]) -> tuple[dict[str, float], dict[str, float]]:
    profile_l2_error: dict[str, float] = {}
    symmetry_residual: dict[str, float] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            continue
        for key, target in (
            ("profile_l2_error", profile_l2_error),
            ("symmetry_residual", symmetry_residual),
        ):
            values = data.get(key)
            if isinstance(values, dict):
                for name, raw in values.items():
                    value = finite_float(raw)
                    if value is not None:
                        target[str(name)] = value
    return profile_l2_error, symmetry_residual


def read_production_audit_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        root = json.load(handle)
    numerics = root.get("numerics") if isinstance(root, dict) else None
    diagnostics = numerics.get("diagnostics") if isinstance(numerics, dict) else None
    audit = diagnostics.get("production_audit") if isinstance(diagnostics, dict) else None
    return audit if isinstance(audit, dict) else {}


def infer_case_name(run_info: Path, history: Path | None, explicit: str | None) -> str:
    if explicit:
        return explicit
    if history is not None:
        name = history.stem
        if name.endswith("_history"):
            return name[: -len("_history")]
        return name
    return run_info.parent.name or "unnamed"


def default_output_path(run_info: Path, history: Path | None, case_name: str) -> Path:
    base_dir = run_info.parent
    if history is not None:
        base_dir = history.parent
    return base_dir / f"{case_name}.audit.json"


def open_history(path: Path | None) -> h5py.File | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return h5py.File(path, "r")


def read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def reproducibility_result_candidates(run_dir: Path) -> list[Path]:
    return [
        run_dir / "reproducibility_check.json",
        run_dir / "reproducibility_result.json",
        run_dir / "reproducibility_results.json",
        run_dir / "same_arch_bitwise.json",
        run_dir / "cross_arch_tolerance.json",
    ]


def load_reproducibility_result(paths: list[Path], run_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in paths + reproducibility_result_candidates(run_dir):
        payload = read_json_object(path)
        if payload is not None:
            out.append(payload)
    return out


def read_reproducibility_summary(run_dir: Path, result_paths: list[Path]) -> dict[str, Any]:
    same_arch_status = "NOT_CHECKED"
    cross_arch_status = "NOT_CHECKED"
    metadata: dict[str, Any] = {}

    metadata_path = run_dir / "cross_arch_metadata.json"
    metadata_payload = read_json_object(metadata_path)
    if metadata_payload is not None:
        metadata = metadata_payload

    for result in load_reproducibility_result(result_paths, run_dir):
        mode = str(result.get("mode", ""))
        status = result.get("status")
        if mode == "same_arch_bitwise" and status in {"PASS", "FAIL"}:
            same_arch_status = str(status)
        elif "same_arch_bitwise" in result and isinstance(result["same_arch_bitwise"], dict):
            nested_status = result["same_arch_bitwise"].get("status")
            if nested_status in {"PASS", "FAIL"}:
                same_arch_status = str(nested_status)

        if mode == "cross_arch_tolerance" and status in {"PASS", "FAIL", "deferred_to_multi_arch_ci"}:
            cross_arch_status = str(status)
        elif result.get("cross_arch_status") in {"PASS", "FAIL", "deferred_to_multi_arch_ci"}:
            cross_arch_status = str(result["cross_arch_status"])

        result_metadata = result.get("cross_arch_metadata")
        if isinstance(result_metadata, dict):
            metadata = result_metadata

    return {
        "same_arch_bitwise": same_arch_status,
        "cross_arch_status": cross_arch_status,
        "cross_arch_metadata": metadata,
    }


def audit_summary(
    run_info_path: Path,
    history_path: Path | None,
    tier: str,
    production_audit_enabled: bool,
    reference_paths: list[Path],
    require_positivity: bool,
    require_gcl: bool,
    reproducibility_result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    with run_info_path.open("r", encoding="utf-8") as handle:
        run_info = json.load(handle)
    termination_reason = str(run_info.get("termination_reason", "unknown"))
    profile_l2_error, reference_symmetry = load_reference(reference_paths)

    history = open_history(history_path)
    try:
        conservation = read_conservation_residual_max(history)
        gcl_residual = read_gcl_residual(history)
        positivity = read_positivity_min(history)
        solver_stats = read_solver_stats(history)
        escape_valves = read_escape_valve_firings(history)
        history_symmetry = read_symmetry_residual(history)
    finally:
        if history is not None:
            history.close()

    symmetry_residual = history_symmetry
    symmetry_residual.update(reference_symmetry)
    missing_required: list[str] = []
    if tier == "A":
        for key, value in conservation.items():
            if value is None:
                missing_required.append(f"conservation_residual_max.{key}")
        if escape_valves is None:
            missing_required.append("escape_valve_firings")
        if require_positivity and positivity is None:
            missing_required.append("positivity_min")
        if require_gcl:
            if gcl_residual is None:
                missing_required.append("gcl")

    if tier == "A" and termination_reason != "t_end_reached":
        audit_status = "TIER_A_FAIL_RUN_ABORTED"
    elif tier == "A" and missing_required:
        audit_status = "TIER_A_FAIL_MISSING_DATA"
    elif not production_audit_enabled and tier == "none":
        audit_status = "NOT_REQUESTED"
    elif tier == "A":
        audit_status = "TIER_A_PASS"
    else:
        audit_status = "PASS"

    return {
        "schema_version": "tenryu.audit_summary.v1",
        "production_audit_enabled": production_audit_enabled,
        "tier": tier,
        "termination_reason": termination_reason,
        "conservation_residual_max": conservation,
        "positivity_min": positivity,
        "solver_stats": solver_stats,
        "escape_valve_firings": escape_valves if escape_valves is not None else {},
        "profile_l2_error": profile_l2_error,
        "symmetry_residual": symmetry_residual,
        "gcl_residual": gcl_residual,
        "reproducibility": read_reproducibility_summary(
            run_info_path.parent,
            reproducibility_result_paths if reproducibility_result_paths is not None else [],
        ),
        "audit_status": audit_status,
        "missing_required_data": missing_required,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-info", required=True, type=Path)
    parser.add_argument("--history-hdf5", "--history", dest="history_hdf5", type=Path)
    parser.add_argument("--frozen-config", type=Path)
    parser.add_argument("--reference", action="append", default=[], type=Path)
    parser.add_argument("--case-name")
    parser.add_argument("--tier", choices=("A", "B", "none"))
    parser.add_argument("--production-audit-enabled", action="store_true")
    parser.add_argument("--require-positivity", action="store_true")
    parser.add_argument("--require-gcl", action="store_true")
    parser.add_argument("--reproducibility-result", action="append", default=[], type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config = read_production_audit_config(args.frozen_config)
    tier = args.tier if args.tier is not None else str(config.get("tier", "none"))
    if tier not in {"A", "B", "none"}:
        print(f"error: invalid production audit tier {tier!r}", file=sys.stderr)
        return 2
    production_audit_enabled = bool(config.get("enabled", False)) or args.production_audit_enabled
    positivity_cfg = config.get("positivity") if isinstance(config.get("positivity"), dict) else {}
    gcl_cfg = config.get("gcl") if isinstance(config.get("gcl"), dict) else {}
    require_positivity = args.require_positivity or bool(positivity_cfg.get("enabled", False))
    require_gcl = args.require_gcl or bool(gcl_cfg.get("enabled", False))
    case_name = infer_case_name(args.run_info, args.history_hdf5, args.case_name)
    output = args.output or default_output_path(args.run_info, args.history_hdf5, case_name)

    payload = audit_summary(
        run_info_path=args.run_info,
        history_path=args.history_hdf5,
        tier=tier,
        production_audit_enabled=production_audit_enabled,
        reference_paths=args.reference,
        require_positivity=require_positivity,
        require_gcl=require_gcl,
        reproducibility_result_paths=args.reproducibility_result,
    )
    write_json(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
