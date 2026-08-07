#!/usr/bin/env python3
"""Run Stage I1 2D RZ weak GXII grey-FLD validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STAGE_ID = "I1"
ALL_MODES = (
    "i1a_grey_fld_1pct",
    "i1a_ale_on_probe",
    "i1_tmat_opacity_coupling",
    "i1b_grey_fld_5pct_powerstress",
    "i1c_grey_fld_10pct_powerstress",
    "i1d_grey_fld_25pct_powerstress",
)
DEFAULT_MODES = (
    "i1a_grey_fld_1pct",
    "i1a_ale_on_probe",
    "i1_tmat_opacity_coupling",
)
INFORMATIONAL_MODES = {"i1a_ale_on_probe"}
NR = 128
NZ = 256
RAYS_PER_BEAM = 72
T_END_DEFAULT_S = 5.0e-11
DT_DEFAULT_S = 2.0e-13
GXII_REFERENCE_E_ERG = 3600.0 * 1.0e7
MODE_ENERGY_FRACTION = {
    "i1a_grey_fld_1pct": 0.01,
    "i1a_ale_on_probe": 0.01,
    "i1_tmat_opacity_coupling": 0.01,
    "i1b_grey_fld_5pct_powerstress": 0.05,
    "i1c_grey_fld_10pct_powerstress": 0.10,
    "i1d_grey_fld_25pct_powerstress": 0.25,
}
MODE_PULSE_DURATION_S = {
    "i1a_grey_fld_1pct": 1.0e-9,
    "i1a_ale_on_probe": 1.0e-9,
    "i1_tmat_opacity_coupling": 1.0e-9,
    "i1b_grey_fld_5pct_powerstress": 1.0e-9,
    "i1c_grey_fld_10pct_powerstress": 1.0e-9,
    "i1d_grey_fld_25pct_powerstress": 1.0e-9,
}
KAPPA_A_CM2_G = 10.0
RHO_VAC_GCC = 1.0e-2
RHO_FLOOR_GCC = 1.0e-12
RHO_VAC_THRESHOLD_GCC = 0.011
E_FLOOR = 1.0e-300
ENERGY_RESIDUAL_MAX = 3.0e-3
CLAMP_MAX = 100
AXIS_CONTRAST_LIMIT = 30.0
AXIS_SHARE_LIMIT = 0.6
VACUUM_FLOOR_FRACTION_LIMIT = 0.10
BOUNDARY_TO_ADJ_P95_LIMIT = 100.0
NUMBER_RE = r"[-+0-9.eE]+"
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")
FLD_TIMING_RE = re.compile(r"\[fld_2d_rz_timing\](?P<body>.*)")
FLD_DIAGNOSTIC_RE = re.compile(r"\[fld_2d_rz_diagnostics_(?:h1|h2|h3|rogue)\](?P<body>.*)")
FLD_TIMING_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")
FLD_DIAGNOSTIC_ROW_KEYS = (
    "fld_cg_true_residual_l2_raw_max",
    "fld_cg_true_residual_max_raw_max",
    "fld_E_solver_raw_max",
    "fld_csr_diag_min_pos_min",
    "fld_csr_diag_max",
    "fld_csr_weak_diag_dom_count_total",
    "fld_csr_nonfinite_count_total",
    "fld_gershgorin_lower_min",
    "fld_gershgorin_upper_max",
    "fld_cg_pAp_min",
    "fld_cg_nonpos_pAp_count_total",
    "fld_cg_nonfinite_count_total",
    "fld_cg_recurrent_resid_last_max",
    "fld_Dcell_min",
    "fld_Dcell_max",
    "fld_Dcell_zero_count_total",
    "fld_Dcell_nonfinite_count_total",
    "fld_face_skip_D_count_total",
    "fld_face_skip_nonfinite_count_total",
    "fld_face_skip_dist_count_total",
    "fld_face_skip_dist_status",
    "fld_diag_fallback_count_total",
    "fld_pair_symm_max_diff",
    "fld_pair_symm_violation_count_total",
    "fld_newton_converged_count_total",
    "fld_newton_cap_hit_count_total",
    "fld_newton_invalid_count_total",
    "fld_newton_resid_abs_max",
    "fld_newton_resid_rel_max",
    "fld_newton_reject_count_total",
    "fld_newton_reject_resid_rel_max",
    "fld_rogue_max_abs_x_raw",
    "fld_rogue_max_abs_x_raw_cell",
    "fld_rogue_max_abs_x_raw_group",
    "fld_rogue_min_x_raw",
    "fld_rogue_min_x_raw_cell",
    "fld_rogue_min_x_raw_group",
    "fld_rogue_max_rad_E",
    "fld_rogue_max_rad_E_cell",
    "fld_rogue_max_rad_E_group",
    "fld_rogue_max_r_true",
    "fld_rogue_max_r_true_cell",
    "fld_rogue_max_r_true_group",
    "fld_rogue_max_E_solver_row",
    "fld_rogue_max_E_solver_row_cell",
    "fld_rogue_max_E_solver_row_group",
)
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_i1_grey_fld_weak\].*?mode=(?P<mode>\S+).*?"
    r"t_end_s=(?P<t_end>{n}).*?laser_input_erg=(?P<laser>{n}).*?"
    r"kappa_a_cm2_g=(?P<kappa>{n})".format(n=NUMBER_RE)
)


@dataclass(frozen=True)
class Mesh:
    nr: int
    nz: int
    r_nodes: Any
    z_nodes: Any
    vol: Any


def parse_csv(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    out = [part.strip().lower().replace("-", "_") for part in value.split(",") if part.strip()]
    bad = [part for part in out if part not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"invalid {label}: {', '.join(bad)}")
    if not out:
        raise argparse.ArgumentTypeError(f"{label} list is empty")
    return out


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed list is empty")
    return seeds


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def laser_input_erg_for(mode: str) -> float:
    return GXII_REFERENCE_E_ERG * MODE_ENERGY_FRACTION[mode]


def case_name(mode: str, seed: int, e_input: float, kappa_a: float = KAPPA_A_CM2_G) -> str:
    return (
        f"i1_{mode}_nr{NR}_nz{NZ}_rays{RAYS_PER_BEAM}"
        f"_e{safe_float_token(e_input)}_ka{safe_float_token(kappa_a)}_seed{seed}"
    )


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=False) + "\n")


def discover_results_dir(outdir: Path) -> Path:
    def has_numbered_plot(path: Path) -> bool:
        return path.exists() and any(
            len(p.stem) >= 5 and p.stem[-4:].isdigit() and p.stem[-5] == "_"
            for p in path.glob("*.h5")
        )

    direct = outdir / "results"
    if has_numbered_plot(direct):
        return direct
    candidates = []
    for path in outdir.parent.glob(f"{outdir.name}_*"):
        candidate = path / "results"
        if has_numbered_plot(candidate):
            candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)
    raise FileNotFoundError(f"no plot files found under {outdir} or {outdir}_*/results")


def locate_plot_files(results_dir: Path, run_id: str) -> list[Path]:
    def plot_index(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    files = sorted(
        (p for p in results_dir.glob(f"{run_id}_*.h5") if p.stem.removeprefix(f"{run_id}_").isdigit()),
        key=plot_index,
    )
    if files:
        return files
    return sorted((p for p in results_dir.glob("*_*.h5") if p.stem.rsplit("_", 1)[-1].isdigit()), key=plot_index)


def reshape_cells(values: Any, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == NR * NZ:
        return arr.reshape((NR, NZ))
    if arr.shape == (NR, NZ):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(NR, NZ)} or flat {NR * NZ}")


def reshape_nodes(values: Any, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == (NR + 1) * (NZ + 1):
        return arr.reshape((NR + 1, NZ + 1))
    if arr.shape == (NR + 1, NZ + 1):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(NR + 1, NZ + 1)}")


def reshape_rad(values: Any, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.shape == (NR * NZ, 1):
        return arr.reshape((NR, NZ))
    if arr.shape == (NR, NZ, 1):
        return arr[:, :, 0]
    if arr.shape == (NR, NZ):
        return arr
    if arr.size == NR * NZ:
        return arr.reshape((NR, NZ))
    raise ValueError(f"{name} has shape {arr.shape}, expected one grey group on {(NR, NZ)}")


def scalar_dataset(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else default


def read_snapshot(path: Path) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        return {
            "path": str(path),
            "t": t_value,
            "step": scalar_dataset(handle, "time_state/step", 0.0),
            "rho": reshape_cells(handle["hydro/rho"][()], "hydro/rho"),
            "Te": reshape_cells(handle["hydro/Te"][()], "hydro/Te"),
            "Ti": reshape_cells(handle["hydro/Ti"][()], "hydro/Ti"),
            "ee": reshape_cells(handle["hydro/ee"][()], "hydro/ee"),
            "ei": reshape_cells(handle["hydro/ei"][()], "hydro/ei"),
            "vol": reshape_cells(handle["hydro/vol"][()], "hydro/vol"),
            "rad_E": reshape_rad(handle["radiation/energy_density"][()], "radiation/energy_density"),
            "r_nodes": reshape_nodes(handle[r_node_path][()], r_node_path),
            "z_nodes": reshape_nodes(handle[z_node_path][()], z_node_path),
            "v_r_nodes": reshape_nodes(handle["mesh/v_r"][()], "mesh/v_r") if "mesh/v_r" in handle else None,
            "v_z_nodes": reshape_nodes(handle["mesh/v_z"][()], "mesh/v_z") if "mesh/v_z" in handle else None,
            "node_mass": reshape_nodes(handle["hydro/node_mass"][()], "hydro/node_mass")
            if "hydro/node_mass" in handle
            else None,
            "E_laser_deposited_state": scalar_dataset(handle, "time_state/E_laser_deposited", 0.0),
            "E_laser_escaped_state": scalar_dataset(handle, "time_state/E_laser_escaped", 0.0),
            "E_rad_escaped_state": scalar_dataset(handle, "time_state/E_rad_escaped", 0.0),
        }


def read_snapshots(results_dir: Path, run_id: str) -> list[dict[str, Any]]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
    return [read_snapshot(path) for path in files]


def read_history_series(results_dir: Path, run_id: str, path: str) -> list[float]:
    import h5py
    import numpy as np

    candidates = list(results_dir.glob(f"{run_id}_history.h5")) + list(results_dir.glob("*_history.h5"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        with h5py.File(candidate, "r") as handle:
            if path in handle:
                return [float(x) for x in np.asarray(handle[path][()], dtype=float).reshape(-1)]
    return []


def parse_fld_timing_metrics(text: str) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for line_re in (FLD_TIMING_RE, FLD_DIAGNOSTIC_RE):
        for match in line_re.finditer(text):
            tokens: dict[str, float] = {}
            for token in FLD_TIMING_TOKEN_RE.finditer(match.group("body")):
                try:
                    tokens[token.group("key")] = float(token.group("value"))
                except ValueError:
                    continue
            if tokens:
                rows.append(tokens)

    def finite_for(key: str) -> list[float]:
        return [
            float(row[key])
            for row in rows
            if key in row and math.isfinite(float(row[key]))
        ]

    face_skip_dist_values = finite_for("face_skip_dist")
    out: dict[str, Any] = {
        "fld_face_skip_dist_status": (
            "not_emitted"
            if not face_skip_dist_values
            else "emitted_nonzero"
            if any(value > 0.0 for value in face_skip_dist_values)
            else "emitted_zero"
        )
    }
    if not rows:
        return out
    values = finite_for("clamp_hits")
    if values:
        out["fld_clamp_hits_total"] = int(sum(values))
    values = finite_for("clamp_E_delta")
    if values:
        out["fld_clamp_energy_delta_total"] = sum(values)
    values = finite_for("min_x_raw")
    if values:
        out["fld_min_x_raw_min"] = min(values)
    values = finite_for("cg_true_resid_l2")
    if values:
        out["fld_cg_true_residual_l2_max"] = max(values)
    values = finite_for("cg_true_resid_max")
    if values:
        out["fld_cg_true_residual_max_max"] = max(values)
    values = finite_for("E_solver")
    if values:
        out["fld_E_solver_max"] = max(abs(value) for value in values)
    values = finite_for("outer_residual")
    if values:
        out["fld_outer_residual_max"] = max(values)
    values = finite_for("outer_iters")
    if values:
        out["fld_outer_iterations_max"] = int(max(values))
    max_token_map = {
        "cg_true_resid_l2_RAW": "fld_cg_true_residual_l2_raw_max",
        "cg_true_resid_max_RAW": "fld_cg_true_residual_max_raw_max",
        "E_solver_RAW": "fld_E_solver_raw_max",
        "diag_max": "fld_csr_diag_max",
        "gershgorin_upper_max": "fld_gershgorin_upper_max",
        "cg_recurrent_resid_last": "fld_cg_recurrent_resid_last_max",
        "Dcell_max": "fld_Dcell_max",
        "pair_symm_max_diff": "fld_pair_symm_max_diff",
        "newton_resid_abs_max": "fld_newton_resid_abs_max",
        "newton_resid_rel_max": "fld_newton_resid_rel_max",
        "newton_reject_resid_rel_max": "fld_newton_reject_resid_rel_max",
    }
    for token_key, out_key in max_token_map.items():
        values = finite_for(token_key)
        if values:
            out[out_key] = max(abs(value) for value in values) if "E_solver" in token_key else max(values)
    for token_key, out_key in (
        ("diag_min_pos", "fld_csr_diag_min_pos_min"),
        ("gershgorin_lower_min", "fld_gershgorin_lower_min"),
        ("cg_pAp_min", "fld_cg_pAp_min"),
        ("Dcell_min", "fld_Dcell_min"),
    ):
        values = finite_for(token_key)
        if values:
            out[out_key] = min(values)
    for token_key, out_key in (
        ("weak_diag_dom_count", "fld_csr_weak_diag_dom_count_total"),
        ("csr_nonfinite_count", "fld_csr_nonfinite_count_total"),
        ("cg_nonpos_pAp_count", "fld_cg_nonpos_pAp_count_total"),
        ("cg_nonfinite_count", "fld_cg_nonfinite_count_total"),
        ("Dcell_zero_count", "fld_Dcell_zero_count_total"),
        ("Dcell_nonfinite_count", "fld_Dcell_nonfinite_count_total"),
        ("face_skip_D", "fld_face_skip_D_count_total"),
        ("face_skip_nf", "fld_face_skip_nonfinite_count_total"),
        ("face_skip_dist", "fld_face_skip_dist_count_total"),
        ("diag_fallback", "fld_diag_fallback_count_total"),
        ("pair_symm_violation_count", "fld_pair_symm_violation_count_total"),
        ("newton_converged", "fld_newton_converged_count_total"),
        ("newton_cap_hit", "fld_newton_cap_hit_count_total"),
        ("newton_invalid", "fld_newton_invalid_count_total"),
        ("newton_reject", "fld_newton_reject_count_total"),
    ):
        values = finite_for(token_key)
        if values:
            out[out_key] = int(sum(values))

    def rogue_triplet(token_key: str, out_prefix: str, use_min: bool = False) -> None:
        best: dict[str, float] | None = None
        for row in rows:
            value = row.get(token_key)
            if value is None or not math.isfinite(float(value)):
                continue
            if best is None:
                best = row
                continue
            best_value = float(best[token_key])
            if (use_min and float(value) < best_value) or (not use_min and float(value) > best_value):
                best = row
        if best is None:
            return
        out[out_prefix] = float(best[token_key])
        cell = best.get(f"{token_key}_cell")
        group = best.get(f"{token_key}_group")
        if cell is not None and math.isfinite(float(cell)):
            out[f"{out_prefix}_cell"] = int(cell)
        if group is not None and math.isfinite(float(group)):
            out[f"{out_prefix}_group"] = int(group)

    rogue_triplet("rogue_max_abs_x_raw", "fld_rogue_max_abs_x_raw")
    rogue_triplet("rogue_min_x_raw", "fld_rogue_min_x_raw", use_min=True)
    rogue_triplet("rogue_max_rad_E", "fld_rogue_max_rad_E")
    rogue_triplet("rogue_max_r_true", "fld_rogue_max_r_true")
    rogue_triplet("rogue_max_E_solver_row", "fld_rogue_max_E_solver_row")
    return out


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out: dict[str, Any] = {
        "n_clamp": sum(int(match.group("count")) for match in CLAMP_RE.finditer(text)),
        "opacity_model_is_tmat": "opacity_model=tmat" in text,
    }
    out.update(parse_fld_timing_metrics(text))
    deck_match = DECK_LINE_RE.search(text)
    if deck_match:
        out["deck_mode"] = deck_match.group("mode")
        out["deck_t_end_s"] = float(deck_match.group("t_end"))
        out["deck_laser_input_erg"] = float(deck_match.group("laser"))
        out["deck_kappa_a_cm2_g"] = float(deck_match.group("kappa"))
    return out


def mesh_from_snapshot(snapshot: dict[str, Any]) -> Mesh:
    return Mesh(nr=NR, nz=NZ, r_nodes=snapshot["r_nodes"], z_nodes=snapshot["z_nodes"], vol=snapshot["vol"])


def cell_centers(snapshot: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np

    rn = np.asarray(snapshot["r_nodes"], dtype=float)
    zn = np.asarray(snapshot["z_nodes"], dtype=float)
    rc = 0.25 * (rn[:-1, :-1] + rn[1:, :-1] + rn[:-1, 1:] + rn[1:, 1:])
    zc = 0.25 * (zn[:-1, :-1] + zn[1:, :-1] + zn[:-1, 1:] + zn[1:, 1:])
    return rc, zc


def compute_E_rad_total(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(np.sum(np.asarray(rad_E, dtype=float) * np.asarray(mesh.vol, dtype=float)))


def compute_U_internal_total(snapshot: dict[str, Any], mesh: Mesh) -> float:
    import numpy as np

    return float(
        np.sum(
            np.asarray(snapshot["rho"], dtype=float)
            * (np.asarray(snapshot["ee"], dtype=float) + np.asarray(snapshot["ei"], dtype=float))
            * np.asarray(mesh.vol, dtype=float)
        )
    )


def compute_K_node_total(snapshot: dict[str, Any]) -> float:
    import numpy as np

    if snapshot.get("node_mass") is None or snapshot.get("v_r_nodes") is None or snapshot.get("v_z_nodes") is None:
        return 0.0
    node_mass = np.asarray(snapshot["node_mass"], dtype=float)
    v_r = np.asarray(snapshot["v_r_nodes"], dtype=float)
    v_z = np.asarray(snapshot["v_z_nodes"], dtype=float)
    return float(0.5 * np.sum(node_mass * (v_r * v_r + v_z * v_z)))


def z_profile(snapshot: dict[str, Any], field: Any) -> tuple[Any, Any]:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    weights = np.asarray(snapshot["vol"], dtype=float)
    zc = cell_centers(snapshot)[1]
    denom = np.maximum(np.sum(weights, axis=0), E_FLOOR)
    profile = np.sum(arr * weights, axis=0) / denom
    centers = np.mean(zc, axis=0)
    return centers, profile


def estimate_shock_z(snapshot: dict[str, Any]) -> float | None:
    import numpy as np

    zc, rho_profile = z_profile(snapshot, snapshot["rho"])
    candidate = np.where((zc > -0.012) & (zc < 0.012))[0]
    if candidate.size < 3:
        candidate = np.arange(rho_profile.size)
    if candidate.size < 3 or not np.all(np.isfinite(rho_profile[candidate])):
        return None
    jumps = np.abs(np.diff(rho_profile[candidate]))
    if jumps.size == 0 or float(np.nanmax(jumps)) <= 0.0:
        return None
    local = int(np.nanargmax(jumps))
    idx0 = int(candidate[local])
    idx1 = int(candidate[min(local + 1, candidate.size - 1)])
    return float(0.5 * (zc[idx0] + zc[idx1]))


def shock_metrics(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
    positions = []
    for snapshot in snapshots:
        z = estimate_shock_z(snapshot)
        if z is not None and math.isfinite(z):
            positions.append((float(snapshot["t"]), z))
    if len(positions) < 2:
        return {"shock_position_count": len(positions), "shock_displacement_cm": None, "shock_speed_cm_s": None}
    t_a, z_a = positions[0]
    t_b, z_b = positions[-1]
    return {
        "shock_position_count": len(positions),
        "shock_z_initial_cm": float(z_a),
        "shock_z_final_cm": float(z_b),
        "shock_displacement_cm": float(z_b - z_a),
        "shock_speed_cm_s": float((z_b - z_a) / max(t_b - t_a, E_FLOOR)),
    }


def final_status(exit_code: int, final_t: float | None, t_end_s: float) -> str:
    if exit_code != 0:
        return "crashed"
    if final_t is None or final_t < 0.999 * t_end_s:
        return "dt_collapse"
    return "completed"


def effective_laser_energies(
    mode: str, results_dir: Path, run_id: str, final: dict[str, Any]
) -> tuple[float, float, float]:
    incident_series = read_history_series(results_dir, run_id, "energy/laser_incident")
    escaped_series = read_history_series(results_dir, run_id, "energy/laser_escaped")
    deposited_series = read_history_series(results_dir, run_id, "energy/laser_deposited")
    incident = sum(max(x, 0.0) for x in incident_series)
    escaped = sum(max(x, 0.0) for x in escaped_series)
    deposited = deposited_series[-1] if deposited_series else final["E_laser_deposited_state"]
    if incident > 0.0:
        return incident, deposited, escaped
    state_total = final["E_laser_deposited_state"] + final["E_laser_escaped_state"]
    if state_total > 0.0:
        return state_total, final["E_laser_deposited_state"], final["E_laser_escaped_state"]
    return laser_input_erg_for(mode), final["E_laser_deposited_state"], final["E_laser_escaped_state"]


def radiation_shape_metrics(first: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    rad = np.asarray(final["rad_E"], dtype=float)
    rho = np.asarray(final["rho"], dtype=float)
    vol = np.asarray(final["vol"], dtype=float)
    finite = rad[np.isfinite(rad)]
    positive = finite[finite > 0.0]
    mean_positive = float(np.mean(positive)) if positive.size else 0.0
    max_rad = float(np.nanmax(rad)) if finite.size else math.inf
    spike_ratio = max_rad / max(mean_positive, E_FLOOR)

    nr, nz = rad.shape
    axis_mask = (
        (np.arange(nr)[:, None] < 3)
        & np.isfinite(rad)
        & (rad > 0.0)
        & (rho > 10.0 * RHO_FLOOR_GCC)
    )
    if np.any(axis_mask):
        axis_peak_field = np.where(axis_mask, rad, -math.inf)
        peak_idx = np.unravel_index(np.argmax(axis_peak_field), rad.shape)
        patch_i = slice(max(0, peak_idx[0] - 2), min(nr, peak_idx[0] + 3))
        patch_j = slice(max(0, peak_idx[1] - 2), min(nz, peak_idx[1] + 3))
        patch_E = rad[patch_i, patch_j].copy()
        local_pi = peak_idx[0] - patch_i.start
        local_pj = peak_idx[1] - patch_j.start
        patch_E[local_pi, local_pj] = np.nan
        finite_patch_E = patch_E[np.isfinite(patch_E)]
        median_neigh = float(np.median(finite_patch_E)) if finite_patch_E.size else 0.0
        axis_peak_local_contrast = float(rad[peak_idx]) / max(median_neigh, E_FLOOR)

        patch_EV = (rad * vol)[patch_i, patch_j]
        peak_EV = float(patch_EV[local_pi, local_pj])
        patch_EV_sum = float(np.sum(patch_EV))
        axis_peak_patch_energy_share = peak_EV / max(patch_EV_sum, E_FLOOR)
    else:
        axis_peak_local_contrast = 0.0
        axis_peak_patch_energy_share = 0.0

    rc, zc = cell_centers(first)
    radius = np.sqrt(rc * rc + zc * zc)
    edge_mask = (np.arange(nr)[:, None] >= nr - 1) | (np.arange(nz)[None, :] >= nz - 1) | (
        np.arange(nz)[None, :] < 1
    )
    low_rho_mask = rho <= RHO_VAC_THRESHOLD_GCC
    rad_vol = rad * vol
    vacuum_floor_energy = float(np.sum(rad_vol[low_rho_mask]))
    total_energy = float(np.sum(rad_vol))
    vacuum_floor_energy_fraction = vacuum_floor_energy / max(total_energy, E_FLOOR)

    boundary_low_density = edge_mask & low_rho_mask
    adjacent_interior_mask = ~edge_mask & ~low_rho_mask
    adjacent_rad = rad[adjacent_interior_mask]
    adjacent_rad = adjacent_rad[np.isfinite(adjacent_rad)]
    adj_E_ref = max(float(np.median(adjacent_rad)) if adjacent_rad.size else 0.0, E_FLOOR)
    ratios = rad[boundary_low_density] / adj_E_ref
    weights = vol[boundary_low_density]
    if ratios.size > 0 and float(np.sum(weights)) > 0.0:
        sorted_idx = np.argsort(ratios)
        sorted_ratios = ratios[sorted_idx]
        sorted_w = weights[sorted_idx]
        cumw = np.cumsum(sorted_w) / np.sum(sorted_w)
        boundary_to_adjacent_p95 = float(sorted_ratios[np.searchsorted(cumw, 0.95)])
    else:
        boundary_to_adjacent_p95 = 0.0

    vacuum_boundary = edge_mask & (radius > 0.011)
    first_boundary = np.asarray(first["rad_E"], dtype=float)[vacuum_boundary]
    final_boundary = rad[vacuum_boundary]
    initial_max = float(np.nanmax(first_boundary)) if first_boundary.size else 0.0
    final_max = float(np.nanmax(final_boundary)) if final_boundary.size else math.inf
    boundary_ratio = final_max / max(initial_max, E_FLOOR)

    return {
        "rad_E_min": float(np.nanmin(rad)) if finite.size else math.inf,
        "rad_E_max": max_rad,
        "rad_E_mean_positive": mean_positive,
        "axis_peak_local_contrast": axis_peak_local_contrast,
        "axis_peak_patch_energy_share": axis_peak_patch_energy_share,
        "rad_E_max_over_mean_positive": spike_ratio,
        "vacuum_floor_energy_fraction": vacuum_floor_energy_fraction,
        "boundary_to_adjacent_rad_E_p95": boundary_to_adjacent_p95,
        "vacuum_boundary_rad_E_initial_max": initial_max,
        "vacuum_boundary_rad_E_final_max": final_max,
        "vacuum_boundary_rad_E_ratio": boundary_ratio,
        "rad_E_nan_count": int(np.count_nonzero(~np.isfinite(rad))),
        "rad_E_negative_count": int(np.count_nonzero(rad < 0.0)),
    }


def compute_metrics(
    mode: str,
    snapshots: list[dict[str, Any]],
    results_dir: Path,
    run_id: str,
    log_metrics: dict[str, Any],
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    first = snapshots[0]
    final = snapshots[-1]
    first_mesh = mesh_from_snapshot(first)
    final_mesh = mesh_from_snapshot(final)
    u0 = compute_U_internal_total(first, first_mesh)
    u1 = compute_U_internal_total(final, final_mesh)
    k0 = compute_K_node_total(first)
    k1 = compute_K_node_total(final)
    rad0 = compute_E_rad_total(first["rad_E"], first_mesh)
    rad1 = compute_E_rad_total(final["rad_E"], final_mesh)
    laser_in, laser_dep, laser_esc = effective_laser_energies(mode, results_dir, run_id, final)
    rad_escaped_series = read_history_series(results_dir, run_id, "energy/radiation_escaped")
    rad_escaped = final["E_rad_escaped_state"] if final["E_rad_escaped_state"] != 0.0 else sum(
        max(x, 0.0) for x in rad_escaped_series
    )
    initial_total = u0 + k0 + rad0
    final_total_with_escape = u1 + k1 + rad1 + rad_escaped + laser_esc
    energy_residual = abs(final_total_with_escape - (initial_total + laser_in)) / max(
        abs(initial_total + laser_in), E_FLOOR
    )

    rho_all = np.concatenate([np.asarray(s["rho"], dtype=float).reshape(-1) for s in snapshots])
    te_all = np.concatenate([np.asarray(s["Te"], dtype=float).reshape(-1) for s in snapshots])
    ti_all = np.concatenate([np.asarray(s["Ti"], dtype=float).reshape(-1) for s in snapshots])
    rho_final = np.asarray(final["rho"], dtype=float)
    te_final = np.asarray(final["Te"], dtype=float)
    ti_final = np.asarray(final["Ti"], dtype=float)
    rad_final = np.asarray(final["rad_E"], dtype=float)
    history_sample_count = max(
        len(read_history_series(results_dir, run_id, "t")),
        len(read_history_series(results_dir, run_id, "cycle")),
        len(read_history_series(results_dir, run_id, "dt")),
        len(read_history_series(results_dir, run_id, "energy/laser_incident")),
        len(read_history_series(results_dir, run_id, "energy/radiation_escaped")),
    )
    history_or_step_count = max(history_sample_count, int(final.get("step", 0)))
    metrics: dict[str, Any] = {
        "final_t_s": float(final["t"]),
        "final_step": int(final.get("step", 0)),
        "completed_fraction": float(final["t"]) / max(float(log_metrics.get("deck_t_end_s", T_END_DEFAULT_S)), E_FLOOR),
        "E_initial_matter_rad": initial_total,
        "E_final_matter_rad_escaped": final_total_with_escape,
        "E_laser_incident": laser_in,
        "E_laser_deposited": laser_dep,
        "E_laser_escaped": laser_esc,
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "E_rad_escaped": rad_escaped,
        "energy_residual": energy_residual,
        "rho_min": float(np.nanmin(rho_all)),
        "rho_max": float(np.nanmax(rho_all)),
        "Te_min_eV": float(np.nanmin(te_all)),
        "Ti_min_eV": float(np.nanmin(ti_all)),
        "nan_count_rho_T": int(
            np.count_nonzero(~np.isfinite(rho_all))
            + np.count_nonzero(~np.isfinite(te_all))
            + np.count_nonzero(~np.isfinite(ti_all))
        ),
        "history_sample_count": history_sample_count,
        "history_or_step_count": history_or_step_count,
        "final_rho_bad_count": int(np.count_nonzero(~np.isfinite(rho_final)) + np.count_nonzero(rho_final <= 0.0)),
        "final_Te_bad_count": int(np.count_nonzero(~np.isfinite(te_final)) + np.count_nonzero(te_final <= 0.0)),
        "final_Ti_bad_count": int(np.count_nonzero(~np.isfinite(ti_final)) + np.count_nonzero(ti_final <= 0.0)),
        "final_rad_E_bad_count": int(np.count_nonzero(~np.isfinite(rad_final)) + np.count_nonzero(rad_final < 0.0)),
        "snapshot_count": len(snapshots),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(log_metrics)
    metrics.update(shock_metrics(snapshots))
    metrics.update(radiation_shape_metrics(first, final))
    return metrics


def evaluate_gates(metrics: dict[str, Any], mode: str) -> list[str]:
    failed: list[str] = []
    if mode == "i1_tmat_opacity_coupling":
        if metrics.get("completed_fraction", 0.0) < 0.99:
            failed.append("i1_tmat_opacity_completion")
        if metrics.get("history_or_step_count", 0) < 10:
            failed.append("i1_tmat_opacity_min_samples")
        if not metrics.get("opacity_model_is_tmat", False):
            failed.append("i1_tmat_opacity_model_evidence")
        if metrics.get("final_rad_E_bad_count", 1) != 0:
            failed.append("i1_tmat_opacity_rad_E_finite_nonnegative")
        if metrics.get("final_rho_bad_count", 1) != 0:
            failed.append("i1_tmat_opacity_rho_positive_finite")
        if metrics.get("final_Te_bad_count", 1) != 0:
            failed.append("i1_tmat_opacity_Te_positive_finite")
        if metrics.get("final_Ti_bad_count", 1) != 0:
            failed.append("i1_tmat_opacity_Ti_positive_finite")
        return failed
    if metrics.get("run_status") != "completed":
        failed.append("i1a_no_crash_or_dt_collapse")
    if metrics.get("n_clamp", math.inf) >= CLAMP_MAX:
        failed.append("i1a_no_runaway_clamps")
    if metrics.get("energy_residual", math.inf) >= ENERGY_RESIDUAL_MAX:
        failed.append("i1a_energy_residual")
    if metrics.get("shock_displacement_cm") is None or abs(float(metrics.get("shock_displacement_cm") or 0.0)) <= 0.0:
        failed.append("i1a_basic_shock_launch")
    if metrics.get("rad_E_nan_count", 1) != 0 or metrics.get("rad_E_negative_count", 1) != 0:
        failed.append("i1a_rad_E_finite_nonnegative")
    if (
        metrics.get("axis_peak_local_contrast", 0.0) > AXIS_CONTRAST_LIMIT
        and metrics.get("axis_peak_patch_energy_share", 0.0) > AXIS_SHARE_LIMIT
    ):
        failed.append("i1a_axial_spike_localized_defect")
    if (
        metrics.get("vacuum_floor_energy_fraction", 0.0) > VACUUM_FLOOR_FRACTION_LIMIT
        and metrics.get("boundary_to_adjacent_rad_E_p95", 0.0) > BOUNDARY_TO_ADJ_P95_LIMIT
    ):
        failed.append("i1a_vacuum_boundary_localized_defect")
    if metrics.get("nan_count_rho_T", 1) != 0:
        failed.append("i1a_no_nan_rho_T")
    if metrics.get("rho_min", 0.0) <= 0.0:
        failed.append("i1a_positive_density")
    return failed


def active_env(args: argparse.Namespace, mode: str, seed: int, outdir: Path) -> dict[str, str]:
    return {
        "TENRYU_I1_MODE": mode,
        "TENRYU_I1_NR": str(NR),
        "TENRYU_I1_NZ": str(NZ),
        "TENRYU_I1_SEED": str(seed),
        "TENRYU_I1_OUTDIR": str(outdir),
        "TENRYU_I1_RAYS_PER_BEAM": str(RAYS_PER_BEAM),
        "TENRYU_I1_LASER_INPUT_ERG": str(laser_input_erg_for(mode)),
        "TENRYU_I1_PULSE_DURATION_S": str(MODE_PULSE_DURATION_S[mode]),
        "TENRYU_I1_KAPPA_A_CM2_G": str(args.kappa_a_cm2_g),
        "TENRYU_I1_T_END_S": str(args.t_end_s),
        "TENRYU_I1_DT_INITIAL_S": str(args.dt_s),
        "TENRYU_I1_DT_MAX_S": str(args.dt_s),
        "TENRYU_I1_PLOT_EVERY_STEPS": str(args.plot_every_steps),
        "TENRYU_I1_MAX_STEPS": str(args.max_steps),
        "TENRYU_I1_VERBOSITY": args.verbosity,
        "TENRYU_I1_ALE": "1" if args.ale else "0",
        "TENRYU_DT_LINEAGE": "1" if args.dt_lineage else "0",
        "TENRYU_DT_LINEAGE_OUTDIR": str(outdir),
    }


def run_one(args: argparse.Namespace, mode: str, seed: int) -> dict[str, Any]:
    e_input = laser_input_erg_for(mode)
    run_id = case_name(mode, seed, e_input, args.kappa_a_cm2_g)
    ladder_row = f"{mode}_nr{NR}_nz{NZ}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    wall_time_s = time.monotonic() - start

    log_metrics = parse_log_metadata(log_path)
    metrics: dict[str, Any] = {**log_metrics, "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    failed_gates: list[str] = []
    failure_class: str | None = None
    result = "PASS"

    try:
        if timed_out:
            metrics["run_status"] = "wall_time_timeout"
            result = "PASS" if mode in INFORMATIONAL_MODES else "FAIL"
            failed_gates = [] if mode in INFORMATIONAL_MODES else ["run_failure"]
            failure_class = None if mode in INFORMATIONAL_MODES else "run_failure"
        elif proc.returncode != 0:
            metrics["run_status"] = final_status(proc.returncode, None, args.t_end_s)
            result = "PASS" if mode in INFORMATIONAL_MODES else "CRASHED"
            failed_gates = [] if mode in INFORMATIONAL_MODES else ["run_failure"]
            failure_class = None if mode in INFORMATIONAL_MODES else "run_failure"
        else:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            snapshots = read_snapshots(results_dir, run_id)
            metrics = compute_metrics(mode, snapshots, results_dir, run_id, log_metrics, wall_time_s, bytes_out)
            t_hist = read_history_series(results_dir, run_id, "t")
            true_final_t = max(float(metrics.get("final_t_s") or 0.0), t_hist[-1] if t_hist else 0.0)
            metrics["true_final_t_s"] = true_final_t
            metrics["run_status"] = final_status(proc.returncode, true_final_t, args.t_end_s)
            if mode in INFORMATIONAL_MODES:
                result = "PASS"
                failed_gates = []
                failure_class = None
            else:
                failed_gates = evaluate_gates(metrics, mode)
                result = "PASS" if not failed_gates else "FAIL"
                failure_class = None if not failed_gates else "gate_failure"
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["analysis_error"] = str(exc)
        metrics["run_status"] = "dt_collapse" if proc.returncode == 0 else "crashed"
        if mode in INFORMATIONAL_MODES:
            result = "PASS"
            failed_gates = []
            failure_class = None
        else:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[run_i1] analysis failed: {exc}\n")

    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": NR,
        "nz": NZ,
        "rays_per_beam": RAYS_PER_BEAM,
        "groups": 1,
        "kappa_a_cm2_g": args.kappa_a_cm2_g,
        "mode_energy_fraction": MODE_ENERGY_FRACTION[mode],
        "mode_pulse_duration_s": MODE_PULSE_DURATION_S[mode],
        "informational": mode in INFORMATIONAL_MODES,
        "result": result,
        "failure_class": failure_class,
        "failed_gates": failed_gates,
        "exit_code": proc.returncode,
        "wall_time_s": wall_time_s,
        "metrics": metrics,
        "env_overrides": env_overrides,
    }
    write_json(Path(metrics_ref), payload)
    return row_from_payload(payload, metrics_ref)


def row_from_payload(payload: dict[str, Any], metrics_ref: str) -> dict[str, Any]:
    metrics = payload["metrics"]
    row = {
        "run_id": payload["run_id"],
        "stage_id": payload["stage_id"],
        "ladder_row": payload["ladder_row"],
        "seed": payload["seed"],
        "deck_path": payload["deck_path"],
        "mode": payload["mode"],
        "informational": payload["informational"],
        "nr": payload["nr"],
        "nz": payload["nz"],
        "rays_per_beam": payload["rays_per_beam"],
        "groups": payload["groups"],
        "kappa_a_cm2_g": payload["kappa_a_cm2_g"],
        "mode_energy_fraction": payload["mode_energy_fraction"],
        "mode_pulse_duration_s": payload["mode_pulse_duration_s"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "run_status": metrics.get("run_status"),
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "final_t_s": metrics.get("final_t_s"),
        "energy_residual": metrics.get("energy_residual"),
        "n_clamp": metrics.get("n_clamp"),
        "shock_displacement_cm": metrics.get("shock_displacement_cm"),
        "axis_peak_local_contrast": metrics.get("axis_peak_local_contrast"),
        "axis_peak_patch_energy_share": metrics.get("axis_peak_patch_energy_share"),
        "rad_E_max_over_mean_positive": metrics.get("rad_E_max_over_mean_positive"),
        "vacuum_floor_energy_fraction": metrics.get("vacuum_floor_energy_fraction"),
        "boundary_to_adjacent_rad_E_p95": metrics.get("boundary_to_adjacent_rad_E_p95"),
        "vacuum_boundary_rad_E_ratio": metrics.get("vacuum_boundary_rad_E_ratio"),
        "fld_clamp_hits_total": metrics.get("fld_clamp_hits_total"),
        "fld_clamp_energy_delta_total": metrics.get("fld_clamp_energy_delta_total"),
        "fld_min_x_raw_min": metrics.get("fld_min_x_raw_min"),
        "fld_cg_true_residual_l2_max": metrics.get("fld_cg_true_residual_l2_max"),
        "fld_cg_true_residual_max_max": metrics.get("fld_cg_true_residual_max_max"),
        "fld_E_solver_max": metrics.get("fld_E_solver_max"),
        "fld_outer_residual_max": metrics.get("fld_outer_residual_max"),
    }
    for key in FLD_DIAGNOSTIC_ROW_KEYS:
        row[key] = metrics.get(key)
    return row


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int]]:
    return [(mode, seed) for mode in args.mode_list for seed in args.seed_list]


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def add_energy_residual_ratios(rows: list[dict[str, Any]]) -> None:
    i1a_by_seed = {
        row["seed"]: row["energy_residual"]
        for row in rows
        if row["mode"] == "i1a_grey_fld_1pct" and isinstance(row.get("energy_residual"), (int, float))
    }
    i1a_residual = next(iter(i1a_by_seed.values()), None)
    for row in rows:
        baseline = i1a_by_seed.get(row["seed"], i1a_residual)
        residual = row.get("energy_residual")
        row["energy_residual_ratio_vs_i1a"] = (
            float(residual) / float(baseline)
            if isinstance(residual, (int, float))
            and isinstance(baseline, (int, float))
            and math.isfinite(float(residual))
            and math.isfinite(float(baseline))
            and float(baseline) != 0.0
            else None
        )


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    add_energy_residual_ratios(rows)
    gating = [row for row in rows if not row["informational"]]
    aggregate = {
        "total_runs": len(rows),
        "gating_runs": len(gating),
        "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
        "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
        "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
    }
    for key in (
        "wall_time_s",
        "energy_residual",
        "energy_residual_ratio_vs_i1a",
        "n_clamp",
        "axis_peak_local_contrast",
        "axis_peak_patch_energy_share",
        "rad_E_max_over_mean_positive",
        "vacuum_floor_energy_fraction",
        "boundary_to_adjacent_rad_E_p95",
        "vacuum_boundary_rad_E_ratio",
    ):
        values = finite_values(rows, key)
        aggregate[f"{key}_max"] = max(values) if values else None
    values = finite_values(rows, "fld_clamp_hits_total")
    aggregate["fld_clamp_hits_total"] = int(sum(values)) if values else None
    values = finite_values(rows, "fld_clamp_energy_delta_total")
    aggregate["fld_clamp_energy_delta_total"] = sum(values) if values else None
    values = finite_values(rows, "fld_min_x_raw_min")
    aggregate["fld_min_x_raw_min"] = min(values) if values else None
    for key in (
        "fld_cg_true_residual_l2_max",
        "fld_cg_true_residual_max_max",
        "fld_E_solver_max",
        "fld_outer_residual_max",
    ):
        values = finite_values(rows, key)
        aggregate[key] = max(values) if values else None
    for key in FLD_DIAGNOSTIC_ROW_KEYS:
        if key == "fld_face_skip_dist_status":
            continue
        if key.startswith("fld_rogue_") or key.endswith("_cell") or key.endswith("_group"):
            continue
        values = finite_values(rows, key)
        if not values:
            aggregate[key] = None
        elif key.endswith("_count_total"):
            aggregate[key] = int(sum(values))
        elif key.endswith("_min") or key.endswith("_min_pos_min") or key == "fld_cg_pAp_min":
            aggregate[key] = min(values)
        else:
            aggregate[key] = max(values)

    def aggregate_rogue(key: str, use_min: bool = False) -> None:
        best: dict[str, Any] | None = None
        for row in rows:
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                continue
            if best is None:
                best = row
                continue
            best_value = float(best[key])
            if (use_min and float(value) < best_value) or (not use_min and float(value) > best_value):
                best = row
        if best is None:
            aggregate[key] = None
            aggregate[f"{key}_cell"] = None
            aggregate[f"{key}_group"] = None
            return
        aggregate[key] = best.get(key)
        aggregate[f"{key}_cell"] = best.get(f"{key}_cell")
        aggregate[f"{key}_group"] = best.get(f"{key}_group")

    aggregate_rogue("fld_rogue_max_abs_x_raw")
    aggregate_rogue("fld_rogue_min_x_raw", use_min=True)
    aggregate_rogue("fld_rogue_max_rad_E")
    aggregate_rogue("fld_rogue_max_r_true")
    aggregate_rogue("fld_rogue_max_E_solver_row")
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": aggregate,
    }
    write_json(Path(args.summary_json), payload)


def print_summary(rows: list[dict[str, Any]]) -> None:
    add_energy_residual_ratios(rows)
    print("Stage I1 weak GXII grey-FLD sweep summary")
    columns = [
        "ladder_row",
        "seed",
        "result",
        "run_status",
        "energy_residual",
        "energy_residual_ratio_vs_i1a",
        "n_clamp",
        "shock_displacement_cm",
        "axis_peak_local_contrast",
        "axis_peak_patch_energy_share",
        "rad_E_max_over_mean_positive",
        "vacuum_floor_energy_fraction",
        "boundary_to_adjacent_rad_E_p95",
        "vacuum_boundary_rad_E_ratio",
        "fld_clamp_hits_total",
        "fld_clamp_energy_delta_total",
        "fld_min_x_raw_min",
        "fld_cg_true_residual_l2_max",
        "fld_cg_true_residual_max_max",
        "fld_E_solver_max",
        "fld_outer_residual_max",
        *FLD_DIAGNOSTIC_ROW_KEYS,
        "failed_gates",
    ]
    print(" ".join(columns))
    for row in rows:
        print(" ".join(str(row.get(column)) for column in columns))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_i1_grey_fld_weak.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/i1_runs")
    parser.add_argument(
        "--modes",
        "--mode-list",
        dest="mode_list",
        type=lambda v: parse_csv(v, ALL_MODES, "mode"),
        default=list(DEFAULT_MODES),
    )
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--smoke", action="store_true", help="kept for parity; selected modes still run at 128x256")
    parser.add_argument("--kappa-a-cm2-g", type=float, default=KAPPA_A_CM2_G)
    parser.add_argument("--t-end-s", type=float, default=T_END_DEFAULT_S)
    parser.add_argument("--dt-s", type=float, default=DT_DEFAULT_S)
    parser.add_argument("--plot-every-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument(
        "--ale",
        action="store_true",
        help="Enable I1 ALE-on (TENRYU_I1_ALE=1) — wires corner-J trigger + volume-rate CFL + trial-volume CFL primitives",
    )
    parser.add_argument(
        "--dt-lineage",
        action="store_true",
        help="Enable dt-lineage JSONL output (TENRYU_DT_LINEAGE=1) for diagnostic probes; outputs to per-run outdir",
    )
    parser.add_argument(
        "--wall-time-s",
        type=float,
        default=None,
        help="Optional wall-time timeout in seconds (subprocess.run timeout). Default: no timeout.",
    )
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/I1/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/I1/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = [run_one(args, mode, seed) for mode, seed in matrix_rows(args)]
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    gating_rows = [row for row in rows if not row["informational"]]
    return 0 if gating_rows and all(row["result"] == "PASS" for row in gating_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
