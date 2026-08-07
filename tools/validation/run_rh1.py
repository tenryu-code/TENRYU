#!/usr/bin/env python3
"""Run Stage RH1 2D RZ hydro+FLD validation sweep."""

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


STAGE_ID = "RH1"
MODES = (
    "shock_tube_grey",
    "planar_radiative_shock",
    "cylindrical_blast_with_rad",
    "optically_thin_precursor",
    "multigroup_tmat",
    "lowrie_edwards_energy_proxy",
    "lowrie_edwards_table_init_shock_frame",
)
STANDARD_MODES = MODES[:4]

FULL_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    mode: ((64, 128),) for mode in MODES
}
SMOKE_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    mode: (meshes[0],) for mode, meshes in FULL_MESHES_BY_MODE.items()
}

NUMBER_RE = r"[-+0-9.eE]+"
FLD_TIMING_RE = re.compile(
    r"\[fld_2d_rz_timing\](?P<body>.*)"
)
FLD_TIMING_TOKEN_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_rh1_hydro_fld\].*?groups=(?P<groups>\d+).*?"
    r"rho_gcc=(?P<rho>{n}).*?kappa_a_cm2_g=(?P<kappa>{n}).*?"
    r"t_end_s=(?P<t_end>{n})".format(n=NUMBER_RE)
)

A_EV = 1.3720e2
E_FLOOR = 1.0e-300
MAX_OUTER_ITERATIONS = 50
ITERATION_CAP_FRACTION = 0.8
ENERGY_TOL = 1.0e-3
MG_TMAT_E_BALANCE_TOL = 1.0e-3
MG_TMAT_MIN_GROUP_OCCUPANCY = 2
PLANAR_SELF_REF_TOL_CM = 5.0e-2
# B2: gate value pending re-freeze after GPU rerun (see b2 note)
LE_PROXY_TOL = 0.20
RH1_SUBSHOCK_REFERENCE_MODEL = "RH1_energy_only_FLD_CED_subshock"
RH1_SUBSHOCK_REFERENCE_TABLE_PATH = Path(
    "tests/verification/data/rh1_subshock_reference/rh1_subshock_M3.json"
)
RH1_SUBSHOCK_REFERENCE_TABLE_KEYS = (
    "x_cm",
    "rho_g_per_cc",
    "u_cm_per_s",
    "T_mat_eV",
    "T_rad_eV",
    "E_r_erg_per_cc",
    "F_r_erg_per_cm2_s",
)
RH1_SUBSHOCK_REFERENCE_TABLE_NESTED_KEYS = {
    "x_cm": "x_cm",
    "rho_g_per_cc": "rho_g_per_cc",
    "u_cm_per_s": "u_cm_per_s",
    "T_mat_eV": "T_eV",
    "T_rad_eV": "T_rad_eV",
    "E_r_erg_per_cc": "E_rad_erg_per_cm3",
    "F_r_erg_per_cm2_s": "F_rad_erg_per_cm2_s",
}


@dataclass(frozen=True)
class Mesh:
    nr: int
    nz: int
    r_nodes: Any
    z_nodes: Any
    vol: Any

    @property
    def dr(self) -> Any:
        import numpy as np

        return np.asarray(self.r_nodes[1:, :-1], dtype=float) - np.asarray(self.r_nodes[:-1, :-1], dtype=float)

    @property
    def dz(self) -> Any:
        import numpy as np

        return np.asarray(self.z_nodes[:-1, 1:], dtype=float) - np.asarray(self.z_nodes[:-1, :-1], dtype=float)

    @property
    def dx_min(self) -> Any:
        import numpy as np

        return np.minimum(np.abs(self.dr), np.abs(self.dz))


def parse_mesh_list(value: str) -> list[tuple[int, int]]:
    out = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        sep = "x" if "x" in item.lower() else ","
        parts = [part.strip() for part in item.lower().split(sep)]
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(f"invalid mesh entry: {item!r}")
        nr = int(parts[0])
        nz = int(parts[1])
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("mesh dimensions must be positive")
        out.append((nr, nz))
    if not out:
        raise argparse.ArgumentTypeError("mesh list is empty")
    return out


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


def mode_groups(mode: str) -> int:
    if mode == "multigroup_tmat":
        return 6
    return 1


def mode_rho_default(mode: str) -> float:
    if mode == "multigroup_tmat":
        return 1.0
    if mode == "cylindrical_blast_with_rad":
        return 1.0e-3
    if mode == "optically_thin_precursor":
        return 1.0e-5
    if mode == "lowrie_edwards_table_init_shock_frame":
        return 1.0
    return 1.0


def mode_kappa_default(mode: str) -> float:
    if mode == "multigroup_tmat":
        return 0.0
    if mode in ("lowrie_edwards_energy_proxy", "lowrie_edwards_table_init_shock_frame"):
        return 1.0
    if mode == "shock_tube_grey":
        return 10.0
    if mode == "planar_radiative_shock":
        return 100.0
    if mode == "optically_thin_precursor":
        return 0.01
    return 10.0


def case_name(mode: str, nr: int, nz: int, rho_gcc: float, kappa_a: float, seed: int, ale: bool) -> str:
    return (
        f"rh1_{mode}_nr{nr}_nz{nz}_rho{safe_float_token(rho_gcc)}"
        f"_ka{safe_float_token(kappa_a)}_g{mode_groups(mode)}_seed{seed}"
        + ("_ale" if ale else "")
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


def reshape_cells(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    if arr.shape == (nr, nz):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr, nz)} or flat {nr * nz}")


def reshape_nodes(values: Any, nr: int, nz: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == (nr + 1) * (nz + 1):
        return arr.reshape((nr + 1, nz + 1))
    if arr.shape == (nr + 1, nz + 1):
        return arr
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr + 1, nz + 1)}")


def reshape_rad(values: Any, nr: int, nz: int, groups: int, name: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.shape == (nr * nz, groups):
        return arr.reshape((nr, nz, groups))
    if arr.shape == (nr, nz, groups):
        return arr
    if groups == 1 and arr.size == nr * nz:
        return arr.reshape((nr, nz, 1))
    if arr.size == nr * nz * groups:
        return arr.reshape((nr, nz, groups))
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr * nz, groups)}")


def scalar_dataset(handle: Any, path: str, default: float | None = None) -> float | None:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    if arr.size == 0:
        return default
    return float(arr[-1])


def read_snapshot(path: Path, nr: int, nz: int, groups: int) -> dict[str, Any]:
    import h5py
    import numpy as np

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        vol = reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol")
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        snap = {
            "path": str(path),
            "t": t_value,
            "rho": reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho"),
            "Te": reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te"),
            "Ti": reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti"),
            "ee": reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee"),
            "ei": reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei"),
            "mass": reshape_cells(handle["hydro/mass"][()], nr, nz, "hydro/mass")
            if "hydro/mass" in handle
            else None,
            "vol": vol,
            "rad_E_groups": reshape_rad(handle["radiation/energy_density"][()], nr, nz, groups, "radiation/energy_density"),
            "r_nodes": reshape_nodes(handle[r_node_path][()], nr, nz, r_node_path),
            "z_nodes": reshape_nodes(handle[z_node_path][()], nr, nz, z_node_path),
            "v_r_nodes": reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r")
            if "mesh/v_r" in handle
            else None,
            "v_z_nodes": reshape_nodes(handle["mesh/v_z"][()], nr, nz, "mesh/v_z")
            if "mesh/v_z" in handle
            else None,
            "node_mass": reshape_nodes(handle["hydro/node_mass"][()], nr, nz, "hydro/node_mass")
            if "hydro/node_mass" in handle
            else None,
            "E_rad_escaped_state": scalar_dataset(handle, "time_state/E_rad_escaped", 0.0),
            "E_numerical_loss_state": scalar_dataset(handle, "time_state/E_numerical_loss", 0.0),
            "E_floor_injected_state": scalar_dataset(handle, "time_state/E_floor_injected", 0.0),
        }
        snap["rad_E"] = np.sum(snap["rad_E_groups"], axis=2)
        return snap


def read_snapshots(results_dir: Path, run_id: str, nr: int, nz: int, groups: int) -> list[dict[str, Any]]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < 2:
        raise RuntimeError("expected at least initial and final plot snapshots")
    return [read_snapshot(path, nr, nz, groups) for path in files]


def read_history_scalar(results_dir: Path, run_id: str, path: str) -> list[float]:
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


def parse_log_metadata(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out: dict[str, Any] = {
        "n_clamp": sum(int(match.group("count")) for match in CLAMP_RE.finditer(text)),
        "fld_outer_iterations_max": None,
        "fld_outer_iterations_series": [],
        "fld_cg_iterations_total_max": None,
        "fld_cg_iterations_max": None,
        "fld_timing_line_count": 0,
    }
    deck_match = DECK_LINE_RE.search(text)
    if deck_match:
        out["deck_groups"] = int(deck_match.group("groups"))
        out["deck_rho_gcc"] = float(deck_match.group("rho"))
        out["deck_kappa_a_cm2_g"] = float(deck_match.group("kappa"))
        out["deck_t_end_s"] = float(deck_match.group("t_end"))
    outer = []
    cg_total = []
    cg_max = []
    for match in FLD_TIMING_RE.finditer(text):
        tokens = {
            token.group("key"): token.group("value")
            for token in FLD_TIMING_TOKEN_RE.finditer(match.group("body"))
        }
        if "outer_iters" in tokens:
            outer.append(int(tokens["outer_iters"]))
        if "cg_iters_total" in tokens:
            cg_total.append(int(tokens["cg_iters_total"]))
        if "cg_iters_max" in tokens:
            cg_max.append(int(tokens["cg_iters_max"]))
    if outer:
        out["fld_outer_iterations_series"] = outer
        out["fld_outer_iterations_max"] = max(outer)
        if cg_total:
            out["fld_cg_iterations_total_max"] = max(cg_total)
        if cg_max:
            out["fld_cg_iterations_max"] = max(cg_max)
        out["fld_timing_line_count"] = len(outer)
    return out


def compute_E_rad_total(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(np.sum(np.asarray(rad_E, dtype=float) * np.asarray(mesh.vol, dtype=float)))


def compute_U_internal_total(rho: Any, ee: Any, ei: Any, mesh: Mesh) -> float:
    import numpy as np

    return float(
        np.sum(
            np.asarray(rho, dtype=float)
            * (np.asarray(ee, dtype=float) + np.asarray(ei, dtype=float))
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


def z_profile(field: Any, vol: Any) -> list[float]:
    import numpy as np

    arr = np.asarray(field, dtype=float)
    weights = np.asarray(vol, dtype=float)
    denom = np.maximum(np.sum(weights, axis=0), E_FLOOR)
    return [float(x) for x in np.sum(arr * weights, axis=0) / denom]


def z_centers(mesh: Mesh) -> list[float]:
    import numpy as np

    z_nodes = np.asarray(mesh.z_nodes, dtype=float)
    centers = 0.25 * (z_nodes[:-1, :-1] + z_nodes[1:, :-1] + z_nodes[:-1, 1:] + z_nodes[1:, 1:])
    return [float(x) for x in np.mean(centers, axis=0)]


def rel_l2(got: Any, ref: Any) -> float:
    import numpy as np

    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if got_arr.size == 0 or ref_arr.size == 0:
        return math.inf
    denom = float(np.sqrt(np.sum(ref_arr * ref_arr)))
    return float(np.sqrt(np.sum((got_arr - ref_arr) ** 2)) / max(denom, E_FLOOR))


def rad_E_to_Trad(rad_E: Any) -> Any:
    import numpy as np

    return np.power(np.maximum(np.asarray(rad_E, dtype=float) / A_EV, 0.0), 0.25)


def _reference_table_values(payload: dict[str, Any], key: str) -> list[float]:
    values = payload.get(key)
    if values is None:
        table = payload.get("table", {})
        values = (
            table.get(RH1_SUBSHOCK_REFERENCE_TABLE_NESTED_KEYS[key])
            if isinstance(table, dict)
            else None
        )
    if values is None:
        raise RuntimeError(f"RH1 subshock reference table missing required key: {key}")
    return [float(value) for value in values]


def rh1_subshock_reference_table_path(
    path: Path = RH1_SUBSHOCK_REFERENCE_TABLE_PATH,
) -> Path:
    table_path = Path(os.environ.get("TENRYU_RH1_SUBSHOCK_REFERENCE_TABLE", str(path)))
    if not table_path.exists():
        raise FileNotFoundError(
            f"RH1 subshock reference table not found: {table_path}; "
            "set TENRYU_RH1_SUBSHOCK_REFERENCE_TABLE to override"
        )
    return table_path


def load_rh1_subshock_reference_table(
    path: Path,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    table = {
        key: _reference_table_values(payload, key)
        for key in RH1_SUBSHOCK_REFERENCE_TABLE_KEYS
    }
    return payload, table


def compute_le_proxy_metrics(
    final: dict[str, Any], mesh: Mesh, le_table_path: Path, mode: str
) -> dict[str, Any]:
    import numpy as np

    ref_payload, ref = load_rh1_subshock_reference_table(le_table_path)

    centers = np.asarray(z_centers(mesh), dtype=float)
    rho_t = np.asarray(z_profile(final["rho"], mesh.vol), dtype=float)
    T_mat_t = np.asarray(z_profile(final["Te"], mesh.vol), dtype=float)
    T_rad_t = np.asarray(z_profile(rad_E_to_Trad(final["rad_E"]), mesh.vol), dtype=float)

    ref_x = np.asarray(ref["x_cm"], dtype=float)
    if mode == "lowrie_edwards_table_init_shock_frame":
        shock_pos_t = None
        z_aligned = centers
        gate_status = "informational_only_per_BC_and_frame_limitation"
        deck_setup = "table_init_shock_frame"
        initial_profile_match = True
        residual_risk = (
            "Informational-only metric. The table-initialized deck starts from the shock-frame RH1 "
            "energy-only FLD-CED subshock "
            "profile, but TENRYU 2D RZ hydro lacks state-supply z boundary conditions, so the Lagrangian "
            "frame can drift during the short-window run. Use this only for trend tracking, not strict "
            "gate enforcement."
        )
    else:
        if centers.size < 3 or rho_t.size < 3:
            raise ValueError("RH1 subshock reference comparison needs at least 3 z cells")
        grad = np.gradient(rho_t, centers)
        if not np.all(np.isfinite(grad)) or float(np.nanmax(np.abs(grad))) <= 0.0:
            raise ValueError("RH1 subshock reference shock front unavailable")

        shock_pos_t = float(centers[int(np.nanargmax(np.abs(grad)))])
        z_aligned = centers - shock_pos_t
        gate_status = "informational_only_per_deck_mismatch"
        deck_setup = "piston_wall_transient"
        initial_profile_match = False
        residual_risk = (
            "Informational-only metric. The piston-transient deck setup does not match the shock-frame "
            "RH1 energy-only FLD-CED subshock reference. T_mat/T_rad profile L2 is expected to exceed "
            "20% in the developing-shock regime. Use this only for trend tracking, not strict gate enforcement. "
            "The kappa->0 calibration shows the gate IS sensitive to radiation coupling "
            "(T_rad L2 worsens 0.61 -> 3.35), so a regression that significantly worsens the L2 metric "
            "should still be reviewed."
        )

    overlap = (z_aligned >= ref_x[0]) & (z_aligned <= ref_x[-1])
    overlap_count = int(np.count_nonzero(overlap))
    if overlap_count < 3:
        raise ValueError("RH1 subshock reference overlap has fewer than 3 z cells")

    ref_rho = np.interp(z_aligned, ref_x, np.asarray(ref["rho_g_per_cc"], dtype=float))
    ref_T_mat = np.interp(z_aligned, ref_x, np.asarray(ref["T_mat_eV"], dtype=float))
    ref_T_rad = np.interp(z_aligned, ref_x, np.asarray(ref["T_rad_eV"], dtype=float))
    domain_span = float(ref_x[-1] - ref_x[0])
    tenryu_span = float(np.max(z_aligned) - np.min(z_aligned))

    return {
        "le_proxy_rho_l2": rel_l2(rho_t[overlap], ref_rho[overlap]),
        "le_proxy_T_mat_l2": rel_l2(T_mat_t[overlap], ref_T_mat[overlap]),
        "le_proxy_T_rad_l2": rel_l2(T_rad_t[overlap], ref_T_rad[overlap]),
        "le_proxy_shock_pos_cm": shock_pos_t,
        "le_proxy_overlap_cell_count": overlap_count,
        "le_proxy_overlap_fraction": float(overlap_count / max(centers.size, 1)),
        "le_proxy_ref_domain_cm": [float(ref_x[0]), float(ref_x[-1])],
        "le_proxy_tenryu_aligned_domain_cm": [float(np.min(z_aligned)), float(np.max(z_aligned))],
        "le_proxy_ref_domain_span_cm": domain_span,
        "le_proxy_tenryu_aligned_span_cm": tenryu_span,
        "le_proxy_table_path": str(le_table_path),
        "le_proxy_reference_model_rh1": ref_payload.get("reference_model_rh1"),
        "le_proxy_reference_model_expected": RH1_SUBSHOCK_REFERENCE_MODEL,
        "le_proxy_reference_schema_rh1": ref_payload.get("schema_version_rh1"),
        "le_proxy_gate_status": gate_status,
        "le_proxy_deck_setup": deck_setup,
        "le_proxy_initial_profile_match": initial_profile_match,
        "le_proxy_reference_setup": "shock_frame_stationary_energy_only_FLD_CED",
        "le_proxy_flux_limiter": "none",
        "le_proxy_eos_assumption": "deck non-TMAT path uses ideal_gas gamma=5/3, A=1, Zbar=1",
        "le_proxy_residual_risk": residual_risk,
    }


def estimate_shock_z(snapshot: dict[str, Any], mesh: Mesh) -> float | None:
    import numpy as np

    profile = np.asarray(z_profile(snapshot["rho"], snapshot["vol"]), dtype=float)
    centers = np.asarray(z_centers(mesh), dtype=float)
    if profile.size < 3 or not np.all(np.isfinite(profile)):
        return None
    jumps = np.abs(np.diff(profile))
    if jumps.size == 0 or float(np.nanmax(jumps)) <= 0.0:
        return None
    idx = int(np.nanargmax(jumps))
    return float(0.5 * (centers[idx] + centers[idx + 1]))


def mesh_from_snapshot(snapshot: dict[str, Any], nr: int, nz: int) -> Mesh:
    return Mesh(nr=nr, nz=nz, r_nodes=snapshot["r_nodes"], z_nodes=snapshot["z_nodes"], vol=snapshot["vol"])


def matter_energy(snapshot: dict[str, Any], mesh: Mesh) -> float:
    return compute_U_internal_total(snapshot["rho"], snapshot["ee"], snapshot["ei"], mesh)


def rel_delta(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), E_FLOOR)


def estimate_terminal_vacuum_boundary_energy(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    # Fallback only: no flux is exported. This records the radiation energy in
    # the last radial shell adjacent to the outer-r vacuum boundary.
    shell = np.asarray(rad_E, dtype=float)[-1:, :]
    vol = np.asarray(mesh.vol, dtype=float)[-1:, :]
    return float(np.sum(np.maximum(shell, 0.0) * vol))


def common_metrics(
    mode: str,
    nr: int,
    nz: int,
    groups: int,
    snapshots: list[dict[str, Any]],
    log_metrics: dict[str, Any],
    wall_time_s: float,
    bytes_out: int,
) -> dict[str, Any]:
    import numpy as np

    first = snapshots[0]
    final = snapshots[-1]
    first_mesh = mesh_from_snapshot(first, nr, nz)
    final_mesh = mesh_from_snapshot(final, nr, nz)
    rad0 = compute_E_rad_total(first["rad_E"], first_mesh)
    rad1 = compute_E_rad_total(final["rad_E"], final_mesh)
    u0 = matter_energy(first, first_mesh)
    u1 = matter_energy(final, final_mesh)
    k0 = compute_K_node_total(first)
    k1 = compute_K_node_total(final)
    escaped_series = read_history_scalar(Path(final["path"]).parent, Path(final["path"]).stem.rsplit("_", 1)[0], "energy/radiation_escaped")
    escaped_from_history = float(sum(max(x, 0.0) for x in escaped_series)) if escaped_series else None
    escaped_from_state = final.get("E_rad_escaped_state")
    escaped = escaped_from_state if escaped_from_state is not None else escaped_from_history
    escaped_source = "time_state/E_rad_escaped" if escaped_from_state is not None else "history/sum(energy/radiation_escaped)"
    if escaped is None:
        escaped = estimate_terminal_vacuum_boundary_energy(final["rad_E"], final_mesh)
        escaped_source = "terminal_outer_rad_E_shell"

    metrics: dict[str, Any] = {
        "a_eV_erg_cm3_eV4": A_EV,
        "groups": groups,
        "rho_min": float(np.nanmin([np.nanmin(s["rho"]) for s in snapshots])),
        "rho_max": float(np.nanmax([np.nanmax(s["rho"]) for s in snapshots])),
        "Te_min_eV": float(np.nanmin([np.nanmin(s["Te"]) for s in snapshots])),
        "Te_max_eV": float(np.nanmax([np.nanmax(s["Te"]) for s in snapshots])),
        "rad_E_min": float(np.nanmin([np.nanmin(s["rad_E"]) for s in snapshots])),
        "rad_E_max": float(np.nanmax([np.nanmax(s["rad_E"]) for s in snapshots])),
        "rad_E_nan_count": int(sum(np.count_nonzero(~np.isfinite(s["rad_E"])) for s in snapshots)),
        "rad_E_negative_count": int(sum(np.count_nonzero(np.asarray(s["rad_E"]) < 0.0) for s in snapshots)),
        "U_internal_initial": u0,
        "U_internal_final": u1,
        "dU_internal_rel": rel_delta(u0, u1),
        "K_node_initial": k0,
        "K_node_final": k1,
        "dK_node_rel": rel_delta(k0, k1),
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "dE_rad_rel": rel_delta(rad0, rad1),
        "E_rad_escaped": float(escaped),
        "E_rad_escaped_source": escaped_source,
        "E_total_initial_matter_rad": u0 + k0 + rad0,
        "E_total_final_matter_rad_escape": u1 + k1 + rad1 + float(escaped),
        "E_balance_rel": abs((u1 + k1 + rad1 + float(escaped)) - (u0 + k0 + rad0)) / max(abs(u0 + k0 + rad0), E_FLOOR),
        "E_terminal_outer_rad_shell": estimate_terminal_vacuum_boundary_energy(final["rad_E"], final_mesh),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(log_metrics)
    if mode == "planar_radiative_shock":
        metrics["shock_z_cm"] = estimate_shock_z(final, final_mesh)
        metrics["shock_z_gate_status"] = "pending_or_skipped"
    return metrics


def compute_multigroup_tmat_metrics(
    nr: int,
    nz: int,
    groups: int,
    snapshots: list[dict[str, Any]],
    common: dict[str, Any],
) -> dict[str, Any]:
    import numpy as np

    final = snapshots[-1]
    final_mesh = mesh_from_snapshot(final, nr, nz)
    rad_groups = np.asarray(final["rad_E_groups"], dtype=float)
    vol = np.asarray(final_mesh.vol, dtype=float)
    group_energy = np.asarray(
        [np.sum(rad_groups[:, :, g] * vol) for g in range(groups)],
        dtype=float,
    )
    total_rad = float(np.sum(group_energy))
    if total_rad > E_FLOOR:
        fractions = group_energy / total_rad
    else:
        fractions = np.zeros_like(group_energy)
    finite_check = all(
        bool(np.all(np.isfinite(np.asarray(s[key], dtype=float))))
        for s in snapshots
        for key in ("rho", "Te", "Ti", "ee", "ei", "rad_E_groups")
    )
    median_energy = float(np.median(group_energy)) if group_energy.size else 0.0
    max_imbalance = (
        float(np.max(group_energy) / median_energy)
        if median_energy > E_FLOOR and group_energy.size
        else math.inf
    )
    return {
        "mg_tmat_E_balance_rel": common.get("E_balance_rel", math.inf),
        "mg_tmat_group_energy_distribution": {
            str(g): float(fractions[g]) for g in range(groups)
        },
        "mg_tmat_group_energy_erg": {
            str(g): float(group_energy[g]) for g in range(groups)
        },
        "mg_tmat_group_occupancy": int(np.count_nonzero(fractions > 0.01)),
        "mg_tmat_max_group_imbalance": max_imbalance,
        "mg_tmat_T_max_eV": common.get("Te_max_eV"),
        "mg_tmat_T_min_eV": common.get("Te_min_eV"),
        "mg_tmat_no_nan": finite_check,
        "mg_tmat_finite_check": finite_check,
        "mg_tmat_opacity_clamp_count": common.get("n_clamp"),
    }


def print_multigroup_tmat_table(run_id: str, metrics: dict[str, Any]) -> None:
    dist = metrics.get("mg_tmat_group_energy_distribution")
    energies = metrics.get("mg_tmat_group_energy_erg")
    if not isinstance(dist, dict) or not isinstance(energies, dict):
        return
    print(f"RH1 multigroup_tmat group energy distribution: {run_id}")
    print("group energy_erg fraction")
    for key in sorted(dist, key=lambda value: int(value)):
        print(f"{key} {energies.get(key)} {dist.get(key)}")


def evaluate_iteration_gate(metrics: dict[str, Any], failed: list[str]) -> None:
    outer_max = metrics.get("fld_outer_iterations_max")
    threshold = ITERATION_CAP_FRACTION * MAX_OUTER_ITERATIONS
    metrics["fld_outer_iterations_gate_threshold"] = threshold
    if outer_max is None:
        failed.append("rh1_fld_iterations_unavailable")
    elif float(outer_max) >= threshold:
        failed.append("rh1_fld_iterations_cap_margin")


def evaluate_gates(mode: str, metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    evaluate_iteration_gate(metrics, failed)
    if metrics.get("rad_E_nan_count", 0) != 0:
        failed.append("rh1_rad_E_finite")
    if metrics.get("rho_min", 0.0) <= 0.0:
        failed.append("rh1_positive_density")
    if mode in ("shock_tube_grey", "cylindrical_blast_with_rad", "optically_thin_precursor"):
        if metrics.get("E_balance_rel", math.inf) > ENERGY_TOL:
            failed.append(f"rh1_{mode}_energy_balance")
    if mode == "optically_thin_precursor":
        if metrics.get("rad_E_negative_count", 0) != 0:
            failed.append("rh1_optically_thin_precursor_nonnegative")
    if mode == "planar_radiative_shock" and metrics.get("shock_z_cm") is None:
        failed.append("rh1_planar_shock_position_unavailable")
    if mode == "multigroup_tmat":
        if metrics.get("mg_tmat_E_balance_rel", math.inf) > MG_TMAT_E_BALANCE_TOL:
            failed.append("rh1_multigroup_tmat_energy_balance")
        if metrics.get("mg_tmat_group_occupancy", 0) < MG_TMAT_MIN_GROUP_OCCUPANCY:
            failed.append("rh1_multigroup_tmat_degenerate_spectrum")
        if not metrics.get("mg_tmat_no_nan", False):
            failed.append("rh1_multigroup_tmat_nan_or_infinite")
    if mode in ("lowrie_edwards_energy_proxy", "lowrie_edwards_table_init_shock_frame"):
        if metrics.get("le_proxy_reference_model_rh1") != RH1_SUBSHOCK_REFERENCE_MODEL:
            failed.append("rh1_le_subshock_reference_model")
    return failed


def active_env(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int, outdir: Path) -> dict[str, str]:
    env = {
        "TENRYU_RH1_MODE": mode,
        "TENRYU_RH1_NR": str(nr),
        "TENRYU_RH1_NZ": str(nz),
        "TENRYU_RH1_SEED": str(seed),
        "TENRYU_RH1_OUTDIR": str(outdir),
        "TENRYU_RH1_MAX_STEPS": str(args.max_steps),
        "TENRYU_RH1_VERBOSITY": args.verbosity,
    }
    if args.t_end_s is not None:
        env["TENRYU_RH1_T_END_S"] = str(args.t_end_s)
    if args.dt_initial_s is not None:
        env["TENRYU_RH1_DT_INITIAL_S"] = str(args.dt_initial_s)
    if args.dt_max_s is not None:
        env["TENRYU_RH1_DT_MAX_S"] = str(args.dt_max_s)
    if args.rho_gcc is not None:
        env["TENRYU_RH1_RHO_GCC"] = str(args.rho_gcc)
    if args.kappa_a_cm2_g is not None:
        env["TENRYU_RH1_KAPPA_A_CM2_G"] = str(args.kappa_a_cm2_g)
    env["TENRYU_RH1_ALE"] = "1" if args.ale else "0"
    return env


def effective_rho(args: argparse.Namespace, mode: str) -> float:
    return float(args.rho_gcc if args.rho_gcc is not None else mode_rho_default(mode))


def effective_kappa(args: argparse.Namespace, mode: str) -> float:
    return float(args.kappa_a_cm2_g if args.kappa_a_cm2_g is not None else mode_kappa_default(mode))


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int) -> dict[str, Any]:
    rho_gcc = effective_rho(args, mode)
    kappa_a = effective_kappa(args, mode)
    groups = mode_groups(mode)
    run_id = case_name(mode, nr, nz, rho_gcc, kappa_a, seed, args.ale)
    ladder_row = f"{mode}_nr{nr}_nz{nz}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(args, mode, nr, nz, seed, outdir)
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"

    start = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run([args.tenryu_bin, "run", args.deck], env=env, stdout=log, stderr=log)
    wall_time_s = time.monotonic() - start

    log_metrics = parse_log_metadata(log_path)
    metrics: dict[str, Any] = {**log_metrics, "wall_time_s": wall_time_s, "output_bytes": output_bytes(outdir)}
    result = "PASS"
    failed_gates: list[str] = []
    failure_class: str | None = None
    if proc.returncode != 0:
        result = "CRASHED"
        failed_gates = ["run_failure"]
        failure_class = "run_failure"
    else:
        try:
            results_dir = discover_results_dir(outdir)
            effective_outdir = results_dir.parent
            bytes_out = output_bytes(effective_outdir)
            if effective_outdir != outdir:
                bytes_out += output_bytes(outdir)
            snapshots = read_snapshots(results_dir, run_id, nr, nz, groups)
            metrics = common_metrics(mode, nr, nz, groups, snapshots, log_metrics, wall_time_s, bytes_out)
            metrics["nr"] = nr
            metrics["nz"] = nz
            ale_rezones = read_history_scalar(results_dir, run_id, "mesh/ale_rezone_invocations")
            if ale_rezones:
                metrics["ale_rezone_invocations"] = int(max(ale_rezones))
            else:
                metrics["ale_rezone_invocations"] = 0
            metrics["ale_enabled"] = bool(args.ale)
            if mode == "multigroup_tmat":
                metrics.update(compute_multigroup_tmat_metrics(nr, nz, groups, snapshots, metrics))
                print_multigroup_tmat_table(run_id, metrics)
            if mode in ("lowrie_edwards_energy_proxy", "lowrie_edwards_table_init_shock_frame"):
                metrics.update(
                    compute_le_proxy_metrics(
                        snapshots[-1],
                        mesh_from_snapshot(snapshots[-1], nr, nz),
                        rh1_subshock_reference_table_path(),
                        mode,
                    )
                )
            failed_gates = evaluate_gates(mode, metrics)
            result = "PASS" if not failed_gates else "FAIL"
            failure_class = None if not failed_gates else "gate_failure"
        except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
            result = "FAIL"
            failed_gates = ["analysis_failure"]
            failure_class = "analysis_failure"
            metrics["analysis_error"] = str(exc)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[run_r1] analysis failed: {exc}\n")

    metrics_ref = f"{args.root.rstrip('/')}/{STAGE_ID}/metrics/{run_id}.json"
    payload = {
        "run_id": run_id,
        "stage_id": STAGE_ID,
        "ladder_row": ladder_row,
        "seed": seed,
        "deck_path": args.deck,
        "mode": mode,
        "nr": nr,
        "nz": nz,
        "groups": groups,
        "rho_gcc": rho_gcc,
        "kappa_a_cm2_g": kappa_a,
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
    return {
        "run_id": payload["run_id"],
        "stage_id": payload["stage_id"],
        "ladder_row": payload["ladder_row"],
        "seed": payload["seed"],
        "deck_path": payload["deck_path"],
        "mode": payload["mode"],
        "nr": payload["nr"],
        "nz": payload["nz"],
        "groups": payload["groups"],
        "rho_gcc": payload["rho_gcc"],
        "kappa_a_cm2_g": payload["kappa_a_cm2_g"],
        "result": payload["result"],
        "failure_class": payload["failure_class"],
        "failed_gates": payload["failed_gates"],
        "exit_code": payload["exit_code"],
        "metrics_ref": metrics_ref,
        "wall_time_s": payload["wall_time_s"],
        "output_bytes": metrics.get("output_bytes"),
        "E_balance_rel": metrics.get("E_balance_rel"),
        "shock_z_cm": metrics.get("shock_z_cm"),
        "shock_z_ref_abs_err_cm": metrics.get("shock_z_ref_abs_err_cm"),
        "rad_E_min": metrics.get("rad_E_min"),
        "rad_E_max": metrics.get("rad_E_max"),
        "dU_internal_rel": metrics.get("dU_internal_rel"),
        "fld_outer_iterations_max": metrics.get("fld_outer_iterations_max"),
        "n_clamp": metrics.get("n_clamp"),
        "ale_enabled": metrics.get("ale_enabled"),
        "ale_rezone_invocations": metrics.get("ale_rezone_invocations"),
        "mg_tmat_E_balance_rel": metrics.get("mg_tmat_E_balance_rel"),
        "mg_tmat_group_occupancy": metrics.get("mg_tmat_group_occupancy"),
        "mg_tmat_max_group_imbalance": metrics.get("mg_tmat_max_group_imbalance"),
        "mg_tmat_T_min_eV": metrics.get("mg_tmat_T_min_eV"),
        "mg_tmat_T_max_eV": metrics.get("mg_tmat_T_max_eV"),
        "mg_tmat_no_nan": metrics.get("mg_tmat_no_nan"),
        "le_proxy_rho_l2": metrics.get("le_proxy_rho_l2"),
        "le_proxy_T_mat_l2": metrics.get("le_proxy_T_mat_l2"),
        "le_proxy_T_rad_l2": metrics.get("le_proxy_T_rad_l2"),
        "le_proxy_shock_pos_cm": metrics.get("le_proxy_shock_pos_cm"),
        "le_proxy_reference_model_rh1": metrics.get("le_proxy_reference_model_rh1"),
        "le_proxy_gate_status": metrics.get("le_proxy_gate_status"),
    }


def load_metrics_payload(metrics_ref: str) -> dict[str, Any]:
    return json.loads(Path(metrics_ref).read_text(encoding="utf-8"))


def apply_planar_self_reference_gate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    planar_rows = [row for row in rows if row["mode"] == "planar_radiative_shock" and row["nr"] == 64 and row["nz"] == 128]
    if not planar_rows:
        return
    if not args.include_planar_self_ref:
        for row in planar_rows:
            payload = load_metrics_payload(row["metrics_ref"])
            payload["metrics"]["shock_z_gate_status"] = "skipped_use_include_planar_self_ref"
            write_json(Path(row["metrics_ref"]), payload)
            updated = row_from_payload(payload, row["metrics_ref"])
            row.clear()
            row.update(updated)
        return

    ref_by_seed = {
        row["seed"]: row
        for row in rows
        if row["mode"] == "planar_radiative_shock" and row["nr"] == 256 and row["nz"] == 512
    }
    for row in planar_rows:
        seed = row["seed"]
        if seed not in ref_by_seed:
            ref_row = run_one(args, "planar_radiative_shock", 256, 512, seed)
            ref_row["ladder_row"] += "_self_ref"
            rows.append(ref_row)
            ref_by_seed[seed] = ref_row
        payload = load_metrics_payload(row["metrics_ref"])
        ref_payload = load_metrics_payload(ref_by_seed[seed]["metrics_ref"])
        shock_z = payload["metrics"].get("shock_z_cm")
        ref_z = ref_payload["metrics"].get("shock_z_cm")
        if shock_z is None or ref_z is None:
            payload["metrics"]["shock_z_gate_status"] = "fail_unavailable"
            if "rh1_planar_self_reference_unavailable" not in payload["failed_gates"]:
                payload["failed_gates"].append("rh1_planar_self_reference_unavailable")
        else:
            err = abs(float(shock_z) - float(ref_z))
            payload["metrics"]["shock_z_reference_cm"] = float(ref_z)
            payload["metrics"]["shock_z_ref_abs_err_cm"] = err
            payload["metrics"]["shock_z_ref_tol_cm"] = PLANAR_SELF_REF_TOL_CM
            payload["metrics"]["shock_z_gate_status"] = "pass" if err <= PLANAR_SELF_REF_TOL_CM else "fail"
            if err > PLANAR_SELF_REF_TOL_CM and "rh1_planar_self_reference_shock_z" not in payload["failed_gates"]:
                payload["failed_gates"].append("rh1_planar_self_reference_shock_z")
        if payload["failed_gates"]:
            payload["result"] = "FAIL"
            payload["failure_class"] = "gate_failure"
        write_json(Path(row["metrics_ref"]), payload)
        updated = row_from_payload(payload, row["metrics_ref"])
        row.clear()
        row.update(updated)


def matrix_rows(args: argparse.Namespace) -> list[tuple[str, int, int, int]]:
    rows = []
    for mode in args.mode_list:
        meshes = tuple(args.mesh_list) if args.mesh_list is not None else (
            SMOKE_MESHES_BY_MODE[mode] if args.smoke else FULL_MESHES_BY_MODE[mode]
        )
        for nr, nz in meshes:
            for seed in args.seed_list:
                rows.append((mode, nr, nz, seed))
    return rows


def finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            values.append(float(value))
    return values


def write_summary(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    aggregate = {
        "total_runs": len(rows),
        "passing_runs": sum(1 for row in rows if row["result"] == "PASS"),
        "crashed_runs": sum(1 for row in rows if row["result"] == "CRASHED"),
        "failed_runs": sum(1 for row in rows if row["result"] == "FAIL"),
        "a_eV_erg_cm3_eV4": A_EV,
    }
    for key in (
        "wall_time_s",
        "E_balance_rel",
        "mg_tmat_E_balance_rel",
        "mg_tmat_group_occupancy",
        "mg_tmat_max_group_imbalance",
        "shock_z_ref_abs_err_cm",
        "rad_E_max",
        "dU_internal_rel",
        "fld_outer_iterations_max",
        "n_clamp",
        "le_proxy_rho_l2",
        "le_proxy_T_mat_l2",
        "le_proxy_T_rad_l2",
    ):
        values = finite_values(rows, key)
        aggregate[f"{key}_max"] = max(values) if values else None
    payload = {
        "stage_id": STAGE_ID,
        "generated_iso": datetime.now(timezone.utc).isoformat(),
        "matrix": rows,
        "aggregate": aggregate,
    }
    write_json(Path(args.summary_json), payload)


def update_stage_json(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    stage_path = Path(args.root) / STAGE_ID / "stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else {"stage_id": STAGE_ID}
    stage["attempted_runs"] = int(stage.get("attempted_runs", 0)) + len(rows)
    stage["passing_runs"] = int(stage.get("passing_runs", 0)) + sum(1 for row in rows if row["result"] == "PASS")
    stage["last_attempt_status"] = "PASS" if rows and all(row["result"] == "PASS" for row in rows) else "FAIL"
    if rows:
        stage["status"] = "PASSED" if all(row["result"] == "PASS" for row in rows) else "IN_PROGRESS"
    write_json(stage_path, stage)


def print_summary(rows: list[dict[str, Any]]) -> None:
    print("Stage RH1 hydro+FLD sweep summary")
    if any(
        row["mode"] in ("lowrie_edwards_energy_proxy", "lowrie_edwards_table_init_shock_frame")
        for row in rows
    ):
        print(f"RH1 subshock reference: {RH1_SUBSHOCK_REFERENCE_MODEL}")
    print(
        "ladder_row seed result E_balance_rel mg_tmat_E_balance_rel "
        "mg_tmat_group_occupancy shock_z_cm shock_z_ref_abs_err_cm "
        "le_proxy_rho_l2 le_proxy_T_mat_l2 le_proxy_T_rad_l2 "
        "le_proxy_reference_model_rh1 le_proxy_gate_status fld_outer_iterations_max failed_gates"
    )
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {E_balance_rel} {mg_tmat_E_balance_rel} "
            "{mg_tmat_group_occupancy} {shock_z_cm} "
            "{shock_z_ref_abs_err_cm} {le_proxy_rho_l2} {le_proxy_T_mat_l2} "
            "{le_proxy_T_rad_l2} {le_proxy_reference_model_rh1} "
            "{le_proxy_gate_status} {fld_outer_iterations_max} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_rh1_hydro_fld.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/rh1_runs")
    parser.add_argument("--modes", "--mode", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(STANDARD_MODES))
    parser.add_argument("--mesh-list", type=parse_mesh_list, default=None)
    parser.add_argument("--seed-list", type=parse_seed_list, default=parse_seed_list("12345"))
    parser.add_argument("--smoke", action="store_true", help="use each mode's smallest grid only")
    parser.add_argument("--rho-gcc", type=float, default=None)
    parser.add_argument("--kappa-a-cm2-g", type=float, default=None)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="verbose")
    parser.add_argument("--include-planar-self-ref", action="store_true")
    parser.add_argument(
        "--ale",
        action="store_true",
        help="Enable RH1 ALE-on (TENRYU_RH1_ALE=1) - wires corner-J trigger + volume-rate CFL + active mesh repair primitives",
    )
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/RH1/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/RH1/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows_spec = matrix_rows(args)
    rows = [run_one(args, mode, nr, nz, seed) for mode, nr, nz, seed in rows_spec]
    apply_planar_self_reference_gate(args, rows)
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    update_stage_json(args, rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
