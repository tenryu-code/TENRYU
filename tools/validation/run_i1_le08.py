#!/usr/bin/env python3
"""Run and analyze the I1 LE08 nED grey radiative-shock verification gate."""

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
LADDER_ROW = "i1_le08_grey_ned_radshock"
A_EV = 1.3720e2
E_FLOOR = 1.0e-300
LOW_PRAD_GATE = 1.0e-3
LOW_FORCE_GATE = 1.0e-3
CONSERVATION_GATE = 1.0e-12
SHOCK_POSITION_RATE_GATE = 1.0
POSTSHOCK_TE_L2_GATE = 0.05
PRECURSOR_TRAD_L2_GATE = 0.10
T0_MACH_REL_GATE = 5.0e-3
T0_PRAD_RATIO_REL_GATE = 5.0e-2
T0_TRAD_TE_REL_GATE = 1.0e-3
INTERIOR_MASK_FRACTION = 0.10
REFERENCE_IDENTITY_MEDIAN_GATE = 1.0e-3
REFERENCE_IDENTITY_P95_GATE = 1.0e-2


@dataclass(frozen=True)
class Snapshot:
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
    r_nodes: Any
    v_r_nodes: Any
    z_nodes: Any | None
    dimension: str


def safe_float_token(value: float) -> str:
    return f"{value:.12g}".replace(".", "p").replace("-", "m")


def parse_grid_list(value: str) -> list[tuple[int, int]]:
    grids: list[tuple[int, int]] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        lowered = item.lower()
        if "x" in lowered or "," in lowered:
            sep = "x" if "x" in lowered else ","
            parts = [part.strip() for part in lowered.split(sep)]
            if len(parts) != 2:
                raise argparse.ArgumentTypeError(f"invalid grid entry {item!r}")
            nr = int(parts[0])
            nz = int(parts[1])
        else:
            nr = int(item)
            nz = 1
        if nr <= 0 or nz <= 0:
            raise argparse.ArgumentTypeError("grid dimensions must be positive")
        grids.append((nr, nz))
    if len(grids) < 3:
        raise argparse.ArgumentTypeError("I1 LE08 requires at least three grids")
    return grids


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def deck_dimension(path: Path) -> str:
    text = read_text(path)
    match = re.search(r"dimension\s*=\s*['\"](1D_SPH|2D_RZ)['\"]", text)
    if not match:
        raise RuntimeError(f"could not detect Main.dimension in {path}")
    return match.group(1)


def validate_grids_for_dimension(grids: list[tuple[int, int]], dimension: str) -> None:
    if dimension == "1D_SPH":
        bad = [f"{nr}x{nz}" for nr, nz in grids if nz != 1]
        if bad:
            raise argparse.ArgumentTypeError(
                "1D_SPH LE08 grids must be one-dimensional entries like '64;128;256', "
                f"got {', '.join(bad)}"
            )


def case_name(mach: float, nr: int, nz: int, seed: int, init_mode: str, dimension: str) -> str:
    if dimension == "1D_SPH":
        return (
            f"i1_le08_grey_ned_radshock_M{safe_float_token(mach)}"
            f"_nr{nr}_seed{seed}_{init_mode}"
        )
    return (
        f"i1_le08_grey_ned_radshock_M{safe_float_token(mach)}"
        f"_nr{nr}_nz{nz}_seed{seed}_{init_mode}"
    )


def discover_actual_outdir(outdir: Path) -> Path:
    if (outdir / "run_info.json").exists():
        return outdir
    candidates: list[Path] = []
    if outdir.parent.exists():
        for path in outdir.parent.glob(f"{outdir.name}_*"):
            if (path / "run_info.json").exists():
                candidates.append(path)
    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)
    return outdir


def discover_results_dir(outdir: Path) -> Path:
    actual = discover_actual_outdir(outdir)
    direct = actual / "results"
    if direct.exists() and any(p.suffix == ".h5" for p in direct.iterdir()):
        return direct
    raise FileNotFoundError(f"no HDF5 results under {actual}/results")


def plot_index(path: Path) -> int:
    suffix = path.stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def locate_plot_files(results_dir: Path, run_id: str) -> list[Path]:
    files = sorted(
        (p for p in results_dir.glob(f"{run_id}_*.h5") if plot_index(p) >= 0),
        key=plot_index,
    )
    if files:
        return files
    return sorted((p for p in results_dir.glob("*_*.h5") if plot_index(p) >= 0), key=plot_index)


def reshape_cells(values: Any, nr: int, nz: int, name: str, dimension: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if dimension == "1D_SPH":
        if arr.shape == (nr,):
            return arr
        if arr.size == nr:
            return arr.reshape((nr,))
        raise ValueError(f"{name} has shape {arr.shape}, expected {(nr,)}")
    if arr.shape == (nr, nz):
        return arr
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr, nz)}")


def reshape_nodes(values: Any, nr: int, nz: int, name: str, dimension: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if dimension == "1D_SPH":
        if arr.shape == (nr + 1,):
            return arr
        if arr.size == nr + 1:
            return arr.reshape((nr + 1,))
        raise ValueError(f"{name} has shape {arr.shape}, expected {(nr + 1,)}")
    if arr.shape == (nr + 1, nz + 1):
        return arr
    if arr.size == (nr + 1) * (nz + 1):
        return arr.reshape((nr + 1, nz + 1))
    raise ValueError(f"{name} has shape {arr.shape}, expected {(nr + 1, nz + 1)}")


def reshape_rad(values: Any, nr: int, nz: int, dimension: str) -> Any:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if dimension == "1D_SPH":
        if arr.shape == (nr,):
            return arr
        if arr.shape == (nr, 1):
            return arr.reshape((nr,))
        if arr.ndim == 2 and arr.shape[0] == nr:
            return np.sum(arr, axis=1)
        if arr.size == nr:
            return arr.reshape((nr,))
        raise ValueError(f"radiation/energy_density has shape {arr.shape}, expected one-group 1D cells")
    if arr.shape == (nr, nz):
        return arr
    if arr.shape == (nr * nz, 1):
        return arr.reshape((nr, nz))
    if arr.ndim == 3 and arr.shape[:2] == (nr, nz):
        return np.sum(arr, axis=2)
    if arr.ndim == 2 and arr.shape[0] == nr * nz:
        return np.sum(arr, axis=1).reshape((nr, nz))
    if arr.size == nr * nz:
        return arr.reshape((nr, nz))
    raise ValueError(f"radiation/energy_density has shape {arr.shape}, expected one-group cells")


def scalar_dataset(handle: Any, path: str, default: float = 0.0) -> float:
    import numpy as np

    if path not in handle:
        return default
    arr = np.asarray(handle[path][()], dtype=float).reshape(-1)
    return float(arr[-1]) if arr.size else default


def read_snapshot(path: Path, nr: int, nz: int, dimension: str) -> Snapshot:
    import h5py

    with h5py.File(path, "r") as handle:
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = scalar_dataset(handle, "time_state/t", t_value)
        r_node_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_node_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        z_nodes = (
            reshape_nodes(handle[z_node_path][()], nr, nz, z_node_path, dimension)
            if dimension == "2D_RZ"
            else None
        )
        return Snapshot(
            path=path,
            t=t_value,
            step=int(round(scalar_dataset(handle, "time_state/step", 0.0))),
            rho=reshape_cells(handle["hydro/rho"][()], nr, nz, "hydro/rho", dimension),
            Te=reshape_cells(handle["hydro/Te"][()], nr, nz, "hydro/Te", dimension),
            Ti=reshape_cells(handle["hydro/Ti"][()], nr, nz, "hydro/Ti", dimension),
            ee=reshape_cells(handle["hydro/ee"][()], nr, nz, "hydro/ee", dimension),
            ei=reshape_cells(handle["hydro/ei"][()], nr, nz, "hydro/ei", dimension),
            Pe=reshape_cells(handle["hydro/Pe"][()], nr, nz, "hydro/Pe", dimension),
            Pi=reshape_cells(handle["hydro/Pi"][()], nr, nz, "hydro/Pi", dimension),
            vol=reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol", dimension),
            rad_E=reshape_rad(handle["radiation/energy_density"][()], nr, nz, dimension),
            rad_F=(
                reshape_rad(handle["radiation/diag_F_first_moment"][()], nr, nz, dimension)
                if "radiation/diag_F_first_moment" in handle
                else None
            ),
            r_nodes=reshape_nodes(handle[r_node_path][()], nr, nz, r_node_path, dimension),
            v_r_nodes=reshape_nodes(handle["mesh/v_r"][()], nr, nz, "mesh/v_r", dimension),
            z_nodes=z_nodes,
            dimension=dimension,
        )


def read_snapshots(
    results_dir: Path,
    run_id: str,
    nr: int,
    nz: int,
    dimension: str,
    min_count: int = 2,
) -> list[Snapshot]:
    files = locate_plot_files(results_dir, run_id)
    if len(files) < min_count:
        raise RuntimeError(f"expected at least {min_count} plot snapshot(s)")
    return [read_snapshot(path, nr, nz, dimension) for path in files]


def profile_centers(snapshot: Snapshot) -> Any:
    import numpy as np

    if snapshot.dimension == "1D_SPH":
        r = np.asarray(snapshot.r_nodes, dtype=float)
        return 0.5 * (r[:-1] + r[1:])
    z = np.asarray(snapshot.z_nodes, dtype=float)
    return np.mean(0.25 * (z[:-1, :-1] + z[1:, :-1] + z[:-1, 1:] + z[1:, 1:]), axis=0)


def profile_average(snapshot: Snapshot, field: Any) -> Any:
    import numpy as np

    if snapshot.dimension == "1D_SPH":
        return np.asarray(field, dtype=float)
    arr = np.asarray(field, dtype=float)
    weights = np.asarray(snapshot.vol, dtype=float)
    denom = np.maximum(np.sum(weights, axis=0), E_FLOOR)
    return np.sum(arr * weights, axis=0) / denom


def profile_velocity(snapshot: Snapshot) -> Any:
    import numpy as np

    vr = np.asarray(snapshot.v_r_nodes, dtype=float)
    if snapshot.dimension == "1D_SPH":
        return 0.5 * (vr[:-1] + vr[1:])
    vr_cell = 0.25 * (vr[:-1, :-1] + vr[1:, :-1] + vr[:-1, 1:] + vr[1:, 1:])
    return profile_average(snapshot, vr_cell)


def profile_cell_widths(snapshot: Snapshot) -> Any:
    import numpy as np

    coord = np.asarray(profile_centers(snapshot), dtype=float)
    if snapshot.dimension == "1D_SPH":
        nodes = np.asarray(snapshot.r_nodes, dtype=float)
        if nodes.size == coord.size + 1:
            return np.diff(nodes)
    if coord.size < 2:
        return np.ones_like(coord)
    edges = np.empty(coord.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (coord[:-1] + coord[1:])
    edges[0] = coord[0] - 0.5 * (coord[1] - coord[0])
    edges[-1] = coord[-1] + 0.5 * (coord[-1] - coord[-2])
    return np.diff(edges)


def midpoint_shock_position(coord: Any, field: Any) -> float | None:
    import numpy as np

    x = np.asarray(coord, dtype=float)
    y = np.asarray(field, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 3:
        return None
    x = x[mask]
    y = y[mask]
    y_min = float(np.nanmin(y))
    y_max = float(np.nanmax(y))
    if not y_max > y_min:
        return None
    midpoint = 0.5 * (y_min + y_max)
    return float(x[int(np.nanargmin(np.abs(y - midpoint)))])


def estimate_shock_position_from_profile(coord: Any, primary: Any, fallback: Any | None = None) -> float | None:
    import numpy as np

    x = np.asarray(coord, dtype=float)
    y = np.asarray(primary, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) >= 3:
        x_valid = x[mask]
        y_valid = y[mask]
        grad = np.gradient(y_valid, x_valid)
        if grad.size and float(np.nanmax(np.abs(grad))) > 0.0:
            return float(x_valid[int(np.nanargmax(np.abs(grad)))])
    if fallback is not None:
        return midpoint_shock_position(coord, fallback)
    return midpoint_shock_position(coord, primary)


def estimate_shock_position(snapshot: Snapshot) -> float | None:
    coord = profile_centers(snapshot)
    Te = profile_average(snapshot, snapshot.Te)
    rho = profile_average(snapshot, snapshot.rho)
    return estimate_shock_position_from_profile(coord, Te, rho)


def fit_shock_velocity(snapshots: list[Snapshot]) -> tuple[float | None, list[float]]:
    import numpy as np

    pairs = [(s.t, estimate_shock_position(s)) for s in snapshots]
    valid = [(t, z) for t, z in pairs if z is not None and math.isfinite(z)]
    if len(valid) < 2:
        return None, []
    t = np.asarray([x[0] for x in valid], dtype=float)
    z = np.asarray([x[1] for x in valid], dtype=float)
    if float(np.max(t) - np.min(t)) <= 0.0:
        return None, [float(v) for v in z]
    coeff = np.polyfit(t, z, deg=1)
    return float(coeff[0]), [float(v) for v in z]


def interp_reference(ref: dict[str, Any], key: str, x: Any) -> Any:
    import numpy as np

    xs = np.asarray(ref["table"]["x_cm"], dtype=float)
    vals = np.asarray(ref["table"][key], dtype=float)
    return np.interp(np.asarray(x, dtype=float), xs, vals)


def le08_schema_is_v3(ref: dict[str, Any]) -> bool:
    version = ref.get("schema_version")
    return version == 3 or str(version).strip() == "3" or str(version).strip().endswith(".v3")


def le08_table_field(table: dict[str, Any], *names: str) -> Any:
    import numpy as np

    for name in names:
        if name in table:
            return np.asarray(table[name], dtype=float)
    raise KeyError(f"missing any reference table field from {names}")


def le08_branch_labels(table: dict[str, Any], x: Any) -> Any:
    import numpy as np

    if "branch" in table:
        return np.asarray(table["branch"], dtype=str)
    return np.where(np.asarray(x, dtype=float) <= 0.0, "upstream", "downstream")


def le08_identity_rel_by_branch(
    ref: dict[str, Any],
    flux_field: str,
    branch: str,
    *,
    branch_oriented_flux: bool,
) -> Any:
    import numpy as np

    table = ref["table"]
    params = ref["parameters"]
    x = le08_table_field(table, "x_cm")
    rho = le08_table_field(table, "rho_g_per_cc")
    E_rad = le08_table_field(table, "E_rad_erg_per_cm3")
    F_ref = le08_table_field(table, flux_field, "F_rad_erg_cm2_s")
    branches = le08_branch_labels(table, x)
    mask = branches == branch
    if np.count_nonzero(mask) < 3:
        return np.asarray([math.inf], dtype=float)
    order = np.argsort(x[mask])
    x_b = x[mask][order]
    rho_b = rho[mask][order]
    E_b = E_rad[mask][order]
    F_b = F_ref[mask][order]
    c_light = float(params["c_cm_per_s"])
    kappa_R = float(params["kappa_R_cm2_per_g"])
    D = c_light / (3.0 * kappa_R * np.maximum(rho_b, E_FLOOR))
    F_from_grad = -D * np.gradient(E_b, x_b)
    if branch_oriented_flux and branch == "downstream":
        F_from_grad = -F_from_grad
    rel = (F_b - F_from_grad) / np.maximum(np.abs(F_b) + np.abs(F_from_grad), E_FLOOR)
    return np.abs(rel[10:-10]) if rel.size > 20 else np.abs(rel)


def le08_identity_stats(
    ref: dict[str, Any],
    flux_field: str,
    *,
    branch_oriented_flux: bool,
) -> dict[str, dict[str, float]]:
    import numpy as np

    rels = {
        branch: le08_identity_rel_by_branch(
            ref,
            flux_field,
            branch,
            branch_oriented_flux=branch_oriented_flux,
        )
        for branch in ("upstream", "downstream")
    }
    rels["total"] = np.concatenate([rels["upstream"], rels["downstream"]])
    return {
        branch: {
            "median_rel": float(np.median(values)),
            "p95_rel": float(np.percentile(values, 95.0)),
        }
        for branch, values in rels.items()
    }


def reference_identity_branch_wise(ref: dict[str, Any]) -> dict[str, Any]:
    primary = le08_identity_stats(
        ref,
        "F_rad_erg_per_cm2_s",
        branch_oriented_flux=False,
    )
    energy_field = "F_rad_energy_invariant_erg_per_cm2_s"
    energy = le08_identity_stats(
        ref,
        energy_field,
        branch_oriented_flux=True,
    )
    passed = all(
        primary[branch]["median_rel"] < REFERENCE_IDENTITY_MEDIAN_GATE
        and primary[branch]["p95_rel"] < REFERENCE_IDENTITY_P95_GATE
        for branch in ("upstream", "downstream")
    )
    return {
        "schema_version": ref.get("schema_version"),
        "schema_v3": le08_schema_is_v3(ref),
        "problem": ref.get("problem"),
        "reference_model": ref.get("reference_model"),
        "thresholds": {
            "median_rel_lt": REFERENCE_IDENTITY_MEDIAN_GATE,
            "p95_rel_lt": REFERENCE_IDENTITY_P95_GATE,
        },
        "passed": passed,
        "upstream": primary["upstream"],
        "downstream": primary["downstream"],
        "total": primary["total"],
        "upstream_median_rel": primary["upstream"]["median_rel"],
        "upstream_p95_rel": primary["upstream"]["p95_rel"],
        "downstream_median_rel": primary["downstream"]["median_rel"],
        "downstream_p95_rel": primary["downstream"]["p95_rel"],
        "total_median_rel": primary["total"]["median_rel"],
        "total_p95_rel": primary["total"]["p95_rel"],
        "energy_invariant_check": energy,
    }


def reference_shock_position(ref: dict[str, Any]) -> float:
    import numpy as np

    table = ref["table"]
    x = np.asarray(table["x_cm"], dtype=float)
    Te = np.asarray(table["T_eV"], dtype=float)
    rho = np.asarray(table["rho_g_per_cc"], dtype=float)
    shock = estimate_shock_position_from_profile(x, Te, rho)
    return 0.0 if shock is None else float(shock)


def reference_frame_coord(snapshot: Snapshot, ref: dict[str, Any]) -> Any:
    import numpy as np

    coord = profile_centers(snapshot)
    ref_x0 = float(ref["table"]["x_cm"][0])
    if snapshot.dimension == "1D_SPH":
        nodes = np.asarray(snapshot.r_nodes, dtype=float)
        origin = float(nodes[0]) - ref_x0 if nodes.size else float(coord[0]) - ref_x0
    else:
        origin = float(np.nanmin(coord)) - ref_x0
    return np.asarray(coord, dtype=float) - origin


def interpolate_le08_table_to_cells(table: dict[str, Any], x_cells: Any) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(x_cells, dtype=float)
    x_ref = np.asarray(table["x_cm"], dtype=float)
    return {
        "rho": np.interp(x, x_ref, np.asarray(table["rho_g_per_cc"], dtype=float)),
        "u": np.interp(x, x_ref, np.asarray(table["u_cm_per_s"], dtype=float)),
        "T": np.interp(x, x_ref, np.asarray(table["T_eV"], dtype=float)),
        "T_rad": np.interp(x, x_ref, np.asarray(table["T_rad_eV"], dtype=float)),
        "P_mat": np.interp(x, x_ref, np.asarray(table["P_mat_dyne_per_cm2"], dtype=float)),
        "P_rad": np.interp(x, x_ref, np.asarray(table["P_rad_dyne_per_cm2"], dtype=float)),
    }


def two_state_shock_position_ref_frame(ref: dict[str, Any]) -> float:
    x = ref["table"]["x_cm"]
    return 0.5 * (float(x[0]) + float(x[-1]))


def build_t0_admissibility_reference(
    ref_json: dict[str, Any],
    x_cells: Any,
    init_mode: str,
    shock_x0: float,
) -> dict[str, Any]:
    import numpy as np

    if init_mode == "reference_table":
        return interpolate_le08_table_to_cells(ref_json["table"], x_cells)
    if init_mode == "two_state":
        up = ref_json["states"]["upstream"]
        dn = ref_json["states"]["downstream"]
        left = np.asarray(x_cells, dtype=float) < shock_x0
        return {
            "rho": np.where(left, float(up["rho"]), float(dn["rho"])),
            "u": np.where(left, float(up["u"]), float(dn["u"])),
            "T": np.where(left, float(up["T"]), float(dn["T"])),
            "T_rad": np.where(left, float(up.get("T_rad", up["T"])), float(dn.get("T_rad", dn["T"]))),
            "P_mat": np.where(left, float(up["P_mat"]), float(dn["P_mat"])),
            "P_rad": np.where(left, float(up["P_rad"]), float(dn["P_rad"])),
        }
    raise ValueError(f"unknown init_mode={init_mode}")


def plateau_mask(
    x: Any,
    shock_x0: float,
    dx: float,
    boundary_frac: float = INTERIOR_MASK_FRACTION,
    shock_exclude_cells: int = 6,
) -> Any:
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.zeros_like(arr, dtype=bool)
    rmin, rmax = float(np.nanmin(arr)), float(np.nanmax(arr))
    interior = (arr > rmin + boundary_frac * (rmax - rmin)) & (
        arr < rmax - boundary_frac * (rmax - rmin)
    )
    away_from_shock = np.abs(arr - shock_x0) > shock_exclude_cells * max(float(dx), E_FLOOR)
    return interior & away_from_shock


def interior_mask(coord: Any, fraction: float = INTERIOR_MASK_FRACTION) -> Any:
    import numpy as np

    x = np.asarray(coord, dtype=float)
    if x.size == 0:
        return np.zeros_like(x, dtype=bool)
    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))
    span = hi - lo
    if not span > 0.0:
        return np.zeros_like(x, dtype=bool)
    margin = fraction * span
    return (x >= lo + margin) & (x <= hi - margin)


def rel_l2(got: Any, ref: Any) -> float:
    import numpy as np

    got_arr = np.asarray(got, dtype=float)
    ref_arr = np.asarray(ref, dtype=float)
    if got_arr.size == 0 or ref_arr.size == 0:
        return math.inf
    denom = float(np.sqrt(np.sum(ref_arr * ref_arr)))
    return float(np.sqrt(np.sum((got_arr - ref_arr) ** 2)) / max(denom, E_FLOOR))


def max_gradient_ratio(z: Any, numerator: Any, denominator: Any) -> float:
    import numpy as np

    z_arr = np.asarray(z, dtype=float)
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    if z_arr.size < 3:
        return math.inf
    grad_num = np.gradient(num, z_arr)
    grad_den = np.gradient(den, z_arr)
    scale = float(np.nanmax(np.abs(grad_den)))
    threshold = max(1.0e-12 * scale, E_FLOOR)
    mask = np.isfinite(grad_num) & np.isfinite(grad_den) & (np.abs(grad_den) >= threshold)
    if not np.any(mask):
        return math.inf
    return float(np.nanmax(np.abs(grad_num[mask]) / np.maximum(np.abs(grad_den[mask]), E_FLOOR)))


def finite_average(values: Any, mask: Any | None = None, default: float = math.nan) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.mean(arr))


def finite_max_abs(values: Any, mask: Any | None = None, default: float = math.inf) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return default
    return float(np.max(arr))


def finite_p95_abs(values: Any, mask: Any | None = None, default: float = math.inf) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = np.abs(arr[np.isfinite(arr)])
    if arr.size == 0:
        return default
    return float(np.percentile(arr, 95.0))


def finite_rms(values: Any, mask: Any | None = None, default: float = math.inf) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.sqrt(np.mean(arr * arr)))


def rel_linf(values: Any, reference: Any, mask: Any | None = None, default: float = math.inf) -> float:
    import numpy as np

    got = np.asarray(values, dtype=float)
    ref = np.asarray(reference, dtype=float)
    local_mask = np.isfinite(got) & np.isfinite(ref)
    if mask is not None:
        local_mask &= np.asarray(mask, dtype=bool)
    if np.count_nonzero(local_mask) == 0:
        return default
    denom = np.maximum(np.abs(ref[local_mask]), E_FLOOR)
    return float(np.max(np.abs(got[local_mask] - ref[local_mask]) / denom))


def median_cell_spacing(x: Any) -> float:
    import numpy as np

    arr = np.asarray(x, dtype=float)
    if arr.size < 2:
        return math.inf
    dx = np.diff(np.sort(arr))
    dx = dx[np.isfinite(dx) & (dx > 0.0)]
    if dx.size == 0:
        return math.inf
    return float(np.median(dx))


def signed_cumulative_integral(coord: Any, integrand: Any, shock_coord: float) -> Any:
    import numpy as np

    x = np.asarray(coord, dtype=float)
    y = np.asarray(integrand, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2 or not math.isfinite(float(shock_coord)):
        return out
    valid_index = np.nonzero(mask)[0]
    order_local = np.argsort(x[mask])
    order = valid_index[order_local]
    xs = x[order]
    ys = y[order]
    cum = np.zeros_like(xs)
    cum[1:] = np.cumsum(0.5 * (ys[:-1] + ys[1:]) * np.diff(xs))
    shock_cum = float(np.interp(shock_coord, xs, cum, left=cum[0], right=cum[-1]))
    out[order] = cum - shock_cum
    return out


def interp_sorted(coord: Any, values: Any, query: Any) -> Any:
    import numpy as np

    x = np.asarray(coord, dtype=float)
    y = np.asarray(values, dtype=float)
    q = np.asarray(query, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        return np.full_like(q, np.nan, dtype=float)
    order = np.argsort(x[mask])
    xs = x[mask][order]
    ys = y[mask][order]
    unique = np.concatenate(([True], np.diff(xs) > 0.0))
    xs = xs[unique]
    ys = ys[unique]
    if xs.size < 2:
        return np.full_like(q, np.nan, dtype=float)
    return np.interp(q, xs, ys, left=np.nan, right=np.nan)


def rel_l2_on_coord(
    got_coord: Any,
    got_values: Any,
    ref_coord: Any,
    ref_values: Any,
    mask: Any,
) -> float:
    import numpy as np

    query = np.asarray(got_coord, dtype=float)
    got = np.asarray(got_values, dtype=float)
    local_mask = np.asarray(mask, dtype=bool) & np.isfinite(query) & np.isfinite(got)
    if np.count_nonzero(local_mask) < 3:
        return math.inf
    ref_interp = interp_sorted(ref_coord, ref_values, query[local_mask])
    finite = np.isfinite(ref_interp)
    if np.count_nonzero(finite) < 3:
        return math.inf
    return rel_l2(got[local_mask][finite], ref_interp[finite])


def reference_tau_m_coordinates(ref: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    table = ref["table"]
    x = np.asarray(table["x_cm"], dtype=float)
    rho = np.asarray(table["rho_g_per_cc"], dtype=float)
    shock = reference_shock_position(ref)
    kappa_R = float(ref["parameters"]["kappa_R_cm2_per_g"])
    return {
        "x_shifted": x - shock,
        "tau_from_shock": signed_cumulative_integral(x, kappa_R * rho, shock),
        "m_from_shock": signed_cumulative_integral(x, rho, shock),
    }


def profile_l2_three_coords(snapshot: Snapshot, ref: dict[str, Any]) -> dict[str, float]:
    import numpy as np

    coord = profile_centers(snapshot)
    raw_x = reference_frame_coord(snapshot, ref)
    Te = profile_average(snapshot, snapshot.Te)
    rho = profile_average(snapshot, snapshot.rho)
    Trad = np.power(np.maximum(profile_average(snapshot, snapshot.rad_E) / A_EV, 0.0), 0.25)
    shock_coord = estimate_shock_position_from_profile(coord, Te, rho)
    ref_shock_coord = reference_shock_position(ref)
    keys = [
        f"{region}_{field}_l2_{axis}"
        for region in ("postshock", "precursor")
        for field in ("Te", "Trad")
        for axis in ("x", "tau", "m")
    ]
    if shock_coord is None:
        return {key: math.inf for key in keys}

    shock_coord_ref_frame = float(np.interp(shock_coord, coord, raw_x))
    x_shifted = np.asarray(raw_x, dtype=float) - shock_coord_ref_frame
    kappa_R = float(ref["parameters"]["kappa_R_cm2_per_g"])
    tau = signed_cumulative_integral(coord, kappa_R * np.asarray(rho, dtype=float), shock_coord)
    mass = signed_cumulative_integral(coord, rho, shock_coord)
    ref_coord = reference_tau_m_coordinates(ref)
    ref_fields = {
        "Te": np.asarray(ref["table"]["T_eV"], dtype=float),
        "Trad": np.asarray(ref["table"]["T_rad_eV"], dtype=float),
    }
    interior = interior_mask(coord)
    post = interior & (x_shifted >= 0.0)
    pre = interior & (x_shifted < 0.0)

    axes = {
        "x": (x_shifted, ref_coord["x_shifted"]),
        "tau": (tau, ref_coord["tau_from_shock"]),
        "m": (mass, ref_coord["m_from_shock"]),
    }
    fields = {"Te": Te, "Trad": Trad}
    out: dict[str, float] = {}
    for region, region_mask in (("postshock", post), ("precursor", pre)):
        for field_name, values in fields.items():
            for axis_name, (got_axis, ref_axis) in axes.items():
                out[f"{region}_{field_name}_l2_{axis_name}"] = rel_l2_on_coord(
                    got_axis,
                    values,
                    ref_axis,
                    ref_fields[field_name],
                    region_mask,
                )
    return out


def profile_metrics(snapshot: Snapshot, ref: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    coord = profile_centers(snapshot)
    Te = profile_average(snapshot, snapshot.Te)
    rho = profile_average(snapshot, snapshot.rho)
    shock_coord = estimate_shock_position_from_profile(coord, Te, rho)
    ref_shock_coord = reference_shock_position(ref)
    if shock_coord is None:
        return {
            "shock_position_cm": None,
            "reference_shock_position_cm": ref_shock_coord,
            "shock_displacement_cm": None,
            "postshock_Te_l2": math.inf,
            "precursor_Trad_l2": math.inf,
            "max_P_rad_over_P_mat": math.inf,
            "max_abs_grad_P_rad_over_abs_grad_P_mat": math.inf,
            "nED_temperature_separation_l_inf": math.inf,
            "reference_nED_temperature_separation_l_inf": math.inf,
            "profile_l2_three_coords": profile_l2_three_coords(snapshot, ref),
        }
    ref_x = np.asarray(ref["table"]["x_cm"], dtype=float)
    raw_x = reference_frame_coord(snapshot, ref)
    shock_coord_ref_frame = float(np.interp(shock_coord, coord, raw_x))
    dx_shock = shock_coord_ref_frame - ref_shock_coord
    aligned_x = raw_x - dx_shock
    overlap = (aligned_x >= ref_x[0]) & (aligned_x <= ref_x[-1])
    interior = interior_mask(coord)
    post = overlap & interior & (aligned_x >= ref_shock_coord)
    pre = overlap & interior & (aligned_x < ref_shock_coord)

    p_mat = profile_average(snapshot, np.asarray(snapshot.Pe) + np.asarray(snapshot.Pi))
    p_rad = profile_average(snapshot, np.asarray(snapshot.rad_E) / 3.0)
    Trad = np.power(np.maximum(profile_average(snapshot, snapshot.rad_E) / A_EV, 0.0), 0.25)
    sep_code = Trad / np.maximum(np.asarray(Te, dtype=float), E_FLOOR) - 1.0
    ref_sep_table = (
        np.asarray(ref["table"]["T_rad_eV"], dtype=float)
        / np.maximum(np.asarray(ref["table"]["T_eV"], dtype=float), E_FLOOR)
        - 1.0
    )
    ref_sep = np.interp(aligned_x, ref_x, ref_sep_table)
    prad_ratio = float(np.nanmax(np.asarray(p_rad) / np.maximum(np.asarray(p_mat), E_FLOOR)))
    grad_ratio = max_gradient_ratio(coord, p_rad, p_mat)

    return {
        "shock_position_cm": shock_coord,
        "reference_shock_position_cm": ref_shock_coord,
        "shock_displacement_cm": dx_shock,
        "postshock_Te_l2": rel_l2(Te[post], interp_reference(ref, "T_eV", aligned_x[post])) if np.count_nonzero(post) >= 3 else math.inf,
        "precursor_Trad_l2": rel_l2(Trad[pre], interp_reference(ref, "T_rad_eV", aligned_x[pre])) if np.count_nonzero(pre) >= 3 else math.inf,
        "profile_l2_three_coords": profile_l2_three_coords(snapshot, ref),
        "max_P_rad_over_P_mat": prad_ratio,
        "max_abs_grad_P_rad_over_abs_grad_P_mat": grad_ratio,
        "nED_temperature_separation_l_inf": rel_linf(sep_code, ref_sep, overlap & interior),
        "reference_nED_temperature_separation_l_inf": finite_max_abs(ref_sep, overlap & interior),
        "overlap_cell_count": int(np.count_nonzero(overlap)),
        "interior_cell_count": int(np.count_nonzero(interior)),
        "masked_overlap_cell_count": int(np.count_nonzero(overlap & interior)),
        "postshock_cell_count": int(np.count_nonzero(post)),
        "precursor_cell_count": int(np.count_nonzero(pre)),
    }


def mean_all_finite_or_inf(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        return math.inf
    return float(np.mean(arr))


def nonfinite_count(values: Any) -> int:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return int(np.count_nonzero(~np.isfinite(arr)))


def aggregate_profile_l2_three_coords(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = [
        f"{region}_{field}_l2_{axis}"
        for region in ("postshock", "precursor")
        for field in ("Te", "Trad")
        for axis in ("x", "tau", "m")
    ]
    return {
        key: mean_all_finite_or_inf([row.get("profile_l2_three_coords", {}).get(key, math.inf) for row in rows])
        for key in keys
    }


def shock_speed_trace(snapshots: list[Snapshot]) -> tuple[list[tuple[int, float, float]], list[dict[str, float | int]]]:
    valid_positions: list[tuple[int, float, float]] = []
    for index, snapshot in enumerate(snapshots):
        position = estimate_shock_position(snapshot)
        if position is None:
            continue
        if math.isfinite(float(snapshot.t)) and math.isfinite(float(position)):
            valid_positions.append((index, float(snapshot.t), float(position)))

    speeds: list[dict[str, float | int]] = []
    for prev, curr in zip(valid_positions, valid_positions[1:]):
        prev_index, prev_t, prev_position = prev
        curr_index, curr_t, curr_position = curr
        dt = curr_t - prev_t
        if not dt > 0.0:
            continue
        speeds.append(
            {
                "start_index": prev_index,
                "end_index": curr_index,
                "start_time_s": prev_t,
                "end_time_s": curr_t,
                "start_position_cm": prev_position,
                "end_position_cm": curr_position,
                "speed_cm_per_s": (curr_position - prev_position) / dt,
            }
        )
    return valid_positions, speeds


def locate_steady_start_index(snapshots: list[Snapshot]) -> dict[str, Any]:
    if not snapshots:
        return {
            "steady_start_index": 0,
            "steady_start_time_s": None,
            "steady_start_method": "no_snapshots",
            "steady_speed_avg_cm_per_s": math.nan,
            "steady_speed_tolerance": 0.01,
            "shock_speed_trace": [],
        }

    valid_positions, speeds = shock_speed_trace(snapshots)
    if not speeds:
        fallback = len(snapshots) - 1
        return {
            "steady_start_index": fallback,
            "steady_start_time_s": float(snapshots[fallback].t),
            "steady_start_method": "fallback_final_snapshot_no_speed_trace",
            "steady_speed_avg_cm_per_s": math.nan,
            "steady_speed_tolerance": 0.01,
            "shock_speed_trace": speeds,
        }

    speed_abs = [abs(float(row["speed_cm_per_s"])) for row in speeds]
    speed_avg = sum(speed_abs) / max(len(speed_abs), 1)
    if speed_avg <= E_FLOOR:
        start = valid_positions[0][0] if valid_positions else 0
        return {
            "steady_start_index": start,
            "steady_start_time_s": float(snapshots[start].t),
            "steady_start_method": "stationary_shock_speed",
            "steady_speed_avg_cm_per_s": speed_avg,
            "steady_speed_tolerance": 0.01,
            "shock_speed_trace": speeds,
        }

    for left, right in zip(speeds, speeds[1:]):
        rel_change = abs(float(left["speed_cm_per_s"]) - float(right["speed_cm_per_s"])) / speed_avg
        if rel_change <= 0.01:
            start = int(left["start_index"])
            return {
                "steady_start_index": start,
                "steady_start_time_s": float(snapshots[start].t),
                "steady_start_method": "adjacent_shock_speed_change_below_1pct",
                "steady_speed_avg_cm_per_s": speed_avg,
                "steady_speed_relative_change": rel_change,
                "steady_speed_tolerance": 0.01,
                "shock_speed_trace": speeds,
            }

    fallback = int(speeds[-1]["end_index"])
    return {
        "steady_start_index": fallback,
        "steady_start_time_s": float(snapshots[fallback].t),
        "steady_start_method": "fallback_final_snapshot_no_steady_pair",
        "steady_speed_avg_cm_per_s": speed_avg,
        "steady_speed_tolerance": 0.01,
        "shock_speed_trace": speeds,
    }


def profile_window_metrics(snapshots: list[Snapshot], ref: dict[str, Any]) -> dict[str, Any]:
    per_snapshot = [profile_metrics(snapshot, ref) for snapshot in snapshots]
    steady = locate_steady_start_index(snapshots)
    start_index = int(steady["steady_start_index"])
    start_index = max(0, min(start_index, len(per_snapshot) - 1)) if per_snapshot else 0
    steady_metrics = per_snapshot[start_index:] if per_snapshot else []

    full_post = [row.get("postshock_Te_l2", math.inf) for row in per_snapshot]
    full_pre = [row.get("precursor_Trad_l2", math.inf) for row in per_snapshot]
    steady_post = [row.get("postshock_Te_l2", math.inf) for row in steady_metrics]
    steady_pre = [row.get("precursor_Trad_l2", math.inf) for row in steady_metrics]

    out = {
        "l2_time_average_source": "phase_aligned_interior_masked_snapshot_mean",
        "full_traversal_snapshot_count": len(per_snapshot),
        "full_traversal_postshock_Te_l2": mean_all_finite_or_inf(full_post),
        "full_traversal_precursor_Trad_l2": mean_all_finite_or_inf(full_pre),
        "full_traversal_profile_l2_three_coords": aggregate_profile_l2_three_coords(per_snapshot),
        "full_traversal_postshock_Te_l2_nonfinite_count": nonfinite_count(full_post),
        "full_traversal_precursor_Trad_l2_nonfinite_count": nonfinite_count(full_pre),
        "steady_window_snapshot_count": len(steady_metrics),
        "steady_window_postshock_Te_l2": mean_all_finite_or_inf(steady_post),
        "steady_window_precursor_Trad_l2": mean_all_finite_or_inf(steady_pre),
        "profile_l2_three_coords": aggregate_profile_l2_three_coords(steady_metrics),
        "steady_window_postshock_Te_l2_nonfinite_count": nonfinite_count(steady_post),
        "steady_window_precursor_Trad_l2_nonfinite_count": nonfinite_count(steady_pre),
    }
    out.update(steady)
    return out


def far_state_masks(
    snapshot: Snapshot,
    shock_coord: float | None = None,
    boundary_frac: float = INTERIOR_MASK_FRACTION,
    shock_exclude_cells: int = 8,
) -> tuple[Any, Any]:
    import numpy as np

    coord = np.asarray(profile_centers(snapshot), dtype=float)
    if coord.size == 0:
        empty = np.zeros_like(coord, dtype=bool)
        return empty, empty
    shock = shock_coord if shock_coord is not None else estimate_shock_position(snapshot)
    if shock is None or not math.isfinite(float(shock)):
        shock = 0.5 * (float(np.nanmin(coord)) + float(np.nanmax(coord)))
    dx = median_cell_spacing(coord)
    mask = plateau_mask(coord, float(shock), dx, boundary_frac, shock_exclude_cells)
    up = mask & (coord < float(shock))
    dn = mask & (coord > float(shock))
    if np.count_nonzero(up) == 0 or np.count_nonzero(dn) == 0:
        n_edge = max(1, int(math.ceil(0.10 * coord.size)))
        up = np.zeros_like(coord, dtype=bool)
        dn = np.zeros_like(coord, dtype=bool)
        up[:n_edge] = True
        dn[-n_edge:] = True
    return up, dn


def matter_energy_flux_profile(snapshot: Snapshot, shock_speed: float) -> Any:
    import numpy as np

    rho = profile_average(snapshot, snapshot.rho)
    u = profile_velocity(snapshot)
    p_mat = profile_average(snapshot, np.asarray(snapshot.Pe) + np.asarray(snapshot.Pi))
    e_int = profile_average(snapshot, np.asarray(snapshot.ee) + np.asarray(snapshot.ei))
    w = np.asarray(u, dtype=float) - float(shock_speed)
    return np.asarray(rho, dtype=float) * w * (
        np.asarray(e_int, dtype=float)
        + 0.5 * w * w
        + np.asarray(p_mat, dtype=float) / np.maximum(np.asarray(rho, dtype=float), E_FLOOR)
    )


def radiation_flux_profile(
    snapshot: Snapshot,
    ref: dict[str, Any],
    shock_speed: float | None,
) -> tuple[Any, str]:
    import numpy as np

    if snapshot.rad_F is not None:
        direct = profile_average(snapshot, snapshot.rad_F)
        if finite_max_abs(direct, default=0.0) > 0.0:
            return direct, "hdf5:radiation/diag_F_first_moment"

    try:
        coord = np.asarray(profile_centers(snapshot), dtype=float)
        raw_x = np.asarray(reference_frame_coord(snapshot, ref), dtype=float)
        aligned_x = raw_x
        shock = estimate_shock_position(snapshot)
        if shock is not None and math.isfinite(float(shock)) and coord.size >= 2:
            shock_coord_ref_frame = float(np.interp(float(shock), coord, raw_x))
            aligned_x = raw_x - (shock_coord_ref_frame - reference_shock_position(ref))
        return interp_reference(ref, "F_rad_erg_per_cm2_s", aligned_x), "reference_table:F_rad_erg_per_cm2_s"
    except (KeyError, ValueError, RuntimeError, FloatingPointError):
        pass

    speed = float(shock_speed) if shock_speed is not None and math.isfinite(float(shock_speed)) else 0.0
    phi_mat = matter_energy_flux_profile(snapshot, speed)
    shock = estimate_shock_position(snapshot)
    up, dn = far_state_masks(snapshot, shock)
    phi_up = finite_average(phi_mat, up)
    phi_dn = finite_average(phi_mat, dn)
    if not (math.isfinite(phi_up) and math.isfinite(phi_dn)):
        return np.full_like(np.asarray(phi_mat, dtype=float), np.nan), "unavailable"
    f_up = float(ref["states"]["upstream"].get("F_rad", 0.0))
    f_dn = float(ref["states"]["downstream"].get("F_rad", 0.0))
    total_energy_flux = 0.5 * ((phi_up + f_up) + (phi_dn + f_dn))
    return total_energy_flux - phi_mat, "steady_energy_flux_inferred"


def compute_closure_defect_metrics(
    snapshot: Snapshot,
    params: dict[str, Any],
    ref: dict[str, Any],
    shock_speed: float | None,
) -> dict[str, Any]:
    import numpy as np

    a_eV = float(params["a_eV_erg_cm3_eV4"])
    c_light = float(params["c_cm_per_s"])
    kappa_R = float(params["kappa_R_cm2_per_g"])

    Te = profile_average(snapshot, snapshot.Te)
    Ti = profile_average(snapshot, snapshot.Ti)
    rho = profile_average(snapshot, snapshot.rho)
    x = profile_centers(snapshot)
    E_rad = profile_average(snapshot, snapshot.rad_E)
    Trad = np.power(np.maximum(np.asarray(E_rad, dtype=float) / a_eV, 0.0), 0.25)

    raw_x = np.asarray(reference_frame_coord(snapshot, ref), dtype=float)
    aligned_x = raw_x
    shock = estimate_shock_position(snapshot)
    if shock is not None and math.isfinite(float(shock)) and np.asarray(x).size >= 2:
        shock_coord_ref_frame = float(np.interp(float(shock), x, raw_x))
        aligned_x = raw_x - (shock_coord_ref_frame - reference_shock_position(ref))
    ref_Te = interp_reference(ref, "T_eV", aligned_x)
    ref_Trad = interp_reference(ref, "T_rad_eV", aligned_x)
    ref_Erad = interp_reference(ref, "E_rad_erg_per_cm3", aligned_x)

    delta_Tr = Trad / np.maximum(np.asarray(Te, dtype=float), E_FLOOR) - 1.0
    ref_delta_Tr = np.asarray(ref_Trad, dtype=float) / np.maximum(np.asarray(ref_Te, dtype=float), E_FLOOR) - 1.0
    delta_Tr_error = delta_Tr - ref_delta_Tr
    delta_ei = np.asarray(Ti, dtype=float) / np.maximum(np.asarray(Te, dtype=float), E_FLOOR) - 1.0

    F_rad, flux_source = radiation_flux_profile(snapshot, ref, shock_speed)
    E_from_Trad = a_eV * Trad**4
    D = c_light / (3.0 * kappa_R * np.maximum(np.asarray(rho, dtype=float), E_FLOOR))
    if np.asarray(x).size >= 2:
        dEdr_rad = np.gradient(E_from_Trad, x)
        dEdr_ref = np.gradient(ref_Erad, x)
    else:
        dEdr_rad = np.full_like(E_from_Trad, np.nan)
        dEdr_ref = np.full_like(E_from_Trad, np.nan)
    F_diff_from_Erad = -D * dEdr_rad
    F_diff_from_reference_Erad = -D * dEdr_ref
    F_rad_arr = np.asarray(F_rad, dtype=float)
    flux_resid_nonnED = (F_rad_arr - F_diff_from_Erad) / np.maximum(
        np.abs(F_rad_arr) + np.abs(F_diff_from_Erad), E_FLOOR
    )
    flux_resid_nED = (F_rad_arr - F_diff_from_reference_Erad) / np.maximum(
        np.abs(F_rad_arr) + np.abs(F_diff_from_reference_Erad), E_FLOOR
    )

    return {
        "delta_Tr": delta_Tr,
        "reference_delta_Tr": ref_delta_Tr,
        "delta_Tr_error": delta_Tr_error,
        "delta_ei": delta_ei,
        "flux_resid_nonnED": flux_resid_nonnED,
        "flux_resid_nED": flux_resid_nED,
        "F_rad": F_rad_arr,
        "F_diff_from_Erad": F_diff_from_Erad,
        "F_diff_from_reference_Erad": F_diff_from_reference_Erad,
        "flux_source": flux_source,
    }


def closure_defect_summary(
    snapshot: Snapshot,
    ref: dict[str, Any],
    shock_speed: float | None,
) -> dict[str, Any]:
    import numpy as np

    metrics = compute_closure_defect_metrics(snapshot, ref["parameters"], ref, shock_speed)
    coord = profile_centers(snapshot)
    Te = profile_average(snapshot, snapshot.Te)
    rho = profile_average(snapshot, snapshot.rho)
    shock = estimate_shock_position_from_profile(coord, Te, rho)
    interior = interior_mask(coord)
    if shock is None:
        post = interior
        pre = interior
    else:
        post = interior & (np.asarray(coord, dtype=float) >= float(shock))
        pre = interior & (np.asarray(coord, dtype=float) < float(shock))

    return {
        "time_s": float(snapshot.t),
        "step": int(snapshot.step),
        "shock_position_cm": None if shock is None else float(shock),
        "flux_source": metrics["flux_source"],
        "max_abs_delta_Tr_interior": finite_max_abs(metrics["delta_Tr"], interior),
        "p95_abs_delta_Tr_interior": finite_p95_abs(metrics["delta_Tr"], interior),
        "max_abs_reference_delta_Tr_interior": finite_max_abs(metrics["reference_delta_Tr"], interior),
        "nED_temperature_separation_l_inf": finite_max_abs(metrics["delta_Tr_error"], interior),
        "p95_abs_nED_temperature_separation": finite_p95_abs(metrics["delta_Tr_error"], interior),
        "l2_delta_Tr_postshock": finite_rms(metrics["delta_Tr"], post),
        "l2_delta_Tr_preshock": finite_rms(metrics["delta_Tr"], pre),
        "max_abs_delta_ei_interior": finite_max_abs(metrics["delta_ei"], interior),
        "p95_abs_delta_ei_interior": finite_p95_abs(metrics["delta_ei"], interior),
        "max_abs_flux_resid_nonnED": finite_max_abs(metrics["flux_resid_nonnED"], interior),
        "max_abs_flux_resid_nED": finite_max_abs(metrics["flux_resid_nED"], interior),
    }


def aggregate_closure_defects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "max_abs_delta_Tr_interior": math.inf,
            "p95_abs_delta_Tr_interior": math.inf,
            "max_abs_reference_delta_Tr_interior": math.inf,
            "nED_temperature_separation_l_inf": math.inf,
            "p95_abs_nED_temperature_separation": math.inf,
            "l2_delta_Tr_postshock": math.inf,
            "l2_delta_Tr_preshock": math.inf,
            "max_abs_delta_ei_interior": math.inf,
            "p95_abs_delta_ei_interior": math.inf,
            "max_abs_flux_resid_nonnED": math.inf,
            "max_abs_flux_resid_nED": math.inf,
            "flux_sources": [],
        }
    max_keys = [
        "max_abs_delta_Tr_interior",
        "p95_abs_delta_Tr_interior",
        "max_abs_reference_delta_Tr_interior",
        "nED_temperature_separation_l_inf",
        "p95_abs_nED_temperature_separation",
        "max_abs_delta_ei_interior",
        "p95_abs_delta_ei_interior",
        "max_abs_flux_resid_nonnED",
        "max_abs_flux_resid_nED",
    ]
    mean_keys = ["l2_delta_Tr_postshock", "l2_delta_Tr_preshock"]
    out: dict[str, Any] = {
        key: max([float(row.get(key, math.inf)) for row in rows] or [math.inf])
        for key in max_keys
    }
    out.update(
        {
            key: mean_all_finite_or_inf([float(row.get(key, math.inf)) for row in rows])
            for key in mean_keys
        }
    )
    out["flux_sources"] = sorted({str(row.get("flux_source", "unknown")) for row in rows})
    return out


def rankine_hugoniot_residuals(
    snapshot: Snapshot,
    shock_speed: float | None,
) -> dict[str, Any]:
    import numpy as np

    if shock_speed is None or not math.isfinite(float(shock_speed)):
        return {"R_J": math.inf, "R_Pi": math.inf, "R_Phi": math.inf}
    rho = profile_average(snapshot, snapshot.rho)
    u = profile_velocity(snapshot)
    P_mat = profile_average(snapshot, np.asarray(snapshot.Pe) + np.asarray(snapshot.Pi))
    e_int = profile_average(snapshot, np.asarray(snapshot.ee) + np.asarray(snapshot.ei))
    shock = estimate_shock_position(snapshot)
    mask_up, mask_dn = far_state_masks(snapshot, shock)

    rho_u = finite_average(rho, mask_up)
    rho_d = finite_average(rho, mask_dn)
    u_u = finite_average(u, mask_up)
    u_d = finite_average(u, mask_dn)
    P_u = finite_average(P_mat, mask_up)
    P_d = finite_average(P_mat, mask_dn)
    e_u = finite_average(e_int, mask_up)
    e_d = finite_average(e_int, mask_dn)
    values = [rho_u, rho_d, u_u, u_d, P_u, P_d, e_u, e_d]
    if any(not math.isfinite(float(value)) for value in values):
        return {"R_J": math.inf, "R_Pi": math.inf, "R_Phi": math.inf}

    w_u = u_u - float(shock_speed)
    w_d = u_d - float(shock_speed)
    J_u = rho_u * w_u
    J_d = rho_d * w_d
    R_J = abs(J_d - J_u) / max(abs(J_u), abs(J_d), E_FLOOR)

    Pi_u = rho_u * w_u * w_u + P_u
    Pi_d = rho_d * w_d * w_d + P_d
    R_Pi = abs(Pi_d - Pi_u) / max(abs(Pi_u), abs(Pi_d), E_FLOOR)

    Phi_u = J_u * (e_u + 0.5 * w_u * w_u + P_u / max(rho_u, E_FLOOR))
    Phi_d = J_d * (e_d + 0.5 * w_d * w_d + P_d / max(rho_d, E_FLOOR))
    R_Phi = abs(Phi_d - Phi_u) / max(abs(Phi_u), abs(Phi_d), E_FLOOR)
    return {"R_J": R_J, "R_Pi": R_Pi, "R_Phi": R_Phi}


def rh_residual_summary(snapshot: Snapshot, shock_speed: float | None) -> dict[str, Any]:
    out = rankine_hugoniot_residuals(snapshot, shock_speed)
    out.update({"time_s": float(snapshot.t), "step": int(snapshot.step)})
    return out


def aggregate_rh_residuals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {
        "max_R_J": max([float(row.get("R_J", math.inf)) for row in rows] or [math.inf]),
        "max_R_Pi": max([float(row.get("R_Pi", math.inf)) for row in rows] or [math.inf]),
        "max_R_Phi": max([float(row.get("R_Phi", math.inf)) for row in rows] or [math.inf]),
    }
    out["gates"] = {key: float(value) < 1.0e-2 for key, value in out.items()}
    return out


def snapshot_dump_arrays(
    snapshot: Snapshot,
    ref: dict[str, Any],
    shock_speed: float | None,
) -> dict[str, Any]:
    import numpy as np

    coord = np.asarray(profile_centers(snapshot), dtype=float)
    rho = profile_average(snapshot, snapshot.rho)
    u = profile_velocity(snapshot)
    Te = profile_average(snapshot, snapshot.Te)
    Ti = profile_average(snapshot, snapshot.Ti)
    E_rad = profile_average(snapshot, snapshot.rad_E)
    Trad = np.power(np.maximum(np.asarray(E_rad, dtype=float) / A_EV, 0.0), 0.25)
    P_mat = profile_average(snapshot, np.asarray(snapshot.Pe) + np.asarray(snapshot.Pi))
    P_rad = np.asarray(E_rad, dtype=float) / 3.0
    shock = estimate_shock_position_from_profile(coord, Te, rho)
    if shock is None:
        shock = 0.5 * (float(np.nanmin(coord)) + float(np.nanmax(coord)))
    kappa_R = float(ref["parameters"]["kappa_R_cm2_per_g"])
    metrics = compute_closure_defect_metrics(snapshot, ref["parameters"], ref, shock_speed)
    return {
        "r_cm": coord,
        "x_shifted_cm": coord - float(shock),
        "tau_from_shock": signed_cumulative_integral(coord, kappa_R * np.asarray(rho, dtype=float), float(shock)),
        "m_from_shock": signed_cumulative_integral(coord, rho, float(shock)),
        "rho": rho,
        "u": u,
        "Te": Te,
        "Ti": Ti,
        "Trad": Trad,
        "E_rad": E_rad,
        "aTe4": A_EV * np.asarray(Te, dtype=float) ** 4,
        "P_mat": P_mat,
        "P_rad": P_rad,
        "F_rad": metrics["F_rad"],
        "F_diff_Erad": metrics["F_diff_from_Erad"],
        "F_diff_reference_Erad": metrics["F_diff_from_reference_Erad"],
        "cell_width": profile_cell_widths(snapshot),
        "shock_index": np.full_like(coord, int(np.nanargmin(np.abs(coord - float(shock)))), dtype=int),
    }


def dump_profile_csv(
    snapshot: Snapshot,
    ref: dict[str, Any],
    out_dir: Path,
    run_id: str,
    label: str,
    shock_speed: float | None,
) -> dict[str, Any]:
    import csv
    import numpy as np

    columns = [
        "r_cm",
        "x_shifted_cm",
        "tau_from_shock",
        "m_from_shock",
        "rho",
        "u",
        "Te",
        "Ti",
        "Trad",
        "E_rad",
        "aTe4",
        "P_mat",
        "P_rad",
        "F_rad",
        "F_diff_Erad",
        "F_diff_reference_Erad",
        "cell_width",
        "shock_index",
    ]
    arrays = snapshot_dump_arrays(snapshot, ref, shock_speed)
    profile_dir = out_dir / "validation_profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{run_id}_{label}.csv"
    n = len(np.asarray(arrays["r_cm"]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for i in range(n):
            writer.writerow([arrays[name][i] for name in columns])
    return {
        "label": label,
        "path": str(path),
        "time_s": float(snapshot.t),
        "step": int(snapshot.step),
        "cell_count": int(n),
    }


def dump_profiles_for_run(
    snapshots: list[Snapshot],
    ref: dict[str, Any],
    out_dir: Path,
    run_id: str,
    shock_speed: float | None,
) -> list[dict[str, Any]]:
    if not snapshots:
        return []
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    t_end = max(float(snapshot.t) for snapshot in snapshots)
    rows: list[dict[str, Any]] = []
    for frac in fractions:
        target = frac * t_end
        snapshot = min(snapshots, key=lambda snap: abs(float(snap.t) - target))
        label = f"t{safe_float_token(frac)}_tend"
        rows.append(dump_profile_csv(snapshot, ref, out_dir, run_id, label, shock_speed))
    return rows


def output_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def run_info_termination(outdir: Path) -> str | None:
    path = discover_actual_outdir(outdir) / "run_info.json"
    if not path.exists():
        return None
    try:
        info = read_json(path)
    except (OSError, json.JSONDecodeError):
        return None
    return info.get("termination_reason")


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
    env = {
        "TENRYU_I1_LE08_M0": str(args.mach),
        "TENRYU_I1_LE08_NR": str(nr),
        "TENRYU_I1_LE08_SEED": str(args.seed),
        "TENRYU_I1_LE08_OUTDIR": str(outdir),
        "TENRYU_I1_LE08_REFERENCE": str(args.reference),
        "TENRYU_I1_LE08_INIT_MODE": args.init_mode,
        "TENRYU_I1_LE08_MAX_STEPS": str(max_steps if max_steps is not None else args.max_steps),
        "TENRYU_I1_LE08_HISTORY_EVERY_STEPS": "1",
        "TENRYU_I1_LE08_VERBOSITY": args.verbosity,
    }
    if dimension == "2D_RZ":
        env["TENRYU_I1_LE08_NZ"] = str(nz)
        env["TENRYU_I1_LE08_MESH_MOTION"] = args.mesh_motion
    effective_t_end_s = t_end_s if t_end_s is not None else args.t_end_s
    effective_dt_initial_s = dt_initial_s if dt_initial_s is not None else args.dt_initial_s
    effective_dt_max_s = dt_max_s if dt_max_s is not None else args.dt_max_s
    if effective_t_end_s is not None:
        env["TENRYU_I1_LE08_T_END_S"] = str(effective_t_end_s)
    if effective_dt_initial_s is not None:
        env["TENRYU_I1_LE08_DT_INITIAL_S"] = str(effective_dt_initial_s)
    if effective_dt_max_s is not None:
        env["TENRYU_I1_LE08_DT_MAX_S"] = str(effective_dt_max_s)
    return env


def nominal_cell_width(snapshot: Snapshot, ref: dict[str, Any], nr: int) -> float:
    import numpy as np

    if snapshot.dimension == "1D_SPH":
        r = np.asarray(snapshot.r_nodes, dtype=float)
        if r.size >= 2:
            widths = np.diff(r)
            finite = widths[np.isfinite(widths) & (widths > 0.0)]
            if finite.size:
                return float(np.median(finite))
    return (float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])) / max(nr, 1)


def reference_t_end_s(ref: dict[str, Any], fraction: float) -> float:
    x = ref["table"]["x_cm"]
    domain = float(x[-1]) - float(x[0])
    u_upstream = abs(float(ref["states"]["upstream"]["u"]))
    return fraction * domain / max(u_upstream, E_FLOOR)


def run_one(
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
    base_run_id = case_name(args.mach, nr, nz, args.seed, args.init_mode, dimension)
    run_id = base_run_id if run_label == "production" else f"{base_run_id}_{run_label}"
    outdir = Path(args.out_root) / run_id
    outdir.mkdir(parents=True, exist_ok=True)
    env_overrides = active_env(
        args,
        nr,
        nz,
        dimension,
        outdir,
        t_end_s=t_end_s,
        max_steps=max_steps,
        dt_initial_s=dt_initial_s,
        dt_max_s=dt_max_s,
    )
    env = os.environ.copy()
    env.update(env_overrides)
    log_path = outdir / "run.log"
    start = time.monotonic()
    timed_out = False
    proc = subprocess.CompletedProcess(args=[args.tenryu_bin, "run", args.deck], returncode=-1)
    with log_path.open("w", encoding="utf-8") as log:
        try:
            proc = subprocess.run(
                [args.tenryu_bin, "run", args.deck],
                env=env,
                stdout=log,
                stderr=log,
                timeout=args.wall_time_s,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
    wall_time_s = time.monotonic() - start

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "case_name": base_run_id,
        "run_label": run_label,
        "dimension": dimension,
        "nr": nr,
        "nz": nz if dimension == "2D_RZ" else None,
        "exit_code": proc.returncode,
        "timed_out": timed_out,
        "wall_time_s": wall_time_s,
        "output_bytes": output_bytes(outdir),
        "termination_reason": run_info_termination(outdir),
        "analysis_error": None,
    }
    if timed_out or proc.returncode != 0:
        metrics["run_status"] = "wall_time_timeout" if timed_out else "crashed"
        metrics["log_tail"] = read_text(log_path)[-4000:]
        return metrics

    try:
        results_dir = discover_results_dir(outdir)
        actual_outdir = results_dir.parent
        metrics["output_bytes"] = output_bytes(actual_outdir)
        snapshots = read_snapshots(results_dir, base_run_id, nr, nz, dimension)
        final = snapshots[-1]
        shock_velocity, shock_positions = fit_shock_velocity(snapshots)
        final_profile = profile_metrics(final, ref)
        window_profile = profile_window_metrics(snapshots, ref)
        steady_start_index = int(window_profile.get("steady_start_index", len(snapshots) - 1))
        steady_start_index = max(0, min(steady_start_index, len(snapshots) - 1))
        closure_rows = [
            dict({"snapshot_index": index}, **closure_defect_summary(snapshot, ref, shock_velocity))
            for index, snapshot in enumerate(snapshots)
        ]
        rh_rows = [
            dict({"snapshot_index": index}, **rh_residual_summary(snapshot, shock_velocity))
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
        metrics["closure_defects"] = aggregate_closure_defects(closure_rows[steady_start_index:])
        metrics["rh_residual_snapshots"] = rh_rows
        metrics["rh_residuals"] = aggregate_rh_residuals(rh_rows[steady_start_index:])
        if args.dump_profiles:
            metrics["profile_dumps"] = dump_profiles_for_run(
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
                "dz_nominal_cm": nominal_cell_width(final, ref, nr if dimension == "1D_SPH" else nz),
            }
        )
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["run_status"] = "analysis_failure"
        metrics["analysis_error"] = str(exc)
        metrics["log_tail"] = read_text(log_path)[-4000:]
    return metrics


def finite_mean(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.mean(arr))


def rel_diff(got: float, ref: float) -> float:
    return abs(float(got) - float(ref)) / max(abs(float(ref)), E_FLOOR)


def state_triplet(code: float, ref: float) -> dict[str, float]:
    return {"code": float(code), "ref": float(ref), "rel_diff": rel_diff(code, ref)}


def edge_state_diagnostics(
    x_ref: Any,
    rho: Any,
    u: Any,
    p_mat: Any,
    Te: Any,
    ref: dict[str, Any],
    side: str,
) -> dict[str, Any]:
    import numpy as np

    x = np.asarray(x_ref, dtype=float)
    span = float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
    n_edge = max(1, int(math.ceil(0.05 * x.size)))
    if side == "upstream":
        cutoff = float(ref["table"]["x_cm"][0]) + 0.05 * span
        mask = x <= cutoff
        fallback = np.zeros_like(x, dtype=bool)
        fallback[:n_edge] = True
        ref_state = ref["states"]["upstream"]
    elif side == "downstream":
        cutoff = float(ref["table"]["x_cm"][-1]) - 0.05 * span
        mask = x >= cutoff
        fallback = np.zeros_like(x, dtype=bool)
        fallback[-n_edge:] = True
        ref_state = ref["states"]["downstream"]
    else:
        raise ValueError(f"unknown edge side {side!r}")
    if np.count_nonzero(mask) == 0:
        mask = fallback
    return {
        "rho": state_triplet(finite_mean(np.asarray(rho, dtype=float)[mask]), float(ref_state["rho"])),
        "u": state_triplet(finite_mean(np.asarray(u, dtype=float)[mask]), float(ref_state["u"])),
        "P_mat": state_triplet(finite_mean(np.asarray(p_mat, dtype=float)[mask]), float(ref_state["P_mat"])),
        "T_e": state_triplet(finite_mean(np.asarray(Te, dtype=float)[mask]), float(ref_state["T"])),
        "cell_count": int(np.count_nonzero(mask)),
    }


def t0_diagnostics(snapshot: Snapshot, ref: dict[str, Any], init_mode: str) -> dict[str, Any]:
    import numpy as np

    x_ref = reference_frame_coord(snapshot, ref)
    rho = profile_average(snapshot, snapshot.rho)
    Te = profile_average(snapshot, snapshot.Te)
    u = profile_velocity(snapshot)
    p_mat = profile_average(snapshot, np.asarray(snapshot.Pe) + np.asarray(snapshot.Pi))
    p_rad = profile_average(snapshot, np.asarray(snapshot.rad_E) / 3.0)
    Trad = np.power(np.maximum(profile_average(snapshot, snapshot.rad_E) / A_EV, 0.0), 0.25)

    shock_x0 = two_state_shock_position_ref_frame(ref) if init_mode == "two_state" else reference_shock_position(ref)
    expected = build_t0_admissibility_reference(ref, x_ref, init_mode, shock_x0)
    dx = median_cell_spacing(x_ref)
    audit_mask = (
        plateau_mask(x_ref, shock_x0, dx, shock_exclude_cells=6)
        if init_mode == "two_state"
        else interior_mask(x_ref)
    )
    if np.count_nonzero(audit_mask) == 0:
        audit_mask = np.ones_like(np.asarray(x_ref, dtype=float), dtype=bool)

    span = float(ref["table"]["x_cm"][-1]) - float(ref["table"]["x_cm"][0])
    if init_mode == "two_state":
        upstream_mask = audit_mask & (np.asarray(x_ref, dtype=float) < shock_x0)
    else:
        upstream_mask = x_ref <= float(ref["table"]["x_cm"][0]) + 0.05 * span
    if np.count_nonzero(upstream_mask) == 0:
        upstream_mask = np.zeros_like(np.asarray(x_ref, dtype=float), dtype=bool)
        upstream_mask[: max(1, int(math.ceil(0.05 * upstream_mask.size)))] = True
    rho_up = finite_mean(np.asarray(rho, dtype=float)[upstream_mask])
    p_up = finite_mean(np.asarray(p_mat, dtype=float)[upstream_mask])
    u_up = finite_mean(np.asarray(u, dtype=float)[upstream_mask])
    gamma = float(ref["parameters"]["gamma"])
    M_code = abs(u_up) / max(math.sqrt(max(gamma * p_up / max(rho_up, E_FLOOR), 0.0)), E_FLOOR)
    M_target = float(ref["parameters"]["M0"])
    M_mismatch_rel = rel_diff(M_code, M_target)

    code_ratio = np.asarray(p_rad, dtype=float) / np.maximum(np.asarray(p_mat, dtype=float), E_FLOOR)
    ref_ratio = np.asarray(expected["P_rad"], dtype=float) / np.maximum(
        np.asarray(expected["P_mat"], dtype=float), E_FLOOR
    )
    max_code_ratio = float(np.nanmax(code_ratio))
    max_ref_ratio = float(np.nanmax(ref_ratio))
    if init_mode == "two_state":
        code_ref_ratio_mismatch = rel_linf(code_ratio, ref_ratio, audit_mask)
        code_ref_ratio = math.nan
    else:
        code_ref_ratio = max_code_ratio / max(max_ref_ratio, E_FLOOR)
        code_ref_ratio_mismatch = abs(code_ref_ratio - 1.0)
    trad_ref_mismatch = float(
        np.nanmax(
            (
                np.abs(np.asarray(Trad, dtype=float) - np.asarray(expected["T_rad"], dtype=float))
                / np.maximum(np.asarray(expected["T_rad"], dtype=float), E_FLOOR)
            )[audit_mask]
        )
    )
    plateau_rel = {
        "rho": rel_linf(rho, expected["rho"], audit_mask),
        "u": rel_linf(u, expected["u"], audit_mask),
        "Te": rel_linf(Te, expected["T"], audit_mask),
        "Trad": rel_linf(Trad, expected["T_rad"], audit_mask),
        "P_mat": rel_linf(p_mat, expected["P_mat"], audit_mask),
        "P_rad_over_P_mat": code_ref_ratio_mismatch,
        "Trad_reference_match": trad_ref_mismatch,
    }

    diagnostics = {
        "init_mode": init_mode,
        "M_code": M_code,
        "M_target": M_target,
        "M_mismatch_rel": M_mismatch_rel,
        "shock_x0_cm_ref_frame": shock_x0,
        "plateau_mask_cell_count": int(np.count_nonzero(audit_mask)),
        "plateau_rel": plateau_rel,
        "max_code_P_rad_over_P_mat_at_t0": max_code_ratio,
        "max_ref_P_rad_over_P_mat_at_t0": max_ref_ratio,
        "code_ref_ratio": code_ref_ratio,
        "code_ref_ratio_mismatch": code_ref_ratio_mismatch,
        "max_abs_T_rad_minus_T_reference_over_T_reference_at_t0": trad_ref_mismatch,
        "max_abs_T_rad_minus_T_e_over_T_e_at_t0": float(
            np.nanmax(
                (
                    np.abs(np.asarray(Trad, dtype=float) - np.asarray(Te, dtype=float))
                    / np.maximum(np.asarray(Te, dtype=float), E_FLOOR)
                )[audit_mask]
            )
        ),
        "upstream_state_code_vs_ref": edge_state_diagnostics(x_ref, rho, u, p_mat, Te, ref, "upstream"),
        "downstream_state_code_vs_ref": edge_state_diagnostics(x_ref, rho, u, p_mat, Te, ref, "downstream"),
    }
    diagnostics["gates"] = {
        "mach_match": M_mismatch_rel <= T0_MACH_REL_GATE,
        "pressure_ratio_match": code_ref_ratio_mismatch <= T0_PRAD_RATIO_REL_GATE,
        "ned_two_state_radiation_admissibility": trad_ref_mismatch <= T0_TRAD_TE_REL_GATE,
    }
    if init_mode == "two_state":
        diagnostics["gates"].update(
            {
                "rho_plateau_match": plateau_rel["rho"] <= 5.0e-3,
                "u_plateau_match": plateau_rel["u"] <= 5.0e-3,
                "Te_plateau_match": plateau_rel["Te"] <= 5.0e-3,
                "P_rad_over_P_mat_plateau_match": plateau_rel["P_rad_over_P_mat"] <= 5.0e-2,
                "Trad_reference_plateau_match": plateau_rel["Trad_reference_match"] <= 1.0e-3,
            }
        )
    diagnostics["passed"] = all(diagnostics["gates"].values())
    return diagnostics


def run_t0_audit_one(args: argparse.Namespace, nr: int, nz: int, dimension: str, ref: dict[str, Any]) -> dict[str, Any]:
    audit_t_end = max(reference_t_end_s(ref, 0.5) * 1.0e-9, 1.0e-18)
    metrics = run_one(
        args,
        nr,
        nz,
        dimension,
        ref,
        run_label="t0_audit",
        t_end_s=audit_t_end,
        max_steps=1,
        dt_initial_s=audit_t_end,
        dt_max_s=audit_t_end,
    )
    metrics["audit_t_end_s"] = audit_t_end
    metrics["audit_passed"] = False
    if metrics.get("exit_code") != 0 or metrics.get("timed_out"):
        return metrics
    try:
        outdir = Path(args.out_root) / str(metrics["run_id"])
        results_dir = discover_results_dir(outdir)
        snapshots = read_snapshots(
            results_dir,
            str(metrics["case_name"]),
            nr,
            nz,
            dimension,
            min_count=1,
        )
        initial = min(snapshots, key=lambda snap: (abs(float(snap.t)), abs(int(snap.step))))
        diagnostics = t0_diagnostics(initial, ref, args.init_mode)
        metrics["diagnostics_t0"] = diagnostics
        metrics["audit_passed"] = bool(diagnostics["passed"])
        metrics["initial_snapshot_path"] = str(initial.path)
    except (OSError, KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
        metrics["analysis_error"] = str(exc)
    return metrics


def aggregate_t0_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    diag_rows = [row.get("diagnostics_t0", {}) for row in rows]
    gate_names = sorted({name for diag in diag_rows for name in diag.get("gates", {})})
    gates = {
        name: all(bool(diag.get("gates", {}).get(name)) for diag in diag_rows) if diag_rows else False
        for name in gate_names
    }
    plateau_keys = sorted({name for diag in diag_rows for name in diag.get("plateau_rel", {})})
    return {
        "max_M_mismatch_rel": max([float(diag.get("M_mismatch_rel", math.inf)) for diag in diag_rows] or [math.inf]),
        "max_code_ref_ratio_mismatch": max(
            [float(diag.get("code_ref_ratio_mismatch", math.inf)) for diag in diag_rows] or [math.inf]
        ),
        "max_abs_T_rad_minus_T_e_over_T_e_at_t0": max(
            [float(diag.get("max_abs_T_rad_minus_T_e_over_T_e_at_t0", math.inf)) for diag in diag_rows] or [math.inf]
        ),
        "max_abs_T_rad_minus_T_reference_over_T_reference_at_t0": max(
            [
                float(diag.get("max_abs_T_rad_minus_T_reference_over_T_reference_at_t0", math.inf))
                for diag in diag_rows
            ]
            or [math.inf]
        ),
        "max_plateau_rel": {
            key: max([float(diag.get("plateau_rel", {}).get(key, math.inf)) for diag in diag_rows] or [math.inf])
            for key in plateau_keys
        },
        "gates": gates,
        "passed": all(gates.values()) and all(bool(row.get("audit_passed")) for row in rows),
    }


def quarter_t_end_summary(row: dict[str, Any], production_aggregate: dict[str, Any]) -> dict[str, Any]:
    l2_gates_passed = (
        float(row.get("postshock_Te_l2", math.inf)) <= POSTSHOCK_TE_L2_GATE
        and float(row.get("precursor_Trad_l2", math.inf)) <= PRECURSOR_TRAD_L2_GATE
    )
    production_l2_passed = (
        float(production_aggregate.get("fine_postshock_Te_l2", math.inf)) <= POSTSHOCK_TE_L2_GATE
        and float(production_aggregate.get("fine_precursor_Trad_l2", math.inf)) <= PRECURSOR_TRAD_L2_GATE
    )
    out = dict(row)
    out.update(
        {
            "diagnostic_t_end_fraction": 0.25,
            "l2_gates_passed": l2_gates_passed,
            "boundary_or_transient_contamination_hint": (not production_l2_passed) and l2_gates_passed,
        }
    )
    return out


def convergence_rate(errors: list[float]) -> float:
    finite_raw = [float(e) for e in errors if math.isfinite(float(e))]
    if len(finite_raw) != len(errors):
        return -math.inf
    if len(finite_raw) < 2:
        return math.inf if finite_raw and max(finite_raw) <= E_FLOOR else -math.inf
    if max(finite_raw) <= E_FLOOR:
        return math.inf
    finite = [max(e, E_FLOOR) for e in finite_raw]
    rates = [math.log(finite[i] / finite[i + 1]) / math.log(2.0) for i in range(len(finite) - 1)]
    return min(rates)


def optional_float(value: Any, default: float = math.inf) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def aggregate(rows: list[dict[str, Any]], ref: dict[str, Any]) -> dict[str, Any]:
    shock_positions = [
        optional_float(row.get("final_shock_position_cm", row.get("shock_position_cm", math.inf)))
        for row in rows
    ]
    shock_errors = [
        abs(shock_positions[i] - shock_positions[i + 1])
        if math.isfinite(shock_positions[i]) and math.isfinite(shock_positions[i + 1])
        else math.inf
        for i in range(len(shock_positions) - 1)
    ]
    shock_rate = convergence_rate(shock_errors)
    fine = rows[-1] if rows else {}
    fine_error_cells = (
        float(shock_errors[-1]) / max(float(fine.get("dz_nominal_cm", math.inf)), E_FLOOR)
        if shock_errors and math.isfinite(float(shock_errors[-1]))
        else math.inf
    )
    fine_postshock_te_l2 = fine.get("steady_window_postshock_Te_l2", fine.get("postshock_Te_l2", math.inf))
    fine_precursor_trad_l2 = fine.get(
        "steady_window_precursor_Trad_l2", fine.get("precursor_Trad_l2", math.inf)
    )
    fine_closure = fine.get("closure_defects", {})
    fine_ned_temp_sep = fine_closure.get(
        "nED_temperature_separation_l_inf",
        fine.get("nED_temperature_separation_l_inf", math.inf),
    )
    fine_rh = fine.get("rh_residuals", {})
    fine_three_coord = fine.get("profile_l2_three_coords", {})
    ref_diag = ref["diagnostics"]
    max_prad = max(
        [float(ref_diag["max_P_rad_over_P_mat"])]
        + [float(row.get("max_P_rad_over_P_mat", math.inf)) for row in rows]
    )
    max_force = max(
        [float(ref_diag["max_abs_grad_P_rad_over_abs_grad_P_mat"])]
        + [float(row.get("max_abs_grad_P_rad_over_abs_grad_P_mat", math.inf)) for row in rows]
    )
    conservation = float(ref_diag["conservation_residual_rel_max"])
    gates = {
        "termination_t_end_reached": all(row.get("termination_reason") == "t_end_reached" for row in rows),
        "gate1_low_prad_only": max_prad <= LOW_PRAD_GATE,
        "gate1_force_ratio_diagnostic": max_force <= LOW_FORCE_GATE,
        "gate2_reference_conservation": conservation <= CONSERVATION_GATE,
        "gate3_shock_velocity_convergence": shock_rate >= SHOCK_POSITION_RATE_GATE,
        "gate4_postshock_Te_l2": optional_float(fine_postshock_te_l2) <= POSTSHOCK_TE_L2_GATE,
        "gate5_precursor_Trad_l2": optional_float(fine_precursor_trad_l2) <= PRECURSOR_TRAD_L2_GATE,
    }
    return {
        "max_P_rad_over_P_mat": max_prad,
        "max_abs_grad_P_rad_over_abs_grad_P_mat": max_force,
        "reference_conservation_residual_rel_max": conservation,
        "shock_position_final_cm": shock_positions,
        "shock_position_pairwise_errors_cm": shock_errors,
        "shock_velocity_errors_cm": shock_errors,
        "shock_velocity_convergence_rate": shock_rate,
        "shock_position_convergence_rate": shock_rate,
        "fine_shock_error_cells": fine_error_cells,
        "fine_postshock_Te_l2": fine_postshock_te_l2,
        "fine_precursor_Trad_l2": fine_precursor_trad_l2,
        "fine_final_postshock_Te_l2": fine.get("final_postshock_Te_l2"),
        "fine_final_precursor_Trad_l2": fine.get("final_precursor_Trad_l2"),
        "nED_temperature_separation_l_inf": fine_ned_temp_sep,
        "fine_nED_temperature_separation_l_inf": fine_ned_temp_sep,
        "fine_reference_nED_temperature_separation_l_inf": fine.get(
            "reference_nED_temperature_separation_l_inf", math.inf
        ),
        "fine_full_traversal_postshock_Te_l2": fine.get("full_traversal_postshock_Te_l2"),
        "fine_full_traversal_precursor_Trad_l2": fine.get("full_traversal_precursor_Trad_l2"),
        "fine_steady_window_snapshot_count": fine.get("steady_window_snapshot_count"),
        "fine_steady_start_time_s": fine.get("steady_start_time_s"),
        "fine_steady_start_method": fine.get("steady_start_method"),
        "closure_defects": fine_closure,
        "nED_closure_defects": fine_closure,
        "fine_closure_defects": fine_closure,
        "fine_max_abs_delta_Tr_interior": fine_closure.get("max_abs_delta_Tr_interior", math.inf),
        "fine_p95_abs_delta_Tr_interior": fine_closure.get("p95_abs_delta_Tr_interior", math.inf),
        "fine_max_abs_reference_delta_Tr_interior": fine_closure.get(
            "max_abs_reference_delta_Tr_interior", math.inf
        ),
        "fine_p95_abs_nED_temperature_separation": fine_closure.get(
            "p95_abs_nED_temperature_separation", math.inf
        ),
        "fine_l2_delta_Tr_postshock": fine_closure.get("l2_delta_Tr_postshock", math.inf),
        "fine_l2_delta_Tr_preshock": fine_closure.get("l2_delta_Tr_preshock", math.inf),
        "fine_max_abs_delta_ei_interior": fine_closure.get("max_abs_delta_ei_interior", math.inf),
        "fine_p95_abs_delta_ei_interior": fine_closure.get("p95_abs_delta_ei_interior", math.inf),
        "fine_max_abs_flux_resid_nonnED": fine_closure.get("max_abs_flux_resid_nonnED", math.inf),
        "fine_max_abs_flux_resid_nED": fine_closure.get("max_abs_flux_resid_nED", math.inf),
        "rh_residuals": fine_rh,
        "fine_rh_residuals": fine_rh,
        "fine_max_R_J": fine_rh.get("max_R_J", math.inf),
        "fine_max_R_Pi": fine_rh.get("max_R_Pi", math.inf),
        "fine_max_R_Phi": fine_rh.get("max_R_Phi", math.inf),
        "profile_l2_three_coords": fine_three_coord,
        "fine_profile_l2_three_coords": fine_three_coord,
        "fine_postshock_Te_l2_x": fine_three_coord.get("postshock_Te_l2_x", math.inf),
        "fine_postshock_Te_l2_tau": fine_three_coord.get("postshock_Te_l2_tau", math.inf),
        "fine_postshock_Te_l2_m": fine_three_coord.get("postshock_Te_l2_m", math.inf),
        "fine_postshock_Trad_l2_x": fine_three_coord.get("postshock_Trad_l2_x", math.inf),
        "fine_postshock_Trad_l2_tau": fine_three_coord.get("postshock_Trad_l2_tau", math.inf),
        "fine_postshock_Trad_l2_m": fine_three_coord.get("postshock_Trad_l2_m", math.inf),
        "fine_precursor_Trad_l2_x": fine_three_coord.get("precursor_Trad_l2_x", math.inf),
        "fine_precursor_Trad_l2_tau": fine_three_coord.get("precursor_Trad_l2_tau", math.inf),
        "fine_precursor_Trad_l2_m": fine_three_coord.get("precursor_Trad_l2_m", math.inf),
        "gates": gates,
        "all_checks_passed": all(
            gates[name]
            for name in (
                "termination_t_end_reached",
                "gate1_low_prad_only",
                "gate2_reference_conservation",
                "gate3_shock_velocity_convergence",
                "gate4_postshock_Te_l2",
                "gate5_precursor_Trad_l2",
            )
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/i1_le08_grey_ned_radshock.py")
    parser.add_argument("--reference", default="tests/verification/data/le08_ned_reference/le08_nED_M2.json")
    parser.add_argument("--mach", type=float, default=2.0)
    parser.add_argument("--grids", type=parse_grid_list, default=parse_grid_list("64;128;256"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--init-mode", choices=("reference_table", "two_state"), default="two_state")
    parser.add_argument("--mesh-motion", choices=("lagrangian", "ale"), default="lagrangian")
    parser.add_argument("--out-root", default="./build/i1_le08_ned_runs")
    parser.add_argument("--summary-json", default="./build/i1_le08_ned_runs/summary.json")
    parser.add_argument("--max-steps", type=int, default=1000000)
    parser.add_argument("--t-end-s", type=float, default=None)
    parser.add_argument("--dt-initial-s", type=float, default=None)
    parser.add_argument("--dt-max-s", type=float, default=None)
    parser.add_argument("--verbosity", choices=("quiet", "normal", "verbose"), default="quiet")
    parser.add_argument("--wall-time-s", type=float, default=None)
    parser.add_argument("--dump-profiles", action="store_true", help="write Wave 5 per-snapshot profile CSV dumps")
    parser.add_argument(
        "--allow-partial-checkpoint",
        action="store_true",
        help="return success for the Phase A partial checkpoint when gates 1-2 and t=0 pass",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    dimension = deck_dimension(Path(args.deck))
    try:
        validate_grids_for_dimension(args.grids, dimension)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if dimension == "1D_SPH" and args.mesh_motion != "lagrangian":
        parser.error("1D_SPH LE08 requires --mesh-motion lagrangian")
    ref = read_json(Path(args.reference))
    ref_identity = reference_identity_branch_wise(ref)
    t0_rows = [run_t0_audit_one(args, nr, nz, dimension, ref) for nr, nz in args.grids]
    t0_aggregate = aggregate_t0_audits(t0_rows)
    rows: list[dict[str, Any]] = []
    quarter_diag: dict[str, Any] | None = None
    if t0_aggregate["passed"]:
        rows = [run_one(args, nr, nz, dimension, ref) for nr, nz in args.grids]
        agg = aggregate(rows, ref)
        fine_nr, fine_nz = args.grids[-1]
        production_t_end = args.t_end_s if args.t_end_s is not None else reference_t_end_s(ref, 3.0)
        diagnostic_t_end = 0.25 * production_t_end
        quarter_row = run_one(
            args,
            fine_nr,
            fine_nz,
            dimension,
            ref,
            run_label="quarter_t_end",
            t_end_s=diagnostic_t_end,
        )
        quarter_diag = quarter_t_end_summary(quarter_row, agg)
    else:
        agg = aggregate(rows, ref)
    payload = {
        "schema_version": "tenryu.i1_le08_grey_ned_radshock.summary.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "stage_id": STAGE_ID,
        "ladder_row": LADDER_ROW,
        "dimension": dimension,
        "deck_path": args.deck,
        "reference_path": args.reference,
        "reference_parameters": ref["parameters"],
        "reference_identity_branch_wise": ref_identity,
        "t0_admissibility": {
            "rows": t0_rows,
            "aggregate": t0_aggregate,
        },
        "matrix": rows,
        "aggregate": agg,
        "quarter_t_end_diagnostic": quarter_diag,
    }
    write_json(Path(args.summary_json), payload)
    print(
        json.dumps(
            {
                "ladder_row": LADDER_ROW,
                "reference_identity_branch_wise": ref_identity,
                "t0_admissibility": t0_aggregate,
                "aggregate": agg,
                "quarter_t_end_diagnostic": quarter_diag,
            },
            indent=2,
            sort_keys=True,
        )
    )
    strict_pass = bool(t0_aggregate["passed"]) and bool(agg["all_checks_passed"])
    partial_pass = (
        bool(args.allow_partial_checkpoint)
        and bool(t0_aggregate["passed"])
        and bool(agg.get("gates", {}).get("termination_t_end_reached"))
        and bool(agg.get("gates", {}).get("gate1_low_prad_only"))
        and bool(agg.get("gates", {}).get("gate2_reference_conservation"))
    )
    return 0 if strict_pass or partial_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
