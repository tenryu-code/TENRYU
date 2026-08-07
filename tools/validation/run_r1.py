#!/usr/bin/env python3
"""Run Stage R1 2D RZ FLD-only validation sweep."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parents[1]
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from marshak_boundary_source_reference import solve_marshak_fld_1d


STAGE_ID = "R1"
MODES = (
    "radial_decoupled",
    "equilibrium_with_leakage",
    "thin_corona",
    "equilibrium_with_leakage_5g",
    "slab_marshak",
    "closed_box_equilibrium",
)

FULL_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    "radial_decoupled": ((64, 128), (128, 256), (256, 512)),
    "equilibrium_with_leakage": ((64, 128), (128, 256)),
    "thin_corona": ((64, 128), (128, 256)),
    "equilibrium_with_leakage_5g": ((64, 128),),
    "slab_marshak": ((64, 128),),
    "closed_box_equilibrium": ((64, 128),),
}
SMOKE_MESHES_BY_MODE: dict[str, tuple[tuple[int, int], ...]] = {
    mode: (meshes[0],) for mode, meshes in FULL_MESHES_BY_MODE.items()
}

NUMBER_RE = r"[-+0-9.eE]+"
FLD_TIMING_RE = re.compile(
    r"\[fld_2d_rz_timing\]\s+outer_iters=(?P<outer>\d+)"
    r"\s+cg_iters_total=(?P<cg_total>-?\d+)"
    r"\s+cg_iters_max=(?P<cg_max>-?\d+)"
    r"\s+early_return=(?P<early>\d+)"
)
CLAMP_RE = re.compile(r"clamp_count=(?P<count>\d+)")
DECK_LINE_RE = re.compile(
    r"\[deck:2d_rz_r1_fld_only\].*?groups=(?P<groups>\d+).*?"
    r"rho_gcc=(?P<rho>{n}).*?kappa_a_cm2_g=(?P<kappa>{n}).*?"
    r"t_end_s=(?P<t_end>{n}).*?cv_e_override=(?P<cv>[^\s]+)".format(n=NUMBER_RE)
)

A_EV = 1.3720e2
C_LIGHT = 2.99792458e10
E_FLOOR = 1.0e-300
MAX_OUTER_ITERATIONS = 50
ITERATION_CAP_FRACTION = 0.8
RADIAL_MATTER_DRIFT_TOL = 1.0e-3
EQUILIBRIUM_ENERGY_TOL = 1.0e-4
EQUILIBRIUM_5G_ENERGY_TOL = 1.0e-3
LTE_RESIDUAL_TOL = 1.0e-3
MARSHAK_SOURCE_ACCOUNTING_TOL = 1.0e-3
MARSHAK_PROFILE_TOL = 0.10
MARSHAK_FLUX_DEFAULT = 1.0e10
MARSHAK_RADIAL_NONUNIFORM_TOL = 0.05
MARSHAK_REFERENCE_STEPS = 20
THIN_CORONA_T_MAX_INIT_EV = 100.0
THIN_CORONA_MAX_PHYSICAL = A_EV * (2.0 * THIN_CORONA_T_MAX_INIT_EV) ** 4


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
    return 5 if mode == "equilibrium_with_leakage_5g" else 1


def mode_rho_default(mode: str) -> float:
    if mode == "thin_corona":
        return 1.0e-7
    if mode in ("equilibrium_with_leakage", "equilibrium_with_leakage_5g", "closed_box_equilibrium"):
        return 1.0e-3
    if mode == "slab_marshak":
        return 1.0
    return 1.0e-4


def mode_kappa_default(mode: str) -> float:
    if mode == "thin_corona":
        return 1.0e-3
    if mode == "closed_box_equilibrium":
        return 1.0e5
    return 1.0


def case_name(mode: str, nr: int, nz: int, rho_gcc: float, kappa_a: float, seed: int) -> str:
    return (
        f"r1_{mode}_nr{nr}_nz{nz}_rho{safe_float_token(rho_gcc)}"
        f"_ka{safe_float_token(kappa_a)}_g{mode_groups(mode)}_seed{seed}"
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
        r_mesh_path = "mesh/x_r" if "mesh/x_r" in handle else "mesh/v_r"
        z_mesh_path = "mesh/x_z" if "mesh/x_z" in handle else "mesh/v_z"
        t_value = float(handle.attrs.get("t", 0.0))
        if "time_state/t" in handle:
            t_value = float(np.asarray(handle["time_state/t"][()]).reshape(-1)[0])
        vol = reshape_cells(handle["hydro/vol"][()], nr, nz, "hydro/vol")
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
            "r_nodes": reshape_nodes(handle[r_mesh_path][()], nr, nz, r_mesh_path),
            "z_nodes": reshape_nodes(handle[z_mesh_path][()], nr, nz, z_mesh_path),
            "E_rad_escaped_state": scalar_dataset(handle, "time_state/E_rad_escaped", None),
            "E_Marshak_in_state": scalar_dataset(handle, "time_state/E_Marshak_in", None),
            "E_numerical_loss_state": scalar_dataset(handle, "time_state/E_numerical_loss", None),
            "E_floor_injected_state": scalar_dataset(handle, "time_state/E_floor_injected", None),
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
        cv = deck_match.group("cv")
        out["deck_cv_e_override"] = None if cv == "None" else float(cv)
    outer = []
    cg_total = []
    cg_max = []
    for match in FLD_TIMING_RE.finditer(text):
        outer.append(int(match.group("outer")))
        cg_total.append(int(match.group("cg_total")))
        cg_max.append(int(match.group("cg_max")))
    if outer:
        out["fld_outer_iterations_series"] = outer
        out["fld_outer_iterations_max"] = max(outer)
        out["fld_cg_iterations_total_max"] = max(cg_total)
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


def compute_interior_mask(rho: Any, kappa_a: float, mesh: Mesh, n_boundary_cells: int = 5) -> Any:
    import numpy as np

    sigma_a = np.asarray(rho, dtype=float) * float(kappa_a)
    optically_thick = sigma_a * mesh.dx_min > 1.0
    far_from_outer_vacuum = np.ones((mesh.nr, mesh.nz), dtype=bool)
    if n_boundary_cells > 0:
        far_from_outer_vacuum[max(mesh.nr - n_boundary_cells, 0) :, :] = False
    return optically_thick & far_from_outer_vacuum


def compute_box_interior_mask(rho: Any, kappa_a: float, mesh: Mesh, n_boundary_cells: int = 5) -> Any:
    import numpy as np

    sigma_a = np.asarray(rho, dtype=float) * float(kappa_a)
    mask = sigma_a * mesh.dx_min > 1.0
    if n_boundary_cells > 0:
        mask[:n_boundary_cells, :] = False
        mask[max(mesh.nr - n_boundary_cells, 0) :, :] = False
        mask[:, :n_boundary_cells] = False
        mask[:, max(mesh.nz - n_boundary_cells, 0) :] = False
    return mask


def compute_lte_residual(rad_E: Any, Te: Any, mask: Any) -> dict[str, Any]:
    import numpy as np

    rad = np.asarray(rad_E, dtype=float)
    lte = A_EV * np.asarray(Te, dtype=float) ** 4
    denom = np.maximum(np.maximum(np.abs(rad), np.abs(lte)), E_FLOOR)
    residual = np.abs(rad - lte) / denom
    masked = residual[np.asarray(mask, dtype=bool)]
    if masked.size == 0:
        return {"lte_residual_max": None, "lte_residual_mean": None, "interior_cell_count": 0}
    return {
        "lte_residual_max": float(np.nanmax(masked)),
        "lte_residual_mean": float(np.nanmean(masked)),
        "interior_cell_count": int(masked.size),
    }


def compute_z_face_area(mesh: Mesh, top: bool) -> float:
    import numpy as np

    r_nodes = np.asarray(mesh.r_nodes, dtype=float)
    j = mesh.nz if top else 0
    r_left = r_nodes[:-1, j]
    r_right = r_nodes[1:, j]
    return float(np.sum(math.pi * np.maximum(r_right * r_right - r_left * r_left, 0.0)))


def volume_weighted_z_profile(field: Any, mesh: Mesh) -> Any:
    import numpy as np

    values = np.asarray(field, dtype=float)
    vol = np.asarray(mesh.vol, dtype=float)
    return np.sum(values * vol, axis=0) / np.maximum(np.sum(vol, axis=0), E_FLOOR)


def z_depth_centers(mesh: Mesh) -> Any:
    import numpy as np

    z_nodes = np.asarray(mesh.z_nodes, dtype=float)
    z_top = z_nodes[0, -1]
    z_centers = 0.5 * (z_nodes[0, :-1] + z_nodes[0, 1:])
    return np.maximum(z_top - z_centers, 0.0)


def relative_l2(profile: Any, reference: Any, weights: Any) -> float:
    import numpy as np

    p = np.asarray(profile, dtype=float)
    r = np.asarray(reference, dtype=float)
    w = np.asarray(weights, dtype=float)
    return float(math.sqrt(np.sum(w * (p - r) ** 2) / max(np.sum(w * r**2), E_FLOOR)))


def centerline_z_profile(field: Any, mesh: Mesh) -> tuple[Any, Any]:
    import numpy as np

    values = np.asarray(field, dtype=float)
    vol = np.asarray(mesh.vol, dtype=float)
    i = mesh.nr // 2
    return values[i, :], vol[i, :]


def radial_nonuniformity_max(field: Any) -> float:
    import numpy as np

    values = np.asarray(field, dtype=float)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)
    return float(np.nanmax(stds / np.maximum(np.abs(means), E_FLOOR)))


def compute_slab_marshak_metrics(
    final: dict[str, Any],
    mesh: Mesh,
    kappa_a: float,
    marshak_in: float,
    cv_e_volume: float | None = None,
) -> dict[str, Any]:
    import numpy as np

    rho_mean = float(np.nanmean(final["rho"]))
    sigma = max(rho_mean * float(kappa_a), 1.0e-300)
    top_area = compute_z_face_area(mesh, top=True)
    flux = MARSHAK_FLUX_DEFAULT
    expected_source = flux * top_area * max(float(final["t"]), 0.0)
    source_rel = abs(marshak_in - expected_source) / max(expected_source, E_FLOOR)
    rad_profile = volume_weighted_z_profile(final["rad_E"], mesh)
    te_profile = volume_weighted_z_profile(final["Te"], mesh)
    top_rad = float(rad_profile[-1])
    bottom_rad = float(rad_profile[0])
    top_te = float(te_profile[-1])
    bottom_te = float(te_profile[0])

    cv_ref = float(cv_e_volume if cv_e_volume is not None else 5.488e2)
    depth = z_depth_centers(mesh)
    L_cm = float(np.nanmax(depth) + 0.5 * np.nanmean(np.asarray(mesh.dz, dtype=float)))
    ref = solve_marshak_fld_1d(
        L_cm=L_cm,
        n_z=mesh.nz,
        sigma_a=sigma,
        rho=rho_mean,
        cv_e_volume=cv_ref,
        F_inc_erg_per_cm2_s=flux,
        t_end_s=max(float(final["t"]), 0.0),
        n_steps=MARSHAK_REFERENCE_STEPS,
        T_init_eV=0.1,
    )
    ref_profile = np.interp(depth, ref["x_centers"], ref["E_r"])
    center_profile, center_weights = centerline_z_profile(final["rad_E"], mesh)
    z_profile_rel_l2 = relative_l2(center_profile, ref_profile, center_weights)
    nonuniformity = radial_nonuniformity_max(final["rad_E"])
    return {
        "marshak_flux_erg_per_cm2_s": flux,
        "marshak_top_area_cm2": top_area,
        "marshak_expected_source_erg": expected_source,
        "marshak_source_accounting_rel": source_rel,
        "marshak_top_rad_E": top_rad,
        "marshak_bottom_rad_E": bottom_rad,
        "marshak_top_to_bottom_rad_E_ratio": top_rad / max(bottom_rad, E_FLOOR),
        "marshak_top_Te_eV": top_te,
        "marshak_bottom_Te_eV": bottom_te,
        "marshak_z_profile_rel_l2": z_profile_rel_l2,
        "marshak_radial_nonuniformity_max": nonuniformity,
        "marshak_reference_scheme": ref["diagnostics"]["scheme"],
        "marshak_reference_n_steps": MARSHAK_REFERENCE_STEPS,
        "marshak_reference_sigma_a_cm_inv": sigma,
        "marshak_reference_cv_e_volume": cv_ref,
    }


def mesh_from_snapshot(snapshot: dict[str, Any], nr: int, nz: int) -> Mesh:
    return Mesh(nr=nr, nz=nz, r_nodes=snapshot["r_nodes"], z_nodes=snapshot["z_nodes"], vol=snapshot["vol"])


def matter_energy(snapshot: dict[str, Any], mesh: Mesh) -> float:
    return compute_U_internal_total(snapshot["rho"], snapshot["ee"], snapshot["ei"], mesh)


def rel_delta(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), E_FLOOR)


def estimate_terminal_vacuum_boundary_energy(rad_E: Any, mesh: Mesh) -> float:
    import numpy as np

    # Fallback only: no flux is exported. This records the radiation energy in
    # the last radial shell adjacent to the only vacuum boundary in the R1 deck.
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
    kappa_a: float,
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
    escaped_series = read_history_scalar(Path(final["path"]).parent, Path(final["path"]).stem.rsplit("_", 1)[0], "energy/radiation_escaped")
    escaped_from_history = float(sum(max(x, 0.0) for x in escaped_series)) if escaped_series else None
    escaped_from_state = final.get("E_rad_escaped_state")
    escaped = escaped_from_state if escaped_from_state is not None else escaped_from_history
    escaped_source = "time_state/E_rad_escaped" if escaped_from_state is not None else "history/sum(energy/radiation_escaped)"
    if escaped is None:
        escaped = estimate_terminal_vacuum_boundary_energy(final["rad_E"], final_mesh)
        escaped_source = "terminal_outer_rad_E_shell"
    marshak_series = read_history_scalar(Path(final["path"]).parent, Path(final["path"]).stem.rsplit("_", 1)[0], "energy/marshak_in")
    marshak_from_history = float(sum(max(x, 0.0) for x in marshak_series)) if marshak_series else None
    marshak_from_state = final.get("E_Marshak_in_state")
    marshak_in = float(marshak_from_state if marshak_from_state is not None else (marshak_from_history or 0.0))

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
        "E_rad_initial": rad0,
        "E_rad_final": rad1,
        "dE_rad_rel": rel_delta(rad0, rad1),
        "E_rad_escaped": float(escaped),
        "E_rad_escaped_source": escaped_source,
        "E_Marshak_in": marshak_in,
        "E_total_initial_matter_rad": u0 + rad0,
        "E_total_final_matter_rad_escape": u1 + rad1 + float(escaped),
        "E_balance_rel": abs((u1 + rad1 + float(escaped)) - (u0 + rad0 + marshak_in))
        / max(max(abs(u0 + rad0), abs(marshak_in)), E_FLOOR),
        "E_terminal_outer_rad_shell": estimate_terminal_vacuum_boundary_energy(final["rad_E"], final_mesh),
        "wall_time_s": wall_time_s,
        "output_bytes": bytes_out,
    }
    metrics.update(log_metrics)
    if mode in ("equilibrium_with_leakage", "equilibrium_with_leakage_5g"):
        mask = compute_interior_mask(final["rho"], kappa_a, final_mesh)
        metrics.update(compute_lte_residual(final["rad_E"], final["Te"], mask))
        metrics["lte_residual_gate_status"] = (
            "skipped_empty_interior_mask"
            if metrics.get("interior_cell_count", 0) <= 0
            else "evaluated"
        )
    if mode == "closed_box_equilibrium":
        mask = compute_box_interior_mask(final["rho"], kappa_a, final_mesh)
        metrics.update(compute_lte_residual(final["rad_E"], final["Te"], mask))
        metrics["lte_residual_gate_status"] = (
            "skipped_empty_interior_mask"
            if metrics.get("interior_cell_count", 0) <= 0
            else "evaluated"
        )
    if mode == "slab_marshak":
        metrics.update(
            compute_slab_marshak_metrics(
                final,
                final_mesh,
                kappa_a,
                marshak_in,
                cv_e_volume=log_metrics.get("deck_cv_e_override"),
            )
        )
    if mode == "thin_corona":
        metrics["thin_corona_rad_E_bound"] = THIN_CORONA_MAX_PHYSICAL
    return metrics


def evaluate_iteration_gate(metrics: dict[str, Any], failed: list[str], *, mode: str) -> None:
    if mode == "slab_marshak":
        metrics["fld_outer_iterations_gate_status"] = "skipped_direct_solver_path"
        return
    outer_max = metrics.get("fld_outer_iterations_max")
    threshold = ITERATION_CAP_FRACTION * MAX_OUTER_ITERATIONS
    metrics["fld_outer_iterations_gate_threshold"] = threshold
    if outer_max is None:
        failed.append("r1_fld_iterations_unavailable")
    elif float(outer_max) >= threshold:
        failed.append("r1_fld_iterations_cap_margin")


def evaluate_gates(mode: str, metrics: dict[str, Any]) -> list[str]:
    failed: list[str] = []
    evaluate_iteration_gate(metrics, failed, mode=mode)
    if metrics.get("rad_E_nan_count", 0) != 0:
        failed.append("r1_rad_E_finite")
    if mode == "radial_decoupled":
        if metrics.get("dU_internal_rel", math.inf) > RADIAL_MATTER_DRIFT_TOL:
            failed.append("r1_radial_matter_frozen")
        metrics["radial_reference_gate_status"] = "pending_or_skipped"
    elif mode == "equilibrium_with_leakage":
        if metrics.get("E_balance_rel", math.inf) > EQUILIBRIUM_ENERGY_TOL:
            failed.append("r1_equilibrium_energy_balance")
        if metrics.get("interior_cell_count", 0) > 0 and metrics.get("lte_residual_max", math.inf) > LTE_RESIDUAL_TOL:
            failed.append("r1_equilibrium_lte_residual")
    elif mode == "equilibrium_with_leakage_5g":
        if metrics.get("E_balance_rel", math.inf) > EQUILIBRIUM_5G_ENERGY_TOL:
            failed.append("r1_equilibrium_5g_energy_balance")
        if metrics.get("interior_cell_count", 0) > 0 and metrics.get("lte_residual_max", math.inf) > LTE_RESIDUAL_TOL:
            failed.append("r1_equilibrium_5g_lte_residual")
    elif mode == "slab_marshak":
        if metrics.get("marshak_source_accounting_rel", math.inf) > MARSHAK_SOURCE_ACCOUNTING_TOL:
            failed.append("r1_slab_marshak_source_accounting")
        if metrics.get("marshak_top_rad_E", 0.0) <= metrics.get("marshak_bottom_rad_E", math.inf):
            failed.append("r1_slab_marshak_top_heating")
        if metrics.get("marshak_z_profile_rel_l2", math.inf) > MARSHAK_PROFILE_TOL:
            failed.append("r1_slab_marshak_z_profile_l2")
        if metrics.get("marshak_radial_nonuniformity_max", math.inf) > MARSHAK_RADIAL_NONUNIFORM_TOL:
            failed.append("r1_slab_marshak_radial_nonuniform")
    elif mode == "closed_box_equilibrium":
        if metrics.get("E_balance_rel", math.inf) > LTE_RESIDUAL_TOL:
            failed.append("r1_closed_box_energy_conservation")
        if metrics.get("interior_cell_count", 0) <= 0:
            failed.append("r1_closed_box_interior_lte_coverage")
        elif metrics.get("lte_residual_max", math.inf) > LTE_RESIDUAL_TOL:
            failed.append("r1_closed_box_lte_residual")
    elif mode == "thin_corona":
        if metrics.get("rad_E_negative_count", 0) != 0:
            failed.append("r1_thin_corona_nonnegative")
        if metrics.get("rad_E_max", math.inf) > THIN_CORONA_MAX_PHYSICAL:
            failed.append("r1_thin_corona_rad_E_bound")
    return failed


def active_env(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int, outdir: Path) -> dict[str, str]:
    env = {
        "TENRYU_R1_MODE": mode,
        "TENRYU_R1_NR": str(nr),
        "TENRYU_R1_NZ": str(nz),
        "TENRYU_R1_SEED": str(seed),
        "TENRYU_R1_OUTDIR": str(outdir),
        "TENRYU_R1_MAX_STEPS": str(args.max_steps),
        "TENRYU_R1_VERBOSITY": args.verbosity,
    }
    if args.t_end_s is not None:
        env["TENRYU_R1_T_END_S"] = str(args.t_end_s)
    if args.dt_initial_s is not None:
        env["TENRYU_R1_DT_INITIAL_S"] = str(args.dt_initial_s)
    if args.dt_max_s is not None:
        env["TENRYU_R1_DT_MAX_S"] = str(args.dt_max_s)
    if args.rho_gcc is not None:
        env["TENRYU_R1_RHO_GCC"] = str(args.rho_gcc)
    if args.kappa_a_cm2_g is not None:
        env["TENRYU_R1_KAPPA_A_CM2_G"] = str(args.kappa_a_cm2_g)
    return env


def effective_rho(args: argparse.Namespace, mode: str) -> float:
    return float(args.rho_gcc if args.rho_gcc is not None else mode_rho_default(mode))


def effective_kappa(args: argparse.Namespace, mode: str) -> float:
    return float(args.kappa_a_cm2_g if args.kappa_a_cm2_g is not None else mode_kappa_default(mode))


def run_one(args: argparse.Namespace, mode: str, nr: int, nz: int, seed: int) -> dict[str, Any]:
    rho_gcc = effective_rho(args, mode)
    kappa_a = effective_kappa(args, mode)
    groups = mode_groups(mode)
    run_id = case_name(mode, nr, nz, rho_gcc, kappa_a, seed)
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
            metrics = common_metrics(mode, nr, nz, groups, snapshots, log_metrics, wall_time_s, bytes_out, kappa_a)
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
        "E_Marshak_in": metrics.get("E_Marshak_in"),
        "lte_residual_max": metrics.get("lte_residual_max"),
        "rad_E_min": metrics.get("rad_E_min"),
        "rad_E_max": metrics.get("rad_E_max"),
        "marshak_source_accounting_rel": metrics.get("marshak_source_accounting_rel"),
        "marshak_z_profile_rel_l2": metrics.get("marshak_z_profile_rel_l2"),
        "marshak_radial_nonuniformity_max": metrics.get("marshak_radial_nonuniformity_max"),
        "dU_internal_rel": metrics.get("dU_internal_rel"),
        "fld_outer_iterations_max": metrics.get("fld_outer_iterations_max"),
        "fld_outer_iterations_gate_status": metrics.get("fld_outer_iterations_gate_status"),
        "n_clamp": metrics.get("n_clamp"),
    }


def load_metrics_payload(metrics_ref: str) -> dict[str, Any]:
    return json.loads(Path(metrics_ref).read_text(encoding="utf-8"))


def downsample_cell_average(fine: Any, fine_mesh: Mesh, nr: int, nz: int) -> Any:
    import numpy as np

    field = np.asarray(fine, dtype=float)
    vol = np.asarray(fine_mesh.vol, dtype=float)
    fr = fine_mesh.nr // nr
    fz = fine_mesh.nz // nz
    if fine_mesh.nr % nr != 0 or fine_mesh.nz % nz != 0:
        raise ValueError("reference grid must be an integer multiple of target grid")
    weighted = field * vol
    weighted_blocks = weighted.reshape(nr, fr, nz, fz).sum(axis=(1, 3))
    vol_blocks = vol.reshape(nr, fr, nz, fz).sum(axis=(1, 3))
    return weighted_blocks / np.maximum(vol_blocks, E_FLOOR)


def radial_reference_errors(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    import numpy as np

    radial_rows = [row for row in rows if row["mode"] == "radial_decoupled" and row["seed"] == args.seed_list[0]]
    by_mesh = {(row["nr"], row["nz"]): row for row in radial_rows if row["exit_code"] == 0}
    if (256, 512) not in by_mesh or (128, 256) not in by_mesh or (64, 128) not in by_mesh:
        return None
    ref_row = by_mesh[(256, 512)]
    ref_payload = load_metrics_payload(ref_row["metrics_ref"])
    ref_run_id = ref_payload["run_id"]
    ref_outdir = Path(args.out_root) / ref_run_id
    ref_results = discover_results_dir(ref_outdir)
    ref_snap = read_snapshots(ref_results, ref_run_id, 256, 512, 1)[-1]
    ref_mesh = mesh_from_snapshot(ref_snap, 256, 512)
    errors: dict[str, float] = {}
    for mesh in ((64, 128), (128, 256)):
        row = by_mesh[mesh]
        payload = load_metrics_payload(row["metrics_ref"])
        run_id = payload["run_id"]
        results_dir = discover_results_dir(Path(args.out_root) / run_id)
        snap = read_snapshots(results_dir, run_id, mesh[0], mesh[1], 1)[-1]
        ref_on_target = downsample_cell_average(ref_snap["rad_E"], ref_mesh, mesh[0], mesh[1])
        denom = np.maximum(np.maximum(np.abs(ref_on_target), np.abs(snap["rad_E"])), E_FLOOR)
        errors[f"nr{mesh[0]}_nz{mesh[1]}"] = float(np.nanmax(np.abs(snap["rad_E"] - ref_on_target) / denom))
    err64 = errors["nr64_nz128"]
    err128 = errors["nr128_nz256"]
    improvement = err64 / max(err128, E_FLOOR)
    return {
        "radial_reference_run_id": ref_run_id,
        "radial_reference_errors": errors,
        "radial_reference_improvement_64_to_128": improvement,
        "radial_reference_gate_pass": bool(err128 < err64),
    }


def apply_radial_reference_gate(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    summary = radial_reference_errors(args, rows)
    if summary is None:
        for row in rows:
            if row["mode"] == "radial_decoupled":
                payload = load_metrics_payload(row["metrics_ref"])
                payload["metrics"]["radial_reference_gate_status"] = "skipped_missing_64_128_256_reference_set"
                write_json(Path(row["metrics_ref"]), payload)
        return
    for row in rows:
        if row["mode"] != "radial_decoupled":
            continue
        payload = load_metrics_payload(row["metrics_ref"])
        payload["metrics"].update(summary)
        payload["metrics"]["radial_reference_gate_status"] = "pass" if summary["radial_reference_gate_pass"] else "fail"
        if not summary["radial_reference_gate_pass"] and "r1_radial_reference_refinement" not in payload["failed_gates"]:
            payload["failed_gates"].append("r1_radial_reference_refinement")
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
        "lte_residual_max",
        "rad_E_max",
        "marshak_source_accounting_rel",
        "marshak_z_profile_rel_l2",
        "marshak_radial_nonuniformity_max",
        "dU_internal_rel",
        "fld_outer_iterations_max",
        "n_clamp",
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
    print("Stage R1 FLD-only sweep summary")
    print("ladder_row seed result E_balance_rel lte_residual_max marshak_source_accounting_rel dU_internal_rel fld_outer_iterations_max failed_gates")
    for row in rows:
        print(
            "{ladder_row} {seed} {result} {E_balance_rel} {lte_residual_max} {marshak_source_accounting_rel} "
            "{dU_internal_rel} {fld_outer_iterations_max} {failed_gates}".format(**row)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tenryu-bin", default="./build/tenryu")
    parser.add_argument("--deck", default="examples/verification/2d_rz_r1_fld_only.py")
    parser.add_argument("--root", default="docs/validation/2d_rz/")
    parser.add_argument("--out-root", default="./build/r1_runs")
    parser.add_argument("--modes", "--mode-list", dest="mode_list", type=lambda v: parse_csv(v, MODES, "mode"), default=list(MODES))
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
    parser.add_argument("--summary-json", default="docs/validation/2d_rz/R1/stats.json")
    parser.add_argument("--runs-jsonl", default="docs/validation/2d_rz/R1/runs.jsonl")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows_spec = matrix_rows(args)
    rows = [run_one(args, mode, nr, nz, seed) for mode, nr, nz, seed in rows_spec]
    apply_radial_reference_gate(args, rows)
    print_summary(rows)
    write_summary(args, rows)
    append_jsonl(Path(args.runs_jsonl), rows)
    update_stage_json(args, rows)
    return 0 if rows and all(row["result"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
