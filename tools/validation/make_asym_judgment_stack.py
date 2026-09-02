#!/usr/bin/env python3
"""Render the zonal-asymmetry validation results as one nine-page PDF."""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm


STRICT_R0_UM = 30.0
LINEARITY_R0_UM = 40.0
LINEARITY_RMS_GATE = 0.03
EPS_NEUTRALITY_FLOOR = 5.0e-5
EPS_NEUTRALITY_FRACTION = 0.01
H2_RELATIVE_GATE = 0.05
FRONT_N_SECTORS = 96
FRONT_RHO_JUMP_FACTOR = 2.0
FRONT_EXAGGERATION = 10.0

P2_ARMS = (
    ("p2m05", -0.005),
    ("p2p05", 0.005),
    ("p2p1", 0.01),
    ("p2p2", 0.02),
)
ARM_ORDER = (
    "a0",
    "p2m05",
    "p2p05",
    "p2p1",
    "p2p2",
    "p4p1",
    "cplx",
)
ARM_LABELS = {
    "a0": "a0",
    "p2m05": "P2 -0.5%",
    "p2p05": "P2 +0.5%",
    "p2p1": "P2 +1%",
    "p2p2": "P2 +2%",
    "p4p1": "P4 +1%",
    "cplx": "complex",
}


@dataclass(frozen=True)
class Snapshot:
    """Cell-centered density data read from one TENRYU snapshot."""

    path: Path
    time: float
    r_cell: np.ndarray
    z_cell: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray | None
    cell_polygons: tuple[np.ndarray, ...] | None
    shell_columns: int | None = None


def _scalar_time(handle: h5py.File) -> float:
    if "time_state/t" in handle:
        values = np.asarray(handle["time_state/t"][()]).reshape(-1)
        if values.size:
            return float(values[0])
    return float(handle.attrs.get("t", 0.0))


def _cell_nverts(handle: h5py.File, n_cells: int) -> np.ndarray | None:
    for name in (
        "mesh/topology/v4/cell_nverts",
        "mesh/topology/v1/cell_nverts",
    ):
        if name in handle:
            values = np.asarray(handle[name][()], dtype=np.int64).reshape(-1)
            if values.size != n_cells:
                raise ValueError(
                    f"{name}: expected {n_cells} entries, got {values.size}"
                )
            return values
    return None


def _csr_cell_geometry(
    handle: h5py.File,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    n_cells: int,
    read_polygons: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...] | None,
] | None:
    topology_group = None
    for candidate in ("mesh/topology/v3", "mesh/topology/v2"):
        offsets_name = f"{candidate}/cell_node_csr_offsets"
        indices_name = f"{candidate}/cell_node_csr_indices"
        if offsets_name in handle or indices_name in handle:
            if offsets_name not in handle or indices_name not in handle:
                raise KeyError(
                    f"incomplete cell-node connectivity under {candidate}"
                )
            topology_group = candidate
            break
    if topology_group is None:
        return None

    offsets = np.asarray(
        handle[f"{topology_group}/cell_node_csr_offsets"][()],
        dtype=np.int64,
    ).reshape(-1)
    indices = np.asarray(
        handle[f"{topology_group}/cell_node_csr_indices"][()],
        dtype=np.int64,
    ).reshape(-1)
    if offsets.size != n_cells + 1:
        raise ValueError(
            f"{topology_group}/cell_node_csr_offsets: expected "
            f"{n_cells + 1} entries, got {offsets.size}"
        )
    if (
        offsets[0] != 0
        or np.any(np.diff(offsets) < 0)
        or offsets[-1] > indices.size
    ):
        raise ValueError(f"invalid cell-node CSR offsets under {topology_group}")

    nverts = _cell_nverts(handle, n_cells)
    r_cell = np.full(n_cells, np.nan, dtype=float)
    z_cell = np.full(n_cells, np.nan, dtype=float)
    polygons: list[np.ndarray] | None = None
    if read_polygons:
        polygons = [np.empty((0, 2), dtype=float) for _ in range(n_cells)]
    for cell in range(n_cells):
        begin = int(offsets[cell])
        end = int(offsets[cell + 1])
        cell_nodes = indices[begin:end]
        if nverts is not None:
            count = int(nverts[cell])
            if count == 0:
                continue
            if count < 3 or count > 8 or count > cell_nodes.size:
                raise ValueError(
                    f"cell {cell}: invalid active vertex count {count}"
                )
            cell_nodes = cell_nodes[:count]
        else:
            cell_nodes = cell_nodes[cell_nodes >= 0]
        if cell_nodes.size < 3:
            raise ValueError(f"cell {cell}: fewer than three connected nodes")
        if cell_nodes.size > 8:
            raise ValueError(f"cell {cell}: more than eight connected nodes")
        if np.any(cell_nodes < 0) or np.any(cell_nodes >= r_nodes.size):
            raise ValueError(f"cell {cell}: node index out of range")
        r_cell[cell] = float(np.mean(r_nodes[cell_nodes]))
        z_cell[cell] = float(np.mean(z_nodes[cell_nodes]))
        if polygons is not None:
            polygons[cell] = np.column_stack(
                (r_nodes[cell_nodes], z_nodes[cell_nodes])
            )
    return r_cell, z_cell, tuple(polygons) if polygons is not None else None


def _structured_cell_centroids(
    handle: h5py.File,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray]:
    nr = int(handle.attrs.get("nr", 0))
    nz = int(handle.attrs.get("nz", 0))
    if r_nodes.ndim == 2 and z_nodes.ndim == 2 and r_nodes.shape == z_nodes.shape:
        node_shape = r_nodes.shape
        if nr == 0:
            nr = node_shape[0] - 1
        if nz == 0:
            nz = node_shape[1] - 1
    else:
        node_shape = (nr + 1, nz + 1)
        if nr <= 0 or nz <= 0 or r_nodes.size != node_shape[0] * node_shape[1]:
            raise ValueError("cannot infer structured node-array dimensions")
        if z_nodes.size != r_nodes.size:
            raise ValueError("mesh/x_r and mesh/x_z sizes differ")
        r_nodes = r_nodes.reshape(node_shape)
        z_nodes = z_nodes.reshape(node_shape)
    if node_shape != (nr + 1, nz + 1):
        raise ValueError(
            f"structured node shape {node_shape} disagrees with nr={nr}, nz={nz}"
        )
    if n_cells != nr * nz:
        raise ValueError(f"hydro/rho has {n_cells} cells, expected {nr * nz}")

    r_cell = 0.25 * (
        r_nodes[:-1, :-1]
        + r_nodes[1:, :-1]
        + r_nodes[1:, 1:]
        + r_nodes[:-1, 1:]
    )
    z_cell = 0.25 * (
        z_nodes[:-1, :-1]
        + z_nodes[1:, :-1]
        + z_nodes[1:, 1:]
        + z_nodes[:-1, 1:]
    )

    nverts = _cell_nverts(handle, n_cells)
    if nverts is not None:
        r_flat = r_cell.reshape(-1)
        z_flat = z_cell.reshape(-1)
        for cell, count_value in enumerate(nverts):
            count = int(count_value)
            if count == 0:
                r_flat[cell] = np.nan
                z_flat[cell] = np.nan
            elif count == 3:
                i, j = divmod(cell, nz)
                r_flat[cell] = float(
                    np.mean(
                        (
                            r_nodes[i, j],
                            r_nodes[i + 1, j],
                            r_nodes[i + 1, j + 1],
                        )
                    )
                )
                z_flat[cell] = float(
                    np.mean(
                        (
                            z_nodes[i, j],
                            z_nodes[i + 1, j],
                            z_nodes[i + 1, j + 1],
                        )
                    )
                )
    return r_cell.reshape(-1), z_cell.reshape(-1)


def read_snapshot(
    path: Path,
    *,
    read_polygons: bool = False,
    read_pressure: bool = False,
) -> Snapshot:
    """Read cell density and centroids using zonal_front_modes.py conventions."""

    with h5py.File(path, "r") as handle:
        for name in ("mesh/x_r", "mesh/x_z", "hydro/rho"):
            if name not in handle:
                raise KeyError(f"{path}: required HDF5 dataset missing: {name}")
        r_nodes = np.asarray(handle["mesh/x_r"][()], dtype=float)
        z_nodes = np.asarray(handle["mesh/x_z"][()], dtype=float)
        rho = np.asarray(handle["hydro/rho"][()], dtype=float).reshape(-1)
        pressure = None
        if read_pressure:
            for name in ("hydro/Pe", "hydro/Pi"):
                if name not in handle:
                    raise KeyError(f"{path}: required HDF5 dataset missing: {name}")
            pe = np.asarray(handle["hydro/Pe"][()], dtype=float).reshape(-1)
            pi = np.asarray(handle["hydro/Pi"][()], dtype=float).reshape(-1)
            if pe.size != rho.size or pi.size != rho.size:
                raise ValueError(f"{path}: pressure and density dataset sizes differ")
            pressure = pe + pi
        if r_nodes.size != z_nodes.size:
            raise ValueError(f"{path}: mesh/x_r and mesh/x_z sizes differ")
        shell_columns = None
        role_name = "mesh/topology/v3/block_role"
        ncol_name = "mesh/topology/v3/block_n_j_cells"
        if role_name in handle and ncol_name in handle:
            roles = np.asarray(handle[role_name][()]).reshape(-1)
            ncols = np.asarray(handle[ncol_name][()]).reshape(-1)
            shell = np.flatnonzero(roles == 5)
            if shell.size:
                shell_columns = int(ncols[shell[0]])
        csr_geometry = _csr_cell_geometry(
            handle,
            r_nodes.reshape(-1),
            z_nodes.reshape(-1),
            rho.size,
            read_polygons,
        )
        if csr_geometry is None:
            if read_polygons:
                raise KeyError(f"{path}: cell-node CSR connectivity is required")
            r_cell, z_cell = _structured_cell_centroids(
                handle, r_nodes, z_nodes, rho.size
            )
            cell_polygons = None
        else:
            r_cell, z_cell, cell_polygons = csr_geometry
        time = _scalar_time(handle)

    valid = np.isfinite(r_cell) & np.isfinite(z_cell) & np.isfinite(rho)
    if cell_polygons is not None:
        valid_indices = np.flatnonzero(valid)
        cell_polygons = tuple(cell_polygons[index] for index in valid_indices)
    return Snapshot(
        path,
        time,
        r_cell[valid],
        z_cell[valid],
        rho[valid],
        pressure[valid] if pressure is not None else None,
        cell_polygons,
        shell_columns,
    )


def _snapshot_paths(run_dir: Path) -> list[Path]:
    suffix = re.compile(r"_(\d{4,})$")
    indexed: list[tuple[int, str, Path]] = []
    if not run_dir.is_dir():
        return []
    for path in run_dir.glob("*.h5"):
        if path.name.endswith("_history.h5"):
            continue
        match = suffix.search(path.stem)
        if match is not None:
            indexed.append((int(match.group(1)), path.name, path))
    indexed.sort()
    return [item[2] for item in indexed]


def _mode_path(modes_root: Path, family: str, arm: str) -> Path:
    prefixes = {"off": "arm_", "pc": "armpc_", "h2": "armh2_"}
    return modes_root / f"{prefixes[family]}{arm}.json"


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _float_series(values: Any) -> np.ndarray:
    if not isinstance(values, list):
        raise ValueError("expected a JSON array")
    return np.asarray(
        [float(value) if value is not None else np.nan for value in values],
        dtype=float,
    )


def _payload_series(payload: dict[str, Any], key: str) -> np.ndarray:
    if key not in payload:
        raise KeyError(f"missing JSON field: {key}")
    return _float_series(payload[key])


def _epsilon(payload: dict[str, Any], degree: int) -> np.ndarray:
    if "b_over_b0" not in payload:
        raise KeyError("missing JSON field: b_over_b0")
    normalized = payload["b_over_b0"]
    if isinstance(normalized, list):
        if degree >= len(normalized):
            raise KeyError(f"b_over_b0 does not contain l={degree}")
        return _float_series(normalized[degree])
    if isinstance(normalized, dict):
        for key in (str(degree), f"l{degree}", f"l={degree}"):
            if key in normalized:
                return _float_series(normalized[key])
    raise KeyError(f"b_over_b0 does not contain l={degree}")


def _paired_series(
    payload: dict[str, Any], values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    time = _payload_series(payload, "time")
    count = min(time.size, values.size)
    return time[:count], values[:count]


def _radius_um(payload: dict[str, Any]) -> np.ndarray:
    return 1.0e4 * _payload_series(payload, "s_f_median")


def _h_f_um(payload: dict[str, Any]) -> np.ndarray:
    return 1.0e4 * _payload_series(payload, "h_f_median")


def _interp(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> np.ndarray:
    count = min(x.size, y.size)
    finite = np.isfinite(x[:count]) & np.isfinite(y[:count])
    if np.count_nonzero(finite) < 2:
        return np.full(query.shape, np.nan, dtype=float)
    x_valid = x[:count][finite]
    y_valid = y[:count][finite]
    order = np.argsort(x_valid, kind="stable")
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    unique_x, first, counts = np.unique(
        x_valid, return_index=True, return_counts=True
    )
    if unique_x.size < 2:
        return np.full(query.shape, np.nan, dtype=float)
    if np.any(counts > 1):
        sums = np.add.reduceat(y_valid, first)
        y_valid = sums / counts
    else:
        y_valid = y_valid[first]
    result = np.interp(query, unique_x, y_valid)
    outside = (query < unique_x[0]) | (query > unique_x[-1])
    result[outside] = np.nan
    return result


def _epsilon_at_radius(
    payload: dict[str, Any], degree: int, radius_um: np.ndarray
) -> np.ndarray:
    return _interp(_radius_um(payload), _epsilon(payload, degree), radius_um)


def _title(page: str, verdict: str) -> str:
    return f"{page} - {verdict}" if verdict else page


def _new_figure(title: str, verdict: str, **kwargs: Any) -> plt.Figure:
    fig = plt.figure(**kwargs)
    fig.suptitle(_title(title, verdict), fontsize=14, fontweight="bold")
    return fig


def _placeholder_figure(
    page_number: int, page_title: str, verdict: str, message: str
) -> plt.Figure:
    fig = _new_figure(
        f"Page {page_number}: {page_title}", verdict, figsize=(11.0, 8.5)
    )
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        f"INPUT UNAVAILABLE\n\n{message}",
        ha="center",
        va="center",
        fontsize=14,
        color="0.35",
        wrap=True,
    )
    return fig


def _missing_axis(ax: plt.Axes, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color="0.4", wrap=True)


def _arm_snapshot_paths(
    arm_root: Path, map_arm: str
) -> tuple[Path, list[Path]]:
    run_dir = arm_root / map_arm
    snapshot_dir = run_dir / "results"
    paths = _snapshot_paths(snapshot_dir)
    if not paths:
        snapshot_dir = run_dir
        paths = _snapshot_paths(snapshot_dir)
    return snapshot_dir, paths


def _page_rho_maps(
    arm_root: Path, map_arm: str, verdict: str, page_label: str = "Page 1"
) -> plt.Figure:
    snapshot_dir, paths = _arm_snapshot_paths(arm_root, map_arm)
    if len(paths) < 8:
        return _placeholder_figure(
            1,
            f"{map_arm} density maps",
            verdict,
            f"Eight indexed snapshots are required; found {len(paths)} in "
            f"{snapshot_dir}",
        )
    chosen_indices = np.linspace(0, len(paths) - 1, 8).round().astype(int)
    snapshots = [
        read_snapshot(paths[index], read_polygons=True) for index in chosen_indices
    ]
    positive = np.concatenate(
        [snapshot.rho[snapshot.rho > 0.0] for snapshot in snapshots]
    )
    if positive.size == 0:
        return _placeholder_figure(
            1,
            f"{map_arm} density maps",
            verdict,
            "Snapshots contain no positive density",
        )
    vmin, vmax = np.percentile(positive, (1.0, 99.0))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin <= 0.0:
        vmin = float(np.min(positive))
        vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = np.nextafter(vmin, np.inf)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(2, 4, figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title(f"{page_label}: {map_arm} density maps", verdict),
        fontsize=14,
        fontweight="bold",
    )
    mappable = None
    for ax, snapshot in zip(axes.flat, snapshots, strict=True):
        if snapshot.cell_polygons is None:
            raise ValueError(f"{snapshot.path}: cell polygons were not loaded")
        positive_mask = snapshot.rho > 0.0
        polygons_um = [
            1.0e4 * polygon
            for polygon, include in zip(
                snapshot.cell_polygons, positive_mask, strict=True
            )
            if include
        ]
        mappable = PolyCollection(
            polygons_um,
            array=snapshot.rho[positive_mask],
            edgecolors="face",
            linewidths=0.4,
            antialiased=True,
            cmap="viridis",
            norm=norm,
            rasterized=True,
        )
        ax.add_collection(mappable, autolim=True)
        ax.autoscale_view()
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {snapshot.time * 1.0e12:.3f} ps", fontsize=9)
        ax.set_xlabel(r"$r$ [$\mu$m]")
        ax.set_ylabel(r"$z$ [$\mu$m]")
    if mappable is not None:
        fig.colorbar(mappable, ax=axes.ravel().tolist(), label=r"$\rho$ [g cm$^{-3}$]")
    return fig


def _running_mean_three(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.full(3, 1.0 / 3.0), mode="valid")


def _jump_front_radius(
    radius: np.ndarray,
    rho: np.ndarray,
    threshold: float,
) -> float:
    finite = np.isfinite(radius) & np.isfinite(rho)
    radius = radius[finite]
    rho = rho[finite]
    if radius.size < 2:
        return float("nan")
    order = np.argsort(radius, kind="stable")
    radius = radius[order]
    rho_smooth = _running_mean_three(rho[order])
    crossings = np.flatnonzero(rho_smooth >= threshold)
    if crossings.size == 0 or crossings[0] == 0:
        return float("nan")
    upper = int(crossings[0])
    lower = upper - 1
    fraction = (
        (threshold - rho_smooth[lower])
        / (rho_smooth[upper] - rho_smooth[lower])
    )
    return float(radius[lower] + fraction * (radius[upper] - radius[lower]))


def _front_shape_diagnostics(
    snapshot: Snapshot,
    previous_front_median: float,
    first_snapshot: bool,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    radius = np.sqrt(
        snapshot.r_cell * snapshot.r_cell
        + snapshot.z_cell * snapshot.z_cell
    )
    theta = np.clip(
        np.arctan2(snapshot.r_cell, snapshot.z_cell), 0.0, np.pi
    )
    if radius.size == 0:
        raise ValueError(f"{snapshot.path}: no finite cells")

    if first_snapshot:
        s_ref = float(np.percentile(radius, 90.0))
    else:
        s_ref = previous_front_median
    upstream_mask = radius < 0.5 * s_ref
    if np.count_nonzero(upstream_mask) >= 50:
        rho_up = float(np.median(snapshot.rho[upstream_mask]))
    else:
        rho_up = float(np.percentile(snapshot.rho, 10.0))

    # Cap the sector count at the shell's angular column count so every
    # sector holds full radial columns: finer sectoring leaves sparse
    # sectors where the 3-point running mean bleeds the shell density
    # across a radial gap and fakes an interior crossing (measured on
    # 48-column mini runs against the 96-sector default).
    n_sectors = FRONT_N_SECTORS
    if snapshot.shell_columns:
        n_sectors = int(min(FRONT_N_SECTORS, max(8, snapshot.shell_columns)))
    sector_edges = np.linspace(0.0, np.pi, n_sectors + 1)
    sector_centers = 0.5 * (sector_edges[:-1] + sector_edges[1:])
    sector = np.floor(theta / (np.pi / n_sectors)).astype(np.int64)
    sector = np.clip(sector, 0, n_sectors - 1)
    front = np.full(n_sectors, np.nan, dtype=float)
    threshold = FRONT_RHO_JUMP_FACTOR * rho_up
    for k in range(n_sectors):
        in_sector = sector == k
        front[k] = _jump_front_radius(
            radius[in_sector], snapshot.rho[in_sector], threshold
        )

    valid_front = front[np.isfinite(front)]
    front_median = (
        float(np.median(valid_front)) if valid_front.size else float("nan")
    )
    sector_mu = np.cos(sector_centers)
    pole_sectors = np.abs(sector_mu) > 0.9
    equator_sectors = np.abs(sector_mu) < 0.2
    pole_front = front[pole_sectors]
    equator_front = front[equator_sectors]
    pole_front = pole_front[np.isfinite(pole_front)]
    equator_front = equator_front[np.isfinite(equator_front)]
    front_difference_um = (
        1.0e4 * (float(np.mean(pole_front)) - float(np.mean(equator_front)))
        if pole_front.size and equator_front.size
        else float("nan")
    )

    pressure_ratio = float("nan")
    if snapshot.pressure is not None:
        cell_mu = np.cos(theta)
        outer_layer = radius > 0.985 * float(np.max(radius))
        finite_pressure = np.isfinite(snapshot.pressure)
        pole_cells = outer_layer & finite_pressure & (np.abs(cell_mu) > 0.9)
        equator_cells = outer_layer & finite_pressure & (np.abs(cell_mu) < 0.2)
        if np.any(pole_cells) and np.any(equator_cells):
            pole_pressure = float(np.mean(snapshot.pressure[pole_cells]))
            equator_pressure = float(np.mean(snapshot.pressure[equator_cells]))
            if equator_pressure != 0.0:
                pressure_ratio = pole_pressure / equator_pressure
    return (
        sector_centers,
        front,
        front_median,
        front_difference_um,
        pressure_ratio,
    )


def _page_front_shape_visibility(
    arm_root: Path,
    map_arm: str,
    analytic_ratio: float,
    analytic_label: str,
    verdict: str,
    page_label: str = "Page 2",
) -> plt.Figure:
    snapshot_dir, paths = _arm_snapshot_paths(arm_root, map_arm)
    if len(paths) < 4:
        return _placeholder_figure(
            2,
            f"front-shape visibility ({map_arm})",
            verdict,
            f"Four indexed snapshots are required; found {len(paths)} in "
            f"{snapshot_dir}",
        )

    chosen_indices = set(
        np.linspace(0, len(paths) - 1, 4).round().astype(int).tolist()
    )
    times: list[float] = []
    front_differences_um: list[float] = []
    pressure_ratios: list[float] = []
    overlay_data: list[tuple[float, np.ndarray, np.ndarray, float]] = []
    previous_front_median = float("nan")
    for index, path in enumerate(paths):
        snapshot = read_snapshot(path, read_pressure=True)
        (
            sector_centers,
            front,
            front_median,
            front_difference_um,
            pressure_ratio,
        ) = _front_shape_diagnostics(
            snapshot, previous_front_median, first_snapshot=index == 0
        )
        previous_front_median = front_median
        times.append(snapshot.time)
        front_differences_um.append(front_difference_um)
        pressure_ratios.append(pressure_ratio)
        if index in chosen_indices:
            exaggerated_front_um = 1.0e4 * front_median * (
                1.0
                + FRONT_EXAGGERATION
                * (front / front_median - 1.0)
            )
            overlay_data.append(
                (
                    snapshot.time,
                    sector_centers.copy(),
                    exaggerated_front_um,
                    1.0e4 * front_median,
                )
            )

    fig = plt.figure(figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title(f"{page_label}: front-shape visibility ({map_arm})", verdict),
        fontsize=14,
        fontweight="bold",
    )
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0))
    polar_ax = fig.add_subplot(grid[:, 0], projection="polar")
    radius_ax = fig.add_subplot(grid[0, 1])
    pressure_ax = fig.add_subplot(grid[1, 1])

    colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(overlay_data)))
    circle_theta = np.linspace(0.0, np.pi, 361)
    for color, (time, theta, front_um, reference_um) in zip(
        colors, overlay_data, strict=True
    ):
        polar_ax.plot(
            theta,
            front_um,
            color=color,
            linewidth=1.8,
            label=f"t = {time * 1.0e12:.3f} ps",
        )
        polar_ax.plot(
            circle_theta,
            np.full(circle_theta.shape, reference_um),
            color=color,
            linestyle="--",
            linewidth=1.0,
            alpha=0.75,
        )
    polar_ax.set_theta_zero_location("N")
    polar_ax.set_theta_direction(-1)
    polar_ax.set_thetamin(0.0)
    polar_ax.set_thetamax(180.0)
    polar_ax.set_title(r"Exaggerated front $r_f(\theta)$", pad=18)
    polar_ax.set_ylabel(r"radius [$\mu$m]", labelpad=28)
    polar_ax.grid(True, alpha=0.3)
    polar_ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=8)
    polar_ax.text(
        0.5,
        1.08,
        f"ASPHERICITY x{FRONT_EXAGGERATION:g}",
        transform=polar_ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="tab:red",
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "tab:red"},
    )

    time_ps = 1.0e12 * np.asarray(times)
    radius_ax.plot(time_ps, front_differences_um, linewidth=1.6)
    radius_ax.axhline(0.0, color="0.5", linewidth=0.8)
    radius_ax.set_xlabel("time [ps]")
    radius_ax.set_ylabel(r"$R_{pole}-R_{equator}$ [$\mu$m]")
    radius_ax.grid(True, alpha=0.3)

    pressure_ax.plot(
        time_ps,
        pressure_ratios,
        linewidth=1.6,
        label="outer shocked layer",
    )
    if np.isfinite(analytic_ratio):
        pressure_ax.axhline(
            analytic_ratio,
            color="tab:red",
            linestyle="--",
            linewidth=1.2,
            label=analytic_label,
        )
    pressure_ax.set_xlabel("time [ps]")
    pressure_ax.set_ylabel(r"$(P_e+P_i)_{pole}/(P_e+P_i)_{equator}$")
    pressure_ax.grid(True, alpha=0.3)
    pressure_ax.legend(fontsize=8)
    return fig


def _page_mesh_structure(
    arm_root: Path,
    map_arm: str,
    verdict: str,
    page_label: str,
    zoom_late: bool,
) -> plt.Figure:
    """Cell-edge wireframe of the mesh at the same times as the density maps.

    zoom_late=False: the 8 linspace-chosen snapshots, full domain, 2x4 grid.
    zoom_late=True: the LAST FOUR of those 8 snapshots, 2x2 grid, window
    limited to 1.15x the 95th-percentile radius of the dense region
    (cells with rho >= 0.5*max rho) so the compressed shell fills the panel.
    """
    from matplotlib.collections import LineCollection

    snapshot_dir, paths = _arm_snapshot_paths(arm_root, map_arm)
    if len(paths) < 8:
        return _placeholder_figure(
            1,
            f"{map_arm} mesh structure",
            verdict,
            f"Eight indexed snapshots are required; found {len(paths)} in "
            f"{snapshot_dir}",
        )
    chosen_indices = np.linspace(0, len(paths) - 1, 8).round().astype(int)
    if zoom_late:
        chosen_indices = chosen_indices[4:]
        fig, axes = plt.subplots(
            2, 2, figsize=(11.0, 8.5), constrained_layout=True
        )
        suffix = " (late-time zoom)"
    else:
        fig, axes = plt.subplots(
            2, 4, figsize=(11.0, 8.5), constrained_layout=True
        )
        suffix = ""
    fig.suptitle(
        _title(f"{page_label}: {map_arm} mesh structure{suffix}", verdict),
        fontsize=14,
        fontweight="bold",
    )
    for ax, index in zip(axes.flat, chosen_indices, strict=True):
        snapshot = read_snapshot(paths[index], read_polygons=True)
        if snapshot.cell_polygons is None:
            raise ValueError(f"{snapshot.path}: cell polygons were not loaded")
        segments = []
        for polygon in snapshot.cell_polygons:
            polygon_um = 1.0e4 * polygon
            count = polygon_um.shape[0]
            for i in range(count):
                segments.append(
                    (polygon_um[i], polygon_um[(i + 1) % count])
                )
        ax.add_collection(
            LineCollection(
                segments,
                colors="black",
                linewidths=0.15,
                alpha=0.7,
                rasterized=True,
            ),
            autolim=True,
        )
        ax.autoscale_view()
        if zoom_late:
            radius = np.sqrt(
                snapshot.r_cell * snapshot.r_cell
                + snapshot.z_cell * snapshot.z_cell
            )
            dense = snapshot.rho >= 0.5 * float(np.max(snapshot.rho))
            if np.any(dense):
                window_um = 1.15e4 * float(
                    np.percentile(radius[dense], 95.0)
                )
                ax.set_xlim(0.0, window_um)
                ax.set_ylim(-window_um, window_um)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"t = {snapshot.time * 1.0e12:.3f} ps", fontsize=9)
        ax.set_xlabel(r"$r$ [$\mu$m]")
        ax.set_ylabel(r"$z$ [$\mu$m]")
    return fig


def _page_modal_growth(
    modes_root: Path,
    verdict: str,
    load: Callable[[Path], dict[str, Any] | None],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 3: modal growth", verdict), fontsize=14, fontweight="bold"
    )
    plotted_top = False
    for arm, _ in P2_ARMS:
        payload = load(_mode_path(modes_root, "off", arm))
        if payload is None:
            continue
        time, epsilon = _paired_series(payload, _epsilon(payload, 2))
        positive = np.abs(epsilon)
        positive[positive <= 0.0] = np.nan
        axes[0].semilogy(
            time * 1.0e12,
            positive,
            label=ARM_LABELS[arm],
            linewidth=1.5,
        )
        plotted_top = True
    null = load(_mode_path(modes_root, "off", "a0"))
    if null is not None:
        time, epsilon = _paired_series(null, _epsilon(null, 2))
        positive = np.abs(epsilon)
        positive[positive <= 0.0] = np.nan
        axes[0].semilogy(
            time * 1.0e12,
            positive,
            "k--",
            label="a0 null, l=2",
            linewidth=1.2,
        )
        plotted_top = True
    if not plotted_top:
        _missing_axis(axes[0], "P2 ladder and a0 mode files are missing")
    else:
        axes[0].set_ylabel(r"$|\epsilon_2|$")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend(ncol=3, fontsize=8)

    plotted_bottom = False
    p4 = load(_mode_path(modes_root, "off", "p4p1"))
    if p4 is not None:
        time, epsilon = _paired_series(p4, _epsilon(p4, 4))
        positive = np.abs(epsilon)
        positive[positive <= 0.0] = np.nan
        axes[1].semilogy(
            time * 1.0e12,
            positive,
            label="P4 +1%, l=4",
            linewidth=1.5,
        )
        plotted_bottom = True
    if null is not None:
        time, epsilon = _paired_series(null, _epsilon(null, 4))
        positive = np.abs(epsilon)
        positive[positive <= 0.0] = np.nan
        axes[1].semilogy(
            time * 1.0e12,
            positive,
            "k--",
            label="a0 null, l=4",
            linewidth=1.2,
        )
        plotted_bottom = True
    if not plotted_bottom:
        _missing_axis(axes[1], "P4 and a0 mode files are missing")
    else:
        axes[1].set_xlabel("time [ps]")
        axes[1].set_ylabel(r"$|\epsilon_4|$")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend(fontsize=8)
    return fig


def _page_ladder_linearity(
    modes_root: Path,
    verdict: str,
    load: Callable[[Path], dict[str, Any] | None],
    summary: dict[str, dict[str, Any]],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 4: P2 ladder linearity", verdict),
        fontsize=14,
        fontweight="bold",
    )
    radius_grid = np.linspace(50.0, STRICT_R0_UM, 201)
    arm_payloads: dict[str, dict[str, Any]] = {}
    for arm, amplitude in P2_ARMS:
        payload = load(_mode_path(modes_root, "off", arm))
        if payload is None:
            continue
        arm_payloads[arm] = payload
        response = _epsilon_at_radius(payload, 2, radius_grid) / amplitude
        ax.plot(radius_grid, response, label=ARM_LABELS[arm], linewidth=1.5)

    ax.set_xlim(50.0, STRICT_R0_UM)
    ax.set_xlabel(r"matched $R_0$ [$\mu$m]")
    ax.set_ylabel(r"$\epsilon_2/a_2$")
    ax.grid(True, alpha=0.3)
    if arm_payloads:
        ax.legend(fontsize=9)
    else:
        _missing_axis(ax, "P2 ladder mode files are missing")
        return fig

    null = load(_mode_path(modes_root, "off", "a0"))
    fit_lines = [f"joint fit at R0 = {LINEARITY_R0_UM:.0f} um"]
    if len(arm_payloads) == len(P2_ARMS) and null is not None:
        query = np.asarray([LINEARITY_R0_UM])
        null_value = float(_epsilon_at_radius(null, 2, query)[0])
        amplitudes: list[float] = []
        responses: list[float] = []
        for arm, amplitude in P2_ARMS:
            value = float(_epsilon_at_radius(arm_payloads[arm], 2, query)[0])
            if np.isfinite(value) and np.isfinite(null_value):
                amplitudes.append(amplitude)
                responses.append(value - null_value)
        if len(amplitudes) == len(P2_ARMS):
            a = np.asarray(amplitudes)
            y = np.asarray(responses)
            design = np.column_stack((a, a * a))
            coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            residual = y - design @ coefficients
            rms = float(np.sqrt(np.mean(residual * residual)))
            scale = abs(float(coefficients[0])) * 0.01
            normalized_rms = rms / scale if scale > 0.0 else float("nan")
            passed = bool(
                np.isfinite(normalized_rms)
                and normalized_rms <= LINEARITY_RMS_GATE
            )
            fit_lines.extend(
                (
                    f"c1 = {coefficients[0]:+.6g}",
                    f"c2 = {coefficients[1]:+.6g}",
                    f"rms = {rms:.3e}",
                    f"rms/(|c1| x 1%) = {normalized_rms:.2%}",
                    f"gate <= {LINEARITY_RMS_GATE:.0%}: "
                    f"{'PASS' if passed else 'FAIL'}",
                )
            )
            summary["linearity"] = {
                "label": "P2 joint-fit residual at R0=40 um",
                "value": f"{normalized_rms:.3%}",
                "gate": f"<= {LINEARITY_RMS_GATE:.0%}",
                "status": "PASS" if passed else "FAIL",
            }
        else:
            fit_lines.append("fit unavailable: R0=40 um is outside an input range")
    else:
        fit_lines.append("fit unavailable: all four P2 arms and a0 are required")
    ax.text(
        0.02,
        0.03,
        "\n".join(fit_lines),
        transform=ax.transAxes,
        va="bottom",
        family="monospace",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return fig


def _page_complex_and_purity(
    modes_root: Path,
    verdict: str,
    load: Callable[[Path], dict[str, Any] | None],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 5: complex response and P4 purity", verdict),
        fontsize=14,
        fontweight="bold",
    )
    complex_payload = load(_mode_path(modes_root, "off", "cplx"))
    if complex_payload is None:
        _missing_axis(axes[0], "arm_cplx.json is missing")
    else:
        for degree in range(2, 9):
            time, epsilon = _paired_series(
                complex_payload, _epsilon(complex_payload, degree)
            )
            axes[0].plot(
                time * 1.0e12, epsilon, label=f"l={degree}", linewidth=1.2
            )
        axes[0].axhline(0.0, color="0.5", linewidth=0.8)
        axes[0].set_xlabel("time [ps]")
        axes[0].set_ylabel(r"$\epsilon_l$")
        axes[0].set_title("Complex arm")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(ncol=2, fontsize=8)

    p4 = load(_mode_path(modes_root, "off", "p4p1"))
    if p4 is None:
        _missing_axis(axes[1], "arm_p4p1.json is missing")
    else:
        query = np.asarray([LINEARITY_R0_UM])
        values = np.asarray(
            [float(_epsilon_at_radius(p4, degree, query)[0]) for degree in range(1, 9)]
        )
        if not np.any(np.isfinite(values)):
            _missing_axis(axes[1], "R0=40 um is outside the P4 input range")
        else:
            degrees = np.arange(1, 9)
            colors = ["tab:orange" if degree == 4 else "tab:blue" for degree in degrees]
            axes[1].bar(degrees, values, color=colors)
            axes[1].axhline(0.0, color="0.3", linewidth=0.8)
            axes[1].set_xticks(degrees)
            axes[1].set_xlabel("Legendre degree l")
            axes[1].set_ylabel(r"$\epsilon_l$")
            axes[1].set_title(r"P4 +1% purity at matched $R_0=40\ \mu$m")
            axes[1].grid(True, axis="y", alpha=0.3)
    return fig


def _parse_key_values(line: str) -> dict[str, str]:
    return dict(re.findall(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)", line))


def _find_arm_logs(arm_root: Path, directory_name: str) -> list[Path]:
    run_dir = arm_root / directory_name
    candidates: list[Path] = []
    if run_dir.is_dir():
        candidates.extend(path for path in run_dir.glob("log*") if path.is_file())
        candidates.extend(path for path in run_dir.glob("*.log") if path.is_file())
    if arm_root.is_dir():
        candidates.extend(
            path for path in arm_root.glob(f"{directory_name}*.log") if path.is_file()
        )
    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve())] = path
    return [unique[key] for key in sorted(unique)]


def _ledger_counts(paths: list[Path]) -> tuple[int, int]:
    total = 0
    applied = 0
    for path in paths:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                if "[band-ale-ledger]" not in line:
                    continue
                total += 1
                fields = _parse_key_values(line)
                if fields.get("applied") == "1":
                    applied += 1
    return total, applied


def _common_time_metrics(
    off: dict[str, Any], pc: dict[str, Any]
) -> tuple[float, float, float, float, int]:
    off_time = _payload_series(off, "time")
    pc_time = _payload_series(pc, "time")
    off_radius = _radius_um(off)
    pc_radius = _radius_um(pc)
    off_eps = _epsilon(off, 2)
    pc_eps = _epsilon(pc, 2)
    off_count = min(off_time.size, off_radius.size, off_eps.size)
    pc_count = min(pc_time.size, pc_radius.size, pc_eps.size)
    off_time = off_time[:off_count]
    off_radius = off_radius[:off_count]
    off_eps = off_eps[:off_count]
    pc_time = pc_time[:pc_count]
    pc_radius = pc_radius[:pc_count]
    pc_eps = pc_eps[:pc_count]
    off_h_f = np.full(off_count, np.nan, dtype=float)
    finite_run_h_f = np.empty(0, dtype=float)
    if "h_f_median" in off:
        available_h_f = _h_f_um(off)
        finite_run_h_f = available_h_f[np.isfinite(available_h_f)]
        h_f_count = min(off_count, available_h_f.size)
        off_h_f[:h_f_count] = available_h_f[:h_f_count]
    finite_h_f = np.isfinite(off_h_f)
    if finite_run_h_f.size:
        h_f_floor = 0.2 * np.median(finite_run_h_f)
        h_f_present = finite_h_f & (off_h_f >= h_f_floor)
    else:
        h_f_present = finite_h_f
    off_h_f[~h_f_present] = np.nan
    finite_off_time = off_time[np.isfinite(off_time)]
    finite_pc_time = pc_time[np.isfinite(pc_time)]
    if finite_off_time.size < 2 or finite_pc_time.size < 2:
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )
    lower = max(float(np.min(finite_off_time)), float(np.min(finite_pc_time)))
    upper = min(float(np.max(finite_off_time)), float(np.max(finite_pc_time)))
    common_time = np.unique(
        np.concatenate(
            (
                off_time[(off_time >= lower) & (off_time <= upper)],
                pc_time[(pc_time >= lower) & (pc_time <= upper)],
            )
        )
    )
    off_r = _interp(off_time, off_radius, common_time)
    pc_r = _interp(pc_time, pc_radius, common_time)
    off_h = _interp(off_time, off_h_f, common_time)
    off_h_present = _interp(
        off_time, h_f_present.astype(float), common_time
    ) == 1.0
    off_e = _interp(off_time, off_eps, common_time)
    pc_e = _interp(pc_time, pc_eps, common_time)
    valid = (
        np.isfinite(off_r)
        & np.isfinite(pc_r)
        & np.isfinite(off_e)
        & np.isfinite(pc_e)
        & (off_r > STRICT_R0_UM)
        & (pc_r > STRICT_R0_UM)
        & (off_r != 0.0)
    )
    if not np.any(valid):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            0,
        )
    relative_radius = np.abs(pc_r[valid] - off_r[valid]) / np.abs(off_r[valid])
    radius_valid = valid & np.isfinite(off_h) & off_h_present & (off_h > 0.0)
    radius_gate_ratio = np.abs(pc_r[radius_valid] - off_r[radius_valid]) / (
        0.05 * off_h[radius_valid]
    )
    epsilon_difference = np.abs(pc_e[valid] - off_e[valid])
    epsilon_gate = EPS_NEUTRALITY_FLOOR + EPS_NEUTRALITY_FRACTION * np.abs(
        off_e[valid]
    )
    epsilon_gate_ratio = epsilon_difference / epsilon_gate
    return (
        float(np.max(relative_radius)),
        float(np.max(radius_gate_ratio)) if radius_gate_ratio.size else float("nan"),
        float(np.max(epsilon_difference)),
        float(np.max(epsilon_gate_ratio)),
        int(np.count_nonzero(valid)),
    )


def _ordered_arms(names: set[str]) -> list[str]:
    known = [name for name in ARM_ORDER if name in names]
    return known + sorted(names - set(known))


def _page_neutrality(
    arm_root: Path,
    modes_root: Path,
    ledger_logs: list[Path],
    verdict: str,
    load: Callable[[Path], dict[str, Any] | None],
    summary: dict[str, dict[str, Any]],
) -> plt.Figure:
    off_names = {
        path.stem[len("arm_") :]
        for path in modes_root.glob("arm_*.json")
        if path.is_file()
    }
    pc_names = {
        path.stem[len("armpc_") :]
        for path in modes_root.glob("armpc_*.json")
        if path.is_file()
    }
    arms = _ordered_arms(off_names & pc_names)
    if not arms:
        return _placeholder_figure(
            6,
            "maintenance OFF vs ON neutrality",
            verdict,
            "No matching arm_<name>.json and armpc_<name>.json pairs were found",
        )

    rows: list[list[str]] = []
    radius_gate_ratios: list[float] = []
    epsilon_gate_ratios: list[float] = []
    for arm in arms:
        off = load(_mode_path(modes_root, "off", arm))
        pc = load(_mode_path(modes_root, "pc", arm))
        if off is None or pc is None:
            continue
        (
            rel_radius,
            radius_ratio,
            epsilon_delta,
            epsilon_ratio,
            samples,
        ) = _common_time_metrics(off, pc)
        logs = _find_arm_logs(arm_root, f"arm_pc_{arm}")
        ledger_total, ledger_applied = _ledger_counts(logs)
        if np.isfinite(radius_ratio):
            radius_gate_ratios.append(radius_ratio)
        if np.isfinite(epsilon_ratio):
            epsilon_gate_ratios.append(epsilon_ratio)
        rows.append(
            [
                ARM_LABELS.get(arm, arm),
                f"{rel_radius:.3e}" if np.isfinite(rel_radius) else "N/A",
                f"{radius_ratio:.3f}" if np.isfinite(radius_ratio) else "N/A",
                f"{epsilon_delta:.3e}" if np.isfinite(epsilon_delta) else "N/A",
                f"{epsilon_ratio:.3f}" if np.isfinite(epsilon_ratio) else "N/A",
                f"{ledger_applied}/{ledger_total}",
                str(samples),
            ]
        )

    fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 6: maintenance OFF vs ON neutrality", verdict),
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")
    columns = (
        "arm",
        "max |dR|/R",
        "max Eq.116 ratio",
        "max |d eps2|",
        "eps gate ratio",
        "ledger applied/total",
        "samples",
    )
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.65)
    for column in range(len(columns)):
        table[(0, column)].set_facecolor("0.88")
        table[(0, column)].set_text_props(fontweight="bold")

    explicit_total, explicit_applied = _ledger_counts(ledger_logs)
    footer = (
        f"Strict window: R0 > {STRICT_R0_UM:.0f} um.  Epsilon gate ratio = "
        f"|d eps2| / ({EPS_NEUTRALITY_FLOOR:.0e} + "
        f"{EPS_NEUTRALITY_FRACTION:.2f}|eps2_off|); gate <= 1.  "
        "Eq.116 gate ratio = |dR| / (0.05 h_f,off); gate <= 1; "
        "frames with degenerate h_f excluded."
    )
    if ledger_logs:
        footer += (
            f"  Explicit --ledger-log files: applied/total = "
            f"{explicit_applied}/{explicit_total}."
        )
    ax.text(0.5, 0.08, footer, ha="center", va="center", fontsize=8.5, wrap=True)

    if epsilon_gate_ratios:
        worst = max(epsilon_gate_ratios)
        passed = worst <= 1.0
        summary["neutrality_eps2"] = {
            "label": "OFF/ON eps2 neutrality (worst arm)",
            "value": f"gate ratio {worst:.3f}",
            "gate": "<= 1",
            "status": "PASS" if passed else "FAIL",
        }
    if radius_gate_ratios:
        worst = max(radius_gate_ratios)
        passed = worst <= 1.0
        summary["neutrality_radius"] = {
            "label": "OFF/ON Eq.116 radius neutrality (guarded worst arm)",
            "value": f"guarded max ratio {worst:.3f}",
            "gate": "|dR| <= 0.05 h_f",
            "status": "PASS" if passed else "FAIL",
        }
    return fig


def _parse_shadow(path: Path) -> dict[str, np.ndarray]:
    step: list[float] = []
    time: list[float] = []
    agree: list[float] = []
    legacy_only: list[float] = []
    pc_only: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if "[band-ale-pc-shadow]" not in line:
                continue
            fields = _parse_key_values(line)
            try:
                agree_value = float(fields["agree"])
                step_value = float(fields.get("step", len(step)))
                legacy_value = float(fields.get("hold_xor_legacy_only", "nan"))
                pc_value = float(fields.get("hold_xor_pc_only", "nan"))
                time_value = float(fields.get("t", "nan"))
            except (KeyError, ValueError):
                continue
            step.append(step_value)
            time.append(time_value)
            agree.append(agree_value)
            legacy_only.append(legacy_value)
            pc_only.append(pc_value)
    return {
        "step": np.asarray(step),
        "time": np.asarray(time),
        "agree": np.asarray(agree),
        "legacy_only": np.asarray(legacy_only),
        "pc_only": np.asarray(pc_only),
    }


def _page_shadow(
    shadow_log: Path | None,
    verdict: str,
    summary: dict[str, dict[str, Any]],
) -> plt.Figure:
    if shadow_log is None or not shadow_log.is_file():
        requested = str(shadow_log) if shadow_log is not None else "--shadow-log"
        return _placeholder_figure(
            7,
            "per-column shadow equivalence",
            verdict,
            f"Shadow log unavailable: {requested}",
        )
    data = _parse_shadow(shadow_log)
    if data["agree"].size == 0:
        return _placeholder_figure(
            7,
            "per-column shadow equivalence",
            verdict,
            f"No [band-ale-pc-shadow] records found in {shadow_log}",
        )
    use_time = np.all(np.isfinite(data["time"]))
    x = data["time"] * 1.0e12 if use_time else data["step"]
    x_label = "time [ps]" if use_time else "step (log has no t field)"
    agree_fraction = float(np.mean(data["agree"] == 1.0))
    passed = agree_fraction == 1.0

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 7: per-column shadow equivalence", verdict),
        fontsize=14,
        fontweight="bold",
    )
    axes[0].step(x, data["agree"], where="post", linewidth=1.2)
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].set_yticks((0, 1))
    axes[0].set_ylabel("agree")
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        0.02,
        0.12,
        f"agree fraction = {agree_fraction:.6f} "
        f"({int(np.count_nonzero(data['agree'] == 1.0))}/{data['agree'].size})\n"
        f"gate = 1.000000: {'PASS' if passed else 'FAIL'}",
        transform=axes[0].transAxes,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axes[1].plot(x, data["legacy_only"], label="hold_xor_legacy_only")
    axes[1].plot(x, data["pc_only"], label="hold_xor_pc_only")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("cumulative count")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    summary["shadow"] = {
        "label": "Shadow decision agreement",
        "value": f"{agree_fraction:.6f}",
        "gate": "= 1.000000",
        "status": "PASS" if passed else "FAIL",
    }
    return fig


def _h2_candidates(modes_root: Path) -> list[str]:
    off_names = {
        path.stem[len("arm_") :]
        for path in modes_root.glob("arm_*.json")
        if path.is_file()
    }
    h2_names = {
        path.stem[len("armh2_") :]
        for path in modes_root.glob("armh2_*.json")
        if path.is_file()
    }
    names = {name for name in off_names & h2_names if name.startswith("p2")}
    preferred = ["p2p1", "p2p2", "p2p05", "p2m05"]
    return [name for name in preferred if name in names] + sorted(
        names - set(preferred)
    )


def _page_resolution(
    modes_root: Path,
    verdict: str,
    load: Callable[[Path], dict[str, Any] | None],
    summary: dict[str, dict[str, Any]],
) -> plt.Figure:
    candidates = _h2_candidates(modes_root)
    if not candidates:
        return _placeholder_figure(
            8,
            "h versus h/2 consistency",
            verdict,
            "No matching P2 arm_<name>.json and armh2_<name>.json pair was found",
        )
    arm = candidates[0]
    coarse = load(_mode_path(modes_root, "off", arm))
    fine = load(_mode_path(modes_root, "h2", arm))
    if coarse is None or fine is None:
        return _placeholder_figure(
            8, "h versus h/2 consistency", verdict, f"Could not load h/2 pair: {arm}"
        )

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title(f"Page 8: h versus h/2 consistency ({ARM_LABELS.get(arm, arm)})", verdict),
        fontsize=14,
        fontweight="bold",
    )
    coarse_time, coarse_eps = _paired_series(coarse, _epsilon(coarse, 2))
    fine_time, fine_eps = _paired_series(fine, _epsilon(fine, 2))
    axes[0].plot(coarse_time * 1.0e12, coarse_eps, label="h", linewidth=1.5)
    axes[0].plot(fine_time * 1.0e12, fine_eps, label="h/2", linewidth=1.5)
    axes[0].set_xlabel("time [ps]")
    axes[0].set_ylabel(r"$\epsilon_2$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    radius_grid = np.linspace(50.0, STRICT_R0_UM, 201)
    coarse_matched = _epsilon_at_radius(coarse, 2, radius_grid)
    fine_matched = _epsilon_at_radius(fine, 2, radius_grid)
    valid = np.isfinite(coarse_matched) & np.isfinite(fine_matched)
    relative = np.full(radius_grid.shape, np.nan)
    relative[valid] = np.abs(coarse_matched[valid] - fine_matched[valid]) / np.maximum(
        np.abs(fine_matched[valid]), EPS_NEUTRALITY_FLOOR
    )
    axes[1].plot(
        radius_grid,
        relative,
        color="0.65",
        linewidth=1.0,
        alpha=0.6,
        label="raw matched-R0 difference (context)",
    )

    def quadratic_fit(
        payload: dict[str, Any],
    ) -> tuple[np.ndarray, float] | None:
        radius = _radius_um(payload)
        epsilon = _epsilon(payload, 2)
        count = min(radius.size, epsilon.size)
        radius = radius[:count]
        epsilon = epsilon[:count]
        fit_mask = (
            np.isfinite(radius)
            & np.isfinite(epsilon)
            & (radius >= STRICT_R0_UM)
            & (radius <= 50.0)
        )
        fit_radius = radius[fit_mask]
        fit_epsilon = epsilon[fit_mask]
        if np.unique(fit_radius).size < 3:
            return None
        centered_radius = fit_radius - LINEARITY_R0_UM
        design = np.column_stack(
            (
                centered_radius * centered_radius,
                centered_radius,
                np.ones_like(fit_radius),
            )
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, fit_epsilon, rcond=None)
        residual_std = float(np.std(fit_epsilon - design @ coefficients))
        return coefficients, residual_std

    coarse_fit = quadratic_fit(coarse)
    fine_fit = quadratic_fit(fine)
    fit_radii = np.asarray([50.0, 45.0, 40.0, 35.0, 30.0])
    if coarse_fit is not None and fine_fit is not None:
        coarse_coefficients, coarse_residual_std = coarse_fit
        fine_coefficients, fine_residual_std = fine_fit

        def evaluate_fit(coefficients: np.ndarray, radius: np.ndarray) -> np.ndarray:
            centered_radius = radius - LINEARITY_R0_UM
            return (
                coefficients[0] * centered_radius * centered_radius
                + coefficients[1] * centered_radius
                + coefficients[2]
            )

        coarse_fit_grid = evaluate_fit(coarse_coefficients, radius_grid)
        fine_fit_grid = evaluate_fit(fine_coefficients, radius_grid)
        fitted_relative = np.abs(coarse_fit_grid - fine_fit_grid) / np.maximum(
            np.abs(fine_fit_grid), EPS_NEUTRALITY_FLOOR
        )
        axes[1].plot(
            radius_grid,
            fitted_relative,
            color="tab:blue",
            linewidth=1.8,
            label="fitted-trajectory difference (quadratic)",
        )
        coarse_gate = evaluate_fit(coarse_coefficients, fit_radii)
        fine_gate = evaluate_fit(fine_coefficients, fit_radii)
        gate_relative = np.abs(coarse_gate - fine_gate) / np.maximum(
            np.abs(fine_gate), EPS_NEUTRALITY_FLOOR
        )
        axes[1].scatter(fit_radii, gate_relative, color="tab:blue", s=22, zorder=3)
        maximum = float(np.max(gate_relative))
        passed = maximum <= H2_RELATIVE_GATE
        axes[1].text(
            0.02,
            0.06,
            f"quadratic fits over [30, 50] um\n"
            f"fit residual std (point-noise scale): "
            f"h={coarse_residual_std:.3e}, "
            f"h/2={fine_residual_std:.3e}\n"
            f"max at R0={{50,45,40,35,30}} um = {maximum:.3%}; "
            f"gate <= {H2_RELATIVE_GATE:.0%}: "
            f"{'PASS' if passed else 'FAIL'}",
            transform=axes[1].transAxes,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
        summary["resolution"] = {
            "label": (
                f"h/h2 fitted-trajectory eps2 consistency "
                f"({ARM_LABELS.get(arm, arm)})"
            ),
            "value": f"fitted max {maximum:.3%}",
            "gate": f"<= {H2_RELATIVE_GATE:.0%}",
            "status": "PASS" if passed else "FAIL",
        }
    else:
        axes[1].text(
            0.02,
            0.08,
            "quadratic fit unavailable: each resolution requires at least "
            "three distinct R0 samples in [30, 50] um",
            transform=axes[1].transAxes,
            wrap=True,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    axes[1].axhline(
        H2_RELATIVE_GATE,
        color="tab:red",
        linestyle="--",
        label=f"{H2_RELATIVE_GATE:.0%} gate",
    )
    axes[1].set_xlim(50.0, STRICT_R0_UM)
    axes[1].set_xlabel(r"matched $R_0$ [$\mu$m]")
    axes[1].set_ylabel(r"$|\epsilon_{2,h}-\epsilon_{2,h/2}| / "
                       r"\max(|\epsilon_{2,h/2}|,5\times10^{-5})$")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    return fig


def _page_summary(
    verdict: str, summary: dict[str, dict[str, Any]]
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11.0, 8.5), constrained_layout=True)
    fig.suptitle(
        _title("Page 9: verdict summary", verdict), fontsize=14, fontweight="bold"
    )
    ax.axis("off")
    order = (
        "linearity",
        "neutrality_radius",
        "neutrality_eps2",
        "shadow",
        "resolution",
    )
    rows: list[list[str]] = []
    for key in order:
        item = summary.get(key)
        if item is None:
            label = {
                "linearity": "P2 joint-fit residual at R0=40 um",
                "neutrality_radius": "OFF/ON Eq.116 radius neutrality (guarded)",
                "neutrality_eps2": "OFF/ON eps2 neutrality",
                "shadow": "Shadow decision agreement",
                "resolution": "h/h2 fitted-trajectory eps2 consistency",
            }[key]
            rows.append([label, "N/A", "input required", "NOT EVALUATED"])
        else:
            rows.append(
                [
                    str(item["label"]),
                    str(item["value"]),
                    str(item["gate"]),
                    str(item["status"]),
                ]
            )
    table = ax.table(
        cellText=rows,
        colLabels=("gate", "computed value", "criterion", "result"),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.42, 0.20, 0.20, 0.18),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)
    for column in range(4):
        table[(0, column)].set_facecolor("0.85")
        table[(0, column)].set_text_props(fontweight="bold")
    for row in range(1, len(rows) + 1):
        status = rows[row - 1][3]
        if status == "PASS":
            table[(row, 3)].set_facecolor("#c8e6c9")
        elif status == "FAIL":
            table[(row, 3)].set_facecolor("#ffcdd2")
        else:
            table[(row, 3)].set_facecolor("#eeeeee")
    ax.text(
        0.5,
        0.12,
        "Summary values are computed from pages 4, 6, 7, and 8. "
        "A missing input remains NOT EVALUATED and does not imply a pass.",
        ha="center",
        va="center",
        fontsize=10,
        wrap=True,
    )
    return fig


def _flatten_rasterized_page(figure: plt.Figure, dpi: int = 300) -> plt.Figure:
    """Render a figure through Agg and rewrap it as one full-page image.

    The PDF mixed-mode renderer (vector page with embedded rasterized
    artists) drops polygon chunks from dense cell-polygon collections
    (measured on matplotlib 3.6.3: striped voids at t=0, while the same
    figure is clean via pure Agg and via pure vector output). Pages that
    contain any rasterized artist are therefore rendered end-to-end through
    Agg and embedded as a single image, so the defective mixed path is
    never exercised.
    """
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi)
    plt.close(figure)
    buffer.seek(0)
    image = plt.imread(buffer)
    height, width = image.shape[:2]
    wrapper = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    axis = wrapper.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.imshow(image, interpolation="none")
    axis.set_axis_off()
    return wrapper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm-root",
        type=Path,
        required=True,
        help="Directory containing arm_off_*, arm_pc_*, and arm_h2_* run directories",
    )
    parser.add_argument(
        "--map-arm",
        default="arm_off_p2p2",
        help="Arm run directory used for density maps and front-shape visibility",
    )
    parser.add_argument(
        "--map-arm-a2",
        type=float,
        default=0.02,
        help="Drive amplitude used for the page 2 analytic g-ratio line",
    )
    parser.add_argument(
        "--modes-root",
        type=Path,
        required=True,
        help=(
            "Directory containing arm_<name>.json, armpc_<name>.json, and "
            "armh2_<name>.json mode files"
        ),
    )
    parser.add_argument(
        "--out-pdf", type=Path, required=True, help="Output nine-page PDF path"
    )
    parser.add_argument(
        "--verdict",
        default="",
        help="One-line verdict text appended to every page title",
    )
    parser.add_argument(
        "--ledger-log",
        type=Path,
        nargs="*",
        default=[],
        help="Additional log files containing [band-ale-ledger] records",
    )
    parser.add_argument(
        "--shadow-log",
        type=Path,
        help="Log file containing [band-ale-pc-shadow] records",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "axes.formatter.useoffset": False,
            "savefig.dpi": 150,
        }
    )
    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    payload_cache: dict[Path, dict[str, Any] | None] = {}

    def load(path: Path) -> dict[str, Any] | None:
        if path not in payload_cache:
            if not path.is_file():
                payload_cache[path] = None
            else:
                try:
                    payload_cache[path] = _load_payload(path)
                except (OSError, ValueError, json.JSONDecodeError):
                    payload_cache[path] = None
        return payload_cache[path]

    summary: dict[str, dict[str, Any]] = {}
    page_builders: list[tuple[str, Callable[[], plt.Figure]]] = [
        (
            f"{args.map_arm} density maps",
            lambda: _page_rho_maps(args.arm_root, args.map_arm, args.verdict),
        ),
        (
            f"front-shape visibility ({args.map_arm})",
            lambda: _page_front_shape_visibility(
                args.arm_root,
                args.map_arm,
                (1.0 + args.map_arm_a2) / (1.0 - 0.5 * args.map_arm_a2),
                "analytic g ratio = "
                f"{(1.0 + args.map_arm_a2) / (1.0 - 0.5 * args.map_arm_a2):.5f}",
                args.verdict,
            ),
        ),
        (
            "modal growth",
            lambda: _page_modal_growth(args.modes_root, args.verdict, load),
        ),
        (
            "P2 ladder linearity",
            lambda: _page_ladder_linearity(
                args.modes_root, args.verdict, load, summary
            ),
        ),
        (
            "complex response and P4 purity",
            lambda: _page_complex_and_purity(args.modes_root, args.verdict, load),
        ),
        (
            "maintenance OFF vs ON neutrality",
            lambda: _page_neutrality(
                args.arm_root,
                args.modes_root,
                args.ledger_log,
                args.verdict,
                load,
                summary,
            ),
        ),
        (
            "per-column shadow equivalence",
            lambda: _page_shadow(args.shadow_log, args.verdict, summary),
        ),
        (
            "h versus h/2 consistency",
            lambda: _page_resolution(args.modes_root, args.verdict, load, summary),
        ),
        ("verdict summary", lambda: _page_summary(args.verdict, summary)),
    ]
    metadata = {
        "Title": "TENRYU zonal-asymmetry judgment stack",
        "Author": "TENRYU validation tools",
        "Creator": "make_asym_judgment_stack.py",
        "CreationDate": dt.datetime(2000, 1, 1),
        "ModDate": dt.datetime(2000, 1, 1),
    }
    with PdfPages(args.out_pdf, metadata=metadata) as pdf:
        for page_number, (page_name, builder) in enumerate(page_builders, start=1):
            try:
                figure = builder()
            except (OSError, KeyError, ValueError, IndexError) as error:
                figure = _placeholder_figure(
                    page_number,
                    page_name,
                    args.verdict,
                    f"{type(error).__name__}: {error}",
                )
            if any(
                bool(artist.get_rasterized())
                for artist in figure.findobj()
            ):
                figure = _flatten_rasterized_page(figure)
            pdf.savefig(figure, dpi=300)
            plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
