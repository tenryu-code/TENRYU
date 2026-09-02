#!/usr/bin/env python3
"""Extract zonal shock-front radii and Legendre modes from HDF5 snapshots."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


_INVALID_REASON_KEYS = (
    "no_cells",
    "no_component",
    "plateau_inner",
    "plateau_outer",
    "flat_plateau",
    "crossing_outside",
    "rho_pass_failed",
    "other",
)


@dataclass(frozen=True)
class Snapshot:
    path: Path
    time: float
    r_cell: np.ndarray
    z_cell: np.ndarray
    rho: np.ndarray
    pressure: np.ndarray | None


def _scalar_time(handle: h5py.File) -> float:
    if "time_state/t" in handle:
        values = np.asarray(handle["time_state/t"][()]).reshape(-1)
        if values.size:
            return float(values[0])
    return float(handle.attrs.get("t", 0.0))


def _cell_nverts(handle: h5py.File, n_cells: int) -> np.ndarray | None:
    for name in ("mesh/topology/v4/cell_nverts", "mesh/topology/v1/cell_nverts"):
        if name in handle:
            values = np.asarray(handle[name][()], dtype=np.int64).reshape(-1)
            if values.size != n_cells:
                raise ValueError(
                    f"{name}: expected {n_cells} entries, got {values.size}")
            return values
    return None


def _csr_cell_centroids(
    handle: h5py.File,
    r_nodes: np.ndarray,
    z_nodes: np.ndarray,
    n_cells: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    topology_group = None
    for candidate in ("mesh/topology/v3", "mesh/topology/v2"):
        offsets_name = f"{candidate}/cell_node_csr_offsets"
        indices_name = f"{candidate}/cell_node_csr_indices"
        if offsets_name in handle or indices_name in handle:
            if offsets_name not in handle or indices_name not in handle:
                raise KeyError(
                    f"incomplete cell-node connectivity under {candidate}")
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
            f"{n_cells + 1} entries, got {offsets.size}")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0) or offsets[-1] > indices.size:
        raise ValueError(f"invalid cell-node CSR offsets under {topology_group}")

    nverts = _cell_nverts(handle, n_cells)
    r_cell = np.full(n_cells, np.nan, dtype=float)
    z_cell = np.full(n_cells, np.nan, dtype=float)
    for cell in range(n_cells):
        begin = int(offsets[cell])
        end = int(offsets[cell + 1])
        cell_nodes = indices[begin:end]
        if nverts is not None:
            count = int(nverts[cell])
            if count == 0:
                continue
            if count < 3 or count > cell_nodes.size:
                raise ValueError(
                    f"cell {cell}: invalid active vertex count {count}")
            cell_nodes = cell_nodes[:count]
        else:
            cell_nodes = cell_nodes[cell_nodes >= 0]
        if cell_nodes.size < 3:
            raise ValueError(f"cell {cell}: fewer than three connected nodes")
        if np.any(cell_nodes < 0) or np.any(cell_nodes >= r_nodes.size):
            raise ValueError(f"cell {cell}: node index out of range")
        r_cell[cell] = float(np.mean(r_nodes[cell_nodes]))
        z_cell[cell] = float(np.mean(z_nodes[cell_nodes]))
    return r_cell, z_cell


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
            f"structured node shape {node_shape} disagrees with nr={nr}, nz={nz}")
    if n_cells != nr * nz:
        raise ValueError(
            f"hydro/rho has {n_cells} cells, expected {nr * nz}")

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
                r_flat[cell] = float(np.mean((
                    r_nodes[i, j], r_nodes[i + 1, j], r_nodes[i + 1, j + 1])))
                z_flat[cell] = float(np.mean((
                    z_nodes[i, j], z_nodes[i + 1, j], z_nodes[i + 1, j + 1])))
    return r_cell.reshape(-1), z_cell.reshape(-1)


def read_snapshot(path: Path, read_pressure: bool) -> Snapshot:
    with h5py.File(path, "r") as handle:
        for name in ("mesh/x_r", "mesh/x_z", "hydro/rho"):
            if name not in handle:
                raise KeyError(f"{path}: required HDF5 dataset missing: {name}")
        if read_pressure:
            for name in ("hydro/Pe", "hydro/Pi"):
                if name not in handle:
                    raise KeyError(
                        f"{path}: required HDF5 dataset missing: {name}")
        r_nodes = np.asarray(handle["mesh/x_r"][()], dtype=float)
        z_nodes = np.asarray(handle["mesh/x_z"][()], dtype=float)
        rho = np.asarray(handle["hydro/rho"][()], dtype=float).reshape(-1)
        if read_pressure:
            pe = np.asarray(handle["hydro/Pe"][()], dtype=float).reshape(-1)
            pi = np.asarray(handle["hydro/Pi"][()], dtype=float).reshape(-1)
            if pe.size != rho.size or pi.size != rho.size:
                raise ValueError(
                    f"{path}: pressure and density dataset sizes differ")
            pressure = pe + pi
        else:
            pressure = None
        if r_nodes.size != z_nodes.size:
            raise ValueError(f"{path}: mesh/x_r and mesh/x_z sizes differ")

        csr_centroids = _csr_cell_centroids(
            handle, r_nodes.reshape(-1), z_nodes.reshape(-1), rho.size)
        if csr_centroids is None:
            r_cell, z_cell = _structured_cell_centroids(
                handle, r_nodes, z_nodes, rho.size)
        else:
            r_cell, z_cell = csr_centroids
        time = _scalar_time(handle)

    valid = np.isfinite(r_cell) & np.isfinite(z_cell) & np.isfinite(rho)
    return Snapshot(
        path=path,
        time=time,
        r_cell=r_cell[valid],
        z_cell=z_cell[valid],
        rho=rho[valid],
        pressure=pressure[valid] if pressure is not None else None,
    )


def _running_mean_three(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values.copy()
    padded = np.pad(values, (1, 1), mode="edge")
    return np.convolve(padded, np.full(3, 1.0 / 3.0), mode="valid")


def _local_radial_spacing(
    radius: np.ndarray,
    crossing: float,
) -> float:
    gaps = np.diff(radius)
    distinct_threshold = 1.0e-3 * (radius[-1] - radius[0])
    distinct_radius = np.concatenate((
        radius[:1],
        radius[1:][gaps > distinct_threshold],
    ))
    crossing_index = int(np.searchsorted(distinct_radius, crossing))
    window_start = max(0, crossing_index - 8)
    window_end = min(distinct_radius.size, crossing_index + 8)
    local_gaps = np.diff(distinct_radius[window_start:window_end])
    local_gaps = local_gaps[
        np.isfinite(local_gaps) & (local_gaps > 0.0)
    ]
    if local_gaps.size == 0:
        return float("nan")
    return float(np.median(local_gaps))


def _front_radius(
    radius: np.ndarray,
    rho: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    finite = np.isfinite(radius) & np.isfinite(rho)
    radius = radius[finite]
    rho = rho[finite]
    if radius.size < 2:
        return float("nan"), float("nan")
    order = np.argsort(radius, kind="stable")
    radius = radius[order]
    rho_smooth = _running_mean_three(rho[order])
    crossings = np.flatnonzero(rho_smooth >= threshold)
    if crossings.size == 0 or crossings[0] == 0:
        return float("nan"), float("nan")
    upper = int(crossings[0])
    lower = upper - 1
    fraction = ((threshold - rho_smooth[lower]) /
                (rho_smooth[upper] - rho_smooth[lower]))
    crossing = float(
        radius[lower] + fraction * (radius[upper] - radius[lower]))
    return crossing, _local_radial_spacing(radius, crossing)


def _isotonic(values: np.ndarray, direction: float) -> np.ndarray:
    block_values: list[float] = []
    block_weights: list[int] = []
    block_starts: list[int] = []
    block_ends: list[int] = []
    for index, value in enumerate(direction * values):
        block_values.append(float(value))
        block_weights.append(1)
        block_starts.append(index)
        block_ends.append(index + 1)
        while (
            len(block_values) >= 2
            and block_values[-2] > block_values[-1]
        ):
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_value = (
                block_values[-2] * block_weights[-2]
                + block_values[-1] * block_weights[-1]
            ) / merged_weight
            block_values[-2:] = [merged_value]
            block_weights[-2:] = [merged_weight]
            block_ends[-2:] = [block_ends[-1]]
            block_starts.pop()

    fitted = np.empty(values.size, dtype=float)
    for value, start, end in zip(
        block_values, block_starts, block_ends, strict=True
    ):
        fitted[start:end] = direction * value
    return fitted


def _midpoint_crossing(
    radius: np.ndarray,
    fitted_log_field: np.ndarray,
    midpoint: float,
    direction: float,
) -> tuple[float, int]:
    oriented_field = direction * fitted_log_field
    oriented_midpoint = direction * midpoint
    if oriented_midpoint < oriented_field[0]:
        return float("nan"), -1
    if oriented_midpoint > oriented_field[-1]:
        return float("nan"), 1
    for index in range(fitted_log_field.size - 1):
        inner = oriented_field[index]
        outer = oriented_field[index + 1]
        if inner <= oriented_midpoint <= outer and inner != outer:
            fraction = ((midpoint - fitted_log_field[index]) /
                        (fitted_log_field[index + 1]
                         - fitted_log_field[index]))
            crossing = radius[index] + fraction * (
                radius[index + 1] - radius[index])
            return float(crossing), 0
    return float("nan"), 0


def _midpoint_front_radius(
    radius: np.ndarray,
    field: np.ndarray,
) -> tuple[float, float, str | None]:
    finite = np.isfinite(radius) & np.isfinite(field) & (field > 0.0)
    radius = radius[finite]
    field = field[finite]
    if radius.size < 7:
        return float("nan"), float("nan"), "no_cells"
    order = np.argsort(radius, kind="stable")
    radius = radius[order]
    log_field = np.log(field[order])

    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = np.gradient(log_field, radius)
    gradient_magnitude = np.abs(gradient)
    if not np.any(np.isfinite(gradient_magnitude)):
        return float("nan"), float("nan"), "no_component"
    ridge = int(np.nanargmax(gradient_magnitude))
    ridge_magnitude = gradient_magnitude[ridge]
    if not np.isfinite(ridge_magnitude) or ridge_magnitude <= 0.0:
        return float("nan"), float("nan"), "no_component"

    component_start = ridge
    component_end = ridge
    component_threshold = 0.5 * ridge_magnitude
    while (
        component_start > 0
        and gradient_magnitude[component_start - 1] >= component_threshold
    ):
        component_start -= 1
    while (
        component_end + 1 < radius.size
        and gradient_magnitude[component_end + 1] >= component_threshold
    ):
        component_end += 1
    component_start = max(0, component_start - 1)
    component_end = min(radius.size - 1, component_end + 1)

    inner_plateau = field[order][max(0, component_start - 4):component_start]
    outer_plateau = field[order][component_end + 1:component_end + 5]
    if inner_plateau.size < 2:
        return float("nan"), float("nan"), "plateau_inner"
    if outer_plateau.size < 2:
        return float("nan"), float("nan"), "plateau_outer"
    inner_plateau_log = float(np.log(np.median(inner_plateau)))
    outer_plateau_log = float(np.log(np.median(outer_plateau)))
    plateau_difference = outer_plateau_log - inner_plateau_log
    if abs(plateau_difference) < 1.0e-12:
        return float("nan"), float("nan"), "flat_plateau"
    direction = float(np.sign(plateau_difference))
    midpoint = 0.5 * (inner_plateau_log + outer_plateau_log)

    window_start = max(0, min(component_start, ridge - 3))
    window_end = min(radius.size - 1, max(component_end, ridge + 3))
    for attempt in range(3):
        window = slice(window_start, window_end + 1)
        fitted = _isotonic(log_field[window], direction)
        crossing, crossing_side = _midpoint_crossing(
            radius[window], fitted, midpoint, direction)
        if np.isfinite(crossing):
            h_front = _local_radial_spacing(radius, crossing)
            if not np.isfinite(h_front):
                return float("nan"), float("nan"), "other"
            return crossing, h_front, None
        if attempt == 2 or crossing_side == 0:
            break
        if crossing_side < 0 and window_start > 0:
            window_start -= 1
        elif crossing_side > 0 and window_end + 1 < radius.size:
            window_end += 1
        else:
            break
    return float("nan"), float("nan"), "crossing_outside"


def _sector_averaged_legendre(
    sector_edges: np.ndarray,
    l_max: int,
) -> np.ndarray:
    mu_inner = np.cos(sector_edges[:-1])
    mu_outer = np.cos(sector_edges[1:])
    weights = mu_inner - mu_outer
    design = np.ones((weights.size, l_max + 1), dtype=float)
    if l_max == 0:
        return design
    p_inner = np.polynomial.legendre.legvander(mu_inner, l_max + 1)
    p_outer = np.polynomial.legendre.legvander(mu_outer, l_max + 1)
    for degree in range(1, l_max + 1):
        antiderivative_difference = (
            p_inner[:, degree + 1]
            - p_inner[:, degree - 1]
            - p_outer[:, degree + 1]
            + p_outer[:, degree - 1]
        ) / (2 * degree + 1)
        design[:, degree] = antiderivative_difference / weights
    return design


def _fit_legendre(
    front: np.ndarray,
    design: np.ndarray,
    sector_weights: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(front)
    if not np.any(valid):
        return np.full(design.shape[1], np.nan, dtype=float)
    sqrt_weights = np.sqrt(sector_weights[valid])
    coefficients, _, _, _ = np.linalg.lstsq(
        design[valid] * sqrt_weights[:, None],
        front[valid] * sqrt_weights,
        rcond=None,
    )
    return coefficients


def _json_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def analyze(
    snapshots: list[Snapshot],
    n_sectors: int,
    l_max: int,
    rho_jump_factor: float,
    front_def: str,
    debug_validity: bool = False,
) -> dict[str, object]:
    sector_edges = np.linspace(0.0, np.pi, n_sectors + 1)
    sector_weights = np.cos(sector_edges[:-1]) - np.cos(sector_edges[1:])
    legendre_design = _sector_averaged_legendre(sector_edges, l_max)

    times: list[float] = []
    b = [[] for _ in range(l_max + 1)]
    b_over_b0 = [[] for _ in range(l_max + 1)]
    b_rho = ([[] for _ in range(l_max + 1)]
             if front_def == "midpoint" else None)
    front_min: list[float | None] = []
    front_median: list[float | None] = []
    front_max: list[float | None] = []
    h_f_median: list[float | None] = []
    n_valid_sectors: list[int] = []
    rho_up_values: list[float] = []
    disagreement_q95 = ([] if front_def == "midpoint" else None)
    rho_pass_success_fraction = ([] if front_def == "midpoint" else None)
    sector_valid_values: list[list[int]] = []
    invalid_reason_counts = ([] if debug_validity else None)
    previous_front_median = float("nan")

    for snapshot_index, snapshot in enumerate(snapshots):
        radius = np.sqrt(
            snapshot.r_cell * snapshot.r_cell
            + snapshot.z_cell * snapshot.z_cell)
        theta = np.clip(
            np.arctan2(snapshot.r_cell, snapshot.z_cell), 0.0, np.pi)
        if radius.size == 0:
            raise ValueError(f"{snapshot.path}: no finite cells")

        if snapshot_index == 0:
            s_ref = float(np.percentile(radius, 90.0))
        else:
            s_ref = previous_front_median
        upstream_mask = radius < 0.5 * s_ref
        if np.count_nonzero(upstream_mask) >= 50:
            rho_up = float(np.median(snapshot.rho[upstream_mask]))
        else:
            rho_up = float(np.percentile(snapshot.rho, 10.0))

        sector = np.floor(theta / (np.pi / n_sectors)).astype(np.int64)
        sector = np.clip(sector, 0, n_sectors - 1)
        front = np.full(n_sectors, np.nan, dtype=float)
        h_front = np.full(n_sectors, np.nan, dtype=float)
        reason_counts = (
            dict.fromkeys(_INVALID_REASON_KEYS, 0) if debug_validity else None
        )
        if front_def == "jump":
            threshold = rho_jump_factor * rho_up
            for k in range(n_sectors):
                in_sector = sector == k
                front[k], h_front[k] = _front_radius(
                    radius[in_sector], snapshot.rho[in_sector], threshold)
                if reason_counts is not None and not np.isfinite(front[k]):
                    reason = (
                        "no_cells" if np.count_nonzero(in_sector) < 2
                        else "other"
                    )
                    reason_counts[reason] += 1
            valid_mask = np.isfinite(front)
            coefficients_rho = None
        else:
            if snapshot.pressure is None:
                raise ValueError(
                    f"{snapshot.path}: pressure was not loaded for midpoint front")
            front_rho = np.full(n_sectors, np.nan, dtype=float)
            for k in range(n_sectors):
                in_sector = sector == k
                front[k], h_front[k], pressure_reason = _midpoint_front_radius(
                    radius[in_sector], snapshot.pressure[in_sector])
                front_rho[k], _, rho_reason = _midpoint_front_radius(
                    radius[in_sector], snapshot.rho[in_sector])
                if reason_counts is not None:
                    if not np.isfinite(front[k]):
                        reason_counts[pressure_reason or "other"] += 1
                    elif not np.isfinite(front_rho[k]):
                        reason_counts["rho_pass_failed"] += 1
                    elif (
                        not np.isfinite(h_front[k])
                        or h_front[k] <= 0.0
                        or pressure_reason is not None
                        or rho_reason is not None
                    ):
                        reason_counts["other"] += 1
            valid_mask = np.isfinite(front)
            rho_valid_mask = np.isfinite(front_rho)
            disagreement_valid_mask = (
                valid_mask
                & rho_valid_mask
                & np.isfinite(h_front)
                & (h_front > 0.0)
            )
            normalized_disagreement = (
                np.abs(
                    front[disagreement_valid_mask]
                    - front_rho[disagreement_valid_mask]
                )
                / h_front[disagreement_valid_mask]
            )
            if normalized_disagreement.size:
                assert disagreement_q95 is not None
                disagreement_q95.append(_json_float(
                    float(np.quantile(normalized_disagreement, 0.95))))
            else:
                assert disagreement_q95 is not None
                disagreement_q95.append(None)
            assert rho_pass_success_fraction is not None
            rho_pass_success_fraction.append(
                float(np.count_nonzero(rho_valid_mask)) / n_sectors)
            coefficients_rho = _fit_legendre(
                front_rho, legendre_design, sector_weights)

        valid_front = front[np.isfinite(front)]
        if valid_front.size:
            current_min = float(np.min(valid_front))
            current_median = float(np.median(valid_front))
            current_max = float(np.max(valid_front))
            current_h_f_median = float(np.median(h_front[valid_mask]))
        else:
            current_min = current_median = current_max = float("nan")
            current_h_f_median = float("nan")
        previous_front_median = current_median

        coefficients = _fit_legendre(
            front, legendre_design, sector_weights)
        if np.isfinite(coefficients[0]) and coefficients[0] != 0.0:
            normalized = coefficients / coefficients[0]
        else:
            normalized = np.full(l_max + 1, np.nan, dtype=float)

        times.append(float(snapshot.time))
        for degree in range(l_max + 1):
            b[degree].append(_json_float(coefficients[degree]))
            b_over_b0[degree].append(_json_float(normalized[degree]))
            if b_rho is not None and coefficients_rho is not None:
                b_rho[degree].append(_json_float(coefficients_rho[degree]))
        front_min.append(_json_float(current_min))
        front_median.append(_json_float(current_median))
        front_max.append(_json_float(current_max))
        h_f_median.append(_json_float(current_h_f_median))
        n_valid_sectors.append(int(valid_front.size))
        rho_up_values.append(rho_up)
        sector_valid_values.append(valid_mask.astype(np.int64).tolist())
        if invalid_reason_counts is not None and reason_counts is not None:
            invalid_reason_counts.append(reason_counts)

    payload: dict[str, object] = {
        "schema": 2,
        "front_def": front_def,
        "time": times,
        "b": b,
        "b_rho": b_rho,
        "b_over_b0": b_over_b0,
        "s_f_min": front_min,
        "s_f_median": front_median,
        "s_f_max": front_max,
        "h_f_median": h_f_median,
        "n_valid_sectors": n_valid_sectors,
        "rho_up": rho_up_values,
        "front_def_disagreement_q95_over_h": disagreement_q95,
        "rho_pass_success_fraction": rho_pass_success_fraction,
        "sector_valid": sector_valid_values,
    }
    if invalid_reason_counts is not None:
        payload["invalid_reason_counts"] = invalid_reason_counts
    return payload


def _float_series(values: list[float | None]) -> np.ndarray:
    return np.asarray([
        float(value) if value is not None else np.nan for value in values
    ], dtype=float)


def write_plot(payload: dict[str, object], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.asarray(payload["time"], dtype=float)
    front_min = _float_series(payload["s_f_min"])
    front_median = _float_series(payload["s_f_median"])
    front_max = _float_series(payload["s_f_max"])

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].fill_between(
        time, front_min, front_max, alpha=0.25, label="min-max envelope")
    axes[0].plot(time, front_median, label="median")
    axes[0].set_ylabel(r"$s_f$ [cm]")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    normalized = payload["b_over_b0"]
    for degree in range(1, len(normalized)):
        axes[1].plot(
            time, _float_series(normalized[degree]), label=rf"$l={degree}$")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel(r"$b_l/b_0$")
    axes[1].grid(True, alpha=0.3)
    if len(normalized) > 1:
        axes[1].legend(ncol=2)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Directory containing <case_name>_NNNN.h5 snapshot files")
    parser.add_argument("--n-sectors", type=int, default=96)
    parser.add_argument("--l-max", type=int, default=32)
    parser.add_argument(
        "--front-def", choices=("midpoint", "jump"), default="jump",
        help=(
            'Front-radius estimator; "midpoint" '
            "(diagnostic: fails its spherical null test on multiblock tier "
            "meshes; do not use for gates pending redesign)"))
    parser.add_argument("--rho-jump-factor", type=float, default=2.0)
    parser.add_argument("--debug-validity", action="store_true")
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--out-png", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_sectors <= 0:
        parser.error("--n-sectors must be positive")
    if args.l_max < 0:
        parser.error("--l-max must be non-negative")
    if args.n_sectors < 2 * (args.l_max + 1):
        parser.error("--n-sectors must be at least 2 * (--l-max + 1)")
    if not np.isfinite(args.rho_jump_factor) or args.rho_jump_factor <= 0.0:
        parser.error("--rho-jump-factor must be finite and positive")

    snapshot_suffix = re.compile(r"_(\d{4,})$")
    indexed_paths: list[tuple[int, Path]] = []
    for path in args.run_dir.glob("*.h5"):
        if path.name.endswith("_history.h5"):
            continue
        match = snapshot_suffix.search(path.stem)
        if match is not None:
            indexed_paths.append((int(match.group(1)), path))
    indexed_paths.sort(key=lambda item: (item[0], item[1].name))
    if not indexed_paths:
        parser.error(
            f"no <case_name>_NNNN.h5 snapshot files found in {args.run_dir}")
    indexed_snapshots = [
        (index, read_snapshot(path, read_pressure=args.front_def == "midpoint"))
        for index, path in indexed_paths
    ]
    snapshots = [
        snapshot
        for _, snapshot in sorted(
            indexed_snapshots,
            key=lambda item: (item[0], item[1].time, item[1].path.name),
        )
    ]
    payload = analyze(
        snapshots,
        n_sectors=args.n_sectors,
        l_max=args.l_max,
        rho_jump_factor=args.rho_jump_factor,
        front_def=args.front_def,
        debug_validity=args.debug_validity,
    )

    json_text = json.dumps(payload, indent=2) + "\n"
    if args.out_json is None:
        sys.stdout.write(json_text)
    else:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json_text, encoding="utf-8")
    if args.out_png is not None:
        write_plot(payload, args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
